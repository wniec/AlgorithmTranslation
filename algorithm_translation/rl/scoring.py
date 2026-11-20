"""Scoring a hand-off at one switch.

A ``SwitchPoint`` freezes everything the reward needs once the warmup has
reached the switch. A ``SwitchScorer`` runs the target optimizer warm-started
from a candidate hand-off (over one or more post-switch seeds) and subtracts the
counterfactual baselines selected by the reward mode:

* ``absolute``   — no baseline: the translated run's own score;
* ``noswitch``   — the source optimizer simply continuing from its full state;
* ``relative``   — the hand-off available without a network, running the same
  target: the computed warm start into CMA-ES (``models/prior.py``), the shared
  population only otherwise. Isolates translation quality;
* ``coldtarget`` — the target optimizer run alone from FE 0, scored over the
  same post-switch window;
* ``mixed``      — alias for ``noswitch,relative``.

A comma-separated list of modes sums their rewards. Each run's segment is scored
by ``rl.reward.run_term`` (AOCC by default). Baselines do not depend on the
action, so they are computed once per (mode, seed) and cached: scoring several
candidates from the same switch (GRPO) runs each counterfactual only once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from algorithm_translation.models.prior import prior_native
from algorithm_translation.rl.reward import run_term
from algorithm_translation.runs import make_optimizer, run_segment, shared_handoff

BASE_REWARD_MODES = ("absolute", "noswitch", "relative", "coldtarget")
REWARD_ALIASES = {"mixed": ["noswitch", "relative"]}


def normalize_reward_modes(reward_mode: str | Sequence[str]) -> list[str]:
    """Flatten a reward mode (one mode, an alias, a comma-separated string or a
    list) into a validated list of base modes."""
    if isinstance(reward_mode, str):
        items = [m.strip() for m in reward_mode.split(",") if m.strip()]
    else:
        items = [str(m).strip() for m in reward_mode]
    if not items:
        raise ValueError("reward_mode is empty")
    modes: list[str] = []
    for m in items:
        modes.extend(REWARD_ALIASES.get(m, [m]))
    for m in modes:
        if m not in BASE_REWARD_MODES:
            raise ValueError(f"unknown reward_mode {m!r}")
    return modes


@dataclass(frozen=True)
class SwitchPoint:
    """The state of an episode at its switch."""

    cfg: dict  # problem dict for the optimizers (see runs.problem_cfg)
    max_fe: int  # total FE budget
    n_fe: int  # FEs already spent when the switch happens
    best_y: float  # incumbent at the switch
    scale: float  # reward range: gap to the optimum at the switch (or probe spread)
    optimum: float | None
    source_algo: str
    target_algo: str
    source_native: dict  # the source optimizer's augmented native state
    n_source: int | None  # population sizes (None = algorithm default)
    n_target: int | None


@dataclass
class ScoredAction:
    reward: float
    components: dict[str, float]  # per-mode reward, each averaged over seeds
    # The first seed's translated run: the episode outcome step() records.
    optimizer: object
    result: dict


@dataclass
class SwitchScorer:
    switch: SwitchPoint
    modes: list[str]
    seeds: list[int]  # post-switch seeds the reward is averaged over
    measure: str = "aocc"
    decades: float = 4.0
    _baselines: dict[tuple[str, int], float] = field(default_factory=dict, repr=False)

    # -- building blocks ------------------------------------------------- #

    def term(self, result: dict, skip: int = 0) -> float:
        """Score one run's segment (see ``rl.reward.run_term``)."""
        return run_term(
            result,
            self.switch.best_y,
            max(self.switch.scale, 1e-12),
            measure=self.measure,
            optimum=self.switch.optimum,
            decades=self.decades,
            skip=skip,
        )

    def run_target(self, payload: dict | None, seed: int):
        """Run the target from the switch to the end of the budget, warm-started
        from ``payload`` (cold when None, in which case it also does not know the
        incumbent). Returns ``(optimizer, result)``."""
        sw = self.switch
        return run_segment(
            sw.target_algo,
            sw.cfg,
            max_fe=sw.max_fe,
            target_fe=sw.max_fe,
            seed=seed,
            n_individuals=sw.n_target,
            n_fe_start=sw.n_fe,
            best_y=sw.best_y if payload is not None else float("inf"),
            payload=payload,
        )

    def relative_baseline_native(self, seed: int = 0) -> dict:
        """The hand-off available without a network: the computed warm start
        into CMA-ES (the lossy default would leave CMA-ES a random mean, i.e. a
        restart), the shared population for every other target."""
        sw = self.switch
        if sw.target_algo == "CMAES":
            return prior_native(
                sw.source_algo, sw.source_native, sw.n_target, np.random.default_rng(seed)
            )
        return shared_handoff(sw.source_native)

    def baseline(self, mode: str, seed: int) -> float:
        """The ``mode`` counterfactual's score under ``seed`` (cached)."""
        key = (mode, int(seed))
        if key not in self._baselines:
            self._baselines[key] = self._run_baseline(mode, seed)
        return self._baselines[key]

    def _run_baseline(self, mode: str, seed: int) -> float:
        sw = self.switch
        if mode == "noswitch":
            _, result = run_segment(
                sw.source_algo,
                sw.cfg,
                max_fe=sw.max_fe,
                target_fe=sw.max_fe,
                seed=seed,
                n_individuals=sw.n_source,
                n_fe_start=sw.n_fe,
                best_y=sw.best_y,
                payload=sw.source_native,
            )
            return self.term(result)
        if mode == "relative":
            _, result = self.run_target(self.relative_baseline_native(seed), seed)
            return self.term(result)
        if mode == "coldtarget":
            # From FE 0 with no incumbent, scored only over the post-switch
            # window so it covers the same evaluations as the translated run.
            opt = make_optimizer(
                sw.target_algo,
                sw.cfg,
                max_fe=sw.max_fe,
                target_fe=sw.max_fe,
                seed=seed,
                n_individuals=sw.n_target,
            )
            return self.term(opt.optimize(), skip=sw.n_fe)
        raise ValueError(f"unknown baseline mode {mode!r}")

    # -- the reward -------------------------------------------------------- #

    def score(self, action_native: dict) -> ScoredAction:
        """Reward of handing ``action_native`` to the target.

        The translated run is repeated over every seed in ``self.seeds``; under
        each seed it is paired with the baselines run under the same seed
        (common random numbers), and every mode's reward is averaged over seeds.
        Modes are summed into the total reward."""
        terms, first = [], None
        for seed in self.seeds:
            opt, result = self.run_target(action_native, seed)
            terms.append(self.term(result))
            if first is None:
                first = (opt, result)
        components = {
            mode: float(
                np.mean(
                    [
                        t if mode == "absolute" else t - self.baseline(mode, seed)
                        for t, seed in zip(terms, self.seeds, strict=True)
                    ]
                )
            )
            for mode in self.modes
        }
        return ScoredAction(
            reward=float(sum(components.values())),
            components=components,
            optimizer=first[0],
            result=first[1],
        )
