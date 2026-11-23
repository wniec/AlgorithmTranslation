"""Gymnasium environment: a BBOB run with one learned optimizer hand-off.

An episode optimizes a single BBOB problem. A random probe fixes the initial
incumbent, the source optimizer warms up to a switch point sampled log-uniformly
over the FE budget, the agent observes the source's internal state and produces
the *target* optimizer's state (the translation), and the target runs from that
warm start to the end of the budget. Each episode is a single step.

* **observation** — the source optimizer's augmented native state, the
  source/target algorithm names, the target's population size, and a small
  progress context for the critic (fraction of budget used, normalized
  best-so-far, optionally a problem-scale channel and ELA features).
* **action** — the target's full native warm-start dict (shared population +
  decoded specific fields), assembled by the policy.
* **reward** — the translated run's score minus the baseline(s) chosen by
  ``reward_mode``; see ``rl.scoring`` for the modes and ``rl.reward`` for how a
  segment is scored.

The observation/action spaces are declared for completeness; the env is driven
by the custom loops in ``rl.ppo`` / ``rl.grpo``, which read the structured
objects directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from algorithm_translation.optimizers.base import sample_switch_points
from algorithm_translation.rl.scoring import SwitchPoint, SwitchScorer, normalize_reward_modes
from algorithm_translation.runs import make_optimizer, problem_cfg
from algorithm_translation.seeding import Stream, derive_seed
from algorithm_translation.state.canonical import augment_native, has_full_state
from algorithm_translation.suite import problem_dim
from algorithm_translation.suite.ela import ELA_DIM, MAX_HISTORY_SAMPLE, compute_ela_features

__all__ = ["TranslationEnv", "sample_episode_start"]


def sample_episode_start(problem, max_fe: int, pop: int, rng, switch_cdb: float = 1.0):
    """Draw an episode's switch point and run its agent-independent probe.

    Shared by ``TranslationEnv.reset`` and the held-out evaluator so both place
    the switch from the same distribution and start the warmup from the same
    kind of probe. Draws from ``rng`` in a fixed order (switch point, then probe
    points).

    ``pop`` is a population-size floor for switch-point spacing (room enough
    for whichever algorithm runs next). Returns ``(switch_fe, x_probe,
    y_probe)``; the probe is already evaluated and costs ``len(y_probe)`` FEs.
    """
    dim = problem.dimension
    n_probe = min(2 * dim + 8, max(max_fe // 4, dim + 1))
    # Leave room after the probe for the source to complete a couple of
    # generations, else it would reach the switch with an empty state.
    fe_min = min(n_probe + 2 * pop, max(max_fe - pop, n_probe + pop))
    # Two boundaries: [switch point, max_fe]; only the first is random.
    switch_fe = int(
        sample_switch_points(2, max_fe, pop, rng, fe_min=fe_min, cdb=switch_cdb)[0]
    )
    lb, ub = problem.lower_bounds, problem.upper_bounds
    x_probe = rng.uniform(lb, ub, size=(n_probe, dim))
    y_probe = np.array([problem(x) for x in x_probe], dtype=float)
    return switch_fe, x_probe, y_probe


class TranslationEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        algo_a: str,
        algo_b: str,
        problem_ids: list[str],
        suite,
        *,
        fe_multiplier: int = 2000,
        n_individuals: int | None = 12,
        n_individuals_a: int | None = None,
        n_individuals_b: int | None = None,
        reward_mode: str | Sequence[str] = "relative",
        switch_cdb: float = 1.0,
        random_start: bool = True,
        scale_feature: bool = False,
        use_ela: bool = False,
        reward_seeds: int = 1,
        reward_measure: str = "aocc",
        aocc_decades: float = 4.0,
        seed: int = 0,
    ):
        """
        reward_mode: one mode, an alias or a list of modes to sum (see
            ``rl.scoring``).
        random_start: draw the switch direction per episode (False: always
            ``algo_a -> algo_b``).
        scale_feature: add log10 of the initial optimum gap to the critic
            context, so the critic can tell problems of different magnitude
            apart (the rest of the context is scale-invariant).
        use_ela: append Exploratory Landscape Analysis features of the points
            evaluated before the switch to the critic context.
        reward_seeds: post-switch seeds the reward is averaged over (paired with
            the baselines per seed); K costs ~K x the target runs for ~1/K of
            the reward variance.
        reward_measure: ``aocc`` (area over the convergence curve over the
            segment) or ``endpoint`` (best-so-far improvement at its end).
        aocc_decades: how many decades below the gap at the switch earn full
            AOCC credit.
        """
        super().__init__()
        if not problem_ids:
            raise ValueError("problem_ids is empty")
        if reward_seeds < 1:
            raise ValueError("reward_seeds must be >= 1")
        if reward_measure not in ("aocc", "endpoint"):
            raise ValueError(f"unknown reward_measure {reward_measure!r}")
        self.reward_mode = reward_mode
        self._component_modes = normalize_reward_modes(reward_mode)
        # A pure 'absolute' reward is scale-varying, so it is not compared or
        # logged per switch direction.
        self.logs_per_direction = self._component_modes != ["absolute"]
        self.algo_a = algo_a
        self.algo_b = algo_b
        self.problem_ids = problem_ids
        self.suite = suite
        self.fe_multiplier = fe_multiplier
        self.n_individuals = n_individuals
        self.n_a = n_individuals_a if n_individuals_a is not None else n_individuals
        self.n_b = n_individuals_b if n_individuals_b is not None else n_individuals
        self.switch_cdb = switch_cdb
        self.random_start = random_start
        self.scale_feature = scale_feature
        self.use_ela = use_ela
        self.context_dim = 2 + int(scale_feature) + (ELA_DIM if use_ela else 0)
        self.reward_seeds = int(reward_seeds)
        self.reward_measure = reward_measure
        self.aocc_decades = float(aocc_decades)
        self._seed = seed
        self.dim = problem_dim(problem_ids[0])

        self.observation_space = spaces.Dict(
            {
                "context": spaces.Box(
                    -np.inf, np.inf, shape=(self.context_dim,), dtype=np.float32
                )
            }
        )
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)

        self._problem_idx = 0  # episodes started so far
        self._order_pass = None  # pass number whose permutation is cached
        self._order = None
        self._reset_episode_state()

    def _problem_at(self, idx: int) -> str:
        """The problem for episode ``idx``. Each pass over ``problem_ids`` uses
        a fresh permutation seeded by (env seed, pass number), so envs with
        different seeds (e.g. parallel rollout workers) walk the problems in
        different orders, while every pass visits each problem exactly once."""
        n = len(self.problem_ids)
        pass_no, pos = divmod(idx, n)
        if self._order_pass != pass_no:
            rng = np.random.default_rng(derive_seed(self._seed, pass_no, Stream.PROBLEM_ORDER))
            self._order = rng.permutation(n)
            self._order_pass = pass_no
        return self.problem_ids[int(self._order[pos])]

    def _reset_episode_state(self):
        self._problem = None
        self._cfg = None
        self._max_fe = 0
        self._n_fe = 0
        self._best_y = float("inf")
        self._best_x = None
        self._range = 1.0  # reward scale, fixed at the switch
        self._optimum = None  # global optimum value, when the suite exposes it
        self._switch_fe = None
        self._segment_seeds: list[int] = []  # post-switch seeds, one per reward seed
        self._scorer: SwitchScorer | None = None
        self._done = False
        self._src_native = None
        self._src_algo = None
        self._tgt_algo = None
        self._hist_x = None  # evaluated points this episode (for ELA)
        self._hist_y = None

    def _n_for(self, algo: str) -> int | None:
        return self.n_a if algo == self.algo_a else self.n_b

    # ------------------------------------------------------------------ #
    # observation                                                         #
    # ------------------------------------------------------------------ #

    def _accumulate_history(self, result: dict) -> None:
        """Append a run's evaluated points to the episode history (for ELA).

        Only the acted trajectory feeds the history; counterfactual runs are
        hypothetical. Keeps the most recent ``MAX_HISTORY_SAMPLE`` points."""
        if not self.use_ela:
            return
        xh = result.get("x_history")
        yh = result.get("y_history")
        if xh is None or yh is None or len(xh) == 0:
            return
        xh = np.asarray(xh, dtype=float)
        yh = np.asarray(yh, dtype=float)
        if self._hist_x is None:
            self._hist_x, self._hist_y = xh, yh
        else:
            self._hist_x = np.concatenate([self._hist_x, xh])[-MAX_HISTORY_SAMPLE:]
            self._hist_y = np.concatenate([self._hist_y, yh])[-MAX_HISTORY_SAMPLE:]

    def _ela(self) -> np.ndarray:
        if self.use_ela and self._hist_x is not None:
            return compute_ela_features(self._hist_x, self._hist_y)
        return np.zeros(ELA_DIM, dtype=np.float32)

    def _observation(self, include_ela: bool = True) -> dict:
        frac = self._n_fe / max(self._max_fe, 1)
        best = self._best_y if np.isfinite(self._best_y) else 0.0
        ctx = [frac, np.tanh(best / max(self._range, 1e-9))]
        if self.scale_feature:
            ctx.append(float(np.log10(max(self._range, 1e-12))))
        if self.use_ela:
            # The terminal observation is never read: skip the pflacco call.
            ela = self._ela() if include_ela else np.zeros(ELA_DIM, dtype=np.float32)
            ctx.extend(float(v) for v in ela)
        return {
            "native": self._src_native,
            "source_algo": self._src_algo,
            "target_algo": self._tgt_algo,
            "target_n": self._n_for(self._tgt_algo),
            "context": np.array(ctx, dtype=np.float32),
        }

    # ------------------------------------------------------------------ #
    # episode                                                             #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        episode = self._problem_idx
        self._problem_idx += 1
        pid = self._problem_at(episode)
        self._reset_episode_state()
        self._problem = self.suite.get_problem(pid)
        dim = self._problem.dimension
        self._cfg = problem_cfg(self._problem)
        self._max_fe = self.fe_multiplier * dim
        # Switch-spacing floor: room for whichever algorithm runs next.
        pop = max(self.n_a or 20, self.n_b or 20)

        rng = np.random.default_rng(derive_seed(self._seed, episode, Stream.EPISODE))
        self._segment_seeds = [
            derive_seed(self._seed, episode, Stream.SEGMENT, j)
            for j in range(self.reward_seeds)
        ]

        if self.random_start and rng.random() < 0.5:
            start_algo, other_algo = self.algo_b, self.algo_a
        else:
            start_algo, other_algo = self.algo_a, self.algo_b

        self._switch_fe, x_probe, y_probe = sample_episode_start(
            self._problem, self._max_fe, pop, rng, self.switch_cdb
        )
        i_best = int(np.argmin(y_probe))
        self._best_y = float(y_probe[i_best])
        self._best_x = x_probe[i_best]
        self._n_fe = len(y_probe)
        if self.use_ela:
            self._hist_x, self._hist_y = x_probe, y_probe

        # Warm up the first algorithm (cold, knowing the probe's best) to build
        # a genuine source state at the switch.
        warmup = make_optimizer(
            start_algo,
            self._cfg,
            max_fe=self._max_fe,
            target_fe=self._switch_fe,
            seed=derive_seed(self._seed, episode, Stream.WARMUP),
            n_individuals=self._n_for(start_algo),
            best_y=self._best_y,
            n_fe_start=self._n_fe,
        )
        warmup.set_data(best_x=self._best_x, best_y=self._best_y)
        result = warmup.optimize()
        self._update_best(result)
        self._n_fe = result.get("n_function_evaluations", self._n_fe)
        self._accumulate_history(result)

        # Reward scale: the remaining gap to the optimum at the switch (or, on
        # suites without a known optimum, the probe's spread).
        optimum = getattr(self._problem, "optimum", None)
        if optimum is not None:
            self._optimum = float(optimum)
            self._range = max(self._best_y - self._optimum, 1e-12)
        else:
            self._range = max(float(np.median(y_probe)) - self._best_y, 1e-5)

        self._src_native = augment_native(warmup, dim)
        self._src_algo = start_algo
        self._tgt_algo = other_algo
        self._scorer = SwitchScorer(
            SwitchPoint(
                cfg=self._cfg,
                max_fe=self._max_fe,
                n_fe=self._n_fe,
                best_y=self._best_y,
                scale=self._range,
                optimum=self._optimum,
                source_algo=start_algo,
                target_algo=other_algo,
                source_native=self._src_native,
                n_source=self._n_for(start_algo),
                n_target=self._n_for(other_algo),
            ),
            modes=self._component_modes,
            seeds=self._segment_seeds,
            measure=self.reward_measure,
            decades=self.aocc_decades,
        )
        return self._observation(), {"problem_id": pid, "dimension": dim}

    def step(self, action_native: dict):
        """Hand ``action_native`` (the target's warm-start dict) to the target,
        run it to the end of the budget and end the episode."""
        scored = self._score(action_native)
        # The first seed's run is the episode outcome. Roles swap only so the
        # terminal observation is well formed; it is never acted on.
        target = self._tgt_algo
        self._update_best(scored.result)
        self._n_fe = scored.result.get("n_function_evaluations", self._n_fe)
        self._src_native = augment_native(scored.optimizer, self.dim)
        self._src_algo = target
        self._tgt_algo = self.algo_a if target == self.algo_b else self.algo_b
        self._done = True

        info = {"best_y": self._best_y, "n_fe": self._n_fe, "reward": scored.reward}
        if len(scored.components) > 1:
            info["reward_parts"] = scored.components
        obs = self._observation(include_ela=False)
        return obs, scored.reward, True, False, info

    def evaluate_action(self, action_native: dict) -> tuple[float, dict]:
        """Score a candidate hand-off WITHOUT advancing the episode: returns
        ``(reward, info)`` as ``step`` would, leaving the env at the switch so
        more candidates can be scored from the same state (GRPO). All candidates
        share the post-switch seeds, and the baselines are run once."""
        scored = self._score(action_native)
        info = {"reward": scored.reward}
        if len(scored.components) > 1:
            info["reward_parts"] = scored.components
        return scored.reward, info

    def _score(self, action_native: dict):
        if self._done or self._scorer is None:
            raise RuntimeError("no switch to score: call reset() first")
        return self._scorer.score(action_native)

    def relative_baseline_native(self, seed: int = 0) -> dict:
        """The current switch's network-free hand-off (see ``rl.scoring``)."""
        return self._scorer.relative_baseline_native(seed)

    def _update_best(self, result: dict):
        new_best = result.get("best_so_far_y", float("inf"))
        if new_best < self._best_y:
            self._best_y = float(new_best)
            self._best_x = result.get("best_so_far_x", self._best_x)

    def source_has_full_state(self) -> bool:
        return has_full_state(self._src_algo, self._src_native)

    def spec(self, seed: int | None = None) -> dict:
        """Picklable constructor kwargs that rebuild this env (the live ioh
        problem is *not* picklable, but it is created lazily from ``suite`` +
        ``problem_ids`` at ``reset``). Used to reconstruct envs inside parallel
        rollout worker processes; ``seed`` overrides the RNG seed so each worker
        walks a different problem/instance/boundary stream."""
        return {
            "algo_a": self.algo_a,
            "algo_b": self.algo_b,
            "problem_ids": self.problem_ids,
            "suite": self.suite,
            "fe_multiplier": self.fe_multiplier,
            "n_individuals": self.n_individuals,
            "n_individuals_a": self.n_a,
            "n_individuals_b": self.n_b,
            "reward_mode": self.reward_mode,
            "switch_cdb": self.switch_cdb,
            "random_start": self.random_start,
            "scale_feature": self.scale_feature,
            "use_ela": self.use_ela,
            "reward_seeds": self.reward_seeds,
            "reward_measure": self.reward_measure,
            "aocc_decades": self.aocc_decades,
            "seed": self._seed if seed is None else seed,
        }
