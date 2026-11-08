"""Gymnasium environment: a BBOB run with learned optimizer hand-offs.

One episode optimizes a single BBOB problem (of a fixed dimension) with exactly
one hand-off: the source optimizer warms up to a switch point sampled
log-uniformly over the FE budget, the agent observes its internal state and
produces the *target* optimizer's internal state (the translation), and the
target runs from that warm start to the end of the budget. Each episode is a
single step.

* **observation** — a dict with the source optimizer's augmented native state,
  the source/target algorithm names, and a small progress context (fraction of
  budget used, normalized best-so-far). The policy turns the native state into
  a ``CanonicalState`` for the encoder; the progress vector is a normalized
  side-input to the critic.
* **action** — the target's full native warm-start dict (shared population +
  decoded specific fields), assembled by the policy from its decoded state.
* **reward** — every mode is ``log_scale(translated improvement) -
  log_scale(baseline improvement)``, differing only in the baseline:
    - ``absolute``: baseline 0 — log-scaled best-so-far improvement over the
      segment.
    - ``relative``: baseline = the hand-off available without a network, running
      the same target: the *computed* warm start for switches into CMA-ES
      (``cat/models/prior.py``), the *lossy default* (shared population only) for
      other targets. Isolates translation quality — what the action controls —
      independent of whether switching was wise.
    - ``noswitch``: baseline = the *source optimizer continuing* (no switch)
      from its full state. Signed switch-worthiness: positive when the switched
      + translated run outperforms staying, negative when it underperforms.
      Outperforming the reference is rewarded, not penalized.
    - ``coldtarget``: baseline = the *target (second) optimizer run cold* from
      the very start of the episode to the end of the budget. The no-switch
      counterpart for the other algorithm — "beat just running the second
      algorithm alone from scratch" rather than "beat continuing the first".
    - ``mixed``: the sum of the ``noswitch`` and ``relative`` rewards — the
      switch is credited both for being worth making (beating the source
      continuing) and for being well translated (beating the lossy hand-off).
      Runs both counterfactual optimizers per step (~3x cost).
  ``reward_mode`` may also be a list (or comma-separated string) of the base
  modes above, in which case the reward is the sum of those modes' component
  rewards (each counterfactual run once); ``mixed`` is just the alias for
  ``["noswitch", "relative"]``. Each component's mean is logged separately.
  All are scaled by the initial gap to the global optimum (fixed at reset).
  Each run's segment is scored by ``reward_measure``: the default ``aocc`` (Area
  Over the Convergence Curve — mean normalized log-precision closed toward the
  optimum over the segment's evaluations, rewarding faster/lower convergence) or
  ``endpoint`` (best-so-far improvement at the segment boundary only). The
  baseline subtraction and mode machinery are identical for both.

The observation/action spaces are declared for completeness but the env is meant
to be driven by the custom loop in ``cat.rl.ppo`` (which reads the structured
objects), not by a generic flat-vector agent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cat.models.prior import prior_native
from cat.optimizers.base import sample_switch_points
from cat.optimizers.portfolio import PORTFOLIO
from cat.rl.reward import (  # re-exported for back-compat (tests import these here)
    _REWARD_EPS,
    _log_scale,
    log_scaled_improvement,
    run_term,
    scaled_improvement,
)
from cat.state.canonical import augment_native, has_full_state, warm_start_optimizer
from cat.suite.ela import ELA_DIM, MAX_HISTORY_SAMPLE, compute_ela_features

__all__ = [
    "_REWARD_EPS",
    "TranslationEnv",
    "_log_scale",
    "log_scaled_improvement",
    "scaled_improvement",
]


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
        super().__init__()
        if not problem_ids:
            raise ValueError("problem_ids is empty")
        # reward_mode may be a single mode, the 'mixed' alias, or a list of
        # modes whose component rewards are summed. Normalize to a flat list of
        # base modes; 'mixed' expands to [noswitch, relative].
        self.reward_mode = reward_mode
        self._component_modes = self._normalize_reward_modes(reward_mode)
        # Per-direction reward logging applies to any comparable baseline; a
        # pure (single) 'absolute' reward is scale-varying and not broken out.
        self.logs_per_direction = self._component_modes != ["absolute"]
        self.algo_a = algo_a
        self.algo_b = algo_b
        self.problem_ids = problem_ids
        self.suite = suite
        self.fe_multiplier = fe_multiplier
        # n_individuals is the shared fallback; n_individuals_a/b let each
        # algorithm run with its own population size.
        self.n_individuals = n_individuals
        self.n_a = n_individuals_a if n_individuals_a is not None else n_individuals
        self.n_b = n_individuals_b if n_individuals_b is not None else n_individuals
        self.switch_cdb = switch_cdb
        self.random_start = random_start
        # scale_feature adds a problem-scale channel (log10 of the fixed initial
        # optimum gap) to the critic context. The default context is otherwise
        # scale-INVARIANT (tanh(best/range)), so the critic has no way to tell
        # problems of different magnitude apart — this exposes that, so it can
        # capture the between-problem return variance (the EV ceiling).
        self.scale_feature = scale_feature
        # use_ela appends 22 Exploratory Landscape Analysis features (pflacco),
        # computed from the points evaluated so far this episode (up to the
        # current switch, capped at MAX_HISTORY_SAMPLE), to the critic context.
        # Unlike the scale channel these are *landscape* descriptors (linearity,
        # multimodality, dispersion, information content), which correlate with
        # the fractional improvement a problem allows — the between-problem
        # structure that caps explained variance. Critic-side only.
        self.use_ela = use_ela
        self.context_dim = 2 + int(scale_feature) + (ELA_DIM if use_ela else 0)
        # Number of optimizer seeds the per-step reward is averaged over (paired
        # per seed for CRN). 1 = single-sample (original). >1 trades ~K x the
        # counterfactual optimizer runs for ~1/K reward variance.
        if reward_seeds < 1:
            raise ValueError("reward_seeds must be >= 1")
        self.reward_seeds = int(reward_seeds)
        # How each run's per-segment score is measured, before the baseline
        # subtraction and log/scale:
        #   'aocc' (default) — Area Over the Convergence Curve: the mean, over
        #     the segment's evaluations, of the normalized log-precision closed
        #     toward the optimum. Rewards converging FASTER/LOWER, so the value
        #     of the hand-off (which lives in the transient) survives even one
        #     long segment, where the endpoint washes out. Bounded [0, 1].
        #   'endpoint' — the original: log-scaled best-so-far improvement at the
        #     segment boundary only.
        if reward_measure not in ("aocc", "endpoint"):
            raise ValueError(f"unknown reward_measure {reward_measure!r}")
        self.reward_measure = reward_measure
        # AOCC log-precision span: how many decades of the gap-at-switch map onto
        # [0, 1]. The floor g_lo = g_hi * 10**(-aocc_decades).
        self.aocc_decades = float(aocc_decades)
        self._seed = seed
        self.dim = self._parse_dim(problem_ids[0])

        # Declared for completeness; the custom PPO loop reads structured objects.
        self.observation_space = spaces.Dict(
            {
                "context": spaces.Box(
                    -np.inf, np.inf, shape=(self.context_dim,), dtype=np.float32
                )
            }
        )
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)

        self._problem_idx = 0
        self._reset_episode_state()

    @staticmethod
    def _parse_dim(pid: str) -> int:
        return int(pid.split("_d")[-1])

    # Base reward modes and the composite aliases that expand into them.
    _BASE_REWARD_MODES = ("absolute", "noswitch", "relative", "coldtarget")
    _REWARD_ALIASES: ClassVar[dict] = {"mixed": ["noswitch", "relative"]}

    @classmethod
    def _normalize_reward_modes(cls, reward_mode) -> list[str]:
        """Flatten ``reward_mode`` (a single mode, the 'mixed' alias, or a list
        of modes to sum) into a list of base modes. A comma-separated string is
        accepted as a list. Validates every resulting mode."""
        if isinstance(reward_mode, str):
            items = [m.strip() for m in reward_mode.split(",") if m.strip()]
        else:
            items = [str(m).strip() for m in reward_mode]
        if not items:
            raise ValueError("reward_mode is empty")
        modes: list[str] = []
        for m in items:
            modes.extend(cls._REWARD_ALIASES.get(m, [m]))
        for m in modes:
            if m not in cls._BASE_REWARD_MODES:
                raise ValueError(f"unknown reward_mode {m!r}")
        return modes

    def _reset_episode_state(self):
        self._problem = None
        self._cfg = None
        self._max_fe = 0
        self._n_fe = 0
        self._best_y = float("inf")
        self._best_x = None
        self._range = 1.0
        self._optimum = None  # global optimum value when the suite exposes it (for AOCC)
        self._switch_fe = None
        self._done = False
        self._src_native = None
        self._src_algo = None
        self._tgt_algo = None
        self._hist_x = None  # accumulated evaluated points this episode (for ELA)
        self._hist_y = None

    # ------------------------------------------------------------------ #

    def _n_for(self, algo: str) -> int | None:
        return self.n_a if algo == self.algo_a else self.n_b

    def _make_optimizer(self, algo: str, target_fe: int, seed_rng: int):
        options = {
            "max_function_evaluations": self._max_fe,
            "target_fe": int(target_fe),
            "best_so_far_y": self._best_y
            if self._best_y < float("inf")
            else float("inf"),
            "seed_rng": seed_rng,
            "verbose": False,
        }
        n = self._n_for(algo)
        if n is not None:
            options["n_individuals"] = n
        opt = PORTFOLIO[algo](self._cfg, options)
        opt.n_function_evaluations = self._n_fe
        return opt

    def _accumulate_history(self, result: dict) -> None:
        """Append a run's evaluated points to the episode history (for ELA).

        Only the *real* (acted) trajectory feeds the history — counterfactual
        baseline runs are hypothetical and must not pollute the landscape
        sample. Keeps the most recent ``MAX_HISTORY_SAMPLE`` points."""
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
            # Absolute problem scale (fixed per episode): lets the critic tell
            # problems of different magnitude apart. obs_rms standardizes it.
            ctx.append(float(np.log10(max(self._range, 1e-12))))
        if self.use_ela:
            # Landscape features of the samples gathered up to this switch.
            # obs_rms standardizes them (their natural scales vary wildly).
            # ``include_ela`` is False for the unused terminal observation, to
            # skip the (expensive) pflacco call where its value is never read.
            ela = self._ela() if include_ela else np.zeros(ELA_DIM, dtype=np.float32)
            ctx.extend(float(v) for v in ela)
        return {
            "native": self._src_native,
            "source_algo": self._src_algo,
            "target_algo": self._tgt_algo,
            "target_n": self._n_for(self._tgt_algo),
            "context": np.array(ctx, dtype=np.float32),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pid = self.problem_ids[self._problem_idx % len(self.problem_ids)]
        self._reset_episode_state()
        self._problem = self.suite.get_problem(pid)
        dim = self._problem.dimension
        self._cfg = {
            "fitness_function": self._problem,
            "ndim_problem": dim,
            "lower_boundary": self._problem.lower_bounds,
            "upper_boundary": self._problem.upper_bounds,
        }
        self._max_fe = self.fe_multiplier * dim
        # Conservative floor for switch-point spacing/probe sizing: room enough
        # regardless of which algo (with its own population size) runs next.
        pop = max(self.n_a or 20, self.n_b or 20)

        base = (self._seed * 1_000_003 + self._problem_idx) % (2**31)
        sw_rng = np.random.default_rng(base)
        self._problem_idx += 1

        # Which algorithm does the cold warmup, i.e. which direction the single
        # switch translates. Randomizing it per episode trains both directions;
        # with random_start=False every episode is algo_a -> algo_b.
        if self.random_start and sw_rng.random() < 0.5:
            start_algo, other_algo = self.algo_b, self.algo_a
        else:
            start_algo, other_algo = self.algo_a, self.algo_b

        # Agent-independent probe: fixes best/scale before the agent acts.
        n_probe = min(2 * dim + 8, max(self._max_fe // 4, dim + 1))
        # The warmup segment must leave room (after the probe) for the source
        # optimizer to complete at least a couple of generations, else it would
        # return an empty state. Bound the earliest switch boundary accordingly.
        fe_min = min(n_probe + 2 * pop, max(self._max_fe - pop, n_probe + pop))
        # Two boundaries: [switch point, max_fe]; the second is always max_fe.
        self._switch_fe = int(
            sample_switch_points(
                2, self._max_fe, pop, sw_rng, fe_min=fe_min, cdb=self.switch_cdb
            )[0]
        )

        lb, ub = self._problem.lower_bounds, self._problem.upper_bounds
        x_probe = sw_rng.uniform(lb, ub, size=(n_probe, dim))
        y_probe = np.array([self._problem(x) for x in x_probe], dtype=float)
        i_best = int(np.argmin(y_probe))
        self._best_y = float(y_probe[i_best])
        self._best_x = x_probe[i_best]
        self._n_fe = n_probe
        if self.use_ela:
            self._hist_x, self._hist_y = x_probe, y_probe  # seed the ELA history

        # Warm up the first algorithm (cold) to build a genuine source state.
        warmup = self._make_optimizer(start_algo, self._switch_fe, base + 1)
        warmup.set_data(best_x=self._best_x, best_y=self._best_y)
        result = warmup.optimize()
        self._update_best(result)
        self._n_fe = result.get("n_function_evaluations", self._n_fe)
        self._accumulate_history(result)

        # Reward scale = remaining distance to the global optimum at the end of
        # the first optimization period (the warmup). Improvements over the rest
        # of the episode are measured as a fraction of this initial gap. Falls
        # back to a probe-spread range on suites that don't expose the optimum.
        optimum = getattr(self._problem, "optimum", None)
        if optimum is not None:
            self._optimum = float(optimum)
            self._range = max(self._best_y - self._optimum, 1e-12)
        else:
            self._optimum = None
            self._range = max(float(np.median(y_probe)) - self._best_y, 1e-5)

        self._src_native = augment_native(warmup, dim)
        self._src_algo = start_algo
        self._tgt_algo = other_algo
        self._done = False
        return self._observation(), {"problem_id": pid, "dimension": dim}

    def step(self, action_native: dict):
        """``action_native`` is the target's full warm-start dict (shared + specific)."""
        if self._done:
            raise RuntimeError("the episode's single switch is done; call reset()")
        target = self._tgt_algo
        prev_best = self._best_y
        target_fe = self._max_fe
        # Compute the reward BEFORE mutating episode state, so every
        # counterfactual is constructed from the same pre-segment best / FE.
        rng = max(self._range, 1e-12)

        # Seed-average the reward. The single-segment reward is a difference of
        # stochastic optimizer runs, so one sample has enormous variance
        # (observed return std >> mean) — which buries both the critic target
        # and the policy gradient in noise. Running the acted segment (and, per
        # mode, each counterfactual baseline) over ``reward_seeds`` seeds and
        # averaging the per-seed component rewards cuts that noise ~1/K. Each
        # seed j pairs the translated run with its baselines under the SAME seed
        # (base_seg_seed + j), preserving the common-random-numbers cancellation
        # within a seed. Seed j == 0 is the run recorded as the episode's
        # outcome (best-so-far, FE count); the other K-1 translated
        # runs serve the reward estimate only, like the counterfactuals — so
        # reward_seeds == 1 reproduces the previous single-sample behaviour.
        base_seg_seed = self._seed + 1
        translated_terms: list[float] = []
        advance_opt = advance_result = None
        for j in range(self.reward_seeds):
            opt = self._make_optimizer(target, target_fe, base_seg_seed + j)
            warm_start_optimizer(opt, action_native)
            result = opt.optimize()
            translated_terms.append(self._run_term(result, prev_best, rng))
            if j == 0:
                advance_opt, advance_result = opt, result

        # Each baseline is a counterfactual improvement, log-scaled the same way
        # as the translated run; the mode selects which reference(s) to subtract.
        #   noswitch: the *source* optimizer continuing (no switch) from its full
        #     state over the same segment — a REFERENCE, not a trajectory to
        #     imitate. Signed: outperforming it is rewarded, underperforming is
        #     penalized, but a genuinely better switch is *not* punished for
        #     deviating (unlike the old -|Δ| form).
        #   relative: the same target run from the hand-off available without a
        #     network (same seed; see relative_baseline_native). Isolates exactly
        #     what the action controls — translation quality, independent of
        #     whether switching was wise.
        #   coldtarget: the *target* (second) optimizer run cold from the very
        #     start of the episode to the end of the budget — the no-switch
        #     counterpart for the *other* algorithm. Like noswitch (signed
        #     switch-worthiness), but the reference is "just run the second
        #     algorithm alone from scratch" rather than "keep running the first".
        #   mixed: an alias for the list [noswitch, relative] — the switch is
        #     credited both for being worth making and for being well translated.
        # A reward_mode that lists several of these modes sums their component
        # rewards; 'mixed' is the [noswitch, relative] case. Every component is
        # itself averaged over the reward_seeds seeds (paired per seed). The
        # per-component means are exposed in reward_parts for separate logging.
        components = {
            mode: float(
                np.mean(
                    [
                        self._reward_component(
                            mode, target, target_fe, base_seg_seed + j,
                            prev_best, rng, translated_terms[j],
                        )
                        for j in range(self.reward_seeds)
                    ]
                )
            )
            for mode in self._component_modes
        }
        reward = float(sum(components.values()))
        reward_parts = components if len(components) > 1 else None

        # Record the seed-0 run as the episode's outcome (best-so-far, FE count,
        # final state). Roles swap only so the returned terminal observation is
        # well formed; it is never acted on.
        self._update_best(advance_result)
        self._n_fe = advance_result.get("n_function_evaluations", self._n_fe)
        self._src_native = augment_native(advance_opt, self.dim)
        self._src_algo = target
        self._tgt_algo = self.algo_a if target == self.algo_b else self.algo_b
        self._done = True

        terminated = True
        info = {"best_y": self._best_y, "n_fe": self._n_fe, "reward": reward}
        if reward_parts is not None:
            info["reward_parts"] = reward_parts
        # The only step is terminal, so its observation is never acted on: skip
        # the (expensive) ELA computation for it.
        obs = self._observation(include_ela=False)
        return obs, reward, terminated, False, info

    def _reward_component(
        self, mode, target, target_fe, seg_seed, prev_best, rng, translated_term
    ) -> float:
        """Single-mode reward: log_scale(translated) - log_scale(baseline), with
        the baseline selected by ``mode``. The additive building block summed
        over ``self._component_modes`` to form the step reward."""
        if mode == "absolute":
            return translated_term
        if mode == "noswitch":
            return translated_term - self._noswitch_term(target_fe, seg_seed, prev_best, rng)
        if mode == "relative":
            return translated_term - self._relative_term(
                target, target_fe, seg_seed, prev_best, rng
            )
        if mode == "coldtarget":
            return translated_term - self._coldtarget_term(
                target, target_fe, seg_seed, prev_best, rng
            )
        raise ValueError(f"unknown reward component {mode!r}")

    def _run_term(self, result: dict, prev_best: float, rng: float) -> float:
        """Score one optimizer run's segment via ``cat.rl.reward.run_term``, with
        this env's measure/optimum/decades — the building block every mode's
        translated/baseline term shares, so they stay comparable within a step."""
        return run_term(
            result,
            prev_best,
            rng,
            measure=self.reward_measure,
            optimum=self._optimum,
            decades=self.aocc_decades,
        )

    def _noswitch_term(self, target_fe, seg_seed, prev_best, rng) -> float:
        """Score of the *source* optimizer continuing (no switch) from its own
        full state over the same segment — the 'noswitch' reference. Does not
        advance the episode."""
        opt = self._make_optimizer(self._src_algo, target_fe, seg_seed)
        warm_start_optimizer(opt, self._src_native)
        return self._run_term(opt.optimize(), prev_best, rng)

    def relative_baseline_native(self, seed: int = 0) -> dict:
        """The current switch's 'relative' reference hand-off. Into CMA-ES it is the
        computed warm start: the lossy default leaves CMA-ES a random mean, which
        amounts to a restart. Other targets get the shared population only."""
        n, target = self._src_native, self._tgt_algo
        if target == "CMAES":
            return prior_native(
                self._src_algo, n, self._n_for(target), np.random.default_rng(seed)
            )
        return {"x": n["x"], "y": n["y"], "best_x": n["best_x"], "best_y": n["best_y"]}

    def _relative_term(self, target, target_fe, seg_seed, prev_best, rng) -> float:
        """Score of ``relative_baseline_native`` running the same target over the
        same segment — the 'relative' reference. Does not advance the episode."""
        opt = self._make_optimizer(target, target_fe, seg_seed)
        warm_start_optimizer(opt, self.relative_baseline_native(seg_seed))
        return self._run_term(opt.optimize(), prev_best, rng)

    def _coldtarget_term(self, target, target_fe, seg_seed, prev_best, rng) -> float:
        """Score of the *target* (second) optimizer running cold from the very
        start (FE 0, no warm start) to the end of the budget — the 'coldtarget'
        reference. Does not advance the episode."""
        opt = self._make_optimizer(target, target_fe, seg_seed)
        # Override the episode-state seeding _make_optimizer applies: run the
        # algorithm from scratch, with no prior best and the full FE budget.
        opt.n_function_evaluations = 0
        opt.best_so_far_y = float("inf")
        return self._run_term(opt.optimize(), prev_best, rng)

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
