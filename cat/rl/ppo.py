"""Custom PPO loop for the state translator.

Standard clipped-surrogate PPO with GAE, specialized in two ways:

* transitions carry **structured, variable-shape** observations and actions
  (the optimizer states), so the update groups them by ``(target_algo, D, N)``
  — same idea as ``cat/data/dataset.py`` — and each minibatch is drawn from one
  shape-homogeneous group;
* the actor objective adds a differentiable **cycle-consistency penalty**
  (``lambda_cycle · A->B->A drift``), realizing the "improvement + cycle penalty"
  objective more sample-efficiently than routing a differentiable quantity
  through the scalar reward. The drift is scored via ``cycle_mode`` — "field"
  (decoded fields, default) or "latent" (re-encoded latents only), mirroring
  ``cat.losses.batch_losses`` — and skipped entirely when ``lambda_cycle == 0``.
"""

from __future__ import annotations

import copy
import math
import multiprocessing as mp
import os
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cat.rl.env import TranslationEnv
from cat.rl.normalize import (
    RunningMeanStd,
    discounted_returns,
    normalize_rewards,
    scale_by_ret_std,
)
from cat.rl.policy import ActorCritic, Step


@dataclass
class PPOConfig:
    updates: int = 50
    rollout_steps: int = 1024
    ppo_epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.25
    # Value-function clip window, decoupled from the policy ratio `clip`. None
    # (the default) disables value clipping — the standard choice, and it frees
    # the critic to track returns that span many units (the old behavior reused
    # `clip`=0.2, which throttled it, worst under --no-norm-reward). Set a float
    # to re-enable (e.g. 0.2 reproduces the old throttle; a larger value is a
    # looser, decoupled window).
    vf_clip: float | None = None
    lambda_cycle: float = 0.05
    cycle_mode: str = "field"  # "field" (decoded fields) or "latent" (re-encoded latents)
    lr: float = 3e-4
    # log_std is a single global scalar (see ActorCritic.log_std): at the shared
    # network lr it can only drift by ~lr per opt.step(), which is too slow to
    # track a moving optimum over the handful of steps in one update. Give it
    # its own Adam param group at lr * std_lr_mult instead.
    std_lr_mult: float = 20.0
    max_grad_norm: float = 1.0
    # Sliding-window rollout buffer: each update collects ~rollout_steps fresh
    # transitions, appends them to a deque of this capacity, and trains on the
    # whole window — so each record is reused across roughly
    # ``buffer_capacity / rollout_steps`` updates. ``None`` keeps only the latest
    # rollout (textbook on-policy PPO). Clamped up to rollout_steps so a full
    # fresh rollout always fits.
    buffer_capacity: int | None = None
    norm_reward: bool = True
    norm_obs: bool = True
    device: str = "cpu"
    seed: int = 42
    log_every: int = 1
    # Rollout collection is the wall-clock bottleneck (the counterfactual
    # optimizer runs in env.step dominate). With n_workers > 1 the rollout is
    # split across that many worker processes, each rebuilding its own envs and
    # running whole episodes with a frozen snapshot of the current policy +
    # normalizers; the main process merges the results and runs the PPO update.
    # 1 keeps the exact single-process path. CPU rollout is assumed.
    n_workers: int = 1


@dataclass
class Transition:
    step: Step
    reward: float
    context: np.ndarray
    adv: float = 0.0
    ret: float = 0.0


@dataclass
class PPOLog:
    history: list[dict] = field(default_factory=list)
    ret_rms: "RunningMeanStd | None" = None
    obs_rms: "RunningMeanStd | None" = None


class RolloutBuffer:
    """Sliding-window store of transitions, retained across PPO updates.

    Mirrors ``DynamicAlgorithmSelection``'s ``RolloutBuffer``: rather than a
    fresh buffer per update (textbook on-policy PPO), it keeps the most recent
    ``capacity`` transitions so each record is reused over several updates. The
    clipped PPO ratio (``exp(logp - old_logp)``) tolerates the mild
    off-policyness of older records still inside the window.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: "deque[Transition]" = deque(maxlen=capacity)

    def clear(self) -> None:
        self._buf.clear()

    def extend(self, transitions: list[Transition]) -> None:
        # deque(maxlen) evicts the oldest transitions once capacity is exceeded.
        self._buf.extend(transitions)

    def as_list(self) -> list[Transition]:
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)


def _gae(rewards, values, gamma, lam):
    T = len(rewards)
    adv = [0.0] * T
    last = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0  # episodes terminate -> 0
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    returns = [adv[t] + values[t] for t in range(T)]
    return adv, returns


def explained_variance(y_pred, y_true) -> float:
    """Fraction of the return variance the value function explains:
    ``1 - Var(y_true - y_pred) / Var(y_true)``. 1.0 is a perfect critic, 0.0 is
    no better than predicting the mean return, and negative is worse than that.
    Diagnostic only (never fed back into training). Returns NaN if the targets
    are (near-)constant, since the ratio is then undefined."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    var_true = y_true.var()
    if var_true < 1e-12:
        return float("nan")
    return float(1.0 - (y_true - y_pred).var() / var_true)


def _collect_episodes(
    ac: ActorCritic,
    envs,
    cfg: PPOConfig,
    ep_rng,
    ret_rms: RunningMeanStd | None,
    obs_rms: RunningMeanStd | None,
    min_steps: int,
    device: str,
    *,
    frozen: bool,
):
    """Run whole episodes until at least ``min_steps`` transitions are gathered.

    ``frozen`` selects how the normalizers are used:

    * ``False`` (single-process path) — ``obs_rms``/``ret_rms`` are the live
      estimators and are updated inline as collection proceeds, exactly as the
      original loop did.
    * ``True`` (parallel worker path) — they are read-only snapshots: the
      context is normalized and the rewards scaled against the frozen stats, and
      the raw contexts + discounted returns are accumulated and returned so the
      main process can merge them into the global estimators once per update.

    Returns ``(transitions, ep_returns, switch_rewards, raw_ctxs, discounted)``.
    The last two are empty unless ``frozen``.
    """
    transitions: list[Transition] = []
    ep_returns: list[float] = []
    switch_rewards: dict[str, list[float]] = defaultdict(list)
    raw_ctxs: list[np.ndarray] = []
    discounted: list[float] = []
    while len(transitions) < min_steps:
        env: TranslationEnv = envs[ep_rng.integers(len(envs))]
        obs, _ = env.reset()
        ep: list[Transition] = []
        rewards, values = [], []
        done = False
        while not done:
            if not env.source_has_full_state():
                break  # degenerate source; abandon this episode
            # Critic side-input: progress scalars.
            raw_ctx = np.asarray(obs["context"], dtype=np.float32)
            if obs_rms is not None:
                if frozen:
                    raw_ctxs.append(raw_ctx)  # merged centrally after the rollout
                else:
                    obs_rms.update(raw_ctx[None])
                ctx = obs_rms.normalize(raw_ctx)
            else:
                ctx = raw_ctx
            switch = f"{obs['source_algo']}->{obs['target_algo']}"
            step = ac.act(obs, device=device, context=ctx)
            next_obs, reward, term, trunc, info = env.step(step.native)
            ep.append(Transition(step=step, reward=reward, context=ctx))
            rewards.append(reward)
            values.append(step.value)
            if env.logs_per_direction:
                switch_rewards[switch].append(reward)  # raw, per switch direction
            # For a summed (multi-mode) reward, also break out each additive
            # component so its mean is logged (reward/<mode>, e.g. reward/noswitch).
            for part, value in info.get("reward_parts", {}).items():
                switch_rewards[part].append(value)
            done = term or trunc
            obs = next_obs
        if not ep:
            continue
        ep_returns.append(float(sum(rewards)))  # report the RAW return
        if ret_rms is None:
            train_rewards = rewards
        elif frozen:
            discounted.extend(discounted_returns(rewards, cfg.gamma))
            train_rewards = scale_by_ret_std(rewards, ret_rms)
        else:
            train_rewards = normalize_rewards(rewards, cfg.gamma, ret_rms)
        adv, ret = _gae(train_rewards, values, cfg.gamma, cfg.gae_lambda)
        for tr, a, r in zip(ep, adv, ret):
            tr.adv, tr.ret = a, r
        transitions.extend(ep)
    return transitions, ep_returns, switch_rewards, raw_ctxs, discounted


def collect_rollout(
    ac: ActorCritic,
    envs,
    cfg: PPOConfig,
    ep_rng,
    ret_rms: RunningMeanStd | None = None,
    obs_rms: RunningMeanStd | None = None,
) -> tuple[list[Transition], list[float], dict[str, list[float]]]:
    """Single-process rollout: run whole episodes until at least rollout_steps
    transitions are gathered, updating the normalizers inline.

    Reward normalization (``ret_rms``) and context/observation normalization
    (``obs_rms``) use running statistics that persist across updates; the
    normalized context is stored in each transition so the PPO update is
    consistent with what the policy saw at collection time.

    The third return value maps each switch type (``"<source>-><target>"``) to
    its raw per-step rewards, for every reward mode whose baseline is comparable
    across the hand-off directions (each a log-scaled improvement over that same
    direction's own baseline: the lossy default for ``relative``, the source
    continuing for ``noswitch``, the cold target run for ``coldtarget``). A pure
    ``absolute`` reward is a raw, scale-varying improvement, so it is not broken
    out per direction (``env.logs_per_direction``). When the reward sums several
    modes (``mixed`` or an explicit list), the same dict additionally carries
    each additive component under its mode name (aggregated over all directions),
    so each part's mean is logged alongside the per-direction totals.
    """
    transitions, ep_returns, switch_rewards, _, _ = _collect_episodes(
        ac, envs, cfg, ep_rng, ret_rms, obs_rms, cfg.rollout_steps, cfg.device,
        frozen=False,
    )
    return transitions, ep_returns, switch_rewards


# --------------------------------------------------------------------------- #
# Parallel rollout: a persistent pool of worker processes, each owning its own
# envs and a copy of the policy, collecting a share of the rollout per update.
# --------------------------------------------------------------------------- #

_WORKER: dict = {}


def _worker_init(ac: ActorCritic, specs, base_seed: int, cfg: PPOConfig, counter):
    """Runs once per worker process. Builds this worker's envs (with a distinct
    seed offset so workers don't duplicate each other's episode stream) and
    stores the policy + a persistent env-selection RNG in module globals."""
    with counter.get_lock():
        wid = counter.value
        counter.value += 1
    # One optimizer run per worker at a time: let each worker use a single
    # thread so N worker processes don't oversubscribe the cores with N * (BLAS
    # threads) — that makes more workers slower, not faster.
    torch.set_num_threads(1)
    ac.to("cpu")
    envs = [TranslationEnv(**spec) for spec in _seeded_specs(specs, wid)]
    _WORKER["ac"] = ac
    _WORKER["cfg"] = cfg
    _WORKER["envs"] = envs
    _WORKER["ep_rng"] = np.random.default_rng(base_seed + 10_000 * (wid + 1))


def _seeded_specs(specs, wid: int):
    """Offset each env spec's seed by the worker id so every worker walks a
    different problem/instance/boundary stream."""
    offset = 1_000_003 * (wid + 1)
    return [{**s, "seed": int(s["seed"]) + offset} for s in specs]


def _worker_collect(payload):
    """Collect ~``n_steps`` transitions with the given policy weights, using
    frozen normalizer snapshots. Runs in a worker process."""
    state_dict, obs_rms, ret_rms, n_steps = payload
    ac: ActorCritic = _WORKER["ac"]
    ac.load_state_dict(state_dict)
    cfg: PPOConfig = _WORKER["cfg"]
    with torch.no_grad():
        trs, ep_returns, switch_rewards, raw_ctxs, disc = _collect_episodes(
            ac, _WORKER["envs"], cfg, _WORKER["ep_rng"], ret_rms, obs_rms,
            n_steps, "cpu", frozen=True,
        )
    return trs, ep_returns, dict(switch_rewards), raw_ctxs, disc


def collect_rollout_parallel(
    executor: ProcessPoolExecutor,
    ac: ActorCritic,
    cfg: PPOConfig,
    ret_rms: RunningMeanStd | None,
    obs_rms: RunningMeanStd | None,
) -> tuple[list[Transition], list[float], dict[str, list[float]]]:
    """Fan the rollout out over ``cfg.n_workers`` worker processes, each using a
    frozen snapshot of the current policy + normalizers, then merge the results
    and fold the new context/return statistics back into the live estimators."""
    n = cfg.n_workers
    per = math.ceil(cfg.rollout_steps / n)
    state_dict = {k: v.detach().cpu() for k, v in ac.state_dict().items()}
    obs_snap = copy.deepcopy(obs_rms)
    ret_snap = copy.deepcopy(ret_rms)
    payload = (state_dict, obs_snap, ret_snap, per)
    results = list(executor.map(_worker_collect, [payload] * n))

    transitions: list[Transition] = []
    ep_returns: list[float] = []
    switch_rewards: dict[str, list[float]] = defaultdict(list)
    all_ctxs: list[np.ndarray] = []
    all_disc: list[float] = []
    for trs, eprs, sw, ctxs, disc in results:
        transitions.extend(trs)
        ep_returns.extend(eprs)
        for k, v in sw.items():
            switch_rewards[k].extend(v)
        all_ctxs.extend(ctxs)
        all_disc.extend(disc)
    # Merge the rollout's new statistics once (workers left the snapshots frozen).
    if obs_rms is not None and all_ctxs:
        obs_rms.update(np.asarray(all_ctxs, dtype=np.float64))
    if ret_rms is not None and all_disc:
        ret_rms.update(np.asarray(all_disc, dtype=np.float64)[:, None])
    return transitions, ep_returns, switch_rewards


def _group_key(tr: Transition):
    s = tr.step.src_state
    return (tr.step.target_algo, s.d, s.n)


def update(ac: ActorCritic, opt, transitions: list[Transition], cfg: PPOConfig) -> dict:
    device = torch.device(cfg.device)
    # Normalize advantages into a local array indexed by position — never mutate
    # Transition.adv, since records persist across updates (sliding-window
    # buffer) and the stored raw GAE must survive for the next update.
    advs = torch.tensor([t.adv for t in transitions], dtype=torch.float32)
    norm_advs = ((advs - advs.mean()) / (advs.std() + 1e-8)).tolist()

    # Critic diagnostic: how much of the return variance the value function
    # explains, using the collection-time value predictions against the GAE
    # return targets. ~0 means the critic predicts little beyond the mean (so
    # the advantages are mostly noise); computed once over the whole window.
    ev = explained_variance(
        [t.step.value for t in transitions], [t.ret for t in transitions]
    )

    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, tr in enumerate(transitions):
        groups[_group_key(tr)].append(i)

    gen = torch.Generator().manual_seed(cfg.seed)
    metrics = defaultdict(float)
    policy_stds: list[float] = []
    # Effective (population-conditioned) exploration std, aggregated over the
    # update's minibatches — this is what actually varies now that the std is
    # state-dependent, so it, not just the per-segment base, shows adaptivity.
    eff_std = {"sum": 0.0, "min": float("inf"), "max": float("-inf"), "n": 0}
    n_batches = 0

    for _ in range(cfg.ppo_epochs):
        for (target_algo, _, _), idxs in groups.items():
            order = torch.randperm(len(idxs), generator=gen).tolist()
            idxs = [idxs[o] for o in order]
            for start in range(0, len(idxs), cfg.minibatch_size):
                mb = idxs[start : start + cfg.minibatch_size]
                states = [transitions[i].step.src_state.to(device) for i in mb]
                source_algo = states[0].algo  # constant within a target-keyed group
                actions = torch.stack([transitions[i].step.action for i in mb]).to(
                    device
                )
                contexts = torch.tensor(
                    np.stack([transitions[i].context for i in mb]),
                    dtype=torch.float32,
                    device=device,
                )
                old_logp = torch.tensor(
                    [transitions[i].step.log_prob for i in mb],
                    dtype=torch.float32,
                    device=device,
                )
                old_value = torch.tensor(
                    [transitions[i].step.value for i in mb],
                    dtype=torch.float32,
                    device=device,
                )
                adv = torch.tensor(
                    [norm_advs[i] for i in mb], dtype=torch.float32, device=device
                )
                ret = torch.tensor(
                    [transitions[i].ret for i in mb], dtype=torch.float32, device=device
                )

                logp, entropy, value, batch, ctx, std = ac.evaluate_actions(
                    states, source_algo, target_algo, actions, contexts
                )
                with torch.no_grad():
                    eff_std["sum"] += float(std.mean())
                    eff_std["min"] = min(eff_std["min"], float(std.min()))
                    eff_std["max"] = max(eff_std["max"], float(std.max()))
                    eff_std["n"] += 1
                ratio = (logp - old_logp).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                # PPO value-clipping + Huber: bounds the value gradient so a
                # critic spike can't dominate the (globally clipped) gradient.
                # The clip window is cfg.vf_clip (a *separate* range from the
                # policy ratio clip): reusing the 0.2 ratio clip here throttles
                # the critic hard when returns span many units (worst with
                # --no-norm-reward), since once the prediction is that far from
                # old_value the pessimistic max() zeroes its gradient. vf_clip
                # <= 0 (or None) disables value clipping entirely.
                vclip = cfg.vf_clip
                if vclip is not None and vclip > 0:
                    v_clipped = old_value + (value - old_value).clamp(-vclip, vclip)
                    vl_unclipped = F.smooth_l1_loss(value, ret, reduction="none")
                    vl_clipped = F.smooth_l1_loss(v_clipped, ret, reduction="none")
                    value_loss = torch.max(vl_unclipped, vl_clipped).mean()
                else:
                    value_loss = F.smooth_l1_loss(value, ret)
                ent = entropy.mean()
                # Skip the A->B->A round trip entirely when it wouldn't affect
                # the loss anyway, rather than computing it just to multiply by 0.
                if cfg.lambda_cycle == 0.0:
                    cycle = value.new_zeros(())
                else:
                    cycle = ac.cycle_drift(batch, ctx, cycle_mode=cfg.cycle_mode)

                loss = (
                    policy_loss
                    + cfg.vf_coef * value_loss
                    - cfg.ent_coef * ent
                    + cfg.lambda_cycle * cycle
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
                opt.step()

                metrics["policy_loss"] += float(policy_loss.detach())
                metrics["value_loss"] += float(value_loss.detach())
                metrics["cycle"] += float(cycle.detach())
                n_batches += 1

    n_batches = max(n_batches, 1)
    out = {k: v / n_batches for k, v in metrics.items()}
    out["explained_variance"] = ev
    # Log the exploration scale (policy std) rather than the summed differential
    # entropy: std is sign-stable and independent of the action dimension, which
    # varies across (PSO/CMA-ES)-target groups. The summary spread ranges over
    # the per-segment stds (ActorCritic.log_std) at the end of the update — the
    # segments are what can actually diverge from one another; each also gets its
    # own named series so the individual field kinds are visible.
    with torch.no_grad():
        for algo, p in ac.log_std.items():
            seg_std = p.exp()
            for name, s in zip(
                ac.translator.decoders[algo].segment_names(), seg_std.tolist()
            ):
                out[f"policy_std/{algo}.{name}"] = s
            policy_stds.extend(seg_std.tolist())
    stds = np.asarray(policy_stds) if policy_stds else np.zeros(1)
    out["policy_std"] = float(stds.mean())
    out["policy_std_median"] = float(np.median(stds))
    out["policy_std_min"] = float(stds.min())
    out["policy_std_max"] = float(stds.max())
    # Effective per-coordinate std (base + population-conditioned residual): its
    # min/max spread over the update is the visible signature of the adaptivity.
    if eff_std["n"]:
        out["policy_std_eff"] = eff_std["sum"] / eff_std["n"]
        out["policy_std_eff_min"] = eff_std["min"]
        out["policy_std_eff_max"] = eff_std["max"]
    return out


def train_ppo(ac: ActorCritic, envs, cfg: PPOConfig, log_fn=None) -> PPOLog:
    """Run PPO. ``log_fn(metrics)`` is called once per update (for W&B; optional).

    The reward/observation normalizers (running statistics) are attached to the
    returned log as ``log.ret_rms`` / ``log.obs_rms`` so they can be checkpointed.
    """
    ac.to(cfg.device)
    other_params = [
        p for n, p in ac.named_parameters() if not n.startswith("log_std.")
    ]
    opt = torch.optim.Adam(
        [
            {"params": other_params},
            {"params": list(ac.log_std.values()), "lr": cfg.lr * cfg.std_lr_mult},
        ],
        lr=cfg.lr,
    )
    ep_rng = np.random.default_rng(cfg.seed)
    log = PPOLog()
    ret_rms = RunningMeanStd(()) if cfg.norm_reward else None
    obs_rms = RunningMeanStd((ac.context_dim,)) if cfg.norm_obs else None
    log.ret_rms, log.obs_rms = ret_rms, obs_rms

    # Persistent sliding-window buffer: each record lives for several updates.
    # Clamp capacity up so at least one full fresh rollout always fits.
    capacity = max(cfg.buffer_capacity or cfg.rollout_steps, cfg.rollout_steps)
    buffer = RolloutBuffer(capacity)

    # Optional parallel rollout: a persistent pool built once and reused across
    # updates, so each worker's envs (and their problem walk) stay alive.
    executor = _make_executor(ac, envs, cfg) if cfg.n_workers > 1 else None
    try:
        _train_loop(ac, envs, cfg, opt, ep_rng, ret_rms, obs_rms, buffer, log,
                    log_fn, executor)
    finally:
        if executor is not None:
            executor.shutdown()
    return log


def _make_executor(ac: ActorCritic, envs, cfg: PPOConfig) -> ProcessPoolExecutor:
    """Build the persistent worker pool. ``spawn`` is used so workers start from
    a clean interpreter (the live ioh problems in ``envs`` are not fork-safe to
    reuse); each worker rebuilds its own envs from the picklable specs."""
    # Children inherit these at spawn and import numpy fresh, so their BLAS
    # (the CMA-ES eigendecomposition) stays single-threaded — see _worker_init.
    # setdefault leaves any explicit user choice untouched; the already-imported
    # parent BLAS is unaffected.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    mp_ctx = mp.get_context("spawn")
    counter = mp_ctx.Value("i", 0)
    specs = [e.spec() for e in envs]
    template = copy.deepcopy(ac).cpu()  # pickled to each worker once, at startup
    return ProcessPoolExecutor(
        max_workers=cfg.n_workers,
        mp_context=mp_ctx,
        initializer=_worker_init,
        initargs=(template, specs, cfg.seed, cfg, counter),
    )


def _train_loop(ac, envs, cfg, opt, ep_rng, ret_rms, obs_rms, buffer, log,
                log_fn, executor):
    pbar = tqdm(range(cfg.updates), desc="training", unit="upd", dynamic_ncols=True)
    for u in pbar:
        if executor is not None:
            fresh, ep_returns, switch_rewards = collect_rollout_parallel(
                executor, ac, cfg, ret_rms, obs_rms
            )
        else:
            fresh, ep_returns, switch_rewards = collect_rollout(
                ac, envs, cfg, ep_rng, ret_rms, obs_rms
            )
        buffer.extend(fresh)
        m = update(ac, opt, buffer.as_list(), cfg)
        m["update"] = u
        # Report the raw return of the freshly collected episodes only, so the
        # learning curve reflects the current policy (not stale window records).
        m["mean_return"] = float(np.mean(ep_returns)) if ep_returns else 0.0
        # Per-switch-direction mean reward (comparable baselines; empty for a
        # pure 'absolute' reward), plus each summed component's mean by mode name.
        for switch, rs in switch_rewards.items():
            m[f"reward/{switch}"] = float(np.mean(rs))
        m["n_transitions"] = len(buffer)
        m["n_fresh"] = len(fresh)
        log.history.append(m)
        if log_fn is not None:
            log_fn(m)
        pbar.set_postfix(
            ret=f"{m['mean_return']:.4f}",
            ev=f"{m['explained_variance']:.3f}",
            std=f"{m['policy_std']:.4f}",
            refresh=False,
        )
        if cfg.log_every and u % cfg.log_every == 0:
            # tqdm.write keeps the per-update log line from corrupting the bar.
            tqdm.write(
                f"update {u:3d} | return {m['mean_return']:.4f} "
                f"| pi {m['policy_loss']:.4f} | vf {m['value_loss']:.4f} "
                f"| ev {m['explained_variance']:.3f} "
                f"| std {m['policy_std']:.4f} | cycle {m['cycle']:.4f}"
            )
    pbar.close()
    return log
