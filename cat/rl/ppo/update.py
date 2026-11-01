"""The PPO update: clipped-surrogate policy loss + (optionally clipped) value
loss + the differentiable cycle-consistency penalty, over shape-homogeneous
minibatches grouped by ``(target_algo, D, N)``."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from cat.rl.policy import ActorCritic
from cat.rl.ppo.config import PPOConfig, Transition


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
                ac.translator.decoders[algo].segment_names(), seg_std.tolist(), strict=True
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
