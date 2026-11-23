"""The PPO update: clipped-surrogate policy loss + (optionally clipped) value
loss, over shape-homogeneous minibatches grouped by ``(target_algo, D, N)``."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from algorithm_translation.rl.policy import ActorCritic
from algorithm_translation.rl.ppo.config import PPOConfig, Transition
from algorithm_translation.seeding import Stream, derive_seed


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


def _batch_inputs(transitions, mb, device):
    states = [transitions[i].step.src_state.to(device) for i in mb]
    contexts = torch.tensor(
        np.stack([transitions[i].context for i in mb]),
        dtype=torch.float32,
        device=device,
    )
    return states, states[0].algo, contexts  # source constant within a group


def _refresh_targets(ac, transitions, groups, cfg, ret_rms, device):
    """Value targets and advantages under the *current* critic and reward scale.

    Records live in the sliding-window buffer for several updates; the
    advantage stored at collection (``r - V_old(s)``, with ``r`` scaled by the
    return std of that moment) goes stale as the critic and ``ret_rms`` move.
    Every episode is a single terminal step (enforced in the rollout), so the
    GAE return is just the scaled reward and the advantage is
    ``r_scaled - V_now(s)`` — cheap to recompute for the whole window each
    update. Returns ``(values, returns, advantages)`` as float32 tensors indexed
    like ``transitions``."""
    scale = float(ret_rms.std.item()) + 1e-8 if ret_rms is not None else 1.0
    rets = torch.tensor([t.reward / scale for t in transitions], dtype=torch.float32)
    values = torch.zeros(len(transitions), dtype=torch.float32)
    with torch.no_grad():
        for idxs in groups.values():
            for start in range(0, len(idxs), cfg.minibatch_size):
                mb = idxs[start : start + cfg.minibatch_size]
                states, source_algo, contexts = _batch_inputs(transitions, mb, device)
                values[mb] = ac.values(states, source_algo, contexts).float().cpu()
    return values, rets, rets - values


def update(
    ac: ActorCritic,
    opt,
    transitions: list[Transition],
    cfg: PPOConfig,
    ret_rms=None,
    update_idx: int = 0,
) -> dict:
    """One PPO update over the buffered ``transitions``.

    ``ret_rms`` is the live return-std estimator (None without reward
    normalization), used to rescale every buffered reward consistently.
    ``update_idx`` varies the minibatch shuffle from update to update."""
    device = torch.device(cfg.device)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, tr in enumerate(transitions):
        groups[_group_key(tr)].append(i)

    values_now, rets_all, advs = _refresh_targets(
        ac, transitions, groups, cfg, ret_rms, device
    )
    norm_advs = ((advs - advs.mean()) / (advs.std() + 1e-8)).tolist()
    rets_list = rets_all.tolist()
    values_list = values_now.tolist()

    # Critic diagnostic: how much of the return variance the value function
    # explains, using the update-start value predictions against the return
    # targets. ~0 means the critic predicts little beyond the mean (so the
    # advantages are mostly noise); computed once over the whole window.
    ev = explained_variance(values_list, rets_list)

    # A fresh shuffle each update (seeding with cfg.seed alone would replay the
    # same minibatch permutation every update).
    gen = torch.Generator().manual_seed(derive_seed(cfg.seed, update_idx, Stream.MINIBATCH))
    metrics = defaultdict(float)
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
                states, source_algo, contexts = _batch_inputs(transitions, mb, device)
                actions = torch.stack([transitions[i].step.action for i in mb]).to(
                    device
                )
                old_logp = torch.tensor(
                    [transitions[i].step.log_prob for i in mb],
                    dtype=torch.float32,
                    device=device,
                )
                old_value = torch.tensor(
                    [values_list[i] for i in mb], dtype=torch.float32, device=device
                )
                adv = torch.tensor(
                    [norm_advs[i] for i in mb], dtype=torch.float32, device=device
                )
                ret = torch.tensor(
                    [rets_list[i] for i in mb], dtype=torch.float32, device=device
                )

                logp, entropy, value, std = ac.evaluate_actions(
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
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
                opt.step()

                metrics["policy_loss"] += float(policy_loss.detach())
                metrics["value_loss"] += float(value_loss.detach())
                n_batches += 1

    n_batches = max(n_batches, 1)
    out = {k: v / n_batches for k, v in metrics.items()}
    out["explained_variance"] = ev
    # Log the exploration scale (std) rather than the summed differential
    # entropy: std is sign-stable and independent of the action dimension.
    out.update(ac.std_summary())
    # Effective per-coordinate std (base + population-conditioned residual): its
    # min/max spread over the update is the visible signature of the adaptivity.
    if eff_std["n"]:
        out["policy_std_eff"] = eff_std["sum"] / eff_std["n"]
        out["policy_std_eff_min"] = eff_std["min"]
        out["policy_std_eff_max"] = eff_std["max"]
    return out
