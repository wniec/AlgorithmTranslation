"""The GRPO update: PPO's clipped surrogate with group-relative advantages and
no critic, over shape-homogeneous minibatches grouped by ``(target_algo, D, N)``."""

from __future__ import annotations

from collections import defaultdict

import torch

from algorithm_translation.rl.grpo.config import GRPOConfig, Sample
from algorithm_translation.rl.policy import ActorCritic
from algorithm_translation.seeding import Stream, derive_seed


def _shape_key(s: Sample):
    st = s.step.src_state
    return (s.step.target_algo, st.d, st.n)


def update(
    ac: ActorCritic, opt, samples: list[Sample], cfg: GRPOConfig, update_idx: int = 0
) -> dict:
    device = torch.device(cfg.device)
    shape_groups: dict[tuple, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        shape_groups[_shape_key(s)].append(i)

    gen = torch.Generator().manual_seed(derive_seed(cfg.seed, update_idx, Stream.MINIBATCH))
    params = ac.actor_parameters()
    sums = defaultdict(float)
    n_batches = 0
    for _ in range(cfg.epochs):
        for (target_algo, _, _), idxs in shape_groups.items():
            order = torch.randperm(len(idxs), generator=gen).tolist()
            idxs = [idxs[o] for o in order]
            for start in range(0, len(idxs), cfg.minibatch_size):
                mb = idxs[start : start + cfg.minibatch_size]
                states = [samples[i].step.src_state.to(device) for i in mb]
                actions = torch.stack([samples[i].step.action for i in mb]).to(device)
                old_logp = torch.tensor(
                    [samples[i].step.log_prob for i in mb], dtype=torch.float32, device=device
                )
                adv = torch.tensor(
                    [samples[i].adv for i in mb], dtype=torch.float32, device=device
                )
                logp, entropy, _ = ac.log_prob_entropy(states, target_algo, actions)
                log_ratio = logp - old_logp
                ratio = log_ratio.exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                ent = entropy.mean()
                loss = policy_loss - cfg.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    sums["policy_loss"] += float(policy_loss)
                    sums["entropy"] += float(ent)
                    # k3 estimator of KL(old || new), and the clipped fraction
                    sums["approx_kl"] += float(((ratio - 1) - log_ratio).mean())
                    sums["clip_frac"] += float(
                        ((ratio - 1).abs() > cfg.clip).float().mean()
                    )
                n_batches += 1

    out = {k: v / max(n_batches, 1) for k, v in sums.items()}
    out.update(ac.std_summary())
    return out
