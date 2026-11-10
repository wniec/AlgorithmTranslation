"""Running mean/std estimators for reward and observation normalization.

``RunningMeanStd`` uses Welford/parallel moments so statistics accumulate stably
across the whole training run (not just one rollout). Two uses, shared by both
the PPO (``algorithm_translation/rl/ppo/``) and TD3 (``algorithm_translation/rl/td3/``) trainers:

* **reward normalization** — rewards are divided by the running std of the
  *discounted return* (the standard PPO/`VecNormalize` trick), which keeps the
  value targets (and, for PPO, advantages) at ~unit scale even though the
  log-scaled, optimum-relative reward can vary a lot across problems.
* **observation normalization** — the scalar context vector fed to the critic /
  Q-network is standardized by its running mean/std. (The structured optimizer
  state is already per-sample normalized inside the network via ``NormContext``.)
"""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        """Update from a batch ``x`` of shape ``(n, *shape)``."""
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def normalize(
        self, x: np.ndarray, center: bool = True, eps: float = 1e-8
    ) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = (x - self.mean) if center else x
        return (out / (self.std + eps)).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.var = np.asarray(d["var"], dtype=np.float64)
        self.count = float(d["count"])


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """The running discounted return ``R_t = gamma * R_{t-1} + r_t`` over one
    episode — the quantity whose std drives reward normalization."""
    R = 0.0
    out = []
    for r in rewards:
        R = gamma * R + r
        out.append(R)
    return out


def normalize_rewards(
    rewards: list[float], gamma: float, ret_rms: RunningMeanStd
) -> list[float]:
    """Scale one episode's rewards by the running std of the discounted return
    (the standard PPO/``VecNormalize`` trick; see module docstring). Divides by
    std only — no mean-centering, since the raw reward's sign/magnitude carries
    real signal (e.g. the default ``noswitch`` env reward is one-sided, always
    <= 0) that a value/Q function should learn to predict, not have subtracted
    away."""
    discounted = discounted_returns(rewards, gamma)
    ret_rms.update(np.asarray(discounted)[:, None])
    return scale_by_ret_std(rewards, ret_rms)


def scale_by_ret_std(rewards: list[float], ret_rms: RunningMeanStd) -> list[float]:
    """Divide rewards by the current running return std *without* updating the
    estimator — used by parallel rollout workers, which share a frozen snapshot
    of ``ret_rms`` for the whole update and let the main process merge the new
    discounted returns once (see ``algorithm_translation.rl.ppo``)."""
    std = float(ret_rms.std.item()) + 1e-8
    return [r / std for r in rewards]
