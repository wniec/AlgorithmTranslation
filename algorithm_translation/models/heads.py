"""Per-algorithm decoder head.

Consumes the shared latent ``z`` and the shared population being handed over
(positions + normalization context) and emits that algorithm's specific fields in raw units:

* PER_DIM (e.g. BOBYQA's model gradient): linear read-out per dimension-token.
* PER_PARTICLE_PER_DIM (velocity, personal-best positions): a per-cell MLP whose
  query for particle *i*, dimension *j* is ``concat(dim_token_j, normalized
  position x_ij)`` — so generated velocities are grounded in the actual swarm
  and the head stays equivariant over both particles and coordinates.
* MATRIX (covariance): ``CovFactorHead`` -> PSD by construction.
* SCALAR (e.g. BOBYQA's trust-region radius): linear read-out from the global
  vector, decoded in log-space
  (always positive after denormalization).

Targets with a computed prior (``algorithm_translation/models/prior``: CMA-ES's
search distribution, BOBYQA's quadratic model) are the exception: the state is
computed from the handed-over points and the head only predicts a residual
around it — a per-coordinate shift (D), a low-rank matrix factor (D x rank) and
a few global log-scale shifts. The action is therefore
``(1 + rank) * D + n_scalar`` numbers and a zero action reproduces the computed
state exactly.

The head also exposes an **action interface** for the RL trainer:
``action_mean`` returns the flat normalized coordinates an RL policy puts a
Gaussian over, and ``build_state`` rebuilds the raw fields from (possibly
noised) coordinates. The deterministic ``forward`` is exactly
``build_state(action_mean(...))``, so both paths share one network.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from algorithm_translation.models.encoder import Latent
from algorithm_translation.models.layers import MLP, CovFactorHead
from algorithm_translation.models.norm import POSITION_LIKE, NormContext
from algorithm_translation.models.prior import PRIORS
from algorithm_translation.optimizers.DE import ARATE
from algorithm_translation.state.canonical import CanonicalState
from algorithm_translation.state.schema import Struct, get_spec


class StateDecoder(nn.Module):
    def __init__(self, algo: str, hidden: int = 64, cov_rank: int = 4):
        super().__init__()
        self.algo = algo
        self.hidden = hidden
        self.cov_rank = cov_rank
        spec = get_spec(algo)
        self.spec = spec

        self.ppd_fields = [f for f in spec.specific if f.struct is Struct.PER_PARTICLE_PER_DIM]
        self.pd_fields = [f for f in spec.specific if f.struct is Struct.PER_DIM]
        self.scalar_fields = [f for f in spec.specific if f.struct is Struct.SCALAR]
        self.set_fields = [f for f in spec.specific if f.struct is Struct.POINT_SET]
        self.cov_field = next((f for f in spec.specific if f.struct is Struct.MATRIX), None)

        self.prior = PRIORS.get(algo)
        if self.prior is not None:
            self.pd_head = nn.Linear(hidden, self.prior.n_shift)  # per-coordinate shifts
            self.scalar_head = nn.Linear(hidden, self.prior.n_scalar)
            if self._uses_factor:
                self.factor_head = nn.Linear(hidden, cov_rank)
            self._small_init()
            return

        if self.pd_fields:
            self.pd_head = nn.Linear(hidden, len(self.pd_fields))
        if self.ppd_fields:
            # input: dim-token (H) + normalized position scalar (1)
            self.ppd_head = MLP([hidden + 1, hidden, len(self.ppd_fields)])
        if self.set_fields:
            # input: dim-token (H) + the slot's seed-position coord (1)
            self.set_head = MLP([hidden + 1, hidden, len(self.set_fields)])
        if self.scalar_fields:
            self.scalar_head = nn.Linear(hidden, len(self.scalar_fields))
        if self.cov_field is not None:
            self.cov_head = CovFactorHead(hidden, rank=cov_rank)

        self._small_init()

    @property
    def _uses_factor(self) -> bool:
        """Whether the prior's action carries a covariance factor block."""
        return self.prior is not None and self.prior.uses_factor and self.cov_rank > 0

    @staticmethod
    def _m_cap(n: int) -> int:
        return max(1, int(round(ARATE * n)))

    def _small_init(self):
        """Start near zero so initial decoded fields (in normalized space) are
        small — keeps the first training steps stable."""
        for head in ("pd_head", "ppd_head", "set_head", "scalar_head", "factor_head"):
            mod = getattr(self, head, None)
            if mod is None:
                continue
            last = mod if isinstance(mod, nn.Linear) else mod.net[-1]
            nn.init.zeros_(last.bias)
            last.weight.data.mul_(0.01)

    # ------------------------------------------------------------------ #
    # action coordinate layout                                            #
    # ------------------------------------------------------------------ #

    def _layout(self, D: int, N: int) -> list[tuple[str, tuple[int, ...]]]:
        """Ordered (segment-name, per-sample shape) of the flat action vector."""
        segs: list[tuple[str, tuple[int, ...]]] = []
        if self.prior is not None:
            segs.append(("pd", (D, self.prior.n_shift)))
            if self._uses_factor:
                segs.append(("cov_L", (D, self.cov_rank)))
            segs.append(("scalar", (self.prior.n_scalar,)))
            return segs
        if self.pd_fields:
            segs.append(("pd", (D, len(self.pd_fields))))
        if self.ppd_fields:
            segs.append(("ppd", (N, D, len(self.ppd_fields))))
        if self.set_fields:
            segs.append(("set", (self._m_cap(N), D, len(self.set_fields))))
        if self.cov_field is not None:
            segs.append(("cov_L", (D, self.cov_rank)))
            segs.append(("cov_d", (D,)))
        if self.scalar_fields:
            segs.append(("scalar", (len(self.scalar_fields),)))
        return segs

    def action_dim(self, D: int, N: int) -> int:
        total = 0
        for _, shape in self._layout(D, N):
            n = 1
            for s in shape:
                n *= s
            total += n
        return total

    def segment_names(self) -> list[str]:
        """Ordered segment names of the flat action. Which segments exist depends
        only on the algorithm's schema — D and N change their *sizes*, not their
        presence — so this is a fixed, small per-algorithm list. The RL policy
        uses it to give each field kind its own exploration std (see
        ``algorithm_translation.rl.policy.ActorCritic.log_std``)."""
        return [name for name, _ in self._layout(1, 1)]

    def segment_sizes(self, D: int, N: int) -> list[int]:
        """Element count of each segment, in ``segment_names()`` order (they sum
        to ``action_dim(D, N)``)."""
        sizes = []
        for _, shape in self._layout(D, N):
            n = 1
            for s in shape:
                n *= s
            sizes.append(n)
        return sizes

    def action_mean(self, z: Latent, positions: Tensor, ctx: NormContext) -> Tensor:
        """Flat normalized action-mean coordinates, shape (B, action_dim)."""
        tokens = z.dim_tokens  # (B, D, H)
        B, D, _ = tokens.shape
        parts: list[Tensor] = []

        if self.prior is not None:
            parts.append(self.pd_head(tokens).reshape(B, -1))  # (B, D*n_shift)
            if self._uses_factor:
                parts.append(self.factor_head(tokens).reshape(B, -1))  # (B, D*rank)
            parts.append(self.scalar_head(z.global_vec))  # (B, n_scalar)
            return torch.cat(parts, dim=-1)

        if self.pd_fields:
            parts.append(self.pd_head(tokens).reshape(B, -1))  # (B, D*n_pd)

        if self.ppd_fields:
            N = positions.shape[1]
            pos_n = ctx.normalize(positions, Struct.PER_PARTICLE_PER_DIM, True)
            tok = tokens.unsqueeze(1).expand(B, N, D, self.hidden)
            cell_in = torch.cat([tok, pos_n.unsqueeze(-1)], dim=-1)
            parts.append(self.ppd_head(cell_in).reshape(B, -1))  # (B, N*D*n_ppd)

        if self.set_fields:
            N = positions.shape[1]
            M = self._m_cap(N)
            seed = self._seed_positions(positions, M)  # (B, M, D)
            seed_n = ctx.normalize(seed, Struct.POINT_SET, True)
            tok = z.dim_tokens.unsqueeze(1).expand(B, M, D, self.hidden)
            cell_in = torch.cat([tok, seed_n.unsqueeze(-1)], dim=-1)
            parts.append(self.set_head(cell_in).reshape(B, -1))  # (B, M*D*n_set)

        if self.cov_field is not None:
            L, d_raw = self.cov_head.factor(tokens)
            parts.append(L.reshape(B, -1))  # (B, D*rank)
            parts.append(d_raw.reshape(B, -1))  # (B, D)

        if self.scalar_fields:
            parts.append(self.scalar_head(z.global_vec))  # (B, n_scalar)

        return torch.cat(parts, dim=-1)

    @staticmethod
    def _seed_positions(positions: Tensor, m: int) -> Tensor:
        """Seed ``m`` point-set slots deterministically by cycling the population
        (slot i <- population[i mod N]); keeps archive decode reproducible."""
        n = positions.shape[1]
        idx = torch.arange(m, device=positions.device) % n
        return positions[:, idx, :]

    def build_state(
        self, coords: Tensor, ctx: NormContext, shared: CanonicalState
    ) -> dict[str, Tensor]:
        """Rebuild raw specific fields from flat (possibly noised) action coords.
        ``shared`` is the source state the action was decoded for."""
        B = coords.shape[0]
        D = ctx.center.shape[-1]
        n = shared.n
        if self.prior is not None:
            k = self.prior.n_shift
            shift = coords[:, : D * k]
            shift = shift if k == 1 else shift.reshape(B, D, k)
            factor_end = D * k + (D * self.cov_rank if self._uses_factor else 0)
            factor = (
                coords[:, D * k : factor_end].reshape(B, D, self.cov_rank)
                if self._uses_factor
                else None
            )
            return self.prior.build(shared, ctx, shift, factor, coords[:, factor_end:])
        out: dict[str, Tensor] = {}
        cov_L: Tensor | None = None
        off = 0

        for name, shape in self._layout(D, n):
            size = 1
            for s in shape:
                size *= s
            chunk = coords[:, off : off + size].reshape(B, *shape)
            off += size

            if name == "pd":
                for i, f in enumerate(self.pd_fields):
                    out[f.name] = ctx.denormalize_field(chunk[..., i], f)
            elif name == "ppd":
                for i, f in enumerate(self.ppd_fields):
                    out[f.name] = ctx.denormalize_field(chunk[..., i], f)
            elif name == "cov_L":
                cov_L = chunk  # assembled together with the diagonal below
            elif name == "cov_d":
                cov_n = self.cov_head.assemble(cov_L, chunk)
                out[self.cov_field.name] = ctx.denormalize_field(cov_n, self.cov_field)
            elif name == "set":
                for i, f in enumerate(self.set_fields):
                    out[f.name] = ctx.denormalize(
                        chunk[..., i], Struct.POINT_SET, f.name in POSITION_LIKE
                    )
            elif name == "scalar":
                for i, f in enumerate(self.scalar_fields):
                    out[f.name] = ctx.denormalize(chunk[:, i], Struct.SCALAR, False)

        return out

    # ------------------------------------------------------------------ #

    def forward(self, z: Latent, shared: CanonicalState, ctx: NormContext) -> dict[str, Tensor]:
        coords = self.action_mean(z, shared.positions, ctx)
        return self.build_state(coords, ctx, shared)
