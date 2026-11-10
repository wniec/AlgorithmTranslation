"""Portfolio of available sub-optimizers.

Add new algorithm classes here. To be selectable in the trainers and
evaluators (``uv run train`` / ``train-td3`` / ``eval`` / ``eval-switch``), an
optimizer also needs a ``StateSpec`` in ``algorithm_translation/state/schema.py``;
the CLIs only offer optimizers that have one (see ``cli.ALGO_CHOICES``). IPSO is
kept for standalone use only: it restarts from a single particle on warm start
and grows its swarm, so it has no fixed-population state to translate.
"""

from algorithm_translation.optimizers.DE import MADDE
from algorithm_translation.optimizers.ES import CMAES
from algorithm_translation.optimizers.PSO import CPSO, IPSO, SPSO, SPSOL
from algorithm_translation.optimizers.TR import BOBYQA

PORTFOLIO: dict = {
    "SPSO": SPSO,
    "SPSOL": SPSOL,
    "IPSO": IPSO,
    "CPSO": CPSO,
    "PSO": SPSO,  # convenient alias: the standard global-best PSO
    "CMAES": CMAES,
    "MADDE": MADDE,
    "BOBYQA": BOBYQA,
}


def get_portfolio(names: list[str]) -> list:
    """Return optimizer classes for the given names.

    Raises ValueError for any unknown name so errors surface early.
    """
    unknown = [n for n in names if n not in PORTFOLIO]
    if unknown:
        raise ValueError(
            f"Unknown optimizer(s): {unknown}. Available: {list(PORTFOLIO)}"
        )
    return [PORTFOLIO[n] for n in names]
