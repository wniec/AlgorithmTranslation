"""Portfolio of available sub-optimizers.

Add new algorithm classes here and they become selectable by name in train_ppo.py
/ evaluate.py. A new optimizer also needs a ``StateSpec`` in
``algorithm_translation/state/schema.py`` before it can participate in translation.
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
