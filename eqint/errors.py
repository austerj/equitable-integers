from abc import ABC


class SolverError(Exception, ABC):
    """Error raised by solver."""

    ...


class BudgetError(SolverError):
    """Error related to the budget provided to the solver."""

    ...
