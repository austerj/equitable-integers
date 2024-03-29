class SolverError(Exception):
    """Error raised by solver."""

    ...


class BudgetError(SolverError):
    """Error related to the budget provided to the solver."""

    ...


class InsufficientBudgetError(BudgetError):
    """Error from insufficient budget."""

    ...


class ExcessBudgetError(BudgetError):
    """Error from excess budget."""


class ConstraintError(BudgetError):
    """Error from invalid constraints."""
