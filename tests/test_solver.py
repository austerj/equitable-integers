import typing

import numpy as np
import pytest
from scipy.optimize import LinearConstraint, minimize

from eqint import errors
from eqint.solver import Bounds, EquitableBudgetAllocator, solve


def scipy_solve(bounds: Bounds, budget: int) -> list[float]:
    """Solve using trust-region constrained method."""
    # define linear constraints
    lb = [b[0] if b[0] is not None else -np.inf for b in bounds] + [budget]
    ub = [b[1] if b[1] is not None else np.inf for b in bounds] + [budget]
    A = np.concatenate([np.eye(len(bounds)), np.ones((1, len(bounds)))])
    constraints = LinearConstraint(A, lb, ub)  # type: ignore
    # define optimization problem
    mean = budget / len(bounds)
    x0 = [mean for _ in range(len(bounds))]  # initial guess of mean
    fun = lambda x: sum([(x_n - mean) ** 2 for x_n in x])  # minimize squared residuals from mean
    return minimize(fun, x0=x0, constraints=constraints, method="trust-constr").x.tolist()


def solution_correct(bounds: Bounds, budget: int) -> bool:
    """Check that a solution is optimal."""
    # solve for allocations
    allocations = solve(bounds, budget)
    return all(
        (
            # all bounds are adhered to
            constraints_met(bounds, allocations),
            # the full budget is allocated
            allocation_full(allocations, budget),
            # integers are optimally distributed
            integers_optimal(bounds, allocations),
        )
    )


def constraints_met(bounds: Bounds, allocations: tuple[int, ...]) -> bool:
    """Check that the full budget is allocated."""
    return all(
        # lower bounds are adhered to
        (a >= b[0] if b[0] is not None else True)
        # upper bounds are adhered to
        and (a <= b[1] if b[1] is not None else True)
        for a, b in zip(allocations, bounds)
    )


def allocation_full(allocations: tuple[int, ...], budget: int) -> bool:
    """Check that the full budget is allocated."""
    return sum(allocations) == budget


def integers_optimal(bounds: Bounds, allocations: tuple[int, ...]) -> bool:
    """Check that no integer value can be moved to produce a more equitable allocation."""
    # no integer can be reallocated in a way that would lead to a more equitable allocation; i.e.
    # the difference between the largest value that is not lower-bounded and the smallest value that
    # is not upper-bounded must be at-most 1
    non_lower_bounded = tuple(a for a, b in zip(allocations, bounds) if b[0] is None or (b[0] is not None and a > b[0]))
    non_upper_bounded = tuple(a for a, b in zip(allocations, bounds) if b[1] is None or (b[1] is not None and a < b[1]))
    if non_lower_bounded and non_upper_bounded:
        upper = max(non_lower_bounded)
        lower = min(non_upper_bounded)
        return abs(upper - lower) <= 1
    # if all values are either lower-bounded or upper-bounded, there is nothing to check as no
    # integers can be reallocated
    return True


# test the functions used for testing
def test_allocation_full():
    # True when allocations sum to budget
    assert allocation_full((10, 10, 10), 30)
    assert allocation_full((10, 5, 12), 27)
    assert allocation_full((-10, -10, 20), 0)

    # False when allocations do not sum to budget
    assert not allocation_full((10, 10, 10), 29)
    assert not allocation_full((10, 5, 12), 28)
    assert not allocation_full((-10, -10, 20), -1)


def test_constraints_met():
    b = (
        (10, 30),
        (None, None),
        (None, 20),
        (20, None),
    )

    # True when values are within constraints
    assert constraints_met(b, (15, 35, 18, 24))
    assert constraints_met(b, (10, 15, 20, 20))
    assert constraints_met(b, (28, 19, -8, 82))
    assert constraints_met(b, (30, -82, 1, 28))

    # False when any constraints are breached
    assert not constraints_met(b, (8, 35, 18, 24))
    assert not constraints_met(b, (31, 15, 20, 20))
    assert not constraints_met(b, (28, 19, 21, 82))
    assert not constraints_met(b, (30, -82, 1, 8))


def test_trivial_constraints():
    # trivial constraints (reducing to constants) behaves as expected
    solver = EquitableBudgetAllocator(((-1, -1), (3, 3), (5, 5), (7, 7)))
    assert solver.solve(-1 + 3 + 5 + 7) == (-1, 3, 5, 7)

    with pytest.raises(errors.InsufficientBudgetError):
        solver.solve(-1 + 3 + 5 + 6)

    with pytest.raises(errors.ExcessBudgetError):
        solver.solve(3 + 5 + 7)


def test_constraint_validation():
    # invalid constraints (lower bound > upper bound) raises error
    with pytest.raises(errors.ConstraintError):
        EquitableBudgetAllocator(((0, -4), (2, 3), (None, 5)))

    with pytest.raises(errors.ConstraintError):
        EquitableBudgetAllocator(((-4, 0), (3, 2), (None, 5)))

    # fixing order of constraints stops error
    EquitableBudgetAllocator(((-4, 0), (2, 3), (None, 5)))


# actual solver tests
def test_solution_bounds():
    # both bounds
    allocator = EquitableBudgetAllocator(((3, 5), (2, 50), (9, 15)))
    assert allocator.lower_bound == 3 + 2 + 9
    assert allocator.upper_bound == 5 + 50 + 15

    # no lower bound
    allocator = EquitableBudgetAllocator(((None, 5), (2, 50), (9, 15)))
    assert allocator.lower_bound == None
    assert allocator.upper_bound == 5 + 50 + 15

    # no upper bound
    allocator = EquitableBudgetAllocator(((3, 5), (2, None), (9, 15)))
    assert allocator.lower_bound == 3 + 2 + 9
    assert allocator.upper_bound == None

    # no bounds
    allocator = EquitableBudgetAllocator(((None, 5), (2, 50), (9, None)))
    assert allocator.lower_bound == None
    assert allocator.upper_bound == None


def test_equality(cases: typing.Sequence[Bounds]):
    # solvers with the same parameters evaluate to be equal
    assert EquitableBudgetAllocator(cases[0]) == EquitableBudgetAllocator(cases[0])
    # solvers with different parameters evaluate to not be equal
    assert EquitableBudgetAllocator(cases[0]) != EquitableBudgetAllocator(cases[1])


def test_simple():
    # testing for some simple hardcoded cases with "simple" solutions
    assert solve(((None, None),), 100) == (100,)
    assert solve(((None, None), (None, None)), 100) == (50, 50)
    assert solve(((5, 10), (5, None)), 100) == (10, 90)
    assert solve(((-5, 10), (5, None)), 10) == (5, 5)
    assert solve(((-5, 10), (5, None)), 2) == (-3, 5)
    assert solve(((-5, 10), (5, None)), 0) == (-5, 5)
    assert solve(((5, 10), (5, 10), (10, 30)), 50) == (10, 10, 30)
    assert solve(((5, 10), (5, 10), (10, 30)), 40) == (10, 10, 20)
    assert solve(((5, 10), (5, 10), (10, 30)), 30) == (10, 10, 10)
    assert solve(((5, 10), (5, 10), (10, 30)), 20) == (5, 5, 10)

    # testing extrapolation of unbounded problems
    assert solve(((None, 10), (5, 10), (10, 30)), -1000) == (-1015, 5, 10)
    assert solve(((None, 10), (5, 10), (10, 30)), 0) == (-15, 5, 10)
    assert solve(((None, 10), (5, 10), (10, 30)), 15) == (0, 5, 10)
    assert solve(((10, None), (5, 10), (-40, 30)), 1000) == (960, 10, 30)
    assert solve(((10, None), (5, 10), (-40, 30)), 0) == (10, 5, -15)
    assert solve(((10, None), (5, 10), (-40, 30)), 50) == (20, 10, 20)
    assert solve(((10, None), (5, 10), (-40, 30)), 60) == (25, 10, 25)
    assert solve(((10, None), (5, 10), (-40, 30)), 80) == (40, 10, 30)

    # budget is above upper bound
    with pytest.raises(errors.ExcessBudgetError):
        solve(((5, 50), (-10, 10)), 61)

    # evaluation at upper bound works
    solve(((5, 50), (-10, 10)), 60)

    # budget is below lower bound
    with pytest.raises(errors.InsufficientBudgetError):
        solve(((5, 50), (-10, 10)), -6)

    # evaluation at lower bound works
    solve(((5, 50), (-10, 10)), -5)


def test_solutions(cases: typing.Sequence[Bounds]):
    # test solutions for variety of bounds and budgets
    for bounds in cases:
        solver = EquitableBudgetAllocator(bounds)
        low = min([*[b[0] for b in solver.bounds if b[0] is not None], -400])
        high = max([*[b[1] for b in solver.bounds if b[1] is not None], 400])
        for budget in range(low - 20, high + 20, int((high - low) / 10)):
            if (lb := solver.lower_bound) and budget < lb:
                with pytest.raises(errors.InsufficientBudgetError):
                    solver.solve(budget)
            elif (ub := solver.upper_bound) and budget > ub:
                with pytest.raises(errors.ExcessBudgetError):
                    solver.solve(budget)
            else:
                assert solution_correct(bounds, budget)


def test_scipy(cases: typing.Sequence[Bounds]):
    # test that real-valued solutions align with trust-region constrained methods
    for bounds in cases:
        solver = EquitableBudgetAllocator(bounds)
        # NOTE: scipy optimizer for large cases is SLOW - only checking one budget per test case
        low = sum([b[0] for b in solver.bounds if b[0] is not None] or [-400])
        high = sum([b[1] for b in solver.bounds if b[1] is not None] or [600])
        budget = (low + high) // 2
        solver_allocations = solver.solve(budget, integer=False)
        scipy_allocations = scipy_solve(bounds, budget)
        assert solver_allocations == pytest.approx(scipy_allocations, abs=1e-2)
