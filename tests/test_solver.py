from eqint.solver import Bounds, EquitableBudgetAllocator

bounds = (
    # bounds 0
    (
        (75, None),
        (45, 65),
        (None, None),
        (40, 55),
        (20, 70),
        (10, 90),
        (None, None),
        (None, 35),
        (None, None),
    ),
    # bounds 1
    (
        (4, 9),
        (None, None),
        (None, 15),
        (23, None),
        (30, None),
        (None, 30),
        (81, 90),
        (None, 35),
        (11, 29),
    ),
)


def solve(bounds: Bounds, budget: int):
    """Solve allocation problem for bounds and budget."""
    return EquitableBudgetAllocator(bounds).solve(budget)


def check_solution(bounds: Bounds, budget: int):
    """Check that a solution is optimal."""
    # solve for allocations
    allocations = EquitableBudgetAllocator(bounds).solve(budget)
    # all bounds are adhered to
    assert constraints_met(bounds, allocations)
    # the full budget is allocated
    assert allocation_full(allocations, budget)
    # integers are optimally distributed
    assert integers_optimal(bounds, allocations)


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
    elif non_lower_bounded:
        # all values are upper-bounded, so we check that the difference among those not
        # lower-bounded is at-most 1
        return abs(max(non_lower_bounded) - min(non_lower_bounded)) <= 1
    else:
        # all values are lower-bounded, so we check that the difference among those not
        # upper-bounded is at-most 1
        return abs(max(non_upper_bounded) - min(non_upper_bounded)) <= 1


def test_hash():
    # solvers with the same parameters evaluate to be equal
    assert EquitableBudgetAllocator(bounds[0]) == EquitableBudgetAllocator(bounds[0])
    # solvers with different parameters evaluate to not be equal
    assert EquitableBudgetAllocator(bounds[0]) != EquitableBudgetAllocator(bounds[1])
