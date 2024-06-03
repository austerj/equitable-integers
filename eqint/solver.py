import itertools
import math
import typing
from bisect import bisect_right
from operator import itemgetter

from eqint import errors

# tuples of bounds provided as problem parameters
Bound = tuple[int | None, int | None]
Bounds = typing.Sequence[Bound]

# solution table mapping budget to (x, rate) pairs
SolutionKeys = tuple[int, ...]
SolutionValues = tuple[tuple[int, int], ...]
SolutionTable = tuple[SolutionKeys, SolutionValues]


class EquitableBudgetAllocator:
    """Solver for the most-equitable allocation of a budget of integers under constraints."""

    __slots__ = ("bounds", "n_lower_unbounded", "n_upper_unbounded", "lower_bound", "upper_bound", "_table")

    def __init__(self, bounds: Bounds):
        # validate constraints
        if any(b[0] is not None and b[1] is not None and b[0] > b[1] for b in bounds):
            raise errors.ConstraintError("Invalid constraint")

        # construct solution table
        self.bounds = bounds
        self._table = _solve_table(self.bounds)

        # number of allocations without lower / upper bounds
        self.n_lower_unbounded = sum(b[0] is None for b in self.bounds)
        self.n_upper_unbounded = sum(b[1] is None for b in self.bounds)

        # lower / upper bounds for the budget solution space
        self.lower_bound = None if self.n_lower_unbounded else sum(b[0] for b in self.bounds)  # type: ignore
        self.upper_bound = None if self.n_upper_unbounded else sum(b[1] for b in self.bounds)  # type: ignore

    def __repr__(self):
        return f"{self.__class__.__name__}(bounds={self.bounds})"

    def __eq__(self, value: object) -> bool:
        return isinstance(value, self.__class__) and all(
            b_self == b_val for b_self, b_val in itertools.zip_longest(self.bounds, value.bounds)
        )

    @property
    def is_unbounded(self):
        """Flag denoting if the problem has no constraints (i.e. an unbounded linear function)."""
        # the table has no entries iff all bounds are None
        return len(self._table[0]) == 0

    def _solve_x(self, budget: int):
        """Compute the (non-integer) solution to x."""
        # if there are no constraints on any allocations, the solution is just the average
        if self.is_unbounded:
            return budget / len(self.bounds)

        # get ref to solution table
        keys, values = self._table

        # find region with binary search
        budget_key = bisect_right(keys, budget) - 1

        # handle exterior of defined solution table
        if budget_key < 0:
            if self.lower_bound:
                raise errors.InsufficientBudgetError("Budget outside solution space: cannot satisfy lower bounds.")
            # if no lower bound, we can extrapolate "backwards" from x at the rate of the number of
            # lower-unbounded allocations
            x, rate = values[0][0], self.n_lower_unbounded
            budget_start = keys[0]
        elif self.upper_bound and budget > self.upper_bound:
            raise errors.ExcessBudgetError("Budget outside solution space: cannot satisfy upper bounds.")
            # if not exceeding an upper bound, we can proceed from the upper (defined) boundary of
            # the solution table and extrapolate forward
        else:
            x, rate = values[budget_key]
            budget_start = keys[budget_key]

        # budget_start + dx * rate = budget <=> dx = (budget - budget_start) / rate
        # NOTE: budget_start == budget <=> exactly at bound value, so dx is 0
        dx = 0 if budget == budget_start else (budget - budget_start) / rate

        return x + dx

    def evaluate(self, x: float) -> tuple[float, ...]:
        """Evaluate the bounded distribution for the specified value of x."""
        return tuple(
            itertools.chain(
                (
                    # lower bound if x is above it
                    b[0] if b[0] is not None and x < b[0]
                    # upper bound if x is below it
                    else b[1] if b[1] is not None and x > b[1]
                    # else value
                    else x
                    for b in self.bounds
                ),
            )
        )

    def solve(self, budget: int):
        """Solve the integer allocation problem and return the bounded items."""
        x = self._solve_x(budget)
        allocations = self.evaluate(x)
        return _distribute_integers(allocations)


def _flatten_bounds(bounds: Bounds) -> list[tuple[int, bool]]:
    """Return (lower, upper) bounds flattened into sorted tuples of (value, is-upper-bound)."""
    lower_bounds = ((b[0], False) for b in bounds if b[0] is not None)
    upper_bounds = ((b[1], True) for b in bounds if b[1] is not None)
    return sorted(itertools.chain(lower_bounds, upper_bounds), key=itemgetter(0))


def _solve_table(bounds: Bounds) -> SolutionTable:
    """Compute budget |-> (x, rate) solution table of linear regions for bounds."""
    # construct lookup table
    flat_bounds = _flatten_bounds(bounds)

    # initialize vars
    budget_start = 0  # constraints are accumulated in loop
    n_lower_unbounded = sum(b[0] is None for b in bounds)
    rate = n_lower_unbounded  # initial rate is 1 per non-lower-bounded element

    # construct intermediary table mapping values of x to rates of budget allocation
    x_table: dict[int, int] = {}
    for b, is_upper in flat_bounds:
        if is_upper:
            # if upper bound: rate decreases
            rate -= 1
        else:
            # if lower bound: rate and start of initial budget region increases
            rate += 1
            budget_start += b
        x_table[b] = rate

    # construct final table mapping budget to x-value and rates on linear sections
    # NOTE: since bounds are pre-sorted, insertion order guarantees that keys are sorted
    keys: list[int] = []
    values: list[tuple[int, int]] = []
    # rate accounts for the contribution from unbounded allocations before the first x
    prev_x, prev_rate = 0, n_lower_unbounded

    for x, rate in x_table.items():
        # accumulate the mapping from regions of budgets to values of x
        budget_start += (x - prev_x) * prev_rate
        keys.append(budget_start)
        values.append((x, rate))
        prev_x, prev_rate = x, rate

    return tuple(keys), tuple(values)


def _distribute_integers(allocations: tuple[float, ...]) -> tuple[int, ...]:
    """Optimally distribute integers from the continuous solution to the allocation problem."""
    # since the bounds are integer, the floored value will not break the constraints
    floored_allocations = [math.floor(a) for a in allocations]

    # NOTE: any binding upper bounds in the original problem will have a difference of 0 to the
    # floored version; since we sort by the difference, these values will not appear before all
    # missing integers have been added
    diff_sorted = sorted(enumerate(f_a - a for f_a, a in zip(floored_allocations, allocations)), key=itemgetter(1))

    # rounding is fine here; allocation sum is (negative) float, but value itself represents an
    # integer (since it solves for integer total budget)
    int_truncation = round(sum(map(itemgetter(1), diff_sorted)))

    # add integers in order of largest deviation to continuous solution
    for i, _ in diff_sorted:
        # >= 0 means all required ints have been added back
        if int_truncation >= 0:
            break
        floored_allocations[i] += 1
        int_truncation += 1

    return tuple(floored_allocations)


def solve(bounds: Bounds, budget: int):
    """Solve the equitable allocation problem for bounds and budget."""
    return EquitableBudgetAllocator(bounds).solve(budget)
