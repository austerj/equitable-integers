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
    """Solver for equitable allocations of a budget of integers under constraints."""

    __slots__ = (
        "bounds",
        "n_lower_unbounded",
        "n_upper_unbounded",
        "lower_bound",
        "upper_bound",
        "is_unbounded",
        "_table",
    )

    def __init__(self, bounds: Bounds) -> None:
        # validate constraints
        if any(b[0] is not None and b[1] is not None and b[0] > b[1] for b in bounds):
            raise errors.ConstraintError("Invalid constraints")

        # number of allocations without lower / upper bounds
        self.bounds = bounds
        self.n_lower_unbounded = sum(b[0] is None for b in self.bounds)
        self.n_upper_unbounded = sum(b[1] is None for b in self.bounds)

        # flag denoting if the problem has no constraints
        self.is_unbounded = all(b[0] is None and b[1] is None for b in self.bounds)

        # lower / upper bounds for the budget solution space
        self.lower_bound = None if self.n_lower_unbounded else sum(b[0] for b in self.bounds)  # type: ignore
        self.upper_bound = None if self.n_upper_unbounded else sum(b[1] for b in self.bounds)  # type: ignore

        # construct solution table
        self._table = self._solve_table()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bounds=({self._bounds_repr}))"

    @property
    def _bounds_repr(self) -> str:
        """String representation of bounds."""
        bound_strs: list[str] = []
        for bound in self.bounds:
            l, r = bound
            bound_strs.append(f"({'-∞' if l is None else l}, {'∞' if r is None else r})")
        return f"({', '.join(bound_strs)})"

    def __eq__(self, value: object) -> bool:
        return isinstance(value, self.__class__) and all(
            b_self == b_val for b_self, b_val in itertools.zip_longest(self.bounds, value.bounds)
        )

    def _solve_x(self, budget: int) -> tuple[float, int]:
        """Return the (non-integer) solution to x and the rate (count of non-binding constraints)."""
        # if there are no constraints on any allocations, the solution is just the mean
        if self.is_unbounded:
            return budget / len(self.bounds), len(self.bounds)

        # get ref to solution table
        keys, values = self._table

        # find region with binary search
        budget_key = bisect_right(keys, budget) - 1

        # handle exterior of defined solution table
        if budget_key < 0:
            if self.lower_bound is not None:
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
        return x if not rate else x + (budget - budget_start) / rate, rate

    def allocations(self, x: float) -> tuple[float, ...]:
        """Evaluate the constrained allocations for the specified value of x."""
        return tuple(
            itertools.chain(
                (
                    # lower bound if x is above it
                    float(b[0]) if b[0] is not None and x < b[0]
                    # upper bound if x is below it
                    else float(b[1]) if b[1] is not None and x > b[1]
                    # else value
                    else x
                    for b in self.bounds
                ),
            )
        )

    def _integer_allocations(self, x: float, n_nonbinding: int) -> tuple[int, ...]:
        """Evaluate the constrained integer allocations for the specified value of x."""
        # want to apply n floors and m-n ceils such that the budget remains exhausted, i.e.
        #   n * floor(x) + (m-n) * ceil(x) = m * x
        # with m being total non-binding constraints
        #
        # if ceil(x) > floor(x):
        #   n = m * (ceil(x) - x) / (ceil(x) - floor(x))
        #     = m * (ceil(x) - x)
        # since ceil(x) - floor(x) = 1
        #
        # if ceil(x) == floor(x):
        #   n in [0,m]
        # hence can reuse above case
        floor_x, ceil_x = math.floor(x), math.ceil(x)
        n_floored = round(n_nonbinding * (ceil_x - x))

        # track the count of floored values
        counter = itertools.count(0)

        return tuple(
            itertools.chain(
                (
                    # lower bound if x is above it
                    b[0] if b[0] is not None and x < b[0]
                    # upper bound if x is below it
                    else b[1] if b[1] is not None and x > b[1]
                    # else floor until count has been reached, then ceil
                    else floor_x if next(counter) < n_floored else ceil_x
                    for b in self.bounds
                ),
            )
        )

    @typing.overload
    def solve(self, budget: int, integer: typing.Literal[True] = ...) -> tuple[int, ...]:
        ...

    @typing.overload
    def solve(self, budget: int, integer: typing.Literal[False] = ...) -> tuple[float, ...]:
        ...

    def solve(self, budget: int, integer: bool = True) -> tuple[typing.Any, ...]:
        """Solve the (integer) allocation problem and return the resulting allocations."""
        x, rate = self._solve_x(budget)
        if not integer:
            return self.allocations(x)
        return self._integer_allocations(x, rate)

    @property
    def flat_bounds(self) -> list[tuple[int, bool]]:
        """Lower- and upper bounds flattened into sorted tuples of (value, is_upper_bound flag)."""
        lower_bounds = ((b[0], False) for b in self.bounds if b[0] is not None)
        upper_bounds = ((b[1], True) for b in self.bounds if b[1] is not None)
        return sorted(itertools.chain(lower_bounds, upper_bounds), key=itemgetter(0))

    def _solve_table(self) -> SolutionTable:
        """Compute budget |-> (x, rate) solution table of linear regions for bounds."""
        # initialize variables
        budget = 0  # lower bounds are accumulated in loop
        rate = self.n_lower_unbounded  # initial rate is 1 per non-lower-bounded element

        # construct intermediary table mapping values of x to rates of budget allocation
        r_table: dict[int, int] = {}
        for x, is_upper in self.flat_bounds:
            if is_upper:
                # if upper bound: rate decreases
                rate -= 1
            else:
                # if lower bound: rate and budget increases
                rate += 1
                budget += x
            r_table[x] = rate

        # construct final table mapping budget to x-value and rates on linear sections
        prev_x, prev_rate = 0, self.n_lower_unbounded
        x_table: dict[int, tuple[int, int]] = {}

        # NOTE: since bounds are pre-sorted, insertion order of dicts ensures that keys are sorted
        for x, rate in r_table.items():
            # accumulate the mapping from regions of budgets to values of x
            budget += (x - prev_x) * prev_rate
            x_table[budget] = (x, rate)
            prev_x, prev_rate = x, rate

        return tuple(x_table.keys()), tuple(x_table.values())


@typing.overload
def solve(bounds: Bounds, budget: int, integer: typing.Literal[True] = ...) -> tuple[int, ...]:
    ...


@typing.overload
def solve(bounds: Bounds, budget: int, integer: typing.Literal[False] = ...) -> tuple[float, ...]:
    ...


def solve(bounds: Bounds, budget: int, integer: bool = True) -> tuple[typing.Any, ...]:
    """Solve the (integer) allocation problem and return the resulting allocations."""
    return EquitableBudgetAllocator(bounds).solve(budget, integer)  # type: ignore
