import itertools
import math
import typing

from matplotlib import ticker

from eqint.solver import Bounds, EquitableBudgetAllocator
from plots import rc_context, savefig, subplots

T = typing.TypeVar("T")
S = typing.TypeVar("S")


def get_domain(bounds: Bounds) -> tuple[int, int]:
    """Get base domain of evaluation (smallest and largest explicit bound values)."""
    start, end = None, None
    for b in itertools.chain.from_iterable(bounds):
        if b is not None:
            start = b if start is None else min(start, b)
            end = b if end is None else max(end, b)
    return typing.cast(int, start), typing.cast(int, end)


def get_segments(xs: typing.Sequence[T], ys: typing.Sequence[S]) -> list[tuple[list[T], list[S]]]:
    """Split sequences into continuous segments."""
    prev_y = None, None
    segments: list[tuple[list[T], list[S]]] = []
    segment_x: list[T] = []
    segment_y: list[S] = []
    for x, y in zip(xs, ys):
        if y == prev_y:
            # skip single-element segments
            if len(segment_x) > 1:
                segments.append((segment_x, segment_y))
            segment_x, segment_y = [], []
        segment_x.append(x)
        segment_y.append(y)
        prev_y = y
    segments.append((segment_x, segment_y))
    return segments


def plot_h(bounds: Bounds):
    """Plot the mapping from x to budgets."""
    # get domain of plot
    start, end = get_domain(bounds)
    padding = 1.5
    xs = range(start, end + 1)

    # evaluate on domain
    solver = EquitableBudgetAllocator(bounds)
    budgets = [sum(solver.allocations(x)) for x in xs]

    # infer rate of change
    def rates_of_change(xs, budgets):
        rates = []
        for i in range(1, len(budgets)):
            d_budget = budgets[i] - budgets[i - 1]
            d_x = xs[i] - xs[i - 1]
            rates.append(d_budget / d_x)
        return rates

    rates = rates_of_change(xs, budgets)

    # show x ticks on all subplots
    f, axs = subplots(3, sharex=True, figsize=(7, 7))
    axs[0].xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    for ax in axs:
        ax.xaxis.set_tick_params(labelbottom=True)

    # plot function evaluation
    axs[0].plot(xs, budgets)
    axs[0].set_title("$h(x)$")
    axs[0].set_xlabel("$x$")
    axs[0].set_ylabel("Budget")

    # plot left extrapolation
    l_xs = [start - padding, start]
    l_budgets = [sum(solver.allocations(x)) for x in l_xs]
    axs[0].plot(l_xs, l_budgets, linestyle="--")
    # plot right extrapolation
    r_xs = [end, end + padding]
    r_budgets = [sum(solver.allocations(x)) for x in r_xs]
    axs[0].plot(r_xs, r_budgets, linestyle="--")

    # plot bounds
    for i, b in enumerate(reversed(bounds)):
        lower = b[0] if b[0] is not None else start - padding
        upper = b[1] if b[1] is not None else end + padding
        axs[1].plot([lower, upper], [i, i], linestyle="--" if b[0] is None or b[1] is None else "-")
        if b[0] is not None:
            axs[1].plot(lower, i, marker="<")
        if b[1] is not None:
            axs[1].plot(upper, i, marker=">")
    axs[1].set_yticks(range(len(bounds)), [f"$a_{{{i+1}}}$" for i in reversed(range(len(bounds)))])
    axs[1].set_title("Bounds")
    axs[1].set_xlabel("$x$")
    axs[1].set_ylabel("Allocation")

    # plot left extrapolated rate
    l_xs = [start - padding, start]
    l_budgets = [sum(solver.allocations(x)) for x in l_xs]
    l_rate = rates_of_change(l_xs, l_budgets)
    axs[2].step([*l_xs, xs[0]], [l_rate[0], l_rate[0], rates[0]], where="post", linestyle="--")

    # plot right extrapolated rate
    r_xs = [end, end + padding]
    r_budgets = [sum(solver.allocations(x)) for x in r_xs]
    r_rate = rates_of_change(r_xs, r_budgets)
    axs[2].step([xs[-1], *r_xs], [rates[-1], r_rate[0], r_rate[0]], where="post", linestyle="--")

    # plot interior rates
    axs[2].step(xs, [*rates, rates[-1]], where="post")
    axs[2].set_title("Rate of change")
    axs[2].set_xlabel("$x$")
    axs[2].set_ylabel("Rate")

    # force ylim
    ylim = axs[0].get_ylim()
    axs[0].set_ylim(ylim[0], math.ceil(ylim[1] / 5) * 5 + padding)

    return f


def plot_h_inv(bounds: Bounds):
    """Plot the (right inverse of h) mapping from budgets to x."""
    # get domain of h
    start, end = get_domain(bounds)
    padding = 2
    xs = range(start, end + 1)

    # evaluate on domain
    solver = EquitableBudgetAllocator(bounds)
    budgets = [sum(solver.allocations(x)) for x in xs]

    f, ax = subplots()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # plot right inverse function evaluation
    for i, (x_segment, budget_segment) in enumerate(get_segments(xs, budgets)):
        # plot segment and mark discontinuity
        ax.plot(budget_segment, x_segment)
        ax.plot(budget_segment[-1], x_segment[-1], marker="o", markerfacecolor="w")
        # mark start of new segment
        if i > 0:
            ax.plot(budget_segment[0], x_segment[0], marker="o")
    ax.set_title("$g(B)$")
    ax.set_xlabel("$B$")
    ax.set_ylabel("$x^*$")

    # plot left / right extrapolation if unbounded
    if any(b[0] is None for b in bounds):
        l_xs = [start - padding, start]
        l_budgets = [sum(solver.allocations(x)) for x in l_xs]
        ax.plot(l_budgets, l_xs, linestyle="--")
    else:
        ax.plot(budgets[0], start, marker="o")
    if any(b[1] is None for b in bounds):
        r_xs = [end - 1, end + padding - 1]
        r_budgets = [sum(solver.allocations(x)) for x in r_xs]
        ax.plot(r_budgets, r_xs, linestyle="--")
    else:
        ax.plot(budgets[-1], end, marker="o")

    # force integer ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    return f


@rc_context
def main():
    bounds = (
        (1, 5),
        (2, 4),
        (None, 3),
        (7, 10),
        (9, 12),
    )
    savefig(plot_h(bounds), "monotonic")
    savefig(plot_h_inv(bounds), "inverse")


if __name__ == "__main__":
    main()
