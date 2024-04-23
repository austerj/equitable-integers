import itertools
import typing

import matplotlib.pyplot as plt
from matplotlib import ticker

from eqint.solver import Bounds, EquitableBudgetAllocator
from plots import rc_context, savefig


def get_domain(bounds: Bounds) -> tuple[int, int]:
    """Get base domain of evaluation (smallest and largest explicit bound values)."""
    start, end = None, None
    for b in itertools.chain.from_iterable(bounds):
        if b is not None:
            start = b if start is None else min(start, b)
            end = b if end is None else max(end, b)
    return typing.cast(int, start), typing.cast(int, end)


def plot_h(bounds: Bounds):
    """Evaluate the mapping from x to budgets."""
    # get domain of plot
    start, end = get_domain(bounds)
    padding = 2
    xs = range(start - padding, end + padding)

    # evaluate on domain
    solver = EquitableBudgetAllocator(bounds)
    budgets = [sum(solver.evaluate(x)) for x in xs]

    # plot function evaluation and bounds
    f, ax = plt.subplots(2, sharex=True)
    ax[0].plot(xs, budgets)
    ax[0].set_title("$h(x)$")
    ax[0].set_xlabel("$x$")
    ax[0].set_ylabel("Budget")
    ax[0].xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax[0].xaxis.set_tick_params(labelbottom=True)

    for i, b in enumerate(reversed(bounds)):
        lower = b[0] if b[0] is not None else start - padding
        upper = b[1] if b[1] is not None else end + padding
        ax[1].plot([lower, upper], [i, i], linestyle="--" if b[0] is None or b[1] is None else "-")
        if b[0] is not None:
            ax[1].plot(lower, i, marker="<")
        if b[1] is not None:
            ax[1].plot(upper, i, marker=">")
    ax[1].set_yticks(range(len(bounds)), [f"$x_{i+1}$" for i in reversed(range(len(bounds)))])
    ax[1].set_title("Bounds")
    ax[1].set_xlabel("$x$")
    ax[1].set_ylabel("Allocation")

    return f


def plot_h_inv(bounds: Bounds):
    """Evaluate the (inverse) mapping from x to budgets."""
    # get domain of h
    start, end = get_domain(bounds)
    padding = 2
    xs = range(start, end)

    # evaluate on domain
    solver = EquitableBudgetAllocator(bounds)
    budgets = [sum(solver.evaluate(x)) for x in xs]

    # infer rate of change
    def rates_of_change(budgets, xs):
        rates = []
        for i in range(1, len(budgets)):
            d_x = xs[i] - xs[i - 1]
            d_budget = budgets[i] - budgets[i - 1]
            rates.append(d_x / d_budget)
        return rates

    rates = rates_of_change(budgets, xs)

    # plot inverse function evaluation and rate of change
    f, ax = plt.subplots(2, sharex=True)
    ax[0].plot(budgets, xs)
    ax[0].set_title("$h^{-1}(B)$")
    ax[0].set_xlabel("B")
    ax[0].set_ylabel("$x^*$")
    ax[0].xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax[0].xaxis.set_tick_params(labelbottom=True)

    # plot left / right extrapolation if unbounded
    if any(b[0] is None for b in bounds):
        l_xs = [start - padding, start]
        l_budgets = [sum(solver.evaluate(x)) for x in l_xs]
        l_rate = rates_of_change(l_budgets, l_xs)
        ax[0].plot(l_budgets, l_xs, linestyle="--")
        ax[1].step([*l_budgets, budgets[0]], [l_rate[0], l_rate[0], rates[0]], where="post", linestyle="--")
    if any(b[1] is None for b in bounds):
        r_xs = [end - 1, end + padding - 1]
        r_budgets = [sum(solver.evaluate(x)) for x in r_xs]
        r_rate = rates_of_change(r_budgets, r_xs)
        ax[0].plot(r_budgets, r_xs, linestyle="--")
        ax[1].step([budgets[-1], *r_budgets], [rates[-1], r_rate[0], r_rate[0]], where="post", linestyle="--")

    ax[1].step(budgets, [*rates, rates[-1]], where="post")
    ax[1].set_title("Rate of change")
    ax[1].set_xlabel("B")
    ax[1].set_ylabel("Rate")

    return f


@rc_context
def main():
    bounds = (
        (1, 7),
        (3, 4),
        (2, 4),
        (None, 3),
        (5, 7),
    )
    savefig(plot_h(bounds), "monotonic")
    savefig(plot_h_inv(bounds), "inverse")


if __name__ == "__main__":
    main()
