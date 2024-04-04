import matplotlib.pyplot as plt
from matplotlib import ticker

from eqint.solver import Bounds, EquitableBudgetAllocator
from plots import rc_context, savefig


def plot_h(bounds: Bounds):
    """Evaluate the mapping from x to budgets."""
    # get domain of plot
    start = min(b[0] for b in bounds if b[0] is not None)
    end = max(b[1] for b in bounds if b[1] is not None)
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


@rc_context
def main():
    bounds = (
        (1, 7),
        (3, 4),
        (2, 4),
        (None, 3),
        (5, 7),
    )
    savefig(plot_h(bounds))


if __name__ == "__main__":
    main()
