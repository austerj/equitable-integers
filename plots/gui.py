from __future__ import annotations

import typing
from abc import ABC

from matplotlib import colors, patches
from matplotlib.axes import Axes

from eqint.solver import Bounds, EquitableBudgetAllocator
from plots import rc_context, savefig, subplots

ALPHA = 0.5
C = [
    colors.to_rgba("#555555", ALPHA),
    colors.to_rgba("#E69F00", ALPHA),
    colors.to_rgba("#56B4E9", ALPHA),
    colors.to_rgba("#009E73", ALPHA),
    colors.to_rgba("#F0E442", ALPHA),
    colors.to_rgba("#0072B2", ALPHA),
    colors.to_rgba("#D55E00", ALPHA),
    colors.to_rgba("#CC79A7", ALPHA),
]


def draw_rect(ax: Axes, y: int, x: int, height: int, width: int, **kwargs):
    patch = patches.Rectangle((x, y), width, height, **kwargs)
    ax.add_patch(patch)


class LayoutElement(ABC):
    def __init__(self, bounds: Bounds | None, **kwargs) -> None:
        self.solver = EquitableBudgetAllocator(bounds if bounds else [(0, None)])
        self.kwargs = {"facecolor": "none", "edgecolor": C[0], **kwargs}

    def draw(self, ax: Axes, y: int, x: int, height: int, width: int, **kwargs) -> None:
        draw_rect(ax, y, x, height, width, **self.kwargs, **kwargs)

    def aggregate_bounds(self, solvers: typing.Sequence[EquitableBudgetAllocator]) -> tuple[int, int | None]:
        """Aggregate lower- and upper bounds."""
        lower_bound = max(solver.lower_bound or 0 for solver in solvers)
        # if any upper bound is None, the aggregated upper bound is also None
        if any(solver.upper_bound is None for solver in solvers):
            upper_bound = None
        else:
            upper_bound = min(solver.upper_bound or 0 for solver in solvers)
        return lower_bound, upper_bound


class Row(LayoutElement):
    """A row element distributing horizontal space to columns."""

    def __init__(
        self,
        columns: typing.Sequence[Column] | None = None,
        *,
        min_height: int = 0,
        max_height: int | None = None,
        **kwargs,
    ) -> None:
        self.columns = columns or []
        self.min_height = min_height
        self.max_height = max_height
        # override height constraints from rows in nested columns
        if self.columns and any(column.rows for column in self.columns):
            min_height, max_height = self.aggregate_bounds([column.solver for column in self.columns])
            self.min_height = max(self.min_height, min_height)
            self.max_height = max_height
        # solver for width allocated to columns
        super().__init__(tuple((col.min_width, col.max_width) for col in self.columns), **kwargs)

    def draw(self, ax: Axes, y: int, x: int, height: int, width: int, **kwargs) -> None:
        if self.columns:
            column_x = x
            for column, column_width in zip(self.columns, self.solver.solve(width)):
                column.draw(ax, y, column_x, height, column_width, **kwargs)
                column_x += column_width
        super().draw(ax, y, x, height, width, **kwargs)


class Column(LayoutElement):
    """A column element distributing vertical space to rows."""

    def __init__(
        self, rows: typing.Sequence[Row] | None = None, *, min_width: int = 0, max_width: int | None = None, **kwargs
    ) -> None:
        self.rows = rows or []
        self.min_width = min_width
        self.max_width = max_width
        # override width constraints from columns in nested rows
        if self.rows and any(row.columns for row in self.rows):
            min_width, max_width = self.aggregate_bounds([row.solver for row in self.rows])
            self.min_width = max(self.min_width, min_width)
            self.max_width = max_width
        # solver for height allocated to rows
        super().__init__(tuple((row.min_height, row.max_height) for row in self.rows), **kwargs)

    def draw(self, ax: Axes, y: int, x: int, height: int, width: int, **kwargs):
        if self.rows:
            row_y = y
            for row, row_height in zip(self.rows, self.solver.solve(height)):
                row.draw(ax, row_y, x, row_height, width, **kwargs)
                row_y += row_height
        super().draw(ax, y, x, height, width, **kwargs)


LAYOUT = Column(
    [
        Row(min_height=60, max_height=60, facecolor=C[0], label="Header"),
        Row(
            [
                Column(min_width=80, max_width=80, facecolor=C[1], label="Toolbar"),
                Column(
                    [
                        Row(),
                        Row(min_height=60, max_height=160),
                    ],
                    max_width=320,
                    facecolor=C[2],
                    label="Panel",
                ),
                # main column
                Column(
                    [
                        # content
                        Row(
                            [
                                Column([Row(min_height=40, max_height=40), Row()]),
                            ],
                            min_height=350,
                            facecolor=C[3],
                            label="Workspace",
                        ),
                        # extra
                        Row(max_height=200, facecolor=C[7], label="Extra"),
                    ],
                    min_width=540,
                ),
                Column(
                    [
                        Row(min_height=120, max_height=240),
                        Row(),
                    ],
                    min_width=220,
                    max_width=300,
                    facecolor=C[5],
                    label="Options",
                ),
            ]
        ),
        Row(min_height=40, max_height=40, facecolor=C[6], label="Footer"),
    ]
)


def plot_layout(res_x: int, res_y: int):
    """Plot layout in fullscreen, side-by-side and quadrants."""
    half_y, half_x = res_y // 2, res_x // 2

    # show x ticks on all subplots
    f, axs = subplots(1, 3, figsize=(9, 2.35))
    for ax in axs:
        ax.spines.top.set_visible(True)
        ax.spines.right.set_visible(True)
        ax.tick_params(bottom=False, left=False, labelleft=False, labelbottom=False)
        ax.set_xlim(0, res_x)
        ax.set_ylim(0, res_y)
        ax.set_aspect("equal")
        ax.set_anchor("N")
        ax.invert_yaxis()

    axs[0].set_title("Fullscreen")
    LAYOUT.draw(axs[0], 0, 0, res_y, res_x)

    axs[1].set_title("Side-by-side")
    # dampened windows
    LAYOUT.draw(axs[1], 0, half_x, res_y, half_x)
    draw_rect(axs[1], 0, half_x, res_y, half_x, facecolor="w", alpha=0.75)
    # main window
    LAYOUT.draw(axs[1], 0, 0, res_y, half_x)

    axs[2].set_title("Quadrants")
    # dampened windows
    LAYOUT.draw(axs[2], half_y, 0, half_y, half_x)
    LAYOUT.draw(axs[2], 0, half_x, half_y, half_x)
    LAYOUT.draw(axs[2], half_y, half_x, half_y, half_x)
    draw_rect(axs[2], half_y, 0, half_y, half_x, facecolor="w", alpha=0.75)
    draw_rect(axs[2], 0, half_x, half_y, half_x, facecolor="w", alpha=0.75)
    draw_rect(axs[2], half_y, half_x, half_y, half_x, facecolor="w", alpha=0.75)
    # main window
    LAYOUT.draw(axs[2], 0, 0, half_y, half_x)

    # get shared figure legend off any axis
    f.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncols=7)

    return f


@rc_context
def main():
    savefig(plot_layout(1920, 1080), "gui")


if __name__ == "__main__":
    main()
