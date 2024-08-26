import inspect
import os
import typing

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure

rc_context = mpl.rc_context(
    {
        # figure
        "figure.figsize": [7, 3],
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.08334,
        "figure.constrained_layout.w_pad": 0.08334,
        # font
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 11,
        "font.size": 11,
        # plot styling
        "axes.prop_cycle": cycler(color=["k"]),
        "lines.color": "black",
        "lines.linewidth": 2,
        # axes
        "axes.linewidth": 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # output
        "savefig.format": "pdf",
    }
)


def savefig(fig: Figure, fname: typing.Optional[str] = None, *args, **kwargs):
    # get path to .out dir
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".out")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # save to .out/{filename}.pdf
    fig.savefig(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            ".out",
            # default to filename (ignoring ".py") of calling module
            os.path.basename(inspect.stack()[1].filename[:-3]) if fname is None else fname,
        ),
        *args,
        **kwargs,
    )


# typed subplots (flattening dimensions of length 1)
ShareType = bool | typing.Literal["none", "all", "row", "col"]


@typing.overload
def subplots(
    nrows: typing.Literal[1] = ...,
    ncols: typing.Literal[1] = ...,
    figsize: tuple[float, float] | None = ...,
    *,
    sharex: ShareType = ...,
    sharey: ShareType = ...,
    **kwargs,
) -> tuple[Figure, Axes]:
    ...


@typing.overload
def subplots(
    nrows: int = ...,
    ncols: typing.Literal[1] = ...,
    figsize: tuple[float, float] | None = ...,
    *,
    sharex: ShareType = ...,
    sharey: ShareType = ...,
    **kwargs,
) -> tuple[Figure, tuple[Axes, ...]]:
    ...


@typing.overload
def subplots(
    nrows: typing.Literal[1] = ...,
    ncols: int = ...,
    figsize: tuple[float, float] | None = ...,
    *,
    sharex: ShareType = ...,
    sharey: ShareType = ...,
    **kwargs,
) -> tuple[Figure, tuple[Axes, ...]]:
    ...


@typing.overload
def subplots(
    nrows: int = ...,
    ncols: int = ...,
    figsize: tuple[float, float] | None = ...,
    *,
    sharex: ShareType = ...,
    sharey: ShareType = ...,
    **kwargs,
) -> tuple[Figure, tuple[tuple[Axes, ...], ...]]:
    ...


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    *,
    sharex: ShareType = True,
    sharey: ShareType = False,
    **kwargs,
) -> tuple[Figure, typing.Any]:
    f, axs_ = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey, squeeze=False, figsize=figsize, **kwargs)
    # return Figure, Axes if only one subplot
    if axs_.size == 1:
        return f, axs_[0, 0]
    # return Figure, tuple[Axes, ...] if one dimension has length 1
    if axs_.shape[0] == 1 or axs_.shape[1] == 1:
        axs_ = axs_.flatten()
        return f, tuple(axs_)
    # return Figure, tuple[tuple[Axes, ...], ...] otherwise
    return f, tuple(tuple(row) for row in axs_)
