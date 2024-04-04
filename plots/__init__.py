import inspect
import os
import typing

import matplotlib as mpl
from cycler import cycler
from matplotlib.figure import Figure

rc_context = mpl.rc_context(
    {
        # figure
        "figure.figsize": [7, 4.5],
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
