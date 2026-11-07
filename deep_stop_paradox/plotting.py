"""Figure style shared by all figure scripts."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .io import path  # noqa: E402

# categorical slots, fixed order (validated reference palette)
SHALLOW = "#2a78d6"     # slot 1, blue: shallow-stop (gas content) schedules
DEEP = "#eb6834"        # slot 2, orange: deep-stop (bubble model) schedules
OPTIMAL = "#1baf7a"     # slot 3, aqua: optimal schedules
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"
NEUTRAL = "#f0efec"

# sequential single-hue ramp (blue 100 -> 700)
_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
        "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", _SEQ)
# diverging blue <-> gray <-> red
DIV = LinearSegmentedColormap.from_list(
    "div_br", ["#104281", "#3987e5", "#b7d3f6", NEUTRAL, "#f4b7b6",
               "#e34948", "#8f1f1f"])


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
    })


def comp_colors(n=16):
    """Colors for compartments ordered from fast (light) to slow (dark)."""
    return [SEQ(0.18 + 0.82 * i / max(n - 1, 1)) for i in range(n)]


def label_panels(axes, x=-0.16, y=1.04):
    for i, ax in enumerate(np.ravel(axes)):
        ax.text(x, y, "(%s)" % "abcdefghijkl"[i], transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom", ha="left",
                color=INK)


def legend_below(fig, handles, labels, ncol=4, y=0.0):
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, y), frameon=False,
               handlelength=2.2, columnspacing=1.6)


def legend_under(ax, handles=None, labels=None, ncol=2, y=-0.24,
                 fontsize=6.8):
    """Legend centred directly under one panel, below its x label."""
    kw = dict(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
              frameon=False, fontsize=fontsize, handlelength=1.8,
              columnspacing=1.2, handletextpad=0.5, borderaxespad=0.0)
    if handles is None:
        return ax.legend(**kw)
    return ax.legend(handles, labels, **kw)


def save(fig, name):
    fig.savefig(path("figures", name + ".pdf"), bbox_inches="tight")
    fig.savefig(path("figures", name + ".png"), dpi=600,
                bbox_inches="tight")
    plt.close(fig)
