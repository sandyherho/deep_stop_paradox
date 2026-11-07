"""Figure 9: the nucleation threshold decides the deep-stop question."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from deep_stop_paradox import io, plotting as pl


def main():
    pl.setup()
    d = io.load_cache("threshold")
    s = io.load_cache("sensitivity")
    n = io.load_cache("nedu")
    rr, lo, hi = n["rr_obs"]
    fam = d["family"]
    t = d["threshold"]
    fig, axs = plt.subplots(2, 2, figsize=(6.8, 6.2))
    styles = ((0, "o", "-"), (1, "s", "--"))
    ax = axs[0, 0]
    ax.axhspan(lo, hi, color=pl.NEUTRAL, zorder=0)
    ax.axhline(rr, color=pl.INK2, ls="--", lw=0.8)
    ax.axhline(1.0, color=pl.INK2, lw=0.6)
    for f, mk, ls in styles:
        m = fam == f
        ax.semilogy(t[m], d["ratio0"][m], ls, marker=mk, ms=3.5,
                    color=pl.INK, mfc="white" if f else pl.INK, lw=1.1)
    ax.set_ylim(0.02, 15)
    ax.set_xlabel("growth threshold (bar)")
    ax.set_ylabel("hazard ratio deep / shallow,\nequal weights")
    ax = axs[0, 1]
    ax.axhline(float(n["beta_obs"][1]), color=pl.INK2, ls="--", lw=0.8)
    ax.axhline(0.0, color=pl.INK2, lw=0.6)
    for f, mk, ls in styles:
        m = fam == f
        ax.plot(t[m], d["beta_c"][m], ls, marker=mk, ms=3.5, color=pl.INK,
                mfc="white" if f else pl.INK, lw=1.1)
    ax.set_xlabel("growth threshold (bar)")
    ax.set_ylabel(r"critical exponent $\beta_c$")
    ax = axs[1, 0]
    m = fam == 0
    ax.step(t[m], d["slow_shallow"][m], where="mid", color=pl.SHALLOW)
    ax.step(t[m], d["slow_deep"][m], where="mid", color=pl.DEEP)
    ax.set_xlabel("growth threshold (bar)")
    ax.set_ylabel("compartments of 54 min\nor slower with bubbles")
    ax = axs[1, 1]
    ax.axhline(1.0, color=pl.INK2, lw=0.6)
    ax.scatter(s["threshold"], s["ratio0"], s=9, color=pl.SHALLOW,
               alpha=0.8, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("growth threshold (bar)")
    ax.set_ylabel("hazard ratio deep / shallow,\nequal weights")
    pl.label_panels(axs, x=-0.22)
    routes = [Line2D([], [], color=pl.INK, marker="o", ms=3.5),
              Line2D([], [], color=pl.INK, marker="s", ms=3.5,
                     mfc="white", ls="--")]
    rlab = ["varying nucleus radius", "varying surface tension"]
    pl.legend_under(axs[0, 0],
                    routes + [Line2D([], [], color=pl.INK2, ls="--"),
                              Patch(color=pl.NEUTRAL)],
                    rlab + ["observed ratio", "95% interval"])
    pl.legend_under(axs[0, 1],
                    routes + [Line2D([], [], color=pl.INK2, ls="--")],
                    rlab + ["bound consistent with the trial"])
    pl.legend_under(axs[1, 0],
                    [Line2D([], [], color=pl.SHALLOW),
                     Line2D([], [], color=pl.DEEP)],
                    ["shallow stops", "deep stops"])
    fig.tight_layout(h_pad=1.2)
    pl.save(fig, "fig09_threshold")


if __name__ == "__main__":
    main()
