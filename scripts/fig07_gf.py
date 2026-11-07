"""Figure 7: gradient-factor low (first-stop depth) in the two-phase model."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from deep_stop_paradox import io, plotting as pl
from deep_stop_paradox.bubble import hazard_weights

BETAS = (-0.5, 0.0, 0.5, 1.0)


def main():
    pl.setup()
    g = io.load_cache("gf")
    lo = g["gf_lo"] * 100
    fig, axs = plt.subplots(2, 2, figsize=(6.8, 6.0))
    cols = [pl.SEQ(0.3 + 0.22 * k) for k in range(len(BETAS))]
    for c_, dive in enumerate(("nedu", "tech45")):
        Q = g[dive + "_Q"][1]                  # GF-high 85
        tst = g[dive + "_tst"]
        ref = np.nanargmax(lo * np.isfinite(tst[1]))
        ax = axs[0, c_]
        for k, b in enumerate(BETAS):
            H = Q @ hazard_weights(b)
            ax.plot(lo, H / H[ref], "o-", color=cols[k], ms=3.5, lw=1.3)
        ax.axhline(1.0, color=pl.INK2, lw=0.6)
        ax.set_xlabel("GF-low (%)")
        ax.set_ylabel("hazard relative to\nGF-low %d" % lo[ref])
        ax = axs[1, c_]
        for a, ls in ((0, "--"), (1, "-")):
            ax.plot(lo, g[dive + "_tst"][a], ls, marker="o", ms=3.5,
                    color=pl.INK, lw=1.2)
        ax.set_xlabel("GF-low (%)")
        ax.set_ylabel("total stop time (min)")
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(lo[::2])
        ax2.set_xticklabels(["%g" % f for f in g[dive + "_first"][1][::2]],
                            fontsize=6.5)
        ax2.set_xlabel("first stop (m), GF-high 85", fontsize=7)
        ax2.grid(False)
    pl.label_panels(axs, x=-0.24, y=1.04)
    handles = [Line2D([], [], color=c, marker="o", ms=3.5) for c in cols]
    labels = [r"$\beta$ = %g" % b for b in BETAS]
    pl.legend_under(axs[0, 0], handles, labels, ncol=4)
    pl.legend_under(axs[0, 1], handles, labels, ncol=4)
    ghi = [Line2D([], [], color=pl.INK, ls="-", marker="o", ms=3.5),
           Line2D([], [], color=pl.INK, ls="--", marker="o", ms=3.5)]
    pl.legend_under(axs[1, 0], ghi, ["GF-high 85", "GF-high 70"])
    pl.legend_under(axs[1, 1], ghi, ["GF-high 85", "GF-high 70"])
    fig.tight_layout(h_pad=1.2)
    pl.save(fig, "fig07_gf")


if __name__ == "__main__":
    main()
