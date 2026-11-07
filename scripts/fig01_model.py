"""Figure 1: the two NEDU schedules in the two-phase model."""

import matplotlib.pyplot as plt
import numpy as np

from deep_stop_paradox import io, plotting as pl
from deep_stop_paradox.constants import (BAR_PER_MSW, HALF_TIMES,
                                         MSW_PER_FSW, P_SURF)
from deep_stop_paradox.bubble import ModelParams

FAST, SLOW = 2, 9          # 12.5 min and 146 min compartments


def main():
    pl.setup()
    d = io.load_cache("nedu")
    prm = ModelParams()
    fig, axs = plt.subplots(2, 2, figsize=(6.8, 5.2))
    handles = []
    for w, col in (("shallow", pl.SHALLOW), ("deep", pl.DEEP)):
        t, z = d[w + "_t"], d[w + "_z"]
        P, R, Q = d[w + "_P"], d[w + "_R"], d[w + "_Qt"]
        ins = t <= d[w + "_runtime"] + 1e-9
        h, = axs[0, 0].plot(t[ins], -z[ins] / MSW_PER_FSW, color=col)
        handles.append(h)
        pa = P_SURF + BAR_PER_MSW * z
        for c, ls in ((FAST, "-"), (SLOW, "--")):
            axs[0, 1].plot(t, P[c] - pa, ls, color=col, lw=1.3)
            axs[1, 0].plot(t, R[c], ls, color=col, lw=1.3)
        axs[1, 1].plot(t, Q.sum(axis=0), color=col)
    ax = axs[0, 0]
    ax.set_xlim(0, 230)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("depth (fsw)")
    ax.set_yticks(np.arange(-175, 1, 25))
    ax.set_yticklabels([str(-v) for v in np.arange(-175, 1, 25)])
    ax = axs[0, 1]
    ax.axhline(0, color=pl.INK2, lw=0.6)
    ax.axhline(prm.threshold, color=pl.INK2, lw=0.6, ls=":")
    ax.text(225, prm.threshold + 0.03, "growth threshold", ha="right",
            fontsize=6.5, color=pl.INK2)
    ax.set_xlim(0, 230)
    ax.set_ylim(-1.0, 1.6)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("tissue N$_2$ minus ambient (bar)")
    ax = axs[1, 0]
    ax.set_yscale("log")
    ax.set_xlim(0, 700)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("bubble radius (um)")
    ax = axs[1, 1]
    ax.set_xlim(0, 1700)
    ax.set_xlabel("time (min)")
    ax.set_ylabel(r"$\sum_i Q_i$ (units of $v_{ref}$ min)")
    pl.label_panels(axs, x=-0.2)
    from matplotlib.lines import Line2D
    sched = ["shallow stops (VVal-18)", "deep stops (BVM(3))"]
    pl.legend_under(axs[0, 0], handles, sched)
    comp = [Line2D([], [], color=pl.SHALLOW, ls="-"),
            Line2D([], [], color=pl.DEEP, ls="-"),
            Line2D([], [], color=pl.INK2, ls="-"),
            Line2D([], [], color=pl.INK2, ls="--")]
    clab = sched + ["%.1f min compartment" % HALF_TIMES[FAST],
                    "%.0f min compartment" % HALF_TIMES[SLOW]]
    pl.legend_under(axs[0, 1], comp, clab)
    pl.legend_under(axs[1, 0], comp, clab)
    pl.legend_under(axs[1, 1], handles, sched)
    fig.set_size_inches(6.8, 6.2)
    fig.tight_layout(h_pad=1.2)
    pl.save(fig, "fig01_model")


if __name__ == "__main__":
    main()
