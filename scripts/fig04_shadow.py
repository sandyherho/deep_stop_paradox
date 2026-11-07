"""Figure 4: shadow prices along the shallow schedule and the marginal
value of moving stop time deeper."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from deep_stop_paradox import io, plotting as pl
from deep_stop_paradox.constants import HALF_TIMES, MSW_PER_FSW

SHOW = [0, 2, 4, 6, 8, 10, 12]


def main():
    pl.setup()
    d = io.load_cache("nedu")
    t = d["shallow_lam_t"]
    lp, lr = d["shallow_lamP"], d["shallow_lamR"]
    cols = pl.comp_colors(16)
    fig, axs = plt.subplots(2, 2, figsize=(6.8, 6.2))
    keep = t <= 230
    for c in SHOW:
        axs[0, 0].plot(t[keep], lp[c, keep], color=cols[c], lw=1.2)
        axs[0, 1].plot(t[keep], lr[c, keep], color=cols[c], lw=1.2)
    ax = axs[0, 0]
    ax.set_xlabel("time (min)")
    ax.set_ylabel("shadow price of tension\n"
                  r"$\lambda_P$ (bar$^{-1}$)")
    ax = axs[0, 1]
    ax.set_xlabel("time (min)")
    ax.set_ylabel("shadow price of radius\n"
                  r"$\lambda_R$ (um$^{-1}$)")
    ax = axs[1, 0]
    e = d["exchange"][0]
    x = np.arange(16)
    ax.bar(x, e, width=0.75,
           color=["#e34948" if v > 0 else "#2a78d6" for v in e])
    ax.axhline(0, color=pl.INK2, lw=0.6)
    ax.set_xticks(x[::3])
    ax.set_xticklabels(["%g" % h for h in HALF_TIMES[::3]])
    ax.set_xlabel("compartment half-time (min)")
    ax.set_ylabel("change in $Q_i$ per minute moved\nfrom 10 to 70 fsw")
    ax = axs[1, 1]
    st = d["stops"][:-1] / MSW_PER_FSW
    ax.plot(st, d["exchange_beta"], "o-", color=pl.INK, ms=4, lw=1.3)
    ax.axhline(float(d["beta_pair"]), color=pl.INK2, ls=":", lw=0.8)
    ax.text(20, float(d["beta_pair"]) - 0.03, "full NEDU redistribution",
            fontsize=6.5, color=pl.INK2, va="top")
    ax.set_xlabel("depth receiving the minute (fsw)")
    ax.set_ylabel(r"critical exponent $\beta_c$")
    pl.label_panels(axs, x=-0.24)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=cols[c], lw=1.4) for c in SHOW]
    labels = ["%g min" % HALF_TIMES[c] for c in SHOW]
    pl.legend_under(axs[0, 0], handles, labels, ncol=4)
    pl.legend_under(axs[0, 1], handles, labels, ncol=4)
    pl.legend_under(axs[1, 0],
                    [Patch(color="#2a78d6"), Patch(color="#e34948")],
                    ["moving deeper lowers $Q_i$",
                     "moving deeper raises $Q_i$"])
    fig.tight_layout(h_pad=1.2)
    pl.save(fig, "fig04_shadow")


if __name__ == "__main__":
    main()
