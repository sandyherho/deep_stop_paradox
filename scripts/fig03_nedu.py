"""Figure 3: compartment hazards and the critical weighting for NEDU."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from deep_stop_paradox import io, plotting as pl
from deep_stop_paradox.bubble import hazard_weights
from deep_stop_paradox.constants import HALF_TIMES


def main():
    pl.setup()
    d = io.load_cache("nedu")
    Qs, Qd = d["shallow_Q"], d["deep_Q"]
    fig, axs = plt.subplots(2, 2, figsize=(6.8, 6.0))
    ax = axs[0, 0]
    ax.semilogx(HALF_TIMES, Qs, "o-", color=pl.SHALLOW, ms=4, lw=1.3)
    ax.semilogx(HALF_TIMES, Qd, "s-", color=pl.DEEP, ms=4, lw=1.3)
    ax.set_xlabel("compartment half-time (min)")
    ax.set_ylabel("hazard integral $Q_i$")
    ax = axs[0, 1]
    dq = Qd - Qs
    cols = ["#e34948" if v > 0 else "#2a78d6" for v in dq]
    x = np.arange(16)
    ax.bar(x, dq, color=cols, width=0.75)
    ax.axhline(0, color=pl.INK2, lw=0.6)
    ax.set_xticks(x[::3])
    ax.set_xticklabels(["%g" % h for h in HALF_TIMES[::3]])
    ax.set_xlabel("compartment half-time (min)")
    ax.set_ylabel("$Q_i$(deep) $-$ $Q_i$(shallow)")
    ax = axs[1, 0]
    b = d["betas"]
    rr, lo, hi = d["rr_obs"]
    ax.axhspan(lo, hi, color=pl.NEUTRAL, zorder=0)
    ax.axhline(rr, color=pl.INK2, lw=0.8, ls="--")
    ax.axhline(1.0, color=pl.INK2, lw=0.6)
    ax.semilogy(b, d["ratio"], color=pl.INK, lw=1.6)
    bc = float(d["beta_pair"])
    ax.axvline(bc, color=pl.INK2, lw=0.8, ls=":")
    ax.text(bc + 0.05, 6.0, r"$\beta_c$ = %.2f" % bc, fontsize=7,
            color=pl.INK)
    ax.set_xlim(b[0], b[-1])
    ax.set_ylim(0.1, 20)
    ax.set_xlabel(r"hazard weighting exponent $\beta$")
    ax.set_ylabel("hazard ratio, deep / shallow")
    ax = axs[1, 1]
    for k, bb in enumerate((-1.0, 0.0, 1.0, 2.0)):
        ax.loglog(HALF_TIMES, hazard_weights(bb), "-",
                  color=pl.SEQ(0.3 + 0.22 * k), lw=1.4)
        ax.text(HALF_TIMES[-1] * 1.08, hazard_weights(bb)[-1],
                r"$\beta$=%g" % bb, fontsize=6.5, va="center",
                color=pl.INK2)
    ax.set_xlim(4, 1400)
    ax.set_xlabel("compartment half-time (min)")
    ax.set_ylabel(r"hazard weight $g_i(\beta)$")
    pl.label_panels(axs, x=-0.2)
    pl.legend_under(axs[0, 0],
                    [Line2D([], [], color=pl.SHALLOW, marker="o", ms=4),
                     Line2D([], [], color=pl.DEEP, marker="s", ms=4)],
                    ["shallow stops", "deep stops"])
    pl.legend_under(axs[0, 1],
                    [Patch(color="#2a78d6"), Patch(color="#e34948")],
                    ["deep stops lower", "deep stops higher"])
    pl.legend_under(axs[1, 0],
                    [Line2D([], [], color=pl.INK, lw=1.6),
                     Line2D([], [], color=pl.INK2, ls="--"),
                     Patch(color=pl.NEUTRAL),
                     Line2D([], [], color=pl.INK2, ls=":")],
                    ["model ratio", "observed ratio", "95% interval",
                     r"critical $\beta_c$"])
    fig.tight_layout(h_pad=1.2)
    pl.save(fig, "fig03_nedu")


if __name__ == "__main__":
    main()
