"""Figure 2: numerical verification of the solver and its gradients."""

import matplotlib.pyplot as plt
import numpy as np

from deep_stop_paradox import io, plotting as pl
from deep_stop_paradox.constants import HALF_TIMES


def main():
    pl.setup()
    d = io.load_cache("verification")
    fig, axs = plt.subplots(2, 2, figsize=(6.8, 5.6))
    ax = axs[0, 0]
    ax.loglog(d["hs"], d["e_central"], "o-", color=pl.SHALLOW, ms=4,
              lw=1.2, label="central difference")
    ax.loglog(d["hs"], d["e_forward"], "s-", color=pl.DEEP, ms=4, lw=1.2,
              label="forward difference")
    h = d["hs"]
    i = np.argmin(np.abs(h - 0.56))
    ax.loglog(h, d["e_central"][i] * (h / h[i]) ** 2, ":", color=pl.INK2,
              lw=0.9, label="slope 2")
    ax.loglog(h, d["e_forward"][i] * (h / h[i]), "--", color=pl.INK2,
              lw=0.9, label="slope 1")
    ax.set_xlabel("finite-difference step (min)")
    ax.set_ylabel("relative difference from\ncostate derivative")
    pl.legend_under(ax, ncol=2)
    ax = axs[0, 1]
    ax.loglog(d["rtols"], d["tol_err"], "o-", color=pl.SHALLOW, ms=4,
              lw=1.2)
    ax.set_xlabel("relative tolerance")
    ax.set_ylabel("relative error in total hazard")
    ax = axs[1, 0]
    ax.semilogx(HALF_TIMES, d["cons_err"], "o", color=pl.SHALLOW, ms=4)
    ax.set_yscale("log")
    ax.set_xlabel("compartment half-time (min)")
    ax.set_ylabel("nitrogen budget residual\n(relative to largest change)")
    ax = axs[1, 1]
    Hw = d["Hw"]
    dev = np.abs(Hw - Hw[2]) / Hw[2]
    keep = dev > 0
    ax.loglog(d["widths"][keep], dev[keep], "o-", color=pl.SHALLOW, ms=4,
              lw=1.2)
    ax.set_xlabel("width of the nucleus floor (um)")
    ax.set_ylabel("relative change in total hazard\n(from 0.02 um)")
    pl.label_panels(axs, x=-0.24)
    fig.tight_layout(h_pad=1.2)
    pl.save(fig, "fig02_verification")


if __name__ == "__main__":
    main()
