"""Figure 6: when does a deep stop pay? Critical weighting over air dives."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from deep_stop_paradox import io, plotting as pl


def main():
    pl.setup()
    d = io.load_cache("phase")
    D, T = d["depths"], d["bts"]
    fig = plt.figure(figsize=(7.2, 3.3))
    axs = [fig.add_axes([0.08, 0.30, 0.25, 0.60]),
           fig.add_axes([0.37, 0.30, 0.25, 0.60]),
           fig.add_axes([0.73, 0.30, 0.25, 0.60])]
    lim = 1.5
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    for ax, key in ((axs[0], "beta_half"), (axs[1], "beta_best")):
        v = d[key]
        m = ax.pcolormesh(np.arange(T.size + 1), np.arange(D.size + 1),
                          np.clip(v, -lim, lim), cmap=pl.DIV, norm=norm,
                          shading="flat")
        for i in range(D.size):
            for j in range(T.size):
                if np.isfinite(v[i, j]):
                    ax.text(j + 0.5, i + 0.5, "%.2f" % v[i, j],
                            ha="center", va="center", fontsize=5.6,
                            color=pl.INK)
        ax.set_xticks(np.arange(T.size) + 0.5)
        ax.set_xticklabels(["%d" % t for t in T])
        ax.set_yticks(np.arange(D.size) + 0.5)
        ax.set_yticklabels(["%d" % z for z in D])
        ax.set_xlabel("bottom time (min)")
        ax.grid(False)
    axs[0].set_ylabel("bottom depth (m)")
    cax = fig.add_axes([0.08, 0.09, 0.54, 0.03])
    cb = fig.colorbar(m, cax=cax, orientation="horizontal")
    cb.set_label(r"critical exponent $\beta_c$; the stop lowers the "
                 r"hazard only for $\beta > \beta_c$", fontsize=7.5)
    ax = axs[2]
    tst = d["tst"]
    for i in range(D.size):
        ax.plot(tst[i], d["beta_half"][i], "o-",
                color=pl.SEQ(0.25 + 0.75 * i / (D.size - 1)), ms=3,
                lw=1.1, label="%d m" % D[i])
    ax.axhline(0, color=pl.INK2, lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("GF 90/85 total stop time (min)")
    ax.set_ylabel(r"$\beta_c$, half-depth stop")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4,
              fontsize=6, handlelength=1.4, columnspacing=0.8)
    axs[1].set_yticklabels([])
    pl.label_panels(axs, x=-0.2, y=1.03)
    pl.save(fig, "fig06_phase")


if __name__ == "__main__":
    main()
