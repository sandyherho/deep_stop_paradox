"""Figure 5: where 174 min of stops should go, against the weighting."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from deep_stop_paradox import io, plotting as pl
from deep_stop_paradox.bubble import hazard_weights


def main():
    pl.setup()
    o = io.load_cache("optimal")
    n = io.load_cache("nedu")
    b = o["betas"]
    X = o["X"]
    zf = o["grid_fsw"]
    Qs, Qd = o["Qs"], o["Qd"]
    fig, axs = plt.subplots(2, 2, figsize=(6.8, 6.4))
    # (a) hazard along the straight path shallow -> deep, all weightings
    ax = axs[0, 0]
    bb = np.linspace(-1.0, 2.0, 121)
    th = o["thetas"]
    Hp = np.array([[np.dot(hazard_weights(x), q) for q in o["Qpath"]]
                   for x in bb])
    rel = np.log10(Hp / Hp[:, :1])
    lim = np.max(np.abs(rel))
    m = ax.pcolormesh(th, bb, rel, cmap=pl.DIV,
                      norm=TwoSlopeNorm(0.0, -lim, lim), shading="gouraud")
    best = th[np.argmin(Hp, axis=1)]
    ax.plot(best, bb, color=pl.INK, lw=1.4)
    ax.axvline(1.0, color=pl.INK2, ls="--", lw=0.8)
    ax.text(1.03, -0.9, "NEDU\ndeep", fontsize=6.5, color=pl.INK2)
    ax.grid(False)
    ax.set_xlabel(r"path parameter $\theta$ (0 shallow, 1 deep stops)")
    ax.set_ylabel(r"weighting exponent $\beta$")
    cb = fig.colorbar(m, ax=ax, pad=0.02)
    cb.set_label(r"log$_{10}$ hazard / shallow stops", fontsize=7)
    # (b) optimal stop times
    ax = axs[0, 1]
    edges = np.r_[b[0] - 0.25, 0.5 * (b[1:] + b[:-1]), b[-1] + 0.25]
    zedges = np.r_[zf + 5, zf[-1] - 5]
    mesh = ax.pcolormesh(edges, zedges, X.T, cmap=pl.SEQ, vmin=0,
                         vmax=np.max(X), shading="flat")
    ax.set_ylim(zedges[-1], zedges[0])
    ax.grid(False)
    ax.set_xlabel(r"weighting exponent $\beta$")
    ax.set_ylabel("stop depth (fsw)")
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("best stop time found (min)", fontsize=7)
    # (c) hazard of the optimum and of the deep schedule
    ax = axs[1, 0]
    hs = np.array([np.dot(hazard_weights(x), Qs) for x in b])
    hd = np.array([np.dot(hazard_weights(x), Qd) for x in b])
    if "H_starts" in o:
        rel = o["H_starts"] / hs[:, None]
        ax.vlines(b, rel.min(axis=1), rel.max(axis=1), color=pl.OPTIMAL,
                  lw=3, alpha=0.35)
    ax.plot(b, o["H"] / hs, "o-", color=pl.OPTIMAL, ms=4)
    ax.plot(b, hd / hs, "--", color=pl.DEEP)
    ax.axhline(1.0, color=pl.SHALLOW, ls="--", lw=1)
    ax.axvline(float(n["beta_pair"]), color=pl.INK2, ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_yticks([0.15, 0.3, 0.5, 1.0, 2.0, 3.0])
    ax.set_yticklabels(["0.15", "0.3", "0.5", "1", "2", "3"])
    ax.minorticks_off()
    ax.set_xlabel(r"weighting exponent $\beta$")
    ax.set_ylabel("hazard relative to shallow stops")
    # (d) deep-stop skew of the optimum
    ax = axs[1, 1]
    ax.plot(b, o["dss"], "o-", color=pl.OPTIMAL, ms=4)
    ax.axhline(float(o["dss_shallow"]), color=pl.SHALLOW, ls="--", lw=1)
    ax.axhline(float(o["dss_deep"]), color=pl.DEEP, ls="--", lw=1)
    ax.axvline(float(n["beta_pair"]), color=pl.INK2, ls=":", lw=0.8)
    ax.set_xlabel(r"weighting exponent $\beta$")
    ax.set_ylabel("deep-stop skew (bar min)")
    pl.label_panels(axs, x=-0.22)
    pl.legend_under(axs[0, 0],
                    [Line2D([], [], color=pl.INK, lw=1.4),
                     Line2D([], [], color=pl.INK2, ls="--")],
                    ["best point on the path", "NEDU deep stops"])
    common = [Line2D([], [], color=pl.OPTIMAL, marker="o", ms=4),
              Line2D([], [], color=pl.SHALLOW, ls="--"),
              Line2D([], [], color=pl.DEEP, ls="--"),
              Line2D([], [], color=pl.INK2, ls=":")]
    clab = ["best distribution found", "NEDU shallow stops",
            "NEDU deep stops", r"$\beta_c$ of the NEDU pair"]
    pl.legend_under(axs[1, 0],
                    common[:1] + [Line2D([], [], color=pl.OPTIMAL, lw=3,
                                         alpha=0.35)] + common[1:],
                    clab[:1] + ["spread over three starts"] + clab[1:])
    pl.legend_under(axs[1, 1], common, clab)
    fig.tight_layout(h_pad=1.2)
    pl.save(fig, "fig05_optimal")


if __name__ == "__main__":
    main()
