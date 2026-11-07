"""Figure 0: geometry of the two-phase model and the numerical scheme.

(a) One perfusion compartment: tissue at nitrogen tension P, fed by
    arterial blood, holding gas nuclei of radius R stabilized at r0.
(b) The quasi-steady diffusion field around one bubble, for a growing
    and a dissolving bubble.
(c) Discretization of a dive into linear depth segments, with the
    forward integration of states and segment transition matrices and
    the backward sweep of costates.
(d) The chain from the segment integrations to the stop-time gradients,
    the critical weighting and the redistribution search.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Rectangle)

from deep_stop_paradox import plotting as pl

TISSUE = "#f3ece4"
BLOOD = "#e8b4b2"
GAS = "#dbe9fb"


def arrow(ax, p, q, color=pl.INK, lw=1.0, style="-|>", ms=8, ls="-",
          **kw):
    a = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms,
                        color=color, lw=lw, linestyle=ls, **kw)
    ax.add_patch(a)
    return a


def panel_geometry(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.3, 1.6), 9.4, 5.1,
                                boxstyle="round,pad=0,rounding_size=0.4",
                                fc=TISSUE, ec=pl.INK2, lw=0.8))
    ax.add_patch(FancyBboxPatch((0.3, 0.3), 9.4, 1.0,
                                boxstyle="round,pad=0,rounding_size=0.3",
                                fc=BLOOD, ec=pl.INK2, lw=0.8))
    ax.text(5.0, 0.8, "capillary blood", ha="center", va="center",
            fontsize=6.5, color=pl.INK)
    ax.text(0.55, 0.8, r"arterial $P_{art}\rightarrow$", fontsize=6,
            va="center")
    ax.text(9.45, 0.8, r"$\rightarrow$ venous $P$", fontsize=6,
            va="center", ha="right")
    arrow(ax, (3.0, 1.35), (3.0, 2.2), lw=0.9, style="<|-|>", ms=7)
    ax.text(3.15, 1.75, r"perfusion $k_i=\ln 2/\tau_i$", fontsize=6,
            va="center")
    ax.text(0.6, 6.25, r"tissue, N$_2$ tension $P_i(t)$", fontsize=6.8)
    rng = np.random.default_rng(4)
    for _ in range(9):
        x, y = rng.uniform(0.9, 4.6), rng.uniform(2.7, 4.8)
        ax.add_patch(Circle((x, y), 0.07, fc="white", ec=pl.INK2, lw=0.5))
    ax.text(0.9, 2.15, r"nuclei, density $N$, floor radius $r_0$",
            fontsize=6)
    # enlarged bubble
    c, R = (7.1, 3.9), 1.15
    for k, a in enumerate((2.15, 1.8, 1.48)):
        ax.add_patch(Circle(c, a, fc="none", ec=pl.SHALLOW, lw=0.6,
                            ls=":", alpha=0.6 + 0.1 * k))
    ax.add_patch(Circle(c, R, fc=GAS, ec=pl.INK, lw=1.0))
    ax.add_patch(Circle(c, 0.35, fc="none", ec=pl.INK2, lw=0.7, ls="--"))
    ax.text(c[0] - 0.55, c[1] + 0.05, r"$r_0$", fontsize=6, ha="center",
            va="center")
    ax.plot([c[0], c[0] + R * np.cos(0.6)], [c[1], c[1] + R * np.sin(0.6)],
            color=pl.INK, lw=0.7)
    ax.text(c[0] + 0.62, c[1] + 0.68, r"$R$", fontsize=7)
    ax.text(c[0], c[1] - 0.6, "N$_2$ at $P_b$\n+ $p_m$", fontsize=5.8,
            ha="center", va="center")
    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False) + 0.2:
        p = (c[0] + 2.1 * np.cos(ang), c[1] + 2.1 * np.sin(ang))
        q = (c[0] + 1.25 * np.cos(ang), c[1] + 1.25 * np.sin(ang))
        if 1.6 < p[1] < 6.6 and 0.4 < p[0] < 9.6:
            arrow(ax, p, q, color=pl.SHALLOW, lw=0.7, ms=6)
    ax.text(0.6, 5.7, r"$P_b = P_{amb} - p_m + 2\sigma/R$", fontsize=6.3)
    ax.text(0.6, 5.15, r"$p_m$: O$_2$ + CO$_2$ + H$_2$O in the bubble",
            fontsize=5.8, color=pl.INK2)


def panel_field(ax):
    r = np.linspace(1.0, 8.0, 300)
    for P, Pb, col, lab in ((1.2, 0.9, pl.SHALLOW, "growing: $P > P_b$"),
                            (0.8, 1.0, pl.DEEP, "dissolving: $P < P_b$")):
        ax.plot(r, P - (P - Pb) / r, color=col, lw=1.6, label=lab)
        ax.axhline(P, color=col, lw=0.6, ls=":")
    ax.axvspan(0.6, 1.0, color=GAS)
    ax.text(0.8, 1.14, "gas", fontsize=6.5, ha="center", rotation=90)
    ax.set_xlim(0.6, 8)
    ax.set_ylim(0.75, 1.25)
    ax.set_xlabel("distance from bubble centre, $r/R$")
    ax.set_ylabel(r"dissolved N$_2$, $p(r)$ (bar)")
    ax.text(7.9, 1.205, r"far field $P$", fontsize=6, ha="right",
            color=pl.SHALLOW)
    ax.text(7.9, 0.79, r"far field $P$", fontsize=6, ha="right",
            color=pl.DEEP, va="top")
    pl.legend_under(ax, ncol=2)


def panel_segments(ax):
    tk = np.array([0, 2, 12, 14, 18, 19, 26, 27, 38, 39, 50])
    zk = np.array([0, 30, 30, 12, 12, 9, 9, 6, 6, 3, 3])
    ax.plot(tk, -zk, color=pl.INK, lw=1.5)
    ax.plot([50, 51.5], [-3, 0], color=pl.INK, lw=1.5)
    ax.plot([51.5, 60], [0, 0], color=pl.INK, lw=1.5, ls=(0, (4, 2)))
    for t in tk[1:]:
        ax.plot([t, t], [-31.5, -30.5], color=pl.INK2, lw=0.6)
    ax.text(3.2, -12, "descent", fontsize=6, rotation=-80, ha="center")
    ax.text(7, -28.5, "bottom", fontsize=6, ha="center")
    ax.text(16.2, -21, "ascent", fontsize=6, rotation=-70, ha="center")
    for (a, b, z, lab) in ((14, 18, -12, r"$\tau_1$"),
                           (19, 26, -9, r"$\tau_2$"),
                           (27, 38, -6, r"$\tau_3$"),
                           (39, 50, -3, r"$\tau_4$")):
        ax.annotate("", (a, z + 1.6), (b, z + 1.6),
                    arrowprops=dict(arrowstyle="<->", lw=0.6,
                                    color=pl.OPTIMAL))
        ax.text(0.5 * (a + b), z + 2.3, lab, fontsize=7, ha="center",
                color=pl.OPTIMAL)
    ax.text(56, 1.5, "24 h at\nthe surface", fontsize=6, ha="center")
    arrow(ax, (2, -36), (58, -36), color=pl.SHALLOW, lw=1.1)
    ax.text(30, -35.2, r"forward: states $y=(P,R,Q)$ and "
            r"transition matrices $\Phi_k$ per segment", fontsize=6,
            ha="center", va="bottom", color=pl.SHALLOW)
    arrow(ax, (58, -41), (2, -41), color=pl.DEEP, lw=1.1)
    ax.text(30, -41.8, r"backward: costates "
            r"$\lambda_k^{\top} = \lambda_{k+1}^{\top}\Phi_k$, "
            r"$\lambda_K = \partial Q/\partial y$",
            fontsize=6, ha="center", va="top", color=pl.DEEP)
    ax.set_xlim(-1, 61)
    ax.set_ylim(-46, 5)
    ax.set_xlabel("time")
    ax.set_ylabel("depth")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def panel_flow(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    boxes = [
        (5.0, 9.0, "Segment integration (LSODA, rtol 1e-7 to 1e-9):\n"
                   r"$\dot y = f(y,t)$, $\dot\Phi = J\,\Phi$, "
                   "compartments decoupled", pl.SHALLOW),
        (5.0, 6.75, "Backward sweep of costates\n"
                    r"$\lambda_P(t)$, $\lambda_R(t)$: shadow prices of "
                    "tension and radius", pl.DEEP),
        (5.0, 4.5, "Exact stop-time gradient\n"
                   r"$\partial Q_i/\partial\tau_j = \lambda_P f_P + "
                   r"\lambda_R f_R + f_Q$ at the end of stop $j$",
         pl.INK2),
        (2.45, 1.6, "Critical weighting $\\beta_c$:\n"
                    r"root of $\Sigma_i\, E_i\, e^{-\beta x_i}=0$" + "\n"
                    "(unique if one sign change)", pl.INK2),
        (7.55, 1.6, "Redistribution at fixed\n"
                    r"$\Sigma_j\, \tau_j = T$: gradient-guided" + "\n"
                    "exchanges, three starts", pl.OPTIMAL),
    ]
    for x, y, txt, col in boxes:
        w = 9.4 if x == 5.0 else 4.6
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.85), w, 1.7,
                                    boxstyle="round,pad=0.02,"
                                    "rounding_size=0.25", fc="white",
                                    ec=col, lw=1.1))
        ax.text(x, y, txt, ha="center", va="center", fontsize=6.1,
                linespacing=1.35)
    for y0, y1 in ((8.15, 7.6), (5.9, 5.35)):
        arrow(ax, (5.0, y0), (5.0, y1), lw=1.0)
    arrow(ax, (4.0, 3.65), (2.6, 2.45), lw=1.0)
    arrow(ax, (6.0, 3.65), (7.4, 2.45), lw=1.0)
    ax.add_patch(Rectangle((0, 0), 10, 10, fc="none", ec="none"))


def main():
    pl.setup()
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.6),
                            gridspec_kw={"height_ratios": [1.0, 1.0]})
    panel_geometry(axs[0, 0])
    panel_field(axs[0, 1])
    panel_segments(axs[1, 0])
    panel_flow(axs[1, 1])
    pl.label_panels(axs, x=-0.06, y=1.02)
    fig.tight_layout(h_pad=1.5, w_pad=1.2)
    pl.save(fig, "fig00_schematic")


if __name__ == "__main__":
    main()
