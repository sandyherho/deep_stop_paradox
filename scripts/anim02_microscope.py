"""Animation 2: inside a fast and a slow tissue, during and after the dive.

Each panel is a 3 mm by 3 mm slice of tissue containing gas nuclei at the
model's number density (1 per mm^3). Around each bubble the dissolved
nitrogen follows the quasi-steady diffusion field of the model,

    p(r) = P - (P - P_b) R / r,

where P is the tissue tension and P_b the nitrogen pressure in the
bubble; the color is p(r) - P, the depletion (cyan, gas flowing into a
growing bubble) or enrichment (amber, gas leaving a dissolving bubble) of
the tissue around each bubble. Bubbles are drawn
at their true radius. Top row: the 12.5 min compartment; bottom row: the
146 min compartment. Left: shallow stops; right: deep stops.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

from deep_stop_paradox import anim_style as A
from deep_stop_paradox import io
from deep_stop_paradox.bubble import ModelParams
from deep_stop_paradox.constants import BAR_PER_MSW, HALF_TIMES, P_SURF
from deep_stop_paradox.profiles import nedu_schedule

FPS = 30
COMPS = (2, 9)
L = 3000.0                 # um
NG = 260
HALO = LinearSegmentedColormap.from_list(
    "halo", ["#9ff6ff", "#2bb3d9", "#105a7a", "#0a1622", "#7a4510",
             "#d98c2b", "#ffd28a"])
VLIM = 0.4


def main():
    A.setup()
    d = io.load_cache("nedu")
    prm = ModelParams()
    sch = {w: nedu_schedule(w) for w in ("shallow", "deep")}
    t_end = max(s.runtime() for s in sch.values()) + 480.0
    warp = A.time_warp([0.0, 32.0, 245.0, t_end], [0.0, 3.0, 21.0, 30.0])
    n_frames = int(30.0 * FPS)
    rng = np.random.default_rng(11)
    # nuclei in a slab one spacing thick: jittered lattice, 1 per mm^3
    g = np.arange(500.0, L, 1000.0)
    cx, cy = np.meshgrid(g, g)
    centers = np.c_[cx.ravel(), cy.ravel()] + rng.normal(0, 140, (9, 2))
    xg = np.linspace(0, L, NG)
    X, Y = np.meshgrid(xg, xg)
    dist = np.min(np.hypot(X[..., None] - centers[:, 0],
                           Y[..., None] - centers[:, 1]), axis=-1)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.text(0.04, 0.955, "Inside the tissue: same stop time, different "
             "bubbles", fontsize=17, fontweight="bold")
    clock = fig.text(0.96, 0.955, "", ha="right", fontsize=14,
                     family="DejaVu Sans Mono")
    panels = {}
    for r_, c in enumerate(COMPS):
        for c_, w in enumerate(("shallow", "deep")):
            ax = fig.add_axes([0.06 + 0.30 * c_, 0.50 - 0.43 * r_ + 0.035,
                               0.27, 0.37])
            im = ax.imshow(np.zeros((NG, NG)), extent=(0, L, 0, L),
                           origin="lower", cmap=HALO, vmin=-VLIM,
                           vmax=VLIM, interpolation="bilinear")
            circs = []
            for (x, y) in centers:
                cc = Circle((x, y), 2.0, fc="#f7fcff", ec="white", lw=0.8,
                            alpha=0.95, zorder=3)
                hi = Circle((x, y), 0.0, fc="white", ec="none", alpha=0.8,
                            zorder=4)
                ax.add_patch(cc)
                ax.add_patch(hi)
                circs.append((cc, hi))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#2a78d6" if w == "shallow" else "#eb6834")
                s.set_linewidth(2.0)
            if r_ == 0:
                ax.set_title("shallow stops (VVal-18)" if w == "shallow"
                             else "deep stops (BVM(3))", fontsize=13,
                             color="#2a78d6" if w == "shallow"
                             else "#eb6834", fontweight="bold")
            if c_ == 0:
                ax.set_ylabel("%g min tissue" % HALF_TIMES[c], fontsize=12)
            ax.plot([L - 1150, L - 150], [150, 150], color="white", lw=2)
            ax.text(L - 650, 230, "1 mm", ha="center", fontsize=9)
            txt = ax.text(100, L - 120, "", va="top", fontsize=10,
                          family="DejaVu Sans Mono",
                          bbox=dict(fc="#06121c", ec="none", alpha=0.6))
            panels[(c, w)] = (im, circs, txt)
    # side panel: depth profiles
    axp = fig.add_axes([0.67, 0.53, 0.29, 0.37])
    axp.set_facecolor("#0a1b28")
    cur = {}
    for w, col in (("shallow", "#2a78d6"), ("deep", "#eb6834")):
        tt, z = d[w + "_t"], d[w + "_z"]
        m = tt <= sch[w].runtime() + 1e-9
        axp.plot(tt[m], -z[m], color=col, lw=1.6)
        cur[w], = axp.plot([], [], "o", color=col, ms=8, mec="white")
    axp.set_xlim(0, 250)
    axp.set_ylim(-55, 3)
    axp.set_xlabel("time (min)")
    axp.set_ylabel("depth (m)")
    axp.set_yticks([-50, -40, -30, -20, -10, 0])
    axp.set_yticklabels(["50", "40", "30", "20", "10", "0"])
    for s in ("top", "right"):
        axp.spines[s].set_visible(False)
    cax = fig.add_axes([0.67, 0.40, 0.29, 0.018])
    sm = plt.cm.ScalarMappable(cmap=HALO, norm=plt.Normalize(-VLIM, VLIM))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("dissolved N$_2$ near a bubble minus tissue tension (bar)"
                 "\ncyan: gas flowing in;  amber: gas flowing out")
    cb.outline.set_edgecolor(A.FG2)
    fig.text(0.67, 0.27, "Bubbles grow only where the tissue is\n"
             "supersaturated beyond the Laplace pressure\n"
             "of the nucleus. Deep stops keep the fast\n"
             "tissue calm but load the slow tissue, whose\n"
             "bubbles keep growing for hours after the dive.",
             va="top", fontsize=12, color=A.FG, linespacing=1.5)

    def interp(w, key, i, t):
        return np.interp(t, d[w + "_t"], d[w + "_" + key][i])

    def frame(k):
        t = float(warp(k / FPS))
        for (c, w), (im, circs, txt) in panels.items():
            z = np.interp(t, d[w + "_t"], d[w + "_z"])
            pa = P_SURF + BAR_PER_MSW * z
            P = interp(w, "P", c, t)
            R = interp(w, "R", c, t)
            pb = pa - prm.p_met + prm.two_sigma / R
            r = np.maximum(dist, R)
            # a nucleus held at its floor exchanges no gas (as in the
            # model), so its halo is switched off by the same gate
            gate = (1.0 if P > pb else
                    1.0 - np.exp(-(R - prm.r0) / prm.floor_width))
            field = -gate * (P - pb) * R / r
            im.set_data(field)
            for cc, hi in circs:
                cc.set_radius(R)
                hi.set_radius(0.35 * R)
                hi.center = (cc.center[0] - 0.3 * R, cc.center[1] + 0.3 * R)
            txt.set_text("P - P_amb = %+.2f bar\nR = %5.1f um"
                         % (P - pa, R))
        for w in cur:
            tt = min(t, sch[w].runtime())
            cur[w].set_data([tt], [-np.interp(tt, d[w + "_t"],
                                              d[w + "_z"])])
        hh, mm = divmod(int(t), 60)
        clock.set_text("t = %d h %02d min" % (hh, mm))

    A.encode(frame, fig, n_frames, "anim02_microscope", fps=FPS)


if __name__ == "__main__":
    main()
