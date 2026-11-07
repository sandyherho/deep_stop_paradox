"""Animation 1: two divers, one total stop time, two distributions.

Left: the NEDU 170 fsw / 30 min air dive on a shot line. The diver on
the left follows the shallow-stops schedule, the one on the right the
deep-stops schedule; their depths are the schedules. Right: the 16
tissue compartments of each diver, colored by nitrogen tension minus
ambient pressure, with the bubble radius from the two-phase model drawn
to a common scale. Bottom right: the hazard integral with equal
compartment weights. The animation continues for 24 hours after
surfacing, when most of the slow-tissue bubbles grow and dissolve.

Divers are drawn about twice their real size; exhaled bubbles expand
by Boyle's law as they rise, at an exaggerated speed.
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Polygon, Rectangle

from deep_stop_paradox import anim_style as A
from deep_stop_paradox import io
from deep_stop_paradox.constants import (BAR_PER_MSW, HALF_TIMES,
                                         MSW_PER_FSW, P_SURF)
from deep_stop_paradox.diver_art import Diver
from deep_stop_paradox.profiles import nedu_schedule

FPS = 30
VT = 0.8
TILE = LinearSegmentedColormap.from_list(
    "tile", ["#2a78d6", "#16365a", "#18222c", "#5a1d1d", "#e34948"])
T_TAIL = 24.0 * 60.0


def stop_info(sched, t):
    """Name of the current phase and the time left in it."""
    tt = 0.0
    for g in sched.segments(include_tail=False):
        if tt + g.dt >= t and g.dt > 0:
            left = tt + g.dt - t
            if g.kind == "hold":
                return "stop %d fsw, %4.1f min left" % (
                    round(g.z0 / MSW_PER_FSW), left)
            return {"descent": "descending", "bottom": "bottom time",
                    "ascent": "ascending"}[g.kind]
        tt += g.dt
    return "on the surface"


def main():
    A.setup()
    d = io.load_cache("nedu")
    sch = {w: nedu_schedule(w) for w in ("shallow", "deep")}
    t_end = max(sch[w].runtime() for w in sch) + T_TAIL
    warp = A.time_warp([0.0, 32.0, 245.0, t_end, t_end],
                       [0.0, 3.0, 21.0, 30.0, 33.0])
    n_frames = int(33.0 * FPS)

    def series(w, key):
        return d[w + "_t"], d[w + "_" + key]

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 9), dpi=100)
    axw = fig.add_axes([0.035, 0.06, 0.43, 0.86])
    zmax = 56.0
    axw.set_xlim(-8, 8)
    axw.set_ylim(-zmax, 4.0)
    A.water_column(axw, -9, 9, 0.0, -zmax)
    axw.add_patch(Rectangle((-9, 0), 18, 4, fc="#0f2a3f", ec="none",
                            zorder=0))
    axw.set_xticks([])
    ticks = np.arange(0, 56, 10)
    axw.set_yticks(-ticks)
    axw.set_yticklabels(["%d m" % v for v in ticks])
    for s in axw.spines.values():
        s.set_visible(False)
    shafts = A.LightShafts(axw, np.linspace(-8, 8, 7), zmax)
    snow = A.Snow(axw, -9, 9, 0, -zmax)
    xs = np.linspace(-9, 9, 300)
    surf, = axw.fill(np.r_[xs, xs[::-1]], np.r_[A.surface_wave(xs, 0),
                                                np.full(300, 4.0)],
                     fc="#0f2a3f", ec="#bfe9f7", lw=1.2, zorder=3)
    # boat and shot line
    hull = Polygon([[-2.6, 0.2], [2.6, 0.2], [2.0, -0.6], [-2.0, -0.6]],
                   fc="#d9dde0", ec="#7a8389", lw=0.8, zorder=4)
    axw.add_patch(hull)
    axw.plot([0, 0], [-0.6, -zmax + 1.2], color="#e9d8a6", lw=1.4,
             zorder=3, alpha=0.9)
    axw.add_patch(Rectangle((-0.5, -zmax + 0.4), 1.0, 0.8, fc="#555",
                            zorder=3))
    divers = {"shallow": Diver(axw, "#2a78d6", 2.6, 12),
              "deep": Diver(axw, "#eb6834", 2.6, 12)}
    xpos = {"shallow": -3.9, "deep": 3.9}
    flip = {"shallow": False, "deep": True}
    lbl = {}
    for w in divers:
        lbl[w] = axw.text(xpos[w], 0, "", ha="center", va="top",
                          fontsize=10.5, color=A.FG, zorder=20)
    puffs = {w: [] for w in divers}
    puff_sc = {w: axw.scatter([], [], s=[], facecolors="none",
                              edgecolors="#e8f8ff", lw=0.8, alpha=0.8,
                              zorder=11) for w in divers}
    fig.text(0.035, 0.955, "Same 174 min of stops, spent two ways",
             fontsize=17, fontweight="bold", color=A.FG)
    clock = fig.text(0.035, 0.925, "", fontsize=14, ha="left",
                     color=A.FG2, family="DejaVu Sans Mono")
    # tissue panels
    tiles = {}
    for k, (w, x0) in enumerate((("shallow", 0.505), ("deep", 0.745))):
        ax = fig.add_axes([x0, 0.36, 0.23, 0.56])
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4.35)
        ax.set_aspect("equal")
        ax.axis("off")
        name = ("shallow stops (VVal-18)" if w == "shallow"
                else "deep stops (BVM(3))")
        ax.text(2.0, 4.25, name, ha="center", va="center", fontsize=12.5,
                color="#2a78d6" if w == "shallow" else "#eb6834",
                fontweight="bold")
        circ, bub = [], []
        for i in range(16):
            r_, c_ = divmod(i, 4)
            cx, cy = 0.5 + c_, 3.5 - r_
            c = Circle((cx, cy), 0.44, fc="#333", ec="#5b6b78", lw=0.8)
            ax.add_patch(c)
            b = Circle((cx, cy), 0.0, fc="#f4fbff", ec="#ffffff", lw=0.6,
                       alpha=0.92)
            ax.add_patch(b)
            ax.text(cx, cy - 0.47, "%g min" % HALF_TIMES[i], ha="center",
                    va="top", fontsize=7.5, color=A.FG2)
            circ.append(c)
            bub.append(b)
        tiles[w] = (circ, bub)
    cax = fig.add_axes([0.53, 0.325, 0.42, 0.012])
    sm = plt.cm.ScalarMappable(cmap=TILE,
                               norm=plt.Normalize(-VT, VT))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("tissue N$_2$ tension minus ambient pressure (bar);"
                 "  white disks: bubble radius, full tile = 150 um",
                 fontsize=10)
    cb.outline.set_edgecolor(A.FG2)
    # hazard panel
    axh = fig.add_axes([0.535, 0.06, 0.41, 0.19])
    axh.set_facecolor("#0a1b28")
    for s in ("top", "right"):
        axh.spines[s].set_visible(False)
    hl = {}
    hmax = 0.0
    for w, col in (("shallow", "#2a78d6"), ("deep", "#eb6834")):
        t, Q = series(w, "Qt")
        H = Q.sum(axis=0)
        hmax = max(hmax, H.max())
        hl[w], = axh.plot([], [], color=col, lw=2.2)
    axh.set_xlim(0, t_end)
    axh.set_ylim(0, hmax * 1.08)
    axh.set_xlabel("time since leaving the surface (min)")
    axh.set_ylabel("hazard (equal weights)")
    ratio_txt = axh.text(0.02, 0.88, "", transform=axh.transAxes,
                         fontsize=11, color=A.FG)
    rng = np.random.default_rng(7)

    def state(w, t):
        tt = d[w + "_t"]
        z = np.interp(t, tt, d[w + "_z"])
        P = np.array([np.interp(t, tt, d[w + "_P"][i]) for i in range(16)])
        R = np.array([np.interp(t, tt, d[w + "_R"][i]) for i in range(16)])
        H = np.interp(t, tt, d[w + "_Qt"].sum(axis=0))
        return z, P, R, H

    def frame(k):
        s = k / FPS
        t = float(warp(s))
        shafts.update(0.8 * s)
        snow.step(0.6)
        surf.set_xy(np.c_[np.r_[xs, xs[::-1]],
                          np.r_[A.surface_wave(xs, 1.5 * s),
                                np.full(300, 4.0)]])
        hs = {}
        for w, dv in divers.items():
            z, P, R, H = state(w, t)
            hs[w] = H
            on_surface = t >= sch[w].runtime() - 1e-6
            y = -z if not on_surface else 0.25 + 0.15 * np.sin(2 * s)
            tilt = 0.0
            dv.place(xpos[w], y, tilt, flip[w],
                     kick=14 * np.sin(4.0 * s + (w == "deep")))
            lbl[w].set_position((xpos[w], y - 1.4))
            lbl[w].set_text("%.1f m\n%s" % (z, stop_info(sch[w], t)))
            # exhaled gas: released at depth, expands as P(z0)/P(z)
            if not on_surface and k % 6 == (0 if w == "shallow" else 3):
                mx, my = dv.mouth(xpos[w], y, tilt, flip[w])
                for _ in range(3):
                    puffs[w].append([mx + rng.normal(0, 0.12),
                                     my + rng.normal(0, 0.1),
                                     -my, rng.uniform(14, 34)])
            keep = []
            for p in puffs[w]:
                p[1] += 0.35 + 0.004 * (p[1] + zmax)
                p[0] += 0.03 * np.sin(p[1])
                if p[1] < -0.2:
                    keep.append(p)
            puffs[w] = keep
            if keep:
                arr = np.array(keep)
                boyle = ((P_SURF + BAR_PER_MSW * arr[:, 2]) /
                         (P_SURF + BAR_PER_MSW * np.maximum(-arr[:, 1], 0)))
                size = arr[:, 3] * boyle ** (2.0 / 3.0)
                puff_sc[w].set_offsets(arr[:, :2])
                puff_sc[w].set_sizes(size)
            else:
                puff_sc[w].set_offsets(np.zeros((0, 2)))
            circ, bub = tiles[w]
            pa = P_SURF + BAR_PER_MSW * z
            for i in range(16):
                circ[i].set_facecolor(TILE((P[i] - pa + VT) / (2 * VT)))
                rr = 0.44 * min(max(R[i] - 2.0, 0.0) / 150.0, 1.0)
                bub[i].set_radius(rr)
            tt = d[w + "_t"]
            m = tt <= t
            hl[w].set_data(tt[m], d[w + "_Qt"].sum(axis=0)[m])
        if hs["shallow"] > 1e-6:
            ratio_txt.set_text("deep / shallow = %.2f"
                               % (hs["deep"] / hs["shallow"]))
        hh, mm = divmod(int(t), 60)
        clock.set_text("t = %d h %02d min" % (hh, mm))

    A.encode(frame, fig, n_frames, "anim01_shotline", fps=FPS)


if __name__ == "__main__":
    main()
