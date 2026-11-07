"""Animation 3: where should 174 min of stops go?

The hazard weighting exponent beta sweeps from tissues-are-equal toward
fast-tissue dominated and back. For each beta the best distribution of
the NEDU total stop time found in the two-phase model (best of three
starts of the exchange search) is shown as the ascent profile and as stop
bars on a shot line, next to its hazard and that of the shallow-stops and
deep-stops schedules. Between computed values of beta the schedules are
interpolated linearly for display only.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from deep_stop_paradox import anim_style as A
from deep_stop_paradox import io
from deep_stop_paradox.bubble import hazard_weights
from deep_stop_paradox.constants import HALF_TIMES
from deep_stop_paradox.diver_art import Diver
from deep_stop_paradox.profiles import nedu_schedule

FPS = 30
SECONDS = 24.0


def main():
    A.setup()
    o = io.load_cache("optimal")
    nd = io.load_cache("nedu")
    betas, X = o["betas"], o["X"]
    ok = np.all(np.isfinite(X), axis=1)
    betas, X, H = betas[ok], X[ok], o["H"][ok]
    zf = o["grid_fsw"]
    base = nedu_schedule("shallow", zf)
    sh = nedu_schedule("shallow", zf).dwell
    dp = nedu_schedule("deep", zf).dwell
    bb = np.linspace(betas[0], betas[-1], 200)
    hs = np.array([np.dot(hazard_weights(b), o["Qs"]) for b in bb])
    hd = np.array([np.dot(hazard_weights(b), o["Qd"]) for b in bb])
    hs_k = np.array([np.dot(hazard_weights(b), o["Qs"]) for b in betas])
    n_frames = int(SECONDS * FPS)
    s = np.linspace(0, 1, n_frames)
    beta_t = betas[0] + (betas[-1] - betas[0]) * 0.5 * (1 - np.cos(
        2 * np.pi * s))

    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.text(0.04, 0.955, "Where should 174 minutes of stops go?",
             fontsize=17, fontweight="bold")
    sub = fig.text(0.04, 0.915, "", fontsize=13, color=A.FG2)
    # shot line with stop bars
    axl = fig.add_axes([0.065, 0.08, 0.27, 0.80])
    zmax = 175
    axl.set_xlim(-60, 70)
    axl.set_ylim(-zmax, 8)
    A.water_column(axl, -60, 70, 0, -zmax)
    axl.add_patch(Rectangle((-60, 0), 130, 8, fc="#0f2a3f", ec="none"))
    axl.plot([0, 0], [0, -zmax + 4], color="#e9d8a6", lw=1.4)
    shafts = A.LightShafts(axl, np.linspace(-50, 60, 6), zmax, seed=5)
    snow = A.Snow(axl, -60, 70, 0, -zmax, n=160)
    axl.set_xticks([])
    axl.set_yticks(-np.arange(0, 176, 25))
    axl.set_yticklabels(["%d" % v for v in np.arange(0, 176, 25)])
    axl.set_ylabel("depth (fsw)")
    for sp in axl.spines.values():
        sp.set_visible(False)
    bars = []
    for z in zf:
        r = Rectangle((2, -z - 3.2), 0, 6.4, fc="#1baf7a", ec="white",
                      lw=0.6, alpha=0.95, zorder=5)
        axl.add_patch(r)
        bars.append(r)
    ghost_s = axl.scatter(sh * 0.6 + 2, -zf, marker="|", s=120,
                          c="#2a78d6", zorder=6)
    ghost_d = axl.scatter(dp * 0.6 + 2, -zf, marker="|", s=120,
                          c="#eb6834", zorder=6)
    del ghost_s, ghost_d
    axl.text(4, -zmax + 12, "bars: best stop times found\n"
             "ticks: NEDU shallow (blue)\n"
             "and deep (orange) stops", fontsize=9, color=A.FG2,
             va="bottom")
    labels = [axl.text(0, -z, "", fontsize=9, va="center", zorder=7)
              for z in zf]
    diver = Diver(axl, "#1baf7a", 13.0, zorder=8)
    # profile panel
    axp = fig.add_axes([0.40, 0.55, 0.56, 0.32])
    axp.set_facecolor("#0a1b28")
    for w, col in (("shallow", "#2a78d6"), ("deep", "#eb6834")):
        t, z = nd[w + "_t"], nd[w + "_z"]
        m = t <= nd[w + "_runtime"] + 1e-9
        axp.plot(t[m], -z[m] / 0.30643, color=col, lw=1.4, alpha=0.55)
    prof, = axp.plot([], [], color="#1baf7a", lw=3.0)
    axp.set_xlim(0, 240)
    axp.set_ylim(-178, 5)
    axp.set_xlabel("time (min)")
    axp.set_ylabel("depth (fsw)")
    axp.set_yticks([-150, -100, -50, 0])
    axp.set_yticklabels(["150", "100", "50", "0"])
    for sp in ("top", "right"):
        axp.spines[sp].set_visible(False)
    # hazard panel
    axh = fig.add_axes([0.40, 0.09, 0.26, 0.34])
    axh.set_facecolor("#0a1b28")
    axh.semilogy(bb, hd / hs, color="#eb6834", lw=2.0, label="deep stops")
    axh.axhline(1.0, color="#2a78d6", lw=2.0, label="shallow stops")
    axh.semilogy(betas, H / hs_k, "o-", color="#1baf7a", lw=2.0, ms=5,
                 label="best found")
    cur, = axh.plot([], [], "o", color="white", ms=10, mec="#1baf7a",
                    mew=2)
    vline = axh.axvline(0, color=A.FG2, lw=0.8)
    axh.set_xlabel(r"hazard weighting exponent $\beta$")
    axh.set_ylabel("hazard / shallow stops")
    axh.set_yticks([0.15, 0.3, 0.5, 1.0, 2.0, 3.0])
    axh.set_yticklabels(["0.15", "0.3", "0.5", "1", "2", "3"])
    axh.minorticks_off()
    axh.legend(loc="upper right", frameon=False, fontsize=10)
    for sp in ("top", "right"):
        axh.spines[sp].set_visible(False)
    # weights panel
    axw = fig.add_axes([0.71, 0.09, 0.25, 0.34])
    axw.set_facecolor("#0a1b28")
    wb = axw.bar(np.arange(16), np.ones(16), color="#9ec5f4", width=0.75)
    axw.set_yscale("log")
    axw.set_ylim(1e-2, 3e2)
    axw.set_yticks([0.01, 0.1, 1, 10, 100])
    axw.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    axw.minorticks_off()
    axw.set_xticks(np.arange(0, 16, 3))
    axw.set_xticklabels(["%g" % h for h in HALF_TIMES[::3]])
    axw.set_xlabel("compartment half-time (min)")
    axw.set_ylabel(r"hazard weight $g_i(\beta)$")
    for sp in ("top", "right"):
        axw.spines[sp].set_visible(False)

    def frame(k):
        b = float(beta_t[k])
        x = np.array([np.interp(b, betas, X[:, j]) for j in range(zf.size)])
        x *= 174.0 / x.sum()
        shafts.update(0.05 * k)
        snow.step(1.5)
        for r, lab, xv, z in zip(bars, labels, x, zf):
            r.set_width(xv * 0.6)
            lab.set_position((4 + xv * 0.6, -z))
            lab.set_text("%.0f min" % xv if xv >= 0.5 else "")
        j = int(np.argmax(x))
        diver.place(-24, -zf[j] + 1.5, 0.0, False,
                    kick=12 * np.sin(0.4 * k))
        sch = base.with_dwell(x)
        t, z = sch.depth_curve(600)
        prof.set_data(t, -z / 0.30643)
        hr = np.interp(b, betas, H / hs_k)
        cur.set_data([b], [hr])
        vline.set_xdata([b, b])
        g = hazard_weights(b)
        for rect, gv in zip(wb, g):
            rect.set_height(gv)
        sub.set_text(r"hazard weighting $\beta$ = %+.2f   (beta < 0: slow "
                     r"tissues count more;  beta > 0: fast tissues count "
                     r"more)" % b)

    A.encode(frame, fig, n_frames, "anim03_redistribution", fps=FPS)


if __name__ == "__main__":
    main()
