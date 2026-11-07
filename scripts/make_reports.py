"""Plain-text report of every number quoted from the simulations."""

import numpy as np

from deep_stop_paradox import io
from deep_stop_paradox.bubble import ModelParams, hazard_weights
from deep_stop_paradox.constants import HALF_TIMES, MSW_PER_FSW


def main():
    L = []
    prm = ModelParams()
    L += ["deep_stop_paradox: results", "=" * 26, "",
          "Reference bubble parameters",
          "  diffusivity %.3g m^2/s, Ostwald %.3g, 2 sigma %.3g bar um,"
          % (prm.diffusivity / 6e13, prm.ostwald, prm.two_sigma),
          "  nucleus radius %.2f um, nuclei %.3g per mm^3, metabolic gas "
          "%.3f bar" % (prm.r0, prm.n_nuclei * 1e9, prm.p_met),
          "  growth threshold (tension minus ambient) %.3f bar"
          % prm.threshold, ""]
    v = io.load_cache("verification")
    i = int(np.argmin(v["e_central"]))
    L += ["1. Verification",
          "  costate vs central difference: best agreement %.1e at h = "
          "%.3g min" % (v["e_central"][i], v["hs"][i]),
          "  hazard error at rtol 1e-8: %.1e (rtol 1e-6: %.1e)"
          % (v["tol_err"][4], v["tol_err"][2]),
          "  nitrogen budget residual, max over compartments: %.1e"
          % np.max(v["cons_err"]),
          "  hazard change over floor widths 0.005-0.1 um: %.1e (relative)"
          % (np.max(np.abs(v["Hw"] - v["Hw"][2])) / v["Hw"][2]), ""]
    n = io.load_cache("nedu")
    Qs, Qd = n["shallow_Q"], n["deep_Q"]
    r0 = Qd.sum() / Qs.sum()
    L += ["2. NEDU 170 fsw / 30 min, 174 min total stop time",
          "  deep-stop skew: shallow %.1f, deep %.1f bar min"
          % (n["shallow_dss"], n["deep_dss"]),
          "  hazard ratio deep/shallow, equal weights (beta=0): %.3f" % r0]
    for b in (-1.0, -0.5, 0.5, 1.0, 2.0):
        w = hazard_weights(b)
        L.append("  hazard ratio at beta=%+.1f: %.3f"
                 % (b, np.dot(w, Qd) / np.dot(w, Qs)))
    L += ["  compartments where deep stops lower Q: %s min"
          % ", ".join("%g" % h for h in HALF_TIMES[Qd < Qs - 1e-6]),
          "  compartments where deep stops raise Q: %s min"
          % ", ".join("%g" % h for h in HALF_TIMES[Qd > Qs + 1e-6]),
          "  sign changes of Q_deep - Q_shallow along half-time: %d"
          % n["sign_changes_pair"],
          "  critical exponent beta_c (equal hazard): %.3f"
          % n["beta_pair"],
          "  observed incidence ratio %.2f (95%% CI %.2f-%.2f)"
          % tuple(n["rr_obs"]),
          "  beta matching the observed ratio %.2f (CI bound %.2f to "
          "below %.2f)" % (n["beta_obs"][0], n["beta_obs"][2],
                           n["beta_obs"][1])]
    fast = np.argmax(n["shallow_R"][2])
    L += ["  peak bubble radius, 12.5 min compartment: shallow %.0f um, "
          "deep %.0f um" % (n["shallow_R"][2].max(), n["deep_R"][2].max()),
          "  peak bubble radius, 146 min compartment: shallow %.0f um, "
          "deep %.0f um" % (n["shallow_R"][9].max(), n["deep_R"][9].max())]
    for w in ("shallow", "deep"):
        t, R = n[w + "_t"], n[w + "_R"][9]
        grown = R > prm.r0 + 1.0
        if np.any(grown):
            L.append("  146 min bubble (%s) lasts until %.0f min after "
                     "surfacing" % (w, t[grown][-1] - n[w + "_runtime"]))
    del fast
    L += ["  marginal move of stop time from 10 fsw (one minute):"]
    for z, b in zip(n["stops"][:-1] / MSW_PER_FSW, n["exchange_beta"]):
        L.append("    to %2.0f fsw: beta_c = %+.3f" % (z, b))
    e = n["exchange"][0]
    L += ["  per-minute change of Q_i, 10 -> 70 fsw: fast (<= 38 min) "
          "%.1f, slow (> 38 min) %.1f" % (e[HALF_TIMES <= 38.3].sum(),
                                          e[HALF_TIMES > 38.3].sum()), ""]
    if io.has_cache("optimal"):
        o = io.load_cache("optimal")
        L += ["3. Optimal redistribution of 174 min (stops 100-10 fsw)"]
        for k, b in enumerate(o["betas"]):
            if not np.all(np.isfinite(o["X"][k])):
                continue
            w = hazard_weights(b)
            hs = np.dot(w, o["Qs"])
            L.append("  beta %+.2f: H/H_shallow %.3f, H_deep/H_shallow "
                     "%.3f, skew %.1f, entropy %.2f bits, first used stop "
                     "%d fsw, KKT %.1e"
                     % (b, o["H"][k] / hs, np.dot(w, o["Qd"]) / hs,
                        o["dss"][k], o["entropy"][k],
                        o["grid_fsw"][np.argmax(o["X"][k] > 0.5)],
                        o["kkt"][k][0]))
            pairs = zip(o["grid_fsw"], o["X"][k])
            L.append("    stops (fsw:min) " + " ".join(
                "%d:%.1f" % (z, x) for z, x in pairs if x > 0.05))
        bb = np.linspace(-1.0, 2.0, 301)
        th = o["thetas"]
        best = np.array([th[np.argmin([np.dot(hazard_weights(x), q)
                                       for q in o["Qpath"]])] for x in bb])
        jumps = np.where(np.diff(best) > 0.2)[0]
        L.append("  straight path shallow -> deep: best theta jumps at "
                 "beta = " + ", ".join("%.2f (%.2f -> %.2f)"
                                       % (bb[j + 1], best[j], best[j + 1])
                                       for j in jumps))
        for x in (0.0, 0.5, 1.0):
            Hp = np.array([np.dot(hazard_weights(x), q) for q in o["Qpath"]])
            k = int(np.argmin(Hp))
            L.append("  path, beta %+.1f: best theta %.2f, hazard %.3f of "
                     "shallow" % (x, th[k], Hp[k] / Hp[0]))
        for k, b in enumerate(o["betas"]):
            L.append("  beta %+.2f: optimum / NEDU deep hazard %.3f"
                     % (b, o["H"][k] / np.dot(hazard_weights(b), o["Qd"])))
        if "H_starts" in o:
            for k, b in enumerate(o["betas"]):
                hs = np.dot(hazard_weights(b), o["Qs"])
                r_ = o["H_starts"][k] / hs
                L.append("  beta %+.2f: optima from starts %s: %s; spread "
                         "%.1f%%; best from %s"
                         % (b, "/".join(str(x) for x in o["starts"]),
                            "/".join("%.3f" % v for v in r_),
                            100 * (r_.max() / r_.min() - 1),
                            o["starts"][o["best_start"][k]]))
        L.append("  (local optima: exchange-stationary to 0.125 min; the "
                 "hazard is piecewise smooth, so marginal hazards are not "
                 "equal at kinks)")
        L.append("")
    if io.has_cache("phase"):
        p = io.load_cache("phase")
        L += ["4. Air dives, GF 90/85 reference (beta_c above which a deep "
              "stop lowers hazard)"]
        for a, D in enumerate(p["depths"]):
            L.append("  %2d m: " % D + "  ".join(
                "%d min %+.2f/%+.2f" % (T, p["beta_half"][a, b],
                                        p["beta_best"][a, b])
                for b, T in enumerate(p["bts"])))
        L.append("  (half-depth stop / Pyle stop); max sign changes %d"
                 % p["sign_changes"].max())
        bh = p["beta_half"]
        L.append("  half-depth stop helps at equal weights in %d of %d dives"
                 % (np.sum(bh < 0), np.sum(np.isfinite(bh))))
        bp = p["beta_best"]
        L.append("  Pyle stop helps at equal weights in %d of %d dives"
                 % (np.sum(bp < 0), np.sum(np.isfinite(bp))))
        nd = p["tst"] <= 3.0 + 1e-9
        L.append("  no-decompression dives (3 min safety stop only): %d;"
                 " half-depth beta_c range %.2f to %.2f"
                 % (nd.sum(), bh[nd].min(), bh[nd].max()))
        L.append("")
    if io.has_cache("gf"):
        g = io.load_cache("gf")
        L += ["5. Gradient-factor low, GF-high 85"]
        for dive in ("nedu", "tech45"):
            Q = g[dive + "_Q"][1]
            for b in (-0.5, 0.0, 0.5, 1.0):
                H = Q @ hazard_weights(b)
                k = int(np.nanargmin(H))
                L.append("  %s beta %+.1f: lowest hazard at GF-low %d (TST "
                         "%d min); GF-low 10 vs 85 hazard ratio %.3f"
                         % (dive, b, round(100 * g["gf_lo"][k]),
                            g[dive + "_tst"][1][k], H[0] / H[-1]))
            L.append("  %s TST at GF-low 10..85: %s" % (
                dive, ", ".join("%d" % t for t in g[dive + "_tst"][1])))
        L.append("")
    if io.has_cache("sensitivity"):
        s = io.load_cache("sensitivity")
        bc = s["beta_c"]
        L += ["6. Sensitivity (%d Latin hypercube samples)" % bc.size,
              "  deep worse than shallow at beta=0 in %d of %d samples"
              % (np.sum(s["ratio0"] > 1), bc.size),
              "  ratio at beta=0: median %.2f, range %.2f-%.2f"
              % (np.median(s["ratio0"]), s["ratio0"].min(),
                 s["ratio0"].max()),
              "  beta_c: median %.2f, 5-95%% %.2f to %.2f; one crossing in "
              "%d samples, none in %d" % (
                  np.nanmedian(bc), np.nanpercentile(bc, 5),
                  np.nanpercentile(bc, 95),
                  np.sum(s["sign_changes"] == 1),
                  np.sum(s["sign_changes"] == 0))]
        from scipy.stats import spearmanr
        for k, nm in enumerate(s["names"]):
            L.append("  rank correlation of beta_c with %s: %+.2f"
                     % (nm, spearmanr(s["X"][:, k], bc,
                                      nan_policy="omit")[0]))
    if io.has_cache("threshold"):
        d = io.load_cache("threshold")
        t, r, fam = d["threshold"], d["ratio0"], d["family"]
        L += ["", "7. Growth threshold (2 sigma / r0 - p_m), NEDU pair"]
        for f, name in ((0, "nucleus radius"), (1, "surface tension")):
            m = fam == f
            tt, rr = t[m], r[m]
            cross = [0.5 * (tt[k] + tt[k + 1]) for k in range(tt.size - 1)
                     if rr[k] > 1 >= rr[k + 1] and np.all(rr[k + 1:] < 1)]
            L.append("  varying %s: ratio at beta=0 is %.2f-%.2f for "
                     "thresholds <= 0.25 bar; last crossing of 1 at about "
                     "%.2f bar; %.2f at 0.40 bar"
                     % (name, rr[tt <= 0.25 + 1e-9].min(),
                        rr[tt <= 0.25 + 1e-9].max(),
                        cross[0] if cross else np.nan,
                        rr[np.argmin(np.abs(tt - 0.40))]))
        a, b = r[fam == 0], r[fam == 1]
        ok = t[fam == 0] <= 0.425
        L.append("  radius and surface-tension routes agree to %.1f%% for "
                 "thresholds <= 0.425 bar (%.0f%% above, where both ratios "
                 "are below 0.05)"
                 % (100 * np.max(np.abs(a - b)[ok] / a[ok]),
                    100 * np.max(np.abs(a - b)[~ok] / a[~ok])))
        bc = d["beta_c"][fam == 0]
        L.append("  beta_c for thresholds <= 0.25 bar: %.2f to %.2f"
                 % (np.nanmin(bc[t[fam == 0] <= 0.25 + 1e-9]),
                    np.nanmax(bc[t[fam == 0] <= 0.25 + 1e-9])))
        s2 = io.load_cache("sensitivity")
        th, r2 = s2["threshold"], s2["ratio0"]
        for lo_, hi_ in ((-1.0, 0.25), (0.25, 0.35), (0.35, 2.0)):
            m = (th > lo_) & (th <= hi_)
            L.append("  samples with threshold in (%.2f, %.2f] bar: %d, "
                     "deep worse in %d, median ratio %.2f"
                     % (max(lo_, th.min() - 1e-9), min(hi_, th.max()),
                        m.sum(), np.sum(r2[m] > 1), np.median(r2[m])))
    io.write_report("results", L)
    print("\n".join(L))


if __name__ == "__main__":
    main()
