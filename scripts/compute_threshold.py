"""The NEDU comparison as a function of the bubble growth threshold.

A nucleus of radius r0 grows only when the tissue tension exceeds the
ambient pressure by more than 2 sigma / r0 - p_m. This threshold decides
whether slow tissues form bubbles at all during an air decompression, and
with it the sign of the deep-stop question. The threshold is scanned in
two independent ways, through the nucleus radius at the reference
surface tension and through the surface tension at the reference radius,
holding all other parameters at their reference values.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from deep_stop_paradox import io
from deep_stop_paradox.bubble import ModelParams, TwoPhaseModel
from deep_stop_paradox.control import critical_beta, sign_changes
from deep_stop_paradox.profiles import nedu_schedule

REF = ModelParams()
# target thresholds (bar), from almost none to well above the slow-tissue
# supersaturation of the NEDU schedules
THRESHOLDS = np.linspace(0.0, 0.55, 23)


def _run(kw):
    prm = ModelParams(**kw)
    m = TwoPhaseModel(prm, rtol=1e-8, atol=1e-11)
    qs = m.run(nedu_schedule("shallow"), sensitivities=False).Q
    qd = m.run(nedu_schedule("deep"), sensitivities=False).Q
    return prm.threshold, qs, qd


def main():
    jobs, family = [], []
    for th in THRESHOLDS:
        # through the radius: r0 = 2 sigma / (threshold + p_m)
        jobs.append({"r0": REF.two_sigma / (th + REF.p_met)})
        family.append(0)
        # through the surface tension: 2 sigma = r0 (threshold + p_m)
        jobs.append({"two_sigma": REF.r0 * (th + REF.p_met)})
        family.append(1)
    with ProcessPoolExecutor(max_workers=2) as pool:
        out = list(pool.map(_run, jobs))
    family = np.array(family)
    thr = np.array([o[0] for o in out])
    Qs = np.array([o[1] for o in out])
    Qd = np.array([o[2] for o in out])
    ratio0 = Qd.sum(axis=1) / Qs.sum(axis=1)
    beta_c = np.array([critical_beta(d) for d in Qd - Qs])
    nsc = np.array([sign_changes(d, 1e-9 * np.abs(d).max())
                    for d in Qd - Qs])
    slow_s = (Qs[:, 6:] > 1.0).sum(axis=1)       # compartments >= 54 min
    slow_d = (Qd[:, 6:] > 1.0).sum(axis=1)
    for f, t, r, b, a, c in zip(family, thr, ratio0, beta_c, slow_s,
                                slow_d):
        print("%s threshold %.3f ratio0 %.3f beta_c %+.3f slow %d/%d"
              % ("r0   " if f == 0 else "sigma", t, r, b, a, c),
              flush=True)
    io.save_cache("threshold", family=family, threshold=thr, Qs=Qs, Qd=Qd,
                  ratio0=ratio0, beta_c=beta_c, sign_changes=nsc,
                  slow_shallow=slow_s, slow_deep=slow_d,
                  r0=np.array([j.get("r0", REF.r0) for j in jobs]),
                  two_sigma=np.array([j.get("two_sigma", REF.two_sigma)
                                      for j in jobs]))
    io.write_csv("threshold_scan",
                 ["varied", "growth_threshold_bar", "r0_um",
                  "two_sigma_bar_um", "ratio_beta0", "beta_c",
                  "slow_compartments_bubbling_shallow",
                  "slow_compartments_bubbling_deep"],
                 [("r0" if f == 0 else "sigma", t, j.get("r0", REF.r0),
                   j.get("two_sigma", REF.two_sigma), r, b, a, c)
                  for f, t, j, r, b, a, c in zip(family, thr, jobs, ratio0,
                                                 beta_c, slow_s, slow_d)])


if __name__ == "__main__":
    main()
