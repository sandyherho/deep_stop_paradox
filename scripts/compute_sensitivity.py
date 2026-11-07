"""Robustness of the NEDU result to the uncertain bubble physics.

A Latin hypercube over the six bubble parameters, log-uniform within
about a factor of two of the reference values (the nucleus radius from
1 to 4 um). For every sample both NEDU schedules are integrated and three
numbers are kept: the hazard ratio deep/shallow at equal weighting
(beta = 0), the critical weighting beta_c at which the two schedules tie,
and the growth threshold 2 sigma / r0 - p_m, the supersaturation a
nucleus needs before it grows.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import qmc

from deep_stop_paradox import io
from deep_stop_paradox.bubble import ModelParams, TwoPhaseModel
from deep_stop_paradox.control import critical_beta, sign_changes
from deep_stop_paradox.profiles import nedu_schedule

PARAMS = [
    # name, low, high, label
    ("diffusivity", 1.0e-9 * 6e13, 4.0e-9 * 6e13, "diffusivity D"),
    ("ostwald", 0.010, 0.030, "solubility lambda"),
    ("two_sigma", 2 * 0.020 * 10.0, 2 * 0.050 * 10.0, "surface tension"),
    ("r0", 1.0, 4.0, "nucleus radius r0"),
    ("n_nuclei", 0.25e-9, 4.0e-9, "nucleus density N"),
    ("p_met", 0.12, 0.25, "metabolic gas p_m"),
]
N_SAMPLES = 128


def _sample(x):
    prm = ModelParams(**{p[0]: x[k] for k, p in enumerate(PARAMS)})
    m = TwoPhaseModel(prm, rtol=1e-8, atol=1e-11)
    qs = m.run(nedu_schedule("shallow"), sensitivities=False).Q
    qd = m.run(nedu_schedule("deep"), sensitivities=False).Q
    return qs, qd, prm.threshold


def main():
    sampler = qmc.LatinHypercube(d=len(PARAMS), seed=20261105)
    u = sampler.random(N_SAMPLES)
    lo = np.log([p[1] for p in PARAMS])
    hi = np.log([p[2] for p in PARAMS])
    X = np.exp(lo + u * (hi - lo))
    with ProcessPoolExecutor(max_workers=2) as pool:
        out = list(pool.map(_sample, X))
    Qs = np.array([o[0] for o in out])
    Qd = np.array([o[1] for o in out])
    thr = np.array([o[2] for o in out])
    ratio0 = Qd.sum(axis=1) / Qs.sum(axis=1)
    beta_c = np.array([critical_beta(d) for d in Qd - Qs])
    nsc = np.array([sign_changes(d, 1e-9 * np.abs(d).max())
                    for d in Qd - Qs])
    for i in range(N_SAMPLES):
        print("%3d threshold %.3f ratio0 %.3f beta_c %.3f"
              % (i, thr[i], ratio0[i], beta_c[i]), flush=True)
    io.save_cache("sensitivity", X=X, names=np.array([p[3] for p in
                                                      PARAMS]),
                  keys=np.array([p[0] for p in PARAMS]), ratio0=ratio0,
                  beta_c=beta_c, sign_changes=nsc, Qs=Qs, Qd=Qd,
                  threshold=thr)
    io.write_csv("sensitivity_samples",
                 [p[0] for p in PARAMS] + ["growth_threshold_bar",
                                           "ratio_beta0", "beta_c"],
                 [tuple(X[i]) + (thr[i], ratio0[i], beta_c[i])
                  for i in range(N_SAMPLES)])


if __name__ == "__main__":
    main()
