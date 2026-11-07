"""Where should the NEDU total stop time go?

The NEDU design fixed the total stop time (174 min) and asked where it
should be spent. Two calculations answer that question in the two-phase
model, on a 10 fsw stop grid from 100 fsw to 10 fsw.

1. The straight path between the two test schedules,
   tau(theta) = (1 - theta) tau_shallow + theta tau_deep, for theta from
   0 (shallow stops) through 1 (deep stops) to the largest value that
   keeps every stop time non-negative. The compartment hazards along the
   path give the total hazard for every weighting at once.
2. The best distribution found for each weighting by gradient-guided
   exchanges of stop time between stops (exact costate gradients), run
   from three starts (the two NEDU schedules and a uniform distribution).
   The hazard is piecewise smooth in the stop times, so the search finds
   local optima; the best of the three is kept and the spread between
   them is stored as a measure of how flat the optimum is.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from deep_stop_paradox import io
from deep_stop_paradox.bubble import TwoPhaseModel, hazard_weights
from deep_stop_paradox.control import exchange_redistribution
from deep_stop_paradox.profiles import deep_stop_skew, nedu_schedule

BETAS = np.array([-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
GRID_FSW = np.arange(100.0, 0.0, -10.0)


def entropy_bits(x):
    p = np.asarray(x, dtype=float)
    p = p[p > 1e-9] / p.sum()
    return float(-np.sum(p * np.log2(p)))


def _model():
    return TwoPhaseModel(rtol=1e-7, atol=1e-10)


def _path_point(x):
    return _model().run(nedu_schedule("shallow", GRID_FSW).with_dwell(x),
                        sensitivities=False).Q


STARTS = ("shallow", "deep", "uniform")


def _optimize(args):
    b, start = args
    m = _model()
    shallow = nedu_schedule("shallow", GRID_FSW)
    x0 = {"shallow": shallow.dwell,
          "deep": nedu_schedule("deep", GRID_FSW).dwell,
          "uniform": np.full(GRID_FSW.size, shallow.tst / GRID_FSW.size)}
    out = exchange_redistribution(m, shallow, b, x0[start])
    return b, start, out


def main():
    shallow = nedu_schedule("shallow", GRID_FSW)
    deep = nedu_schedule("deep", GRID_FSW)
    ds, dd = shallow.dwell, deep.dwell
    neg = dd < ds
    theta_max = float(np.min(ds[neg] / (ds[neg] - dd[neg])))
    thetas = np.linspace(0.0, theta_max, 30)
    path = [ds + t * (dd - ds) for t in thetas]
    with ProcessPoolExecutor(max_workers=2) as pool:
        Qpath = np.array(list(pool.map(_path_point, path)))
        print("path done", flush=True)
        jobs = [(b, st) for b in BETAS for st in STARTS]
        runs = list(pool.map(_optimize, jobs))
    Qs, Qd = Qpath[0], _path_point(dd)
    n = ds.size
    ns = len(STARTS)
    Xs = np.zeros((BETAS.size, ns, n))
    Hs = np.zeros((BETAS.size, ns))
    gaps = np.zeros((BETAS.size, ns))
    Qall = np.zeros((BETAS.size, ns, 16))
    for b, st, r in runs:
        i, k = int(np.argmin(np.abs(BETAS - b))), STARTS.index(st)
        Xs[i, k], Hs[i, k], Qall[i, k] = r["x"], r["H"], r["Q"]
        gaps[i, k] = r["kkt_equal"]
    best = np.argmin(Hs, axis=1)
    idx = np.arange(BETAS.size)
    X, H, Qopt = Xs[idx, best], Hs[idx, best], Qall[idx, best]
    kkt = np.c_[gaps[idx, best], np.zeros(BETAS.size)]
    nit = np.zeros(BETAS.size, dtype=int)
    chunk = np.zeros(BETAS.size)
    for i, b in enumerate(BETAS):
        hs = np.dot(hazard_weights(b), Qs)
        print("beta %+.2f  best start %-8s H/H_shallow %.4f  (starts: %s)"
              % (b, STARTS[best[i]], H[i] / hs,
                 ", ".join("%.4f" % v for v in Hs[i] / hs)),
              np.round(X[i], 1), flush=True)
    dss = np.array([deep_stop_skew(shallow.with_dwell(x)) for x in X])
    ent = np.array([entropy_bits(x) for x in X])
    dss_path = np.array([deep_stop_skew(shallow.with_dwell(x))
                         for x in path])
    io.save_cache("optimal", betas=BETAS, X=X, H=H, kkt=kkt, nit=nit,
                  starts=np.array(STARTS), X_starts=Xs, H_starts=Hs,
                  gap_starts=gaps, best_start=best,
                  chunk=chunk, Qopt=Qopt, Qs=Qs, Qd=Qd,
                  stops=shallow.stops, grid_fsw=GRID_FSW, tst=shallow.tst,
                  dss=dss, entropy=ent, dss_shallow=deep_stop_skew(shallow),
                  dss_deep=deep_stop_skew(deep), thetas=thetas,
                  Qpath=Qpath, dss_path=dss_path)
    io.write_csv("optimal_schedules",
                 ["beta"] + ["tau_%dfsw" % z for z in GRID_FSW]
                 + ["H_opt_over_H_shallow", "deep_stop_skew",
                    "entropy_bits"],
                 [(BETAS[i],) + tuple(X[i]) +
                  (H[i] / np.dot(hazard_weights(BETAS[i]), Qs), dss[i],
                   ent[i]) for i in range(BETAS.size)])
    io.write_csv("redistribution_path",
                 ["theta", "deep_stop_skew"] + ["Q_%02d" % (k + 1)
                                                for k in range(16)],
                 [(t, s) + tuple(q) for t, s, q in zip(thetas, dss_path,
                                                       Qpath)])


if __name__ == "__main__":
    main()
