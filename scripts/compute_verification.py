"""Numerical checks of the two-phase solver and its exact gradients.

1. Taylor test: central and forward finite differences of the hazard
   against the costate (shadow-price) derivative, over step sizes.
2. Tolerance convergence of the total hazard.
3. Nitrogen conservation: content change minus the perfusion integral.
4. Insensitivity to the width of the smooth nucleus floor.
"""

import numpy as np

from deep_stop_paradox import io
from deep_stop_paradox.bubble import (ModelParams, TwoPhaseModel, hazard,
                                      nitrogen_content)
from deep_stop_paradox.constants import BAR_PER_MSW, P_H2O, P_SURF
from deep_stop_paradox.profiles import nedu_schedule


def _budget(sched, rtol=1e-10, atol=1e-13):
    """Return t, residual(t) per compartment, and the content scale."""
    from scipy.integrate import solve_ivp
    from deep_stop_paradox.bubble import _Rhs, initial_state
    prm = ModelParams()
    n = prm.n
    P, R = initial_state(prm)
    Q = np.zeros(n)
    F = np.zeros(n)
    c0 = nitrogen_content(prm, 0.0, P, R)
    ts, res, cmax = [], [], np.zeros(n)
    t0 = 0.0
    for g in sched.segments():
        if g.dt <= 0:
            continue
        rhs = _Rhs(prm, g.z0, g.rate)

        def f(t, y, rhs=rhs, g=g):
            P = y[:n]
            base = rhs.f(t, y[:3 * n])
            z = g.z0 + g.rate * t
            p_art = prm.f_n2 * (P_SURF + BAR_PER_MSW * z - P_H2O)
            return np.concatenate([base, prm.k * (p_art - P)])
        sol = solve_ivp(f, (0.0, g.dt), np.concatenate([P, R, Q, F]),
                        method="LSODA", rtol=rtol, atol=atol,
                        t_eval=np.linspace(0.0, g.dt, 50))
        Y = sol.y
        z = g.z0 + g.rate * sol.t
        c = nitrogen_content(prm, z[None, :], Y[:n], Y[n:2 * n])
        ts.append(t0 + sol.t)
        res.append((c - c0[:, None]) - Y[3 * n:])
        cmax = np.maximum(cmax, np.max(np.abs(c - c0[:, None]), axis=1))
        y = Y[:, -1]
        P, R, Q, F = y[:n], y[n:2 * n], y[2 * n:3 * n], y[3 * n:]
        t0 += g.dt
    return np.concatenate(ts), np.concatenate(res, axis=1), cmax


def main():
    sched = nedu_schedule("shallow")
    m = TwoPhaseModel(rtol=1e-10, atol=1e-13)
    base = m.run(sched, sensitivities=True)
    j = 3                                    # the 40 fsw stop
    g_exact = base.dQ_dtau[j]
    H_exact = float(np.sum(g_exact))
    hs = np.logspace(-4, 0.5, 13)
    e_central, e_forward = [], []
    Q0 = base.Q
    for h in hs:
        e = np.eye(sched.stops.size)[j]
        qp = m.run(sched.with_dwell(sched.dwell + h * e),
                   sensitivities=False).Q
        qm = m.run(sched.with_dwell(sched.dwell - h * e),
                   sensitivities=False).Q
        e_central.append(abs(np.sum(qp - qm) / (2 * h) - H_exact)
                         / abs(H_exact))
        e_forward.append(abs(np.sum(qp - Q0) / h - H_exact)
                         / abs(H_exact))
        print("h %.2e central %.2e forward %.2e"
              % (h, e_central[-1], e_forward[-1]), flush=True)

    # tolerance convergence
    ref = hazard(TwoPhaseModel(rtol=1e-11, atol=1e-14).run(
        sched, sensitivities=False), 0.0)
    rtols = np.array([1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
    tol_err = []
    for rt in rtols:
        H = hazard(TwoPhaseModel(rtol=rt, atol=rt * 1e-3).run(
            sched, sensitivities=False), 0.0)
        tol_err.append(abs(H - ref) / ref)
    # nitrogen conservation: integrate the perfusion flux as an extra
    # state and compare with the change in total nitrogen content
    t_all, resid_all, scale = _budget(sched)
    cons_err = np.max(np.abs(resid_all), axis=1) / scale
    t, resid = t_all, resid_all
    # floor width
    widths = np.array([0.005, 0.01, 0.02, 0.05, 0.1])
    Hw = []
    for w in widths:
        Hw.append(hazard(TwoPhaseModel(ModelParams(floor_width=w),
                                       rtol=1e-9, atol=1e-12).run(
            sched, sensitivities=False), 0.0))
    Hw = np.array(Hw)
    io.save_cache("verification", hs=hs, e_central=np.array(e_central),
                  e_forward=np.array(e_forward), rtols=rtols,
                  tol_err=np.array(tol_err), cons_err=cons_err,
                  t=t, resid=resid, scale=scale, widths=widths, Hw=Hw,
                  g_exact=g_exact)
    io.write_csv("verification_taylor", ["h_min", "rel_err_central",
                                         "rel_err_forward"],
                 zip(hs, e_central, e_forward))


if __name__ == "__main__":
    main()
