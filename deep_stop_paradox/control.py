r"""Optimal distribution of stop time and the critical hazard weighting.

The decision variables are the stop times :math:`\tau_j \ge 0` on a fixed
grid of stop depths. Two problems are solved:

* Redistribution (the design of the NEDU trial): minimize the hazard
  :math:`H_\beta(\tau) = \sum_i g_i(\beta) Q_i(\tau)` at a fixed total
  stop time :math:`\sum_j \tau_j = T`, by gradient-guided exchanges.
* The marginal (exchange) value of moving stop time from a shallow stop
  ``a`` to a deeper stop ``b``,
  :math:`E_i = \partial Q_i/\partial\tau_b - \partial Q_i/\partial\tau_a`,
  which is negative when the move lowers the hazard of compartment ``i``.

Because :math:`g_i(\beta) = (\tau_i/\bar\tau)^{-\beta}`, the weighted sum
:math:`\sum_i g_i E_i = \sum_i E_i e^{-\beta x_i}` with
:math:`x_i = \ln(\tau_i/\bar\tau)` is an exponential sum. By Laguerre's
extension of Descartes' rule of signs, its number of real zeros in
:math:`\beta` is at most the number of sign changes of :math:`E_i`
ordered by half-time. When deep time helps the fast compartments and
hurts the slow ones there is exactly one sign change, hence a unique
critical exponent :math:`\beta_c`.
"""

import numpy as np
from scipy.optimize import brentq

from .bubble import hazard_weights
from .constants import HALF_TIMES


def sign_changes(v, tol=0.0):
    """Number of sign changes in ``v`` ignoring entries with |v| <= tol."""
    v = np.asarray(v, dtype=float)
    s = np.sign(v[np.abs(v) > tol])
    return int(np.sum(s[1:] != s[:-1]))


def critical_beta(dvec, lo=-6.0, hi=8.0, half_times=HALF_TIMES):
    r"""Root of :math:`\sum_i g_i(\beta)\,d_i = 0`, or ``nan`` if none.

    For ``dvec = Q_deep - Q_shallow`` this is the weighting at which two
    schedules carry equal hazard; for an exchange vector it is the
    weighting at which moving time deeper is neutral at the margin.
    """
    d = np.asarray(dvec, dtype=float)

    def f(b):
        w = hazard_weights(b, half_times)
        return float(np.dot(w, d)) / float(np.sum(w * np.abs(d)) + 1e-300)
    grid = np.linspace(lo, hi, 281)
    vals = np.array([f(b) for b in grid])
    idx = np.where(np.sign(vals[1:]) != np.sign(vals[:-1]))[0]
    if idx.size == 0:
        return np.nan
    i = idx[0]
    return brentq(f, grid[i], grid[i + 1], xtol=1e-10)


def exchange(res, j_from, j_to):
    """Per-compartment marginal value of moving time ``j_from -> j_to``."""
    return res.dQ_dtau[j_to] - res.dQ_dtau[j_from]


def project_simplex(v, total):
    """Euclidean projection of ``v`` onto {x >= 0, sum(x) = total}."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - total
    k = np.arange(1, v.size + 1)
    rho = np.nonzero(u - css / k > 0)[0][-1]
    theta = css[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def exchange_redistribution(model, base, beta, x0, tst=None, chunk0=4.0,
                            min_chunk=0.125, tol=1e-3, maxiter=200,
                            log=None):
    r"""Minimize :math:`H_\beta` at fixed total stop time by exchanges.

    The method follows the optimality condition directly: at an optimum
    every used stop has the same marginal hazard
    :math:`\partial H/\partial\tau_j`, and no unused stop has a lower
    one. At each step the exact gradient picks the donor (the used stop
    whose last minute saves the least hazard) and the receiver (the stop
    whose next minute saves the most), and a block of stop time is moved
    between them. The block grows after a successful move and is halved
    after a failed one; the search ends when the block is below
    ``min_chunk`` minutes or the marginal hazards agree to ``tol``.

    The hazard is piecewise smooth in the stop times (bubbles in a
    compartment switch on and off at the growth threshold), which stalls
    quasi-Newton methods; this exchange search needs only monotone
    decrease of the true hazard.
    """
    w = hazard_weights(beta, model.p.half_times)
    tst = float(np.sum(x0)) if tst is None else float(tst)
    x = project_simplex(np.asarray(x0, dtype=float) * tst / np.sum(x0),
                        tst)

    def evaluate(xx):
        r = model.run(base.with_dwell(xx), sensitivities=True)
        return float(np.dot(w, r.Q)), r.dQ_dtau @ w, r.Q

    H, g, Q = evaluate(x)
    gscale = float(np.mean(np.abs(g)))
    chunk = chunk0
    hist = []
    for it in range(maxiter):
        used = x > 1e-9
        donor = int(np.argmax(np.where(used, g, -np.inf)))
        recv = int(np.argmin(g))
        gap = (g[donor] - g[recv]) / gscale
        hist.append((it, H, gap, chunk))
        if log and it % 5 == 0:
            log("    it %3d H %.7g gap %.2e chunk %.3f"
                % (it, H, gap, chunk))
        if gap < tol or chunk < min_chunk or donor == recv:
            break
        amt = min(chunk, x[donor])
        xn = x.copy()
        xn[donor] -= amt
        xn[recv] += amt
        Hn, gn, Qn = evaluate(xn)
        if Hn < H:
            x, H, g, Q = xn, Hn, gn, Qn
            chunk = min(1.5 * chunk, 32.0)
        else:
            chunk *= 0.5
    used = x > 1e-9
    mu = float(np.mean(g[used]))
    return {"x": x, "H": H, "Q": Q, "grad": g, "mu": mu,
            "kkt_equal": float(np.max(np.abs(g[used] - mu))) / gscale,
            "kkt_inactive": (float(np.min(g[~used] - mu)) / gscale
                             if np.any(~used) else np.inf),
            "nit": len(hist), "history": hist, "final_chunk": chunk}
