r"""Two-phase (dissolved gas plus bubble) decompression model.

Every perfusion compartment ``i`` carries a nitrogen tension
:math:`P_i` and a population of identical gas nuclei of number density
:math:`N` whose radius is :math:`R_i`. The nuclei are stabilized at the
radius :math:`r_0` and cannot dissolve below it; above it they grow or
shrink by diffusion.

Gas in a bubble is at the ambient pressure plus the Laplace pressure,
and contains the metabolic gases (O2, CO2, H2O) at the fixed partial
pressure :math:`p_m`, so the nitrogen pressure in the bubble is

.. math::  P_b = P_a(t) - p_m + 2\sigma/R .

Quasi-steady diffusion of nitrogen to a sphere, with the ideal-gas
content :math:`P_b R^3`, gives the classical radius equation

.. math::

    \dot R = \frac{D\lambda\,(P - P_b) - \tfrac13 R^2 \dot P_a}
                  {R\,(P_a - p_m) + \tfrac43 \sigma}.

The nitrogen the bubbles gain is lost by the tissue, so that

.. math::

    \dot P = k\,(P_{art} - P) - \frac{4\pi N}{3\lambda}
             \frac{d}{dt}\big(P_b R^3\big),

which conserves the total nitrogen of each compartment exactly, apart
from perfusion. The hazard of each compartment is the integral of its
bubble volume fraction :math:`v_i = N\tfrac43\pi(R_i^3 - r_0^3)`,
measured in units of :math:`v_{ref}`:

.. math::  \dot Q_i = v_i / v_{ref}.

The compartments are independent once the depth path is given, so all
16 are integrated together but their Jacobian is block diagonal.

The solver integrates, segment by segment, the state and its first
order sensitivity to the state at the start of the segment (the
segment transition matrix). Composing these matrices backwards gives
the shadow prices (costates) of tissue tension and bubble radius, and
from them the exact derivative of every compartment hazard with respect
to every stop time.
"""

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.integrate import solve_ivp

from . import constants as C


@dataclass
class ModelParams:
    diffusivity: float = C.DIFFUSIVITY          # um^2/min
    ostwald: float = C.OSTWALD                  # -
    two_sigma: float = C.TWO_SIGMA              # bar um
    r0: float = C.R_NUCLEUS                     # um
    n_nuclei: float = C.N_NUCLEI                # um^-3
    p_met: float = C.P_METABOLIC                # bar
    v_ref: float = C.V_REF                      # -
    floor_width: float = C.FLOOR_WIDTH          # um
    f_n2: float = C.F_N2_AIR                    # -
    half_times: np.ndarray = field(
        default_factory=lambda: C.HALF_TIMES.copy())

    @property
    def k(self):
        return np.log(2.0) / np.asarray(self.half_times)

    @property
    def n(self):
        return int(np.size(self.half_times))

    def varied(self, **kw):
        return replace(self, **kw)

    @property
    def threshold(self):
        """Supersaturation (bar) needed to grow a nucleus."""
        return self.two_sigma / self.r0 - self.p_met


def hazard_weights(beta, half_times=C.HALF_TIMES):
    r"""Hazard weight per unit bubble volume.

    :math:`g_i = (\tau_i/\bar\tau)^{-\beta}`.

    ``beta = 0`` gives every compartment the same weight; ``beta > 0``
    weights fast compartments more heavily. The pivot is the geometric
    mean half-time, so the weights have unit geometric mean.
    """
    tau = np.asarray(half_times, dtype=float)
    tau_gm = np.exp(np.mean(np.log(tau)))
    return (tau / tau_gm) ** (-float(beta))


# --------------------------------------------------------------------- #
# Right-hand side and Jacobian
# --------------------------------------------------------------------- #
class _Rhs:
    """Vectorized right-hand side on one linear depth segment."""

    def __init__(self, prm, z0, rate):
        self.p = prm
        self.z0 = z0
        self.rate = rate
        self.n = prm.n
        self.k = prm.k
        self.A = prm.diffusivity * prm.ostwald
        self.kap = 4.0 * np.pi * prm.n_nuclei / (3.0 * prm.ostwald)
        self.c_v = prm.n_nuclei * 4.0 * np.pi / 3.0 / prm.v_ref

    def core(self, t, P, R):
        p = self.p
        s = p.two_sigma
        z = self.z0 + self.rate * t
        pa = C.P_SURF + C.BAR_PER_MSW * z
        dp = C.BAR_PER_MSW * self.rate
        part = p.f_n2 * (pa - C.P_H2O)
        B = pa - p.p_met
        pb = B + s / R
        delta = P - pb
        num = self.A * delta - R * R * dp / 3.0
        den = R * B + 2.0 * s / 3.0
        F = num / den
        dF_dP = self.A / den
        dF_dR = ((self.A * s / R ** 2 - 2.0 * R * dp / 3.0) * den
                 - num * B) / den ** 2
        # smooth floor: shrinking stops at the nucleus radius
        e = np.exp(np.minimum(-(R - p.r0) / p.floor_width, 30.0))
        neg = F < 0.0
        G = np.where(neg, 1.0 - e, 1.0)
        Rdot = F * G
        dR_dP = G * dF_dP
        dR_dR = G * dF_dR + np.where(neg, F * e / p.floor_width, 0.0)
        # nitrogen moved into the bubbles (bar um^3 / min per bubble)
        Cc = 3.0 * R * R * B + 2.0 * s * R
        m = R ** 3 * dp + Rdot * Cc
        dm_dP = Cc * dR_dP
        dm_dR = 3.0 * R * R * dp + Cc * dR_dR + Rdot * (6.0 * R * B
                                                        + 2.0 * s)
        Pdot = self.k * (part - P) - self.kap * m
        dP_dP = -self.k - self.kap * dm_dP
        dP_dR = -self.kap * dm_dR
        Qdot = self.c_v * (R ** 3 - p.r0 ** 3)
        dQ_dR = 3.0 * self.c_v * R * R
        return (Pdot, Rdot, Qdot, dP_dP, dP_dR, dR_dP, dR_dR, dQ_dR, m)

    # -- plain state ------------------------------------------------------
    def f(self, t, y):
        n = self.n
        P, R = y[:n], y[n:2 * n]
        out = self.core(t, P, R)
        return np.concatenate(out[:3])

    # -- state with sensitivities ----------------------------------------
    def f_aug(self, t, y):
        n = self.n
        P, R = y[:n], y[n:2 * n]
        (Pd, Rd, Qd, jpp, jpr, jrp, jrr, jqr, _) = self.core(t, P, R)
        S = y[3 * n:].reshape(6, n)
        spp, srp, sqp, spr, srr, sqr = S
        dS = np.stack([jpp * spp + jpr * srp,
                       jrp * spp + jrr * srp,
                       jqr * srp,
                       jpp * spr + jpr * srr,
                       jrp * spr + jrr * srr,
                       jqr * srr])
        return np.concatenate([Pd, Rd, Qd, dS.ravel()])

    def _blocks(self, t, y):
        n = self.n
        P, R = y[:n], y[n:2 * n]
        out = self.core(t, P, R)
        return out[3:8]

    def jac(self, t, y):
        n = self.n
        jpp, jpr, jrp, jrr, jqr = self._blocks(t, y)
        i = np.arange(n)
        rows = np.concatenate([i, i, n + i, n + i, 2 * n + i])
        cols = np.concatenate([i, n + i, i, n + i, n + i])
        vals = np.concatenate([jpp, jpr, jrp, jrr, jqr])
        J = np.zeros((3 * n, 3 * n))
        J[rows, cols] = vals
        return J

    def jac_aug(self, t, y):
        """Block Jacobian; the term (dJ/dy) S is dropped.

        It is used only by the Newton iteration of the implicit method,
        so the omission changes the convergence rate, not the solution.
        """
        n = self.n
        jpp, jpr, jrp, jrr, jqr = self._blocks(t, y)
        i = np.arange(n)
        rr, cc, vv = [], [], []

        def put(r, c, v):
            rr.append(r)
            cc.append(c)
            vv.append(v)
        # base state (P, R, Q)
        put(i, i, jpp)
        put(i, n + i, jpr)
        put(n + i, i, jrp)
        put(n + i, n + i, jrr)
        put(2 * n + i, n + i, jqr)
        # two sensitivity columns, each a copy of the 3x3 block
        for col0 in (3 * n, 6 * n):
            a, b, q = col0 + i, col0 + n + i, col0 + 2 * n + i
            put(a, a, jpp)
            put(a, b, jpr)
            put(b, a, jrp)
            put(b, b, jrr)
            put(q, b, jqr)
        J = np.zeros((9 * n, 9 * n))
        J[np.concatenate(rr), np.concatenate(cc)] = np.concatenate(vv)
        return J


# --------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------- #
@dataclass
class RunResult:
    schedule: object
    segments: list
    entry: list                     # state (P, R) at the start of segments
    phi: list                       # per-segment 2x2 + Q-row transition
    Q: np.ndarray                   # final hazard integral per compartment
    dQ_dtau: np.ndarray = None      # (n_stops, n_comp)
    lam_hold: list = None           # costates after each hold
    F_hold: list = None             # vector field at the end of each hold
    dense: dict = None


def initial_state(prm):
    P = np.full(prm.n, prm.f_n2 * (C.P_SURF - C.P_H2O))
    R = np.full(prm.n, prm.r0)
    return P, R


class TwoPhaseModel:
    """Integrates the two-phase model along a :class:`Schedule`."""

    def __init__(self, params=None, rtol=1e-8, atol=1e-11,
                 method="LSODA"):
        self.p = params if params is not None else ModelParams()
        self.rtol = rtol
        self.atol = atol
        self.method = method

    # ------------------------------------------------------------------
    def _atol(self, aug):
        n = self.p.n
        a = np.concatenate([np.full(n, self.atol),
                            np.full(n, self.atol * 10.0),
                            np.full(n, self.atol * 10.0)])
        if aug:
            a = np.concatenate([a, np.full(6 * n, self.atol * 10.0)])
        return a

    def run(self, schedule, sensitivities=True, dense=False,
            n_dense=60, state0=None, split=None):
        """Integrate the whole exposure.

        Parameters
        ----------
        schedule : Schedule
        sensitivities : bool
            Also integrate the segment transition matrices and return
            the exact stop-time derivatives ``dQ_dtau``.
        dense : bool
            Store a time series sampled at ``n_dense`` points per segment
            (for figures and animations).
        """
        prm = self.p
        n = prm.n
        segs = schedule.segments(split=split)
        P, R = initial_state(prm) if state0 is None else state0
        Q = np.zeros(n)
        entry, phi = [], []
        dts, dz, dP, dR, dQ = [], [], [], [], []
        t_abs = 0.0
        for g in segs:
            entry.append((P.copy(), R.copy()))
            rhs = _Rhs(prm, g.z0, g.rate)
            if g.dt <= 0.0:
                phi.append(None)
                continue
            if sensitivities:
                y0 = np.concatenate([P, R, Q,
                                     np.ones(n), np.zeros(n), np.zeros(n),
                                     np.zeros(n), np.ones(n), np.zeros(n)])
                fun, jac = rhs.f_aug, rhs.jac_aug
            else:
                y0 = np.concatenate([P, R, Q])
                fun, jac = rhs.f, rhs.jac
            t_eval = (np.linspace(0.0, g.dt, n_dense) if dense else None)
            sol = solve_ivp(fun, (0.0, g.dt), y0, method=self.method,
                            jac=jac, rtol=self.rtol,
                            atol=self._atol(sensitivities),
                            t_eval=t_eval)
            if not sol.success or not np.all(np.isfinite(sol.y)):
                # fall back to the implicit Runge-Kutta method
                sol = solve_ivp(fun, (0.0, g.dt), y0, method="Radau",
                                jac=jac, rtol=self.rtol,
                                atol=self._atol(sensitivities),
                                t_eval=t_eval)
                if not sol.success or not np.all(np.isfinite(sol.y)):
                    raise RuntimeError("integration failed: "
                                       + sol.message)
            y = sol.y[:, -1]
            P, R, Q = y[:n].copy(), y[n:2 * n].copy(), y[2 * n:3 * n].copy()
            if sensitivities:
                S = y[3 * n:].reshape(6, n)
                phi.append(S.copy())
            else:
                phi.append(None)
            if dense:
                dts.append(t_abs + sol.t)
                dz.append(g.z0 + g.rate * sol.t)
                dP.append(sol.y[:n])
                dR.append(sol.y[n:2 * n])
                dQ.append(sol.y[2 * n:3 * n])
            t_abs += g.dt
        res = RunResult(schedule, segs, entry, phi, Q)
        if dense:
            res.dense = {"t": np.concatenate(dts), "z": np.concatenate(dz),
                         "P": np.concatenate(dP, axis=1),
                         "R": np.concatenate(dR, axis=1),
                         "Q": np.concatenate(dQ, axis=1)}
        if sensitivities:
            self._dwell_gradient(res)
        return res

    # ------------------------------------------------------------------
    def _dwell_gradient(self, res):
        r"""Exact :math:`\partial Q_i/\partial\tau_j` for every stop.

        Extending hold ``j`` by :math:`\delta\tau` at its end perturbs the
        state entering the next segment by :math:`f(y_j)\,\delta\tau` and
        adds :math:`\dot Q(y_j)\,\delta\tau` of hazard. The effect on the
        final hazard of compartment ``i`` is the costate (shadow price)
        :math:`\lambda_i` of the state after the hold contracted with
        the vector field there.
        """
        prm = self.p
        n = prm.n
        segs = res.segments
        nst = res.schedule.stops.size
        lamP = np.zeros(n)
        lamR = np.zeros(n)
        dQ = np.zeros((nst, n))
        lam_hold = [None] * nst
        F_hold = [None] * nst
        lam_seg = [None] * len(segs)
        # walk backwards; lam is the costate at the START of segment s
        for s in range(len(segs) - 1, -1, -1):
            g = segs[s]
            if g.kind == "hold" and g.end:
                j = g.stop
                # state at the end of the hold = entry of segment s + 1
                P, R = res.entry[s + 1]
                rhs = _Rhs(prm, g.z0, 0.0)
                out = rhs.core(0.0, P, R)
                Pd, Rd, Qd = out[0], out[1], out[2]
                dQ[j] = lamP * Pd + lamR * Rd + Qd
                lam_hold[j] = (lamP.copy(), lamR.copy())
                F_hold[j] = (Pd, Rd, Qd)
            S = res.phi[s]
            if S is None:
                lam_seg[s] = (lamP.copy(), lamR.copy())
                continue
            spp, srp, sqp, spr, srr, sqr = S
            newP = lamP * spp + lamR * srp + sqp
            newR = lamP * spr + lamR * srr + sqr
            lamP, lamR = newP, newR
            lam_seg[s] = (lamP.copy(), lamR.copy())
        res.dQ_dtau = dQ
        res.lam_seg = lam_seg
        res.lam_hold = lam_hold
        res.F_hold = F_hold
        res.lam0 = (lamP, lamR)


def hazard(res_or_Q, beta, h0=1.0, half_times=C.HALF_TIMES):
    """Total hazard ``h0 * sum_i g_i Q_i`` for hazard exponent ``beta``."""
    Q = res_or_Q.Q if hasattr(res_or_Q, "Q") else np.asarray(res_or_Q)
    return h0 * float(np.dot(hazard_weights(beta, half_times), Q))


def p_dcs(H):
    """Probability of decompression sickness from the total hazard."""
    return 1.0 - np.exp(-np.asarray(H))


def nitrogen_content(prm, z, P, R):
    r"""Nitrogen per unit tissue volume divided by the solubility (bar).

    :math:`P + \frac{4\pi N}{3\lambda} P_b R^3`. In the model it changes
    only through perfusion, :math:`k (P_{art} - P)`; the gas held in the
    stabilized nuclei at :math:`R = r_0` is included.
    """
    pa = C.P_SURF + C.BAR_PER_MSW * np.asarray(z)
    pb = pa - prm.p_met + prm.two_sigma / R
    return P + 4.0 * np.pi * prm.n_nuclei / (3.0 * prm.ostwald) * pb * R ** 3
