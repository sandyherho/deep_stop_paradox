"""ZH-L16C decompression schedules with gradient factors.

This is the dissolved-gas algorithm that most dive computers implement.
It is used here only to produce reference schedules; the risk of every
schedule is evaluated with the two-phase model.

The tolerated ambient pressure of compartment ``i`` at gradient factor
``gf`` is

    P_tol = (P_i - a_i gf) / (gf / b_i - gf + 1),

GF-low applies at the first stop and GF-high at the surface, with linear
interpolation in depth in between. Stops are on a 3 m grid, the last stop
is at 3 m and stop times are whole minutes.
"""

import numpy as np

from . import constants as C
from .profiles import Schedule, stop_grid


def _load_linear(P, k, pi0, rate_pi, dt):
    """Schreiner solution for a linear change of inspired pressure."""
    e = np.exp(-k * dt)
    return (pi0 + rate_pi * (dt - 1.0 / k)
            - (pi0 - P - rate_pi / k) * e)


def _segment(P, k, z0, z1, dt, f_n2):
    if dt <= 0:
        return P
    pi0 = f_n2 * (C.ambient(z0) - C.P_H2O)
    rate = f_n2 * C.BAR_PER_MSW * (z1 - z0) / dt
    return _load_linear(P, k, pi0, rate, dt)


def ceiling(P, gf):
    """Shallowest tolerated depth (msw) at gradient factor ``gf``."""
    a, b = C.A_ZHL16C, C.B_ZHL16C
    p_tol = (P - a * gf) / (gf / b - gf + 1.0)
    return max(float(np.max((p_tol - C.P_SURF) / C.BAR_PER_MSW)), 0.0)


def gf_schedule(depth, bottom_time, gf_lo=0.3, gf_hi=0.85,
                descent_rate=18.0, ascent_rate=9.0, last_stop=3.0,
                step=3.0, f_n2=C.F_N2_AIR, grid_first=None, max_stop=999):
    """Plan a ZH-L16C GF decompression.

    Parameters
    ----------
    grid_first : float or None
        Deepest depth of the stop grid of the returned schedule. Stops
        deeper than the first required stop get zero dwell, so that the
        same grid can be shared by schedules with different first stops.
    """
    k = np.log(2.0) / C.HALF_TIMES
    P = np.full(C.N_COMP, f_n2 * (C.P_SURF - C.P_H2O))
    t_desc = depth / descent_rate
    P = _segment(P, k, 0.0, depth, t_desc, f_n2)
    P = _segment(P, k, depth, depth, bottom_time - t_desc, f_n2)
    # first stop: ascend until the GF-low ceiling is reached
    z = depth
    first = None
    while z > last_stop:
        znext = z - step if z % step < 1e-9 else step * np.floor(z / step)
        znext = max(znext, last_stop)
        Pn = _segment(P, k, z, znext, (z - znext) / ascent_rate, f_n2)
        if ceiling(Pn, gf_lo) > znext + 1e-9:
            first = step * np.ceil(z / step - 1e-9)
            first = min(first, z)
            break
        P, z = Pn, znext
    if first is None:
        first = last_stop
    # move to the first stop depth on the grid
    P = _segment(P, k, z, first, (z - first) / ascent_rate, f_n2)
    z = first
    stops = stop_grid(first, last_stop, step)
    dwell = np.zeros(stops.size)

    def gf_at(zz):
        if first <= 0:
            return gf_hi
        return gf_hi + (gf_lo - gf_hi) * zz / first
    for j, zs in enumerate(stops):
        z_next = stops[j + 1] if j + 1 < stops.size else 0.0
        t = 0
        while ceiling(P, gf_at(z_next)) > z_next + 1e-9:
            P = _segment(P, k, zs, zs, 1.0, f_n2)
            t += 1
            if t > max_stop:
                raise RuntimeError("stop longer than max_stop")
        dwell[j] = t
        P = _segment(P, k, zs, z_next, (zs - z_next) / ascent_rate, f_n2)
    if grid_first is not None and grid_first > stops[0]:
        full = stop_grid(grid_first, last_stop, step)
        d = np.zeros(full.size)
        d[full.size - stops.size:] = dwell
        stops, dwell = full, d
    return Schedule(depth, bottom_time, stops, dwell, descent_rate,
                    ascent_rate, label="ZH-L16C GF %d/%d"
                    % (round(100 * gf_lo), round(100 * gf_hi)),
                    meta={"gf_lo": gf_lo, "gf_hi": gf_hi, "first": first})
