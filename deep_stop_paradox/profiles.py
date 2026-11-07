"""Dive profiles as sequences of linear depth segments.

A profile is a list of :class:`Segment` objects. Each segment moves the
diver linearly from ``z0`` to ``z1`` (msw) in ``dt`` minutes. Holds at a
decompression stop carry the index of the stop in ``stop``; this is what
the dwell-time gradients refer to.

Bottom time follows the U.S. Navy convention: it runs from leaving the
surface to leaving the bottom, so it includes the descent. Stop times do
not include travel between stops.
"""

from dataclasses import dataclass, field

import numpy as np

from .constants import MSW_PER_FSW, SURFACE_TAIL


@dataclass
class Segment:
    dt: float
    z0: float
    z1: float
    kind: str
    stop: int = -1
    end: bool = True            # last piece of a subdivided segment

    @property
    def rate(self):
        return 0.0 if self.dt <= 0 else (self.z1 - self.z0) / self.dt


@dataclass
class Schedule:
    """A square-profile dive followed by a staged decompression."""

    depth: float                     # bottom depth (msw)
    bottom_time: float               # min, includes descent
    stops: np.ndarray                # stop depths (msw), deepest first
    dwell: np.ndarray                # stop times (min)
    descent_rate: float = 18.0       # msw/min
    ascent_rate: float = 9.0         # msw/min
    tail: float = SURFACE_TAIL       # post-surface window (min)
    label: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.stops = np.asarray(self.stops, dtype=float)
        self.dwell = np.asarray(self.dwell, dtype=float)
        if self.stops.size != self.dwell.size:
            raise ValueError("stops and dwell must have equal length")
        if np.any(np.diff(self.stops) >= 0):
            raise ValueError("stops must be strictly decreasing")

    @property
    def tst(self):
        """Total stop time (min)."""
        return float(np.sum(self.dwell))

    def with_dwell(self, dwell):
        return Schedule(self.depth, self.bottom_time, self.stops.copy(),
                        np.asarray(dwell, dtype=float), self.descent_rate,
                        self.ascent_rate, self.tail, self.label,
                        dict(self.meta))

    def segments(self, include_tail=True, split=None):
        """Return the list of segments of the whole exposure.

        ``split`` (min) subdivides long segments into equal pieces; this
        changes nothing in the solution but gives the costates at more
        points in time.
        """
        segs = []
        t_desc = self.depth / self.descent_rate
        segs.append(Segment(t_desc, 0.0, self.depth, "descent"))
        segs.append(Segment(max(self.bottom_time - t_desc, 0.0),
                            self.depth, self.depth, "bottom"))
        z = self.depth
        for j, (zs, tau) in enumerate(zip(self.stops, self.dwell)):
            segs.append(Segment((z - zs) / self.ascent_rate, z, zs,
                                "ascent"))
            segs.append(Segment(float(tau), zs, zs, "hold", j))
            z = zs
        segs.append(Segment(z / self.ascent_rate, z, 0.0, "ascent"))
        if include_tail:
            segs.append(Segment(self.tail, 0.0, 0.0, "surface"))
        if split is None:
            return segs
        out = []
        for g in segs:
            m = max(int(np.ceil(g.dt / split)), 1)
            for q in range(m):
                za = g.z0 + (g.z1 - g.z0) * q / m
                zb = g.z0 + (g.z1 - g.z0) * (q + 1) / m
                out.append(Segment(g.dt / m, za, zb, g.kind, g.stop,
                                   q == m - 1))
        return out

    def runtime(self):
        """Time from leaving the surface to surfacing (min)."""
        return sum(s.dt for s in self.segments(include_tail=False))

    def depth_curve(self, n=2000, include_tail=False, t_extra=0.0):
        """Sample (t, z) along the profile."""
        segs = self.segments(include_tail=False)
        tk = np.concatenate([[0.0], np.cumsum([s.dt for s in segs])])
        zk = np.array([segs[0].z0] + [s.z1 for s in segs])
        t = np.linspace(0.0, tk[-1] + t_extra, n)
        return t, np.interp(t, tk, zk)


def deep_stop_skew(sched):
    """Deep-stop skew (bar min) as defined in NEDU TR 11-06.

    The area between the ambient-pressure trace of the ascent and the
    surface pressure, from leaving the bottom to surfacing, expressed
    relative to the area of an ascent that does the whole stop time at
    the shallowest stop. Larger values mean stop time spent deeper.
    """
    def area(s):
        segs = s.segments(include_tail=False)[2:]
        a = 0.0
        for g in segs:
            a += 0.1 * 0.5 * (g.z0 + g.z1) * g.dt
        return a
    ref = sched.with_dwell(np.r_[np.zeros(sched.stops.size - 1),
                                 sched.tst])
    return area(sched) - area(ref)


# --------------------------------------------------------------------- #
# The NEDU deep-stops trial (Doolette, Gerth and Gault 2011, NEDU TR
# 11-06): 170 fsw for 30 min on air, descent 60 fsw/min, ascent
# 30 fsw/min, 174 min total stop time.
# --------------------------------------------------------------------- #
NEDU_FSW = np.array([70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0])
NEDU_SHALLOW = np.array([0.0, 0.0, 0.0, 9.0, 20.0, 52.0, 93.0])
NEDU_DEEP = np.array([12.0, 17.0, 15.0, 18.0, 23.0, 17.0, 72.0])
NEDU_DCS = {"shallow": (3, 192), "deep": (10, 198)}


def nedu_schedule(which="shallow", grid_fsw=None):
    """NEDU test schedule on an optional finer stop grid (fsw)."""
    if grid_fsw is None:
        grid_fsw = NEDU_FSW
    grid_fsw = np.asarray(grid_fsw, dtype=float)
    dwell = np.zeros(grid_fsw.size)
    src = NEDU_SHALLOW if which == "shallow" else NEDU_DEEP
    for zf, tau in zip(NEDU_FSW, src):
        dwell[np.argmin(np.abs(grid_fsw - zf))] += tau
    return Schedule(depth=170.0 * MSW_PER_FSW, bottom_time=30.0,
                    stops=grid_fsw * MSW_PER_FSW, dwell=dwell,
                    descent_rate=60.0 * MSW_PER_FSW,
                    ascent_rate=30.0 * MSW_PER_FSW,
                    label="NEDU " + which)


def stop_grid(first, last=3.0, step=3.0):
    """Stop depths from ``first`` down to ``last`` (msw), deepest first."""
    n = int(round((first - last) / step))
    return last + step * np.arange(n, -1, -1, dtype=float)
