"""Gradient-factor schedules evaluated in the two-phase model.

GF-low sets the depth of the first stop and is the setting divers use to
add deep stops. For two air dives (the NEDU 52 m / 30 min exposure and a
technical 45 m / 25 min dive) GF-low is swept at fixed GF-high and every
schedule is evaluated with the two-phase model. The compartment hazards
are stored so that any weighting can be applied afterwards.
"""

import numpy as np

from deep_stop_paradox import io
from deep_stop_paradox.bubble import TwoPhaseModel
from deep_stop_paradox.buhlmann import gf_schedule
from deep_stop_paradox.constants import MSW_PER_FSW
from deep_stop_paradox.profiles import deep_stop_skew

DIVES = {
    "nedu": dict(depth=170.0 * MSW_PER_FSW, bottom_time=30.0,
                 descent_rate=60.0 * MSW_PER_FSW,
                 ascent_rate=30.0 * MSW_PER_FSW),
    "tech45": dict(depth=45.0, bottom_time=25.0, descent_rate=18.0,
                   ascent_rate=9.0),
}
GF_LO = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85])
GF_HI = (0.70, 0.85)


def main():
    m = TwoPhaseModel(rtol=1e-8, atol=1e-11)
    out = {"gf_lo": GF_LO, "gf_hi": np.array(GF_HI)}
    rows = []
    for name, kw in DIVES.items():
        Q = np.zeros((len(GF_HI), GF_LO.size, 16))
        tst = np.zeros((len(GF_HI), GF_LO.size))
        dss = np.zeros_like(tst)
        first = np.zeros_like(tst)
        for a, hi in enumerate(GF_HI):
            for b, lo in enumerate(GF_LO):
                if lo > hi:
                    Q[a, b] = np.nan
                    tst[a, b] = dss[a, b] = first[a, b] = np.nan
                    continue
                s = gf_schedule(gf_lo=lo, gf_hi=hi, **kw)
                r = m.run(s, sensitivities=False)
                Q[a, b] = r.Q
                tst[a, b] = s.tst
                dss[a, b] = deep_stop_skew(s)
                first[a, b] = s.meta["first"]
                rows.append((name, hi, lo, s.tst, first[a, b], dss[a, b])
                            + tuple(r.Q))
                print(name, hi, lo, s.tst, first[a, b], flush=True)
        out[name + "_Q"] = Q
        out[name + "_tst"] = tst
        out[name + "_dss"] = dss
        out[name + "_first"] = first
    io.save_cache("gf", **out)
    io.write_csv("gf_sweep", ["dive", "gf_hi", "gf_lo", "tst_min",
                              "first_stop_msw", "deep_stop_skew_bar_min"]
                 + ["Q_%02d" % (i + 1) for i in range(16)], rows)


if __name__ == "__main__":
    main()
