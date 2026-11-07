"""Critical hazard weighting across air dives (depth, bottom time).

For each dive, the reference is a ZH-L16C GF 90/85 schedule with at least
a 3 min stop at 3 m, placed on a 3 m stop grid that reaches down to 80% of
the bottom depth. One sensitivity run gives the exact marginal value of
moving stop time from the 3 m stop to every deeper stop. For each move,
the critical weighting beta_c is the exponent above which the move lowers
the total hazard; below it, the move raises it. Two moves are reported:
to the stop nearest half the bottom depth (the deep stop proposed for
recreational no-decompression dives), and to the stop nearest the midpoint
between the bottom and the first GF 90/85 stop (a Pyle stop).
"""

import numpy as np

from deep_stop_paradox import io
from deep_stop_paradox.bubble import TwoPhaseModel, hazard_weights
from deep_stop_paradox.buhlmann import gf_schedule
from deep_stop_paradox.control import critical_beta, exchange, sign_changes

DEPTHS = np.array([24.0, 30.0, 36.0, 42.0, 48.0, 54.0, 60.0])
BTS = np.array([10.0, 15.0, 20.0, 30.0, 40.0, 50.0])


def main():
    m = TwoPhaseModel(rtol=1e-8, atol=1e-11)
    nd, nb = DEPTHS.size, BTS.size
    beta_half = np.full((nd, nb), np.nan)
    beta_best = np.full((nd, nb), np.nan)
    z_best = np.full((nd, nb), np.nan)
    tst = np.zeros((nd, nb))
    first = np.zeros((nd, nb))
    nsc = np.zeros((nd, nb), dtype=int)
    # sign of the weighted exchange at large beta (+1: the move raises
    # the hazard when fast tissues dominate, -1: it lowers it)
    side = np.zeros((nd, nb, 2))
    for a, D in enumerate(DEPTHS):
        for b, T in enumerate(BTS):
            grid_first = 3.0 * np.floor(0.8 * D / 3.0)
            s = gf_schedule(D, T, 0.9, 0.85, grid_first=grid_first)
            dw = s.dwell.copy()
            dw[-1] = max(dw[-1], 3.0)
            s = s.with_dwell(dw)
            tst[a, b] = s.tst
            first[a, b] = s.meta["first"]
            r = m.run(s, sensitivities=True)
            last = s.stops.size - 1
            bc = []
            for j in range(last):
                e = exchange(r, last, j)
                bc.append(critical_beta(e))
                nsc[a, b] = max(nsc[a, b],
                                sign_changes(e, 1e-9 * np.abs(e).max()))
            bc = np.array(bc)
            jh = int(np.argmin(np.abs(s.stops[:last] - 0.5 * D)))
            beta_half[a, b] = bc[jh]
            w3 = hazard_weights(3.0)
            side[a, b, 0] = np.sign(np.dot(w3, exchange(r, last, jh)))
            zp = 0.5 * (D + first[a, b])
            jp = int(np.argmin(np.abs(s.stops[:last] - zp)))
            beta_best[a, b] = bc[jp]
            z_best[a, b] = s.stops[jp]
            side[a, b, 1] = np.sign(np.dot(w3, exchange(r, last, jp)))
            print("%2d m %2d min TST %4d  beta_half %.3f beta_pyle %.3f"
                  " at %.0f m" % (D, T, tst[a, b], beta_half[a, b],
                                  beta_best[a, b], z_best[a, b]),
                  flush=True)
    io.save_cache("phase", depths=DEPTHS, bts=BTS, beta_half=beta_half,
                  beta_best=beta_best, z_best=z_best, tst=tst, first=first,
                  sign_changes=nsc, side=side)
    rows = []
    for a, D in enumerate(DEPTHS):
        for b, T in enumerate(BTS):
            rows.append((D, T, tst[a, b], first[a, b], beta_half[a, b],
                         beta_best[a, b], z_best[a, b]))
    io.write_csv("phase_map", ["depth_msw", "bottom_time_min", "tst_min",
                               "gf_first_stop_msw", "beta_c_half_depth",
                               "beta_c_pyle_stop", "pyle_stop_msw"], rows)


if __name__ == "__main__":
    main()
