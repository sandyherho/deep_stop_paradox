"""The NEDU deep-stops trial replayed in the two-phase model.

* Dense trajectories of both test schedules (figures and animations).
* Hazard of each compartment and the critical weighting at which the
  two schedules carry equal total hazard.
* Shadow prices (costates) along the shallow schedule and the marginal
  value of moving stop time from 10 fsw to every deeper stop.
* The range of weightings consistent with the observed DCS incidence.
"""

import numpy as np
from scipy.stats import norm

from deep_stop_paradox import io
from deep_stop_paradox.bubble import TwoPhaseModel, hazard_weights
from deep_stop_paradox.control import critical_beta, exchange, sign_changes
from deep_stop_paradox.profiles import (NEDU_DCS, deep_stop_skew,
                                        nedu_schedule)


def ratio_beta(Qs, Qd, target):
    """Weighting at which H_deep / H_shallow equals ``target``."""
    return critical_beta(Qd - target * Qs)


def main():
    m = TwoPhaseModel(rtol=1e-9, atol=1e-12)
    out = {}
    for w in ("shallow", "deep"):
        s = nedu_schedule(w)
        r = m.run(s, sensitivities=True, dense=True, n_dense=40, split=1.0)
        d = r.dense
        out[w + "_t"] = d["t"]
        out[w + "_z"] = d["z"]
        out[w + "_P"] = d["P"]
        out[w + "_R"] = d["R"]
        out[w + "_Qt"] = d["Q"]
        out[w + "_Q"] = r.Q
        out[w + "_dQ"] = r.dQ_dtau
        out[w + "_dss"] = deep_stop_skew(s)
        out[w + "_runtime"] = s.runtime()
        # costates at the start of each 1-min piece
        segs = r.segments
        tk = np.concatenate([[0.0], np.cumsum([g.dt for g in segs])])[:-1]
        lp = np.array([lam[0] for lam in r.lam_seg]).T
        lr = np.array([lam[1] for lam in r.lam_seg]).T
        out[w + "_lam_t"] = tk
        out[w + "_lamP"] = lp
        out[w + "_lamR"] = lr
        print(w, "done", flush=True)
    Qs, Qd = out["shallow_Q"], out["deep_Q"]
    dQ = Qd - Qs
    out["beta_pair"] = critical_beta(dQ)
    out["sign_changes_pair"] = sign_changes(dQ, 1e-9 * np.abs(dQ).max())
    betas = np.linspace(-1.5, 2.5, 161)
    out["betas"] = betas
    out["ratio"] = np.array([np.dot(hazard_weights(b), Qd)
                             / np.dot(hazard_weights(b), Qs)
                             for b in betas])
    # observed relative risk with a Katz log interval
    (a, n1), (c, n2) = NEDU_DCS["deep"], NEDU_DCS["shallow"]
    rr = (a / n1) / (c / n2)
    se = np.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n2)
    z = norm.ppf(0.975)
    rr_lo, rr_hi = rr * np.exp(-z * se), rr * np.exp(z * se)
    out["rr_obs"] = np.array([rr, rr_lo, rr_hi])
    out["beta_obs"] = np.array([ratio_beta(Qs, Qd, rr),
                                ratio_beta(Qs, Qd, rr_lo),
                                ratio_beta(Qs, Qd, rr_hi)])
    # marginal value of moving 1 min from 10 fsw to each deeper stop
    stops = nedu_schedule("shallow").stops
    last = stops.size - 1
    ex = np.array([exchange(_res_like(out, "shallow"), last, j)
                   for j in range(last)])
    out["exchange"] = ex
    out["exchange_beta"] = np.array([critical_beta(e) for e in ex])
    out["exchange_sign_changes"] = np.array(
        [sign_changes(e, 1e-9 * np.abs(e).max()) for e in ex])
    out["stops"] = stops
    io.save_cache("nedu", **out)
    io.write_csv("nedu_compartments",
                 ["half_time_min", "Q_shallow", "Q_deep", "dQ_deep_minus"
                  "_shallow", "exchange_10_to_70fsw"],
                 zip(m.p.half_times, Qs, Qd, dQ, ex[0]))
    io.write_csv("nedu_ratio", ["beta", "H_deep_over_H_shallow"],
                 zip(betas, out["ratio"]))
    print("beta_pair", out["beta_pair"], "obs", out["beta_obs"])


class _res_like:
    def __init__(self, out, w):
        self.dQ_dtau = out[w + "_dQ"]


if __name__ == "__main__":
    main()
