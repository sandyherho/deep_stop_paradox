"""Figure 8: robustness of the NEDU result to the bubble parameters."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from deep_stop_paradox import io, plotting as pl


def main():
    pl.setup()
    d = io.load_cache("sensitivity")
    n = io.load_cache("nedu")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    ax = axs[0]
    ax.hist(d["ratio0"], bins=16, color=pl.SHALLOW, ec="white", lw=0.5)
    ax.axvline(1.0, color=pl.INK2, lw=0.8)
    ax.set_xlabel(r"hazard ratio deep/shallow at $\beta$ = 0")
    ax.set_ylabel("samples")
    ax = axs[1]
    bc = d["beta_c"]
    ax.hist(bc[np.isfinite(bc)], bins=16, color=pl.SHALLOW, ec="white",
            lw=0.5)
    ax.axvline(float(n["beta_pair"]), color=pl.INK2, ls=":", lw=0.9)
    ax.set_xlabel(r"critical exponent $\beta_c$")
    ax = axs[2]
    pretty = {"diffusivity": "diffusivity $D$",
              "ostwald": r"solubility $\lambda$",
              "two_sigma": r"surface tension $\sigma$",
              "r0": "nucleus radius $r_0$",
              "n_nuclei": "nucleus density $N$",
              "p_met": "metabolic gases $p_m$"}
    names = [pretty[str(k)] for k in d["keys"]]
    rho = [spearmanr(d["X"][:, k], bc, nan_policy="omit")[0]
           for k in range(len(names))]
    y = np.arange(len(names))
    ax.barh(y, rho, color=["#e34948" if r > 0 else "#2a78d6" for r in rho],
            height=0.6)
    ax.axvline(0, color=pl.INK2, lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(-1, 1)
    ax.set_xlabel(r"rank correlation with $\beta_c$")
    ax.grid(axis="y", visible=False)
    pl.label_panels(axs, x=-0.25, y=1.03)
    fig.tight_layout()
    pl.save(fig, "fig08_sensitivity")


if __name__ == "__main__":
    main()
