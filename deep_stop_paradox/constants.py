"""Physical constants and the reference parameter set.

Units used throughout the library
---------------------------------
time      minutes (min)
pressure  bar (absolute)
depth     metres of sea water (msw), with 1 msw = 0.1 bar
length    micrometres (um) for bubble radii

The compartment half-times and the Buhlmann coefficients are those of the
ZH-L16C nitrogen table (version 1b, first compartment 5 min). They are
used twice: as the perfusion time constants of the two-phase model and,
separately, to build gradient-factor schedules for comparison.
"""

import numpy as np

# --------------------------------------------------------------------- #
# Environment and breathing gas
# --------------------------------------------------------------------- #
P_SURF = 1.01325          # bar, sea-level pressure
BAR_PER_MSW = 0.1         # bar per metre of sea water (EN 13319)
P_H2O = 0.0627            # bar, alveolar water vapour (Buhlmann)
F_N2_AIR = 0.79           # nitrogen fraction of air

# fsw -> msw so that 33.066 fsw = 1 atm = 10.1325 msw
MSW_PER_FSW = P_SURF / 33.066 / BAR_PER_MSW

# --------------------------------------------------------------------- #
# ZH-L16C (1b) nitrogen compartments
# --------------------------------------------------------------------- #
HALF_TIMES = np.array([5.0, 8.0, 12.5, 18.5, 27.0, 38.3, 54.3, 77.0,
                       109.0, 146.0, 187.0, 239.0, 305.0, 390.0, 498.0,
                       635.0])
A_ZHL16C = np.array([1.1696, 1.0000, 0.8618, 0.7562, 0.6200, 0.5043,
                     0.4410, 0.4000, 0.3750, 0.3500, 0.3295, 0.3065,
                     0.2835, 0.2610, 0.2480, 0.2327])
B_ZHL16C = np.array([0.5578, 0.6514, 0.7222, 0.7825, 0.8126, 0.8434,
                     0.8693, 0.8910, 0.9092, 0.9222, 0.9319, 0.9403,
                     0.9477, 0.9544, 0.9602, 0.9653])
N_COMP = HALF_TIMES.size
TAU_REF = 40.0            # min, pivot half-time of the hazard weights

# --------------------------------------------------------------------- #
# Bubble physics (reference values; all are varied in the sensitivity
# analysis)
# --------------------------------------------------------------------- #
# N2 diffusivity in aqueous tissue, 2.0e-9 m^2/s, in um^2/min
DIFFUSIVITY = 2.0e-9 * 1.0e12 * 60.0
# Ostwald solubility of N2 in aqueous tissue at 37 C (dimensionless)
OSTWALD = 0.015
# 2 * surface tension, 2 * 0.030 N/m, expressed in bar um
TWO_SIGMA = 2.0 * 0.030 / 1.0e5 * 1.0e6
# Radius of the stabilized gas nuclei (um)
R_NUCLEUS = 2.0
# Number density of nuclei, 1 per mm^3, in um^-3
N_NUCLEI = 1.0e-9
# Partial pressure of O2 + CO2 + H2O inside a tissue bubble (bar)
P_METABOLIC = 0.053 + 0.061 + 0.0627
# Reference bubble volume fraction that makes the hazard O(1)
V_REF = 1.0e-3
# Width of the smooth floor that keeps nuclei at R >= R_NUCLEUS (um)
FLOOR_WIDTH = 0.02

# Post-surface observation window over which hazard is accumulated
SURFACE_TAIL = 24.0 * 60.0


def ambient(z):
    """Ambient pressure (bar) at depth ``z`` (msw)."""
    return P_SURF + BAR_PER_MSW * np.asarray(z, dtype=float)


def inspired_n2(z, f_n2=F_N2_AIR):
    """Alveolar (arterial) nitrogen pressure (bar) at depth ``z``."""
    return f_n2 * (ambient(z) - P_H2O)
