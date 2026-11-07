"""Run the whole pipeline: simulations, figures, reports, animations.

Usage::

    python scripts/run_all.py            # everything
    python scripts/run_all.py --no-anim  # skip the animations

Cached simulation results in ``outputs/cache`` are reused by the figure
and report scripts. Delete ``outputs/cache`` to recompute from scratch.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

COMPUTE = ["compute_verification", "compute_nedu", "compute_phase",
           "compute_gf", "compute_sensitivity", "compute_threshold",
           "compute_optimal"]
FIGURES = ["fig00_schematic", "fig01_model", "fig02_verification",
           "fig03_nedu", "fig04_shadow", "fig05_optimal", "fig06_phase",
           "fig07_gf", "fig08_sensitivity", "fig09_threshold"]
REPORTS = ["make_reports"]
ANIMS = ["anim01_shotline", "anim02_microscope", "anim03_redistribution"]


def run(name):
    t0 = time.time()
    print("== %s" % name, flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    subprocess.run([sys.executable, os.path.join(HERE, name + ".py")],
                   check=True, env=env)
    print("   %.0f s" % (time.time() - t0), flush=True)


def main():
    steps = COMPUTE + FIGURES + REPORTS
    if "--no-anim" not in sys.argv:
        steps += ANIMS
    for s in steps:
        run(s)


if __name__ == "__main__":
    main()
