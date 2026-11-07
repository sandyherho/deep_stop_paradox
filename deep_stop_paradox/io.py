"""Output paths, cache and plain-text reports."""

import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
DIRS = {k: os.path.join(OUT, k) for k in
        ("figures", "animations", "data", "reports", "cache")}


def path(kind, name):
    d = DIRS[kind]
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def save_cache(name, **arrays):
    np.savez_compressed(path("cache", name + ".npz"), **arrays)


def load_cache(name):
    with np.load(path("cache", name + ".npz"), allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def has_cache(name):
    return os.path.exists(path("cache", name + ".npz"))


def write_csv(name, header, rows):
    with open(path("data", name + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(["%.8g" % v if isinstance(v, (float, np.floating))
                        else v for v in r])


def write_report(name, lines):
    with open(path("reports", name + ".txt"), "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")
