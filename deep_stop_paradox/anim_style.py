"""Scenery and encoding shared by the animation scripts.

Light shafts, drifting particles and the sky are decorative. Everything
that carries a number (depths, tissue tensions, bubble radii, hazards)
comes from the model.
"""

import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from .io import path  # noqa: E402

BG = "#06121c"
FG = "#e8eef2"
FG2 = "#9fb3c2"
WATER = LinearSegmentedColormap.from_list(
    "water", ["#7fd3e6", "#2a9bbf", "#136f96", "#0b4a6e", "#072f49",
              "#041b2c"])
TISSUE = LinearSegmentedColormap.from_list(
    "tissue", ["#184f95", "#3987e5", "#9ec5f4", "#f3ece4", "#f4b7b6",
               "#e34948", "#8f1f1f"])


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": FG,
        "axes.labelcolor": FG,
        "axes.edgecolor": FG2,
        "xtick.color": FG2,
        "ytick.color": FG2,
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
    })


def water_column(ax, x0, x1, ztop, zbot, n=400):
    """Paint a depth-graded water background between ``ztop`` and
    ``zbot`` (both negative-down y coordinates)."""
    grad = np.linspace(0, 1, n)[:, None] * np.ones((1, 4))
    ax.imshow(grad, extent=(x0, x1, zbot, ztop), cmap=WATER,
              aspect="auto", zorder=0, interpolation="bicubic")


class LightShafts:
    """Slowly swaying translucent shafts from the surface."""

    def __init__(self, ax, xs, depth, seed=1):
        rng = np.random.default_rng(seed)
        self.ax = ax
        self.depth = depth
        self.base = np.asarray(xs, dtype=float)
        self.w = rng.uniform(0.4, 1.4, self.base.size)
        self.ph = rng.uniform(0, 2 * np.pi, self.base.size)
        self.polys = []
        for _ in self.base:
            p = Polygon([[0, 0], [0, 0], [0, 0]], closed=True, fc="#dff6ff",
                        ec="none", alpha=0.06, zorder=1)
            ax.add_patch(p)
            self.polys.append(p)

    def update(self, phase):
        for i, p in enumerate(self.polys):
            x = self.base[i] + 0.6 * np.sin(0.7 * phase + self.ph[i])
            sl = 0.10 + 0.04 * np.sin(0.5 * phase + 2 * self.ph[i])
            w = self.w[i]
            p.set_xy([[x - w / 2, 0], [x + w / 2, 0],
                      [x + w / 2 + sl * self.depth * 0.6, -self.depth],
                      [x - w / 2 + sl * self.depth * 0.2, -self.depth]])
            p.set_alpha(0.035 + 0.03 * (1 + np.sin(phase + self.ph[i])))


class Snow:
    """Drifting particles (marine snow)."""

    def __init__(self, ax, x0, x1, ztop, zbot, n=220, seed=3):
        rng = np.random.default_rng(seed)
        self.x0, self.x1, self.ztop, self.zbot = x0, x1, ztop, zbot
        self.x = rng.uniform(x0, x1, n)
        self.y = rng.uniform(zbot, ztop, n)
        self.v = rng.uniform(0.02, 0.08, n)
        self.s = rng.uniform(0.5, 3.0, n)
        self.sc = ax.scatter(self.x, self.y, s=self.s, c="#d8f0ff",
                             alpha=0.35, lw=0, zorder=2)

    def step(self, dt=1.0):
        self.y -= self.v * dt
        self.x += 0.02 * np.sin(self.y * 0.7) * dt
        low = self.y < self.zbot
        self.y[low] = self.ztop
        self.sc.set_offsets(np.c_[self.x, self.y])


def surface_wave(x, phase, amp=0.25):
    return amp * (np.sin(0.9 * x + phase)
                  + 0.4 * np.sin(2.3 * x - 1.7 * phase))


def time_warp(knots_model, knots_video):
    """Monotone piecewise-linear map from video seconds to model time."""
    km = np.asarray(knots_model, dtype=float)
    kv = np.asarray(knots_video, dtype=float)
    return lambda s: np.interp(s, kv, km)


def encode(anim_func, fig, n_frames, name, fps=30, gif_width=800,
           gif_fps=12):
    """Render frames to MP4 with FFmpeg and derive a GIF from it."""
    from matplotlib.animation import FFMpegWriter
    preview = os.environ.get("ANIM_PREVIEW")
    if preview:
        # render a few frames to PNG for inspection, no video
        for k in [int(v) for v in preview.split(",")]:
            anim_func(k)
            fig.savefig(path("cache", "%s_frame%04d.png" % (name, k)),
                        dpi=fig.dpi)
        plt.close(fig)
        return
    mp4 = path("animations", name + ".mp4")
    gif = path("animations", name + ".gif")
    w = FFMpegWriter(fps=fps, codec="libx264", bitrate=6000,
                     extra_args=["-pix_fmt", "yuv420p", "-preset",
                                 "veryfast", "-crf", "20"])
    with w.saving(fig, mp4, dpi=fig.dpi):
        for k in range(n_frames):
            anim_func(k)
            w.grab_frame()
            if k % 100 == 0:
                print("  frame %d / %d" % (k, n_frames), flush=True)
    pal = mp4 + ".palette.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf",
                    "fps=%d,scale=%d:-1:flags=lanczos,"
                    "palettegen=max_colors=160" %
                    (gif_fps, gif_width), pal], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-i",
                    pal, "-lavfi",
                    "fps=%d,scale=%d:-1:flags=lanczos[x];[x][1:v]"
                    "paletteuse=dither=bayer:bayer_scale=5" %
                    (gif_fps, gif_width), gif], check=True)
    os.remove(pal)
    plt.close(fig)
