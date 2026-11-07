"""Side-view scuba diver drawn from patches in a local frame.

The diver is built once, in metres, horizontal and prone with the head
at +x, and then placed with an affine transform every frame. Only the
fins move (a slow flutter kick). The drawing carries no physics; the
diver's depth in the animations comes from the model.
"""

import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon
from matplotlib.transforms import Affine2D

SUIT = "#1f2a33"
SUIT_HI = "#2f3f4b"
SKIN = "#c99a7a"
TANK = "#aab6bf"
TANK_HI = "#d7dee3"
STRAP = "#11181d"


class Diver:
    """A diver made of matplotlib patches."""

    def __init__(self, ax, accent="#2a78d6", scale=1.0, zorder=10):
        self.ax = ax
        self.scale = scale
        self.accent = accent
        self.z = zorder
        self.parts = []          # (patch, kind)
        self._build()

    def _add(self, p, kind="body", z=0):
        p.set_zorder(self.z + z)
        self.ax.add_patch(p)
        self.parts.append((p, kind))
        return p

    def _build(self):
        a = self.accent
        # tank on the back (y > 0 is the back)
        self._add(FancyBboxPatch((-0.02, 0.13), 0.72, 0.17,
                                 boxstyle="round,pad=0,rounding_size=0.08",
                                 fc=TANK, ec="#6b7780", lw=0.6), z=0)
        self._add(FancyBboxPatch((0.02, 0.24), 0.62, 0.035,
                                 boxstyle="round,pad=0,rounding_size=0.02",
                                 fc=TANK_HI, ec="none"), z=0.1)
        self._add(Polygon([[0.70, 0.18], [0.80, 0.20], [0.80, 0.25],
                           [0.70, 0.25]], fc="#58636b", ec="none"), z=0.1)
        # torso with BCD
        self._add(Ellipse((0.32, 0.02), 0.86, 0.34, fc=SUIT, ec="none"),
                  z=1)
        self._add(Ellipse((0.36, 0.07), 0.62, 0.22, fc=SUIT_HI,
                          ec="none"), z=1.1)
        for xs in (0.15, 0.48):
            self._add(Polygon([[xs, -0.15], [xs + 0.05, -0.15],
                               [xs + 0.05, 0.19], [xs, 0.19]],
                              fc=STRAP, ec="none"), z=1.2)
        # head, hood, mask and regulator
        self._add(Circle((0.86, 0.0), 0.13, fc=SUIT, ec="none"), z=2)
        self._add(Polygon([[0.88, -0.13], [1.00, -0.10], [1.00, -0.02],
                           [0.90, -0.02]], fc="#9fd3ff", ec=STRAP,
                          lw=0.8, alpha=0.95), z=2.1)
        self._add(Circle((0.97, 0.05), 0.035, fc="#3a3a3a", ec="none"),
                  z=2.1)
        self._add(Polygon([[0.92, 0.02], [0.99, 0.03], [0.99, 0.08],
                           [0.92, 0.07]], fc=SKIN, ec="none"), z=2.05)
        # hose from first stage to mouth
        th = np.linspace(0.0, np.pi, 12)
        hose = np.c_[0.82 + 0.13 * np.cos(th) * 0.9,
                     0.20 + 0.06 * np.sin(th)]
        self._add(Polygon(np.r_[hose, hose[::-1] + [0, -0.02]],
                          fc="#2b2b2b", ec="none", closed=True), z=1.9)
        # arms (forearms folded under the chest, as in a good trim)
        self._add(Polygon([[0.56, -0.06], [0.70, -0.14], [0.50, -0.21],
                           [0.42, -0.14]], fc=SUIT_HI, ec="none"), z=1.5)
        self._add(Polygon([[0.44, -0.18], [0.28, -0.22], [0.30, -0.16],
                           [0.46, -0.13]], fc=SUIT, ec="none"), z=1.6)
        # legs: thighs and shins (static), fins (kick)
        self._add(Polygon([[0.0, 0.10], [-0.48, 0.07], [-0.48, -0.06],
                           [0.0, -0.13]], fc=SUIT, ec="none"), z=0.8)
        self.shin = self._add(Polygon([[-0.46, 0.07], [-0.86, 0.06],
                                       [-0.86, -0.03], [-0.46, -0.06]],
                                      fc=SUIT_HI, ec="none"), "shin", 0.7)
        self.fin = self._add(Polygon([[-0.84, 0.05], [-1.50, 0.12],
                                      [-1.56, 0.02], [-1.50, -0.07],
                                      [-0.84, -0.03]], fc=a,
                                     ec="#0b0b0b", lw=0.4, alpha=0.95),
                             "fin", 0.6)

    def place(self, x, y, angle=0.0, flip=False, kick=0.0):
        """Place the diver.

        ``angle`` (deg) tilts the body, ``flip`` mirrors it so that the
        head points to -x, and ``kick`` (deg) rotates the fins about the
        ankle.
        """
        sx = -self.scale if flip else self.scale
        base = (Affine2D().scale(sx, self.scale).rotate_deg(angle)
                .translate(x, y) + self.ax.transData)
        ankle = Affine2D().rotate_deg_around(-0.86, 0.01, kick)
        knee = Affine2D().rotate_deg_around(-0.47, 0.0, 0.35 * kick)
        for p, kind in self.parts:
            if kind == "fin":
                p.set_transform(ankle + knee + base)
            elif kind == "shin":
                p.set_transform(knee + base)
            else:
                p.set_transform(base)

    def mouth(self, x, y, angle=0.0, flip=False):
        """World position of the regulator exhaust for bubble release."""
        sx = -self.scale if flip else self.scale
        tr = (Affine2D().scale(sx, self.scale).rotate_deg(angle)
              .translate(x, y))
        return tr.transform((0.95, 0.12))
