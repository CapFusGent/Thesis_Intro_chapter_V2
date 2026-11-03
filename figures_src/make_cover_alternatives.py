"""Two alternative title-page backgrounds (US letter, 300 dpi).

B "Field lines": dark UGent-blue page, a torus drawn only with helical
  magnetic field lines that glow towards the plasma core. Calm and technical.
C "Signals": a torus traced by monitoring time series; lines start healthy
  (smooth, blue) and degrade towards the end (noisy, orange-red), with a
  health-indicator curve crossing the page. Ties fusion to predictive
  maintenance.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

W, H = 8.5, 11.0
ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(3)
UGENT = "#1E4C9A"


def rot(p, ax, ay, az):
    cx, sx, cy, sy, cz, sz = (np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay),
                              np.cos(az), np.sin(az))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return p @ (Rz @ Ry @ Rx).T


def field_line(R, r, q, phase, n=4000, turns=6):
    u = np.linspace(0, 2 * np.pi * turns, n)
    v = phase + u / q
    return np.stack([(R + r * np.cos(v)) * np.cos(u),
                     (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)], -1)


def segments(p2, values):
    pts = p2.reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], 1), (values[:-1] + values[1:]) / 2


# ---------------------------------------------------------------- B
fig = plt.figure(figsize=(W, H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-1.42, 1.42)
ax.set_ylim(-1.84, 1.84)
ax.axis("off")
bg = np.linspace(0, 1, 512).reshape(-1, 1)
ax.imshow(bg, extent=(-1.42, 1.42, -1.84, 1.84), aspect="auto", zorder=0,
          cmap=LinearSegmentedColormap.from_list("bg", ["#0B1E44", UGENT,
                                                        "#0B1E44"]))
glow = LinearSegmentedColormap.from_list("g", ["#3A66AB", "#9BC4FF",
                                               "#FFFFFF", "#FFD166"])
ang = (np.deg2rad(38), 0, np.deg2rad(8))
for i, (r, q) in enumerate([(0.62, 3.3), (0.5, 2.9), (0.38, 2.5),
                            (0.26, 2.1), (0.14, 1.7)]):
    for ph in np.linspace(0, 2 * np.pi, 3, endpoint=False):
        p = rot(field_line(1.0, r, q, ph + i), *ang)
        depth = p[:, 2]
        seg, d = segments(p[:, :2], depth)
        core = 1 - r / 0.62
        lc = LineCollection(seg, cmap=glow, norm=plt.Normalize(-1.6, 1.6),
                            linewidths=0.6 + 0.9 * core,
                            alpha=0.55 + 0.4 * core)
        lc.set_array(d + core * 1.2)
        ax.add_collection(lc)
fig.savefig(ROOT / "background_fieldlines.jpg", dpi=300,
            pil_kwargs={"quality": 92})
plt.close(fig)

# ---------------------------------------------------------------- C
fig = plt.figure(figsize=(W, H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-1.42, 1.42)
ax.set_ylim(-1.84, 1.84)
ax.axis("off")
fig.patch.set_facecolor("white")
health = LinearSegmentedColormap.from_list("h", [UGENT, "#2A9D8F", "#E08A00",
                                                 "#C23B4E"])
ang = (np.deg2rad(40), 0, np.deg2rad(-10))
n_lines = 70
for k in range(n_lines):
    v0 = 2 * np.pi * k / n_lines
    u = np.linspace(0, 2 * np.pi, 1400)
    age = u / (2 * np.pi)                       # 0 healthy -> 1 degraded
    noise = (0.004 + 0.06 * age ** 3) * rng.standard_normal(u.size)
    noise = np.convolve(noise, np.ones(5) / 5, mode="same")
    r = 0.5 + noise
    p = np.stack([(1 + r * np.cos(v0)) * np.cos(u),
                  (1 + r * np.cos(v0)) * np.sin(u), r * np.sin(v0)], -1)
    p = rot(p, *ang)
    seg, a = segments(p[:, :2], age)
    front = (p[:, 2].mean() + 1) / 2
    lc = LineCollection(seg, cmap=health, norm=plt.Normalize(0, 1),
                        linewidths=0.7, alpha=0.35 + 0.55 * front)
    lc.set_array(a)
    ax.add_collection(lc)
# health-indicator curve across the lower part of the page
t = np.linspace(-1.42, 1.42, 600)
hi = -1.45 + 0.4 * (1 - ((t + 1.42) / 2.84) ** 2.4)
hi += 0.012 * rng.standard_normal(t.size)
seg, a = segments(np.stack([t, hi], -1), (t + 1.42) / 2.84)
lc = LineCollection(seg, cmap=health, norm=plt.Normalize(0, 1),
                    linewidths=2.2)
lc.set_array(a)
ax.add_collection(lc)
ax.plot([-1.42, 1.42], [-1.6, -1.6], color="#C23B4E", lw=0.8, ls="--")
fig.savefig(ROOT / "background_signals.jpg", dpi=300,
            pil_kwargs={"quality": 92})
plt.close(fig)
print("done")
