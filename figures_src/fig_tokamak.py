"""Schematic of a tokamak: plasma torus, toroidal and poloidal field coils,
central solenoid and one helical field line. Original drawing in the thesis
style (2-D projection, painter's ordering by depth)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from thesis_style import apply, C, FULL, savefig

apply()


def rot(p, ax_, az):
    cx, sx, cz, sz = np.cos(ax_), np.sin(ax_), np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return p @ (Rx @ Rz).T


VIEW = (np.deg2rad(-68), np.deg2rad(20))
R0, a, kappa = 1.0, 0.36, 1.6      # major radius, minor radius, elongation


def dshape(t, scale=1.0, tri=0.35):
    """D-shaped poloidal cross section (R, Z)."""
    Rr = R0 + scale * a * np.cos(t + tri * np.sin(t))
    Z = scale * kappa * a * np.sin(t)
    return Rr, Z


def to3d(Rr, Z, phi):
    return np.stack([Rr * np.cos(phi), Rr * np.sin(phi), Z], -1)


fig, ax = plt.subplots(figsize=(FULL, 0.62 * FULL))
ax.set_aspect("equal")
ax.axis("off")
items = []   # (depth, kind, data)

# plasma surface, cut open over a sector so the inside is visible
t = np.linspace(0, 2 * np.pi, 41)
phis = np.linspace(np.deg2rad(285), np.deg2rad(285 + 295), 70)
for i in range(len(phis) - 1):
    for j in range(len(t) - 1):
        q = []
        for (pp, tt) in [(phis[i], t[j]), (phis[i + 1], t[j]),
                         (phis[i + 1], t[j + 1]), (phis[i], t[j + 1])]:
            Rr, Z = dshape(tt)
            q.append(to3d(Rr, Z, pp))
        q = rot(np.array(q), *VIEW)
        shade = 0.55 + 0.45 * np.cos(t[j] - 0.9)
        col = (1.0, 0.62 + 0.25 * shade, 0.35 + 0.35 * shade)
        items.append((q[:, 2].mean(), "poly", (q[:, :2], col, col, 0.4, 1.0)))
# cut face of the plasma at phi = 40 deg (nested surfaces)
for s_, col in [(1.0, "#F6B26B"), (0.7, "#F9CB8F"), (0.4, "#FCE3B8")]:
    Rr, Z = dshape(np.linspace(0, 2 * np.pi, 80), s_)
    for cut in (285, 285 + 295):
        p = rot(to3d(Rr, Z, np.deg2rad(cut)), *VIEW)
        items.append((p[:, 2].mean() + 0.3 + 0.02 * (1 - s_), "poly",
                      (p[:, :2], col, C["orange"], 0.5, 1.0)))

# toroidal field coils: D-shaped bands around the plasma
for phi in np.deg2rad(np.arange(0, 360, 36) + 18):
    Rr, Z = dshape(np.linspace(0, 2 * np.pi, 60), scale=1.55, tri=0.45)
    p = rot(to3d(Rr, Z, phi), *VIEW)
    for k in range(len(p) - 1):
        seg = p[k:k + 2]
        items.append((seg[:, 2].mean(), "line", (seg[:, :2], C["blue"], 2.2)))

# poloidal field coils (horizontal rings) and central solenoid
for Rc, Zc, lw in [(1.75, 0.55, 1.8), (1.75, -0.55, 1.8), (1.25, 1.05, 1.6),
                   (1.25, -1.05, 1.6)]:
    phi = np.linspace(0, 2 * np.pi, 200)
    p = rot(to3d(np.full_like(phi, Rc), np.full_like(phi, Zc), phi), *VIEW)
    for k in range(len(p) - 1):
        items.append((p[k:k + 2, 2].mean(), "line",
                      (p[k:k + 2, :2], C["teal"], lw)))
cs = []
for zc in np.linspace(-0.9, 0.9, 10):
    phi = np.linspace(0, 2 * np.pi, 80)
    p = rot(to3d(np.full_like(phi, 0.22), np.full_like(phi, zc), phi), *VIEW)
    for k in range(len(p) - 1):
        items.append((p[k:k + 2, 2].mean(), "line",
                      (p[k:k + 2, :2], C["purple"], 1.4)))

# one helical field line on a flux surface
u = np.concatenate([np.linspace(np.deg2rad(286), np.deg2rad(395), 500),
                    [np.nan],
                    np.linspace(np.deg2rad(150), np.deg2rad(224), 350)])
Rr, Z = dshape(6.0 * u, 1.02)
fl = rot(to3d(Rr, Z, u), *VIEW)

items.sort(key=lambda x: x[0])
for _, kind, d in items:
    if kind == "poly":
        xy, fc, ec, lw, al = d
        ax.add_patch(plt.Polygon(xy, fc=fc, ec=ec, lw=lw, alpha=al,
                                 zorder=2))
    else:
        xy, col, lw = d
        ax.plot(xy[:, 0], xy[:, 1], color=col, lw=lw, solid_capstyle="butt",
                zorder=2)

# field line on top of the plasma where it faces the viewer
vis = fl[:, 2] > -0.05
xs = np.where(vis, fl[:, 0], np.nan)
ys = np.where(vis, fl[:, 1], np.nan)
ax.plot(xs, ys, color=C["red"], lw=1.0, zorder=3)
ax.autoscale_view()
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
ax.set_xlim(x0 - 0.75, x1 + 0.9)
lab = dict(fontsize=8, va="center")
ax.annotate("toroidal field coils", xy=(1.45, 0.75), xytext=(x1 + 0.1, 1.05),
            color=C["blue"], arrowprops=dict(arrowstyle="-", color=C["grey"],
            lw=0.5), **lab)
ax.annotate("poloidal field coils", xy=(1.72, -0.22), xytext=(x1 + 0.1, 0.25),
            color="#1F7A70", arrowprops=dict(arrowstyle="-", color=C["grey"],
            lw=0.5), **lab)
ax.annotate("plasma", xy=(0.95, -0.2), xytext=(x1 + 0.1, -0.55),
            color="#9A5F00", arrowprops=dict(arrowstyle="-", color=C["grey"],
            lw=0.5), **lab)
ax.annotate("central solenoid\n(drives the plasma current)", xy=(-0.02, -0.55),
            xytext=(x0 - 0.7, -1.05), color=C["purple"],
            arrowprops=dict(arrowstyle="-", color=C["grey"], lw=0.5), **lab)
ax.annotate("helical magnetic\nfield line", xy=(-0.95, 0.05),
            xytext=(x0 - 0.7, 0.55), color=C["red"],
            arrowprops=dict(arrowstyle="-", color=C["grey"], lw=0.5), **lab)
savefig(fig, "tokamak_schematic")
print("done")
