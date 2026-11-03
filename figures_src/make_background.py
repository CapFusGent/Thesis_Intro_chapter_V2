"""Title-page background: a mosaic tokamak in two panels.

Left: tilted torus covered in coloured tiles, wrapped by helical coil bands,
with glowing plasma rings. Right: view into the vessel, curved tile wall,
plasma streaks and tiles breaking loose.

Pure matplotlib (2-D projection + painter's algorithm), so it is fully
reproducible. Output: ../background.jpg at US-letter size, 300 dpi (the book
class default paper, as used by \\BackgroundPic).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import to_rgb

rng = np.random.default_rng(7)
PAL = ["#E63946", "#F4A300", "#F7E03C", "#6CC04A", "#2BB673",
       "#2FA4E7", "#3B4CC0", "#7B3FA0", "#E0368F", "#F37021"]
EDGE = "#151515"
W, H = 8.5, 11.0
OUT = Path(__file__).resolve().parent.parent / "background.jpg"


def tile_colours(n):
    return [PAL[i] for i in rng.integers(0, len(PAL), n)]


def rot(points, ax, ay, az):
    """Rotate Nx3 points by angles about x, y, z."""
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return points @ (Rz @ Ry @ Rx).T


def project(p, d=6.0):
    """Weak perspective: returns 2-D points and depth (larger = nearer)."""
    f = d / (d - p[..., 2])
    return p[..., :2] * f[..., None], p[..., 2]


def torus_quads(R, r, nu, nv, u0=0, u1=2 * np.pi, v0=0, v1=2 * np.pi):
    u = np.linspace(u0, u1, nu + 1)
    v = np.linspace(v0, v1, nv + 1)
    U, V = np.meshgrid(u, v, indexing="ij")
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    P = np.stack([X, Y, Z], -1)
    quads = []
    for i in range(nu):
        for j in range(nv):
            quads.append([P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1]])
    return np.array(quads)


def ribbon_quads(R, r, turns, phase, width, n=140, u_span=2 * np.pi):
    """Helical band around the torus (a coil winding), as small quads."""
    u = np.linspace(0, u_span, n + 1)
    v = phase + turns * u
    rr = r * 1.18
    centre = np.stack([(R + rr * np.cos(v)) * np.cos(u),
                       (R + rr * np.cos(v)) * np.sin(u),
                       rr * np.sin(v)], -1)
    # band direction: along the toroidal direction, offset +/- width
    tang = np.stack([-np.sin(u), np.cos(u), np.zeros_like(u)], -1)
    a = centre - tang * width / 2
    b = centre + tang * width / 2
    quads = [[a[i], a[i + 1], b[i + 1], b[i]] for i in range(n)]
    return np.array(quads)


def draw_quads(ax, quads3d, colours, lw=0.9, shade=True, alpha=1.0):
    pts, depth = project(quads3d)
    order = np.argsort(depth.mean(axis=1))          # far to near
    cols = []
    for k in order:
        c = np.array(to_rgb(colours[k]))
        if shade:
            # mild depth shading, keeps the flat poster look
            t = (depth[k].mean() + 1.6) / 3.2
            c = np.clip(c * (0.72 + 0.28 * t), 0, 1)
        cols.append(c)
    pc = PolyCollection(pts[order], facecolors=cols, edgecolors=EDGE,
                        linewidths=lw, alpha=alpha, zorder=2)
    ax.add_collection(pc)


def glow_line(ax, x, y, colour, width=2.0, layers=7, alpha=0.5, z=5):
    for k in range(layers, 0, -1):
        ax.plot(x, y, color=colour, lw=width * k * 1.6,
                alpha=alpha / (k * 1.6), solid_capstyle="round", zorder=z)
    ax.plot(x, y, color="white", lw=width * 0.6, alpha=0.9, zorder=z + 1)


# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(W, H), dpi=300)
fig.patch.set_facecolor("white")
gap = 0.012
axL = fig.add_axes([0, 0, 0.5 - gap / 2, 1])
axR = fig.add_axes([0.5 + gap / 2, 0, 0.5 - gap / 2, 1])
for ax in (axL, axR):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# ---------------- Left panel: the torus from outside ----------------------
axL.set_xlim(-1.45, -0.08)
axL.set_ylim(-1.75, 1.75)
ang = (np.deg2rad(42), np.deg2rad(-12), np.deg2rad(90))   # tilt and turn
R, r = 1.0, 0.46

tq = torus_quads(R, r, 44, 20)
tq = rot(tq.reshape(-1, 3), *ang).reshape(tq.shape)
cols = tile_colours(len(tq))

coil_q, coil_c = [], []
for turns, phase, wid in [(1.0, 0.3, 0.16), (1.0, 3.4, 0.14)]:
    q = ribbon_quads(R, r, turns, phase, wid)
    q = rot(q.reshape(-1, 3), *ang).reshape(q.shape)
    coil_q.append(q)
    # colour runs along the band in short segments
    coil_c.extend(tile_colours(len(q)))
for u0 in np.deg2rad([25, 70, 115, 160, 200, 245, 290, 335]):
    v = np.linspace(0, 2 * np.pi, 25)
    rr = r * 1.22
    du = 0.09
    q = []
    for k in range(24):
        pts = []
        for vv, uu in [(v[k], u0 - du), (v[k + 1], u0 - du),
                       (v[k + 1], u0 + du), (v[k], u0 + du)]:
            pts.append([(R + rr * np.cos(vv)) * np.cos(uu),
                        (R + rr * np.cos(vv)) * np.sin(uu), rr * np.sin(vv)])
        q.append(pts)
    q = rot(np.array(q).reshape(-1, 3), *ang).reshape(-1, 4, 3)
    coil_q.append(q)
    coil_c.extend(tile_colours(len(q)))
coil_q = np.concatenate(coil_q)

allq = np.concatenate([tq, coil_q])
allc = cols + list(coil_c)
lws = None
draw_quads(axL, allq, allc, lw=0.7)

# plasma rings: poloidal cross sections that glow
for u0 in [np.deg2rad(150), np.deg2rad(205)]:
    v = np.linspace(0, 2 * np.pi, 300)
    ring = np.stack([(R + 0.75 * r * np.cos(v)) * np.cos(u0),
                     (R + 0.75 * r * np.cos(v)) * np.sin(u0),
                     0.75 * r * np.sin(v)], -1)
    ring = rot(ring, *ang)
    p, _ = project(ring)
    for shift, colr in [(0, "#FF7AD9"), (0.025, "#9BE7FF"), (-0.02, "#FFFFFF")]:
        glow_line(axL, p[:, 0] + shift, p[:, 1] + shift * 0.5, colr,
                  width=1.6, alpha=0.35)

# ---------------- Right panel: inside the vessel --------------------------
axR.set_xlim(0, 1)
axR.set_ylim(0, 1.294)        # panel aspect: (W/2) x H

# curved wall: tiles on concentric elliptical bands converging to a centre
cx, cy = 0.72, 0.62
nb, ns = 13, 30
th = np.linspace(0, 2 * np.pi, ns + 1)
rad = np.geomspace(0.07, 1.6, nb + 1)
wall, wcol = [], []
for i in range(nb):
    for j in range(ns):
        r0, r1 = rad[i], rad[i + 1]
        t0, t1 = th[j], th[j + 1]
        sq = 1.25        # vertical stretch
        poly = [[cx + r0 * np.cos(t0), cy + sq * r0 * np.sin(t0)],
                [cx + r1 * np.cos(t0), cy + sq * r1 * np.sin(t0)],
                [cx + r1 * np.cos(t1), cy + sq * r1 * np.sin(t1)],
                [cx + r0 * np.cos(t1), cy + sq * r0 * np.sin(t1)]]
        wall.append(poly)
        wcol.append(PAL[rng.integers(len(PAL))])
axR.add_collection(PolyCollection(wall, facecolors=wcol, edgecolors=EDGE,
                                  linewidths=0.9, zorder=1))

# plasma: many glowing elliptical arcs around the magnetic axis
glow_cols = ["#FFD166", "#FF7AD9", "#9BE7FF", "#B8F2A0", "#FFFFFF", "#C9A7FF"]
for k in range(46):
    a = rng.uniform(0.12, 0.34)
    b = a * rng.uniform(1.1, 1.7)
    t0 = rng.uniform(0, 2 * np.pi)
    span = rng.uniform(1.2, 3.6)
    t = np.linspace(t0, t0 + span, 120)
    tilt = rng.uniform(-0.5, 0.5)
    x = a * np.cos(t)
    y = b * np.sin(t)
    xr = 0.73 + x * np.cos(tilt) - y * np.sin(tilt)
    yr = 0.62 + x * np.sin(tilt) + y * np.cos(tilt)
    glow_line(axR, xr, yr, glow_cols[k % len(glow_cols)],
              width=rng.uniform(0.5, 1.1), layers=4, alpha=0.45, z=6)
# tiles breaking loose from the wall (lower left) and drifting into plasma
n_frag = 160
fx = rng.uniform(0.0, 0.42, n_frag)
fy = rng.uniform(0.0, 0.42, n_frag) * (1 - fx * 1.2)
frag = []
for x0, y0 in zip(fx, fy):
    w_, h_ = rng.uniform(0.012, 0.035), rng.uniform(0.02, 0.05)
    a_ = np.deg2rad(rng.uniform(-35, 35))
    c, s = np.cos(a_), np.sin(a_)
    pts = np.array([[0, 0], [w_, 0], [w_, h_], [0, h_]])
    pts = pts @ np.array([[c, -s], [s, c]]).T + [x0, y0]
    frag.append(pts)
axR.add_collection(PolyCollection(frag, facecolors=tile_colours(n_frag),
                                  edgecolors=EDGE, linewidths=0.6, zorder=7))
# a few iridescent (glassy) tiles on the wall
for (x0, y0) in [(0.06, 0.36), (0.11, 0.36), (0.42, 0.29), (0.47, 0.29),
                 (0.33, 1.08)]:
    grad = np.linspace(0, 1, 64).reshape(1, -1)
    axR.imshow(grad, extent=(x0, x0 + 0.045, y0, y0 + 0.085), cmap="cool",
               alpha=0.55, zorder=8, aspect="auto")
    axR.add_patch(matplotlib.patches.Rectangle((x0, y0), 0.045, 0.085,
                  fill=False, ec=EDGE, lw=0.8, zorder=9))

fig.savefig(OUT, dpi=300, pil_kwargs={"quality": 92})
fig.savefig("/tmp/background_preview.png", dpi=60)
print("saved", OUT)
