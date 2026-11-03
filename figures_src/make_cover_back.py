"""Back cover (side B, 'maintenance view') matching the field-lines front.

Same torus and view as the front, mirrored left-right, with: a degrading
sector (frayed amber/red field lines), sensor points on the vessel coloured
by risk, and a health-indicator curve with failure threshold and
intervention window. Output: ../background_back.jpg (same size as front).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import make_cover_pdm as m          # reuses page(), field_lines(), glow(), ...

rng = np.random.default_rng(21)
fig, ax = m.page()
# mirror the torus (not the page): the back shows the other side of the machine
_rot = m.rot


def _rot_mirror(p, *a):
    q = _rot(p, *a)
    q[..., 0] *= -1
    return q


m.rot = _rot_mirror
centre, width = np.deg2rad(320), np.deg2rad(55)


def fray(u):
    d = np.angle(np.exp(1j * (u - centre)))
    return np.clip(1 - np.abs(d) / width, 0, 1) ** 1.5


m.field_lines(ax, fray=fray, rng=rng)

# sensor ring on the vessel; risk peaks where the field lines fray
nu, nv = 36, 14
U, V = np.meshgrid(np.linspace(0, 2 * np.pi, nu, endpoint=False),
                   np.linspace(0, 2 * np.pi, nv, endpoint=False))
U, V = U.ravel(), V.ravel()
r = 0.70
P = m.rot(np.stack([(1 + r * np.cos(V)) * np.cos(U), (1 + r * np.cos(V)) *
                    np.sin(U), r * np.sin(V)], -1), *m.ANG)
du = np.angle(np.exp(1j * (U - centre)))
risk = np.exp(-du ** 2 / 0.12) * (0.55 + 0.45 * np.cos(V + 0.4) ** 2)
risk = np.maximum(risk, 0.4 * np.exp(-np.angle(np.exp(1j * (U - np.deg2rad(150))))
                                     ** 2 / 0.05) * (np.cos(V) > 0.6))
front = (P[:, 2] - P[:, 2].min()) / np.ptp(P[:, 2])
for k in np.argsort(P[:, 2]):
    col = m.RISK(risk[k])
    if risk[k] > 0.4:
        m.glow(ax, P[k, 0], P[k, 1], col, s=12 + 40 * risk[k], z=4)
    else:
        ax.scatter(P[k, 0], P[k, 1], s=7 + 14 * front[k], color=col,
                   alpha=0.35 + 0.5 * front[k], lw=0, zorder=4)

m.rot = _rot
# health indicator: observed (solid), forecast (dashed), threshold, window
XL = m.XL
t = np.linspace(XL[0], XL[1], 700)
x = (t - XL[0]) / (XL[1] - XL[0])
base = -1.22 - 0.38 * x ** 2.6
hi = base + 0.008 * rng.standard_normal(t.size)
now = 0.66
obs = x <= now
ax.add_collection(LineCollection(m.segs(np.stack([t[obs], hi[obs]], -1)),
                  colors="#DCE8FF", linewidths=1.4, alpha=0.9, zorder=4))
ax.plot(t[~obs], base[~obs], color="#FFD166", lw=1.2, ls=(0, (4, 3)),
        zorder=4)
lo, hi_b = base * 1.0, base * 1.0
spread = 0.12 * np.clip((x - now) / (1 - now), 0, 1)
ax.fill_between(t[~obs], base[~obs] - spread[~obs], base[~obs] + spread[~obs],
                color="#FFD166", alpha=0.10, lw=0, zorder=3)
thr = -1.55
ax.plot(XL, [thr, thr], color="#E63946", lw=0.9, ls=(0, (6, 4)), alpha=0.9,
        zorder=3)
x_fail = (-(thr + 1.22) / 0.38) ** (1 / 2.6)
t_fail = XL[0] + x_fail * (XL[1] - XL[0])
t_now = XL[0] + now * (XL[1] - XL[0])
m.glow(ax, t_now, -1.22 - 0.38 * now ** 2.6, "#9BC4FF", s=10)
ax.plot([t_fail, t_fail], [thr - 0.04, thr + 0.04], color="#E63946", lw=1.2,
        zorder=4)
ax.annotate("", xy=(t_fail - 0.02, -1.08), xytext=(t_now + 0.02, -1.08),
            arrowprops=dict(arrowstyle="<->", color="#FFD166", lw=0.8))
ax.text((t_now + t_fail) / 2, -1.04, "RUL", color="#FFD166", ha="center",
        va="bottom", fontsize=11, family="serif")
fig.savefig(m.ROOT / "background_back.jpg", dpi=300,
            pil_kwargs={"quality": 92})
plt.close(fig)
print("done")
