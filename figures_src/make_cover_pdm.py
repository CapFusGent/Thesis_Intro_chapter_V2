"""Cover variants on the field-lines design that add the PdM / risk idea.

B1 "Early warning": one toroidal sector of the field lines frays (jitter and
   colour shift to amber/red); a health-indicator curve with failure threshold
   and shaded intervention window runs across the lower page.
B2 "Risk map": field-line torus with sensor points on the vessel, coloured by
   risk (mostly calm blue, a few amber/red hotspots with a glow).
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
UGENT = "#1E4C9A"
XL, YL = (-1.42, 1.42), (-1.84, 1.84)
BG = LinearSegmentedColormap.from_list("bg", ["#0B1E44", UGENT, "#0B1E44"])
COOL = LinearSegmentedColormap.from_list("c", ["#3A66AB", "#9BC4FF", "#FFFFFF"])
RISK = LinearSegmentedColormap.from_list("r", ["#9BC4FF", "#FFD166",
                                               "#F4A300", "#E63946"])
ANG = (np.deg2rad(38), 0, np.deg2rad(8))


def rot(p, ax, ay, az):
    cx, sx, cy, sy, cz, sz = (np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay),
                              np.cos(az), np.sin(az))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return p @ (Rz @ Ry @ Rx).T


def page():
    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    ax.axis("off")
    ax.imshow(np.linspace(0, 1, 512).reshape(-1, 1), extent=(*XL, *YL),
              aspect="auto", cmap=BG, zorder=0)
    return fig, ax


def segs(p2):
    pts = p2.reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], 1)


def field_lines(ax, fray=None, rng=None):
    """Draw nested helical field lines; `fray(u)` in [0,1] adds damage."""
    for i, (r, q) in enumerate([(0.62, 3.3), (0.5, 2.9), (0.38, 2.5),
                                (0.26, 2.1), (0.14, 1.7)]):
        for ph in np.linspace(0, 2 * np.pi, 3, endpoint=False):
            u = np.linspace(0, 2 * np.pi * 6, 4000)
            v = ph + i + u / q
            dmg = fray(np.mod(u, 2 * np.pi)) if fray else np.zeros_like(u)
            jit = 0.0
            if fray is not None:
                jit = dmg * 0.022 * rng.standard_normal(u.size)
                jit = np.convolve(jit, np.ones(7) / 7, mode="same")
            rr = r + jit
            p = np.stack([(1 + rr * np.cos(v)) * np.cos(u),
                          (1 + rr * np.cos(v)) * np.sin(u), rr * np.sin(v)], -1)
            p = rot(p, *ANG)
            core = 1 - r / 0.62
            s = segs(p[:, :2])
            depth = (p[:-1, 2] + p[1:, 2]) / 2
            if fray is None or dmg.max() == 0:
                lc = LineCollection(s, cmap=COOL, norm=plt.Normalize(-1.6, 1.6),
                                    linewidths=0.6 + 0.9 * core,
                                    alpha=0.55 + 0.4 * core)
                lc.set_array(depth + core * 1.2)
            else:
                d = (dmg[:-1] + dmg[1:]) / 2
                lc = LineCollection(s, cmap=RISK, norm=plt.Normalize(0, 1),
                                    linewidths=0.6 + 0.9 * core,
                                    alpha=0.55 + 0.4 * core)
                lc.set_array(d)
            lc.set_zorder(2)
            ax.add_collection(lc)


def glow(ax, x, y, col, s=40, z=5):
    for k, a in zip([9, 6, 3.5, 2], [0.06, 0.1, 0.18, 0.4]):
        ax.scatter(x, y, s=s * k, color=col, alpha=a, lw=0, zorder=z)
    ax.scatter(x, y, s=s * 0.6, color="white", alpha=0.95, lw=0, zorder=z + 1)


def main():
    # ---------------------------------------------------------------- B1
    rng = np.random.default_rng(11)
    fig, ax = page()
    centre, width = np.deg2rad(320), np.deg2rad(55)


    def fray(u):
        d = np.angle(np.exp(1j * (u - centre)))
        return np.clip(1 - np.abs(d) / width, 0, 1) ** 1.5


    field_lines(ax, fray=fray, rng=rng)
    # health indicator across the lower page
    t = np.linspace(XL[0], XL[1], 700)
    x = (t - XL[0]) / (XL[1] - XL[0])
    hi = -1.18 - 0.42 * x ** 2.6 + 0.008 * rng.standard_normal(t.size)
    obs = x <= 0.68
    ax.add_collection(LineCollection(segs(np.stack([t[obs], hi[obs]], -1)),
                      colors="#DCE8FF", linewidths=1.4, alpha=0.9, zorder=4))
    ax.plot(t[~obs], (-1.18 - 0.42 * x ** 2.6)[~obs], color="#FFD166", lw=1.2,
            ls=(0, (4, 3)), zorder=4)
    thr = -1.52
    ax.plot(XL, [thr, thr], color="#E63946", lw=0.9, ls=(0, (6, 4)), alpha=0.9,
            zorder=3)
    x_fail = ((-(thr + 1.18) / 0.42) ** (1 / 2.6))
    t_fail = XL[0] + x_fail * (XL[1] - XL[0])
    t_now = XL[0] + 0.68 * (XL[1] - XL[0])
    for k, a in enumerate(np.linspace(0.02, 0.16, 12)):
        x0 = t_fail - 0.34 + k * 0.022
        ax.fill_between([x0, x0 + 0.022], thr - 0.06, -1.02, color="#FFD166",
                        alpha=a, lw=0, zorder=1)
    glow(ax, t_now, -1.18 - 0.42 * 0.68 ** 2.6, "#9BC4FF", s=10)
    fig.savefig(ROOT / "background_earlywarning.jpg", dpi=300,
                pil_kwargs={"quality": 92})
    plt.close(fig)

    # ---------------------------------------------------------------- B2
    rng = np.random.default_rng(5)
    fig, ax = page()
    field_lines(ax)
    # sensor points on the outer flux surface (vessel), coloured by risk
    nu, nv = 36, 14
    U, V = np.meshgrid(np.linspace(0, 2 * np.pi, nu, endpoint=False),
                       np.linspace(0, 2 * np.pi, nv, endpoint=False))
    U, V = U.ravel(), V.ravel()
    r = 0.70
    P = rot(np.stack([(1 + r * np.cos(V)) * np.cos(U), (1 + r * np.cos(V)) *
                      np.sin(U), r * np.sin(V)], -1), *ANG)
    risk = np.full(U.size, 0.0)
    for uc, vc, amp in [(np.deg2rad(300), np.deg2rad(-20), 1.0),
                        (np.deg2rad(200), np.deg2rad(70), 0.6),
                        (np.deg2rad(40), np.deg2rad(-90), 0.45)]:
        du = np.angle(np.exp(1j * (U - uc)))
        dv = np.angle(np.exp(1j * (V - vc)))
        risk = np.maximum(risk, amp * np.exp(-(du ** 2 / 0.08 + dv ** 2 / 0.35)))
    front = (P[:, 2] - P[:, 2].min()) / np.ptp(P[:, 2])
    for k in np.argsort(P[:, 2]):
        col = RISK(risk[k])
        if risk[k] > 0.35:
            glow(ax, P[k, 0], P[k, 1], col, s=14 + 46 * risk[k], z=4)
        else:
            ax.scatter(P[k, 0], P[k, 1], s=8 + 16 * front[k], color=col,
                       alpha=0.4 + 0.55 * front[k], lw=0, zorder=4)
    fig.savefig(ROOT / "background_riskmap.jpg", dpi=300,
                pil_kwargs={"quality": 92})
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
