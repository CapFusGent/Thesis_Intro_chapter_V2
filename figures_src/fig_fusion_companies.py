"""Private fusion companies (FIA Global Fusion Industry Report 2026).

One bubble per company: x = year founded, row = country of headquarters,
colour and shape = design family (same encoding as the world device map),
area = total declared funding (hollow = not disclosed). Companies above
US$1 billion are named.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from thesis_style import apply, C, FULL, savefig

apply()
df = pd.read_csv(Path(__file__).parent / "data" / "fia2026_companies.csv",
                 comment="#")

FAM = {
    "tokamak": ("Tokamak", C["blue"], "o"),
    "spherical tokamak": ("Spherical tokamak", "#5F86C1", "h"),
    "stellarator": ("Stellarator", C["orange"], "D"),
    "laser / inertial": ("Laser / inertial", C["red"], "*"),
    "magnetic mirror": ("Magnetic mirror", C["purple"], "s"),
    "pinch": ("Pinch", C["olive"], "^"),
    "other": ("Other concepts", C["teal"], "v"),
}

order = df.country.value_counts()
countries = list(order.index)                    # most companies first
# rows: USA gets a taller band
band = {c: (3.0 if c == "USA" else 0.85) for c in countries}
ypos, y = {}, 0.0
for c in countries:
    ypos[c] = y + band[c] / 2
    y += band[c]


def size(f):
    return 6 if np.isnan(f) else 4 + 220 * np.sqrt(f / 3e9)


LABEL_OFF = {"TAE": (10, 0, "left"), "SHINE": (-9, 0, "right"),
             "Helion": (10, 2, "left"), "CFS": (-12, 0, "right"),
             "Pacific Fusion": (0, 12, "center")}
fig, ax = plt.subplots(figsize=(FULL, 0.82 * FULL))
for c in countries:
    sub = df[df.country == c].sort_values("founded")
    n = len(sub)
    # spread companies founded in similar years vertically inside the band
    offs = np.zeros(n)
    if n > 1:
        h = band[c] * 0.78
        levels = np.linspace(-h / 2, h / 2, 7 if c == "USA" else 3)
        pattern = [3, 0, 5, 2, 6, 1, 4] if c == "USA" else [1, 0, 2]
        offs = np.array([levels[pattern[k % len(pattern)]] for k in range(n)])
    for (_, d), o in zip(sub.iterrows(), offs):
        lab, col, mk = FAM[d.family]
        f = d.funding_usd
        ms = size(f) * (2.2 if mk == "*" else 1.0)
        ax.scatter(d.founded, ypos[c] + o, s=ms, marker=mk,
                   facecolor="white" if np.isnan(f) else col + "CC",
                   edgecolor=col, linewidth=0.7, zorder=3)
        if not np.isnan(f) and f >= 1e9:  # at least US$1 bn
            name = (d.company.replace(" Technologies", "")
                    .replace("Commonwealth Fusion Systems", "CFS"))
            dx, dy, ha = LABEL_OFF.get(name, (8, 0, "left"))
            ax.annotate(name, (d.founded, ypos[c] + o), xytext=(dx, dy),
                        textcoords="offset points", fontsize=6.8, ha=ha,
                        va="center", color=C["ink"])
for i, c in enumerate(countries):
    if i % 2 == 0:
        ax.axhspan(ypos[c] - band[c] / 2, ypos[c] + band[c] / 2,
                   color="#F4F6FA", lw=0, zorder=0)
ax.set_yticks([ypos[c] for c in countries])
ax.set_yticklabels([f"{c} ({order[c]})" for c in countries], fontsize=7.5)
ax.set_ylim(y, 0)
ax.set_xlim(1983, 2027)
ax.set_xlabel("Year founded")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)

fam_h = [Line2D([], [], ls="", marker=m, ms=5.5 if m != "*" else 8,
                mfc=col, mec=col, label=l) for l, col, m in FAM.values()]
size_h = [plt.scatter([], [], s=size(v), facecolor="#BDBDBD", edgecolor=C["grey"],
                      lw=0.6, label=lab)
          for v, lab in [(1e8, "US\\$0.1 bn"), (1e9, "US\\$1 bn"),
                         (3e9, "US\\$3 bn")]]
size_h.append(plt.scatter([], [], s=size(np.nan), facecolor="white",
                          edgecolor=C["grey"], lw=0.6, label="not disclosed"))
l1 = ax.legend(handles=fam_h, loc="lower left", fontsize=6.8, frameon=False,
               bbox_to_anchor=(0.0, 0.0), title="Design", title_fontsize=7,
               alignment="left", labelspacing=0.35, handletextpad=0.3)
ax.add_artist(l1)
ax.legend(handles=size_h, loc="lower left", fontsize=6.8, frameon=False,
          bbox_to_anchor=(0.27, 0.0), title="Declared funding",
          title_fontsize=7, alignment="left", labelspacing=0.9,
          handletextpad=0.6, borderpad=0.6)
savefig(fig, "fusion_private_companies")
print(order.to_dict(), df.family.value_counts().to_dict())
