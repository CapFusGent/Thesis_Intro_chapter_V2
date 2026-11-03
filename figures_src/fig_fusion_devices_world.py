"""Fusion devices of the world as a tile map.

Each country is a square tile placed roughly at its geographic position; each
device is one symbol. Shape and colour = design family, fill = status
(filled: operating, half: under construction, hollow: planned), black centre
dot = DEMO-class power plant design. Data: IAEA World Survey of Fusion
Devices 2022 (figures_src/data/iaea_fusion_devices_2022.csv).
Set as a full landscape page (sidewaysfigure).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from thesis_style import apply, C, savefig

apply()
df = pd.read_csv(Path(__file__).parent / "data" / "iaea_fusion_devices_2022.csv",
                 comment="#")

FAM = {  # family: (label, colour, marker, size)
    "tokamak": ("Tokamak", C["blue"], "o", 30),
    "spherical tokamak": ("Spherical tokamak", "#5F86C1", "h", 34),
    "stellarator": ("Stellarator / heliotron", C["orange"], "D", 24),
    "laser / inertial": ("Laser / inertial", C["red"], "*", 58),
    "magnetic mirror": ("Magnetic mirror", C["purple"], "s", 24),
    "pinch": ("Pinch / reversed-field pinch", C["olive"], "^", 32),
    "other": ("Other alternative concepts", C["teal"], "v", 32),
}
ORDER = list(FAM)
STATUS = ["operational", "under construction", "planned"]

# tile positions (column, row); rows grow downwards
POS = {
    "Canada": (1, 0), "United States": (1, 1), "Costa Rica": (1, 2),
    "Brazil": (2, 3),
    "European Union": (5, 0), "Sweden": (7, 0),
    "United Kingdom": (5, 1), "Denmark": (6, 1),
    "Russian Federation": (9, 0),
    "France": (5, 2), "Germany": (6, 2), "Czech Republic": (7, 2),
    "Ukraine": (8, 2), "Kazakhstan": (10, 1),
    "Portugal": (4, 3), "Spain": (5, 3), "Switzerland": (6, 3),
    "Italy": (7, 3), "Iran": (9, 3), "Pakistan": (10, 3),
    "China": (12, 2), "Republic of Korea": (13, 2), "Japan": (14, 2),
    "Libya": (6, 4), "Egypt": (7, 4), "India": (11, 4), "Thailand": (12, 4),
    "Australia": (14, 5),
}
SHORT = {"United States": "USA", "Russian Federation": "Russia",
         "United Kingdom": "UK", "Republic of Korea": "Korea",
         "Czech Republic": "Czechia", "European Union": "EU"}
missing = set(df.country) - set(POS)
assert not missing, missing

PER_ROW = 7
TILE = 1.0
fig = plt.figure(figsize=(8.6, 4.55))
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(0.45, 15.1)
ax.set_ylim(5.55, -0.55)

for country, (cx, cy) in POS.items():
    sub = df[df.country == country].copy()
    sub["f"] = sub.configuration.map(ORDER.index)
    sub["s"] = sub.status.map(STATUS.index)
    sub = sub.sort_values(["demo", "f", "s"])
    ax.add_patch(FancyBboxPatch((cx - 0.46, cy - 0.46), 0.92, 0.92,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                fc="#F4F6FA", ec="#D5DCE8", lw=0.6, zorder=0))
    ax.text(cx - 0.40, cy - 0.33, SHORT.get(country, country), fontsize=6.3,
            va="center", ha="left", color=C["ink"])
    ax.text(cx + 0.40, cy - 0.33, str(len(sub)), fontsize=6.3, va="center",
            ha="right", color=C["grey"])
    n = len(sub)
    rows = int(np.ceil(n / PER_ROW))
    for k, (_, d) in enumerate(sub.iterrows()):
        i, j = k % PER_ROW, k // PER_ROW
        x = cx - 0.36 + i * 0.12
        y = cy - 0.15 + j * 0.115
        lab, col, mk, ms = FAM[d.configuration]
        if d.status == "operational":
            fc = col
        elif d.status == "under construction":
            fc = col + "66"
        else:
            fc = "white"
        ax.scatter(x, y, s=ms * 0.55, marker=mk, facecolor=fc, edgecolor=col,
                   linewidth=0.9, zorder=3)
        if d.demo:
            ax.scatter(x, y, s=2.2, color=C["ink"], zorder=4)
        if d.device in ("JET",):
            ax.scatter(x, y, s=ms * 1.9, marker="o", facecolor="none",
                       edgecolor=C["ink"], linewidth=0.7, zorder=4)

# callouts for a few landmark devices
ax.annotate("JET", xy=(4.645, 0.85), xytext=(3.9, 0.62), fontsize=7,
            arrowprops=dict(arrowstyle="-", color=C["grey"], lw=0.5))

# ---- legend in the empty "Atlantic" area (left) and status below it
fam_handles = [Line2D([], [], ls="", marker=FAM[f][2],
                      ms=5.5 if FAM[f][2] != "*" else 8, mfc=FAM[f][1],
                      mec=FAM[f][1], label=FAM[f][0]) for f in ORDER]
st_handles = [Line2D([], [], ls="", marker="o", ms=5.5, mfc=C["grey"],
                     mec=C["grey"], label="operating"),
              Line2D([], [], ls="", marker="o", ms=5.5, mfc=C["grey"] + "66",
                     mec=C["grey"], label="under construction"),
              Line2D([], [], ls="", marker="o", ms=5.5, mfc="white",
                     mec=C["grey"], label="planned"),
              Line2D([], [], ls="", marker=r"$\odot$", ms=7, color=C["ink"],
                     label="DEMO-class plant design")]
l1 = ax.legend(handles=fam_handles + [Line2D([], [], ls="")] + st_handles,
               loc="lower left", ncol=2, fontsize=6.8, handletextpad=0.4,
               labelspacing=0.45, columnspacing=1.0, frameon=False,
               bbox_to_anchor=(0.0, 0.0),
               title="Design (shape and colour) and status (fill)",
               title_fontsize=7, alignment="left")
# ---- totals per family (small bar summary, top right of the map area)
# summary bars in the empty south-west corner (data coords -> figure coords)
def data_to_fig(x, y):
    return fig.transFigure.inverted().transform(ax.transData.transform((x, y)))
(x0, y0), (x1, y1) = data_to_fig(8.6, 5.5), data_to_fig(10.95, 4.05)
bx = fig.add_axes([x0, y0, x1 - x0, y1 - y0])
counts = (df.groupby(["configuration", "status"]).size().unstack(fill_value=0)
          .reindex(index=ORDER, columns=STATUS, fill_value=0))
left = np.zeros(len(ORDER))
for st, alpha in zip(STATUS, [1.0, 0.4, 0.0]):
    vals = counts[st].values
    for yi, f in enumerate(ORDER):
        col = FAM[f][1]
        bx.barh(yi, vals[yi], left=left[yi], height=0.66,
                color=col if alpha else "white", alpha=max(alpha, 1.0) if alpha in (0.0, 1.0) else 1.0,
                ec=col, lw=0.6) if alpha != 0.4 else bx.barh(
            yi, vals[yi], left=left[yi], height=0.66, color=col + "66", ec=col, lw=0.6)
    left += vals
for yi, f in enumerate(ORDER):
    bx.text(left[yi] + 0.8, yi, str(int(left[yi])), va="center", fontsize=6.5,
            color=C["grey"])
bx.set_yticks(range(len(ORDER)))
bx.set_yticklabels([FAM[f][0].split(" /")[0].replace("Other alternative concepts", "Other")
                    for f in ORDER], fontsize=6.5)
bx.invert_yaxis()
bx.set_xlim(0, 75)
bx.set_xticks([0, 25, 50, 75])
bx.tick_params(axis="x", labelsize=6.5, length=2)
bx.tick_params(axis="y", length=0)
bx.set_title(f"{len(df)} devices by design", fontsize=7.5, loc="left")
savefig(fig, "fusion_devices_world")
print(counts.sum(axis=1).to_dict(), len(df))
