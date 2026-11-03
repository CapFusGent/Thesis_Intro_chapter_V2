"""Private fusion industry: companies founded per year and cumulative count.

Data: figures_src/data/fia2024_companies.csv, read from the FIA Global Fusion
Industry Report 2024 (Figs. 4 and 15). Companies with more than US$200M of
investment are highlighted.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from thesis_style import apply, C, FULL, savefig

apply()
df = pd.read_csv(Path(__file__).parent / "data" / "fia2024_companies.csv",
                 comment="#")
years = np.arange(1990, 2025)
per_year = df.groupby("founded").size().reindex(years, fill_value=0)
big = df[df.invest_200m == 1].groupby("founded").size().reindex(years,
                                                                 fill_value=0)
cum = per_year.cumsum()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FULL, 0.78 * FULL), sharex=True,
                               gridspec_kw=dict(height_ratios=[1, 1.25],
                                                hspace=0.2))
# (a) cumulative number of companies
ax1.step(years, cum, where="post", color=C["blue"])
ax1.fill_between(years, cum, step="post", color=C["blue"], alpha=0.08, lw=0)
ax1.set_ylabel("Companies\n(cumulative)")
ax1.set_ylim(0, 50)
ax1.grid(True, axis="y")
ax1.text(2023.6, cum.iloc[-1] + 1.5, f"{cum.iloc[-1]}", ha="right", fontsize=8,
         color=C["blue"])
ax1.text(0.0, 1.04, "(a)", transform=ax1.transAxes, fontsize=9)

# (b) founded per year, highlighting >US$200M companies
ax2.bar(years, per_year - big, bottom=big, width=0.75, color="#B3C6E4",
        label="other companies")
ax2.bar(years, big, width=0.75, color=C["blue"],
        label=r"raised $>$US\$200M")
ax2.set_ylabel("Companies\nfounded")
ax2.set_ylim(0, 11)
ax2.set_yticks([0, 2, 4, 6, 8])
ax2.text(0.0, 1.02, "(b)", transform=ax2.transAxes, fontsize=9)
ax2.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92))
ax2.set_xlim(1990.5, 2024.5)
ax2.set_xticks(range(1992, 2025, 4))
ax2.tick_params(axis="x", length=2)

# name the highlighted companies, vertically above their year's bar
short = {"Commonwealth Fusion Systems": "CFS", "TAE Technologies": "TAE",
         "SHINE Technologies": "SHINE"}
for y, grp in df[df.invest_200m == 1].groupby("founded"):
    names = ", ".join(short.get(c, c) for c in grp.company)
    ax2.text(y, per_year[y] + 0.25, names, rotation=90, ha="center",
             va="bottom", fontsize=7, color=C["blue"])
savefig(fig, "fusion_private_industry")
print(per_year[per_year > 0].to_dict(), "total", cum.iloc[-1])
