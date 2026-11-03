"""Capital-related cost of electricity as a function of plant availability.

For a plant whose cost is dominated by capital and fixed O&M costs, the cost
per MWh scales as 1/A (Eq. in Section 2.5). Normalised to A = 0.75, a typical
assumption for commercial fusion plants. Marks the EU-DEMO minimum (30 %) and
the 75-85 % range assumed in commercial studies.
"""
import numpy as np
import matplotlib.pyplot as plt
from thesis_style import apply, C, FULL, savefig

apply()
A = np.linspace(0.15, 0.95, 400)
ref = 0.75
rel = ref / A

fig, ax = plt.subplots(figsize=(FULL, 0.5 * FULL))
ax.axvspan(0.75, 0.85, color=C["teal"], alpha=0.12, lw=0)
ax.text(0.80, 4.55, "commercial\nassumptions", ha="center", va="top",
        fontsize=7.5, color="#1F7A70")
ax.plot(A, rel, color=C["blue"])
for a, lab, dx, ha in [(0.30, "EU-DEMO minimum", 0.02, "left"),
                       (0.75, "", 0, "left")]:
    ax.plot(a, ref / a, "o", color=C["blue"], ms=4.5, zorder=3)
ax.text(0.325, 2.62, "EU-DEMO minimum (30 %):\n2.5 times the cost per MWh",
        fontsize=7.5, color=C["ink"], va="bottom")
ax.axhline(1, color=C["grey"], lw=0.6, ls=":")
ax.set_xlim(0.15, 0.95)
ax.set_ylim(0, 4.6)
ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ax.set_xticklabels([f"{int(t*100)}" for t in ax.get_xticks()])
ax.set_xlabel("Plant availability $A$ (%)")
ax.set_ylabel("Capital-related cost per MWh\n(relative to $A$ = 75 %)")
ax.grid(True, axis="y")
savefig(fig, "lcoe_availability")
print("done")
