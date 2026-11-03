"""Chapter 2 schematics: RUL definition, maintenance strategies, prognostic
spectrum, RUL model families, industrial revolutions timeline.

All are conceptual diagrams redrawn in the thesis style. Keep the source
citation in the caption where the concept is adapted from someone else
(e.g. MathWorks for the RUL model families, Sufian et al. for the timeline).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from thesis_style import apply, C, FULL, savefig

apply()


def health_axes(ax, xlabel="Time", ylabel="Health indicator"):
    """Bare schematic axes: arrows, no ticks, labels at the ends."""
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(1, 0, ">", color=C["ink"], transform=ax.transAxes, clip_on=False,
            ms=3.5)
    ax.plot(0, 1, "^", color=C["ink"], transform=ax.transAxes, clip_on=False,
            ms=3.5)
    ax.set_xlabel(xlabel, loc="right")
    ax.set_ylabel(ylabel, loc="top")


def degradation(t, t_onset=0.25, t_fail=1.0, h0=1.0, h_fail=0.2):
    """Smooth health curve: flat, then accelerating decline to h_fail."""
    x = np.clip((t - t_onset) / (t_fail - t_onset), 0, None)
    return h0 - (h0 - h_fail) * x ** 2


# ---------------------------------------------------------------------------
# 1. Health indicator, threshold, useful life and RUL
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(FULL, 0.5 * FULL))
t = np.linspace(0, 1.0, 400)
t1, tf = 0.62, 1.0
thr = 0.2
obs = t <= t1
ax.plot(t[obs], degradation(t[obs]), color=C["blue"])
# predicted future: median and band
tp = np.linspace(t1, 1.12, 200)
ax.plot(tp, degradation(tp), color=C["blue"], ls="--", lw=1.1)
ax.fill_between(tp, degradation(tp, t_fail=1.12), degradation(tp, t_fail=0.9),
                color=C["blue"], alpha=0.12, lw=0)
ax.axhline(thr, color=C["red"], lw=0.9)
ax.text(0.02, thr + 0.03, "failure threshold", color=C["red"], fontsize=8)
ax.plot(t1, degradation(t1), "o", color=C["blue"], ms=4.5, zorder=4)
ax.axvline(t1, color=C["grey"], lw=0.6, ls=":")
ax.axvline(tf, color=C["grey"], lw=0.6, ls=":")
ax.text(t1, 1.07, "$t_1$ (now)", ha="center", fontsize=8)
ax.text(tf, 1.07, "$t_{\\mathrm{failure}}$", ha="center", fontsize=8)
y_ul, y_rul = 0.07, 0.07
ax.annotate("", xy=(0, -0.04), xytext=(tf, -0.04),
            arrowprops=dict(arrowstyle="<->", color=C["ink"], lw=0.6))
ax.text(tf / 2, -0.01, "useful life", ha="center", fontsize=8)
ax.annotate("", xy=(t1, y_rul), xytext=(tf, y_rul),
            arrowprops=dict(arrowstyle="<->", color=C["blue"], lw=0.8))
ax.text((t1 + tf) / 2, y_rul + 0.03, "RUL$(t_1)$", ha="center", fontsize=8,
        color=C["blue"])
ax.text(0.05, 0.93, "observed", fontsize=8, color=C["blue"])
ax.text(0.86, 0.55, "predicted\n(with uncertainty)", fontsize=8,
        color=C["blue"])
ax.set_xlim(0, 1.15)
ax.set_ylim(-0.08, 1.12)
health_axes(ax)
savefig(fig, "RUL")

# ---------------------------------------------------------------------------
# 2. Reactive, preventive and predictive maintenance in one figure
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(FULL, 0.36 * FULL), sharey=True)
titles = ["(a) Reactive", "(b) Preventive", "(c) Predictive"]
t = np.linspace(0, 1, 300)
h = degradation(t, t_onset=0.2, t_fail=0.9, h_fail=0.15)
for ax, title in zip(axs, titles):
    ax.axhline(0.15, color=C["red"], lw=0.7)
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.1)
    health_axes(ax, xlabel="", ylabel="")

# (a) run to failure
m = t <= 0.9
axs[0].plot(t[m], h[m], color=C["blue"])
axs[0].plot(0.9, 0.15, "x", color=C["red"], ms=6, mew=1.5)
axs[0].text(0.62, 0.22, "failure,\nthen repair", fontsize=7.5, ha="right",
            color=C["red"])

# (b) fixed interval: intervene early, life left unused
tp = 0.55
m = t <= tp
axs[1].plot(t[m], h[m], color=C["blue"])
axs[1].plot(t[~m], h[~m], color=C["light"], ls="--", lw=1.0)
axs[1].plot([tp, tp], [h[m][-1], 1.0], color=C["teal"], lw=1.0)
axs[1].plot(tp, 1.0, "o", color=C["teal"], ms=3.5)
axs[1].annotate("", xy=(tp, 0.07), xytext=(0.9, 0.07),
                arrowprops=dict(arrowstyle="<->", color=C["grey"], lw=0.6))
axs[1].text(0.5, 0.2, "unused\nlife", ha="right", fontsize=7.5,
            color=C["grey"])
axs[1].text(tp - 0.03, 1.03, "scheduled", fontsize=7.5, ha="right",
            color=C["teal"])

# (c) condition-based forecast: intervene just before predicted failure
tn, tm = 0.6, 0.8
m = t <= tn
axs[2].plot(t[m], h[m], color=C["blue"])
mp = (t >= tn) & (t <= tm)
axs[2].plot(t[mp], h[mp], color=C["blue"], ls="--", lw=1.1)
axs[2].plot(t[t >= tm], h[t >= tm], color=C["light"], ls="--", lw=1.0)
axs[2].axvspan(0.74, 0.84, color=C["teal"], alpha=0.15, lw=0)
axs[2].plot([tm, tm], [h[t >= tm][0], 1.0], color=C["teal"], lw=1.0)
axs[2].plot(tm, 1.0, "o", color=C["teal"], ms=3.5)
axs[2].text(tm - 0.06, 1.03, "planned", fontsize=7.5, color=C["teal"],
            ha="right")
axs[2].plot(tn, h[m][-1], "o", color=C["blue"], ms=3.5)
axs[2].text(tn - 0.03, h[m][-1] - 0.12, "now", fontsize=7.5, ha="right")
axs[0].set_ylabel("Health", loc="top")
for ax in axs:
    ax.set_xlabel("Time", loc="right")
fig.subplots_adjust(wspace=0.15)
savefig(fig, "maintenance_strategies")

# ---------------------------------------------------------------------------
# 3. Maintenance maturity ladder (replaces the 3-D bar chart)
# Ordinal, not measured: bars show increasing information used for decisions.
# ---------------------------------------------------------------------------
labels = ["No\nmaintenance", "Reactive", "Preventive", "Predictive",
          "Proactive", "Self-\nmaintenance"]
ramp = ["#D6E0F0", "#B3C6E4", "#8CA8D4", "#5F86C1", "#3A66AB", "#1E4C9A"]
fig, ax = plt.subplots(figsize=(FULL, 0.42 * FULL))
x = np.arange(len(labels))
ax.bar(x, x + 1, color=ramp, width=0.78, edgecolor="white", linewidth=1.5)
ax.bar(x[3], 4, color=ramp[3], width=0.78, edgecolor=C["ink"], linewidth=0.9)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.tick_params(axis="x", length=0)
ax.set_ylabel("Information used for\nmaintenance decisions", fontsize=8.5)
ax.annotate("focus of\nthis thesis", xy=(3, 4.05), xytext=(2.0, 5.2),
            fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="->", color=C["ink"], lw=0.6))
ax.set_ylim(0, 6.6)
savefig(fig, "main_strategies")

# ---------------------------------------------------------------------------
# 4. Prognostic modelling spectrum
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(FULL, 0.33 * FULL))
ax.set_xlim(0, 1)
ax.set_ylim(-0.05, 1)
ax.axis("off")
grad = np.linspace(0, 1, 256).reshape(1, -1)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("spec", [C["blue"], "#EDEDED",
                                                  C["orange"]])
ax.imshow(grad, extent=(0.02, 0.98, 0.62, 0.74), aspect="auto", cmap=cmap)
ax.text(0.02, 0.80, "more physical knowledge", fontsize=8, color=C["blue"])
ax.text(0.98, 0.80, "more operational data", fontsize=8, color="#9A5F00",
        ha="right")
groups = [
    (0.16, "Model-based", "physical degradation\nmodels, Kalman and\nparticle filters"),
    (0.50, "Hybrid", "physics-informed neural\nnetworks, simulator +\nresidual learning"),
    (0.84, "Data-driven", "similarity, degradation\nand survival models,\nmachine learning"),
]
for xg, name, ex in groups:
    ax.plot([xg, xg], [0.56, 0.62], color=C["ink"], lw=0.6)
    ax.text(xg, 0.50, name, ha="center", va="top", fontsize=8.5,
            fontweight="bold")
    ax.text(xg, 0.32, ex, ha="center", va="top", fontsize=7.5,
            color=C["grey"], linespacing=1.3)
savefig(fig, "prognostic_spectrum")

# ---------------------------------------------------------------------------
# 5. RUL estimator families by data availability (concept after MathWorks)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(FULL, 0.46 * FULL))
ax.set_xlim(0, 1)
ax.set_ylim(0.25, 1)
ax.axis("off")


def box(x, y, text, fc="white", ec=C["ink"], w=0.2, h=0.09, bold=False,
        tc=None, fs=7.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.01",
                                fc=fc, ec=ec, lw=0.7))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc or C["ink"])


def link(x0, y0, x1, y1, ym=None):
    ym = (y0 + y1) / 2 if ym is None else ym
    ax.plot([x0, x0, x1, x1], [y0, ym, ym, y1], color=C["grey"], lw=0.6,
            zorder=0)


leaves = ["Hash\nsimilarity", "Pairwise\nsimilarity", "Residual\nsimilarity",
          "Linear\ndegradation", "Exponential\ndegradation",
          "Reliability\nsurvival", "Covariate\nsurvival"]
lx = np.linspace(0.075, 0.925, 7)
fam = [("Similarity models", [0, 1, 2], "run-to-failure histories\nof similar assets"),
       ("Degradation models", [3, 4], "condition indicator and\nknown failure threshold"),
       ("Survival models", [5, 6], "failure times, with or\nwithout covariates")]
box(0.5, 0.94, "What data are available?", bold=True, w=0.34, h=0.08)
for name, idx, cond in fam:
    xf = float(np.mean(lx[idx]))
    link(0.5, 0.90, xf, 0.66, ym=0.86)
    ax.text(xf, 0.765, cond, ha="center", va="center", fontsize=7,
            color=C["grey"], linespacing=1.25,
            bbox=dict(fc="white", ec="none", pad=0.8))
    box(xf, 0.62, name, fc="#E3EAF5", ec=C["blue"], w=0.245, h=0.08,
        bold=True, tc=C["blue"], fs=7.2)
    for i in idx:
        link(xf, 0.58, lx[i], 0.39, ym=0.50)
for x, lab in zip(lx, leaves):
    box(x, 0.33, lab, w=0.118, h=0.12, fs=6.8)
savefig(fig, "rul_families")
print("done")

# ---------------------------------------------------------------------------
# 6. Four industrial revolutions (concept after Sufian et al. 2021)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(FULL, 0.34 * FULL))
ax.set_xlim(0, 4.3)
ax.set_ylim(0, 0.72)
ax.axis("off")
ax.annotate("", xy=(4.3, 0.45), xytext=(0, 0.45),
            arrowprops=dict(arrowstyle="->", color=C["ink"], lw=0.7))
steps = [(1784, "Industry 1.0", "mechanisation,\nwater and steam power"),
         (1870, "Industry 2.0", "electrification,\nmass production"),
         (1969, "Industry 3.0", "electronics, IT,\nautomation"),
         (2011, "Industry 4.0", "cyber-physical systems,\nIoT, data analytics")]
ramp = ["#8CA8D4", "#5F86C1", "#3A66AB", "#1E4C9A"]
for i, ((yr, name, desc), col) in enumerate(zip(steps, ramp)):
    xp = 0.55 + i * 1.05
    ax.plot(xp, 0.45, "o", color=col, ms=6, zorder=3)
    ax.text(xp, 0.58, name, ha="center", fontsize=8.5, fontweight="bold",
            color=col)
    ax.text(xp, 0.34, f"c. {yr}", ha="center", va="top", fontsize=7.5)
    ax.text(xp, 0.20, desc, ha="center", va="top", fontsize=7.5,
            color=C["grey"], linespacing=1.25)
savefig(fig, "Timeline_industrial")
print("timeline done")
