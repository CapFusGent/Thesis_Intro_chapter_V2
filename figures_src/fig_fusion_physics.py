"""Chapter 1 physics figures: cross sections, Coulomb barrier, binding energy.

All three are computed, not traced from a source, so they can be restyled
freely. Sources for the numbers are given in the comments and should be cited
in the captions.
"""
import numpy as np
import matplotlib.pyplot as plt
from thesis_style import apply, C, FULL, savefig

apply()

# ---------------------------------------------------------------------------
# 1. Fusion cross sections
# Parametrisation from the NRL Plasma Formulary (Huba), "Fusion" section:
#   sigma(E) = (A5 + A2 / ((A4 - A3*E)**2 + 1)) / (E * (exp(A1/sqrt(E)) - 1))
# sigma in barn, E = laboratory energy of the incident deuteron in keV.
# Converted here to centre-of-mass energy and m^2 (1 b = 1e-28 m^2).
# ---------------------------------------------------------------------------
NRL = {
    # name: (A1, A2, A3, A4, A5, m_target/(m_d+m_target))
    "DT":   (45.95, 50200.0, 1.368e-2, 1.076, 409.0, 3 / 5),
    "DDa":  (46.097, 372.0, 4.36e-4, 1.220, 0.0, 1 / 2),   # D(d,n)3He
    "DDb":  (47.88, 482.0, 3.08e-4, 1.177, 0.0, 1 / 2),    # D(d,p)T
    "DHe3": (89.27, 25900.0, 3.98e-3, 1.297, 647.0, 3 / 5),
}


def sigma_cm(key, e_cm):
    a1, a2, a3, a4, a5, f = NRL[key]
    e_lab = e_cm / f
    s_barn = (a5 + a2 / ((a4 - a3 * e_lab) ** 2 + 1)) / (
        e_lab * (np.exp(a1 / np.sqrt(e_lab)) - 1))
    return s_barn * 1e-28


E = np.logspace(0, 3, 600)            # 1 keV ... 1 MeV, centre of mass
curves = [
    (r"D–T", sigma_cm("DT", E), C["blue"]),
    (r"D–$^{3}$He", sigma_cm("DHe3", E), C["orange"]),
    (r"D–D (both branches)", sigma_cm("DDa", E) + sigma_cm("DDb", E), C["teal"]),
]

fig, ax = plt.subplots(figsize=(FULL, 0.62 * FULL))
for name, s, col in curves:
    ax.loglog(E, s, color=col)
ax.set_xlim(1, 1000)
ax.set_ylim(1e-32, 1e-27)
ax.set_xlabel("Centre-of-mass energy (keV)")
ax.set_ylabel(r"Cross section $\sigma$ (m$^2$)")
# Direct labels instead of a legend box
ax.text(14, 1.6e-28, "D–T", color=C["blue"], fontsize=9)
ax.text(300, 2.2e-28, r"D–$^{3}$He", color=C["orange"], fontsize=9)
ax.text(330, 2.5e-30, "D–D", color=C["teal"], fontsize=9)
# Secondary axis: temperature equivalent, T = E / k_B
k_B = 8.617e-8   # keV per K
sec = ax.secondary_xaxis("top", functions=(lambda e: e / k_B / 1e6,
                                           lambda t: t * 1e6 * k_B))
sec.set_xlabel(r"Equivalent temperature ($10^{6}$ K)")
sec.spines["top"].set_visible(True)
ax.grid(True, which="major", axis="y")
savefig(fig, "cross_section_fusion")

# ---------------------------------------------------------------------------
# 2. Coulomb barrier (schematic, arbitrary units)
# Square nuclear well of depth V0 inside radius R, Coulomb 1/r outside.
# ---------------------------------------------------------------------------
R, V0, k = 1.0, 1.3, 1.0
r = np.linspace(0.02, 6, 1000)
coul = k / r
total = np.where(r < R, -V0, coul)

fig, ax = plt.subplots(figsize=(FULL, 0.55 * FULL))
ax.plot(r, coul, color=C["grey"], lw=0.9, ls="--")
ax.plot(r[r >= R], total[r >= R], color=C["blue"])
ax.plot([0, R, R], [-V0, -V0, k / R], color=C["blue"])
ax.axhline(0, color=C["ink"], lw=0.6)
E_p = 0.30                                        # particle energy
r_turn = k / E_p
ax.plot([r_turn, 6], [E_p, E_p], color=C["red"], lw=1.2)
ax.plot([R, r_turn], [E_p, E_p], color=C["red"], lw=1.0, ls=":")
ax.annotate("", xy=(R + 0.03, 1.0), xytext=(R + 0.6, 1.3),
            arrowprops=dict(arrowstyle="->", color=C["ink"], lw=0.6))
ax.text(R + 0.65, 1.3, r"Coulomb barrier $\propto Z_1 Z_2 / R$", va="center")
ax.text(3.95, E_p + 0.09, "kinetic energy of the incoming nucleus",
        color=C["red"], fontsize=8)
ax.text((R + r_turn) / 2, E_p - 0.17, "tunnelling", color=C["red"],
        fontsize=8, ha="center")
ax.text(2.0, 0.68, "Coulomb repulsion", color=C["grey"], fontsize=8)
ax.text(0.5, -V0 + 0.15, "strong-force\nattraction", fontsize=8, ha="center")
ax.set_xlim(0, 6)
ax.set_ylim(-1.6, 1.75)
ax.set_xticks([R])
ax.set_xticklabels([r"$R$"])
ax.set_yticks([0])
ax.set_xlabel("Distance between nuclei $r$")
ax.set_ylabel("Potential energy $V(r)$")
savefig(fig, "coulomb_barrier")

# ---------------------------------------------------------------------------
# 3. Binding energy per nucleon
# Curve: semi-empirical mass formula along the valley of stability
# (coefficients in MeV: aV 15.75, aS 17.8, aC 0.711, aA 23.7, aP 11.18).
# Points: measured values (AME2020) for selected nuclides.
# ---------------------------------------------------------------------------
def semf_b_per_a(A):
    Z = A / (2 + 0.0155 * A ** (2 / 3))       # most stable Z for given A
    aV, aS, aC, aA = 15.75, 17.8, 0.711, 23.7
    B = aV * A - aS * A ** (2 / 3) - aC * Z * (Z - 1) / A ** (1 / 3) \
        - aA * (A - 2 * Z) ** 2 / A
    return B / A

A = np.linspace(12, 245, 500)
meas = {  # nuclide: (A, B/A in MeV)
    r"$^{2}$H": (2, 1.112), r"$^{3}$H": (3, 2.827), r"$^{3}$He": (3, 2.573),
    r"$^{4}$He": (4, 7.074), r"$^{6}$Li": (6, 5.332), r"$^{7}$Li": (7, 5.606),
    r"$^{12}$C": (12, 7.680), r"$^{16}$O": (16, 7.976),
    r"$^{56}$Fe": (56, 8.790), r"$^{235}$U": (235, 7.591),
    r"$^{238}$U": (238, 7.570),
}
offs = {r"$^{2}$H": (6, -2), r"$^{3}$H": (6, 2), r"$^{3}$He": (6, -9),
        r"$^{4}$He": (6, -3), r"$^{6}$Li": (6, -6), r"$^{7}$Li": (6, 0),
        r"$^{12}$C": (-17, 1), r"$^{16}$O": (6, -1), r"$^{56}$Fe": (-6, 6),
        r"$^{235}$U": (-22, 6), r"$^{238}$U": (2, -11)}

fig, ax = plt.subplots(figsize=(FULL, 0.6 * FULL))
ax.axvspan(0, 56, color=C["blue"], alpha=0.06, lw=0)
ax.axvspan(56, 250, color=C["orange"], alpha=0.06, lw=0)
ax.plot(A, semf_b_per_a(A), color=C["grey"], lw=1.0,
        label="Semi-empirical mass formula")
for name, (a, b) in meas.items():
    ax.plot(a, b, "o", color=C["blue"], ms=4, zorder=3)
    ax.annotate(name, (a, b), xytext=offs[name], textcoords="offset points",
                fontsize=8)
ax.text(18, 1.0, "fusion releases energy\n(binding increases)", fontsize=8,
        color=C["blue"])
ax.text(120, 1.0, "fission releases energy\n(binding increases)", fontsize=8,
        color="#9A5F00")
ax.set_xlim(0, 250)
ax.set_ylim(0, 9.5)
ax.set_xlabel("Mass number $A$")
ax.set_ylabel("Binding energy per nucleon (MeV)")
ax.legend(loc="lower right", bbox_to_anchor=(1, 0.18))
savefig(fig, "nuclear_binding")
print("done")
