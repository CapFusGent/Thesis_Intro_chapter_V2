"""Shared figure style for the thesis.

Every figure script imports this module, so fonts, colours, line weights and
sizes stay identical across chapters. Figures are saved as vector PDF and
included in LaTeX at their natural width (no rescaling), so the 9-10 pt figure
text matches the 10 pt body text of the book class.

Usage:
    from thesis_style import apply, C, FULL, savefig
    apply()
    fig, ax = plt.subplots(figsize=(FULL, 0.5 * FULL))
    ...
    savefig(fig, "name")          # writes ../Figures/name.pdf
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- Font: Latin Modern Roman, the same family as the LaTeX body text --------
_LM = Path("/usr/share/texmf/fonts/opentype/public/lm")
for f in ["lmroman10-regular.otf", "lmroman10-italic.otf",
          "lmroman10-bold.otf", "lmroman10-bolditalic.otf"]:
    p = _LM / f
    if p.exists():
        font_manager.fontManager.addfont(str(p))

# --- Sizes (inches). Default book class text block is about 4.8 in wide. ----
FULL = 4.8          # \includegraphics{...} at full text width
HALF = 2.35         # side-by-side panels

# --- Colours: UGent blue first, then a validated categorical order ----------
# Validated with the dataviz palette checker (CVD and normal-vision separation
# pass). Use slots in this order; never cycle. Orange is light: always label it.
C = {
    "blue":   "#1E4C9A",   # UGent blue, primary series
    "orange": "#E08A00",
    "teal":   "#2A9D8F",
    "red":    "#C23B4E",
    "purple": "#7B5EA7",
    "olive":  "#8A9A2B",
    # neutrals
    "ink":    "#1A1A1A",   # text, axes
    "grey":   "#6B6B6B",   # secondary text, annotations
    "light":  "#BDBDBD",   # reference lines, uncertainty bands
    "grid":   "#E6E6E6",
}
SERIES = [C[k] for k in ["blue", "orange", "teal", "red", "purple", "olive"]]

FIG_DIR = Path(__file__).resolve().parent.parent / "Figures"


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "LM Roman 10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": C["ink"],
        "axes.labelcolor": C["ink"],
        "text.color": C["ink"],
        "xtick.color": C["ink"],
        "ytick.color": C["ink"],
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": C["grid"],
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def savefig(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"{name}.pdf"
    fig.savefig(out)
    fig.savefig(Path("/tmp") / f"{name}.png", dpi=200)   # preview only
    plt.close(fig)
    return out
