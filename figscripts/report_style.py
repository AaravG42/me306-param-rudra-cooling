"""House style for the ME 306 report figures.

Follows the figures4papers conventions (github.com/ChenLiu-1996/figures4papers, scientific-figure-making):
minimal spines (top/right off), frameless legends, semantic blue-green-red-neutral palette, explicit limits,
vector export. Sizes are scaled down to the report's 15.2 cm text width, and the font is Inter so that figure
text matches the report's heading face.

Colour roles (used the same way in every figure):
  OPT   blue_main    optimised / recommended / bypass recirculation
  BASE  red_strong   C-DAC design set-points / traditional enclosed aisle / baseline
  AIR   blue_second  anything moved by air (air heat, fans)
  LIQ   teal         anything moved by liquid (cold plates, TCS)
  REFR  violet       refrigeration (chillers)
  REJ   green_3      heat rejection (tower fans)
  PUMP  neutral dark pumps
  GREY  neutral      context / reference
"""
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "neutral_dark": "#767676", "ink": "#272727",
    "highlight": "#FFD700", "teal": "#42949E", "violet": "#9A4D8E",
}
OPT, BASE = PALETTE["blue_main"], PALETTE["red_strong"]
AIR, LIQ, REFR, REJ, PUMP, GREY = (PALETTE["blue_secondary"], PALETTE["teal"], PALETTE["violet"],
                                   PALETTE["green_3"], PALETTE["neutral_dark"], PALETTE["neutral"])

TEXTWIDTH_IN = 6.0          # 15.2 cm
FULL = (TEXTWIDTH_IN, 2.45)
FULL_TALL = (TEXTWIDTH_IN, 2.9)
HALF = (TEXTWIDTH_IN * 0.52, 2.45)

@dataclass(frozen=True)
class FigureStyle:
    font_size: float = 8.5
    axes_linewidth: float = 0.9
    font_family: tuple = ("STIXGeneral", "DejaVu Serif")

def apply_publication_style(style: FigureStyle = FigureStyle()):
    plt.rcParams.update({
        "font.family": list(style.font_family), "mathtext.fontset": "stix", "font.size": style.font_size,
        "axes.labelsize": style.font_size, "axes.titlesize": style.font_size,
        "xtick.labelsize": style.font_size - 0.5, "ytick.labelsize": style.font_size - 0.5,
        "legend.fontsize": style.font_size - 0.8, "legend.frameon": False,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": style.axes_linewidth, "xtick.major.width": style.axes_linewidth,
        "ytick.major.width": style.axes_linewidth, "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.edgecolor": PALETTE["ink"], "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"], "ytick.color": PALETTE["ink"], "text.color": PALETTE["ink"],
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 6,
        "lines.linewidth": 1.6, "lines.markersize": 4,
        "axes.grid": False, "axes.axisbelow": True,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "figure.dpi": 150, "savefig.dpi": 300,
    })

def panel_label(ax, letter, x=-0.14, y=1.06):
    """(a), (b) ... in the panel's top-left, same everywhere."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=plt.rcParams["font.size"] + 0.5,
            fontweight="bold", va="bottom", ha="left")

def finalize_figure(fig, out_dir, name, pad=0.6):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=pad)
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight", pad_inches=0.02)
        paths.append(p)
    plt.close(fig)
    return paths
