"""
Matplotlib style for the A0 poster figures.

Every poster figure is built at the size it will be printed, so a 22 pt label on screen is a 22 pt
label on the poster. Never scale a saved figure in the poster layout: change its size in the script
and re-run instead.

    from poster_style import apply_poster_style, new_poster_fig, save_poster_fig

    apply_poster_style()
    fig, ax = new_poster_fig(10, 7)
    ...
    save_poster_fig(fig, "my_figure", 10, 7)    # figures/poster/my_figure.svg and .png (300 dpi)

Place the files at 100 % in the poster: the SVG says its size in points, the PNG carries 300 dpi in
its header, so both import at width_in x height_in.
"""
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.text import Text
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures" / "poster"

# Data colours, at most four. WATER and TUE_RED are the two series colours: checked with a
# colour-blind simulation (protan, deutan, tritan) they stay clearly apart on white, and both have
# at least 3:1 contrast with it. GREY is for context (reference lines, earlier periods).
# WATER_LIGHT is too pale to carry a series on its own: use it only as a fill behind a WATER line.
WATER = "#1c5cab"        # water and flooding
TUE_RED = "#c81919"      # accent: the one thing the reader should notice. TU/e red, RGB 200 25 25:
                         # check against the official poster template and change it here if needed
GREY = "#767676"
WATER_LIGHT = "#8fb8e8"
PALETTE = {"water": WATER, "accent": TUE_RED, "context": GREY, "water_light": WATER_LIGHT}

# Neutrals for text and axes (not data colours)
INK = "#1a1a1a"
INK_SOFT = "#4d4d4d"
GRID = "#dcdcdc"
WHITE = "#ffffff"

MIN_FONT_PT = 20   # poster rule: no text smaller than this
PNG_DPI = 300


def apply_poster_style() -> None:
    """Set matplotlib rcParams for poster figures. Sizes are points at the printed size."""
    mpl.rcParams.update({
        # text
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 22,
        "text.color": INK,
        "axes.titlesize": 28, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 20,
        "figure.titlesize": 28, "figure.titleweight": "bold",
        "axes.labelsize": 24, "axes.labelcolor": INK, "axes.labelpad": 12,
        "xtick.labelsize": 22, "ytick.labelsize": 22,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "legend.fontsize": 22, "legend.title_fontsize": 22, "legend.frameon": False,
        # data marks
        "axes.prop_cycle": cycler(color=[WATER, TUE_RED, GREY]),
        "lines.linewidth": 3.5, "lines.markersize": 11, "lines.solid_capstyle": "round",
        "patch.linewidth": 0,
        # axes: no top/right spines, light horizontal grid behind the data
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK_SOFT, "axes.linewidth": 1.5,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
        "xtick.major.width": 1.5, "ytick.major.width": 1.5, "xtick.major.size": 8, "ytick.major.size": 8,
        "xtick.major.pad": 8, "ytick.major.pad": 8,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 1.2,
        # white background everywhere
        "figure.facecolor": WHITE, "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE, "savefig.edgecolor": WHITE, "savefig.transparent": False,
        # layout and output. A 'tight' bbox would crop or grow the canvas and change the printed size,
        # so the layout engine fits everything inside the figure instead.
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.12, "figure.constrained_layout.w_pad": 0.12,
        "savefig.bbox": "standard",
        "savefig.dpi": PNG_DPI,
        "svg.fonttype": "none",   # text stays text in the SVG, so it can be edited in the poster
        "pdf.fonttype": 42,
    })


def new_poster_fig(width_in: float, height_in: float, **subplots_kw):
    """plt.subplots at the final printed size. Use this so the layout is designed at that size."""
    return plt.subplots(figsize=(width_in, height_in), **subplots_kw)


def text_below(fig, min_pt: float = MIN_FONT_PT) -> list[tuple[str, float]]:
    """Visible, non-empty text in `fig` smaller than `min_pt`, as (text, size) pairs."""
    fig.canvas.draw()   # tick labels only exist after a draw
    return [(t.get_text(), t.get_fontsize()) for t in fig.findobj(Text)
            if t.get_visible() and t.get_text().strip() and t.get_fontsize() < min_pt]


def save_poster_fig(fig, name: str, width_in: float, height_in: float, out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    """
    Save `fig` at exactly width_in x height_in inches to out_dir/<name>.svg and out_dir/<name>.png (300 dpi).

    The figure should already be that size (see new_poster_fig). If it isn't, it is resized here: text
    keeps its point size and the layout reflows, so check the result for overlaps.
    """
    w, h = fig.get_size_inches()
    if abs(w - width_in) > 1e-6 or abs(h - height_in) > 1e-6:
        warnings.warn(f"{name}: built at {w:g} x {h:g} in, saved at {width_in:g} x {height_in:g} in. "
                      "Build figures at their final size so the layout is checked at that size.")
        fig.set_size_inches(width_in, height_in)

    small = text_below(fig)
    if small:
        warnings.warn(f"{name}: text below {MIN_FONT_PT} pt: {small}")

    out_dir.mkdir(parents=True, exist_ok=True)
    svg, png = out_dir / f"{name}.svg", out_dir / f"{name}.png"
    fig.savefig(svg, format="svg", metadata={"Date": None})   # no timestamp, so unchanged figures give unchanged files
    fig.savefig(png, format="png", dpi=PNG_DPI)

    expected = (round(width_in * PNG_DPI), round(height_in * PNG_DPI))
    with Image.open(png) as im:
        if im.size != expected:
            raise RuntimeError(f"{png.name} is {im.size} px, expected {expected} px")
    return svg, png
