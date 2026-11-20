"""Shared look of the figures in docs/figures: colours, fonts, the figure frame, saving, diagram boxes.

Every figure is 9 inches wide (864 CSS px), drawn on a light surface with a thin border so that it reads the
same on a white and on a dark page. No text is smaller than 10 pt, which stays above 9 pt when the page scales
the image down to 800 px. Text is written to the SVG as outlines, so the layout does not depend on the fonts of
the machine that shows it.

Every figure module has the same three members: ``NAME`` (the file name), ``texts()`` (title, caption, alt text
and source note, with the numbers read from docs/figures/data where they come from data) and ``make()`` (draws
and saves). Output is byte-for-byte reproducible for a given matplotlib version: the SVG carries no date and a
fixed hash salt, the PNG no time stamp.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = FIG_DIR / "data"


@dataclass(frozen=True)
class Texts:
    """The words of one figure. ``title`` may hold a line break where the figure breaks it."""

    title: str
    caption: str
    alt: str
    source: str


def read_json(name: str) -> Any:
    """Read docs/figures/data/<name>."""
    return json.loads((DATA_DIR / name).read_text())


def count(n: int) -> str:
    """A count as figure text: words up to ten, digits above."""
    words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
    return words[n] if 0 <= n <= 10 else f"{n:,}"


def signed(x: float, digits: int = 5) -> str:
    """A difference with its sign, for figure text: +0.00032, −0.00017 (true minus sign)."""
    return f"{x:+.{digits}f}".replace("-", "−")

WIDTH = 9.0  # inches
MARGIN = 0.30  # inches between the border and the content
PNG_DPI = 192  # twice the 96 dpi of a CSS pixel
MIN_FONT = 10.0
EDGE = 0.1  # inches: no text closer than this to the left or right border

# Surfaces and ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"

# Hues. Blue and orange are the only two identity colours; the blue ramp is used where order matters.
# Checked on this surface with a colour-vision-deficiency validator (OKLab distance x 100 under protanopia and
# deuteranopia): blue against orange 24.7 (target 8 or more); the four ramp steps are monotone in lightness with
# gaps of 0.06 or more; every pair of outcome fills that can touch in experiments-outcomes is 15 or more apart for
# normal vision and 8 or more under both deficiencies. Yellow and the light blue are below 3:1 against the
# surface, so they never carry a value without a printed label.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
YELLOW = "#eda100"
BLUE_250 = "#86b6ef"
BLUE_400 = "#3987e5"
BLUE_550 = "#1c5cab"
BLUE_700 = "#0d366b"
BLUE_TINT = "#cde2fb"
ORANGE_TINT = "#fbdccf"

# Diagram boxes: fill and outline per meaning.
BOX = {
    "data": (NEUTRAL, MUTED),
    "blue": (BLUE_TINT, BLUE),
    "orange": (ORANGE_TINT, ORANGE),
    "plain": (SURFACE, AXIS),
}

FS_TITLE = 13.5
FS_TEXT = 10.5
FS_SMALL = 10.0
LINE = 0.19  # inches between text lines at 10 to 10.5 pt


def apply() -> None:
    """Set the matplotlib defaults shared by all figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FS_TEXT,
            "text.color": INK,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.labelsize": FS_TEXT,
            "axes.linewidth": 0.8,
            "axes.facecolor": SURFACE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.labelsize": FS_SMALL,
            "ytick.labelsize": FS_SMALL,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "svg.fonttype": "path",
            "svg.hashsalt": "avito-ranker-figures",
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def figure(height: float) -> Figure:
    """A 9-inch-wide figure of the given height (inches) on the light surface with a thin border."""
    apply()
    fig = plt.figure(figsize=(WIDTH, height))
    fig.patch.set_facecolor(SURFACE)
    fig.patch.set_edgecolor(AXIS)
    fig.patch.set_linewidth(2.0)  # half of it falls outside the canvas
    return fig


def header(fig: Figure, title: str, subtitle: str | None = None) -> float:
    """Write the title (and subtitle) at the top left. Returns the height in inches that the header takes."""
    w, h = fig.get_size_inches()
    y = 0.24
    fig.text(MARGIN / w, 1 - y / h, title, fontsize=FS_TITLE, fontweight="bold", color=INK, va="top", ha="left",
             linespacing=1.25)
    y += (title.count("\n") + 1) * FS_TITLE * 1.25 / 72 + 0.07
    if subtitle:
        fig.text(MARGIN / w, 1 - y / h, subtitle, fontsize=FS_TEXT, color=INK2, va="top", ha="left",
                 linespacing=1.3)
        y += (subtitle.count("\n") + 1) * FS_TEXT * 1.3 / 72 + 0.05
    return y


def source(fig: Figure, text: str) -> None:
    """Write the one-line source note at the bottom left."""
    w, h = fig.get_size_inches()
    fig.text(MARGIN / w, 0.15 / h, text, fontsize=FS_SMALL, color=INK2, va="bottom", ha="left")


def axes(fig: Figure, left: float, bottom: float, right: float, top: float) -> Axes:
    """Add axes whose plot area leaves the given margins (inches) to the figure edges."""
    w, h = fig.get_size_inches()
    return fig.add_axes((left / w, bottom / h, (w - left - right) / w, (h - bottom - top) / h))


def canvas(fig: Figure) -> Axes:
    """Axes that cover the whole figure, in inches from the bottom left. For box-and-arrow diagrams."""
    w, h = fig.get_size_inches()
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.patch.set_alpha(0)
    return ax


def hgrid(ax: Axes) -> None:
    """Hairline horizontal grid, no left spine."""
    ax.grid(axis="y")
    ax.spines["left"].set_visible(False)


def box(ax: Axes, x: float, y: float, w: float, h: float, title: str | None = None, lines: tuple[str, ...] = (),
        kind: str = "blue", align: str = "center", dashed: bool = False, pill: bool = False) -> None:
    """A rounded box with a bold first line and regular lines below. (x, y) is the lower left corner, in inches."""
    fill, edge = BOX[kind]
    r = min(h / 2, 0.16) if pill else 0.05
    ax.add_patch(
        FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r},rounding_size={r}",
                       facecolor=fill, edgecolor=edge, linewidth=1.2, linestyle=(0, (4, 3)) if dashed else "-",
                       zorder=2))
    n = (1 if title else 0) + len(lines)
    if n == 0:
        return
    top = y + h / 2 + (n - 1) * LINE / 2
    tx = x + w / 2 if align == "center" else x + 0.11
    rows = ([(title, "bold", FS_TEXT)] if title else []) + [(s, "normal", FS_SMALL) for s in lines]
    renderer = ax.figure.canvas.get_renderer()
    for i, (s, weight, size) in enumerate(rows):
        t = ax.text(tx, top - i * LINE, s, fontsize=size, fontweight=weight, color=INK, ha=align, va="center",
                    zorder=3)
        width = t.get_window_extent(renderer).width / ax.figure.dpi
        assert width <= w - 0.12, f"text wider than its box by {width - w + 0.12:.2f} in: {s!r}"


def arrow(ax: Axes, pts: list[tuple[float, float]], head: bool = True, color: str = INK2, lw: float = 1.2) -> None:
    """A polyline through pts (inches) that ends in an arrow head."""
    xs, ys = zip(*pts)
    if len(pts) > 2 or not head:
        end = len(pts) if not head else len(pts) - 1
        ax.plot(xs[:end], ys[:end], color=color, lw=lw, zorder=1)
    if head:
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=1,
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": lw, "shrinkA": 0, "shrinkB": 0,
                                "mutation_scale": 11})


def key(ax: Axes, x: float, y: float, items: list[tuple[str, str]], gap: float = 0.28) -> None:
    """A row of colour keys for a diagram: (box kind, meaning). (x, y) is the left end of the row."""
    renderer = ax.figure.canvas.get_renderer()
    for kind, label in items:
        fill, edge = BOX[kind]
        ax.add_patch(FancyBboxPatch((x + 0.03, y - 0.06), 0.2, 0.12, boxstyle="round,pad=0.03,rounding_size=0.03",
                                    facecolor=fill, edgecolor=edge, linewidth=1.2))
        t = ax.text(x + 0.36, y, label, fontsize=FS_SMALL, color=INK2, va="center", ha="left")
        width = t.get_window_extent(renderer).width / ax.figure.dpi
        x += 0.36 + width + gap


def save(fig: Figure, name: str, texts: Texts) -> None:
    """Write the source note, then docs/figures/<name>.svg and <name>.png (2x).

    Refuses text below the minimum size. The SVG carries the title and the alt text as metadata.
    """
    source(fig, texts.source)
    fig.set_dpi(PNG_DPI)  # measure text as the PNG renders it (hinting widens text at low resolution)
    fig.canvas.draw()  # places axis labels and tick labels, so their extents are final
    renderer = fig.canvas.get_renderer()
    w, h = fig.get_size_inches() * fig.dpi
    edge = EDGE * fig.dpi
    hidden = set()  # labels of ticks outside the view: matplotlib keeps them but does not draw them
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            eps = (hi - lo) * 1e-9
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - eps <= tick.get_loc() <= hi + eps:
                    hidden.update((id(tick.label1), id(tick.label2)))
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text().strip() and t.get_visible() and id(t) not in hidden:
            assert t.get_fontsize() >= MIN_FONT, (name, t.get_text(), t.get_fontsize())
            bb = t.get_window_extent(renderer)
            assert edge <= bb.x0 and bb.x1 <= w - edge and 0 <= bb.y0 and bb.y1 <= h, \
                f"{name}: text runs outside the frame: {t.get_text()[:60]!r}"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.svg", metadata={"Date": None, "Creator": "scripts/figures",
                                                   "Title": texts.title.replace("\n", " "),
                                                   "Description": texts.alt})
    fig.savefig(FIG_DIR / f"{name}.png", dpi=PNG_DPI, metadata={"Software": "scripts/figures"})
    plt.close(fig)
