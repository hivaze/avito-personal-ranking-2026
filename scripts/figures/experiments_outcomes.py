"""experiments-outcomes: number of catalogued ideas per theme, split by what came of them."""

from __future__ import annotations

import pandas as pd
from matplotlib.patches import Rectangle

import style

NAME = "experiments-outcomes"
SOURCE = "Data: docs/figures/data/experiments_outcomes.csv, counted from docs/07-what-failed.md by count_catalogue.py."

# column, legend text, fill, label colour inside the fill
OUTCOMES = [
    ("adopted", "adopted", style.BLUE, "#ffffff"),
    ("helped_then_superseded", "helped, then superseded", style.BLUE_250, style.INK),
    ("no_effect", "no effect", style.MUTED, "#ffffff"),
    ("hurt", "hurt", style.RED, "#ffffff"),
    ("inconclusive", "inconclusive", style.YELLOW, style.INK),
    ("not_run", "not run", style.AXIS, style.INK),
]
BAR = 0.56  # bar thickness as a share of the row
GAP_PT = 1.5  # surface gap between segments, in points


def load() -> pd.DataFrame:
    d = pd.read_csv(style.DATA_DIR / "experiments_outcomes.csv")
    cols = [c for c, *_ in OUTCOMES]
    assert (d[cols].sum(axis=1) == d.total).all(), "a theme's outcomes do not add up to its total"
    return d.sort_values(["total", "adopted"], ascending=[False, False]).reset_index(drop=True)


def texts() -> style.Texts:
    d = load()
    cols = [c for c, *_ in OUTCOMES]
    n, sums = int(d.total.sum()), d[cols].sum()
    top = [f"{r.label} {r.total}" for r in d.head(5).itertuples()]
    top[0] += " ideas"
    last = d.iloc[-1]
    no_effect_first = sum(all(r.no_effect > r[c] for c in cols if c != "no_effect") for _, r in d.iterrows())
    most_adopted, most_hurt = d.loc[d.adopted.idxmax()], d.loc[d.hurt.idxmax()]
    return style.Texts(
        title=f"Of {n} catalogued ideas, {sums.adopted} were adopted; {sums.no_effect} had no measurable effect",
        caption=(f"The {n} entries of the experiment catalogue by theme and outcome. Several entries bundle many "
                 "arms (13 block designs, 63 tricks, 95 rule families), so the number of distinct things tried is "
                 "larger."),
        alt=(f"Horizontal stacked bars for {len(d)} themes, longest first: {'; '.join(top)}; down to {last.label} "
             f"{last.total}. Each bar is split into adopted, helped then superseded, no effect, hurt, inconclusive "
             f"and not run. No effect is the largest part of {no_effect_first} of the {len(d)} bars; "
             f"{most_adopted.label} has the most adopted ideas, {most_adopted.adopted}; {most_hurt.label} has the "
             f"most that hurt, {most_hurt.hurt}."),
        source=SOURCE,
    )


def make() -> None:
    d, t = load(), texts()
    cols = [c for c, *_ in OUTCOMES]
    # light blue and light grey are too close to sit side by side; no row puts them there
    assert not ((d.helped_then_superseded > 0) & (d.no_effect + d.hurt + d.inconclusive == 0) & (d.not_run > 0)).any()

    fig = style.figure(7.45)
    head = style.header(fig, t.title, "Ideas per theme, split by outcome. Numbers inside the bars are counts; "
                                      "a segment without a number is one idea.")
    left, right, bottom = 3.5, 0.55, 0.98
    ax = style.axes(fig, left, bottom, right, head + 0.72)
    n = len(d)
    x_max = int(d.total.max()) + 1  # the total of the longest bar is printed just past it
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(0, x_max)
    ax.set_yticks(range(n))
    ax.set_yticklabels(d.label, fontsize=style.FS_SMALL, color=style.INK)
    ax.tick_params(axis="y", pad=8)
    ax.set_xticks(range(0, x_max, 5))
    ax.grid(axis="x")
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color(style.AXIS)
    ax.set_xlabel("Number of ideas")

    plot_w_pt = (fig.get_size_inches()[0] - left - right) * 72
    gap = GAP_PT * x_max / plot_w_pt  # the surface gap in data units
    for i, r in d.iterrows():
        x = 0.0
        for col, _, fill, ink in OUTCOMES:
            v = int(r[col])
            if v == 0:
                continue
            ax.add_patch(Rectangle((x + gap / 2, i - BAR / 2), v - gap, BAR, facecolor=fill, edgecolor="none",
                                   zorder=3))
            if v >= 2:
                ax.text(x + v / 2, i, str(v), fontsize=style.FS_SMALL, color=ink, ha="center", va="center",
                        zorder=4)
            x += v
        ax.text(x + 0.45, i, str(int(r.total)), fontsize=style.FS_SMALL, fontweight="bold", color=style.INK,
                ha="left", va="center")

    # legend: two rows of three, in stack order, with the totals
    w, h = fig.get_size_inches()
    totals = d[cols].sum()
    for k, (col, text, fill, _) in enumerate(OUTCOMES):
        x = (style.MARGIN + (0.0, 2.75, 5.5)[k % 3]) / w
        y = 1 - (head + 0.16 + 0.26 * (k // 3)) / h
        fig.patches.append(Rectangle((x, y - 0.065 / h), 0.26 / w, 0.13 / h, transform=fig.transFigure,
                                     facecolor=fill, edgecolor="none"))
        fig.text(x + 0.36 / w, y, f"{text}: {int(totals[col])}", fontsize=style.FS_SMALL, color=style.INK2,
                 va="center", ha="left")

    style.save(fig, NAME, t)


if __name__ == "__main__":
    make()
