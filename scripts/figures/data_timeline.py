"""data-timeline: the calendar of the splits and how old the popularity inputs are in each evaluation set."""

from __future__ import annotations

import pandas as pd
from matplotlib.patches import Rectangle

import style

NAME = "data-timeline"
TITLE = ("Test feeds are served 7 to 10 days after the last train day:\n"
         "popularity inputs are 9 to 11 days old there, 1.75 days on the holdout")
CAPTION = (
    "Calendar of the data in Moscow time, September 2026, and the mean age of the popularity inputs in each "
    "evaluation set: days since a candidate item was last seen in train, over items seen in train. Each line ends "
    "at the mean time of the set's feeds and is as long as that age."
)
ALT = (
    "Top: a calendar strip from 6 to 29 September 2026. Train covers 6 to 14 and 17 to 19 September, validation "
    "Tuesday 22 and Thursday 24, test 26 to 29; the public part of test ends on Sunday 27 at 11:32 and the rest "
    "is private. Bottom: four horizontal lines on the same day scale, 1.75 days for the holdout, 5.69 for "
    "validation, 9.34 for public and 11.23 for private; the last three all start around 17 September, inside "
    "the last train days."
)
SOURCE = "Data: docs/figures/data/calendar.csv, staleness.csv. Feed counts and times from the flattened splits."

COLORS = {"holdout": style.BLUE_250, "validation": style.BLUE_400, "public": style.BLUE_550,
          "private": style.BLUE_700}
ROW_TEXT = {
    "holdout": "Holdout: 30,000 random-order train feeds",
    "validation": "Validation: 14,423 feeds, Tue 22 and Thu 24 Sep",
    "public": "Public: 8,562 test feeds, up to Sun 27 Sep 11:32",
    "private": "Private: 20,033 test feeds after the cut",
}


def texts() -> style.Texts:
    return style.Texts(TITLE, CAPTION, ALT, SOURCE)


def make() -> None:
    cal = pd.read_csv(style.DATA_DIR / "calendar.csv", parse_dates=["date_msk"])
    st = pd.read_csv(style.DATA_DIR / "staleness.csv", parse_dates=["mean_feed_time_msk"])
    day0 = cal.date_msk.min()
    n_days = len(cal)
    assert n_days == 24 and cal.train_feeds.sum() == 3151208 and cal.test_feeds.sum() == 28595

    fig = style.figure(5.75)
    head = style.header(fig, TITLE, "Calendar of the data (Moscow time, September 2026) and the age of the "
                                    "popularity inputs in each evaluation set.")
    ax = style.canvas(fig)
    height = fig.get_size_inches()[1]
    cw = (style.WIDTH - 2 * style.MARGIN) / n_days  # width of one day

    def x_of(day: float) -> float:
        return style.MARGIN + day * cw

    # ---- calendar strip
    y_cell = height - head - 0.95
    h_cell = 0.44
    gap = 0.014
    for i, r in cal.iterrows():
        parts = []
        if r.split == "train":
            parts = [(0.0, 1.0, style.AXIS)]
        elif r.split == "validation":
            parts = [(0.0, 1.0, COLORS["validation"])]
        elif r.split == "test":
            p = r.public_part_of_day
            parts = [(0.0, p, COLORS["public"]), (p, 1.0, COLORS["private"])]
        for a, b, color in parts:
            if b > a:
                ax.add_patch(Rectangle((x_of(i + a) + gap, y_cell), (b - a) * cw - 2 * gap, h_cell, facecolor=color,
                                       edgecolor="none"))
        weekend = r.weekday in ("Sat", "Sun")
        ax.text(x_of(i + 0.5), y_cell - 0.17, str(r.date_msk.day), fontsize=style.FS_SMALL, ha="center",
                va="center", color=style.INK if r.split != "none" else style.INK2)
        ax.text(x_of(i + 0.5), y_cell - 0.37, r.weekday[0], fontsize=style.FS_SMALL, ha="center", va="center",
                color=style.INK if weekend else style.INK2, fontweight="bold" if weekend else "normal")
    ax.text(style.MARGIN, y_cell - 0.6, "Day of September and weekday, weekends in bold. A day without a block "
                                        "has at most 42 feeds.", fontsize=style.FS_SMALL, color=style.INK2, va="center")

    def group(first: float, last: float, text: str) -> None:
        """Bracket and name over the days from first to last (day numbers, fractions allowed)."""
        y = y_cell + h_cell + 0.1
        ax.plot([x_of(first) + gap, x_of(last) - gap], [y, y], color=style.INK2, lw=1.0)
        ax.text((x_of(first) + x_of(last)) / 2, y + 0.09, text, fontsize=style.FS_SMALL, color=style.INK,
                ha="center", va="bottom")

    cut = 21 + cal.public_part_of_day.iloc[21]
    group(0, 14, "Train: 3,151,208 feeds on 12 days")
    group(16, 19, "Validation")
    group(20, cut, "public")
    group(cut, 24, "private")
    ax.text(x_of(20) - 0.04, y_cell + h_cell + 0.19, "Test:", fontsize=style.FS_SMALL, color=style.INK, ha="right",
            va="bottom")

    # ---- age of the popularity inputs, on the same day scale
    y_top = y_cell - 1.05
    ax.text(style.MARGIN, y_top, "Age of the popularity inputs: mean days since a candidate item was last seen "
                                 "in train", fontsize=style.FS_TEXT, fontweight="bold", color=style.INK, va="center")
    train_end = cal[cal.split == "train"].index.max() + 1
    row = 0.36
    y_rows = [y_top - 0.42 - k * row for k in range(len(st))]
    y_guide = y_rows[-1] - 0.34
    ax.plot([x_of(train_end)] * 2, [y_guide - 0.08, y_rows[1] + 0.26], color=style.AXIS, lw=1.0, zorder=1)
    ax.text(x_of(train_end) + 0.07, y_guide, "end of the last train day, Sat 19 Sep", fontsize=style.FS_SMALL,
            color=style.INK2, ha="left", va="center")
    for y, (_, r) in zip(y_rows, st.iterrows()):
        end = (r.mean_feed_time_msk - day0) / pd.Timedelta(days=1)
        start = end - r.staleness_days
        color = COLORS[r.feed_set]
        ax.plot([x_of(start), x_of(end)], [y, y], color=color, lw=5, solid_capstyle="butt", zorder=3)
        ax.plot([x_of(end)], [y], marker="o", markersize=10, color=color, markeredgecolor=style.SURFACE,
                markeredgewidth=1.6, zorder=4)
        value = f"{r.staleness_days:.2f} days"
        if r.feed_set == "holdout":
            ax.text(x_of(end) + 0.14, y, f"{value}   {ROW_TEXT[r.feed_set]}", fontsize=style.FS_SMALL,
                    color=style.INK, ha="left", va="center")
        else:
            ax.text(style.MARGIN, y, ROW_TEXT[r.feed_set], fontsize=style.FS_SMALL, color=style.INK, ha="left",
                    va="center")
            ax.text(x_of(end) - 0.12, y + 0.15, value, fontsize=style.FS_SMALL, color=style.INK, ha="right",
                    va="center")
    ax.text(style.MARGIN, y_guide - 0.34, "Each line ends at the mean time of the set's feeds; its length is the "
                                          "age. Items never seen in train are left out.",
            fontsize=style.FS_SMALL, color=style.INK2, va="center")

    style.save(fig, NAME, texts())


if __name__ == "__main__":
    make()
