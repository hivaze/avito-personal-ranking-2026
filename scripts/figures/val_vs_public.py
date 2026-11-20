"""val-vs-public: validation NDCG against public NDCG for the scored files, with the level line."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

import style

NAME = "val-vs-public"

ZOOM_X = (0.3985, 0.4037)
ZOOM_Y = (0.3920, 0.39565)
# members_train_on (first word) -> marker colour, shape, filled, legend text; in legend order, two per row
GROUPS = {
    "hold": (style.BLUE, "o", True, "members hold out 30,000 train feeds"),
    "all": (style.ORANGE, "s", True, "members trained on all train feeds"),
    "not": (style.MUTED, "o", True, "first two files: member training not recorded"),
    "mixed:": (style.ORANGE, "s", False, "union of 68 members, 44 of them trained on all train feeds"),
}


def load() -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(style.DATA_DIR / "public_scores.csv")
    d = d[d.val_usable == 1].copy()
    d["group"] = d.members_train_on.str.split().str[0]
    assert set(d.group) == set(GROUPS)
    line = style.read_json("level_line.json")
    d["resid"] = d.public_ndcg - (line["intercept"] + line["slope"] * d.val_ndcg)
    assert (d.in_level_fit == 1).sum() == line["files_in_fit"]
    return d, line


def picks() -> dict[int, pd.Series]:
    d, _ = load()
    return {p["pick"]: d[d.short_id == p["short_id"]].iloc[0] for p in style.read_json("final_picks.json")["picks"]}


def texts() -> style.Texts:
    d, line = load()
    a, b, sd = line["intercept"], line["slope"], line["residual_sd"]
    alld, hold = d[d.group.isin(["all", "mixed:"])], d[d.group == "hold"]
    below = int((alld.resid < 0).sum())
    pk = picks()
    p1, p2 = pk[1], pk[2]
    side = {True: "above", False: "below"}
    return style.Texts(
        title=(f"Validation predicts the public score of a file to about {sd};\n"
               f"files built on all-data members land {-alld.resid.mean():.4f} below the line on average"),
        caption=(f"Validation NDCG against public NDCG for the {len(d)} scored files that have a clean validation "
                 f"score. The line public = {a} + {b} × validation was fitted on the {line['files_in_fit']} files "
                 f"scored by 14 November (residual sd {sd}). On the right, {below} of the {len(alld)} files whose "
                 "members were trained on all train feeds (the 68-member union included) sit below the line, by "
                 f"{-alld.resid.mean():.4f} on average; the {len(hold)} files whose members hold out 30,000 train "
                 f"feeds average {style.signed(hold.resid.mean())}."),
        alt=(f"Two scatter plots of public NDCG against validation NDCG. Left: {len(d)} files from validation "
             f"{d.val_ndcg.min():.3f} to {d.val_ndcg.max():.3f} lie along a straight line from public "
             f"{d.public_ndcg.min():.3f} to {d.public_ndcg.max():.3f}. Right: enlargement of validation "
             f"{ZOOM_X[0]} to {ZOOM_X[1]}; files whose members hold out 30,000 train feeds scatter on both sides of "
             f"the line within about 0.001, and {below} of the {len(alld)} files built on all-data members are "
             f"below it. Pick 1, {p1.short_id}, is {side[p1.resid > 0]} the line at validation {p1.val_ndcg:.5f} "
             f"and public {p1.public_ndcg:.5f}; pick 2, {p2.short_id}, is {side[p2.resid > 0]} it at "
             f"{p2.val_ndcg:.5f} and {p2.public_ndcg:.5f}."),
        source=("Data: docs/figures/data/public_scores.csv, level_line.json. Scores: validation 14,423 feeds, "
                "public 8,562."),
    )


def scatter(ax, d: pd.DataFrame, size: float) -> None:
    for g, (color, marker, filled, _) in GROUPS.items():
        m = d[d.group == g]
        if filled:
            ax.scatter(m.val_ndcg, m.public_ndcg, s=size, color=color, marker=marker, edgecolor=style.SURFACE,
                       linewidth=1.4, zorder=4)
        else:
            ax.scatter(m.val_ndcg, m.public_ndcg, s=size * 0.62, facecolor=style.SURFACE, marker=marker,
                       edgecolor=color, linewidth=1.8, zorder=5)


def label(ax, d: pd.DataFrame, short_id: str, text: str, x: float, y: float, ha: str = "left",
          va: str = "center", relpos: tuple[float, float] = (0.5, 0.5)) -> None:
    r = d[d.short_id == short_id].iloc[0]
    ax.annotate(text, xy=(r.val_ndcg, r.public_ndcg), xytext=(x, y), ha=ha, va=va, fontsize=style.FS_SMALL,
                color=style.INK, linespacing=1.25, zorder=6,
                arrowprops={"arrowstyle": "-", "color": style.MUTED, "lw": 0.9, "shrinkA": 3, "shrinkB": 5,
                            "relpos": relpos})


def make() -> None:
    (d, line), t, pick = load(), texts(), picks()
    a, b, sd = line["intercept"], line["slope"], line["residual_sd"]
    all_data = d[d.group.isin(["all", "mixed:"])]
    inside = d.val_ndcg.between(*ZOOM_X) & d.public_ndcg.between(*ZOOM_Y)
    assert inside[all_data.index].all(), "a file with all-data members falls outside the enlarged panel"

    mix = {f["id"]: f for f in style.read_json("ensemble.json")["files"]}
    size = {k: f"{mix[pick[k].short_id]['wide_students']} + {mix[pick[k].short_id]['popularity_light_students']} "
               "students" for k in pick}
    if size[1] == size[2] and pick[1].members_train_on == pick[2].members_train_on == "all train feeds":
        picks_text = f"{pick[1].short_id} and {pick[2].short_id}, both {size[1]} on all train feeds"
    else:
        picks_text = (f"{pick[1].short_id} ({size[1]}, {pick[1].members_train_on}) and {pick[2].short_id} "
                      f"({size[2]}, {pick[2].members_train_on})")
    fig = style.figure(6.2)
    head = style.header(fig, t.title, f"One point per scored file. Line: public = {a} + {b} × validation, "
                                      f"fitted on the {line['files_in_fit']} files scored by 14 Nov.\n"
                                      f"Final picks: {picks_text}.")
    left = style.axes(fig, 0.85, 1.05, 5.05, head + 0.8)
    right = style.axes(fig, 4.95, 1.05, 0.3, head + 0.8)

    xs = np.array([0.380, 0.4045])
    for ax in (left, right):
        ax.grid(True)
        ax.plot(xs, a + b * xs, color=style.INK2, lw=1.2, zorder=3)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.set_xlabel("Validation NDCG")
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    # ---- left: all files
    left.set_title(f"All {len(d)} files: one line fits them", fontsize=style.FS_TEXT, fontweight="bold", loc="left",
                   pad=8)
    left.set_xlim(0.3800, 0.4045)
    left.set_ylim(0.3722, 0.3968)
    left.xaxis.set_major_locator(MultipleLocator(0.005))
    left.yaxis.set_major_locator(MultipleLocator(0.005))
    left.set_ylabel("Public NDCG")
    scatter(left, d, 40)
    left.add_patch(Rectangle((ZOOM_X[0], ZOOM_Y[0]), ZOOM_X[1] - ZOOM_X[0], ZOOM_Y[1] - ZOOM_Y[0], fill=False,
                             edgecolor=style.MUTED, lw=0.9, zorder=2))
    left.text(ZOOM_X[0] - 0.0004, ZOOM_Y[1], "enlarged\non the right", fontsize=style.FS_SMALL, color=style.INK2,
              ha="right", va="top", linespacing=1.25)
    left.text(0.3912, 0.3794, "level line,\nresidual sd 0.0006", fontsize=style.FS_SMALL, color=style.INK2,
              ha="left", va="top", linespacing=1.25)
    left.annotate("", xy=(0.3895, a + b * 0.3895), xytext=(0.3911, 0.3789),
                  arrowprops={"arrowstyle": "-", "color": style.MUTED, "lw": 0.9, "shrinkA": 0, "shrinkB": 3})

    # ---- right: the final cluster
    below = int((all_data.resid < 0).sum())
    right.set_title(f"{below} of the {len(all_data)} all-data files are below the line", fontsize=style.FS_TEXT,
                    fontweight="bold", loc="left", pad=8)
    right.set_xlim(*ZOOM_X)
    right.set_ylim(*ZOOM_Y)
    right.xaxis.set_major_locator(MultipleLocator(0.001))
    right.yaxis.set_major_locator(MultipleLocator(0.001))
    right.fill_between(xs, a + b * xs - sd, a + b * xs + sd, color=style.NEUTRAL, zorder=1, lw=0)
    scatter(right, d, 62)
    right.text(0.39995, 0.39412, "grey band:\nline ± 0.0006", fontsize=style.FS_SMALL, color=style.INK2, ha="right",
               va="bottom", linespacing=1.25)
    # the picks sit 0.0001 apart: both labels end at the same x, the higher label goes to the higher point
    upper, lower = sorted(pick, key=lambda k: -pick[k].public_ndcg)
    for k, y in ((upper, 0.39552), (lower, 0.39530)):
        label(right, d, pick[k].short_id, f"pick {k}, {pick[k].short_id}", 0.4017, y, ha="right",
              relpos=(1.0, 0.5))
    label(right, d, "FD16P20", "FD16P20,\nearlier pick 1", 0.40245, 0.39385, ha="left", relpos=(0.0, 0.5))
    label(right, d, "M2", "best on validation\non 12 Nov; 0.0011\nbelow the line", 0.40135, 0.39246, ha="left")

    # legend: two rows above the panels
    w, h = fig.get_size_inches()
    for i, (color, marker, filled, text) in enumerate(GROUPS.values()):
        x = (style.MARGIN + (0.0, 3.55)[i % 2]) / w
        y = 1 - (head + 0.13 + 0.23 * (i // 2)) / h
        fig.add_artist(Line2D([x + 0.07 / w], [y], marker=marker, linestyle="none", transform=fig.transFigure,
                              markersize=7.5 if filled else 6, markerfacecolor=color if filled else style.SURFACE,
                              markeredgecolor=style.SURFACE if filled else color,
                              markeredgewidth=1.2 if filled else 1.8))
        fig.text(x + 0.22 / w, y, text, fontsize=style.FS_SMALL, color=style.INK2, va="center", ha="left")

    style.save(fig, NAME, t)


if __name__ == "__main__":
    make()
