"""near-copies-noise: the scored files of the final family on public, with paired standard errors."""

from __future__ import annotations

import math

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

import style

NAME = "near-copies-noise"
REF = "FD16P20"

# horizontal layout, inches from the left edge of the figure
LABEL_W = 4.25  # file names and descriptions
PLOT_L = style.MARGIN + LABEL_W + 0.2
PLOT_R = 2.0  # right margin of the plot area: values column
ROW = 0.47  # inches per file


def earlier_picks() -> dict[str, str]:
    """short_id -> ", pick 1 until 18 Nov" for the files that were a final pick before the present two."""
    fp = style.read_json("final_picks.json")
    out = {}
    for k in (1, 2):
        e = fp.get(f"earlier_pick_{k}")
        if e:
            out[e["short_id"]] = f", pick {k} until {pd.Timestamp(e[f'pick_{k}_until']):%-d %b}"
    return out


def load() -> pd.DataFrame:
    d = pd.read_csv(style.DATA_DIR / "near_copies.csv").sort_values("public_ndcg", ascending=False)
    d = d.reset_index(drop=True)
    picks = {p["pick"]: p["short_id"] for p in style.read_json("final_picks.json")["picks"]}
    assert {int(r.final_pick): r.short_id for r in d.itertuples() if r.final_pick} == picks
    ref = float(d[d.short_id == REF].public_ndcg.iloc[0])
    d["z"] = (d.public_ndcg - ref) / d.paired_se_vs_fd16p20
    return d


def texts() -> style.Texts:
    d = load()
    n, ref = len(d), float(d[d.short_id == REF].public_ndcg.iloc[0])
    zmax = d.z.abs().max()
    assert zmax < 2, "a file of the family is more than 2 paired standard errors from FD16P20"
    bound = math.ceil(zmax * 10) / 10
    pick = {int(r.final_pick): r for r in d.itertuples() if r.final_pick}
    se = d.paired_se_vs_fd16p20.dropna()
    top, bottom = d.iloc[0], d.iloc[-1]
    rank = {r.short_id: i + 1 for i, r in d.iterrows()}
    order = {k: "highest" if rank[r.short_id] == 1 else f"number {rank[r.short_id]}" for k, r in pick.items()}
    days = pd.read_csv(style.DATA_DIR / "public_scores.csv").set_index("short_id").loc[d.short_id, "date_utc"]
    first, last = (pd.Timestamp(x) for x in (days.min(), days.max()))
    return style.Texts(
        title=(f"The public board cannot separate the {n} files of the final family:\n"
               f"every difference to {REF} is within {bound:.1f} paired standard errors"),
        caption=(f"Public NDCG (8,562 feeds) of the {n} scored files built as full-view distilled students plus "
                 f"popularity-light students, scored {first:%-d} to {last:%-d %B %Y}. Bars show one and two "
                 f"paired standard errors of each file's difference to {REF}, pick 1 until 18 November; every "
                 f"two-standard-error bar reaches the score of {REF}. Short feeds have fewer than 50 candidates, "
                 "long feeds 50 or more."),
        alt=(f"Dot plot of {n} files against public NDCG from {d.public_ndcg.min():.5f} to "
             f"{d.public_ndcg.max():.5f}. Pick 1, {pick[1].short_id}, is {order[1]} at {pick[1].public_ndcg:.5f}; "
             f"pick 2, {pick[2].short_id}, is {order[2]} at {pick[2].public_ndcg:.5f}; "
             f"{REF} at {ref:.5f} is number {rank[REF]} and carries the vertical reference line. Every other file "
             f"has error bars of one and two paired standard errors, {se.min():.5f} to {se.max():.5f} per "
             f"standard error, and every two-standard-error bar crosses the line at {REF}. The largest "
             f"differences are {style.signed(top.public_ndcg - ref)} ({style.signed(top.z, 1)} standard errors) and "
             f"{style.signed(bottom.public_ndcg - ref)} ({style.signed(bottom.z, 1)})."),
        source=(f"Data: docs/figures/data/near_copies.csv. Paired SE as logged; for files of 17-{last:%-d} Nov "
                "recomputed the same way."),
    )


def make() -> None:
    d, t = load(), texts()
    ref = float(d[d.short_id == REF].public_ndcg.iloc[0])
    n = len(d)
    mean, sd = d.public_ndcg.mean(), d.public_ndcg.std()

    fig = style.figure(1.75 + 0.42 + n * ROW + 1.0)
    head = style.header(fig, t.title, f"Public NDCG of each file. Thick bar: ±1 paired standard error of its "
                                      f"difference to {REF}. Thin bar: ±2.\nThe {n} scores average {mean:.5f} "
                                      f"with a standard deviation of {sd:.5f}. Short feeds: under 50 candidates.")
    ax = style.axes(fig, PLOT_L, 1.0, PLOT_R, head + 0.42)
    lo = (d.public_ndcg - 2 * d.paired_se_vs_fd16p20).min()
    hi = (d.public_ndcg + 2 * d.paired_se_vs_fd16p20).max()
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(math.floor(lo * 1e4 - 0.5) / 1e4, math.ceil(hi * 1e4 + 0.5) / 1e4)
    ax.xaxis.set_major_locator(MultipleLocator(0.001))
    ax.xaxis.set_minor_locator(MultipleLocator(0.0005))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(axis="x", which="both")
    ax.tick_params(which="minor", length=0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Public NDCG")
    ax.axvline(ref, color=style.INK2, lw=1.0, zorder=2)

    to_fig = ax.get_yaxis_transform()  # x in axes fraction, y in data
    w = fig.get_size_inches()[0]
    plot_w = w - PLOT_L - PLOT_R
    x_label = -(PLOT_L - style.MARGIN) / plot_w  # left edge of the row labels
    x_value = 1 + 0.15 / plot_w
    renderer = fig.canvas.get_renderer()
    earlier = earlier_picks()
    assert REF in earlier
    for i, r in d.iterrows():
        color = style.BLUE if r.final_pick else style.MUTED
        if pd.notna(r.paired_se_vs_fd16p20):
            se = r.paired_se_vs_fd16p20
            ax.plot([r.public_ndcg - 2 * se, r.public_ndcg + 2 * se], [i, i], color=color, lw=1.3, zorder=3)
            ax.plot([r.public_ndcg - se, r.public_ndcg + se], [i, i], color=color, lw=4.5, zorder=3,
                    solid_capstyle="butt")
        ax.scatter([r.public_ndcg], [i], s=70, color=color, edgecolor=style.SURFACE, linewidth=1.5, zorder=4)
        name = r.short_id + (f", pick {int(r.final_pick)}" if r.final_pick else "")
        name += earlier.get(r.short_id, "")
        ax.text(x_label, i - 0.2, name, transform=to_fig, fontsize=style.FS_TEXT, fontweight="bold",
                color=style.INK, ha="left", va="center")
        detail = ax.text(x_label, i + 0.2, r.label, transform=to_fig, fontsize=style.FS_SMALL, color=style.INK2,
                         ha="left", va="center")
        assert detail.get_window_extent(renderer).width / fig.dpi <= LABEL_W, f"description too long: {r.label!r}"
        ax.text(x_value, i - 0.2, f"{r.public_ndcg:.5f}", transform=to_fig, fontsize=style.FS_TEXT,
                color=style.INK, ha="left", va="center")
        diff = ("reference" if pd.isna(r.paired_se_vs_fd16p20)
                else f"{style.signed(r.public_ndcg - ref)} ± {r.paired_se_vs_fd16p20:.5f}")
        ax.text(x_value, i + 0.2, diff, transform=to_fig, fontsize=style.FS_SMALL, color=style.INK2, ha="left",
                va="center")
        if i:
            ax.axhline(i - 0.5, color=style.GRID, lw=0.8, zorder=1, xmin=x_label,
                       xmax=1 + (PLOT_R - style.MARGIN) / plot_w, clip_on=False)

    # column heads
    y_head = -0.55
    ax.text(x_label, y_head, "File", transform=to_fig, fontsize=style.FS_SMALL, color=style.INK2, ha="left",
            va="bottom")
    ax.text(x_value, y_head, "Public NDCG and\ndifference ± paired SE", transform=to_fig, fontsize=style.FS_SMALL,
            color=style.INK2, ha="left", va="bottom", linespacing=1.2)
    ax.text(ref, y_head, REF, fontsize=style.FS_SMALL, color=style.INK2, ha="center", va="bottom")

    # colour key under the axis
    fy = 0.66 / fig.get_size_inches()[1]
    for x, color, text in ((style.MARGIN + 0.06, style.BLUE, "the two final picks"),
                           (style.MARGIN + 2.0, style.MUTED, "other files of the family")):
        fig.add_artist(Line2D([x / w + 0.008], [fy], marker="o", linestyle="none", markersize=8, color=color,
                              markeredgecolor=style.SURFACE, markeredgewidth=1.2, transform=fig.transFigure))
        fig.text(x / w + 0.024, fy, text, fontsize=style.FS_SMALL, color=style.INK2, ha="left", va="center")

    style.save(fig, NAME, t)


if __name__ == "__main__":
    make()
