"""public-progress: every scored file in the order its score was logged, with the team-best step line."""

from __future__ import annotations

import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

import style

NAME = "public-progress"

DAY_GAP = 1.0
Y_DETAIL = (0.39195, 0.39865)


def load() -> pd.DataFrame:
    d = pd.read_csv(style.DATA_DIR / "public_scores.csv")
    day = d.date_utc.rank(method="dense").astype(int) - 1
    d["x"] = d.order + day * DAY_GAP
    return d


def board() -> pd.Series:
    """The last fetched board snapshot."""
    lb = pd.read_csv(style.DATA_DIR / "leaderboard.csv")
    return lb[lb.kind == "fetched"].iloc[-1]


def picks(d: pd.DataFrame) -> dict[int, pd.Series]:
    return {p["pick"]: d[d.short_id == p["short_id"]].iloc[0] for p in style.read_json("final_picks.json")["picks"]}


def texts() -> style.Texts:
    d, snap = load(), board()
    snap_time = pd.Timestamp(snap.snapshot_utc)
    days = pd.to_datetime(d.date_utc)
    best = d[d.is_team_best == 1]
    top = best.iloc[-1]
    day1 = best[best.date_utc == d.date_utc.min()]
    gain, gain1 = top.public_ndcg - best.public_ndcg.iloc[0], day1.public_ndcg.iloc[-1] - best.public_ndcg.iloc[0]
    n_days = (pd.Timestamp(top.date_utc) - days.min()).days + 1
    p = picks(d)
    assert top.short_id == p[2].short_id, "the alt text assumes pick 2 is the team best"
    later = best[best.date_utc > d.date_utc.min()]
    v = {r.short_id: f"{r.public_ndcg:.5f}" for r in later.itertuples()}
    n_first = len(d[d.date_utc == d.date_utc.min()])
    return style.Texts(
        title=(f"Public NDCG went from {best.public_ndcg.iloc[0]:.4f} to {top.public_ndcg:.4f} in "
               f"{style.count(n_days)} days;\n"
               f"{gain1:.3f} of the {gain:.3f} came on the first day"),
        caption=(f"All {len(d)} scored files in the order their public score was logged, {days.min():%-d} to "
                 f"{days.max():%-d %B %Y} (UTC). The blue step line is the team best; the ringed points are the two "
                 f"final picks, {p[1].short_id} and {p[2].short_id}. The two horizontal lines are places 1 and 2 on the "
                 f"last fetched board snapshot ({snap_time:%-d %B}, {snap_time:%H:%M} UTC), with the team at rank "
                 f"{int(snap.team_rank)} of {int(snap.teams)}."),
        alt=(f"Two stacked scatter plots of public NDCG against submission order. Top: all {len(d)} files; the team "
             f"best climbs from {best.public_ndcg.iloc[0]:.4f} to {day1.public_ndcg.iloc[-1]:.4f} over the "
             f"{style.count(n_first)} files of {days.min():%-d %B} and then flattens. Bottom: "
             f"detail of {days[d.date_utc > d.date_utc.min()].min():%-d} to {days.max():%-d %B} between "
             f"{Y_DETAIL[0]:.3f} and {Y_DETAIL[1]:.4f}; the team best steps up {style.count(len(later))} more "
             f"times, to "
             f"{v['G3']} and then {v['C3']} with pools of rankers trained on clicks, {v['W8']} with distilled "
             f"students, {v['W8P20']} with the popularity-light blend W8P20, {v[p[1].short_id]} with {p[1].short_id} "
             f"(pick 1) on {pd.Timestamp(p[1].date_utc):%-d %B} and {top.public_ndcg:.5f} with {top.short_id} "
             f"(pick 2) on {pd.Timestamp(top.date_utc):%-d %B}. On the "
             f"board of {snap_time:%-d %B}, place 2 is at {snap.rank2_ndcg:.6f}, the leader at "
             f"{snap.rank1_ndcg:.6f}, and the team is at rank {int(snap.team_rank)} of {int(snap.teams)}."),
        source=(f"Data: docs/figures/data/public_scores.csv, leaderboard.csv. Board last fetched "
                f"{snap_time:%-d %b %Y, %H:%M} UTC."),
    )


def draw_points(ax, d: pd.DataFrame, x_end: float, ring: float) -> None:
    best = d[d.is_team_best == 1]
    ax.step(list(best.x) + [x_end], list(best.public_ndcg) + [best.public_ndcg.iloc[-1]], where="post",
            color=style.BLUE, lw=2, zorder=3)
    rest = d[d.is_team_best == 0]
    ax.scatter(rest.x, rest.public_ndcg, s=46, color=style.MUTED, edgecolor=style.SURFACE, linewidth=1.5, zorder=4)
    ax.scatter(best.x, best.public_ndcg, s=58, color=style.BLUE, edgecolor=style.SURFACE, linewidth=1.5, zorder=5)
    for r in picks(d).values():
        ax.scatter([r.x], [r.public_ndcg], s=ring, facecolor="none", edgecolor=style.INK, linewidth=1.3, zorder=6)


def note(ax, d: pd.DataFrame, short_id: str, text: str, x: float, y: float, ha: str = "left",
         va: str = "center", line: bool = True, relpos: tuple[float, float] = (0.5, 0.5)) -> None:
    """Label one file. (x, y) is where the text goes: x in slots relative to the point, y in NDCG.

    ``relpos`` is the point of the text box the line starts from, (0, 0) its lower left corner."""
    r = d[d.short_id == short_id].iloc[0]
    ax.annotate(text, xy=(r.x, r.public_ndcg), xytext=(r.x + x, y), ha=ha, va=va, fontsize=style.FS_SMALL,
                color=style.INK, zorder=7, linespacing=1.25,
                arrowprops={"arrowstyle": "-", "color": style.MUTED, "lw": 0.9, "shrinkA": 3, "shrinkB": 7,
                            "relpos": relpos}
                if line else None)


def spread(fig, ax, labels: list, lo: float, hi: float, gap_in: float = 0.15) -> None:
    """Move centred labels apart along x until neighbours are at least ``gap_in`` inches apart, inside lo..hi."""
    slot = (ax.get_position().width * fig.get_size_inches()[0]) / (hi - lo)  # inches per x unit
    renderer = fig.canvas.get_renderer()
    half = [t.get_window_extent(renderer).width / fig.dpi / slot / 2 for t in labels]
    x = [t.get_position()[0] for t in labels]
    for i in range(1, len(x)):  # push right
        x[i] = max(x[i], x[i - 1] + half[i - 1] + half[i] + gap_in / slot)
    x[-1] = min(x[-1], hi - half[-1])
    for i in range(len(x) - 2, -1, -1):  # then back from the right edge
        x[i] = min(x[i], x[i + 1] - half[i + 1] - half[i] - gap_in / slot)
    assert x[0] - half[0] >= lo, "the day labels do not fit under the panel"
    for t, xi in zip(labels, x):
        t.set_x(xi)


def make() -> None:
    d, snap, t = load(), board(), texts()
    leader, second = float(snap.rank1_ndcg), float(snap.rank2_ndcg)
    p = picks(d)

    fig = style.figure(7.3)
    head = style.header(fig, t.title, "Every scored file in the order its score was logged. Blue: a new best score. "
                                      "Grey: every other file.\nRinged: the two final picks.")
    x0, x1 = d.x.min() - 1.0, d.x.max() + 1.0
    top = style.axes(fig, 0.85, 4.1, 0.3, head + 0.12)
    bot = style.axes(fig, 0.85, 0.95, 0.3, 3.5)

    for ax, ring in ((top, 150), (bot, 190)):
        ax.set_xlim(x0, x1)
        style.hgrid(ax)
        ax.set_xticks([])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.set_ylabel("Public NDCG")
        for v in (leader, second):
            ax.axhline(v, color=style.INK2, lw=1.0, zorder=2)
        draw_points(ax, d, x1, ring)
        for _, g in d.groupby("date_utc"):  # hairlines between days
            if g.x.min() > d.x.min():
                ax.axvline(g.x.min() - (1 + DAY_GAP) / 2, color=style.GRID, lw=0.8, zorder=1)

    # ---- top: everything
    top.set_ylim(0.3722, 0.4008)
    top.yaxis.set_major_locator(MultipleLocator(0.005))
    top.axhspan(*Y_DETAIL, color=style.NEUTRAL, zorder=0)
    top.text(x0 + 0.3, leader + 0.0003, f"#1 {leader:.6f}", fontsize=style.FS_SMALL, color=style.INK2,
             va="bottom")
    top.text(x0 + 0.3, second - 0.0003, f"#2 {second:.6f}", fontsize=style.FS_SMALL, color=style.INK2,
             va="top")
    top.text(x1 - 0.3, leader + 0.0003, f"board of {pd.Timestamp(snap.snapshot_utc):%-d %b}: rank "
             f"{int(snap.team_rank)} of {int(snap.teams)} teams", fontsize=style.FS_SMALL, color=style.INK,
             ha="right", va="bottom")
    note(top, d, "v3", "First file, 0.3739: neural rankers and tree models", 1.5, 0.3739, line=False)
    note(top, d, "v10", "Four ten-epoch rankers, 0.3879", 2.2, 0.3838)
    note(top, d, "v13 / Cd", "Pool of 19 rankers from three recipes, 0.3921", 2.6, 0.3884)
    top.text(x1 - 0.3, Y_DETAIL[0] - 0.0008, "the shaded band is enlarged below", fontsize=style.FS_SMALL,
             color=style.INK2, ha="right", va="top")

    # ---- bottom: the band between 0.392 and 0.3986
    bot.set_ylim(*Y_DETAIL)
    bot.yaxis.set_major_locator(MultipleLocator(0.001))
    bot.text(x0 + 0.3, leader + 0.00008, f"#1 {leader:.6f}", fontsize=style.FS_SMALL, color=style.INK2,
             va="bottom")
    close = leader - second < 0.0004  # the #2 label would cover the #1 line: put it under its own line
    bot.text(x0 + 0.3, second + (-0.00008 if close else 0.00008), f"#2 {second:.6f}", fontsize=style.FS_SMALL,
             color=style.INK2, va="top" if close else "bottom")
    note(bot, d, "G3", "Pool of 20 rankers\ntrained on clicks, 0.39282", 0.2, 0.39345, ha="center", va="bottom")
    note(bot, d, "C3", "8 popularity-light\nrankers added, 0.39345", -1.3, 0.39415, ha="center", va="bottom")
    note(bot, d, "W8", "8 distilled students, 0.39449:\n+0.00103 over the best file before", 1.2, 0.39315,
         ha="left", va="center")
    p1, p2 = p[1], p[2]
    w8p20 = d[d.short_id == "W8P20"].iloc[0]
    note(bot, d, "W8P20", f"W8P20, {w8p20.public_ndcg:.5f}: the 8 students mixed with\n4 popularity-light "
                          "students (pick 2 until 19 Nov)", x0 + 0.3 - w8p20.x, 0.39570, ha="left",
         relpos=(1.0, 0.5))
    note(bot, d, p1.short_id, f"Pick 1, {p1.public_ndcg:.5f}: 16 + 4 students,\npopularity-light weight 0.4 on\n"
                              f"short feeds ({p1.short_id})", -11.0, 0.39750, ha="right", va="top",
         relpos=(1.0, 0.0))
    note(bot, d, p2.short_id, f"Pick 2 and best score, {p2.public_ndcg:.5f}:\n16 + 4 students, the 16 "
                              f"trained\nwith pseudo-feeds ({p2.short_id})",
         x1 - 0.3 - p2.x, 0.39750, ha="right", va="top")

    day_labels = [bot.text(g.x.mean(), Y_DETAIL[0] - 0.00028,
                           f"{pd.Timestamp(day).strftime('%-d Nov')}\n{len(g)} files", fontsize=style.FS_SMALL,
                           color=style.INK2, ha="center", va="top", linespacing=1.3)
                  for day, g in d.groupby("date_utc")]  # day labels under the lower panel
    spread(fig, bot, day_labels, x0, x1)

    style.save(fig, NAME, t)


if __name__ == "__main__":
    make()
