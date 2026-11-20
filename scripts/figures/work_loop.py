"""work-loop: how the nine days were run, and the closed loop that an automated harness would add."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, Rectangle

import style

NAME = "work-loop"
TITLE = ("The lead session stopped after each request and waited for the person;\n"
         "an automated harness would close the loop and add its own checks")
SOURCE = "Data: docs/figures/data/work_loop.json."

# columns (inches): the left column holds whoever stands outside the agents' work, the right part the loop
XL, WL = style.MARGIN, 2.3
XF = XL + WL + 0.45  # left edge of the loop area
XR = style.WIDTH - style.MARGIN  # right edge of the loop area
PAD = 0.12  # between the frame of the closed loop and its boxes
GAP_X = 0.42
WB = (XR - XF - 2 * PAD - GAP_X) / 2  # width of a loop box
X1 = XF + PAD
X2 = X1 + WB + GAP_X
H3, H4 = 0.72, 0.9  # box heights for 3 and 4 lines
GAP_A, GAP_B = 0.46, 0.28  # between the rows of the upper and of the lower panel


def texts() -> style.Texts:
    m = style.read_json("work_loop.json")
    n, days = m["requests_from_the_person"], style.count(m["days"])
    return style.Texts(
        title=TITLE,
        caption=(
            f"Top: how the {days} days were run. The person sent about {n} short requests and questions. After each request the "
            "lead session wrote down results and options and stopped until the next message. Bottom: the design "
            "of a stronger setup, a loop that proposes, runs, evaluates, keeps or drops and archives without a "
            "message, with a second model as critic and a scheduled look back over the decision ledger. The lower "
            "panel is a proposal; none of it ran."
        ),
        alt=(
            "Two box-and-arrow diagrams, one above the other. Top, how this entry was run: a loop from the person, "
            f"who sends a short request (about {n} in {days} days), to the lead session (one session, its context "
            f"summarised {m['context_summaries']} times, a status log and a decision ledger as memory), to "
            f"about {m['workflows']} multi-agent workflows, to experiments on GPUs, to results and options "
            f"written down, and back to the person, who uploads up to {m['uploads_per_day']} files a day and "
            "passes the public score back. A marker in the person's box says that the loop waits there for the "
            "next message. Bottom, what an automated harness would add: a closed loop of five steps inside a "
            "frame, propose (checked against the catalogue of closed ideas), run (with a noise probe), evaluate "
            "(on validation and the holdout, under a rule written first), keep or drop, and archive, from which "
            "the next round starts without a message. Outside the frame stand the person, who sets the goal and "
            "the budget and uploads the files, a second model as critic, and a scheduled look back over the "
            "ledger."
        ),
        source=SOURCE,
    )


def block(ax: Axes, x: float, yc: float, w: float, rows: list[tuple[str, bool]]) -> None:
    """Centred text lines around the height yc in a column of width w that starts at x; (text, bold) per line."""
    renderer = ax.figure.canvas.get_renderer()
    top = yc + (len(rows) - 1) * style.LINE / 2
    for i, (s, bold) in enumerate(rows):
        t = ax.text(x + w / 2, top - i * style.LINE, s, fontsize=style.FS_TEXT if bold else style.FS_SMALL,
                    fontweight="bold" if bold else "normal", color=style.INK, ha="center", va="center", zorder=3)
        width = t.get_window_extent(renderer).width / ax.figure.dpi
        assert width <= w - 0.12, f"text wider than its column by {width - w + 0.12:.2f} in: {s!r}"


def link(ax: Axes, x0: float, x1: float, y: float) -> None:
    """A horizontal connector with a head at both ends: an exchange with the loop, not a step of it."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y), zorder=1,
                arrowprops={"arrowstyle": "<|-|>", "color": style.INK2, "lw": 1.2, "shrinkA": 0, "shrinkB": 0,
                            "mutation_scale": 10})


def heading(ax: Axes, y: float, title: str, note: str) -> float:
    """The heading of a panel and one line under it. Returns the y of the top of the panel's content."""
    ax.text(XL, y, title, fontsize=style.FS_TEXT, fontweight="bold", color=style.INK, va="center")
    ax.text(XL, y - style.LINE - 0.02, note, fontsize=style.FS_SMALL, color=style.INK2, va="center")
    return y - style.LINE - 0.02 - 0.24


def make() -> None:
    m, t = style.read_json("work_loop.json"), texts()
    assert m["lead_sessions"] == 1
    n, days = m["requests_from_the_person"], style.count(m["days"])

    fig = style.figure(8.72)
    head = style.header(fig, t.title)
    ax = style.canvas(fig)
    y = fig.get_size_inches()[1] - head - 0.3

    # ---- upper panel: the loop of this entry runs through the person
    top = heading(ax, y, "How this entry was run",
                  "One interactive session. Every round starts with a message from the person.")
    y1 = top - H4  # row 1
    y2 = y1 - GAP_A - H4  # row 2
    style.box(ax, XL, y2, WL, top - y2, kind="orange")
    block(ax, XL, y1 + H4 / 2, WL, [("The person", True), ("sends a short request:", False),
                                    (f"about {n} in {days} days", False)])
    block(ax, XL, y2 + H4 / 2, WL, [(f"uploads up to {m['uploads_per_day']} files a day,", False),
                                    ("passes the public score back", False)])
    # the marker of the wait, between the two blocks of the person's box
    xw, ww, hw = XL + 0.1, WL - 0.2, 0.52
    yw = (y1 + y2 + H4) / 2 - 0.05 - hw / 2  # midway between the two text blocks
    ax.add_patch(FancyBboxPatch((xw + 0.1, yw + 0.1), ww - 0.2, hw - 0.2,
                                boxstyle="round,pad=0.1,rounding_size=0.1", facecolor=style.SURFACE,
                                edgecolor=style.ORANGE, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2.5))
    for dx in (0.0, 0.085):  # a pause sign: two bars
        ax.add_patch(Rectangle((xw + 0.13 + dx, yw + hw / 2 - 0.085), 0.045, 0.17, facecolor=style.ORANGE,
                               edgecolor="none", zorder=3))
    block(ax, xw + 0.25, yw + hw / 2, ww - 0.3, [("The loop waits here", True), ("for the next message", False)])

    style.box(ax, X1, y1, WB, H4, "Lead session", (f"one session for the {days} days,",
                                                   f"context summarised {m['context_summaries']} times,",
                                                   "memory: status log and ledger"))
    style.box(ax, X2, y1, WB, H4, "Multi-agent workflows", (f"about {m['workflows']} named in the status log,",
                                                            "agents in parallel"))
    style.box(ax, X2, y2, WB, H4, "Experiments", ("training and evaluation", "on GPUs"))
    style.box(ax, X1, y2, WB, H4, "Results and options", ("written down,", "then the session stops"), kind="data")
    style.arrow(ax, [(XL + WL, y1 + H4 / 2), (X1, y1 + H4 / 2)])
    style.arrow(ax, [(X1 + WB, y1 + H4 / 2), (X2, y1 + H4 / 2)])
    style.arrow(ax, [(X2 + WB / 2, y1), (X2 + WB / 2, y2 + H4)])
    style.arrow(ax, [(X2, y2 + H4 / 2), (X1 + WB, y2 + H4 / 2)])
    style.arrow(ax, [(X1, y2 + H4 / 2), (XL + WL, y2 + H4 / 2)])

    # ---- lower panel: a closed loop, with the person and two checks outside it
    top = heading(ax, y2 - 0.5, "What an automated harness would add",
                  "A closed loop that needs no message. A proposal: none of it ran.")
    r1 = top - PAD - H3
    r2 = r1 - GAP_B - H3
    r3 = r2 - GAP_B - H3
    bottom = r3 - PAD
    ax.add_patch(FancyBboxPatch((XF + 0.05, bottom + 0.05), XR - XF - 0.1, top - bottom - 0.1,
                                boxstyle="round,pad=0.05,rounding_size=0.08", facecolor=style.SURFACE,
                                edgecolor=style.MUTED, linewidth=1.0, zorder=0.5))
    style.box(ax, X1, r1, WB, H3, "Propose", ("checked against the", "catalogue of closed ideas"))
    style.box(ax, X2, r1, WB, H3, "Run", ("with a noise probe: the", "reference on the same seeds"))
    style.box(ax, X2, r2, WB, H3, "Evaluate", ("on validation and the holdout,", "under a rule written first"))
    style.box(ax, X2, r3, WB, H3, "Keep or drop", ("by that rule",))
    style.box(ax, X1, r3, WB, H3, "Archive", ("every result, in the ledger", "and the catalogue"), kind="data")
    style.arrow(ax, [(X1 + WB, r1 + H3 / 2), (X2, r1 + H3 / 2)])
    style.arrow(ax, [(X2 + WB / 2, r1), (X2 + WB / 2, r2 + H3)])
    style.arrow(ax, [(X2 + WB / 2, r2), (X2 + WB / 2, r3 + H3)])
    style.arrow(ax, [(X2, r3 + H3 / 2), (X1 + WB, r3 + H3 / 2)])
    xa = X1 + 0.42  # the return of the loop, with its note in the free cell
    style.arrow(ax, [(xa, r3 + H3), (xa, r1)])
    for i, s in enumerate(("The next round starts", "without a message")):
        ax.text(xa + 0.2, r2 + H3 / 2 + (0.5 - i) * style.LINE, s, fontsize=style.FS_SMALL, fontweight="bold",
                color=style.INK, va="center", ha="left")

    style.box(ax, XL, r1, WL, H3, "The person", ("sets the goal and budget,", "uploads the files"),
              kind="orange")
    style.box(ax, XL, r2, WL, H3, "Second model as critic", ("reads each plan", "and each result"), dashed=True)
    style.box(ax, XL, r3, WL, H3, "Scheduled look back", ("over the ledger, once a day,", "in a fresh context"),
              dashed=True)
    for r in (r1, r2, r3):
        link(ax, XL + WL, XF, r + H3 / 2)

    # ---- key
    yk = bottom - 0.36
    style.key(ax, XL, yk, [("orange", "the person"), ("blue", "work of agents"), ("data", "written records")])
    fill, edge = style.BOX["blue"]
    xk = ax.texts[-1].get_window_extent(fig.canvas.get_renderer()).x1 / fig.dpi + 0.28
    ax.add_patch(FancyBboxPatch((xk + 0.03, yk - 0.06), 0.2, 0.12, boxstyle="round,pad=0.03,rounding_size=0.03",
                                facecolor=fill, edgecolor=edge, linewidth=1.2, linestyle=(0, (3, 2))))
    ax.text(xk + 0.36, yk, "check from outside the loop", fontsize=style.FS_SMALL, color=style.INK2, va="center",
            ha="left")
    style.save(fig, NAME, t)


if __name__ == "__main__":
    make()
