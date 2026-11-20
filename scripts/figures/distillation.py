"""distillation: how the training target of a student is made, and what it bought."""

from __future__ import annotations

import json

import style

NAME = "distillation"
TITLE = ("Half clicks, half teacher: one student gained 0.0026 on validation,\n"
         "the first file of students 0.0010 on the public board")
CAPTION = (
    "How a student's training target is made. The logits of 11 or 25 rankers trained on clicks are centred per "
    "feed and averaged; the target of a feed is half the normalised clicks and half the softmax of that teacher. "
    "Gains: one student against the same ranker trained on clicks (width 128, 4 seeds), and the first file of 8 "
    "full-view students against the best earlier file."
)
ALT = (
    "Box-and-arrow diagram. Top row: 25 rankers trained on clicks produce logits on 246.7 million train "
    "candidates, which are centred per feed and averaged into a teacher file. Middle row: the clicks of a feed "
    "and the softmax of the teacher scores of that feed enter the training target with weight 0.5 each. Below: "
    "the student, the same ranker at width 192, is trained on that target. Three figures at the bottom: plus "
    "0.00257 validation NDCG and plus 0.00200 holdout NDCG for one student, plus 0.00103 public NDCG for the "
    "first file of students."
)
SOURCE = "Data: docs/figures/data/distillation.json, from the decision log entries of 14 Nov 2026."

X1, W1 = style.MARGIN, 2.7
X2, W2 = 3.3, 2.4
X3, W3 = 6.0, 2.7
H = 0.72


def texts() -> style.Texts:
    return style.Texts(TITLE, CAPTION, ALT, SOURCE)


def make() -> None:
    m = json.loads((style.DATA_DIR / "distillation.json").read_text())
    groups = m["teacher_rankers"]
    n_all = sum(g["count"] for g in groups)
    n_small = sum(g["count"] for g in groups if g["in_teacher_of_11"])
    assert (n_small, n_all) == (11, 25)
    wt = m["teacher_weight"]
    val, hold, pub = m["gains"]

    fig = style.figure(7.2)
    head = style.header(fig, TITLE)
    ax = style.canvas(fig)
    top = fig.get_size_inches()[1] - head - 0.22

    # ---- row 1: from the rankers to the teacher file
    y1 = top - H
    style.box(ax, X1, y1, W1, H, "Teacher rankers", (f"{n_all} rankers trained on clicks;",
                                                     "30,000 train feeds held out"))
    style.box(ax, X2, y1, W2, H, "Their logits",
              (f"on all {m['train_candidates'] / 1e6:.1f} M train", "candidate rows"), kind="data")
    style.box(ax, X3, y1, W3, H, "Teacher file", ("logits centred per feed, then",
                                                  f"averaged over {n_small} or {n_all} rankers"), kind="data")
    style.arrow(ax, [(X1 + W1, y1 + H / 2), (X2, y1 + H / 2)])
    style.arrow(ax, [(X2 + W2, y1 + H / 2), (X3, y1 + H / 2)])

    # ---- row 2: the target of one feed
    y2 = y1 - 0.34 - H
    style.box(ax, X1, y2, W1, H, "Clicks of a train feed", ("y / sum(y)",), kind="data")
    style.box(ax, X2, y2, W2, H, "Training target", (f"{wt} · clicks + {wt} · teacher",), kind="orange")
    style.box(ax, X3, y2, W3, H, "Teacher scores of the feed", ("softmax over its candidates",), kind="data")
    style.arrow(ax, [(X3 + W3 / 2, y1), (X3 + W3 / 2, y2 + H)])
    style.arrow(ax, [(X1 + W1, y2 + H / 2), (X2, y2 + H / 2)])
    style.arrow(ax, [(X3, y2 + H / 2), (X2 + W2, y2 + H / 2)])
    for x in ((X1 + W1 + X2) / 2, (X2 + W2 + X3) / 2):
        ax.text(x, y2 + H / 2 + 0.15, f"{wt}", fontsize=style.FS_SMALL, color=style.INK, ha="center", va="center")

    # ---- row 3: the student, with notes on both sides
    h3 = 0.9
    y3 = y2 - 0.34 - h3
    style.box(ax, X2, y3, W2, h3, "Student", (f"same ranker, width {m['student_width']};", "listwise cross-entropy",
                                              "to the target"), kind="orange")
    style.arrow(ax, [(X2 + W2 / 2, y2), (X2 + W2 / 2, y3 + h3)])

    def notes(x: float, rows: list[tuple[str, bool]]) -> None:
        y = y3 + h3 - 0.1
        for k, (text, bold) in enumerate(rows):
            y -= 0.08 if bold and k else 0
            ax.text(x, y, text, fontsize=style.FS_SMALL, fontweight="bold" if bold else "normal",
                    color=style.INK if bold else style.INK2, va="center", ha="left")
            y -= style.LINE

    notes(X1, [
        (f"Teacher of {n_small}", True),
        (f"{groups[0]['count']} rankers with a re-run cooldown,", False),
        (f"{groups[1]['count']} trained in one run", False),
        (f"Teacher of {n_all}", True),
        (f"the {n_small}, {groups[2]['count']} with a GRU history", False),
        (f"encoder, {groups[3]['count']} popularity-light", False),
    ])
    f = m["students_in_final_files"]["FD16P20"]
    notes(X3 + 0.12, [
        ("In the final file FD16P20", True),
        (f"{f['wide_taught_by_11']} full-view students per teacher file,", False),
        (f"{f['popularity_light']} popularity-light, teacher of {f['popularity_light_teacher']}", False),
        ("Otherwise as the base rankers", True),
        ("same inputs, 10 epochs, auxiliary", False),
        ("display-order loss", False),
    ])

    # ---- what it bought
    y4 = y3 - 0.6
    ax.text(X1, y4, "What it bought", fontsize=style.FS_TEXT, fontweight="bold", color=style.INK, va="center")
    w_tile, ht = 2.7, 1.18
    yt = y4 - 0.2 - ht
    tiles = [
        (f"+{val['delta']:.5f} ± {val['se']:.5f}", ("validation NDCG,", "one student against the same",
                                                    "ranker trained on clicks")),
        (f"+{hold['delta']:.5f} ± {hold['se']:.5f}", ("holdout NDCG,", "the same comparison")),
        (f"+{pub['delta']:.5f}", ("public NDCG, the first file of", "8 full-view students against",
                                  "the best file before")),
    ]
    for k, (number, rows) in enumerate(tiles):
        x = X1 + k * (w_tile + 0.15)
        style.box(ax, x, yt, w_tile, ht, kind="plain")
        ax.text(x + 0.14, yt + ht - 0.27, number, fontsize=14, fontweight="bold", color=style.INK, va="center",
                zorder=3)
        for j, text in enumerate(rows):
            ax.text(x + 0.14, yt + ht - 0.56 - j * style.LINE, text, fontsize=style.FS_SMALL, color=style.INK2,
                    va="center", zorder=3)
    ax.text(X1, yt - 0.22, f"Student comparison: {val['conditions']}, teacher of {n_small}. Public: "
                           f"{pub['conditions']}.", fontsize=style.FS_SMALL, color=style.INK2, va="center")

    style.key(ax, X1, yt - 0.56, [("blue", "rankers trained on clicks"), ("data", "data"),
                                  ("orange", "target and student")])
    style.save(fig, NAME, texts())


if __name__ == "__main__":
    make()
