"""ensemble: from member predictions to the ranks in the submission file."""

from __future__ import annotations

import style

NAME = "ensemble"


def load() -> tuple[dict, dict]:
    """The mix numbers and the file shown (pick 1)."""
    m = style.read_json("ensemble.json")
    pick1 = next(p for p in style.read_json("final_picks.json")["picks"] if p["pick"] == 1)
    shown = next(f for f in m["files"] if f["id"] == pick1["short_id"])
    assert shown["role"] == "pick 1" and shown["val_final"] == pick1["val_ndcg"]
    return m, shown


def listing(names: list[str]) -> str:
    """'A', 'A and B', 'A, B and C'."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def texts() -> style.Texts:
    m, f = load()
    w = (f"w = {f['w_short']} on feeds of fewer than {m['short_feed_limit']} candidates and {f['w_long']} on the "
         "others")
    others = [g for g in m["files"] if g is not f]
    assert all(g["w_short"] == g["w_long"] for g in others)
    return style.Texts(
        title=("Each final file is a fixed-weight mix of two student pools and two branches;\n"
               f"pick 1 gives the popularity-light pool {f['w_short']} on short feeds and {f['w_long']} on long ones"),
        caption=("How a final file is assembled from member predictions. Scores are z-scored within each feed before "
                 "every average. In each branch the popularity-light pool enters at weight w and the full-view pool "
                 f"at 1 − w, with {w} in pick 1, {f['id']} (w = {others[0]['w_short']} on every feed in "
                 f"{listing([g['id'] for g in others])}). The fine-tuned and train-only branches are mixed "
                 f"{m['weight_finetuned']} / {m['weight_train_only']}, and the file holds the rank within the feed. "
                 f"Validation NDCG is given for {f['id']}."),
        alt=("Box-and-arrow diagram. On the left, two groups, fine-tuned predictions and train-only predictions, "
             f"each with a pool of {f['wide_students']} full-view students and a pool of "
             f"{f['popularity_light_students']} popularity-light students. In each group the two pools are mixed "
             f"with weights 1 − w and w into a branch, {w}: validation NDCG {f['val_finetuned_branch']:.5f} for the "
             f"fine-tuned branch and {f['val_train_only_branch']:.5f} for the train-only branch. The branches are "
             f"mixed {m['weight_finetuned']} and {m['weight_train_only']} into the final score, validation NDCG "
             f"{f['val_final']:.5f}, which is ranked within each feed and written to the submission file of "
             f"{m['submission_rows']:,} rows."),
        source=("Data: docs/figures/data/ensemble.json. Validation: 14,423 feeds; fine-tuned predictions are "
                "out-of-fold."),
    )


XA, WA = style.MARGIN, 2.45
XB, WB = 3.75, 2.2
XC, WC = 6.6, 2.1
H2, H3, H4 = 0.53, 0.72, 0.9


def make() -> None:
    (m, fd), t = load(), texts()
    p2, old1, old2 = (next(g for g in m["files"] if g["role"] == role)
                      for role in ("pick 2", "pick 1 until 18 Nov", "pick 2 until 19 Nov"))
    wf, wt = m["weight_finetuned"], m["weight_train_only"]

    fig = style.figure(7.6 + style.LINE + 0.02)  # six note lines
    head = style.header(fig, t.title)
    ax = style.canvas(fig)
    top = fig.get_size_inches()[1] - head - 0.2

    def group(y_head: float, name: str, detail: str, branch: str, val: float) -> float:
        """One branch: two pools and their mix. Returns the y of the centre of the mix box."""
        ax.text(XA, y_head, name, fontsize=style.FS_TEXT, fontweight="bold", color=style.INK, va="center")
        ax.text(XA, y_head - style.LINE, detail, fontsize=style.FS_SMALL, color=style.INK2, va="center")
        y_wide = y_head - style.LINE - 0.2 - H2
        y_light = y_wide - 0.14 - H2
        style.box(ax, XA, y_wide, WA, H2, f"{fd['wide_students']} full-view students",
                  ("mean of within-feed z-scores",))
        style.box(ax, XA, y_light, WA, H2, f"{fd['popularity_light_students']} popularity-light students",
                  ("mean of within-feed z-scores",), kind="orange")
        yc = (y_wide + H2 + y_light) / 2
        style.box(ax, XB, yc - H4 / 2, WB, H4, branch, ("(1 − w) · z(full-view)", "+ w · z(popularity-light)",
                                                        f"validation NDCG {val:.5f}"), kind="data")
        xm = (XA + WA + XB) / 2
        for y_pool, dy, weight in ((y_wide + H2 / 2, 0.2, "1 − w"), (y_light + H2 / 2, -0.2, "w")):
            style.arrow(ax, [(XA + WA, y_pool), (xm, y_pool), (xm, yc + dy), (XB, yc + dy)])
            ax.text((XA + WA + xm) / 2, y_pool + (0.13 if dy > 0 else -0.13), weight, fontsize=style.FS_SMALL,
                    color=style.INK, ha="center", va="center")
        return yc

    y1 = group(top - 0.1, "Fine-tuned predictions",
               f"validation: out-of-fold; test: mean of {m['finetune_models_per_test_row']} fold models",
               "Fine-tuned branch", fd["val_finetuned_branch"])
    y2 = group(y1 - H2 - 0.14 - 0.55, "Train-only predictions", "the students as trained, no validation labels",
               "Train-only branch", fd["val_train_only_branch"])

    # ---- final mix, rank, file
    ym = (y1 + y2) / 2
    style.box(ax, XC, ym - H4 / 2, WC, H4, "Final score", (f"{wf} · z(fine-tuned)", f"+ {wt} · z(train-only)",
                                                           f"validation NDCG {fd['val_final']:.5f}"), kind="data")
    xm = (XB + WB + XC) / 2
    for yb, dy, weight in ((y1, 0.2, wf), (y2, -0.2, wt)):
        style.arrow(ax, [(XB + WB, yb), (xm, yb), (xm, ym + dy), (XC, ym + dy)])
        ax.text(xm + 0.09, yb + (-0.13 if dy > 0 else 0.13), f"{weight}", fontsize=style.FS_SMALL, color=style.INK,
                ha="left", va="center")
    y_rank = ym - H4 / 2 - 0.26 - H3
    style.box(ax, XC, y_rank, WC, H3, "Rank within feed", ("exact ties broken by", "a fixed random key"), kind="data")
    y_file = y_rank - 0.26 - H2
    style.box(ax, XC, y_file, WC, H2, "Submission file", (f"{m['submission_rows']:,} rows",), kind="data", pill=True)
    style.arrow(ax, [(XC + WC / 2, ym - H4 / 2), (XC + WC / 2, y_rank + H3)])
    style.arrow(ax, [(XC + WC / 2, y_rank), (XC + WC / 2, y_file + H2)])

    # ---- notes
    yn = min(y_file, y2 - H2 - 0.07) - 0.36
    for text in (
        "z(x) = (x − mean) / standard deviation over the candidates of one feed; every mix is z-scored again.",
        f"w = {fd['w_short']} on feeds of fewer than {m['short_feed_limit']} candidates, {fd['w_long']} on the "
        "others: one float32 weight per candidate row.",
        f"{fd['id']} (pick 1), shown here: {fd['wide_students']} + {fd['popularity_light_students']} students "
        f"of width {fd['popularity_light_width']}, trained on {fd['members_train_on']}.",
        f"{p2['id']} (pick 2): {p2['wide_students']} + {p2['popularity_light_students']} students on "
        f"{p2['members_train_on']}, the {p2['wide_students']} trained with pseudo-feeds added; w = {p2['w_short']}.",
        f"{old1['id']}, pick 1 until 18 Nov: the members of pick 1 with w = {old1['w_short']} on every feed; "
        f"validation NDCG {old1['val_final']:.5f}.",
        f"{old2['id']}, pick 2 until 19 Nov: {old2['wide_students']} full-view + {old2['popularity_light_students']} "
        f"popularity-light students of width {old2['popularity_light_width']}; training holds out 30,000 feeds.",
    ):
        ax.text(XA, yn, text, fontsize=style.FS_SMALL, color=style.INK2, va="center")
        yn -= style.LINE + 0.02

    style.key(ax, XA, yn - 0.16, [("blue", "full-view students"), ("orange", "popularity-light students"),
                                  ("data", "mixes and the file")])
    style.save(fig, NAME, t)


if __name__ == "__main__":
    make()
