"""architecture: the ranker of the final members, with the widths read from the member checkpoints."""

from __future__ import annotations

import json

from matplotlib.patches import FancyBboxPatch

import style

NAME = "architecture"
TITLE = ("The ranker: five attention blocks of width 192 over the user's history\n"
         "and the candidates of the feed, 2.1 million parameters")
CAPTION = (
    "The ranker used by every final member. Each candidate token is built from the item vector and 104 dense "
    "columns, picks up values from its 16 nearest history items, and then passes twice through a cross-attention "
    "block over the encoded user and history and a self-attention block over the other candidates of the feed, which the lab calls set attention."
)
ALT = (
    "Box-and-arrow diagram in two columns. Left: the user vector and up to 256 history events become a user token "
    "and history tokens of width 192, encoded by one self-attention block. Right: up to 300 candidates per feed "
    "become a dense block of 104 columns and a candidate token of width 192; a retrieval step adds values from "
    "the 16 nearest history items; two pairs of blocks follow, each a cross-attention block and a self-attention block over the candidates; a score head "
    "gives one score per candidate and an auxiliary head, used in training only, predicts the display order."
)
SOURCE = "Data: docs/figures/data/model.json, read from the model code and the checkpoints of the final members."

# columns (inches)
XL, WU, WH = style.MARGIN, 1.32, 2.36  # left column: user sub-column, history sub-column
XH = XL + WU + 0.12
XLR = XH + WH  # right edge of the left column
XR, WR = 4.5, 4.2  # right column
LANE_RETR, LANE_CTX = XLR + 0.3, XLR + 0.14  # vertical lanes in the gutter
H3, H4, H2, HP = 0.72, 0.9, 0.53, 0.34  # box heights for 3, 4 and 2 lines, and for an output pill
GAP = 0.2


def chain(*dims: int) -> str:
    return "→".join(str(v) for v in dims)


def texts() -> style.Texts:
    return style.Texts(TITLE, CAPTION, ALT, SOURCE)


def make() -> None:
    m = json.loads((style.DATA_DIR / "model.json").read_text())
    d, br = m["width"], m["dense_breakdown"]
    as_values = br["inputs"] - br["inputs_kept_only_as_rank_z"]
    assert br["computed_in_model"] + br["user0_flags"] + as_values + br["rank_z_columns"] == m["dense_columns"]
    emb = m["embedding_dim"]

    fig = style.figure(8.65)
    head = style.header(fig, TITLE)
    ax = style.canvas(fig)
    y = fig.get_size_inches()[1] - head - 0.22

    # ---- row A: inputs
    ya = y - H3
    style.box(ax, XL, ya, WU, H3, "User vector u", (f"{emb}-d",), kind="data")
    style.box(ax, XH, ya, WH, H3, "History", (f"up to {m['history_events']} newest events: item",
                                              "vector, type, source, category"), kind="data")
    style.box(ax, XR, ya, WR, H3, f"Candidates: {m['min_candidates']} to {m['max_candidates']} per feed",
              (f"item vector e ({emb}-d), category, f1 to f5,", f"{m['precomputed_inputs']} pre-computed inputs"),
              kind="data")

    # ---- row B: tokens of the context, dense block
    yb = ya - GAP - H4
    style.box(ax, XL, yb, WU, H4, "User token", ("Linear", chain(emb, d)))
    style.box(ax, XH, yb, WH, H4, "History tokens", (f"Linear {chain(emb, d)} + type,", "source, category, recency",
                                                     "embeddings; LayerNorm"))
    style.box(ax, XR, yb, WR, H4, f"Dense block X: {m['dense_columns']} columns",
              (f"{br['computed_in_model']} computed in the model, {br['user0_flags']} user-0 columns,",
               f"{as_values} inputs, {br['rank_z_columns']} rank / z-score columns;",
               f"standardised, clamped to ±{m['dense_clamp']}"))

    # ---- row C: history encoder, candidate token
    yc = yb - GAP - H3
    style.box(ax, XL, yc, XLR - XL, H3, "History encoder",
              (f"{m['history_encoder_blocks']} self-attention block over", "[user token, history tokens]"),
              kind="orange")
    style.box(ax, XR, yc, WR, H3, "Candidate token",
              (f"Linear {chain(m['candidate_projection_in'], d)} on [e, e·u] + category embedding",
               f"+ MLP {chain(*m['dense_mlp'])} on X; LayerNorm; + domain"))

    # ---- row D: retrieval
    yd = yc - GAP - H3
    r = m["retrieval"]
    style.box(ax, XR, yd, WR, H3, "Retrieval over the history",
              (f"{r['top_k']} nearest history items (cosine); keys {chain(emb, r['key_dim'])};",
               f"values MLP {chain(*r['value_mlp'])}, added to the token"))

    # ---- row E: the two cross + set pairs
    pad = 0.11
    yf_top = yd - GAP
    y_cross = yf_top - pad - H2
    y_set = y_cross - 0.16 - H2
    yf = y_set - pad
    ax.add_patch(FancyBboxPatch((XR - 0.0 + 0.05, yf + 0.05), WR - 0.1, yf_top - yf - 0.1,
                                boxstyle="round,pad=0.05,rounding_size=0.08", facecolor=style.SURFACE,
                                edgecolor=style.MUTED, linewidth=1.0, zorder=0.5))
    xi, wi = XR + pad, WR - 2 * pad - 0.55
    style.box(ax, xi, y_cross, wi, H2, "Cross-attention block", ("candidate → [user, history], cosine bias",),
              kind="orange")
    style.box(ax, xi, y_set, wi, H2, "Self-attention block", ("over the feed's candidates, no positions",),
              kind="orange")
    ax.text(XR + WR - 0.33, (yf + yf_top) / 2, f"× {m['cross_set_pairs']}", fontsize=13, fontweight="bold",
            color=style.INK, ha="center", va="center", zorder=3)

    # ---- row F: heads and outputs
    yh = yf - GAP - H3
    w_aux = 1.62
    x_score, w_score = XR + w_aux + 0.14, WR - w_aux - 0.14
    style.box(ax, XR, yh, w_aux, H3, "Auxiliary head", ("same MLP,", "training only"), dashed=True)
    style.box(ax, x_score, yh, w_score, H3, "Score head (32-bit)",
              (f"MLP {chain(*m['head_mlp'])} on token", f"+ domain; Linear {chain(*m['linear_on_dense'])} on X"))
    yo = yh - GAP - HP
    style.box(ax, XR, yo, w_aux, HP, lines=("display order",), kind="data", pill=True)
    style.box(ax, x_score, yo, w_score, HP, lines=("score per candidate",), kind="data", pill=True)

    # ---- arrows
    def down(x: float, y_from: float, y_to: float) -> None:
        style.arrow(ax, [(x, y_from), (x, y_to)])

    xu, xh, xr = XL + WU / 2, XH + WH / 2, XR + WR / 2
    down(xu, ya, yb + H4)
    down(xh, ya, yb + H4)
    down(xu, yb, yc + H3)
    down(xh, yb, yc + H3)
    down(xr, ya, yb + H4)
    down(xr, yb, yc + H3)
    down(xr, yc, yd + H3)
    x_in = xi + wi / 2
    down(x_in, yd, y_cross + H2)
    down(x_in, y_cross, y_set + H2)
    style.arrow(ax, [(x_in, y_set), (x_in, yf - GAP / 2), (XR + w_aux / 2, yf - GAP / 2), (XR + w_aux / 2, yh + H3)])
    style.arrow(ax, [(x_in, yf - GAP / 2), (x_score + w_score / 2, yf - GAP / 2), (x_score + w_score / 2, yh + H3)])
    down(XR + w_aux / 2, yh, yo + HP)
    down(x_score + w_score / 2, yh, yo + HP)
    # raw history item vectors to the retrieval; encoded context to the cross-attention block
    style.arrow(ax, [(XLR, ya + H3 / 2), (LANE_RETR, ya + H3 / 2), (LANE_RETR, yd + H3 / 2), (XR, yd + H3 / 2)])
    style.arrow(ax, [(XLR, yc + H3 / 2), (LANE_CTX, yc + H3 / 2), (LANE_CTX, y_cross + H2 / 2),
                     (xi, y_cross + H2 / 2)])
    # the dense block also feeds the linear part of the score head
    lane = XR + WR + 0.13
    style.arrow(ax, [(XR + WR, yb + H4 / 2), (lane, yb + H4 / 2), (lane, yh + H3 / 2), (XR + WR, yh + H3 / 2)])

    # ---- notes in the free part of the left column
    p = m["parameters"]
    notes = [
        ("Every attention block", True),
        (f"width {d}, {m['heads']} heads, feed-forward {m['feed_forward_factor'] * d},", False),
        (f"dropout {m['dropout']}, pre-LayerNorm, gated output,", False),
        ("separate key and value matrices;", False),
        (f"about {round(m['parameters_per_attention_block'], -2):,} parameters each.", False),
        ("", False),
        (f"Whole ranker: {p['full_view']:,} parameters", True),
        (f"Popularity-light view: {m['popularity_light']['precomputed_inputs']} inputs,", False),
        (f"{m['popularity_light']['dense_columns']} columns, {p['popularity_light_width_192']:,} parameters.", False),
        ("", False),
        ("Domain embedding", True),
        (f"{m['domain_rows']} rows. Training mixes the first two by the", False),
        ("exploration probability of the feed; the", False),
        ("fine-tune uses the third.", False),
        ("", False),
        ("Fine-tune only", True),
        (f"MLP {chain(*m['finetune_extras']['mlp'])} on {m['finetune_extras']['inputs']} extra inputs,", False),
        ("added to the candidate token, zero-initialised.", False),
    ]
    yn = yc - 0.34
    for text, bold in notes:
        if text:
            ax.text(XL, yn, text, fontsize=style.FS_TEXT if bold else style.FS_SMALL,
                    fontweight="bold" if bold else "normal", color=style.INK if bold else style.INK2, va="center")
        yn -= style.LINE if text else style.LINE * 0.55

    style.key(ax, XL, 0.62, [("data", "input or output"), ("blue", "projection, MLP or head"),
                            ("orange", "attention block")])
    style.save(fig, NAME, texts())


if __name__ == "__main__":
    make()
