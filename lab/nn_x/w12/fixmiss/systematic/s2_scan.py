"""Step 2: slice scan. For each slice: actual / expected positives on half A and B separately (z from the re-draw
null of that slice's count), share of the NDCG loss vs the share under re-drawn labels, loss per positive.

SYSTEMATIC = same-sign |z| >= 3 on both halves AND (loss share > re-draw share at z >= 2, for under-predicted
slices). Over-predicted slices (A/E < 1 on both halves at |z| >= 3) are flagged OVER: their cost falls on positives
outside the slice, so the loss-share test does not apply to them.

usage: s2_scan.py [final|ft|to]   -> data/scan_<name>.parquet and a printed table
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import *  # noqa: F401,F403

name = sys.argv[1] if len(sys.argv) > 1 else "final"
F, M, off = layout()
half = np.load(HERE / "data/half.npy")
fid = np.repeat(np.arange(len(M)), M)
S = np.load(HERE / f"data/setup_{name}.npz")
E, L, rows, loss = S["E"], S["L"], S["rows"].astype(np.int64), S["loss"].astype(np.float64)
R = rows.shape[0]
D = load_table()
y = D["label"].to_numpy().astype(np.float64)
hr = half[fid]
hrows = hr[rows]                       # half of each re-drawn positive
Ltot_rd = loss.sum(1)


def col(c):
    return D[c].to_numpy().astype(np.float64)


# ---------- derived columns ----------
cni = col("cat_newest_idx")
nov_cls = np.select([cni < 0, cni == 0, cni < 8, cni < 32], ["never", "newest", "h1-7", "h8-31"], "h32+")
nov8 = (cni < 0) | (cni >= 8)
Mr = M[fid].astype(np.float64)
feed_new_share = np.bincount(fid, nov8, len(M)) / M
past_rate = np.where(col("past_npos") > 0, col("past_nnew") / np.maximum(col("past_npos"), 1), np.nan)
der = {
    "nov8": nov8, "nov_cls": nov_cls, "cold": col("xp_imp_c50") == 0,
    "cat_share_feed": col("cat_n_in_feed") / Mr, "disagree_ft_to": col("s_ft") - col("s_to"),
    "feed_new_share": feed_new_share[fid], "past_novel_rate": past_rate,
    "item_clock": col("item_id"), "Mrow": Mr,
}


def qbins(x, q=10):
    x = np.asarray(x, np.float64)
    ok = ~np.isnan(x)
    edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, q + 1)[1:-1]))
    b = np.searchsorted(edges, x, side="right").astype(np.int64)
    lab = np.array([f"q{i}" for i in range(len(edges) + 1)] + ["nan"])
    b[~ok] = len(edges) + 1 - 1 + 1 - 1  # placeholder, replaced below
    out = lab[np.minimum(b, len(edges))]
    out = np.where(ok, out, "nan")
    lo = np.r_[-np.inf, edges]; hi = np.r_[edges, np.inf]
    desc = {f"q{i}": f"[{lo[i]:.3g},{hi[i]:.3g})" for i in range(len(edges) + 1)}
    desc["nan"] = "nan"
    return out, desc


slices = []   # (family, label, mask, level)


def add_cont(name_, x, q=10, level="cand"):
    b, desc = qbins(x, q)
    for v in np.unique(b):
        slices.append((name_, f"{v} {desc[v]}", b == v, level))


def add_disc(name_, x, level="cand", min_rows=2000):
    x = np.asarray(x)
    for v in np.unique(x):
        m = x == v
        if m.sum() >= min_rows:
            slices.append((name_, str(v), m, level))


for c in ["ucos", "hmax", "h1", "h5", "h32", "hmean5", "c_contact", "c_fav", "c_order", "c_lead", "c_vfeed",
          "c_vsearch", "cand_maxcos", "f1", "f3", "f4", "f5", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo",
          "it_hist_et4_luo", "xp_imp_c50", "seq_cv_invd20", "pre2_fe_z", "pre2_new_z", "cat_hist_share", "gap_next",
          "gap_top", "hmax_idx"]:
    add_cont(c, col(c))
for c in ["cat_share_feed", "disagree_ft_to", "item_clock"]:
    add_cont(c, der[c])
add_disc("cat", col("cat").astype(int))
add_disc("hmax_type", col("hmax_type").astype(int)); add_disc("hmax_src", col("hmax_src").astype(int))
add_disc("n_hdup95", np.minimum(col("n_hdup95"), 3).astype(int)); add_disc("n_cand_dup95", np.minimum(col("n_cand_dup95"), 3).astype(int))
add_disc("cat_user_rank", np.minimum(col("cat_user_rank"), 6).astype(int)); add_disc("nov_cls", nov_cls); add_disc("nov8", nov8)
add_disc("same_cat_newest", col("same_cat_newest").astype(int)); add_disc("f2_zero", col("f2") == 0)
add_disc("f2_home", col("f2_home").astype(int)); add_disc("cold", der["cold"])
add_disc("cat_actions", np.minimum(col("cat_actions"), 3).astype(int))
rk = col("rank")
add_disc("rank", np.select([rk == 1, rk <= 3, rk <= 10, rk <= 30, rk <= 100], ["1", "2-3", "4-10", "11-30", "31-100"], "101+"))
cr = col("cat_rank_in_feed")
add_disc("cat_rank_in_feed", np.select([cr == 1, cr <= 3, cr <= 10], ["1", "2-3", "4-10"], "11+"))
# feed-level (A/E is 1 by construction; loss share only)
add_disc("M", np.select([Mr <= 30, Mr <= 50, Mr <= 100, Mr <= 200], ["005-030", "031-050", "051-100", "101-200"], "201-300"), "feed")
add_cont("H", col("H"), 5, "feed"); add_disc("guest", col("guest").astype(int), "feed")
add_disc("uvec_missing", col("uvec_missing").astype(int), "feed"); add_disc("weekday", col("weekday").astype(int), "feed")
add_disc("hour4", (col("hour") // 4).astype(int), "feed"); add_disc("newest_type", col("newest_type").astype(int), "feed")
add_disc("newest_src", col("newest_src").astype(int), "feed")
for c in ["h_ent", "h_nov32", "h_nov32r", "h_brk10", "h_topsh", "n_feed_cats", "n_hist_cats32"]:
    add_cont(c, col(c), 5, "feed")
add_cont("feed_new_share", der["feed_new_share"], 5, "feed"); add_cont("past_novel_rate", past_rate, 3, "feed")
# interactions: novelty x explorer signals / feed context; cold / age x feed size
Mb = np.where(Mr < 50, "M<50", np.where(Mr <= 150, "M50-150", "M>150"))
for c, xx in [("h_nov32", col("h_nov32")), ("h_nov32r", col("h_nov32r")), ("h_ent", col("h_ent")),
              ("h_brk10", col("h_brk10")), ("h_topsh", col("h_topsh")), ("n_hist_cats32", col("n_hist_cats32")),
              ("H", col("H")), ("feed_new_share", der["feed_new_share"]), ("past_novel_rate", past_rate)]:
    b, desc = qbins(xx, 5 if c != "past_novel_rate" else 3)
    for v in np.unique(b):
        for nv in (True, False):
            slices.append((f"nov8={int(nv)}x{c}", f"{v} {desc[v]}", (b == v) & (nov8 == nv), "cand"))
for nv in (True, False):
    for mb in np.unique(Mb):
        slices.append((f"nov8={int(nv)}xM", mb, (nov8 == nv) & (Mb == mb), "cand"))
    for g in (0, 1):
        slices.append((f"nov8={int(nv)}xguest", str(g), (nov8 == nv) & (col("guest") == g), "cand"))
    for nt in (1, 2, 3, 4, 10):
        slices.append((f"nov8={int(nv)}xnewest_type", str(nt), (nov8 == nv) & (col("newest_type") == nt), "cand"))
    for ns in range(0, 8):
        slices.append((f"nov8={int(nv)}xnewest_src", str(ns), (nov8 == nv) & (col("newest_src") == ns), "cand"))
for cl in np.unique(nov_cls):
    for mb in np.unique(Mb):
        slices.append(("nov_clsxM", f"{cl}|{mb}", (nov_cls == cl) & (Mb == mb), "cand"))
f4b, f4d = qbins(col("f4"), 5)
for v in np.unique(f4b):
    for mb in np.unique(Mb):
        slices.append(("f4xM", f"{v} {f4d[v]}|{mb}", (f4b == v) & (Mb == mb), "cand"))
for cd in (True, False):
    for mb in np.unique(Mb):
        slices.append(("coldxM", f"{int(cd)}|{mb}", (der["cold"] == cd) & (Mb == mb), "cand"))

# ---------- statistics ----------
Lact_tot = L.sum()
recs = []
for fam, lab, m, level in slices:
    m = np.asarray(m, bool)
    if m.sum() < 500:
        continue
    r = {"family": fam, "slice": lab, "level": level, "rows": int(m.sum()), "feeds": int(np.unique(fid[m]).size)}
    mr = m[rows]                                    # (R, P): re-drawn positive falls in slice
    for h in (0, 1):
        mh = m & (hr == h)
        a = y[mh].sum(); e = E[mh].sum()
        null = (mr & (hrows == h)).sum(1)
        sd = null.std()
        r[f"act_{'AB'[h]}"] = a; r[f"exp_{'AB'[h]}"] = e
        r[f"ae_{'AB'[h]}"] = a / e if e > 0 else np.nan
        r[f"z_{'AB'[h]}"] = (a - e) / sd if sd > 0 else 0.0
    r["ae_all"] = y[m].sum() / E[m].sum()
    la = L[m].sum()
    lsh_rd = (loss * mr).sum(1) / Ltot_rd
    r["loss_share"] = la / Lact_tot; r["loss_share_rd"] = lsh_rd.mean()
    r["z_loss"] = (r["loss_share"] - lsh_rd.mean()) / max(lsh_rd.std(), 1e-12)
    npos_rd = mr.sum(1)
    r["lpp"] = la / max(y[m].sum(), 1)              # loss per positive in the slice
    r["lpp_rd"] = ((loss * mr).sum(1) / np.maximum(npos_rd, 1)).mean()
    recs.append(r)

T = pl.DataFrame(recs)
both = (np.sign(T["z_A"]) == np.sign(T["z_B"])) & (T["z_A"].abs() >= 3) & (T["z_B"].abs() >= 3)
T = T.with_columns(pl.Series("cal_both", both.to_numpy()),
                   pl.Series("systematic", (both & (T["ae_all"] > 1) & (T["z_loss"] >= 2)).to_numpy()),
                   pl.Series("over", (both & (T["ae_all"] < 1)).to_numpy()))
T.write_parquet(HERE / f"data/scan_{name}.parquet")
pl.Config.set_tbl_rows(400); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(40)
show = ["family", "slice", "rows", "act_A", "ae_A", "z_A", "ae_B", "z_B", "loss_share", "loss_share_rd", "z_loss", "lpp", "lpp_rd"]
print(f"[{name}] slices scanned {T.height}; cal_both {int(T['cal_both'].sum())}; SYSTEMATIC {int(T['systematic'].sum())}; OVER {int(T['over'].sum())}")
print("\n=== slices with same-sign |z|>=3 on both halves ===")
print(T.filter(pl.col("cal_both")).sort(pl.col("z_A").abs() + pl.col("z_B").abs(), descending=True).select(show)
      .with_columns(pl.col(pl.Float64).round(4)))
print("\n=== feed-level slices: loss share vs re-draw share (|z_loss| >= 3) ===")
print(T.filter((pl.col("level") == "feed") & (pl.col("z_loss").abs() >= 3)).select(show).with_columns(pl.col(pl.Float64).round(4)))
