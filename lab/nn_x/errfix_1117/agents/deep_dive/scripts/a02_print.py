"""Print feeds in full (top 10 + positives, all columns) for the lenses; output to text files."""
import sys
from util import *

OUT = "<repo>/lab/nn_x/errfix_1117/agents/deep_dive/scripts/"
df = load()
fl = pl.read_parquet(OUT + "feedlevel.parquet")
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
which = sys.argv[1]

COLS1 = ["rank", "label", "score", "s_ft", "s_to", "gap_next", "cat", "cat_rank_in_feed", "cat_n_in_feed",
         "f1", "f2", "f2_home", "f3", "f4", "f5", "xp_imp_c50", "it_hist_all_luo", "it_hist_et1_luo",
         "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo", "seq_cv_invd20", "pre2_fe_z", "pre2_new_z"]
COLS2 = ["rank", "label", "ucos", "hmax", "hmax_idx", "hmax_type", "hmax_src", "h1", "h5", "h10", "h32",
         "hmean5", "n_hdup95", "c_contact", "c_order", "c_fav", "c_lead", "c_vfeed", "c_vsearch",
         "cat_views", "cat_actions", "cat_newest_idx", "cat_user_rank", "same_cat_newest", "cat_hist_share",
         "cand_maxcos", "n_cand_dup95", "item_id"]
SHORT = {"cat_rank_in_feed": "crk", "cat_n_in_feed": "cn", "f2_home": "hm", "xp_imp_c50": "xp",
         "it_hist_all_luo": "ihA", "it_hist_et1_luo": "ih1", "it_hist_et2_luo": "ih2", "it_hist_et3_luo": "ih3",
         "it_hist_et4_luo": "ih4", "seq_cv_invd20": "cv", "pre2_fe_z": "pfe", "pre2_new_z": "pnw",
         "gap_next": "gap", "hmax_idx": "hix", "hmax_type": "hty", "hmax_src": "hsr", "hmean5": "hm5",
         "n_hdup95": "hd95", "c_contact": "cCon", "c_order": "cOrd", "c_fav": "cFav", "c_lead": "cLd",
         "c_vfeed": "cVf", "c_vsearch": "cVs", "cat_views": "cV", "cat_actions": "cA", "cat_newest_idx": "cNi",
         "cat_user_rank": "cUr", "same_cat_newest": "scn", "cat_hist_share": "cSh", "cand_maxcos": "cmx",
         "n_cand_dup95": "cd95", "label": "y", "score": "sc"}


def fmt(v):
    if v is None:
        return "."
    if isinstance(v, bool):
        return "T" if v else "-"
    if isinstance(v, float):
        if v != v:
            return "nan"
        return f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
    return str(v)


def table(sub, cols):
    hdr = [SHORT.get(c, c) for c in cols]
    rows = [[fmt(v) for v in r] for r in sub.select(cols).iter_rows()]
    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(hdr)]
    lines = [" ".join(h.rjust(w[i]) for i, h in enumerate(hdr))]
    for r in rows:
        lines.append(" ".join(x.rjust(w[i]) for i, x in enumerate(r)))
    return "\n".join(lines)


def show(fidx):
    f = df.filter(pl.col("fidx") == fidx).sort("rank")
    r0 = f.row(0, named=True)
    fr = fl.filter(pl.col("fidx") == fidx).row(0, named=True)
    head = (f"=== feed {fidx} user {r0['user_id']} M={r0['M']} H={r0['H']} guest={r0['guest']} uvec0={r0['uvec_missing']} "
            f"hour={r0['hour']} wd={r0['weekday']} newest(type={r0['newest_type']},src={r0['newest_src']},cat={r0['newest_cat']}) "
            f"hcats32={r0['n_hist_cats32']} feedcats={r0['n_feed_cats']} npos={fr['npos']} best={fr['best_pos_rank']} ndcg={fr['ndcg']:.3f}")
    # category composition of feed
    cc = f.group_by("cat").agg(pl.len(), pl.col("label").sum(), pl.col("rank").min().alias("minrk"),
                               pl.col("cat_user_rank").first(), pl.col("cat_views").first(),
                               pl.col("cat_actions").first(), pl.col("cat_newest_idx").first()).sort("len", descending=True)
    ccs = " ".join(f"{r[0]}:{r[1]}/{r[2]}/r{r[3]}/u{r[4]}/v{r[5]}a{r[6]}n{r[7]}" for r in cc.iter_rows())
    sub = f if f.height <= 40 else pl.concat([f.head(10), f.filter((pl.col("label") == 1) & (pl.col("rank") > 10))])
    return "\n".join([head, "cats (cat:n/pos/minrank/userrank/views/actions/newestidx): " + ccs,
                      table(sub, COLS1), table(sub, COLS2), ""])


if which == "a":
    qa = fl["gap12"].quantile(0.75)
    sel = fl.filter((pl.col("lab_r1") == 0) & (pl.col("gap12") >= qa) & (pl.col("best_pos_rank") >= 30))["fidx"].to_numpy()
elif which == "b":
    sel = df.filter((pl.col("M") < 120) & (pl.col("label") == 1) & (pl.col("rank") >= 50))["fidx"].unique().to_numpy()
elif which == "s":
    sel = fl.filter((pl.col("M") < 40) & (pl.col("best_pos_rank") >= 8))["fidx"].to_numpy()
elif which == "a2":
    qa = fl["gap12"].quantile(0.75)
    sel = fl.filter((pl.col("lab_r1") == 0) & (pl.col("gap12") >= qa) & (pl.col("best_pos_rank") >= 30))["fidx"].to_numpy()
elif which == "g":
    sel = fl.filter(pl.col("best_pos_rank") <= 3)["fidx"].to_numpy()
n = int(sys.argv[3]) if len(sys.argv) > 3 else 25
sel = rng.choice(np.sort(sel), size=min(n, len(sel)), replace=False)
with open(OUT + f"feeds_{which}_{sys.argv[2] if len(sys.argv) > 2 else 1}.txt", "w") as fh:
    for x in sel:
        fh.write(show(int(x)) + "\n")
print("wrote", len(sel))
