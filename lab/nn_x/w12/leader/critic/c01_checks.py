"""Critic checks for the leader synthesis (2026-11-18). CPU only; writes only under this directory.
(a) board arithmetic (inputs: past_audits/board_update.py, docs/07 family stats);
(b) nats -> NDCG conversion behind "0.009-0.017 nats" (data_channels/logs/t0_harness_v2.log, p4_battery_v2.log);
(c) where the lost-history rows sit in the best file's order (is the channel 'at the wrong end of the list'?),
    plus a no-fit overlay ceiling (ex_lost rows pushed to the top of their feed);
(d) size of the transductive transfer set against the U20 precedent (reports/NEXT_1115/u20wit.md section 1.4).
"""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np
import polars as pl
sys.path.insert(0, "<repo>/lab/nn_x/w12/leader/data_channels")
from dc_lib import Split, FLAT, log  # noqa: E402

# ---------------- (a) board
lead = 0.3982556
b13 = {"r2": 0.395115, "r3": 0.393778, "us": 0.393206}
b18 = {"r2": 0.3972749, "r3": 0.3964761, "r4": 0.3960596, "r5": 0.3958928, "r6": 0.3954462, "r7": 0.3953650, "us": 0.3952092}
above13 = [k for k, v in b13.items() if v > 0.3955 and k != "us"]
above18 = [k for k, v in b18.items() if v > 0.3955 and k != "us"]
print("(a) rivals (not the leader) above 0.3955: 13 Nov", above13, "| 18 Nov", above18, "->", len(above18), "new")
print("    rivals above our 0.3952092 on 18 Nov:", [k for k, v in b18.items() if v > b18['us']])
print("    leader minus r2..r7:", " ".join("%s %.5f" % (k, lead - b18[k]) for k in ("r2", "r3", "r4", "r5", "r6", "r7")))
fam_mean, fam_sd = 0.39480, 0.00021  # docs/07 row 'Reading experimental files of one family' (14 files)
print("    leader minus our family mean %.5f (our best-of-14 optimism %.5f)" % (lead - fam_mean, b18["us"] - fam_mean))
rng = np.random.default_rng(0)
for K in (14, 30, 60, 100):
    print("    E[max]-mean at our own sd 0.00021, K=%d: +%.5f" % (K, rng.standard_normal((200000, K)).max(1).mean() * fam_sd))
for k in ("r2", "r3", "r4"):
    lo, hi = b18[k] - fam_mean - 0.00125, b18[k] - fam_mean - 0.00043
    print("    %s true lead over our family mean if its optimism is +0.00043..+0.00125: %+.5f..%+.5f" % (k, lo, hi))

# ---------------- (b) nats -> NDCG
for name, dn, dl in (("planted 2% positives", 0.01184, 0.03516), ("later-snapshot leak", 0.00452, 0.02480)):
    print("(b) %-22s NDCG per nat %.3f -> +0.003 NDCG needs %.4f nats/click" % (name, dn / dl, 0.003 / (dn / dl)))
print("    lost history 0.00115 nats = %.0f%%-%.0f%% of that; f5 echo 0.00457 = %.0f%%-%.0f%%" %
      (100 * 0.00115 / 0.0165, 100 * 0.00115 / 0.0089, 100 * 0.00457 / 0.0165, 100 * 0.00457 / 0.0089))

# ---------------- (c) lost-history rows in the best file's order
v = Split("val")
trf = pl.read_parquet(f"{FLAT}/train/feeds.parquet", columns=["user_id"])
tr_u = trf["user_id"].to_numpy()
users = np.setdiff1d(v.user, [0])
sel = np.isin(tr_u, users)
hf = np.load(f"{FLAT}/train/hist_fidx.npy", mmap_mode="r")
rows = np.flatnonzero(sel[np.asarray(hf)])
H = pl.DataFrame({"u": tr_u[np.asarray(hf)[rows]],
                  "item": np.asarray(np.load(f"{FLAT}/train/hist_item_id.npy", mmap_mode="r")[rows])}).unique()
cur = pl.DataFrame({"g": v.hfid, "item": v.hitem}).unique().with_columns(pl.lit(True).alias("incur"))
cand = pl.DataFrame({"r": np.arange(v.N), "g": v.fid, "item": v.item, "u": v.user[v.fid]})
L = (cand.filter(pl.col("u") != 0).join(H.with_columns(pl.lit(True).alias("inlost")), on=["u", "item"], how="inner")
     .join(cur, on=["g", "item"], how="left").filter(pl.col("incur").is_null()))
ex = np.zeros(v.N, bool); ex[L["r"].to_numpy()] = True
y = v.y
log("ex_lost rows", ex.sum(), "positive rate %.4f" % y[ex].mean(), "(p5 log: 635 rows, 0.0378)")
s = v.best()
# relative rank within feed: 0 = top, 1 = bottom
order = np.lexsort((-s, v.fid))
rank = np.empty(v.N, np.int64); rank[order] = np.arange(v.N) - np.repeat(v.off, v.M)
rel = rank / np.maximum(np.repeat(v.M, v.M) - 1, 1)
print("(c) mean relative rank (0 top, 1 bottom): all rows %.3f | all positives %.3f | ex_lost rows %.3f | ex_lost positives %.3f (n %d)"
      % (rel.mean(), rel[y > 0].mean(), rel[ex].mean(), rel[ex & (y > 0)].mean(), int((ex & (y > 0)).sum())))
print("    ex_lost positives already ranked first: %d of %d; in the top 10%% of their feed: %d"
      % (int(((rank == 0) & ex & (y > 0)).sum()), int((ex & (y > 0)).sum()), int(((rel <= 0.1) & ex & (y > 0)).sum())))
base = v.ndcg(s)
s2 = s + 100.0 * ex  # overlay: lost-history candidates to the top, original order among them
d = v.ndcg(s2) - base
print("    overlay (no fit) ex_lost to the top: dNDCG %+.5f +- %.5f over all val feeds; feeds touched %d"
      % (d.mean(), d.std() / np.sqrt(len(d)), int((np.bincount(v.fid, weights=ex, minlength=v.n) > 0).sum())))

# ---------------- (d) transfer-set size against the U20 precedent
u20_unlab = 624752          # u20wit.md 1.4: fold-0 feeds given teacher-only targets
vt = 14423 + 28595
print("(d) val+test feeds %d = %.1f%% of the U20 unlabelled fold; one cross-fitted half ~%d = %.1f%%"
      % (vt, 100 * vt / u20_unlab, vt // 2, 100 * vt / 2 / u20_unlab))
print("    linear scaling of the U20 4-seed pool read (+0.00069 val / -0.00017 holdout) to one half: %+.5f / %+.5f"
      % (0.00069 * vt / 2 / u20_unlab, -0.00017 * vt / 2 / u20_unlab))
print("    at 13x value per in-period feed (top1 h3 GBDT): %+.5f / %+.5f"
      % (13 * 0.00069 * vt / 2 / u20_unlab, -13 * 0.00017 * vt / 2 / u20_unlab))
log("done")
