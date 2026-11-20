"""T5: the member-equal PER-FEED z-mean at small M. For M=5..10 feeds a per-feed z-score is computed from 5-10 values,
so each member's within-feed scale is noisy and the pool weights members by 1/sd_feed. Alternatives evaluated by M
bucket on val (paired vs the current G3 construction): (a) global z per member (one mean/sd over all candidates),
(b) rank mean, (c) raw-logit mean (TO logits share a loss scale; FT logits too).
Also: exact ties in the G3 val / test pool arrays (pessimistic metric), and the FT/TO branch blend by M bucket (descriptive)."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "12")
import numpy as np
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
P = "<repo>/preds"
fam = json.load(open("<repo>/nn_x/stack/hfit/cache/families.json"))
members = fam["F10"] + fam["F7x"] + fam["F7x70"]
f, M, off = feed_meta("val"); M = M.astype(np.int64); y = np.load(FLAT / "val/cand_target.npy")
fid = np.repeat(np.arange(len(M)), M)
out = []
def say(s): print(s, flush=True); out.append(s)

def feed_z(s):
    s = s.astype(np.float64)
    mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-6
    return (s - mu[fid]) / sd[fid]
def feed_rank(s):
    o = np.lexsort((s, fid)); r = np.empty(len(s)); r[o] = np.arange(len(s)) - np.repeat(off, M)
    return r / np.maximum(M[fid] - 1, 1)
def glob_z(s):
    s = s.astype(np.float64); return (s - s.mean()) / s.std()

ft = [np.load(f"{P}/nn_{v}_n2cpv3_val_oof.npy").astype(np.float64) for v in members]
to = [np.load(f"{P}/nn_{v}_trainonly_val.npy").astype(np.float64) for v in members]
ref_ft, ref_to = np.load(f"{P}/ens_G3_ft_val.npy"), np.load(f"{P}/ens_G3_to_val.npy")
cur_ft = np.mean([feed_z(s) for s in ft], 0); cur_to = np.mean([feed_z(s) for s in to], 0)
say(f"rebuild check: corr(ft branch, ens_G3_ft_val) {np.corrcoef(cur_ft, ref_ft)[0, 1]:.6f}; corr(to branch, ens_G3_to_val) {np.corrcoef(cur_to, ref_to)[0, 1]:.6f}")
cur = 0.5 * feed_z(cur_ft) + 0.5 * feed_z(cur_to)
nd_cur = ndcg_per_feed(cur, y, off, M)
say(f"G3 rebuilt val {nd_cur.mean():.5f}")
# ties
for name, arr in [("val pool", cur), ("ens_G3 test (0.5z ft+0.5z to) ", None)]:
    if arr is None:
        ft_t, to_t = np.load(f"{P}/ens_G3_ft_test.npy"), np.load(f"{P}/ens_G3_to_test.npy")
        ftt, Mt, offt = feed_meta("test"); Mt = Mt.astype(np.int64); fidt = np.repeat(np.arange(len(Mt)), Mt)
        def fz_t(s):
            s = s.astype(np.float64); mu = np.bincount(fidt, s, len(Mt)) / Mt
            sd = np.sqrt(np.maximum(np.bincount(fidt, s * s, len(Mt)) / Mt - mu ** 2, 0)) + 1e-6
            return (s - mu[fidt]) / sd[fidt]
        arr = (0.5 * fz_t(ft_t) + 0.5 * fz_t(to_t)).astype(np.float32); fi = fidt
    else:
        arr = arr.astype(np.float32); fi = fid
    o = np.lexsort((arr, fi)); a = arr[o]; ties = (a[1:] == a[:-1]) & (fi[o][1:] == fi[o][:-1])
    say(f"exact ties (float32) in {name}: {ties.sum()} pairs")
variants = {"global_z": (lambda s: glob_z(s)), "rank": feed_rank, "raw": (lambda s: s.astype(np.float64))}
buckets = [(5, 10), (10, 20), (20, 50), (50, 100), (100, 301)]
say("variant | " + " | ".join(f"M[{a},{b}) delta+-SE" for a, b in buckets) + " | all")
for name, fn in variants.items():
    vft = np.mean([fn(s) for s in ft], 0); vto = np.mean([fn(s) for s in to], 0)
    v = 0.5 * feed_z(vft) + 0.5 * feed_z(vto)
    nd = ndcg_per_feed(v, y, off, M); d = nd - nd_cur
    cells = []
    for a, b in buckets:
        s = (M >= a) & (M < b); cells.append(f"{d[s].mean():+.5f}+-{d[s].std() / np.sqrt(s.sum()):.5f} (n={s.sum()})")
    say(f"{name:9s} | " + " | ".join(cells) + f" | {d.mean():+.5f}+-{d.std() / np.sqrt(len(d)):.5f}")
# FT vs TO branch by bucket (descriptive, known from INSIGHTS; here for the record with the member-level spread)
nd_ft = ndcg_per_feed(cur_ft, y, off, M); nd_to = ndcg_per_feed(cur_to, y, off, M)
say("branch NDCG by M bucket (ft | to | 0.5/0.5): " + " ; ".join(f"[{a},{b}) {nd_ft[(M >= a) & (M < b)].mean():.4f}|{nd_to[(M >= a) & (M < b)].mean():.4f}|{nd_cur[(M >= a) & (M < b)].mean():.4f}" for a, b in buckets))
# how noisy is the per-feed sd at small M across members: coefficient of variation of member sd within a feed
sds = np.array([np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - (np.bincount(fid, s, len(M)) / M) ** 2, 0)) for s in to])  # [members, feeds]
cv = sds.std(0) / sds.mean(0)
say("within-feed sd spread across TO members (CV of member sd, mean by bucket): " + " ; ".join(f"[{a},{b}) {cv[(M >= a) & (M < b)].mean():.3f}" for a, b in buckets))
open("<repo>/nn_x/audit/weakspots/t5_pool_norm_by_M.txt", "w").write("\n".join(out) + "\n")
