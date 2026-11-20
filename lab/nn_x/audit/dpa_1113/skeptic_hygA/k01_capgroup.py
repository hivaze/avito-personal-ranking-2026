"""Skeptic check of P1 HYG-A mechanism (1): does the ordering destroyed by the +-10 attach_xt cap carry
information that the G3 pool does not already have?

For each mass column c in {ctx_hn_soft05, ctx_hs_k128, ctx_hs_e1, ctx_hs_t10}:
  capped group of a feed = rows with raw c >= 10 (the trainer sees exactly 10.0 for all of them).
  (a) pairwise accuracy among (pos, neg) pairs INSIDE the capped group: raw c vs G3 score (ties 0.5).
  (b) NDCG delta if the capped rows keep the set of G3 positions they occupy but are re-ordered by raw c
      (upper bound of "restore the order the cap destroyed" if a model trusted raw c fully inside the group).
  (c) oracle: same positions, positives first (absolute ceiling of any within-capped-group re-ordering).
Val: G3 submission score z(0.5 z(FT) + 0.5 z(TO)).  Holdout: member-equal z-mean of the 20 G3 TO members.
CPU only; reads project files, writes nothing outside this directory.
"""
import os, sys
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, numba as nb
from pathlib import Path
ROOT = Path("<repo>")
sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed
from common import feed_meta
PIPE = Path("/tmp/scratchpad/dpa/pipeline/out")
MASS = ["ctx_hn_soft05", "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_t10"]


@nb.njit(parallel=True, cache=False)
def feed_z(x, off, M):
    out = np.zeros(len(x))
    for i in nb.prange(len(M)):
        o = off[i]; m = M[i]; s = 0.0
        for k in range(m):
            s += x[o + k]
        mu = s / m; v = 0.0
        for k in range(m):
            v += (x[o + k] - mu) ** 2
        sd = np.sqrt(v / m)
        for k in range(m):
            out[o + k] = (x[o + k] - mu) / (sd + 1e-9) if sd > 1e-12 else 0.0
    return out


@nb.njit(cache=False)
def pair_acc(raw, s, y, off, M, cap):
    """pairs (pos, neg) inside the capped group: counts where raw orders correctly, where s does, both, total."""
    n_raw = 0.0; n_s = 0.0; n_tot = 0.0; n_rawtie = 0.0; nfeeds = 0
    for i in range(len(M)):
        o = off[i]; m = M[i]; has = False
        for p in range(m):
            if y[o + p] <= 0 or raw[o + p] < cap:
                continue
            for q in range(m):
                if y[o + q] > 0 or raw[o + q] < cap:
                    continue
                has = True
                n_tot += 1
                if raw[o + p] > raw[o + q]:
                    n_raw += 1
                elif raw[o + p] == raw[o + q]:
                    n_raw += 0.5; n_rawtie += 1
                if s[o + p] > s[o + q]:
                    n_s += 1
                elif s[o + p] == s[o + q]:
                    n_s += 0.5
        if has:
            nfeeds += 1
    return n_raw, n_s, n_tot, n_rawtie, nfeeds


def reorder(s, key, grp, off, M):
    """Rows in grp keep the multiset of scores s they hold within their feed, re-assigned in descending key order."""
    out = s.copy()
    fid = np.repeat(np.arange(len(M)), M)
    idx = np.flatnonzero(grp)
    f = fid[idx]
    # process feed by feed (only feeds with >= 2 group rows matter)
    cnt = np.bincount(f, minlength=len(M))
    starts = np.concatenate([[0], np.cumsum(cnt)])
    for i in np.flatnonzero(cnt >= 2):
        r = idx[starts[i]:starts[i + 1]]
        sv = np.sort(s[r])[::-1]
        order = np.argsort(-key[r], kind="stable")
        out[r[order]] = sv
    return out


def run(tag, s, y, off, M, R):
    base = ndcg_per_feed(s, y, off, M)
    print(f"== {tag}: base NDCG {base.mean():.5f}  feeds {len(M)}  rows {len(y)}")
    for c, raw in R.items():
        grp = raw >= 10.0
        fid = np.repeat(np.arange(len(M)), M)
        g2 = np.bincount(fid[grp], minlength=len(M)) >= 2
        nr, ns, nt, ntie, nf = pair_acc(raw, s, y, off, M, 10.0)
        d_raw = ndcg_per_feed(reorder(s, raw, grp, off, M), y, off, M) - base
        d_orc = ndcg_per_feed(reorder(s, y + 1e-9 * s, grp, off, M), y, off, M) - base
        d_rev = ndcg_per_feed(reorder(s, -raw, grp, off, M), y, off, M) - base
        se = d_raw.std() / np.sqrt(len(d_raw))
        print(f"  {c:14s} rows>=10 {grp.mean():.4f} feeds>=2 {g2.mean():.3f} | mixed-label pairs {int(nt)} in {nf} feeds "
              f"({nf/len(M):.4f}) | pair acc raw {nr/max(nt,1):.3f} (raw ties {ntie/max(nt,1):.3f}) vs pool {ns/max(nt,1):.3f} | "
              f"NDCG re-order by raw {d_raw.mean():+.5f} (SE {se:.5f}), by -raw {d_rev.mean():+.5f}, oracle {d_orc.mean():+.5f}")


# ---------------- val
f, Mv, offv = feed_meta("val")
yv = np.load(ROOT / "data/flat/val/cand_target.npy").astype(np.float64)
b = feed_z(0.5 * feed_z(np.load(ROOT / "preds/ens_G3_ft_val.npy").astype(np.float64), offv, Mv)
           + 0.5 * feed_z(np.load(ROOT / "preds/ens_G3_to_val.npy").astype(np.float64), offv, Mv), offv, Mv)
Rv = {c: np.nan_to_num(np.load(ROOT / f"nn/feat_cache/val/{c}.npy").astype(np.float64)) for c in MASS}
run("val G3", b, yv, offv, Mv, Rv)
# single members (the proposal's evidence is member-level)
for mem in ["r6_cd_m02_r101", "v23_gru_pre2s_s501"]:
    sm = np.load(ROOT / f"preds/nn_{mem}_trainonly_val.npy").astype(np.float64)
    run(f"val member {mem} (TO)", sm, yv, offv, Mv, {"ctx_hn_soft05": Rv["ctx_hn_soft05"], "ctx_hs_e1": Rv["ctx_hs_e1"]})

# ---------------- holdout
import torch
CK = torch.load(ROOT / "nn_x/r6/ckpt/r6_cd_m02_r101.pt", map_location="cpu", weights_only=False)
XT = CK["cfg"]["xt_names"]
hm = np.load(PIPE / "hold_meta.npz")
Mh, offh = hm["M"], hm["off"]
yh = np.load(ROOT / "preds/holdout/holdout_y.npy").astype(np.float64)
hmeta = np.load(ROOT / "preds/holdout/holdout_meta.npy")
assert (hmeta[0] == Mh).all() and (hmeta[1] == offh).all()
MEM = [f"v23_gru_pre2s_s50{i}" for i in range(1, 7)] + [f"r6_cd_m02_r10{i}" for i in range(1, 8)] + [f"r6_cd_m02_p70k_r10{i}" for i in range(1, 8)]
bh = feed_z(np.mean([feed_z(np.load(ROOT / f"preds/holdout/nn_{m}_holdout.npy").astype(np.float64), offh, Mh) for m in MEM], 0), offh, Mh)
Rh_all = np.load(PIPE / "hold_raw.npy", mmap_mode="r")
print("hold_raw shape", Rh_all.shape, "xt names", len(XT))
Rh = {c: np.nan_to_num(np.asarray(Rh_all[:, XT.index(c)]).astype(np.float64)) for c in MASS}
run("holdout G3-20 TO", bh, yh, offh, Mh, Rh)
