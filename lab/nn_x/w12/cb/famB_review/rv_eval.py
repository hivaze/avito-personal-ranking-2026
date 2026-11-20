"""famB_review: INDEPENDENT evaluation code (does not import nats.py, pc_common, cbx or any famB script).
Labels / groups from raw flat files, W16 rebuilt from the 16 member prediction files.
  NDCG: binary gain, pessimistic ties (negatives first inside a tie run), feeds without a click -> 1 (project convention), numpy lexsort.
  NATS: per-feed listwise softmax loss (target y / n_pos), Newton fit of 1 and 2 coefficients on the witness, gain = loss([zW]) - loss([zW, zX]).
  MEMBER RULE: NDCG(0.9 zW + 0.1 z(X)) - NDCG(zW), paired per feed, pooled z = (z_val + z_hold) / sqrt(2).
usage: python rv_eval.py            (prints the table of every arm and bundle the verdict rests on, writes out/rv_eval.json)
"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import polars as pl

ROOT = "<repo>"
HERE = f"{ROOT}/nn_x/w12/cb/famB_review"
FLAT = f"{ROOT}/data/flat"
W5P = f"{ROOT}/nn_x/w5/preds"
W16 = [f"w5_e10_kd05w192_s{s}" for s in range(951, 959)] + [f"w5_e10_kd05t25w192_s{s}" for s in range(951, 959)]


def hold_index():
    """own derivation of the 30K holdout feed set (definition read in cbx.train_sets / nn/train.py)."""
    f = pl.read_parquet(f"{FLAT}/train/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64)
    pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    return f, M, hold


class W:
    """a witness: labels y, feed sizes M, offsets off, feed id per row fid, zW (W16 pool z)."""
    pass


def zfeed(x, w):
    x = np.asarray(x, np.float64)
    s1 = np.add.reduceat(x, w.off); mu = s1 / w.M
    d = x - mu[w.fid]
    sd = np.sqrt(np.add.reduceat(d * d, w.off) / w.M) + 1e-9
    return d / sd[w.fid]


def load_witness(name):
    w = W(); w.name = name
    if name == "val":
        f = pl.read_parquet(f"{FLAT}/val/feeds.parquet")
        w.M = f["M"].to_numpy().astype(np.int64)
        w.y = np.load(f"{FLAT}/val/cand_target.npy").astype(np.float64)
    else:
        f, M, hold = hold_index()
        off_all = np.r_[0, np.cumsum(M)[:-1]]
        w.M = M[hold]
        rows = np.concatenate([np.arange(off_all[i], off_all[i] + M[i]) for i in hold])
        w.y = np.asarray(np.load(f"{FLAT}/train/cand_target.npy", mmap_mode="r")[rows], np.float64)
        w.hold_fidx = hold
    w.off = np.r_[0, np.cumsum(w.M)[:-1]].astype(np.int64)
    w.fid = np.repeat(np.arange(len(w.M)), w.M)
    w.n = int(w.M.sum()); assert len(w.y) == w.n
    w.npos = np.bincount(w.fid, w.y, len(w.M))
    acc = np.zeros(w.n)
    for m in W16:
        p = f"{W5P}/nn_{m}_trainonly_val.npy" if name == "val" else f"{W5P}/holdout/nn_{m}_holdout.npy"
        x = np.load(p); assert x.shape == (w.n,), (p, x.shape)
        acc += zfeed(x, w)
    w.zW = zfeed(acc / len(W16), w)
    # idcg table
    disc = 1.0 / np.log2(np.arange(2, w.M.max() + 2))
    w.disc = disc; w.cdisc = np.r_[0, np.cumsum(disc)]
    return w


def ndcg(s, w):
    """per-feed NDCG, pessimistic ties, feeds without a click -> 1."""
    s = np.asarray(s, np.float32).astype(np.float64)
    o = np.lexsort((w.y, -s, w.fid))                # by feed, then score descending, then label ascending (negatives first in a tie)
    rank = np.arange(w.n) - w.off[w.fid]            # the sorted array keeps feeds contiguous, so position - feed offset = rank
    dcg = np.bincount(w.fid, w.y[o] * w.disc[rank], len(w.M))
    idcg = w.cdisc[w.npos.astype(np.int64)]
    out = np.ones(len(w.M)); k = w.npos > 0
    out[k] = dcg[k] / idcg[k]
    return out


def clogit(X, w, iters=50):
    """Newton fit of the within-feed softmax model; returns (beta, per-feed loss over feeds with a click)."""
    keep = w.npos > 0
    rk = keep[w.fid]
    X = np.asarray(X, np.float64)[rk]; y = w.y[rk]
    M = w.M[keep]; fid = np.repeat(np.arange(len(M)), M); off = np.r_[0, np.cumsum(M)[:-1]]
    t = y / w.npos[keep][fid]
    p = X.shape[1]; b = np.zeros(p)

    def loss_grad_hess(b):
        s = X @ b
        mx = np.maximum.reduceat(s, off)
        e = np.exp(s - mx[fid]); den = np.add.reduceat(e, off)
        pr = e / den[fid]
        lse = mx + np.log(den)
        L = lse - np.bincount(fid, t * s, len(M))                 # per-feed loss
        g = ((pr - t)[:, None] * X).sum(0)
        mu = np.stack([np.bincount(fid, pr * X[:, j], len(M)) for j in range(p)], 1)
        H = (X * pr[:, None]).T @ X - mu.T @ mu
        return L, g, H
    for _ in range(iters):
        L, g, H = loss_grad_hess(b)
        step = np.linalg.solve(H, g)
        b = b - step
        if np.abs(step).max() < 1e-10:
            break
    L, g, H = loss_grad_hess(b)
    return b, L, keep


def pse(d):
    d = np.asarray(d, np.float64)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def read_model(xs, w, base=None, sub=None):
    """xs: list of raw score arrays (1 = single arm, more = bundle, member-equal mean of per-feed z).
    sub: optional per-feed boolean mask (subset read, coefficients refitted on the subset like read_m50)."""
    zx = np.mean([zfeed(x, w) for x in xs], 0)
    nW = ndcg(w.zW, w); nX = ndcg(zx, w)
    nmix = ndcg(0.9 * w.zW + 0.1 * zfeed(zx, w), w)
    r = {}
    ww = w
    if sub is not None:
        ww = subset(w, sub); zxs = zx[sub[w.fid]]
    else:
        zxs = zx
    _, l1, k = clogit(ww.zW[:, None], ww)
    b2, l2, _ = clogit(np.stack([ww.zW, zxs], 1), ww)
    d = l1 - l2
    r["nats"], r["nats_se"] = pse(d); r["n_click_feeds"] = int(len(d)); r["beta"] = [float(v) for v in b2]
    r["nats_perfeed"] = d; r["keep"] = k
    m = np.ones(len(w.M), bool) if sub is None else sub
    r["level"], r["level_se"] = pse(nX[m]); r["w16"] = float(nW[m].mean())
    r["mix"], r["mix_se"] = pse((nmix - nW)[m]); r["n_feeds"] = int(m.sum())
    r["nd"] = nX
    # rho: mean within-feed Pearson of z(X) and zW
    zz = zfeed(zx, w)
    r["rho"] = float((np.bincount(w.fid, zz * w.zW, len(w.M)) / w.M)[m].mean())
    return r


def subset(w, fm):
    s = W(); rk = fm[w.fid]
    s.M = w.M[fm]; s.y = w.y[rk]; s.off = np.r_[0, np.cumsum(s.M)[:-1]].astype(np.int64); s.fid = np.repeat(np.arange(len(s.M)), s.M)
    s.n = int(s.M.sum()); s.npos = np.bincount(s.fid, s.y, len(s.M)); s.zW = w.zW[rk]; s.disc = w.disc; s.cdisc = w.cdisc
    return s


FB = f"{ROOT}/nn_x/w12/cb/famB/preds"
B0 = f"{ROOT}/nn_x/w12/cb/infra/out/ts_lr020_s0"
ARMS = {"b0": [B0], "b0_s1": [f"{FB}/b0_s1"], "b3_abc_s0": [f"{FB}/b3_abc_s0"], "b3_abc_s1": [f"{FB}/b3_abc_s1"],
        "b1_mcc4_s0": [f"{FB}/b1_mcc4_s0"], "b1_mcc2_s0": [f"{FB}/b1_mcc2_s0"], "b2_item_s0": [f"{FB}/b2_item_s0"],
        "bundle(b3_abc_s0+b3_abc_s1)": [f"{FB}/b3_abc_s0", f"{FB}/b3_abc_s1"],
        "bundle(b0+b0_s1+b3_abc_s0+b3_abc_s1)": [B0, f"{FB}/b0_s1", f"{FB}/b3_abc_s0", f"{FB}/b3_abc_s1"],
        "bundle(b0+b3_abc_s0)": [B0, f"{FB}/b3_abc_s0"], "bundle(b0+b0_s1)": [B0, f"{FB}/b0_s1"],
        "bundle(b0+b1_mcc4_s0)": [B0, f"{FB}/b1_mcc4_s0"], "bundle(b0+b1_mcc4_s0+b3_abc_s0)": [B0, f"{FB}/b1_mcc4_s0", f"{FB}/b3_abc_s0"]}

if __name__ == "__main__":
    wv, wh = load_witness("val"), load_witness("hold")
    # alignment cross-checks against the files of record (read only)
    yh_ref = np.load(f"{W5P}/holdout/holdout_y.npy"); Mh, oh = np.load(f"{W5P}/holdout/holdout_meta.npy")
    print("holdout labels identical to holdout_y.npy:", bool((yh_ref == wh.y).all()), "| M identical:", bool((Mh == wh.M).all()))
    zc_v = np.load(f"{ROOT}/nn_x/w12/cb/infra/cache/w16_z_val.npy"); zc_h = np.load(f"{ROOT}/nn_x/w12/cb/infra/cache/w16_z_hold.npy")
    print("my W16 z vs infra cache: max abs diff val %.2e hold %.2e" % (np.abs(zc_v - wv.zW).max(), np.abs(zc_h - wh.zW).max()))
    print("W16 level val %.5f (feeds %d, with a click %d) holdout %.5f (feeds %d, with a click %d)" % (
        ndcg(wv.zW, wv).mean(), len(wv.M), int((wv.npos > 0).sum()), ndcg(wh.zW, wh).mean(), len(wh.M), int((wh.npos > 0).sum())))
    m50 = wv.M >= 50
    res = {}; store = {}
    for tag, files in ARMS.items():
        rv = read_model([np.load(f + "_val.npy") for f in files], wv)
        rh = read_model([np.load(f + "_hold.npy") for f in files], wh)
        r5 = read_model([np.load(f + "_val.npy") for f in files], wv, sub=m50)
        zp = (rv["mix"] / rv["mix_se"] + rh["mix"] / rh["mix_se"]) / np.sqrt(2)
        prom = rv["nats"] >= 0.0016 and rh["nats"] >= 0.0016
        mp = rv["mix"] >= 0 and rh["mix"] >= 0 and zp >= 2
        print(f"{tag}: level {rv['level']:.5f} / {rh['level']:.5f} | rho {rv['rho']:.3f} / {rh['rho']:.3f} | NATS {rv['nats']:+.5f}+-{rv['nats_se']:.5f} / "
              f"{rh['nats']:+.5f}+-{rh['nats_se']:.5f} -> {'PROMISING' if prom else 'not promising'} | member 0.9/0.1: val {rv['mix']:+.5f}+-{rv['mix_se']:.5f}, "
              f"holdout {rh['mix']:+.5f}+-{rh['mix_se']:.5f}, pooled z {zp:+.2f} -> {'PASS' if mp else 'no pass'} | val M>=50 ({r5['n_feeds']} feeds): "
              f"NATS {r5['nats']:+.5f}+-{r5['nats_se']:.5f}, member {r5['mix']:+.5f}+-{r5['mix_se']:.5f}", flush=True)
        store[tag] = (rv, rh)
        res[tag] = {sp: {k: v for k, v in r.items() if k not in ("nats_perfeed", "keep", "nd")} for sp, r in (("val", rv), ("hold", rh), ("val_m50", r5))}
        res[tag]["pooled_z"] = float(zp); res[tag]["promising"] = bool(prom); res[tag]["member_pass"] = bool(mp)
    # paired differences the verdict quotes
    def pair(a, b, label):
        out = {}
        for i, sp in enumerate(("val", "hold")):
            ra, rb = store[a][i], store[b][i]
            out[sp] = dict(nats=pse(ra["nats_perfeed"] - rb["nats_perfeed"]), level=pse(ra["nd"] - rb["nd"]))
        print(f"PAIR {label}: NATS val {out['val']['nats'][0]:+.5f}+-{out['val']['nats'][1]:.5f} hold {out['hold']['nats'][0]:+.5f}+-{out['hold']['nats'][1]:.5f} | "
              f"level val {out['val']['level'][0]:+.5f}+-{out['val']['level'][1]:.5f} hold {out['hold']['level'][0]:+.5f}+-{out['hold']['level'][1]:.5f}")
        res["PAIR " + label] = out
    pair("b3_abc_s0", "b0", "b3_abc_s0 - b0")
    pair("b3_abc_s1", "b0_s1", "b3_abc_s1 - b0_s1")
    pair("bundle(b3_abc_s0+b3_abc_s1)", "bundle(b0+b0_s1)", "B3x2 - B0x2")
    pair("b1_mcc4_s0", "b0", "b1_mcc4_s0 - b0")
    pair("b2_item_s0", "b0", "b2_item_s0 - b0")
    os.makedirs(f"{HERE}/out", exist_ok=True)
    json.dump(res, open(f"{HERE}/out/rv_eval.json", "w"), indent=1)
