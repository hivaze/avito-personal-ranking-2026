"""staleft_review: independent re-derivation of the staleft pilot numbers from its prediction files. CPU only.
Own NDCG (numpy lexsort, pessimistic ties), own per-feed z, own weighted mean / SE (pmix weights are the project's).
  cd <repo> && CUDA_VISIBLE_DEVICES="" ... .venv/bin/python nn_x/w12/stale/staleft_review/rederive.py
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["NUMBA_CACHE_DIR"] = "<repo>/nn_x/w12/stale/staleft_review/numba_cache"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("<repo>")
PR = ROOT / "nn_x/w12/stale/staleft/preds"
W5 = ROOT / "nn_x/w5/preds"
AB = ROOT / "nn_x/abl/preds"
FLAT = ROOT / "data/flat"
OUT = ROOT / "nn_x/w12/stale/staleft_review/out"
OUT.mkdir(exist_ok=True)
BOUND = 1790497920
lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


class Split:
    def __init__(self, name, labels=True):
        f = pl.read_parquet(FLAT / name / "feeds.parquet")
        self.M = f["M"].to_numpy().astype(np.int64)
        self.off = f["cand_off"].to_numpy().astype(np.int64)
        self.n = len(self.M)
        assert (self.off == np.concatenate([[0], np.cumsum(self.M)[:-1]])).all()
        self.fid = np.repeat(np.arange(self.n), self.M)
        self.pos = np.arange(self.M.sum()) - self.off[self.fid]
        self.ts = f["timestamp"].to_numpy().astype(np.int64)
        if labels:
            self.y = (np.load(FLAT / name / "cand_target.npy") > 0).astype(np.int8)
            self.npos = np.bincount(self.fid, self.y, self.n).astype(np.int64)
            disc = 1.0 / np.log2(np.arange(self.M.max()) + 2.0)
            cs = np.concatenate([[0.0], np.cumsum(disc)])
            self.idcg = cs[self.npos]
            self.disc = disc

    def z(self, x):
        x = np.asarray(x, np.float64)
        mu = np.bincount(self.fid, x, self.n) / self.M
        var = np.bincount(self.fid, x * x, self.n) / self.M - mu ** 2
        sd = np.sqrt(np.maximum(var, 0)) + 1e-9
        return (x - mu[self.fid]) / sd[self.fid]

    def ndcg(self, s):
        """per-feed NDCG, binary gain, pessimistic ties: within a tie the negatives come first."""
        s = np.asarray(s, np.float32)  # same precision as the files / the project tools: ties are float32 ties
        order = np.lexsort((self.y, -s.astype(np.float64), self.fid))
        gain = self.y[order] * self.disc[self.pos]
        dcg = np.bincount(self.fid, gain, self.n)  # order is grouped by fid, positions are self.pos
        out = np.ones(self.n)
        ok = self.npos > 0
        out[ok] = dcg[ok] / self.idcg[ok]
        return out

    def corr_top1(self, a, b):
        za, zb = self.z(a), self.z(b)
        c = np.bincount(self.fid, za * zb, self.n) / self.M
        # top-1: argmax per feed (first max)
        def am(x):
            o = np.lexsort((self.pos, -np.asarray(x, np.float64), self.fid))
            return o[self.off] - self.off
        return c, (am(a) != am(b)).astype(np.float64)


V = Split("val")
T = Split("test", labels=False)
pub = T.ts < BOUND
big = V.M >= 50
sys.path.insert(0, str(ROOT / "src"))
import pmix  # noqa: E402

Wt = pmix.weights()
wpub, wpriv = np.asarray(Wt["public"], float), np.asarray(Wt["private"], float)


def ms(d, w=None):
    d = np.asarray(d, float)
    if w is None:
        return d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    m = (w * d).sum() / w.sum()
    neff = w.sum() ** 2 / (w ** 2).sum()
    return m, np.sqrt((w * (d - m) ** 2).sum() / w.sum() / neff)


def fm(d, w=None):
    m, s = ms(d, w)
    return f"{m:+.5f} +- {s:.5f} (z {m / s:+.2f})"


def row(d):
    return f"all {fm(d)} | M>=50 {fm(d[big])} | M<50 {fm(d[~big])} | pmix-pub {fm(d, wpub)} | pmix-priv {fm(d, wpriv)}"


def ld(m, tag, k):
    return np.load(PR / f"nn_{m}_sft_{tag}_{k}.npy")


def zm(arrs, S=V):
    return np.mean([S.z(a) for a in arrs], 0)


def fin(a, b, S=V):
    return 0.5 * S.z(a) + 0.5 * S.z(b)


A = ("kd05w192", "kd05t25w192")
PILOT = [f"w5_e10_{a}_s{s}" for s in (951, 952) for a in A]
REP = [f"w5_e10_{a}_s{s}" for s in (953, 954) for a in A]
W8 = PILOT + REP
P1 = [f"abl_e10_pop1kd05t25_s{s}" for s in range(951, 955)]


def to_pools(members):
    tf = zm([np.load(W5 / f"nn_{m}_trainonly_val.npy") for m in members])
    ts = zm([ld(m, "p0", "val_e0_stale") for m in members])
    return tf, ts


def levels(members, tag, tf, ts):
    a = zm([ld(m, tag, "val_oof") for m in members])
    b = zm([ld(m, tag, "val_oof_stale") for m in members])
    return {"ft_fresh": V.ndcg(a), "ft_stale": V.ndcg(b), "fin_fresh": V.ndcg(fin(a, tf)), "fin_stale": V.ndcg(fin(b, ts)),
            "mem_fresh": np.mean([V.ndcg(ld(m, tag, "val_oof")) for m in members], 0),
            "mem_stale": np.mean([V.ndcg(ld(m, tag, "val_oof_stale")) for m in members], 0)}


def block(name, members, tags):
    P(f"\n===== {name}: {len(members)} members, {V.n} val feeds ({int(big.sum())} with M>=50) =====")
    tf, ts = to_pools(members)
    ntf, nts = V.ndcg(tf), V.ndcg(ts)
    P(f"TO pool fresh {ntf.mean():.5f} stale {nts.mean():.5f} drop {fm(nts - ntf)}")
    c = levels(members, "p0", tf, ts)
    P(f"control p0 levels: FT fresh {c['ft_fresh'].mean():.5f} FT stale {c['ft_stale'].mean():.5f} final fresh {c['fin_fresh'].mean():.5f} final stale {c['fin_stale'].mean():.5f}")
    P(f"control: final stale-fresh {fm(c['fin_stale'] - c['fin_fresh'])}; FT branch stale-fresh {fm(c['ft_stale'] - c['ft_fresh'])}; "
      f"FT half worth fresh {fm(c['fin_fresh'] - ntf)} stale {fm(c['fin_stale'] - nts)}")
    res = {"p0": c}
    for t in tags:
        if not all((PR / f"nn_{m}_sft_{t}_val_oof_stale.npy").exists() for m in members):
            P(f"{t}: incomplete"); continue
        v = levels(members, t, tf, ts)
        res[t] = v
        P(f"{t} levels: FT fresh {v['ft_fresh'].mean():.5f} FT stale {v['ft_stale'].mean():.5f} final fresh {v['fin_fresh'].mean():.5f} final stale {v['fin_stale'].mean():.5f}")
        for key, nm in [("fin_fresh", "FINAL fresh"), ("fin_stale", "FINAL stale"), ("ft_fresh", "FT fresh"), ("ft_stale", "FT stale"), ("mem_fresh", "member fresh"), ("mem_stale", "member stale")]:
            P(f"  {t} {nm:13s} {row(v[key] - c[key])}")
        P(f"  {t} FINAL mix     {row(0.5 * (v['fin_fresh'] - c['fin_fresh']) + 0.5 * (v['fin_stale'] - c['fin_stale']))}")
        a, b = ms(v["fin_fresh"] - c["fin_fresh"]), ms(v["fin_stale"] - c["fin_stale"])
        P(f"  {t} RULE: fresh {a[0]:+.5f} (z {a[0] / a[1]:+.2f}) stale {b[0]:+.5f} (z {b[0] / b[1]:+.2f}) -> {'PASS' if a[0] >= 0 and b[0] >= 2 * b[1] else 'FAIL'}")
    return res, (tf, ts)


def test_shift(members, tags):
    P(f"\n-- test shift ({len(members)} members): FT branch corr pub/priv | top-1 changed pub/priv || final corr | top-1 --")
    tp0 = zm([ld(m, "p0", "test") for m in members], T)
    tto = zm([np.load(W5 / f"nn_{m}_trainonly_test.npy") for m in members], T)
    for t in tags:
        if not all((PR / f"nn_{m}_sft_{t}_test.npy").exists() for m in members):
            continue
        tp = zm([ld(m, t, "test") for m in members], T)
        c, t1 = T.corr_top1(tp, tp0)
        fc, f1 = T.corr_top1(fin(tp, tto, T), fin(tp0, tto, T))
        P(f"{t}: FT corr {c[pub].mean():.5f} / {c[~pub].mean():.5f} | top-1 {t1[pub].mean():.4f} / {t1[~pub].mean():.4f} || final corr {fc[pub].mean():.5f} / {fc[~pub].mean():.5f} | top-1 {f1[pub].mean():.4f} / {f1[~pub].mean():.4f}")


if __name__ == "__main__":
    rp, _ = block("PILOT 4-pool (s951, s952)", PILOT, ["p05", "p10", "p0b"])
    test_shift(PILOT, ["p05", "p10", "p0b"])
    rr, _ = block("REPLICATION 4-pool (s953, s954)", REP, ["p10", "p05"])
    r8, (tf8, ts8) = block("POOLED W8", W8, ["p10", "p05"])
    test_shift(W8, ["p05", "p10"])
    P(f"\nW8 pooled: p10 - p05 FINAL mix {row(0.5 * (r8['p10']['fin_fresh'] - r8['p05']['fin_fresh']) + 0.5 * (r8['p10']['fin_stale'] - r8['p05']['fin_stale']))}")
    # stored 3-seed reference
    s3 = zm([np.load(W5 / f"nn_{m}_n2cpv3s3_val_oof.npy") for m in W8])
    n3 = V.ndcg(fin(s3, tf8))
    P(f"\nW8 stored 3-seed final {n3.mean():.5f}; 1-seed p0 - stored 3-seed: {row(r8['p0']['fin_fresh'] - n3)}")
    P(f"FT half worth, stored 3-seed W8 (final - TO pool): {row(n3 - V.ndcg(tf8))}")
    # ---- W8P20 form ----
    P("\n===== W8P20 form =====")
    key = {"fresh": "val_oof", "stale": "val_oof_stale"}
    tof = {"fresh": (tf8, zm([np.load(AB / f"nn_{m}_trainonly_val.npy") for m in P1])),
           "stale": (ts8, zm([ld(m, "p0", "val_e0_stale") for m in P1]))}
    ftp = {v: zm([ld(m, "p0", key[v]) for m in P1]) for v in key}
    ftp_stored = zm([np.load(AB / f"nn_{m}_n2cpv3s3_val_oof.npy") for m in P1])

    def f20(tag, v, pl_ft=None):
        a = zm([ld(m, tag, key[v]) for m in W8])
        pf = ftp[v] if pl_ft is None else pl_ft
        return V.ndcg(0.5 * V.z(0.8 * V.z(a) + 0.2 * V.z(pf)) + 0.5 * V.z(0.8 * V.z(tof[v][0]) + 0.2 * V.z(tof[v][1])))

    c20 = {v: f20("p0", v) for v in key}
    P(f"control W8P20 form fresh {c20['fresh'].mean():.5f} stale {c20['stale'].mean():.5f}; W8 form fresh {r8['p0']['fin_fresh'].mean():.5f} stale {r8['p0']['fin_stale'].mean():.5f}")
    P(f"P20 admixture worth (control branches): fresh {row(c20['fresh'] - r8['p0']['fin_fresh'])}")
    P(f"P20 admixture worth (control branches): stale {row(c20['stale'] - r8['p0']['fin_stale'])}")
    for t in ("p10", "p05"):
        d = {v: f20(t, v) - c20[v] for v in key}
        for v in key:
            P(f"{t} W8P20 FINAL {v:5s}: {row(d[v])}")
        P(f"{t} W8P20 FINAL mix  : {row(0.5 * d['fresh'] + 0.5 * d['stale'])}")
        # interaction: is the P20-form delta different from the W8-form delta? (paired)
        for v, kk in (("fresh", "fin_fresh"), ("stale", "fin_stale")):
            P(f"{t} INTERACTION (P20-form delta - W8-form delta) {v}: {row(d[v] - (r8[t][kk] - r8['p0'][kk]))}")
        # with the stored pop-light FT branch on fresh val
        dd = f20(t, "fresh", ftp_stored) - f20("p0", "fresh", ftp_stored)
        P(f"{t} W8P20 FINAL fresh with STORED pop-light FT branch: {row(dd)}")
    np.savez(OUT / "perfeed.npz", **{f"w8_{t}_{k}": r8[t][k] for t in r8 for k in ("fin_fresh", "fin_stale", "ft_fresh", "ft_stale")})
    (OUT / "rederive.txt").write_text("\n".join(lines) + "\n")
