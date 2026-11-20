"""veto: shared read-out helpers (CPU, read-only on every project file; writes nothing outside nn_x/w12/stale/veto).

Pool score = member mean of per-feed z (same definition as src/stack_val.feed_rank_z[:,1], re-implemented here so that no numba
cache file is written under src/). NDCG = numba copy of nn_x/w12/recheck/pool_composition/pc_common._ndcg (pessimistic ties).
pmix weights = src/pmix.weights() (imported; pure numpy / polars).
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_k, "12")
import numpy as np
import numba as nb
import polars as pl
from pathlib import Path

ROOT = Path("<repo>")
HERE = ROOT / "nn_x/w12/stale/veto"
DIRS = [ROOT / "nn_x/w5/preds", ROOT / "nn_x/abl/preds", ROOT / "preds", ROOT / "nn_x/r6/preds", ROOT / "nn_x/solvers/preds",
        ROOT / "nn_x/w11/width/preds", ROOT / "nn_x/w10/fpcd/preds", ROOT / "nn_x/r7/preds"]


def find(v, kind):
    """kind: val | hold | stale (stored GPU bf16) | cstale / cval (CPU fp32, this directory) | ft | ft3."""
    if kind == "cstale":
        p = HERE / "preds" / f"cpu_{v}_staleval.npy"; return p if p.exists() else None
    if kind == "cval":
        p = HERE / "preds" / f"cpu_{v}_val.npy"; return p if p.exists() else None
    rel = {"val": f"nn_{v}_trainonly_val.npy", "hold": f"holdout/nn_{v}_holdout.npy", "stale": f"stale/nn_{v}_staleval.npy",
           "ft": f"nn_{v}_n2cpv3_val_oof.npy", "ft3": f"nn_{v}_n2cpv3s3_val_oof.npy", "test": f"nn_{v}_trainonly_test.npy"}[kind]
    for d in DIRS:
        if (d / rel).exists():
            return d / rel
    return None


@nb.njit(parallel=True, cache=True)
def _ndcg(s, y, off, M):
    n = len(M); out = np.ones(n, np.float64)
    for f in nb.prange(n):
        o, m = off[f], M[f]
        sv = s[o:o + m]; yv = y[o:o + m]
        npos = 0
        for k in range(m):
            if yv[k] > 0: npos += 1
        if npos == 0: continue
        order = np.argsort(-sv.astype(np.float64) + 0.0, kind='mergesort')
        dcg = 0.0
        i = 0
        while i < m:
            j = i
            while j + 1 < m and sv[order[j + 1]] == sv[order[i]]: j += 1
            if j == i:
                if yv[order[i]] > 0: dcg += 1.0 / np.log2(i + 2.0)
            else:
                cp = 0
                for k in range(i, j + 1):
                    if yv[order[k]] > 0: cp += 1
                for k in range(j - cp + 1, j + 1): dcg += 1.0 / np.log2(k + 2.0)
            i = j + 1
        idcg = 0.0
        for k in range(npos): idcg += 1.0 / np.log2(k + 2.0)
        out[f] = dcg / idcg
    return out


class Ctx:
    def __init__(self):
        fv = pl.read_parquet(ROOT / "data/flat/val/feeds.parquet")
        self.Mv = fv["M"].to_numpy().astype(np.int64); self.ov = fv["cand_off"].to_numpy().astype(np.int64)
        self.yv = np.load(ROOT / "data/flat/val/cand_target.npy").astype(np.float32)
        self.fidv = np.repeat(np.arange(len(self.Mv)), self.Mv)
        Mh, oh = np.load(ROOT / "nn_x/w5/preds/holdout/holdout_meta.npy"); self.Mh = Mh.astype(np.int64); self.oh = oh.astype(np.int64)
        self.yh = np.load(ROOT / "nn_x/w5/preds/holdout/holdout_y.npy").astype(np.float32)
        self.fidh = np.repeat(np.arange(len(self.Mh)), self.Mh)
        # val slices
        luo = np.load(ROOT / "nn/feat_cache/val/it_hist_all_luo.npy", mmap_mode="r")
        cold = np.nan_to_num(np.asarray(luo, np.float64), nan=0.0) == 0
        npos = np.bincount(self.fidv, self.yv, len(self.Mv)); ncold = np.bincount(self.fidv, self.yv * cold, len(self.Mv))
        self.sl = {"M>=50": self.Mv >= 50, "M<50": self.Mv < 50, "coldpos": (npos > 0) & (ncold == npos),
                   "warmpos": (npos > 0) & (ncold == 0)}
        # pmix weights (src/pmix.py)
        sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eda/errors/segments"))
        import pmix
        W = pmix.weights(); self.wpub = W["public"]; self.wpriv = W["private"]
        # holdout days (r6_eval.setup: the same 30,000 feeds)
        tr = pl.read_parquet(ROOT / "data/flat/train/feeds.parquet", columns=["timestamp", "M"])
        pr = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
        trM = tr["M"].to_numpy(); pr[trM < 50] = 0.0
        cand = np.where((pr > 0.9) & (trM >= 50))[0]
        hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
        assert (trM[hold] == self.Mh).all(), "holdout identity mismatch"
        day = (tr["timestamp"].to_numpy()[hold] + 3 * 3600) // 86400
        self.hday = day
        self.hl = {"late>=Sep13": day >= 20709, "Sep17-19": day >= 20713, "Sep6-8": day <= 20704}
        self._c = {}

    def z(self, sp, x):
        fid, M = (self.fidv, self.Mv) if sp == "val" else (self.fidh, self.Mh)
        s = np.asarray(x, np.float64)
        mu = np.bincount(fid, s, len(M)) / M
        sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-9
        return ((s - mu[fid]) / sd[fid]).astype(np.float32)

    def zfile(self, v, kind):
        key = (v, kind)
        if key not in self._c:
            p = find(v, kind)
            if p is None:
                raise FileNotFoundError(f"{kind} file of {v}")
            self._c[key] = self.z("hold" if kind == "hold" else "val", np.load(p))
        return self._c[key]

    def has(self, names, kind):
        return all(find(v, kind) is not None for v in names)

    def pool(self, spec, kind):
        """spec = list of member names (member-equal) or list of (weight, [names]) blocks: sum_k w_k z(mean z of block k)."""
        sp = "hold" if kind == "hold" else "val"
        if spec and isinstance(spec[0], tuple):
            acc = 0.0
            for w, names in spec:
                acc = acc + w * self.z(sp, np.mean([self.zfile(v, kind) for v in names], 0))
            return np.asarray(acc, np.float32)
        return np.mean([self.zfile(v, kind) for v in spec], 0).astype(np.float32)

    def nv(self, s):
        return _ndcg(np.ascontiguousarray(s, np.float32), self.yv, self.ov, self.Mv)

    def nh(self, s):
        return _ndcg(np.ascontiguousarray(s, np.float32), self.yh, self.oh, self.Mh)


def ms(d, w=None):
    """(mean, se) of a per-feed delta; weighted form = src/pmix.wmean_se."""
    if w is None:
        return float(d.mean()), float(d.std() / np.sqrt(len(d)))
    m = np.average(d, weights=w); neff = w.sum() ** 2 / (w ** 2).sum()
    return float(m), float(np.sqrt(np.average((d - m) ** 2, weights=w) / neff))


def f(t):
    return "n/a" if t is None else f"{t[0]:+.5f} +- {t[1]:.5f} (z {t[0] / t[1] if t[1] > 0 else 0:+.2f})"


def flat(spec):
    return [v for _, ns in spec for v in ns] if spec and isinstance(spec[0], tuple) else list(spec)


def read(c, label, A, B, kinds=("val", "stale", "cstale", "hold"), sufA="", sufB=None):
    """Paired read of pool A against pool B. Suffix '@last' is applied to member names when given."""
    sufB = sufA if sufB is None else sufB
    ap = lambda spec, s: [(w, [v + s for v in ns]) for w, ns in spec] if spec and isinstance(spec[0], tuple) else [v + s for v in spec]
    A2, B2 = ap(A, sufA), ap(B, sufB)
    r = {"label": label, "nA": len(flat(A2)), "nB": len(flat(B2))}
    if "val" in kinds and c.has(flat(A2) + flat(B2), "val"):
        a, b = c.nv(c.pool(A2, "val")), c.nv(c.pool(B2, "val")); d = a - b
        r["levelA"], r["levelB"] = float(a.mean()), float(b.mean())
        r["val"] = ms(d)
        for k, m in c.sl.items(): r[k] = ms(d[m])
        r["pmix_pub"] = ms(d, c.wpub); r["pmix_priv"] = ms(d, c.wpriv)
    for kind in ("stale", "cstale"):
        if kind in kinds and c.has(flat(A2) + flat(B2), kind):
            a, b = c.nv(c.pool(A2, kind)), c.nv(c.pool(B2, kind)); d = a - b
            r[kind] = ms(d); r[kind + "_levelA"], r[kind + "_levelB"] = float(a.mean()), float(b.mean())
            r[kind + "_priv"] = ms(d, c.wpriv)
            for k, m in c.sl.items(): r[kind + "_" + k] = ms(d[m])
    if "hold" in kinds and c.has(flat(A2) + flat(B2), "hold"):
        a, b = c.nh(c.pool(A2, "hold")), c.nh(c.pool(B2, "hold")); d = a - b
        r["hold"] = ms(d); r["hold_levelA"], r["hold_levelB"] = float(a.mean()), float(b.mean())
        for k, m in c.hl.items(): r["hold_" + k] = ms(d[m])
    return r


def show(r):
    g = lambda k: f(r.get(k))
    print(f"== {r['label']}  (A {r['nA']} members, B {r['nB']})")
    if "val" in r:
        print(f"   val {g('val')} [A {r['levelA']:.5f} B {r['levelB']:.5f}] | M>=50 {g('M>=50')} | M<50 {g('M<50')}")
        print(f"   coldpos {g('coldpos')} | warmpos {g('warmpos')} | pmix-pub {g('pmix_pub')} | pmix-priv {g('pmix_priv')}")
    for kind in ("stale", "cstale"):
        if kind in r:
            print(f"   {kind}-val {g(kind)} [A {r[kind + '_levelA']:.5f} B {r[kind + '_levelB']:.5f}] | priv-weighted {g(kind + '_priv')}"
                  f" | M>=50 {g(kind + '_M>=50')} | M<50 {g(kind + '_M<50')}")
    if "hold" in r:
        print(f"   holdout {g('hold')} [A {r['hold_levelA']:.5f} B {r['hold_levelB']:.5f}] | late>=Sep13 {g('hold_late>=Sep13')} | Sep17-19 {g('hold_Sep17-19')} | Sep6-8 {g('hold_Sep6-8')}")
    sys.stdout.flush()
