"""famS common code: CatBoost students on teacher targets (family S of nn_x/w12/cb/PREDECLARED.txt). CPU only (cbx enforces it).
Raw-embedding numeric blocks for S3 (no label, no NN output):
  variant "raw": cand (128 int8 dims) + element-wise products cand x unit(user), cand x unit(hmean), cand x unit(h5)  -> 512 columns
  variant "pca": 32-dim PCA of cand (fitted on train-side FIT rows of xm, unsupervised) + products in that PCA basis   -> 128 columns
Products are scaled so that the sum over the 128 raw dims = 100 x cosine(cand, feed vector)."""
import os, sys, time, json
import numpy as np
sys.path.insert(0, "<repo>/nn_x/w12/cb/infra")
import cbx  # noqa: E402
from cbx import log  # noqa: E402

HERE = cbx.ROOT / "nn_x/w12/cb/famS"
PREDS = HERE / "preds"
CACHE = HERE / "cache"
KINDS = ("user", "hmean", "h5")
NPC = 32


def block_names(variant):
    if variant == "raw":
        return [f"e{j}" for j in range(128)] + [f"ex{k}_{j}" for k in KINDS for j in range(128)]
    if variant == "pca":
        return [f"pe{j}" for j in range(NPC)] + [f"px{k}_{j}" for k in KINDS for j in range(NPC)]
    if variant == "cand":
        return [f"e{j}" for j in range(128)]
    raise KeyError(variant)


def pca_basis():
    """(mu, V): mean and the 32 leading principal directions of the candidate vectors of the xm train-side FIT feeds (3M-row sample)."""
    p = CACHE / "pca32.npz"
    if not p.exists():
        tr = cbx.load("xm")
        fit, _ = cbx.trainside_split(tr)
        rows = np.flatnonzero(np.repeat(fit, tr.M))
        rg = np.random.default_rng(7)
        idx = np.sort(rg.choice(rows, 3_000_000, replace=False))
        E = np.asarray(tr.emb_cand_i8[idx], np.float64)
        mu = E.mean(0)
        C = (E - mu).T @ (E - mu) / len(E)
        w, U = np.linalg.eigh(C)
        o = np.argsort(-w)
        V = U[:, o[:NPC]]
        tmp = CACHE / f"pca32.tmp.{os.getpid()}.npz"
        np.savez(tmp, mu=mu.astype(np.float32), V=V.astype(np.float32), evr=(w[o] / w.sum()).astype(np.float32))
        os.replace(tmp, p)
        log(f"pca32 fitted on {len(E)} xm fit rows; explained variance of 32 comps {w[o][:NPC].sum() / w.sum():.3f}")
    z = np.load(p)
    return z["mu"], z["V"]


def _unit(f):
    f = np.asarray(f, np.float32)
    n = np.linalg.norm(f, axis=1, keepdims=True)
    return np.where(n > 0, f / np.maximum(n, 1e-6), 0.0).astype(np.float32)


def build_block(ds, variant):
    """float16 (n_rows, n_cols), Fortran order (contiguous columns)."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    names = block_names(variant)
    n = ds.n_rows
    out = np.empty((n, len(names)), np.float16, order="F")
    e = ds.emb_cand_i8
    gid = ds.gid
    fh = {k: _unit(ds.emb_feed(k)) for k in KINDS} if variant != "cand" else {}
    if variant == "pca":
        mu, V = pca_basis()
        fhp = {k: fh[k] @ V for k in KINDS}
    step = 400_000

    def one(a):
        b = min(a + step, n)
        ec = np.asarray(e[a:b], np.float32)
        g = gid[a:b]
        if variant in ("raw", "cand"):
            out[a:b, :128] = ec
            for i, k in enumerate(KINDS if variant == "raw" else ()):
                out[a:b, 128 * (i + 1):128 * (i + 2)] = ec * fh[k][g] * np.float32(100.0 / 256.0)
        else:
            out[a:b, :NPC] = (ec - mu) @ V
            ep = (ec @ V) * np.float32(1.0 / 256.0)
            for i, k in enumerate(KINDS):
                out[a:b, NPC * (i + 1):NPC * (i + 2)] = ep * fhp[k][g] * np.float32(100.0)
    with ThreadPoolExecutor(cbx.THREADS) as ex:
        list(ex.map(one, range(0, n, step)))
    assert np.isfinite(out[:: max(1, n // 100000)]).all()
    log(f"block {variant} for {ds.name}: {out.shape} float16 in {time.time() - t0:.0f}s rss {cbx.rss_gb()[0]:.1f} GB")
    return out


_BLK = {}


def block(ds, variant):
    key = (ds.name, variant)
    if key not in _BLK:
        _BLK.clear()            # one data set in RAM at a time
        _BLK[key] = build_block(ds, variant)
    return _BLK[key]


def extra_cols(variant):
    """dict name -> callable(ds) for cbx.make_pool(extra=...): one pool_kw serves train, selection, val, holdout and test pools."""
    return {nm: (lambda d, j=j: block(d, variant)[:, j]) for j, nm in enumerate(block_names(variant))}


def kd_soft_label(tr, w=0.5, T=1.0):
    """the NN students' listwise KD target: (1 - w) y / sum(y) + w softmax(teacher_mean11 / T) inside the feed (labels >= 0, sum 1 per feed).
    Feeds without a click: softmax(teacher) only."""
    y = np.asarray(tr.y, np.float64)
    s = np.add.reduceat(y, tr.off)
    soft = np.asarray(tr.teacher("mean11", "soft", T), np.float64)
    sy = s[tr.gid]
    lab = np.where(sy > 0, (1 - w) * y / np.maximum(sy, 1) + w * soft, soft)
    return lab.astype(np.float32), int((s == 0).sum())
