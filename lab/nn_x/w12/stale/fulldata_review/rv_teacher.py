"""Reviewer: are the teacher rows on the 30,000 holdout feeds finite and OUT-OF-SAMPLE for every teacher component?"""
import numpy as np, polars as pl, torch, time, os, glob
R = "<repo>"; K = f"{R}/nn_x/w12/kd/out"
def P(*a): print(*a, flush=True)
f = pl.read_parquet(f"{R}/data/flat/train/feeds.parquet"); Mt = f["M"].to_numpy().astype(np.int64); offt = f["cand_off"].to_numpy().astype(np.int64)
p = np.nan_to_num(np.load(f"{R}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0); p[Mt < 50] = 0.0
cand = np.where((p > 0.9) & (Mt >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
rest = np.setdiff1d(cand, hold); ins = np.sort(np.random.default_rng(4242).choice(rest, 30000, replace=False))   # in-sample comparison feeds, same rule
rows = lambda ids: np.concatenate([np.arange(o, o + m) for o, m in zip(offt[ids], Mt[ids])])
hr, ir = rows(hold), rows(ins)
yt = np.load(f"{R}/data/flat/train/cand_target.npy", mmap_mode="r"); yh = np.asarray(yt[hr], np.float64); yi = np.asarray(yt[ir], np.float64)
def nd(s, t, M):
    n = len(M); fid = np.repeat(np.arange(n), M); off = np.concatenate([[0], np.cumsum(M)[:-1]])
    o = np.lexsort((t, -s.astype(np.float64), fid)); rk = np.arange(len(s)) - np.repeat(off, M); d = 1.0 / np.log2(rk + 2.0)
    dcg = np.bincount(fid, weights=t[o] * d, minlength=n); npos = np.bincount(fid, weights=t, minlength=n)
    idcg = np.bincount(fid, weights=(rk < np.repeat(npos, M)) * d, minlength=n); return dcg / idcg
def cen(s, M):
    fid = np.repeat(np.arange(len(M)), M); return s - (np.bincount(fid, weights=s) / M)[fid]
Mh, Mi = Mt[hold], Mt[ins]
comp = {}
P("component | n ckpt | NDCG holdout rows | NDCG 30,000 other draw-set TRAIN feeds | gap (in-sample - holdout) +- SE(unpaired)")
for t in ("a", "b", "c", "d", "e", "f", "g1", "g2", "p1", "p2"):
    m = np.load(f"{K}/teacher_{t}.npy", mmap_mode="r"); n = int(open(f"{K}/teacher_{t}.n").read())
    h = np.asarray(m[hr], np.float64); i = np.asarray(m[ir], np.float64); comp[t] = (h, i, n)
    a, b = nd(h, yh, Mh), nd(i, yi, Mi)
    P(f"{t:3s} | {n} | {a.mean():.5f} | {b.mean():.5f} | {b.mean() - a.mean():+.5f} +- {np.sqrt(a.var() / len(a) + b.var() / len(b)):.5f}")
for name, parts, n in (("teacher_mean11", "abcdef", 11), ("teacher_mean25", ["a", "b", "c", "d", "e", "f", "g1", "g2", "p1", "p2"], 25)):
    m = np.load(f"{K}/{name}.npy", mmap_mode="r")
    assert m.shape[0] == int(Mt.sum())
    nf = 0
    for s in range(0, m.shape[0], 20_000_000):
        nf += int((~np.isfinite(m[s:s + 20_000_000])).sum())
    h = np.asarray(m[hr], np.float64); i = np.asarray(m[ir], np.float64)
    hs = sum(comp[t][0] for t in parts) / n; assert sum(comp[t][2] for t in parts) == n
    fid = np.repeat(np.arange(30000), Mh); sd = np.sqrt(np.bincount(fid, weights=cen(h, Mh) ** 2) / Mh)
    a, b = nd(h, yh, Mh), nd(i, yi, Mi)
    P(f"{name}: rows {m.shape[0]}, non-finite in whole file {nf}; holdout rows non-finite {int((~np.isfinite(h)).sum())}; feeds with constant teacher {int((sd < 1e-6).sum())}; "
      f"max |file - sum(components)/{n}| on holdout rows {np.abs(h - hs).max():.2e}; NDCG holdout {a.mean():.5f}, in-sample feeds {b.mean():.5f}, gap {b.mean() - a.mean():+.5f}")
# members' OWN saved holdout prediction files vs the component files (per-feed centred)
own = {"e": [f"{R}/nn_x/w5/preds/holdout/nn_w5_e10_base_s954_holdout.npy", f"{R}/nn_x/w5/preds/holdout/nn_w5_e10_base_s951_holdout.npy"],
       "f": [f"{R}/nn_x/w5/preds/holdout/nn_w5_e10_base_s952_holdout.npy"],
       "p1": [f"{R}/nn_x/abl/preds/holdout/nn_abl_e10_pop1_s95{k}_holdout.npy" for k in (1, 2, 3, 4)]}
for t, fs in own.items():
    if all(os.path.exists(x) for x in fs):
        s = sum(np.load(x).astype(np.float64) for x in fs)
        P(f"component {t} holdout rows vs the members' own saved holdout files (centred per feed): max abs diff {np.abs(cen(comp[t][0], Mh) - cen(s, Mh)).max():.2e}, corr {np.corrcoef(cen(comp[t][0], Mh), cen(s, Mh))[0, 1]:.8f}")
    else:
        P(f"component {t}: own holdout files missing: {[x for x in fs if not os.path.exists(x)]}")
# checkpoint provenance
cks = []
for t in ("a", "b", "c", "d", "e", "f", "g1", "g2", "p1", "p2"):
    cks += [l.split()[0] for l in open(f"{K}/dump_{t}.log") if l.split() and l.split()[0].endswith(".pt")]
P(f"{len(cks)} teacher checkpoints")
bad = 0
for c in cks:
    a = torch.load(c, map_location="cpu", weights_only=False)["args"]
    init = a.get("init") or ""
    line = f"{c.replace(R + '/', '')}: holdout {a.get('holdout')} val_mix {a.get('val_mix', 0)} kd_file {a.get('kd_file', '') or '-'} train_feeds {a.get('train_feeds')} init {init.replace(R + '/', '') or '-'}"
    if init:
        ip = init if init.startswith("/") else None
        cand_paths = [ip] if ip else glob.glob(f"{R}/nn*/ckpt/{init}.pt") + glob.glob(f"{R}/nn_x/*/ckpt/{init}.pt")
        for q in cand_paths:
            b = torch.load(q, map_location="cpu", weights_only=False)["args"]
            line += f" | snapshot {q.replace(R + '/', '')}: holdout {b.get('holdout')} val_mix {b.get('val_mix', 0)} kd_file {b.get('kd_file', '') or '-'} init {b.get('init') or '-'}"
            bad += b.get("holdout") != 30000
    bad += a.get("holdout") != 30000 or bool(a.get("val_mix", 0)) or bool(a.get("kd_file", ""))
    P(line)
P(f"provenance problems: {bad}")
