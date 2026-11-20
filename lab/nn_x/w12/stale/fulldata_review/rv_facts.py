"""Reviewer's own re-derivation of the facts (no import of the study's code; src/metrics only as a cross-check of my NDCG)."""
import numpy as np, polars as pl, json, sys, time
R = "<repo>"
t0 = time.time()
def P(*a):
    print(*a, flush=True)
def feeds(split):
    f = pl.read_parquet(f"{R}/data/flat/{split}/feeds.parquet")
    return f["M"].to_numpy().astype(np.int64), f["cand_off"].to_numpy().astype(np.int64), f["user_id"].to_numpy()
def my_ndcg(s, t, M):
    """binary gain, pessimistic ties, per feed; s, t flat in feed order (contiguous feeds of sizes M)."""
    n = len(M); fid = np.repeat(np.arange(n), M); off = np.concatenate([[0], np.cumsum(M)[:-1]])
    o = np.lexsort((t, -s.astype(np.float64), fid))
    rk = np.arange(len(s)) - np.repeat(off, M)
    d = 1.0 / np.log2(rk + 2.0)
    dcg = np.bincount(fid, weights=t[o] * d, minlength=n)
    npos = np.bincount(fid, weights=t, minlength=n)
    idcg = np.bincount(fid, weights=(rk < np.repeat(npos, M)) * d, minlength=n)
    out = np.ones(n); m = idcg > 0; out[m] = dcg[m] / idcg[m]
    return out
def zfeed(s, M):
    n = len(M); fid = np.repeat(np.arange(n), M)
    mu = np.bincount(fid, weights=s, minlength=n) / M
    var = np.bincount(fid, weights=(s - mu[fid]) ** 2, minlength=n) / M
    return (s - mu[fid]) / np.sqrt(np.maximum(var, 1e-12))[fid]
def pse(d):
    return f"{d.mean():+.5f} +- {d.std(ddof=1) / np.sqrt(len(d)):.5f} (n {len(d)})"
Mt, offt, uidt = feeds("train"); Mv, offv, uidv = feeds("val")
ntr, ncand = len(Mt), int(Mt.sum())
P(f"train feeds {ntr}, candidates {ncand}; val feeds {len(Mv)}, candidates {int(Mv.sum())}")
pr = np.load(f"{R}/eda/labels_position/out/train_prandom.npz")
assert (pr["fidx"] == np.arange(ntr)).all()
p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[Mt < 50] = 0.0
cand = np.where((p > 0.9) & (Mt >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
hrows = np.concatenate([np.arange(o, o + m) for o, m in zip(offt[hold], Mt[hold])])
expl = np.where(p > 0.5)[0]
P(f"holdout: {len(hold)} feeds = {100 * len(hold) / ntr:.3f}% of feeds; rows {len(hrows)} = {100 * len(hrows) / ncand:.3f}% of candidates; mean M holdout {Mt[hold].mean():.1f} vs all {Mt.mean():.1f}")
P(f"draw set (p>0.9, M>=50): {len(cand)} -> holdout share {100 * 30000 / len(cand):.1f}%; exploration-like (p>0.5): {len(expl)} -> holdout share {100 * np.isin(hold, expl).sum() / len(expl):.2f}%; all holdout have p>0.5: {bool((p[hold] > 0.5).all())}")
def nb(ids):
    Mp = (Mt[ids] + 15) // 16 * 16; tot = 0
    for mp in np.unique(Mp):
        B = max(1, min(512, 32768 // int(mp))); tot += -(-int((Mp == mp).sum()) // B)
    return tot
allf = np.arange(ntr); woh = np.setdiff1d(allf, hold)
s0, s1 = nb(woh), nb(allf)
P(f"steps/epoch without holdout {s0} (x10 = {10 * s0}; t_cool {round(0.8 * 10 * s0)}); with holdout {s1} (x10 = {10 * s1}; t_cool {round(0.8 * 10 * s1)}); re-cool 2 epochs: {2 * s0} -> {2 * s1}")
P(f"cooldown pool: without holdout {int((p[woh] > 0.5).sum())}; with {int((p > 0.5).sum())}")
yt = np.load(f"{R}/data/flat/train/cand_target.npy", mmap_mode="r"); yv = np.load(f"{R}/data/flat/val/cand_target.npy").astype(np.float64)
# feeds without a positive
cs = np.concatenate([[0], np.cumsum(np.asarray(yt, dtype=np.int64))]); npos_t = cs[offt + Mt] - cs[offt]
csv_ = np.concatenate([[0], np.cumsum(yv)]); npos_v = csv_[offv + Mv] - csv_[offv]
P(f"train feeds with 0 positives: {int((npos_t == 0).sum())}; val feeds with 0 positives: {int((npos_v == 0).sum())}")
yh = np.asarray(yt[hrows], dtype=np.float64); Mh = Mt[hold]
# cross-check my NDCG against src/metrics on val with a random score
sys.path.insert(0, f"{R}/src"); import metrics
rs = np.random.default_rng(0).normal(size=len(yv)).round(1)
assert np.allclose(my_ndcg(rs, yv, Mv), metrics.ndcg_per_feed(rs, yv, offv, Mv)); P("my NDCG == src/metrics on a tied random score (val): ok")
# ---- W16 pools: selected vs @last
W16 = [f"w5_e10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in range(951, 959)]
D = f"{R}/nn_x/w5/preds"
def pool(names, kind):
    acc = None
    for v in names:
        if kind == "val":
            x = np.load(f"{D}/nn_{v}_trainonly_val.npy").astype(np.float64); z = zfeed(x, Mv)
        else:
            x = np.load(f"{D}/holdout/nn_{v}_holdout.npy").astype(np.float64); z = zfeed(x, Mh)
        acc = z if acc is None else acc + z
    return acc / len(names)
sel_v = my_ndcg(pool(W16, "val"), yv, Mv); last_v = my_ndcg(pool([v + "@last" for v in W16], "val"), yv, Mv)
sel_h = my_ndcg(pool(W16, "hold"), yh, Mh); last_h = my_ndcg(pool([v + "@last" for v in W16], "hold"), yh, Mh)
P(f"W16 16-pool val: selected {sel_v.mean():.5f} @last {last_v.mean():.5f}; selected - @last {pse(sel_v - last_v)}")
P(f"W16 16-pool holdout: selected {sel_h.mean():.5f} @last {last_h.mean():.5f}; selected - @last {pse(sel_h - last_h)}")
# member level + kept steps
ms_v, ms_h, lastv_m, lasth_m = [], [], [], []
for v in W16:
    a = my_ndcg(np.load(f"{D}/nn_{v}_trainonly_val.npy").astype(np.float64), yv, Mv).mean(); b = my_ndcg(np.load(f"{D}/nn_{v}@last_trainonly_val.npy").astype(np.float64), yv, Mv).mean()
    c = my_ndcg(np.load(f"{D}/holdout/nn_{v}_holdout.npy").astype(np.float64), yh, Mh).mean(); d = my_ndcg(np.load(f"{D}/holdout/nn_{v}@last_holdout.npy").astype(np.float64), yh, Mh).mean()
    ms_v.append(a - b); ms_h.append(c - d); lastv_m.append(b); lasth_m.append(d)
P(f"member level selected - @last: holdout mean {np.mean(ms_h):+.5f} (min {np.min(ms_h):+.5f}, max {np.max(ms_h):+.5f}); val mean {np.mean(ms_v):+.5f} (min {np.min(ms_v):+.5f} max {np.max(ms_v):+.5f})")
P(f"@last members: val SD {np.std(lastv_m, ddof=1):.5f} (mean {np.mean(lastv_m):.5f}); holdout SD {np.std(lasth_m, ddof=1):.5f} (mean {np.mean(lasth_m):.5f}; min {np.min(lasth_m):.5f} max {np.max(lasth_m):.5f})")
# ---- null band: random disjoint 8-pool splits (4 + 4 per arm) of the 16 @last members, val
Z = np.stack([zfeed(np.load(f"{D}/nn_{v}@last_trainonly_val.npy").astype(np.float64), Mv) for v in W16])
rng = np.random.default_rng(777); ds = []
for _ in range(60):
    a1 = rng.permutation(8)[:4]; a2 = 8 + rng.permutation(8)[:4]; A = np.concatenate([a1, a2]); B = np.setdiff1d(np.arange(16), A)
    ds.append(my_ndcg(Z[A].mean(0), yv, Mv).mean() - my_ndcg(Z[B].mean(0), yv, Mv).mean())
ds = np.array(ds); s8 = np.sqrt((ds ** 2).mean())
P(f"null: 60 random 8 vs 8 splits (4+4 per arm), val: RMS {s8:.5f}, max |d| {np.abs(ds).max():.5f} -> s16 = s8/sqrt(2) = {s8 / np.sqrt(2):.5f}; 3 s16 = {3 * s8 / np.sqrt(2):.5f}")
json.dump(dict(s8=float(s8)), open(f"{R}/nn_x/w12/stale/fulldata_review/logs/rv_facts.json", "w"))
P(f"done {time.time() - t0:.0f}s")
