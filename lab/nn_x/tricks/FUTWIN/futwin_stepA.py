"""FUTWIN Step A closure readout (oracle; CPU only). See PREDECLARED.txt in this directory for the frozen rule.

u_fut  : share of the user's FUTURE-window events (next train snapshot >= 24 h later, minus the current history,
         multiset difference on (item_id, event_type)) in the candidate's category  -- oracle, NOT a feature
u_past : share of the current history events in the candidate's category
u_echo : 1[candidate category in the categories of this feed's positives]  -- label-echo control
Each is added to the G3 train-only holdout pool as zf(pool) + 0.1 * zf(u) and scored with paired per-feed NDCG.
Writes RESULT.txt / result.json next to this file. Reads only data/flat, eda p_random and preds/holdout."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat/train"
PH = ROOT / "preds/holdout"
OUT = ROOT / "nn_x/tricks/FUTWIN"
W_MAIN = 0.1
W_INFO = [0.05, 0.2, 0.3]
G3 = ([f"v23_gru_pre2s_s{i}" for i in range(501, 507)] + [f"r6_cd_m02_r{i}" for i in range(101, 108)]
      + [f"r6_cd_m02_p70k_r{i}" for i in range(101, 108)])
assert len(G3) == 20

lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


t0 = time.time()
# ---------------- holdout feed index (nn_x/r6/train.py / holdout_eval.py rule) ----------------
f = pl.read_parquet(FLAT / "feeds.parquet")
Mtr = f["M"].to_numpy().astype(np.int64)
off_tr = f["cand_off"].to_numpy().astype(np.int64)
Htr = f["H"].to_numpy().astype(np.int64)
hoff_tr = f["hist_off"].to_numpy().astype(np.int64)
uid = f["user_id"].to_numpy().astype(np.int64)
ts = f["timestamp"].to_numpy().astype(np.int64)
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
assert (pr["fidx"] == np.arange(len(Mtr))).all()
prand = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
prand[Mtr < 50] = 0.0
cand = np.where((prand > 0.9) & (Mtr >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
Mh, offh_src = Mtr[hold], off_tr[hold]
offh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
meta = np.load(PH / "holdout_meta.npy")
assert (meta[0] == Mh).all() and (meta[1] == offh).all(), "holdout index mismatch"
rows = np.concatenate([np.arange(o, o + m) for o, m in zip(offh_src, Mh)])
ctgt = np.load(FLAT / "cand_target.npy", mmap_mode="r")
ccat = np.load(FLAT / "cand_category_id.npy", mmap_mode="r")
citem = np.load(FLAT / "cand_item_id.npy", mmap_mode="r")
y = np.asarray(ctgt[rows]).astype(np.float64)
yh = np.load(PH / "holdout_y.npy").astype(np.float64)
assert (y == yh).all(), "holdout labels mismatch"
cat_c = np.asarray(ccat[rows]).astype(np.int64)
item_c = np.asarray(citem[rows]).astype(np.int64)
n = len(hold)
fid = np.repeat(np.arange(n), Mh)
P(f"holdout: {n} feeds, {len(rows)} candidates, positives {int(y.sum())}, feeds with >=1 positive "
  f"{int((np.bincount(fid, y, n) > 0).sum())}  [{time.time() - t0:.0f}s]")

# ---------------- earliest later train feed of the same non-zero user with gap >= 24 h ----------------
order = np.lexsort((np.arange(len(uid)), ts, uid))
key_s = uid[order] * (1 << 32) + ts[order]
tgt = uid[hold] * (1 << 32) + ts[hold] + 86400
pos = np.searchsorted(key_s, tgt, side="left")
okp = pos < len(key_s)
nb = np.full(n, -1, np.int64)
nb[okp] = order[pos[okp]]
valid = (uid[hold] != 0) & (nb >= 0)
valid[valid] &= uid[nb[valid]] == uid[hold][valid]
gap = np.where(valid, ts[np.maximum(nb, 0)] - ts[hold], -1)
assert (gap[valid] >= 86400).all()
P(f"holdout feeds with user_id 0: {(uid[hold] == 0).sum()}; kept set S (later train feed >= 24 h): {valid.sum()} "
  f"({valid.mean():.3f}); gap median {np.median(gap[valid]) / 3600:.1f} h, p10 {np.percentile(gap[valid], 10) / 3600:.1f} h, "
  f"p90 {np.percentile(gap[valid], 90) / 3600:.1f} h")

# ---------------- per-candidate features ----------------
hitem = np.load(FLAT / "hist_item_id.npy", mmap_mode="r")
het = np.load(FLAT / "hist_event_type.npy", mmap_mode="r")
hcat = np.load(FLAT / "hist_category_id.npy", mmap_mode="r")
NC = 64
u_fut = np.zeros(len(rows))
u_fut_np = np.zeros(len(rows))
u_past = np.zeros(len(rows))
u_echo = np.zeros(len(rows))
nW = np.zeros(n, np.int64)
nW_np = np.zeros(n, np.int64)
Ha = np.zeros(n, np.int64)
Hb = np.zeros(n, np.int64)
echo_in_W = np.zeros(n, np.int64)  # future-window events on this feed's own positive items
t1 = time.time()
for i in range(n):
    a = hold[i]
    sl = slice(offh[i], offh[i] + Mh[i])
    cc = cat_c[sl]
    yy = y[sl]
    # u_echo (all feeds: needed only on S, but cheap)
    if yy.sum() > 0:
        pc = np.unique(cc[yy == 1])
        u_echo[sl] = np.isin(cc, pc).astype(np.float64)
    ha0, ha1 = hoff_tr[a], hoff_tr[a] + Htr[a]
    Ha[i] = Htr[a]
    if Htr[a] > 0:
        hc_a = np.asarray(hcat[ha0:ha1]).astype(np.int64)
        cnt_a = np.bincount(hc_a, minlength=NC).astype(np.float64)
        u_past[sl] = cnt_a[cc] / Htr[a]
    if not valid[i]:
        continue
    b = nb[i]
    hb0, hb1 = hoff_tr[b], hoff_tr[b] + Htr[b]
    Hb[i] = Htr[b]
    kA = (np.asarray(hitem[ha0:ha1]).astype(np.int64) << 4) | np.asarray(het[ha0:ha1]).astype(np.int64)
    itB = np.asarray(hitem[hb0:hb1]).astype(np.int64)
    kB = (itB << 4) | np.asarray(het[hb0:hb1]).astype(np.int64)
    cB = np.asarray(hcat[hb0:hb1]).astype(np.int64)
    if len(kB) == 0:
        continue
    # occurrence rank of each B event within its key, newest (smallest hpos) first
    srt = np.argsort(kB, kind="stable")
    ks = kB[srt]
    first = np.r_[True, ks[1:] != ks[:-1]]
    grp_start = np.maximum.accumulate(np.where(first, np.arange(len(ks)), 0))
    occ = np.empty(len(kB), np.int64)
    occ[srt] = np.arange(len(ks)) - grp_start
    uk, cntB = np.unique(kB, return_counts=True)
    if len(kA):
        ua, ca = np.unique(kA, return_counts=True)
        j = np.searchsorted(ua, uk)
        jj = np.minimum(j, len(ua) - 1)
        cntA_u = np.where((j < len(ua)) & (ua[jj] == uk), ca[jj], 0)
    else:
        cntA_u = np.zeros(len(uk), np.int64)
    excess_u = np.maximum(cntB - cntA_u, 0)
    excess = excess_u[np.searchsorted(uk, kB)]
    fut = occ < excess
    nW[i] = fut.sum()
    if nW[i] > 0:
        cnt_w = np.bincount(cB[fut], minlength=NC).astype(np.float64)
        u_fut[sl] = cnt_w[cc] / nW[i]
    # info: remove events on this feed's own positive items from the window
    posit = np.unique(item_c[sl][yy == 1])
    onpos = np.isin(itB, posit) & fut
    echo_in_W[i] = onpos.sum()
    fut_np = fut & ~onpos
    nW_np[i] = fut_np.sum()
    if nW_np[i] > 0:
        cnt_w = np.bincount(cB[fut_np], minlength=NC).astype(np.float64)
        u_fut_np[sl] = cnt_w[cc] / nW_np[i]
P(f"features built in {time.time() - t1:.0f}s")
S = valid
P(f"S: {S.sum()} feeds; |W| median {np.median(nW[S]):.0f}, mean {nW[S].mean():.1f}, share |W|=0 {(nW[S] == 0).mean():.3f}, "
  f"share |W|>=10 {(nW[S] >= 10).mean():.3f}; H_a median {np.median(Ha[S]):.0f}, H_b median {np.median(Hb[S]):.0f}")
hasp = np.bincount(fid, y, n) > 0
P(f"S feeds with >=1 positive: {(S & hasp).sum()}; of those, window contains an event on an own positive item: "
  f"{((echo_in_W > 0) & S & hasp).sum()} ({((echo_in_W > 0) & S & hasp).sum() / max((S & hasp).sum(), 1):.3f})")


# ---------------- pool and scoring ----------------
def zf(s):
    s = np.asarray(s, np.float64)
    mu = np.bincount(fid, s, n) / Mh
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / Mh - mu ** 2, 0)) + 1e-9
    return (s - mu[fid]) / sd[fid]


pool = np.mean([zf(np.load(PH / f"nn_{m}_holdout.npy")) for m in G3], 0)
nd_pool = ndcg_per_feed(pool, y, offh, Mh)
P(f"G3 train-only holdout pool NDCG (all 30K): {nd_pool.mean():.5f} (expected 0.32379)")
assert abs(nd_pool.mean() - 0.32379) < 1e-5, "G3 holdout pool does not reproduce"
base = zf(pool)
nd_base = ndcg_per_feed(base, y, offh, Mh)
assert np.allclose(nd_base, nd_pool)
P(f"G3 pool NDCG on S: {nd_base[S].mean():.5f}; off S: {nd_base[~S].mean():.5f}")

feats = {"u_fut": u_fut, "u_past": u_past, "u_echo": u_echo, "u_fut_nopos": u_fut_np}
zfe = {k: zf(v) for k, v in feats.items()}
res = {"n_hold": int(n), "n_S": int(S.sum()), "pool_all": float(nd_pool.mean()), "pool_S": float(nd_base[S].mean())}
d = {}


def stat(x):
    x = np.asarray(x, np.float64)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return m, se, m / se if se > 0 else 0.0


P("\n== paired per-feed NDCG delta vs G3 pool on S (w = weight on zf(feature); w = 0.1 is the pre-declared one)")
for w in [W_MAIN] + W_INFO:
    for k in feats:
        nd = ndcg_per_feed(base + w * zfe[k], y, offh, Mh)
        dd = (nd - nd_base)[S]
        d[(k, w)] = dd
        m, se, z = stat(dd)
        tag = "" if w == W_MAIN and k != "u_fut_nopos" else "  [info]"
        P(f"w {w:4.2f} {k:12s} delta {m:+.5f} +- {se:.5f} (z {z:+.2f}); NDCG on S {nd[S].mean():.5f}; "
          f"feeds up {(dd > 0).mean():.3f} down {(dd < 0).mean():.3f}{tag}")
        res[f"{k}@{w}"] = dict(delta=m, se=se, z=z, ndcg_S=float(nd[S].mean()))

dfp = d[("u_fut", W_MAIN)] - d[("u_past", W_MAIN)]
m_fp, se_fp, z_fp = stat(dfp)
P(f"\ndelta(u_fut) - delta(u_past) @0.1 = {m_fp:+.5f} +- {se_fp:.5f} (z {z_fp:+.2f})")
dfe = d[("u_fut", W_MAIN)] - d[("u_echo", W_MAIN)]
m_fe, se_fe, z_fe = stat(dfe)
P(f"delta(u_fut) - delta(u_echo) @0.1 = {m_fe:+.5f} +- {se_fe:.5f} (z {z_fe:+.2f})")
res["fut_minus_past"] = dict(delta=m_fp, se=se_fp, z=z_fp)
res["fut_minus_echo"] = dict(delta=m_fe, se=se_fe, z=z_fe)

# ---------------- within-feed correlations ----------------
Sc = S[fid]


def wcorr(a_, b_, mask_feeds):
    za, zb = zf(a_), zf(b_)
    va = np.bincount(fid, za * za, n) / Mh > 0.5
    vb = np.bincount(fid, zb * zb, n) / Mh > 0.5
    fm = mask_feeds & va & vb
    cm = fm[fid]
    pooled = float(np.sum(za[cm] * zb[cm]) / np.sqrt(np.sum(za[cm] ** 2) * np.sum(zb[cm] ** 2)))
    per = (np.bincount(fid, za * zb, n) / Mh)[fm]
    return pooled, float(per.mean()), float(np.median(per)), int(fm.sum())


P("\n== within-feed corr (Pearson of per-feed z over candidates; feeds where both vary)")
for a_, b_ in [("u_fut", "u_past"), ("u_fut", "u_echo"), ("u_past", "u_echo"), ("u_fut_nopos", "u_past"),
               ("u_fut", "pool"), ("u_past", "pool"), ("u_echo", "pool")]:
    A_ = feats[a_] if a_ != "pool" else pool
    B_ = feats[b_] if b_ != "pool" else pool
    pc, pm, pmed, nf = wcorr(A_, B_, S)
    P(f"corr({a_}, {b_}): pooled {pc:+.4f}, per-feed mean {pm:+.4f}, median {pmed:+.4f} over {nf} feeds")
    res[f"corr_{a_}_{b_}"] = dict(pooled=pc, mean=pm, median=pmed, n=nf)

# ---------------- info slices ----------------
P("\n== info slices @0.1 (delta +- SE; n feeds)")
sl_defs = {"|W|=0": nW == 0, "|W| 1-9": (nW >= 1) & (nW < 10), "|W| 10-49": (nW >= 10) & (nW < 50),
           "|W|>=50": nW >= 50, "gap 24-48h": (gap >= 86400) & (gap < 2 * 86400),
           "gap 2-4d": (gap >= 2 * 86400) & (gap < 4 * 86400), "gap >=4d": gap >= 4 * 86400,
           "has positive": hasp, "window has own-positive event": echo_in_W > 0,
           "has positive, no own-positive event in window": hasp & (echo_in_W == 0)}
for name, msk in sl_defs.items():
    mS = msk[S]
    if mS.sum() < 30:
        continue
    out = []
    for k in ["u_fut", "u_past", "u_echo", "u_fut_nopos"]:
        m, se, _ = stat(d[(k, W_MAIN)][mS])
        out.append(f"{k} {m:+.5f}+-{se:.5f}")
    P(f"{name:48s} n {mS.sum():6d} | " + " | ".join(out))

# ---------------- decision ----------------
mf, sef, zfu = stat(d[("u_fut", W_MAIN)])
mp = stat(d[("u_past", W_MAIN)])[0]
me = stat(d[("u_echo", W_MAIN)])[0]
c1 = (mf >= 0.005) and (zfu >= 3)
c2 = (mf - mp) >= 0.003
c3 = mf >= 0.30 * me
verdict = "DEFERRED (all three pre-declared conditions hold)" if (c1 and c2 and c3) else \
    "CLOSED for good (category-intent aux family; FUTWIN Step B / PFD unfunded)"
P("\n== DECISION (pre-declared, w = 0.1)")
P(f"(1) delta(u_fut) = {mf:+.5f} >= +0.005 and z = {zfu:+.2f} >= 3: {c1}")
P(f"(2) delta(u_fut) - delta(u_past) = {mf - mp:+.5f} >= +0.003: {c2}")
P(f"(3) delta(u_fut) = {mf:+.5f} >= 0.30 * delta(u_echo) = {0.3 * me:+.5f} (ratio {mf / me if me else float('nan'):.3f}): {c3}")
P(f"VERDICT: {verdict}")
res.update(c1=bool(c1), c2=bool(c2), c3=bool(c3), verdict=verdict, ratio_fut_echo=float(mf / me) if me else None)
P(f"[total {time.time() - t0:.0f}s]")
json.dump(res, open(OUT / "result.json", "w"), indent=1, default=float)
open(OUT / "run_log.txt", "w").write("\n".join(lines) + "\n")
