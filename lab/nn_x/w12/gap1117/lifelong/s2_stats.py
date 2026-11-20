"""Step 2 (label-free): coverage, age gap and parity of the extension blocks of out/align_<split>.npz.
  python s2_stats.py > out_s2_stats.txt
No label is read. Holdout = the fixed 30,000 train feeds (align_lib.Flat.holdout, same draw as train.py --holdout 30000).
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_k] = "12"
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hx_lib import BOUND, KEEP, OUTD, SPL, Flat  # noqa: E402

F = Flat()
hold, _ = F.holdout()
Z = {s: np.load(f"{OUTD}/align_{s}.npz") for s in SPL}
KS = Z["train"]["KS"]
STAT = list(Z["train"]["STAT"])
ix = {k: i for i, k in enumerate(STAT)}


def q(x, p=(5, 25, 50, 75, 95), fmt="{:.1f}"):
    if len(x) == 0:
        return "-"
    return " / ".join(fmt.format(v) for v in np.percentile(x, p))


def ks_stat(a, b):
    """two-sample Kolmogorov-Smirnov distance."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a, b = np.sort(a), np.sort(b)
    g = np.unique(np.r_[a, b])
    return float(np.abs(np.searchsorted(a, g, side="right") / len(a) - np.searchsorted(b, g, side="right") / len(b)).max())


def event_offsets(cnt):
    """cnt [n, nk] -> start offset of the (feed, k) block in the feed-major / k-minor concatenation."""
    flat = cnt.reshape(-1).astype(np.int64)
    off = np.r_[0, np.cumsum(flat)[:-1]]
    return off.reshape(cnt.shape)


def block_types(split, j, feeds_mask):
    """event types / emb-missing / category of the T events of the selected feeds for k index j."""
    z = Z[split]
    cnt = z["cnt"]
    off = event_offsets(cnt)
    sel = np.where(feeds_mask)[0]
    idx = np.concatenate([np.arange(off[i, j], off[i, j] + cnt[i, j]) for i in sel]) if len(sel) else np.zeros(0, np.int64)
    sc, row = z["ev_sc"][idx], z["ev_row"][idx]
    et, emiss = np.zeros(len(idx), np.int64), np.zeros(len(idx), bool)
    emb = {}
    for c in range(3):
        m = sc == c
        if m.any():
            r = row[m]
            o = np.argsort(r)
            tmp = np.empty(len(r), np.int64)
            tmp[o] = np.asarray(F.et[SPL[c]][r[o]])
            et[m] = tmp
            e = np.load(f"<repo>/data/flat/{SPL[c]}/hist_emb_idx.npy", mmap_mode="r")
            tmp2 = np.empty(len(r), np.int64)
            tmp2[o] = np.asarray(e[r[o]])
            emiss[m] = tmp2 < 0
    per_feed = np.repeat(np.arange(len(sel)), cnt[sel, j])
    vshare = np.bincount(per_feed, weights=(et == 1), minlength=len(sel)) / np.maximum(cnt[sel, j], 1)
    return et, emiss, sc, vshare


def stored_view_share(split, feeds):
    """share of views (type 1) in the stored history of each feed (format marker: train-format keeps ~2x the views)."""
    o, h = F.hoff[split][feeds], F.H[split][feeds]
    out = np.zeros(len(feeds))
    for i, (a, b) in enumerate(zip(o, h)):
        out[i] = (np.asarray(F.et[split][a:a + b]) == 1).mean() if b else np.nan
    return out


print("# gap1117 / lifelong: coverage, age gap and parity of the extension blocks (label-free)\n")
SETS = [("train (all)", "train", None), ("holdout (train rows)", "train", hold), ("val", "val", None), ("test (all)", "test", None),
        ("test public-like", "test", "pub"), ("test private-like", "test", "priv")]
summary = {}
for j, k in enumerate(KS):
    print(f"\n## k = {k:g} days: snapshot = most recent admissible strictly earlier snapshot at least {k:g} days older than the feed")
    print("| set | feeds | capped | has snapshot | A (aligned) of has | B (end-anchored) of A | extended (A, B, >= 1 T event) | extended among capped | extended among uncapped | "
          "mean extra events per extended feed (capped / uncapped) | extra events 5/25/50/75/95% (extended) | share with all 128 | mean extra events per feed | gap to snapshot used, days 5/25/50/75/95% (extended feeds) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for nm, split, sub in SETS:
        z = Z[split]
        S, cnt = z["S"], z["cnt"]
        n_all = len(S)
        if sub is None:
            m = np.ones(n_all, bool)
        elif isinstance(sub, str):
            pub = F.ts["test"] < BOUND
            m = pub if sub == "pub" else ~pub
        else:
            m = np.zeros(n_all, bool)
            m[sub] = True
        H = F.H[split]
        cap = H == 256
        has = (S[:, j, ix["has"]] > 0) & m
        A = has & (S[:, j, ix["A"]] > 0)
        B = A & (S[:, j, ix["B"]] > 0)
        ext = B & (cnt[:, j] > 0)
        nE = cnt[:, j]
        gap = S[:, j, ix["gap"]]
        n = m.sum()
        print(f"| {nm} | {n} | {(cap & m).sum() / n:.4f} | {has.sum() / n:.4f} | {A.sum() / max(has.sum(), 1):.3f} | {B.sum() / max(A.sum(), 1):.3f} | "
              f"**{ext.sum() / n:.4f}** ({ext.sum()}) | {(ext & cap).sum() / max((cap & m).sum(), 1):.4f} | {(ext & ~cap).sum() / max((~cap & m).sum(), 1):.4f} | "
              f"{nE[ext].mean() if ext.any() else 0:.1f} ({nE[ext & cap].mean() if (ext & cap).any() else 0:.1f} / {nE[ext & ~cap].mean() if (ext & ~cap).any() else 0:.1f}) | "
              f"{q(nE[ext], fmt='{:.0f}')} | {(nE[ext] == KEEP).mean() if ext.any() else 0:.3f} | {nE[m].sum() / n:.2f} | {q(gap[ext], fmt='{:.2f}')} |")
        summary[(nm, k)] = dict(ext=ext, cap=cap, nE=nE, gap=gap, has=has, A=A, B=B, m=m)
# ---------------------------------------------------------------- gap of the snapshot used, all feeds that have one
print("\n## Age gap between the feed and the snapshot used (all feeds with an admissible snapshot; days; 5/10/25/50/75/90/95%)")
for k in (0.0, 7.0):
    for nm in ("train (all)", "holdout (train rows)", "val", "test (all)", "test private-like"):
        d = summary[(nm, k)]
        print(f"k={k:g} {nm}: has {d['has'].sum()}: {q(d['gap'][d['has']], (5, 10, 25, 50, 75, 90, 95), '{:.2f}')}; "
              f"extended {d['ext'].sum()}: {q(d['gap'][d['ext']], (5, 10, 25, 50, 75, 90, 95), '{:.2f}')}")
# ---------------------------------------------------------------- train coverage by day
print("\n## Train rows: extended share by train day (k = 0 / 3 / 5 / 7)")
t = F.ts["train"]
day = ((t - t.min()) // 86400).astype(int)
rows = []
for dd in range(day.max() + 1):
    md = day == dd
    rows.append(f"d{dd}: " + " / ".join(f"{summary[('train (all)', k)]['ext'][md].mean():.3f}" for k in (0.0, 3.0, 5.0, 7.0)) + f" (n {md.sum()})")
print("; ".join(rows))
# ---------------------------------------------------------------- parity: train(k) vs test(k=0) and test(k)
print("\n## Parity of the extension: train rows built with k vs test rows (KS distance of the per-feed distributions; extended feeds only)")
print("| train k | test k | train extended share | gap KS | extra-events KS | extra events mean train / test | T view share mean train / test | view-share KS | "
      "T events without embedding train / test | stored-history view share KS (format/user mix) | capped share of extended train / test |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
te0 = summary[("test (all)", 0.0)]
cache_types = {}


def types_for(nm, split, k):
    key = (nm, k)
    if key not in cache_types:
        d = summary[key]
        j = list(KS).index(k)
        et, emiss, sc, vsh = block_types(split, j, d["ext"])
        svs = stored_view_share(split, np.where(d["ext"])[0][:: max(1, d["ext"].sum() // 20000)])
        cache_types[key] = (et, emiss, vsh, svs)
    return cache_types[key]


for ktr in (0.0, 3.0, 5.0, 7.0, 10.0):
    for kte in (0.0, ktr) if ktr > 0 else (0.0,):
        tr = summary[("train (all)", ktr)]
        te = summary[("test (all)", kte)]
        et_tr, em_tr, vs_tr, svs_tr = types_for("train (all)", "train", ktr)
        et_te, em_te, vs_te, svs_te = types_for("test (all)", "test", kte)
        print(f"| {ktr:g} | {kte:g} | {tr['ext'].mean():.4f} | {ks_stat(tr['gap'][tr['ext']], te['gap'][te['ext']]):.3f} | "
              f"{ks_stat(tr['nE'][tr['ext']], te['nE'][te['ext']]):.3f} | {tr['nE'][tr['ext']].mean():.1f} / {te['nE'][te['ext']].mean():.1f} | "
              f"{vs_tr.mean():.3f} / {vs_te.mean():.3f} | {ks_stat(vs_tr, vs_te):.3f} | {em_tr.mean():.3f} / {em_te.mean():.3f} | "
              f"{ks_stat(svs_tr, svs_te):.3f} | {tr['cap'][tr['ext']].mean():.3f} / {te['cap'][te['ext']].mean():.3f} |")
# val against test, for reference
for kv in (0.0, 7.0):
    v = summary[("val", kv)]
    et_v, em_v, vs_v, svs_v = types_for("val", "val", kv)
    et_te, em_te, vs_te, svs_te = types_for("test (all)", "test", kv)
    te = summary[("test (all)", kv)]
    print(f"| val {kv:g} | {kv:g} | {v['ext'].mean():.4f} | {ks_stat(v['gap'][v['ext']], te['gap'][te['ext']]):.3f} | "
          f"{ks_stat(v['nE'][v['ext']], te['nE'][te['ext']]):.3f} | {v['nE'][v['ext']].mean():.1f} / {te['nE'][te['ext']].mean():.1f} | "
          f"{vs_v.mean():.3f} / {vs_te.mean():.3f} | {ks_stat(vs_v, vs_te):.3f} | {em_v.mean():.3f} / {em_te.mean():.3f} | {ks_stat(svs_v, svs_te):.3f} | "
          f"{v['cap'][v['ext']].mean():.3f} / {te['cap'][te['ext']].mean():.3f} |")
# ---------------------------------------------------------------- reweighting option: gap-bin importance weights train(k=0) -> test(k=0)
print("\n## Reweighting option: importance weights on train extended feeds (k = 0) so that their gap matches test's (gap bins, days)")
bins = np.array([0, 0.5, 1, 2, 3, 5, 7, 10, 14, 30])
tr, te = summary[("train (all)", 0.0)], summary[("test (all)", 0.0)]
htr = np.histogram(np.clip(tr["gap"][tr["ext"]], 0, 29.9), bins)[0].astype(float)
hte = np.histogram(np.clip(te["gap"][te["ext"]], 0, 29.9), bins)[0].astype(float)
ptr, pte = htr / htr.sum(), hte / hte.sum()
w = np.where(ptr > 0, pte / np.maximum(ptr, 1e-12), 0.0)
print("bin (days) | train share | test share | weight")
for b0, b1, a, b, ww in zip(bins[:-1], bins[1:], ptr, pte, w):
    print(f"[{b0:g}, {b1:g}) | {a:.4f} | {b:.4f} | {ww:.2f}")
wf = w[np.clip(np.digitize(np.clip(tr["gap"][tr["ext"]], 0, 29.9), bins) - 1, 0, len(w) - 1)]
ess = wf.sum() ** 2 / (wf ** 2).sum()
miss = pte[(ptr == 0)].sum()
print(f"effective sample size of the weighted train extended feeds: {ess:.0f} of {len(wf)} ({ess / len(wf):.3f}); "
      f"test mass in bins with no train feed: {miss:.4f} (test gap > {bins[np.where(ptr > 0)[0].max() + 1]:g} d cannot be matched)")
print(f"test extended feeds with gap >= 14 d (beyond every train gap): {(te['gap'][te['ext']] >= 14).mean():.3f}")
np.savez(f"{OUTD}/s2_summary.npz", **{f"{nm}|{k}|{key}": v for (nm, k), d in summary.items() for key, v in d.items() if key in ("ext", "nE", "gap")})
print("\ndone")
