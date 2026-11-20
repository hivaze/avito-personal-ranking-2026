"""Step 2b (label-free): (i) positive check of the single-snapshot alignment (cap pretended at 192, the snapshot cut to 192 too;
truth = stored events 193-256), 3,000 capped feeds per split; (ii) parity rows restricted to capped feeds; (iii) dose-bin
reweighting of train extended feeds to the test dose; (iv) the generic train / test history-format difference over ALL capped
feeds (to separate it from anything specific to the extension). No label is read.
  python s2b_parity_extra.py > out_s2b_parity_extra_v2.txt  (v2: placement error = events not matched to the truth that are stored among the newest 192, as in the refill; v1 output counted every repeat)
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_k] = "12"
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hx_lib import DATA_OUT, SPL, Flat, pick_snapshot, union_timeline  # noqa: E402

F = Flat()
hold, _ = F.holdout()
rng = np.random.default_rng(20261117)


def ks_stat(a, b):
    a, b = np.sort(a), np.sort(b)
    g = np.unique(np.r_[a, b])
    return float(np.abs(np.searchsorted(a, g, side="right") / len(a) - np.searchsorted(b, g, side="right") / len(b)).max())


def inter(a, b):
    ua, ca = np.unique(a, return_counts=True)
    ub, cb = np.unique(b, return_counts=True)
    _, xa, xb = np.intersect1d(ua, ub, return_indices=True)
    return int(np.minimum(ca[xa], cb[xb]).sum())


print("## (i) positive check of the single-snapshot aligner (cap pretended 192; snapshot also cut to 192; truth = stored 193-256)")
for nm, split, pool in (("holdout", "train", hold), ("train (non-holdout)", "train", None), ("val", "val", None), ("test", "test", None)):
    H, u = F.H[split], F.uid[split]
    if pool is None:
        pool = np.arange(len(H))
    c = pool[(H[pool] == 256) & (u[pool] != 0)]
    c = rng.permutation(c)
    rows, done = [], 0
    for f in c:
        snap = pick_snapshot(F, split, f, 0.0)
        if snap is None:
            continue
        done += 1
        U, st = union_timeline(F, split, f, cap=192, cap_snap=192, snaps=[(snap[0], snap[1])])
        if st["n_merged"] < 1 or st["deep_anchor"] < 184 or st["n_T"] < 1:
            rows.append([0, 0, 0, 0, 0])
        else:
            rec = U["key"][st["iL"] + 1:][:64]
            truth_all, _ = F.seq(SPL.index(split), f)
            truth = truth_all[192:256]
            newest = truth_all[:192]
            m = len(rec)
            # placement error: recovered events left after removing those matched to the truth 193-256 (multiset), that are stored among the newest 192
            ur, cr = np.unique(rec, return_counts=True)
            ut, ct = np.unique(truth, return_counts=True)
            left = cr.copy()
            _, xr, xt_ = np.intersect1d(ur, ut, return_indices=True)
            left[xr] -= np.minimum(cr[xr], ct[xt_])
            rest = np.repeat(ur, left)
            rows.append([m, int((rec == truth[:m]).sum()), inter(rec, truth), inter(rest, newest), int(m == 64)])
        if done >= 3000:
            break
    r = np.array(rows)
    ok = r[:, 0] > 0
    ev = r[ok, 0].sum()
    print(f"{nm}: feeds with a snapshot {done}; with >= 1 recovered event {ok.sum()}; events {ev}: exact position {r[ok, 1].sum() / ev:.3f}; "
          f"true event 193-256 (any order) {r[ok, 2].sum() / ev:.3f}; NOT a true event 193-256 but stored among the newest 192 (placement error) {r[ok, 3].sum() / ev:.3f}")

print("\n## (ii) parity restricted to capped feeds (H = 256), k = 0 and k = 7, train vs test (extended feeds; written arrays)")


def load(split, var):
    d = f"{DATA_OUT}/{split}" + ("" if var == "k0" else f"/{var}")
    n = np.load(f"{d}/ext_n.npy").astype(np.int64)
    off = np.load(f"{d}/ext_off.npy")
    et = np.load(f"{d}/ext_event_type.npy")
    emb = np.load(f"{d}/ext_emb_idx.npy")
    gap = np.load(f"{d}/ext_gap_days.npy")
    cap = np.load(f"{d}/ext_capped.npy")
    fe = np.repeat(np.arange(len(n)), n)
    vs = np.bincount(fe, weights=(et == 1), minlength=len(n)) / np.maximum(n, 1)
    ms = np.bincount(fe, weights=(emb < 0), minlength=len(n)) / np.maximum(n, 1)
    return dict(n=n, vs=vs, ms=ms, gap=gap, cap=cap)


Dd = {(s, v): load(s, v) for s in SPL for v in ("k0", "k7")}
n_tr = len(F.H["train"])
print("| train var | test var | train extended share (all feeds) | of capped | dose KS | dose mean train / test | gap KS | view-share KS | view share train / test | no-embedding share train / test |")
print("|---|---|---|---|---|---|---|---|---|---|")
for vtr, vte in (("k0", "k0"), ("k7", "k0"), ("k7", "k7")):
    a, b = Dd[("train", vtr)], Dd[("test", vte)]
    ea, eb = (a["n"] > 0) & a["cap"], (b["n"] > 0) & b["cap"]
    print(f"| {vtr} capped | {vte} capped | {ea.mean():.4f} | {ea.sum() / a['cap'].sum():.4f} | {ks_stat(a['n'][ea], b['n'][eb]):.3f} | "
          f"{a['n'][ea].mean():.1f} / {b['n'][eb].mean():.1f} | {ks_stat(a['gap'][ea], b['gap'][eb]):.3f} | {ks_stat(a['vs'][ea], b['vs'][eb]):.3f} | "
          f"{a['vs'][ea].mean():.3f} / {b['vs'][eb].mean():.3f} | {a['ms'][ea].mean():.3f} / {b['ms'][eb].mean():.3f} |")
for vv in ("k0",):
    a, b = Dd[("val", vv)], Dd[("test", vv)]
    ea, eb = (a["n"] > 0) & a["cap"], (b["n"] > 0) & b["cap"]
    print(f"| val {vv} capped | {vv} capped | {ea.mean():.4f} | {ea.sum() / a['cap'].sum():.4f} | {ks_stat(a['n'][ea], b['n'][eb]):.3f} | "
          f"{a['n'][ea].mean():.1f} / {b['n'][eb].mean():.1f} | {ks_stat(a['gap'][ea], b['gap'][eb]):.3f} | {ks_stat(a['vs'][ea], b['vs'][eb]):.3f} | "
          f"{a['vs'][ea].mean():.3f} / {b['vs'][eb].mean():.3f} | {a['ms'][ea].mean():.3f} / {b['ms'][eb].mean():.3f} |")
for nm, sp, sub in (("holdout", "train", hold),):
    a = Dd[(sp, "k0")]
    m = np.zeros(len(a["n"]), bool)
    m[sub] = True
    e = (a["n"] > 0) & a["cap"] & m
    print(f"holdout capped-only k0: extended {e.sum()} = {e.sum() / m.sum():.4f} of holdout feeds, {e.sum() / (a['cap'] & m).sum():.4f} of capped holdout feeds")
for sp in ("val", "test"):
    a = Dd[(sp, "k0")]
    e = (a["n"] > 0) & a["cap"]
    print(f"{sp} capped-only k0: extended {e.sum()} = {e.mean():.4f} of feeds, {e.sum() / a['cap'].sum():.4f} of capped feeds; mean events {a['n'][e].mean():.1f}")
pub = F.ts["test"] < 1790497920
a = Dd[("test", "k0")]
for nm, m in (("public-like", pub), ("private-like", ~pub)):
    e = (a["n"] > 0) & a["cap"] & m
    print(f"test {nm} capped-only k0: {e.sum() / m.sum():.4f} of feeds")

print("\n## (iii) dose-bin reweighting of train capped extended feeds (k = 0) to the test dose distribution")
a, b = Dd[("train", "k0")], Dd[("test", "k0")]
ea, eb = (a["n"] > 0) & a["cap"], (b["n"] > 0) & b["cap"]
bins = np.array([1, 8, 16, 32, 64, 96, 128, 129])
ha = np.histogram(a["n"][ea], bins)[0] / ea.sum()
hb = np.histogram(b["n"][eb], bins)[0] / eb.sum()
w = hb / np.maximum(ha, 1e-12)
print("dose bin | train | test | weight")
for b0, b1, x, y, ww in zip(bins[:-1], bins[1:], ha, hb, w):
    print(f"[{b0}, {b1}) | {x:.4f} | {y:.4f} | {ww:.2f}")
wf = w[np.clip(np.digitize(a["n"][ea], bins) - 1, 0, len(w) - 1)]
print(f"ESS {wf.sum() ** 2 / (wf ** 2).sum() / len(wf):.3f} of {len(wf)} feeds; weights {w.min():.2f}..{w.max():.2f}")

print("\n## (iv) generic history-format difference over ALL capped feeds (not specific to the extension): stored view share")
for vs_sp in ("train", "val", "test"):
    H = F.H[vs_sp]
    c = np.where(H == 256)[0]
    c = rng.choice(c, min(20000, len(c)), replace=False)
    sh = np.array([(np.asarray(F.et[vs_sp][F.hoff[vs_sp][f]:F.hoff[vs_sp][f] + 256]) == 1).mean() for f in c])
    Dd[("svs", vs_sp)] = sh
    print(f"{vs_sp}: all capped feeds, stored view share mean {sh.mean():.3f}, 5/50/95% {np.percentile(sh, [5, 50, 95]).round(3).tolist()}")
print(f"KS train vs test (all capped feeds) {ks_stat(Dd[('svs', 'train')], Dd[('svs', 'test')]):.3f}; val vs test {ks_stat(Dd[('svs', 'val')], Dd[('svs', 'test')]):.3f}")
print("done")
