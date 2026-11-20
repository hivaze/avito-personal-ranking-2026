"""Part 3: exploration traffic. (a) Is production order a function of f1..f5 (train, production vs random-like feeds)?
(b) Is the val/test candidate SET ranker-shaped (positives concentrate at the top of f1 within the feed)? (c) Does the value
of production-order inputs differ Tue vs Thu on val? (d) Label-free drift by day on val and test: pool agreement,
score-vs-f1 agreement, confidence, cold top-1 share; and whether it is explained by cold share. Writes out/s03_exploration.log."""
from lib import *  # noqa: F401,F403
import io
import numba as nb

buf = io.StringIO()


def say(*a):
    print(*a); print(*a, file=buf)


@nb.njit(parallel=True)
def feed_spearman(off, M, a, b):
    out = np.empty(len(M))
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        x = a[o:o + m]; y = b[o:o + m]
        rx = np.argsort(np.argsort(x)).astype(np.float64); ry = np.argsort(np.argsort(y)).astype(np.float64)
        mx = rx.mean(); my = ry.mean()
        sx = np.sqrt(((rx - mx) ** 2).sum()); sy = np.sqrt(((ry - my) ** 2).sum())
        out[f] = ((rx - mx) * (ry - my)).sum() / (sx * sy + 1e-12)
    return out


@nb.njit(parallel=True)
def feed_top1(off, M, a):
    out = np.empty(len(M), np.int64)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        out[f] = o + np.argmax(a[o:o + m])
    return out


@nb.njit(parallel=True)
def feed_rank_of(off, M, a, idx):
    """0-based rank (desc) of candidate idx[f] in feed f under score a"""
    out = np.empty(len(M), np.int64)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        v = a[idx[f]]
        r = 0
        for k in range(m):
            if a[o + k] > v:
                r += 1
        out[f] = r
    return out


dv, dt = descriptors("val"), descriptors("test")
fv, Mv, offv = feed_meta("val"); ft, Mt, offt = feed_meta("test")
y = np.load(FLAT / "val/cand_target.npy")

say("=" * 120)
say("3A. TRAIN: is production position a function of f1..f5? within-feed Spearman(-pos, f_k) on production feeds (p_random < 0.05)")
say("    and random-like feeds (p_random > 0.9), 20K feeds each, M >= 20; plus NDCG of f1 vs the production order on train labels.")
say("=" * 120)
ftr = pl.read_parquet(FLAT / "train/feeds.parquet")
Mtr = ftr["M"].to_numpy().astype(np.int64); otr = ftr["cand_off"].to_numpy().astype(np.int64)
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz"); prand = np.nan_to_num(pr["p_random"].astype(np.float64), nan=0.0)
rng = np.random.default_rng(0)
prod = np.where((prand < 0.05) & (Mtr >= 20))[0]; rnd_ = np.where((prand > 0.9) & (Mtr >= 20))[0]
sel = {"production": rng.choice(prod, 20000, replace=False), "random-like": rng.choice(rnd_, 20000, replace=False)}
pos = np.load(FLAT / "train/cand_pos.npy", mmap_mode="r"); ytr = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
F = {k: np.load(FLAT / f"train/cand_f{k}.npy", mmap_mode="r") for k in range(1, 6)}
for tag, idx in sel.items():
    idx = np.sort(idx)
    rows = np.concatenate([np.arange(otr[i], otr[i] + Mtr[i]) for i in idx])
    M_ = Mtr[idx]; off_ = np.r_[0, np.cumsum(M_)[:-1]]
    p_ = -np.asarray(pos[rows], np.float64); y_ = np.asarray(ytr[rows], np.float64)
    line = f"  {tag:12s} (n {len(idx)}, mean M {M_.mean():.0f}, pos rate {y_.mean():.4f}): "
    for k in range(1, 6):
        fk = np.asarray(F[k][rows], np.float64)
        line += f"rho(-pos,f{k}) {feed_spearman(off_, M_, p_, fk).mean():+.3f}  "
    f1 = np.asarray(F[1][rows], np.float64)
    nd_pos = ndcg_per_feed(p_, y_, off_, M_).mean(); nd_f1 = ndcg_per_feed(f1 + 1e-6 * rng.standard_normal(len(f1)), y_, off_, M_).mean()
    nd_rand = ndcg_per_feed(rng.standard_normal(len(f1)), y_, off_, M_).mean()
    line += f"| NDCG(train labels): production order {nd_pos:.4f}  f1 {nd_f1:.4f}  random {nd_rand:.4f}"
    say(line)
say("  (p_random comes from eda/labels_position: the posterior that a train feed's display order was random; random-like feeds are the holdout pool.)")

say("")
say("=" * 120)
say("3B. VAL/TEST CANDIDATE SET IS RANKER-SHAPED: within-feed percentile of f1 (and f3) of the positives on val, by day; share of")
say("    positives in the top-10% of f1; f1-alone NDCG by day vs random. On test (label-free): the same f1 percentile of G3's top-1.")
say("=" * 120)
f1v = np.load(FLAT / "val/cand_f1.npy").astype(np.float64); f3v = np.load(FLAT / "val/cand_f3.npy").astype(np.float64)
fidv = np.repeat(np.arange(len(Mv)), Mv)
rank_f1 = feed_spearman  # placeholder to keep namespace tidy
from scipy.stats import rankdata  # noqa: E402


def feed_pct(off, M, a):
    out = np.empty(len(a))
    for o, m in zip(off, M):
        out[o:o + m] = (rankdata(-a[o:o + m], method="average") - 1) / max(m - 1, 1)
    return out


pct1 = feed_pct(offv, Mv, f1v); pct3 = feed_pct(offv, Mv, f3v)
day = dv["day"].to_numpy(); posrows = y == 1
rng2 = np.random.default_rng(1)
for dn, dval in [("Tue", 0), ("Thu", 2), ("all", None)]:
    m = np.ones(len(Mv), bool) if dval is None else day == dval
    rm = m[fidv] & posrows
    nd1 = ndcg_per_feed(f1v + 1e-7 * rng2.standard_normal(len(f1v)), y, offv, Mv)[m]
    nd3 = ndcg_per_feed(f3v + 1e-7 * rng2.standard_normal(len(f1v)), y, offv, Mv)[m]
    ndr = ndcg_per_feed(rng2.standard_normal(len(f1v)), y, offv, Mv)[m]
    say(f"  val {dn:4s}: positives' f1 percentile mean {pct1[rm].mean():.3f} (top-10% share {(pct1[rm] <= .1).mean():.3f}); f3 pct {pct3[rm].mean():.3f} | "
        f"NDCG f1 {nd1.mean():.4f}±{nd1.std() / np.sqrt(m.sum()):.4f} f3 {nd3.mean():.4f} random {ndr.mean():.4f}")
nd1_all = ndcg_per_feed(f1v + 1e-7 * rng2.standard_normal(len(f1v)), y, offv, Mv)
dT = nd1_all[day == 2].mean() - nd1_all[day == 0].mean()
say(f"  f1 NDCG Thu - Tue {dT:+.4f} ± {np.sqrt(nd1_all[day == 2].var() / (day == 2).sum() + nd1_all[day == 0].var() / (day == 0).sum()):.4f} (unpaired)")

say("")
say("=" * 120)
say("3C. LABEL-FREE DRIFT BY DAY (val Tue/Thu; test Sat/Sun/Mon/Tue; public/private): mean within-feed Spearman between pools and")
say("    with f1 / f3 / log LUO; G3 top-1 cold share; G3 confidence (top-1 z); plus the per-feed regression on cold share + log M")
say("    (if drift is explained by cold share, pmix already accounts for it).")
say("=" * 120)
S = {}
for nm in ["G3", "M2", "Pall", "Cd", "PF7x", "S2", "TOonly", "G3ft", "G3to"]:
    S[nm] = {"val": pool_scores(nm, "val"), "test": pool_scores(nm, "test")}
f1t = np.load(FLAT / "test/cand_f1.npy").astype(np.float64); f3t = np.load(FLAT / "test/cand_f3.npy").astype(np.float64)
luov = np.load(FC / "val/it_hist_all_luo.npy").astype(np.float64); luot = np.load(FC / "test/it_hist_all_luo.npy").astype(np.float64)
groups = []
di_v = dv["dayidx"].to_numpy(); di_t = dt["dayidx"].to_numpy(); pub = dt["public"].to_numpy() == 1
for u, nm in [(20718, "val Tue"), (20720, "val Thu")]:
    groups.append((nm, "val", di_v == u))
groups.append(("val all", "val", np.ones(len(Mv), bool)))
for u, nm in [(20722, "test Sat"), (20723, "test Sun"), (20724, "test Mon"), (20725, "test Tue")]:
    groups.append((nm, "test", di_t == u))
groups += [("test public", "test", pub), ("test private", "test", ~pub)]
stats = {}
for split, (off, M, f1, f3, luo) in {"val": (offv, Mv, f1v, f3v, luov), "test": (offt, Mt, f1t, f3t, luot)}.items():
    g3 = S["G3"][split]
    st = {}
    for nm in ["M2", "Pall", "Cd", "PF7x", "S2", "TOonly"]:
        st[f"rho(G3,{nm})"] = feed_spearman(off, M, g3, S[nm][split])
    st["rho(FT,TO)"] = feed_spearman(off, M, S["G3ft"][split], S["G3to"][split])
    st["rho(G3,f1)"] = feed_spearman(off, M, g3, f1)
    st["rho(G3,f3)"] = feed_spearman(off, M, g3, f3)
    st["rho(G3,logLUO)"] = feed_spearman(off, M, g3, np.log1p(luo))
    t1 = feed_top1(off, M, g3)
    st["top1 cold"] = (luo[t1] == 0).astype(float)
    st["top1 z"] = g3[t1]
    st["top1 f1pct"] = feed_pct(off, M, f1)[t1]
    st["top1 agree(G3,Pall)"] = (feed_top1(off, M, S["Pall"][split]) == t1).astype(float)
    st["top1 agree(G3,M2)"] = (feed_top1(off, M, S["M2"][split]) == t1).astype(float)
    st["top1 agree(FT,TO)"] = (feed_top1(off, M, S["G3ft"][split]) == feed_top1(off, M, S["G3to"][split])).astype(float)
    stats[split] = st
keys = list(stats["val"].keys())
say(f"  {'stat':22s}" + "".join(f"{g[0]:>13s}" for g in groups))
for k in keys:
    say(f"  {k:22s}" + "".join(f"{stats[g[1]][k][g[2]].mean():13.4f}" for g in groups))
say("  SE of a group mean ~ sd/sqrt(n): e.g. rho(G3,Pall) sd {:.3f} -> SE {:.4f} at n 7000".format(stats["val"]["rho(G3,Pall)"].std(), stats["val"]["rho(G3,Pall)"].std() / np.sqrt(7000)))

say("")
say("  Per-feed regression of each stat on [cold_share, log M, H256, uid0] fitted on VAL and applied to each group's descriptors;")
say("  'resid' = group mean - predicted (0 means the day drift is explained by composition). Columns as above.")
import numpy.linalg as la  # noqa: E402


def design(d):
    return np.column_stack([np.ones(len(d)), d["cold_share"].to_numpy(), np.log(d["M"].to_numpy()), d["H256"].to_numpy(), d["uid0"].to_numpy()])


Xv_, Xt_ = design(dv), design(dt)
for k in keys:
    b, *_ = la.lstsq(Xv_, stats["val"][k], rcond=None)
    line = f"  {k:22s}"
    for g in groups:
        X = Xv_ if g[1] == "val" else Xt_
        line += f"{stats[g[1]][k][g[2]].mean() - (X[g[2]] @ b).mean():+13.4f}"
    say(line + f"   [b cold {b[1]:+.3f} logM {b[2]:+.3f}]")

say("")
say("3D. On val: paired delta (pool - G3) per feed vs label-free disagreement decile (rho(G3,pool)); E|d| rises as rho falls;")
say("    projected |d| scale on public/private from their own rho distribution (the 'DIS' model of pubdiag/noise).")
for nm in ["Pall", "M2", "Cd", "TOonly"]:
    d = per_feed(nm) - per_feed("G3"); rho = stats["val"][f"rho(G3,{nm})"]
    edges = np.quantile(rho, np.linspace(0, 1, 6))
    bins_v = np.clip(np.searchsorted(edges, rho, side="right") - 1, 0, 4)
    rho_t = stats["test"][f"rho(G3,{nm})"]; bins_t = np.clip(np.searchsorted(edges, rho_t, side="right") - 1, 0, 4)
    ed = np.array([d[bins_v == i].mean() for i in range(5)]); ea = np.array([np.abs(d[bins_v == i]).mean() for i in range(5)])
    mix_p = np.bincount(bins_t[pub], minlength=5) / pub.sum(); mix_q = np.bincount(bins_t[~pub], minlength=5) / (~pub).sum()
    say(f"  {nm:6s} E[d|rho quintile] {np.round(ed, 5)} E|d| {np.round(ea, 4)} | quintile mix val uniform, public {np.round(mix_p, 2)}, private {np.round(mix_q, 2)} "
        f"-> projected d public {mix_p @ ed:+.5f} private {mix_q @ ed:+.5f} (val {d.mean():+.5f})")

(OUT / "s03_exploration.log").write_text(buf.getvalue())
print("written", OUT / "s03_exploration.log")
