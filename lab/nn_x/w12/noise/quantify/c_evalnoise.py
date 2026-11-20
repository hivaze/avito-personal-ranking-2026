"""(C) evaluation noise: SE of one file's mean NDCG at the val / public / private sizes, and paired SE between files.

Per-feed NDCG on validation; SE at size n = sd / sqrt(n) (checked by bootstrap). The public / private composition is
emulated by re-weighting val feeds to the test part's joint (M bin x cold-share bin) mix (the lab's pmix weights,
re-implemented here from lab/eda/errors/segments, label-free). The label-draw part of each SE comes from labels re-drawn
from the best file's calibrated softmax (out/a_draws_t1.npz for single files; a joint re-draw here for pairs).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT_SEG := Path("<repo>/lab/eda/errors/segments")))
from common import *  # noqa: F401,F403
from segdefs import seg_defs  # noqa: E402

t0 = time.time()
F, M, off = layout()
y = labels()
fid = np.repeat(np.arange(len(M)), M)
npos = np.bincount(fid, y, len(M)).astype(np.int64)
n_val = len(M)


def mix_weights():
    d = pl.read_parquet(ROOT_SEG / "feed_table.parquet")
    te = pl.read_parquet(ROOT_SEG / "test_segments.parquet")
    Ft, Mt, _ = layout("test")
    assert (d["M"].to_numpy() == M).all()
    ts = Ft["timestamp"].to_numpy()[te["fidx"].to_numpy()]
    S, St = seg_defs(d), seg_defs(te, test=True)
    kv = np.char.add(np.char.add(S["M"].astype(str), "|"), S["cold_share"].astype(str))
    kt = np.char.add(np.char.add(St["M"].astype(str), "|"), St["cold_share"].astype(str))
    out = {}
    for name, m in [("public", ts < 1790497920), ("private", ts >= 1790497920)]:
        w = np.zeros(len(kv))
        for k in np.unique(kv):
            w[kv == k] = (kt[m] == k).mean() / (kv == k).mean()
        out[name] = w / w.mean()
        out[name + "_n"] = int(m.sum())
    return out


W = mix_weights()
assert W["public_n"] == N_PUB and W["private_n"] == N_PRIV
print(f"test parts: public {W['public_n']} feeds, private {W['private_n']} feeds; n_eff of val under the public mix "
      f"{W['public'].sum() ** 2 / (W['public'] ** 2).sum():.0f}, private mix {W['private'].sum() ** 2 / (W['private'] ** 2).sum():.0f}")


def wsd(x, w=None):
    if w is None:
        return float(x.std(ddof=1))
    m = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - m) ** 2, weights=w)))


PUBLIC = {"FD16P20_m04_02kd": 0.3952092, "FD16P20kd": 0.394892, "FD24P20kd": 0.394905, "FD48P20kd": 0.39484072,
          "W8P20kd": 0.3950094, "G3": 0.3928199}
files = {k: pool_score(k) for k in PUBLIC}
cb = np.load(PRED / "cb_tr300k_none_val.npy").astype(np.float64)
files["cb_tr300k_none"] = cb
ND = {k: ndcg_per_feed(v, y, off, M) for k, v in files.items()}
for k, v in ND.items():
    print(f"  {k:20s} val {v.mean():.5f}  public-mix {np.average(v, weights=W['public']):.5f}  private-mix "
          f"{np.average(v, weights=W['private']):.5f}  public board {PUBLIC.get(k, float('nan')):.5f}")

# ---------------------------------------------------------------- single file
best = ND[BEST]
rng = np.random.default_rng(0)
B = 4000
boot = np.array([best[rng.integers(0, n_val, n_val)].mean() for _ in range(B)])
print(f"\nsingle file {BEST}: per-feed sd {wsd(best):.4f}; SE at n=14,423 analytic {wsd(best) / np.sqrt(n_val):.5f}, "
      f"bootstrap ({B}) {boot.std(ddof=1):.5f}")
Z = np.load(OUT / "a_draws_t1.npz")
v_draw = Z["v"]                       # per-feed label-draw variance at the calibrated temperature
for name, n, w in [("val", n_val, None), ("public", N_PUB, W["public"]), ("private", N_PRIV, W["private"])]:
    sd = wsd(best, w)
    vd = v_draw.mean() if w is None else np.average(v_draw, weights=w)
    # bootstrap at the target size with composition: resample n feeds with probability proportional to w
    pr = None if w is None else w / w.sum()
    bt = np.array([best[rng.choice(n_val, n, p=pr)].mean() for _ in range(1000)])
    print(f"  {name:8s} n={n:6d}: SE {sd / np.sqrt(n):.5f} (bootstrap {bt.std(ddof=1):.5f}); label-draw part {np.sqrt(vd / n):.5f} "
          f"= {vd / sd ** 2:.3f} of the variance; feed-sampling part {np.sqrt(max(sd ** 2 - vd, 0) / n):.5f}")
se_pub, se_priv = wsd(best, W["public"]) / np.sqrt(N_PUB), wsd(best, W["private"]) / np.sqrt(N_PRIV)
print(f"  public - private of one file (same model, disjoint feeds): sd {np.hypot(se_pub, se_priv):.5f} around the "
      f"composition offset {np.average(best, weights=W['public']) - np.average(best, weights=W['private']):+.5f}")

# ---------------------------------------------------------------- pairs
rks = {k: model_rank(v, off, M) for k, v in files.items()}
x_best = files[BEST] / float(Z["T_A"])


@nb.njit(cache=True)
def joint_redraw(x, rkm, off, M, npos, R, seed, disc):
    """Same label draws (Plackett-Luce from softmax(x), K per feed) scored under several rankings rkm[j]."""
    np.random.seed(seed)
    J = rkm.shape[0]
    Fn = len(M)
    out = np.zeros((R, J))
    idcg = np.zeros(Fn)
    for f in range(Fn):
        for j in range(npos[f]):
            idcg[f] += disc[j]
    for r in range(R):
        for f in range(Fn):
            o = off[f]; m = M[f]; k = npos[f]
            g = np.empty(m)
            for i in range(m):
                g[i] = x[o + i] - np.log(-np.log(np.random.random() + 1e-300) + 1e-300)
            top = np.argsort(-g)[:k]
            for j in range(J):
                d = 0.0
                for i in range(k):
                    d += disc[rkm[j, o + top[i]]]
                out[r, j] += d / idcg[f]
    return out / Fn


keys = list(files)
rkm = np.stack([rks[k] for k in keys])
JR = joint_redraw(x_best, rkm, off, M, npos, 300, 2026, DISC)
print(f"\n(joint re-draw, 300 draws from the best file's softmax, done {time.time() - t0:.0f}s)")
pairs = [(BEST, "FD16P20kd"), ("FD16P20kd", "FD24P20kd"), ("FD16P20kd", "FD48P20kd"), ("FD24P20kd", "FD48P20kd"),
         (BEST, "FD48P20kd"), (BEST, "W8P20kd"), (BEST, "G3"), (BEST, "cb_tr300k_none")]
print("pairs: val diff | corr per-feed | paired SE at val / public(mix) / private(mix) | label-draw part of the val paired SE | public board diff (z at public paired SE)")
for a, b in pairs:
    d = ND[a] - ND[b]
    sv = wsd(d) / np.sqrt(n_val); sp = wsd(d, W["public"]) / np.sqrt(N_PUB); sq = wsd(d, W["private"]) / np.sqrt(N_PRIV)
    ja, jb = keys.index(a), keys.index(b)
    ld = (JR[:, ja] - JR[:, jb]).std(ddof=1)
    pb = PUBLIC.get(a, np.nan) - PUBLIC.get(b, np.nan)
    print(f"  {a:17s} - {b:16s} {d.mean():+.5f} | {np.corrcoef(ND[a], ND[b])[0, 1]:.3f} | {sv:.5f} / {sp:.5f} / {sq:.5f} | "
          f"{ld:.5f} ({(ld / sv) ** 2:.2f} of var) | {pb:+.5f} ({pb / sp:+.1f})")
dd = ND[BEST] - ND["FD16P20kd"]
bt = np.array([dd[rng.integers(0, n_val, n_val)].mean() for _ in range(B)])
print(f"  bootstrap check, {BEST} - FD16P20kd at val size: {bt.std(ddof=1):.5f}")

# the leader's gap on public, read against pair SEs
gap = 0.3982556 - 0.3952092
print(f"\nleader minus pick 1 on public: {gap:+.5f}")
for a, b in [(BEST, "W8P20kd"), (BEST, "G3"), (BEST, "cb_tr300k_none")]:
    d = ND[a] - ND[b]
    sp = wsd(d, W["public"]) / np.sqrt(N_PUB); sq = wsd(d, W["private"]) / np.sqrt(N_PRIV)
    from scipy.stats import norm
    pflip = norm.cdf(-gap / np.hypot(sp, sq))
    print(f"  with the paired SE of a pair as far apart as {a} vs {b} (public {sp:.5f}, private {sq:.5f}): z {gap / sp:.1f}; "
          f"P(private order flips | true gap = public gap) ~ {norm.cdf(-gap / sq):.4f}; P(flip) if the public gap is itself a draw ~ {pflip:.4f}")
print(f"done {time.time() - t0:.0f}s")
