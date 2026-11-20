"""Step 6: (I) private-risk slices for the KD files: cold share (train-derived, drifts val -> public -> private), hour, user type.
(J) control-variate test: does a label-free model-expected NDCG delta reduce the SE of a paired val delta? (PPI-type estimator)
Writes out/s6_cold_cv.log."""
from lib import *
import io
buf = io.StringIO()


def say(*a):
    print(*a, flush=True); print(*a, file=buf)


z = np.load(OUT / "nd_val.npz"); names = list(z["names"]); ND = dict(zip(names, z["nd"]))
fv, Mv, ov = meta("val"); pub, tday, Mt = test_parts()
dv = pl.read_parquet(ROOT / "nn_x/audit/valpub/semantics/out/desc_val.parquet"); dt = pl.read_parquet(ROOT / "nn_x/audit/valpub/semantics/out/desc_test.parquet")
assert (dv["M"].to_numpy() == Mv).all() and (dt["M"].to_numpy() == Mt).all()
f = lambda d: f"{d.mean():+.5f}+-{se(d):.5f}" if len(d) > 1 else "n/a"
say("== I. slices of the KD deltas on val, with the slice shares on val / public / private")
cold_v = dv["cold_share"].to_numpy(); cold_t = dt["cold_share"].to_numpy()
SL = [("cold<0.30", lambda c: c < 0.30), ("cold 0.30-0.45", lambda c: (c >= 0.30) & (c < 0.45)), ("cold>=0.45", lambda c: c >= 0.45)]
for a, b in [("W8", "C3"), ("S16", "C3"), ("W8", "G3"), ("W8", "S16"), ("C3", "G3")]:
    d = ND[a] - ND[b]; say(f" {a}-{b}: all {f(d)}")
    for lab, fn in SL:
        m = fn(cold_v); say(f"    {lab:15s} share val {m.mean():.3f} public {fn(cold_t[pub]).mean():.3f} private {fn(cold_t[~pub]).mean():.3f} | delta {f(d[m])}")
    # M x cold post-stratification to public / private (cells with < 30 val feeds are merged into the M margin)
    mb = np.digitize(Mv, [20, 50, 100, 200]); cb = np.digitize(cold_v, [0.30, 0.45]); cell = mb * 3 + cb
    mbt = np.digitize(Mt, [20, 50, 100, 200]); cbt = np.digitize(cold_t, [0.30, 0.45]); cellt = mbt * 3 + cbt
    for lab, msk in (("public", pub), ("private", ~pub)):
        tot = 0.0; var = 0.0
        for c in np.unique(cellt[msk]):
            sh = (cellt[msk] == c).mean(); mm = cell == c
            if mm.sum() < 30:
                mm = mb == c // 3
            tot += sh * d[mm].mean(); var += sh ** 2 * d[mm].var(ddof=1) / mm.sum()
        say(f"    M x cold post-stratified to {lab}: {tot:+.5f}+-{np.sqrt(var):.5f}")
hour = dv["hour"].to_numpy(); tu = dv["train_user"].to_numpy(); u0 = dv["uid0"].to_numpy()
d = ND["W8"] - ND["C3"]
say(" W8-C3 by other descriptors: " + " | ".join([f"hour<10 {f(d[hour < 10])}", f"hour 10-17 {f(d[(hour >= 10) & (hour < 18)])}", f"hour>=18 {f(d[hour >= 18])}", f"train_user {f(d[tu == 1])}", f"new user {f(d[tu == 0])}", f"uid0 {f(d[u0 == 1])}"]))
npos = dv["npos"].to_numpy()
say(" W8-C3 by number of positives: " + " | ".join(f"npos={k} (n {(npos == k).sum()}) {f(d[npos == k])}" for k in (0, 1, 2, 3)) + f" | npos>=4 {f(d[npos >= 4])}")

# ---------------------------------------------------------------- J. control variate
say("\n== J. CONTROL VARIATE (label-free model-expected delta) for paired val deltas")
y = val_y(); fid = np.repeat(np.arange(len(Mv)), Mv)
def ranks(score):
    order = np.lexsort((-score, fid)); r = np.empty(len(score), np.int64); r[order] = np.arange(len(score)) - np.repeat(ov, Mv); return r
def soft(zs, T):
    e = np.exp(T * (zs - np.maximum.reduceat(zs, ov)[fid])); return e / np.add.reduceat(e, ov)[fid]
for a, b, refs in [("W8", "C3", ("S16", "K5")), ("W8", "S16", ("C3", "K5")), ("C3", "G3", ("W8", "S16"))]:
    za = zf("val", final_score(a, "val")); zb = zf("val", final_score(b, "val"))
    ga = 1 / np.log2(ranks(za) + 2.0); gb = 1 / np.log2(ranks(zb) + 2.0)
    d = ND[a] - ND[b]
    for rname in refs + ("avg(a,b)",):
        zr = 0.5 * (za + zb) if rname == "avg(a,b)" else zf("val", final_score(rname, "val"))
        # temperature by single-positive likelihood on val (in-sample for T only; one scalar)
        best = None
        for T in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            p = soft(zr, T); ll = (y * np.log(p + 1e-12)).sum() / y.sum()
            if best is None or ll > best[0]: best = (ll, T, p)
        ll, T, p = best
        cv = np.bincount(fid, p * (ga - gb), len(Mv))            # expected single-positive NDCG delta under the label model
        rho = np.corrcoef(d, cv)[0, 1]; beta = np.cov(d, cv)[0, 1] / cv.var(ddof=1)
        adj = d - beta * (cv - cv.mean())
        say(f"  {a}-{b}, label model = softmax({T} * z[{rname}]): corr(d, expected d) {rho:+.3f}; beta {beta:.2f}; SE {se(d):.5f} -> {se(adj):.5f} (variance reduction {1 - adj.var() / d.var():.3f}); mean expected delta {cv.mean():+.5f} vs realised {d.mean():+.5f}")
say("  reading: variance reduction = corr^2; a per-feed realised NDCG delta is one draw of the clicked item, so its expectation explains almost none of it")
(OUT / "s6_cold_cv.log").write_text(buf.getvalue())
