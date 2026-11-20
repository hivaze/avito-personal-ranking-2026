"""Stage 1: streaming per-feature screen. For every generated feature:
  - inner-CV binned Newton gain (1-D and x base-rank group) inside each user half and on full val (selection scores)
  - nested linear tilt  base_z + w * z_feed(x): w chosen on one user half by NDCG (grid incl. 0), scored on the other
  - nested binned tilt  base_z + eta * d[bin(x)] / T  (1-D GAM step), same nesting
  - holdout read of both tilts, fitted on full val
usage: screen.py <tag> <fams comma> [shard nsh]"""
import sys, time, json
import numpy as np
from lib import *
from lib import _rk, _z

tag = sys.argv[1]; fams = sys.argv[2].split(",")
pairsel = None
if len(sys.argv) > 3:
    sh, nsh = int(sys.argv[3]), int(sys.argv[4])
    pairsel = lambda i: i % nsh == sh
load_stats()
V = DS("val"); Hd = DS("hold")
WG = np.array([-0.16, -0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08, 0.16], np.float32)
ETA = [0.25, 0.5, 1.0]
NB = 24


def fitT(D, mask_f):
    best = (1e30, 1.0)
    for T in np.arange(0.6, 4.01, 0.1):
        _, _, loss = softmax_gh(D.off, D.M, (T * D.zb).astype(np.float32), D.y)
        l = loss[mask_f].sum()
        if l < best[0]:
            best = (l, float(T))
    return best[1]


hA = V.half == 0; hB = ~hA
TA, TB = fitT(V, hA), fitT(V, hB)
Trow = np.where(np.repeat(hA, V.M), TA, TB).astype(np.float32)
g, h, _ = softmax_gh(V.off, V.M, Trow * V.zb, V.y)
g = g.astype(np.float64); h = h.astype(np.float64)
qf = (V.half * 2 + V.inner).astype(np.int64); qrow = np.repeat(qf, V.M)
rkb = _rk(V.off, V.M, -V.zb) * np.repeat(np.maximum(V.M - 1, 1), V.M)
rg = np.digitize(rkb, [2.5, 9.5, 29.5]).astype(np.int64)
qr = qrow * 4 + rg
base_v = V.ndcg(V.zb); base_h = Hd.ndcg(Hd.zb)
nA, nB = hA.sum(), hB.sum()
print(f"[{tag}] T {TA:.2f}/{TB:.2f} base val {base_v.mean():.5f} hold {base_h.mean():.5f}", flush=True)


def cvgain(Gp, Hp, Gq, Hq, lam):
    d = Gp / (Hp + lam)
    return float((d * Gq - 0.5 * d * d * Hq).sum())


def one(x, xh):
    r = {}
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    xh = np.nan_to_num(xh, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    # ---- bins
    ed = np.unique(np.quantile(x[::7], np.linspace(0, 1, NB + 1)[1:-1]))
    idx = np.searchsorted(ed, x, side="right")
    G = np.bincount(qr * NB + idx, weights=g, minlength=16 * NB).reshape(4, 4, NB)
    Hs = np.bincount(qr * NB + idx, weights=h, minlength=16 * NB).reshape(4, 4, NB)
    lam = 0.05 * Hs.sum() / (4 * NB) + 2.0
    G1, H1 = G.sum(1), Hs.sum(1)  # [quarter, bin]
    for nm, (p, q) in {"A": (0, 1), "B": (2, 3)}.items():
        r[f"cv1_{nm}"] = cvgain(G1[p], H1[p], G1[q], H1[q], lam) + cvgain(G1[q], H1[q], G1[p], H1[p], lam)
        r[f"cv2_{nm}"] = cvgain(G[p], Hs[p], G[q], Hs[q], lam / 2) + cvgain(G[q], Hs[q], G[p], Hs[p], lam / 2)
    GA, HA, GB, HB = G[:2].sum(0), Hs[:2].sum(0), G[2:].sum(0), Hs[2:].sum(0)
    r["cv1_F"] = cvgain(GA.sum(0), HA.sum(0), GB.sum(0), HB.sum(0), lam) + cvgain(GB.sum(0), HB.sum(0), GA.sum(0), HA.sum(0), lam)
    r["cv2_F"] = cvgain(GA, HA, GB, HB, lam / 2) + cvgain(GB, HB, GA, HA, lam / 2)
    # ---- linear tilt
    fz = _z(V.off, V.M, x)
    if float(np.abs(fz).max()) < 1e-6:
        r.update(lin_d=0.0, lin_se=0.0, lin_z=0.0, lin_h=0.0, lin_hz=0.0, bin_d=0.0, bin_se=0.0, bin_z=0.0, bin_h=0.0, bin_hz=0.0, wA=0.0, wB=0.0, wF=0.0, const=1)
        return r
    Dm = np.stack([V.ndcg(V.zb + w * fz) - base_v for w in WG])
    mA = Dm[:, hA].mean(1); mB = Dm[:, hB].mean(1); mF = Dm.mean(1)
    d = np.zeros(V.nf)
    iA, iB, iF = mA.argmax(), mB.argmax(), mF.argmax()
    wA = WG[iA] if mA[iA] > 0 else 0.0; wB = WG[iB] if mB[iB] > 0 else 0.0; wF = WG[iF] if mF[iF] > 0 else 0.0
    if wA != 0: d[hB] = Dm[iA, hB]
    if wB != 0: d[hA] = Dm[iB, hA]
    se = d.std() / np.sqrt(V.nf)
    r.update(lin_d=float(d.mean()), lin_se=float(se), lin_z=float(d.mean() / se) if se > 0 else 0.0, wA=float(wA), wB=float(wB), wF=float(wF),
             lin_ins=float(mF.max()), linA=float(mA.max()), linB=float(mB.max()))
    if wF != 0:
        dh = Hd.ndcg(Hd.zb + wF * _z(Hd.off, Hd.M, xh)) - base_h
        r.update(lin_h=float(dh.mean()), lin_hz=float(dh.mean() / (dh.std() / np.sqrt(Hd.nf) + 1e-12)))
    else:
        r.update(lin_h=0.0, lin_hz=0.0)
    # ---- binned tilt (1-D)
    d = np.zeros(V.nf)
    for (Gf, Hf, T, fit, ev) in ((GA.sum(0), HA.sum(0), TA, hA, hB), (GB.sum(0), HB.sum(0), TB, hB, hA)):
        db = (Gf / (Hf + lam)).astype(np.float32)[idx] / T
        best = (0.0, None)
        for eta in ETA:
            dd = V.ndcg(V.zb + eta * db) - base_v
            if dd[fit].mean() > best[0]:
                best = (dd[fit].mean(), dd)
        if best[1] is not None:
            d[ev] = best[1][ev]
    se = d.std() / np.sqrt(V.nf)
    r.update(bin_d=float(d.mean()), bin_se=float(se), bin_z=float(d.mean() / se) if se > 0 else 0.0)
    Gf, Hf = (GA + GB).sum(0), (HA + HB).sum(0)
    dbv = (Gf / (Hf + lam)).astype(np.float32)
    best = (0.0, 0.0)
    for eta in ETA:
        m = (V.ndcg(V.zb + eta * dbv[idx] / (0.5 * (TA + TB))) - base_v).mean()
        if m > best[0]:
            best = (m, eta)
    if best[1] > 0:
        ih = np.searchsorted(ed, xh, side="right")
        dh = Hd.ndcg(Hd.zb + best[1] * dbv[ih] / (0.5 * (TA + TB))) - base_h
        r.update(bin_h=float(dh.mean()), bin_hz=float(dh.mean() / (dh.std() / np.sqrt(Hd.nf) + 1e-12)))
    else:
        r.update(bin_h=0.0, bin_hz=0.0)
    return r


def stream():
    gv = gen(V, fams, pairsel=pairsel); gh = gen(Hd, fams, pairsel=pairsel)
    for (fam, n, x), (_, n2, xh) in zip(gv, gh):
        assert n == n2
        yield fam, n, x, xh
    if "CTRL" in fams:  # label-free noise controls: N(0,1) and within-feed permutations of real inputs
        rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
        for i in range(int(sys.argv[5]) if len(sys.argv) > 5 else 200):
            if i % 2 == 0:
                yield "CTRL", f"noise{tag}{i}", rng.standard_normal(V.n).astype(np.float32), rng.standard_normal(Hd.n).astype(np.float32)
            else:
                b = BASE[(i // 2) % len(BASE)]
                def perm(D):
                    key = D.fid + rng.random(D.n)
                    return D.get(b)[np.argsort(key, kind="stable")]
                yield "CTRL", f"perm{tag}{i}({b})", perm(V), perm(Hd)


out = open(OUT / f"{SPREF}_{tag}.jsonl", "w")
t0 = time.time(); k = 0
for fam, n, x, xh in stream():
    r = one(x, xh); r.update(fam=fam, name=n)
    out.write(json.dumps(r) + "\n"); out.flush(); k += 1
    if k % 50 == 0:
        print(f"[{tag}] {k} feats {time.time() - t0:.0f}s last {n} lin_z {r['lin_z']:.2f} bin_z {r['bin_z']:.2f}", flush=True)
print(f"[{tag}] done {k} feats {time.time() - t0:.0f}s", flush=True)
