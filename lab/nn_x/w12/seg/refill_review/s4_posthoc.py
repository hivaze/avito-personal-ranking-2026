"""POST-HOC (after the declared read). (i) the study's stored scores of passes I did not re-run, read with my statistics; (ii) stability of sub-rows; (iii) fill buckets."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from rr_lib import HERE, ROOT, cstat, my_ndcg
STUDY = f"{ROOT}/nn_x/w12/seg/refill/out"
MEMBERS = ["w5_e10_kd05w192_s951", "w5_e10_kd05t25w192_s951", "w5_e10_base_s951", "w5_e10_base_s952"]
def take(path, rows):
    x = np.load(path, mmap_mode="r"); o = np.argsort(rows, kind="stable"); out = np.empty(len(rows), x.dtype); out[o] = x[rows[o]]; return out
def f(x, c):
    mu, se = cstat(x, c); return f"{mu:+.5f} +- {se:.5f} (z {mu / se:+.2f}, n {len(x)})"
class St:  # the study's files
    def __init__(self, name):
        split = "train" if name == "holdout" else "val"
        self.m = m = np.load(f"{STUDY}/meta_{name}.npz")
        self.M, self.loff, self.n, self.uid = m["M"], m["loff"], len(m["M"]), m["uid"].astype(np.int64)
        self.y = take(f"{ROOT}/data/flat/{split}/cand_target.npy", m["rows"]).astype(np.float64)
        self.sc = [np.load(f"{STUDY}/scores_{name}_{v}.npz") for v in MEMBERS]
        self.c = {}
    def nd(self, p, who=(0, 1, 2, 3)):
        if (p, who) in self.c: return self.c[(p, who)]
        fi = self.m["feeds_" + p]
        rows = np.concatenate([np.arange(self.loff[k], self.loff[k] + self.M[k]) for k in fi]); Mf = self.M[fi]; fid = np.repeat(np.arange(len(fi)), Mf)
        zs = 0
        for i in who:
            x = self.sc[i][p][rows].astype(np.float64); mu = np.bincount(fid, x) / Mf
            zs = zs + (x - mu[fid]) / (np.sqrt(np.maximum(np.bincount(fid, x * x) / Mf - mu ** 2, 0)) + 1e-6)[fid]
        out = np.full(self.n, np.nan); out[fi] = my_ndcg(zs / len(who), self.y[rows], Mf); self.c[(p, who)] = out; return out
S = {k: St(k) for k in ("holdout", "val")}
print("## (i) the study's stored scores, my statistics, 4-pool (passes I did not re-run)")
for tag in "TU":
    acc = {}
    for name, s in S.items():
        rd = np.zeros(s.n, bool); rd[s.m["qi"][s.m["read" + tag]]] = True
        mp = rd & ~np.isnan(s.nd("dP" + tag))
        acc[name] = dict(c=s.uid[rd], cp=s.uid[mp], d1=(s.nd("d1" + tag) - s.nd("a"))[rd], d2=(s.nd("d2" + tag) - s.nd("a"))[rd], dP=(s.nd("dP" + tag) - s.nd("a"))[mp],
                         c192=(s.nd("c192" + tag) - s.nd("b192"))[rd], a192=(s.nd("a") - s.nd("b192"))[rd])
        a = acc[name]
        print(f"   {tag} {name:8s}: d1 - a {f(a['d1'], a['c'])} | d2 - a {f(a['d2'], a['c'])} | dP - a {f(a['dP'], a['cp'])} | c192 - b192 {f(a['c192'], a['c'])} | a - b192 {f(a['a192'], a['c'])}")
    cat = lambda k, ck="c": (np.concatenate([acc[x][k] for x in acc]), np.concatenate([acc[x][ck] for x in acc]))
    print(f"   {tag} pooled  : d1 - a {f(*cat('d1'))} | d2 - a {f(*cat('d2'))} | dP - a {f(*cat('dP', 'cp'))}")
for name, s in S.items():
    al = np.ones(s.n, bool)
    g, den = (s.nd("g") - s.nd("b")), (s.nd("b192") - s.nd("b"))
    print(f"   calibration g, {name}, all {s.n} capped feeds: g - b {f(g, s.uid)} | b192 - b {f(den, s.uid)} | ratio {g.mean() / den.mean():.2f} | a - b192 {f(s.nd('a') - s.nd('b192'), s.uid)} | a - b {f(s.nd('a') - s.nd('b'), s.uid)}")
print("\n## (iii) c - b by fill bucket, study events, my inference and statistics, 4-pool, holdout + val pooled")
R = {}
for name in ("holdout", "val"):
    m = np.load(f"{HERE}/out/rv_meta_{name}.npz"); split = "train" if name == "holdout" else "val"
    y = take(f"{ROOT}/data/flat/{split}/cand_target.npy", m["rows"]).astype(np.float64)
    sc = [np.load(f"{HERE}/out/rv_scores_{name}_{v}.npz") for v in MEMBERS]
    def nd(p, who=(0, 1, 2, 3), m=m, sc=sc, y=y):
        fi = m["feeds_" + p]; rows = np.concatenate([np.arange(m["c0"][k], m["c0"][k] + m["M"][k]) for k in fi]); Mf = m["M"][fi]; fid = np.repeat(np.arange(len(fi)), Mf)
        zs = 0
        for i in who:
            x = sc[i][p][rows].astype(np.float64); mu = np.bincount(fid, x) / Mf
            zs = zs + (x - mu[fid]) / (np.sqrt(np.maximum(np.bincount(fid, x * x) / Mf - mu ** 2, 0)) + 1e-6)[fid]
        out = np.full(len(m["M"]), np.nan); out[fi] = my_ndcg(zs / len(who), y[rows], Mf); return out
    R[name] = dict(m=m, nd=nd, uid=m["uid"].astype(np.int64))
for k in ("Ts", "Us"):
    row = []
    for lo, hi in ((1, 31), (32, 63), (64, 127), (128, 128)):
        xs, cs = [], []
        for name, r in R.items():
            mk = r["m"]["read_" + k] & (r["m"]["nrec_" + k] >= lo) & (r["m"]["nrec_" + k] <= hi)
            if "cb" + k not in r: r["cb" + k] = r["nd"]("c" + k) - r["nd"]("b")
            xs.append(r["cb" + k][mk]); cs.append(r["uid"][mk])
        row.append(f"{lo}-{hi}: {f(np.concatenate(xs), np.concatenate(cs))}")
    print(f"   {k}: " + " | ".join(row))
print("\n## (ii) stability: hard-label family and 4-pool numerators, study events vs my events, on the feeds common to both read sets (design 1)")
for name, r in R.items():
    m = r["m"]; zs, zm = np.load(f"{STUDY}/align_{name}.npz"), np.load(f"{HERE}/out/myalign_{name}.npz")
    both = m["read_Ts"] & m["read_Tm"]
    same = np.array([np.array_equal(zs["T_sc"][i][:128].astype(np.int64), zm["T_sc"][i]) and np.array_equal(zs["T_row"][i], zm["T_row"][i]) for i in range(len(both))])
    for who, lab in (((0, 1, 2, 3), "pool4"), ((2, 3), "hard"), ((0, 1), "distilled")):
        cs_, cm_, b_ = r["nd"]("cTs", who), r["nd"]("cTm", who), r["nd"]("b", who)
        d = (cs_ - cm_)
        print(f"   {name} {lab:9s}: common feeds {int(both.sum())} (identical recovered events {int((both & same).sum())}); cTs - cTm on identical-event feeds: mean {np.nanmean(d[both & same]):+.6f}, feeds with a different NDCG {int((np.abs(d[both & same]) > 1e-12).sum())}; "
              f"on feeds with different events ({int((both & ~same).sum())}): {f(d[both & ~same], r['uid'][both & ~same])}; contribution to the read-set mean {d[both & ~same].sum() / both.sum():+.5f}")
