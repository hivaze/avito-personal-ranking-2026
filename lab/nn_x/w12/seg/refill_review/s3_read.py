"""Reviewer step 3: the labelled read (labels are read HERE only), statistics of DECLARATION block 0.   python s3_read.py > out_s3.txt
"""
import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_k, "12")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

from rr_lib import HERE, ROOT, boot_ratio, cstat, my_ndcg

sys.path.insert(0, f"{ROOT}/src")
from metrics import ndcg_per_feed  # cross-check of my_ndcg only

MEMBERS = ["w5_e10_kd05w192_s951", "w5_e10_kd05t25w192_s951", "w5_e10_base_s951", "w5_e10_base_s952"]
WHO = {"pool4": [0, 1, 2, 3], "distilled": [0, 1], "hard": [2, 3], "m0 kd05w192": [0], "m1 kd05t25w192": [1], "m2 base951": [2], "m3 base952": [3]}
STUDY = f"{ROOT}/nn_x/w12/seg/refill/out"


def mm_rows(path, rows):
    x = np.load(path, mmap_mode="r")
    o = np.argsort(rows, kind="stable")
    out = np.empty(len(rows), x.dtype)
    out[o] = x[rows[o]]
    return out


class S:
    def __init__(self, name):
        self.name = name
        split = "train" if name == "holdout" else "val"
        m = np.load(f"{HERE}/out/rv_meta_{name}.npz")
        self.m = m
        self.M, self.c0, self.n = m["M"], m["c0"], len(m["M"])
        self.y = mm_rows(f"{ROOT}/data/flat/{split}/cand_target.npy", m["rows"]).astype(np.float64)
        self.sc = [np.load(f"{HERE}/out/rv_scores_{name}_{v}.npz") for v in MEMBERS]
        self.clu = m["uid"].astype(np.int64) + (0 if name == "holdout" else 0)  # same user_id = same cluster across sets
        self.cache = {}

    def nd(self, p, who):
        """per-feed NDCG [n] (NaN outside the pass) of the member-equal mean of per-feed score z."""
        if (p, who) in self.cache:
            return self.cache[(p, who)]
        fi = self.m["feeds_" + p]
        rows = np.concatenate([np.arange(self.c0[f], self.c0[f] + self.M[f]) for f in fi])
        Mf = self.M[fi]
        fid = np.repeat(np.arange(len(fi)), Mf)
        zs = 0
        for i in WHO[who]:
            x = self.sc[i][p][rows].astype(np.float64)
            assert not np.isnan(x).any(), (p, who)
            mu = np.bincount(fid, x) / Mf
            sd = np.sqrt(np.maximum(np.bincount(fid, x * x) / Mf - mu ** 2, 0)) + 1e-6
            zs = zs + (x - mu[fid]) / sd[fid]
        out = np.full(self.n, np.nan)
        out[fi] = my_ndcg(zs / len(WHO[who]), self.y[rows], Mf)
        self.cache[(p, who)] = out
        return out

    def d(self, pa, pb, who, mask):
        x = (self.nd(pa, who) - self.nd(pb, who))[mask]
        assert not np.isnan(x).any(), (pa, pb)
        return x


def f(x, clu):
    mu, se = cstat(x, clu)
    return f"{mu:+.5f} +- {se:.5f} (z {mu / se:+.2f}, n {len(x)})" if se > 0 else f"{mu:+.5f} (n {len(x)})"


def rule(lab, num, den, clu, val=None):
    mu, se = cstat(num, clu)
    dm, dse = cstat(den, clu)
    lo, hi = boot_ratio(num, den, clu)
    rho = mu / dm
    s = (f"   {lab:44s} n {len(num):5d} | numerator {mu:+.5f} +- {se:.5f} (z {mu / se:+.2f}) | denominator {dm:+.5f} +- {dse:.5f} (z {dm / dse:+.2f}) | rho {rho:+.3f} [my bootstrap 95% {lo:+.3f}, {hi:+.3f}]"
         f" | numerator upper bound {mu + 1.96 * se:+.5f} vs 25% of denominator {0.25 * dm:+.5f}")
    lab_out = None
    if val is not None:
        vm, vse = cstat(*val)
        opn = rho >= 0.5 and mu / se >= 2 and vm > 0 and dm > 0
        cls = (mu + 1.96 * se) < 0.25 * dm and dm > 0
        lab_out = "OPEN" if opn else ("CLOSE" if cls else "UNDECIDED")
        s += f" | val numerator {vm:+.5f} +- {vse:.5f} | thresholds: {lab_out}"
    print(s, flush=True)
    return mu, se, dm, dse, rho, lab_out


D = {k: S(k) for k in ("holdout", "val")}
print("## 0. integrity (label-free unless stated)")
for name, s in D.items():
    # my NDCG vs src/metrics.py on pass a
    rows_all = np.arange(len(s.y))
    x = s.sc[0]["a"].astype(np.float64)
    mine = my_ndcg(x, s.y, s.M)
    ref = ndcg_per_feed(x, s.y, s.c0, s.M)
    print(f"   {name}: my NDCG vs src/metrics.py on pass a, member 0: max |diff| {np.abs(mine - ref).max():.2e} [labels]; feeds {s.n}; passes {sorted(k[6:] for k in s.m.files if k.startswith('feeds_'))}")
    # my scores vs the study's stored scores for the same passes and candidates
    sm = np.load(f"{STUDY}/meta_{name}.npz")
    pos = np.searchsorted(sm["rows"], s.m["rows"])
    assert (sm["rows"][pos] == s.m["rows"]).all()
    for i, v in enumerate(MEMBERS):
        ss = np.load(f"{STUDY}/scores_{name}_{v}.npz")
        row = []
        for mp, sp in (("a", "a"), ("b", "b"), ("cTs", "cT"), ("cUs", "cU"), ("amTs", "amT"), ("amUs", "amU")):
            a, b = s.sc[i][mp], ss[sp][pos]
            ok = ~np.isnan(a) & ~np.isnan(b)
            row.append(f"{mp}: n {int(ok.sum())}, only-mine {int((~np.isnan(a) & np.isnan(b)).sum())}, only-study {int((np.isnan(a) & ~np.isnan(b)).sum())}, corr {np.corrcoef(a[ok], b[ok])[0, 1]:.6f}, max|d| {np.abs(a[ok] - b[ok]).max():.1e}")
        print(f"   {name} {v}: my scores vs the study's stored scores: " + " | ".join(row))

print("\n## 1. THE RULE on my inference and my statistics")
RES = {}
for k, lab in (("Ts", "V1  design 1 (T), STUDY events, study read set"), ("Tm", "V2  design 1 (T), MY aligner, my read set"),
               ("Us", "    design 2 (U), STUDY events, study read set"), ("Um", "    design 2 (U), MY aligner, my read set")):
    print(f"  -- {lab}")
    for who in ("pool4", "distilled", "hard"):
        parts = {}
        for name, s in D.items():
            mk = s.m["read_" + k]
            parts[name] = (s.d("c" + k, "b", who, mk), s.d("a", "b", who, mk), s.clu[mk])
        r = rule(f"{who} HOLDOUT", *parts["holdout"], val=(parts["val"][0], parts["val"][2]))
        rv = rule(f"{who} val", *parts["val"])
        pn = np.concatenate([parts[x][0] for x in parts])
        pd_ = np.concatenate([parts[x][1] for x in parts])
        pc = np.concatenate([parts[x][2] for x in parts])
        rp = rule(f"{who} holdout + val", pn, pd_, pc)
        RES[(k, who)] = (r, rv, rp)
    print("   single members, numerator c - b (holdout | val): " + " ; ".join(
        f"{w}: " + " | ".join(f"{cstat(D[nm].d('c' + k, 'b', w, D[nm].m['read_' + k]), D[nm].clu[D[nm].m['read_' + k]])[0]:+.5f}" for nm in D) for w in WHO if w.startswith("m")))
    # full fill
    for name, s in D.items():
        mk = s.m["read_" + k] & (s.m["nrec_" + k] >= 128)
        rule(f"pool4 {name} FULL FILL (n = 128)", s.d("c" + k, "b", "pool4", mk), s.d("a", "b", "pool4", mk), s.clu[mk])

print("\n## 2. PLACEBO (my donor draw) and the length-matched stored reference, study events, pool4 (and families)")
for k in ("Ts", "Us"):
    for who in ("pool4", "distilled", "hard"):
        acc = {}
        for name, s in D.items():
            mk = s.m["read_" + k]
            me = mk & ~np.isnan(s.nd("e" + k, who))
            acc[name] = dict(clu=s.clu[me], cb=s.d("c" + k, "b", who, me), eb=s.d("e" + k, "b", who, me), ce=s.d("c" + k, "e" + k, who, me),
                             clu_r=s.clu[mk], cam=s.d("c" + k, "am" + k, who, mk), amb=s.d("am" + k, "b", who, mk))
            print(f"   {k} {who:9s} {name:8s}: e - b {f(acc[name]['eb'], acc[name]['clu'])} | c - e {f(acc[name]['ce'], acc[name]['clu'])} | am - b {f(acc[name]['amb'], acc[name]['clu_r'])} | c - am {f(acc[name]['cam'], acc[name]['clu_r'])}")
        cat = lambda key, ck: (np.concatenate([acc[x][key] for x in acc]), np.concatenate([acc[x][ck] for x in acc]))
        print(f"   {k} {who:9s} pooled  : e - b {f(*cat('eb', 'clu'))} | c - e {f(*cat('ce', 'clu'))} | am - b {f(*cat('amb', 'clu_r'))} | c - am {f(*cat('cam', 'clu_r'))}")

print("\n## 3. ARTEFACT CONTROLS, study events, pool4: bL = length inputs only; cN = recovered events without their item embedding; cX = recovered events that restore items of the removed stored events 129-256 dropped")
for k in ("Ts", "Us"):
    t = k[0]
    acc = {}
    for name, s in D.items():
        mk = s.m["read_" + k]
        nfull, nx = s.m["nval_c" + k] - 128, s.m["nval_cX" + t] - 128
        acc[name] = dict(clu=s.clu[mk], cb=s.d("c" + k, "b", "pool4", mk), bL=s.d("bL" + t, "b", "pool4", mk), cN=s.d("cN" + t, "b", "pool4", mk), ccN=s.d("c" + k, "cN" + t, "pool4", mk),
                         cX=s.d("cX" + t, "b", "pool4", mk), ccX=s.d("c" + k, "cX" + t, "pool4", mk), ab=s.d("a", "b", "pool4", mk))
        a = acc[name]
        print(f"   {k} {name:8s}: c - b {f(a['cb'], a['clu'])} | bL - b {f(a['bL'], a['clu'])} | cN - b {f(a['cN'], a['clu'])} | c - cN {f(a['ccN'], a['clu'])} | cX - b {f(a['cX'], a['clu'])} | c - cX {f(a['ccX'], a['clu'])} "
              f"| events kept by cX {nx.sum()} of {nfull.sum()} ({100 * nx.sum() / nfull.sum():.1f}%)")
        # feeds where cX dropped at least one event
        ch = (nx < nfull)
        mk_i = np.where(mk)[0]
        sub = np.zeros(s.n, bool)
        sub[mk_i[ch]] = True
        print(f"      feeds where cX drops >= 1 event: {int(ch.sum())}: c - b {f(s.d('c' + k, 'b', 'pool4', sub), s.clu[sub])} | cX - b {f(s.d('cX' + t, 'b', 'pool4', sub), s.clu[sub])} | c - cX {f(s.d('c' + k, 'cX' + t, 'pool4', sub), s.clu[sub])}")
    cat = lambda key: (np.concatenate([acc[x][key] for x in acc]), np.concatenate([acc[x]["clu"] for x in acc]))
    print(f"   {k} pooled  : c - b {f(*cat('cb'))} | bL - b {f(*cat('bL'))} | cN - b {f(*cat('cN'))} | c - cN {f(*cat('ccN'))} | cX - b {f(*cat('cX'))} | c - cX {f(*cat('ccX'))}")
    # the rule with cX in place of c
    rule(f"{k} pool4 HOLDOUT with cX in place of c", acc["holdout"]["cX"], acc["holdout"]["ab"], acc["holdout"]["clu"], val=(acc["val"]["cX"], acc["val"]["clu"]))

print("\n## 4. SIZE [D], plain form: mean c - b over the read set x label-free share of feeds in the read set")
sh = {"Ts": dict(val=0.0959, pub=0.0929, pri=0.0815), "Us": dict(val=0.1302, pub=0.1221, pri=0.1087), "Tm": dict(val=0.0955, pub=0.0920, pri=0.0817), "Um": dict(val=0.1295, pub=0.1209, pri=0.1077)}
for k in ("Ts", "Us", "Tm", "Um"):
    row = []
    for src in ("holdout", "val", "pooled"):
        if src == "pooled":
            x = np.concatenate([D[nm].d("c" + k, "b", "pool4", D[nm].m["read_" + k]) for nm in D])
            cl = np.concatenate([D[nm].clu[D[nm].m["read_" + k]] for nm in D])
        else:
            x, cl = D[src].d("c" + k, "b", "pool4", D[src].m["read_" + k]), D[src].clu[D[src].m["read_" + k]]
        mu, se = cstat(x, cl)
        row.append(f"effect from {src} {mu:+.5f} +- {se:.5f} -> private-like {sh[k]['pri'] * mu:+.5f} +- {sh[k]['pri'] * se:.5f}, public-like {sh[k]['pub'] * mu:+.5f}, val {sh[k]['val'] * mu:+.5f}")
    print(f"   {k}: " + " | ".join(row))
