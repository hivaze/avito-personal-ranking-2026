"""Step 3 / 4: read of the refill replay by the statistics of DECLARATION.txt block 0 (labels are read HERE only).
  python s3_read.py > out_s3.txt
"""
import json
import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_k, "12")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

ROOT = "<repo>"
sys.path.insert(0, f"{ROOT}/src")
from metrics import ndcg_per_feed  # noqa: E402

from align_lib import BOUND, OUT, Flat  # noqa: E402

MEMBERS = ["w5_e10_kd05w192_s951", "w5_e10_kd05t25w192_s951", "w5_e10_base_s951", "w5_e10_base_s952"]
WHO = {"pool4": [0, 1, 2, 3], "distilled": [0, 1], "hard": [2, 3], "m:kd05w192_s951": [0], "m:kd05t25w192_s951": [1], "m:base_s951": [2], "m:base_s952": [3]}
BUCK = [("1-31", 1, 31), ("32-63", 32, 63), ("64-127", 64, 127), ("128", 128, 10 ** 9)]
NB = 2000
RES = {}


def take(path, rows):
    x = np.load(path, mmap_mode="r")
    o = np.argsort(rows, kind="stable")
    out = np.empty(len(rows), x.dtype)
    out[o] = x[rows[o]]
    return out


class Set:
    def __init__(self, name):
        self.name = name
        split = "train" if name == "holdout" else "val"
        m = np.load(OUT / f"meta_{name}.npz")
        self.m = m
        self.M, self.loff, self.uid, self.n = m["M"], m["loff"], m["uid"].astype(np.int64), len(m["M"])
        self.y = take(f"{ROOT}/data/flat/{split}/cand_target.npy", m["rows"]).astype(np.float64)
        self.fid = np.repeat(np.arange(self.n), self.M)
        self.sc = [np.load(OUT / f"scores_{name}_{v}.npz") for v in MEMBERS]
        self.passes = [k[6:] for k in m.files if k.startswith("feeds_")]
        # clusters: user; every user_id-0 feed its own cluster. Offset so that holdout and val share user clusters.
        self.clu = np.where(self.uid == 0, -(np.arange(self.n) + 1) - (10 ** 9 if name == "val" else 0), self.uid)
        self.N = {}
        self.qi = m["qi"]
        self.nf = {"T": np.full(self.n, 0), "U": np.full(self.n, 0)}
        self.read = {"T": np.zeros(self.n, bool), "U": np.zeros(self.n, bool)}
        for tag in "TU":
            self.nf[tag][self.qi] = np.minimum(m["n" + tag], 128)
            self.read[tag][self.qi[m["read" + tag]]] = True

    def z(self, s, lf):
        """per-feed z of flat scores restricted to feeds lf (blend.py feed_z)."""
        rows = np.concatenate([np.arange(self.loff[f], self.loff[f] + self.M[f]) for f in lf])
        x = s[rows].astype(np.float64)
        assert not np.isnan(x).any()
        M = self.M[lf]
        fid = np.repeat(np.arange(len(lf)), M)
        mu = np.bincount(fid, x, len(lf)) / M
        sd = np.sqrt(np.maximum(np.bincount(fid, x * x, len(lf)) / M - mu ** 2, 0)) + 1e-6
        return (x - mu[fid]) / sd[fid], rows, M

    def ndcg(self, p, who):
        """per-feed NDCG [n] (NaN outside the pass) of the member-equal z-mean of `who` on pass p."""
        key = (p, who)
        if key not in self.N:
            lf = self.m["feeds_" + p]
            zs = None
            for i in WHO[who]:
                zi, rows, M = self.z(self.sc[i][p], lf)
                zs = zi if zs is None else zs + zi
            offs = np.r_[0, np.cumsum(M)[:-1]]
            out = np.full(self.n, np.nan)
            out[lf] = ndcg_per_feed(zs / len(WHO[who]), self.y[rows], offs, M)
            self.N[key] = out
        return self.N[key]


def cstat(x, clu):
    """mean, cluster-robust SE, n."""
    n = len(x)
    if n == 0:
        return np.nan, np.nan, 0
    mu = x.mean()
    _, inv = np.unique(clu, return_inverse=True)
    r = np.bincount(inv, weights=x - mu)
    g = len(r)
    return mu, float(np.sqrt((r ** 2).sum() * g / max(g - 1, 1)) / n), n


def boot_ratio(num, den, clu, seed=11):
    """percentile 95% interval of mean(num) / mean(den), users resampled."""
    _, inv = np.unique(clu, return_inverse=True)
    g = inv.max() + 1
    sn, sd = np.bincount(inv, weights=num, minlength=g), np.bincount(inv, weights=den, minlength=g)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(g, np.ones(g) / g, size=NB).astype(np.float64)
    r = (w @ sn) / (w @ sd)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)), float((w @ sd <= 0).mean())


def f(mu, se):
    return f"{mu:+.5f} +- {se:.5f} (z {mu / se:+.2f})" if se and se > 0 else f"{mu:+.5f}"


def diff(S, pa, pb, who, mask):
    x = (S.ndcg(pa, who) - S.ndcg(pb, who))[mask]
    assert not np.isnan(x).any(), (pa, pb)
    return x


def cat(parts):
    """concatenate (values, clusters) over sets."""
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def rule_line(tag, name, num, den, clu, val_num=None, label=""):
    mu, se, n = cstat(num, clu)
    dm, dse, _ = cstat(den, clu)
    lo, hi, bad = boot_ratio(num, den, clu)
    rho = mu / dm
    s = (f"   {label:34s} n {n:5d} | numerator {f(mu, se)} | denominator {f(dm, dse)} | rho {rho:+.3f} [boot 95% {lo:+.3f}, {hi:+.3f}]"
         f" | upper bound of the numerator {mu + 1.96 * se:+.5f} (1.645: {mu + 1.645 * se:+.5f}) vs 25% of the denominator {0.25 * dm:+.5f}")
    verdict = None
    if val_num is not None:
        vm, vse, vn = val_num
        opn = (rho >= 0.5) and (mu / se >= 2) and (vm > 0) and dm > 0
        cls = (mu + 1.96 * se) < 0.25 * dm and dm > 0
        verdict = "OPEN" if opn else ("CLOSE" if cls else "UNDECIDED")
        s += f" | val numerator {f(vm, vse)} (n {vn}) | by the rule's thresholds: {verdict}"
    print(s)
    return dict(n=n, num=mu, num_se=se, den=dm, den_se=dse, rho=rho, lo=lo, hi=hi, verdict=verdict)


def main():
    S = {k: Set(k) for k in ("holdout", "val")}
    # ---------------- sanity: the review's truncation costs on all capped val feeds (hard-label members base s951, s952)
    print("## 0. Sanity [M]: truncation cost on ALL capped feeds (pass b / b192 minus a), compare review.md section 3.1")
    for name in ("val", "holdout"):
        s = S[name]
        allm = np.ones(s.n, bool)
        for who in ("hard", "distilled", "pool4"):
            row = []
            for p in ("b192", "b"):
                if who == "pool4":
                    x = diff(s, p, "a", who, allm)
                else:  # the review reports the mean of the members' own NDCG deltas
                    x = np.mean([diff(s, p, "a", w, allm) for w in WHO if w.startswith("m:") and WHO[w][0] in WHO[who]], 0)
                row.append(f"{'K=192' if p == 'b192' else 'K=128'} {f(*cstat(x, s.clu)[:2])}")
            print(f"   {name:8s} {who:10s} ({'pool' if who == 'pool4' else 'mean of member deltas'}) n {s.n}: " + " | ".join(row))
        print(f"   {name}: NDCG of pass a on all capped feeds: pool4 {np.nanmean(s.ndcg('a', 'pool4')):.5f}; members " + ", ".join(f"{np.nanmean(s.ndcg('a', w)):.5f}" for w in WHO if w.startswith('m:')))
    # ---------------- calibration g on all capped feeds
    print("\n## 1. Calibration g [M]: newest 128 + stored events 193-256 shown in slots 129-192, against the true newest 192 (rho_g = [g - b] / [b192 - b])")
    for name in ("holdout", "val"):
        s = S[name]
        for lab, mask in (("all capped", np.ones(s.n, bool)), ("read set U", s.read["U"]), ("read set T", s.read["T"])):
            for who in ("pool4", "distilled", "hard"):
                num, den = diff(s, "g", "b", who, mask), diff(s, "b192", "b", who, mask)
                r = rule_line("g", name, num, den, s.clu[mask], label=f"{name} {lab} {who}")
                if who == "pool4":
                    RES[f"g_{name}_{lab}"] = r
            x = diff(s, "g", "b192", "pool4", mask)
            print(f"   {name} {lab}: g - b192 (pool4) {f(*cstat(x, s.clu[mask])[:2])}; a - b192 {f(*cstat(diff(s, 'a', 'b192', 'pool4', mask), s.clu[mask])[:2])}")
    # ---------------- the rule
    for tag, dn in (("T", "DESIGN 1 (literal): events OLDER than the stored 256th event"), ("U", "DESIGN 2: all old-only events of the earlier snapshots, newest first")):
        print(f"\n## 2{tag}. {dn}")
        c, am, e = "c" + tag, "am" + tag, "e" + tag
        for who in ("pool4", "distilled", "hard"):
            print(f"  -- {who}")
            for sub_lab, lo_, hi_ in [("literal: all filled (n >= 1)", 1, 10 ** 9), ("full fill (n = 128)", 128, 10 ** 9)] + [(f"filled {b[0]}", b[1], b[2]) for b in BUCK[:3]]:
                out = {}
                for name in ("val", "holdout"):
                    s = S[name]
                    mask = s.read[tag] & (s.nf[tag] >= lo_) & (s.nf[tag] <= hi_)
                    num, den = diff(s, c, "b", who, mask), diff(s, "a", "b", who, mask)
                    out[name] = (num, den, s.clu[mask], mask)
                vnum = cstat(out["val"][0], out["val"][2])
                r = rule_line(tag, "holdout", *out["holdout"][:3], val_num=vnum, label=f"HOLDOUT {sub_lab}")
                rv = rule_line(tag, "val", *out["val"][:3], label=f"val     {sub_lab}")
                pn, pc = cat([(out[k][0], out[k][2]) for k in out])
                pd_, _ = cat([(out[k][1], out[k][2]) for k in out])
                rp = rule_line(tag, "pooled", pn, pd_, pc, label=f"holdout + val {sub_lab}")
                RES[f"rule_{tag}_{who}_{sub_lab}"] = dict(holdout=r, val=rv, pooled=rp)
                if sub_lab.startswith("literal"):
                    # ceiling and length-matched share
                    for name in ("holdout", "val"):
                        s, mask = S[name], out[name][3]
                        amb = diff(s, am, "b", who, mask)
                        ce = rule_line(tag, name, amb, out[name][1], out[name][2], label=f"{name} CEILING [am - b]/[a - b]")
                        vn2 = cstat(out["val"][0], out["val"][2]) if name == "holdout" else None
                        rm = rule_line(tag, name, out[name][0], amb, out[name][2], val_num=vn2, label=f"{name} LENGTH-MATCHED [c - b]/[am - b]")
                        x = diff(s, c, am, who, mask)
                        print(f"   {name}: c - am (recovered minus stored events in the same slots) {f(*cstat(x, s.clu[mask])[:2])}; mean fill {s.nf[tag][mask].mean():.1f}")
                        RES[f"ceil_{tag}_{who}_{name}"] = ce
                        RES[f"matched_{tag}_{who}_{name}"] = rm
            # placebo
            for name in ("holdout", "val"):
                s = S[name]
                mask = s.read[tag] & ~np.isnan(s.ndcg(e, who))
                num, den = diff(s, e, "b", who, mask), diff(s, "a", "b", who, mask)
                r = rule_line(tag, name, num, den, s.clu[mask], label=f"{name} PLACEBO e in place of c")
                x = diff(s, c, e, who, mask)
                mu, se, n = cstat(x, s.clu[mask])
                print(f"   {name}: c - e (own old events minus another user's) {f(mu, se)}; c - b on the same feeds {f(*cstat(diff(s, c, 'b', who, mask), s.clu[mask])[:2])}")
                RES[f"placebo_{tag}_{who}_{name}"] = dict(e=r, c_minus_e=mu, c_minus_e_se=se)
            # 192 + 64 line
            for name in ("holdout", "val"):
                s = S[name]
                for sub_lab, lo_ in (("192+64 line, n >= 1", 1), ("192+64 line, n >= 64", 64)):
                    mask = s.read[tag] & (s.nf[tag] >= lo_)
                    num, den = diff(s, "c192" + tag, "b192", who, mask), diff(s, "a", "b192", who, mask)
                    r = rule_line(tag, name, num, den, s.clu[mask], label=f"{name} {sub_lab}")
                    RES[f"l192_{tag}_{who}_{name}_{lo_}"] = r
                    if lo_ == 1:
                        amb = diff(s, "am192" + tag, "b192", who, mask)
                        rule_line(tag, name, num, amb, s.clu[mask], label=f"{name} 192+64 length-matched")
            # d passes
            for name in ("holdout", "val"):
                s = S[name]
                for sub_lab, lo_ in (("n >= 1", 1), ("n >= 64", 64), ("n = 128", 128)):
                    mask = s.read[tag] & (s.nf[tag] >= lo_)
                    mp = mask & ~np.isnan(s.ndcg("dP" + tag, who))
                    d1, d2 = cstat(diff(s, "d1" + tag, "a", who, mask), s.clu[mask]), cstat(diff(s, "d2" + tag, "a", who, mask), s.clu[mask])
                    dp = cstat(diff(s, "dP" + tag, "a", who, mp), s.clu[mp])
                    dd = cstat(diff(s, "d1" + tag, "dP" + tag, who, mp), s.clu[mp])
                    print(f"   {name} pass d, {sub_lab} (n {d1[2]}): d1 - a {f(*d1[:2])} | d2 - a {f(*d2[:2])} | placebo dP - a {f(*dp[:2])} | d1 - dP {f(*dd[:2])}")
                    RES[f"d_{tag}_{who}_{name}_{lo_}"] = dict(d1=d1[:2], d2=d2[:2], dP=dp[:2], d1_dP=dd[:2], n=d1[2])
        # members: numerator only
        print("  -- single members, literal numerator c - b / denominator a - b (holdout | val)")
        for who in WHO:
            if who.startswith("m:"):
                row = []
                for name in ("holdout", "val"):
                    s = S[name]
                    mask = s.read[tag]
                    row.append(f"{f(*cstat(diff(s, c, 'b', who, mask), s.clu[mask])[:2])} / {f(*cstat(diff(s, 'a', 'b', who, mask), s.clu[mask])[:2])}")
                print(f"   {who:22s} " + " | ".join(row))
    # ---------------- size
    print("\n## 4. SIZE [D]: effect per feed of a fill bucket x share of the segment's feeds in the bucket (label-free alignment), summed")
    F = Flat()
    al = {k: np.load(OUT / f"align_{k}.npz") for k in ("val", "test")}
    tst = F.ts["test"]
    segs = {"val": ("val", np.ones(len(al["val"]["fid"]), bool), int(al["val"]["n_all"])),
            "test public-like days": ("test", tst[al["test"]["fid"]] < BOUND, int((tst < BOUND).sum())),
            "test private-like days": ("test", tst[al["test"]["fid"]] >= BOUND, int((tst >= BOUND).sum())),
            "test all": ("test", np.ones(len(al["test"]["fid"]), bool), len(tst))}
    for k, (src, m, ntot) in segs.items():
        print(f"   {k}: feeds {ntot}; qualifying (H = 256, user_id != 0, strictly earlier snapshot) {100 * m.sum() / ntot:.2f}%")
    for tag in "TU":
        for eff_lab, pa, pb in (("N(c) - N(b): worth of the recovered events in trained slots (upper-side proxy for a retrained model)", "c" + tag, "b"),
                                ("N(d1) - N(a): direct, model not trained for it", "d1" + tag, "a")):
            print(f"  -- design {tag}, effect = {eff_lab}")
            for src_lab, names in (("holdout + val", ("holdout", "val")), ("holdout only", ("holdout",)), ("val only", ("val",))):
                eff = []
                for bl, lo_, hi_ in BUCK:
                    parts = []
                    for name in names:
                        s = S[name]
                        mask = s.read[tag] & (s.nf[tag] >= lo_) & (s.nf[tag] <= hi_)
                        parts.append((diff(s, pa, pb, "pool4", mask), s.clu[mask]))
                    x, cl = cat(parts)
                    eff.append(cstat(x, cl))
                line = f"   effects from {src_lab}: " + "; ".join(f"n {BUCK[i][0]}: {f(*eff[i][:2])} (feeds {eff[i][2]})" for i in range(4))
                print(line)
                for k, (src, m, ntot) in segs.items():
                    z = al[src]
                    st = z["stats"]
                    rd = (z["A"] & z["B"] & (st[:, 6] >= 1)) if tag == "T" else (z["A"] & (st[:, 7] >= 1))
                    nfill = np.minimum(st[:, 6] if tag == "T" else st[:, 7], 128)
                    tot, var, shares = 0.0, 0.0, []
                    for i, (bl, lo_, hi_) in enumerate(BUCK):
                        sh = (m & rd & (nfill >= lo_) & (nfill <= hi_)).sum() / ntot
                        shares.append(sh)
                        tot += sh * eff[i][0]
                        var += (sh * eff[i][1]) ** 2
                    se = np.sqrt(var)
                    print(f"      {k:24s} read-set share {100 * sum(shares):5.2f}% (by bucket {' / '.join(f'{100 * x:.2f}' for x in shares)}%) -> pool-level {tot:+.5f} +- {se:.5f}  [95%: {tot - 1.96 * se:+.5f}, {tot + 1.96 * se:+.5f}]")
                    RES[f"size_{tag}_{pa}_{src_lab}_{k}"] = dict(size=tot, se=se, share=sum(shares))
            # plain form on the holdout / val: mean over all qualifying feeds (zero outside the read set) x qualifying share
            for name in ("holdout", "val"):
                s = S[name]
                qm = np.zeros(s.n, bool)
                qm[s.qi] = True
                x = np.zeros(s.n)
                x[s.read[tag]] = diff(s, pa, pb, "pool4", s.read[tag])
                mu, se, n = cstat(x[qm], s.clu[qm])
                print(f"   plain form, {name}: mean effect over all {n} qualifying feeds (zeros outside the read set) {f(mu, se)}; x 14.35% (test) = {0.1435 * mu:+.5f} +- {0.1435 * se:.5f}; x 14.05% (private-like) = {0.1405 * mu:+.5f} +- {0.1405 * se:.5f}")
                RES[f"plain_{tag}_{pa}_{name}"] = dict(mu=mu, se=se, n=n)
    json.dump(RES, open(OUT / "s3_results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
