"""INVENTORY B, step 6: CONDITIONAL score test = "does the column say something that the pool AND the ranker's own inputs do not?"

Why: the plain likelihood ratio of lens.py reads 20-48 on val for inputs the ranker ALREADY has (out/negctl_existing_inputs.txt): on the
post-train period the pool's weighting of its own inputs is off, and any column correlated with them inherits that. So the null model here is
    click ~ conditional logit on [z(pool)] + CONTROL BLOCK,
control block = the 31 distinct pre-computed inputs of the ranker (cvnd_pre2s) + the in-model columns that have a copy in data/feats
(f1..f5, cos_u, history similarities, per-event-type similarities, category counts / shares / recency, feed category share), each as a
within-feed z, the main ones also as a within-feed rank. A candidate column is then judged by the Rao score test for adding it:
    cz  = U / sqrt(user-clustered variance of the efficient score)          (N(0,1) under "no information beyond pool + inputs")
    b1  = one-step coefficient, dnats = one-step loglik gain per feed.
Calibration inside this script: 300 label-free placebo columns; positive controls = pools trained WITHOUT a block, block removed from the
control block, its columns offered as candidates; negative control = a column the ablated pool does read.
Usage: s06_cond.py val|hold TO16|FILE     Output: out/cond_<split>_<base>.jsonl, out/cond_<split>_<base>.json (calibration), log on stdout.
CPU only, 12 threads, read-only."""
import sys, json, time
sys.path.insert(0, "<repo>/reports/INPUTS_1116/residual_rows")
from lens import *
import cand_specs as cs

sp, base = sys.argv[1], sys.argv[2]; t0 = time.time()
POP8 = "seq_cv_invd20 seq_cv_pop seq_cv_norm it_hist_all_luo it_hist_et1_luo it_hist_et2_luo it_hist_et3_luo it_hist_et4_luo".split()
PRE2 = "pre2_fe_z pre2_new_z pre2_hmax32".split()
CTX = ("ctx_nd_hcent ctx_nd_knn_f1 ctx_nd_knn_h ctx_nd_knn_h_dev ctx_nd_knn_u ctx_nd_knn_u_dev ctx_nd_max ctx_nd_top5h ctx_nd_top5u ctx_nd_ucent "
       "ctx_hn_top1w ctx_hn_rrw ctx_hn_bestrk ctx_hn_margin ctx_hn_soft05 ctx_hn_e1_rrw ctx_hs_mutual ctx_hs_k128 ctx_hs_e1 ctx_hs_t10").split()
INMODEL = ("f1 f2 f3 f4 f5 emb_missing item_norm cos_u dot_u seq_max_all seq_max_new8 maxsim32 meansim32 attn32 seq_top3 "
           + " ".join(f"cos_het{t}" for t in range(1, 11)) + " cat_cnt_all cat_share_all " + " ".join(f"cat_cnt_et{t}" for t in range(1, 11)) + " cat_min_hpos feed_cat_share").split()
RK = "f1 f2 f3 f4 f5 cos_u seq_max_all attn32 seq_top3 cat_cnt_all seq_cv_invd20 it_hist_all_luo pre2_fe_z pre2_new_z pre2_hmax32".split()
STORE = {n: s for g, n, s in cs.specs() if g == "store"}


@nb.njit(parallel=True, cache=True)
def clogit_big(X, y, off, M, theta, use, T):
    """as lens.clogit, with the information matrix accumulated per thread chunk (K up to ~100). Also returns row probabilities and per-feed E_p[X]."""
    n = len(M); K = X.shape[1]
    ll = np.zeros(n); G = np.zeros((n, K)); Hc = np.zeros((T, K, K)); p_row = np.zeros(X.shape[0], np.float32); MU = np.zeros((n, K), np.float32)
    for c in nb.prange(T):
        lo = (n * c) // T; hi = (n * (c + 1)) // T
        mu = np.zeros(K); S2 = np.zeros((K, K))
        for f in range(lo, hi):
            o = off[f]; m = M[f]; npos = 0.0
            for i in range(m): npos += y[o + i]
            s = np.empty(m); mx = -1e30
            for i in range(m):
                v = 0.0
                for k in range(K): v += X[o + i, k] * theta[k]
                s[i] = v
                if v > mx: mx = v
            Z = 0.0
            for i in range(m):
                s[i] = np.exp(s[i] - mx); Z += s[i]
            mu[:] = 0.0
            for i in range(m):
                p = s[i] / Z; p_row[o + i] = p
                for k in range(K): mu[k] += p * X[o + i, k]
            for k in range(K): MU[f, k] = mu[k]
            if npos == 0.0 or not use[f]: continue
            S2[:, :] = 0.0; l = 0.0
            for i in range(m):
                p = s[i] / Z; yi = y[o + i]
                if yi > 0:
                    l += yi * np.log(p)
                    for k in range(K): G[f, k] += yi * X[o + i, k]
                for k in range(K):
                    xk = p * X[o + i, k]
                    for q in range(k, K): S2[k, q] += xk * X[o + i, q]
            ll[f] = l
            for k in range(K):
                G[f, k] -= npos * mu[k]
                for q in range(k, K): Hc[c, k, q] += npos * (S2[k, q] - mu[k] * mu[q])
    H = Hc.sum(0)
    for k in range(K):
        for q in range(k): H[k, q] = H[q, k]
    return ll, G, H, p_row, MU


def fit_big(D, X, use=None, theta0=None, iters=40, ridge=1e-6):
    K = X.shape[1]; use = np.ones(D["n"], np.bool_) if use is None else use
    th = np.zeros(K) if theta0 is None else np.asarray(theta0, np.float64).copy()
    if theta0 is None: th[0] = 1.0
    r = clogit_big(X, D["y"], D["off"], D["M"], th, use, 24); L = r[0].sum()
    for it in range(iters):
        g = r[1].sum(0) - ridge * th * 0; Hs = r[2] + ridge * np.eye(K) * np.trace(r[2]) / K
        step = np.linalg.solve(Hs, g); t = 1.0
        while True:
            th2 = th + t * step; r2 = clogit_big(X, D["y"], D["off"], D["M"], th2, use, 24); L2 = r2[0].sum()
            if L2 >= L - 1e-9 or t < 1e-4: break
            t *= 0.5
        done = abs(L2 - L) < 1e-8 * max(1.0, abs(L)); th, r, L = th2, r2, L2
        if done: break
    return dict(theta=th, L=L, ll=r[0], G=r[1], H=r[2], p=r[3], MU=r[4])


def control_block(D, exclude=()):
    names, cols = [], []
    for n in POP8 + CTX + PRE2 + INMODEL:
        if n in exclude: continue
        F = forms(cs.load(STORE[n][sp], D), D)
        if F["varshare"] < 0.01: continue
        cand = [(n + ":z", F["z"])] + ([(n + ":rk", F["rk"])] if n in RK else [])
        for nm, c in cand:
            dup = False
            for c2 in cols[-12:]:
                if abs(np.corrcoef(c, c2)[0, 1]) > 0.999: dup = True; break
            if not dup: names.append(nm); cols.append(c)
    return names, cols


class Null:
    """null model = pool + control block; score test for one more column"""
    def __init__(self, D, names, cols):
        self.D = D; self.names = ["pool"] + names
        self.X = np.ascontiguousarray(np.stack([D["base"]] + cols, 1).astype(np.float32))
        f0 = fit(D, []); self.L_pool = f0["L"]
        self.r = fit_big(D, self.X, theta0=[f0["theta"][0]] + [0.0] * len(cols))
        self.fid = np.repeat(np.arange(D["n"]), D["M"]); self.npos = np.bincount(self.fid, D["y"], D["n"])
        self.rw = (self.npos[self.fid] * self.r["p"]).astype(np.float32)           # row weight npos_f * p_i
        self.Hi = np.linalg.inv(self.r["H"] + 1e-6 * np.eye(self.X.shape[1]) * np.trace(self.r["H"]) / self.X.shape[1])
        self.GA = self.r["G"] @ self.Hi                                                # per-feed score of the null model times I_cc^-1
        self.nMU = self.npos[:, None] * self.r["MU"].astype(np.float64)

    def score(self, x):
        D = self.D; x = np.asarray(x, np.float32); p = self.r["p"]
        mx = np.bincount(self.fid, p * x, D["n"])
        g = np.bincount(self.fid, D["y"] * x, D["n"]) - self.npos * mx
        Ixx = float((self.npos * (np.bincount(self.fid, p * x * x, D["n"]) - mx * mx)).sum())
        Ixc = (self.rw * x) @ self.X - mx @ self.nMU
        Uf = g - self.GA @ Ixc; U = float(Uf.sum())
        V = Ixx - float(Ixc @ self.Hi @ Ixc)
        Uc = np.bincount(D["clus"], Uf); Vr = float((Uc ** 2).sum())
        if V <= 1e-9 * max(Ixx, 1e-12) or Vr <= 0:
            return dict(cz=0.0, cz_model=0.0, b1=0.0, dnats=0.0, r2=1.0)
        return dict(cz=U / np.sqrt(Vr), cz_model=U / np.sqrt(V), b1=U / V, dnats=0.5 * U * U / V / D["n"], r2=1.0 - V / max(Ixx, 1e-12))

    def nested_block(self):
        """what re-weighting the ranker's own inputs is worth in NDCG (5 user-grouped folds), against the pool alone"""
        D = self.D; d = np.zeros(D["n"])
        for k in range(5):
            tr = D["fold"] != k; r = fit_big(D, self.X, use=tr, theta0=self.r["theta"], iters=6)
            s = (self.X @ r["theta"].astype(np.float32)).astype(np.float32); e = ndcg(s, D); d[~tr] = e[~tr] - D["ndcg0"][~tr]
        return d


def read_cand(N, x):
    F = forms(x, N.D); out = dict(nanshare=F["nanshare"], varshare=F["varshare"])
    if F["varshare"] < 0.01: out["skip"] = "constant within feeds"; return out
    for k in ("z", "rk", "nan"):
        if k in F:
            for a, b in N.score(F[k]).items(): out[f"{a}_{k}"] = float(b)
    return out


@nb.njit(cache=True)
def perm_within(x, off, M, seed):
    np.random.seed(seed); out = x.copy()
    for f in range(len(M)):
        o = off[f]; m = M[f]
        for i in range(m - 1, 0, -1):
            j = np.random.randint(0, i + 1); t = out[o + i]; out[o + i] = out[o + j]; out[o + j] = t
    return out


D = load_split(sp, base)
if sp == "val":
    import pmix
    D["wpriv"] = pmix.weights()["private"]
print(f"[{time.time() - t0:.0f}s] {sp} / base {base}: {D['n']} feeds, pool NDCG {D['ndcg0'].mean():.5f}", flush=True)
names, cols = control_block(D)
N = Null(D, names, cols)
cal = dict(split=sp, base=base, K=len(names), names=names, pool_ndcg=float(D["ndcg0"].mean()), LR_block=float(2 * (N.r["L"] - N.L_pool)), nats_block=float((N.r["L"] - N.L_pool) / D["n"]))
d = N.nested_block(); cal.update(nest_block=float(d.mean()), nest_block_se=float(d.std() / np.sqrt(len(d))))
if "wpriv" in D: cal["nest_block_pmix"] = float(np.average(d, weights=D["wpriv"]))
print(f"[{time.time() - t0:.0f}s] control block: {len(names)} columns of the ranker's own inputs. LR(block | pool) {cal['LR_block']:.1f} ({len(names)} df), nats/feed {cal['nats_block']:+.5f};"
      f" nested NDCG of re-weighting the ranker's own inputs {cal['nest_block']:+.5f} +- {cal['nest_block_se']:.5f}" + (f" (pmix-private {cal['nest_block_pmix']:+.5f})" if 'nest_block_pmix' in cal else ""), flush=True)

# ---- placebos
rng = np.random.default_rng(20261117); pl_ = []
REAL = ["f4", "f5", "cos_u", "seq_max_all", "it_hist_all_luo", "tte_it_ctr", "trd_kde_sday", "clu_pmi1024_mean", "ctx_nd_knn_h", "cat_cnt_all", "ixh_good", "seq_cv_invd20", "pre2_fe_z", "tte_it_imp", "attn32"]
Zr = {c: forms(cs.load(STORE[c][sp], D), D)["z"] for c in REAL}
for i in range(150):
    pl_.append(N.score(rng.standard_normal(len(D["y"])).astype(np.float32))["cz"])
for i in range(150):
    pl_.append(N.score(perm_within(Zr[REAL[i % len(REAL)]].astype(np.float64), D["off"], D["M"], 5000 + i))["cz"])
pl_ = np.array(pl_); cal["placebo_cz"] = pl_.tolist()
print(f"[{time.time() - t0:.0f}s] placebo (300 label-free columns): cz mean {pl_.mean():+.2f} sd {pl_.std():.2f} max |cz| {np.abs(pl_).max():.2f}; share |cz| >= 2: {(np.abs(pl_) >= 2).mean():.3f} (N(0,1): 0.046)", flush=True)

# ---- candidates
outp = OUT / f"cond_{sp}_{base}.jsonl"; fo = open(outp, "w"); n = 0
for grp, name, src in cs.specs():
    if sp not in src: continue
    try:
        r = read_cand(N, cs.load(src[sp], D))
    except Exception as e:
        r = dict(error=repr(e)[:200])
    r.update(key=f"{grp}:{name}", grp=grp, name=name)
    fo.write(json.dumps(r) + "\n"); n += 1
    if n % 60 == 0: fo.flush(); print(f"   [{time.time() - t0:.0f}s] {n} candidates", flush=True)
fo.close()
json.dump(cal, open(OUT / f"cond_{sp}_{base}.json", "w"))

# ---- positive / negative controls (TO16 runs only): pools trained WITHOUT a block
if base == "TO16":
    pf = lambda v: (W5 if v.startswith("w5_") else AB) / (f"nn_{v}_trainonly_val.npy" if sp == "val" else f"holdout/nn_{v}_holdout.npy")
    CT = [("pop8_student_w192", [f"abl_e10_pop1kd05w192_s{s}" for s in (951, 952, 953, 954)], POP8, POP8 + ["xp_imp_c50"], "popularity-light wide students (read xp_imp_c50, not the 8 popularity inputs)"),
          ("pop8_hard10ep", [f"abl_e10_xdrop_pop_s{s}" for s in (951, 952, 953, 954)], POP8, POP8, "hard-label members without the 8 popularity inputs"),
          ("pre2_hard10ep", [f"abl_e10_xdrop_pre2_s{s}" for s in (951, 952)], PRE2, PRE2, "hard-label members without the encoder scalars")]
    cal["controls"] = {}
    for tag, members, excl, cands, note in CT:
        Dc = dict(D); Dc["base"] = zpool([pf(v) for v in members], D["off"], D["M"]); Dc["ndcg0"] = ndcg(Dc["base"], Dc)
        nm, cl = control_block(Dc, exclude=set(excl)); Nc = Null(Dc, nm, cl); res = {}
        print(f"\n== CONTROL {tag}: {note}; pool NDCG {Dc['ndcg0'].mean():.5f}; control block without the removed inputs ({len(nm)} columns)", flush=True)
        for c in cands:
            F = forms(cs.load(STORE[c][sp], Dc), Dc); s = Nc.score(F["z"]); s2 = Nc.score(F["rk"]); res[c] = dict(z=s, rk=s2)
            print(f"   {c:18s} cz {s['cz']:+7.2f} (rank form {s2['cz']:+7.2f})  one-step coef {s['b1']:+.4f}  dnats/feed {s['dnats']:+.6f}  R2 with pool+controls {s['r2']:.2f}"
                  + ("   <- NEGATIVE control: this pool reads it" if c == "xp_imp_c50" else ""), flush=True)
        cal["controls"][tag] = res
    json.dump(cal, open(OUT / f"cond_{sp}_{base}.json", "w"))
print(f"[{time.time() - t0:.0f}s] done", flush=True)
