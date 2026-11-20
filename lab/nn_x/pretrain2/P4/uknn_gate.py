"""P4 (E5-reduced) stage 0: cross-user trajectory-neighbour label transfer, val + holdout, CPU only (12 threads).

Design and gate: nn_x/pretrain2/P4/PREDECLARED.txt (written before any P4 number was computed).
Bank = exploration-like train feeds (p_random > 0.5, M >= 50) of a fixed-length trailing window (5 data days, gap >= 3 days),
excluding the holdout feeds, user 0 and every user present in val / test / holdout. Key = user_emb (one entry per user,
mean of the user's in-window feeds), value = the user's shown candidates with labels. Query = the feed's own user_emb;
k = 32 nearest bank users, weights softmax(10 cos). Signals s_pos / s_neg / contrast at item_id and cl4096 level.
Lens = nested residual on G3 (user-grouped 5-fold, user-0 singletons, 2 fold seeds), val leak-free + holdout replica.
Outputs: uknn_gate.log, uknn_gate.json, signals_{val,holdout}.npz, deltas_{val,holdout}.npz (all in this directory).
"""
import os
for _k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[_k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time, json
from pathlib import Path
import numpy as np, polars as pl, numba as nb

ROOT = Path("<repo>")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eda/errors/synthesis"))
from metrics import ndcg_per_feed  # noqa: E402
from qt import leak_mask  # noqa: E402
from pmix import weights as pmix_weights, wmean_se  # noqa: E402

FLAT = ROOT / "data/flat"; FEATS = ROOT / "data/feats"; PREDS = ROOT / "preds"; CLU = ROOT / "data/clu_work"
OUT = ROOT / "nn_x/pretrain2/P4"
MSK = 3 * 3600
DAY0 = (1788642048 + MSK) // 86400  # Sep 6 2026 MSK = day 0
K_NN = 32; TEMP = 10.0; WIN = 5; GAP = 3
G3_MEMBERS = ([f"v23_gru_pre2s_s50{i}" for i in range(1, 7)] + [f"r6_cd_m02_r10{i}" for i in range(1, 8)]
              + [f"r6_cd_m02_p70k_r10{i}" for i in range(1, 8)])
T0 = time.time()
LOG = open(OUT / "uknn_gate.log", "w")
RES = {}


def log(*a):
    s = f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def day_name(d):
    import datetime as dt
    return (dt.date(2026, 9, 6) + dt.timedelta(days=int(d))).strftime("%b %d")


def meta(split):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    ts = f["timestamp"].to_numpy().astype(np.int64)
    return dict(user=f["user_id"].to_numpy().astype(np.int64), ts=ts, day=(ts + MSK) // 86400 - DAY0,
                M=f["M"].to_numpy().astype(np.int64), off=f["cand_off"].to_numpy().astype(np.int64), n=len(f))


def feed_rows(off, M, feeds):
    m = M[feeds]; st = off[feeds]
    cs = np.concatenate([[0], np.cumsum(m)[:-1]])
    return np.repeat(st - cs, m) + np.arange(m.sum())


def feed_z(x, fid, n):
    x = np.asarray(x, np.float64)
    c = np.bincount(fid, minlength=n).astype(np.float64)
    mu = np.bincount(fid, x, n) / np.maximum(c, 1)
    v = np.bincount(fid, (x - mu[fid]) ** 2, n) / np.maximum(c, 1)
    sd = np.sqrt(v); sd[sd < 1e-9] = 1.0
    return (x - mu[fid]) / sd[fid]


def mse(d):
    d = np.asarray(d, np.float64)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def fmt(d):
    m, s = mse(d)
    return f"{m:+.5f} ± {s:.5f} (z {m / s if s > 0 else 0:+.2f})"


def l2n(E):
    E = E.astype(np.float32)
    nr = np.linalg.norm(E, axis=1, keepdims=True); nr[nr == 0] = 1.0
    return E / nr


# ---------------------------------------------------------------- metadata
tr = meta("train"); va = meta("val"); te = meta("test")
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
assert (pr["fidx"] == np.arange(tr["n"])).all()
p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[tr["M"] < 50] = 0.0
hcand = np.where((p > 0.9) & (tr["M"] >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(hcand, min(30000, len(hcand)), replace=False))
hm = np.load(PREDS / "holdout/holdout_meta.npy"); hM = hm[0].astype(np.int64); hoff = hm[1].astype(np.int64)
assert len(hold) == len(hM) and (tr["M"][hold] == hM).all()
log(f"train feeds {tr['n']:,}; val {va['n']:,}; test {te['n']:,}; holdout {len(hold):,} (rng 12345 rule reproduced, M matches)")

cnt_day = np.bincount(tr["day"], minlength=16)
DATA_DAYS = np.nonzero(cnt_day >= 1000)[0]
log("train feeds per day:", {day_name(d): int(c) for d, c in enumerate(cnt_day) if c}, "| data days:", [day_name(d) for d in DATA_DAYS])

excl_users = np.unique(np.concatenate([va["user"], te["user"], tr["user"][hold], [0]]))
bankable = (p > 0.5) & (tr["M"] >= 50)
n_bl = int(bankable.sum())
bankable[hold] = False
n_noh = int(bankable.sum())
bankable &= ~np.isin(tr["user"], excl_users)
log(f"bankable train feeds: p>0.5 & M>=50 {n_bl:,}; minus holdout {n_noh:,}; minus user0 / val / test / holdout users {int(bankable.sum()):,}")


def window(D):
    dd = DATA_DAYS[DATA_DAYS <= D - GAP]
    return tuple(int(x) for x in dd[-WIN:]) if len(dd) >= WIN else None


# ---------------------------------------------------------------- train arrays (memmap)
c_item = np.load(FLAT / "train/cand_item_id.npy", mmap_mode="r")
c_tgt = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
c_cl = np.load(CLU / "cand_cl_4096_train.npy", mmap_mode="r")
ue_tr = np.load(FLAT / "train/user_emb.npy", mmap_mode="r")
assert len(c_cl) == len(c_item)


def reduce_user_keys(row_u, keys, tgt, nu):
    """unique (bank user, key) with any-click flag -> CSR (ptr over users, keys, pos)."""
    o = np.lexsort((keys, row_u))
    ku, kk, tt = row_u[o], keys[o], tgt[o]
    new = np.ones(len(o), bool); new[1:] = (ku[1:] != ku[:-1]) | (kk[1:] != kk[:-1])
    st = np.nonzero(new)[0]
    pos = np.maximum.reduceat(tt, st) > 0
    uu = ku[st]
    ptr = np.searchsorted(uu, np.arange(nu + 1))
    return ptr.astype(np.int64), kk[st].astype(np.int64), pos


BANKS = {}


def build_bank(days):
    if days in BANKS:
        return BANKS[days]
    t = time.time()
    fe = np.nonzero(bankable & np.isin(tr["day"], days))[0]
    ub, uinv = np.unique(tr["user"][fe], return_inverse=True)
    nu = len(ub)
    E = l2n(np.asarray(ue_tr[fe]))
    Kb = np.zeros((nu, 128), np.float64); np.add.at(Kb, uinv, E)
    Kb = l2n(Kb)
    last_ts = np.zeros(nu, np.int64); np.maximum.at(last_ts, uinv, tr["ts"][fe])
    nfe = np.bincount(uinv, minlength=nu)
    rows = feed_rows(tr["off"], tr["M"], fe)
    row_u = np.repeat(uinv, tr["M"][fe]).astype(np.int64)
    it = np.asarray(c_item[rows]).astype(np.int64); cl = np.asarray(c_cl[rows]).astype(np.int64)
    tg = np.asarray(c_tgt[rows]).astype(np.int8)
    bi = reduce_user_keys(row_u, it, tg, nu)
    okc = cl >= 0
    bc = reduce_user_keys(row_u[okc], cl[okc], tg[okc], nu)
    b = dict(days=days, n_feeds=len(fe), n_users=nu, K=Kb, last_ts=last_ts, item=bi, cl=bc, nrows=len(rows),
             ctr=float(tg.mean()), feeds_per_user=float(nfe.mean()))
    log(f"bank {[day_name(d) for d in days]}: {len(fe):,} feeds, {nu:,} users ({nfe.mean():.2f} feeds/user), {len(rows):,} shown rows "
        f"(CTR {tg.mean():.4f}), item keys {len(bi[1]):,}, cl keys {len(bc[1]):,} [{time.time() - t:.0f}s]")
    BANKS[days] = b
    return b


def knn(Q, Kb, k=K_NN, chunk=2048):
    nq = len(Q)
    idx = np.empty((nq, k), np.int64); sim = np.empty((nq, k), np.float32)
    Kt = np.ascontiguousarray(Kb.T.astype(np.float32))
    for s in range(0, nq, chunk):
        S = Q[s:s + chunk].astype(np.float32) @ Kt
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        sv = np.take_along_axis(S, part, 1)
        o = np.argsort(-sv, 1)
        idx[s:s + chunk] = np.take_along_axis(part, o, 1); sim[s:s + chunk] = np.take_along_axis(sv, o, 1)
    return idx, sim


@nb.njit(parallel=True, cache=False)
def transfer(qfeeds, qoff, qM, qkeys, nbr, w, ptr, bkeys, bpos, out_p, out_n):
    for i in nb.prange(len(qfeeds)):
        q = qfeeds[i]
        L = 0
        for j in range(nbr.shape[1]):
            u = nbr[i, j]; L += ptr[u + 1] - ptr[u]
        kk = np.empty(L, np.int64); wp = np.empty(L); wn = np.empty(L)
        c = 0
        for j in range(nbr.shape[1]):
            u = nbr[i, j]; wj = w[i, j]
            for t in range(ptr[u], ptr[u + 1]):
                kk[c] = bkeys[t]
                if bpos[t]:
                    wp[c] = wj; wn[c] = 0.0
                else:
                    wp[c] = 0.0; wn[c] = wj
                c += 1
        o = np.argsort(kk)
        uk = np.empty(L, np.int64); up = np.zeros(L); un = np.zeros(L); nu = 0
        for t in range(L):
            a = o[t]
            if nu == 0 or kk[a] != uk[nu - 1]:
                uk[nu] = kk[a]; up[nu] = 0.0; un[nu] = 0.0; nu += 1
            up[nu - 1] += wp[a]; un[nu - 1] += wn[a]
        ukv = uk[:nu]
        for r in range(qoff[q], qoff[q] + qM[q]):
            key = qkeys[r]
            if key < 0:
                continue
            ps = np.searchsorted(ukv, key)
            if ps < nu and ukv[ps] == key:
                out_p[r] = up[ps]; out_n[r] = un[ps]


SIG = ["s_pos_item", "s_neg_item", "contrast_item", "s_pos_cl", "s_neg_cl", "contrast_cl"]
PRIMARY = ["s_neg_item", "contrast_item", "s_neg_cl", "contrast_cl"]


def run_queries(name, qdays, Q, q_ts, qoff, qM, q_item, q_cl):
    """qdays: per query feed day; returns signals dict, eligibility mask, neighbour diagnostics."""
    n = len(qM); nrow = int(qM.sum())
    sp_i = np.zeros(nrow); sn_i = np.zeros(nrow); sp_c = np.zeros(nrow); sn_c = np.zeros(nrow)
    elig = np.zeros(n, bool); gaps = []; topcos = np.full(n, np.nan); wins = {}
    for D in np.unique(qdays):
        wd = window(int(D))
        qf = np.nonzero(qdays == D)[0]
        if wd is None:
            log(f"  {name} day {day_name(D)}: {len(qf):,} feeds, no full {WIN}-data-day window with gap >= {GAP} -> no bank")
            continue
        wins.setdefault(wd, []).append(int(D))
        elig[qf] = True
    for wd, Ds in wins.items():
        b = build_bank(wd)
        qf = np.nonzero(np.isin(qdays, Ds))[0]
        t = time.time()
        idx, sim = knn(Q[qf], b["K"])
        ww = np.exp(TEMP * (sim - sim[:, :1])); ww /= ww.sum(1, keepdims=True)
        topcos[qf] = sim.mean(1)
        gaps.append(((q_ts[qf][:, None] - b["last_ts"][idx]) / 86400.0).ravel())
        transfer(qf.astype(np.int64), qoff, qM, q_item, idx, ww, b["item"][0], b["item"][1], b["item"][2], sp_i, sn_i)
        transfer(qf.astype(np.int64), qoff, qM, q_cl, idx, ww, b["cl"][0], b["cl"][1], b["cl"][2], sp_c, sn_c)
        log(f"  {name} query days {[day_name(d) for d in Ds]} -> bank {[day_name(d) for d in wd]}: {len(qf):,} feeds, "
            f"top-32 cos mean {sim.mean():.4f} (top1 {sim[:, 0].mean():.4f}, 32nd {sim[:, -1].mean():.4f}), "
            f"max weight mean {ww.max(1).mean():.4f} [{time.time() - t:.0f}s]")
    gaps = np.concatenate(gaps) if gaps else np.zeros(0)
    S = dict(s_pos_item=sp_i, s_neg_item=sn_i, contrast_item=sp_i - sn_i, s_pos_cl=sp_c, s_neg_cl=sn_c, contrast_cl=sp_c - sn_c)
    return S, elig, gaps, topcos


def coverage(ev, fid, n, feeds_mask):
    a = np.bincount(fid, ev.astype(np.float64), n)[feeds_mask]; m = np.bincount(fid, minlength=n).astype(np.float64)[feeds_mask]
    r = a.sum() / m.sum()
    se = np.sqrt(((a - r * m) ** 2).sum()) / m.sum()
    return float(r), float(se)


# ---------------------------------------------------------------- val queries
log("=== VAL queries")
vy = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
v_item = np.load(FLAT / "val/cand_item_id.npy").astype(np.int64)
v_cl = np.load(CLU / "cand_cl_4096_val.npy").astype(np.int64)
Qv = l2n(np.load(FLAT / "val/user_emb.npy"))
SV, eligv, gapv, cosv = run_queries("val", va["day"], Qv, va["ts"], va["off"], va["M"], v_item, v_cl)
assert eligv.all()

# ---------------------------------------------------------------- holdout queries
log("=== HOLDOUT queries")
hrows = feed_rows(tr["off"], tr["M"], hold)
hy = np.load(PREDS / "holdout/holdout_y.npy").astype(np.float64)
assert len(hy) == len(hrows) and (np.asarray(c_tgt[hrows]).astype(np.float64) == hy).all()
h_item = np.asarray(c_item[hrows]).astype(np.int64); h_cl = np.asarray(c_cl[hrows]).astype(np.int64)
Qh = l2n(np.asarray(ue_tr[hold]))
SH, eligh, gaph, cosh = run_queries("holdout", tr["day"][hold], Qh, tr["ts"][hold], hoff, hM, h_item, h_cl)
log(f"holdout eligible feeds {eligh.sum():,} of {len(hold):,} (days {[day_name(d) for d in np.unique(tr['day'][hold][eligh])]})")
np.savez_compressed(OUT / "signals_val.npz", **{k: v.astype(np.float16) for k, v in SV.items()})
np.savez_compressed(OUT / "signals_holdout.npz", eligible=eligh, **{k: v.astype(np.float16) for k, v in SH.items()})

# ---------------------------------------------------------------- bases
nv = va["n"]; fidv = np.repeat(np.arange(nv), va["M"])
nh = len(hold); fidh = np.repeat(np.arange(nh), hM)
keep = ~leak_mask()
zG3v = feed_z(np.load(PREDS / "ens_G3_final_val.npy"), fidv, nv)
acc = None
for m in G3_MEMBERS:
    z = feed_z(np.load(PREDS / f"holdout/nn_{m}_holdout.npy"), fidh, nh)
    acc = z if acc is None else acc + z
zG3h = feed_z(acc / len(G3_MEMBERS), fidh, nh)
basev = ndcg_per_feed(zG3v, vy, va["off"], va["M"]); baseh = ndcg_per_feed(zG3h, hy, hoff, hM)
log(f"G3 val NDCG all {mse(basev)[0]:.5f} ± {mse(basev)[1]:.5f}; leak-free ({keep.sum():,}) {mse(basev[keep])[0]:.5f} ± {mse(basev[keep])[1]:.5f}")
log(f"G3 holdout NDCG all {mse(baseh)[0]:.5f} ± {mse(baseh)[1]:.5f} (expect 0.32379); eligible ({eligh.sum():,}) "
    f"{mse(baseh[eligh])[0]:.5f} ± {mse(baseh[eligh])[1]:.5f}")
RES["base"] = dict(val_all=mse(basev), val_leakfree=mse(basev[keep]), hold_all=mse(baseh), hold_elig=mse(baseh[eligh]))

# ---------------------------------------------------------------- descriptive diagnostics
log("=== DIAGNOSTICS")
RES["bank"] = {"/".join(day_name(d) for d in k): dict(n_feeds=b["n_feeds"], n_users=b["n_users"], rows=b["nrows"], ctr=b["ctr"],
                                                      feeds_per_user=b["feeds_per_user"]) for k, b in BANKS.items()}
qs = [5, 25, 50, 75, 95]
gq_v = np.percentile(gapv, qs); gq_h = np.percentile(gaph, qs)
log(f"query-to-neighbour gap (days, neighbour's latest in-window feed) q{qs}: val {np.round(gq_v, 2).tolist()} | holdout {np.round(gq_h, 2).tolist()}")
log(f"mean top-32 cos: val {np.nanmean(cosv):.4f} ± {np.nanstd(cosv) / np.sqrt(np.isfinite(cosv).sum()):.4f} | "
    f"holdout(eligible) {np.nanmean(cosh[eligh]):.4f} ± {np.nanstd(cosh[eligh]) / np.sqrt(eligh.sum()):.4f}")
RES["gap_days_q"] = dict(q=qs, val=gq_v.tolist(), holdout=gq_h.tolist())
RES["coverage"] = {}
for lens, S, fid, n, fm in [("val", SV, fidv, nv, keep), ("holdout_elig", SH, fidh, nh, eligh)]:
    rec = {}
    for lev in ["item", "cl"]:
        ev = (S[f"s_pos_{lev}"] + S[f"s_neg_{lev}"]) > 0
        rec[f"any_{lev}"] = coverage(ev, fid, n, fm)
        rec[f"pos_{lev}"] = coverage(S[f"s_pos_{lev}"] > 0, fid, n, fm)
        rec[f"neg_{lev}"] = coverage(S[f"s_neg_{lev}"] > 0, fid, n, fm)
    rec["feeds_with_any_item_evidence"] = mse(np.bincount(fid, ((S["s_pos_item"] + S["s_neg_item"]) > 0).astype(float), n)[fm] > 0)
    RES["coverage"][lens] = rec
    log(f"coverage {lens}: " + "; ".join(f"{k} {v[0]:.4f} ± {v[1]:.4f}" for k, v in rec.items()))

# label rate given evidence (descriptive; val labels used only descriptively, no selection)
for lens, S, yy, fid, n, fm in [("val", SV, vy, fidv, nv, keep), ("holdout_elig", SH, hy, fidh, nh, eligh)]:
    rm = fm[fid]
    base_rate = yy[rm].mean()
    parts = [f"base {base_rate:.4f}"]
    for lev in ["item", "cl"]:
        for nm_, cond in [("pos>0", S[f"s_pos_{lev}"] > 0), ("neg>0&pos=0", (S[f"s_neg_{lev}"] > 0) & (S[f"s_pos_{lev}"] == 0)),
                          ("none", (S[f"s_pos_{lev}"] + S[f"s_neg_{lev}"]) == 0)]:
            mm = rm & cond
            parts.append(f"{lev}:{nm_} n={mm.sum():,} rate {yy[mm].mean() if mm.any() else float('nan'):.4f}")
    log(f"label rate by evidence {lens}: " + "; ".join(parts))

# within-feed corr with cos_u (val leak-free rows)
cosu = np.load(FEATS / "val/cos_u.npy").astype(np.float64)
zc = feed_z(np.nan_to_num(cosu, nan=np.nanmean(cosu)), fidv, nv)
rmv = keep[fidv]
RES["corr_cos_u"] = {}
for k in SIG:
    zx = feed_z(SV[k], fidv, nv)
    cc = float(np.corrcoef(zx[rmv], zc[rmv])[0, 1])
    RES["corr_cos_u"][k] = cc
log("within-feed corr (pooled within-feed z) with cos_u, val leak-free: " + "; ".join(f"{k} {v:+.4f}" for k, v in RES["corr_cos_u"].items()))
zg = zG3v
RES["corr_G3"] = {k: float(np.corrcoef(feed_z(SV[k], fidv, nv)[rmv], zg[rmv])[0, 1]) for k in SIG}
log("within-feed corr with z(G3), val leak-free: " + "; ".join(f"{k} {v:+.4f}" for k, v in RES["corr_G3"].items()))

# single-feature NDCG (pessimistic ties, evaluator definition)
RES["single"] = {}
for k in SIG:
    a = ndcg_per_feed(SV[k], vy, va["off"], va["M"])[keep]; b = ndcg_per_feed(SH[k], hy, hoff, hM)[eligh]
    RES["single"][k] = dict(val=mse(a), hold=mse(b))
    log(f"single-feature NDCG {k:14s} val(leak-free) {mse(a)[0]:.5f} ± {mse(a)[1]:.5f} | holdout(elig) {mse(b)[0]:.5f} ± {mse(b)[1]:.5f}")
rnd = ndcg_per_feed(np.zeros(len(vy)), vy, va["off"], va["M"])[keep]
log(f"  reference: constant score (all ties, pessimistic) val {mse(rnd)[0]:.5f}")

# ---------------------------------------------------------------- cell AUC (stage0_gate.py definition, leak-free val)
log("=== CELL ('right category, buried', nn_x/r3/stage0_gate.py definition)")
cat = np.load(FLAT / "val/cand_category_id.npy").astype(np.int64)
ck = fidv * 64 + cat; uk, inv = np.unique(ck, return_inverse=True)
lse = np.log(np.bincount(inv, np.exp(zG3v)))
kf = uk // 64; o = np.lexsort((-lse, kf)); cr = np.empty(len(uk), np.int64); cr[o] = np.arange(len(uk)) - np.searchsorted(kf[o], kf[o])
pcr = cr[inv]
o2 = np.lexsort((vy, -zG3v, ck)); wr = np.empty(len(vy), np.int64); wr[o2] = np.arange(len(vy)) - np.searchsorted(ck[o2], ck[o2])
cellpos = (vy > 0) & (pcr == 0) & (wr >= 3) & keep[fidv]
pi, ni = [], []
for g in np.unique(fidv[cellpos]):
    s, m = va["off"][g], va["M"][g]
    pp = np.nonzero(cellpos[s:s + m])[0] + s
    neg = np.nonzero(vy[s:s + m] == 0)[0] + s
    for p_ in pp:
        q = neg[(cat[neg] == cat[p_]) & (zG3v[neg] > zG3v[p_])]
        pi.append(np.full(len(q), p_)); ni.append(q)
pi = np.concatenate(pi); ni = np.concatenate(ni)
log(f"cell positives {cellpos.sum():,} (leak-free), beater pairs {len(pi):,}, feeds {len(np.unique(fidv[pi])):,}")


def pair_auc(x):
    x = np.asarray(x, np.float64)
    ok = np.isfinite(x[pi]) & np.isfinite(x[ni])
    d = x[pi][ok] - x[ni][ok]
    a = (d > 0) + 0.5 * (d == 0)
    # feed-clustered SE of the pair AUC
    fg = fidv[pi][ok]; uf, fi = np.unique(fg, return_inverse=True)
    sa = np.bincount(fi, a); sn = np.bincount(fi).astype(np.float64); r = sa.sum() / sn.sum()
    se = np.sqrt(((sa - r * sn) ** 2).sum()) / sn.sum()
    return float(r), float(se), float(ok.mean()), float((d != 0).mean())


ctrl = {"-log1p(it_hist_all_luo)": -np.log1p(np.load(FEATS / "val/it_hist_all_luo.npy").astype(np.float64)),
        "-maxsim32": -np.load(FEATS / "val/maxsim32.npy").astype(np.float64)}
RES["cell_auc"] = {}
for k, x in list(ctrl.items()) + [(k, SV[k]) for k in SIG]:
    a = pair_auc(x)
    RES["cell_auc"][k] = a
    log(f"cell pair AUC {k:26s} {a[0]:.4f} ± {a[1]:.4f} (pairs with finite values {a[2]:.3f}, non-tied {a[3]:.3f})")

# ---------------------------------------------------------------- nested residual on G3
log("=== NESTED RESIDUAL on G3")
GRID = np.array([0, .02, .05, .1, .15, .2, .3, .5, .75, 1.0]); GRID = np.concatenate([GRID, -GRID[1:]])  # 0 first: ties -> w = 0


def groups_of(users):
    g = users.copy(); z0 = g == 0
    g[z0] = -np.arange(1, z0.sum() + 1)
    return g


def make_folds(groups, seed):
    u, inv = np.unique(groups, return_inverse=True)
    return (np.random.default_rng(seed).permutation(len(u)) % 5)[inv]


FOLDS_V = [make_folds(groups_of(va["user"]), s) for s in (0, 1)]
FOLDS_H = [make_folds(groups_of(tr["user"][hold]), s) for s in (0, 1)]


def nd_grid(zb, zx, y, off, M):
    return np.stack([ndcg_per_feed(zb + w * zx, y, off, M) for w in GRID])


def nested_fixed(nd, base, mask, folds):
    d_all, ws = [], []
    for fold in folds:
        oof = np.zeros(len(base))
        for k in range(5):
            trm = (fold != k) & mask; tem = fold == k
            i = int(np.argmax(nd[:, trm].mean(1)))
            oof[tem] = nd[i, tem] - base[tem]; ws.append(float(GRID[i]))
        d_all.append(oof)
    curve = (nd[:, mask] - base[mask]).mean(1)
    return np.mean(d_all, 0), ws, curve


def nested_ridge(zb, Z, y, fid, off, M, base, mask, folds, alpha=1.0):
    n = len(M)
    ydm = y - (np.bincount(fid, y, n) / M)[fid]
    X = np.column_stack([Z, zb])
    d_all, ws, betas = [], [], []
    for fold in folds:
        oof = np.zeros(n)
        for k in range(5):
            trm = (fold != k) & mask; tem = fold == k
            rr = trm[fid]
            Xt = X[rr]; A = Xt.T @ Xt + alpha * np.eye(X.shape[1]); beta = np.linalg.solve(A, Xt.T @ ydm[rr])
            betas.append(beta.tolist())
            zd = feed_z(Z @ beta[:Z.shape[1]], fid, n)
            nd = np.stack([ndcg_per_feed(zb + w * zd, y, off, M) for w in GRID])
            i = int(np.argmax(nd[:, trm].mean(1)))
            oof[tem] = nd[i, tem] - base[tem]; ws.append(float(GRID[i]))
        d_all.append(oof)
    return np.mean(d_all, 0), ws, betas


wpriv = pmix_weights()["private"]
RES["nested"] = {}
DV, DH, DHA = {}, {}, {}
ZV = {k: feed_z(SV[k], fidv, nv) for k in SIG}
ZH = {k: feed_z(SH[k], fidh, nh) for k in SIG}
for k in SIG:
    t = time.time()
    ndv = nd_grid(zG3v, ZV[k], vy, va["off"], va["M"]); ndh = nd_grid(zG3h, ZH[k], hy, hoff, hM)
    dv, wsv, cv = nested_fixed(ndv, basev, keep, FOLDS_V)
    dh, wsh, ch = nested_fixed(ndh, baseh, eligh, FOLDS_H)
    dha, wsha, _ = nested_fixed(ndh, baseh, np.ones(nh, bool), FOLDS_H)
    DV[k], DH[k], DHA[k] = dv, dh, dha
    pm, pse = wmean_se(dv[keep], wpriv[keep])
    rec = dict(val=mse(dv[keep]), val_M_lt50=mse(dv[keep & (va["M"] < 50)]), val_M_ge50=mse(dv[keep & (va["M"] >= 50)]),
               pmix=(float(pm), float(pse)), hold_elig=mse(dh[eligh]), hold_all=mse(dha), w_val=wsv, w_hold=wsh,
               full_fit_val_best=(float(cv.max()), float(GRID[int(np.argmax(cv))])),
               full_fit_hold_best=(float(ch.max()), float(GRID[int(np.argmax(ch))])))
    RES["nested"][k] = rec
    log(f"{k:14s} val {fmt(dv[keep])} | M<50 {fmt(dv[keep & (va['M'] < 50)])} M>=50 {fmt(dv[keep & (va['M'] >= 50)])} | "
        f"pmix {pm:+.5f} ± {pse:.5f} | holdout(elig) {fmt(dh[eligh])} | holdout(all 30K) {fmt(dha)} | "
        f"w val {sorted(set(wsv))} hold {sorted(set(wsh))} | full-fit best val {cv.max():+.5f}@{GRID[int(np.argmax(cv))]:+.2f} "
        f"hold {ch.max():+.5f}@{GRID[int(np.argmax(ch))]:+.2f} [{time.time() - t:.0f}s]")

# joint ridge arm (secondary)
t = time.time()
JS = ["s_pos_item", "s_neg_item", "s_pos_cl", "s_neg_cl"]
dvj, wsj, bj = nested_ridge(zG3v, np.column_stack([ZV[k] for k in JS]), vy, fidv, va["off"], va["M"], basev, keep, FOLDS_V)
dhj, wshj, bhj = nested_ridge(zG3h, np.column_stack([ZH[k] for k in JS]), hy, fidh, hoff, hM, baseh, eligh, FOLDS_H)
pm, pse = wmean_se(dvj[keep], wpriv[keep])
RES["nested"]["JOINT_ridge"] = dict(val=mse(dvj[keep]), pmix=(float(pm), float(pse)), hold_elig=mse(dhj[eligh]), w_val=wsj, w_hold=wshj,
                                    beta_val_mean=np.mean(bj, 0).tolist(), beta_hold_mean=np.mean(bhj, 0).tolist())
DV["JOINT_ridge"], DH["JOINT_ridge"] = dvj, dhj
log(f"JOINT_ridge    val {fmt(dvj[keep])} | pmix {pm:+.5f} ± {pse:.5f} | holdout(elig) {fmt(dhj[eligh])} | w val {sorted(set(wsj))} "
    f"hold {sorted(set(wshj))} | beta(val, [{', '.join(JS)}, zG3]) {np.round(np.mean(bj, 0), 5).tolist()} [{time.time() - t:.0f}s]")
np.savez_compressed(OUT / "deltas_val.npz", keep=keep, **DV)
np.savez_compressed(OUT / "deltas_holdout.npz", eligible=eligh, **DH, **{f"{k}_all": v for k, v in DHA.items()})

# ---------------------------------------------------------------- gate
log("=== GATE (PREDECLARED.txt)")
verdicts = {}
for k in PRIMARY:
    r = RES["nested"][k]
    vm, vs = r["val"]; hmn = r["hold_elig"][0]
    task_pass = (vm >= 0.0010) and (vm >= 3 * vs) and (hmn > 0)
    extras = dict(pmix_ok=r["pmix"][0] >= -r["pmix"][1], corr_cos_u_ok=abs(RES["corr_cos_u"][k]) < 0.8,
                  cell_ok=all(RES["cell_auc"][k][0] > RES["cell_auc"][c][0] for c in ctrl))
    v = "PASS" if task_pass and all(extras.values()) else ("PASS-CONDITIONAL" if task_pass else "FAIL")
    verdicts[k] = dict(task_gate=bool(task_pass), extras={a: bool(b) for a, b in extras.items()}, verdict=v)
    log(f"{k:14s} val {vm:+.5f} ± {vs:.5f} (>= +0.0010: {vm >= 0.001}; >= 3 SE: {vm >= 3 * vs}) holdout(elig) {hmn:+.5f} (> 0: {hmn > 0}) "
        f"-> task gate {'PASS' if task_pass else 'FAIL'}; extras {extras} -> {v}")
overall = "PASS" if any(v["verdict"] == "PASS" for v in verdicts.values()) else (
    "PASS-CONDITIONAL" if any(v["verdict"] == "PASS-CONDITIONAL" for v in verdicts.values()) else "CLOSED")
RES["gate"] = dict(arms=verdicts, overall=overall)
log(f"P4 VERDICT: {overall}")
json.dump(RES, open(OUT / "uknn_gate.json", "w"), indent=1, default=float)
log("done")
