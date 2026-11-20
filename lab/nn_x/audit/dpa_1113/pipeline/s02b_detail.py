"""Input pipeline audit, step 2b: targeted measurements behind the suspicious items of s02.
 A. attention-mass inputs (ctx_hn_soft05, ctx_hs_k128/e1/t10): cap-10 binding, positives at the cap, within-feed ties at
    the cap, compression of the bulk in standardised units.
 B. popularity rel features: PSI of the trainer's mid-rank / z (cold rows tied) vs a warm-only rank, val->pub/priv.
 C. pre2 family: PSI on present rows only; Sep 6-8 regime share in trn / holdout (per feed).
 D. seq_cv_pop builder cap (3000).
 E. categorical inputs: PSI of candidate category, history event type / source / (type, source) / category, H.
CPU only. Output out/s02b.log
"""
import os, sys
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s02_stats import *  # noqa
import numba as nb

LOG = open(OUT / "s02b.log", "w")


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


@nb.njit(parallel=True, cache=True)
def warm_rank(x, off, M):
    """rank among warm rows (x > 0) in [0, 1] (mid-rank ties), cold rows -> -1 (sentinel); 0.5 if 1 warm row."""
    out = np.full(len(x), -1.0)
    for i in nb.prange(len(M)):
        o = off[i]; m = M[i]; nw = 0
        for k in range(m):
            nw += x[o + k] > 0
        for k in range(m):
            v = x[o + k]
            if v <= 0:
                continue
            if nw == 1:
                out[o + k] = 0.5; continue
            less = 0; eq = 0
            for j in range(m):
                u = x[o + j]
                if u <= 0:
                    continue
                if u < v:
                    less += 1
                elif u == v:
                    eq += 1
            out[o + k] = (less + 0.5 * (eq - 1)) / (nw - 1)
    return out


@nb.njit(parallel=True, cache=True)
def ties_at_cap(x, y, off, M, cap):
    nF = len(M); nc = np.zeros(nF); pc = np.zeros(nF); npos = np.zeros(nF)
    for i in nb.prange(nF):
        o = off[i]
        for k in range(M[i]):
            if x[o + k] >= cap:
                nc[i] += 1
                pc[i] += y[o + k]
            npos[i] += y[o + k]
    return nc, pc, npos


def main():
    D = {}
    for sub in ["trn", "hold", "val", "test"]:
        X, R, mt = load(sub)
        if sub == "test":
            pubf = mt["ts"] < BOUND
            for nm, sel in [("pub", pubf), ("priv", ~pubf)]:
                rows = np.repeat(sel, mt["M"]); Ms = mt["M"][sel]
                D[nm] = dict(X=np.asarray(X[rows]), R=np.asarray(R[rows]), M=Ms,
                             off=np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64), y=None, ts=mt["ts"][sel])
        else:
            D[sub] = dict(X=np.asarray(X), R=np.asarray(R), M=mt["M"], off=mt["off"], y=mt["y"] if len(mt["y"]) else None,
                          ts=mt["ts"], fidx=mt["fidx"])
    xi = {n: j for j, n in enumerate(XT)}
    # A ---------------------------------------------------------------------------------------------------------------
    log("== A. attention-mass inputs: cap 10 (attach_xt clip) ==")
    for nm in ["ctx_hn_soft05", "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_t10"]:
        for s in ["trn", "hold", "val", "priv"]:
            r = D[s]["R"][:, xi[nm]].astype(np.float64); r0 = np.nan_to_num(r, nan=0.0)
            line = f"  {nm:14s} {s:5s} rows>=10 {np.mean(r0 >= 10):.4f}"
            nc, pc, npos = ties_at_cap(r0, (D[s]["y"] if D[s]["y"] is not None else np.zeros(len(r0))).astype(np.float64),
                                       D[s]["off"], D[s]["M"], 10.0)
            line += f" | feeds with >=1 / >=2 capped {np.mean(nc >= 1):.3f} / {np.mean(nc >= 2):.3f}"
            if D[s]["y"] is not None:
                y = D[s]["y"]
                line += f" | CTR capped {y[r0 >= 10].mean():.4f} vs all {y.mean():.4f}; share of positives capped {pc.sum()/npos.sum():.4f}"
                fe = (nc >= 2) & (pc >= 1)
                line += f"; feeds with a positive among >=2 capped {fe.mean():.4f}"
            # bulk compression: standardised width of the p10..p90 band
            j = NAMES.index("xt:" + nm)
            q = np.quantile(np.clip(r0, -10, 10), [0.1, 0.5, 0.9])
            line += f" | p10/p50/p90 raw {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f} -> std units width {(q[2]-q[0])/FS[j]:.3f}"
            log(line)
        # log-version for comparison
        r0 = np.nan_to_num(D["val"]["R"][:, xi[nm]].astype(np.float64), nan=0.0)
        lr = np.log1p(r0); qq = np.quantile(lr, [0.1, 0.9])
        log(f"  {nm:14s} val log1p version: p10..p90 width in its own std units {(qq[1]-qq[0])/lr.std():.3f}; p999/p50 raw {np.quantile(r0,0.999)/max(np.median(r0),1e-9):.1f}")
    # B ---------------------------------------------------------------------------------------------------------------
    log("\n== B. popularity rel inputs: trainer mid-rank (cold rows tied) vs warm-only rank; PSI ==")
    for nm in ["it_hist_all_luo", "seq_cv_invd20", "it_hist_et1_luo", "seq_cv_pop"]:
        wr = {s: warm_rank(np.nan_to_num(D[s]["R"][:, xi[nm]].astype(np.float64)), D[s]["off"], D[s]["M"]) for s in D}
        line = f"  {nm:16s} warm-only rank PSI trn->val {psi(wr['trn'], wr['val']):.3f} trn->priv {psi(wr['trn'], wr['priv']):.3f} " \
               f"val->pub {psi(wr['val'], wr['pub']):.3f} val->priv {psi(wr['val'], wr['priv']):.3f} hold->val {psi(wr['hold'], wr['val']):.3f}"
        # warm rows only PSI
        wo = {s: wr[s][wr[s] >= 0] for s in wr}
        line += f" | warm rows only val->priv {psi(wo['val'], wo['priv']):.4f}"
        if ("xt:" + nm + "_rkf") in NAMES:
            j = NAMES.index("xt:" + nm + "_rkf"); jz = NAMES.index("xt:" + nm + "_zf")
            line += f" | trainer rk val->priv {psi(D['val']['X'][:, j].astype(np.float64), D['priv']['X'][:, j].astype(np.float64)):.3f}" \
                    f" z {psi(D['val']['X'][:, jz].astype(np.float64), D['priv']['X'][:, jz].astype(np.float64)):.3f}"
            # value of a cold row's rank: mean by split
            for s in ["trn", "hold", "val", "pub", "priv"]:
                cold = np.nan_to_num(D[s]["R"][:, xi[nm]]) <= 0
                line += f" | {s} cold-row rk {D[s]['X'][cold, j].mean():.3f} warm-row rk {D[s]['X'][~cold, j].mean():.3f} cold {cold.mean():.3f}"
        log(line)
        np.save(OUT / f"val_warmrank_{nm}.npy", wr["val"]); np.save(OUT / f"hold_warmrank_{nm}.npy", wr["hold"])
    # C ---------------------------------------------------------------------------------------------------------------
    log("\n== C. pre2 family ==")
    for nm in ["pre2_fe_s", "pre2_new_s", "pre2_hmax32", "pre2_fe_z"]:
        p = {s: D[s]["R"][:, xi[nm]].astype(np.float64) for s in D}
        log(f"  {nm:12s} PSI present rows trn->val {psi(p['trn'], p['val']):.3f} trn->priv {psi(p['trn'], p['priv']):.3f} "
            f"hold->val {psi(p['hold'], p['val']):.3f} val->priv {psi(p['val'], p['priv']):.4f}")
    for s in ["trn", "hold"]:
        pn = np.isnan(D[s]["R"][:, xi["pre2_fe"] if "pre2_fe" in xi else xi["pre2_fe_s"]])
        fid = np.repeat(np.arange(len(D[s]["M"])), D[s]["M"])
        fr = np.bincount(fid, weights=pn.astype(float), minlength=len(D[s]["M"])) / D[s]["M"]
        day = (D[s]["ts"].astype(np.int64) + 10800) // 86400; d0 = day.min()
        log(f"  {s}: rows pre2 NaN {pn.mean():.4f}; feeds all-NaN {np.mean(fr == 1):.4f}, partial {np.mean((fr > 0) & (fr < 1)):.4f}; "
            f"feeds by day(Moscow) from Sep 6: " + " ".join(f"{int(dd-d0)}:{np.mean(day == dd):.3f}/{fr[day == dd].mean():.2f}" for dd in np.unique(day)))
    # D ---------------------------------------------------------------------------------------------------------------
    log("\n== D. builder caps ==")
    for nm in ["seq_cv_pop", "it_hist_all_luo"]:
        mx = max(np.nanmax(D[s]["R"][:, xi[nm]]) for s in D)
        log(f"  {nm}: max {mx:.0f}; share at max " + " ".join(f"{s} {np.mean(D[s]['R'][:, xi[nm]] >= mx):.5f}" for s in D))
    # E ---------------------------------------------------------------------------------------------------------------
    log("\n== E. categorical inputs (PSI over category frequencies) ==")
    FL = ROOT / "data/flat"

    def cat_dist(split, fe, name, nmax):
        f = pl.read_parquet(FL / split / "feeds.parquet")
        if name.startswith("cand"):
            o = f["cand_off"].to_numpy()[fe]; L = f["M"].to_numpy()[fe]
        else:
            o = f["hist_off"].to_numpy()[fe]; L = f["H"].to_numpy()[fe]
        rows = np.concatenate([np.arange(a, a + l) for a, l in zip(o, L)])
        a = np.asarray(np.load(FL / split / f"{name}.npy", mmap_mode="r")[rows]).astype(np.int64)
        return np.bincount(a, minlength=nmax)[:nmax] / len(a), a, rows

    def psi_c(p, q):
        p = np.clip(p, 1e-5, None); q = np.clip(q, 1e-5, None)
        return float(((q - p) * np.log(q / p)).sum())
    sp = {"trn": ("train", np.load(OUT / "trn_meta.npz")["fidx"]), "hold": ("train", np.load(OUT / "hold_meta.npz")["fidx"]),
          "val": ("val", np.arange(14423))}
    tm = np.load(OUT / "test_meta.npz"); sp["pub"] = ("test", np.where(tm["ts"] < BOUND)[0]); sp["priv"] = ("test", np.where(tm["ts"] >= BOUND)[0])
    for name, nmax in [("cand_category_id", 60), ("hist_category_id", 60), ("hist_event_type", 12), ("hist_event_source", 32)]:
        P = {k: cat_dist(sp[k][0], sp[k][1], name, nmax)[0] for k in sp}
        log(f"  {name:18s} PSI trn->val {psi_c(P['trn'], P['val']):.4f} trn->priv {psi_c(P['trn'], P['priv']):.4f} "
            f"hold->val {psi_c(P['hold'], P['val']):.4f} val->pub {psi_c(P['val'], P['pub']):.4f} val->priv {psi_c(P['val'], P['priv']):.4f}; "
            f"#codes seen val {int((P['val'] > 0).sum())}")
    # joint (type, source)
    J = {}
    for k in sp:
        _, et, rows = cat_dist(sp[k][0], sp[k][1], "hist_event_type", 12)
        es = np.asarray(np.load(FL / sp[k][0] / "hist_event_source.npy", mmap_mode="r")[rows]).astype(np.int64)
        J[k] = np.bincount(et * 32 + es, minlength=12 * 32) / len(et)
    nz = (J["val"] > 0).sum(); top = np.sort(J["val"])[::-1]
    log(f"  (event_type, source) joint: {nz} pairs seen on val; top-10 cover {top[:10].sum():.3f}; pairs with >=0.1% {np.sum(J['val'] >= 0.001)}; "
        f"PSI trn->val {psi_c(J['trn'], J['val']):.4f} val->priv {psi_c(J['val'], J['priv']):.4f}")
    for k in ["trn", "hold", "val", "pub", "priv"]:
        Hh = pl.read_parquet(FL / sp[k][0] / "feeds.parquet")["H"].to_numpy()[sp[k][1]]
        Mm = pl.read_parquet(FL / sp[k][0] / "feeds.parquet")["M"].to_numpy()[sp[k][1]]
        log(f"  {k:5s} H: mean {Hh.mean():.1f} share H=0 {np.mean(Hh == 0):.4f} share H=256 {np.mean(Hh == 256):.3f} | M mean {Mm.mean():.1f} median {np.median(Mm):.0f} share M>=50 {np.mean(Mm >= 50):.3f}")


if __name__ == "__main__":
    main()
