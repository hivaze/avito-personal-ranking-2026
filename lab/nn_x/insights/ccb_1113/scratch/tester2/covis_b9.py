"""C05 (IDF / IUF / Swing / BM25 co-vis counts) + C11 (typed implicit rating cvx) on ONE B9 co-visitation index.
Copy of feats_v2/seq_/feat_covis.py logic (window +-20, CAP 3000 latest source occurrences of c, same non-zero user skipped,
w_x = 0.95^hpos of x's newest occurrence in the feed's history), source rule changed to B9 (PREDECLARED_ccb.txt):
  val row of day q: train de-duplicated user sequences (latest train feed per non-zero user + every user-0 train feed, = sources.py
     source_fidx(None)) + every val feed of a day < q (other users; test never a source);
  holdout row of train day d: sources.py source_fidx(d) (latest train feed with day < d per non-zero user + user-0 feeds day < d).
One union index holds, per non-zero user, the latest feed of each train day (valid for query day q iff day < q <= the user's next
train day) + user-0 train feeds + val feeds (valid iff day < q); the CAP counts VALID postings from the newest backwards, i.e. it
equals the per-day index of feat_covis. df / pop / N_seq are per query day (valid sequences only).
Columns: C05 cv_iuf, cv_idf, cv_bm25, cv_invd20_idfdamp, swing_max; C11 cvx_g1, cvx_g2, cvx_g3, cvx_intent, cvx_tw;
INFORMATION: cv_invd20 (the plain B9 invd20).
  python covis_b9.py [nfeeds_val_test]
"""
import sys
sys.path.insert(0, "/tmp/scratchpad/ccb/tester2")
from harness import *  # noqa: F401,F403
from emb_util import ranges

NAMES = ["cv_iuf", "cv_idf", "cv_bm25", "cv_invd20_idfdamp", "swing_max", "cvx_g1", "cvx_g2", "cvx_g3", "cvx_intent",
         "cvx_tw", "cv_invd20"]
CAP, WIN, UCAP, NLAST = 3000, 20, 200, 32
BIG = 1 << 30
MSK = 10800


@nb.njit(parallel=True, cache=True)
def gather(sel, hoff, H, src, out_off, out):
    for q in nb.prange(len(sel)):
        f = sel[q]; o = out_off[q]; ho = hoff[f]
        for j in range(H[f]):
            out[o + j] = src[ho + j]


@nb.njit(parallel=True, cache=True)
def seg_sort_unique(dense, starts, lens, sitems, sulen):
    for s in nb.prange(len(lens)):
        st = starts[s]; L = lens[s]
        tmp = np.sort(dense[st:st + L])
        n = 0
        for i in range(L):
            if i == 0 or tmp[i] != tmp[i - 1]:
                sitems[st + n] = tmp[i]; n += 1
        sulen[s] = n


@nb.njit(cache=True)
def stage_counts(dense, starts, lens, sitems, sulen, k0, k1, nst, pop, df, nseq):
    for s in range(len(lens)):
        a = k0[s]; bb = k1[s]
        if a >= bb:
            continue
        nseq[a] += 1
        if bb < nst:
            nseq[bb] -= 1
        st = starts[s]
        for i in range(lens[s]):
            x = dense[st + i]
            pop[a, x] += 1
            if bb < nst:
                pop[bb, x] -= 1
        for i in range(sulen[s]):
            x = sitems[st + i]
            df[a, x] += 1
            if bb < nst:
                df[bb, x] -= 1


@nb.njit(cache=True)
def inter_count(sitems, a0, la, b0, lb):
    i = 0; j = 0; c = 0
    while i < la and j < lb:
        x = sitems[a0 + i]; y = sitems[b0 + j]
        if x == y:
            c += 1; i += 1; j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    return c


@nb.njit(parallel=True, cache=True)
def query(fsel, fstage, fday, fuser, coff, M, hoff, H, citem, hitem, hpos, htype,
          items, starts, lens, susers, slo, shi, sitems, sulen, uitems, ustart, ucnt, p_seq, p_pos,
          pop, df, nseq, out, work):
    for q in nb.prange(len(fsel)):
        fi = fsel[q]; kst = fstage[q]; qd = fday[q]; u = fuser[q]
        h = H[fi]; ho = hoff[fi]
        hs = np.empty(h, np.int64)
        for j in range(h):
            hs[j] = hitem[ho + j]
        order = np.argsort(hs)
        nu = 0
        uh = np.empty(h, np.int64); uw = np.empty(h); g1 = np.zeros(h); g2 = np.zeros(h); g3 = np.zeros(h)
        umin = np.empty(h, np.int64)
        for jj in range(h):
            j = order[jj]
            w = 0.95 ** hpos[ho + j]; ty = htype[ho + j]
            if not (nu > 0 and uh[nu - 1] == hs[j]):
                uh[nu] = hs[j]; uw[nu] = 0.0; g1[nu] = 0.0; g2[nu] = 0.0; g3[nu] = 0.0; umin[nu] = 1 << 30; nu += 1
            k = nu - 1
            if w > uw[k]:
                uw[k] = w
            if ty == 1:
                if w > g1[k]:
                    g1[k] = w
            elif ty >= 2 and ty <= 4:
                if w > g2[k]:
                    g2[k] = w
            elif ty >= 5:
                if w > g3[k]:
                    g3[k] = w
            if hpos[ho + j] < umin[k]:
                umin[k] = hpos[ho + j]
        idf = np.zeros(nu); dmp = np.zeros(nu)
        Ns = nseq[kst] if kst >= 0 else 0
        for k in range(nu):
            kk = np.searchsorted(uitems, uh[k])
            dfx = 0; px = 0
            if kk < len(uitems) and uitems[kk] == uh[k] and kst >= 0:
                dfx = df[kst, kk]; px = pop[kst, kk]
            idf[k] = np.log((Ns + 1.0) / (1.0 + dfx))
            dmp[k] = 1.0 / np.log(2.0 + px)
        slot = np.full(nu, -1, np.int64)
        if nu > 0:
            o2 = np.argsort(umin[:nu])
            for j in range(min(NLAST, nu)):
                slot[o2[j]] = j
        hitc = np.zeros(nu, np.int64); touched = np.empty(nu, np.int64)
        U = np.empty((NLAST, UCAP), np.int64); ul = np.zeros(NLAST, np.int64)
        co = coff[fi]
        for i in range(M[fi]):
            r = co + i
            for z in range(out.shape[1]):
                out[r, z] = 0.0
            if nu == 0 or kst < 0:
                continue
            c = citem[r]
            k = np.searchsorted(uitems, c)
            if k >= len(uitems) or uitems[k] != c:
                continue
            ps = ustart[k]; pe = ustart[k] + ucnt[k]
            lo_ = ps; hi_ = pe          # first posting with slo >= qd
            while lo_ < hi_:
                mid = (lo_ + hi_) // 2
                if slo[p_seq[mid]] < qd:
                    lo_ = mid + 1
                else:
                    hi_ = mid
            end = lo_
            a_iuf = 0.0; a_idf = 0.0; a_dmp = 0.0; a_inv = 0.0; a1 = 0.0; a2 = 0.0; a3 = 0.0
            nt = 0; nvalid = 0
            for z in range(NLAST):
                ul[z] = 0
            e = end - 1
            while e >= ps:
                s = p_seq[e]
                e -= 1
                if shi[s] < qd:
                    continue
                nvalid += 1
                if nvalid > CAP:
                    break
                if u != 0 and susers[s] == u:
                    continue
                p = p_pos[e + 1]; st = starts[s]; L = lens[s]
                wlo = max(0, p - WIN); whi = min(L, p + WIN + 1)
                liuf = 1.0 / np.log(2.0 + L)
                for qq in range(wlo, whi):
                    if qq == p:
                        continue
                    x = items[st + qq]
                    if x == c:
                        continue
                    kk = np.searchsorted(uh[:nu], x)
                    if kk < nu and uh[kk] == x:
                        d = abs(qq - p); w = uw[kk]
                        a_inv += w / d
                        a_iuf += w / d * liuf
                        a_idf += w * idf[kk] / d
                        a_dmp += w * dmp[kk] / d
                        a1 += g1[kk] / d; a2 += g2[kk] / d; a3 += g3[kk] / d
                        if hitc[kk] == 0:
                            touched[nt] = kk; nt += 1
                        hitc[kk] += 1
                        sl = slot[kk]
                        if sl >= 0 and ul[sl] < UCAP and (ul[sl] == 0 or U[sl, ul[sl] - 1] != s):
                            U[sl, ul[sl]] = s; ul[sl] += 1
            bm = 0.0
            for t in range(nt):
                kk = touched[t]; n = hitc[kk]
                bm += idf[kk] * (n * 2.2) / (n + 1.2)
                hitc[kk] = 0
            sw = 0.0
            if a_inv > 0:
                for sl in range(NLAST):
                    m = ul[sl]
                    if m < 2:
                        continue
                    acc = 0.0
                    for a in range(m):
                        sa = U[sl, a]
                        for bq in range(a + 1, m):
                            sb = U[sl, bq]
                            acc += 1.0 / (1.0 + inter_count(sitems, starts[sa], sulen[sa], starts[sb], sulen[sb]))
                            work[q] += 1
                    if acc > sw:
                        sw = acc
            out[r, 0] = a_iuf; out[r, 1] = a_idf; out[r, 2] = bm; out[r, 3] = a_dmp; out[r, 4] = sw
            out[r, 5] = a1; out[r, 6] = a2; out[r, 7] = a3
            out[r, 8] = a2 / (a1 + a2 + a3 + 1e-9); out[r, 9] = a1 + 6.0 * a2 + 3.0 * a3
            out[r, 10] = a_inv


def build_index():
    t = time.time()
    ft = pl.read_parquet(FLAT / "train/feeds.parquet").with_columns(
        day=((pl.col("timestamp").cast(pl.Int64) + MSK) // 86400))
    nz = ft.filter(pl.col("user_id") != 0).sort(["user_id", "timestamp", "fidx"]).group_by(["user_id", "day"]).agg(
        pl.col("fidx").last()).sort(["user_id", "day"])
    nz = nz.with_columns(hi=pl.col("day").shift(-1).over("user_id").fill_null(BIG))
    z0 = ft.filter(pl.col("user_id") == 0).select("user_id", "day", "fidx").with_columns(hi=pl.lit(BIG, pl.Int64))
    tr = pl.concat([nz.select("user_id", "day", "fidx", pl.col("hi").cast(pl.Int64)),
                    z0.select("user_id", "day", "fidx", "hi")]).sort("fidx")
    fvv = pl.read_parquet(FLAT / "val/feeds.parquet").with_columns(day=((pl.col("timestamp").cast(pl.Int64) + MSK) // 86400))
    parts = [("train", tr["fidx"].to_numpy().astype(np.int64), tr["user_id"].to_numpy(), tr["day"].to_numpy(), tr["hi"].to_numpy()),
             ("val", fvv["fidx"].to_numpy().astype(np.int64), fvv["user_id"].to_numpy(), fvv["day"].to_numpy(),
              np.full(len(fvv), BIG, np.int64))]
    its, lens, users, los, his = [], [], [], [], []
    for split, sel, us, lo, hi in parts:
        f = pl.read_parquet(FLAT / split / "feeds.parquet")
        H = f["H"].to_numpy().astype(np.int64); ho = f["hist_off"].to_numpy().astype(np.int64)
        h = H[sel]; off = np.concatenate([[0], np.cumsum(h)[:-1]]).astype(np.int64)
        it = np.empty(int(h.sum()), np.int32)
        gather(sel, ho, H, np.load(FLAT / split / "hist_item_id.npy", mmap_mode="r"), off, it)
        its.append(it); lens.append(h); users.append(us.astype(np.int64)); los.append(lo.astype(np.int64)); his.append(hi.astype(np.int64))
        log(f"index part {split}: {len(sel)} seqs, {int(h.sum())} events ({time.time()-t:.0f}s)")
    I = dict(items=np.concatenate(its), lens=np.concatenate(lens), users=np.concatenate(users),
             lo=np.concatenate(los), hi=np.concatenate(his))
    del its
    I["starts"] = np.concatenate([[0], np.cumsum(I["lens"])[:-1]]).astype(np.int64)
    assert (np.diff(I["lo"]) >= 0).all(), "sequence lo-days must be monotone (fidx time order)"
    order = np.argsort(I["items"], kind="stable")
    si = I["items"][order]
    brk = np.empty(len(si), np.bool_); brk[0] = True; brk[1:] = si[1:] != si[:-1]
    ustart = np.flatnonzero(brk).astype(np.int64)
    I["uitems"] = si[ustart].astype(np.int64)
    I["ucnt"] = np.diff(np.append(ustart, len(si))).astype(np.int64); I["ustart"] = ustart
    dense_sorted = (np.cumsum(brk) - 1).astype(np.int32)
    dense = np.empty(len(si), np.int32); dense[order] = dense_sorted
    del si, brk, dense_sorted
    seq = np.repeat(np.arange(len(I["lens"]), dtype=np.int32), I["lens"])
    I["p_seq"] = seq[order]
    I["p_pos"] = (np.arange(len(order), dtype=np.int64) - np.repeat(I["starts"], I["lens"]))[order].astype(np.int16)
    del order, seq
    I["sitems"] = np.empty(len(dense), np.int32); I["sulen"] = np.empty(len(I["lens"]), np.int64)
    seg_sort_unique(dense, I["starts"], I["lens"], I["sitems"], I["sulen"])
    I["dense"] = dense
    log(f"index: {len(I['lens'])} seqs, {len(I['items'])} events, {len(I['uitems'])} items ({time.time()-t:.0f}s)")
    return I


def stage_stats(I, QD):
    t = time.time()
    nst = len(QD)
    k0 = np.searchsorted(QD, I["lo"], side="right").astype(np.int64)
    k1 = np.searchsorted(QD, np.minimum(I["hi"], 1 << 40), side="right").astype(np.int64)
    pop = np.zeros((nst, len(I["uitems"])), np.int32); df = np.zeros((nst, len(I["uitems"])), np.int32)
    nseq = np.zeros(nst, np.int64)
    stage_counts(I["dense"], I["starts"], I["lens"], I["sitems"], I["sulen"], k0, k1, nst, pop, df, nseq)
    np.cumsum(pop, axis=0, out=pop); np.cumsum(df, axis=0, out=df); nseq = np.cumsum(nseq)
    log(f"stage stats: {nst} query days {QD.tolist()}; N_seq per stage {nseq.tolist()} ({time.time()-t:.0f}s)")
    return pop, df, nseq


def run(I, pop, df, nseq, QD, split, sel, log_every=None):
    t = time.time()
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    H = f["H"].to_numpy().astype(np.int64)[sel]; ho = f["hist_off"].to_numpy().astype(np.int64)[sel]
    M = f["M"].to_numpy().astype(np.int64)[sel]; co = f["cand_off"].to_numpy().astype(np.int64)[sel]
    day = (f["timestamp"].to_numpy().astype(np.int64)[sel] + MSK) // 86400
    user = f["user_id"].to_numpy().astype(np.int64)[sel]
    hr = ranges(ho, H); cr = ranges(co, M)
    A = lambda n, r: np.asarray(np.load(FLAT / split / f"{n}.npy", mmap_mode="r")[r])  # noqa: E731
    citem = A("cand_item_id", cr).astype(np.int64); hitem = A("hist_item_id", hr).astype(np.int64)
    hpos = A("hist_hpos", hr).astype(np.int64); htype = A("hist_event_type", hr).astype(np.int64)
    hoff = np.concatenate([[0], np.cumsum(H)[:-1]]).astype(np.int64); coff = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
    k = np.searchsorted(QD, day); assert (QD[k] == day).all()
    # a stage with no valid source sequence (first train day) -> zeros, as feat_covis run_train
    stage = np.where(nseq[k] > 0, k, -1).astype(np.int64)
    out = np.zeros((len(citem), len(NAMES)), np.float64); work = np.zeros(len(sel), np.int64)
    query(np.arange(len(sel), dtype=np.int64), stage, day, user, coff, M, hoff, H, citem, hitem, hpos, htype,
          I["items"], I["starts"], I["lens"], I["users"], I["lo"], I["hi"], I["sitems"], I["sulen"], I["uitems"],
          I["ustart"], I["ucnt"], I["p_seq"], I["p_pos"], pop, df, nseq, out, work)
    log(f"query {split}: {len(sel)} feeds, {len(out)} rows, swing pair-ops {work.sum():.3e} (max/feed {work.max():.3e}), "
        f"{(stage < 0).sum()} feeds with empty source ({time.time()-t:.0f}s)")
    return out


if __name__ == "__main__":
    nb.set_num_threads(12)
    LOG = open(HERE / "logs/covis.log", "a")
    log = log_to(LOG)
    log("==== covis_b9 (C05 + C11)", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), sys.argv)
    ntest = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    QD = np.unique(np.concatenate([day_v, HOLD_DAY_ABS])).astype(np.int64)
    I = build_index()
    pop, df, nseq = stage_stats(I, QD)
    if ntest:
        sel = np.random.default_rng(1).choice(nF, ntest, replace=False); sel.sort()
        o = run(I, pop, df, nseq, QD, "val", sel)
        for i, n in enumerate(NAMES):
            log(f"  test {n:20s} nonzero {np.mean(o[:, i] != 0):.4f} mean {o[:, i].mean():.4g}")
        selh = np.sort(np.random.default_rng(1).choice(HOLD, ntest, replace=False))
        o = run(I, pop, df, nseq, QD, "train", selh)
    Xv = run(I, pop, df, nseq, QD, "val", np.arange(nF)); assert len(Xv) == len(yv)
    np.save(FEATD / "covis_val.npy", Xv.astype(np.float32))
    Xh = run(I, pop, df, nseq, QD, "train", HOLD); assert len(Xh) == len(yh)
    np.save(FEATD / "covis_hold.npy", Xh.astype(np.float32))
    old = np.load(ROOT / "data/feats/val/seq_cv_invd20.npy")
    log(f"INFO B9 cv_invd20 vs on-disk seq_cv_invd20 (timestamp-causal, test-including source): corr "
        f"{np.corrcoef(Xv[:, 10], old)[0, 1]:.4f}; nonzero share B9 {np.mean(Xv[:, 10] > 0):.4f} vs disk {np.mean(old > 0):.4f}")
    log("BUILT")
