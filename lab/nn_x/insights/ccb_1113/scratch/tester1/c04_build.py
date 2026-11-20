"""C04 co-vis variants + OTTO-7th NB log-lift. Copy of feats_v2/seq_/feat_covis.py (Index / _covis) with the B9 source:
  val rows: all-train source (sources.source_fidx(None)) + OTHER users' val feed histories on STRICTLY EARLIER MSK days
            (test never a source); same non-zero user's sequences skipped.
  holdout rows (train day d): train source of days < d (sources.source_fidx(d), = feat_covis run_train), same user skipped.
Columns: seq_cv_w5, seq_cv_fwd5, seq_cv_t1, seq_cv_seqs (feat_covis definitions, CAP 3000 latest occurrences, window +-20 /
d <= 5) and nb_loglift = log P(c) + sum_{i=1..10} ((11-i)/10) * log[(n(h_i,type_i -> c) + 0.1) / (n(h_i,type_i) * P(c) + 0.1)],
h_i = i-th newest history event; n(h,type -> c) = co-occurrences of c with (h,type) within +-5 positions in the allowed source
(counted over c's latest CAP allowed occurrences, scaled by n_c / scanned when capped); n_c, n(h,type) = exact allowed
occurrence counts; P(c) = n_c / N, log P(c) = log((n_c + 1)/(N + 1)), N = allowed source events (per query day; the same-user
exclusion is not subtracted from N, a feed-constant approximation).
Outputs out/c04_<col>_{val,hold}.npy."""
import os, sys, time
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
import numpy as np, polars as pl, numba as nb
from pathlib import Path
HERE = Path(__file__).resolve().parent; OUT = HERE / "out"
ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"
MSK = 3 * 3600; CAP = 3000
COLS = ["seq_cv_w5", "seq_cv_fwd5", "seq_cv_t1", "seq_cv_seqs", "nb_loglift"]
T0 = time.time()
_F = {}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def feeds(split):
    if split not in _F:
        _F[split] = pl.read_parquet(FLAT / split / "feeds.parquet")
    return _F[split]


def arr(split, name):
    return np.load(FLAT / split / f"{name}.npy", mmap_mode="r")


def day_of(ts):
    return (np.asarray(ts, dtype=np.int64) + MSK) // 86400


def train_days():
    d = day_of(feeds("train")["timestamp"].to_numpy())
    u, c = np.unique(d, return_counts=True)
    return u[c > 1000]


def source_fidx(cutoff_day=None):
    f = feeds("train").with_columns(day=((pl.col("timestamp").cast(pl.Int64) + MSK) // 86400))
    if cutoff_day is not None:
        f = f.filter(pl.col("day") < int(cutoff_day))
    nz = f.filter(pl.col("user_id") != 0).sort(["user_id", "timestamp", "fidx"]).group_by("user_id").agg(
        pl.col("fidx").last())["fidx"].to_numpy()
    z = f.filter(pl.col("user_id") == 0)["fidx"].to_numpy()
    return np.sort(np.concatenate([nz, z])).astype(np.int64)


@nb.njit(parallel=True)
def _gather(sel, hist_off, H, src_item, src_type, out_off, items, types):
    for q in nb.prange(len(sel)):
        f = sel[q]; o = out_off[q]; ho = hist_off[f]
        for j in range(H[f]):
            items[o + j] = src_item[ho + j]
            types[o + j] = src_type[ho + j]


class Index:
    def __init__(self, parts):
        """parts: list of (split, fidx array, constrained: bool). Constrained sequences are usable only by rows of a LATER day."""
        its, tys, lens, users, days, cons = [], [], [], [], [], []
        for split, sel, constrained in parts:
            f = feeds(split)
            H = f["H"].to_numpy().astype(np.int64); ho = f["hist_off"].to_numpy()
            h = H[sel]
            off = np.concatenate([[0], np.cumsum(h)[:-1]]).astype(np.int64)
            it = np.empty(h.sum(), np.int32); ty = np.empty(h.sum(), np.int8)
            _gather(sel, ho, H, np.asarray(arr(split, "hist_item_id")), np.asarray(arr(split, "hist_event_type")), off, it, ty)
            its.append(it); tys.append(ty); lens.append(h)
            users.append(f["user_id"].to_numpy()[sel]); days.append(day_of(f["timestamp"].to_numpy()[sel]))
            cons.append(np.full(len(sel), constrained, np.bool_))
        self.items = np.concatenate(its); self.types = np.concatenate(tys)
        self.lens = np.concatenate(lens).astype(np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.lens)[:-1]]).astype(np.int64)
        self.users = np.concatenate(users).astype(np.int64)
        self.days = np.concatenate(days).astype(np.int64); self.cons = np.concatenate(cons)
        seq = np.repeat(np.arange(len(self.lens), dtype=np.int32), self.lens)
        pos = (np.arange(len(self.items), dtype=np.int64) - np.repeat(self.starts, self.lens)).astype(np.int16)
        order = np.argsort(self.items, kind="stable")
        si = self.items[order]
        self.uitems, ustart, ucnt = np.unique(si, return_index=True, return_counts=True)
        self.ustart = ustart.astype(np.int64); self.ucnt = ucnt.astype(np.int64)
        self.p_seq = seq[order]; self.p_pos = pos[order]
        # N per query day: events in unconstrained sequences + constrained sequences of earlier days
        self.n_uncon = int(self.lens[~self.cons].sum())
        cd = self.days[self.cons]; cl = self.lens[self.cons]
        self.cdays = np.unique(cd) if len(cd) else np.zeros(0, np.int64)
        self.cday_tot = np.array([cl[cd == d].sum() for d in self.cdays], np.int64)
        del order, si, seq, pos
        log(f"index: {len(self.lens)} seqs, {len(self.items)} events, {len(self.uitems)} items")

    def N_for_day(self, d):
        return self.n_uncon + int(self.cday_tot[self.cdays < d].sum())


@nb.njit(parallel=True)
def _covis(sel, coff, M, hoff, H, fuser, fday, Nday, citem, hitem, hpos, htype, items, types, starts, lens, susers, sdays,
           scons, uitems, ustart, ucnt, p_seq, p_pos, ooff, out):
    for q in nb.prange(len(sel)):
        fi = sel[q]
        h = H[fi]; ho = hoff[fi]; u = fuser[fi]; dd = fday[fi]; N = Nday[q]
        hs = np.empty(h, np.int32)
        for j in range(h):
            hs[j] = hitem[ho + j]
        order = np.argsort(hs)
        nu = 0
        uh = np.empty(h, np.int32); uw = np.empty(h, np.float64)
        for jj in range(h):
            j = order[jj]
            w = 0.95 ** hpos[ho + j]
            if nu > 0 and uh[nu - 1] == hs[j]:
                if w > uw[nu - 1]:
                    uw[nu - 1] = w
            else:
                uh[nu] = hs[j]; uw[nu] = w; nu += 1
        # 10 newest history events (hpos 0..9) and their exact allowed (item, type) counts
        nh = 0
        h10 = np.empty(10, np.int64); t10 = np.empty(10, np.int64); n10 = np.zeros(10)
        for j in range(h):
            if hpos[ho + j] < 10:
                pj = hpos[ho + j]
                h10[pj] = hitem[ho + j]; t10[pj] = htype[ho + j]
                if pj + 1 > nh:
                    nh = pj + 1
        for i in range(nh):
            k = np.searchsorted(uitems, h10[i])
            if k >= len(uitems) or uitems[k] != h10[i]:
                continue
            for e in range(ustart[k], ustart[k] + ucnt[k]):
                s = p_seq[e]
                if u != 0 and susers[s] == u:
                    continue
                if scons[s] and sdays[s] >= dd:
                    continue
                if types[starts[s] + p_pos[e]] == t10[i]:
                    n10[i] += 1.0
        co = coff[fi]; m = M[fi]; oo = ooff[q]
        cnt = np.zeros(10)
        for i in range(m):
            r = co + i; ro = oo + i
            for kk in range(5):
                out[ro, kk] = 0.0
            c = citem[r]
            k = np.searchsorted(uitems, c)
            found = k < len(uitems) and uitems[k] == c
            # exact allowed occurrence count n_c
            nc = 0.0
            if found:
                for e in range(ustart[k], ustart[k] + ucnt[k]):
                    s = p_seq[e]
                    if u != 0 and susers[s] == u:
                        continue
                    if scons[s] and sdays[s] >= dd:
                        continue
                    nc += 1.0
            Pc = nc / max(N, 1.0)
            lp = np.log((nc + 1.0) / (N + 1.0))
            for ii in range(10):
                cnt[ii] = 0.0
            nscan = 0.0
            if found and nc > 0:
                pc = ucnt[k]
                # latest CAP allowed occurrences (postings are in source order; scan from the end)
                nseq = 0.0
                e = ustart[k] + pc - 1
                while e >= ustart[k] and nscan < CAP:
                    s = p_seq[e]
                    if (u != 0 and susers[s] == u) or (scons[s] and sdays[s] >= dd):
                        e -= 1
                        continue
                    nscan += 1.0
                    p = p_pos[e]; st = starts[s]; L = lens[s]
                    is1 = types[st + p] == 1
                    hit = False
                    if nu > 0:
                        lo = max(0, p - 20); hi_ = min(L, p + 21)
                        for qq in range(lo, hi_):
                            if qq == p:
                                continue
                            x = items[st + qq]
                            if x == c:
                                continue
                            d = abs(qq - p)
                            if d <= 5:
                                tx = types[st + qq]
                                for ii in range(nh):
                                    if h10[ii] == x and t10[ii] == tx:
                                        cnt[ii] += 1.0
                            kk = np.searchsorted(uh[:nu], x)
                            if kk < nu and uh[kk] == x:
                                w = uw[kk]
                                if d <= 5:
                                    hit = True
                                    out[ro, 0] += w
                                    if qq > p:
                                        out[ro, 1] += w
                                    if is1:
                                        out[ro, 2] += w
                    if hit:
                        nseq += 1.0
                    e -= 1
                out[ro, 3] = nseq
            scale = nc / nscan if nscan > 0 else 1.0
            s_nb = lp
            for ii in range(nh):
                s_nb += ((10.0 - ii) / 10.0) * np.log((cnt[ii] * scale + 0.1) / (n10[ii] * Pc + 0.1))
            out[ro, 4] = s_nb


def run_queries(idx, split, sel, out, ooff):
    f = feeds(split)
    fday = day_of(f["timestamp"].to_numpy())
    Nday = np.array([idx.N_for_day(fday[i]) for i in sel], np.float64) if len(idx.cdays) else np.full(len(sel), float(idx.n_uncon))
    _covis(np.asarray(sel, np.int64), f["cand_off"].to_numpy().astype(np.int64), f["M"].to_numpy().astype(np.int64),
           f["hist_off"].to_numpy().astype(np.int64), f["H"].to_numpy().astype(np.int64), f["user_id"].to_numpy().astype(np.int64),
           fday.astype(np.int64), Nday, np.asarray(arr(split, "cand_item_id")), np.asarray(arr(split, "hist_item_id")),
           np.asarray(arr(split, "hist_hpos")), np.asarray(arr(split, "hist_event_type")).astype(np.int64),
           idx.items, idx.types, idx.starts, idx.lens, idx.users, idx.days, idx.cons, idx.uitems, idx.ustart, idx.ucnt,
           idx.p_seq, idx.p_pos, np.asarray(ooff, np.int64), out)


def save(tag, out):
    for i, c in enumerate(COLS):
        np.save(OUT / f"c04_{c}_{tag}.npy", out[:, i].astype(np.float32))
    log(f"{tag}: " + " ".join(f"{c} nz {np.mean(out[:, i] != 0):.3f} mean {out[:, i].mean():.3f}" for i, c in enumerate(COLS)))


which = sys.argv[1] if len(sys.argv) > 1 else "val,hold"
if "val" in which:
    fv = feeds("val")
    vdays = day_of(fv["timestamp"].to_numpy())
    log(f"val days {np.unique(vdays, return_counts=True)}")
    idx = Index([("train", source_fidx(None), False), ("val", np.arange(len(fv)), True)])
    Mv = fv["M"].to_numpy().astype(np.int64)
    out = np.zeros((int(Mv.sum()), len(COLS)), np.float64)
    run_queries(idx, "val", np.arange(len(fv)), out, fv["cand_off"].to_numpy())
    save("val", out)
    del idx
if "hold" in which:
    sys.path.insert(0, str(HERE))
    from h import holdout_ids
    hold = holdout_ids()[0]
    ft = feeds("train")
    Mt = ft["M"].to_numpy().astype(np.int64)
    hoff = np.concatenate([[0], np.cumsum(Mt[hold])[:-1]]).astype(np.int64)
    d = day_of(ft["timestamp"].to_numpy())[hold]
    days = train_days()
    di = np.searchsorted(days, d, side="right") - 1
    out = np.zeros((int(Mt[hold].sum()), len(COLS)), np.float64)
    for i, c in enumerate(days):
        q = np.nonzero(di == i)[0]
        if len(q) == 0:
            continue
        src = source_fidx(c)
        if len(src) == 0:
            log("day", c, "empty source -> zeros", len(q)); continue
        idx = Index([("train", src, False)])
        run_queries(idx, "train", hold[q], out, hoff[q])
        log("holdout day", c, "feeds", len(q))
        del idx
    save("hold", out)
log("DONE")
