"""C09 RP3beta whole-sequence walk (PREDECLARED_ccb.txt C09, B9 source). Build only; writes cache/c09_{val,hold}.npz.
Source sequences = feat_covis.py / sources.py convention (copied): de-duplicated user histories = latest train feed per non-zero
user + every user-0 feed, restricted by the B9 day rule:
  val rows (day D): all-train source + OTHER users' val feeds with day < D (test never); same non-zero user skipped.
  holdout rows (train day d): train source of days < d (sources.source_fidx(d)); same non-zero user skipped.
For candidate c: over c's latest 500 valid source occurrences (postings in index order = feat_covis 'latest'), each distinct
source sequence s containing c contributes sum_{x in s, x != c, x in the feed's history} 0.95^hpos(x) / (len(s) * deg(x))
(hpos of x's newest history occurrence; deg = posting count of x in the B9 source of that query day).
rp3b05 = sum / deg(c)^0.5, p3a = sum (beta 0). Cold c (no posting) -> 0 for both."""
import os
import sys
sys.dont_write_bytecode = True
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
from pathlib import Path
import numba as nb
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
FLAT = Path("<repo>/data/flat")
MSK = 3 * 3600
T0 = time.time()
NT = 12


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def day_of(ts):
    return (np.asarray(ts, dtype=np.int64) + MSK) // 86400


def source_fidx(ft, cutoff_day=None):  # copy of feats_v2/seq_/sources.py source_fidx
    f = ft.with_columns(day=((pl.col("timestamp").cast(pl.Int64) + MSK) // 86400))
    if cutoff_day is not None:
        f = f.filter(pl.col("day") < cutoff_day)
    nz = f.filter(pl.col("user_id") != 0).sort(["user_id", "timestamp", "fidx"]).group_by("user_id").agg(
        pl.col("fidx").last())["fidx"].to_numpy()
    z = f.filter(pl.col("user_id") == 0)["fidx"].to_numpy()
    return np.sort(np.concatenate([nz, z])).astype(np.int64)


@nb.njit(parallel=True, cache=True)
def gather(sel, hist_off, H, src, out_off, out):
    for q in nb.prange(len(sel)):
        f = sel[q]
        o = out_off[q]
        ho = hist_off[f]
        for j in range(H[f]):
            out[o + j] = src[ho + j]


@nb.njit(cache=True)
def counting_postings(ev, G, starts, lens):
    cnt = np.zeros(G, np.int64)
    for i in range(len(ev)):
        cnt[ev[i]] += 1
    pst = np.zeros(G, np.int64)
    acc = 0
    for g in range(G):
        pst[g] = acc
        acc += cnt[g]
    fill = pst.copy()
    p_seq = np.empty(len(ev), np.int32)
    for s in range(len(lens)):
        st = starts[s]
        for j in range(lens[s]):
            g = ev[st + j]
            p_seq[fill[g]] = s
            fill[g] += 1
    return cnt, pst, p_seq


@nb.njit(parallel=True, cache=True)
def rp3(sel, coff, M, hoff, H, fuser, fslot, cand_d, hist_d, hist_hpos, ev, sstart, slen, suser, sday, fday,
        pst, pcnt, p_seq, deg2, G, nthr, cap, out):
    for t in nb.prange(nthr):
        wd = np.zeros(G, np.float64)
        flag = np.zeros(G, np.uint8)
        touched = np.empty(512, np.int64)
        for q in range(t, len(sel), nthr):
            fi = sel[q]
            u = fuser[fi]
            ds = fslot[q]
            fd = fday[q]
            h = H[fi]
            ho = hoff[fi]
            nt = 0
            for j in range(h):
                g = hist_d[ho + j]
                if g < 0:
                    continue
                w = 0.95 ** hist_hpos[ho + j]
                if w > wd[g]:
                    wd[g] = w
                touched[nt] = g
                nt += 1
            for j in range(nt):
                g = touched[j]
                if flag[g] == 0:
                    dg = deg2[ds, g]
                    wd[g] = wd[g] / dg if dg > 0 else 0.0
                    flag[g] = 1
            co = coff[fi]
            for i in range(M[fi]):
                r = co + i
                out[r, 0] = 0.0
                out[r, 1] = 0.0
                out[r, 2] = 0.0
                c = cand_d[r]
                if c < 0 or nt == 0:
                    continue
                n_occ = 0
                last_s = -1
                tot = 0.0
                nseq = 0
                e = pst[c] + pcnt[c] - 1
                while e >= pst[c] and n_occ < cap:
                    s = p_seq[e]
                    e -= 1
                    if u != 0 and suser[s] == u:
                        continue
                    if sday[s] >= fd:
                        continue
                    n_occ += 1
                    if s == last_s:
                        continue
                    last_s = s
                    nseq += 1
                    st = sstart[s]
                    L = slen[s]
                    acc = 0.0
                    for qq in range(L):
                        x = ev[st + qq]
                        if x != c:
                            acc += wd[x]
                    tot += acc / L
                dc = deg2[ds, c]
                out[r, 0] = tot / np.sqrt(dc) if dc > 0 else 0.0
                out[r, 1] = tot
                out[r, 2] = nseq
            for j in range(nt):
                g = touched[j]
                wd[g] = 0.0
                flag[g] = 0


class Index:
    def __init__(self, parts, guniq_n, hist_dense, feeds_tab):
        """parts: list of (split, fidx array, day of each sequence (int64; train source -> -1))."""
        evs, lens, users, days = [], [], [], []
        for split, sel, sd in parts:
            f = feeds_tab[split]
            H = f["H"].to_numpy().astype(np.int64)
            ho = f["hist_off"].to_numpy().astype(np.int64)
            h = H[sel]
            off = np.concatenate([[0], np.cumsum(h)[:-1]]).astype(np.int64)
            ev = np.empty(h.sum(), np.int32)
            gather(sel, ho, H, hist_dense[split], off, ev)
            evs.append(ev)
            lens.append(h)
            users.append(f["user_id"].to_numpy()[sel].astype(np.int64))
            days.append(sd)
        self.ev = np.concatenate(evs)
        self.lens = np.concatenate(lens).astype(np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.lens)[:-1]]).astype(np.int64)
        self.users = np.concatenate(users)
        self.days = np.concatenate(days).astype(np.int64)
        assert self.ev.min() >= 0
        self.cnt, self.pst, self.p_seq = counting_postings(self.ev, guniq_n, self.starts, self.lens)
        log(f"index: {len(self.lens)} seqs, {len(self.ev)} events, {int((self.cnt > 0).sum())} items")


def main():
    nb.set_num_threads(NT)
    ft = pl.read_parquet(FLAT / "train/feeds.parquet")
    fv = pl.read_parquet(FLAT / "val/feeds.parquet")
    feeds_tab = {"train": ft, "val": fv}
    # global dense item ids over all train + val history items
    ht = np.load(FLAT / "train/hist_item_id.npy")
    hv = np.load(FLAT / "val/hist_item_id.npy")
    guniq = np.unique(np.concatenate([np.unique(ht), np.unique(hv)]))
    G = len(guniq)
    hist_dense = {"train": np.searchsorted(guniq, ht).astype(np.int32), "val": np.searchsorted(guniq, hv).astype(np.int32)}
    del ht, hv
    log(f"dense ids: {G} items")

    def cand_dense(items):
        k = np.searchsorted(guniq, items)
        k = np.minimum(k, G - 1)
        return np.where(guniq[k] == items, k, -1).astype(np.int64)

    # ---------------- val ----------------
    vday = day_of(fv["timestamp"].to_numpy())
    src_tr = source_fidx(ft, None)
    idx = Index([("train", src_tr, np.full(len(src_tr), -1, np.int64)), ("val", np.arange(len(fv)), vday)], G, hist_dense, feeds_tab)
    vdays = np.unique(vday)
    # deg per query day slot: train part + val feeds with day < D
    ntr_ev = int(idx.lens[:len(src_tr)].sum())
    ev_tr = idx.ev[:ntr_ev]
    deg_tr = np.bincount(ev_tr, minlength=G).astype(np.float64)
    deg2 = np.zeros((len(vdays), G), np.float64)
    seq_day_ev = np.repeat(idx.days, idx.lens)
    for j, D in enumerate(vdays):
        m = (seq_day_ev >= 0) & (seq_day_ev < D)
        deg2[j] = deg_tr + np.bincount(idx.ev[m], minlength=G)
    del seq_day_ev
    fslot = np.searchsorted(vdays, vday).astype(np.int64)
    cand_v = cand_dense(np.load(FLAT / "val/cand_item_id.npy"))
    out_v = np.zeros((len(cand_v), 3), np.float64)
    sel = np.arange(len(fv), dtype=np.int64)
    rp3(sel, fv["cand_off"].to_numpy().astype(np.int64), fv["M"].to_numpy().astype(np.int64),
        fv["hist_off"].to_numpy().astype(np.int64), fv["H"].to_numpy().astype(np.int64), fv["user_id"].to_numpy().astype(np.int64),
        fslot, cand_v, hist_dense["val"].astype(np.int64), np.load(FLAT / "val/hist_hpos.npy").astype(np.int64), idx.ev, idx.starts,
        idx.lens, idx.users, idx.days, vday[sel].astype(np.int64), idx.pst, idx.cnt, idx.p_seq, deg2, G, NT, 500, out_v)
    log(f"val done: share rp3>0 {np.mean(out_v[:, 1] > 0):.4f}, cand in source {np.mean(cand_v >= 0):.4f}")
    np.savez(HERE / "cache/c09_val.npz", rp3b05=out_v[:, 0], p3a=out_v[:, 1], nseq=out_v[:, 2])
    del idx, deg2, out_v

    # ---------------- holdout (per train day) ----------------
    Mt = ft["M"].to_numpy().astype(np.int64)
    p = np.nan_to_num(np.load("<repo>/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
    p[Mt < 50] = 0.0
    HOLD = np.sort(np.random.default_rng(12345).choice(np.where((p > 0.9) & (Mt >= 50))[0], 30000, replace=False))
    Mh = Mt[HOLD]
    hoff_local = np.concatenate([[0], np.cumsum(Mh)[:-1]]).astype(np.int64)
    tday = day_of(ft["timestamp"].to_numpy())
    hd = tday[HOLD]
    cand_t_items = np.load(FLAT / "train/cand_item_id.npy", mmap_mode="r")
    coff_t = ft["cand_off"].to_numpy().astype(np.int64)
    out_h = np.zeros((int(Mh.sum()), 3), np.float64)
    hpos_t = np.load(FLAT / "train/hist_hpos.npy", mmap_mode="r")
    hpos_t = np.asarray(hpos_t).astype(np.int16)
    hist_t64 = hist_dense["train"]
    for d in np.unique(hd):
        qi = np.nonzero(hd == d)[0]                     # positions in HOLD
        src = source_fidx(ft, int(d))
        if len(src) == 0:
            log(f"hold day {d}: empty source -> zeros ({len(qi)} feeds)")
            continue
        idx = Index([("train", src, np.full(len(src), -1, np.int64))], G, hist_dense, feeds_tab)
        deg2 = idx.cnt.astype(np.float64)[None, :]
        # local candidate arrays for these feeds: build a compact feed table (cand rows local to out_h)
        sel = HOLD[qi].astype(np.int64)
        rows = np.concatenate([np.arange(coff_t[f], coff_t[f] + Mt[f]) for f in sel])
        cd_local = cand_dense(np.asarray(cand_t_items[rows]))
        # map: feed q -> local offset into cd_local / out block
        Mq = Mt[sel]
        offq = np.concatenate([[0], np.cumsum(Mq)[:-1]]).astype(np.int64)
        outq = np.zeros((len(rows), 3), np.float64)
        qsel = np.arange(len(sel), dtype=np.int64)
        rp3(qsel, offq, Mq, ft["hist_off"].to_numpy().astype(np.int64)[sel], ft["H"].to_numpy().astype(np.int64)[sel],
            ft["user_id"].to_numpy().astype(np.int64)[sel], np.zeros(len(sel), np.int64), cd_local, hist_t64,
            hpos_t, idx.ev, idx.starts, idx.lens, idx.users, idx.days, np.full(len(sel), 1 << 40, np.int64),
            idx.pst, idx.cnt, idx.p_seq, deg2, G, NT, 500, outq)
        for j, qpos in enumerate(qi):
            out_h[hoff_local[qpos]:hoff_local[qpos] + Mh[qpos]] = outq[offq[j]:offq[j] + Mq[j]]
        log(f"hold day {d}: {len(qi)} feeds, share rp3>0 {np.mean(outq[:, 1] > 0):.4f}")
        del idx, deg2
    np.savez(HERE / "cache/c09_hold.npz", rp3b05=out_h[:, 0], p3a=out_h[:, 1], nseq=out_h[:, 2])
    log("DONE")


if __name__ == "__main__":
    main()
