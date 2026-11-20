# Builder of the item-to-item co-visitation counts (the seq_cv_* columns), the one collaborative signal of the
# sequence family. Three of its columns (seq_cv_invd20, seq_cv_pop, seq_cv_norm) are among the 34 inputs of the
# final ranker. Train rows of day d see only train feeds of earlier days; val and test rows see all of train plus
# earlier val/test histories; sequences of the same user are skipped.
# It was called through the driver lab/feats_v2/seq_/run_cv.py (from lab/feats_v2/seq_/build_keep.sh):
#   python run_cv.py val,test,train 16 cv
# It wrote data/feats/{train,val,test}/seq_cv_*.npy.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/feats_v2/seq_/feat_covis.py
"""Item-item co-visitation (OTTO style) between the candidate and the feed's history items.

Source sequences = de-duplicated user histories (sources.py):
  val/test rows: all-train source + every val/test feed history with timestamp < own feed timestamp (causal, so the
                 definition is identical for val and test and does not depend on how much future data exists);
  train rows of day d: train source strictly before day d (expanding window; no val/test).
Sequences of the same non-zero user are skipped. For candidate c we scan c's occurrences in the source (most recent
CAP occurrences) and, for each neighbour x within +-20 positions that is in the feed's history (recency weight
w = 0.95^hpos of x's newest occurrence there), accumulate:
  seq_cv_invd20  sum w/d                                 (d = position distance in the source sequence)
  seq_cv_w5      sum w           (d <= 5)
  seq_cv_fwd5    sum w           (d <= 5, c newer than x in the source sequence: "c followed x")
  seq_cv_t1      sum w           (d <= 5, c's source occurrence is event_type 1)
  seq_cv_seqs    # source sequences with a hit (d <= 5)
  seq_cv_pop     # scanned occurrences of c in the source (0 = cold item)
  seq_cv_norm    seq_cv_invd20 / sqrt(seq_cv_pop + 1)
Cold items (not in the source) get 0 (pop = 0 tells the model the 0 is "unknown").
Usage: python feat_covis.py <split>[,..] [threads]
"""
import sys
import numba as nb
import numpy as np
from seq_lib import *
from sources import train_days, source_fidx, day_of

CAP = 3000
WIN = 20
NAMES = ["seq_cv_invd20", "seq_cv_w5", "seq_cv_fwd5", "seq_cv_t1", "seq_cv_seqs", "seq_cv_pop", "seq_cv_norm"]


@nb.njit(parallel=True, cache=True)
def _gather(sel, hist_off, H, src_item, src_type, out_off, items, types):
    for q in nb.prange(len(sel)):
        f = sel[q]; o = out_off[q]; ho = hist_off[f]
        for j in range(H[f]):
            items[o + j] = src_item[ho + j]
            types[o + j] = src_type[ho + j]


class Index:
    def __init__(self, parts):
        """parts: list of (split, fidx array, ts_constrained: bool)."""
        self.parts = parts
        its, tys, lens, users, tss, cons = [], [], [], [], [], []
        for split, sel, constrained in parts:
            f = feeds(split)
            H = f["H"].to_numpy().astype(np.int64); ho = f["hist_off"].to_numpy()
            h = H[sel]
            off = np.concatenate([[0], np.cumsum(h)[:-1]]).astype(np.int64)
            it = np.empty(h.sum(), np.int32); ty = np.empty(h.sum(), np.int8)
            _gather(sel, ho, H, np.asarray(arr(split, "hist_item_id")), np.asarray(arr(split, "hist_event_type")),
                    off, it, ty)
            its.append(it); tys.append(ty); lens.append(h)
            users.append(f["user_id"].to_numpy()[sel]); tss.append(f["timestamp"].to_numpy()[sel].astype(np.int64))
            cons.append(np.full(len(sel), constrained, np.bool_))
        self.items = np.concatenate(its); self.types = np.concatenate(tys)
        self.lens = np.concatenate(lens).astype(np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.lens)[:-1]]).astype(np.int64)
        self.users = np.concatenate(users).astype(np.int64)
        self.ts = np.concatenate(tss); self.cons = np.concatenate(cons)
        seq = np.repeat(np.arange(len(self.lens), dtype=np.int32), self.lens)
        pos = (np.arange(len(self.items), dtype=np.int64) - np.repeat(self.starts, self.lens)).astype(np.int16)
        order = np.argsort(self.items, kind="stable")
        si = self.items[order]
        self.uitems, ustart, ucnt = np.unique(si, return_index=True, return_counts=True)
        self.ustart = ustart.astype(np.int64); self.ucnt = ucnt.astype(np.int64)
        self.p_seq = seq[order]; self.p_pos = pos[order]
        del order, si, seq, pos
        log(f"index: {len(self.lens)} seqs, {len(self.items)} events, {len(self.uitems)} items")

    def cluster_codes(self, K):
        """int16 cluster id (clu_ k-means, K) of every source event, -1 if no embedding."""
        out = []
        for split, sel, _ in self.parts:
            f = feeds(split)
            H = f["H"].to_numpy().astype(np.int64); ho = f["hist_off"].to_numpy()
            h = H[sel]
            off = np.concatenate([[0], np.cumsum(h)[:-1]]).astype(np.int64)
            e = np.empty(h.sum(), np.int32); dummy = np.empty(h.sum(), np.int8)
            _gather(sel, ho, H, np.asarray(arr(split, "hist_emb_idx")), np.asarray(arr(split, "hist_event_type")),
                    off, e, dummy)
            cl = np.load(CLU / f"item_cl_{K}_{split}.npy")
            out.append(np.where(e >= 0, cl[np.maximum(e, 0)], -1).astype(np.int16))
        return np.concatenate(out)


@nb.njit(parallel=True, cache=True)
def _covis(sel, coff, M, hoff, H, fuser, fts, citem, hitem, hpos, items, types, starts, lens, susers, sts, scons,
           uitems, ustart, ucnt, p_seq, p_pos, out):
    for q in nb.prange(len(sel)):
        fi = sel[q]
        h = H[fi]; ho = hoff[fi]; u = fuser[fi]; t = fts[fi]
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
        co = coff[fi]; m = M[fi]
        for i in range(m):
            r = co + i
            for k in range(7):
                out[r, k] = 0.0
            c = citem[r]
            k = np.searchsorted(uitems, c)
            if k >= len(uitems) or uitems[k] != c or nu == 0:
                continue
            pc = ucnt[k]
            ps = ustart[k] + max(0, pc - 3000)
            pe = ustart[k] + pc
            npop = 0.0; nseq = 0.0
            for e in range(ps, pe):
                s = p_seq[e]
                if u != 0 and susers[s] == u:
                    continue
                if scons[s] and sts[s] >= t:
                    continue
                npop += 1.0
                p = p_pos[e]; st = starts[s]; L = lens[s]
                lo = max(0, p - 20); hi_ = min(L, p + 21)
                hit = False
                is1 = types[st + p] == 1
                for qq in range(lo, hi_):
                    if qq == p:
                        continue
                    x = items[st + qq]
                    if x == c:
                        continue
                    kk = np.searchsorted(uh[:nu], x)
                    if kk < nu and uh[kk] == x:
                        d = abs(qq - p)
                        w = uw[kk]
                        out[r, 0] += w / d
                        if d <= 5:
                            hit = True
                            out[r, 1] += w
                            if qq > p:
                                out[r, 2] += w
                            if is1:
                                out[r, 3] += w
                if hit:
                    nseq += 1.0
            out[r, 4] = nseq
            out[r, 5] = npop
            out[r, 6] = out[r, 0] / np.sqrt(npop + 1.0)


def run_queries(idx, split, sel, out):
    f = feeds(split)
    _covis(np.asarray(sel, np.int64), f["cand_off"].to_numpy(), f["M"].to_numpy().astype(np.int64),
           f["hist_off"].to_numpy(), f["H"].to_numpy().astype(np.int64), f["user_id"].to_numpy().astype(np.int64),
           f["timestamp"].to_numpy().astype(np.int64), np.asarray(arr(split, "cand_item_id")),
           np.asarray(arr(split, "hist_item_id")), np.asarray(arr(split, "hist_hpos")),
           idx.items, idx.types, idx.starts, idx.lens, idx.users, idx.ts, idx.cons,
           idx.uitems, idx.ustart, idx.ucnt, idx.p_seq, idx.p_pos, out)


def run_train():
    N = len(arr("train", "cand_fidx"))
    out = np.zeros((N, len(NAMES)), np.float32)
    f = feeds("train")
    d = day_of(f["timestamp"].to_numpy())
    days = train_days()
    di = np.searchsorted(days, d, side="right") - 1
    for i, c in enumerate(days):
        sel = np.nonzero(di == i)[0]
        src = source_fidx(c)
        if len(src) == 0 or len(sel) == 0:
            log("day", c, "empty source -> zeros", len(sel))
            continue
        idx = Index([("train", src, False)])
        run_queries(idx, "train", sel, out)
        log("day", c, "queries", len(sel))
        del idx
    for i, nm in enumerate(NAMES):
        save("train", nm, out[:, i])


def run_vt(splits, transductive=True):
    parts = [("train", source_fidx(None), False)]
    if transductive:
        for s in ("val", "test"):
            parts.append((s, np.arange(len(feeds(s))), True))
    idx = Index(parts)
    for split in splits:
        N = len(arr(split, "cand_fidx"))
        out = np.zeros((N, len(NAMES)), np.float32)
        run_queries(idx, split, np.arange(len(feeds(split))), out)
        log(split, "done")
        for i, nm in enumerate(NAMES):
            save(split, nm, out[:, i])


if __name__ == "__main__":
    nb.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 24)
    sp = sys.argv[1].split(",")
    vt = [s for s in sp if s != "train"]
    if vt:
        run_vt(vt)
    if "train" in sp:
        run_train()
