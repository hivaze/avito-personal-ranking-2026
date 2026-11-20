# A zero-GPU trigger test of wave 11. Before training a ranker with a next-item auxiliary task we
# asked whether the pool's per-feed error is related to how well a sequence encoder predicts the
# user's hidden recent events: the partial correlation of the two, controlling for feed size,
# history length and cold share, on late holdout feeds of users absent from the encoder's corpus.
# How it was run: cd <project root> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python
# -I nn_x/w11/ssl/trigger/e2_trigger.py score w2r_v3, then ... readout (from the docstring).
# What it produced: out/readout.txt (copy: results/wave11_ssl_trigger_readout.txt): partial
# correlation -0.0012 +- 0.0139 on 5,191 feeds against a bar of +0.05, so the arm was not run.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w11/ssl/trigger/e2_trigger.py
"""SSL-WIDTH Step -1: the E2 reopen trigger (PRETRAIN_PLAN_1113 section 5; SOLVER_POWER_1113 section 6.3 item 4), CPU only.

Partial correlation, over post-window holdout feeds, between the per-feed NDCG of the G3 train-only pool and the hidden-event
recall@10 of a windowed pre2 encoder, controlling for M, H and cold share. Rule and definitions: nn_x/abl/ledger/ledger.md
section 11 (pre-declared with `date -u` BEFORE this script's first `score` run).

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python -I nn_x/w11/ssl/trigger/e2_trigger.py score w2r_v3
      -> out/recall_w2r_v3.npz  (per late-holdout feed: status, hidden / prefix / target counts, recall@10 / @50 for K = 16 and 8)
  ... e2_trigger.py score w3r_v3                       (information column: in-sample for 88% of the late feeds)
  ... e2_trigger.py readout                            -> out/readout.txt + out/readout.json (the pre-declared verdict)

Construction (per encoder w{k}r_v3, corpus = data/pretrain/win{k} rebuilt from data/flat/train exactly as build_corpus_window.py:
background = latest train snapshot per non-zero user with feed time < T_k plus every user-0 feed < T_k; in-window = earliest
snapshot per non-zero user in [T_k, E_k) plus every user-0 feed in the window; verified against win{k}/info.json and the
checkpoint's ID-table size):
  late slice   = the 8,279 holdout feeds with timestamp >= 2026-09-17 00:00 MSK (the 'late' holdout column of the 16:10 rule amendment)
  status       = nocorpus (non-zero user with NO sequence in the corpus) | user0 | hascorpus (user has a corpus snapshot) | insample_self
  hidden       = the newest min(K, H) history events of the feed (K = 16 primary, 8 information); prefix = the older H - K events
  targets      = distinct items of hidden events with event_type 1 and event_source 1 whose item does not occur in the prefix
                 (the encoder's own fe-target rule: next NEW feed-originated open)
  universe     = items with >= 1 fe target occurrence in the corpus (item_cnt_fe > 0, build_corpus.targets rule) UNION all targets
  ranking      = fe-head cosine logit of the state after the prefix (BOS state when the prefix is empty) against every universe item,
                 prefix items removed; recall@k = |targets in top-k| / |targets|
"""
import datetime as dt
import json
import os
import sys
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only: run with CUDA_VISIBLE_DEVICES=''"
NT = int(os.environ.get("OMP_NUM_THREADS", "12") or 12)
import numba as nb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

nb.set_num_threads(NT)
torch.set_num_threads(NT)
R = "<repo>"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/out"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f"{R}/pretrain")
sys.path.insert(0, f"{R}/src")
from seqenc import SeqEncoder  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402


# pretrain/build_corpus.py gather_rev / targets, copied verbatim (that module sets 32 numba threads at import; the cap here is 12)
@nb.njit(parallel=True, cache=False)
def gather_rev(hist_off, H, src_list, out_off, dst):
    # copy hist rows of each selected feed reversed (newest-first -> old->new)
    for s in nb.prange(len(H)):
        o, h, d = hist_off[s], H[s], out_off[s]
        for k in range(h):
            dst[d + k] = src_list[o + h - 1 - k]


@nb.njit(parallel=True, cache=False)
def targets(off, item, et, es, tgt_fe, tgt_new):
    for s in nb.prange(len(off) - 1):
        a, b = off[s], off[s + 1]
        L = b - a
        fo = np.empty(L, np.int32)
        for j in range(L):
            fo[j] = j
            x = item[a + j]
            for i in range(j):
                if item[a + i] == x:
                    fo[j] = i
                    break
        for t in range(L):
            tgt_fe[a + t] = -1
            tgt_new[a + t] = -1
            for j in range(t + 1, L):
                if fo[j] == j:
                    tgt_new[a + t] = item[a + j]
                    break
            for j in range(t + 1, L):
                if et[a + j] == 1 and es[a + j] == 1 and fo[j] > t:
                    tgt_fe[a + t] = item[a + j]
                    break

MSK = dt.timezone(dt.timedelta(hours=3))
T = lambda d: int(dt.datetime.fromisoformat(d + "T00:00:00").replace(tzinfo=MSK).timestamp())  # noqa: E731
WIN = {1: (T("2026-09-09"), T("2026-09-12")), 2: (T("2026-09-12"), T("2026-09-17")), 3: (T("2026-09-17"), 2**31 - 1)}
LATE_T = T("2026-09-17")
K_PRIMARY, K_INFO = 16, 8
TOPK = 50
G3 = [f"v23_gru_pre2s_s{s}" for s in range(501, 507)] + [f"r6_cd_m02_r{s}" for s in range(101, 108)] + \
     [f"r6_cd_m02_p70k_r{s}" for s in range(101, 108)]
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)


def feeds():
    f = pl.read_parquet(f"{R}/data/flat/train/feeds.parquet")
    g = lambda c: f[c].to_numpy().astype(np.int64)  # noqa: E731
    return dict(ts=g("timestamp"), u=g("user_id"), H=g("H"), hoff=g("hist_off"), M=g("M"), coff=g("cand_off"))


def holdout_ids(F):
    """The fixed 30K random-like train holdout (nn/holdout_eval.py selection), verified against preds/holdout/holdout_meta.npy."""
    pr = np.nan_to_num(np.load(f"{R}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
    pr[F["M"] < 50] = 0.0
    cand = np.where((pr > 0.9) & (F["M"] >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    meta = np.load(f"{R}/preds/holdout/holdout_meta.npy")
    assert np.array_equal(meta[0], F["M"][hold]) and np.array_equal(meta[1], np.concatenate([[0], np.cumsum(F["M"][hold])[:-1]]))
    return hold


def corpus_feeds(F, k):
    a, e = WIN[k]
    u, ts = F["u"], F["ts"]
    nz = u != 0
    idx = np.where((ts < a) & nz)[0]
    o = idx[np.lexsort((idx, ts[idx]))]
    last = {}
    for i in o:
        last[u[i]] = i
    bg = np.sort(np.concatenate([np.fromiter(last.values(), np.int64, len(last)), np.where((ts < a) & ~nz)[0]]))
    wm = (ts >= a) & (ts < e)
    idx = np.where(wm & nz)[0]
    o = idx[np.lexsort((idx, ts[idx]))]
    first = {}
    for i in o:
        if u[i] not in first:
            first[u[i]] = i
    win = np.sort(np.concatenate([np.fromiter(first.values(), np.int64, len(first)), np.where(wm & ~nz)[0]]))
    users = set(last) | set(first)
    return bg, win, users


def rebuild_corpus(F, k, G):
    """Token arrays (old -> new) of the w{k}r corpus; item counts, fe-target counts, per-item category (Items.cat rule)."""
    bg, win, users = corpus_feeds(F, k)
    info = json.load(open(f"{R}/data/pretrain/win{k}/info.json"))
    ids = np.concatenate([bg, win])
    H = F["H"][ids]
    keep = H > 0
    ids, H = ids[keep], H[keep]
    assert int((F["H"][bg] > 0).sum()) == info["n_bg_seqs"] and int((F["H"][win] > 0).sum()) == info["n_win_seqs"] and int(H.sum()) == info["tokens"]
    hoff = F["hoff"][ids]
    ooff = np.concatenate([[0], np.cumsum(H)[:-1]]).astype(np.int64)
    tok = {}
    for c in ["item_id", "category_id", "event_type", "event_source"]:
        src = np.load(f"{R}/data/flat/train/hist_{c}.npy")
        dst = np.empty(int(H.sum()), src.dtype)
        gather_rev(hoff, H, src, ooff, dst)
        tok[c] = dst
        del src
    off = np.concatenate([[0], np.cumsum(H)]).astype(np.int64)
    tf = np.empty(len(tok["item_id"]), np.int32)
    tn = np.empty(len(tok["item_id"]), np.int32)
    targets(off, tok["item_id"], tok["event_type"], tok["event_source"], tf, tn)
    cnt = np.bincount(tok["item_id"], minlength=G).astype(np.int32)
    cnt_fe = np.bincount(tf[tf >= 0], minlength=G).astype(np.int32)
    assert int((cnt >= 3).sum()) == info["id_items_ge3"], (int((cnt >= 3).sum()), info["id_items_ge3"])
    cat = np.zeros(G, np.int8)
    cat[tok["item_id"]] = tok["category_id"]
    for s in ["train", "val", "test"]:
        cat[np.load(f"{R}/data/flat/{s}/cand_item_id.npy")] = np.load(f"{R}/data/flat/{s}/cand_category_id.npy")
    log(f"w{k}r corpus rebuilt: {len(ids)} seqs, {off[-1]} tokens, id items {(cnt >= 3).sum()}, fe-target items {(cnt_fe > 0).sum()}, "
        f"users {len(users)}")
    return dict(cnt=cnt, cnt_fe=cnt_fe, cat=cat, users=users, bg=bg, win=win)


def eval_set(F, hold, users, bg, win):
    """Status of every late holdout feed w.r.t. the corpus."""
    late = hold[F["ts"][hold] >= LATE_T]
    cset = set(bg.tolist()) | set(win.tolist())
    status = np.empty(len(late), dtype=object)
    for j, fi in enumerate(late):
        uu = F["u"][fi]
        if uu == 0:
            status[j] = "user0" if fi not in cset else "insample_self"
        elif fi in cset:
            status[j] = "insample_self"
        elif uu in users:
            status[j] = "hascorpus"
        else:
            status[j] = "nocorpus"
    return late, status.astype(str)


class CpuEncoder:
    """The pretrained SeqEncoder on CPU with the ID table left memory-mapped (no 9-18 GB copy)."""

    def __init__(self, tag, cnt):
        ck = torch.load(f"{R}/data/pretrain/ckpt_{tag}.pt", map_location="cpu", mmap=True, weights_only=False)
        a = ck["args"]
        self.a = a
        sd = ck["model"]
        heads = ["fe", "new"] if a["w_new"] > 0 else ["fe"]
        self.m = SeqEncoder(0, d=a["d"], nl=a["nl"], nh=a["nh"], d_id=a["d_id"], drop=a["drop"], heads=heads, id_bias=False)
        dense = {k: v for k, v in sd.items() if not k.startswith(("id_emb", "id_proj", "id_bias"))}
        self.m.load_state_dict(dense, strict=True)
        self.m.eval()
        self.id_w = sd["id_emb.weight"]  # mmap [n_id+1, d_id]
        self.id_proj = sd["id_proj.weight"].clone()  # [d, d_id]
        self.id_bias = sd["id_bias.weight"].clone() if "id_bias.weight" in sd else None
        ids = np.where(cnt >= a["min_cnt"])[0]
        assert len(ids) + 1 == self.id_w.shape[0], (len(ids) + 1, self.id_w.shape)
        self.slot = np.zeros(len(cnt), np.int32)
        self.slot[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
        self.tab = np.load(f"{R}/data/pretrain/item_tab2.npy", mmap_mode="r")
        self.has = np.load(f"{R}/data/pretrain/item_has2.npy")
        assert a["item_tab"] == "item_tab2"
        log(f"encoder {tag}: {a['d']}d x {a['nl']} blocks, d_id {a['d_id']}, {len(ids)} ID rows, id_bias {self.id_bias is not None}")

    @torch.no_grad()
    def item_rep(self, items, cat, valid=None):
        """items int64 array (any shape); returns rep [..., d] fp32 (content + category + ID, as the deployed 'fe' scores)."""
        it = np.asarray(items, np.int64)
        shp = it.shape
        flat = it.reshape(-1)
        emb = torch.from_numpy(np.ascontiguousarray(self.tab[flat]))
        has = torch.from_numpy(self.has[flat].copy())
        if valid is not None:
            has = has & torch.from_numpy(np.asarray(valid).reshape(-1))
        r = self.m.item_rep(emb, has, torch.from_numpy(cat[flat].astype(np.int64)), torch.zeros(len(flat), dtype=torch.long))
        sl = self.slot[flat]
        nz = np.where(sl > 0)[0]
        if len(nz):
            idv = self.id_w[torch.from_numpy(sl[nz].astype(np.int64))]
            r[torch.from_numpy(nz)] += idv @ self.id_proj.T
        return r.reshape(*shp, -1)

    @torch.no_grad()
    def states(self, seqs, cat, batch_tokens=200_000):
        """seqs = list of (items, et, es) old -> new; returns the fe query [n, d] of the state after the last event (BOS if empty)."""
        n = len(seqs)
        lens = np.array([len(s[0]) for s in seqs])
        Q = torch.zeros((n, self.m.d))
        order = np.argsort(lens, kind="stable")
        i = 0
        while i < n:
            L = max(1, int(lens[order[min(n - 1, i)]]))
            B = max(1, batch_tokens // (L + 1))
            ids = order[i:i + B]
            Lx = max(1, int(lens[ids].max()))
            it = np.zeros((len(ids), Lx), np.int64); et = np.zeros((len(ids), Lx), np.int64); es = np.zeros((len(ids), Lx), np.int64)
            mask = np.zeros((len(ids), Lx), bool)
            for r_, j in enumerate(ids):
                a_, b_, c_ = seqs[j]
                it[r_, :len(a_)] = a_; et[r_, :len(a_)] = b_; es[r_, :len(a_)] = c_; mask[r_, :len(a_)] = True
            rep = self.item_rep(it, cat, valid=mask)
            h = self.m.encode(rep, torch.from_numpy(et), torch.from_numpy(es))
            last = torch.from_numpy(lens[ids].astype(np.int64))
            hl = h[torch.arange(len(ids)), last]
            Q[torch.from_numpy(ids)] = self.m.query(hl, "fe")
            i += B
        return Q

    @torch.no_grad()
    def out_vecs(self, items, cat, chunk=1_000_000):
        V = torch.empty((len(items), self.m.d), dtype=torch.float32)
        for a in range(0, len(items), chunk):
            V[a:a + chunk] = self.m.out_vec(self.item_rep(items[a:a + chunk], cat))
            if (a // chunk) % 5 == 0:
                log(f"   out vecs {a + min(chunk, len(items) - a)} / {len(items)}")
        return V


def hidden_targets(F, late, K):
    hit = np.load(f"{R}/data/flat/train/hist_item_id.npy", mmap_mode="r")
    het = np.load(f"{R}/data/flat/train/hist_event_type.npy", mmap_mode="r")
    hes = np.load(f"{R}/data/flat/train/hist_event_source.npy", mmap_mode="r")
    seqs, tg, pref_items = [], [], []
    n_hid = np.zeros(len(late), np.int64)
    for fi in late:
        o, h = F["hoff"][fi], F["H"][fi]
        nh = min(K, h)
        it = np.asarray(hit[o:o + h]); et = np.asarray(het[o:o + h]); es = np.asarray(hes[o:o + h])
        hid_it, hid_et, hid_es = it[:nh], et[:nh], es[:nh]
        pre_it, pre_et, pre_es = it[nh:][::-1].copy(), et[nh:][::-1].copy(), es[nh:][::-1].copy()  # old -> new
        t_ = np.unique(hid_it[(hid_et == 1) & (hid_es == 1)])
        t_ = t_[~np.isin(t_, pre_it)]
        seqs.append((pre_it.astype(np.int64), pre_et.astype(np.int64), pre_es.astype(np.int64)))
        tg.append(t_.astype(np.int64)); pref_items.append(np.unique(pre_it).astype(np.int64))
        n_hid[len(seqs) - 1] = nh
    return seqs, tg, pref_items, n_hid


@torch.no_grad()
def recall(enc, Q, universe, cat, tg, pref_items, feed_batch=1024, item_chunk=400_000):
    """Top-TOPK universe items per feed (prefix items removed) -> recall@10 / @50 per feed (NaN when no target)."""
    n = Q.shape[0]
    V = enc.out_vecs(universe, cat)
    log(f"universe out vectors {tuple(V.shape)}")
    upos = {int(x): i for i, x in enumerate(universe.tolist())} if len(universe) < 5_000_000 else None
    uni_sorted = universe  # sorted ascending
    top_v = torch.full((n, TOPK), -1e9); top_i = torch.full((n, TOPK), -1, dtype=torch.long)
    for a in range(0, n, feed_batch):
        b = min(n, a + feed_batch)
        q = Q[a:b]
        # prefix items of this batch -> universe positions (for masking)
        rows, cols = [], []
        for r_, j in enumerate(range(a, b)):
            p = pref_items[j]
            pos = np.searchsorted(uni_sorted, p)
            ok = (pos < len(uni_sorted)) & (uni_sorted[np.minimum(pos, len(uni_sorted) - 1)] == p)
            pos = pos[ok]
            rows.append(np.full(len(pos), r_)); cols.append(pos)
        rows = np.concatenate(rows); cols = np.concatenate(cols)
        order = np.argsort(cols, kind="stable"); rows, cols = rows[order], cols[order]
        tv, ti = top_v[a:b], top_i[a:b]
        for c0 in range(0, len(uni_sorted), item_chunk):
            c1 = min(len(uni_sorted), c0 + item_chunk)
            s = q @ V[c0:c1].T  # [B, C]
            lo, hi = np.searchsorted(cols, c0), np.searchsorted(cols, c1)
            if hi > lo:
                s[torch.from_numpy(rows[lo:hi]), torch.from_numpy(cols[lo:hi] - c0)] = -1e9
            cv, ci = torch.topk(s, min(TOPK, c1 - c0), dim=1)
            mv = torch.cat([tv, cv], 1); mi = torch.cat([ti, ci + c0], 1)
            sel_v, sel = torch.topk(mv, TOPK, dim=1)
            tv, ti = sel_v, mi.gather(1, sel)
        top_v[a:b], top_i[a:b] = tv, ti
        log(f"   scored feeds {b} / {n}")
    del V
    r10 = np.full(n, np.nan); r50 = np.full(n, np.nan); hit10 = np.zeros(n, np.int64)
    top_items = uni_sorted[top_i.numpy().clip(min=0)]
    for j in range(n):
        if len(tg[j]) == 0:
            continue
        r10[j] = np.isin(tg[j], top_items[j, :10]).mean(); r50[j] = np.isin(tg[j], top_items[j, :TOPK]).mean()
        hit10[j] = int(np.isin(tg[j], top_items[j, :10]).sum())
    return r10, r50, hit10


def score(tag):
    k = int(tag[1])
    F = feeds()
    hold = holdout_ids(F)
    G = len(np.load(f"{R}/data/pretrain/item_has2.npy", mmap_mode="r"))
    C = rebuild_corpus(F, k, G)
    late, status = eval_set(F, hold, C["users"], C["bg"], C["win"])
    vals, cnts = np.unique(status, return_counts=True)
    log(f"late holdout feeds {len(late)}: " + ", ".join(f"{v} {c}" for v, c in zip(vals, cnts)))
    enc = CpuEncoder(tag, C["cnt"])
    res = dict(late=late, status=status, H=F["H"][late], M=F["M"][late], user=F["u"][late], ts=F["ts"][late])
    uni_base = np.where(C["cnt_fe"] > 0)[0].astype(np.int64)
    for K in (K_PRIMARY, K_INFO):
        seqs, tg, pref_items, n_hid = hidden_targets(F, late, K)
        n_tgt = np.array([len(t) for t in tg])
        log(f"K {K}: feeds with >= 1 target {(n_tgt > 0).sum()}, targets/feed {n_tgt[n_tgt > 0].mean():.2f}, prefix 0 {(np.array([len(s[0]) for s in seqs]) == 0).sum()}")
        universe = np.unique(np.concatenate([uni_base] + tg))
        log(f"K {K}: universe {len(universe)} items ({len(uni_base)} corpus fe-target items + {len(universe) - len(uni_base)} target-only)")
        Q = enc.states(seqs, C["cat"])
        log(f"K {K}: states encoded")
        r10, r50, hit10 = recall(enc, Q, universe, C["cat"], tg, pref_items)
        ok = n_tgt > 0
        log(f"K {K}: recall@10 mean {np.nanmean(r10):.4f} (nonzero share {(r10[ok] > 0).mean():.3f}), recall@50 {np.nanmean(r50):.4f}")
        res.update({f"r10_K{K}": r10, f"r50_K{K}": r50, f"hit10_K{K}": hit10, f"ntgt_K{K}": n_tgt, f"nhid_K{K}": n_hid,
                    f"npref_K{K}": np.array([len(s[0]) for s in seqs]), f"universe_K{K}": len(universe)})
    np.savez(f"{OUT}/recall_{tag}.npz", **res)
    log(f"saved {OUT}/recall_{tag}.npz")


# ----------------------------------------------------------------------------------------------------- readout
def feed_z(s, M):
    fid = np.repeat(np.arange(len(M)), M)
    mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-6
    return (s - mu[fid]) / sd[fid]


def g3_per_feed():
    meta = np.load(f"{R}/preds/holdout/holdout_meta.npy"); M, offs = meta[0], meta[1]
    y = np.load(f"{R}/preds/holdout/holdout_y.npy").astype(np.float64)
    Z = np.zeros(len(y))
    for m in G3:
        Z += feed_z(np.load(f"{R}/preds/holdout/nn_{m}_holdout.npy").astype(np.float64), M)
    return ndcg_per_feed(Z / len(G3), y, offs, M)


def partial_corr(x, y, Z):
    """Pearson corr of the OLS residuals of x and y on Z (Z includes the intercept)."""
    bx, *_ = np.linalg.lstsq(Z, x, rcond=None); by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    ex, ey = x - Z @ bx, y - Z @ by
    return float(np.corrcoef(ex, ey)[0, 1]), ex, ey


def boot_se(x, y, Z, clusters, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    uc, inv = np.unique(clusters, return_inverse=True)
    groups = [np.where(inv == g)[0] for g in range(len(uc))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        vals.append(partial_corr(x[idx], y[idx], Z[idx])[0])
    return float(np.std(vals, ddof=1)), vals


def controls(M, H, cold, extra=None):
    cols = [np.ones(len(M)), M, np.log1p(M), H, np.log1p(H), cold]
    if extra is not None:
        cols += list(extra)
    return np.column_stack(cols).astype(np.float64)


def readout():
    F = feeds()
    hold = holdout_ids(F)
    nd = g3_per_feed()
    late_mask = F["ts"][hold] >= LATE_T
    nd_late = nd[late_mask]
    late = hold[late_mask]
    # cold share = share of the feed's candidates with it_hist_all_luo == 0 (LUO-cold, the ranker's own exposure input)
    luo = np.load(f"{R}/data/feats/train/it_hist_all_luo.npy", mmap_mode="r")
    cold = np.array([float((np.asarray(luo[F["coff"][fi]:F["coff"][fi] + F["M"][fi]]) == 0).mean()) for fi in late])
    lines, J = [], {}
    rows = []

    def block(name, tag, K, mask_fn, extra_fn=None, rk="r10", primary=False):
        d = np.load(f"{OUT}/recall_{tag}.npz", allow_pickle=True)
        assert np.array_equal(d["late"], late)
        r = d[f"{rk}_K{K}"]; ntgt = d[f"ntgt_K{K}"]; npref = d[f"npref_K{K}"]
        m = mask_fn(d["status"]) & (ntgt > 0) & np.isfinite(r)
        if m.sum() < 30:  # e.g. w3r has no 'user0' status (its corpus holds every user-0 W3 feed): nothing to read
            lines.append(f"{name}: n {int(m.sum())} < 30 feeds, not read")
            return None
        x, y = nd_late[m], r[m]
        Z = controls(d["M"][m].astype(np.float64), d["H"][m].astype(np.float64), cold[m],
                     extra=None if extra_fn is None else extra_fn(ntgt[m], npref[m]))
        rp, ex, ey = partial_corr(x, y, Z)
        cl = np.where(d["user"][m] == 0, -(1 + d["late"][m]), d["user"][m])  # user 0 is a pooled bucket: one cluster per feed
        se, vals = boot_se(x, y, Z, cl, n_boot=2000 if primary else 500)
        se_fisher = 1.0 / np.sqrt(m.sum() - (Z.shape[1] - 1) - 3)
        raw = float(np.corrcoef(x, y)[0, 1])
        from scipy.stats import spearmanr
        rho = float(spearmanr(ex, ey).correlation)
        # readability: mean G3 NDCG residual by recall class
        res_by = {"recall=0": float(ex[y == 0].mean()) if (y == 0).any() else float("nan"),
                  "recall>0": float(ex[y > 0].mean()) if (y > 0).any() else float("nan")}
        row = dict(name=name, tag=tag, K=K, metric=rk, n=int(m.sum()), n_users=int(len(np.unique(d["user"][m]))), r_partial=rp,
                   se_boot=se, se_fisher=float(se_fisher), lower2se=rp - 2 * se, z=rp / se if se > 0 else float("nan"), r_raw=raw,
                   spearman_partial=rho, recall_mean=float(y.mean()), recall_nonzero_share=float((y > 0).mean()),
                   g3_ndcg_mean=float(x.mean()), resid_by_recall=res_by, targets_per_feed=float(ntgt[m].mean()),
                   universe=int(d[f"universe_K{K}"]), primary=primary)
        if primary:
            row["verdict_strict"] = "PASS" if (rp - 2 * se > 0.05) else "FAIL"
            row["verdict_lenient_info"] = "PASS" if (rp > 0.05 and rp > 2 * se) else "FAIL"
            row["boot_quantiles"] = [float(np.quantile(vals, q)) for q in (0.025, 0.5, 0.975)]
        rows.append(row)
        lines.append(f"{'PRIMARY ' if primary else ''}{name}: n {row['n']} feeds / {row['n_users']} users | partial corr {rp:+.4f} "
                     f"+- {se:.4f} (boot; Fisher {se_fisher:.4f}) | lower 2-SE bound {rp - 2 * se:+.4f} | z {row['z']:+.2f} | raw corr {raw:+.4f} | "
                     f"Spearman(partial) {rho:+.4f} | recall@{rk[1:]} mean {y.mean():.4f} (nonzero {100 * (y > 0).mean():.1f}%) | targets/feed "
                     f"{ntgt[m].mean():.2f} | G3 NDCG {x.mean():.4f}; residual by recall=0 / >0: {res_by['recall=0']:+.4f} / {res_by['recall>0']:+.4f}")
        return row

    nocorp = lambda st: st == "nocorpus"  # noqa: E731
    allok = lambda st: np.ones(len(st), bool)  # noqa: E731
    ext = lambda nt, npf: (np.log1p(nt.astype(np.float64)), np.log1p(npf.astype(np.float64)))  # noqa: E731
    prim = block("w2r_v3 | late slice, users absent from the w2r corpus | K 16 | recall@10 | controls M, log M, H, log H, cold share",
                 "w2r_v3", 16, nocorp, primary=True)
    lines.append(f"\nVERDICT (pre-declared rule, ledger section 11: PASS iff r_partial - 2 SE_boot > 0.05): {prim['verdict_strict']}"
                 f"   [lenient reading r > 0.05 and r > 2 SE, information only: {prim['verdict_lenient_info']}]\n")
    lines.append("Information columns (never gates):")
    block("w2r_v3 | same set | K 16 | recall@50", "w2r_v3", 16, nocorp, rk="r50")
    block("w2r_v3 | same set | K 8 | recall@10", "w2r_v3", 8, nocorp)
    block("w2r_v3 | same set | K 16 | recall@10 | + controls log(1+n_targets), log(1+prefix length)", "w2r_v3", 16, nocorp, extra_fn=ext)
    block("w2r_v3 | ALL late feeds with a target (incl. user0 and users with a w2r corpus snapshot) | K 16 | recall@10", "w2r_v3", 16, allok)
    block("w2r_v3 | late user0 feeds only | K 16 | recall@10", "w2r_v3", 16, lambda st: st == "user0")
    block("w2r_v3 | late feeds of users WITH a w2r corpus snapshot (hidden events possibly in-sample) | K 16 | recall@10", "w2r_v3", 16,
          lambda st: st == "hascorpus")
    if os.path.exists(f"{OUT}/recall_w3r_v3.npz"):
        block("w3r_v3 | ALL late feeds with a target (IN-SAMPLE for the 'insample_self' 88%: the literal w3r reading) | K 16 | recall@10",
              "w3r_v3", 16, allok)
        block("w3r_v3 | late feeds whose own snapshot is a w3r training sequence (insample_self) | K 16 | recall@10", "w3r_v3", 16,
              lambda st: st == "insample_self")
        block("w3r_v3 | late user0 feeds (not in the w3r corpus) | K 16 | recall@10", "w3r_v3", 16, lambda st: st == "user0")
        block("w3r_v3 | late feeds with a w3r corpus snapshot of the user but not in-sample themselves (hascorpus) | K 16 | recall@10",
              "w3r_v3", 16, lambda st: st == "hascorpus")
    txt = "\n".join(lines)
    print(txt)
    hdr = (f"E2 TRIGGER READOUT {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}; G3 train-only pool = member-equal z-mean of "
           f"{len(G3)} members on the 30K holdout, per-feed NDCG; late slice = {len(late)} feeds (ts >= 2026-09-17 MSK); cold share = "
           f"share of candidates with it_hist_all_luo == 0 (mean {cold.mean():.3f})\n")
    open(f"{OUT}/readout.txt", "w").write(hdr + txt + "\n")
    json.dump(dict(header=hdr, rows=rows), open(f"{OUT}/readout.json", "w"), indent=1, default=float)
    log(f"wrote {OUT}/readout.txt")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "score":
        score(sys.argv[2])
    elif mode == "readout":
        readout()
    else:
        raise SystemExit("usage: e2_trigger.py score <w2r_v3|w3r_v3> | readout")
