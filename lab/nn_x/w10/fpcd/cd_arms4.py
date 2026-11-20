"""r6 wave-4 cooldown-DATA arms (reports/RESEARCH_SWEEP_1112.md P5 / P4 / P3), used by nn_x/r6/train.py.
Every arm is m02 (--cooldown_mix 0.02) plus ONE change, judged vs m02 (r6_cd_m02_r<root>, seed 811) by r6_eval_cdw4.py.
All flags are OFF at their train.py defaults; train.py builds the wave-4 objects only behind `if a.hcap or a.rec_q > 0 or
a.rho_file` and uses them only through guards / conditional expressions that reduce to the previous code when they are
None, so a flag-off run executes exactly the previous code path (tests/test_cdw4.py: AST proof + end-to-end parity).
Imports only numpy/torch (no GPU guard), so the unit tests call the very functions train.py uses.

  --hcap "1:110,4:117,6:41,8:42,10:40"   P5 history-format parity (review_integrity INTEGRITY-1). From the cooldown start
        (step t_cool; step 0 in the CD harness) every TRAINING batch (pool and non-pool) is gathered from a capped copy
        of the train histories: per feed and event type t, only the newest cap_t events of type t are kept, the kept
        events are compacted in their stored (newest-first) order and H becomes the kept count, i.e. exactly what
        SplitData.gather returns on a split whose stored histories were built with these caps (tests: equality with a
        numpy-built capped split). The holdout evaluation used for checkpoint SELECTION reads the same capped histories
        (train.py: eval_subset on the capped view). Val / test are never changed (they already have this format), and
        holdout_eval.py (the read-out's holdout preds) keeps the stored histories, as for m02.
        Caps (derived from data by hcap_caps.py -> logs/hcap_caps.txt; val+test per-feed type counts vs train):
          types capped = those whose train p99.9 exceeds the val/test p99.9 by > 10%: 1, 4, 6, 8, 10 (types 2/3/5/7/9 match);
          type 1: 110 = end of the val/test plateau (100-109: ~2,200 feeds per count; 110: 741, 111: 299); 2.5% of
                  val/test feeds exceed it vs 29.5% of train feeds (p99.9 174 is a thin tail, not a construction cap);
          types 4/6/8/10: the val/test p99.9 (max of val and test): 117 / 41 / 42 / 40.
        Caveat (as wave-2 aug): precomputed xt columns (co-vis, LUO, ctx_hn/hs) still describe the stored history.
  --rec_q Q (0 = off), --rec_days N (3)   P4 recency: at each NON-pool cooldown step, with probability Q the batch is
        drawn from a reshuffled stream over the training feeds (holdout excluded) of the last N train days (MSK days
        holding >= 1% of train feeds; train = Sep 6-14 + Sep 17-19, so N=3 -> Sep 17-19 from Sep 17 00:00 MSK, incl. the 34
        feeds just after midnight of Sep 20); otherwise the regular batch is used. Pool steps (m02 replay) are decided
        first by the unchanged mix RNG, so the pool share and the pool steps are the m02 ones.
  --rho_file F (""), --rho_keep K (0.7)   P3 learnability-selected cooldown (RHO-LOSS / JEST): F = rho/excess_r<root>.npz
        (rho_score.py: excess = L_snap - L_ref per production train feed, L = listwise CE / log M). Every NON-pool cooldown
        batch is drawn from a reshuffled stream over {top K of the production feeds by excess} + {the pool-eligible
        feeds, p_random > 0.5, which have no excess and stay at their m02 dose}; total steps and schedule unchanged
        (~2.8 passes over the kept set in the 2-epoch CD cooldown).
"""
import copy
import datetime

import numpy as np
import torch

PROD_P = 0.5  # production feeds = training feeds with p_random <= 0.5 (= not pool-eligible at the nn/train.py pool_p)
HCAP_F7 = "1:110,4:117,6:41,8:42,10:40"  # the P5 arm caps (hcap_caps.py)


# ------------------------------------------------------------------------------------------------ P5 hcap
def parse_caps(s):
    caps = {}
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            t, c = tok.split(":")
            caps[int(t)] = int(c)
    assert caps and all(1 <= t <= 10 and c >= 1 for t, c in caps.items()), f"bad --hcap spec {s!r}"
    return caps


@torch.no_grad()
def cap_drop_mask(h_et, hist_off, H, caps):
    """bool [n_hist]: True for events beyond their per-type cap. Rank of an event within (its feed, its type) in stored
    order (newest first), so the newest cap_t events of each type are kept. Needs contiguous history segments
    (hist_off == exclusive cumsum of H, asserted). Device-agnostic (GPU in train.py, CPU in the tests)."""
    n, nh = H.shape[0], h_et.shape[0]
    dev = h_et.device
    Hl = H.long()
    excl = torch.cumsum(Hl, 0) - Hl
    assert torch.equal(hist_off.long(), excl) and int(Hl.sum()) == nh, "history segments must be contiguous"
    feed_ev = torch.repeat_interleave(torch.arange(n, device=dev, dtype=torch.int32), Hl)
    start = hist_off.long().clamp_max(max(nh - 1, 0))
    drop = torch.zeros(nh, dtype=torch.bool, device=dev)
    for t, c in sorted(caps.items()):
        m = h_et == t
        cs = torch.cumsum(m, 0, dtype=torch.int32)
        cs.sub_(m.to(torch.int32))  # exclusive count of type t before each event
        base = cs[start]  # exclusive count at each feed's first event
        cs.sub_(base[feed_ev])  # rank of the event within (feed, type t) for type-t events
        drop |= m & (cs >= c)
        del cs, base, m
    return drop, feed_ev


@torch.no_grad()
def cap_history_view(sd, caps):
    """Shallow copy of a SplitData / MiniSplit whose history arrays are the capped, compacted histories (see module
    doc). Candidate arrays, item table, xt, user arrays are shared (no copy). Returns (view, stats)."""
    drop, feed_ev = cap_drop_mask(sd.h_et, sd.hist_off, sd.H, caps)
    keep = ~drop
    ndrop = torch.bincount(feed_ev[drop].long(), minlength=sd.n)
    H2 = sd.H - ndrop.to(sd.H.dtype)
    off2 = torch.cumsum(H2.long(), 0) - H2.long()
    v = copy.copy(sd)
    v.h_emb, v.h_cat, v.h_et, v.h_es = sd.h_emb[keep], sd.h_cat[keep], sd.h_et[keep], sd.h_es[keep]
    if getattr(sd, "h_item", None) is not None:
        v.h_item = sd.h_item[keep]
    v.H = H2
    v.H_np = H2.cpu().numpy().astype(np.int64)
    v.hist_off = off2.to(sd.hist_off.dtype)
    v.n_hist = int(H2.sum())
    v.hcap = dict(caps)
    st = dict(caps=dict(caps), events_before=int(sd.n_hist), events_after=v.n_hist,
              events_dropped_share=float(drop.float().mean()), feeds_changed=int((ndrop > 0).sum()),
              feeds_changed_share=float((ndrop > 0).float().mean()), mean_H_before=float(sd.H.float().mean()),
              mean_H_after=float(H2.float().mean()),
              dropped_by_type={int(t): int((drop & (sd.h_et == t)).sum()) for t in sorted(caps)})
    del drop, feed_ev, keep
    return v, st


# ------------------------------------------------------------------------------------------------ P4 rec
def recent_feeds(feeds, ts, n_days=3, min_share=0.01, tz_hours=3):
    """Training feeds of the last n_days train days (MSK = UTC+3). Days = calendar days holding >= min_share of all train
    feeds (drops the 42 / 34 spill-over feeds of Sep 15 / Sep 20); the boundary is 00:00 MSK of the first of the last
    n_days such days, and every feed at or after it is recent. Returns (sorted feed ids, boundary_ts, [dates])."""
    ts = np.asarray(ts)
    day = (ts + tz_hours * 3600) // 86400
    u, c = np.unique(day, return_counts=True)
    full = u[c >= min_share * len(ts)]
    last = full[-n_days:]
    assert len(last) == n_days and np.all(np.diff(last) == 1), f"last {n_days} train days not consecutive: {last}"
    boundary = int(last[0]) * 86400 - tz_hours * 3600
    rec = np.sort(feeds[ts[feeds] >= boundary])
    dates = [str(datetime.date(1970, 1, 1) + datetime.timedelta(days=int(d))) for d in last]
    return rec, boundary, dates


# ------------------------------------------------------------------------------------------------ P3 rho
def production_feeds(feeds, prand_np):
    return feeds[prand_np[feeds] <= PROD_P]


@torch.no_grad()
def per_feed_ce(model, data, feed_ids, cand_budget=65536, max_feeds=1024):
    """Per-feed listwise softmax CE (train.py's listwise_loss term of one feed: -sum_i y_i log p_i / sum_i y_i) divided by
    log M, in eval mode (no dropout) with the predict() forward (bf16 autocast, use_pos=False), for feed_ids (sorted
    1-D int array of data's feed indices). The domain token comes from data.dom (None -> the model's inference domain).
    Returns float64 [len(feed_ids)]; NaN for a feed without positives."""
    feed_ids = np.asarray(feed_ids)
    assert np.all(np.diff(feed_ids) > 0), "feed_ids must be sorted and unique"
    out = np.full(len(feed_ids), np.nan)
    was_training = model.training
    model.eval()
    for fb in data.batches(feed_ids, cand_budget, max_feeds, shuffle=False):
        b = data.gather(fb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(b, use_pos=False)
        m, y = b["cmask"], b["y"]
        logp = torch.log_softmax(lg.float().masked_fill(~m, -1e4), 1)
        ys = y.sum(1)
        ce = -(y * logp).sum(1) / ys.clamp_min(1)
        ce = torch.where(ys > 0, ce / torch.log(b["M"].double().clamp_min(2)).float(), torch.full_like(ce, float("nan")))
        out[np.searchsorted(feed_ids, np.asarray(fb))] = ce.double().cpu().numpy()
    if was_training:
        model.train()
    return out


def rho_stream_feeds(feeds, prand_np, hold, path, keep, init=None):
    """Regular cooldown stream of the rho arm: top `keep` of the production feeds by excess (ties by feed id) + every
    pool-eligible feed (p_random > 0.5). Checks that the file was scored on exactly this run's production feeds (same
    holdout exclusion) and, if given, with this run's --init snapshot. Returns (sorted feed ids, stats)."""
    assert 0.5 <= keep < 1.0, f"--rho_keep must be in [0.5, 1) (RESEARCH_SWEEP P3: never below 0.5): {keep}"
    z = np.load(path, allow_pickle=False)
    prod = production_feeds(feeds, prand_np)
    fidx, exc = z["fidx"].astype(np.int64), z["excess"].astype(np.float64)
    assert np.array_equal(fidx, prod), f"{path}: scored feeds != this run's production feeds (holdout / p_random rule)"
    assert np.isfinite(exc).all(), f"{path}: non-finite excess"
    if init:
        assert str(z["snap"]) == init, f"{path} was scored with {z['snap']}, run inits from {init}"
    n_keep = int(round(keep * len(prod)))
    order = np.lexsort((fidx, -exc))  # excess descending, ties by feed id
    kept = np.sort(fidx[order[:n_keep]])
    pool = feeds[prand_np[feeds] > PROD_P]
    stream = np.union1d(kept, pool)
    if hold is not None:
        assert not np.isin(stream, hold).any(), "LEAK GUARD: holdout feeds in the rho cooldown stream"
    st = dict(file=str(path), n_prod=len(prod), n_keep=n_keep, keep=keep, n_pool=len(pool), n_stream=len(stream),
              excess_cut=float(exc[order[n_keep - 1]]), excess_mean_kept=float(exc[order[:n_keep]].mean()),
              excess_mean_dropped=float(exc[order[n_keep:]].mean()) if n_keep < len(prod) else float("nan"))
    return stream, st


# ------------------------------------------------------------------------------------------------ cooldown stream
class CoolStream:
    """Source of the NON-pool cooldown batches for rec / rho (train.py calls next(fb) only on non-pool cooldown steps,
    after the unchanged m02 pool decision). rec: with prob rec_q a batch from the recent stream, else fb (the regular
    batch of that step). rho: a batch from the rho stream (replaces fb). Own RNG (seed + 2): the mix RNG, the pool
    stream and the regular batch order of m02 are untouched. Streams are reshuffled with tr.batches when exhausted."""

    def __init__(self, tr, cand_budget, max_feeds, rng, rec=None, rec_q=0.0, rho=None):
        self.tr, self.cb, self.mf, self.rng = tr, cand_budget, max_feeds, rng
        self.rec, self.rec_q, self.rho = rec, float(rec_q), rho
        assert rec is not None or rho is not None
        if rec is not None:
            assert 0.0 < self.rec_q < 1.0, f"--rec_q must be in (0, 1): {rec_q}"
        self.st = {"rec": [[], 0, 0], "rho": [[], 0, 0]}  # [batches, next index, passes]
        self.n = dict(regular=0, recent=0, rho=0)

    def _draw(self, k):
        s = self.st[k]
        if s[1] >= len(s[0]):
            s[0], s[1], s[2] = self.tr.batches(self.rec if k == "rec" else self.rho, self.cb, self.mf, shuffle=True,
                                               rng=self.rng), 0, s[2] + 1
        fb = s[0][s[1]]
        s[1] += 1
        return fb

    def next(self, fb):
        if self.rec is not None and self.rng.random() < self.rec_q:
            self.n["recent"] += 1
            return self._draw("rec")
        if self.rho is not None:
            self.n["rho"] += 1
            return self._draw("rho")
        self.n["regular"] += 1
        return fb

    def summary(self):
        tot = max(sum(self.n.values()), 1)
        return (f"wave-4 stream (non-pool cooldown batches): regular {self.n['regular']} recent {self.n['recent']} "
                f"({self.n['recent'] / tot:.3f}) rho {self.n['rho']} ({self.n['rho'] / tot:.3f}); passes recent "
                f"{self.st['rec'][2]} rho {self.st['rho'][2]}")


# ------------------------------------------------------------------------------------------------ train.py entry
def setup_wave4(a, tr, feeds, prand_np, hold, log):
    """Builds (capped train view or None, CoolStream or None) for train.py. Called only when a wave-4 flag is on."""
    assert a.cooldown_mix > 0, "wave-4 arms are m02 + one change: they need --cooldown_mix > 0 (and --holdout)"
    assert hold is not None, "wave-4 arms need --holdout (leak guards and the selection holdout)"
    tr_cap = stream = None
    if a.hcap:
        caps = parse_caps(a.hcap)
        tr_cap, st = cap_history_view(tr, caps)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # release the transient blocks of the capping pass (~6-10 GB on the full split)
        log(f"wave-4 hcap: caps {caps}; events {st['events_before']} -> {st['events_after']} (dropped "
            f"{st['events_dropped_share']:.4f}, by type {st['dropped_by_type']}); feeds changed {st['feeds_changed']} "
            f"({st['feeds_changed_share']:.4f}); mean H {st['mean_H_before']:.2f} -> {st['mean_H_after']:.2f}; capped "
            f"histories on every training batch from the cooldown start and on the SELECTION holdout; val/test unchanged"
            + (f"; gpu reserved {torch.cuda.memory_reserved() / 2**30:.1f} GB" if torch.cuda.is_available() else ""))
    rec = rho = None
    if a.rec_q > 0:
        ts = tr.feeds["timestamp"].to_numpy()
        rec, boundary, dates = recent_feeds(feeds, ts, a.rec_days)
        assert not np.isin(rec, hold).any(), "LEAK GUARD: holdout feeds in the recent stream"
        log(f"wave-4 rec: q {a.rec_q} of non-pool cooldown batches from days {dates} (ts >= {boundary}, 00:00 MSK): "
            f"{len(rec)} feeds ({len(rec) / len(feeds):.4f} of training feeds; pool-eligible "
            f"{int((prand_np[rec] > PROD_P).sum())}); holdout feeds in stream 0")
    if a.rho_file:
        rho, st = rho_stream_feeds(feeds, prand_np, hold, a.rho_file, a.rho_keep, a.init)
        log(f"wave-4 rho: {st}")
    if rec is not None or rho is not None:
        stream = CoolStream(tr, a.cand_budget, a.max_feeds, np.random.default_rng(a.seed + 2), rec, a.rec_q, rho)
    return tr_cap, stream
