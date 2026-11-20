"""nn_x/w9/hs1 WAVE-9 arm HS1 = feed-surface HISTORY VIEW (scratchpad w9/hs1.json; CURRENT_STATUS 2026-11-13 13:09 / 13:11),
used by train.py, finetune.py, holdout_eval.py and stale_pred.py of this private copy of nn_x/w5. OFF by default: every flag
is '' at its train.py default and every call site sits behind `if a.hist_src or a.hist_et:` (train.py / finetune.py) or is the
identity `view_for(sd, args, cache) -> sd` (holdout_eval.py / stale_pred.py) when the checkpoint's stored args carry no view,
so a flag-off run executes exactly the nn_x/w5 code path (tests/test_hs1.py: AST proof + CPU end-to-end identity).

  --hist_src "1"            HS1: every GPU-side history channel (cand x history cosine features, category-affinity counts,
                            sim_bias, TabR retrieval, the history encoder) is built from the events whose event_source is in
                            the list ('1' = the main-page recommendation feed, the target's surface; CODES_DECODING_1113 s2).
  --hist_et "2,3,4,7,9"     HS1b (information arm): the actual Avito 'model 5' filter = target ACTIONS of the listed event
                            types from any source, plain views (type 1) excluded. Both flags given = events matching BOTH.

Mechanics (history_view = a clone of cd_arms4.cap_history_view with drop = ~isin(h_es, keep_src) [& ~isin(h_et, keep_et)]):
  the kept events are compacted in their stored (newest-first) order, H / H_np / hist_off / n_hist are recomputed and
  h_emb / h_cat / h_et / h_es (/ h_item) are the compacted arrays; candidate arrays, the item table, xt, user arrays and
  the domain mixture are SHARED with the original split (shallow copy). SplitData.gather then reads the newest max_hist
  kept events of every feed, so training, the selection holdout, val, test, the fine-tune, holdout_eval.py and stale_pred.py
  all see the same slice. The xt inputs (seq_cv, ctx_nd, ctx_hn/hs, pre2, LUO popularity) and user_emb are untouched.
  A feed whose slice is EMPTY (H -> 0; 2.97% of val / 2.37% of test feeds for --hist_src 1, 1.35% / 0.86% of which are
  already empty in the full history) goes through the same path as an empty-history feed of the base: hmask / hemb_ok all
  False, he_raw 0, smax / lse / top3 / per-type max = -1.5, counts 0, log1p(H) 0, retrieval all mass on the learned sink,
  history encoder context = the user token only.
  The view is driven by the CHECKPOINT (ck['args']['hist_src'] / ['hist_et']; also stored in ck['cfg'] so that a script that
  does not know the view, e.g. nn_x/w5/finetune.py, fails loudly in Ranker(**cfg) instead of silently scoring the full history).
Imports only numpy/torch (no GPU guard), so the unit tests call the very functions the scripts use.
"""
import copy

import numpy as np
import torch

HIST_VIEW_KEYS = ("hist_src", "hist_et")


def parse_codes(s, lo, hi, name):
    """'' -> (); '1' -> (1,); '2,3,4' -> (2, 3, 4); sorted, unique, each in [lo, hi]."""
    if s is None or str(s).strip() == "":
        return ()
    out = tuple(sorted({int(t) for t in str(s).split(",") if t.strip()}))
    assert out and all(lo <= c <= hi for c in out), f"bad --{name} spec {s!r} (codes {lo}..{hi})"
    return out


def view_spec(args):
    """(keep_src, keep_et) from an argparse Namespace or a stored-args / cfg dict; None when no view is set."""
    g = (lambda k: args.get(k, "")) if isinstance(args, dict) else (lambda k: getattr(args, k, ""))
    src, et = parse_codes(g("hist_src"), 0, 30, "hist_src"), parse_codes(g("hist_et"), 1, 10, "hist_et")
    return (src, et) if (src or et) else None


@torch.no_grad()
def history_view(sd, keep_src=(), keep_et=()):
    """Shallow copy of a SplitData / MiniSplit whose history arrays hold only the events with event_source in keep_src (if
    given) AND event_type in keep_et (if given), compacted in stored (newest-first) order (see module doc). Needs contiguous
    history segments (hist_off == exclusive cumsum of H, asserted; true for train / val / test). Returns (view, stats)."""
    keep_src, keep_et = tuple(keep_src or ()), tuple(keep_et or ())
    assert keep_src or keep_et, "history_view: nothing to keep"
    n, nh, dev = sd.n, sd.n_hist, sd.h_es.device
    Hl = sd.H.long()
    assert torch.equal(sd.hist_off.long(), torch.cumsum(Hl, 0) - Hl) and int(Hl.sum()) == nh, \
        "history segments must be contiguous"
    keep = torch.ones(nh, dtype=torch.bool, device=dev)
    if keep_src:
        keep &= torch.isin(sd.h_es, torch.tensor(keep_src, dtype=sd.h_es.dtype, device=dev))
    if keep_et:
        keep &= torch.isin(sd.h_et, torch.tensor(keep_et, dtype=sd.h_et.dtype, device=dev))
    feed_ev = torch.repeat_interleave(torch.arange(n, device=dev, dtype=torch.int32), Hl)
    nkeep = torch.bincount(feed_ev[keep].long(), minlength=n)
    H2 = nkeep.to(sd.H.dtype)
    off2 = torch.cumsum(H2.long(), 0) - H2.long()
    v = copy.copy(sd)
    v.h_emb, v.h_cat, v.h_et, v.h_es = sd.h_emb[keep], sd.h_cat[keep], sd.h_et[keep], sd.h_es[keep]
    if getattr(sd, "h_item", None) is not None:
        v.h_item = sd.h_item[keep]
    v.n_hist = int(H2.sum())
    if v.n_hist == 0:  # (tiny test splits only) keep one masked row: gather clamps its reads to n_hist - 1
        v.h_emb, v.h_cat, v.h_et, v.h_es = sd.h_emb[:1], sd.h_cat[:1], sd.h_et[:1], sd.h_es[:1]
        if getattr(sd, "h_item", None) is not None:
            v.h_item = sd.h_item[:1]
        v.n_hist = 1
    v.H = H2
    v.H_np = H2.cpu().numpy().astype(np.int64)
    v.hist_off = off2.to(sd.hist_off.dtype)
    v.hist_view = dict(src=keep_src, et=keep_et)
    et_all = torch.bincount(sd.h_et.long(), minlength=11)
    et_kept = torch.bincount(v.h_et.long()[: int(H2.sum())], minlength=11)
    empty_before, empty_after = sd.H == 0, H2 == 0
    st = dict(keep_src=keep_src, keep_et=keep_et, feeds=int(n), events_before=int(nh), events_after=int(H2.sum()),
              kept_share=float(H2.sum()) / max(int(nh), 1),
              feeds_changed=int((H2 != sd.H).sum()), feeds_changed_share=float((H2 != sd.H).float().mean()),
              feeds_empty_before=int(empty_before.sum()), feeds_empty_before_share=float(empty_before.float().mean()),
              feeds_empty_after=int(empty_after.sum()), feeds_empty_after_share=float(empty_after.float().mean()),
              feeds_emptied=int((empty_after & ~empty_before).sum()),
              feeds_emptied_share=float((empty_after & ~empty_before).float().mean()),
              mean_H_before=float(sd.H.float().mean()), mean_H_after=float(H2.float().mean()),
              kept_by_type={t: int(et_kept[t]) for t in range(1, 11)},
              all_by_type={t: int(et_all[t]) for t in range(1, 11)})
    del keep, feed_ev, nkeep
    return v, st


def apply_history_views(args, splits, log):
    """train.py / finetune.py entry: (views of the given splits), one log line each; frees the transient GPU blocks."""
    spec = view_spec(args)
    assert spec is not None, "apply_history_views called without a view"
    out = []
    for sd in splits:
        v, st = history_view(sd, *spec)
        if torch.cuda.is_available() and torch.device(sd.device).type == "cuda":
            torch.cuda.empty_cache()
        log(f"wave-9 HS1 history view on {sd.split}: keep_src {spec[0]} keep_et {spec[1]}; events {st['events_before']} -> "
            f"{st['events_after']} (kept {st['kept_share']:.4f}); feeds changed {st['feeds_changed']} "
            f"({st['feeds_changed_share']:.4f}); mean H {st['mean_H_before']:.2f} -> {st['mean_H_after']:.2f}; empty feeds "
            f"before {st['feeds_empty_before']} ({st['feeds_empty_before_share']:.4f}) -> after {st['feeds_empty_after']} "
            f"({st['feeds_empty_after_share']:.4f}), emptied by the view {st['feeds_emptied']} ({st['feeds_emptied_share']:.4f}); "
            f"kept by type {st['kept_by_type']}; every GPU-side history channel of this split reads the view (xt / user_emb "
            "untouched)" + (f"; gpu reserved {torch.cuda.memory_reserved() / 2**30:.1f} GB"
                            if torch.cuda.is_available() and torch.device(sd.device).type == "cuda" else ""))
        out.append(v)
    return tuple(out)


def view_for(sd, args, cache):
    """holdout_eval.py / stale_pred.py entry: the split itself when the checkpoint's stored args carry no view (identity, the
    nn_x/w5 path); otherwise the cached view for that spec with the split's current xt / dom / xe shared."""
    spec = view_spec(args)
    if spec is None:
        return sd
    assert getattr(sd, "c_item", None) is None, "history views: item-ID tables are not supported (id_buckets 0 in every recipe)"
    key = (sd.split, spec)
    if key not in cache:
        cache[key] = history_view(sd, *spec)[0]
        if torch.cuda.is_available() and torch.device(sd.device).type == "cuda":
            torch.cuda.empty_cache()
    v = cache[key]
    for k in ("xt", "dom", "xe", "_xt_names"):
        if hasattr(sd, k):
            setattr(v, k, getattr(sd, k))
    return v


def check_init_view(a, cfg):
    """--init consistency: a warm start from a checkpoint trained with a view needs the same --hist_src / --hist_et."""
    want = view_spec({k: cfg.get(k, "") for k in HIST_VIEW_KEYS})
    have = view_spec(a)
    assert want == have, f"--init checkpoint view {want} != this run's --hist_src / --hist_et {have}"
