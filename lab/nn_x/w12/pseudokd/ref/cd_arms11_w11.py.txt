"""nn_x/w11/pseudo PSEUDO-DIRECT (reports/SOLVER_POWER_1113.md section 6.3 item 3; CURRENT_STATUS 2026-11-13 23:51): pseudo-feed
co-training, used by nn_x/w11/pseudo/train.py through the flags below. Written 2026-11-14 ~00:10 UTC. OFF at the train.py
defaults (--pseudo_ratio 0): train.py builds the object only behind `if a.pseudo_ratio > 0:` and reads it only through
`if pseudo is not None ...` blocks, so a flag-off run executes exactly the nn_x/w5 code path (tests/test_pseudo.py: AST proof
against tests/ref_w5_train.py.txt + end-to-end batch / logit / log / checkpoint identity on the mini split).

THE SPEC (section 6.3 item 3, construction): "Pseudo-feeds built on the GPU from the batch's own histories. Target: a new
type-1/source-1 event at t <= 63. History: the events older than t + g, g ~ U[16, 64], recency re-indexed. Negatives: the source
feed's non-clicked candidates. user_emb zeroed (a learned no-user vector); a fourth domain embedding; f1-f5 and the 34
pre-computed columns on their missing path; K9 weight 0; listwise CE weight 0.3; plateau only. The cooldown stays on real
feeds for an absolute 19,670 steps (never cooldown_frac x total). Holdout users and later same-user snapshots are never a
source. The pseudo domain gets its own context-free linear head on [content, category], so the shared pathway is not paid to
learn the past-click marginal (shortcut AUC 0.590 is not removed by logQ). Report a set-attention-masked variant on pseudo
rows as information (set leak 0.608 vs 0.582)."

CONSTRUCTION, per training step of the PLATEAU (step < t_cool; never on cooldown / pool / val batches), from the gathered REAL
batch b of B feeds (history newest-first: index 0 = the most recent event; K = padded history length <= max_hist):
  draws      P = ceil(R B) (source feed, t, g) draws, R = --pseudo_ratio; the source feeds are a cyclic random permutation of
             the batch rows, so at an integer R every row is a source exactly R times; g ~ U{gap_min..gap_max} (16..64) per draw.
  source     a row is a source only if its user is not a HOLDOUT user (user ids of the --holdout feeds; user 0 = pooled bucket,
             never matched). "Later same-user snapshots" are val / test rows, which this train-only stage never reads.
  target     t ~ uniform over the ELIGIBLE indices t in [0, tmax] (tmax 63) of the source row: a valid event with an embedding,
             event_type 1 from event_source 1 (feed-originated item view), NEW = its item occurs in no OLDER event of the loaded
             history (index > t; items identified by the item-table row, which is one-to-one with item_id), NOT A CANDIDATE of the
             source feed (its item is none of the feed's M candidates, clicked or not: the positive of a pseudo-feed is never a
             candidate of its source feed), and t + g <= H_cap - 1 (at least one history event remains older than the gap).
  history    the source row's events at indices >= t + g (the newest kept event is exactly g positions older than the target;
             everything at indices < t + g, i.e. the target and the gap, is dropped), re-indexed to 0.. (recency embedding, the
             retrieval recency bucket and the newest-first order all follow), H' = H_cap - t - g, padded to a multiple of 64.
  candidates slot 0 = the target item (its embedding and category from the history event); slots 1.. = the source feed's
             NON-CLICKED candidates (y == 0) in their original order, minus the ones whose item occurs in the history at an
             index < t + g (the user DID interact with them after the pseudo request time: masked out as unlabelled, not used as
             negatives); a draw with no remaining negative is dropped; M' = 1 + n_neg, padded to a multiple of 16.
  inputs     f1-f5 = 0 and every --xt column = 0 (incl. has:pre2_fe = 0) on EVERY candidate of a pseudo-feed, target and
             negatives alike: the target has none of these inputs (they exist for candidate rows only) and putting only the
             target on the missing path would mark it (a shortcut far above the 0.59 content shortcut); the within-feed z / rank
             of these columns are then constant. user_emb = 0 (un = 0: cos_u = dot_u = 0, the user token = u_lin's bias, a learned
             no-user vector); user0 flag and fidx of the source row; domain one-hot row 2 (the "exploration" row of the existing
             3-row domain table, which the recipe never trains (tr.dom[:, 2] = 0) nor uses at inference (infer_domain 1): a
             dedicated embedding without a model change); no production positions (use_pos=False; the recipe has --no_pos).
  head       --pseudo_head 1: logit += lin(normalised content embedding) + cat_emb[category] on pseudo rows only (zero-init;
             parameters outside the model's state_dict so every checkpoint stays loadable by finetune / holdout_eval /
             stale_pred; its own AdamW group (no decay, the run's lr schedule); saved to ckpt/<v>_pseudo_head.pt at the end).
  loss       loss = loss_real + W * listwise_loss(lg_p, y_p, cmask_p, fw_p, norm=a.loss_norm, denom=a.max_feeds), W =
             --pseudo_w (0.3): the SAME normalisation and the SAME constant denominator as the real loss (--loss_norm feed:
             sum over feeds / max_feeds, so one pseudo-feed weighs W times one real feed). fw_p = None when the real fw is
             None (the recipe: --prand_base -1), else prand_base + 1 (pseudo rows are random-like). K9 aux: never on pseudo
             rows (the real aux term is built before combine(); model.aux_out of the pseudo forward is ignored). TRUNC offsets,
             R-Drop / SAM second passes and --intent are not combined (asserted). CLIPLOG's main := main + W loss_p (aux = K9).
  variant    --pseudo_noset 1 (information): the candidate-set self-attention blocks are skipped on pseudo rows (model.use_set
             False during the pseudo forward only), removing the request-time set cue (set-centroid AUC 0.608 vs 0.582).
  shapes     the pseudo batch always has ceil(R B) rows (dropped draws become zero-weight duplicates, pseudo_valid False), M' padded to the
             source batch's Mx (or Mx + 16), H' to a multiple of 64: a small, stable set of (P, M, K) shapes (kernel-selection warm-up).
  steps      the pseudo forward / backward is an EXTRA pass inside the same step: the step count, t_cool, the cooldown length
             (19,670 steps at the recipe's 98,350 total) and the batch sequence are unchanged; pseudo-feeds stop at t_cool.
  stats      per eval: steps with pseudo, pseudo-feeds built (= positives: exactly one per pseudo-feed), mean M', mean H',
             mean t, mean g, draws dropped (no eligible target / no negative), source rows blocked (holdout users), mean
             pseudo loss per feed ('wave-11 PSEUDO' log line); cumulative totals saved with the head.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from model import N_CAT, listwise_loss

PSEUDO_DOM = 2  # the pseudo domain = row 2 of the 3-row domain table (never used by the recipe otherwise)


class PseudoHead(nn.Module):
    """Context-free linear head on [content, category] for pseudo rows (zero-init: an exact no-op at step 0)."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(128, 1)
        self.cat_emb = nn.Embedding(N_CAT, 1)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        nn.init.zeros_(self.cat_emb.weight)

    def forward(self, ce, ccat):
        return self.lin(ce.float()).squeeze(-1) + self.cat_emb(ccat).squeeze(-1)


def eligible_targets(b, cemb, tmax):
    """[B, T] bool, T = min(tmax + 1, K): index t of the (newest-first) history is an eligible pseudo target of the row.
    cemb [B, M] = item-table row of each candidate (-1 none; padded slots arbitrary, masked by cmask)."""
    het, hes, hv, hidx, hmask = b["het"], b["hes"], b["hemb_ok"], b["hidx"], b["hmask"]
    B, K = het.shape
    T = min(tmax + 1, K)
    dev = het.device
    tgt = hidx[:, :T]
    base = hmask[:, :T] & hv[:, :T] & (het[:, :T] == 1) & (hes[:, :T] == 1)
    older = torch.arange(K, device=dev)[None, None, :] > torch.arange(T, device=dev)[None, :, None]  # [1,T,K]: j > t
    seen = ((tgt[:, :, None] == hidx[:, None, :]) & hv[:, None, :] & older).any(-1)  # the item occurs in an older event
    incand = ((tgt[:, :, None] == cemb[:, None, :]) & b["cmask"][:, None, :]).any(-1)  # the item is a candidate of the feed
    return base & ~seen & ~incand


def draw_sources(B, ratio, gen, dev):
    """[P] source rows, P = ceil(ratio B): a cyclic random permutation (every row exactly R times at an integer R)."""
    P = int(math.ceil(ratio * B))
    nrep = int(math.ceil(P / B))
    return torch.cat([torch.randperm(B, generator=gen, device=dev) for _ in range(nrep)])[:P]


def build_pseudo_batch(b, cemb, src_ok, ratio, gmin, gmax, tmax, max_hist, gen):
    """Pseudo batch dict (same keys the model reads from a gathered batch) + diagnostics; None if no pseudo-feed can be built.
    b = the gathered REAL training batch; cemb [B, M] candidate item-table rows; src_ok [B] bool (row may be a source)."""
    dev = b["het"].device
    B, M = b["cmask"].shape
    K = b["het"].shape[1]
    Hc = b["H"]  # [B] capped history length (gather: clamp_max(max_hist))
    elig = eligible_targets(b, cemb, tmax) & src_ok[:, None]  # [B,T]
    T = elig.shape[1]
    src = draw_sources(B, ratio, gen, dev)  # [P]
    P0 = src.shape[0]
    g = torch.randint(gmin, gmax + 1, (P0,), generator=gen, device=dev)
    tt = torch.arange(T, device=dev)
    ok = elig[src] & ((tt[None, :] + g[:, None]) <= (Hc[src] - 1)[:, None])  # [P,T]
    score = torch.rand(P0, T, generator=gen, device=dev).masked_fill(~ok, -1.0)
    t = score.argmax(1)
    keep = ok.any(1)
    n_drop_tgt = (~keep).sum()  # tensor (no host sync)
    src, t, g = src[keep], t[keep], g[keep]
    # negatives: the source feed's non-clicked candidates whose item does not occur at a history index < t + g
    off = t + g  # [P] first kept history index
    kk = torch.arange(K, device=dev)
    fut = ((cemb[src][:, :, None] == b["hidx"][src][:, None, :]) & b["hemb_ok"][src][:, None, :]
           & (kk[None, None, :] < off[:, None, None])).any(-1)  # [P,M]
    negmask = b["cmask"][src] & (b["y"][src] == 0) & ~fut
    n_neg = negmask.sum(1)
    keep2 = n_neg >= 1
    n_drop_neg = (~keep2).sum()
    if not bool(keep2.any()):
        return None, dict(n_drop_tgt=n_drop_tgt, n_drop_neg=n_drop_neg, n_blocked=(~src_ok).sum())
    src, t, g, off, negmask, n_neg = src[keep2], t[keep2], g[keep2], off[keep2], negmask[keep2], n_neg[keep2]
    # SHAPE STABILITY: the pseudo batch always has P0 = ceil(R B) rows (the number of draws); the rows of dropped draws are
    # duplicates of kept rows with an all-zero y (loss weight 0, loss 0: no gradient), flagged pseudo_valid = False. Candidate
    # padding = the source batch's Mx (a multiple of 16) or Mx + 16; history padding = a multiple of 64. Every new (P, M, K) shape
    # costs a kernel-selection warm-up on the GPU (data.py), and an unpadded P would take hundreds of distinct values.
    Pv = src.shape[0]
    pad = torch.arange(P0, device=dev) % Pv
    valid = torch.arange(P0, device=dev) < Pv
    src, t, g, off, negmask, n_neg = src[pad], t[pad], g[pad], off[pad], negmask[pad], n_neg[pad]
    P = P0
    Mp = 1 + n_neg
    Mx = M if int(Mp.max()) <= M else M + 16
    order = torch.argsort((~negmask).to(torch.int8), dim=1, stable=True)  # negatives first, original order
    s = (torch.arange(Mx, device=dev) - 1).clamp(0, M - 1)
    pos = order[:, s]  # [P,Mx] source candidate slot of pseudo slot (slot 0 overwritten by the target)
    cmask_p = torch.arange(Mx, device=dev)[None, :] < Mp[:, None]
    rows = src[:, None].expand(-1, Mx)
    ce_raw = b["ce_raw"][rows, pos].clone()
    cmiss = b["cmiss"][rows, pos].clone()
    ccat = b["ccat"][rows, pos].clone()
    ce_raw[:, 0] = b["he_raw"][src, t]
    cmiss[:, 0] = False
    ccat[:, 0] = b["hcat"][src, t]
    ce_raw = ce_raw * cmask_p[..., None].to(ce_raw.dtype)
    cmiss = cmiss | ~cmask_p
    ccat = torch.where(cmask_p, ccat, torch.zeros_like(ccat))
    y = torch.zeros(P, Mx, device=dev, dtype=b["y"].dtype)
    y[:, 0] = valid.to(y.dtype)  # padded (invalid) rows: no positive -> loss weight 0
    # history: indices >= t + g, re-indexed to 0..
    Hp = Hc[src] - off  # [P] >= 1
    Kp = min(max(-(-int(Hp.max()) // 64) * 64, 64), max_hist)
    k = torch.arange(Kp, device=dev)
    hmask_p = k[None, :] < Hp[:, None]
    hsrc = (k[None, :] + off[:, None]).clamp_max(K - 1)  # [P,Kp]
    hrows = src[:, None].expand(-1, Kp)
    he_raw = b["he_raw"][hrows, hsrc] * hmask_p[..., None].to(b["he_raw"].dtype)
    hemb_ok = b["hemb_ok"][hrows, hsrc] & hmask_p
    z = torch.zeros((), dtype=torch.long, device=dev)
    hcat = torch.where(hmask_p, b["hcat"][hrows, hsrc], z)
    het = torch.where(hmask_p, b["het"][hrows, hsrc], z)
    hes = torch.where(hmask_p, b["hes"][hrows, hsrc], z)
    hidx = torch.where(hmask_p, b["hidx"][hrows, hsrc], torch.full((), -1, dtype=b["hidx"].dtype, device=dev))
    bp = dict(user0=b["user0"][src], fidx=b["fidx"][src], M=Mp, H=Hp, cmask=cmask_p, hmask=hmask_p,
              ce_raw=ce_raw, cmiss=cmiss, ccat=ccat, cf=torch.zeros(P, Mx, 5, device=dev, dtype=b["cf"].dtype),
              he_raw=he_raw, hemb_ok=hemb_ok, hcat=hcat, het=het, hes=hes,
              ue_raw=torch.zeros(P, b["ue_raw"].shape[1], device=dev, dtype=b["ue_raw"].dtype), hidx=hidx, y=y,
              pseudo_src=src, pseudo_t=t, pseudo_g=g, pseudo_valid=valid)
    if "dom" in b:
        dom = torch.zeros(P, b["dom"].shape[1], device=dev, dtype=b["dom"].dtype)
        dom[:, PSEUDO_DOM] = 1.0
        bp["dom"] = dom
    if "xt" in b:
        bp["xt"] = torch.zeros(P, Mx, b["xt"].shape[-1], device=dev, dtype=b["xt"].dtype)
    if "citem" in b:  # hashed item-ID residual inputs (not in the recipe): same construction on the global item ids
        citem = b["citem"][rows, pos].clone()
        citem[:, 0] = b["hitem"][src, t]
        bp["citem"] = torch.where(cmask_p, citem, torch.zeros_like(citem))
        bp["hitem"] = torch.where(hmask_p, b["hitem"][hrows, hsrc], torch.zeros_like(b["hitem"][hrows, hsrc]))
    return bp, dict(n_drop_tgt=n_drop_tgt, n_drop_neg=n_drop_neg, n_blocked=(~src_ok).sum())


class PseudoTrainer:
    """Builds and trains on pseudo-feeds inside the training step (module docstring). Lives on the model's device."""

    def __init__(self, a, model, tr, src_ok, opts, t_end, head_seed):
        assert a.pseudo_ratio > 0
        dev = next(model.parameters()).device  # the model's device (CPU in the unit / e2e tests)
        self.ratio, self.w = float(a.pseudo_ratio), float(a.pseudo_w)
        self.gmin, self.gmax, self.tmax = int(a.pseudo_gap_min), int(a.pseudo_gap_max), int(a.pseudo_tmax)
        self.max_hist = int(a.max_hist)
        self.noset = bool(a.pseudo_noset)
        self.t_end = int(t_end)  # pseudo-feeds on steps < t_end (= t_cool: plateau only)
        self.tr = tr
        self.src_ok = src_ok.to(dev)  # [tr.n] bool: the feed may be a pseudo source (not a holdout user)
        self.dev = dev
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(int(a.seed) + 1_100_011)
        self.a = a
        self.head = None
        if a.pseudo_head:
            with torch.random.fork_rng(devices=[]):  # the head is zero-init: the fork only keeps the global RNG untouched
                torch.manual_seed(head_seed)
                self.head = PseudoHead().to(dev)
            opts[0].add_param_group({"params": list(self.head.parameters()), "weight_decay": 0.0})
        self.cum = dict(steps=0, feeds=0, sumM=0, sumH=0, sumt=0, sumg=0, drop_tgt=0, drop_neg=0, blocked=0, draws=0)
        self.acc = torch.zeros(10, device=dev, dtype=torch.float64)  # per-eval accumulators (no host sync per step)
        # acc: 0 steps_with_pseudo, 1 feeds, 2 sum M', 3 sum H', 4 sum t, 5 sum g, 6 drop_tgt, 7 drop_neg, 8 blocked rows, 9 loss sum
        self.n_draws = 0
        self.n_steps_seen = 0
        self.last_P = torch.zeros((), device=dev)

    def build(self, b):
        cemb = self.tr.c_emb[b["ci"]].long()  # [B,M] item-table row of each candidate (padding: masked by cmask)
        src_ok = self.src_ok[b["fidx"]]
        return build_pseudo_batch(b, cemb, src_ok, self.ratio, self.gmin, self.gmax, self.tmax, self.max_hist, self.gen)

    def pseudo_loss(self, model, a, bp, prand):
        """Forward of the pseudo batch through the SAME model (+ the context-free head on pseudo rows); the listwise loss with
        the real loss's normalisation and denominator. Returns (loss_p, logits)."""
        X = model.dense(bp)
        saved = model.use_set
        if self.noset:
            model.use_set = False
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
                lg = model(bp, X, use_pos=False)
        finally:
            model.use_set = saved
        if self.head is not None:
            lg = lg + self.head(bp["ce"], bp["ccat"])
        fw = None if prand is None or a.prand_base < 0 else torch.full((lg.shape[0],), a.prand_base + 1.0, device=lg.device)
        return listwise_loss(lg, bp["y"], bp["cmask"], fw, norm=a.loss_norm, denom=a.max_feeds), lg

    def combine(self, model, a, b, loss, prand, cliplog):
        """loss_real + W * loss_pseudo for the REAL batch b (called on plateau steps only, after the real loss is complete)."""
        self.n_steps_seen += 1
        bp, info = self.build(b)
        self.acc[6] += info["n_drop_tgt"].double()
        self.acc[7] += info["n_drop_neg"].double()
        self.acc[8] += info["n_blocked"].double()
        self.n_draws += int(math.ceil(self.ratio * b["cmask"].shape[0]))
        if bp is None:
            self.last_P = torch.zeros((), device=self.dev)
            return loss
        loss_p, _ = self.pseudo_loss(model, a, bp, prand)
        v = bp["pseudo_valid"]
        vf = v.to(torch.float64)
        self.last_P = v.sum()  # tensor (no host sync in the training loop)
        self.acc[0] += 1
        self.acc[1] += vf.sum()
        self.acc[2] += (bp["M"].double() * vf).sum()
        self.acc[3] += (bp["H"].double() * vf).sum()
        self.acc[4] += (bp["pseudo_t"].double() * vf).sum()
        self.acc[5] += (bp["pseudo_g"].double() * vf).sum()
        self.acc[9] += loss_p.detach().double() * a.max_feeds if a.loss_norm != "batch" else loss_p.detach().double() * vf.sum()
        term = self.w * loss_p
        if cliplog is not None and cliplog.main is not None:  # CLIPLOG main / aux split: main = the listwise terms, aux = K9
            cliplog.main = cliplog.main + term
        return loss + term

    def summary(self, step):
        v = self.acc.cpu().numpy()
        n_st, n_f = int(v[0]), int(v[1])
        s = (f"wave-11 PSEUDO step {step}: {n_st} steps with pseudo-feeds of {self.n_steps_seen} plateau steps since the last eval; "
             f"{n_f} pseudo-feeds (= positives) from {self.n_draws} draws (dropped: no eligible target {int(v[6])}, no negative "
             f"{int(v[7])}; source rows blocked as holdout users {int(v[8])}); mean M' {v[2] / max(n_f, 1):.1f} H' "
             f"{v[3] / max(n_f, 1):.1f} t {v[4] / max(n_f, 1):.1f} g {v[5] / max(n_f, 1):.1f}; mean pseudo CE per feed "
             f"{v[9] / max(n_f, 1):.4f} (weight {self.w}, ratio {self.ratio}, gap {self.gmin}-{self.gmax}, tmax {self.tmax}, "
             f"head {'on' if self.head is not None else 'off'}, noset {int(self.noset)}, pseudo until step {self.t_end})")
        for k_, i in (("steps", 0), ("feeds", 1), ("sumM", 2), ("sumH", 3), ("sumt", 4), ("sumg", 5), ("drop_tgt", 6),
                      ("drop_neg", 7), ("blocked", 8)):
            self.cum[k_] += int(v[i])
        self.cum["draws"] += self.n_draws
        self.acc.zero_()
        self.n_draws, self.n_steps_seen = 0, 0
        return s

    def save(self, path, log):
        torch.save({"head": None if self.head is None else {k: v.cpu() for k, v in self.head.state_dict().items()},
                    "cum": dict(self.cum), "ratio": self.ratio, "w": self.w, "gap": (self.gmin, self.gmax), "tmax": self.tmax,
                    "noset": self.noset, "t_end": self.t_end}, path)
        c = self.cum
        log(f"wave-11 PSEUDO final: {c['steps']} steps with pseudo-feeds, {c['feeds']} pseudo-feeds from {c['draws']} draws, mean M' "
            f"{c['sumM'] / max(c['feeds'], 1):.1f} H' {c['sumH'] / max(c['feeds'], 1):.1f} t {c['sumt'] / max(c['feeds'], 1):.1f} g "
            f"{c['sumg'] / max(c['feeds'], 1):.1f}; dropped {c['drop_tgt']} / {c['drop_neg']}; blocked rows {c['blocked']}; head saved to {path}")


def holdout_user_mask(tr, hold):
    """[tr.n] bool numpy: the feed may be a pseudo source = its user is not a holdout-feed user (user 0 never matched)."""
    ok = np.ones(tr.n, dtype=bool)
    if hold is not None and len(hold):
        hu = np.unique(tr.user_id[hold])
        hu = hu[hu != 0]
        ok = ~np.isin(tr.user_id, hu) | (tr.user_id == 0)
    return ok


def setup_pseudo(a, model, tr, feeds, hold, opts, t_cool, total, log):
    """Called once when --pseudo_ratio > 0: guards, the holdout-user source mask, the trainer (+ head in opts[0])."""
    assert a.pseudo_ratio > 0
    assert a.loss == "softmax", "--pseudo_ratio: the listwise softmax loss only"
    assert a.intent == 0, "--pseudo_ratio: not combined with --intent"
    assert not a.trunc_off and not a.trunc_lifts, "--pseudo_ratio: not combined with TRUNC-PROD (no production order on pseudo-feeds)"
    assert a.rdrop_cd <= 0 and a.sam_cd <= 0, "--pseudo_ratio: not combined with the wave-7 R-Drop / SAM second passes"
    assert a.val_mix <= 0 and not a.hzcd and not a.hcap, "--pseudo_ratio: not combined with valcd / hzcd / hcap views"
    assert a.arch != "din", "--pseudo_ratio: arch tf / gru only"
    assert 1 <= a.pseudo_gap_min <= a.pseudo_gap_max, "--pseudo_gap_min/max: 1 <= min <= max"
    assert 0 <= a.pseudo_tmax < a.max_hist, "--pseudo_tmax must be < --max_hist"
    assert a.pseudo_w > 0, "--pseudo_w must be > 0 with --pseudo_ratio > 0"
    assert 0 < t_cool <= total, f"--pseudo_ratio: pseudo-feeds are plateau-only and the plateau is empty (t_cool {t_cool} of {total} steps)"
    assert a.sched == "wsd", "--pseudo_ratio: the plateau / cooldown split is defined by the WSD schedule"
    ok_np = holdout_user_mask(tr, hold)
    n_blocked = int((~ok_np[feeds]).sum())
    pseudo = PseudoTrainer(a, model, tr, torch.from_numpy(ok_np), opts, t_cool, head_seed=int(a.seed) + 7919 * 11)
    n_head = 0 if pseudo.head is None else sum(p.numel() for p in pseudo.head.parameters())
    log(f"wave-11 PSEUDO-DIRECT: --pseudo_ratio {a.pseudo_ratio} (ceil(R B) draws per plateau step, steps 0..{t_cool - 1} of {total}; "
        f"the cooldown {total - t_cool} steps stays on real feeds), target = new type-1/source-1 event at t <= {a.pseudo_tmax}, "
        f"g ~ U[{a.pseudo_gap_min}, {a.pseudo_gap_max}], history = events at index >= t + g re-indexed, negatives = the source feed's "
        f"non-clicked candidates (minus items seen at index < t + g), f1-f5 / xt = 0, user_emb = 0, domain row {PSEUDO_DOM}, "
        f"K9 aux 0, loss += {a.pseudo_w} * listwise(norm {a.loss_norm}, denom {a.max_feeds}); context-free head "
        f"{'ON (' + str(n_head) + ' params, AdamW no-decay group)' if pseudo.head is not None else 'off'}; set attention on pseudo rows "
        f"{'MASKED (information variant)' if pseudo.noset else 'on'}; source rows blocked as holdout users: {n_blocked} of "
        f"{len(feeds)} training feeds ({len(np.unique(tr.user_id[hold])) if hold is not None else 0} holdout users)")
    return pseudo
