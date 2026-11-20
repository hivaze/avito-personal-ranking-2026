"""nn_x/w5 WAVE-5 arms and hygiene (reports/INSIGHTS_1112.md section 3.1 items 1 TRUNC-PROD, 3 AUXF, 8 hygiene), used by
nn_x/w5/train.py. Every flag is OFF at its train.py default; train.py builds the wave-5 objects only behind
`if a.trunc_off or a.aux_norm != "batch" or a.cliplog` and uses every wave-5 object / flag only through guards (or one
conditional expression) that reduce to the nn_x/r6 code when off, so a flag-off run executes exactly the nn_x/r6/train.py
code path (tests/test_w5.py: AST proof against a frozen copy of nn_x/r6/train.py + end-to-end batch / log / checkpoint
identity). Imports only numpy/torch (no GPU guard), so the unit tests call the very functions train.py uses.

  --trunc_off A (0 = off; arm value 1.0)   TRUNC-PROD (data_usage E1 + labels_loss E2). A training-only additive offset on
        the MAIN listwise logits of every TRAINING batch:
            lg[f, i] += A * L[g_f, rkend_i] * (1 - p_soft_f)   for rkend_i in {0, 1, 2}; 0 for rkend >= 3,
        rkend_i = TRUE rank from the end of the shown feed (0 = last shown item) from
            nn_x/insights/data_usage/out/nn_artifacts/train_rkend_true.npy (int16, flat train candidate order; NOT M-1-pos:
            85.5% of train feeds have position gaps),
        L = the log click lift of rkend 0/1/2 vs rkend >= 3 from .../nn_artifacts/trunc_lift.json:
            group 0 'prod_M<50'  (every M<50 feed: production by the NN rule p := 0 for M<50): 1.0038 / 0.5449 / 2.0131
            group 1 'prod_M>=50' (M>=50 and NN p_random <= 0.5):                                 2.8071 / -0.1950 / -0.2199
            group 2 random-like  (NN p_random > 0.5, i.e. the m02 pool-eligible feeds):          0 (no offset; the
                  'random_like' lifts in the json are NOT used: an end-of-feed offset on exploration feeds is harmful,
                  labels_loss A3 -0.0011 holdout z -3.7; the GBDT proxy arm 'tprod' also left them out),
        p_soft_f = the EDA posterior p_random of eda/labels_position/out/train_prandom.npz with NaN -> 0 and WITHOUT the
            NN's M<50 -> 0 rule, so the ~214K exploration feeds hidden among the short feeds get ~0 offset.
        Never applied to the K9 aux (a separate head, model.aux_out), to eval / selection-holdout / predict (they call
        the model directly in common.py and never see the offset), to val or test. Holdout feeds (never trained on) get 0.
        Leakage: uses only train-row production positions + train labels (the lift table) of production train feeds.
  --trunc_lifts PATH ('' = off; only with --trunc_off > 0; arms trunc2 / truncp2 in lane.sh)   TRUNC-PROD with PER-M-BUCKET
        lifts: the same offset a * L[g_f, rkend_i] * (1 - p_soft_f), but g_f = the M bucket of a production feed
        (buckets and lifts from PATH, e.g. nn_x/w5/trunc_lifts_mbucket.json written by review/make_trunc_lifts.py:
        [1,16] [17,32] [33,49] [50,inf), (1-p_soft)-weighted lifts of production train feeds, holdout + pool excluded)
        instead of the two groups M<50 / M>=50 of trunc_lift.json. Random-like (NN p_random > 0.5, M >= 50) and holdout
        feeds: 0, as before. Off (the default) = the code above, unchanged (TruncOffset takes the original branch).
  --aux_norm feed (default 'batch' = off), --aux_norm_c C (1.695)   AUXF (labels_loss E1 = REVIEW_SYNTHESIS P8). The K9
        aux of the live recipe is a per-batch weighted MEAN, aux_w * sum_f (1-p_f) CE_aux_f / sum_f (1-p_f); AUXF makes it
        a per-FEED sum like the main loss: aux_w * C * sum_f (1-p_f) CE_aux_f / max_feeds. p_f = the NN p_random (M<50 ->
        0, as the live fw_aux). C = 1.695 keeps the epoch-mean aux weight of the live recipe (labels_loss
        out/10_aux_norm_const.txt; re-derived exactly from SplitData.batches in tests/test_w5.py C3). train.py logs this
        run's realised epoch-mean multiplier (C * mean_b sum(1-p)/max_feeds over its regular batches; ~1.00).
  --cliplog N (0 = off)   CLIPLOG (labels_loss E3; diagnostic only): every step records the PRE-clip gradient norm (the
        return value of clip_grad_norm_) into a GPU buffer (no sync); every N steps the main-loss and aux-loss gradient
        norms are computed with torch.autograd.grad(retain_graph=True) (no .grad written, no RNG drawn) and stored with
        the batch's Mpad. Summary at every eval; logs/<v>_cliplog.npz at the end. Decision rule (INSIGHTS item 1):
        clipping is closed as an issue iff < 20% of plateau steps (warmup <= step < t_cool) are clipped (norm > 1).
  Hygiene (INSIGHTS item 8; helpers below, flags in train.py):
    --save_last         keep the LAST checkpoint as well as the selected one: ckpt/<v>@last.pt, preds/nn_<v>@last_trainonly_val.npy,
                        preds/holdout/nn_<v>@last_holdout.npy (and preds/nn_<v>@last_trainonly_test.npy unless --no_test).
    --save_holdout      write preds/holdout/nn_<v>_holdout.npy of the selected checkpoint from the training process
                        (nn_x/r7 port; identical numerics to holdout_eval.py, which stays the fallback).
    --select_min_step K checkpoint selection only among evals at step >= K (judge at k >= 12k: CD / 2-epoch runs use 12000).
    CDSEED              (lane.sh, not a flag) every wave-5 CD run of root r uses --seed 900 + r for EVERY arm, including the
                        new m02 reference arm 'm02ref' (INSIGHTS item 8: one data order per root, shared by the arms).
"""
import hashlib
import json
import time

import numpy as np
import torch

ROOT = "<repo>"
ART = f"{ROOT}/nn_x/insights/data_usage/out/nn_artifacts"
RKEND_PATH = f"{ART}/train_rkend_true.npy"
LIFT_PATH = f"{ART}/trunc_lift.json"
MB_LIFT_PATH = f"{ROOT}/nn_x/w5/trunc_lifts_mbucket.json"  # --trunc_lifts table of the trunc2 / truncp2 arms (lane.sh)
PRANDOM_PATH = f"{ROOT}/eda/labels_position/out/train_prandom.npz"
CAND_POS_PATH = f"{ROOT}/data/flat/train/cand_pos.npy"
AUXF_C = 1.695  # labels_loss out/10_aux_norm_const.txt (c = regular batches per epoch / sum_b sum_f (1-p_f) / 512)
POOL_P = 0.5  # NN pool / production threshold on the NN p_random (nn_x/r6 POOL_P_DEFAULT)
GROUPS = ("prod_M<50", "prod_M>=50")


# ------------------------------------------------------------------------------------------------ TRUNC-PROD
def load_lifts(path=LIFT_PATH):
    """L [3, 4] float64: rows prod_M<50 / prod_M>=50 / random-like (zeros), columns rkend 0 / 1 / 2 / >=3 (zero)."""
    j = json.load(open(path))
    L = np.zeros((3, 4))
    for g, nm in enumerate(GROUPS):
        L[g, :3] = [j[nm][f"rkend{k}"] for k in range(3)]
    return L


def p_soft_all(path=PRANDOM_PATH, n=None):
    """EDA posterior p_random for EVERY train feed (NaN -> 0, no M<50 rule)."""
    z = np.load(path)
    if n is not None:
        assert len(z["fidx"]) == n and (z["fidx"] == np.arange(n)).all(), "train_prandom.npz not in train feed order"
    return np.nan_to_num(z["p_random"].astype(np.float64), nan=0.0)


def trunc_groups(M, prand_nn):
    """0 = production M<50, 1 = production M>=50 (NN p <= 0.5), 2 = random-like (NN p > 0.5: no offset)."""
    M = np.asarray(M)
    return np.where(M < 50, 0, np.where(prand_nn > POOL_P, 2, 1)).astype(np.int64)


def trunc_coef(a, M, prand_nn, p_soft, L, hold=None):
    """Per-feed offset table coef [n_feeds, 4] float32: coef[f, k] = a * L[g_f, k] * (1 - p_soft_f) (k = min(rkend, 3);
    column 3 and group 2 are 0). Holdout feeds -> 0. Computed in float64, cast once to float32."""
    g = trunc_groups(M, prand_nn)
    coef = a * L[g] * (1.0 - p_soft)[:, None]
    if hold is not None:
        coef[hold] = 0.0
    return coef.astype(np.float32), g


class TruncOffset:
    """Holds rk3 = min(rkend, 3) per train candidate (uint8, device) and the per-feed coef table (float32, device).
    offset(b) -> [B, Mx] float32 offsets of a gathered training batch (0 on padding); depends on b['fidx'], b['ci'],
    b['cmask'] only, so it is the same for the regular split and any history view (hcap) of it."""

    def __init__(self, a_off, tr, prand_nn, hold, log=print, rkend_path=RKEND_PATH, lift_path=LIFT_PATH,
                 prandom_path=PRANDOM_PATH, mb_path=""):
        t0 = time.time()
        rk = np.load(rkend_path, mmap_mode="r")
        assert rk.shape == (tr.n_cand,), f"rkend file has {rk.shape}, train split has {tr.n_cand} candidates"
        rk = np.asarray(rk)
        nz = int((rk == 0).sum())
        assert nz == tr.n and int(rk.min()) >= 0 and int(rk.max()) < int(tr.M_np.max()), \
            f"rkend sanity: {nz} zeros for {tr.n} feeds, range [{rk.min()}, {rk.max()}]"
        # cheap alignment probe (the full check is tests/test_w5.py C1): in 2000 random feeds rkend == #candidates of the
        # feed shown later (+ the later flat rows among equal shown positions: the builder's tie rule; 15 train feeds
        # have duplicated positions)
        cp = np.load(CAND_POS_PATH, mmap_mode="r")
        assert cp.shape == rk.shape, "cand_pos / rkend length mismatch"
        bad = 0
        for f in np.random.default_rng(0).choice(tr.n, min(2000, tr.n), replace=False):
            o, m = int(tr.cand_off_np[f]), int(tr.M_np[f])
            pf = np.asarray(cp[o:o + m]).astype(np.int64)
            eq = np.triu(pf[None, :] == pf[:, None], k=1).sum(1)
            bad += int(not np.array_equal(rk[o:o + m], (pf[None, :] > pf[:, None]).sum(1) + eq))
        assert bad == 0, f"rkend misaligned with the train candidate order in {bad}/2000 probe feeds"
        rk3 = np.minimum(rk, 3).astype(np.uint8)
        del rk
        if mb_path:  # --trunc_lifts (default '' = off: the original code below runs unchanged)
            self._init_mbucket(a_off, tr, prand_nn, hold, log, rk3, mb_path, prandom_path, t0)
            return
        self.L = load_lifts(lift_path)
        p_soft = p_soft_all(prandom_path, tr.n)
        coef, g = trunc_coef(a_off, tr.M_np, prand_nn, p_soft, self.L, hold)
        dev = tr.device
        self.rk3 = torch.from_numpy(rk3).to(dev)
        self.coef = torch.from_numpy(coef).to(dev)
        self.a = a_off
        w = (1.0 - p_soft)
        gh = g.copy()
        if hold is not None:
            gh[hold] = 3
        msg = "; ".join(f"{nm} {int((gh == k).sum())} feeds, mean (1-p_soft) {w[gh == k].mean():.3f}"
                        for k, nm in enumerate(("prod M<50", "prod M>=50", "random-like (0)", "holdout (0)")) if (gh == k).any())
        log(f"wave-5 TRUNC-PROD: a {a_off}; lifts prod M<50 {np.round(self.L[0, :3], 4).tolist()} / prod M>=50 "
            f"{np.round(self.L[1, :3], 4).tolist()} (rkend 0/1/2; random-like 0); {msg}; non-zero (feed, rkend) offsets "
            f"{int(np.count_nonzero(coef[:, :3]))}; main listwise logits of training batches only; {time.time() - t0:.0f}s")

    def _init_mbucket(self, a_off, tr, prand_nn, hold, log, rk3, mb_path, prandom_path, t0):
        """--trunc_lifts: per-M-bucket production lifts (same rk3 / coef layout, so offset() is shared)."""
        self.lo, self.L = load_lifts_mbucket(mb_path)
        p_soft = p_soft_all(prandom_path, tr.n)
        coef, g = trunc_coef_mb(a_off, tr.M_np, prand_nn, p_soft, self.lo, self.L, hold)
        dev = tr.device
        self.rk3 = torch.from_numpy(rk3).to(dev)
        self.coef = torch.from_numpy(coef).to(dev)
        self.a = a_off
        nb = len(self.lo)
        w = 1.0 - p_soft
        gh = g.copy()
        if hold is not None:
            gh[hold] = nb + 1
        names = [f"prod {x}" for x in bucket_names(self.lo)] + ["random-like (0)", "holdout (0)"]
        msg = "; ".join(f"{nm} {int((gh == k).sum())} feeds, mean (1-p_soft) {w[gh == k].mean():.3f}"
                        for k, nm in enumerate(names) if (gh == k).any())
        lifts = " / ".join(f"{nm} {np.round(self.L[i, :3], 4).tolist()}" for i, nm in enumerate(bucket_names(self.lo)))
        md5 = hashlib.md5(open(mb_path, "rb").read()).hexdigest()
        log(f"wave-5 TRUNC-PROD: a {a_off}; M-bucket lifts (--trunc_lifts {mb_path}, md5 {md5}): {lifts} (rkend 0/1/2; "
            f"random-like 0); {msg}; non-zero (feed, rkend) offsets {int(np.count_nonzero(coef[:, :3]))}; main listwise "
            f"logits of training batches only; {time.time() - t0:.0f}s")

    def offset(self, b):
        k = self.rk3[b["ci"]].long()  # [B, Mx]
        return torch.gather(self.coef[b["fidx"]], 1, k) * b["cmask"]


# ------------------------------------------------------------------------------------------------ TRUNC-PROD M-bucket lifts
def load_lifts_mbucket(path):
    """--trunc_lifts: (lo [nb] int64, L [nb + 1, 4] float64) from a review/make_trunc_lifts.py json. Row b = production
    feeds with lo[b] <= M < lo[b + 1] (last bucket open-ended) with the json's applied lifts rkend0/1/2; row nb =
    random-like (zeros); column 3 (rkend >= 3) zero. Buckets must start at M = 1 and be contiguous."""
    bk = json.load(open(path))["buckets"]
    lo = np.array([int(b["lo"]) for b in bk], np.int64)
    hi = [b["hi"] for b in bk]
    assert len(bk) >= 1 and lo[0] == 1 and hi[-1] is None and all(
        hi[i] is not None and int(hi[i]) >= lo[i] and lo[i + 1] == int(hi[i]) + 1 for i in range(len(bk) - 1)), \
        f"--trunc_lifts {path}: buckets must be contiguous from M=1 with an open last bucket, got {list(zip(lo, hi))}"
    L = np.zeros((len(bk) + 1, 4))
    for i, b in enumerate(bk):
        L[i, :3] = [float(b[f"rkend{k}"]) for k in range(3)]
    assert np.isfinite(L).all(), f"--trunc_lifts {path}: non-finite lift"
    return lo, L


def trunc_groups_mb(M, prand_nn, lo):
    """M bucket b (production feeds, by the trunc_groups rule: M<50 always, M>=50 iff NN p <= 0.5) or len(lo) (random-like:
    NN p > 0.5 and M >= 50, no offset)."""
    M = np.asarray(M)
    b = np.searchsorted(lo, M, side="right") - 1
    assert (b >= 0).all(), "M below the first bucket"
    return np.where(M < 50, b, np.where(prand_nn > POOL_P, len(lo), b)).astype(np.int64)


def trunc_coef_mb(a, M, prand_nn, p_soft, lo, L, hold=None):
    """trunc_coef with the M-bucket groups: coef[f, k] = a * L[bucket_f, k] * (1 - p_soft_f); random-like / holdout /
    column 3 zero. float64, cast once to float32."""
    g = trunc_groups_mb(M, prand_nn, lo)
    coef = a * L[g] * (1.0 - p_soft)[:, None]
    if hold is not None:
        coef[hold] = 0.0
    return coef.astype(np.float32), g


def bucket_names(lo):
    return [f"M{lo[i]}-{lo[i + 1] - 1}" for i in range(len(lo) - 1)] + [f"M>={lo[-1]}"]


# ------------------------------------------------------------------------------------------------ AUXF
def auxf_multiplier(batches, prand_nn, c=AUXF_C, max_feeds=512):
    """Realised epoch-mean aux multiplier of AUXF over a list of regular batches: mean_b c * sum_{f in b} (1-p_f) / max_feeds.
    The live recipe has multiplier 1 on every batch (weighted mean); AUXF keeps its epoch mean at ~1."""
    s = np.array([(1.0 - prand_nn[fb]).sum() for fb in batches])
    return float(c * s.mean() / max_feeds), float(len(batches) / (s.sum() / max_feeds))


# ------------------------------------------------------------------------------------------------ CLIPLOG
class ClipLog:
    """Pre-clip gradient norm of every step (GPU buffer, no sync) + main / aux gradient-norm split every `every` steps."""

    def __init__(self, total, every, warmup, t_cool, dev):
        self.every, self.warmup, self.t_cool, self.total = every, warmup, t_cool, total
        self.gn = torch.full((total,), float("nan"), device=dev)
        self.pool = np.zeros(total, bool)
        self.mpad = np.zeros(total, np.int16)
        self.split_rows = []  # (step, mpad, is_pool, main_norm, aux_norm)
        self.main = None

    def split(self, model, loss, step, b, is_pool):
        params = [p for p in model.parameters() if p.requires_grad]
        out = []
        for part in (self.main, loss - self.main):
            gs = torch.autograd.grad(part, params, retain_graph=True, allow_unused=True)
            out.append(float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None))))
        self.split_rows.append((step, int(b["cmask"].shape[1]), bool(is_pool), out[0], out[1]))

    def record(self, step, gn, b, is_pool):
        self.gn[step] = gn.detach().float()
        self.main = None
        self.pool[step] = is_pool
        self.mpad[step] = b["cmask"].shape[1]

    def _phase_stats(self, upto):
        g = self.gn[:upto].cpu().numpy()
        s = np.arange(upto)
        out = {}
        for nm, m in (("warmup", s < self.warmup), ("plateau", (s >= self.warmup) & (s < self.t_cool)),
                      ("cooldown", s >= self.t_cool), ("pool", self.pool[:upto])):
            v = g[m & np.isfinite(g)]
            if len(v):
                out[nm] = dict(n=int(len(v)), clipped_share=float((v > 1.0).mean()), median=float(np.median(v)),
                               p90=float(np.quantile(v, 0.9)), max=float(v.max()))
        return out

    def summary(self, upto):
        st = self._phase_stats(upto)
        s = "; ".join(f"{k} n {v['n']} clipped {v['clipped_share']:.3f} median {v['median']:.3f} p90 {v['p90']:.3f}"
                      for k, v in st.items())
        r = [x for x in self.split_rows if x[0] < upto][-20:]
        if r:
            ratio = np.median([x[4] / max(x[3], 1e-12) for x in r])
            s += f"; aux/main grad-norm ratio (median of last {len(r)} splits) {ratio:.3f}"
        return f"wave-5 CLIPLOG (pre-clip grad norm, clip 1.0): {s}"

    def save(self, path, upto):
        st = self._phase_stats(upto)
        rows = np.array(self.split_rows, dtype=np.float64).reshape(-1, 5)
        byM = {}
        for mp in np.unique(rows[:, 1]).astype(int):
            m = rows[:, 1] == mp
            byM[int(mp)] = dict(n=int(m.sum()), main=float(np.median(rows[m, 3])), aux=float(np.median(rows[m, 4])),
                                aux_over_main=float(np.median(rows[m, 4] / np.maximum(rows[m, 3], 1e-12))))
        plateau = st.get("plateau", {}).get("clipped_share", float("nan"))
        verdict = ("clipping CLOSED as an issue (< 20% of plateau steps clipped)" if plateau < 0.2 else
                   "clipping ACTIVE (>= 20% of plateau steps clipped): one CD arm with clip 5.0 (labels_loss E3)"
                   if plateau >= 0.2 else "no plateau steps")
        np.savez(path, gn=self.gn[:upto].cpu().numpy(), is_pool=self.pool[:upto], mpad=self.mpad[:upto], split=rows,
                 summary=np.array(json.dumps(dict(phases=st, by_mpad=byM, verdict=verdict))))
        return st, byM, verdict


# ------------------------------------------------------------------------------------------------ hygiene
def setup_wave5(a, tr, feeds, prand, hold, all_batches, t_cool, total, log):
    """(TruncOffset or None, ClipLog or None) for train.py; logs the AUXF setting. Called only when a wave-5 arm flag is on."""
    trunc = cliplog = None
    prand_nn = prand.cpu().numpy().astype(np.float64) if prand is not None else np.zeros(tr.n)
    mb_path = getattr(a, "trunc_lifts", "") or ""  # --trunc_lifts ('' = off; older train.py copies have no such flag)
    if mb_path:
        assert a.trunc_off, "--trunc_lifts only replaces the TRUNC-PROD lift table: it needs --trunc_off > 0"
    if a.trunc_off:
        assert a.loss == "softmax", "--trunc_off acts on the listwise softmax loss only"
        assert hold is not None and prand is not None, "--trunc_off needs --holdout / p_random (production groups)"
        assert not (a.aug_cand or a.aug_hist), "--trunc_off is not combined with the wave-2 pool augmentation"
        trunc = TruncOffset(a.trunc_off, tr, prand_nn, hold, log, mb_path=mb_path)
    if a.aux_norm != "batch":
        assert a.aux_norm == "feed" and a.aux_pos > 0 and a.aux_mode == "listwise", "--aux_norm feed: K9 listwise aux only"
        reg = all_batches[:min(len(all_batches), max(1, len(tr.batches(feeds, a.cand_budget, a.max_feeds, shuffle=False))))]
        mult, c_run = auxf_multiplier(reg, prand_nn, a.aux_norm_c, a.max_feeds)
        log(f"wave-5 AUXF: K9 aux = aux_w(step) * {a.aux_norm_c} * sum_f (1-p_f) CE_aux_f / {a.max_feeds} (was / sum_f "
            f"(1-p_f)); this run's first-epoch regular batches: epoch-mean aux multiplier {mult:.4f} (exact constant for "
            f"them {c_run:.4f}); pool batches get their production mass")
    if a.cliplog:
        assert a.cliplog > 0
        cliplog = ClipLog(total, a.cliplog, a.warmup, t_cool, tr.device)
        log(f"wave-5 CLIPLOG: pre-clip grad norm every step, main/aux split every {a.cliplog} steps (autograd.grad, "
            f"no effect on training)")
    return trunc, cliplog


def holdout_preds(predict, model, tr, hold, hold_rows):
    """Exactly holdout_eval.py: predict(model, tr, hold)[rows] without TF32 (restored afterwards)."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return predict(model, tr, hold)[hold_rows].astype(np.float32)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def write_holdout(name, sh, tr, hold, hold_rows, preds_dir, ndcg):
    hd = preds_dir / "holdout"
    hd.mkdir(exist_ok=True)
    np.save(hd / f"nn_{name}_holdout.npy", sh)
    M_h = tr.M_np[hold]
    meta = np.stack([M_h, np.concatenate([[0], np.cumsum(M_h)[:-1]])])
    if not (hd / "holdout_meta.npy").exists():
        np.save(hd / "holdout_meta.npy", meta)
        np.save(hd / "holdout_y.npy", tr.target_np[hold_rows])
    assert (np.load(hd / "holdout_meta.npy") == meta).all(), "holdout meta mismatch"
    return ndcg(sh, tr.target_np[hold_rows], meta[1], M_h)


def save_holdout(a, model, tr, hold, hold_rows, preds_dir, predict, ndcg, log):
    assert hold is not None and a.holdout == 30000, "--save_holdout needs --holdout 30000 (the fixed holdout)"
    sh = holdout_preds(predict, model, tr, hold, hold_rows)
    nd = write_holdout(a.variant, sh, tr, hold, hold_rows, preds_dir, ndcg)
    log(f"wave-5 saved holdout preds of the selected checkpoint {sh.shape} (holdout NDCG {nd:.5f})")


def save_last(a, model, va, tr, hold, hold_rows, steps_done, ckpt_dir, preds_dir, predict, eval_ndcg, ndcg, log):
    """Last-step checkpoint + its val (TF32 as train.py's val predict) and holdout (no TF32) predictions.
    Returns a CPU copy of the last state dict (for the optional last-checkpoint test predictions)."""
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    name = f"{a.variant}@last"
    torch.save({"state_dict": sd, "cfg": model.cfg, "args": vars(a), "step": steps_done}, ckpt_dir / f"{name}.pt")
    pv = predict(model, va)
    np.save(preds_dir / f"nn_{name}_trainonly_val.npy", pv.astype(np.float32))
    msg = f"val_ndcg {eval_ndcg(pv, va):.5f}"
    if hold is not None and a.holdout == 30000:
        sh = holdout_preds(predict, model, tr, hold, hold_rows)
        msg += f" holdout_ndcg {write_holdout(name, sh, tr, hold, hold_rows, preds_dir, ndcg):.5f}"
    log(f"wave-5 LAST checkpoint (step {steps_done}) saved as {name}: {msg}")
    return sd


def save_last_test(a, model, last_sd, te, preds_dir, predict, log):
    cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(last_sd)
    pt = predict(model, te)
    np.save(preds_dir / f"nn_{a.variant}@last_trainonly_test.npy", pt.astype(np.float32))
    model.load_state_dict(cur)
    log(f"wave-5 saved last-checkpoint test preds {pt.shape}")
