"""nn_x/w5 WAVE-5 hzcd arm: popularity-parity cooldown (reports/INSIGHTS_1112.md section 3.1 item 6; AUDIT_1112 section 3.2
row 1; pre-flight nn_x/w5/hzcd/preflight.txt, PASS 2026-11-12), used by nn_x/w5/train.py. OFF at the train.py default
(--hzcd absent): train.py builds the hzcd view only behind `if a.hzcd` and reads it only through ONE conditional expression
that reduces to the previous gather when the view is None, so a flag-off run executes exactly the previous nn_x/w5 code path
(tests/test_hzcd.py: AST proof against tests/ref_pre_hzcd/train.py.txt + end-to-end batch / logit / log / checkpoint
identity; tests/test_w5.py and tests/test_trunc_lifts.py keep their own proofs by stripping the hzcd statements first,
tests/strip_hzcd.py). Imports only numpy/torch (no GPU guard), so the unit tests call the very functions train.py uses.

  --hzcd   From the cooldown start (step t_cool; step 0 in the CD harness: --cooldown_frac 1.0 right after --init) every
        TRAINING batch (regular and pool) is gathered through HzTrainView: SplitData.gather of the train split, then the
        LUO popularity columns of b['xt'] (it_hist_all_luo, it_hist_et1_luo, it_hist_et2_luo, it_hist_et3_luo,
        it_hist_et4_luo = xt columns 3-7 of the presets cvraw / cvnd / cvnd_pre2s; taken from the model's xt_names, so
        --init checkpoints are covered) are overwritten by the horizon-limited counts data/feats_new/train/
        it_hist_hz_{all,et1,et2,et3,et4}.npy (feat_new/it_hist_hz.py, reports/featnew_it_hist_hz.md: for a listing created
        at or after the feed's lagged cutoff the occurrences are counted over the val/test snapshots plus a 1/37 sparse sample
        of the train snapshots, i.e. the coverage regime of val/test rows; older listings keep the as-built LUO count; the
        val/test hz files are byte-identical to the luo files). Pre-flight (hzcd/preflight.txt): train rows move toward
        val's joint age x coverage distribution (3-7 d zero rate 0.13 -> 0.35 vs val 0.62; 12-cell JS 0.073 -> 0.027),
        16.7% of train rows change for the 'all' column; 10-24 d rows are unchanged by construction (lag <= 8 d).
        Scaling: IDENTICAL to the luo columns' own path data.SplitData.attach_xt (xt_transform below reproduces its
        branches; for names starting with it_hist: log1p(max(x, 0)), NaN -> 0, stored float16, gathered as xt[ci].float()).
        Downstream the hz values then pass through exactly what the luo values pass through: model.dense_features
        (ID-input dropout xt_drop, which is 0 in F7; the in-model within-feed pct-rank + z of it_hist_all_luo via xt_rel)
        and the dense normalisation f_mean / f_std (fitted on luo batches at model build, or loaded with --init) -- the
        hz counts live on the same log1p-count scale, so the same normalisation is the intended parity.
        NEVER applied to: feature_stats (model build), the selection holdout (eval_subset reads tr), val / test predict,
        save_last / save_holdout / holdout_eval.py (all read tr, the untouched split, or va / te).
        Extra output with --hzcd (INSIGHTS item 6 'score the holdout both as built and with hz inputs'): the selected
        checkpoint's holdout preds scored THROUGH the hz view, preds/holdout/nn_<v>_holdout_hz.npy (same numerics as the
        wave-5 hygiene holdout_preds: common.predict without TF32); the as-built file nn_<v>_holdout.npy is written as before.
        Holdout feeds have hz lag 0 (feat_new/hz_common.py), so their hz rows differ from luo only for same-day listings.
        lane.sh arms: B:hzcd (= B:m02ref + --hzcd, roots 101-107) and E:hzcd (= E:base + --hzcd, seeds 951-954).
"""
import time

import numpy as np
import torch

ROOT = "<repo>"
HZ_DIR = f"{ROOT}/data/feats_new"  # feat_new/it_hist_hz.py outputs (train / val / test)
LUO_TO_HZ = {"it_hist_all_luo": "it_hist_hz_all", "it_hist_et1_luo": "it_hist_hz_et1", "it_hist_et2_luo": "it_hist_hz_et2",
             "it_hist_et3_luo": "it_hist_hz_et3", "it_hist_et4_luo": "it_hist_hz_et4"}


def hz_path(split, name):
    return f"{HZ_DIR}/{split}/{name}.npy"


def luo_cols(xt_names):
    """[(xt column index, luo name, hz name)] for the LUO popularity columns among the model's xt_names."""
    cols = [(i, n, LUO_TO_HZ[n]) for i, n in enumerate(xt_names or []) if n in LUO_TO_HZ]
    assert cols, f"--hzcd: no LUO popularity column among the xt inputs {xt_names}"
    return cols


def xt_transform(name, x):
    """data.SplitData.attach_xt's fixed, split-independent transform of one feature as float32 numpy (same branches in the
    same order; the caller casts with torch .to(device, float16) exactly as attach_xt does). Names starting with it_hist
    (luo AND hz) take the count branch log1p(max(x, 0)); NaN -> 0."""
    x = np.asarray(x).astype(np.float32)
    if name.startswith("has:") or name.endswith("_rk"):
        pass
    elif name.endswith("_z"):
        x = np.clip(x, -5, 5)
    elif name.startswith(("seq_cv", "it_hist")):  # counts
        x = np.log1p(np.maximum(x, 0))
    else:
        x = np.clip(x, -10, 10)
    return np.nan_to_num(x, nan=0.0)


def load_hz(split, cols, device, rows=None):
    """[n_rows, len(cols)] float16 tensor on `device` of the hz columns of `split` (all rows, or the flat candidate rows
    `rows`), transformed exactly as attach_xt transforms the luo columns."""
    out = []
    for _, _, hz_n in cols:
        x = np.load(hz_path(split, hz_n), mmap_mode="r")
        x = np.asarray(x if rows is None else x[rows])
        out.append(torch.from_numpy(xt_transform(hz_n, x)).to(device, torch.float16))
        del x
    return torch.stack(out, 1)


class HzTrainView:
    """gather() of the train split with the LUO xt columns replaced by the hz columns. Every other attribute / method
    (batches, scatter, n, n_cand, device, M_np, cand_off_np, target_np, ...) is the split's own, delegated, so
    common.predict / eval_subset can score through the view (used only by save_holdout_hz)."""

    def __init__(self, tr, cols, hz):
        assert tr.xt is not None, "HzTrainView needs a split with attached xt features"
        assert hz.shape == (tr.n_cand, len(cols)) and hz.dtype == tr.xt.dtype and hz.device == tr.xt.device, \
            f"hz tensor {tuple(hz.shape)} {hz.dtype} {hz.device} vs xt {tuple(tr.xt.shape)} {tr.xt.dtype} {tr.xt.device}"
        assert all(0 <= c < tr.xt.shape[1] for c, _, _ in cols)
        self.tr, self.cols, self.hz = tr, list(cols), hz
        self.col_idx = [int(c) for c, _, _ in cols]
        self.hzcd = True

    def __getattr__(self, k):  # only reached for names HzTrainView itself does not define
        return getattr(self.tr, k)

    @torch.no_grad()
    def gather(self, fidx_np, with_pos=False):
        b = self.tr.gather(fidx_np, with_pos)
        xt = b["xt"]  # a fresh tensor (SplitData.gather: self.xt[ci].float()), so the in-place column copies are safe
        v = self.hz[b["ci"]].float()  # [B, Mx, len(cols)]; padded positions read the feed's first row, as xt does
        for k, c in enumerate(self.col_idx):
            xt[..., c].copy_(v[..., k])  # strided copy_ (deterministic; no index_put / index_copy)
        return b


@torch.no_grad()
def hz_stats(tr, view):
    """Per column: zero rate luo / hz and the share of changed rows over the whole split (column by column: no large temp)."""
    st = []
    for k, (c, luo_n, hz_n) in enumerate(view.cols):
        a, h = tr.xt[:, c], view.hz[:, k]
        st.append(dict(col=c, luo=luo_n, hz=hz_n, zero_luo=float((a == 0).float().mean()), zero_hz=float((h == 0).float().mean()),
                       changed=float((a != h).float().mean())))
        del a, h
    return st


def setup_hzcd(a, tr, xt_names, hold, t_cool, total, log):
    """HzTrainView of the train split for train.py. Called only behind `if a.hzcd`."""
    assert xt_names, "--hzcd needs xt inputs holding the LUO popularity columns (presets cvraw / cvnd / cvnd_pre2s)"
    assert not getattr(a, "hcap", ""), "--hzcd is not combined with --hcap (both replace the training-batch gather)"
    assert t_cool < total, "--hzcd: the cooldown never starts (t_cool >= total steps)"
    assert hold is not None, "--hzcd needs --holdout (the as-built vs hz holdout read-out)"
    t0 = time.time()
    cols = luo_cols(xt_names)
    hz = load_hz("train", cols, tr.device)
    view = HzTrainView(tr, cols, hz)
    st = hz_stats(tr, view)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    msg = "; ".join(f"col {s['col']} {s['luo']} -> {s['hz']}: zero rate {s['zero_luo']:.3f} -> {s['zero_hz']:.3f}, rows changed "
                    f"{s['changed']:.4f}" for s in st)
    log(f"wave-5 hzcd: from step {t_cool} (of {total}) every TRAINING batch (regular and pool) reads the horizon-limited "
        f"popularity {HZ_DIR}/train/it_hist_hz_*.npy in place of the LUO xt columns, same attach_xt scaling (log1p, fp16); "
        f"{msg}; feature_stats / selection holdout / val / test / save_holdout unchanged (luo); extra file "
        f"preds/holdout/nn_<v>_holdout_hz.npy; hz tensor {hz.numel() * hz.element_size() / 2**30:.2f} GB; {time.time() - t0:.0f}s")
    return view


def save_holdout_hz(a, model, view, hold, hold_rows, preds_dir, predict, ndcg, log):
    """Holdout preds of the selected checkpoint scored THROUGH the hz view (the wave-5 hygiene holdout_preds numerics:
    predict without TF32, restored afterwards), written as preds/holdout/nn_<v>_holdout_hz.npy next to the as-built file."""
    assert view is not None and hold is not None
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        sh = predict(model, view, hold)[hold_rows].astype(np.float32)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    hd = preds_dir / "holdout"
    hd.mkdir(exist_ok=True)
    np.save(hd / f"nn_{a.variant}_holdout_hz.npy", sh)
    M_h = view.M_np[hold]
    offs = np.concatenate([[0], np.cumsum(M_h)[:-1]])
    nd = ndcg(sh, view.target_np[hold_rows], offs, M_h)
    log(f"wave-5 hzcd: holdout preds of the selected checkpoint scored with the hz inputs {sh.shape} -> "
        f"preds/holdout/nn_{a.variant}_holdout_hz.npy (holdout NDCG with hz inputs {nd:.5f}; the as-built holdout file is unchanged)")
    return nd
