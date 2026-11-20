"""Fine-tune a train-only checkpoint on val with 5-fold GroupKFold (by user_id).

FULLVAL copy (nn_x/w12/stale/fulldata, 2026-11-15; reports/STALE_1115/fulldata.md section 5): nn_x/abl/popfree/finetune.py (frozen copy
ref/finetune_abl.py.txt, md5 in ref/md5.txt) + ONE flag, OFF by default and guarded so that a flag-off run executes exactly the
nn_x/abl/popfree/finetune.py code path (test_ft_full.py: AST proof + textual diff). Helper ft_full.py:
  --full_val   every model is fine-tuned on ALL val feeds (one fit set, `--seeds` models, torch / numpy seeds 0..seeds-1);
               a.folds is forced to 1. NO out-of-fold prediction exists: the val predictions are IN-SAMPLE and are written as
               preds/nn_<out>_valINSAMPLE.npy (never as ..._val_oof.npy); every "OOF" figure of the log is in-sample and is
               announced as such; <out>_ft.json gets insample_* keys instead of oof_* keys. Test predictions
               (nn_<out>_test.npy = mean of the `--seeds` full-val models, final EMA weights) as before.
               Run with PYTHONPATH=<code dir>:  cd <repo>/nn_x/abl/popfree && PYTHONPATH=<repo>/nn_x/abl python <repo>/nn_x/w12/stale/fulldata/finetune_full_abl.py --variant <v> --out <v>_n2cpv3fv15 ... --full_val --seeds 15
Previous docstring:

nn_x/abl/popfree copy (2026-11-13 ~04:00 UTC, pop-free diversity family, pre-declared CURRENT_STATUS 03:50 UTC) of
nn_x/w5/finetune.py (= nn/finetune.py, md5 1c74e4dc...). Differences, all outside the training loop:
  * sys.path puts nn_x/abl FIRST, so common / data / blend / model are the nn_x/abl harness modules (model.py is
    byte-identical to nn_x/w5's; data.py / common.py differ only in the feat_cache symlink and the output root):
    checkpoints are read from nn_x/abl/ckpt, predictions go to nn_x/abl/preds, logs to nn_x/abl/logs.
  * honour_xt_exclude (default --xt_check on): nn_x/abl/train.py stores its args (incl. --xt / --xt_exclude) and the
    model cfg (xt_names AFTER the exclusion, xt_rel / xt_idcols / xt_relonly indices into that list, n_dense) in the
    checkpoint. The expected list and indices are recomputed from the args with the very helper train.py used
    (abl_arms.xt_exclude_names) and the run is refused on any mismatch; attach_xt / Ranker(**cfg) then use exactly
    that reduced xt input set (the w5 code path below is unchanged). --xt_check off = the w5 code path verbatim.
  * pop1 extension (2026-11-13 ~07:15 UTC, lead pop1 10-epoch family pre-declared 07:10): honour_xt_exclude also
    honours the checkpoint's --xt_add (nn_x/abl train.py: columns appended LAST after --xt_exclude, with the same helper
    abl_arms.xt_add_names and the same `reserved` = XT_REL + XT_RELONLY of the preset). A checkpoint without --xt_add
    (no key or '') takes exactly the previous path and log line.
Previous nn/finetune.py docstring:

Val is unbiased (random display order) and has the test distribution, so a short low-LR fine-tune adapts the
model. For every fold we fine-tune for `--epochs` epochs, recording held-out-fold and test predictions after
each epoch. The epoch count is chosen by the OOF NDCG over all val feeds (one global choice for all folds),
and the test prediction is the mean of the 5 fold models at that epoch.

Example:
  CUDA_VISIBLE_DEVICES=N python nn/finetune.py --variant base --epochs 8 --lr 1e-4
Outputs:
  preds/nn_<variant>_val_oof.npy, preds/nn_<variant>_test.npy, nn/logs/<variant>_ft.json
"""
import argparse
import copy
import math
import sys
import time

sys.path.insert(0, "<repo>/nn_x/abl")  # nn_x/abl/popfree: the nn_x/abl harness modules (ckpt / preds / logs there)

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from common import CKPT, LOGS, PREDS, ROOT, Logger, eval_ndcg, predict, save_json
from data import SplitData, feat_exists
from blend import feed_z
from ft_full import FullVal  # FULLVAL --full_val (nn_x/w12/stale/fulldata; OFF by default)
from model import N_DENSE, Ranker, bce_loss, dense_features, listwise_loss
from abl_arms import xt_exclude_names, xt_rel_extend  # nn_x/abl/popfree: the helpers train.py used for --xt_exclude
from abl_arms import xt_add_names  # nn_x/abl/pop1: the helper train.py used for --xt_add
from train import XT_PRESETS, XT_REL, XT_RELONLY  # nn_x/abl/popfree: the preset tables train.py built xt_names from


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="base", help="checkpoint name in nn/ckpt (train-only model)")
    p.add_argument("--out", default=None, help="output variant name (default: same as --variant)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.02)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--cand_budget", type=int, default=8192)
    p.add_argument("--max_feeds", type=int, default=128)
    p.add_argument("--loss", default="softmax", choices=["softmax", "bce", "both"])
    p.add_argument("--loss_norm", default="batch", choices=["batch", "feed", "idcg"],
                   help="listwise loss normalisation (K1): per-batch mean / equal per-feed weight / x K/IDCG(K)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", type=int, default=1, help="fine-tune seeds per fold (averaged)")
    p.add_argument("--drop", type=float, default=None, help="override dropout during fine-tune")
    p.add_argument("--from_scratch", action="store_true", help="ignore checkpoint weights (val-only baseline)")
    p.add_argument("--extra", default="", help="comma list of data/feats names, or 'item' preset, injected via a "
                                                   "zero-init branch (val/test only features)")
    p.add_argument("--extra_lr_mult", type=float, default=10.0)
    p.add_argument("--add_random", type=int, default=0,
                   help="mix this many random-order (exploration) train feeds into each fine-tune epoch")
    p.add_argument("--pseudo", default="", help="N6: teacher run name; uses preds/nn_<teacher>_test_folds.npy "
                                                "(per-fold test predictions, so fold k's teacher never saw val fold k)")
    p.add_argument("--w_test", type=float, default=0.3, help="loss weight of the pseudo-labelled test batches")
    p.add_argument("--pseudo_n", type=int, default=12000, help="test feeds sampled per fine-tune epoch")
    p.add_argument("--pseudo_T", type=float, default=1.0, help="temperature of softmax(teacher z) soft targets")
    p.add_argument("--decay", default="const", choices=["const", "cos"],
                   help="lr after warmup: constant (default) or cosine to 0 over all epochs (nnx fta recipe)")
    p.add_argument("--ema", type=float, default=0.0,
                   help=">0: weight EMA with this beta; the saved OOF/test predictions come from the final EMA weights")
    p.add_argument("--ema_start", type=float, default=0.5, help="EMA start, in epochs")
    p.add_argument("--u0_singleton", action="store_true",
                   help="GroupKFold: every user-0 (unknown user) feed is its own group instead of one pooled group")
    p.add_argument("--select", default="snapshot", choices=["snapshot", "best"],
                   help="saved predictions: snapshot mean over epochs 1..E (default) or best OOF epoch")
    p.add_argument("--xt_check", default="on", choices=["on", "off"],
                   help="nn_x/abl/popfree: verify the checkpoint cfg against its saved --xt / --xt_exclude args "
                        "(honour_xt_exclude); off = the nn_x/w5 finetune.py code path verbatim")
    p.add_argument("--full_val", action="store_true",
                   help="FULLVAL (ft_full.py): fine-tune every model on ALL val feeds (--seeds models, no folds, no OOF); val "
                        "predictions are IN-SAMPLE and written as ..._valINSAMPLE.npy")
    return p.parse_args()


ITEM_PRESET = ["it_cand_cnt_split", "it_cand_cnt_split_rk", "it_cand_cnt_split_z", "it_cand_cnt_split_gap",
               "it_cand_cnt_train", "it_hist_all", "it_hist_all_rk", "it_hist_all_z", "it_hist_all_gap",
               *[f"it_hist_et{i}" for i in range(1, 11)], "it_tr_coec", "it_tr_coec_rk", "it_tr_coec_z",
               "it_tr_coec_gap", "it_tr_ctr", "it_tr_pos", "u_tr_ctr", "u_tr_feeds", "hour", "centrality"]

# raw within-split candidate counts have a different scale in test (4 days) than in val (3 days): keep only
# their within-feed relative versions
ITEM_SAFE = [n for n in ITEM_PRESET if n not in ("it_cand_cnt_split", "it_cand_cnt_split_gap")]
# leave-same-user-out popularity replaces it_hist_* (which include the same user's later snapshots)
ITEM_LUO = [n for n in ITEM_SAFE if not n.startswith("it_hist_")] + [
    "it_hist_all_luo", "it_hist_all_luo_rk", "it_hist_all_luo_z", "it_hist_all_luo_gap",
    *[f"it_hist_et{i}_luo" for i in range(1, 11)]]
ALS_PRESET = ["als_exp2", "als_exp5", "als_exp10", "als_last10", "als_et1_exp5", "als_samecat_exp10", "als_has",
              "als_exp5_rk", "als_exp5_z"]
COVIS_PRESET = ["seq_cv_invd20", "seq_cv_invd20_rk", "seq_cv_invd20_z", "seq_cv_w5", "seq_cv_fwd5", "seq_cv_t1",
                "seq_cv_seqs", "seq_cv_pop", "seq_cv_norm"]
SEQ_PRESET = ["seq_max_all", "seq_top3", "seq_dcos_hl5", "seq_nd90_cnt", "seq_argmax_hpos"]
CTX_PRESET = ["ctx_nd_c80", "ctx_nd_c90", "ctx_nd_c95", "ctx_nd_dupb_f1", "ctx_nd_dupb_h", "ctx_nd_dupb_u",
              "ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev", "ctx_nd_knn_u", "ctx_nd_knn_u_dev",
              "ctx_nd_max", "ctx_nd_mean", "ctx_nd_top5h", "ctx_nd_top5u", "ctx_nd_ucent"]
# audit after the later-history finding: drop in-split candidate counts (they include the same user's later feeds)
# and keep only train-period, own-history, causal or leave-same-user-out sources
ITEM_CLEAN = [n for n in ITEM_LUO if not n.startswith("it_cand_cnt_split")]
TTE_PRESET = ["tte_c_ctr", "tte_it_coec", "tte_it_ctr", "tte_it_imp", "tte_it_pos", "tte_u_coec", "tte_u_ctr",
              "tte_u_imp", "tte_uc_ctr", "tte_uc_imp", "tte_ui_imp", "tte_ui_pos"]


def honour_xt_exclude(ck, log):
    """nn_x/abl/popfree: the fine-tuned model must use exactly the checkpoint's reduced xt input set.
    nn_x/abl/train.py saves its args (incl. --xt / --xt_exclude / --xt_rel_extra / --user0) and the model cfg (xt_names
    after the exclusion, xt_rel / xt_idcols / xt_relonly as indices into that list, n_dense) in the checkpoint
    (torch.save({"state_dict", "cfg", "args"})). Recompute the expected list and indices from the args with the helpers
    train.py used and refuse any mismatch, so attach_xt(cfg xt_names) / Ranker(**cfg) below are provably the trained
    inputs. Returns the expected xt_names (None = no xt)."""
    args, cfg = ck["args"], ck["cfg"]
    xt, excl, rel_extra = args.get("xt", ""), args.get("xt_exclude", ""), args.get("xt_rel_extra", "")
    xadd = args.get("xt_add", "") or ""  # nn_x/abl pop1 (absent in checkpoints written before the flag existed)
    if args.get("init"):  # --init runs take their inputs from the init checkpoint cfg (train.py refuses the flags there)
        assert not excl and not rel_extra and not xadd
        log(f"abl xt check: --init run ({args['init']}): inputs fixed by the checkpoint cfg, "
            f"{len(cfg.get('xt_names') or [])} xt columns, n_dense {cfg['n_dense']}")
        return cfg.get("xt_names")
    quiet = lambda *s: None
    exp = (XT_PRESETS[xt] if xt in XT_PRESETS else [n for n in xt.split(",") if n]) or None
    if excl:
        exp = xt_exclude_names(exp, excl, quiet)
    if xadd:  # pop1: exactly train.py's hook (after --xt_exclude, before the XT_REL / XT_RELONLY / idcols lookups)
        exp = xt_add_names(exp, xadd, quiet, reserved=XT_REL.get(xt, []) + XT_RELONLY.get(xt, []))
    exp = exp or None
    got_names = cfg.get("xt_names") or None
    assert got_names == exp, f"cfg xt_names {got_names} != --xt {xt!r} minus --xt_exclude {excl!r} plus --xt_add {xadd!r} = {exp}"
    if exp:
        rel = [exp.index(n) for n in XT_REL.get(xt, []) if n in exp] or None
        if rel_extra:
            rel = xt_rel_extend(rel, exp, rel_extra, quiet)
        idc = [i for i, n in enumerate(exp) if n.startswith(("seq_cv", "it_hist"))] or None
        ronly = [exp.index(n) for n in XT_RELONLY.get(xt, []) if n in exp] or None
    else:
        rel = idc = ronly = None
    n_dense = (N_DENSE + 2 if args.get("user0") else N_DENSE) + len(exp or []) + 2 * len(rel or []) - len(ronly or [])
    got = (cfg.get("xt_rel"), cfg.get("xt_idcols"), cfg.get("xt_relonly"), cfg["n_dense"])
    assert got == (rel, idc, ronly, n_dense), \
        f"cfg (xt_rel, xt_idcols, xt_relonly, n_dense) {got} != recomputed {(rel, idc, ronly, n_dense)}"
    dropped = [n for n in excl.split(",") if n]
    log(f"abl xt_exclude honoured: --xt {xt!r} --xt_exclude {len(dropped)} column(s) {dropped} -> {len(exp or [])} xt "
        f"inputs, n_dense {n_dense} (xt_rel {rel}, xt_idcols {idc}, xt_relonly {ronly}); attach_xt / Ranker use the cfg"
        + (f"; abl xt_add honoured: {[n for n in xadd.split(',') if n]} appended last" if xadd else ""))
    return exp


def build_model(ck, dev, drop=None, from_scratch=False, n_extra=0):
    cfg = dict(ck["cfg"])
    if drop is not None:
        cfg["drop"] = drop
    m = Ranker(**cfg)
    if n_extra:
        m.add_extra(n_extra)
    m = m.to(dev)
    sd = ck["state_dict"]
    if from_scratch:
        m.set_norm(sd["f_mean"], sd["f_std"])
    else:
        m.load_state_dict(sd, strict=not n_extra)
    if m.id_buckets:  # N4: item-ID table frozen during fine-tune, no ID-dropout
        m.id_emb.weight.requires_grad_(False)
        m.id_drop = 0.0
    if m.domain:  # K2: val/test = exploration domain, initialised from the random-like embedding
        with torch.no_grad():
            m.dom.weight[2] = m.dom.weight[1]
        m.infer_domain = 2
    return m


def main():
    a = parse()
    out = a.out or a.variant
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    log = Logger(LOGS / f"{out}_ft.log")
    log(f"==== {time.ctime()} args: {vars(a)}")
    fv = None  # FULLVAL (ft_full.py): stays None unless --full_val
    if a.full_val:  # FULLVAL: one fit set = ALL val feeds, --seeds models, a.folds forced to 1 (checks inside)
        fv = FullVal(a, out, log)
    ck = torch.load(CKPT / f"{a.variant}.pt", map_location=dev, weights_only=False)
    xt_excluded = []
    if a.xt_check == "on":  # nn_x/abl/popfree: refuse a checkpoint whose cfg does not match its saved --xt / --xt_exclude
        honour_xt_exclude(ck, log)
        xt_excluded = [n for n in ck["args"].get("xt_exclude", "").split(",") if n]
    va = SplitData("val", dev, max_hist=ck["args"]["max_hist"])
    te = SplitData("test", dev, max_hist=ck["args"]["max_hist"])
    for sd in (va, te):
        if ck["cfg"].get("xt_names"):
            sd.attach_xt(ck["cfg"]["xt_names"])
        if ck["cfg"].get("id_buckets"):
            sd.load_item_ids()
    trd, rand_pool = None, None
    if a.add_random > 0:
        trd = SplitData("train", dev, max_hist=ck["args"]["max_hist"])
        if ck["cfg"].get("xt_names"):
            trd.attach_xt(ck["cfg"]["xt_names"])
        if ck["cfg"].get("id_buckets"):
            trd.load_item_ids()
        pr = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"], nan=0.0)
        rand_pool = np.where((pr > 0.5) & (trd.M_np >= 50))[0]
        prm = np.where(trd.M_np >= 50, pr, 0.0).astype(np.float32)
        pt = torch.from_numpy(prm).to(dev)
        trd.dom = torch.stack([1 - pt, pt, torch.zeros_like(pt)], 1)
        log(f"mixing {a.add_random} of {len(rand_pool)} random-order train feeds into every fine-tune epoch")
    presets = {"item": ITEM_PRESET, "item_tte": ITEM_PRESET + TTE_PRESET, "item_safe": ITEM_SAFE,
               "item_luo": ITEM_LUO, "n2": ITEM_LUO + ALS_PRESET + COVIS_PRESET,
               "n2full": ITEM_LUO + ALS_PRESET + COVIS_PRESET + SEQ_PRESET + CTX_PRESET,
               "n2clean": ITEM_CLEAN + COVIS_PRESET + SEQ_PRESET + CTX_PRESET,
               "n2clean_als": ITEM_CLEAN + ALS_PRESET + COVIS_PRESET + SEQ_PRESET + CTX_PRESET}
    PRE = ["pre_fe", "pre_fe_z", "pre_new", "pre_new_z", "pre_hmax32", "pre_hmax8", "pre_fe_noid", "pre_id_contrib",
           "pre_idmax32"]  # leak-safe pretrained sequence encoder outputs (val/test only)
    presets["n2ca_pre"] = presets["n2clean_als"] + PRE
    presets["n2c_pre"] = presets["n2clean"] + PRE
    # W3 hygiene: fixed-parity id contribution, no train-period statistics (it_tr_*/u_tr_*/it_cand_cnt_train) or hour
    presets["n2c_pre_v2"] = [n for n in presets["n2c_pre"] if n != "hour" and not n.startswith(
        ("it_tr_", "u_tr_", "it_cand_cnt_train"))]
    presets["n2c_pre_v2"] = [("pre_id_contrib_v2" if n == "pre_id_contrib" else n) for n in presets["n2c_pre_v2"]]
    # W3b bundle on top of n2c_pre_v2: prune the 14 names that no longer exist on disk, explicit presence flag for
    # pre_idmax32, pre_hmax32/8 -2 sentinel -> NaN + one flag (identical rows for both), drop seq_cv_norm
    # (rho 0.9998 with seq_cv_invd20), signed log1p of it_hist_all_luo_z (derived files: nn/build_w3b_feats.py)
    PHANTOM = {"seq_cv_invd20_z", "seq_cv_w5", "seq_cv_fwd5", "seq_cv_t1", "seq_cv_seqs", "seq_dcos_hl5", "seq_nd90_cnt",
               "ctx_nd_c80", "ctx_nd_c90", "ctx_nd_c95", "ctx_nd_dupb_f1", "ctx_nd_dupb_h", "ctx_nd_dupb_u", "ctx_nd_mean"}
    REN = {"pre_hmax32": "pre_hmax32_c", "pre_hmax8": "pre_hmax8_c", "it_hist_all_luo_z": "it_hist_all_luo_zsl"}
    presets["n2c_pre_v3"] = [REN.get(n, n) for n in presets["n2c_pre_v2"] if n not in PHANTOM and n != "seq_cv_norm"] + [
        "pre_idmax32_has", "pre_hmax32_sent"]
    # nn_x/abl/popfree (pre-declared amendments CURRENT_STATUS 04:33 + 04:44 UTC 2026-11-13): n2c_pre_v3 minus the 7 raw
    # popularity columns that the pop-free family drops from its xt inputs AND their 10 derived channels (rank / signed-log z /
    # gap, the et5-et10 counts, seq_cv_invd20_rk), so the nopop fine-tune branch is popularity-free end to end: 42 -> 25
    NOPOP = ["it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo",
             "seq_cv_invd20", "seq_cv_pop",  # the 7 raw columns
             "it_hist_all_luo_rk", "it_hist_all_luo_zsl", "it_hist_all_luo_gap", *[f"it_hist_et{i}_luo" for i in range(5, 11)],
             "seq_cv_invd20_rk"]  # the 10 derived channels (amendment 2)
    presets["n2c_pre_v3_nopop"] = [n for n in presets["n2c_pre_v3"] if n not in NOPOP]
    assert len(NOPOP) == 17 and len(presets["n2c_pre_v3"]) == 42 and len(presets["n2c_pre_v3_nopop"]) == 25, \
        (len(NOPOP), len(presets["n2c_pre_v3"]), len(presets["n2c_pre_v3_nopop"]))
    assert not any(n.startswith(("it_hist", "seq_cv")) for n in presets["n2c_pre_v3_nopop"])
    # group ablations of n2clean_als (R2 step 1)
    full = presets["n2clean_als"]
    groups = {"als": set(ALS_PRESET), "nbr": set(CTX_PRESET) | {"centrality"},
              "trstat": {n for n in full if n.startswith(("it_tr_", "u_tr_", "it_cand_cnt_train"))},
              "cv": {n for n in full if n.startswith(("it_hist_", "seq_cv_"))}}
    for g, members in groups.items():
        presets[f"n2ca_no{g}"] = [n for n in full if n not in members]
        presets[f"n2ca_only{g}"] = [n for n in full if n in members]
    names = presets[a.extra] if a.extra in presets else [n for n in a.extra.split(",") if n and n != "none"]
    miss = [n for n in names if not all(feat_exists(sp, n) for sp in ("val", "test"))]
    if miss:  # feature dirs get pruned by other agents: drop what no longer exists, and say so
        log(f"extra features missing on disk, dropped: {miss}")
        names = [n for n in names if n not in miss]
    if names:
        st = va.attach_extra(names)
        te.attach_extra(names, st)
        log(f"extra features ({len(names)}): {names}")
    if a.extra == "n2c_pre_v3_nopop":  # nn_x/abl/popfree: the final nopop list and its length, next to the generic line above
        log(f"abl nopop preset: n2c_pre_v3 ({len(presets['n2c_pre_v3'])} names) minus {NOPOP} -> {len(names)} extra inputs "
            f"(final list): {names}")
    if xt_excluded:  # nn_x/abl/popfree, information only: the pre-declared n2cpv3 recipe's val/test-only extra branch
        back = [n for n in names if n in xt_excluded]
        log(f"abl popfree note: {len(back)} of the {len(xt_excluded)} excluded xt columns re-enter through the fine-tune "
            f"extra branch (val/test-only zero-init inputs, pre-declared recipe): {back}; the train-only member and the "
            f"xt inputs stay pop-free")
    E = a.epochs
    oof = np.zeros((E + 1, va.n_cand), np.float32)
    test = np.zeros((E + 1, te.n_cand), np.float32)
    test_folds = np.zeros((a.folds, te.n_cand), np.float32)  # per-fold snapshot test predictions (teachers for N6)
    oof_fin = np.zeros(va.n_cand, np.float32)  # final EMA weights (with --ema)
    test_fin = np.zeros(te.n_cand, np.float32)
    test_fin_folds = np.zeros((a.folds, te.n_cand), np.float32)
    teacher = None
    if a.pseudo:
        teacher = np.load(PREDS / f"nn_{a.pseudo}_test_folds.npy").astype(np.float32)
        log(f"pseudo-labels from {a.pseudo}: w_test {a.w_test}, {a.pseudo_n} test feeds/epoch, T {a.pseudo_T}")
    gkf = GroupKFold(n_splits=a.folds) if fv is None else None
    t0 = time.time()
    fold_curves = []
    fold_groups = va.user_id.astype(np.int64)
    if a.u0_singleton:  # user 0 pools unrelated anonymous feeds: do not force them all into one fold
        u0 = fold_groups == 0
        fold_groups = np.where(u0, -1 - np.arange(va.n), fold_groups)
        log(f"user-0 feeds as singleton groups: {int(u0.sum())}")
    for k, (tr_f, ho_f) in enumerate(gkf.split(np.arange(va.n), groups=fold_groups) if fv is None else fv.splits(va.n)):
        curve = []
        for s in range(a.seeds):
            torch.manual_seed(1000 * k + s)
            rng = np.random.default_rng(1000 * k + s)
            model = build_model(ck, dev, a.drop, a.from_scratch, len(names))
            if s == 0:  # epoch 0 = the starting (train-only) model
                oof[0] += predict(model, va, ho_f)
                test[0] += predict(model, te) / a.folds
            ex = [p for n, p in model.named_parameters() if n.startswith("extra.")]
            base = [p for n, p in model.named_parameters() if not n.startswith("extra.")]
            groups = [{"params": base, "mult": 1.0}] + ([{"params": ex, "mult": a.extra_lr_mult}] if ex else [])
            groups = [dict(g, params=[p for p in g["params"] if p.requires_grad]) for g in groups]
            opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.wd)
            step = 0
            S_total, ema, ema_params, ema_step0 = None, None, None, None
            params = list(model.parameters())
            tz = None if teacher is None else torch.from_numpy(teacher[k]).to(dev)  # fold-safe teacher scores
            for ep in range(1, E + 1):
                model.train()
                jobs = [(va, fb) for fb in va.batches(tr_f, a.cand_budget, a.max_feeds, shuffle=True, rng=rng)]
                if tz is not None:
                    smp = rng.choice(te.n, min(a.pseudo_n, te.n), replace=False)
                    jobs += [(te, fb) for fb in te.batches(smp, a.cand_budget, a.max_feeds, shuffle=True, rng=rng)]
                if trd is not None:  # mix in a fresh sample of random-order train feeds every epoch
                    smp = rng.choice(rand_pool, min(a.add_random, len(rand_pool)), replace=False)
                    jobs += [(trd, fb) for fb in trd.batches(smp, a.cand_budget, a.max_feeds, shuffle=True, rng=rng)]
                jobs = [jobs[i] for i in rng.permutation(len(jobs))]
                if S_total is None:
                    S_total = len(jobs) * E
                    ema_step0 = int(a.ema_start * len(jobs))
                for src, fb in jobs:
                    lr_now = a.lr * min(1.0, (step + 1) / a.warmup)
                    if a.decay == "cos":
                        lr_now *= 0.5 * (1 + math.cos(math.pi * min(step, S_total) / S_total))
                    for g in opt.param_groups:
                        g["lr"] = lr_now * g["mult"]
                    b = src.gather(fb)  # train batches carry no "xe" -> the extra branch is skipped for them
                    X = model.dense(b)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg = model(b, X, use_pos=False)
                    m = b["cmask"]
                    if src is te:  # soft pseudo-label target softmax(z_teacher / T) over the feed
                        y = torch.softmax((tz[b["ci"]] / a.pseudo_T).masked_fill(~m, -1e4), 1) * m
                        loss = a.w_test * listwise_loss(lg, y, m)
                    elif a.loss == "softmax":
                        y = b["y"]
                        loss = listwise_loss(lg, y, m, norm=a.loss_norm, denom=a.max_feeds)
                    elif a.loss == "bce":
                        loss = bce_loss(lg, b["y"], m)
                    else:
                        loss = listwise_loss(lg, b["y"], m) + bce_loss(lg, b["y"], m)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    if a.ema > 0 and step >= ema_step0:
                        with torch.no_grad():
                            if ema is None:  # start: copy of the current weights
                                for attr in ("aux_out", "intent_logp"):  # non-leaf forward caches block deepcopy
                                    if hasattr(model, attr):
                                        delattr(model, attr)
                                ema = copy.deepcopy(model).float()
                                ema_params = list(ema.parameters())
                            else:  # theta_bar <- beta * theta_bar + (1 - beta) * theta
                                torch._foreach_lerp_(ema_params, params, 1.0 - a.ema)
                    step += 1
                p = predict(model, va, ho_f)
                oof[ep] += p / a.seeds
                pt = predict(model, te)
                test[ep] += pt / (a.folds * a.seeds)
                test_folds[k] += feed_z(pt, te.M_np) / (E * a.seeds)
                if s == 0:
                    curve.append(eval_ndcg(p, va, ho_f))
            if ema is not None:  # final EMA weights -> the saved predictions
                oof_fin += predict(ema, va, ho_f) / a.seeds
                pt = predict(ema, te)
                test_fin += pt / (a.folds * a.seeds)
                test_fin_folds[k] += feed_z(pt, te.M_np) / a.seeds
        fold_curves.append([eval_ndcg(oof[0], va, ho_f)] + curve)
        log(f"fold {k}: ndcg by epoch (seed0) {np.round(fold_curves[-1], 4).tolist()} {(time.time() - t0) / 60:.1f} min")
    if fv is not None:  # FULLVAL: every 'OOF' figure below is IN-SAMPLE (all val feeds were trained on)
        fv.warn(log)
    scores = [eval_ndcg(oof[e], va) for e in range(E + 1)]
    best_e = int(np.argmax(scores))
    log(f"OOF ndcg by epoch: {np.round(scores, 5).tolist()}")
    log(f"best epoch {best_e}: OOF ndcg {scores[best_e]:.5f} (epoch0 = train-only {scores[0]:.5f})")
    # snapshot ensemble: mean of per-feed z-scored predictions over epochs 1..E (no epoch selection on OOF)
    snap_v = np.mean([feed_z(oof[e], va.M_np) for e in range(1, E + 1)], 0).astype(np.float32)
    snap_t = np.mean([feed_z(test[e], te.M_np) for e in range(1, E + 1)], 0).astype(np.float32)
    snap = eval_ndcg(snap_v, va)
    log(f"snapshot mean of epochs 1..{E}: OOF ndcg {snap:.5f}")
    fin = None
    if a.ema > 0:
        fin = eval_ndcg(oof_fin, va)
        log(f"final EMA weights: OOF ndcg {fin:.5f}")
        np.save(PREDS / (f"nn_{out}_val_oof.npy" if fv is None else fv.val_name), oof_fin)
        np.save(PREDS / f"nn_{out}_test.npy", test_fin)
        np.save(PREDS / f"nn_{out}_test_folds.npy", test_fin_folds.astype(np.float16))
    elif a.select == "snapshot":
        np.save(PREDS / (f"nn_{out}_val_oof.npy" if fv is None else fv.val_name), snap_v)
        np.save(PREDS / f"nn_{out}_test.npy", snap_t)
        np.save(PREDS / f"nn_{out}_test_folds.npy", test_folds.astype(np.float16))
    else:
        np.save(PREDS / (f"nn_{out}_val_oof.npy" if fv is None else fv.val_name), oof[best_e])
        np.save(PREDS / f"nn_{out}_test.npy", test[best_e])
    save_json(LOGS / f"{out}_ft.json", dict(args=vars(a), oof_by_epoch=scores, best_epoch=best_e,
                                            oof_ndcg_best_epoch=scores[best_e], oof_ndcg_snapshot=snap,
                                            saved="ema_final" if a.ema > 0 else a.select,
                                            oof_ndcg_ema_final=fin, fold_curves=fold_curves,
                                            minutes=(time.time() - t0) / 60))
    if fv is not None:  # FULLVAL: rewrite the summary with insample_* keys, check that no ..._val_oof.npy was written
        fv.finish(LOGS, PREDS, log)


if __name__ == "__main__":
    main()
