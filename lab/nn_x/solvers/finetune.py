raise SystemExit("disabled in nn_x/solvers copy: hard-codes shared output paths")
"""Fine-tune a train-only checkpoint on val with 5-fold GroupKFold (by user_id).

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
import time

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from common import CKPT, LOGS, PREDS, ROOT, Logger, eval_ndcg, predict, save_json
from data import SplitData, feat_exists
from blend import feed_z
from model import Ranker, bce_loss, dense_features, listwise_loss


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
    ck = torch.load(CKPT / f"{a.variant}.pt", map_location=dev, weights_only=False)
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
    gkf = GroupKFold(n_splits=a.folds)
    t0 = time.time()
    fold_curves = []
    fold_groups = va.user_id.astype(np.int64)
    if a.u0_singleton:  # user 0 pools unrelated anonymous feeds: do not force them all into one fold
        u0 = fold_groups == 0
        fold_groups = np.where(u0, -1 - np.arange(va.n), fold_groups)
        log(f"user-0 feeds as singleton groups: {int(u0.sum())}")
    for k, (tr_f, ho_f) in enumerate(gkf.split(np.arange(va.n), groups=fold_groups)):
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
        np.save(PREDS / f"nn_{out}_val_oof.npy", oof_fin)
        np.save(PREDS / f"nn_{out}_test.npy", test_fin)
        np.save(PREDS / f"nn_{out}_test_folds.npy", test_fin_folds.astype(np.float16))
    elif a.select == "snapshot":
        np.save(PREDS / f"nn_{out}_val_oof.npy", snap_v)
        np.save(PREDS / f"nn_{out}_test.npy", snap_t)
        np.save(PREDS / f"nn_{out}_test_folds.npy", test_folds.astype(np.float16))
    else:
        np.save(PREDS / f"nn_{out}_val_oof.npy", oof[best_e])
        np.save(PREDS / f"nn_{out}_test.npy", test[best_e])
    save_json(LOGS / f"{out}_ft.json", dict(args=vars(a), oof_by_epoch=scores, best_epoch=best_e,
                                            oof_ndcg_best_epoch=scores[best_e], oof_ndcg_snapshot=snap,
                                            saved="ema_final" if a.ema > 0 else a.select,
                                            oof_ndcg_ema_final=fin, fold_curves=fold_curves,
                                            minutes=(time.time() - t0) / 60))


if __name__ == "__main__":
    main()
