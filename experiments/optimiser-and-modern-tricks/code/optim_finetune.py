# Fine-tune on the validation days used by the optimiser group: 5 folds by user, with the switches added for
# this study (--decay cos, --ema, --ema_start, --lpft_epochs, --muon).
# Adopted recipe for checkpoints trained with Muon, from sh/worker.log (lab/archive/nnx/optim, one GPU):
#   NNX_FEATS=.../nnx/optim/feats python finetune.py --variant optim_muwsd_s23 --out fta_optim_muwsd_s23 \
#     --extra $(cat n2clean_frozen.txt) --lr 5e-5 --epochs 4 --decay cos --ema 0.99 --ema_start 0.5
# Produced out-of-fold val predictions and test predictions; out-of-fold val 0.38833 for this run.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/archive/nnx/optim/finetune.py
#
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
from data import FEATS, SplitData
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
    p.add_argument("--select", default="snapshot", choices=["snapshot", "best"],
                   help="saved predictions: snapshot mean over epochs 1..E (default) or best OOF epoch "
                        "(ignored with --ema: then the final EMA weights are saved)")
    # ---- C3 (nnx/optim): fine-tune recipe; defaults = the production recipe
    p.add_argument("--decay", default="const", choices=["const", "cos"],
                   help="lr after warmup: constant (production) or cosine to 0 over the (phase-B) epochs")
    p.add_argument("--ema", type=float, default=0.0, help=">0: weight EMA with this beta; OOF/test predicted once "
                                                          "from the final EMA weights")
    p.add_argument("--ema_start", type=float, default=0.5, help="EMA start, in (phase-B) epochs")
    p.add_argument("--lpft_epochs", type=int, default=0,
                   help="LP-FT phase A: train only extra.*, head.*, wide.* for this many epochs first")
    p.add_argument("--lpft_lr", type=float, default=5e-4, help="phase-A lr (constant after warmup)")
    p.add_argument("--muon", action="store_true", help="C2 closeout: Muon (RMS-matched, same lr) for the hidden 2-D "
                                                       "Linear weights in fine-tune (matches a Muon-pretrained model)")
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


class MultiOpt:
    """Several optimizers stepped together; param_groups is the concatenation (each group keeps its 'mult')."""

    def __init__(self, opts):
        self.opts = opts
        self.param_groups = [g for o in opts for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()


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
    names = presets[a.extra] if a.extra in presets else [n for n in a.extra.split(",") if n]
    miss = [n for n in names if not all((FEATS / sp / f"{n}.npy").exists() for sp in ("val", "test"))]
    if miss:  # feature dirs get pruned by other agents: drop what no longer exists, and say so
        log(f"extra features missing on disk, dropped: {miss}")
        names = [n for n in names if n not in miss]
    if names:
        st = va.attach_extra(names)
        te.attach_extra(names, st)
        log(f"extra features ({len(names)}): {names}")
    E = a.lpft_epochs + a.epochs  # total epochs (phase A + phase B)
    oof = np.zeros((E + 1, va.n_cand), np.float32)
    oof_ema = np.zeros((E + 1, va.n_cand), np.float32)  # per-epoch EMA OOF (diagnostics only)
    oof_fin = np.zeros(va.n_cand, np.float32)  # final EMA weights
    test_fin = np.zeros(te.n_cand, np.float32)
    test_fin_folds = np.zeros((a.folds, te.n_cand), np.float32)
    test = np.zeros((E + 1, te.n_cand), np.float32)
    test_folds = np.zeros((a.folds, te.n_cand), np.float32)  # per-fold snapshot test predictions (teachers for N6)
    teacher = None
    if a.pseudo:
        teacher = np.load(PREDS / f"nn_{a.pseudo}_test_folds.npy").astype(np.float32)
        log(f"pseudo-labels from {a.pseudo}: w_test {a.w_test}, {a.pseudo_n} test feeds/epoch, T {a.pseudo_T}")
    gkf = GroupKFold(n_splits=a.folds)
    t0 = time.time()
    fold_curves = []
    for k, (tr_f, ho_f) in enumerate(gkf.split(np.arange(va.n), groups=va.user_id)):
        curve = []
        for s in range(a.seeds):
            torch.manual_seed(1000 * k + s)
            rng = np.random.default_rng(1000 * k + s)
            model = build_model(ck, dev, a.drop, a.from_scratch, len(names))
            if s == 0:  # epoch 0 = the starting (train-only) model
                oof[0] += predict(model, va, ho_f)
                test[0] += predict(model, te) / a.folds
            trainable0 = {n for n, p in model.named_parameters() if p.requires_grad}

            def make_opt(phase):
                if phase == "A":  # LP-FT: new branch + head only (the backbone is frozen)
                    for n, p in model.named_parameters():
                        p.requires_grad_(n in trainable0 and n.startswith(("extra.", "head.", "wide.")))
                    groups = [{"params": [p for p in model.parameters() if p.requires_grad], "mult": 1.0}]
                    return torch.optim.AdamW(groups, lr=a.lpft_lr, weight_decay=a.wd)
                for n, p in model.named_parameters():
                    p.requires_grad_(n in trainable0)
                ex = [p for n, p in model.named_parameters() if n.startswith("extra.")]
                base = [p for n, p in model.named_parameters() if not n.startswith("extra.")]
                groups = [{"params": base, "mult": 1.0}] + ([{"params": ex, "mult": a.extra_lr_mult}] if ex else [])
                groups = [dict(g, params=[p for p in g["params"] if p.requires_grad]) for g in groups]
                if a.muon:  # hidden 2-D Linear weights -> Muon, the rest -> AdamW, both keep the group lr multipliers
                    mids = {id(mod.weight) for mod in model.modules()
                            if isinstance(mod, torch.nn.Linear) and min(mod.weight.shape) > 1}
                    gm = [dict(g, params=[p for p in g["params"] if id(p) in mids]) for g in groups]
                    ga = [dict(g, params=[p for p in g["params"] if id(p) not in mids]) for g in groups]
                    return MultiOpt([torch.optim.AdamW([g for g in ga if g["params"]], lr=a.lr, weight_decay=a.wd),
                                     torch.optim.Muon([g for g in gm if g["params"]], lr=a.lr, weight_decay=a.wd,
                                                      momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw")])
                return torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.wd)

            phase = "A" if a.lpft_epochs > 0 else "B"
            opt = make_opt(phase)
            step = 0  # step counter within the current phase
            S_B = None  # phase-B total steps (for the cosine decay), set at its first epoch
            ema, ema_params, ema_step0 = None, None, None
            params = [p for p in model.parameters()]
            tz = None if teacher is None else torch.from_numpy(teacher[k]).to(dev)  # fold-safe teacher scores
            for ep in range(1, E + 1):
                model.train()
                if phase == "A" and ep > a.lpft_epochs:
                    phase, step = "B", 0
                    opt = make_opt("B")
                jobs = [(va, fb) for fb in va.batches(tr_f, a.cand_budget, a.max_feeds, shuffle=True, rng=rng)]
                if tz is not None:
                    smp = rng.choice(te.n, min(a.pseudo_n, te.n), replace=False)
                    jobs += [(te, fb) for fb in te.batches(smp, a.cand_budget, a.max_feeds, shuffle=True, rng=rng)]
                if trd is not None:  # mix in a fresh sample of random-order train feeds every epoch
                    smp = rng.choice(rand_pool, min(a.add_random, len(rand_pool)), replace=False)
                    jobs += [(trd, fb) for fb in trd.batches(smp, a.cand_budget, a.max_feeds, shuffle=True, rng=rng)]
                jobs = [jobs[i] for i in rng.permutation(len(jobs))]
                if phase == "B" and S_B is None:
                    S_B = len(jobs) * a.epochs
                    ema_step0 = int(a.ema_start * len(jobs))
                for src, fb in jobs:
                    if phase == "A":
                        lr_now = a.lpft_lr * min(1.0, (step + 1) / a.warmup)
                    else:
                        lr_now = a.lr * min(1.0, (step + 1) / a.warmup)
                        if a.decay == "cos":
                            lr_now *= 0.5 * (1 + math.cos(math.pi * min(step, S_B) / S_B))
                    for g in opt.param_groups:
                        g["lr"] = lr_now * g["mult"]
                    b = src.gather(fb)  # train batches carry no "xe" -> the extra branch is skipped for them
                    X = dense_features(b, model.cfg["n_dense"])
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
                    if a.ema > 0 and phase == "B" and step >= ema_step0:
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
                if ema is not None and s == 0:
                    oof_ema[ep] += predict(ema, va, ho_f)
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
    ema_scores, fin = None, None
    if a.ema > 0:
        ema_scores = [eval_ndcg(oof_ema[e], va) if oof_ema[e].any() else float("nan") for e in range(E + 1)]
        fin = eval_ndcg(oof_fin, va)
        log(f"EMA OOF ndcg by epoch (diagnostic): {np.round(ema_scores, 5).tolist()}")
        log(f"final EMA weights: OOF ndcg {fin:.5f}")
        np.save(PREDS / f"nn_{out}_val_oof.npy", feed_z(oof_fin, va.M_np).astype(np.float32))
        np.save(PREDS / f"nn_{out}_test.npy", feed_z(test_fin, te.M_np).astype(np.float32))
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
                                            saved="ema_final" if a.ema > 0 else a.select, fold_curves=fold_curves,
                                            ema_oof_by_epoch=ema_scores, oof_ndcg_ema_final=fin,
                                            minutes=(time.time() - t0) / 60))


if __name__ == "__main__":
    main()
