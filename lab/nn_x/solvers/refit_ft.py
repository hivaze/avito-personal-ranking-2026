"""R5 refit (review_ideas #8): fine-tune on a given val feed subset and predict another (or test).

Training loop = nn/finetune.py (n2cpv3 recipe: extra n2c_pre_v3, AdamW lr 5e-5, wd 0.02, warmup 30, cos decay,
EMA 0.99 from 0.5 epochs, 4 epochs, final EMA weights), copied line by line minus the unused pseudo/add_random
branches and the per-epoch predictions (predict() is RNG-free, so skipping it does not change training).

Modes (checkpoints are read-only from nn/ckpt; outputs go to nn_x/solvers/preds/refit/):
  check  : original 5-fold split + original seeds (1000*k) for the folds in --check_folds; saves held-out preds
  nested : outer GroupKFold(5); per outer fold k: inner 5-fold fine-tunes on the other 80% (mean on fold k) and one
           refit on all of the other 80% (pred on fold k)
  full   : one fine-tune on all val feeds (seed 5000), predicts test -> nn_<v>_n2cpv3full_test.npy

  CUDA_VISIBLE_DEVICES=N python nn_x/solvers/refit_ft.py --mode nested --variants v15_cd_all_e10,v14_s93
"""
import argparse
import copy
import math
import time

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from common import LOGS, PREDS, ROOT, Logger, eval_ndcg, predict
from data import SplitData, feat_exists
from model import Ranker, listwise_loss

SRC_CKPT = ROOT / "nn" / "ckpt"
OUT = PREDS / "refit"
OUT.mkdir(parents=True, exist_ok=True)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["check", "nested", "full"])
    p.add_argument("--variants", required=True, help="comma list of checkpoint names in nn/ckpt")
    p.add_argument("--check_folds", default="0")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--wd", type=float, default=0.02)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--cand_budget", type=int, default=8192)
    p.add_argument("--max_feeds", type=int, default=128)
    p.add_argument("--extra_lr_mult", type=float, default=10.0)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--ema_start", type=float, default=0.5)
    p.add_argument("--extra", default="n2c_pre_v3")
    p.add_argument("--full_seed", type=int, default=5000)
    p.add_argument("--tag", default="n2cpv3full")
    return p.parse_args()


# ---- extra-feature presets: verbatim from nn/finetune.py (only the chain needed for n2c_pre_v3) ----
ITEM_PRESET = ["it_cand_cnt_split", "it_cand_cnt_split_rk", "it_cand_cnt_split_z", "it_cand_cnt_split_gap",
               "it_cand_cnt_train", "it_hist_all", "it_hist_all_rk", "it_hist_all_z", "it_hist_all_gap",
               *[f"it_hist_et{i}" for i in range(1, 11)], "it_tr_coec", "it_tr_coec_rk", "it_tr_coec_z",
               "it_tr_coec_gap", "it_tr_ctr", "it_tr_pos", "u_tr_ctr", "u_tr_feeds", "hour", "centrality"]
ITEM_SAFE = [n for n in ITEM_PRESET if n not in ("it_cand_cnt_split", "it_cand_cnt_split_gap")]
ITEM_LUO = [n for n in ITEM_SAFE if not n.startswith("it_hist_")] + [
    "it_hist_all_luo", "it_hist_all_luo_rk", "it_hist_all_luo_z", "it_hist_all_luo_gap",
    *[f"it_hist_et{i}_luo" for i in range(1, 11)]]
COVIS_PRESET = ["seq_cv_invd20", "seq_cv_invd20_rk", "seq_cv_invd20_z", "seq_cv_w5", "seq_cv_fwd5", "seq_cv_t1",
                "seq_cv_seqs", "seq_cv_pop", "seq_cv_norm"]
SEQ_PRESET = ["seq_max_all", "seq_top3", "seq_dcos_hl5", "seq_nd90_cnt", "seq_argmax_hpos"]
CTX_PRESET = ["ctx_nd_c80", "ctx_nd_c90", "ctx_nd_c95", "ctx_nd_dupb_f1", "ctx_nd_dupb_h", "ctx_nd_dupb_u",
              "ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev", "ctx_nd_knn_u", "ctx_nd_knn_u_dev",
              "ctx_nd_max", "ctx_nd_mean", "ctx_nd_top5h", "ctx_nd_top5u", "ctx_nd_ucent"]
ITEM_CLEAN = [n for n in ITEM_LUO if not n.startswith("it_cand_cnt_split")]


def preset(name):
    n2clean = ITEM_CLEAN + COVIS_PRESET + SEQ_PRESET + CTX_PRESET
    PRE = ["pre_fe", "pre_fe_z", "pre_new", "pre_new_z", "pre_hmax32", "pre_hmax8", "pre_fe_noid", "pre_id_contrib",
           "pre_idmax32"]
    n2c_pre = n2clean + PRE
    v2 = [n for n in n2c_pre if n != "hour" and not n.startswith(("it_tr_", "u_tr_", "it_cand_cnt_train"))]
    v2 = [("pre_id_contrib_v2" if n == "pre_id_contrib" else n) for n in v2]
    PHANTOM = {"seq_cv_invd20_z", "seq_cv_w5", "seq_cv_fwd5", "seq_cv_t1", "seq_cv_seqs", "seq_dcos_hl5", "seq_nd90_cnt",
               "ctx_nd_c80", "ctx_nd_c90", "ctx_nd_c95", "ctx_nd_dupb_f1", "ctx_nd_dupb_h", "ctx_nd_dupb_u", "ctx_nd_mean"}
    REN = {"pre_hmax32": "pre_hmax32_c", "pre_hmax8": "pre_hmax8_c", "it_hist_all_luo_z": "it_hist_all_luo_zsl"}
    v3 = [REN.get(n, n) for n in v2 if n not in PHANTOM and n != "seq_cv_norm"] + ["pre_idmax32_has", "pre_hmax32_sent"]
    return {"n2c_pre_v3": v3}[name]


def build_model(ck, dev, n_extra):  # nn/finetune.py build_model with drop=None, from_scratch=False
    cfg = dict(ck["cfg"])
    m = Ranker(**cfg)
    if n_extra:
        m.add_extra(n_extra)
    m = m.to(dev)
    m.load_state_dict(ck["state_dict"], strict=not n_extra)
    if m.id_buckets:
        m.id_emb.weight.requires_grad_(False)
        m.id_drop = 0.0
    if m.domain:
        with torch.no_grad():
            m.dom.weight[2] = m.dom.weight[1]
        m.infer_domain = 2
    return m


def fine_tune(a, ck, dev, va, n_extra, tr_f, seed):
    """One fine-tune on val feeds tr_f; returns the final EMA model (identical loop to nn/finetune.py)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(ck, dev, n_extra)
    E = a.epochs
    ex = [p for n, p in model.named_parameters() if n.startswith("extra.")]
    base = [p for n, p in model.named_parameters() if not n.startswith("extra.")]
    groups = [{"params": base, "mult": 1.0}] + ([{"params": ex, "mult": a.extra_lr_mult}] if ex else [])
    groups = [dict(g, params=[p for p in g["params"] if p.requires_grad]) for g in groups]
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.wd)
    step = 0
    S_total, ema, ema_params, ema_step0 = None, None, None, None
    params = list(model.parameters())
    for ep in range(1, E + 1):
        model.train()
        jobs = [(va, fb) for fb in va.batches(tr_f, a.cand_budget, a.max_feeds, shuffle=True, rng=rng)]
        jobs = [jobs[i] for i in rng.permutation(len(jobs))]
        if S_total is None:
            S_total = len(jobs) * E
            ema_step0 = int(a.ema_start * len(jobs))
        for src, fb in jobs:
            lr_now = a.lr * min(1.0, (step + 1) / a.warmup)
            lr_now *= 0.5 * (1 + math.cos(math.pi * min(step, S_total) / S_total))
            for g in opt.param_groups:
                g["lr"] = lr_now * g["mult"]
            b = src.gather(fb)
            X = model.dense(b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(b, X, use_pos=False)
            m = b["cmask"]
            loss = listwise_loss(lg, b["y"], m, norm="batch", denom=a.max_feeds)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if a.ema > 0 and step >= ema_step0:
                with torch.no_grad():
                    if ema is None:
                        for attr in ("aux_out", "intent_logp"):
                            if hasattr(model, attr):
                                delattr(model, attr)
                        ema = copy.deepcopy(model).float()
                        ema_params = list(ema.parameters())
                    else:
                        torch._foreach_lerp_(ema_params, params, 1.0 - a.ema)
            step += 1
    return ema, step


def groups_u0(user_id):
    g = user_id.astype(np.int64)
    u0 = g == 0
    return np.where(u0, -1 - np.arange(len(g)), g)


def main():
    a = parse()
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    log = Logger(LOGS / f"r5_refit_{a.mode}.log")
    log(f"==== {time.ctime()} args: {vars(a)}")
    names = preset(a.extra)
    variants = [v for v in a.variants.split(",") if v]
    cache = {}
    for v in variants:
        t0 = time.time()
        ck = torch.load(SRC_CKPT / f"{v}.pt", map_location=dev, weights_only=False)
        mh = ck["args"]["max_hist"]
        key = (mh, tuple(ck["cfg"].get("xt_names") or ()), bool(ck["cfg"].get("id_buckets")))
        if key not in cache:  # reload splits only when the input spec changes
            cache.clear()
            torch.cuda.empty_cache()
            va = SplitData("val", dev, max_hist=mh)
            te = SplitData("test", dev, max_hist=mh) if a.mode == "full" else None
            for sd in [s for s in (va, te) if s is not None]:
                if ck["cfg"].get("xt_names"):
                    sd.attach_xt(ck["cfg"]["xt_names"])
                if ck["cfg"].get("id_buckets"):
                    sd.load_item_ids()
            miss = [n for n in names if not all(feat_exists(sp, n) for sp in ("val", "test"))]
            assert not miss, f"missing extra features: {miss}"
            st = va.attach_extra(names)
            if te is not None:
                te.attach_extra(names, st)
            cache[key] = (va, te)
        va, te = cache[key]
        n_extra = len(names)
        grp = groups_u0(va.user_id)
        allf = np.arange(va.n)
        if a.mode == "check":
            ref = np.load(ROOT / "preds" / f"nn_{v}_n2cpv3_val_oof.npy")
            folds = list(GroupKFold(n_splits=5).split(allf, groups=grp))
            for k in [int(x) for x in a.check_folds.split(",")]:
                tr_f, ho_f = folds[k]
                ema, steps = fine_tune(a, ck, dev, va, n_extra, tr_f, 1000 * k)
                p = predict(ema, va, ho_f)
                rows = np.concatenate([np.arange(va.cand_off_np[f], va.cand_off_np[f] + va.M_np[f]) for f in ho_f])
                c = np.corrcoef(p[rows], ref[rows])[0, 1]
                log(f"{v} check fold {k}: steps {steps} ndcg new {eval_ndcg(p, va, ho_f):.5f} "
                    f"ref {eval_ndcg(ref, va, ho_f):.5f} corr {c:.6f} maxabs {np.abs(p[rows] - ref[rows]).max():.4f} "
                    f"{(time.time() - t0) / 60:.1f} min")
                np.save(OUT / f"check_{v}_fold{k}.npy", p)
        elif a.mode == "nested":
            P_i = np.zeros(va.n_cand, np.float32)
            P_ii = np.zeros(va.n_cand, np.float32)
            outer = np.full(va.n, -1, np.int64)
            info = []
            for k, (R, T) in enumerate(GroupKFold(n_splits=5).split(allf, groups=grp)):
                outer[T] = k
                inner = list(GroupKFold(n_splits=5).split(R, groups=grp[R]))
                acc = np.zeros(va.n_cand, np.float32)
                st_i = []
                for j, (itr, _iho) in enumerate(inner):
                    ema, steps = fine_tune(a, ck, dev, va, n_extra, R[itr], 1000 * j + 10 * k + 1)
                    acc += predict(ema, va, T) / 5
                    st_i.append(steps)
                    del ema
                ema, steps_r = fine_tune(a, ck, dev, va, n_extra, R, 9000 + 10 * k + 1)
                pr = predict(ema, va, T)
                del ema
                P_i += acc  # rows outside T are 0 in both
                P_ii += pr
                n_i, n_ii = eval_ndcg(acc, va, T), eval_ndcg(pr, va, T)
                n_iii = eval_ndcg(0.5 * acc + 0.5 * pr, va, T)
                info.append(dict(k=k, n_T=len(T), n_R=len(R), steps_inner=st_i, steps_refit=steps_r,
                                 ndcg_i=n_i, ndcg_ii=n_ii, ndcg_iii=n_iii))
                log(f"{v} outer {k}: |R| {len(R)} |T| {len(T)} steps inner {st_i} refit {steps_r} | "
                    f"(i) {n_i:.5f} (ii) {n_ii:.5f} (iii) {n_iii:.5f} {(time.time() - t0) / 60:.1f} min")
            assert (outer >= 0).all()
            np.savez(OUT / f"nested_{v}.npz", P_i=P_i, P_ii=P_ii, outer=outer, info=np.array(str(info)))
            log(f"{v} nested ALL: (i) {eval_ndcg(P_i, va):.5f} (ii) {eval_ndcg(P_ii, va):.5f} "
                f"(iii) {eval_ndcg(0.5 * P_i + 0.5 * P_ii, va):.5f} {(time.time() - t0) / 60:.1f} min")
        else:
            ema, steps = fine_tune(a, ck, dev, va, n_extra, allf, a.full_seed)
            pt = predict(ema, te)
            pv = predict(ema, va)  # in-sample (information only)
            del ema
            np.save(OUT / f"nn_{v}_{a.tag}_test.npy", pt)
            np.save(OUT / f"nn_{v}_{a.tag}_val_insample.npy", pv)
            log(f"{v} full refit: steps {steps}, in-sample val ndcg {eval_ndcg(pv, va):.5f} "
                f"{(time.time() - t0) / 60:.1f} min")
        del ck
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
