"""Train the neural ranker on the (position-biased) train split, select by full NDCG on val.

Example:
  CUDA_VISIBLE_DEVICES=N python nn/train.py --variant base --epochs 1
Outputs:
  nn/ckpt/<variant>.pt                         best (by val NDCG) checkpoint
  preds/nn_<variant>_trainonly_{val,test}.npy  predictions of that checkpoint (flat cand order)
  nn/logs/<variant>.log / .json                training log and summary
"""
import argparse
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CKPT, LOGS, PREDS, ROOT, Logger, eval_ndcg, eval_subset, feature_stats, predict, save_json
from data import SplitData
from model import N_DENSE, Ranker, bce_loss, dense_features, listwise_loss


XT_PRESETS = {"cv": ["seq_cv_invd20", "seq_cv_w5", "seq_cv_fwd5", "seq_cv_t1", "seq_cv_seqs", "seq_cv_pop",
                      "seq_cv_norm", "seq_cv_invd20_rk", "seq_cv_invd20_z", "it_hist_all_luo", "it_hist_all_luo_rk",
                      "it_hist_all_luo_z", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo"]}
# + item cross-feed context (day-cutoff for train rows): feed-size stats, production-position prior, f1 prior
XT_PRESETS["cvix"] = XT_PRESETS["cv"] + ["ctx_ix_mlogM", "ctx_ix_slogM", "ctx_ix_dlogM", "ctx_ix_mrpos", "ctx_ix_mlpos",
                                         "ctx_ix_mf1", "ctx_ix_df1", "ctx_ix_mlogM_rk", "ctx_ix_mrpos_rk", "ctx_ix_mf1_rk",
                                         "has:ctx_ix_mlogM"]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="base")
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--hist_layers", type=int, default=1)
    p.add_argument("--drop", type=float, default=0.1)
    p.add_argument("--no_hist", action="store_true")
    p.add_argument("--no_set", action="store_true")
    p.add_argument("--no_pos", action="store_true", help="disable the position tower")
    p.add_argument("--loss", default="softmax", choices=["softmax", "bce", "both"])
    p.add_argument("--loss_norm", default="batch", choices=["batch", "feed", "idcg"],
                   help="listwise loss normalisation (K1): per-batch mean / equal per-feed weight / x K/IDCG(K)")
    p.add_argument("--max_hist", type=int, default=256)
    p.add_argument("--train_feeds", type=int, default=0, help="subsample train feeds (0 = all)")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.02)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--cand_budget", type=int, default=32768)
    p.add_argument("--max_feeds", type=int, default=512)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_test", action="store_true")
    p.add_argument("--aux_pos", type=float, default=0.0, help="weight of auxiliary production-position regression")
    p.add_argument("--aux_mode", default="mse", choices=["mse", "listwise"],
                   help="aux production-order target: MSE on pos/(M-1) or listwise CE toward softmax(-pos/tau) (K9)")
    p.add_argument("--aux_tau", type=float, default=10.0)
    p.add_argument("--ff", type=int, default=2, help="FFN width multiplier")
    p.add_argument("--sim_bias", type=int, default=0, nargs="?", const=1,
                   help="1: per-head scaled raw cosine bias in target attention; 2: + per-event-type scale/prior, "
                        "same-category and recency biases")
    p.add_argument("--prand_base", type=float, default=-1.0,
                   help=">=0: per-feed loss weight = prand_base + p_random (exploration/random-order train feeds, "
                        "eda/labels_position/out/train_prandom.npz; p=0 for M<50 or NaN); aux loss weighted by 1-p")
    p.add_argument("--only_random", type=float, default=-1.0,
                   help=">=0: train only on feeds with p_random > this value (e.g. stage-2 continued training)")
    p.add_argument("--keep", default="best", choices=["best", "last"],
                   help="checkpoint to keep: best val eval (default) or the last step (no selection on val)")
    p.add_argument("--intent", type=float, default=0.0,
                   help="N3: weight of the category-intent aux loss (predict the category distribution of the "
                        "feed's positives from user+history; log P(cand category) is fed to the candidates)")
    p.add_argument("--xt", default="", help="train-stage extra features: preset 'cv' (leak-safe co-visitation + "
                                             "leave-user-out popularity) or a comma list of data/feats names")
    p.add_argument("--arch", default="tf", choices=["tf", "gru", "din"],
                   help="tf: transformer history encoder + cross/set attention; gru: bi-GRU history encoder; "
                        "din: DIN MLP attention + SiLU pooling + DCNv2 cross head, no set attention")
    p.add_argument("--id_buckets", type=int, default=0, help="N4: hashed item-ID residual table size (0 = off)")
    p.add_argument("--id_drop", type=float, default=0.5)
    p.add_argument("--user0", action="store_true", help="K5: add is_user0 and is_user0*log(1+H) dense features")
    p.add_argument("--domain", action="store_true",
                   help="K2: traffic-domain token (train: production soft-mixed to random-like by p_random)")
    p.add_argument("--holdout", type=int, default=0,
                   help="item 4: hold out this many random-like train feeds (p_random>0.9, M>=50, fixed seed) "
                        "from training and evaluate on them")
    p.add_argument("--select_on", default="val", choices=["val", "holdout"], help="checkpoint selection metric")
    p.add_argument("--init", default="", help="initialize weights (and architecture) from nn/ckpt/<name>.pt")
    p.add_argument("--pos_mode", default="both", choices=["both", "rev", "pos"],
                   help="position tower terms: g(pos)+g'(M-1-pos) / only end-of-feed g'(M-1-pos) / only g(pos)")
    p.add_argument("--time_limit", type=float, default=0, help="stop training after this many minutes")
    # ---- nnx/optim experiments (all default to the baseline behaviour)
    p.add_argument("--ema", default="", help="C1: comma list of weight-EMA betas; each EMA is scored at every eval "
                                             "after --ema_start and keeps its own holdout-selected best")
    p.add_argument("--ema_start", type=float, default=0.5, help="C1: fraction of total steps at which the EMAs start")
    p.add_argument("--muon", action="store_true", help="C2: Muon (RMS-matched) for hidden 2-D Linear weights, "
                                                       "AdamW for the rest; implies split k/v projections")
    p.add_argument("--sched", default="cos", choices=["cos", "wsd"], help="C4: lr schedule (wsd = 1-sqrt cooldown)")
    p.add_argument("--cooldown_frac", type=float, default=0.2, help="C4: fraction of steps in the WSD cooldown")
    p.add_argument("--cooldown_mix", type=float, default=0.0,
                   help="C4: probability of drawing an exploration-pool batch at each cooldown step")
    p.add_argument("--aux_anneal", type=float, default=-1.0,
                   help="C4: >=0: aux weight annealed from --aux_pos to this value with the cooldown lr")
    p.add_argument("--distill", default="", help="C5: path of frozen-teacher train logits (fp16, flat cand order)")
    p.add_argument("--distill_w", type=float, default=0.5)
    p.add_argument("--distill_tau", type=float, default=1.0)
    return p.parse_args()


def main():
    a = parse()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    log = Logger(LOGS / f"{a.variant}.log")
    log(f"==== {time.ctime()} args: {vars(a)}")
    tr = SplitData("train", dev, max_hist=a.max_hist)
    va = SplitData("val", dev, max_hist=a.max_hist)
    rng = np.random.default_rng(a.seed)
    feeds = np.arange(tr.n)
    if a.train_feeds:
        feeds = np.sort(rng.choice(tr.n, a.train_feeds, replace=False))
    prand = None
    hold, hold_rows = None, None
    if a.prand_base >= 0 or a.only_random >= 0 or a.domain or a.holdout:
        pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
        assert (pr["fidx"] == np.arange(tr.n)).all()
        prand_np = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
        prand_np[tr.M_np < 50] = 0.0  # detector unreliable for short feeds
        prand = torch.from_numpy(prand_np).to(dev)
        if a.holdout:
            cand = np.where((prand_np > 0.9) & (tr.M_np >= 50))[0]
            hold = np.sort(np.random.default_rng(12345).choice(cand, min(a.holdout, len(cand)), replace=False))
            hold_rows = np.concatenate([np.arange(o, o + m) for o, m in zip(tr.cand_off_np[hold], tr.M_np[hold])])
            feeds = np.setdiff1d(feeds, hold)
            log(f"holdout: {len(hold)} random-like train feeds excluded from training")
        if a.only_random >= 0:
            feeds = feeds[prand_np[feeds] > a.only_random]
        log(f"p_random: mean {prand_np.mean():.4f}, feeds>0.5 {(prand_np > 0.5).sum()}, training feeds {len(feeds)}")
    if a.init:
        ck0 = torch.load(CKPT / f"{a.init}.pt", map_location=dev, weights_only=False)
        xt_names, id_buckets = ck0["cfg"].get("xt_names"), ck0["cfg"].get("id_buckets", 0)
    else:
        xt_names = (XT_PRESETS[a.xt] if a.xt in XT_PRESETS else [n for n in a.xt.split(",") if n]) or None
        if xt_names:  # keep only features that exist for all three splits (feature dirs get pruned)
            miss = [n for n in xt_names if not all((ROOT / "data/feats" / sp / f"{n.removeprefix('has:')}.npy").exists()
                                                   for sp in ("train", "val", "test"))]
            xt_names = [n for n in xt_names if n not in miss] or None
            if miss:
                log(f"xt features missing on disk, dropped: {miss}")
        id_buckets = a.id_buckets
        assert not xt_names or a.user0, "--xt requires --user0 (dense feature layout)"
    if xt_names:
        for sd in (tr, va):
            sd.attach_xt(xt_names)
        log(f"train-stage extra features ({len(xt_names)}): {xt_names}")
    if id_buckets:
        for sd in (tr, va):
            sd.load_item_ids()
    if a.init:
        ck = ck0
        model = Ranker(**ck["cfg"]).to(dev)
        model.load_state_dict(ck["state_dict"])
        log(f"initialized from {a.init}: {ck['cfg']}")
    else:
        model = Ranker(d=a.d, heads=a.heads, hist_layers=a.hist_layers, blocks=a.blocks, drop=a.drop,
                       use_hist=not a.no_hist, use_set=not a.no_set, use_pos_tower=not a.no_pos,
                       aux_pos=a.aux_pos > 0, ff=a.ff, sim_bias=a.sim_bias,
                       pos_mode=a.pos_mode, n_dense=(N_DENSE + 2 if a.user0 else N_DENSE) + len(xt_names or []),
                       domain=a.domain, intent=a.intent > 0, xt_names=xt_names, id_buckets=id_buckets,
                       id_drop=a.id_drop, arch=a.arch, split_kv=a.muon).to(dev)
        mean, std = feature_stats(tr, feeds, n_dense=model.cfg["n_dense"])
        model.set_norm(mean, std)
    if model.domain:  # train feeds: production, soft-mixed toward random-like by p_random (M>=50)
        tr.dom = torch.stack([1 - prand, prand, torch.zeros_like(prand)], 1)
    log(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    muon_names = set()
    if a.muon:  # Moonlight rule: every hidden 2-D Linear weight (not embeddings, not 1-output layers)
        assert model.cfg.get("split_kv"), "--muon needs split k/v projections"
        muon_names = {f"{mn}.weight" for mn, mod in model.named_modules()
                      if isinstance(mod, nn.Linear) and min(mod.weight.shape) > 1}
    rest = [(n, p) for n, p in model.named_parameters() if n not in muon_names]
    decay = [p for n, p in rest if p.ndim >= 2 and "emb" not in n]
    no_decay = [p for n, p in rest if not (p.ndim >= 2 and "emb" not in n)]
    opts = [torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                              lr=a.lr, betas=(0.9, 0.98))]
    if a.muon:
        P_muon = [p for n, p in model.named_parameters() if n in muon_names]
        opts.append(torch.optim.Muon(P_muon, lr=a.lr, weight_decay=a.wd, momentum=0.95, nesterov=True,
                                     adjust_lr_fn="match_rms_adamw"))
        log(f"muon: {len(P_muon)} matrices, {sum(p.numel() for p in P_muon) / 1e6:.3f}M params; adamw: "
            f"{sum(p.numel() for _, p in rest) / 1e6:.3f}M params")
    n_ep = math.ceil(a.epochs)
    epoch_batches = [tr.batches(feeds, a.cand_budget, a.max_feeds, shuffle=True, rng=rng) for _ in range(n_ep)]
    all_batches = [b for ep in epoch_batches for b in ep]
    total = int(len(epoch_batches[0]) * a.epochs)
    all_batches = all_batches[:total]
    log(f"train feeds {len(feeds)}, steps/epoch {len(epoch_batches[0])}, total steps {total}")

    t_cool = int(round((1 - a.cooldown_frac) * total))  # C4: start of the WSD cooldown

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        if a.sched == "wsd":  # warmup-stable-decay with the 1-sqrt cooldown to the same 3% floor (Hagele 2024)
            if s < t_cool:
                return a.lr
            return a.lr * (0.03 + 0.97 * (1 - math.sqrt((s - t_cool) / max(1, total - t_cool))))
        return a.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, total - a.warmup))))

    def aux_w(s):  # C4: K9 weight annealed with the cooldown lr: aux_pos -> aux_anneal
        if a.aux_anneal < 0 or a.sched != "wsd" or s < t_cool:
            return a.aux_pos
        r = (lr_at(s) / a.lr - 0.03) / 0.97
        return a.aux_anneal + (a.aux_pos - a.aux_anneal) * r

    # C4: exploration pool for the cooldown (random-order train feeds, holdout already excluded from `feeds`)
    mix_rng, expl, expl_i = None, [], 0
    if a.cooldown_mix > 0:
        assert prand is not None
        pool = feeds[prand_np[feeds] > 0.5]
        mix_rng = np.random.default_rng(a.seed + 1)
        expl = tr.batches(pool, a.cand_budget, a.max_feeds, shuffle=True, rng=mix_rng)
        log(f"cooldown mix: p={a.cooldown_mix} from step {t_cool}, exploration pool {len(pool)} feeds, "
            f"{len(expl)} batches")
    n_mixed = 0

    # C5: frozen-teacher logits (flat train cand order)
    tz = None
    if a.distill:
        tz = torch.from_numpy(np.load(a.distill)).to(dev)
        assert tz.shape[0] == tr.n_cand and tz.dtype == torch.float16
        log(f"distill: teacher {a.distill}, w {a.distill_w}, tau {a.distill_tau}")
    kd_run = 0.0

    # C1: weight EMAs (fp32 copies), started at ema_start * total
    emas = []
    for tok in [t for t in a.ema.split(",") if t]:
        em = copy.deepcopy(model).float()
        for q_ in em.parameters():
            q_.requires_grad_(False)
        emas.append(dict(beta=float(tok), tag=f"ema{tok}", model=em, params=list(em.parameters()),
                         best=(-1.0, None, -1), hist=[]))
    ema_start = int(a.ema_start * total)
    params = list(model.parameters())
    if emas:
        log(f"ema: betas {[e['beta'] for e in emas]} from step {ema_start}")

    best = (-1.0, None, -1)
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    model.train()
    hist = []
    for step, fb in enumerate(all_batches):
        for opt in opts:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
        if mix_rng is not None and step >= t_cool and mix_rng.random() < a.cooldown_mix:
            if expl_i >= len(expl):  # reshuffle the exploration pool when exhausted
                expl, expl_i = tr.batches(pool, a.cand_budget, a.max_feeds, shuffle=True, rng=mix_rng), 0
            fb = expl[expl_i]
            expl_i += 1
            n_mixed += 1
        b = tr.gather(fb, with_pos=True)
        X = dense_features(b, model.cfg["n_dense"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(b, X, use_pos=True)
        y, m = b["y"], b["cmask"]
        fw = None if prand is None or a.prand_base < 0 else a.prand_base + prand[b["fidx"]]
        if a.loss == "softmax":
            loss = listwise_loss(lg, y, m, fw, norm=a.loss_norm, denom=a.max_feeds)
        elif a.loss == "bce":
            loss = bce_loss(lg, y, m)
        else:
            loss = listwise_loss(lg, y, m) + bce_loss(lg, y, m)
        if a.intent > 0 and getattr(model, "intent", False):
            q = torch.zeros(y.shape[0], model.intent_logp.shape[1], device=y.device).scatter_add_(1, b["ccat"], y)
            k = q.sum(1)
            li = -(q / k.clamp_min(1)[:, None] * model.intent_logp).sum(1)
            loss = loss + a.intent * (li * (k > 0)).sum() / (k > 0).sum().clamp_min(1)
        if tz is not None:  # C5: listwise KD, KL(softmax(zT/tau) || softmax(zS/tau)) per feed, equal feed weight
            tau = a.distill_tau
            zt = tz[b["ci"]].float()
            lpt = torch.log_softmax((zt / tau).masked_fill(~m, -1e4), 1)
            lps = torch.log_softmax((lg.float() / tau).masked_fill(~m, -1e4), 1)
            kl = (lpt.exp() * (lpt - lps) * m).sum(1)
            lkd = kl.sum() / a.max_feeds
            loss = loss + a.distill_w * tau ** 2 * lkd
            kd_run += lkd.item() if step % 50 == 0 else 0.0
        w_aux = aux_w(step)
        if a.aux_pos > 0:
            if a.aux_mode == "mse":
                pct = b["pos"].float() / (b["M"][:, None].float() - 1).clamp_min(1)
                am = m.float() if prand is None else m * (1 - prand[b["fidx"]])[:, None]  # random feeds: no prod order
                loss = loss + w_aux * (((model.aux_out - pct) ** 2) * am).sum() / am.sum().clamp_min(1)
            else:  # K9: listwise CE toward softmax(-pos/tau) (top-heavy, like the production objective)
                tgt = torch.softmax((-b["pos"].float() / a.aux_tau).masked_fill(~m, -1e4), 1) * m
                lp = torch.log_softmax(model.aux_out.masked_fill(~m, -1e4), 1)
                fw_aux = torch.ones_like(tgt[:, 0]) if prand is None else 1 - prand[b["fidx"]]
                loss = loss + w_aux * (-(tgt * lp).sum(1) * fw_aux).sum() / fw_aux.sum().clamp_min(1e-6)
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in opts:
            opt.step()
        if emas and step >= ema_start:
            with torch.no_grad():
                for e in emas:
                    if step == ema_start:
                        torch._foreach_copy_(e["params"], params)
                    else:  # theta_bar <- beta * theta_bar + (1 - beta) * theta
                        torch._foreach_lerp_(e["params"], params, 1.0 - e["beta"])
        run_loss += loss.item() if step % 50 == 0 else 0.0
        run_n += 1 if step % 50 == 0 else 0
        last = step == total - 1
        if (step + 1) % a.eval_every == 0 or last:
            pv = predict(model, va)
            nd = eval_ndcg(pv, va)
            ndh = eval_subset(model, tr, hold, hold_rows) if hold is not None else float("nan")
            el = (time.time() - t0) / 60
            extra = ""
            if tz is not None:
                extra += f" kd {kd_run / max(run_n, 1):.4f}"
            if mix_rng is not None:
                extra += f" mixed {n_mixed} w_aux {w_aux:.3f}"
            log(f"step {step + 1}/{total} loss {run_loss / max(run_n, 1):.4f} val_ndcg {nd:.5f} "
                f"holdout_ndcg {ndh:.5f} lr {lr_at(step):.2e} {el:.1f} min{extra}")
            hist.append(dict(step=step + 1, loss=run_loss / max(run_n, 1), val_ndcg=nd, holdout_ndcg=ndh, minutes=el))
            kd_run = 0.0
            for e in emas:
                if step < ema_start:
                    continue
                pve = predict(e["model"], va)
                nde = eval_ndcg(pve, va)
                ndhe = eval_subset(e["model"], tr, hold, hold_rows) if hold is not None else float("nan")
                log(f"    {e['tag']}: val_ndcg {nde:.5f} holdout_ndcg {ndhe:.5f} (raw {nd:.5f} / {ndh:.5f})")
                e["hist"].append(dict(step=step + 1, val_ndcg=nde, holdout_ndcg=ndhe))
                sele = ndhe if a.select_on == "holdout" else nde
                if sele > e["best"][0]:
                    e["best"] = (sele, copy.deepcopy(e["model"].state_dict()), step + 1)
            sel = ndh if a.select_on == "holdout" else nd
            run_loss, run_n = 0.0, 0
            if sel > best[0] or (a.keep == "last" and last):
                best = (sel, copy.deepcopy(model.state_dict()), step + 1)
            if a.time_limit and el > a.time_limit:
                log("time limit reached")
                break
    train_min = (time.time() - t0) / 60
    model.load_state_dict(best[1])
    torch.save({"state_dict": best[1], "cfg": model.cfg, "args": vars(a)}, CKPT / f"{a.variant}.pt")
    pv = predict(model, va)
    nd = eval_ndcg(pv, va)
    np.save(PREDS / f"nn_{a.variant}_trainonly_val.npy", pv.astype(np.float32))
    ndh_best = float("nan")
    if hold is not None:  # per-feed holdout scores (pairable across runs: hold is sorted, rows align)
        ndh_best, sch = eval_subset(model, tr, hold, hold_rows, return_scores=True)
        np.savez(PREDS / f"nn_{a.variant}_holdout.npz", scores=sch.astype(np.float32),
                 y=tr.target_np[hold_rows].astype(np.float32), M=tr.M_np[hold])
    log(f"BEST step {best[2]} val_ndcg {nd:.5f} holdout_ndcg {ndh_best:.5f}; train time {train_min:.1f} min")
    ema_summary = {}
    for e in emas:  # C1: each EMA's own holdout-selected best -> ckpt / val preds / holdout scores
        if e["best"][1] is None:
            continue
        name = f"{a.variant}_{e['tag']}"
        e["model"].load_state_dict(e["best"][1])
        torch.save({"state_dict": e["best"][1], "cfg": model.cfg, "args": vars(a)}, CKPT / f"{name}.pt")
        pve = predict(e["model"], va)
        nde = eval_ndcg(pve, va)
        np.save(PREDS / f"nn_{name}_trainonly_val.npy", pve.astype(np.float32))
        ndhe = float("nan")
        if hold is not None:
            ndhe, sche = eval_subset(e["model"], tr, hold, hold_rows, return_scores=True)
            np.savez(PREDS / f"nn_{name}_holdout.npz", scores=sche.astype(np.float32),
                     y=tr.target_np[hold_rows].astype(np.float32), M=tr.M_np[hold])
        log(f"BEST {e['tag']}: step {e['best'][2]} val_ndcg {nde:.5f} holdout_ndcg {ndhe:.5f}")
        ema_summary[e["tag"]] = dict(best_step=e["best"][2], val_ndcg=nde, holdout_ndcg=ndhe, history=e["hist"])
    if not a.no_test:
        del tr
        torch.cuda.empty_cache()
        te = SplitData("test", dev, max_hist=a.max_hist)
        if xt_names:
            te.attach_xt(xt_names)
        if id_buckets:
            te.load_item_ids()
        pt = predict(model, te)
        np.save(PREDS / f"nn_{a.variant}_trainonly_test.npy", pt.astype(np.float32))
        log(f"saved test preds {pt.shape}")
    save_json(LOGS / f"{a.variant}.json", dict(args=vars(a), best_step=best[2], val_ndcg=nd, holdout_ndcg=ndh_best,
                                                train_minutes=train_min, history=hist, ema=ema_summary,
                                                n_mixed=n_mixed))


if __name__ == "__main__":
    main()
