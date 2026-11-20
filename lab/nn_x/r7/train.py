"""Train the neural ranker on the (position-biased) train split, select by full NDCG on val.

Example:
  CUDA_VISIBLE_DEVICES=N python nn/train.py --variant base --epochs 1
Outputs:
  nn/ckpt/<variant>.pt                         best (by val NDCG) checkpoint
  preds/nn_<variant>_trainonly_{val,test}.npy  predictions of that checkpoint (flat cand order)
  nn/logs/<variant>.log / .json                training log and summary
nn_x/r7 copy: outputs go to nn_x/r7/{ckpt,logs,preds} (common.py); variant names must start with r7_. Every new flag
(r7 arms and the arms ported from nn_x/r6) defaults to OFF = the nn/train.py run (reports/r7_build.md).
"""
import argparse
import copy
import math
import os
import time

import numpy as np
import torch

from common import CKPT, LOGS, PREDS, ROOT, Logger, eval_ndcg, eval_subset, feature_stats, ndcg, predict, save_json
from data import SplitData, feat_exists
from model import N_DENSE, Ranker, bce_loss, dense_features, listwise_loss, r7_diagnostics


XT_PRESETS = {"cv": ["seq_cv_invd20", "seq_cv_w5", "seq_cv_fwd5", "seq_cv_t1", "seq_cv_seqs", "seq_cv_pop",
                      "seq_cv_norm", "seq_cv_invd20_rk", "seq_cv_invd20_z", "it_hist_all_luo", "it_hist_all_luo_rk",
                      "it_hist_all_luo_z", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo"]}
# + item cross-feed context (day-cutoff for train rows): feed-size stats, production-position prior, f1 prior
XT_PRESETS["cvix"] = XT_PRESETS["cv"] + ["ctx_ix_mlogM", "ctx_ix_slogM", "ctx_ix_dlogM", "ctx_ix_mrpos", "ctx_ix_mlpos",
                                         "ctx_ix_mf1", "ctx_ix_df1", "ctx_ix_mlogM_rk", "ctx_ix_mrpos_rk", "ctx_ix_mf1_rk",
                                         "has:ctx_ix_mlogM"]
XT_PRESETS["cvraw"] = ["seq_cv_invd20", "seq_cv_pop", "seq_cv_norm", "it_hist_all_luo", "it_hist_et1_luo",
                       "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo"]
XT_PRESETS["cvnd"] = XT_PRESETS["cvraw"] + ["ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev",
                                             "ctx_nd_knn_u", "ctx_nd_knn_u_dev", "ctx_nd_max", "ctx_nd_top5h",
                                             "ctx_nd_top5u", "ctx_nd_ucent", "ctx_hn_top1w", "ctx_hn_rrw",
                                             "ctx_hn_bestrk", "ctx_hn_margin", "ctx_hn_soft05", "ctx_hn_e1_rrw",
                                             "ctx_hs_mutual", "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_t10"]
# (cvnd_pre / cvnd_prez removed: the train-row pre_* leaked through content-table membership and are quarantined)
XT_REL = {"cvraw": ["seq_cv_invd20", "it_hist_all_luo"], "cvnd": ["seq_cv_invd20", "it_hist_all_luo"]}
# CPU-fixed leak-free pre features (NaN where it_hist_all_luo == 0, z recomputed over warm candidates)
XT_PRESETS["cvnd_prew"] = XT_PRESETS["cvnd"] + ["prew_fe_z", "prew_new_z", "prew_fe", "prew_new", "prew_hmax32",
                                                "has:prew_fe"]
XT_REL["cvnd_prew"] = ["seq_cv_invd20", "it_hist_all_luo", "prew_fe", "prew_new", "prew_hmax32"]
XT_PRESETS["cvnd_pre2"] = XT_PRESETS["cvnd"] + ["pre2_fe_z", "pre2_new_z", "pre2_fe", "pre2_new", "pre2_hmax32",
                                                "pre2_id_contrib", "has:pre2_fe"]  # id_contrib parity fixed upstream
XT_REL["cvnd_pre2"] = ["seq_cv_invd20", "it_hist_all_luo", "pre2_fe", "pre2_new", "pre2_hmax32", "pre2_id_contrib"]
XT_RELONLY = {"cvnd_prew": ["prew_fe", "prew_new", "prew_hmax32"], "cvnd_pre2": ["pre2_fe", "pre2_new", "pre2_hmax32", "pre2_id_contrib"]}
# FR-1 fix: cvnd_pre2 fed raw pre2_fe/new (median ~15) through attach_xt's +-10 clip (rank channel mostly ties) and
# included pre2_id_contrib (density-skewed no-ID share). cvnd_pre2s uses the clip-safe (x+20)/6 copies, no id_contrib.
XT_PRESETS["cvnd_pre2s"] = XT_PRESETS["cvnd"] + ["pre2_fe_z", "pre2_new_z", "pre2_fe_s", "pre2_new_s", "pre2_hmax32",
                                                 "has:pre2_fe"]
XT_REL["cvnd_pre2s"] = ["seq_cv_invd20", "it_hist_all_luo", "pre2_fe_s", "pre2_new_s", "pre2_hmax32"]
XT_RELONLY["cvnd_pre2s"] = ["pre2_fe_s", "pre2_new_s", "pre2_hmax32"]
XT_PRESETS["cvhn"] = XT_PRESETS["cv"] + ["ctx_hn_top1w", "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_hn_margin", "ctx_hn_soft05",
                                         "ctx_hn_e1_rrw", "ctx_hs_mutual", "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_t10"]
# + candidate competition for the user's recent history items (reciprocal-NN / soft attention mass, feed-internal)
XT_PRESETS["cvixhn"] = XT_PRESETS["cvix"] + ["ctx_hn_top1w", "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_hn_margin",
                                             "ctx_hn_soft05", "ctx_hn_e1_rrw", "ctx_hs_mutual", "ctx_hs_k128",
                                             "ctx_hs_e1", "ctx_hs_t10"]


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
    p.add_argument("--attn_gate", action="store_true", help="gated attention (elementwise sigmoid output gate)")
    p.add_argument("--retr_m", type=int, default=0, help="nnx A4: TabR-style retrieval over the top-m history items")
    p.add_argument("--muon", action="store_true", help="nnx C2: Muon (RMS-matched to AdamW) for hidden 2-D Linear "
                                                      "weights (needs split k/v projections, set automatically)")
    p.add_argument("--sched", default="cos", choices=["cos", "wsd"], help="nnx C4: lr schedule (wsd = 1-sqrt cooldown)")
    p.add_argument("--cooldown_frac", type=float, default=0.2, help="nnx C4: fraction of steps in the WSD cooldown")
    p.add_argument("--cooldown_mix", type=float, default=0.0,
                   help="nnx C4: probability of drawing an exploration-feed batch at each cooldown step")
    p.add_argument("--aux_anneal", type=float, default=-1.0,
                   help="nnx C4: >=0: aux weight annealed from --aux_pos to this value with the cooldown lr")
    p.add_argument("--id_buckets", type=int, default=0, help="N4: hashed item-ID residual table size (0 = off)")
    p.add_argument("--id_drop", type=float, default=0.5)
    p.add_argument("--id_hist_only", action="store_true", help="item-ID embeddings only in history tokens")
    p.add_argument("--xt_drop", type=float, default=0.0,
                   help="ID-input dropout: blank all --xt inputs of this fraction of train candidates (cold simulation)")
    p.add_argument("--user0", action="store_true", help="K5: add is_user0 and is_user0*log(1+H) dense features")
    p.add_argument("--domain", action="store_true",
                   help="K2: traffic-domain token (train: production soft-mixed to random-like by p_random)")
    p.add_argument("--holdout", type=int, default=0,
                   help="item 4: hold out this many random-like train feeds (p_random>0.9, M>=50, fixed seed) "
                        "from training and evaluate on them")
    p.add_argument("--select_on", default="val", choices=["val", "holdout"], help="checkpoint selection metric")
    p.add_argument("--save_steps", default="", help="W1: comma list of steps (eval steps) at which to also save "
                                                     "ckpt/<variant>@<step>.pt (snapshot ensembles)")
    p.add_argument("--init", default="", help="initialize weights (and architecture) from nn/ckpt/<name>.pt")
    p.add_argument("--pos_mode", default="both", choices=["both", "rev", "pos"],
                   help="position tower terms: g(pos)+g'(M-1-pos) / only end-of-feed g'(M-1-pos) / only g(pos)")
    p.add_argument("--time_limit", type=float, default=0, help="stop training after this many minutes")
    # ---- nn_x/r6 arms (ported verbatim; reports/r6_design.md); defaults reproduce the nn/train.py run exactly
    p.add_argument("--attn_drop", type=float, default=-1.0,
                   help="r6 adrop0: attention-probability dropout in every attention block (<0: same as --drop)")
    p.add_argument("--retr_prod", action="store_true", help="r6 a4w: add the product term T3(k_c * k_j) to A4 values")
    p.add_argument("--retr_uniq", action="store_true",
                   help="r6 a4u: A4 retrieval over distinct items (newest occurrence), fp32 scores, repeat-count value")
    p.add_argument("--cretr_m", type=int, default=0, help="r6 cset: TabR retrieval over the top-m same-feed candidates")
    p.add_argument("--set_sim_bias", action="store_true", help="r6 cset: cosine prior in the candidate set attention")
    p.add_argument("--typed_cross", action="store_true",
                   help="r6 typed: last cross block attends to the user token + target actions (event_type 1, source 1)")
    # ---- nn_x/r7 arms (reports/r7_research.md §5, reports/r7_build.md); defaults reproduce the nn/train.py run exactly
    p.add_argument("--muon_inputs", default="muon", choices=["muon", "adamw"],
                   help="r7 muon_in: optimiser of the input projections (c_mlp.0, c_lin, h_lin, u_lin, retr.wk, ...)")
    p.add_argument("--muon_momentum", type=float, default=0.95, help="r7 batch2s: Muon momentum (Nesterov)")
    p.add_argument("--adam_betas", default="0.9,0.98", help="r7 batch2s: AdamW betas")
    p.add_argument("--res_drop", type=float, default=-1.0,
                   help="r7 resdrop0: residual / FFN-hidden / retrieval dropout (<0: same as --drop); attention "
                        "probabilities keep --drop (see --attn_drop)")
    p.add_argument("--pbld", type=int, default=0, help="r7 pbld: RealMLP periodic embeddings, # frequencies (0 = off)")
    p.add_argument("--hc", type=int, default=0, help="r7 hc4: dynamic hyper-connections, # streams (0 = off)")
    p.add_argument("--dwa", action="store_true", help="r7 dwa: DenseFormer depth-weighted averages")
    p.add_argument("--diff_attn", default="", choices=["", "cross"], help="r7 diff2: differential target attention")
    p.add_argument("--ffn", default="gelu", choices=["gelu", "swiglu"], help="r7 swiglu: FFN form (matched params)")
    p.add_argument("--norm", default="ln", choices=["ln", "derf"], help="r7 derf: block / head normalisation")
    p.add_argument("--hist_extra", type=int, default=0, help="r7 hl2: extra zero-init history layers")
    p.add_argument("--periln", action="store_true", help="r7 periln: Peri-LN output norms on every sublayer")
    p.add_argument("--ftok", type=int, default=0, help="r7 ftok: feature-group tokens (8 = cvnd_pre2s groups)")
    p.add_argument("--postln", action="store_true", help="r7 postln (reserve): Post-LN + DeepNorm")
    p.add_argument("--save_holdout", action="store_true",
                   help="r7: also write preds/holdout/nn_<variant>_holdout.npy for the selected checkpoint (same file and "
                        "order as holdout_eval.py; avoids a second train-split load). Needs --holdout 30000")
    return p.parse_args()


R7_MODEL_FLAGS = ("attn_drop", "retr_prod", "retr_uniq", "cretr_m", "set_sim_bias", "typed_cross", "res_drop", "pbld",
                  "hc", "dwa", "diff_attn", "ffn", "norm", "hist_extra", "periln", "ftok", "postln")
R7_MODEL_DEFAULTS = dict(attn_drop=-1.0, retr_prod=False, retr_uniq=False, cretr_m=0, set_sim_bias=False,
                         typed_cross=False, res_drop=-1.0, pbld=0, hc=0, dwa=False, diff_attn="", ffn="gelu", norm="ln",
                         hist_extra=0, periln=False, ftok=0, postln=False)
# r7 parameters that must stay out of weight decay although they are 2-D (HC matrices, PBLD frequencies / phases / biases)
R7_NO_DECAY_PREFIX = ("hc.", "pbld.w1", "pbld.b1", "pbld.b2")


def build_opts(model, a, log=print):
    """Optimiser partition of nn/train.py (Muon for hidden 2-D Linear weights, AdamW for the rest) plus the r7 options:
    --muon_inputs adamw moves model.input_proj_names() to the AdamW decay group; R7_NO_DECAY_PREFIX keeps the 2-D HC / PBLD
    tensors out of weight decay; --muon_momentum / --adam_betas (defaults 0.95 / 0.9,0.98 = nn/). Returns (opts, info)."""
    muon_names = set()
    if a.muon:  # Moonlight rule: every hidden 2-D Linear weight (not embeddings, not 1-output layers)
        assert model.cfg.get("split_kv"), "--muon needs split k/v projections"
        muon_names = {f"{mn}.weight" for mn, mod in model.named_modules()
                      if isinstance(mod, torch.nn.Linear) and min(mod.weight.shape) > 1}
        if a.muon_inputs == "adamw":  # r7 muon_in: input projections -> AdamW decay group (same lr, schedule, wd)
            inp = set(model.input_proj_names())
            assert inp <= muon_names, f"input projections not on Muon: {inp - muon_names}"
            muon_names -= inp
            log(f"muon_in: input projections moved to AdamW (decay): {sorted(inp)}")
    else:
        assert a.muon_inputs == "muon", "--muon_inputs adamw needs --muon"
    betas = tuple(float(x) for x in a.adam_betas.split(","))
    assert len(betas) == 2
    rest = [(n, p) for n, p in model.named_parameters() if n not in muon_names]
    is_decay = lambda n, p: p.ndim >= 2 and "emb" not in n and not n.startswith(R7_NO_DECAY_PREFIX)  # noqa: E731
    decay = [p for n, p in rest if is_decay(n, p)]
    no_decay = [p for n, p in rest if not is_decay(n, p)]
    log(f"adamw: decay {len(decay)} tensors / {sum(p.numel() for p in decay)} params, no-decay {len(no_decay)} tensors / "
        f"{sum(p.numel() for p in no_decay)} params, betas {betas}")
    log(f"adamw decay names: {[n for n, p in rest if is_decay(n, p)]}")
    opts = [torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                              lr=a.lr, betas=betas)]
    if a.muon:
        P_muon = [p for n, p in model.named_parameters() if n in muon_names]
        opts.append(torch.optim.Muon(P_muon, lr=a.lr, weight_decay=a.wd, momentum=a.muon_momentum, nesterov=True,
                                     adjust_lr_fn="match_rms_adamw"))
        log(f"muon: {len(P_muon)} matrices, {sum(p.numel() for p in P_muon) / 1e6:.3f}M params, momentum {a.muon_momentum}")
    info = dict(muon=sorted(muon_names), adamw_decay=[n for n, p in rest if is_decay(n, p)],
                adamw_no_decay=[n for n, p in rest if not is_decay(n, p)])
    return opts, info


def main():
    a = parse()
    assert a.variant.startswith("r7_"), "nn_x/r7 copy: variant names must start with r7_"
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
    if a.init:  # with --init the model is rebuilt from the checkpoint cfg, so model flags would be ignored silently
        assert all(getattr(a, k) == v for k, v in R7_MODEL_DEFAULTS.items()), \
            "nn_x/r7: model flags have no effect with --init (architecture comes from the checkpoint cfg)"
        ck0 = torch.load(a.init if a.init.startswith("/") else CKPT / f"{a.init}.pt", map_location=dev,
                         weights_only=False)  # an absolute path reads any checkpoint (read-only)
        xt_names, id_buckets = ck0["cfg"].get("xt_names"), ck0["cfg"].get("id_buckets", 0)
    else:
        xt_names = (XT_PRESETS[a.xt] if a.xt in XT_PRESETS else [n for n in a.xt.split(",") if n]) or None
        if xt_names:  # keep only features that exist for all three splits (feature dirs get pruned)
            miss = [n for n in xt_names if not all(feat_exists(sp, n.removeprefix("has:"))
                                                   for sp in ("train", "val", "test"))]
            # nn_x/r7: feat_exists sees only nn/feat_cache (read-only), so a silent drop here would make an r7 run
            # differ from its nn/ control; fail loudly instead
            assert not miss, f"nn_x/r7: xt features missing from nn/feat_cache (would be silently dropped): {miss}"
        id_buckets = a.id_buckets
        assert not xt_names or a.user0, "--xt requires --user0 (dense feature layout)"
        # raw-count presets: within-feed rank/z of these columns are computed in the model (after ID-input dropout)
        xt_rel = [xt_names.index(n) for n in XT_REL.get(a.xt, []) if xt_names and n in xt_names] or None
        xt_idcols = [i for i, n in enumerate(xt_names or []) if n.startswith(("seq_cv", "it_hist"))] or None
        xt_relonly = [xt_names.index(n) for n in XT_RELONLY.get(a.xt, []) if xt_names and n in xt_names] or None
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
                       pos_mode=a.pos_mode, n_dense=(N_DENSE + 2 if a.user0 else N_DENSE) + len(xt_names or []) + 2 * len(xt_rel or [])
                       - len(xt_relonly or []),
                       domain=a.domain, intent=a.intent > 0, xt_names=xt_names, id_buckets=id_buckets,
                       id_drop=a.id_drop, arch=a.arch, id_hist_only=a.id_hist_only,
                       xt_rel=xt_rel, xt_drop=a.xt_drop, xt_idcols=xt_idcols, xt_relonly=xt_relonly,
                       attn_gate=a.attn_gate, split_kv=a.muon, retr_m=a.retr_m,
                       attn_drop=a.attn_drop if a.attn_drop >= 0 else None, retr_prod=a.retr_prod,
                       retr_uniq=a.retr_uniq, cretr_m=a.cretr_m, set_sim_bias=a.set_sim_bias,
                       typed_cross=a.typed_cross, res_drop=a.res_drop if a.res_drop >= 0 else None, pbld=a.pbld,
                       hc=a.hc, dwa=a.dwa, diff_attn=a.diff_attn, ffn=a.ffn, norm=a.norm, hist_extra=a.hist_extra,
                       periln=a.periln, ftok=a.ftok, postln=a.postln).to(dev)
        model.eval()
        mean, std = feature_stats(tr, feeds, dense_fn=model.dense)
        model.train()
        model.set_norm(mean, std)
    if model.domain:  # train feeds: production, soft-mixed toward random-like by p_random (M>=50)
        tr.dom = torch.stack([1 - prand, prand, torch.zeros_like(prand)], 1)
    log(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M ({sum(p.numel() for p in model.parameters())})")
    opts, _ = build_opts(model, a, log)
    n_ep = math.ceil(a.epochs)
    epoch_batches = [tr.batches(feeds, a.cand_budget, a.max_feeds, shuffle=True, rng=rng) for _ in range(n_ep)]
    all_batches = [b for ep in epoch_batches for b in ep]
    total = int(len(epoch_batches[0]) * a.epochs)
    all_batches = all_batches[:total]
    log(f"train feeds {len(feeds)}, steps/epoch {len(epoch_batches[0])}, total steps {total}")

    t_cool = int(round((1 - a.cooldown_frac) * total))  # start of the WSD cooldown

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        if a.sched == "wsd":  # warmup-stable-decay with the 1-sqrt cooldown to the same 3% floor
            if s < t_cool:
                return a.lr
            return a.lr * (0.03 + 0.97 * (1 - math.sqrt((s - t_cool) / max(1, total - t_cool))))
        return a.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, total - a.warmup))))

    def aux_w(s):  # aux weight annealed with the cooldown lr: aux_pos -> aux_anneal
        if a.aux_anneal < 0 or a.sched != "wsd" or s < t_cool:
            return a.aux_pos
        r = (lr_at(s) / a.lr - 0.03) / 0.97
        return a.aux_anneal + (a.aux_pos - a.aux_anneal) * r

    mix_rng, expl, expl_i, pool = None, [], 0, None
    cd_loss = [[0.0, 0], [0.0, 0]]  # [regular, pool] running (sum, n) of the training loss during the cooldown
    if a.cooldown_mix > 0:  # exploration pool for the cooldown (random-order train feeds, holdout excluded)
        assert prand is not None, "--cooldown_mix needs p_random (use --holdout or --domain)"
        pool = feeds[prand_np[feeds] > 0.5]
        mix_rng = np.random.default_rng(a.seed + 1)
        expl = tr.batches(pool, a.cand_budget, a.max_feeds, shuffle=True, rng=mix_rng)
        log(f"cooldown mix: p={a.cooldown_mix} from step {t_cool}, exploration pool {len(pool)} feeds")

    best = (-1.0, None, -1)
    diag0 = r7_diagnostics(model)
    log(f"r7 init diagnostics: {diag0}")
    torch.cuda.reset_peak_memory_stats(dev)
    t_start = time.time()
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    model.train()
    hist = []
    for step, fb in enumerate(all_batches):
        for opt in opts:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
        is_pool = False
        if mix_rng is not None and step >= t_cool and mix_rng.random() < a.cooldown_mix:
            if expl_i >= len(expl):  # reshuffle the exploration pool when exhausted
                expl, expl_i = tr.batches(pool, a.cand_budget, a.max_feeds, shuffle=True, rng=mix_rng), 0
            fb = expl[expl_i]
            expl_i += 1
            is_pool = True
        b = tr.gather(fb, with_pos=True)
        X = model.dense(b)
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
        if a.aux_pos > 0:
            if a.aux_mode == "mse":
                pct = b["pos"].float() / (b["M"][:, None].float() - 1).clamp_min(1)
                am = m.float() if prand is None else m * (1 - prand[b["fidx"]])[:, None]  # random feeds: no prod order
                loss = loss + aux_w(step) * (((model.aux_out - pct) ** 2) * am).sum() / am.sum().clamp_min(1)
            else:  # K9: listwise CE toward softmax(-pos/tau) (top-heavy, like the production objective)
                tgt = torch.softmax((-b["pos"].float() / a.aux_tau).masked_fill(~m, -1e4), 1) * m
                lp = torch.log_softmax(model.aux_out.masked_fill(~m, -1e4), 1)
                fw_aux = torch.ones_like(tgt[:, 0]) if prand is None else 1 - prand[b["fidx"]]
                loss = loss + aux_w(step) * (-(tgt * lp).sum(1) * fw_aux).sum() / fw_aux.sum().clamp_min(1e-6)
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in opts:
            opt.step()
        if step >= t_cool and (is_pool or step % 5 == 0):  # cooldown diagnostics: pool vs regular batch loss
            cd_loss[int(is_pool)][0] += loss.item()
            cd_loss[int(is_pool)][1] += 1
        run_loss += loss.item() if step % 50 == 0 else 0.0
        run_n += 1 if step % 50 == 0 else 0
        last = step == total - 1
        if (step + 1) % a.eval_every == 0 or last:
            pv = predict(model, va)
            nd = eval_ndcg(pv, va)
            ndh = eval_subset(model, tr, hold, hold_rows) if hold is not None else float("nan")
            el = (time.time() - t0) / 60
            log(f"step {step + 1}/{total} loss {run_loss / max(run_n, 1):.4f} val_ndcg {nd:.5f} "
                f"holdout_ndcg {ndh:.5f} lr {lr_at(step):.2e} {el:.1f} min"
                + (f" cd_loss_reg {cd_loss[0][0] / max(cd_loss[0][1], 1):.4f} cd_loss_pool "
                   f"{cd_loss[1][0] / max(cd_loss[1][1], 1):.4f}" if step >= t_cool else ""))
            cd_loss = [[0.0, 0], [0.0, 0]]
            hist.append(dict(step=step + 1, loss=run_loss / max(run_n, 1), val_ndcg=nd, holdout_ndcg=ndh, minutes=el))
            sel = ndh if a.select_on == "holdout" else nd
            if a.save_steps and str(step + 1) in a.save_steps.split(","):
                torch.save({"state_dict": copy.deepcopy(model.state_dict()), "cfg": model.cfg, "args": vars(a),
                            "step": step + 1, "val_ndcg": nd, "holdout_ndcg": ndh}, CKPT / f"{a.variant}@{step + 1}.pt")
            run_loss, run_n = 0.0, 0
            if sel > best[0] or (a.keep == "last" and last):
                best = (sel, copy.deepcopy(model.state_dict()), step + 1)
            if a.time_limit and el > a.time_limit:
                log("time limit reached")
                break
    train_min = (time.time() - t0) / 60
    peak = dict(peak_alloc_gb=torch.cuda.max_memory_allocated(dev) / 2**30,
                peak_reserved_gb=torch.cuda.max_memory_reserved(dev) / 2**30)
    log(f"r7 end-of-training diagnostics (last step): {r7_diagnostics(model)}")
    log(f"r7 peak GPU memory: {peak}; {train_min / max(total, 1) * 1000:.3f} min / 1000 steps")
    model.load_state_dict(best[1])
    torch.save({"state_dict": best[1], "cfg": model.cfg, "args": vars(a)}, CKPT / f"{a.variant}.pt")
    pv = predict(model, va)
    nd = eval_ndcg(pv, va)
    np.save(PREDS / f"nn_{a.variant}_trainonly_val.npy", pv.astype(np.float32))
    if a.save_holdout:  # identical to holdout_eval.py: predict(model, tr, hold)[rows], holdout feeds sorted
        assert hold is not None and a.holdout == 30000, "--save_holdout needs --holdout 30000 (the fixed holdout)"
        hd = PREDS / "holdout"
        hd.mkdir(exist_ok=True)
        torch.backends.cuda.matmul.allow_tf32 = False  # holdout_eval.py runs without TF32: identical numerics
        sh = predict(model, tr, hold)[hold_rows].astype(np.float32)
        torch.backends.cuda.matmul.allow_tf32 = True
        np.save(hd / f"nn_{a.variant}_holdout.npy", sh)
        M_h = tr.M_np[hold]
        meta = np.stack([M_h, np.concatenate([[0], np.cumsum(M_h)[:-1]])])
        if not (hd / "holdout_meta.npy").exists():
            np.save(hd / "holdout_meta.npy", meta)
            np.save(hd / "holdout_y.npy", tr.target_np[hold_rows])
        assert (np.load(hd / "holdout_meta.npy") == meta).all(), "holdout meta mismatch"
        log(f"saved holdout preds {sh.shape} -> {hd / f'nn_{a.variant}_holdout.npy'} (holdout NDCG "
            f"{ndcg(sh, tr.target_np[hold_rows], meta[1], M_h):.5f})")
    diag = r7_diagnostics(model)
    log(f"r7 selected-checkpoint diagnostics: {diag}")
    log(f"BEST step {best[2]} val_ndcg {nd:.5f}; train time {train_min:.1f} min")
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
    save_json(LOGS / f"{a.variant}.json", dict(args=vars(a), best_step=best[2], val_ndcg=nd,
                                                train_minutes=train_min, history=hist, total_steps=total,
                                                gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), start_unix=t_start,
                                                n_params=sum(p.numel() for p in model.parameters()), diag=diag,
                                                diag_init=diag0,
                                                **peak))


if __name__ == "__main__":
    main()
