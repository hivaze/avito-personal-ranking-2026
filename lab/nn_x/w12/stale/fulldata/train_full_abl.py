"""Train the neural ranker on the (position-biased) train split, select by full NDCG on val.

FULLDATA copy (nn_x/w12/stale/fulldata, 2026-11-15; reports/STALE_1115/fulldata.md): nn_x/abl/train.py (frozen copy
ref/train_abl.py.txt, md5 in ref/md5.txt) + ONE flag, OFF by default and guarded so that a flag-off run executes exactly
the nn_x/abl/train.py code path (test_full.py: AST proof + textual diff + end-to-end CPU identity). Helper cd_full.py:
  --holdout_in_train     the 30,000 holdout feeds are identified exactly as before (hold / hold_rows, the exploration posterior,
                         every log line up to "holdout: ...") but are NOT removed from the training feeds nor from the cooldown
                         exploration pool; the kept checkpoint is the LAST step (no selection: the holdout is in-sample and val
                         is never a selection metric here); the holdout NDCG of every eval is logged on its own line with the
                         word IN-SAMPLE and the regular holdout_ndcg field is nan; holdout predictions are written as
                         preds/holdout/nn_<v>[@last]_holdoutINSAMPLE.npy (never as ..._holdout.npy); val / test predictions,
                         --save_last and --save_steps as before. Run from the code directory with PYTHONPATH=<code dir>:
                           cd <repo>/nn_x/abl && PYTHONPATH=<repo>/nn_x/abl python <repo>/nn_x/w12/stale/fulldata/train_full_abl.py --variant ... --holdout_in_train
Previous docstring:

nn_x/abl copy (ablation screen, 2026-11-13; ledger nn_x/abl/ledger/ledger.md section 4): nn_x/w5/train.py (frozen copy
tests/ref_train_w5.py.txt, md5 in tests/w5_source_md5.txt) + two input flags, both OFF at the default '' and guarded so
that a flag-off run executes exactly the nn_x/w5 code path (tests/test_abl.py: AST proof + end-to-end identity). Outputs
go to nn_x/abl/{ckpt,logs,preds}; variant names must start with abl_. Helpers in abl_arms.py:
  --xt_exclude n1,n2,...   xdrop_* arms: drop the named columns from the --xt preset list BEFORE the XT_REL / XT_RELONLY /
                           xt_idcols lookups (their rank / z channels vanish with them; n_dense follows)
  --xt_rel_extra n1,...    xrel_more arm: within-feed pct-rank + z channels (model.dense_features xt_rel) also for the
                           named preset columns, appended after XT_REL[preset] (+2 dense inputs per name)
Both act on the preset list, so they are refused with --init (the checkpoint cfg fixes the inputs there).
Second wave (2026-11-13 ~04:00 UTC, ABLATION_PLAN_1113 section 3.3; helper abl_retr.py, OFF at the default 0, same guarding):
  --retr_cat M             retr_cat arm: a SECOND HistRetrieval (M slots) over the candidate's SAME-CATEGORY history items,
                           zero-init output Z (flag-on at init bit-identical to flag-off), wired behind the base
                           retrieval by hooks; needs --retr_m > 0; refused with --init; cfg['retr_cat'] = M in checkpoints
                           (load them with abl_retr.load_ranker)
pop1 arm (2026-11-13 ~06:10 UTC, nn_x/codes/followup/RESULT.txt section (4); ledger section 7; helper abl_arms.py, OFF at
the default '', same guarding; the AST proof in tests/test_abl.py strips exactly one more import / add_argument / guard):
  --xt_add n1,n2,...       append the named feature columns LAST to the --xt list, right after --xt_exclude and BEFORE the
                           feat_exists miss check; XT_REL / XT_RELONLY / xt_idcols stay as without it (indices unchanged),
                           n_dense + len(names); refused with --init; the checkpoint cfg carries the extended xt_names
Previous nn_x/w5 docstring:

nn_x/w5 copy (wave 5, 2026-11-12; reports/INSIGHTS_1112.md section 3.1 items 1 / 3 / 8): nn_x/r6/train.py (frozen copy
tests/ref_train_r6.py.txt) + the wave-5 flags below, every one OFF by default and guarded so that a flag-off run executes
exactly the nn_x/r6 code path (tests/test_w5.py: AST proof + end-to-end identity). Outputs go to nn_x/w5/{ckpt,logs,preds};
variant names must start with w5_. Helpers in cd_arms5.py:
  --trunc_off A          TRUNC-PROD: training-only offset A * L[group, rkend] * (1 - p_soft) on the MAIN listwise logits
  --trunc_lifts PATH     TRUNC-PROD with per-M-bucket production lifts from PATH (trunc_lifts_mbucket.json; needs
                         --trunc_off > 0; '' = off = the two-group trunc_lift.json table)
  --aux_norm feed        AUXF: K9 aux = aux_w * --aux_norm_c (1.695) * sum_f (1-p_f) CE_aux_f / max_feeds
  --cliplog N            CLIPLOG: pre-clip grad norm every step, main / aux grad-norm split every N steps (diagnostic)
  --save_last / --save_holdout / --select_min_step K   hygiene: keep the last checkpoint (+ val / holdout / test preds),
                         write the selected checkpoint's holdout preds, select only among evals at step >= K
  --hzcd                 (cd_arms6.py, INSIGHTS item 6) popularity-parity cooldown: from step t_cool every TRAINING batch
                         reads data/feats_new/train/it_hist_hz_* in place of the it_hist_*_luo xt columns (same attach_xt
                         scaling); val / test / holdout evaluation unchanged; extra preds/holdout/nn_<v>_holdout_hz.npy
  --val_mix P --val_fold K  (cd_arms7.py, 11-12 ~23:35 UTC, idea D1 "val-in-cooldown") VALCD: from step t_cool each NON-pool
                         cooldown step trains w.p. P on a batch of LABELLED VAL feeds of the folds != K (nn/finetune.py's
                         user-grouped 5 folds; fold K = this run's OOF read) with the random-like domain token and
                         p_random 1 (aux weight 0); the m02 pool stream is unchanged; selection on the holdout
Previous nn_x/r6 docstring:

nn_x/r6 copy: outputs go to nn_x/r6/{ckpt,logs,preds} (common.py); variant names must start with r6_. New flags (all
default OFF = the nn/train.py run): --attn_drop, --retr_prod, --retr_uniq, --cretr_m, --set_sim_bias, --typed_cross
(model options, see model.py) and --init <absolute path> (read-only warm start from any checkpoint file).
Wave-2 cooldown-continuation flags (R6_PLAN 4.1, helpers in cd_arms.py; all OFF by default, guarded so that a flag-off
run executes the previous code path): --drop_cd, --pool_p, --aug_cand, --aug_hist (tests/test_cdw2.py).
Wave-4 cooldown-data flags (RESEARCH_SWEEP_1112 P5/P4/P3, helpers in cd_arms4.py; all OFF by default, same guarding;
tests/test_cdw4.py): --hcap (history-format parity), --rec_q/--rec_days (recency), --rho_file/--rho_keep (RHO-LOSS).

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

from cd_arms import POOL_P_DEFAULT, aug_enabled, augment_pool_batch, exploration_pool, set_dropout
from cd_arms4 import setup_wave4
from cd_arms5 import save_holdout, save_last, save_last_test, setup_wave5
from cd_arms6 import save_holdout_hz, setup_hzcd  # wave-5 hzcd (INSIGHTS_1112 item 6; OFF by default)
from cd_full import FullData  # FULLDATA --holdout_in_train (nn_x/w12/stale/fulldata; OFF by default)
from cd_arms7 import setup_valcd  # wave-5 valcd (val-in-cooldown, D1; OFF by default)
from abl_arms import xt_exclude_names, xt_rel_extend  # nn_x/abl input flags (OFF by default; ledger section 4)
from abl_retr import setup_retr_cat  # nn_x/abl second wave: same-category retrieval (OFF by default; ledger section 6)
from abl_arms import xt_add_names  # nn_x/abl pop1: append xt columns (OFF by default; ledger section 7)
from abl_arms import xt_add_relonly_idx  # nn_x/abl ENCODER-DOSE pre2noid: rank/z-only added columns (OFF by default; ledger section 10)
from encdose.kv_probe.kv_adapter import setup_kv64  # nn_x/abl ENCODER-DOSE (C): zero-init gated K/V adapter (OFF by default; ledger section 10.C)
from encdose.kv_probe.kv_data import attach_kv64  # nn_x/abl ENCODER-DOSE (C): 64-d encoder states in the batches (OFF by default; ledger section 10.C)
from common import CKPT, LOGS, PREDS, ROOT, Logger, eval_ndcg, eval_subset, feature_stats, predict, save_json
from common import ndcg  # wave-5 (metrics.ndcg via common; holdout NDCG of the saved holdout preds)
from data import SplitData, feat_exists
from model import N_DENSE, Ranker, bce_loss, dense_features, listwise_loss


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
    # ---- nn_x/r6 arms (reports/r6_design.md); defaults reproduce the nn/train.py run exactly
    p.add_argument("--attn_drop", type=float, default=-1.0,
                   help="r6 adrop0: attention-probability dropout in every attention block (<0: same as --drop)")
    p.add_argument("--retr_prod", action="store_true", help="r6 a4w: add the product term T3(k_c * k_j) to A4 values")
    p.add_argument("--retr_uniq", action="store_true",
                   help="r6 a4u: A4 retrieval over distinct items (newest occurrence), fp32 scores, repeat-count value")
    p.add_argument("--retr_tcnt", action="store_true",
                   help="r6 a4u+: with --retr_uniq, per-event-type occurrence counts of each retrieved item (value + logit)")
    p.add_argument("--cretr_m", type=int, default=0, help="r6 cset: TabR retrieval over the top-m same-feed candidates")
    p.add_argument("--set_sim_bias", action="store_true", help="r6 cset: cosine prior in the candidate set attention")
    p.add_argument("--typed_cross", action="store_true",
                   help="r6 typed: last cross block attends to the user token + target actions (event_type 1, source 1)")
    # ---- r6 wave-2 cooldown-continuation arms (R6_PLAN 4.1 "Wave 2", cd_arms.py); defaults = OFF (previous code path)
    p.add_argument("--drop_cd", type=float, default=-1.0,
                   help="wave-2 d0/d20: >=0: dropout p on EVERY dropout site (Block FFN + residual, retrieval value MLP "
                        "and attention-probability dropout) from the cooldown start (step t_cool; step 0 with "
                        "--cooldown_frac 1.0, i.e. right after --init). Exact; the checkpoint cfg keeps its drop")
    p.add_argument("--kd_file", default="", help="w12 KD: npy of teacher main logits per train candidate (flat order)")
    p.add_argument("--kd_w", type=float, default=0.5)
    p.add_argument("--kd_T", type=float, default=1.0)
    p.add_argument("--pool_p", type=float, default=POOL_P_DEFAULT,
                   help="wave-2 pp02: cooldown exploration pool = training feeds (holdout excluded; M>=50 since "
                        "p_random is 0 for M<50) with p_random > this (default 0.5 = the nn/train.py pool)")
    p.add_argument("--aug_cand", type=float, default=0.0,
                   help="wave-2 aug (r6_external R2): on cooldown POOL batches only, per feed q ~ U(aug_cand, 1) and each "
                        "negative candidate kept w.p. q, all positives kept (0 = off; values >= 1 are also a no-op)")
    p.add_argument("--aug_hist", type=float, default=0.0,
                   help="wave-2 aug (R2): on cooldown POOL batches only, drop each history event w.p. this (0 = off)")
    # ---- r6 wave-4 cooldown-data arms (RESEARCH_SWEEP_1112 P5/P4/P3, cd_arms4.py); defaults = OFF (previous code path)
    p.add_argument("--hcap", default="",
                   help="wave-4 hcap (P5): per-event-type history caps 'type:cap,...' (newest kept, compacted) on every "
                        "training batch from the cooldown start and on the selection holdout; '' = off")
    p.add_argument("--rec_q", type=float, default=0.0,
                   help="wave-4 rec33 (P4): prob. that a NON-pool cooldown batch comes from the last --rec_days train days "
                        "(holdout excluded); 0 = off")
    p.add_argument("--rec_days", type=int, default=3, help="wave-4 rec33: number of last train days (MSK) (with --rec_q)")
    p.add_argument("--rho_file", default="",
                   help="wave-4 rho70 (P3): excess file (rho_score.py) of this --init snapshot; NON-pool cooldown batches "
                        "sample the top --rho_keep production feeds by excess + the pool-eligible feeds; '' = off")
    p.add_argument("--rho_keep", type=float, default=0.7, help="wave-4 rho70: kept share of production feeds")
    # ---- nn_x/w5 wave-5 arms + hygiene (INSIGHTS_1112 3.1 items 1 / 3 / 8, cd_arms5.py); defaults = OFF (r6 code path)
    p.add_argument("--trunc_off", type=float, default=0.0,
                   help="wave-5 TRUNC-PROD: a > 0 adds a * L[group, rkend] * (1 - p_soft) (rkend 0-2, production groups "
                        "M<50 / M>=50, trunc_lift.json) to the main listwise logits of TRAINING batches only; 0 = off")
    p.add_argument("--trunc_lifts", default="",
                   help="wave-5 TRUNC-PROD M-bucket lifts: json (review/make_trunc_lifts.py, e.g. "
                        "nn_x/w5/trunc_lifts_mbucket.json) whose per-M-bucket production lifts replace the M<50 / M>=50 "
                        "groups of --trunc_off (needs --trunc_off > 0); '' = off")
    p.add_argument("--aux_norm", default="batch", choices=["batch", "feed"],
                   help="wave-5 AUXF: K9 aux normalisation; batch = live weighted per-batch mean (off), feed = "
                        "aux_norm_c * sum_f (1-p_f) CE_aux_f / max_feeds")
    p.add_argument("--aux_norm_c", type=float, default=1.695,
                   help="wave-5 AUXF constant (keeps the epoch-mean aux weight; labels_loss out/10_aux_norm_const.txt)")
    p.add_argument("--cliplog", type=int, default=0,
                   help="wave-5 CLIPLOG: N > 0 logs the pre-clip grad norm every step and the main / aux grad norms every "
                        "N steps (diagnostic; no effect on training); 0 = off")
    p.add_argument("--save_last", action="store_true",
                   help="wave-5 hygiene: also keep the LAST checkpoint ckpt/<v>@last.pt with its val / holdout (/ test) preds")
    p.add_argument("--save_holdout", action="store_true",
                   help="wave-5 hygiene (r7 port): write preds/holdout/nn_<v>_holdout.npy of the selected checkpoint")
    p.add_argument("--select_min_step", type=int, default=0,
                   help="wave-5 hygiene: checkpoint selection only among evals at step >= this (0 = off)")
    p.add_argument("--hzcd", action="store_true",
                   help="wave-5 hzcd (INSIGHTS item 6, cd_arms6.py): from the cooldown start (step t_cool; step 0 with "
                        "--cooldown_frac 1.0) every TRAINING batch (regular and pool) reads the horizon-limited popularity "
                        "data/feats_new/train/it_hist_hz_{all,et1..et4} in place of the it_hist_*_luo xt columns (same "
                        "attach_xt scaling); feature_stats, selection holdout, val / test and the saved holdout preds are "
                        "unchanged; extra file preds/holdout/nn_<v>_holdout_hz.npy")
    p.add_argument("--val_mix", type=float, default=0.0,
                   help="wave-5 valcd (D1 val-in-cooldown, cd_arms7.py): P > 0: from the cooldown start every NON-pool step "
                        "trains w.p. P on a batch of LABELLED VAL feeds of the folds != --val_fold (nn/finetune.py's "
                        "user-grouped 5-fold split; the own fold is never trained on = the OOF read), random-like domain "
                        "token, p_random 1 (aux weight 0); the m02 pool draws / batches are unchanged; 0 = off")
    p.add_argument("--val_fold", type=int, default=-1, help="wave-5 valcd: the held-out (OOF) val fold 0-4 (with --val_mix)")
    # ---- nn_x/abl ablation-screen input flags (abl_arms.py; ledger section 4 arms xdrop_pop / xdrop_ctxnd / xrel_more)
    p.add_argument("--xt_exclude", default="",
                   help="nn_x/abl xdrop_*: comma list of --xt preset columns to drop before the XT_REL / XT_RELONLY / "
                        "xt_idcols lookups (their rank / z channels go with them); '' = off (the full preset)")
    p.add_argument("--xt_rel_extra", default="",
                   help="nn_x/abl xrel_more: comma list of preset columns that ALSO get within-feed pct-rank + z channels "
                        "(appended after XT_REL[preset]; +2 dense inputs each); '' = off")
    p.add_argument("--retr_cat", type=int, default=0,
                   help="nn_x/abl second wave retr_cat (abl_retr.py): M > 0 adds a second HistRetrieval with M slots over "
                        "the candidate's SAME-CATEGORY history items (zero-init output; needs --retr_m > 0); 0 = off")
    p.add_argument("--xt_add", default="",
                   help="nn_x/abl pop1: comma list of feature columns appended LAST to the --xt list (after --xt_exclude, "
                        "before the miss check; XT_REL / XT_RELONLY / xt_idcols unchanged; +1 dense input each); '' = off")
    p.add_argument("--xt_add_relonly", default="",
                   help="nn_x/abl ENCODER-DOSE pre2noid (ledger section 10): comma list of --xt_add columns that get within-feed "
                        "rank / z channels and whose raw column is removed (the XT_RELONLY treatment; appended after the kept "
                        "XT_REL entries; net +1 dense input each); '' = off")
    p.add_argument("--kv64", type=int, default=0,
                   help="nn_x/abl ENCODER-DOSE (C) (encdose/kv_probe): 1 = attach the frozen pre2 encoder's 64-d states "
                        "(data/feats/<split>/kv64_*.npy) to every batch and a zero-init gated K/V adapter to the cross blocks "
                        "(bit-identical to the base at init); 0 = off")
    p.add_argument("--holdout_in_train", action="store_true",
                   help="FULLDATA (cd_full.py): keep the --holdout feeds identified as before but do NOT remove them from the "
                        "training feeds nor from the cooldown pool; checkpoint = the LAST step (no selection); holdout NDCG "
                        "logged as IN-SAMPLE only; holdout preds written as ..._holdoutINSAMPLE.npy")
    return p.parse_args()


def main():
    a = parse()
    assert a.variant.startswith("abl_"), "nn_x/abl copy: variant names must start with abl_"
    if a.xt_exclude or a.xt_rel_extra or a.retr_cat or a.xt_add or a.xt_add_relonly or a.kv64:  # nn_x/abl: the flags act on the preset list / architecture, which --init replaces by the cfg
        assert not a.init, "nn_x/abl: --xt_exclude / --xt_rel_extra / --retr_cat / --xt_add / --xt_add_relonly / --kv64 have no effect with --init (inputs and architecture come from the checkpoint cfg)"
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
    fd = None  # FULLDATA (cd_full.py): stays None unless --holdout_in_train
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
            if a.holdout_in_train:  # FULLDATA: remember the training feeds BEFORE the holdout is removed (checks + a.keep = 'last')
                fd = FullData(a, feeds, hold, hold_rows, tr, log)
            feeds = np.setdiff1d(feeds, hold)
            log(f"holdout: {len(hold)} random-like train feeds excluded from training")
            if fd is not None:  # FULLDATA: put them back (hold / hold_rows stay as identified above)
                feeds = fd.put_back(feeds, log)
        if a.only_random >= 0:
            feeds = feeds[prand_np[feeds] > a.only_random]
        log(f"p_random: mean {prand_np.mean():.4f}, feeds>0.5 {(prand_np > 0.5).sum()}, training feeds {len(feeds)}")
    if a.init:  # r6 review fix: with --init the model is rebuilt from the checkpoint cfg, so r6 model flags would be ignored
        assert a.attn_drop < 0 and not (a.retr_prod or a.retr_uniq or a.retr_tcnt or a.cretr_m or a.set_sim_bias
                                        or a.typed_cross), \
            "nn_x/r6: r6 model flags have no effect with --init (architecture comes from the checkpoint cfg)"
        ck0 = torch.load(a.init if a.init.startswith("/") else CKPT / f"{a.init}.pt", map_location=dev,
                         weights_only=False)  # r6: an absolute path reads any checkpoint (read-only)
        xt_names, id_buckets = ck0["cfg"].get("xt_names"), ck0["cfg"].get("id_buckets", 0)
    else:
        xt_names = (XT_PRESETS[a.xt] if a.xt in XT_PRESETS else [n for n in a.xt.split(",") if n]) or None
        if a.xt_exclude:  # nn_x/abl xdrop_*: drop columns BEFORE the miss check and the XT_REL / XT_RELONLY / idcols lookups
            xt_names = xt_exclude_names(xt_names, a.xt_exclude, log)
        if a.xt_add:  # nn_x/abl pop1: append columns LAST, after --xt_exclude and BEFORE the miss check (XT_REL / XT_RELONLY / idcols unchanged)
            xt_names = xt_add_names(xt_names, a.xt_add, log, reserved=XT_REL.get(a.xt, []) + XT_RELONLY.get(a.xt, []))
        if xt_names:  # keep only features that exist for all three splits (feature dirs get pruned)
            miss = [n for n in xt_names if not all(feat_exists(sp, n.removeprefix("has:"))
                                                   for sp in ("train", "val", "test"))]
            # r6 review fix: r6 data.feat_exists sees only nn/feat_cache (no data/feats fallback as in nn/data.py), so a
            # silent drop here would make an r6 run differ from its nn/ / sx control; fail loudly instead
            assert not miss, f"nn_x/r6: xt features missing from nn/feat_cache (would be silently dropped): {miss}"
            xt_names = [n for n in xt_names if n not in miss] or None
            if miss:
                log(f"xt features missing on disk, dropped: {miss}")
        id_buckets = a.id_buckets
        assert not xt_names or a.user0, "--xt requires --user0 (dense feature layout)"
        # raw-count presets: within-feed rank/z of these columns are computed in the model (after ID-input dropout)
        xt_rel = [xt_names.index(n) for n in XT_REL.get(a.xt, []) if xt_names and n in xt_names] or None
        if a.xt_rel_extra:  # nn_x/abl xrel_more: rank / z channels also for the named preset columns (after XT_REL's)
            xt_rel = xt_rel_extend(xt_rel, xt_names, a.xt_rel_extra, log)
        xt_idcols = [i for i, n in enumerate(xt_names or []) if n.startswith(("seq_cv", "it_hist"))] or None
        xt_relonly = [xt_names.index(n) for n in XT_RELONLY.get(a.xt, []) if xt_names and n in xt_names] or None
        if a.xt_add_relonly:  # nn_x/abl ENCODER-DOSE pre2noid: added columns used only through their within-feed rank / z (ledger section 10)
            xt_rel, xt_relonly = xt_add_relonly_idx(xt_rel, xt_relonly, xt_names, a.xt_add, a.xt_add_relonly, log)
    if xt_names:
        for sd in (tr, va):
            sd.attach_xt(xt_names)
        log(f"train-stage extra features ({len(xt_names)}): {xt_names}")
    if a.kv64:  # nn_x/abl ENCODER-DOSE (C): the exported 64-d encoder states ride along in every train / val batch (ledger section 10.C)
        for sd in (tr, va):
            attach_kv64(sd, ROOT / "data/feats" / sd.split)
            log(f"kv64: attached {sd.split} states {tuple(sd.kv64_states.shape)} fp16, exported feeds {int((sd.kv64_off >= 0).sum())} of {len(sd.kv64_off)}")
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
                       typed_cross=a.typed_cross, retr_tcnt=a.retr_tcnt).to(dev)
        model.eval()
        mean, std = feature_stats(tr, feeds, dense_fn=model.dense)
        model.train()
        model.set_norm(mean, std)
        if a.retr_cat:  # nn_x/abl retr_cat: second (same-category) retrieval, built AFTER the Ranker and BEFORE the optimisers
            model = setup_retr_cat(model, a, log)
        if a.kv64:  # nn_x/abl ENCODER-DOSE (C): zero-init gated K/V adapter, built AFTER the Ranker (and retr_cat) and BEFORE the optimisers
            model = setup_kv64(model, log)
    if model.domain:  # train feeds: production, soft-mixed toward random-like by p_random (M>=50)
        tr.dom = torch.stack([1 - prand, prand, torch.zeros_like(prand)], 1)
    log(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    aug_on = aug_enabled(a.aug_cand, a.aug_hist)  # wave-2 flags: checks only (no RNG draw, no state change)
    if aug_on or a.pool_p != POOL_P_DEFAULT:
        assert a.cooldown_mix > 0, "--pool_p / --aug_* act on cooldown pool batches: they need --cooldown_mix > 0"
    if aug_on:
        assert model.arch == "tf", "--aug_* (R2): arch tf only (the GRU packs by b['H'], which R2 leaves unchanged)"
    if a.drop_cd >= 0:
        assert a.drop_cd < 1.0, "--drop_cd must be in [0, 1)"
    muon_names = set()
    if a.muon:  # Moonlight rule: every hidden 2-D Linear weight (not embeddings, not 1-output layers)
        assert model.cfg.get("split_kv"), "--muon needs split k/v projections"
        muon_names = {f"{mn}.weight" for mn, mod in model.named_modules()
                      if isinstance(mod, torch.nn.Linear) and min(mod.weight.shape) > 1}
    rest = [(n, p) for n, p in model.named_parameters() if n not in muon_names]
    decay = [p for n, p in rest if p.ndim >= 2 and "emb" not in n]
    no_decay = [p for n, p in rest if not (p.ndim >= 2 and "emb" not in n)]
    opts = [torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                              lr=a.lr, betas=(0.9, 0.98))]
    if a.muon:
        P_muon = [p for n, p in model.named_parameters() if n in muon_names]
        opts.append(torch.optim.Muon(P_muon, lr=a.lr, weight_decay=a.wd, momentum=0.95, nesterov=True,
                                     adjust_lr_fn="match_rms_adamw"))
        log(f"muon: {len(P_muon)} matrices, {sum(p.numel() for p in P_muon) / 1e6:.3f}M params")
    n_ep = math.ceil(a.epochs)
    epoch_batches = [tr.batches(feeds, a.cand_budget, a.max_feeds, shuffle=True, rng=rng) for _ in range(n_ep)]
    all_batches = [b for ep in epoch_batches for b in ep]
    total = int(len(epoch_batches[0]) * a.epochs)
    all_batches = all_batches[:total]
    log(f"train feeds {len(feeds)}, steps/epoch {len(epoch_batches[0])}, total steps {total}")
    if fd is not None:  # FULLDATA: holdout feeds per epoch in the regular batches (count check)
        fd.steps_note(epoch_batches, total, a, tr, log)

    t_cool = int(round((1 - a.cooldown_frac) * total))  # start of the WSD cooldown
    assert a.drop_cd < 0 or t_cool < total, "--drop_cd: the cooldown never starts (t_cool >= total steps)"

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

    kd = None
    if a.kd_file:  # w12 KD (port of nn_x/w5/train.py)
        assert a.loss == "softmax"
        kd = torch.from_numpy(np.load(a.kd_file).astype(np.float32)).to(dev)
        assert kd.shape[0] == tr.n_cand
        log(f"w12 KD: teacher {a.kd_file} w {a.kd_w} T {a.kd_T}")
    mix_rng, expl, expl_i, pool = None, [], 0, None
    cd_loss = [[0.0, 0], [0.0, 0]]  # [regular, pool] running (sum, n) of the training loss during the cooldown
    if a.cooldown_mix > 0:  # exploration pool for the cooldown (random-order train feeds, holdout excluded)
        assert prand is not None, "--cooldown_mix needs p_random (use --holdout or --domain)"
        pool = exploration_pool(feeds, prand_np, a.pool_p, hold if fd is None else None)  # == feeds[prand_np[feeds] > 0.5] at the default
        if fd is not None:  # FULLDATA: the pool keeps its holdout feeds (counted and logged)
            fd.pool_note(pool, log)
        mix_rng = np.random.default_rng(a.seed + 1)
        expl = tr.batches(pool, a.cand_budget, a.max_feeds, shuffle=True, rng=mix_rng)
        log(f"cooldown mix: p={a.cooldown_mix} from step {t_cool}, exploration pool {len(pool)} feeds")
        if a.pool_p != POOL_P_DEFAULT or aug_on:
            log(f"wave-2: pool p_random > {a.pool_p} ({len(pool)} feeds, min M {tr.M_np[pool].min()}, holdout feeds in "
                f"pool {0 if hold is None else int(np.isin(pool, hold).sum())}); aug_cand {a.aug_cand} aug_hist "
                f"{a.aug_hist} on pool batches only ({'ON' if aug_on else 'off'})")
    tr_cap, cool_stream = None, None  # wave-4 (cd_arms4.py): both stay None unless a wave-4 flag is on
    if a.hcap or a.rec_q > 0 or a.rho_file:
        tr_cap, cool_stream = setup_wave4(a, tr, feeds, prand_np, hold, log)
    trunc, cliplog = None, None  # wave-5 (cd_arms5.py): both stay None unless a wave-5 arm flag is on
    if a.trunc_off or a.trunc_lifts or a.aux_norm != "batch" or a.cliplog:
        trunc, cliplog = setup_wave5(a, tr, feeds, prand, hold, all_batches, t_cool, total, log)
    if a.select_min_step:  # wave-5 hygiene: the selection window must contain the last step
        assert 0 < a.select_min_step <= total, f"--select_min_step {a.select_min_step} > total steps {total}"
        log(f"wave-5 selection: checkpoint chosen only among evals at step >= {a.select_min_step}")
    tr_hz = None  # wave-5 hzcd (cd_arms6.py): stays None unless --hzcd
    if a.hzcd:
        tr_hz = setup_hzcd(a, tr, xt_names, hold, t_cool, total, log)
    valcd = None  # wave-5 valcd (cd_arms7.py): stays None unless --val_mix > 0
    if a.val_mix > 0:  # wave-5 valcd: labelled val feeds (folds != val_fold) as an additive cooldown pool; p_random extended by 1s
        valcd, prand = setup_valcd(a, tr, va, prand, hold, t_cool, total, log, eval_ndcg)

    best = (-1.0, None, -1)
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    model.train()
    hist = []
    for step, fb in enumerate(all_batches):
        if a.drop_cd >= 0 and step == t_cool:  # wave-2 d0/d20: cooldown dropout on every site (cd_arms.set_dropout)
            nd_, na_ = set_dropout(model, a.drop_cd)
            log(f"drop_cd: dropout p={a.drop_cd} from step {step} on {nd_} nn.Dropout modules and {na_} "
                f"attention-probability sites (model cfg unchanged: drop {model.cfg['drop']})")
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
        if cool_stream is not None and not is_pool and step >= t_cool:  # wave-4 rec / rho: non-pool cooldown batches
            fb = cool_stream.next(fb)
        b = (tr_hz if tr_hz is not None and step >= t_cool else  # wave-5 hzcd: cooldown training batches read hz
             (tr_cap if tr_cap is not None and step >= t_cool else tr)).gather(fb, with_pos=True)
        if valcd is not None and step >= t_cool and valcd.draw(is_pool):  # wave-5 valcd: this non-pool step trains on val
            b, is_pool = valcd.gather(valcd.next()), True
        if is_pool and aug_on:  # wave-2 aug (R2): cooldown pool batches only; eval/predict never call this
            augment_pool_batch(b, a.aug_cand, a.aug_hist)
        X = model.dense(b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(b, X, use_pos=True)
        if trunc is not None:  # wave-5 TRUNC-PROD: training-only offset on the MAIN listwise logits (never K9 / eval)
            lg = lg + trunc.offset(b).to(lg.dtype)
        y, m = b["y"], b["cmask"]
        fw = None if prand is None or a.prand_base < 0 else a.prand_base + prand[b["fidx"]]
        if a.loss == "softmax":
            if kd is not None:
                qt = torch.softmax((kd[b["ci"]] / a.kd_T).masked_fill(~m, -1e4), 1) * m
                y = (1 - a.kd_w) * y / y.sum(1, keepdim=True).clamp_min(1) + a.kd_w * qt * (y.sum(1, keepdim=True) > 0)
            loss = listwise_loss(lg, y, m, fw, norm=a.loss_norm, denom=a.max_feeds)
        elif a.loss == "bce":
            loss = bce_loss(lg, y, m)
        else:
            loss = listwise_loss(lg, y, m) + bce_loss(lg, y, m)
        if cliplog is not None:  # wave-5 CLIPLOG: handle of the main loss (main / aux gradient-norm split)
            cliplog.main = loss
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
                loss = loss + aux_w(step) * (-(tgt * lp).sum(1) * fw_aux).sum() / (
                    a.max_feeds / a.aux_norm_c if a.aux_norm == "feed" else fw_aux.sum().clamp_min(1e-6))  # wave-5 AUXF
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        if cliplog is not None and step % a.cliplog == 0:  # wave-5 CLIPLOG: autograd.grad only (.grad untouched)
            cliplog.split(model, loss, step, b, is_pool)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if cliplog is not None:
            cliplog.record(step, gn, b, is_pool)
        for opt in opts:
            opt.step()
        if valcd is not None and valcd.is_val:  # wave-5 valcd: val-batch loss (diagnostic)
            valcd.add_loss(loss.item())
        if step >= t_cool and (is_pool or step % 5 == 0):  # cooldown diagnostics: pool vs regular batch loss
            cd_loss[int(is_pool)][0] += loss.item()
            cd_loss[int(is_pool)][1] += 1
        run_loss += loss.item() if step % 50 == 0 else 0.0
        run_n += 1 if step % 50 == 0 else 0
        last = step == total - 1
        if (step + 1) % a.eval_every == 0 or last:
            pv = predict(model, va)
            nd = eval_ndcg(pv, va)
            ndh = eval_subset(model, tr_cap if tr_cap is not None else tr, hold, hold_rows) if hold is not None else float("nan")
            if fd is not None:  # FULLDATA: logged on its own line as IN-SAMPLE; the regular field below shows nan
                ndh = fd.insample(step + 1, ndh, log)
            el = (time.time() - t0) / 60
            log(f"step {step + 1}/{total} loss {run_loss / max(run_n, 1):.4f} val_ndcg {nd:.5f} "
                f"holdout_ndcg {ndh:.5f} lr {lr_at(step):.2e} {el:.1f} min"
                + (f" cd_loss_reg {cd_loss[0][0] / max(cd_loss[0][1], 1):.4f} cd_loss_pool "
                   f"{cd_loss[1][0] / max(cd_loss[1][1], 1):.4f}" if step >= t_cool else ""))
            cd_loss = [[0.0, 0], [0.0, 0]]
            if cool_stream is not None:
                log(cool_stream.summary())
            if cliplog is not None:
                log(cliplog.summary(step + 1))
            if valcd is not None:  # wave-5 valcd: OOF (fold k) vs in-sample val NDCG of this eval
                log(valcd.summary(f"step {step + 1}", pv))
            hist.append(dict(step=step + 1, loss=run_loss / max(run_n, 1), val_ndcg=nd, holdout_ndcg=ndh, minutes=el))
            sel = ndh if a.select_on == "holdout" else nd
            if a.select_min_step and step + 1 < a.select_min_step:  # wave-5 hygiene: no selection before this step
                sel = -math.inf
            if fd is not None:  # FULLDATA: no selection at all; a.keep == 'last' (set by FullData) keeps the LAST step
                sel = -math.inf
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
    if cliplog is not None:  # wave-5 CLIPLOG summary + per-step / per-Mpad arrays
        cl_st, cl_m, cl_v = cliplog.save(LOGS / f"{a.variant}_cliplog.npz", step + 1)
        log(f"wave-5 CLIPLOG final: {cl_st}; by Mpad (median main / aux grad norm): {cl_m}; {cl_v}")
    if a.save_last:  # wave-5 hygiene: keep the last checkpoint as well as the selected one
        last_sd = save_last(a, model, va, tr, hold if fd is None else None, hold_rows, step + 1, CKPT, PREDS, predict, eval_ndcg, ndcg, log)
        if fd is not None:  # FULLDATA: the last checkpoint's holdout preds under the IN-SAMPLE name
            fd.save_insample(f"{a.variant}@last", model, tr, PREDS, predict, ndcg, log)
    model.load_state_dict(best[1])
    torch.save({"state_dict": best[1], "cfg": model.cfg, "args": vars(a)}, CKPT / f"{a.variant}.pt")
    pv = predict(model, va)
    nd = eval_ndcg(pv, va)
    np.save(PREDS / f"nn_{a.variant}_trainonly_val.npy", pv.astype(np.float32))
    log(f"BEST step {best[2]} val_ndcg {nd:.5f}; train time {train_min:.1f} min")
    if valcd is not None:  # wave-5 valcd: the selected checkpoint's OOF vs in-sample val NDCG
        log(valcd.summary(f"selected step {best[2]}", pv))
    if a.save_holdout and fd is None:  # wave-5 hygiene (r7 port): holdout preds of the selected checkpoint, identical to holdout_eval.py
        save_holdout(a, model, tr, hold, hold_rows, PREDS, predict, ndcg, log)
    if a.hzcd and fd is None:  # wave-5 hzcd: the selected checkpoint's holdout preds scored through the hz view (extra diagnostic file)
        save_holdout_hz(a, model, tr_hz, hold, hold_rows, PREDS, predict, ndcg, log)
    if fd is not None:  # FULLDATA: kept (= last) checkpoint's holdout preds (as built, and through the hz view with --hzcd), IN-SAMPLE names
        fd.finish(a, model, tr, tr_hz, best[2], total, PREDS, LOGS, predict, ndcg, log)
    if not a.no_test:
        del tr
        if a.hzcd:  # wave-5 hzcd: drop the view's reference to the train split so that `del tr` frees it before test loads
            tr_hz = None
        torch.cuda.empty_cache()
        te = SplitData("test", dev, max_hist=a.max_hist)
        if xt_names:
            te.attach_xt(xt_names)
        if id_buckets:
            te.load_item_ids()
        pt = predict(model, te)
        np.save(PREDS / f"nn_{a.variant}_trainonly_test.npy", pt.astype(np.float32))
        log(f"saved test preds {pt.shape}")
        if a.save_last:  # wave-5 hygiene: test preds of the last checkpoint too
            save_last_test(a, model, last_sd, te, PREDS, predict, log)
    save_json(LOGS / f"{a.variant}.json", dict(args=vars(a), best_step=best[2], val_ndcg=nd,
                                                train_minutes=train_min, history=hist))


if __name__ == "__main__":
    main()
