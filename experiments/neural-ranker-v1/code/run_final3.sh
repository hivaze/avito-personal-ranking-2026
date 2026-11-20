#!/usr/bin/env bash
# Driver that records how the round-2 ensemble (nn_final3) was built: 13 train-stage runs with the
# co-visitation and popularity inputs in three architectures (transformer, GRU, DIN-style attention),
# the second stage on random-order feeds, the fine-tune on validation and the equal-weight ensemble
# with two members of the previous round.
# Usage as written in the file: CUDA_VISIBLE_DEVICES=N bash nn/run_final3.sh. lab/reports/nn_report.md
# names this file as the record of the recipe.
# Produced preds/nn_final3_*: the 15 members read val 0.38828 (fine-tuned + 0.2 x train-only) and
# 0.38704 (train-only, no validation labels).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn/run_final3.sh
#
# Reproduce nn_final3 (round 2). Usage: CUDA_VISIBLE_DEVICES=N bash nn/run_final3.sh
# Train stage: v2_feed_k9 recipe (sim bias, listwise production-order distillation, equal per-feed loss weight,
# domain token, user-0 inputs, checkpoint chosen on the fixed 30K random-like train holdout) + leak-safe
# co-visitation / leave-user-out popularity inputs (--xt cv | cvraw | cvhn | cvnd); architectures tf / bi-GRU / DIN.
# Then stage 2 on random-order train feeds + val fine-tune with the n2c_pre branch (n2clean + pretrained-encoder
# features, no ALS), and an equal-weight per-feed z-mean ensemble (+0.2 x the train-only ensemble).
# The v2 members s2_v2_feed_k9 / s2_v2_k9 come from run_final_v2.sh (stage 1 + stage 2 only are needed).
# Feature files are read through the hard-linked snapshot nn/feat_cache/ (upstream pruning cannot break inference).
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
C="--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout --epochs 2 --eval_every 1000"
$PY train.py --variant v4_cv            $C --xt cv    --loss_norm feed --seed 23
$PY train.py --variant v4_cv_s33        $C --xt cv    --loss_norm idcg --seed 33
$PY train.py --variant v4_cv_s38        $C --xt cv    --loss_norm feed --seed 38
$PY train.py --variant v7_q0            $C --xt cvraw --loss_norm feed --seed 23
$PY train.py --variant v8_nd_q0         $C --xt cvnd  --loss_norm feed --seed 23
$PY train.py --variant v8_nd_q15        $C --xt cvnd  --loss_norm feed --xt_drop 0.15 --seed 23
$PY train.py --variant v4_cv_gru        $C --xt cv    --loss_norm feed --arch gru --seed 31
$PY train.py --variant v4_cv_gru_s35    $C --xt cv    --loss_norm feed --arch gru --seed 35
$PY train.py --variant v4_cv_gru_s37    $C --xt cv    --loss_norm feed --arch gru --seed 37
$PY train.py --variant v4_cv_din        $C --xt cv    --loss_norm feed --arch din --no_set --loss both --seed 32
$PY train.py --variant v4_cv_din_s34    $C --xt cv    --loss_norm feed --arch din --no_set --loss both --seed 34
$PY train.py --variant v6_cvhn          $C --xt cvhn  --loss_norm feed --seed 23
$PY train.py --variant v6_cvhn_gru      $C --xt cvhn  --loss_norm feed --arch gru --seed 36
R2="v4_cv v4_cv_s33 v4_cv_s38 v7_q0 v8_nd_q0 v8_nd_q15 v4_cv_gru v4_cv_gru_s35 v4_cv_gru_s37 v4_cv_din v4_cv_din_s34 v6_cvhn v6_cvhn_gru"
EXTRA=n2c_pre SUF=n2cp bash stage2_ft_v3.sh $R2                      # -> s2_<v>, s2_<v>_n2cp
EXTRA=n2c_pre SUF=n2cp bash ft_list.sh s2_v2_feed_k9 s2_v2_k9
M=""; for v in $R2; do M="$M s2_$v"; done; M="$M s2_v2_feed_k9 s2_v2_k9"
FT=""; for m in $M; do FT="$FT,${m}_n2cp"; done
$PY ensemble.py --out final3 --w_trainonly 0.2 --variants "${FT#,}" --trainonly_variants "$(echo $M | tr ' ' ',')"
