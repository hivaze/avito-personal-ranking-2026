#!/usr/bin/env bash
# wait for the 15 fold teachers, dump each fold's 3 models on all train candidates (some GPUs), assemble the OOF teacher
cd <repo>/nn_x/w12/kd; L=<repo>/nn_x/w5/logs; C=<repo>/nn_x/w5/ckpt
until [ $(ls $L/w5_e10_xf[0-4]_s95[1-3].json 2>/dev/null | wc -l) -ge 15 ]; do sleep 120; done; date -u
for k in <ids>; do CUDA_VISIBLE_DEVICES=$((k+2)) OMP_NUM_THREADS=12 <repo>/.venv/bin/python dump_teacher.py xf$k $C/w5_e10_xf${k}_s951.pt $C/w5_e10_xf${k}_s952.pt $C/w5_e10_xf${k}_s953.pt > out/dump_xf$k.log 2>&1 & done; wait
tail -qn 1 out/dump_xf?.log
OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES="" <repo>/.venv/bin/python assemble_oof.py
for k in 0 1 2 3 4; do grep -h "holdout_ndcg" $L/w5_e10_xf${k}_s951.log | tail -n 1 | cut -c1-140; done
