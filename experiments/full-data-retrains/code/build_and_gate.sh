#!/usr/bin/env bash
# Builds one all-data submission from its members and runs the pre-submission gate on it (CPU only).
# Steps: resolve the member list, collect each member's fine-tuned and train-only predictions, build the
# pool (0.8 wide + 0.2 popularity-light per branch, then 0.5 fine-tuned + 0.5 train-only), run the gate
# against the team-best file as reference, print the sha256. Nothing is submitted.
# Usage as in the file: bash fd_finish.sh FD16P20   (FD16P20 = 16 wide + 4 popularity-light students).
# Produced subs/cand_FD16P20kd_r4.parquet and nn_x/w12/kd/out/gate_FD16P20kd.txt.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/kd/fd/fd_finish.sh
#
# fd_finish.sh <KEY> [check]: build + gate one round-4 file from its members (CPU only). KEY = a build_r4.py key (FD8P20 | FDX8P20:hz | FD16P20 | FDX16P20:hz | ...).
#   check : only list which member files are present / missing (no build).
# Steps: (1) member list from build_r4.py's spec; (2) copy each member's FT + train-only prediction files into preds/ and its train / FT logs into nn/logs/
# (cp --update=none; sources nn_x/w5 or nn_x/abl by prefix); (3) build_r4.py <KEY> build (writes preds/ens_<tag>_* and subs/cand_<tag>_r4.parquet; refuses to overwrite);
# (4) src/presub_gate.py against the W8P20 file as reference (member audit with --ft_suffix n2cpv3s3); (5) sha256. Nothing is submitted.
set -uo pipefail
cd <repo> || exit 1; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
K=${1:?key}; mode=${2:-build}; tag=${K//:/_}; tag=${tag//@m/_m}; tag=${tag//./}kd   # same tag as build_r4.py ('@m0.2_0.4' -> '_m02_04')
M=$(.venv/bin/python - "$K" <<'EOF'
import sys, re
src = open('<repo>/nn_x/w12/kd/build_r4.py').read()
head = src[src.index('S8 = range'):src.index('fv, Mv, ov = feed_meta')]
ns = {}; exec('from pathlib import Path\n' + head, ns)
base, pop, w = ns['spec'](sys.argv[1].split('@m')[0]); print(','.join(base + pop))
EOF
) || { echo "cannot resolve members of $K"; exit 1; }
miss=0
for v in ${M//,/ }; do
  case $v in w5_*) H=nn_x/w5 ;; abl_*) H=nn_x/abl ;; *) echo "unknown harness for $v"; exit 1 ;; esac
  for f in n2cpv3s3_val_oof n2cpv3s3_test trainonly_val trainonly_test; do
    [ -f preds/nn_${v}_$f.npy ] || cp --update=none $H/preds/nn_${v}_$f.npy preds/ 2>/dev/null
    [ -f preds/nn_${v}_$f.npy ] || { echo "MISSING nn_${v}_$f.npy"; miss=$((miss + 1)); }
  done
  for l in $v.log ${v}_n2cpv3s3_ft.log; do [ -f nn/logs/$l ] || cp --update=none $H/logs/$l nn/logs/ 2>/dev/null; [ -f nn/logs/$l ] || echo "log missing: $l"; done
done
echo "$K: $(echo ${M//,/ } | wc -w) members, $miss prediction files missing"
# log copies taken while a run was still training are byte prefixes of the complete log (review 2026-11-15): replace such copies by the complete source before the gate reads them
.venv/bin/python nn_x/w12/kd/fd/refresh_log_copies.py "$K" apply | tail -n 1
[ "$mode" = check ] && exit 0
[ $miss -eq 0 ] || { echo "not built: files missing"; exit 2; }
.venv/bin/python nn_x/w12/kd/build_r4.py "$K" build 2>&1 | cut -c1-700
[ -f subs/cand_${tag}_r4.parquet ] || { echo "build failed"; exit 3; }
.venv/bin/python src/presub_gate.py --sub subs/cand_${tag}_r4.parquet --val "0.5*ens_${tag}_ft_val+0.5*ens_${tag}_to_val" --test "0.5*ens_${tag}_ft_test+0.5*ens_${tag}_to_test" \
  --ref_val "0.5*ens_W8P20kd_ft_val+0.5*ens_W8P20kd_to_val" --ref_test "0.5*ens_W8P20kd_ft_test+0.5*ens_W8P20kd_to_test" --members $M --ft_suffix n2cpv3s3 > nn_x/w12/kd/out/gate_${tag}.txt 2>&1
echo "gate exit $?"; grep -E "^D val|M<50|M>=50|Tue|Thu|LEAK|^E |^F |GATE|FAIL|WARN" nn_x/w12/kd/out/gate_${tag}.txt | cut -c1-220; sha256sum subs/cand_${tag}_r4.parquet
