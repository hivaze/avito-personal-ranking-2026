#!/usr/bin/env bash
# legacy_layout.sh on|off|status
#
# The lab tree was written for a flat layout in which nn/, nn_x/, reports/, data/, preds/ and subs/
# sat at the top of the repository, and most of its 2,500 scripts spell absolute paths into that
# layout. `on` recreates the old top-level names as relative symlinks into lab/ and artifacts/, so a
# historical script can run unchanged; `off` removes exactly those symlinks again. Nothing else is
# touched: the script refuses to remove anything that is not one of its own symlinks.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# old top-level name -> where it lives now (relative to the repository root)
LINKS=(
  "nn:lab/nn" "nn_x:lab/nn_x" "reports:lab/reports" "eda:lab/eda" "solvers:lab/solvers"
  "feats_v2:lab/feats_v2" "feat_new:lab/feat_new" "pretrain:lab/pretrain" "logs:lab/logs"
  "archive:lab/archive" "src:lab/src"
  "data:artifacts/data" "preds:artifacts/predictions" "subs:artifacts/submissions"
)

case "${1:-status}" in
  on)
    for pair in "${LINKS[@]}"; do
      name=${pair%%:*}; target=${pair#*:}
      if [ -L "$name" ]; then continue; fi
      if [ -e "$name" ]; then echo "refusing: $name exists and is not a symlink"; exit 1; fi
      [ -e "$target" ] || { echo "missing target $target"; exit 1; }
      ln -s "$target" "$name"
    done
    echo "legacy layout on" ;;
  off)
    for pair in "${LINKS[@]}"; do
      name=${pair%%:*}; target=${pair#*:}
      if [ -L "$name" ] && [ "$(readlink "$name")" = "$target" ]; then rm "$name"; fi
    done
    echo "legacy layout off" ;;
  status)
    for pair in "${LINKS[@]}"; do
      name=${pair%%:*}
      if [ -L "$name" ]; then echo "$name -> $(readlink "$name")"; fi
    done ;;
  *) echo "usage: $0 on|off|status"; exit 2 ;;
esac
