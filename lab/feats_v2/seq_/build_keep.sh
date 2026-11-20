#!/bin/bash
# Rebuild exactly the final seq_ KEEP family (26 features) for all splits.
set -e
cd <repo>/feats_v2/seq_
PY=<repo>/.venv/bin/python
export SEQ_KEEP_ONLY=1
$PY feat_hist.py val,test,train 16 E
$PY run_cv.py val,test,train 16 cv
$PY feat_rel.py val,test,train 16
