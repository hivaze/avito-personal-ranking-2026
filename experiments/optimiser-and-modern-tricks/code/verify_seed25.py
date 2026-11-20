# Replication on a fresh seed. Compares the two adopted changes trained with seed 25 (gate + retrieval; Muon
# + cooldown schedule) with a baseline of seed 25 that had been registered before either was judged, then
# pools seeds 23, 24 and 25. The holdout z-score uses the run-to-run sd 0.00085 measured in recheck.py.
# Run on CPU from lab/nn_x/verify after run_a.sh and run_b.sh (the two GPU chains) had finished; the command
# line is not on record and its printed output was not saved. Its numbers are in lab/reports/nnx_VERDICT.md.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/verify/s25.py
#
"""Fresh seed-25 confirmation: paired per-feed deltas vs the pre-registered shared baseline feat_base_s25, then 3-seed pooling."""
import sys, numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
FL = "<repo>/data/flat"
f = pl.read_parquet(f"{FL}/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
y = np.load(f"{FL}/val/cand_target.npy").astype(np.float64)
O, A, F, V = ("<repo>/preds_x/" + g for g in ("optim", "attn", "feat", "verify"))
def pv(p): return ndcg_per_feed(np.load(p + "_trainonly_val.npy"), y, off, M)
def ph(p):
    h = np.load(p + "_holdout.npz"); Mh = h["M"].astype(np.int64); oh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
    return ndcg_per_feed(h["scores"], h["y"], oh, Mh)
TRAIN_SD_HOLD = 0.00085  # run-to-run sd of a holdout delta between same-config seeds (null calibration)
def report(name, pairs):
    for k, fn in (("val", pv), ("hold", ph)):
        D = [fn(t) - fn(b) for b, t in pairs]
        per = [d.mean() for d in D]
        d = np.mean(D, 0); se = d.std(ddof=1) / np.sqrt(len(d))
        extra = ""
        if k == "hold":
            sec = np.sqrt(se ** 2 + TRAIN_SD_HOLD ** 2 / len(D)); extra = f" z_cal={d.mean()/sec:+.2f}"
        print(f"{name:40s} {k:4s} n={len(D)} d={d.mean():+.5f} se={se:.5f} z={d.mean()/se:+.2f}{extra} per-seed {['%+.5f' % x for x in per]}")
s25b, s25s2 = F + "/nn_feat_base_s25", V + "/nn_s2_feat_base_s25"
print("seed-25 baseline: val %.5f hold %.5f | stage 2: val %.5f hold %.5f" % (pv(s25b).mean(), ph(s25b).mean(), pv(s25s2).mean(), ph(s25s2).mean()))
report("A1+A4 vs base (s25 only)", [(s25b, V + "/nn_attn_gate_retr_s25")])
report("A1+A4 vs base (s23,s24,s25)", [(O + "/nn_base_s23", A + "/nn_attn_gate_retr_s23"), (O + "/nn_base_s24", A + "/nn_attn_gate_retr_s24"), (s25b, V + "/nn_attn_gate_retr_s25")])
report("MuonWSD vs base+s2 (s25 only)", [(s25s2, V + "/nn_optim_muwsd_s25")])
report("MuonWSD vs base (s25 only)", [(s25b, V + "/nn_optim_muwsd_s25")])
report("MuonWSD vs base+s2 (s23,s24,s25)", [(O + "/nn_s2_base_s23", O + "/nn_optim_muwsd_s23"), (O + "/nn_s2_base_s24", O + "/nn_optim_muwsd_s24"), (s25s2, V + "/nn_optim_muwsd_s25")])
report("stage2 on base (s25)", [(s25b, s25s2)])
