# The rule for the three CatBoost experiments of lab/nn_x/cb26, written before any of them ran, and
# the function that applies it. The docstring is the rule: a component counts as useful only if
# adding it to the 20-member neural pool gains more than 2 SE on val, does not lose on the holdout,
# and has no val slice below -2 SE and no loss at the private feed mix.
# How it was used: imported by eval_cb.py in the same lab folder (usage: eval_cb.py <pool> <tags>).
# The command lines are not on record; the outputs are logs/eval_e1e2.log and logs/eval_e3.log.
# What it produced: the verdict lines of results/cb26_eval_e1e2.log and results/cb26_eval_e3.log.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/cb26/cbrule.py
"""cb26 PRE-DECLARATION + acceptance-rule implementation.
Written 2026-11-12 14:10:29 UTC, BEFORE any experiment in nn_x/cb26 was run. The docstring is the rule.

Question (user): did we try adding tuned CatBoost solvers? What is still possible?

Reference pools
- PRIMARY reference: G3 = F10+F7x+F7x70 (20 members), preds/ens_G3_{ft,to}_{val,test}.npy, pool = 0.5 z(ft) + 0.5 z(to).
  Val 0.40032, holdout (train-only member z-mean) 0.32379.
- Replication reference: PF7x (38 members). Used only if something passes on G3.
- Holdout pool = member-equal z-mean of the same members' preds/holdout/nn_<m>_holdout.npy (train-only).

Experiments (bounded, in this order)
E1 HR: holdout-trained residual CatBoost.
   - Training data: the 30K random-like train holdout (trH). It is the only labelled set besides val on which the train-only members are honest.
   - Model: CatBoost GPU with baseline = beta * z(pool_to). beta comes from a 1-D conditional logit on the training rows.
   - Features: z(pool_to), member disagreement (std of member z), family z-means, M, H, user0, and leak-safe, parity-OK, train-available tabular features.
     Excluded: non-LUO it_hist, it_tr_*, u_tr_*, it_cand_cnt_* (INVENTORY-2 parity break), tte_* (gap mismatch), and pre_*/als_* (quarantined / not on train).
   - Folds: 5 user-grouped folds on trH (user 0 as singletons). Each fold model trains on 3 folds, early-stops on 1 and predicts the 5th, giving an honest holdout OOF.
   - Val and test = mean over the 5 fold models of Delta(x). Val is never used for fitting or stopping.
   - PRIMARY E1 score: S = beta*z(pool) + Delta(x), where pool = the full 0.5/0.5 mix.
   - Secondary: to-branch replacement, 0.5 z(ft) + 0.5 z(beta*z(to) + Delta).
E2 VR: val-nested residual CatBoost on the full pool (task 1a).
   - Same feature design, plus z(ft) and z(to) separately.
   - Folds: the n2cpv3 folds (GroupKFold(5) on user_id, user 0 as singletons).
   - Each outer fold k: train on 3 folds, early-stop on 1, predict fold k. Test = pool_test + mean over folds of Delta.
E3 XM: CatBoost member trained on the train-period exploration subset (task 1b).
   - Data: p_random > 0.5 and M >= 50 train feeds, holdout excluded. Features: the strongest leak-safe tabular set.
   - Training: YetiRank on GPU, early-stopped on user-half A of val, reported on half B.
   - Evaluation: val, holdout, and eq-gain with the pool at pre-declared weights w in {0.05, 0.1, 0.2} (on per-feed z), plus the strength-diversity-law prediction.
Optional, only if E1 or E2 shows a positive signal > 1 SE: E1+E2 combined (trH + val train folds).

ACCEPTANCE RULE (applies to every CatBoost component). A component is USEFUL only if ALL hold:
 1. Adding it to the pool improves nested/OOF val NDCG vs the reference pool by > 2 SE.
    This is the paired per-feed delta over all 14,423 val feeds, SE = std/sqrt(n).
 2. When holdout predictions exist (E1 OOF on trH, E3 on trH), it does not hurt the holdout: paired delta >= 0.
    A negative point estimate fails.
 3. No val slice < -2 SE. Slices: M<50, M>=50, Tue, Thu, user0 / known user, and feed cold-share terciles.
 4. The pmix private-mix delta (src/pmix.py; report-only + veto, CURRENT_STATUS 12:42) is not < -1 SE, if computable.
 Further rules:
 - E3 "adding it" = the equal z-mix at the best pre-declared weight. Because that weight is selected, it must clear 2 SE at the
   selected weight AND be > 0 at w=0.05.
 - With more than 2 arms in an experiment, a pass at only 2-2.5 SE is reported as "not robust".
 - Nothing is chosen on public LB.

PRIOR (stated before running): E2 and E3 are expected to fail. History: stackers were null on v11, GBDT members sit at
0.366-0.370, the break-even at rho ~0.8 is ~0.388, and the residual feature scan (INVENTORY-3) found nothing > 2 SE.
E1 is the only untried design: about 3x the training feeds of val-only stacking, a clean val evaluation and a new training
domain. Its prior is low, about 15%.
"""
import sys
import numpy as np
import polars as pl
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402

DATA = "<repo>/nn_x/cb26/data"


def pse(d):
    d = np.asarray(d, np.float64)
    return float(d.mean()), float(d.std() / np.sqrt(len(d)))


def val_slices():
    """Boolean masks over val feeds (label-free descriptors)."""
    f, M, off = feed_meta("val")
    ts = f["timestamp"].to_numpy()
    day = (ts + 10800) // 86400
    uid = f["user_id"].to_numpy()
    sl = {"M<50": M < 50, "M>=50": M >= 50, "Tue": day < 20720, "Thu": day >= 20720, "user0": uid == 0, "known": uid != 0}
    try:
        cs = np.load(f"{DATA}/val_cold_share.npy")
        q = np.quantile(cs, [1 / 3, 2 / 3])
        sl["cold_lo"] = cs <= q[0]; sl["cold_mid"] = (cs > q[0]) & (cs <= q[1]); sl["cold_hi"] = cs > q[1]
    except FileNotFoundError:
        pass
    return sl


def judge(name, d_val, d_hold=None, d_pmix=None, extra_cond=True):
    """d_val: per-feed val delta (component-added minus reference). Returns (pass, text)."""
    m, s = pse(d_val)
    ok1 = m > 2 * s
    lines = [f"[{name}] val delta {m:+.5f} +- {s:.5f} (z {m / s:+.2f}) -> rule1 {'OK' if ok1 else 'FAIL'}"]
    ok2 = True
    if d_hold is not None:
        hm, hs = pse(d_hold)
        ok2 = hm >= 0
        lines.append(f"   holdout delta {hm:+.5f} +- {hs:.5f} (z {hm / hs:+.2f}) -> rule2 {'OK' if ok2 else 'FAIL'}")
    ok3 = True
    for k, msk in val_slices().items():
        sm, ss = pse(d_val[msk])
        bad = sm < -2 * ss
        ok3 &= not bad
        lines.append(f"   slice {k:9s} n={msk.sum():5d} {sm:+.5f} +- {ss:.5f}{'  <-- < -2SE' if bad else ''}")
    ok4 = True
    if d_pmix is not None:
        pm, ps = d_pmix
        ok4 = pm >= -ps
        lines.append(f"   pmix private {pm:+.5f} +- {ps:.5f} -> rule4 {'OK' if ok4 else 'VETO'}")
    ok = ok1 and ok2 and ok3 and ok4 and extra_cond
    lines.append(f"   ==> {'USEFUL (passes)' if ok else 'NOT useful (fails)'}")
    return ok, "\n".join(lines)
