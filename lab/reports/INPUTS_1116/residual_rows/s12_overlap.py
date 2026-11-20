"""INVENTORY B, step 12: do the three collaborative constructions carry the SAME unread information?  Val only (ALS and the graph walks have no
holdout arrays).  Conditional score z of each column given pool + the ranker's own inputs, and given that plus another family (z and rank form).
Families: RP3 = C09 p3a, rp3b05 | WALK = int_gnn walk_train columns 0, 1 | ALS = als_exp5, als_exp5_rk, als_last10 | DENS = kn_cos1, kn_cos20.
Output: out/overlap.txt.  CPU only, read-only."""
import sys
sys.path.insert(0, "<repo>/reports/INPUTS_1116/residual_rows")
sys.argv = [sys.argv[0], "val", "TO16"]
from lens import *
import cand_specs as cs, condlib
SP = {f"{g}:{n}": s for g, n, s in cs.specs()}
D = load_split("val", "TO16"); condlib.sp = "val"; nm, cl = condlib.control_block(D)
def col(k):
    if k.startswith("als"):
        x = np.load(ROOT / f"data/feats/val/{k}.npy", mmap_mode="r"); return forms(np.asarray(x, np.float64), D)
    return forms(cs.load(SP[k]["val"], D), D)
FAM = {"RP3": ["ccb:c09_p3a", "ccb:c09_rp3b05"], "WALK": ["lit_gnn:walk_train_c0", "lit_gnn:walk_train_c1"], "ALS": ["als_exp5", "als_exp5_rk", "als_last10"],
       "DENS": ["cbB_c:kn_cos1", "cbB_c:kn_cos20"], "BT": ["ccb:c06_theta_l3"], "XT": ["itemprior:ip_x_val_T"], "S12": ["codes:ex_X3_s1s2_lf", "codes:ex_n_s2_s1s2_lf"], "GAP": ["timegap:tg_gap_first"]}
F = {k: col(k) for v in FAM.values() for k in v}
out = open(OUT / "overlap.txt", "w")
def P(s): print(s, flush=True); out.write(s + "\n")
N0 = condlib.Null(D, nm, cl)
P("val, train-only pool. cz (z form / rank form) of each column given pool + the ranker's own inputs [+ the family named in the column head]")
heads = ["-", "RP3", "WALK", "ALS", "DENS", "BT", "S12"]
NU = {"-": N0}
for h in heads[1:]:
    extra = [F[k][f] for k in FAM[h] for f in ("z", "rk")]
    NU[h] = condlib.Null(D, nm + [f"{h}{i}" for i in range(len(extra))], cl + extra)
P(f"{'column':26s} " + " ".join(f"{('given +' + h) if h != '-' else 'given inputs':>16s}" for h in heads))
for fam, ks in FAM.items():
    for k in ks:
        line = f"{k:26s} "
        for h in heads:
            if h == fam: line += f"{'(in block)':>16s} "; continue
            a = NU[h].score(F[k]["z"])["cz"]; b = NU[h].score(F[k]["rk"])["cz"]; line += f"{a:+7.2f} /{b:+6.2f}  "
        P(line)
