"""Cross-fitted tests of the first candidate rules (params chosen on 4 folds, evaluated on the 5th)."""
import itertools
from lib import *

D = Data()
xp = D.col("xp_imp_c50"); f4 = D.col("f4"); f5 = D.col("f5"); H = D.col("H"); cat = D.col("cat")
guest = D.col("guest") > 0; uvm = D.col("uvec_missing") > 0
cur = D.col("cat_user_rank")
s = D.s
NV = 0


def run(name, gens, ws):
    """gens: dict label -> g array; grid = all (label, w)."""
    global NV
    grid = [(k, w) for k in gens for w in ws]
    NV += len(grid)
    r = D.crossfit(lambda p: s + p[1] * gens[p[0]] + 1e-7 * s, grid)
    best = max(r["insample"], key=lambda t: t[1])
    print(f"{name:28s} CF {r['cf']:+.5f} se {r['se']:.5f} z {r['z']:5.2f} | in-sample best {best[0]} {best[1]:+.5f} ({best[2]:.5f}) | chosen {r['chosen']}")
    return r


W_NEG = [-0.05, -0.1, -0.2, -0.3, -0.5, -0.8, -1.2]
W_POS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
# R1 fresh in weak-signal feeds
gens = {}
for t, wk in itertools.product([0.05, 0.1, 0.15, 0.2], ["guest", "guest|uvm", "guest|uvm|H<30", "H<30"]):
    weak = {"guest": guest, "guest|uvm": guest | uvm, "guest|uvm|H<30": guest | uvm | (H < 30), "H<30": H < 30}[wk]
    gens[(t, wk)] = ((f4 < t) & weak).astype(float)
run("R1 fresh x weak feed", gens, W_NEG)
for k, g in list(gens.items())[:0]:
    pass
# R1 single fixed form for reference
run("R1a fresh<.1 guest only", {"a": ((f4 < 0.1) & guest).astype(float)}, W_NEG)
run("R1b fresh<.1 guest|uvm", {"a": ((f4 < 0.1) & (guest | uvm)).astype(float)}, W_NEG)
# R2 fresh in vacancies/services
gens = {(t, cs): ((f4 < t) & np.isin(cat, cs)).astype(float) for t in [0.05, 0.1, 0.2] for cs in [(48,), (50,), (48, 50)]}
run("R2 fresh x cat48/50", gens, W_NEG)
run("R2a fresh<.1 cat48|50", {"a": ((f4 < 0.1) & np.isin(cat, [48, 50])).astype(float)}, W_NEG)
# R3 low engagement at high exposure
gens = {(a, b, seg): ((f5 < a) & (xp > b) & m).astype(float) for a in [0.1, 0.2, 0.3] for b in [2.5, 3.0, 3.5]
        for seg, m in [("all", np.ones(len(s), bool)), ("re10|11", np.isin(cat, [10, 11]))]}
run("R3 f5 low & xp high", gens, W_NEG)
run("R3a f5<.2&xp>3 all", {"a": ((f5 < 0.2) & (xp > 3)).astype(float)}, W_NEG)
run("R3b f5<.2&xp>3 cat10|11", {"a": ((f5 < 0.2) & (xp > 3) & np.isin(cat, [10, 11])).astype(float)}, W_NEG)
# R4 cold in the user's top category
gens = {(cr,): ((xp == 0) & np.isin(cur, cr)).astype(float) for cr in [(1,), (1, 2), (1, 2, 3)]}
run("R4 cold x top cat", gens, W_POS)
run("R4a cold&cat_user_rank1", {"a": ((xp == 0) & (cur == 1)).astype(float)}, W_POS)
# R5 fresh in cold-heavy feeds
cs = D.feedwise((xp == 0).astype(float), np.mean)
gens = {(t, c): ((f4 < t) & (cs > c)).astype(float) for t in [0.1, 0.2] for c in [0.4, 0.5, 0.6]}
run("R5 fresh x cold-heavy feed", gens, W_NEG)
print("variants so far", NV)
