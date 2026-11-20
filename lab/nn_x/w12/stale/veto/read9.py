"""veto read 9 (resume run): reverse-check items added from the component ledger. Rule: PREDECLARED.txt item 4 + ADDENDUM R2.
Output: logs/read9.txt (stdout) + out/read9.json"""
import json
import sys
sys.path.insert(0, "<repo>/nn_x/w12/stale/veto")
from vlib import *

c = Ctx(); R = []
rng = lambda p, a, b: [f"{p}{i}" for i in range(a, b + 1)]


def go(label, A, B, **kw):
    miss = [v for v in flat(A) + flat(B) if find(v, "val") is None]
    if miss:
        print(f"== {label}: MISSING val files {miss}"); return None
    r = read(c, label, A, B, kinds=("val", "hold"), **kw); show(r); R.append(r); return r


F7 = rng("v17_pre2s_s", 101, 107); F9 = rng("v22_pre2s_lr1_s", 401, 406); F12 = rng("v26_d192_pre2s_s", 101, 107); F10 = rng("v23_gru_pre2s_s", 501, 506)
import glob, os
f12 = sorted(os.path.basename(p)[3:-18] for p in glob.glob(str(ROOT / "preds/nn_v26_d192_pre2s_s*_trainonly_val.npy")))
print("F12 members on disk:", f12)
print("##### (i) lr 2e-3 (F7) vs lr 1e-3 (F9), 10 epochs, hard labels, mix 0.1")
go("F7 x6 (lr 2e-3) vs F9 x6 (lr 1e-3)", F7[:6], F9)
go("F7 x7 vs F9 x6 (all members)", F7, F9)
print("##### (ii) hard-label width: F12 (d192) vs F7 (d128)")
go(f"F12 x{len(f12)} (d192, hard) vs F7 x{len(f12)}", f12, F7[:len(f12)])
print("##### (iii) GRU family F10 vs F7")
go("F10 x6 (GRU) vs F7 x6", F10, F7[:6])
json.dump(R, open(HERE / "out/read9.json", "w"), indent=1, default=float)
