"""famB_review: combined read of the reduced-scale diagnostic replicates (rv_diag.py seed 11, rv_diag2.py seeds 12, 13).
Per arm: bundle = member-equal mean of per-feed z of the 3 replicate models (30K-feed fit samples); NATS over the W16 pool (own code, rv_eval), paired per-feed differences."""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB_review")
import numpy as np, rv_eval as E
O = "<repo>/nn_x/w12/cb/famB_review/out"
f = lambda arm, seed, sp: np.load(f"{O}/diag_{arm}_{sp}.npy" if seed == 11 else f"{O}/diag_{arm}_s{seed}_{sp}.npy")
wv, wh = E.load_witness("val"), E.load_witness("hold")
W = {"val": wv, "hold": wh}
res = {}; R = {}


def rd(tag, arm, seeds, hold_kind="hold"):
    R[tag] = {"val": E.read_model([f(arm, s, "val") for s in seeds], wv), "hold": E.read_model([f(arm, s, hold_kind) for s in seeds], wh)}
    v, h = R[tag]["val"], R[tag]["hold"]
    res[tag] = dict(seeds=list(seeds), val=dict(level=v["level"], nats=v["nats"], se=v["nats_se"]), hold=dict(level=h["level"], nats=h["nats"], se=h["nats_se"]))
    print(f"{tag} (replicates {list(seeds)}): level {v['level']:.5f} / {h['level']:.5f} | NATS {v['nats']:+.5f}+-{v['nats_se']:.5f} / {h['nats']:+.5f}+-{h['nats_se']:.5f}", flush=True)


def pair(a, b):
    o = {}
    for sp in ("val", "hold"):
        o[sp] = dict(nats=E.pse(R[a][sp]["nats_perfeed"] - R[b][sp]["nats_perfeed"]), level=E.pse(R[a][sp]["nd"] - R[b][sp]["nd"]))
    res[f"PAIR {a} - {b}"] = o
    print(f"PAIR {a} - {b}: NATS val {o['val']['nats'][0]:+.5f}+-{o['val']['nats'][1]:.5f} hold {o['hold']['nats'][0]:+.5f}+-{o['hold']['nats'][1]:.5f} | "
          f"level val {o['val']['level'][0]:+.5f}+-{o['val']['level'][1]:.5f} hold {o['hold']['level'][0]:+.5f}+-{o['hold']['level'][1]:.5f}", flush=True)


S3 = (11, 12, 13); S2 = (12, 13)
for arm in ("R", "A", "F", "K", "L", "D"):
    rd(f"{arm}x3", arm, S3)
for a, b in (("Ax3", "Rx3"), ("Fx3", "Rx3"), ("Kx3", "Rx3"), ("Lx3", "Rx3"), ("Ax3", "Fx3"), ("Ax3", "Kx3"), ("Kx3", "Fx3"), ("Lx3", "Ax3"), ("Dx3", "Rx3"), ("Fx3", "Dx3"), ("Ax3", "Dx3")):
    pair(a, b)
rd("Ax2", "A", S2); rd("Rx2", "R", S2); rd("Lx2", "L", S2); rd("Lx2_holdlag", "L", S2, hold_kind="holdlag")
pair("Lx2", "Ax2"); pair("Lx2_holdlag", "Ax2"); pair("Lx2_holdlag", "Rx2"); pair("Ax2", "Rx2")
# per replicate: L - A and K - F, to show the spread between fit samples
for s in S3:
    for arm in ("R", "A", "F", "K", "L"):
        R[f"{arm}{s}"] = {"val": E.read_model([f(arm, s, "val")], wv), "hold": E.read_model([f(arm, s, "hold")], wh)}
    pair(f"L{s}", f"A{s}"); pair(f"K{s}", f"R{s}"); pair(f"F{s}", f"R{s}"); pair(f"A{s}", f"R{s}")
json.dump(res, open(f"{O}/rv_diag_read.json", "w"), indent=1)
print("DONE")
