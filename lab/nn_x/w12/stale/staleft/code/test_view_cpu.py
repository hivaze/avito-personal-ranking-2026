"""CPU sanity test of the stale view (no GPU, no label read): builds the fresh and the stale val SplitData exactly as
finetune_stale.py does and checks which model-scale columns differ + the feed-level mix on one batch."""
import os, sys, copy
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
torch.set_num_threads(12)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, "<repo>/nn_x/w12/stale/staleft")
import data as Dm
from stale_map import STALE_XE, STALE_XT, N2C_PRE_V3, XT_NAMES
STALE = {"on": None}
_fp = Dm.feat_path
def fp(split, name):
    if STALE["on"] is not None and split == "val" and name in STALE["on"]:
        return STALE["on"][name]
    return _fp(split, name)
Dm.feat_path = fp
dev = torch.device("cpu")
va = Dm.SplitData("val", dev, max_hist=256)
va.attach_xt(XT_NAMES); st = va.attach_extra(N2C_PRE_V3)
vs = copy.copy(va)
STALE["on"] = STALE_XT; vs.xt = None; vs.attach_xt(XT_NAMES)
STALE["on"] = STALE_XE; vs.xe = None; vs.attach_extra(N2C_PRE_V3, st)
STALE["on"] = None
assert va.xt is not vs.xt and va.xe is not vs.xe and va.c_emb is vs.c_emb
dx = (vs.xt.float() - va.xt.float()).abs().mean(0).numpy(); de = (vs.xe - va.xe).abs().mean(0).numpy()
print("xt columns changed:", [(n, round(float(d), 4)) for n, d in zip(XT_NAMES, dx) if d > 0])
print("xe columns changed:", [(n, round(float(d), 4)) for n, d in zip(N2C_PRE_V3, de) if d > 0])
print("xe switched but unchanged:", [n for n, d in zip(N2C_PRE_V3, de) if d == 0 and n in STALE_XE])
print("rows with any xt change", float(((vs.xt != va.xt).any(1)).float().mean()), "any xe change", float(((vs.xe != va.xe).any(1)).float().mean()))
fb = np.arange(40, 72)
b = va.gather(fb)
rng_s = np.random.default_rng(777000)
sm = rng_s.random(len(fb)) < 0.5
smt = torch.from_numpy(sm).to(dev)[:, None, None]
xt2 = torch.where(smt, vs.xt[b["ci"]].float(), b["xt"]); xe2 = torch.where(smt, vs.xe[b["ci"]], b["xe"])
bs = vs.gather(fb)
assert torch.equal(xt2[sm], bs["xt"][sm]) and torch.equal(xt2[~sm], b["xt"][~sm]) and torch.equal(xe2[sm], bs["xe"][sm]) and torch.equal(xe2[~sm], b["xe"][~sm])
print("feed-level mix on one batch: OK;", int(sm.sum()), "of", len(fb), "feeds stale; shapes", tuple(b["xt"].shape), tuple(b["xe"].shape))
