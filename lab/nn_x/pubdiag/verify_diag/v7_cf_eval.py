"""Evaluate the SHIFT-2 replay: TO-pool deltas on val with base vs notest it_hist_*_luo inputs."""
from v_lib import *
I = D / "infer_cf"
g = groups_val()
roots = range(101, 108)
FM = {"F5cd": FAM["F5cd"], "F6": FAM["F6"], "F7": FAM["F7"], "F7b": FAM["F7b"], "F7x": FAM["F7x"], "F7x70": FAM["F7x70"]}
def pool(fams, mode):
    ms = [m for f in fams for m in FM[f]]
    return np.mean([z("val", np.load(I / f"{m}_{mode}_val.npy")) for m in ms], 0)
R = {}
for mode in ["base", "cf"]:
    for k, fams in {"Cd": ["F5cd", "F6", "F7"], "M2": ["F7x", "F7x70"], "F7": ["F7"], "F7b": ["F7b"], "F7x": ["F7x"], "F7x70": ["F7x70"],
                    "G3noF10": ["F7x", "F7x70"]}.items():
        R[(k, mode)] = pf("val", pool(fams, mode))
for k in ["Cd", "M2", "F7", "F7b", "F7x"]:
    d = R[(k, "cf")] - R[(k, "base")]
    print(f"{k:4s} TO pool: base {R[(k,'base')].mean():.5f} notest {R[(k,'cf')].mean():.5f} shift {d.mean():+.5f}+-{se_clu(d,g):.5f}")
print()
for a, b in [("M2", "Cd"), ("F7b", "F7"), ("F7x", "F7b"), ("F7x", "F7"), ("M2", "F7")]:
    db = R[(a, "base")] - R[(b, "base")]; dc = R[(a, "cf")] - R[(b, "cf")]; dd = dc - db
    print(f"{a}-{b} TO: base {db.mean():+.5f}+-{se_clu(db,g):.5f}  notest {dc.mean():+.5f}+-{se_clu(dc,g):.5f}  change {dd.mean():+.5f}+-{se_clu(dd,g):.5f}")
# per-member root-paired check
for fa, fb in [("F7b", "F7"), ("F7x", "F7")]:
    ch = []
    for i in range(7):
        ma, mb = FM[fa][i], FM[fb][i]
        x = {mode: pf("val", np.load(I / f"{ma}_{mode}_val.npy")) - pf("val", np.load(I / f"{mb}_{mode}_val.npy")) for mode in ["base", "cf"]}
        ch.append((x["cf"] - x["base"]).mean())
    print(f"{fa}-{fb} member change under notest per root: {np.round(ch,5).tolist()}  mean {np.mean(ch):+.5f}")
# consistency: base inference vs saved preds
mx = max(float(np.abs(np.load(I / f"{m}_base_val.npy") - np.load(member_file(m, 'val', 'to'))).max()) for f in FM for m in FM[f])
print("max |base replay - saved trainonly val| over 40 members:", mx)
