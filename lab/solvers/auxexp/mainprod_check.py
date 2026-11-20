import os, sys, json
os.environ["AUX_THREADS"]="16"
sys.path.insert(0,"<repo>/solvers")
import CPU3_aux_head_member_W6 as C, numpy as np
V=C.Val()
ds=[]; rows={}
for m in C.MEMBERS:
    a=V.pf(np.load(C.rpath(m,"val","main_rl"))); b=V.pf(np.load(C.rpath(m,"val","main_prod")))
    d=b-a; ds.append(d); sl=V.slices(d); rows[m]=sl
    print(f"{m:16s} main_prod - main_rl: {C.fmt(sl)}")
d=np.mean(ds,0); sl=V.slices(d)
print("pooled over 12 members (per-feed mean of deltas):", C.fmt(sl))
nm=np.array([r["all"][0] for r in rows.values()]); nc=np.array([r["allcold"][0] for r in rows.values()])
print(f"member-level: all mean {nm.mean():+.5f} sd {nm.std(ddof=1):.5f} (+ in {int((nm>0).sum())}/12); allcold mean {nc.mean():+.5f} sd {nc.std(ddof=1):.5f} (+ in {int((nc>0).sum())}/12)")
print("allcold share of val feeds", V.allcold.mean())
json.dump({"per_member":rows,"pooled":sl},open("<repo>/solvers/out/CPU3_aux_head_member_W6/mainprod_vs_mainrl.json","w"),indent=1)
