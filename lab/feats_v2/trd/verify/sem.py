import numpy as np
from pathlib import Path
F=Path("<repo>/data/feats"); FL=Path("<repo>/data/flat")
names=sorted(p.stem for p in (F/"val").glob("trd_*.npy"))
for sp in ["train","test"]:
    s=sorted(p.stem for p in (F/sp).glob("trd_*.npy")); assert s==names, (sp, set(s)^set(names))
print("name sets identical across splits:", len(names))
ytr=np.load(FL/"train/cand_target.npy", mmap_mode="r"); n=len(ytr)
rng=np.random.default_rng(1); idx=np.sort(rng.choice(n,4_000_000,replace=False)); yt=np.asarray(ytr[idx]).astype(float)
yv=np.load(FL/"val/cand_target.npy").astype(float)
print("pos rate train sample", yt.mean().round(4), "val", yv.mean().round(4))
for nm in ["trd_kde_sday","trd_kde_e1_sday","trd_n85_sday","trd_max_sday","trd_skde_sday","trd_dsr_sday","trd_kde_s6h","trd_dsr_s6h"]:
    v=np.load(F/"val"/f"{nm}.npy"); t=np.load(F/"train"/f"{nm}.npy",mmap_mode="r")[idx]
    qs=np.nanquantile(v,np.linspace(0,1,6)); qs[0]=-np.inf; qs[-1]=np.inf
    bv=np.digitize(v,qs[1:-1]); bt=np.digitize(t,qs[1:-1])
    lv=[yv[bv==b].mean()/yv.mean() for b in range(5)]; lt=[yt[bt==b].mean()/yt.mean() for b in range(5)]
    print(f"{nm:18s} val lift by quintile {np.round(lv,2)}  train lift {np.round(lt,2)}  train share/bin {np.round(np.bincount(bt,minlength=5)/len(bt),2)}")
