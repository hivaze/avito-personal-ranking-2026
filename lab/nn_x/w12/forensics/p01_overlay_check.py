import os; os.environ["OMP_NUM_THREADS"]="12"; os.environ["POLARS_MAX_THREADS"]="12"
import sys; sys.path.insert(0,"<repo>/src")
import numpy as np, polars as pl
from common import *
from collections import Counter
R="<repo>/"
a=pl.read_parquet(R+"subs/cand_G3_later_overlay.parquet"); 
print(a.schema, a.height)
import glob
g=[p for p in glob.glob(R+"subs/*.parquet") if "G_F10F7xF7x70" in p or "G3_final" in p]; print(g)
for split in ["val","test"]:
    f,M,off=feed_meta(split); print(split, f.columns)
    uid=f["user_id"].to_numpy(); ts=f["timestamp"].to_numpy()
    hf=np.load(FLAT/f"{split}/hist_fidx.npy"); hi=np.load(FLAT/f"{split}/hist_item_id.npy")
    het=np.load(FLAT/f"{split}/hist_event_type.npy"); hes=np.load(FLAT/f"{split}/hist_event_source.npy"); hp=np.load(FLAT/f"{split}/hist_hpos.npy")
    hoff=f["hist_off"].to_numpy(); H=f["H"].to_numpy()
    cand=np.load(FLAT/f"{split}/cand_item_id.npy")
    y=np.load(FLAT/f"{split}/cand_target.npy") if split=="val" else None
    by={}
    for i in np.nonzero(uid!=0)[0]: by.setdefault(int(uid[i]),[]).append(i)
    cnt=Counter(); nfeeds=0; nflag=0; pos=0; npairs=0; newev=[]; flagfeeds=[]
    for u,l in by.items():
        if len(l)<2: continue
        l=sorted(l,key=lambda i:ts[i])
        for x in range(len(l)):
            for z in range(x+1,len(l)):
                A,B=l[x],l[z]
                if ts[B]<=ts[A]: continue
                npairs+=1
                ca=cand[off[A]:off[A]+M[A]]
                hb=hi[hoff[B]:hoff[B]+H[B]]
                m=np.isin(hb,ca)
                # number of new events in B relative to A: find A's first hist item in B
                ha=hi[hoff[A]:hoff[A]+H[A]]
                if m.any():
                    flagfeeds.append(A)
                    for k in np.nonzero(m)[0]:
                        lab = int(y[off[A]+np.nonzero(ca==hb[k])[0][0]]) if y is not None else -1
                        cnt[(int(het[hoff[B]+k]),int(hes[hoff[B]+k]),lab)]+=1
    print(split,"pairs",npairs,"feeds flagged",len(set(flagfeeds)))
    agg=Counter()
    for (et,es,lab),c in cnt.items(): agg[(et,es if es<=2 else 9,lab)]+=c
    for k in sorted(agg): print("  et,src,label",k,agg[k])
    if split=="test":
        pub=ts<1790497920
        ff=np.array(sorted(set(flagfeeds))); print(" flagged feeds public",pub[ff].sum(),"private",(~pub[ff]).sum())
        np.save("test_flagged_feeds.npy",ff)
