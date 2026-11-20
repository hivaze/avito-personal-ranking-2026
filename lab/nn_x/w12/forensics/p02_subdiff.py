import os; os.environ["POLARS_MAX_THREADS"]="12"
import sys; sys.path.insert(0,"<repo>/src")
import numpy as np, polars as pl
from common import *
R="<repo>/"
a=pl.read_parquet(R+"subs/cand_G3_later_overlay.parquet"); b=pl.read_parquet(R+"subs/cand_G_F10F7xF7x70_memeq.parquet")
print((a["feed_id"]==b["feed_id"]).all(), (a["item_id"]==b["item_id"]).all())
d=(a["score"]!=b["score"]).to_numpy(); print("rows differing",d.sum())
fd=np.unique(a["feed_id"].to_numpy()[d]); print("feeds differing",len(fd))
f,M,off=feed_meta("test"); ts=f["timestamp"].to_numpy(); fid=f["feed_id"].to_numpy()
pub=ts<1790497920
m=np.isin(fid,fd); print("public",(m&pub).sum(),"private",(m&~pub).sum())
ff=np.load("test_flagged_feeds.npy"); print("my flagged in differing:",np.isin(fid[ff],fd).sum(),len(ff))
# rank of flagged items in base vs overlay within differing feeds
sa=a["score"].to_numpy(); sb=b["score"].to_numpy()
# check sub row order equals flat order
cand=np.load(FLAT/"test/cand_item_id.npy"); print("row order = flat:", (a["item_id"].to_numpy()==cand).all(), (a["feed_id"].to_numpy()==np.repeat(fid,M)).all())
rk=[]
for i in np.nonzero(m)[0][:2000]:
    s0=sb[off[i]:off[i]+M[i]]; s1=sa[off[i]:off[i]+M[i]]
    ch=s0!=s1
    r0=(s0[:,None]<s0[None,:]).sum(0)[ch]; r1=(s1[:,None]<s1[None,:]).sum(0)[ch]
    rk.append((M[i],ch.sum(),r0.mean(),r1.mean()))
rk=np.array(rk); print("mean M, nflag, base rank(0=top), overlay rank:",rk.mean(0)); print(rk[:10])
print("sample sub rows in sample_submission order?")
s=pl.read_parquet(R+"data/competition/sample_submission.parquet"); print((s["feed_id"]==a["feed_id"]).all(),(s["item_id"]==a["item_id"]).all())
