import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","POLARS_MAX_THREADS"): os.environ[k]="12"
os.environ["CUDA_VISIBLE_DEVICES"]=""
import sys; sys.path.insert(0,"/tmp/scratchpad/dpa/validation")
import numpy as np, polars as pl
from lib import meta, day, holdout_index, _HOLD
f, M, off = meta("train")
print(f.columns)
d = day(f["timestamp"].to_numpy())
u, c = np.unique(d, return_counts=True)
m50 = M >= 50
for a, b in zip(u, c):
    s = d == a
    print(a, b, "M>=50", int((s & m50).sum()), "M<50", int((s & ~m50).sum()))
hold = holdout_index()
p = _HOLD["p_random"]; pool = _HOLD["pool134"]
late = d >= 20713
print("holdout late", int(late[hold].sum()), "pool134 late", int(late[pool].sum()))
print("overlap hold/pool", len(np.intersect1d(hold, pool)))
# exploration share by day among all feeds (p_random>0.9 among M>=50)
pr = np.load("<repo>/eda/labels_position/out/train_prandom.npz")
praw = pr["p_random"].astype(np.float32)
print("p_random NaN share (all)", float(np.isnan(praw).mean()), "M<50 NaN share", float(np.isnan(praw[~m50]).mean()))
for a in u:
    s = (d == a) & m50
    if s.sum()==0: continue
    pp = np.nan_to_num(praw[s], nan=-1)
    print(a, "M>=50 feeds", int(s.sum()), "p>0.9", int((pp>0.9).sum()), "p<0.1", int(((pp>=0)&(pp<0.1)).sum()), "nan", int((pp<0).sum()))
# user ids of late eval feeds vs users in val/test
uid = f["user_id"].to_numpy()
ev = np.union1d(hold[late[hold]], pool[late[pool]])
print("T5 eval feeds", len(ev), "distinct users", len(np.unique(uid[ev])), "uid0", int((uid[ev]==0).sum()))
fv,_,_ = meta("val"); ft,_,_ = meta("test")
later = set(fv["user_id"].to_numpy().tolist()) | set(ft["user_id"].to_numpy().tolist())
ue = uid[ev]
print("T5 eval feeds whose user has a val/test feed (later snapshot exists)", int(np.isin(ue[ue!=0], np.array(sorted(later))).sum()), "of", int((ue!=0).sum()))
# users of T5 eval with LATER train feeds (days > feed day)
fd = d
import collections
