import sys, numpy as np, polars as pl
sys.path.insert(0, "<repo>/feats_v2/trd"); sys.path.insert(0, "<repo>/src")
from trd_lib import psi, single_ndcg
from pathlib import Path
F = Path("<repo>/data/feats"); FL = Path("<repo>/data/flat")
names = sorted(p.stem for p in (F/"val").glob("trd_*.npy"))
MSK=3*3600; PUB=1790497920
meta = {}
for sp in ["train","val","test"]:
    f = pl.read_parquet(FL/sp/"feeds.parquet"); ts=f["timestamp"].to_numpy().astype(np.int64)
    cf = np.load(FL/sp/"cand_fidx.npy", mmap_mode="r")
    meta[sp] = (len(cf), ts, f)
# row-level timestamp for val/test
def row_ts(sp):
    n, ts, f = meta[sp]; cf = np.load(FL/sp/"cand_fidx.npy"); return ts[cf]
tv = row_ts("val"); tt = row_ts("test")
hv = ((tv+MSK)%86400)//3600; ht = ((tt+MSK)%86400)//3600
dt_ = (tt+MSK)//86400; dow = (dt_+3)%7  # 1970-01-01 Thu -> dow 0=Mon
wk_t = dow>=5
pub = tt < PUB
edge_v = (hv<3)|(hv>=21)
# test: truncated edges only Sat 00-03 and Tue 21-24
d0 = dt_.min(); d1 = dt_.max()
edge_t_trunc = ((dt_==d0)&(ht<3))|((dt_==d1)&(ht>=21))
edge_t_full = ((ht<3)|(ht>=21)) & ~edge_t_trunc
rng = np.random.default_rng(0)
ntr = meta["train"][0]; samp = np.sort(rng.choice(ntr, 3_000_000, replace=False))
y = np.load(FL/"val/cand_target.npy"); fv = pl.read_parquet(FL/"val/feeds.parquet")
M = fv["M"].to_numpy(); off = fv["cand_off"].to_numpy()
rows=[]
for nm in names:
    a = {sp: np.load(F/sp/f"{nm}.npy", mmap_mode="r") for sp in ["train","val","test"]}
    ok = all(len(a[sp])==meta[sp][0] and a[sp].dtype==np.float32 for sp in a)
    v = np.asarray(a["val"]); t = np.asarray(a["test"]); tr = np.asarray(a["train"][samp])
    nan = [float(np.isnan(x).mean()) for x in (tr,v,t)]
    nun = [len(np.unique(x[np.isfinite(x)][:2_000_000])) for x in (tr,v,t)]
    q = lambda x: np.nanquantile(x,[.1,.5,.9,.99])
    sa, sd = single_ndcg(v, y, off, M)
    r = dict(f=nm, ok=ok, nan_tr=nan[0], nan_v=nan[1], nan_t=nan[2], nuniq_min=min(nun),
             psi_v_pub=psi(v,t[pub]), psi_v_priv=psi(v,t[~pub]), psi_pub_priv=psi(t[pub],t[~pub]),
             psi_wkday_wkend=psi(t[~wk_t],t[wk_t]), psi_v_tr=psi(v,tr),
             psi_edge_v_vs_testfull=psi(v[edge_v], t[edge_t_full]), psi_int_v_vs_t=psi(v[~edge_v], t[~((ht<3)|(ht>=21))]),
             med_edge_v=float(np.nanmedian(v[edge_v])), med_edge_tfull=float(np.nanmedian(t[edge_t_full])), med_int_v=float(np.nanmedian(v[~edge_v])),
             q_tr=np.round(q(tr),4).tolist(), q_v=np.round(q(v),4).tolist(), q_pub=np.round(q(t[pub]),4).tolist(), q_priv=np.round(q(t[~pub]),4).tolist(),
             nd_asc=sa, nd_desc=sd)
    rows.append(r); print(r, flush=True)
import json; json.dump(rows, open(sys.argv[1],"w"), indent=1)
print("edge share val", edge_v.mean(), "test trunc", edge_t_trunc.mean(), "test full-edge", edge_t_full.mean())
