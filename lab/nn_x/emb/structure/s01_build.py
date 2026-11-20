"""Build per-item aggregates for the TRAIN embedding table and draw the item sample used by the probes.

Per emb_idx (train table, N = 81,557,944): n_cand (train candidate impressions), n_pos (train clicks, production traffic),
mean f1..f5 over candidate rows, category (cand, else history), history-event counts per event_type 1..10,
a representative candidate row -> feed time and f4 -> posting time t_post (f4 = exact age, feats_v2/life_/post_time.py).
Sample: 1,500,000 emb rows drawn uniformly (seed 0) among items that appear as a train candidate (so f1..f5 exist).
"""
import sys, time
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/feats_v2/life_")
from post_time import age_hours

R = "<repo>/data/flat/train/"
OUT = "<repo>/nn_x/emb/structure/cache/"
N = 81_557_944
t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

ce = np.load(R + "cand_emb_idx.npy")
ok = ce >= 0
cev = ce[ok]
n_cand = np.bincount(cev, minlength=N).astype(np.int32)
tg = np.load(R + "cand_target.npy")[ok]
n_pos = np.bincount(cev, weights=tg.astype(np.float64), minlength=N).astype(np.float32)
del tg
log("n_cand items", (n_cand > 0).sum(), "rows", ok.sum())
fm = {}
for k in range(1, 6):
    f = np.load(R + f"cand_f{k}.npy")[ok]
    s = np.bincount(cev, weights=f.astype(np.float64), minlength=N)
    fm[k] = (s / np.maximum(n_cand, 1)).astype(np.float32)
    del f, s
    log("f", k)
cat = np.zeros(N, np.int8)
cc = np.load(R + "cand_category_id.npy")[ok]
cat[cev] = cc
del cc
rep = np.full(N, -1, np.int64)
rep[cev] = np.nonzero(ok)[0]
del cev
log("rep")
# history counts per event type
he = np.load(R + "hist_emb_idx.npy")
het = np.load(R + "hist_event_type.npy")
hc = np.load(R + "hist_category_id.npy")
hok = he >= 0
hev = he[hok]
m0 = cat == 0
cat2 = np.zeros(N, np.int8); cat2[hev] = hc[hok]
cat[m0] = cat2[m0]
del cat2, hc
n_hist = np.bincount(hev, minlength=N).astype(np.int32)
key = hev.astype(np.int64) * 11 + het[hok]
cnt = np.bincount(key, minlength=N * 11).reshape(N, 11)[:, 1:].astype(np.int32)
del key, he, het, hok, hev
log("hist", n_hist.sum(), "items with hist", (n_hist > 0).sum())

# sample
rng = np.random.default_rng(0)
cand_items = np.nonzero(n_cand > 0)[0]
S = np.sort(rng.choice(cand_items, 1_500_000, replace=False))
log("cand-item universe", len(cand_items), "hist-only items", ((n_cand == 0) & (n_hist > 0)).sum(), "neither", ((n_cand == 0) & (n_hist == 0)).sum())
# representative row -> feed time, f4 at that appearance, t_post
feeds = pl.read_parquet(R + "feeds.parquet")
ts = feeds["timestamp"].to_numpy().astype(np.float64)
uid = feeds["user_id"].to_numpy()
rr = rep[S]
cfi = np.load(R + "cand_fidx.npy", mmap_mode="r")
f4r = np.load(R + "cand_f4.npy", mmap_mode="r")
iid = np.load(R + "cand_item_id.npy", mmap_mode="r")
fi = np.asarray(cfi[rr]); tf = ts[fi]
knots = np.load("<repo>/data/feats/life_f4_age_knots.npy")
age_h = age_hours(np.asarray(f4r[rr]), knots)
t_post = tf - age_h * 3600.0
emb = np.memmap("<repo>/data/competition/train/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=(N, 128))
E = np.asarray(emb[S])
log("emb loaded", E.shape)
np.savez(OUT + "train_sample.npz", idx=S, E=E, item_id=np.asarray(iid[rr]), cat=cat[S], n_cand=n_cand[S], n_pos=n_pos[S],
         f1=fm[1][S], f2=fm[2][S], f3=fm[3][S], f4=fm[4][S], f5=fm[5][S], f4_rep=np.asarray(f4r[rr]), t_feed=tf, t_post=t_post,
         n_hist=n_hist[S], hcnt=cnt[S], user_rep=uid[fi])
# a uniform sample over the whole table (incl. history-only items) for the global stats
U = np.sort(rng.choice(N, 500_000, replace=False))
np.savez(OUT + "train_uniform.npz", idx=U, E=np.asarray(emb[U]), cat=cat[U], n_cand=n_cand[U], n_hist=n_hist[U], hcnt=cnt[U])
log("done")
