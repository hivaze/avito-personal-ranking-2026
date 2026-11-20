"""B3 pre-screen, step (ii): block-level, TRAIN feeds only (no val / holdout / test label is read).
LightGBM lambdarank on a random subset of the xm FIT feeds, read on the 13,527 xm SELECTION feeds (users never fitted) by the screen statistic
with z_oof in the role of the pool:  nats(s) = loss(z_oof) - loss(z_oof, z(s))  per feed (conditional logit, nats.clogit).
  --mode joint : models on [z_oof, 177] (reference) and [z_oof, 177, block]   (the DECLARED drop rule: increment < 1 paired SE -> block dropped)
  --mode arm   : models on [177] and [177, block] (no NN input: what the B3 arm is), read with the same statistic
  --mode small : models on [z_oof] and [z_oof, block] only
usage: python -u b3_prescreen.py --mode joint --blocks a b [c d] [--nfit 30000] [--rounds 300]"""
import argparse, json, sys, time
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB"); sys.path.insert(0, "<repo>/nn_x/w12/cb")
from fb_common import *
from nats import clogit
import lightgbm as lgb

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["joint", "arm", "small"], default="joint"); ap.add_argument("--blocks", nargs="+", default=["a", "b"])
ap.add_argument("--nfit", type=int, default=30000); ap.add_argument("--rounds", type=int, default=300); ap.add_argument("--all", action="store_true")
a = ap.parse_args()
tr = cbx.load("xm"); fit, sel = cbx.trainside_split(tr)
rg = np.random.default_rng(11); fi = np.where(fit)[0]; kf = np.zeros(tr.n_feeds, bool); kf[rg.choice(fi, a.nfit, replace=False)] = True
rf, rs = np.repeat(kf, tr.M), np.repeat(sel, tr.M)
y = np.asarray(tr.y, np.float32); zo = cbx.feed_z(tr.off, tr.M, np.ascontiguousarray(cbx.nn_input(tr), np.float32))
Ms = tr.M[sel]; offs = np.r_[0, np.cumsum(Ms)[:-1]].astype(np.int64); ys = y[rs].astype(np.float64); zs = zo[rs].astype(np.float64)
_, _, l0 = clogit(zs[:, None], ys, offs, Ms); nf = int((l0 != 0).sum())
X = tr.X
blocks = {b: (json.load(open(FEAT / f"{b}_names.json")), np.load(FEAT / f"{b}_xm.npy", mmap_mode="r")) for b in a.blocks}
log(f"fit feeds {a.nfit} rows {int(rf.sum())} | selection feeds {int(sel.sum())} rows {int(rs.sum())} | mode {a.mode} | blocks { {b: len(v[0]) for b, v in blocks.items()} }")


def mat(rows, bl):
    parts = []
    if a.mode in ("joint", "small"):
        parts.append(zo[rows][:, None])
    if a.mode in ("joint", "arm"):
        parts.append(np.asarray(X[rows]))
    for b in bl:
        parts.append(np.asarray(blocks[b][1][rows]))
    return np.concatenate(parts, 1).astype(np.float32)


def run(tag, bl):
    t0 = time.time()
    Xf, Xs = mat(rf, bl), mat(rs, bl)
    ds = lgb.Dataset(Xf, label=y[rf], group=tr.M[kf], free_raw_data=True)
    prm = dict(objective="lambdarank", learning_rate=0.1, num_leaves=63 if a.mode != "small" else 31, min_data_in_leaf=200, feature_fraction=0.5 if a.mode != "small" else 1.0,
               lambda_l2=5.0, lambdarank_truncation_level=300, max_bin=255, num_threads=cbx.THREADS, verbose=-1, seed=0, deterministic=True)
    m = lgb.train(prm, ds, num_boost_round=a.rounds)
    s = m.predict(Xs, num_threads=cbx.THREADS).astype(np.float32)
    z = cbx.feed_z(offs, Ms, np.ascontiguousarray(s)).astype(np.float64)
    b, se_, l1 = clogit(np.stack([zs, z], 1), ys, offs, Ms)
    nd = cbx.ndcg_fast(s, ys.astype(np.float32), offs, Ms)
    d = l0 - l1
    imp = m.feature_importance("gain"); tot = imp.sum(); nb0 = Xf.shape[1] - sum(len(blocks[b_][0]) for b_ in bl); share = {}
    o = nb0
    for b_ in bl:
        k = len(blocks[b_][0]); share[b_] = float(imp[o:o + k].sum() / tot)
        top = sorted(zip(imp[o:o + k] / tot, blocks[b_][0]), reverse=True)[:6]; share[b_ + "_top"] = [(n, round(float(v), 4)) for v, n in top]; o += k
    log(f"[{tag}] cols {Xf.shape[1]} | selection NDCG {nd.mean():.5f} | nats over z_oof {d.sum() / nf:+.5f} +- {np.std(d) / np.sqrt(len(d)) * len(d) / nf:.5f} (beta z {b[1] / se_[1]:+.1f}) | gain share {json.dumps(share)} | {time.time() - t0:.0f}s")
    return d, nd


res = {}
d0, n0 = run("reference", [])
todo = [[b] for b in a.blocks] + ([list(a.blocks)] if len(a.blocks) > 1 and a.all else [])
for bl in todo:
    tag = "+".join(bl); d, nd = run(tag, bl)
    inc = d - d0; dn = nd - n0
    m_, s_ = inc.sum() / nf, np.std(inc) / np.sqrt(len(inc)) * len(inc) / nf
    res[tag] = dict(nats_inc=float(m_), nats_inc_se=float(s_), ndcg_inc=float(dn.mean()), ndcg_inc_se=float(dn.std() / np.sqrt(len(dn))))
    log(f"BLOCK {tag}: increment over the reference {m_:+.5f} +- {s_:.5f} nats (z {m_ / s_:+.1f}); selection NDCG {dn.mean():+.5f} +- {dn.std() / np.sqrt(len(dn)):.5f} -> "
        f"{'KEEP' if m_ >= s_ else 'DROP'} (rule: increment >= 1 paired SE)")
json.dump(res, open(HERE / f"out/b3_prescreen_{a.mode}_{'_'.join(a.blocks)}.json", "w"), indent=1)
