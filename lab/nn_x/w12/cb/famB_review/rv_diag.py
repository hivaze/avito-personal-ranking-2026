"""famB_review: REDUCED-SCALE tree diagnostic (small fits of a few minutes; no model of record, no candidate).
Question 1 (verdict item 4): is B3's gain over B0 carried by the click-FREE columns (density + exposure counts) or by the CLICK statistics?
Question 2 (the lag-matched refit the study did not run): does training on test-lag statistics change the read?
LightGBM lambdarank, 63 leaves, lr 0.1, 300 rounds, no subsampling, on NFIT random xm feeds (hard click labels; train labels only), same rows for every arm:
  R   = the 177 features
  D   = R + kn_cos1 + kn_cos20                  (the two label-free density columns)
  F   = R + every click-free column             (density, impression counts / shares, kernel mass, neighbour expected clicks, recency)
  K   = R + every click-statistic column
  A   = R + all 70 columns                      (= the B3 input set), fresh train columns (as the study's arm)
  L   = R + all 70 columns, train rows from feat/lag_{a,b,c}_xm.npy (test lag 6-9 d); read on val (natural lag), holdout fresh AND holdout lagged
Loader: cbx (infra data loader only). Evaluation: rv_eval.py (own code).  No val / holdout label is used for fitting or selection (fixed 300 rounds).
usage: python -u rv_diag.py [NFIT=30000]"""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CBX_CACHE_EXTRA", "<repo>/nn_x/w12/cb/famB_review/cache")
sys.path.insert(0, "<repo>/nn_x/w12/cb/infra"); sys.path.insert(0, "<repo>/nn_x/w12/cb/famB_review")
import numpy as np
import cbx
import lightgbm as lgb
import rv_eval as E

NFIT = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
FEAT = "<repo>/nn_x/w12/cb/famB/feat"; HERE = "<repo>/nn_x/w12/cb/famB_review"
t0 = time.time()
def log(*a): print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

names = {b: json.load(open(f"{FEAT}/{b}_names.json")) for b in "abc"}
allc = [(b, j, n) for b in "abc" for j, n in enumerate(names[b])]
DENS = {"kn_cos1", "kn_cos20"}
FREE = DENS | {"kn_ex20", "ua_ns", "ua_dt_last", "ue_k05_all", "ue_k20_all", "ue_top5_cos", "ua_item_dt"} | {n for _, _, n in allc if n.endswith("_imp") or n.endswith("_imprw") or n.endswith("_share")}
groups = {"R": [], "D": [c for c in allc if c[2] in DENS], "F": [c for c in allc if c[2] in FREE], "K": [c for c in allc if c[2] not in FREE], "A": allc}
log({k: len(v) for k, v in groups.items()}, "click-free:", sorted(FREE & {n for _, _, n in allc}))

tr = cbx.load("xm")
fit, sel = cbx.trainside_split(tr)
rg = np.random.default_rng(11); pick = np.zeros(tr.n_feeds, bool); pick[rg.choice(np.where(fit)[0], NFIT, replace=False)] = True
rm = np.repeat(pick, tr.M); Mfit = tr.M[pick]
Xtr = np.asarray(tr.X[rm], np.float32); ytr = np.asarray(tr.y, np.float32)[rm]
log("fit feeds", NFIT, "rows", Xtr.shape, "clicks", float(ytr.sum()))
blk = {(pre, b): np.load(f"{FEAT}/{pre}{b}_xm.npy", mmap_mode="r") for pre in ("", "lag_") for b in "abc"}
ev = {}
for s in ("val", "hold"):
    d = cbx.load(s)
    ev[s] = dict(X=np.asarray(d.X, np.float32), blk={b: np.load(f"{FEAT}/{b}_{s}.npy", mmap_mode="r") for b in "abc"})
ev["hold"]["lag"] = {b: np.load(f"{FEAT}/lag_{b}_hold.npy", mmap_mode="r") for b in "abc"}
wv, wh = E.load_witness("val"), E.load_witness("hold")
log("witnesses ready")
P = dict(objective="lambdarank", metric="None", num_leaves=63, learning_rate=0.1, min_data_in_leaf=100, max_bin=255, num_threads=12, verbose=-1, seed=0,
         deterministic=True, force_row_wise=True, lambdarank_truncation_level=30)


def cols_of(cs, src):
    return np.stack([np.asarray(src[b][:, j], np.float32) for b, j, _ in cs], 1) if cs else None


def run(tag, cs, pre=""):
    t1 = time.time()
    add = np.stack([np.asarray(blk[(pre, b)][:, j][rm], np.float32) for b, j, _ in cs], 1) if cs else None
    X = Xtr if add is None else np.concatenate([Xtr, add], 1)
    m = lgb.train(P, lgb.Dataset(X, ytr, group=Mfit, free_raw_data=True), num_boost_round=300)
    out = {}
    for s in ("val", "hold"):
        a = cols_of(cs, ev[s]["blk"]); out[s] = m.predict(ev[s]["X"] if a is None else np.concatenate([ev[s]["X"], a], 1), num_threads=12)
    if pre:
        a = cols_of(cs, ev["hold"]["lag"]); out["hold_lag"] = m.predict(np.concatenate([ev["hold"]["X"], a], 1), num_threads=12)
    np.save(f"{HERE}/out/diag_{tag}_val.npy", out["val"].astype(np.float32)); np.save(f"{HERE}/out/diag_{tag}_hold.npy", out["hold"].astype(np.float32))
    log(f"fit {tag}: {X.shape[1]} features, {time.time() - t1:.0f}s")
    return out


res = {}; reads = {}
def read(tag, pv, ph):
    rv, rh = E.read_model([pv], wv), E.read_model([ph], wh); reads[tag] = (rv, rh)
    res[tag] = dict(val=dict(level=rv["level"], nats=rv["nats"], nats_se=rv["nats_se"]), hold=dict(level=rh["level"], nats=rh["nats"], nats_se=rh["nats_se"]))
    line = f"{tag}: level {rv['level']:.5f} / {rh['level']:.5f} | NATS {rv['nats']:+.5f}+-{rv['nats_se']:.5f} / {rh['nats']:+.5f}+-{rh['nats_se']:.5f}"
    if tag != "R" and "R" in reads:
        for i, sp in enumerate(("val", "hold")):
            dn = E.pse(reads[tag][i]["nats_perfeed"] - reads["R"][i]["nats_perfeed"]); dl = E.pse(reads[tag][i]["nd"] - reads["R"][i]["nd"])
            res[tag][sp].update(d_nats=dn, d_level=dl)
        line += (f" | minus R: NATS val {res[tag]['val']['d_nats'][0]:+.5f}+-{res[tag]['val']['d_nats'][1]:.5f} hold {res[tag]['hold']['d_nats'][0]:+.5f}+-{res[tag]['hold']['d_nats'][1]:.5f}"
                 f" | level val {res[tag]['val']['d_level'][0]:+.5f}+-{res[tag]['val']['d_level'][1]:.5f} hold {res[tag]['hold']['d_level'][0]:+.5f}+-{res[tag]['hold']['d_level'][1]:.5f}")
    log(line)
    json.dump(res, open(f"{HERE}/out/rv_diag.json", "w"), indent=1)


for tag in ("R", "A", "F", "K", "D"):
    o = run(tag, groups[tag]); read(tag, o["val"], o["hold"])
o = run("L", allc, pre="lag_"); read("L", o["val"], o["hold"]); read("L_holdlag", o["val"], o["hold_lag"])
# direct paired reads between arms
def pair(a, b):
    out = {}
    for i, sp in enumerate(("val", "hold")):
        out[sp] = dict(nats=E.pse(reads[a][i]["nats_perfeed"] - reads[b][i]["nats_perfeed"]), level=E.pse(reads[a][i]["nd"] - reads[b][i]["nd"]))
    log(f"PAIR {a} - {b}: NATS val {out['val']['nats'][0]:+.5f}+-{out['val']['nats'][1]:.5f} hold {out['hold']['nats'][0]:+.5f}+-{out['hold']['nats'][1]:.5f} | "
        f"level val {out['val']['level'][0]:+.5f}+-{out['val']['level'][1]:.5f} hold {out['hold']['level'][0]:+.5f}+-{out['hold']['level'][1]:.5f}")
    res[f"PAIR {a}-{b}"] = out
pair("A", "F"); pair("A", "K"); pair("F", "D"); pair("L", "A"); pair("L_holdlag", "A")
json.dump(res, open(f"{HERE}/out/rv_diag.json", "w"), indent=1)
log("DONE")
