"""Follow-up (4) POP-FREE FREE CHECK (report §8 D1; PREDECLARED.txt 05:47:47 UTC).
Cells (15_residual_f5): n = tte_it_imp + it_hist_all_luo; NOISY-HIGH = f5 >= q90(f5 over val rows) & n < 5; RELIABLE-HIGH = f5 >= q90
& n >= 5; MID/LOW = 0 < f5 < q90; NO-DATA = f5 == 0. act/exp of a score = sum y / sum expected in the cell, expected = within-feed
decile (of that score) positive rate renormalised per feed to its actual positives. User-clustered SE; differences vs comparators
with a 200-rep user-cluster bootstrap.
Pools (member-equal): to = rz-mean of present members' train-only preds; ft = rz-mean of FT preds; pool = 0.5 z(ft) + 0.5 z(to) if
every present member has FT, else to only. Also reported: the nopop-FT alias (n2cpv3np) when present, the same-seed pop-ful base
pool (nn_x/w5 w5_e10_base_s95x train-only), G3 final and G3 train-only.
PREDICTION HOLDS iff the pop-free pool's act/exp on NOISY-HIGH < 0.95. Output: out/f04_popfree.log, out/f04_popfree.json."""
import sys, json, time
sys.path.insert(0, "<repo>/nn_x/codes/followup")
from fu_common import *

T0 = time.time()
LOGF = open(OUT / "f04_popfree.log", "w")


def log(s=""):
    print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()


V = Split("val")
y, fid, n = V.y, V.fid, V.n
PA = ROOT / "nn_x/abl/preds"; PW = ROOT / "nn_x/w5/preds"
SEEDS = [951, 952, 953, 954]


def find(name):
    for d in (P, PA):
        if (d / name).exists():
            return d / name
    return None


mem = {}
for s in SEEDS:
    v = f"abl_e10_xdrop_pop_s{s}"
    mem[s] = dict(to=find(f"nn_{v}_trainonly_val.npy"), ft=P / f"nn_{v}_n2cpv3_val_oof.npy" if (P / f"nn_{v}_n2cpv3_val_oof.npy").exists() else None,
                  ftnp=find(f"nn_{v}_n2cpv3np_val_oof.npy") or (P / f"nn_{v}np_n2cpv3_val_oof.npy" if (P / f"nn_{v}np_n2cpv3_val_oof.npy").exists() else None),
                  base=PW / f"nn_w5_e10_base_s{s}_trainonly_val.npy" if (PW / f"nn_w5_e10_base_s{s}_trainonly_val.npy").exists() else None)
    log(f"seed {s}: " + ", ".join(f"{k}={'yes (' + str(p.parent.relative_to(ROOT)) + ')' if p else 'no'}" for k, p in mem[s].items()))
present = [s for s in SEEDS if mem[s]["to"] is not None]
log(f"present pop-free members (train-only): {present}")
assert present, "no pop-free member present"


def pool(paths):
    return V.rz(np.mean([V.rz(np.load(p)) for p in paths], 0))


SC = {}
SC["G3 final"] = np.load(P / "ens_G3_final_val.npy").astype(np.float64)
SC["G3 train-only"] = np.load(P / "ens_G3_to_val.npy").astype(np.float64)
to = pool([mem[s]["to"] for s in present])
SC[f"popfree TO ({len(present)})"] = to
ft_all = all(mem[s]["ft"] is not None for s in present)
if ft_all:
    SC[f"popfree FT n2cpv3 0.5ft+0.5to ({len(present)})"] = 0.5 * pool([mem[s]["ft"] for s in present]) + 0.5 * to
if all(mem[s]["ftnp"] is not None for s in present):
    SC[f"popfree FT n2cpv3np 0.5ft+0.5to ({len(present)})"] = 0.5 * pool([mem[s]["ftnp"] for s in present]) + 0.5 * to
bs = [s for s in present if mem[s]["base"] is not None]
if bs:
    SC[f"pop-ful base w5_e10 TO ({len(bs)}, same seeds)"] = pool([mem[s]["base"] for s in bs])
PRIMARY = f"popfree FT n2cpv3 0.5ft+0.5to ({len(present)})" if ft_all else f"popfree TO ({len(present)})"
log(f"PRIMARY pop-free pool (pre-declared rule): {PRIMARY}")

f5 = np.load(FLAT / "val/cand_f5.npy").astype(np.float64)
nexp = np.nan_to_num(np.load(FEATS / "val/tte_it_imp.npy").astype(np.float64)) + np.nan_to_num(np.load(FEATS / "val/it_hist_all_luo.npy").astype(np.float64))
q90 = np.quantile(f5, 0.9)
CELLS = {"NOISY-HIGH (f5>=q90, n<5)": (f5 >= q90) & (nexp < 5), "RELIABLE-HIGH (f5>=q90, n>=5)": (f5 >= q90) & (nexp >= 5),
         "MID/LOW (0<f5<q90)": (f5 > 0) & (f5 < q90), "NO-DATA (f5==0)": f5 == 0}
log(f"f5 q90 {q90:.4f}; cell row shares: " + "; ".join(f"{k} {m.mean():.4f} (pos rate {y[m].mean():.4f})" for k, m in CELLS.items()))


def expected(s):
    o = np.lexsort((s, fid)); fs_ = np.r_[True, fid[o][1:] != fid[o][:-1]]
    rk = np.arange(len(o)) - np.flatnonzero(fs_)[np.cumsum(fs_) - 1]
    pct = np.empty(len(o)); pct[o] = rk / np.maximum(V.M[fid[o]] - 1, 1)
    dec = np.clip((pct * 10).astype(int), 0, 9)
    rate = np.bincount(dec, y, minlength=10) / np.bincount(dec, minlength=10); ex = rate[dec]
    return ex * (np.bincount(fid, y, minlength=n) / np.maximum(np.bincount(fid, ex, minlength=n), 1e-12))[fid]


EX = {k: expected(s) for k, s in SC.items()}
ND = {k: V.nd(s).mean() for k, s in SC.items()}
G = V.cl.max() + 1
rng = np.random.default_rng(0)
BW = rng.multinomial(G, np.full(G, 1.0 / G), size=200).astype(np.float64)   # cluster bootstrap weights
RES = {"q90": float(q90), "present": present, "primary": PRIMARY, "ndcg": {k: float(v) for k, v in ND.items()}, "cells": {}}
log("\nact/exp (+- user-clustered SE) per cell; [diff vs G3 final +- bootstrap SE] [diff vs G3 train-only] [diff vs same-seed base]")
for cn, m in CELLS.items():
    log(f"--- {cn}: rows {m.sum():,}, positives {int(y[m].sum()):,}")
    ycl = np.bincount(V.cl, (y * m)[:], G) if False else np.bincount(V.cl[fid], y * m, G)
    rec = {}
    for k in SC:
        ecl = np.bincount(V.cl[fid], EX[k] * m, G)
        r = ycl.sum() / ecl.sum()
        num = np.bincount(fid, y * m, n); den = np.bincount(fid, EX[k] * m, n)
        _, se = ratio_cl(num, den, V.cl)
        boot = (BW @ ycl) / (BW @ ecl)
        rec[k] = dict(ae=float(r), se=float(se), boot=boot)
    refs = [("G3 final", "G3 final"), ("G3 train-only", "G3 train-only")] + \
           ([(f"pop-ful base w5_e10 TO ({len(bs)}, same seeds)", "base")] if bs else [])
    for k in SC:
        ds = []
        for rk, lab in refs:
            if rk == k:
                continue
            d = rec[k]["boot"] - rec[rk]["boot"]
            ds.append(f"vs {lab} {rec[k]['ae'] - rec[rk]['ae']:+.3f} +- {d.std():.3f}")
        log(f"   {k:52s} NDCG {ND[k]:.5f} | act/exp {rec[k]['ae']:.3f} +- {rec[k]['se']:.3f} | " + " | ".join(ds))
    RES["cells"][cn] = {k: dict(ae=v["ae"], se=v["se"]) for k, v in rec.items()}
    if cn.startswith("NOISY"):
        RES["noisy_primary"] = rec[PRIMARY]["ae"]; RES["noisy_primary_se"] = rec[PRIMARY]["se"]
        RES["noisy_g3"] = rec["G3 final"]["ae"]
        RES["noisy_diff_vs_g3"] = float(rec[PRIMARY]["ae"] - rec["G3 final"]["ae"])
        RES["noisy_diff_vs_g3_se"] = float((rec[PRIMARY]["boot"] - rec["G3 final"]["boot"]).std())
holds = RES["noisy_primary"] < 0.95
RES["prediction_holds"] = bool(holds)
log(f"\nPREDICTION (pop-free act/exp on NOISY-HIGH < 0.95): {RES['noisy_primary']:.3f} +- {RES['noisy_primary_se']:.3f} "
    f"(G3 {RES['noisy_g3']:.3f}; diff {RES['noisy_diff_vs_g3']:+.3f} +- {RES['noisy_diff_vs_g3_se']:.3f}) -> {'HOLDS' if holds else 'DOES NOT HOLD'}")
json.dump(RES, open(OUT / "f04_popfree.json", "w"), indent=1, default=float)
log(f"done in {time.time()-T0:.0f}s")
