"""famS_review step 1: every number the famS verdict rests on, recomputed with rv_own (own NDCG, own conditional logit, witnesses and W16 rebuilt from raw files).
Evaluation only, 4 threads. Writes out/rv_reads.json and prints one line per read."""
import json, sys, time
import numpy as np
sys.path.insert(0, "<repo>/nn_x/w12/cb/famS_review")
import rv_own as o
from rv_own import ROOT, FAM, INFRA, pse

t0 = time.time()
W = o.witnesses()
print("holdout rebuild check:", W["hold"]["check"], flush=True)
for sp in ("val", "hold"):
    w = W[sp]; npos = np.add.reduceat(w["y"], w["off"])
    print(f"{sp}: feeds {len(w['M'])} rows {int(w['M'].sum())} feeds without click {int((npos == 0).sum())} M>=50 feeds {int((w['M'] >= 50).sum())} M min {w['M'].min()}", flush=True)
ZW = {sp: o.w16_pool(sp, W[sp]) for sp in ("val", "hold")}
for sp in ("val", "hold"):
    c = np.load(INFRA / f"cache/w16_z_{sp}.npy")
    print(f"W16 {sp}: my rebuild vs infra cache max abs diff {np.abs(c - ZW[sp]).max():.2e}", flush=True)
E0 = {sp: o.ndcg(ZW[sp], W[sp]) for sp in ("val", "hold")}
print(f"W16 level val {E0['val'].mean():.5f} holdout {E0['hold'].mean():.5f}  ({time.time() - t0:.0f}s)", flush=True)
CLs = {sp: o.CL(W[sp]) for sp in ("val", "hold")}
L0 = {sp: CLs[sp].fit([ZW[sp]])[1] for sp in ("val", "hold")}
SUF = {"val": "val", "hold": "hold"}


def raw(tag, sp):
    for d in (FAM / "preds", FAM / "staged", INFRA / "out"):
        p = d / f"{tag}_{SUF[sp]}.npy"
        if p.exists():
            x = np.load(p)
            assert x.shape == ZW[sp].shape and np.isfinite(x).all(), p
            return x
    raise FileNotFoundError(tag)


EXT = {"lgb_K134k_r600": ("nn_x/w12/recheck/long_bets/out/gbdt_K_xm_134118_val.npy", "nn_x/w12/recheck/long_bets/out/gbdt_K_xm_134118_trH.npy"),
       "lgb_KL0_60k_r2400": ("nn_x/w12/cb/audit/out/KL0_60k_r2400_val.npy", "nn_x/w12/cb/audit/out/KL0_60k_r2400_trH.npy")}
_Z = {}


def z(name, sp):
    """per-feed z of a tag; 'a+b+c' = member-equal mean of the members' z (NOT re-standardised, as nats.py)."""
    if (name, sp) not in _Z:
        if "+" in name:
            _Z[(name, sp)] = np.mean([z(t, sp) for t in name.split("+")], 0)
        elif name in EXT:
            _Z[(name, sp)] = o.zfeed(np.load(ROOT / EXT[name][0 if sp == "val" else 1]), W[sp]["off"], W[sp]["M"])
        else:
            _Z[(name, sp)] = o.zfeed(raw(name, sp), W[sp]["off"], W[sp]["M"])
    return _Z[(name, sp)]


_LEV = {}


def level(name, sp):
    if (name, sp) not in _LEV:
        if "+" in name or name in EXT:
            s = z(name, sp) if "+" in name else np.load(ROOT / EXT[name][0 if sp == "val" else 1])
        else:
            s = raw(name, sp)          # raw float32 predictions: the ties of the tree model are kept
        _LEV[(name, sp)] = o.ndcg(s, W[sp])
    return _LEV[(name, sp)]


def rho_feed(a, b, w):
    fid = o.fid_of(w["M"])            # both are per-feed z (mean 0, sd 1): per-feed Pearson = mean of the product
    return float((np.add.reduceat(a * b, w["off"]) / w["M"]).mean())


_G = {}


def gain(name, sp):
    """per-feed nats gain of [zW, zX] over [zW] on the feeds with a click."""
    if (name, sp) not in _G:
        b, l2 = CLs[sp].fit([ZW[sp], z(name, sp)])
        _G[(name, sp)] = (L0[sp] - l2, b)
    return _G[(name, sp)]


def screen(name):
    r = {"tag": name}
    for sp in ("val", "hold"):
        w = W[sp]; zx = z(name, sp); zw = ZW[sp]
        d, b = gain(name, sp)
        lv = level(name, sp)
        zx1 = o.zfeed(zx, w["off"], w["M"])
        dm = o.ndcg(0.9 * zw + 0.1 * zx1, w) - E0[sp]
        nat, nse = pse(d)
        mm, ms = pse(dm)
        r[sp] = dict(level=float(lv.mean()), rho_global=float(np.corrcoef(zx, zw)[0, 1]), rho_feed=rho_feed(zx1, zw, w), nats=nat, nats_se=nse,
                     beta=[float(v) for v in b], mix01=mm, mix01_se=ms, gap=float((E0[sp] - lv).mean()), gap_se=pse(E0[sp] - lv)[1])
        for wt in (0.05, 0.2):
            r[sp][f"mix{wt}"] = float((o.ndcg((1 - wt) * zw + wt * zx1, w) - E0[sp]).mean())
    v, h = r["val"], r["hold"]
    r["promising"] = bool(v["nats"] >= 0.0016 and h["nats"] >= 0.0016)
    r["pooled_z"] = float((v["mix01"] / v["mix01_se"] + h["mix01"] / h["mix01_se"]) / np.sqrt(2))
    r["member_pass"] = bool(v["mix01"] >= 0 and h["mix01"] >= 0 and r["pooled_z"] >= 2)
    print(f"{name}: level {v['level']:.5f} / {h['level']:.5f} | rho {v['rho_global']:.3f} / {h['rho_global']:.3f} (per-feed {v['rho_feed']:.3f} / {h['rho_feed']:.3f}) | "
          f"NATS {v['nats']:+.5f}+-{v['nats_se']:.5f} / {h['nats']:+.5f}+-{h['nats_se']:.5f} -> {'PROMISING' if r['promising'] else 'not promising'} | "
          f"member 0.9/0.1: val {v['mix01']:+.5f}+-{v['mix01_se']:.5f}, holdout {h['mix01']:+.5f}+-{h['mix01_se']:.5f}, pooled z {r['pooled_z']:+.2f} -> "
          f"{'PASS' if r['member_pass'] else 'no pass'} | w0.05 {v['mix0.05']:+.5f}/{h['mix0.05']:+.5f} w0.2 {v['mix0.2']:+.5f}/{h['mix0.2']:+.5f}", flush=True)
    return r


def paired_level(a, b):
    r = {}
    for sp in ("val", "hold"):
        d = level(a, sp) - level(b, sp); r[sp], r[sp + "_se"] = pse(d)
    r["pooled_z"] = float((r["val"] / r["val_se"] + r["hold"] / r["hold_se"]) / np.sqrt(2))
    print(f"[level {a} - {b}] val {r['val']:+.5f}+-{r['val_se']:.5f} | holdout {r['hold']:+.5f}+-{r['hold_se']:.5f} | pooled z {r['pooled_z']:+.2f}", flush=True)
    return r


def natsdiff(x, h):
    r = {}
    for sp in ("val", "hold"):
        d = gain(x, sp)[0] - gain(h, sp)[0]; r[sp], r[sp + "_se"] = pse(d)
    r["pooled_z"] = float((r["val"] / r["val_se"] + r["hold"] / r["hold_se"]) / np.sqrt(2))
    print(f"[NATS({x}) - NATS({h})] val {r['val']:+.5f}+-{r['val_se']:.5f} | holdout {r['hold']:+.5f}+-{r['hold_se']:.5f} | pooled z {r['pooled_z']:+.2f}", flush=True)
    return r


def joint(s, h):
    r = {}
    for sp in ("val", "hold"):
        zs, zh, zw = z(s, sp), z(h, sp), ZW[sp]; c = CLs[sp]
        _, ls = c.fit([zw, zs]); _, lh = c.fit([zw, zh]); _, lsh = c.fit([zw, zs, zh])
        r[sp] = {"S|W16,H": pse(lh - lsh), "H|W16,S": pse(ls - lsh), "W16+S+H": pse(L0[sp] - lsh), "corr(S,H)": float(np.corrcoef(zs, zh)[0, 1])}
    print(f"[joint S={s} H={h}] S | W16,H: {r['val']['S|W16,H'][0]:+.5f}+-{r['val']['S|W16,H'][1]:.5f} / {r['hold']['S|W16,H'][0]:+.5f}+-{r['hold']['S|W16,H'][1]:.5f} ; "
          f"H | W16,S: {r['val']['H|W16,S'][0]:+.5f}+-{r['val']['H|W16,S'][1]:.5f} / {r['hold']['H|W16,S'][0]:+.5f}+-{r['hold']['H|W16,S'][1]:.5f} ; "
          f"W16+S+H {r['val']['W16+S+H'][0]:+.5f} / {r['hold']['W16+S+H'][0]:+.5f} ; corr(S,H) {r['val']['corr(S,H)']:.3f} / {r['hold']['corr(S,H)']:.3f}", flush=True)
    return r


def m50(name):
    w = W["val"]; r = {"tag": name}
    zx = z(name, "val"); zw = ZW["val"]; zx1 = o.zfeed(zx, w["off"], w["M"])
    dm = o.ndcg(0.9 * zw + 0.1 * zx1, w) - E0["val"]; lv = level(name, "val")
    for nm, k in (("M>=50", w["M"] >= 50), ("M<50", w["M"] < 50)):
        c = o.CL(w, feeds=k)
        _, l1 = c.fit([zw]); _, l2 = c.fit([zw, zx])
        r[nm] = dict(n=int(k.sum()), n_click=c.nf, level=float(lv[k].mean()), w16=float(E0["val"][k].mean()), nats=pse(l1 - l2), mix01=pse(dm[k]),
                     gap=pse((E0["val"] - lv)[k]))
        print(f"[{name}] val {nm} n={r[nm]['n']} level {r[nm]['level']:.5f} (W16 {r[nm]['w16']:.5f}, gap {r[nm]['gap'][0]:.5f}+-{r[nm]['gap'][1]:.5f}) | "
              f"NATS {r[nm]['nats'][0]:+.5f}+-{r[nm]['nats'][1]:.5f} | member 0.9/0.1 {r[nm]['mix01'][0]:+.5f}+-{r[nm]['mix01'][1]:.5f}", flush=True)
    return r


A = "s1f_rmse_d8_lr030"
H = "ts_lr020_s0"
B3 = f"{A}_s0+{A}_s1+{A}_s2"
ARMS = [H, "s1_rmse_d8_lr010_s0", f"{A}_s0", f"{A}_s1", f"{A}_s2", B3, f"{A}_s1long", "s3f_pca_d8_lr030_s0", "s2_rmse_d10_lr030_s0", "s4_qsm_kdmix_lr030_s0",
        "s4b_qsm_soft_lr030_s0", "s4y_yeti_teacher_lr030_s0", f"{A}_s0+{H}", "bundle_s1f_3seeds_z+" + H, "bundle_s1f_3seeds_z", "lgb_K134k_r600", "lgb_KL0_60k_r2400"]
out = {"screen": {}, "level": {}, "natsdiff": {}, "joint": {}, "m50": {}, "holdcheck": W["hold"]["check"]}
print("\n== SCREEN lines (own code)")
for t in ARMS:
    out["screen"][t] = screen(t)
# saved bundle file against my own z-mean of the three seeds
for sp in ("val", "hold"):
    print(f"saved bundle file vs my z-mean of 3 seeds, {sp}: max abs diff {np.abs(raw('bundle_s1f_3seeds_z', sp) - z(B3, sp)).max():.2e}")
print("\n== paired level reads")
for a, b in ((B3, H), (B3, "lgb_K134k_r600"), (B3, "lgb_KL0_60k_r2400"), ("s3f_pca_d8_lr030_s0@721", f"{A}_s0@721"), ("s3f_pca_d8_lr030_s0@500", f"{A}_s0@500"),
             ("s3f_pca_d8_lr030_s0@300", f"{A}_s0@300"), ("s4b_qsm_soft_lr030_s0", f"{A}_s0"), ("s4b_qsm_soft_lr030_s0", "s1_rmse_d8_lr010_s0"), ("s4b_qsm_soft_lr030_s0", B3),
             ("s4b_qsm_soft_lr030_s0", f"{A}_s1long"), ("s2_rmse_d10_lr030_s0", f"{A}_s0@721"), ("s2_rmse_d10_lr030_s0@300", f"{A}_s0@300"), ("s1_rmse_d8_lr010_s0", f"{A}_s0"),
             (f"{A}_s1long@1761", f"{A}_s1long@1400"), ("s4_qsm_kdmix_lr030_s0", H), ("s4y_yeti_teacher_lr030_s0", H), (f"{A}_s1long", "lgb_KL0_60k_r2400"),
             (f"{A}_s1", f"{A}_s1long@886")):
    try:
        out["level"][f"{a} - {b}"] = paired_level(a, b)
    except FileNotFoundError as e:
        print("missing", e)
print("\n== paired NATS(X) - NATS(hard)")
for t in ("s1_rmse_d8_lr010_s0", f"{A}_s0", "s3f_pca_d8_lr030_s0", "s2_rmse_d10_lr030_s0", "s4_qsm_kdmix_lr030_s0", "s4b_qsm_soft_lr030_s0", "s4y_yeti_teacher_lr030_s0", B3, f"{A}_s1long"):
    out["natsdiff"][t] = natsdiff(t, H)
print("\n== joint reads")
for s in (B3, f"{A}_s0", "s4b_qsm_soft_lr030_s0", "s3f_pca_d8_lr030_s0"):
    out["joint"][s] = joint(s, H)
print("\n== val by M bin")
for t in (B3, H, f"{A}_s0", "s4b_qsm_soft_lr030_s0", f"{A}_s1long"):
    out["m50"][t] = m50(t)
(ROOT / "nn_x/w12/cb/famS_review/out/rv_reads.json").write_text(json.dumps(out, indent=1))
print(f"DONE {time.time() - t0:.0f}s")
