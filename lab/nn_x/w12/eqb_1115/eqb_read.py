"""EQB file read, the ONE labelled read (ledger block 2026-11-15T23:08:02Z item Q6; details nn_x/w12/eqb_1115/PREDECLARED.txt, written before the predictions).
CPU only. Pooling / z / metric conventions are IMPORTED from nn_x/w12/kd/build_r4.py (run in its print-only mode for the key FD16P20: no 'build' argument, nothing written).
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 ... .venv/bin/python nn_x/w12/eqb_1115/eqb_read.py   -> stdout (logs/read_val.txt) + out/read_val.json"""
import io
import json
import sys
import contextlib
from pathlib import Path

assert __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
R = Path("<repo>"); HERE = R / "nn_x/w12/eqb_1115"; G = HERE / "preds"
sys.argv = ["build_r4.py", "FD16P20"]  # print-only mode of the builder
sys.path.insert(0, str(R / "nn_x/w12/kd")); sys.path.insert(0, str(R / "feat_new"))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import build_r4 as B4  # noqa: E402
import numpy as np  # noqa: E402
import hz_common as H  # noqa: E402
print("build_r4.py FD16P20 (print-only) says:"); print("   " + buf.getvalue().strip().replace("\n", "\n   ")[:1500])
P = B4.P; Z = B4.Z["val"]; ev = B4.ev; wts = B4.wts; Mv = B4.Mv; ov = B4.ov; yv = B4.yv; pmix = B4.pmix
base, pop, w = B4.spec("FD16P20")
MEM = [f"w5_fd10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in range(951, 959)]
assert base == MEM and w == 0.2 and pop == [f"abl_fd10_pop1kd05w192_s{s}" for s in range(951, 955)]
n = len(Mv); fid = np.repeat(np.arange(n), Mv)
ft_stored = np.load(P / "ens_FD16P20kd_ft_val.npy"); to_stored = np.load(P / "ens_FD16P20kd_to_val.npy")
pop_to = B4.pool(pop, "val")[1]
raw = {k: [np.load(G / f"gpu_{m}_{k}_val.npy") for m in MEM] for k in ("base", "B")}
def TO(kind): return (1 - w) * Z(np.mean([Z(p) for p in raw[kind]], 0)) + w * Z(pop_to)
to = {k: TO(k) for k in raw}; fil = {k: B4.fin("val", (ft_stored, to[k])) for k in raw}
e_to = {k: ev(to[k]) for k in raw}; e_f = {k: ev(fil[k]) for k in raw}
# ---- path check (label-free part + the level of the reference)
ident = [bool(np.array_equal(p, np.load(P / f"nn_{m}_trainonly_val.npy"))) for m, p in zip(MEM, raw["base"])]
ref_stored = ev(B4.fin("val", (ft_stored, to_stored)))
print(f"\nPATH CHECK: base re-predictions identical to the stored train-only val files for {sum(ident)} of 16 members; "
      f"TO(base) vs stored ens_FD16P20kd_to_val.npy max|d| {np.abs(to['base'] - to_stored).max():.2e}; "
      f"FT branch of build_r4 vs stored file max|d| {np.abs(B4.B['val'][0] - ft_stored).max():.2e}")
print(f"   reference (same path, stored inputs): FILE val {e_f['base'].mean():.5f} (stored-file FD16P20 final {ref_stored.mean():.5f}, build_r4 {B4.E_ALL['FD16P20'][0].mean():.5f}) | "
      f"train-only branch {e_to['base'].mean():.5f} | arm B: FILE {e_f['B'].mean():.5f}, train-only branch {e_to['B'].mean():.5f}")
ch = np.mean([(a != b) for a, b in zip(raw["B"], raw["base"])]); print(f"   rows whose member prediction changes (mean over members): {ch:.3f}")
# ---- slices
ts = B4.fv["timestamp"].to_numpy().astype(np.int64); day = H.day_of(ts)
cnt = np.load(R / "data/feats/val/it_hist_all_luo.npy"); y = yv.astype(np.float64)
npos = np.bincount(fid, y, n); coldpos = np.bincount(fid, y * (cnt <= 0), n); big = Mv >= 50
SL = [("val (all feeds)", np.ones(n, bool), None), ("val M<50", ~big, None), ("val M>=50", big, None), ("pmix-public", np.ones(n, bool), wts["public"]),
      ("pmix-private", np.ones(n, bool), wts["private"]), ("Tue (MSK day 20718)", day == 20718, None), ("Thu (MSK day 20720)", day == 20720, None),
      ("positives all cold", (npos > 0) & (coldpos == npos), None), ("no cold positive", (npos > 0) & (coldpos == 0), None)]
def rd(d, m, wt):
    if wt is None:
        x = d[m]; mu, se = float(x.mean()), float(x.std() / np.sqrt(len(x))); nn = int(m.sum())
    else:
        mu, se = pmix.wmean_se(d, wt); mu, se = float(mu), float(se); nn = int(round(wt.sum() ** 2 / (wt ** 2).sum()))
    return mu, se, nn
OUT = {"members": MEM, "identical_base": int(sum(ident)), "level": {"file_base": float(e_f["base"].mean()), "file_B": float(e_f["B"].mean()),
       "to_base": float(e_to["base"].mean()), "to_B": float(e_to["B"].mean()), "stored_file": float(ref_stored.mean())}}
for name, ee in (("FILE", e_f), ("TRAIN-ONLY BRANCH", e_to)):
    d = ee["B"] - ee["base"]; print(f"\n{name}: arm B (half dose) minus same-path reference, paired per feed"); OUT[name] = {}
    for lab, m, wt in SL:
        mu, se, nn = rd(d, m, wt); OUT[name][lab] = (mu, se, nn)
        print(f"   {lab:22s} {mu:+.5f} +- {se:.5f}  (z {mu / se:+.2f}, {'ESS' if wt is not None else 'n'} {nn})")
    print(f"   feeds changed {float((d != 0).mean()):.3f}; paired public SE (sd / sqrt(8562)) {d.std() / np.sqrt(8562):.5f}")
# ---- members
print("\nMEMBERS (each alone, arm B minus base):"); OUT["member"] = {}
for m, pb, pa in zip(MEM, raw["base"], raw["B"]):
    d = ev(pa) - ev(pb); mu, se = float(d.mean()), float(d.std() / np.sqrt(n)); OUT["member"][m] = (mu, se)
    print(f"   {m:28s} {mu:+.5f} +- {se:.5f} (z {mu / se:+.2f})")
mm = np.array([v[0] for v in OUT["member"].values()])
print(f"   positive members {int((mm > 0).sum())} of 16; mean {mm.mean():+.5f}, sd across members {mm.std(ddof=1):.5f}; kd05w192 x8 mean {mm[:8].mean():+.5f}, kd05t25w192 x8 mean {mm[8:].mean():+.5f}")
OUT["member_summary"] = {"positive": int((mm > 0).sum()), "mean": float(mm.mean()), "sd": float(mm.std(ddof=1))}
# ---- rule
F = OUT["FILE"]; c1 = F["val (all feeds)"][0] >= 0; c2 = F["pmix-private"][0] >= F["pmix-private"][1]; c3 = F["val M<50"][0] >= -F["val M<50"][1]
OUT["rule"] = {"val>=0": bool(c1), "pmix_private>=+1SE": bool(c2), "M<50>=-1SE": bool(c3), "BUILD": bool(c1 and c2 and c3)}
print(f"\nRULE (ledger 23:08:02Z Q6): FILE val delta >= 0: {c1} ({F['val (all feeds)'][0]:+.5f}) | pmix-private >= +1 SE: {c2} ({F['pmix-private'][0]:+.5f} vs SE {F['pmix-private'][1]:.5f}, "
      f"z {F['pmix-private'][0] / F['pmix-private'][1]:+.2f}) | val M<50 >= -1 SE: {c3} ({F['val M<50'][0]:+.5f} vs -{F['val M<50'][1]:.5f})  ->  {'BUILD' if c1 and c2 and c3 else 'CLOSED'}")
(HERE / "out").mkdir(exist_ok=True); (HERE / "out" / "read_val.json").write_text(json.dumps(OUT, indent=1))
