"""veto read 10 (resume run): the basis of the non-staleness re-open item (aux weight) re-read with the seeds that exist now, and the
selected-vs-@last read of the student re-cools with every finished run. Information; rules unchanged (two-witness rule for aux).
Output: logs/read10.txt + out/read10.json"""
import glob, json, os, re, sys
sys.path.insert(0, "<repo>/nn_x/w12/stale/veto")
from vlib import *
c = Ctx(); R = []
def seeds(arm):
    out = []
    for p in sorted(glob.glob(str(ROOT / f"nn_x/w5/preds/nn_w5_{arm}_s*_trainonly_val.npy"))):
        m = re.search(rf"nn_w5_{re.escape(arm)}_s(\d+)_trainonly_val\.npy$", p)
        if m and find(f"w5_{arm}_s{m.group(1)}", "hold") is not None and os.path.exists(ROOT / f"nn_x/w5/logs/w5_{arm}_s{m.group(1)}.json"):
            out.append(int(m.group(1)))
    return out
def go(label, A, B, **kw):
    r = read(c, label, A, B, kinds=("val", "hold"), **kw); show(r); R.append(r); return r
print("##### aux weight under KD (kwaux0 = --aux_pos 0) vs kd05w192, same seeds, finished runs only")
s = seeds("e10_kwaux0"); print("kwaux0 seeds finished:", s)
go(f"kwaux0 x{len(s)} vs kd05w192 same seeds", [f"w5_e10_kwaux0_s{i}" for i in s], [f"w5_e10_kd05w192_s{i}" for i in s])
print("##### aux weight 1.0, hard labels: 7 re-cool roots and the 10-epoch pair")
go("re-cool ap1 x7 vs m02ref x7 [selected]", [f"w5_cd_ap1_r{i}" for i in range(101, 108)], [f"w5_cd_m02ref_r{i}" for i in range(101, 108)])
go("re-cool ap1 x7 vs m02ref x7 [@last]", [f"w5_cd_ap1_r{i}" for i in range(101, 108)], [f"w5_cd_m02ref_r{i}" for i in range(101, 108)], sufA="@last")
go("e10 ap1 x2 vs base x2", ["w5_e10_ap1_s951", "w5_e10_ap1_s952"], ["w5_e10_base_s951", "w5_e10_base_s952"])
print("##### student re-cools of the harness: selected minus @last, every finished run (ref, ref2)")
runs = []
for arm in ("kcd_ref", "lcd_ref", "kcd_ref2", "lcd_ref2", "kcd_hz", "lcd_hz"):
    for i in seeds(arm):
        v = f"w5_{arm}_s{i}"
        if find(v + "@last", "val") is not None and find(v + "@last", "hold") is not None: runs.append(v)
print("finished re-cool runs with both checkpoints:", len(runs), runs)
ref = [v for v in runs if "_hz_" not in v]; hz = [v for v in runs if "_hz_" in v]
if ref:
    go(f"student ref / ref2 re-cools x{len(ref)}: pool S - L", ref, ref, sufA="", sufB="@last")
    dm = np.mean([c.nv(c.zfile(v, "val")) - c.nv(c.zfile(v + "@last", "val")) for v in ref], 0)
    dh = np.mean([c.nh(c.zfile(v, "hold")) - c.nh(c.zfile(v + "@last", "hold")) for v in ref], 0)
    print(f"   member-level mean S - L over {len(ref)} runs: val {f(ms(dm))} | coldpos {f(ms(dm[c.sl['coldpos']]))} | warmpos {f(ms(dm[c.sl['warmpos']]))} | pmix-priv {f(ms(dm, c.wpriv))} | holdout {f(ms(dh))}")
    R.append({"label": f"member-level S-L student re-cools x{len(ref)}", "val": ms(dm), "coldpos": ms(dm[c.sl['coldpos']]), "warmpos": ms(dm[c.sl['warmpos']]), "pmix_priv": ms(dm, c.wpriv), "hold": ms(dh)})
print("hz re-cools finished:", hz)
print("##### harness control with every finished pair: ref re-cools vs the original runs of the same snapshots")
pairs = [(v, v.replace("w5_kcd_ref_", "w5_e10_kd05w192_").replace("w5_lcd_ref_", "w5_e10_kd05t25w192_")) for v in runs if "_ref_" in v]
if pairs:
    go(f"ref re-cools x{len(pairs)} vs originals", [a for a, _ in pairs], [b for _, b in pairs])
json.dump(R, open(HERE / "out/read10.json", "w"), indent=1, default=float)
