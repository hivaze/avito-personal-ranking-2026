"""audit 4: ceiling of the distilled tree student. Reads the KL* probes (out/KL*_r<round>_{val,trH}.npy). Light: 2 threads."""
from au_common import *
light(2)
m = Meta(); P = {sp: w16_pool(m, sp) for sp in ("val", "hold")}; nd = {sp: ndcg_fast(P[sp], m.y[sp], m.off[sp], m.M[sp]) for sp in ("val", "hold")}
rec = {sp: ndcg_fast(m.z(sp, np.load(LB / f"gbdt_K_xm_60000_{'val' if sp == 'val' else 'trH'}.npy")), m.y[sp], m.off[sp], m.M[sp]) for sp in ("val", "hold")}
print(f"record student K_xm_60000 r600: val {rec['val'].mean():.5f} holdout {rec['hold'].mean():.5f}")
for tag in sorted({p.name.split("_r")[0] for p in OUT.glob("KL*_r*_val.npy")}):
    rounds = sorted(int(p.name.split("_r")[1].split("_")[0]) for p in OUT.glob(f"{tag}_r*_val.npy"))
    E = {}; Zs = {}
    for r in rounds:
        for sp, suf in (("val", "val"), ("hold", "trH")):
            z = m.z(sp, np.load(OUT / f"{tag}_r{r}_{suf}.npy")); Zs[(r, sp)] = z; E[(r, sp)] = ndcg_fast(z, m.y[sp], m.off[sp], m.M[sp])
    print(f"\n== {tag}: {json.load(open(OUT / f'{tag}.json'))['params']}")
    base = 600 if 600 in rounds else rounds[0]
    for r in rounds:
        s = f"  round {r:5d}: val {E[(r,'val')].mean():.5f} hold {E[(r,'hold')].mean():.5f} | vs round {base}: val {fmt(E[(r,'val')] - E[(base,'val')])} hold {fmt(E[(r,'hold')] - E[(base,'hold')])}"
        s += f" | vs record r600: val {fmt(E[(r,'val')] - rec['val'])} hold {fmt(E[(r,'hold')] - rec['hold'])}"
        rv, rh = corr(Zs[(r, 'val')], P['val']), corr(Zs[(r, 'hold')], P['hold']); gv = (nd['val'] - E[(r, 'val')]).mean(); gh = (nd['hold'] - E[(r, 'hold')]).mean()
        bv = ndcg_fast(0.9 * P['val'] + 0.1 * Zs[(r, 'val')], m.y['val'], m.off['val'], m.M['val']) - nd['val']; bh = ndcg_fast(0.9 * P['hold'] + 0.1 * Zs[(r, 'hold')], m.y['hold'], m.off['hold'], m.M['hold']) - nd['hold']
        s += f" | rho {rv:.4f}/{rh:.4f} gap {gv:.5f}/{gh:.5f} law {0.0186*(1-rv)-0.378*gv:+.5f}/{0.0186*(1-rh)-0.378*gh:+.5f} refit {0.0103*(1-rv)-0.304*gv:+.5f}/{0.0103*(1-rh)-0.304*gh:+.5f} | 0.9/0.1 blend val {fmt(bv)} hold {fmt(bh)}"
        print(s)
    last = rounds[-1]
    if len(rounds) >= 2:
        print(f"  last interval ({rounds[-2]} -> {last}): val {fmt(E[(last,'val')] - E[(rounds[-2],'val')])} hold {fmt(E[(last,'hold')] - E[(rounds[-2],'hold')])}")
if (OUT / "KL0_60k_r2400_val.npy").exists() and list(OUT.glob("KL1_60k_r*_val.npy")):
    r1 = max(int(p.name.split("_r")[1].split("_")[0]) for p in OUT.glob("KL1_60k_r*_val.npy"))
    for sp, suf in (("val", "val"), ("hold", "trH")):
        a = ndcg_fast(m.z(sp, np.load(OUT / f"KL1_60k_r{r1}_{suf}.npy")), m.y[sp], m.off[sp], m.M[sp]); b = ndcg_fast(m.z(sp, np.load(OUT / f"KL0_60k_r2400_{suf}.npy")), m.y[sp], m.off[sp], m.M[sp])
        print(f"KL1 r{r1} - KL0 r2400 on {sp}: {fmt(a - b)}")
