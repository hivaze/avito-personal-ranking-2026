"""veto read 7: holdout-selected minus @last for the finished STUDENT re-cools of the K / L harness (the runs the hz arm will be read
against). Stored predictions only; information for the reverse check of checkpoint selection. Output logs/read7.txt + out/read7.json"""
import glob, json, sys
sys.path.insert(0, "<repo>/nn_x/w12/stale/veto")
from vlib import *
c = Ctx(); R = []
runs = []
for p in sorted(glob.glob(str(ROOT / "nn_x/w5/logs/w5_[kl]cd_*_s95?.json"))):
    v = Path(p).stem; j = json.load(open(p))
    if all(find(v + s, k) is not None for s in ("", "@last") for k in ("val", "hold")): runs.append((v, j["best_step"]))
print("runs:", [(v, b) for v, b in runs])
names = [v for v, _ in runs]; diff = [v for v, b in runs if b != 19670]
for lab, sel in (("all finished K / L re-cools", names), ("those whose selected step != last step", diff), ("ref only", [v for v in names if v.endswith(tuple(f"ref_s{s}" for s in range(951, 959)))])):
    if not sel: continue
    out = {"label": f"S - L member-level mean, {lab} ({len(sel)} runs)"}
    for kind, ev in (("val", c.nv), ("hold", c.nh)):
        d = np.mean([ev(c.zfile(v, kind)) - ev(c.zfile(v + "@last", kind)) for v in sel], 0); out[kind] = ms(d)
        if kind == "val":
            out["pmix_priv"] = ms(d, c.wpriv); out["pmix_pub"] = ms(d, c.wpub)
            for k, m in c.sl.items(): out[k] = ms(d[m])
        else:
            out["hold_late"] = ms(d[c.hl["late>=Sep13"]])
    print(f"== {out['label']}\n   " + " | ".join(f"{k} {f(v)}" for k, v in out.items() if k != "label")); R.append(out)
    r = read(c, f"S - L POOL, {lab} ({len(sel)} runs)", sel, sel, sufA="", sufB="@last", kinds=("val", "hold")); show(r); R.append(r)
json.dump(R, open(HERE / "out/read7.json", "w"), indent=1, default=float)
