"""famB_review: side-by-side of the study's stored numbers (famB/out/final_table.json, read_m50.json) and my independent recomputation (out/rv_eval.json). Writes out/rv_compare.md."""
import json
S = json.load(open("<repo>/nn_x/w12/cb/famB/out/final_table.json")); M = json.load(open("<repo>/nn_x/w12/cb/famB_review/out/rv_eval.json"))
key = lambda t: S[t]["tag"].replace("BUNDLE", "bundle")
rows = []; worst = dict(level=0, nats=0, se=0, mix=0, z=0)
f = lambda v, h, d=5: f"{v:+.{d}f} / {h:+.{d}f}"
for t, s in S.items():
    m = M[key(t)]
    for k, a, b in (("level", "level", "level"), ("nats", "nats", "nats"), ("se", "nats_se", "nats_se"), ("mix", "mix01", "mix")):
        worst[k] = max(worst[k], abs(s["val"][a] - m["val"][b]), abs(s["hold"][a] - m["hold"][b]))
    worst["z"] = max(worst["z"], abs(s["pooled_z"] - m["pooled_z"]))
    assert s["promising"] == m["promising"] and s["member_pass"] == m["member_pass"]
    rows.append(f"| `{key(t)}` level val / holdout | {s['val']['level']:.5f} / {s['hold']['level']:.5f} | {m['val']['level']:.5f} / {m['hold']['level']:.5f} | yes |")
    rows.append(f"| `{key(t)}` NATS val / holdout (± SE) | {s['val']['nats']:+.5f} ± {s['val']['nats_se']:.5f} / {s['hold']['nats']:+.5f} ± {s['hold']['nats_se']:.5f} | "
                f"{m['val']['nats']:+.5f} ± {m['val']['nats_se']:.5f} / {m['hold']['nats']:+.5f} ± {m['hold']['nats_se']:.5f} | yes, not PROMISING |")
    rows.append(f"| `{key(t)}` member rule 0.9 / 0.1 val / holdout, pooled z | {s['val']['mix01']:+.5f} ± {s['val']['mix01_se']:.5f} / {s['hold']['mix01']:+.5f} ± {s['hold']['mix01_se']:.5f}, {s['pooled_z']:+.2f} | "
                f"{m['val']['mix']:+.5f} ± {m['val']['mix_se']:.5f} / {m['hold']['mix']:+.5f} ± {m['hold']['mix_se']:.5f}, {m['pooled_z']:+.2f} | yes, no pass |")
print("largest absolute difference over the 13 rows:", {k: float(f"{v:.2e}") for k, v in worst.items()})
try:
    R5 = json.load(open("<repo>/nn_x/w12/cb/famB/out/read_m50.json")); print("read_m50 keys:", list(R5)[:12]); print(json.dumps(R5[list(R5)[0]])[:500])
except Exception as e:
    print("read_m50:", e)
open("<repo>/nn_x/w12/cb/famB_review/out/rv_compare.md", "w").write("\n".join(rows) + "\n")
print(len(rows), "rows written")
