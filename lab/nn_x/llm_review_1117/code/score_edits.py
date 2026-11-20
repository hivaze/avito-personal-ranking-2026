"""Score blind edits of eval feeds against the hidden validation labels.

usage: score_edits.py EDITS.json [EDITS2.json ...]
Each file: {"feeds": [{"feed": "E00-03", "moves": [{"from_rank": 37, "to_rank": 2}], "confidence": ..., "reason": ...}]}
or a list of such objects. Feeds missing from the files keep the model's order.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cards import Split  # noqa: E402

HERE = Path(__file__).parent
key = json.load(open(HERE / "val/private/eval_key.json"))
s = Split("val")
disc = 1.0 / np.log2(np.arange(2, 302))


def ndcg_order(t: np.ndarray) -> float:
    """NDCG of a strict order given the labels in that order."""
    dcg = (t * disc[:len(t)]).sum()
    ideal = disc[:int(t.sum())].sum()
    return float(dcg / ideal) if ideal > 0 else 1.0


def apply(m: int, moves: list[dict]) -> tuple[list[int], int]:
    seen, valid = set(), []
    for mv in moves:
        a, b = int(mv["from_rank"]), int(mv["to_rank"])
        if 1 <= a <= m and 1 <= b <= m and a not in seen and a != b:
            seen.add(a); valid.append((a - 1, b - 1))
    rest = [x for x in range(m) if x not in seen]
    for a, b in sorted(valid, key=lambda v: v[1]):
        rest.insert(min(b, len(rest)), a)
    return rest, len(moves) - len(valid)


entries = []
for p in sys.argv[1:]:
    d = json.load(open(p))
    for obj in (d if isinstance(d, list) else [d]):
        entries += obj["feeds"]
by_feed = {e["feed"]: e for e in entries}
unknown = [f for f in by_feed if f not in key]
print(f"entries {len(entries)} distinct feeds {len(by_feed)} unknown labels {unknown[:5]}")

rows_out, invalid = [], 0
for lab, e in by_feed.items():
    if lab not in key:
        continue
    i = key[lab]
    order = s.order(i)
    t = s.target[order].astype(float)
    new, bad = apply(len(order), e.get("moves", []))
    invalid += bad
    base, after = ndcg_order(t), ndcg_order(t[new])
    moved_pos = [int(t[int(mv["from_rank"]) - 1]) for mv in e.get("moves", []) if 1 <= int(mv["from_rank"]) <= len(order)]
    rows_out.append(dict(feed=lab, conf=e.get("confidence", "?"), n_moves=len(e.get("moves", [])), base=base, after=after,
                         delta=after - base, moved=len(moved_pos), moved_pos=sum(moved_pos),
                         up=[(int(mv["from_rank"]), int(mv["to_rank"])) for mv in e.get("moves", [])]))

d = np.array([r["delta"] for r in rows_out])
n = len(d)
print(f"feeds scored {n}, invalid moves dropped {invalid}")
print(f"model NDCG on these feeds {np.mean([r['base'] for r in rows_out]):.5f} -> edited {np.mean([r['after'] for r in rows_out]):.5f}")
print(f"mean delta per feed {d.mean():+.5f} +- {d.std(ddof=1) / np.sqrt(n):.5f} (SE)")
edited = [r for r in rows_out if r["n_moves"] > 0]
print(f"edited feeds {len(edited)} ({len(edited) / n:.1%}); improved {sum(r['delta'] > 1e-12 for r in edited)}, "
      f"worse {sum(r['delta'] < -1e-12 for r in edited)}, same {sum(abs(r['delta']) <= 1e-12 for r in edited)}")
mv = sum(r["moved"] for r in rows_out); mp = sum(r["moved_pos"] for r in rows_out)
print(f"moved items {mv}, of which positive {mp} ({mp / max(mv, 1):.1%})")
by_conf = defaultdict(list)
for r in rows_out:
    by_conf[r["conf"]].append(r)
for c in ["high", "medium", "low", "none"]:
    rr = by_conf.get(c, [])
    if rr:
        dd = np.array([r["delta"] for r in rr])
        mvc = sum(r["moved"] for r in rr); mpc = sum(r["moved_pos"] for r in rr)
        print(f"  conf {c:6s}: feeds {len(rr):4d}  sum delta {dd.sum():+.3f}  mean {dd.mean():+.4f}  "
              f"moved {mvc} positive {mpc}")
# moves to the top 3 only
top = [(r, u) for r in rows_out for u in r["up"] if u[1] <= 3]
print(f"moves into the top 3: {len(top)}")
json.dump(rows_out, open(HERE / "val/private/last_scored.json", "w"), indent=0)
