"""Lead only: audit the analysts' tool calls and written code for any touch of sealed data.

usage: audit.py TRANSCRIPT_DIR
Flags: any tool input or analyst-written file that mentions the private folder, a label file, a
prediction or pool file, another analyst's folder (by path), or a non-allowed write location.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

BASE = "<repo>/lab/nn_x/errfix_1117"
LENSES = ["head", "category", "within_cat", "segments", "history_intent", "popularity_fresh", "residual_model", "deep_dive"]
FORBIDDEN = [r"errfix_1117/private", r"cand_target", r"holdout_y", r"artifacts/predictions", r"ens_[A-Za-z0-9]+_(ft|to)_",
             r"confirm\.parquet", r"holdout\.parquet", r"test\.parquet", r"lab/nn_x/w5/preds", r"lab/nn_x/abl/preds"]
root = Path(sys.argv[1])
calls, flagged = Counter(), []
for p in sorted(root.rglob("agent-*.jsonl")):
    text = p.read_text()
    lens = next((l for l in LENSES if f"agents/{l}/" in text[:20000]), "?")
    for line in text.splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = rec.get("message") or {}
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for c in content:
            if not (isinstance(c, dict) and c.get("type") == "tool_use"):
                continue
            name, inp = c.get("name"), json.dumps(c.get("input", {}))
            calls[(lens, name)] += 1
            for pat in FORBIDDEN:
                if re.search(pat, inp):
                    flagged.append((lens, name, pat, inp[:300]))
            for other in LENSES:
                if other != lens and f"agents/{other}/" in inp:
                    flagged.append((lens, name, f"other folder {other}", inp[:300]))
            if name in ("Write", "Edit", "NotebookEdit"):
                fp = c.get("input", {}).get("file_path", "")
                if not fp.startswith(f"{BASE}/agents/{lens}/"):
                    flagged.append((lens, name, "write outside own folder", fp))
# code the analysts wrote
for lens in LENSES:
    for f in Path(f"{BASE}/agents/{lens}").rglob("*"):
        if f.is_file() and f.suffix in (".py", ".sh", ".txt", ".json"):
            t = f.read_text(errors="ignore")
            for pat in FORBIDDEN:
                if re.search(pat, t):
                    flagged.append((lens, "file", pat, str(f)))
print("tool calls by lens:")
for (lens, name), n in sorted(calls.items()):
    print(f"  {lens:18s} {name:18s} {n}")
print("FLAGGED:", len(flagged))
for f in flagged[:80]:
    print("  ", f)
