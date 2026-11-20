"""List every tool call of the workflow's agents and flag anything other than Read on the allowed files.

usage: audit_tools.py TRANSCRIPT_DIR
"""
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
ALLOWED_PREFIX = "/tmp/scratchpad/llmfix/"
ALLOWED_TOOLS = {"Read", "StructuredOutput"}
calls, flagged, files = Counter(), [], Counter()
for p in sorted(root.rglob("agent-*.jsonl")):
    for line in open(p):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = rec.get("message") or {}
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, dict) and c.get("type") == "tool_use":
                name, inp = c.get("name"), c.get("input", {})
                calls[name] += 1
                fp = inp.get("file_path", "")
                if name == "Read":
                    files[fp.replace(ALLOWED_PREFIX, "")] += 1
                bad = name not in ALLOWED_TOOLS or (name == "Read" and (
                    not fp.startswith(ALLOWED_PREFIX) or "/private/" in fp or "score" in fp or "audit" in fp))
                if bad:
                    flagged.append((p.name, name, json.dumps(inp)[:200]))
print("tool calls:", dict(calls))
print("files read:", dict(files))
print("FLAGGED:", len(flagged))
for f in flagged[:50]:
    print("  ", f)
