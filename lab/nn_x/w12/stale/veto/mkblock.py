"""veto: write the CPU stale-val block of reports/STALE_1115/veto.md from out/read4.json (between the two HTML comment markers, or in
place of the token CPU_STALE_BLOCK on the first call). Run after read4.py. Edits only my own report file."""
import json
import time
from pathlib import Path

ROOT = Path("<repo>"); HERE = ROOT / "nn_x/w12/stale/veto"; REP = ROOT / "reports/STALE_1115/veto.md"
R = json.load(open(HERE / "out/read4.json"))
f = lambda t: "n/a" if t is None else f"{t[0]:+.5f} +- {t[1]:.5f}"
z = lambda t: f"{t[0] / t[1]:+.2f}" if t and t[1] > 0 else "n/a"
cs = [r for r in R if r["label"].startswith("[CPU stale]")]
lv = [r for r in R if r["label"].startswith("LEVEL")]
done = sorted(p.name[4:-13] for p in (HERE / "preds").glob("cpu_*_staleval.npy"))
L = ["<!-- cpu-block-start -->",
     f"State at {time.strftime('%FT%TZ', time.gmtime())}: {len(done)} CPU stale-val files of mine exist (`nn_x/w12/stale/veto/preds/`); "
     "reads that need a file not yet written are absent from the table, not estimated.", ""]
if cs:
    L += ["| CPU stale-val read (14,423 feeds, CPU against CPU) | stale-val delta | z | private-weighted | M>=50 | M<50 | coldpos | warmpos |", "|---|---|---|---|---|---|---|---|"]
    for r in cs:
        L.append(f"| {r['label'][12:]} | {f(r['cstale'])} [{r['levelA']:.5f} vs {r['levelB']:.5f}] | {z(r['cstale'])} | {f(r['cstale_priv'])} | "
                 f"{f(r['cstale_M>=50'])} | {f(r['cstale_M<50'])} | {f(r['cstale_coldpos'])} | {f(r['cstale_warmpos'])} |")
    L.append("")
if lv:
    L += ["Pool level under private-like inputs (CPU stale against the stored bf16 fresh file, so each drop carries the numerics offset of about +-0.0003; "
          "read the difference between rows):", "", "| pool | stale-val | fresh val (stored) | drop |", "|---|---|---|---|"]
    for r in lv:
        L.append(f"| {r['label'][6:]} | {r['stale']:.5f} | {r['fresh']:.5f} | {f(r['drop'])} |")
    L.append("")
L.append("<!-- cpu-block-end -->")
blk = "\n".join(L)
s = REP.read_text()
if "CPU_STALE_BLOCK" in s:
    s = s.replace("CPU_STALE_BLOCK", blk)
else:
    a, b = s.index("<!-- cpu-block-start -->"), s.index("<!-- cpu-block-end -->") + len("<!-- cpu-block-end -->")
    s = s[:a] + blk + s[b:]
REP.write_text(s)
print(blk)
