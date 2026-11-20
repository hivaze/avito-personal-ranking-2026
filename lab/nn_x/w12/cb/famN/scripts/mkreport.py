"""Assemble reports/CATBOOST_1115/famN.md = scripts/report_head.txt + the tables of mktable.py + scripts/report_n3.txt + scripts/report_static.txt.
usage: CUDA_VISIBLE_DEVICES="" CBX_THREADS=4 python mkreport.py   (run read.py --n 30 and --n 6 first: they write logs/compare_n*.json)"""
import os, subprocess, sys
H = os.path.dirname(os.path.abspath(__file__))
OUT = "<repo>/reports/CATBOOST_1115/famN.md"


def table(nk, mode):
    r = subprocess.run([sys.executable, os.path.join(H, "mktable.py"), str(nk), mode], capture_output=True, text=True, env=os.environ)
    assert r.returncode == 0, r.stderr[-2000:]
    return "\n".join(l for l in r.stdout.splitlines() if "Warning" not in l).strip() + "\n"


rd = lambda f: open(os.path.join(H, f)).read().rstrip() + "\n"
parts = [rd("report_head.txt"),
         "\n## 2. One row per arm\n\nReferences first. \"trees\" = selected tree count per seed. d = arm minus its same-loss reference, paired per feed and seed-paired (section 5); "
         "NATS = gain of the shared screen `nats.py` over the W16 train-only pool (bar 0.0016 on val AND holdout).\n\n",
         table(30, "arms"), "\n", table(6, "arms"),
         "\n## 3. The nats.py numbers of every (arm, seed)\n\nThe literal one-line outputs of the tool are in the process logs (`grep NATS nn_x/w12/cb/famN/logs/p*.log`) and in "
         "`nn_x/w12/cb/famN/logs/nats_lines.txt`; this table carries the same numbers.\n\n",
         table(30, "seeds"), "\n", table(6, "seeds"),
         "\nrho in these tables is the `cbx.Evaluator` value (mean within-feed correlation); the `nats.py` line prints one Pearson over all rows, 0.003 to 0.008 higher on val. "
         "A tag that appears twice in `logs/results.jsonl` (`Yref_n30k_s1`, `Y_txt_s_n30k_s0`) has its discarded od100 run first; the last line counts.\n\n",
         rd("report_n3.txt"), "\n", rd("report_static.txt"),
         "\n## 12. Appendix: the literal `nats.py` lines\n\n```\n", open(os.path.join(H, "..", "logs", "nats_lines.txt")).read().rstrip(), "\n```\n"]
open(OUT, "w").write("".join(parts))
print("written", OUT, sum(len(p) for p in parts), "chars")
