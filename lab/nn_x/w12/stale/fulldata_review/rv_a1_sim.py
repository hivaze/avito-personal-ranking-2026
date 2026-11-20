"""Reviewer: does accept_twins.py A1 'arguments == the reference run's' survive the 04:52 change of nn_x/w5/train.py?
Replicates the 6 lines of accept_twins.a1 that build `diff` / `late` (copied logic, own inputs):
  ra = args of a REAL reference re-cool started after 04:52 (first log line), ta = args the STUDY's train_full.py parser gives the RH twin command."""
import ast, sys, subprocess, shlex, os, argparse
R = "<repo>"; FD = f"{R}/nn_x/w12/stale/fulldata"
def parser(path):
    tree = ast.parse(open(path).read()); fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    ns = {"argparse": argparse, "POOL_P_DEFAULT": 0.5}; exec(compile(ast.Module([fn], []), path, "exec"), ns)
    def run(argv):
        old = sys.argv; sys.argv = ["x"] + argv
        try:
            while True:
                try: return vars(ns["parse"]())
                except NameError as e: ns[str(e).split("'")[1]] = {}
        finally: sys.argv = old
    return run
for copy_path, label in ((f"{FD}/train_full.py", "study's copy (source e4b5f0d7)"), (f"{R}/nn_x/w12/stale/fulldata_review/rederived/train_full.py", "re-derived copy (source 26d254d0)")):
    run = parser(copy_path); dflt = run(["--variant", "x"])
    for st, arm, s, refv in (("RH", "kd05w192", 951, "w5_kcd_hz_s951"), ("RH", "kd05t25w192", 953, "w5_lcd_hz_s953"), ("R", "kd05w192", 951, "w5_kcd_ref_s951")):
        line = subprocess.run(["bash", f"{FD}/lane_full.sh", "0", f"{st}:{arm}:{s}"], capture_output=True, text=True, env=dict(os.environ, DRY="1")).stdout.strip()
        ta = run(shlex.split(line.split(") ", 1)[1])[2:]); ta["keep"] = "last"
        l = open(f"{R}/nn_x/w5/logs/{refv}.log").readline(); ra = ast.literal_eval(l[l.index("args: ") + 6:])
        skip = {"variant", "holdout_in_train", "keep"}
        diff = sorted(k for k in set(ra) | set(ta) if k not in skip and ra.get(k, "<absent>") != ta.get(k, "<absent>"))
        late = [k for k in diff if k not in ra and k in dflt and ta[k] == dflt[k]]
        diff = [k for k in diff if k not in late]
        print(f"{label}: {st}:{arm}:{s} vs reference {refv} (started {l[5:29]}): differing keys {dict((k, (ra.get(k, '<absent>'), ta.get(k, '<absent>'))) for k in diff)} -> A1 arguments item {'PASS' if not diff else 'DEFECT (false alarm)'}; late {late}")
