"""Reviewer: lane_full.sh commands vs the ORIGINAL members' logged arguments (own extraction of parse() from the copies)."""
import ast, argparse, subprocess, shlex, sys, os
R = "<repo>"; FD = f"{R}/nn_x/w12/stale/fulldata"
def parser_of(path):
    tree = ast.parse(open(path).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    consts = [n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, (ast.Constant, ast.Dict, ast.List, ast.Tuple))]
    ns = {"argparse": argparse}
    for c in consts:
        try: exec(compile(ast.Module([c], []), path, "exec"), ns)
        except Exception: pass
    for _ in range(20):
        try:
            exec(compile(ast.Module([fn], []), path, "exec"), ns); break
        except NameError as e:
            raise
    return ns
def run_parse(ns, argv):
    old = sys.argv; sys.argv = ["x"] + argv
    try:
        while True:
            try: return vars(ns["parse"]())
            except NameError as e:
                nm = str(e).split("'")[1]; ns[nm] = 0.5 if nm == "POOL_P_DEFAULT" else {}
    finally: sys.argv = old
def dry(jobs, env=None):
    e = dict(os.environ, DRY="1", **(env or {}))
    out = subprocess.run(["bash", f"{FD}/lane_full.sh", "0", jobs], capture_output=True, text=True, env=e)
    assert out.returncode == 0, out.stderr
    return [l for l in out.stdout.splitlines() if l.startswith("GPU N:")]
def logged(h, v):
    l = open(f"{R}/nn_x/{h}/logs/{v}.log").readline()
    return ast.literal_eval(l[l.index("args: ") + 6:])
ns_w, ns_a = parser_of(f"{FD}/train_full.py"), parser_of(f"{FD}/train_full_abl.py")
defaults_w, defaults_a = run_parse(ns_w, ["--variant", "x"]), run_parse(ns_a, ["--variant", "x"])
jobs = [("F", a, s, "w5") for a in ("kd05w192", "kd05t25w192") for s in range(951, 959)] + [("P", a, s, "abl") for a in ("pop1kd05t25", "pop1kd05w192") for s in range(951, 955)]
bad = 0
for st, arm, s, h in jobs:
    line = dry(f"{st}:{arm}:{s}")[0]
    toks = shlex.split(line.split(") ", 1)[1])
    script = toks[1]; argv = toks[2:]
    ns, dfl = (ns_w, defaults_w) if h == "w5" else (ns_a, defaults_a)
    got = run_parse(ns, argv); ref = logged(h, f"{h}_e10_{arm}_s{s}")
    assert got["variant"] == f"{h}_fd10_{arm}_s{s}" and got["holdout_in_train"] is True and script.endswith("train_full.py" if h == "w5" else "train_full_abl.py")
    diff = [k for k in got if k not in ("variant", "holdout_in_train") and (got[k] != ref[k] if k in ref else got[k] != dfl[k])]
    extra = [k for k in ref if k not in got]
    newer = [k for k in got if k not in ref and k != "holdout_in_train"]
    bad += bool(diff or extra)
    print(f"{st}:{arm}:{s}: {len(got)} args; differing from the member's logged args: {diff}; keys only in the member's log: {extra}; keys newer than the log (at parser default): {newer}; seed {got['seed']} kd {os.path.basename(got['kd_file'])} d {got['d']} holdout {got['holdout']} keep {got['keep']} epochs {got['epochs']}")
# re-cool forms against lane.sh K:ref / L:ref DRY
for arm, st in (("kd05w192", "K"), ("kd05t25w192", "L")):
    for s in range(951, 959):
        a = shlex.split(dry(f"R:{arm}:{s}")[0].split(") ", 1)[1])[2:]
        o = subprocess.run(["bash", f"{R}/nn_x/w5/lane.sh", "0", f"{st}:ref:{s}"], capture_output=True, text=True, env=dict(os.environ, DRY="1")).stdout.strip()
        b = shlex.split(o.split(": ", 1)[1])[2:]
        ok = a[2:-1] == b[2:] and a[-1] == "--holdout_in_train" and a[1] == f"w5_fdcd_{arm}_s{s}" and b[1] == f"w5_{st.lower()}cd_ref_s{s}"
        bad += not ok
        print(f"R:{arm}:{s} == lane.sh {st}:ref:{s} + --holdout_in_train: {ok}")
print("ALL PASS" if not bad else f"PROBLEMS: {bad}")
