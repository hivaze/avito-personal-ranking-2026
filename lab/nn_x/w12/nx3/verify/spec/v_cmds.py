"""Independent verifier: dry command of every job of queue_spec_jobs.txt vs the LOGGED args of its reference member.
Own code: AST-extract parse() of the trainer that will run, parse the dry command with it, literal_eval the args line of the reference log."""
import ast, re, subprocess, sys, shlex, os, hashlib
R = "<repo>"
NX = f"{R}/nn_x/w12/nx3"
BUNDLE = "nx_rp3_z,nx_rp3_rk,has:nx_rp3_z"


def parser_of(path):
    src = open(path).read()
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"]
    assert len(fn) == 1, path
    code = ast.get_source_segment(src, fn[0])
    ns = {"argparse": __import__("argparse"), "POOL_P_DEFAULT": 0.5}
    exec(code, ns)
    return ns["parse"]


def parse_with(path, argv):
    old = sys.argv
    sys.argv = ["x"] + argv
    try:
        return vars(parser_of(path)())
    finally:
        sys.argv = old


def defaults_of(path):
    return parse_with(path, [])


def logged_args(log, which=0):
    out = []
    for ln in open(log, errors="replace"):
        m = re.match(r"^==== (.*?) args: (\{.*\})\s*$", ln)
        if m:
            out.append((m.group(1), ast.literal_eval(m.group(2))))
    return out


def dry_fn(arm, seed, g=0):
    o = subprocess.run(["bash", f"{NX}/lane_full_nx3.sh", str(g), f"FN:{arm}:{seed}"], capture_output=True, text=True,
                       env={**os.environ, "DRY": "1", "W5_GPUS": "0 1 2 3 4 5 6 7"})
    assert o.returncode == 0, o.stderr
    lines = [l for l in o.stdout.splitlines() if l.strip()]
    assert len(lines) == 1, lines
    m = re.match(r"^GPU (\d+): \(cd (\S+) && PYTHONPATH=(\S+)\) (\S+) (\S+) (.*)$", lines[0])
    assert m, lines[0]
    return dict(gpu=m.group(1), cwd=m.group(2), pp=m.group(3), py=m.group(4), script=m.group(5), argv=shlex.split(m.group(6)), raw=lines[0])


def dry_b(seed):
    """static extraction from launch_nx3_kd_a7.sh (the script has no DRY form)"""
    src = open(f"{NX}/launch_nx3_kd_a7.sh").read()
    m = re.search(r"\nsetsid nohup (\S+) (train\.py) (--variant \$v .*?)\n > logs/\$v\.stdout 2>&1 < /dev/null 9>&- &\n", src, re.S)
    assert m, "command block not found"
    block = (m.group(3) + "\n").replace("\\\n", " ").strip()
    assert "$arm" not in block
    mv = re.search(r"\nv=(\S+)\n", src); assert mv.group(1) == "abl_e10_${arm}kd05w192_s$s", mv.group(1)
    mx = re.search(r"\nNX3=(\S+)\n", src); nx = mx.group(1)
    assert re.search(r"case \$arm in nx3\) XA=\$NX3 ;; \*\) echo \"arm must be nx3\"; exit 1 ;; esac", src)
    v = f"abl_e10_nx3kd05w192_s{seed}"
    block = block.replace("$v", v).replace("$XA", nx).replace("$s", str(seed))
    assert "$" not in block, block
    mcd = re.search(r"\ncd (\S+)\n", src)
    return dict(cwd=mcd.group(1), py=m.group(1), script=m.group(2), argv=shlex.split(block), raw=f"(cd {mcd.group(1)}) {m.group(1)} {m.group(2)} {block}")


def cmp(new, ref, allowed_new_only, allowed_ref_only, allowed_diff):
    """returns list of problems"""
    bad = []
    for k in sorted(set(new) | set(ref)):
        if k in new and k in ref:
            if new[k] != ref[k] or type(new[k]) != type(ref[k]):
                if k in allowed_diff and allowed_diff[k](new[k], ref[k]):
                    continue
                bad.append(f"DIFF {k}: new {new[k]!r} ({type(new[k]).__name__}) ref {ref[k]!r} ({type(ref[k]).__name__})")
        elif k in new:
            if k in allowed_new_only and new[k] == allowed_new_only[k]:
                continue
            bad.append(f"NEW-ONLY {k}: {new[k]!r}")
        else:
            if k in allowed_ref_only and ref[k] == allowed_ref_only[k]:
                continue
            bad.append(f"REF-ONLY {k}: {ref[k]!r}")
    return bad


ABL_ONLY_OFF = {"xt_exclude": "", "xt_rel_extra": "", "retr_cat": 0, "xt_add_relonly": "", "kv64": 0, "xt_add": BUNDLE}
W5_ONLY_OFF = {"aug_all": False, "xfold": -1, "xfold_unlab": False, "prand2": "", "rdrop_cd": 0.0, "sam_cd": 0.0}

jobs = [l.strip() for l in open(f"{NX}/logs/queue_spec_jobs.txt") if l.strip() and not l.lstrip().startswith("#")]
print("jobs:", len(jobs))
TFA = f"{R}/nn_x/w12/stale/fulldata/train_full_abl.py"
TF = f"{R}/nn_x/w12/stale/fulldata/train_full.py"
allbad = 0
seen = set()
for j in jobs:
    p = j.split(":")
    if p[0] == "FN":
        arm, seed = p[1], int(p[2])
        d = dry_fn(arm, seed)
        assert d["script"] == TFA and d["cwd"] == f"{R}/nn_x/abl" and d["pp"] == f"{R}/nn_x/abl" and d["py"] == f"{R}/.venv/bin/python", d
        new = parse_with(TFA, d["argv"])
        refarm = arm.replace("nx3", "", 1)
        refv = f"w5_fd10_{refarm}_s{seed}"
        newv = f"abl_fd10_{arm}_s{seed}"
        la = logged_args(f"{R}/nn_x/w5/logs/{refv}.log")
    else:
        seed = int(p[1])
        d = dry_b(seed)
        assert d["cwd"] == f"{R}/nn_x/abl" and d["script"] == "train.py" and d["py"] == f"{R}/.venv/bin/python", d
        new = parse_with(f"{R}/nn_x/abl/train.py", d["argv"])
        refv = f"w5_e10_kd05w192_s{seed}"
        newv = f"abl_e10_nx3kd05w192_s{seed}"
        la = logged_args(f"{R}/nn_x/w5/logs/{refv}.log")
    assert newv not in seen; seen.add(newv)
    assert len(la) == 1, (refv, len(la))
    ref = la[0][1]
    assert ref["variant"] == refv
    bad = cmp(new, ref, ABL_ONLY_OFF, W5_ONLY_OFF, {"variant": lambda a, b: a == newv and b == refv})
    # explicit facts
    facts = []
    facts.append(("xt_add", new.get("xt_add") == BUNDLE))
    facts.append(("variant", new["variant"] == newv))
    if p[0] == "FN":
        facts.append(("holdout_in_train", new.get("holdout_in_train") is True and ref.get("holdout_in_train") is True and "--holdout_in_train" in d["argv"]))
        T = "teacher_mean25" if "t25" in arm else "teacher_mean11"
        facts.append(("kd_file", new["kd_file"] == f"{R}/nn_x/w12/kd/out/{T}.npy" == ref["kd_file"]))
    else:
        facts.append(("no holdout_in_train key/flag", "holdout_in_train" not in new and "--holdout_in_train" not in d["argv"] and "holdout_in_train" not in ref))
        facts.append(("holdout 30000 select_on holdout", new["holdout"] == 30000 and new["select_on"] == "holdout" and ref["holdout"] == 30000))
        facts.append(("kd_file", new["kd_file"] == f"{R}/nn_x/w12/kd/out/teacher_mean11.npy" == ref["kd_file"]))
    fb = [n for n, ok in facts if not ok]
    shared = len(set(new) & set(ref))
    print(f"{j:28s} {newv:34s} vs {refv:28s} ({la[0][0]}) keys new {len(new)} ref {len(ref)} shared {shared} "
          f"new-only {sorted(set(new) - set(ref))} ref-only {sorted(set(ref) - set(new))} -> {'OK' if not bad and not fb else 'BAD'}")
    if j in ("FN:nx3kd05w192:951", "FN:nx3kd05t25w192:951", "B:951"):
        print("   DRY:", d["raw"])
    for b in bad: print("   ", b)
    for b in fb: print("    FACT FAIL", b)
    allbad += len(bad) + len(fb)
print("TOTAL PROBLEMS:", allbad)

# ---- parser defaults: w5 vs abl, and the full-data copies
print("\n== parser defaults ==")
pairs = [(f"{R}/nn_x/w5/train.py", f"{R}/nn_x/abl/train.py"), (TF, TFA), (f"{R}/nn_x/abl/train.py", TFA), (f"{R}/nn_x/w5/train.py", TF)]
for a, b in pairs:
    da, db = defaults_of(a), defaults_of(b)
    diff = {k: (da[k], db[k]) for k in set(da) & set(db) if da[k] != db[k] or type(da[k]) != type(db[k])}
    print(os.path.relpath(a, R), "vs", os.path.relpath(b, R), ": shared", len(set(da) & set(db)), "default diffs", diff,
          "| only-first", {k: da[k] for k in sorted(set(da) - set(db))}, "| only-second", {k: db[k] for k in sorted(set(db) - set(da))})

# ---- same-harness precedent: stored abl_fd10_pop1kd05w192_s951 (train_full_abl.py) and abl_e10_pop1kd05w192_s951 (abl/train.py)
print("\n== same-harness precedents ==")
for newj, prec in (("FN:nx3kd05w192:951", f"{R}/nn_x/abl/logs/abl_fd10_pop1kd05w192_s951.log"), ("B:951", f"{R}/nn_x/abl/logs/abl_e10_pop1kd05w192_s951.log"),
                   ("FN:nx3kd05t25w192:951", f"{R}/nn_x/abl/logs/abl_fd10_pop1kd05t25_s951.log")):
    p = newj.split(":")
    if p[0] == "FN":
        d = dry_fn(p[1], int(p[2])); new = parse_with(TFA, d["argv"])
    else:
        d = dry_b(int(p[1])); new = parse_with(f"{R}/nn_x/abl/train.py", d["argv"])
    if not os.path.exists(prec):
        print(newj, "precedent log missing", prec); continue
    la = logged_args(prec)
    ref = la[0][1]
    diffs = {k: (new.get(k, "<absent>"), ref.get(k, "<absent>")) for k in sorted(set(new) | set(ref)) if new.get(k, "<absent>") != ref.get(k, "<absent>")}
    print(newj, "vs", os.path.basename(prec), "(", la[0][0], ") n_args_lines", len(la), "keys", len(new), len(ref), "diffs:", diffs)
