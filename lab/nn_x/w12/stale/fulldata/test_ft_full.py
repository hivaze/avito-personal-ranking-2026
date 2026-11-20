"""Static + unit proofs for finetune_full.py / finetune_full_abl.py (--full_val). CPU only, no data load, seconds.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python nn_x/w12/stale/fulldata/test_ft_full.py

U0 for both copies (nn_x/w5/finetune.py and nn_x/abl/popfree/finetune.py):
  (a) frozen reference == ref/md5.txt == the live file;
  (b) AST(copy minus the FULLVAL statements) == AST(reference), module docstring dropped. Stripped set, exact and counted:
      1 `from ft_full import FullVal`, 1 add_argument("--full_val"), 1 `fv = None`, 1 `if a.full_val:` (no else),
      2 `if fv is not None:` (no else), 5 conditional expressions `X if fv is None else Y` -> X;
  (c) parser: --full_val default False, all other defaults equal; (d) no fv / FullVal / a.full_val node survives the strip.
U1 FullVal unit behaviour: a.folds -> 1; splits() = one pair (all, all); refusals; finish() renames the oof_* keys and
   refuses when a ..._val_oof.npy of the same --out exists.
Writes logs/test_ft_full.log.
"""
import argparse
import ast
import collections
import hashlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == ""
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "<repo>"
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

LOG = open(f"{HERE}/logs/test_ft_full.log", "w")
FAIL = []
PAIRS = [("w5", f"{HERE}/finetune_full.py", f"{HERE}/ref/finetune_w5.py.txt", f"{ROOT}/nn_x/w5/finetune.py"),
         ("abl", f"{HERE}/finetune_full_abl.py", f"{HERE}/ref/finetune_abl.py.txt", f"{ROOT}/nn_x/abl/popfree/finetune.py")]
EXPECT = {"import": 1, "add_argument": 1, "fv_none": 1, "if_flag": 1, "if_fv": 2, "ifexp": 5}


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def check(name, ok, info=""):
    P(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}")
    if not ok:
        FAIL.append(name)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def is_fv_cmp(n, op):
    return (isinstance(n, ast.Compare) and isinstance(n.left, ast.Name) and n.left.id == "fv" and len(n.ops) == 1
            and isinstance(n.ops[0], op) and isinstance(n.comparators[0], ast.Constant) and n.comparators[0].value is None)


def is_flag(n):
    return isinstance(n, ast.Attribute) and n.attr == "full_val" and isinstance(n.value, ast.Name) and n.value.id == "a"


class Strip(ast.NodeTransformer):
    def __init__(self):
        self.c = collections.Counter()

    def visit_ImportFrom(self, n):
        if n.module == "ft_full":
            self.c["import"] += 1
            return None
        return n

    def visit_Expr(self, n):
        v = n.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "add_argument" and v.args
                and isinstance(v.args[0], ast.Constant) and v.args[0].value == "--full_val"):
            self.c["add_argument"] += 1
            return None
        return self.generic_visit(n)

    def visit_Assign(self, n):
        if (len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "fv"
                and isinstance(n.value, ast.Constant) and n.value.value is None):
            self.c["fv_none"] += 1
            return None
        return self.generic_visit(n)

    def visit_If(self, n):
        if is_flag(n.test) or is_fv_cmp(n.test, ast.IsNot):
            assert not n.orelse
            self.c["if_flag" if is_flag(n.test) else "if_fv"] += 1
            return None
        return self.generic_visit(n)

    def visit_IfExp(self, n):
        if is_fv_cmp(n.test, ast.Is):
            self.c["ifexp"] += 1
            return self.visit(n.body)
        return self.generic_visit(n)


def drop_doc(tree):
    b = tree.body
    if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
        tree.body = b[1:]
    return tree


def load_parse(path):
    tree = ast.parse(open(path).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    ns = {"argparse": argparse}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "ft_parse", "exec"), ns)

    def parse(argv):
        old = sys.argv
        sys.argv = ["finetune.py"] + argv
        try:
            return vars(ns["parse"]())
        finally:
            sys.argv = old
    return parse


P("== U0 code identity with the flag absent")
want = dict(line.split()[::-1] for line in open(f"{HERE}/ref/md5.txt").read().splitlines())
for tag, copy, ref, live in PAIRS:
    P(f" -- {tag}: {os.path.basename(copy)} vs {os.path.relpath(live, ROOT)}")
    check(f"{tag} (a) frozen reference == ref/md5.txt == live file", md5(ref) == want[os.path.basename(ref)] == md5(live), md5(live))
    st = Strip()
    stripped = ast.fix_missing_locations(st.visit(drop_doc(ast.parse(open(copy).read()))))
    refa = drop_doc(ast.parse(open(ref).read()))
    check(f"{tag} (b) stripped statement counts == expected", dict(st.c) == EXPECT, dict(st.c))
    check(f"{tag} (b) AST(copy minus FULLVAL statements) == AST(reference)", ast.dump(stripped) == ast.dump(refa), f"{len(ast.dump(refa))} chars")
    left = [n for n in ast.walk(stripped) if (isinstance(n, ast.Name) and n.id in ("fv", "FullVal")) or is_flag(n)]
    check(f"{tag} (d) no fv / FullVal / a.full_val node survives the strip", not left)
    d_new, d_ref = load_parse(copy)([]), load_parse(ref)([])
    check(f"{tag} (c) --full_val default False; all {len(d_ref)} other defaults == reference", d_new.pop("full_val") is False and d_new == d_ref)
P("== U1 FullVal unit behaviour")
import ft_full as FF  # noqa: E402

logs = []


def args(**k):
    d = dict(pseudo="", add_random=0, ema=0.99, select="snapshot", folds=5, seeds=3)
    d.update(k)
    return types.SimpleNamespace(**d)


a = args()
fv = FF.FullVal(a, "w5_x_n2cpv3fv3", logs.append)
(pair,) = list(fv.splits(14423))
check("a.folds -> 1; splits() = one (fit, eval) pair, both all 14,423 feeds; val file name carries INSAMPLE",
      a.folds == 1 and np.array_equal(pair[0], np.arange(14423)) and np.array_equal(pair[1], pair[0]) and fv.val_name == "nn_w5_x_n2cpv3fv3_valINSAMPLE.npy")
for k, v, o in (("pseudo", "t", "w5_x_fv"), ("add_random", 100, "w5_x_fv"), ("select", "best", "w5_x_fv"), ("x", 0, "w5_x_n2cpv3s3"), ("x", 0, "w5_x_n2cpv3")):
    try:
        FF.FullVal(args(**({k: v} if k != "x" else {}), **({"ema": 0.0} if k == "select" else {})), o, logs.append)
        ok = False
    except AssertionError:
        ok = True
    check(f"refused: {k}={v!r} out={o}", ok)
with tempfile.TemporaryDirectory(dir=f"{HERE}/logs") as td:
    td = Path(td)
    (td / "w5_x_n2cpv3fv3_ft.json").write_text(json.dumps(dict(args={}, oof_by_epoch=[0.39, 0.41], best_epoch=1, oof_ndcg_best_epoch=0.41, oof_ndcg_snapshot=0.41, oof_ndcg_ema_final=0.41)))
    np.save(td / fv.val_name, np.zeros(3))
    np.save(td / "nn_w5_x_n2cpv3fv3_test.npy", np.zeros(3))
    fv.finish(td, td, logs.append)
    j = json.loads((td / "w5_x_n2cpv3fv3_ft.json").read_text())
    check("finish(): oof_* keys renamed to insample_*, full_val True, epoch-0 train-only NDCG kept",
          not [k for k in j if k.startswith("oof_")] and j["full_val"] is True and j["train_only_val_ndcg_epoch0"] == 0.39 and "insample_by_epoch" in j)
    np.save(td / "nn_w5_x_n2cpv3fv3_val_oof.npy", np.zeros(3))
    try:
        fv.finish(td, td, logs.append)
        ok = False
    except AssertionError:
        ok = True
    check("finish() refuses when a ..._val_oof.npy of the same --out exists", ok)
P(f"\n{'ALL PASS' if not FAIL else 'FAILED: ' + str(FAIL)}")
sys.exit(1 if FAIL else 0)
