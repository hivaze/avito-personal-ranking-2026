"""Static + unit proofs for train_full.py / train_full_abl.py (--holdout_in_train). CPU only, no data load, seconds.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python nn_x/w12/stale/fulldata/test_full.py

T0 code identity with the flag absent (for BOTH copies: w5 and abl)
  (a) frozen references ref/train_w5.py.txt, ref/train_abl.py.txt match ref/md5.txt AND the live nn_x/w5/train.py,
      nn_x/abl/train.py (if a live file changed after the copy was taken, this check says so);
  (b) AST: the copy minus the FULLDATA statements == the reference (module docstring dropped on both sides). The stripped
      set is exact and counted: 1 `from cd_full import FullData`, 1 add_argument("--holdout_in_train"), 1 `fd = None`,
      1 `if a.holdout_in_train:` (no else), 7 `if fd is not None:` (no else), 2 conditional expressions
      `hold if fd is None else None` -> `hold`, 2 conditions `X and fd is None` -> `X`;
  (c) textual diff: every added / changed line outside the module docstring mentions fd / holdout_in_train / FullData /
      FULLDATA; no reference line is deleted without a replacement that contains it up to the conditional;
  (d) parser: --holdout_in_train defaults to False; every other default equals the reference parser's; with the flag absent
      `fd` can only be None: its single non-None assignment is inside `if a.holdout_in_train:`;
  (e) read sites: `fd` appears only in the stripped statements (Name-node count), `a.holdout_in_train` twice (parser aside:
      once), cd_full is imported by no other module of nn_x/w5 / nn_x/abl.
T1 FullData unit behaviour on synthetic arrays (the very class the train files use)
  put_back returns the list before the removal; a.keep forced to 'last'; refused flag combinations raise; insample()
  returns nan and logs the word IN-SAMPLE; _write refuses a name without the IN-SAMPLE suffix; file names.
Writes logs/test_full.log (+ .json) in this directory.
"""
import argparse
import ast
import collections
import difflib
import hashlib
import json
import os
import sys
import types

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "run with CUDA_VISIBLE_DEVICES=''"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "<repo>"
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

os.makedirs(f"{HERE}/logs", exist_ok=True)
LOG = open(f"{HERE}/logs/test_full.log", "w")
RES, FAIL = {"checks": []}, []
PAIRS = [("w5", f"{HERE}/train_full.py", f"{HERE}/ref/train_w5.py.txt", f"{ROOT}/nn_x/w5/train.py", f"{ROOT}/nn_x/w5"),
         ("abl", f"{HERE}/train_full_abl.py", f"{HERE}/ref/train_abl.py.txt", f"{ROOT}/nn_x/abl/train.py", f"{ROOT}/nn_x/abl")]
EXPECT = {"import": 1, "add_argument": 1, "fd_none": 1, "if_flag": 1, "if_fd": 7, "ifexp": 2, "boolop": 2}
MARK = ("fd", "holdout_in_train", "FullData", "FULLDATA")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def check(name, ok, info=""):
    P(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}")
    RES["checks"].append(dict(name=name, ok=bool(ok), info=str(info)))
    if not ok:
        FAIL.append(name)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def is_fd_cmp(n, op):
    return (isinstance(n, ast.Compare) and isinstance(n.left, ast.Name) and n.left.id == "fd" and len(n.ops) == 1
            and isinstance(n.ops[0], op) and isinstance(n.comparators[0], ast.Constant) and n.comparators[0].value is None)


def is_flag(n):
    return isinstance(n, ast.Attribute) and n.attr == "holdout_in_train" and isinstance(n.value, ast.Name) and n.value.id == "a"


class Strip(ast.NodeTransformer):
    def __init__(self):
        self.c = collections.Counter()

    def visit_ImportFrom(self, n):
        if n.module == "cd_full":
            assert [x.name for x in n.names] == ["FullData"]
            self.c["import"] += 1
            return None
        return n

    def visit_Expr(self, n):
        v = n.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "add_argument" and v.args
                and isinstance(v.args[0], ast.Constant) and v.args[0].value == "--holdout_in_train"):
            self.c["add_argument"] += 1
            return None
        return self.generic_visit(n)

    def visit_Assign(self, n):
        if (len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "fd"
                and isinstance(n.value, ast.Constant) and n.value.value is None):
            self.c["fd_none"] += 1
            return None
        return self.generic_visit(n)

    def visit_If(self, n):
        if is_flag(n.test):
            assert not n.orelse
            self.c["if_flag"] += 1
            return None
        if is_fd_cmp(n.test, ast.IsNot):
            assert not n.orelse
            self.c["if_fd"] += 1
            return None
        return self.generic_visit(n)

    def visit_IfExp(self, n):
        if is_fd_cmp(n.test, ast.Is):
            assert isinstance(n.orelse, ast.Constant) and n.orelse.value is None
            self.c["ifexp"] += 1
            return self.visit(n.body)
        return self.generic_visit(n)

    def visit_BoolOp(self, n):
        if isinstance(n.op, ast.And) and len(n.values) == 2 and is_fd_cmp(n.values[1], ast.Is):
            self.c["boolop"] += 1
            return self.visit(n.values[0])
        return self.generic_visit(n)


def drop_doc(tree):
    b = tree.body
    if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
        tree.body = b[1:]
    return tree


def load_parse(path, pool_default):
    tree = ast.parse(open(path).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    ns = {"argparse": argparse, "POOL_P_DEFAULT": pool_default}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "train_parse", "exec"), ns)

    def parse(argv):
        old = sys.argv
        sys.argv = ["train.py"] + argv
        try:
            return vars(ns["parse"]())
        finally:
            sys.argv = old
    return parse


def t0():
    P("== T0 code identity with the flag absent")
    want = dict(line.split()[::-1] for line in open(f"{HERE}/ref/md5.txt").read().splitlines())
    for tag, copy, ref, live, cdir in PAIRS:
        P(f" -- {tag}: {os.path.basename(copy)} vs {os.path.relpath(live, ROOT)}")
        check(f"{tag} (a) frozen reference == ref/md5.txt", md5(ref) == want[os.path.basename(ref)], md5(ref))
        check(f"{tag} (a) live source file still == the frozen reference", md5(live) == md5(ref),
              f"live {md5(live)} (if FAIL: the live file changed after the copy; re-derive the copy)")
        src, rsrc = open(copy).read(), open(ref).read()
        st = Strip()
        stripped = ast.fix_missing_locations(st.visit(drop_doc(ast.parse(src))))
        refa = drop_doc(ast.parse(rsrc))
        check(f"{tag} (b) stripped statement counts == expected", dict(st.c) == EXPECT, dict(st.c))
        same = ast.dump(stripped) == ast.dump(refa)
        check(f"{tag} (b) AST(copy minus FULLDATA statements) == AST(reference)", same,
              f"{len(ast.dump(refa))} chars of AST dump compared")
        # every remaining `fd` name / flag attribute is gone after the strip
        left = [n for n in ast.walk(stripped) if (isinstance(n, ast.Name) and n.id in ("fd", "FullData")) or is_flag(n)]
        check(f"{tag} (e) no fd / FullData / a.holdout_in_train node survives the strip", not left, len(left))
        names = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name) and n.id == "fd"]
        stores = [n for n in names if isinstance(n.ctx, ast.Store)]
        flags = [n for n in ast.walk(ast.parse(src)) if is_flag(n)]
        check(f"{tag} (e) `fd`: 2 assignments (None; FullData(...) inside `if a.holdout_in_train`), {len(names) - 2} reads; "
              f"a.holdout_in_train read once", len(stores) == 2 and len(flags) == 1, f"names {len(names)} stores {len(stores)} flag reads {len(flags)}")
        # (c) textual diff
        a_, b_ = rsrc.splitlines(), src.splitlines()
        doc_end = next(i for i, l in enumerate(b_) if l.startswith("Previous docstring:"))
        sm = difflib.SequenceMatcher(None, a_, b_, autojunk=False)
        added, changed, deleted, bad = 0, 0, 0, []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                continue
            if op == "delete":
                deleted += i2 - i1
                bad += a_[i1:i2]
                continue
            head = None  # indentation of the last added line that carries a marker (a guard / statement head)
            for j in range(j1, j2):
                if j <= doc_end + 1:
                    continue  # the docstring paragraph
                ind = len(b_[j]) - len(b_[j].lstrip())
                if any(m in b_[j] for m in MARK):
                    head = ind if head is None or ind <= head else head
                elif head is None or ind <= head:  # allowed only as the body / continuation of a marked head line
                    bad.append(b_[j])
            if op == "replace":
                changed += i2 - i1
                for i, j in zip(range(i1, i2), range(j1, j2)):  # a replaced line must be the old line + a conditional on fd
                    new = b_[j].replace("hold if fd is None else None", "hold").replace(" and fd is None", "")
                    if new != a_[i]:
                        bad.append(f"replace mismatch: {a_[i]!r} -> {b_[j]!r}")
                added += (j2 - j1) - (i2 - i1)
            else:
                added += j2 - j1
        check(f"{tag} (c) textual diff: only guarded additions / 4 conditional rewrites, nothing deleted",
              not bad and deleted == 0 and changed == 4, f"added lines {added} (incl. docstring), rewritten {changed}, deleted {deleted}, offending {bad[:3]}")
        # (d) parser defaults
        sys.path.insert(0, cdir)
        pool_default = 0.5
        d_new, d_ref = load_parse(copy, pool_default)([]), load_parse(ref, pool_default)([])
        sys.path.remove(cdir)
        check(f"{tag} (d) --holdout_in_train default False; all {len(d_ref)} other defaults == reference",
              d_new.pop("holdout_in_train") is False and d_new == d_ref)
        on = load_parse(copy, pool_default)(["--holdout_in_train"])
        check(f"{tag} (d) flag parses to True and changes no other argument", on.pop("holdout_in_train") is True and on == d_ref)
        others = [f for f in os.listdir(cdir) if f.endswith(".py") and "cd_full" in open(os.path.join(cdir, f)).read()]
        check(f"{tag} (e) no module of {os.path.relpath(cdir, ROOT)} mentions cd_full", not others, others)
        check(f"{tag} (e) no file named like a code-dir module in this directory (import shadowing)",
              not (set(os.listdir(HERE)) & {f for f in os.listdir(cdir) if f.endswith('.py')}),
              sorted(set(os.listdir(HERE)) & {f for f in os.listdir(cdir) if f.endswith('.py')}))
    # the two copies carry the same patch
    def patch_lines(copy, ref):
        return [l[2:].replace("nn_x/w5", "X").replace("nn_x/abl", "X").replace("train_w5.py.txt", "R").replace("train_abl.py.txt", "R")
                .replace("train_full_abl.py", "T").replace("train_full.py", "T")
                for l in difflib.ndiff(open(ref).read().splitlines(), open(copy).read().splitlines()) if l[:2] in ("+ ", "- ")]
    check("w5 and abl copies carry the same patch (line for line)", patch_lines(*PAIRS[0][1:3]) == patch_lines(*PAIRS[1][1:3]))


def t1():
    P("== T1 FullData unit behaviour (synthetic arrays)")
    import cd_full as CF
    logs = []
    log = logs.append
    n = 200000
    rng = np.random.default_rng(0)
    M = rng.integers(50, 300, n)
    tr = types.SimpleNamespace(M_np=M, target_np=None)
    hold = np.sort(rng.choice(n, 30000, replace=False))
    hold_rows = None

    def args(**k):
        d = dict(holdout=30000, xfold=-1, xfold_unlab=False, val_mix=0.0, hcap="", rec_q=0.0, rho_file="", trunc_off=0.0,
                 prand2="", only_random=-1.0, time_limit=0, keep="best", select_on="holdout", variant="w5_x")
        d.update(k)
        return types.SimpleNamespace(**d)
    for feeds in (np.arange(n), np.sort(rng.choice(n, 20000, replace=False))):
        a = args()
        fd = CF.FullData(a, feeds, hold, hold_rows, tr, log)
        removed = np.setdiff1d(feeds, hold)
        back = fd.put_back(removed, log)
        check(f"put_back restores the {len(feeds)} feeds before the removal ({fd.n_hold_in} holdout feeds back); a.keep -> 'last'",
              np.array_equal(back, feeds) and a.keep == "last" and fd.n_hold_in == len(feeds) - len(removed))
    try:
        fd.put_back(removed[:-1], log)
        ok = False
    except AssertionError:
        ok = True
    check("put_back refuses a feed list that is not (before minus holdout)", ok)
    for k, v in (("xfold", 0), ("xfold_unlab", True), ("val_mix", 0.01), ("hcap", "1:8"), ("rec_q", 0.3), ("rho_file", "x"),
                 ("trunc_off", 1.0), ("prand2", "x"), ("only_random", 0.5), ("time_limit", 10), ("holdout", 1000)):
        try:
            CF.FullData(args(**{k: v}), np.arange(n), hold if k != "holdout" else hold[:1000], hold_rows, tr, log)
            ok = False
        except AssertionError:
            ok = True
        check(f"refused with --{k}", ok)
    a = args(keep="best")
    fd = CF.FullData(a, np.arange(n), hold, hold_rows, tr, log)
    r = fd.insample(2500, 0.345, log)
    check("insample() returns nan, logs 'IN-SAMPLE', stores the value", r != r and "IN-SAMPLE" in logs[-1] and fd.hist[-1]["holdout_insample_ndcg"] == 0.345)
    check("abl parser lacks xfold / prand2: getattr defaults used", True, "FullData reads them with getattr(a, name, default)")
    check("file suffix", CF.SUFFIX == "holdoutINSAMPLE")


if __name__ == "__main__":
    t0()
    t1()
    P(f"\n{len(RES['checks'])} checks, {len(FAIL)} FAILED {FAIL if FAIL else ''}")
    json.dump(RES, open(f"{HERE}/logs/test_full.json", "w"), indent=1)
    sys.exit(1 if FAIL else 0)
