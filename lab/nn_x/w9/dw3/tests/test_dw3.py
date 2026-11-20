"""Tests of the nn_x/w9/dw3 WAVE-9 arm DW3 (random-order-domain-weighted plateau; cd_arms9.py; train.py --prand_norm /
--prand_until_cool, both OFF by default). Written 2026-11-13 ~13:45 UTC.

  cpu part:  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w9/dw3/tests/test_dw3.py cpu
  gpu part:  cd <repo> && CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=12 .venv/bin/python nn_x/w9/dw3/tests/test_dw3.py gpu
             (ONE GPU of the DW3 grant 6 / 7; 6 sequential e2e subprocesses of 39 steps each)

CPU part
  C0 code. (a) module copies: model / data / cd_arms* / finetune / holdout_eval / stale_pred / blend / tests/e2e_w5 / mini_split
     byte-identical to nn_x/w5 (md5s recorded at copy time in tests/w5_source_md5.txt, re-checked against the live nn_x/w5 files);
     common.py == nn_x/w5/common.py up to the output-root constant (AST equal after the path substitution); lane.sh differs from
     nn_x/w5/lane.sh only in the documented lines (root, allowed GPUs, dw3 arm case, RUNNING file, w9_ stage-A names, smoke rename).
     (b) AST proof: train.py minus the DW3 statements (tests/strip_dw3.py, exact set asserted) and minus the variant-prefix assert
     == the frozen nn_x/w5/train.py (tests/ref_w5_train.py.txt; md5 == the live nn_x/w5/train.py) minus its assert, docstrings
     stripped; the new assert accepts exactly the prefixes w5_ / w9_. (c) parse(): both flags OFF by default; the lane DRY commands:
     A:base / E:base have the DW3 guard False and prand_base -1; A:dw3 / E:dw3 == base + exactly '--prand_base 0.3 --prand_norm
     --prand_until_cool'; E:base == nn_x/w5 lane.sh E:base token for token; A:base == the w5 A:base except the w9_ variant name.
     (d) call sites: setup_dw3 once behind its guard, dw3.weight once behind `if dw3 is not None` placed right after the unchanged
     `fw = ...` statement and before `if a.loss == "softmax"`; no other module imports cd_arms9.
  C1 statement identity (MiniSplit, CPU): the exact `fw = ...` + `if dw3 is not None:` statements of train.py exec'd on real mini
     train batches: flags off == the frozen w5 statement bitwise (prand_base -1 -> None; prand_base 0.3 -> the same tensor);
     flags on: step < t_cool -> (0.3 + p) / mean (rtol 1e-6), step >= t_cool -> None, pool batches -> None.
  C2 realised dose (FULL training-feed set, CPU; train.py's holdout construction replicated exactly): 3,121,208 training feeds,
     134,118 pool feeds; (0.3 + p) / mean has mean 1 to 1e-9 over the training feeds; the realised mean per-feed weight over the
     training batches of the first epoch (SplitData.batches with the run's rng seed 23) == 1.0000 (every feed once) and over the
     first 1,000 batches within 1.00 +- 0.02; per-batch mean distribution by Mpad reported; DW3.weight on a pool batch / any
     cooldown step -> None. Writes the dose numbers to logs/test_dw3_dose.json.
  C3 loss scale (information): listwise_loss(norm=feed) with the normalised weights vs None on a mini batch (ratio ~1) and vs the
     unnormalised 0.3 + p (ratio ~0.35 = why --prand_norm exists).
GPU part (tests/e2e_dw3.py; 39 steps each)
  G1 cooldown-only (--init v17_pre2s_s101@77500, cooldown_frac 1.0, cooldown_mix 0.2, evals at 20 / 39, as nn_x/w5 test_w5 G2):
     frozen w5 train.py vs new flags OFF: identical batch sequence, train-mode logits, loss-input logits, logged lines, bit-identical
     checkpoint and val preds, w None on every call. New flags ON (t_cool = 0): bit-identical to flags OFF (the cooldown is the base's
     by construction), w None on every call, the DW3 setup lines logged.
  G2 from scratch (A_BASE recipe at --epochs 0.004 = 39 steps, t_cool 31, cooldown_mix 0.5, warmup 10, seed 23): frozen w5 vs new
     flags OFF identical as in G1 (plateau + cooldown + pool batches). Flags ON vs OFF: identical batch sequence; on every plateau
     step w == (0.3 + p[fidx]) * scale (rtol 1e-6, never None), on every cooldown step (regular and pool) w None; step-0 logits
     identical (same init, same batch), later logits and the checkpoint differ (the mechanism is live); the feed-weighted mean of
     the plateau weights reported.
  Test variants w5_t9e2e_*; all their outputs are deleted afterwards.
Writes nn_x/w9/dw3/logs/test_dw3_<part>.log (+ .json).
"""
import ast
import collections
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import time

PART = sys.argv[1] if len(sys.argv) > 1 else "cpu"
assert PART in ("cpu", "gpu"), "usage: test_dw3.py cpu|gpu"
_CVD = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
if PART == "cpu":
    assert _CVD == "", "cpu part: run with CUDA_VISIBLE_DEVICES=''"
else:
    assert _CVD in ("6", "7"), "gpu part: CUDA_VISIBLE_DEVICES=<6 or 7, the DW3 grant>"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
NT = min(12, int(os.environ.get("OMP_NUM_THREADS", "12")))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(NT)
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)  # nn_x/w9/dw3
ROOT = "<repo>"
W5 = f"{ROOT}/nn_x/w5"
PY = f"{ROOT}/.venv/bin/python"
SCR = f"{D}/logs/test_scratch"
os.makedirs(SCR, exist_ok=True)
sys.path.insert(0, HERE)
from strip_dw3 import EXPECT, strip_dw3  # noqa: E402

LOG = open(f"{D}/logs/test_dw3_{PART}.log", "w")
RES, FAILS, N = {}, [], [0]


def P(s):
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def check(name, ok, info=""):
    N[0] += 1
    ok = bool(ok)
    P(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" -- {info}" if info else ""))
    if not ok:
        FAILS.append(name)
    return ok


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def strip_doc(tree):
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr) \
                and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:] or [ast.Pass()]
    return tree


class DropVariantAssert(ast.NodeTransformer):
    def __init__(self):
        self.dropped = []

    def visit_Assert(self, n):
        if "variant.startswith" in ast.unparse(n.test):
            self.dropped.append(ast.unparse(n))
            return None
        return n


def lane_dry(script, gpu, jobs, env=None):
    r = subprocess.run(["bash", script, gpu, jobs], capture_output=True, text=True, env={**os.environ, "DRY": "1", **(env or {})})
    assert r.returncode == 0, r.stderr
    return [ln.split(": ", 1)[1] for ln in r.stdout.strip().splitlines() if ln.startswith("GPU ")]


def parse_cmd(cmd):
    """train.py's own parser on a DRY command (argv after 'train.py')."""
    sys.path.insert(0, D)
    import importlib
    tr = importlib.import_module("train_parse_stub")
    return tr.parse_args(cmd.split("train.py ", 1)[1].split())


def _make_parse_stub():
    """A stub module holding train.py's parse() only (no torch / common import; CPU-safe): the parser lines are copied by AST."""
    tree = ast.parse(open(f"{D}/train.py").read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    cd = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("XT_PRESETS", "XT_REL", "XT_RELONLY") for t in n.targets)]
    body = "import argparse\n" + "\n".join(ast.unparse(n) for n in cd) + "\n" + ast.unparse(fn).replace("return p.parse_args()", "return p") \
        + "\n\ndef parse_args(argv):\n    return parse().parse_args(argv)\n"
    body = body.replace("POOL_P_DEFAULT", "0.5")
    open(f"{D}/train_parse_stub.py", "w").write(body)


# ------------------------------------------------------------------------------------------------ CPU part
def c0_code():
    P("\n== C0 code")
    rec = {}
    for ln in open(f"{HERE}/w5_source_md5.txt"):
        h, f = ln.split()
        rec[f] = h
    same = ["model.py", "data.py", "cd_arms.py", "cd_arms4.py", "cd_arms5.py", "cd_arms6.py", "cd_arms7.py", "cd_arms8.py",
            "finetune.py", "holdout_eval.py", "stale_pred.py", "blend.py", "tests/e2e_w5.py", "tests/mini_split.py"]
    bad = [f for f in same if not (md5(f"{D}/{f}") == rec[f] == md5(f"{W5}/{f}"))]
    check("(a) byte-identical module copies (md5 == recorded == live nn_x/w5)", not bad, f"{len(same)} files" + (f"; differ: {bad}" if bad else ""))
    check("(a) frozen reference tests/ref_w5_train.py.txt == live nn_x/w5/train.py (md5)", md5(f"{HERE}/ref_w5_train.py.txt") == rec["train.py"] == md5(f"{W5}/train.py"), rec["train.py"])
    c_new = open(f"{D}/common.py").read().replace('ROOT / "nn_x" / "w9" / "dw3"', 'ROOT / "nn_x" / "w5"')
    check("(a) common.py == nn_x/w5/common.py up to the output root (AST, docstrings/comments ignored)",
          ast.dump(strip_doc(ast.parse(c_new))) == ast.dump(strip_doc(ast.parse(open(f"{W5}/common.py").read()))))
    a_ = [ln.replace("<repo>/nn_x/w5", "@W") for ln in open(f"{W5}/lane.sh").read().splitlines()]
    b_ = [ln.replace("<repo>/nn_x/w9/dw3", "@W") for ln in open(f"{D}/lane.sh").read().splitlines()]
    diff = [ln for ln in difflib.unified_diff(a_, b_, lineterm="", n=0) if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
    exp = {"-ALLOWED=${W5_GPUS:-\"0 1 2 4\"}", "+ALLOWED=${W5_GPUS:-\"6 7\"}", "-TL=@W/trunc_lifts_mbucket.json",
           "+TL=<repo>/nn_x/w5/trunc_lifts_mbucket.json", "-RUN=logs/RUNNING_w5.txt"}
    unexpected = [ln for ln in diff if ln not in exp and not (ln.startswith("+#") or "A:dw3|E:dw3" in ln or "RUN=RUNNING.txt" in ln
                                                                 or "v=w9_${arm}_s${x}" in ln or 'v=w5_${arm}_s${x}' in ln
                                                                 or "w5_smoke_" in ln)]
    check("(a) lane.sh differs from nn_x/w5/lane.sh only in the documented lines", not unexpected and len(diff) >= 8, f"{len(diff)} diff lines" + (f"; unexpected {unexpected}" if unexpected else ""))
    # (b) AST proof
    new_src, ref_src = open(f"{D}/train.py").read(), open(f"{HERE}/ref_w5_train.py.txt").read()
    t_new, removed = strip_dw3(ast.parse(new_src))
    check("(b) strip_dw3 removed exactly the expected DW3 statement set", removed == EXPECT, dict(removed))
    dv_new, dv_ref = DropVariantAssert(), DropVariantAssert()
    t_new, t_ref = strip_doc(dv_new.visit(t_new)), strip_doc(dv_ref.visit(ast.parse(ref_src)))
    check("(b) train.py minus DW3 statements minus the variant assert == frozen nn_x/w5/train.py (AST)", ast.dump(t_new) == ast.dump(t_ref))
    check("(b) exactly one variant assert each; new accepts exactly ('w5_', 'w9_'), ref 'w5_'",
          len(dv_new.dropped) == 1 == len(dv_ref.dropped) and "a.variant.startswith(('w5_', 'w9_'))" in dv_new.dropped[0]
          and "a.variant.startswith('w5_')" in dv_ref.dropped[0], dv_new.dropped[0][:60])
    if ast.dump(t_new) != ast.dump(t_ref):
        for i, (x, y) in enumerate(zip(ast.unparse(t_new).splitlines(), ast.unparse(t_ref).splitlines())):
            if x != y:
                P(f"     first AST difference at unparsed line {i}: NEW {x[:100]!r} | REF {y[:100]!r}")
                break
    # (c) parser defaults + lane DRY commands
    _make_parse_stub()
    a_def = parse_cmd("train.py --variant w9_x")
    check("(c) parse(): --prand_norm / --prand_until_cool OFF by default, prand_base -1",
          a_def.prand_norm is False and a_def.prand_until_cool is False and a_def.prand_base == -1.0)
    lane, lane5 = f"{D}/lane.sh", f"{W5}/lane.sh"
    cmds = dict(zip(["A:base", "A:dw3", "E:base", "E:dw3"], lane_dry(lane, "6", "A:base:23 A:dw3:23 E:base:951 E:dw3:951")))
    ref5 = dict(zip(["A:base", "E:base"], lane_dry(lane5, "0", "A:base:23 E:base:951", env={"W5_GPUS": "0"})))
    ARM = "--prand_base 0.3 --prand_norm --prand_until_cool"
    for st in ("A", "E"):
        ab, ad = parse_cmd(cmds[f"{st}:base"]), parse_cmd(cmds[f"{st}:dw3"])
        check(f"(c) lane {st}:base: DW3 guard False, prand_base -1", not (ab.prand_norm or ab.prand_until_cool) and ab.prand_base < 0)
        check(f"(c) lane {st}:dw3: guard True, prand_base 0.3, both flags on", ad.prand_norm and ad.prand_until_cool and ad.prand_base == 0.3)
        base_tok = [t.replace("_base_s", "_dw3_s") if t.startswith(("w9_base_s", "w5_e10_base_s")) else t for t in cmds[f"{st}:base"].split()]
        check(f"(c) lane {st}:dw3 == {st}:base (variant renamed) + exactly '{ARM}'", cmds[f"{st}:dw3"].split() == base_tok + ARM.split())
    check("(c) lane E:base == nn_x/w5 lane.sh E:base token for token", cmds["E:base"].split() == ref5["E:base"].split())
    check("(c) lane A:base == nn_x/w5 lane.sh A:base except the variant name w9_base_s23 vs w5_base_s23",
          cmds["A:base"].split() == [t.replace("w5_base_s23", "w9_base_s23") for t in ref5["A:base"].split()] and "--variant w9_base_s23" in cmds["A:base"])
    e_args = vars(parse_cmd(cmds["E:dw3"]))
    e_base = json.load(open(f"{W5}/logs/w5_e10_base_s951.json"))["args"]
    dif = {k: (e_args.get(k), e_base.get(k)) for k in e_base if e_args.get(k) != e_base.get(k)}
    later = {k: e_args[k] for k in set(e_args) - set(e_base)}  # flags added to nn_x/w5 after the base ran (all at their OFF defaults)
    check("(c) E:dw3 args vs the live w5_e10_base_s951 stored args: differ only in variant and prand_base; later-added w5 flags at their defaults",
          set(dif) == {"variant", "prand_base"} and later == dict(hzcd=False, val_mix=0.0, val_fold=-1, rdrop_cd=0.0, sam_cd=0.0, trunc_lifts="",
                                                                   prand_norm=True, prand_until_cool=True), f"{dif} | later-added {later}")
    # (d) call sites
    tree = ast.parse(new_src)
    calls = collections.Counter()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            s = ast.unparse(n.func)
            if s in ("setup_dw3", "dw3.weight"):
                calls[s] += 1
    main = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    loop = [n for n in ast.walk(main) if isinstance(n, ast.For) and ast.unparse(n.target) == "(step, fb)" or (isinstance(n, ast.For) and ast.unparse(n.target) == "step, fb")][0]
    body = loop.body
    i_fw = [i for i, s in enumerate(body) if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "fw"][0]
    fw_stmt = ast.unparse(body[i_fw])
    ref_fw = [ast.unparse(s) for s in ast.walk(ast.parse(ref_src)) if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "fw"]
    nxt, nxt2 = body[i_fw + 1], body[i_fw + 2]
    check("(d) setup_dw3 called once, dw3.weight called once", calls["setup_dw3"] == 1 and calls["dw3.weight"] == 1, dict(calls))
    check("(d) the `fw = ...` statement is the reference's, `if dw3 is not None:` follows it, then `if a.loss == 'softmax':`",
          fw_stmt in ref_fw and isinstance(nxt, ast.If) and ast.unparse(nxt.test) == "dw3 is not None" and len(nxt.body) == 1
          and ast.unparse(nxt.body[0]) == "fw = dw3.weight(fw, step, is_pool)" and isinstance(nxt2, ast.If) and ast.unparse(nxt2.test) == "a.loss == 'softmax'")
    others = [f for f in os.listdir(D) if f.endswith(".py") and f not in ("train.py", "cd_arms9.py", "train_parse_stub.py") and "cd_arms9" in open(f"{D}/{f}").read()]
    check("(d) no other module imports cd_arms9", not others, str(others))
    RES["c0"] = dict(removed=dict(removed), lane_diff=diff)


def _fw_statements():
    """The exact `fw = ...` assignment and the `if dw3 is not None:` block of train.py's training loop (source strings)."""
    tree = ast.parse(open(f"{D}/train.py").read())
    main = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    loop = [n for n in ast.walk(main) if isinstance(n, ast.For) and "step" in ast.unparse(n.target)][0]
    i = [k for k, s in enumerate(loop.body) if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "fw"][0]
    ref = ast.parse(open(f"{HERE}/ref_w5_train.py.txt").read())
    rfw = [s for s in ast.walk(ref) if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "fw"][0]
    return ast.unparse(loop.body[i]), ast.unparse(loop.body[i + 1]), ast.unparse(rfw)


class _A:
    def __init__(self, **k):
        self.__dict__.update(dict(prand_base=-1.0, prand_norm=False, prand_until_cool=False, loss="softmax"))
        self.__dict__.update(k)


def c1_statements():
    P("\n== C1 statement identity on MiniSplit train batches (CPU)")
    sys.path.insert(0, D)
    from mini_split import MiniSplit, pick_train_feeds
    from cd_arms9 import setup_dw3
    feeds = pick_train_feeds(n_each=40, seed=5)
    ms = MiniSplit("train", feeds, xt_names=None)
    rng = np.random.default_rng(0)
    bl = ms.batches(np.arange(ms.n), 32768, 512, shuffle=True, rng=rng)
    bs = [ms.gather(fb, with_pos=True) for fb in bl]
    s_fw, s_if, s_ref = _fw_statements()
    P(f"   statements: {s_fw!r}\n               {s_if.splitlines()[0]!r} ...\n   reference:  {s_ref!r}")
    prand = ms.prand
    logs = []
    exec_new = lambda ns: (exec(s_fw, ns), exec(s_if, ns), ns["fw"])[2]  # noqa: E731
    exec_ref = lambda ns: (exec(s_ref, ns), ns["fw"])[1]  # noqa: E731
    # flags OFF, prand_base -1 and 0.3: new == ref bitwise
    ok_off = True
    for pb in (-1.0, 0.3):
        a = _A(prand_base=pb)
        for st, b in enumerate(bs):
            n1 = dict(prand=prand, a=a, b=b, dw3=None, step=st, is_pool=False)
            n2 = dict(prand=prand, a=a, b=b)
            f1, f2 = exec_new(n1), exec_ref(n2)
            ok_off &= (f1 is None and f2 is None) if pb < 0 else (f1 is not None and torch.equal(f1, f2) and f1.shape == (len(b["fidx"]),))
    check("flags OFF (dw3 None): new statements == frozen w5 statement bitwise for prand_base -1 (None) and 0.3 (tensor)", ok_off, f"{len(bs)} batches")
    # flags ON
    t_cool, total = 5, 8
    a = _A(prand_base=0.3, prand_norm=True, prand_until_cool=True)
    dw3 = setup_dw3(a, ms, prand, np.arange(ms.n), t_cool, total, logs.append)
    p_np = prand.numpy().astype(np.float64)
    m = (0.3 + p_np).mean()
    ok_on, n_pl, n_cd = True, 0, 0
    for st in range(total):
        for b in bs:
            for is_pool in (False, True):
                ns = dict(prand=prand, a=a, b=b, dw3=dw3, step=st, is_pool=is_pool)
                f = exec_new(ns)
                if st < t_cool and not is_pool:
                    exp = torch.from_numpy(((0.3 + p_np[b["fidx"].numpy()]) / m).astype(np.float32))
                    ok_on &= f is not None and torch.allclose(f, exp, rtol=1e-6, atol=0) and abs(float(f.mean()) - float(exp.mean())) < 1e-6
                    n_pl += 1
                elif st >= t_cool:
                    ok_on &= f is None
                    n_cd += 1
    check("flags ON: step < t_cool -> fw == (0.3 + p) / mean(0.3 + p) (rtol 1e-6); step >= t_cool (regular and pool) -> None", ok_on, f"{n_pl} plateau / {n_cd} cooldown evaluations")
    check("flags ON: setup log reports the scale and the until_cool range", len(logs) == 2 and "scale" in logs[0] and "--prand_until_cool" in logs[1], logs[0][:120])
    # norm only / until_cool only
    d_n = setup_dw3(_A(prand_base=0.3, prand_norm=True), ms, prand, np.arange(ms.n), t_cool, total, lambda s: None)
    d_u = setup_dw3(_A(prand_base=0.3, prand_until_cool=True), ms, prand, np.arange(ms.n), t_cool, total, lambda s: None)
    b = bs[0]
    f0 = 0.3 + prand[b["fidx"]]
    check("--prand_norm alone: weights scaled on every step (also >= t_cool); --prand_until_cool alone: scale 1 (same tensor), None from t_cool",
          d_n.weight(f0, t_cool + 1, False) is not None and torch.allclose(d_n.weight(f0, 0, False), f0 / m, rtol=1e-6)
          and d_u.weight(f0, 0, False) is f0 and d_u.weight(f0, t_cool, False) is None and d_u.weight(None, 0, False) is None)
    RES["c1"] = dict(n_batches=len(bs), mini_mean=float(m))


def c2_dose():
    P("\n== C2 realised dose on the full training-feed set (CPU; train.py's holdout construction replicated)")
    sys.path.insert(0, D)
    import data as Da
    from cd_arms9 import DW3
    pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    M = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["M"])["M"].to_numpy().astype(np.int64)
    assert (pr["fidx"] == np.arange(len(p))).all()
    p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
    feeds = np.setdiff1d(np.arange(len(p)), hold)
    pool = feeds[p[feeds] > 0.5]
    check("training feeds 3,121,208 and pool 134,118 (== the w5_e10_base log)", len(feeds) == 3121208 and len(pool) == 134118, f"{len(feeds)} / {len(pool)}")
    pt = p[feeds].astype(np.float64)
    w = (0.3 + pt) / (0.3 + pt).mean()
    pm = pt > 0.5
    short = M[feeds] < 50
    dose = dict(prand_base=0.3, n_train=int(len(feeds)), n_pool=int(len(pool)), mean_p=float(pt.mean()), mean_w=float(w.mean()),
                pool_share_weight=float(w[pm].sum() / w.sum()), pool_share_count=float(pm.mean()), pool_mean_w=float(w[pm].mean()),
                prod_mean_w=float(w[~pm].mean()), short_mean_w=float(w[short].mean()), max_w=float(w.max()), sd_w=float(w.std()))
    P(f"   dose: {json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in dose.items()})}")
    check("mean per-feed weight over the training feeds == 1 (1e-9)", abs(w.mean() - 1) < 1e-9)
    check("pool (p > 0.5) weight share 10-20% (expected 14.0%), pool mean weight 3.26x, production 0.90x, M<50 0.86x",
          0.10 < dose["pool_share_weight"] < 0.20 and 3.0 < dose["pool_mean_w"] < 3.5 and 0.85 < dose["prod_mean_w"] < 0.95)
    # the run's first-epoch batches: rng seed 23 exactly as train.py (no --train_feeds: the first rng draw is the batch shuffle)
    class Stub:
        M_np = M
    rng = np.random.default_rng(23)
    bl = Da.SplitData.batches(Stub(), feeds, 32768, 512, shuffle=True, rng=rng)
    wt = np.zeros(len(p))
    wt[feeds] = w
    bm = np.array([wt[fb].mean() for fb in bl])
    bn = np.array([len(fb) for fb in bl])
    mp = np.array([int(-(-M[fb].max() // 16) * 16) for fb in bl])
    epoch_mean = float((bm * bn).sum() / bn.sum())
    first = slice(0, 1000)
    m1000 = float((bm[first] * bn[first]).sum() / bn[first].sum())
    P(f"   epoch 1: {len(bl)} batches (== 9,835 steps/epoch: {len(bl) == 9835}); realised mean per-feed weight over the epoch {epoch_mean:.5f}; "
      f"over the first 1,000 batches {m1000:.4f}; mean of per-batch means {bm.mean():.4f}; per-batch mean min {bm.min():.3f} median {np.median(bm):.3f} max {bm.max():.3f}")
    for lo, hi in ((16, 48), (48, 64), (64, 128), (128, 256), (256, 100000)):
        s = (mp >= lo) & (mp < hi)
        if s.any():
            P(f"     Mpad [{lo},{hi}): {s.sum()} batches, feeds/batch {bn[s].mean():.0f}, per-batch mean weight {bm[s].mean():.3f} (min {bm[s].min():.3f} max {bm[s].max():.3f})")
    check("realised mean per-feed weight on the training batches of epoch 1 == 1.0000 (every training feed once)", abs(epoch_mean - 1) < 1e-6 and len(bl) == 9835, f"{epoch_mean:.6f}")
    check("realised mean per-feed weight over the first 1,000 training batches within 1.00 +- 0.02", abs(m1000 - 1) <= 0.02, f"{m1000:.4f}")
    d = DW3(1 / (0.3 + pt).mean(), True, 15736)
    f0 = torch.from_numpy((0.3 + p[bl[0]]).astype(np.float32))
    check("pool batch (is_pool, step >= t_cool) -> weight None; any cooldown step -> None; plateau step -> scaled tensor",
          d.weight(f0, 15736, True) is None and d.weight(f0, 19669, False) is None and d.weight(f0, 15735, False) is not None)
    dose.update(epoch_mean=epoch_mean, first1000_mean=m1000, batch_mean_of_means=float(bm.mean()), n_batches=len(bl), t_cool_2ep=15736, t_cool_10ep=78680)
    json.dump(dose, open(f"{D}/logs/test_dw3_dose.json", "w"), indent=1)
    RES["c2"] = dose


def c3_loss_scale():
    P("\n== C3 loss scale on a mini batch (information)")
    sys.path.insert(0, D)
    from mini_split import MiniSplit, pick_train_feeds
    from model import listwise_loss
    ms = MiniSplit("train", pick_train_feeds(n_each=30, seed=7), xt_names=None)
    rng = np.random.default_rng(1)
    bl = ms.batches(np.arange(ms.n), 32768, 512, shuffle=True, rng=rng)
    torch.manual_seed(0)
    p_all = ms.prand.numpy().astype(np.float64)
    m = (0.3 + p_all).mean()
    r0 = r1 = 0.0
    n = 0
    for fb in bl:
        b = ms.gather(fb, with_pos=True)
        lg = torch.randn(b["cmask"].shape) * 0.5
        y, mk = b["y"], b["cmask"]
        fw_raw = 0.3 + ms.prand[b["fidx"]]
        l_none = listwise_loss(lg, y, mk, None, norm="feed", denom=512).item()
        l_norm = listwise_loss(lg, y, mk, fw_raw / m, norm="feed", denom=512).item()
        l_raw = listwise_loss(lg, y, mk, fw_raw, norm="feed", denom=512).item()
        r0 += l_norm / l_none
        r1 += l_raw / l_none
        n += 1
    P(f"   {n} mini batches: loss(normalised fw) / loss(None) mean {r0 / n:.3f}; loss(raw 0.3 + p) / loss(None) mean {r1 / n:.3f} (mini mean(0.3+p) {m:.3f})")
    check("loss scale: raw weights shrink the main loss by ~mean(0.3+p); normalised weights keep it near the base's scale", r1 / n < 0.8 * r0 / n)
    RES["c3"] = dict(ratio_norm=r0 / n, ratio_raw=r1 / n)


# ------------------------------------------------------------------------------------------------ GPU part
INIT = f"{ROOT}/nn/ckpt/v17_pre2s_s101@77500.pt"
E2E_CD = ("--init {init} --epochs 0.004 --lr 2e-3 --warmup 0 --sched wsd --cooldown_frac 1.0 --muon --aux_pos 0.5 "
          "--aux_mode listwise --aux_tau 10 --aux_anneal 0.1 --cooldown_mix 0.2 --loss_norm feed --holdout 30000 "
          "--select_on holdout --eval_every 20 --seed 811 --no_test")
E2E_SC = ("--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout "
          "--loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --lr 1e-3 --sched wsd --cooldown_frac 0.2 --cooldown_mix 0.5 "
          "--aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 0.004 --eval_every 20 --no_test --seed 23 --warmup 10")
ARM = ["--prand_base", "0.3", "--prand_norm", "--prand_until_cool"]


def outputs(v):
    fs = [f"{D}/logs/{v}.log", f"{D}/logs/{v}.json", f"{D}/logs/{v}_cliplog.npz"]
    for x in (v, f"{v}@last"):
        fs += [f"{D}/ckpt/{x}.pt", f"{D}/preds/nn_{x}_trainonly_val.npy", f"{D}/preds/nn_{x}_trainonly_test.npy", f"{D}/preds/holdout/nn_{x}_holdout.npy"]
    return fs


def clean(v):
    for f in outputs(v):
        if os.path.exists(f):
            os.remove(f)


def e2e_run(train_file, v, base_args, extra):
    rec = f"{SCR}/rec_{v}.npz"
    cmd = [PY, f"{HERE}/e2e_dw3.py", train_file, rec, "--variant", v] + base_args.format(init=INIT).split() + extra
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=D, env={**os.environ, "OMP_NUM_THREADS": str(NT), "MKL_NUM_THREADS": str(NT),
                                                                          "OPENBLAS_NUM_THREADS": str(NT), "POLARS_MAX_THREADS": str(NT)})
    open(f"{SCR}/{v}.stdout", "w").write(r.stdout + "\n" + r.stderr)
    P(f"  ran {v} ({time.time() - t0:.0f}s, rc {r.returncode})")
    assert r.returncode == 0, r.stderr[-3000:]
    return dict(np.load(rec, allow_pickle=True)), open(f"{D}/logs/{v}.log").read()


def metric_lines(log):
    return [re.sub(r" [\d.]+ min", "", ln) for ln in log.splitlines() if ln.startswith("step ") or ln.startswith("BEST")]


def ck_equal(a, b):
    ca = torch.load(a, map_location="cpu", weights_only=False)["state_dict"]
    cb = torch.load(b, map_location="cpu", weights_only=False)["state_dict"]
    return ca.keys() == cb.keys() and all(torch.equal(ca[k], cb[k]) for k in ca)


def identical(ra, rb, la, lb, va, vb, tag):
    nb = len(ra["off"]) - 1
    check(f"{tag}: identical training-batch sequence", np.array_equal(ra["fidx"], rb["fidx"]) and np.array_equal(ra["off"], rb["off"]) and nb == 39, f"{nb} batches")
    check(f"{tag}: identical train-mode logits and loss-input logits", np.array_equal(ra["lg_model"], rb["lg_model"]) and np.array_equal(ra["lg_loss"], rb["lg_loss"]))
    ma, mb = metric_lines(la), metric_lines(lb)
    check(f"{tag}: identical logged loss / val / holdout / BEST lines", ma == mb and len(ma) >= 3, mb[-1] if mb else "")
    check(f"{tag}: bit-identical checkpoint and val preds", ck_equal(f"{D}/ckpt/{va}.pt", f"{D}/ckpt/{vb}.pt")
          and np.array_equal(np.load(f"{D}/preds/nn_{va}_trainonly_val.npy"), np.load(f"{D}/preds/nn_{vb}_trainonly_val.npy")))


def g_e2e():
    P("\n== G1 cooldown-only e2e (frozen w5 train.py vs new; 39 CD steps from v17_pre2s_s101@77500, cooldown_mix 0.2)")
    ref_copy = f"{SCR}/train_w5_frozen.py"
    open(ref_copy, "w").write(open(f"{HERE}/ref_w5_train.py.txt").read())
    names = ["w5_t9e2e_ref_cd", "w5_t9e2e_new_cd", "w5_t9e2e_on_cd", "w5_t9e2e_ref_sc", "w5_t9e2e_new_sc", "w5_t9e2e_on_sc"]
    for v in names:
        clean(v)
    try:
        ra, la = e2e_run(ref_copy, names[0], E2E_CD, [])
        rb, lb = e2e_run(f"{D}/train.py", names[1], E2E_CD, [])
        identical(ra, rb, la, lb, names[0], names[1], "flags OFF vs frozen w5 (cooldown-only)")
        check("flags OFF: w None on every listwise_loss call (both runs)", int(ra["has_w"].sum()) == 0 == int(rb["has_w"].sum()) and len(rb["has_w"]) == 39)
        rc, lc = e2e_run(f"{D}/train.py", names[2], E2E_CD, ARM)
        identical(rc, rb, lc, lb, names[2], names[1], "flags ON cooldown-only (t_cool 0) vs flags OFF")
        check("flags ON cooldown-only: w None on every call; DW3 setup lines logged", int(rc["has_w"].sum()) == 0 and "wave-9 DW3: prand_base 0.3" in lc and "--prand_until_cool" in lc)
        P("\n== G2 from-scratch e2e (A_BASE recipe, 39 steps: t_cool 31, cooldown_mix 0.5, evals at 20 / 39)")
        rd, ld = e2e_run(ref_copy, names[3], E2E_SC, [])
        re_, le = e2e_run(f"{D}/train.py", names[4], E2E_SC, [])
        identical(rd, re_, ld, le, names[3], names[4], "flags OFF vs frozen w5 (from scratch)")
        check("from scratch flags OFF: w None on every call; 39 loss calls", int(re_["has_w"].sum()) == 0 and len(re_["has_w"]) == 39)
        rf, lf = e2e_run(f"{D}/train.py", names[5], E2E_SC, ARM)
        check("flags ON from scratch: identical batch sequence to flags OFF", np.array_equal(rf["fidx"], re_["fidx"]) and np.array_equal(rf["off"], re_["off"]))
        t_cool = int(re.search(r"cooldown mix: p=0.5 from step (\d+)", lf).group(1))
        pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
        p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
        M = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["M"])["M"].to_numpy()
        p[M < 50] = 0.0
        scale = float(re.search(r"scale ([\d.]+)", lf).group(1))
        m_exact = 1 / (0.3 + p[np.setdiff1d(np.arange(len(p)), np.sort(np.random.default_rng(12345).choice(np.where((p > 0.9) & (M >= 50))[0], 30000, replace=False)))].astype(np.float64)).mean()
        ok_w, wsum, wn, n_pl, n_cd = True, 0.0, 0, 0, 0
        for k in range(len(rf["has_w"])):
            fb = rf["fidx"][rf["off"][k]:rf["off"][k + 1]]
            wk = rf["w"][rf["w_off"][k]:rf["w_off"][k + 1]]
            if k < t_cool:
                exp = (0.3 + p[fb]) * np.float32(m_exact)
                ok_w &= bool(rf["has_w"][k] == 1 and len(wk) == len(fb) and np.allclose(wk, exp, rtol=1e-5, atol=1e-6))
                wsum += wk.sum()
                wn += len(wk)
                n_pl += 1
            else:
                ok_w &= bool(rf["has_w"][k] == 0)
                n_cd += 1
        check(f"flags ON: every plateau step (< {t_cool}) w == (0.3 + p[fidx]) * scale (rtol 1e-5); every cooldown step (regular and pool) w None",
              ok_w and n_pl == t_cool and n_cd == 39 - t_cool, f"{n_pl} plateau / {n_cd} cooldown calls; logged scale {scale:.4f} vs exact {m_exact:.4f}; "
              f"feed-weighted mean of the plateau weights {wsum / max(wn, 1):.4f} over {wn} feeds")
        check("flags ON: logged scale == 1 / mean(0.3 + p) over the training feeds (4 decimals)", abs(scale - m_exact) < 6e-5)
        s0 = rf["shapes"][0].prod()
        check("flags ON: step-0 logits identical to flags OFF (same init / batch), later logits and the checkpoint differ (mechanism live)",
              np.array_equal(rf["lg_model"][:s0], re_["lg_model"][:s0]) and not np.array_equal(rf["lg_model"], re_["lg_model"])
              and not ck_equal(f"{D}/ckpt/{names[5]}.pt", f"{D}/ckpt/{names[4]}.pt"))
        check("flags ON: n2 loss calls == n model forwards == 39; norm 'feed' on every call", int(rf["n_loss"]) == int(rf["n_model"]) == 39 and all(x == "feed" for x in rf["norm"]))
        RES["g"] = dict(t_cool=t_cool, scale=scale, plateau_mean_w=wsum / max(wn, 1), n_pl=n_pl, n_cd=n_cd)
    finally:
        for v in names:
            clean(v)
        for v in names:
            for f in (f"{D}/logs/{v}_holdout.stdout",):
                if os.path.exists(f):
                    os.remove(f)


if __name__ == "__main__":
    t0 = time.time()
    P(f"test_dw3 {PART} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} in {D}")
    if PART == "cpu":
        c0_code()
        c1_statements()
        c2_dose()
        c3_loss_scale()
    else:
        g_e2e()
    P(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed in {time.time() - t0:.0f}s" + (f"; FAILED: {FAILS}" if FAILS else ""))
    json.dump(dict(part=PART, passed=N[0] - len(FAILS), total=N[0], fails=FAILS, res=RES), open(f"{D}/logs/test_dw3_{PART}.json", "w"), indent=1, default=str)
    sys.exit(1 if FAILS else 0)
