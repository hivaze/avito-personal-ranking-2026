"""Tests of the nn_x/w10/r2 WAVE-10 arm R2 XOFF (examination offset on exploration training feeds; cd_arms10.py; train.py --xoff,
OFF by default). Written 2026-11-13 ~17:00 UTC (harness = nn_x/w9/dw3/tests/test_dw3.py adapted).

  cpu part:  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w10/r2/tests/test_r2.py cpu
  gpu part:  cd <repo> && CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=12 .venv/bin/python nn_x/w10/r2/tests/test_r2.py gpu
             (ONE GPU of the R2 grant 4 / 5; 6 sequential e2e subprocesses of 39 steps each)

CPU part
  C0 code. (a) module copies byte-identical to nn_x/w5 (md5 == recorded == live); common.py == nn_x/w5/common.py up to the output
     root; lane.sh differs only in the documented lines. (b) AST proof: train.py minus the XOFF statements (tests/strip_r2.py, exact
     set asserted) minus the variant-prefix assert == the frozen nn_x/w5/train.py (tests/ref_w5_train.py.txt; md5 == live). (c) parse():
     --xoff 0 by default, --xoff_table = the premise table; lane DRY: A:base / E:base xoff 0; A:r2 / E:r2 == base + exactly '--xoff 0.5';
     E:base == nn_x/w5 E:base token for token; E:r2 args vs the stored w5_e10_base_s951 args differ only in variant / xoff (+ later w5
     flags at their defaults). (d) call sites: setup_xoff once behind `if a.xoff > 0`, xoff.offset once behind `if xoff is not None`
     placed right after the TRUNC `if` and before `y, m = ...`; no other module imports cd_arms10.
  C1 offset on MiniSplit train batches (CPU): offset(b) == a * g[clamp(pos, 0, 299)] * cmask on exploration feeds (NN p > 0.5), exactly 0 on
     production (p <= 0.5), M<50 and holdout feeds and on padding; shape / dtype; the table's g[0] > 0 > g[299]; setup_xoff refusals
     (--trunc_off, --val_mix, rdrop/sam, aug); the log line carries the md5.
  C2 full training-feed set (CPU; train.py's holdout construction replicated): coef > 0 on exactly 134,118 training feeds (4.30%), 0 on
     the 30K holdout; realised dose over the epoch-1 batches of seed 23 (share of batches with >= 1 exploration feed (0.68: M-grouped batches), mean |offset| per
     exploration row at a = 0.5, mean offset over exploration positives / negatives). Writes logs/test_r2_dose.json.
GPU part (tests/e2e_r2.py; 39 steps each)
  G1 cooldown-only (--init v17_pre2s_s101@77500, cooldown_frac 1.0, cooldown_mix 0.2): frozen w5 train.py vs flag OFF: identical batch
     sequence, train-mode logits, loss-input logits, logged lines, bit-identical checkpoint and val preds. Flag ON: identical batch
     sequence; on every step lg_loss - lg_model == 0.5 * g[pos] on the exploration feeds' candidates (bf16 tolerance) and exactly 0 on
     every other candidate; pool batches (all exploration feeds) offset everywhere; the XOFF setup line logged.
  G2 from scratch (A_BASE recipe at --epochs 0.004 = 39 steps, t_cool 31, cooldown_mix 0.5, seed 23): frozen vs flag OFF identical as in
     G1; flag ON vs OFF: identical batch sequence, step-0 model logits identical, loss-input logits differ exactly by the offset on
     every step, later logits and the checkpoint differ (the mechanism is live).
  Test variants w5_t10e2e_*; all their outputs are deleted afterwards.
Writes nn_x/w10/r2/logs/test_r2_<part>.log (+ .json).
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
assert PART in ("cpu", "gpu"), "usage: test_r2.py cpu|gpu"
_CVD = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
if PART == "cpu":
    assert _CVD == "", "cpu part: run with CUDA_VISIBLE_DEVICES=''"
else:
    assert _CVD in ("4", "5"), "gpu part: CUDA_VISIBLE_DEVICES=<4 or 5, the R2 grant>"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
NT = min(12, int(os.environ.get("OMP_NUM_THREADS", "12")))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(NT)
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)  # nn_x/w10/r2
ROOT = "<repo>"
W5 = f"{ROOT}/nn_x/w5"
PY = f"{ROOT}/.venv/bin/python"
SCR = f"{D}/logs/test_scratch"
os.makedirs(SCR, exist_ok=True)
sys.path.insert(0, HERE)
sys.path.insert(0, D)
from strip_r2 import EXPECT, strip_r2  # noqa: E402
from cd_arms10 import XOFF_TABLE, MAX_POS, XOff, load_table, setup_xoff, xoff_coef  # noqa: E402

LOG = open(f"{D}/logs/test_r2_{PART}.log", "w")
RES, FAILS, N = {}, [], [0]
ARM = ["--xoff", "0.5"]
A_OFF = 0.5


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
    import importlib
    tr = importlib.import_module("train_parse_stub")
    return tr.parse_args(cmd.split("train.py ", 1)[1].split())


def _make_parse_stub():
    """A stub module holding train.py's parse() only (no torch / common import; CPU-safe): the parser lines are copied by AST."""
    tree = ast.parse(open(f"{D}/train.py").read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    cd = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("XT_PRESETS", "XT_REL", "XT_RELONLY") for t in n.targets)]
    body = "import argparse\n" + f"XOFF_TABLE = {XOFF_TABLE!r}\n" + "\n".join(ast.unparse(n) for n in cd) + "\n" \
        + ast.unparse(fn).replace("return p.parse_args()", "return p") + "\n\ndef parse_args(argv):\n    return parse().parse_args(argv)\n"
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
    c_new = open(f"{D}/common.py").read().replace('ROOT / "nn_x" / "w10" / "r2"', 'ROOT / "nn_x" / "w5"')
    check("(a) common.py == nn_x/w5/common.py up to the output root (AST, docstrings/comments ignored)",
          ast.dump(strip_doc(ast.parse(c_new))) == ast.dump(strip_doc(ast.parse(open(f"{W5}/common.py").read()))))
    a_ = [ln.replace("<repo>/nn_x/w5", "@W") for ln in open(f"{W5}/lane.sh").read().splitlines()]
    b_ = [ln.replace("<repo>/nn_x/w10/r2", "@W") for ln in open(f"{D}/lane.sh").read().splitlines()]
    diff = [ln for ln in difflib.unified_diff(a_, b_, lineterm="", n=0) if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
    exp = {"-ALLOWED=${W5_GPUS:-\"0 1 2 4\"}", "+ALLOWED=${W5_GPUS:-\"4 5\"}", "-TL=@W/trunc_lifts_mbucket.json",
           "+TL=<repo>/nn_x/w5/trunc_lifts_mbucket.json", "-RUN=logs/RUNNING_w5.txt"}
    unexpected = [ln for ln in diff if ln not in exp and not (ln.startswith("+#") or "A:r2|E:r2" in ln or "RUN=RUNNING.txt" in ln
                                                                 or "v=w10_${arm}_s${x}" in ln or 'v=w5_${arm}_s${x}' in ln
                                                                 or "w5_smoke_" in ln)]
    check("(a) lane.sh differs from nn_x/w5/lane.sh only in the documented lines", not unexpected and len(diff) >= 8, f"{len(diff)} diff lines" + (f"; unexpected {unexpected}" if unexpected else ""))
    # (b) AST proof
    new_src, ref_src = open(f"{D}/train.py").read(), open(f"{HERE}/ref_w5_train.py.txt").read()
    t_new, removed = strip_r2(ast.parse(new_src))
    check("(b) strip_r2 removed exactly the expected XOFF statement set", removed == EXPECT, dict(removed))
    dv_new, dv_ref = DropVariantAssert(), DropVariantAssert()
    t_new, t_ref = strip_doc(dv_new.visit(t_new)), strip_doc(dv_ref.visit(ast.parse(ref_src)))
    check("(b) train.py minus XOFF statements minus the variant assert == frozen nn_x/w5/train.py (AST)", ast.dump(t_new) == ast.dump(t_ref))
    check("(b) exactly one variant assert each; new accepts exactly ('w5_', 'w10_'), ref 'w5_'",
          len(dv_new.dropped) == 1 == len(dv_ref.dropped) and "a.variant.startswith(('w5_', 'w10_'))" in dv_new.dropped[0]
          and "a.variant.startswith('w5_')" in dv_ref.dropped[0], dv_new.dropped[0][:60])
    if ast.dump(t_new) != ast.dump(t_ref):
        for i, (x, y) in enumerate(zip(ast.unparse(t_new).splitlines(), ast.unparse(t_ref).splitlines())):
            if x != y:
                P(f"     first AST difference at unparsed line {i}: NEW {x[:100]!r} | REF {y[:100]!r}")
                break
    # (c) parser defaults + lane DRY commands
    _make_parse_stub()
    a_def = parse_cmd("train.py --variant w10_x")
    check("(c) parse(): --xoff 0.0 by default, --xoff_table = the premise table", a_def.xoff == 0.0 and a_def.xoff_table == XOFF_TABLE and os.path.exists(XOFF_TABLE))
    lane, lane5 = f"{D}/lane.sh", f"{W5}/lane.sh"
    cmds = dict(zip(["A:base", "A:r2", "E:base", "E:r2"], lane_dry(lane, "4", "A:base:23 A:r2:23 E:base:951 E:r2:951")))
    ref5 = dict(zip(["A:base", "E:base"], lane_dry(lane5, "0", "A:base:23 E:base:951", env={"W5_GPUS": "0"})))
    for st in ("A", "E"):
        ab, ad = parse_cmd(cmds[f"{st}:base"]), parse_cmd(cmds[f"{st}:r2"])
        check(f"(c) lane {st}:base: xoff 0 (guard False)", ab.xoff == 0.0)
        check(f"(c) lane {st}:r2: xoff 0.5 (guard True), default table", ad.xoff == 0.5 and ad.xoff_table == XOFF_TABLE)
        base_tok = [t.replace("_base_s", "_r2_s") if t.startswith(("w10_base_s", "w5_e10_base_s")) else t for t in cmds[f"{st}:base"].split()]
        check(f"(c) lane {st}:r2 == {st}:base (variant renamed) + exactly '--xoff 0.5'", cmds[f"{st}:r2"].split() == base_tok + ARM)
    check("(c) lane E:base == nn_x/w5 lane.sh E:base token for token", cmds["E:base"].split() == ref5["E:base"].split())
    check("(c) lane A:base == nn_x/w5 lane.sh A:base except the variant name w10_base_s23 vs w5_base_s23",
          cmds["A:base"].split() == [t.replace("w5_base_s23", "w10_base_s23") for t in ref5["A:base"].split()] and "--variant w10_base_s23" in cmds["A:base"])
    e_args = vars(parse_cmd(cmds["E:r2"]))
    e_base = json.load(open(f"{W5}/logs/w5_e10_base_s951.json"))["args"]
    dif = {k: (e_args.get(k), e_base.get(k)) for k in e_base if e_args.get(k) != e_base.get(k)}
    later = {k: e_args[k] for k in set(e_args) - set(e_base)}  # flags added to nn_x/w5 after the base ran (all at their OFF defaults) + the arm
    check("(c) E:r2 args vs the live w5_e10_base_s951 stored args: differ only in variant; later-added w5 flags at their defaults; xoff 0.5",
          set(dif) == {"variant"} and later == dict(hzcd=False, val_mix=0.0, val_fold=-1, rdrop_cd=0.0, sam_cd=0.0, trunc_lifts="", xoff=0.5, xoff_table=XOFF_TABLE),
          f"{dif} | later-added {later}")
    # the abl / w9 stage-S references: stored args == the A:r2 command minus the arm (effective flags)
    a_cmd = vars(parse_cmd(cmds["A:base"]))
    for nm, path in (("abl_base_s23", f"{ROOT}/nn_x/abl/logs/abl_base_s23.json"), ("abl_base_s24", f"{ROOT}/nn_x/abl/logs/abl_base_s24.json"),
                     ("w9_base_s23", f"{ROOT}/nn_x/w9/dw3/logs/w9_base_s23.json"), ("w9_base_s24", f"{ROOT}/nn_x/w9/dw3/logs/w9_base_s24.json")):
        st_args = json.load(open(path))["args"]
        OFF = (False, None, 0, 0.0, -1, -1.0, "")
        d2 = {k: (a_cmd.get(k), st_args.get(k)) for k in set(a_cmd) | set(st_args) if k not in ("variant", "seed", "xoff_table") and a_cmd.get(k) != st_args.get(k)
              and not (a_cmd.get(k) in OFF and st_args.get(k) in OFF)}  # xoff_table: a later-added flag (its default is inert at xoff 0)
        check(f"(c) reference {nm}: stored args == lane A:base (every non-default flag equal; seed {st_args.get('seed')})", not d2 and st_args.get("seed") == int(nm[-2:]) and st_args.get("xoff", 0) == 0, str(d2))
    # (d) call sites
    tree = ast.parse(new_src)
    calls = collections.Counter()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            s = ast.unparse(n.func)
            if s in ("setup_xoff", "xoff.offset"):
                calls[s] += 1
    main = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    loop = [n for n in ast.walk(main) if isinstance(n, ast.For) and "step" in ast.unparse(n.target)][0]
    body = loop.body
    i_tr = [i for i, s in enumerate(body) if isinstance(s, ast.If) and ast.unparse(s.test) == "trunc is not None"][0]
    nxt, nxt2 = body[i_tr + 1], body[i_tr + 2]
    check("(d) setup_xoff called once, xoff.offset called once", calls["setup_xoff"] == 1 and calls["xoff.offset"] == 1, dict(calls))
    check("(d) `if xoff is not None:` follows the TRUNC `if` and precedes `y, m = b['y'], b['cmask']`",
          isinstance(nxt, ast.If) and ast.unparse(nxt.test) == "xoff is not None" and len(nxt.body) == 1
          and ast.unparse(nxt.body[0]) == "lg = lg + xoff.offset(b).to(lg.dtype)" and isinstance(nxt2, ast.Assign) and "b['y']" in ast.unparse(nxt2))
    setup_ifs = [n for n in ast.walk(main) if isinstance(n, ast.If) and ast.unparse(n.test) == "a.xoff > 0"]
    check("(d) setup guard `if a.xoff > 0:` once, body = `xoff = setup_xoff(a, tr, prand, hold, log)`",
          len(setup_ifs) == 1 and len(setup_ifs[0].body) == 1 and ast.unparse(setup_ifs[0].body[0]) == "xoff = setup_xoff(a, tr, prand, hold, log)")
    others = [f for f in os.listdir(D) if f.endswith(".py") and f not in ("train.py", "cd_arms10.py", "train_parse_stub.py") and "cd_arms10" in open(f"{D}/{f}").read()]
    check("(d) no other module imports cd_arms10", not others, str(others))
    RES["c0"] = dict(removed=dict(removed), lane_diff=diff)


class _A:
    def __init__(self, **k):
        self.__dict__.update(dict(xoff=A_OFF, xoff_table=XOFF_TABLE, loss="softmax", aug_cand=0, aug_hist=0, val_mix=0.0, rdrop_cd=0.0, sam_cd=0.0, trunc_off=0.0))
        self.__dict__.update(k)


def c1_offset():
    P("\n== C1 offset on MiniSplit train batches (CPU)")
    from mini_split import MiniSplit, pick_train_feeds
    feeds = pick_train_feeds(n_each=40, seed=5)
    ms = MiniSplit("train", feeds, xt_names=None)
    g, md = load_table(XOFF_TABLE)
    check("table: 300 finite entries, g[0] > 0 is the max, g[299] < 0; g[0] / g[1] / g[299]", g.shape == (300,) and g[0] > 0 and g[-1] < 0, f"{g[0]:.3f} / {g[1]:.3f} / {g[-1]:.3f}; md5 {md}")
    # a fake holdout = 3 of the exploration mini feeds
    p_np = ms.prand.numpy().astype(np.float64)
    expl = np.where(p_np > 0.5)[0]
    hold = np.sort(expl[:3])
    logs = []
    xo = XOff(A_OFF, ms, p_np, hold, logs.append, table_path=XOFF_TABLE)
    check("setup log line carries a, the table md5 and the exploration-feed count", len(logs) == 1 and md in logs[0] and f"a {A_OFF}" in logs[0] and f"{int((p_np > 0.5).sum()) - 3} feeds" in logs[0], logs[0][:160])
    rng = np.random.default_rng(0)
    bl = ms.batches(np.arange(ms.n), 32768, 512, shuffle=True, rng=rng)
    cpos = ms.c_pos.numpy().astype(np.int64)
    ok_shape = ok_val = ok_zero = ok_pad = True
    n_on = n_off = 0
    for fb in bl:
        b = ms.gather(fb, with_pos=True)
        o = xo.offset(b)
        ok_shape &= o.shape == b["cmask"].shape and o.dtype == torch.float32
        exp = np.zeros(o.shape, np.float32)
        for r, f in enumerate(fb):
            m = int(ms.M_np[f])
            if p_np[f] > 0.5 and f not in hold:
                pos = cpos[ms.cand_off_np[f]:ms.cand_off_np[f] + m]
                exp[r, :m] = A_OFF * g[np.clip(pos, 0, MAX_POS - 1)]
                n_on += 1
            else:
                n_off += 1
        ok_val &= np.allclose(o.numpy(), exp, rtol=1e-6, atol=1e-7)
        ok_zero &= bool((o[~b["cmask"]] == 0).all())
        ok_pad &= bool((o.numpy()[exp == 0] == 0).all())
    check("offset(b) == a * g[clamp(pos)] * cmask on exploration feeds (p > 0.5, not holdout); exactly 0 on production / M<50 / holdout feeds and padding",
          ok_shape and ok_val and ok_zero and ok_pad, f"{len(bl)} batches; {n_on} exploration rows on, {n_off} rows off")
    # refusals
    def refused(**k):
        try:
            setup_xoff(_A(**k), ms, ms.prand, hold, lambda s: None)
            return False
        except AssertionError:
            return True
    check("setup_xoff refuses --trunc_off / --val_mix / rdrop / sam / aug / --loss bce and needs a holdout",
          refused(trunc_off=1.0) and refused(val_mix=0.01) and refused(rdrop_cd=1.0) and refused(sam_cd=0.05) and refused(aug_cand=1) and refused(loss="bce")
          and not refused())
    xo2 = setup_xoff(_A(), ms, ms.prand, hold, lambda s: None)
    check("setup_xoff builds the same coef table as XOff (a = 0.5 on p > 0.5 minus holdout)", torch.equal(xo2.coef, xo.coef) and float(xo.coef.max()) == A_OFF and float(xo.coef[hold].abs().sum()) == 0)
    RES["c1"] = dict(n_batches=len(bl), n_on=n_on, n_off=n_off, md5=md)


def c2_dose():
    P("\n== C2 realised dose on the full training-feed set (CPU; train.py's holdout construction replicated)")
    import data as Da
    pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    f = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["M", "cand_off"])
    M, off = f["M"].to_numpy().astype(np.int64), f["cand_off"].to_numpy().astype(np.int64)
    assert (pr["fidx"] == np.arange(len(p))).all()
    p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
    feeds = np.setdiff1d(np.arange(len(p)), hold)
    coef = xoff_coef(A_OFF, p.astype(np.float64), hold)
    on = coef > 0
    check("coef > 0 on exactly 134,118 training feeds (4.30% of 3,121,208), 0 on every holdout feed and every M<50 feed",
          int(on[feeds].sum()) == 134118 and len(feeds) == 3121208 and not on[hold].any() and not on[M < 50].any(), f"{int(on[feeds].sum())} / {len(feeds)}")
    g, md = load_table(XOFF_TABLE)
    cpos = np.load(f"{ROOT}/data/flat/train/cand_pos.npy", mmap_mode="r")
    ctg = np.load(f"{ROOT}/data/flat/train/cand_target.npy", mmap_mode="r")
    pool = feeds[on[feeds]]
    rows = np.concatenate([np.arange(a, a + b) for a, b in zip(off[pool], M[pool])])
    pos = np.clip(np.asarray(cpos[rows]).astype(np.int64), 0, MAX_POS - 1)
    y = np.asarray(ctg[rows]).astype(np.int64)
    o = A_OFF * g[pos]
    dose = dict(a=A_OFF, table_md5=md, n_train=int(len(feeds)), n_pool=int(len(pool)), pool_share=float(on[feeds].mean()), n_pool_rows=int(len(rows)),
                mean_offset_rows=float(o.mean()), mean_abs_offset_rows=float(np.abs(o).mean()), mean_offset_pos=float(o[y == 1].mean()),
                mean_offset_neg=float(o[y == 0].mean()), max_offset=float(o.max()), min_offset=float(o.min()))
    class Stub:
        M_np = M
    rng = np.random.default_rng(23)
    bl = Da.SplitData.batches(Stub(), feeds, 32768, 512, shuffle=True, rng=rng)
    nb_on = np.array([int(on[fb].sum()) for fb in bl])
    dose.update(n_batches=len(bl), share_batches_with_expl=float((nb_on > 0).mean()), mean_expl_feeds_per_batch=float(nb_on.mean()))
    P(f"   dose: {json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in dose.items()})}")
    check("epoch-1 batches (seed 23): 9,835 batches; 50-90% of the batches hold >= 1 exploration feed (batches are M-grouped: M<50 batches hold none), ~13.6 per batch on average",
          len(bl) == 9835 and 0.5 < dose["share_batches_with_expl"] < 0.9 and 5 < dose["mean_expl_feeds_per_batch"] < 30)
    check("offset at a = 0.5: max +0.48 (pos 0), min about -0.09; positives get a higher mean offset than negatives (the curve explains the examination lift)",
          0.4 < dose["max_offset"] < 0.6 and -0.2 < dose["min_offset"] < 0 and dose["mean_offset_pos"] > dose["mean_offset_neg"])
    json.dump(dose, open(f"{D}/logs/test_r2_dose.json", "w"), indent=1)
    RES["c2"] = dose


# ------------------------------------------------------------------------------------------------ GPU part
INIT = f"{ROOT}/nn/ckpt/v17_pre2s_s101@77500.pt"
E2E_CD = ("--init {init} --epochs 0.004 --lr 2e-3 --warmup 0 --sched wsd --cooldown_frac 1.0 --muon --aux_pos 0.5 "
          "--aux_mode listwise --aux_tau 10 --aux_anneal 0.1 --cooldown_mix 0.2 --loss_norm feed --holdout 30000 "
          "--select_on holdout --eval_every 20 --seed 811 --no_test")
E2E_SC = ("--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout "
          "--loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --lr 1e-3 --sched wsd --cooldown_frac 0.2 --cooldown_mix 0.5 "
          "--aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 0.004 --eval_every 20 --no_test --seed 23 --warmup 10")


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
    cmd = [PY, f"{HERE}/e2e_r2.py", train_file, rec, "--variant", v] + base_args.format(init=INIT).split() + extra
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


def offset_check(r, tag, coef, g, cpos):
    """lg_loss - lg_model == coef[fidx] * g[pos] on real candidates (bf16 tolerance), exactly 0 where the coef is 0; per-step counts."""
    ok_on = ok_off = True
    n_on = n_off = n_pool_b = 0
    maxerr = 0.0
    for k in range(len(r["off"]) - 1):
        fb = r["fidx"][r["off"][k]:r["off"][k + 1]]
        B, Mx = r["shapes"][k]
        ci = r["ci"][k * 0:0]  # placeholder (ci stored flat per batch below)
    # ci / logits are stored flat in batch order with shapes
    sidx = np.concatenate([[0], np.cumsum(r["shapes"].prod(1))])
    for k in range(len(r["off"]) - 1):
        fb = r["fidx"][r["off"][k]:r["off"][k + 1]]
        B, Mx = r["shapes"][k]
        ci = r["ci"][sidx[k]:sidx[k + 1]].reshape(B, Mx)
        lm = r["lg_model"][sidx[k]:sidx[k + 1]].reshape(B, Mx)
        ll = r["lg_loss"][sidx[k]:sidx[k + 1]].reshape(B, Mx)
        real = ci >= 0
        pos = np.clip(np.asarray(cpos[np.maximum(ci, 0)]).astype(np.int64), 0, MAX_POS - 1)
        exp = coef[fb][:, None] * g[pos] * real
        d = ll - lm
        on = (exp != 0) & real
        off = (exp == 0) & real
        if on.any():
            err = np.abs(d[on] - exp[on])
            tol = 0.02 + 0.01 * np.abs(ll[on])  # bf16 addition of the offset to bf16 logits
            ok_on &= bool((err <= tol).all())
            maxerr = max(maxerr, float((err / tol).max()))
            n_on += int(on.sum())
        ok_off &= bool((d[off] == 0).all())
        n_off += int(off.sum())
        n_pool_b += int((coef[fb] > 0).all())
    check(f"{tag}: lg_loss - lg_model == 0.5 * g[pos] on exploration-feed candidates (bf16 tol) and exactly 0 elsewhere",
          ok_on and ok_off and n_on > 0, f"{n_on} candidates with an offset, {n_off} without; max err/tol {maxerr:.2f}; {n_pool_b} all-exploration (pool) batches")
    return n_on, n_off, n_pool_b


def g_e2e():
    P("\n== G1 cooldown-only e2e (frozen w5 train.py vs new; 39 CD steps from v17_pre2s_s101@77500, cooldown_mix 0.2)")
    ref_copy = f"{SCR}/train_w5_frozen.py"
    open(ref_copy, "w").write(open(f"{HERE}/ref_w5_train.py.txt").read())
    pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    M = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["M"])["M"].to_numpy()
    p[M < 50] = 0.0
    hold = np.sort(np.random.default_rng(12345).choice(np.where((p > 0.9) & (M >= 50))[0], 30000, replace=False))
    coef = xoff_coef(A_OFF, p.astype(np.float64), hold)
    g, md = load_table(XOFF_TABLE)
    cpos = np.load(f"{ROOT}/data/flat/train/cand_pos.npy", mmap_mode="r")
    names = ["w5_t10e2e_ref_cd", "w5_t10e2e_new_cd", "w5_t10e2e_on_cd", "w5_t10e2e_ref_sc", "w5_t10e2e_new_sc", "w5_t10e2e_on_sc"]
    for v in names:
        clean(v)
    try:
        ra, la = e2e_run(ref_copy, names[0], E2E_CD, [])
        rb, lb = e2e_run(f"{D}/train.py", names[1], E2E_CD, [])
        identical(ra, rb, la, lb, names[0], names[1], "flag OFF vs frozen w5 (cooldown-only)")
        check("flag OFF: loss-input logits == model logits on every step (no offset)", np.array_equal(rb["lg_loss"], rb["lg_model"]) and len(rb["has_w"]) == 39)
        rc, lc = e2e_run(f"{D}/train.py", names[2], E2E_CD, ARM)
        check("flag ON cooldown-only: identical batch sequence to flag OFF", np.array_equal(rc["fidx"], rb["fidx"]) and np.array_equal(rc["off"], rb["off"]))
        n_on, n_off, n_pb = offset_check(rc, "flag ON cooldown-only", coef, g, cpos)
        check("flag ON cooldown-only: XOFF setup line logged with the table md5; step-0 model logits identical to flag OFF (same init / batch)",
              "wave-10 XOFF: a 0.5" in lc and md in lc and np.array_equal(rc["lg_model"][:rc["shapes"][0].prod()], rb["lg_model"][:rb["shapes"][0].prod()]))
        check("flag ON cooldown-only: the m02 pool batches (all exploration feeds) carry the offset on every candidate; later logits / checkpoint differ",
              n_pb >= 3 and not np.array_equal(rc["lg_model"], rb["lg_model"]) and not ck_equal(f"{D}/ckpt/{names[2]}.pt", f"{D}/ckpt/{names[1]}.pt"))
        P("\n== G2 from-scratch e2e (A_BASE recipe, 39 steps: t_cool 31, cooldown_mix 0.5, evals at 20 / 39)")
        rd, ld = e2e_run(ref_copy, names[3], E2E_SC, [])
        re_, le = e2e_run(f"{D}/train.py", names[4], E2E_SC, [])
        identical(rd, re_, ld, le, names[3], names[4], "flag OFF vs frozen w5 (from scratch)")
        check("from scratch flag OFF: loss-input logits == model logits; 39 loss calls", np.array_equal(re_["lg_loss"], re_["lg_model"]) and int(re_["n_loss"]) == 39)
        rf, lf = e2e_run(f"{D}/train.py", names[5], E2E_SC, ARM)
        check("flag ON from scratch: identical batch sequence to flag OFF", np.array_equal(rf["fidx"], re_["fidx"]) and np.array_equal(rf["off"], re_["off"]))
        n_on2, n_off2, n_pb2 = offset_check(rf, "flag ON from scratch", coef, g, cpos)
        s0 = rf["shapes"][0].prod()
        check("flag ON from scratch: step-0 model logits identical to flag OFF, later logits and the checkpoint differ (mechanism live); norm 'feed' on every call",
              np.array_equal(rf["lg_model"][:s0], re_["lg_model"][:s0]) and not np.array_equal(rf["lg_model"], re_["lg_model"])
              and not ck_equal(f"{D}/ckpt/{names[5]}.pt", f"{D}/ckpt/{names[4]}.pt") and all(x == "feed" for x in rf["norm"]))
        check("flag ON: n loss calls == n model forwards == 39; eval / predict paths untouched (same counts as flag OFF)",
              int(rf["n_loss"]) == int(rf["n_model"]) == 39 and int(rf["n_eval"]) == int(re_["n_eval"]) and int(rf["n_predict"]) == int(re_["n_predict"]))
        RES["g"] = dict(cd_on=n_on, cd_off=n_off, cd_pool_batches=n_pb, sc_on=n_on2, sc_off=n_off2, sc_pool_batches=n_pb2)
    finally:
        for v in names:
            clean(v)
            for f in (f"{D}/logs/{v}_holdout.stdout",):
                if os.path.exists(f):
                    os.remove(f)


if __name__ == "__main__":
    t0 = time.time()
    P(f"test_r2 {PART} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} in {D}")
    if PART == "cpu":
        c0_code()
        c1_offset()
        c2_dose()
    else:
        g_e2e()
    P(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed in {time.time() - t0:.0f}s" + (f"; FAILED: {FAILS}" if FAILS else ""))
    json.dump(dict(part=PART, passed=N[0] - len(FAILS), total=N[0], fails=FAILS, res=RES), open(f"{D}/logs/test_r2_{PART}.json", "w"), indent=1, default=str)
    sys.exit(1 if FAILS else 0)
