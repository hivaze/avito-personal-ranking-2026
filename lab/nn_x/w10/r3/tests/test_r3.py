"""Tests of the nn_x/w10/r3 WAVE-10 arm R3 (listwise CE + --bce_w * feed-normalised BCE; cd_arms10.py; OFF by default).
Written 2026-11-13 ~17:10 UTC (pattern: nn_x/w9/dw3/tests/test_dw3.py + nn_x/w9/catgrad/tests/test_w9_catgrad.py).

  cpu part:  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w10/r3/tests/test_r3.py cpu
  gpu part:  cd <repo> && CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=12 .venv/bin/python nn_x/w10/r3/tests/test_r3.py gpu
             (ONE GPU of the R3 grant 0 / 1; one W5_SMOKE=1 lane run with the flag ON, ~3-5 min)

CPU part
  C0 code. (a) module copies: model / data / cd_arms* / finetune / holdout_eval / stale_pred / blend / tests/e2e_w5 / mini_split
     byte-identical to nn_x/w5 (md5s recorded at copy time in tests/w5_source_md5.txt, re-checked against the live nn_x/w5 files);
     common.py == nn_x/w5/common.py up to the output-root constant (AST equal after the path substitution); lane.sh differs from
     nn_x/w5/lane.sh only in the documented lines. (b) AST proof: train.py minus the R3 statements (tests/strip_r3.py, exact set
     asserted) and minus the variant-prefix assert == the frozen nn_x/w5/train.py (tests/ref_w5_train.py.txt; md5 == the live
     nn_x/w5/train.py) minus its assert, docstrings stripped; the new assert accepts exactly the prefixes w5_ / w10_.
     (c) parse(): --bce_w 0.0 by default; lane DRY commands: A:r3 / E:r3 == A:base / E:base + exactly '--bce_w 5', A:r3w20 == A:base +
     '--bce_w 20'; E:base == nn_x/w5 lane.sh E:base token for token; A:base == the w5 A:base except the w10_ variant name; E:r3 args vs
     the stored w5_e10_base_s951 args differ only in variant and bce_w. (d) call sites: setup_bce once behind `if a.bce_w > 0`,
     bce_feed_loss once behind `if bce_w > 0`, placed right after the `if a.loss == 'softmax'` chain and before the cliplog
     `cliplog.main = loss` statement; no other module imports cd_arms10.
  C1 unit (MiniSplit, real train batches, CPU): cd_arms10.bce_feed_loss == a brute-force numpy reference (per feed: mean BCE over the
     valid candidates; weights = (fw or 1) * [ys > 0]; norm feed = sum / denom, batch = weighted mean, idcg = K/IDCG(K) factor)
     for fw None and a random fw, every norm; the exact train.py statements (`fw = ...`, the loss chain, `if bce_w > 0:`) exec'd on
     real batches: bce_w 0 -> loss bitwise == the frozen w5 statements' loss; bce_w 5 -> loss == listwise + 5 * bce_feed_loss (rel 1e-6)
     with BOTH calls receiving the SAME fw object, norm == a.loss_norm and denom == a.max_feeds (prand_base -1 -> fw None; 0.3 -> the
     same tensor); d loss / d logits == d listwise / d logits + 5 * d bce / d logits (rel 1e-5).
  C2 CPU end-to-end (tests/e2e_cpu.py, miniature split, 2 mini epochs with cooldown + pool batches, evals every 10 steps, save_last /
     save_holdout / cliplog): frozen nn_x/w5 code vs the new code flag OFF: identical batch sequence, train-mode logits, loss-input
     logits, logged lines, bit-identical selected + @last checkpoints, val and holdout preds, stored args + {bce_w: 0.0}. Flag ON
     (--bce_w 5): same batch sequence; step-0 logits identical; on EVERY step bce_feed_loss called once with the same w (bitwise) /
     norm / denom as listwise_loss; later logits and the checkpoint differ (mechanism live); the wave-10 log line; a 4th run
     (--bce_w 5 --aux_pos 0): the scalar sent to .backward() == listwise + 5 * bce on every step (rel 1e-6).
  C3 scale (information, logs/test_r3_scale.json): the trained nn_x/w5 checkpoint w5_e10_base_s951.pt (selected) and a fresh init on
     real train batches: mean per-feed listwise CE vs mean per-feed BCE, and the logit-gradient norms of the two terms; the BCE
     gradient share at bce_w 1 / 5 / 20 (the dose argument of the pre-declaration).
GPU part
  G1 plumbing smoke: W5_SMOKE=1 lane.sh <gpu> A:r3:23 -> w10_smoke_r3_s23 (20K train feeds, 1 epoch, evals every 25 steps): the lane
     completes; json / log / selected + @last checkpoints, val + holdout preds written; args bce_w 5; wave-10 log line; val NDCG > 0.30;
     finite preds. Outputs deleted afterwards except logs/smoke_r3.log (a copy of the run log; no number of it enters any rule).
Writes nn_x/w10/r3/logs/test_r3_<part>.log (+ .json).
"""
import ast
import collections
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import types

PART = sys.argv[1] if len(sys.argv) > 1 else "cpu"
assert PART in ("cpu", "gpu"), "usage: test_r3.py cpu|gpu"
_CVD = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
if PART == "cpu":
    assert _CVD == "", "cpu part: run with CUDA_VISIBLE_DEVICES=''"
else:
    assert _CVD in ("0", "1"), "gpu part: CUDA_VISIBLE_DEVICES=<0 or 1, the R3 grant>"
NT = min(12, int(os.environ.get("OMP_NUM_THREADS", "12")))
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(NT)
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)  # nn_x/w10/r3
ROOT = "<repo>"
W5 = f"{ROOT}/nn_x/w5"
PY = f"{ROOT}/.venv/bin/python"
SCR = f"{D}/logs/test_scratch"
os.makedirs(SCR, exist_ok=True)
sys.path.insert(0, HERE)
from strip_r3 import EXPECT, strip_r3  # noqa: E402

LOG = open(f"{D}/logs/test_r3_{PART}.log", "w")
RES, FAILS, N = {}, [], [0]
DOSE = "5"
E2E = ("--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout "
       "--loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --lr 1e-3 --sched wsd --cooldown_frac 0.5 --cooldown_mix 0.3 "
       "--aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 2 --eval_every 10 --no_test --save_last --save_holdout "
       "--select_min_step 10 --cliplog 5 --warmup 2 --cand_budget 4096 --max_feeds 8 --seed 23")


def P(s):
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def check(name, ok, info=""):
    N[0] += 1
    ok = bool(ok)
    if not ok:
        FAILS.append(name)
    P(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({info})" if info != "" else ""))
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


def _make_parse_stub():
    """A stub module holding train.py's parse() only (no torch / common import; CPU-safe): the parser lines are copied by AST."""
    tree = ast.parse(open(f"{D}/train.py").read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    cd = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("XT_PRESETS", "XT_REL", "XT_RELONLY") for t in n.targets)]
    body = "import argparse\n" + "\n".join(ast.unparse(n) for n in cd) + "\n" + ast.unparse(fn).replace("return p.parse_args()", "return p") \
        + "\n\ndef parse_args(argv):\n    return parse().parse_args(argv)\n"
    body = body.replace("POOL_P_DEFAULT", "0.5")
    open(f"{SCR}/train_parse_stub.py", "w").write(body)


def parse_cmd(cmd):
    """train.py's own parser on a DRY command (argv after 'train.py')."""
    if SCR not in sys.path:
        sys.path.insert(0, SCR)
    import importlib
    tr = importlib.import_module("train_parse_stub")
    return tr.parse_args(cmd.split("train.py ", 1)[1].split())


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
    c_new = open(f"{D}/common.py").read().replace('ROOT / "nn_x" / "w10/r3"', 'ROOT / "nn_x" / "w5"')
    check("(a) common.py == nn_x/w5/common.py up to the output root (AST, docstrings/comments ignored)",
          ast.dump(strip_doc(ast.parse(c_new))) == ast.dump(strip_doc(ast.parse(open(f"{W5}/common.py").read()))))
    a_ = [ln.replace("<repo>/nn_x/w5", "@W") for ln in open(f"{W5}/lane.sh").read().splitlines()]
    b_ = [ln.replace("<repo>/nn_x/w10/r3", "@W") for ln in open(f"{D}/lane.sh").read().splitlines()]
    diff = [ln for ln in difflib.unified_diff(a_, b_, lineterm="", n=0) if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
    exp = {"-ALLOWED=${W5_GPUS:-\"0 1 2 4\"}", "+ALLOWED=${W5_GPUS:-\"0 1\"}", "-TL=@W/trunc_lifts_mbucket.json",
           "+TL=<repo>/nn_x/w5/trunc_lifts_mbucket.json", "-RUN=logs/RUNNING_w5.txt"}
    unexpected = [ln for ln in diff if ln not in exp and not (ln.startswith("+#") or "A:r3|E:r3" in ln or "A:r3w20" in ln or "RUN=RUNNING.txt" in ln
                                                                 or "v=w10_${arm}_s${x}" in ln or 'v=w5_${arm}_s${x}' in ln
                                                                 or "w5_smoke_" in ln)]
    check("(a) lane.sh differs from nn_x/w5/lane.sh only in the documented lines", not unexpected and len(diff) >= 8, f"{len(diff)} diff lines" + (f"; unexpected {unexpected}" if unexpected else ""))
    # (b) AST proof
    new_src, ref_src = open(f"{D}/train.py").read(), open(f"{HERE}/ref_w5_train.py.txt").read()
    t_new, removed = strip_r3(ast.parse(new_src))
    check("(b) strip_r3 removed exactly the expected R3 statement set", removed == EXPECT, dict(removed))
    dv_new, dv_ref = DropVariantAssert(), DropVariantAssert()
    t_new, t_ref = strip_doc(dv_new.visit(t_new)), strip_doc(dv_ref.visit(ast.parse(ref_src)))
    check("(b) train.py minus R3 statements minus the variant assert == frozen nn_x/w5/train.py (AST)", ast.dump(t_new) == ast.dump(t_ref))
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
    check("(c) parse(): --bce_w 0.0 by default (loss softmax, loss_norm batch unchanged)", a_def.bce_w == 0.0 and a_def.loss == "softmax" and a_def.loss_norm == "batch")
    lane, lane5 = f"{D}/lane.sh", f"{W5}/lane.sh"
    cmds = dict(zip(["A:base", "A:r3", "A:r3w20", "E:base", "E:r3"], lane_dry(lane, "0", "A:base:23 A:r3:23 A:r3w20:23 E:base:951 E:r3:951")))
    ref5 = dict(zip(["A:base", "E:base"], lane_dry(lane5, "0", "A:base:23 E:base:951", env={"W5_GPUS": "0"})))
    for st, arm, w in (("A", "r3", "5"), ("E", "r3", "5"), ("A", "r3w20", "20")):
        ab, ad = parse_cmd(cmds[f"{st}:base"]), parse_cmd(cmds[f"{st}:{arm}"])
        check(f"(c) lane {st}:base: bce_w 0; {st}:{arm}: bce_w {w}", ab.bce_w == 0.0 and ad.bce_w == float(w))
        base_tok = [t.replace("_base_s", f"_{arm}_s") if t.startswith(("w10_base_s", "w5_e10_base_s")) else t for t in cmds[f"{st}:base"].split()]
        check(f"(c) lane {st}:{arm} == {st}:base (variant renamed) + exactly '--bce_w {w}'", cmds[f"{st}:{arm}"].split() == base_tok + ["--bce_w", w])
    check("(c) lane E:base == nn_x/w5 lane.sh E:base token for token", cmds["E:base"].split() == ref5["E:base"].split())
    check("(c) lane A:base == nn_x/w5 lane.sh A:base except the variant name w10_base_s23 vs w5_base_s23",
          cmds["A:base"].split() == [t.replace("w5_base_s23", "w10_base_s23") for t in ref5["A:base"].split()] and "--variant w10_base_s23" in cmds["A:base"])
    e_args = vars(parse_cmd(cmds["E:r3"]))
    e_base = json.load(open(f"{W5}/logs/w5_e10_base_s951.json"))["args"]
    dif = {k: (e_args.get(k), e_base.get(k)) for k in e_base if e_args.get(k) != e_base.get(k)}
    later = {k: e_args[k] for k in set(e_args) - set(e_base)}  # flags added to nn_x/w5 after the base ran (all at their OFF defaults) + bce_w
    check("(c) E:r3 args vs the live w5_e10_base_s951 stored args: differ only in variant; later-added w5 flags at their defaults; bce_w 5 new",
          set(dif) == {"variant"} and later == dict(hzcd=False, val_mix=0.0, val_fold=-1, rdrop_cd=0.0, sam_cd=0.0, trunc_lifts="", bce_w=5.0), f"{dif} | later-added {later}")
    # (d) call sites
    tree = ast.parse(new_src)
    calls = collections.Counter()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            s = ast.unparse(n.func)
            if s in ("setup_bce", "bce_feed_loss"):
                calls[s] += 1
    main = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    loop = [n for n in ast.walk(main) if isinstance(n, ast.For) and ast.unparse(n.target).strip("()") == "step, fb"][0]
    body = loop.body
    i_fw = [i for i, s in enumerate(body) if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "fw"][0]
    chain, blk, nxt = body[i_fw + 1], body[i_fw + 2], body[i_fw + 3]
    check("(d) setup_bce called once, bce_feed_loss called once", calls["setup_bce"] == 1 and calls["bce_feed_loss"] == 1, dict(calls))
    check("(d) `if bce_w > 0:` follows the `if a.loss == 'softmax'` chain (right after `fw = ...`) and precedes `if cliplog is not None:`; its body is the one loss statement",
          isinstance(chain, ast.If) and ast.unparse(chain.test) == "a.loss == 'softmax'" and isinstance(blk, ast.If) and ast.unparse(blk.test) == "bce_w > 0"
          and len(blk.body) == 1 and ast.unparse(blk.body[0]) == "loss = loss + bce_w * bce_feed_loss(lg, y, m, fw, norm=a.loss_norm, denom=a.max_feeds)"
          and isinstance(nxt, ast.If) and ast.unparse(nxt.test) == "cliplog is not None" and "cliplog.main = loss" in ast.unparse(nxt))
    setup = [n for n in ast.walk(main) if isinstance(n, ast.If) and ast.unparse(n.test) == "a.bce_w > 0"]
    check("(d) the setup block `if a.bce_w > 0:` holds exactly `bce_w = setup_bce(a, log)`", len(setup) == 1 and len(setup[0].body) == 1 and ast.unparse(setup[0].body[0]) == "bce_w = setup_bce(a, log)")
    others = [f for f in os.listdir(D) if f.endswith(".py") and f not in ("train.py", "cd_arms10.py") and "cd_arms10" in open(f"{D}/{f}").read()]
    check("(d) no other module imports cd_arms10", not others, str(others))
    RES["c0"] = dict(removed=dict(removed), lane_diff=diff)


def _loop_statements(src):
    """(fw statement, loss chain, [bce block]) of train.py's training loop, unparsed."""
    tree = ast.parse(src)
    main = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    loop = [n for n in ast.walk(main) if isinstance(n, ast.For) and ast.unparse(n.target).strip("()") == "step, fb"][0]
    body = loop.body
    i_fw = [i for i, s in enumerate(body) if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "fw"][0]
    out = [ast.unparse(body[i_fw]), ast.unparse(body[i_fw + 1])]
    if isinstance(body[i_fw + 2], ast.If) and ast.unparse(body[i_fw + 2].test) == "bce_w > 0":
        out.append(ast.unparse(body[i_fw + 2]))
    return out


def brute_bce(lg, y, m, w, norm, denom):
    """numpy reference of cd_arms10.bce_feed_loss (float64)."""
    lg, y, m = lg.astype(np.float64), y.astype(np.float64), m.astype(np.float64)
    l_c = (np.maximum(lg, 0) - lg * y + np.log1p(np.exp(-np.abs(lg)))) * m
    l = l_c.sum(1) / np.maximum(m.sum(1), 1)
    ys = y.sum(1)
    valid = (ys > 0).astype(np.float64)
    w = (np.ones_like(l) if w is None else w.astype(np.float64)) * valid
    if norm == "batch":
        return (l * w).sum() / max(w.sum(), 1e-6)
    if norm == "idcg":
        idcg = np.concatenate([[1.0], np.cumsum(1.0 / np.log2(np.arange(2, 400)))])
        k = np.clip(ys.astype(np.int64), 0, 398)
        w = w * ys / np.maximum(idcg[k], 1e-6) * (k > 0)
    return (l * w).sum() / denom


def c1_unit():
    P("\n== C1 unit: cd_arms10.bce_feed_loss vs brute force; the exact train.py loop statements on real mini batches")
    sys.path.insert(0, D)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # common.py import-time guard only; nothing touches CUDA here
    from mini_split import MiniSplit, pick_train_feeds
    import model as Mo
    import cd_arms10 as C10
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ms = MiniSplit("train", pick_train_feeds(n_each=24, seed=3), xt_names=None)
    rng = np.random.default_rng(0)
    batches = ms.batches(np.arange(ms.n), 32768, 512, shuffle=True, rng=rng)
    g = torch.Generator().manual_seed(0)
    worst = collections.defaultdict(float)
    n_b = 0
    for fb in batches[:6]:
        b = ms.gather(fb, with_pos=True)
        y, m = b["y"], b["cmask"]
        lg = (torch.randn(y.shape, generator=g) * 2 - 3).to(torch.bfloat16).float()  # bf16-representable logits (like the autocast output)
        fw = (0.3 + ms.prand[b["fidx"]]).float()
        for w_ in (None, fw):
            for norm, denom in (("feed", 512), ("batch", None), ("idcg", 512), ("feed", 8)):
                val = float(C10.bce_feed_loss(lg, y, m, w_, norm=norm, denom=denom))
                ref = brute_bce(lg.numpy(), y.numpy(), m.numpy(), None if w_ is None else w_.numpy(), norm, denom)
                worst[norm] = max(worst[norm], abs(val - ref) / max(abs(ref), 1e-9))
        n_b += 1
    check(f"bce_feed_loss == brute-force reference on {n_b} real batches, fw None / (0.3 + p), norms feed / batch / idcg, denom 512 / 8 (rel <= 1e-5)",
          all(v <= 1e-5 for v in worst.values()), {k: f"{v:.1e}" for k, v in worst.items()})
    # the exact train.py statements
    new_st, ref_st = _loop_statements(open(f"{D}/train.py").read()), _loop_statements(open(f"{HERE}/ref_w5_train.py.txt").read())
    check("train.py loop: `fw = ...` and the loss chain are the frozen w5 statements verbatim; the R3 block is the 3rd statement",
          new_st[:2] == ref_st and len(new_st) == 3 and len(ref_st) == 2, new_st[2].replace("\n", " ")[:90])
    calls = []

    def ll(logit, y, mask, w=None, norm="batch", denom=None):
        calls.append(("listwise", w, norm, denom))
        return Mo.listwise_loss(logit, y, mask, w, norm=norm, denom=denom)

    def bfl(logit, y, mask, w=None, norm="batch", denom=None):
        calls.append(("bce", w, norm, denom))
        return C10.bce_feed_loss(logit, y, mask, w, norm=norm, denom=denom)

    ok_bit, ok_sum, ok_args, ok_grad, n_c = True, True, True, True, 0
    worst_sum, worst_grad = 0.0, 0.0
    for fb in batches[:6]:
        b = ms.gather(fb, with_pos=True)
        y, m = b["y"], b["cmask"]
        for prand_base, prand in ((-1.0, ms.prand), (0.3, ms.prand), (-1.0, None)):
            for norm in ("feed", "batch"):
                a = types.SimpleNamespace(loss="softmax", loss_norm=norm, max_feeds=512, prand_base=prand_base)
                lg0 = (torch.randn(y.shape, generator=g) * 2 - 3).to(torch.bfloat16).float()
                outs = {}
                for bce_w in (0.0, 5.0):
                    lg = lg0.clone().requires_grad_(True)
                    ns = dict(a=a, prand=prand, b=b, lg=lg, y=y, m=m, bce_w=bce_w, listwise_loss=ll, bce_feed_loss=bfl, bce_loss=Mo.bce_loss)
                    calls.clear()
                    exec("\n".join(new_st), ns)
                    loss = ns["loss"]
                    loss.backward()
                    outs[bce_w] = (loss.detach().clone(), lg.grad.clone(), ns["fw"], list(calls))
                # frozen statements
                lg = lg0.clone().requires_grad_(True)
                ns = dict(a=a, prand=prand, b=b, lg=lg, y=y, m=m, listwise_loss=ll, bce_loss=Mo.bce_loss)
                calls.clear()
                exec("\n".join(ref_st), ns)
                ref_loss = ns["loss"]
                ref_loss.backward()
                ref_grad = lg.grad.clone()
                l0, g0, fw0, c0 = outs[0.0]
                l5, g5, fw5, c5 = outs[5.0]
                ok_bit &= bool(torch.equal(l0, ref_loss.detach()) and torch.equal(g0, ref_grad) and len(c0) == 1 and c0[0][0] == "listwise")
                bce = C10.bce_feed_loss(lg0, y, m, fw5, norm=norm, denom=512).detach()
                exp = ref_loss.detach() + 5.0 * bce
                worst_sum = max(worst_sum, float(abs(l5 - exp) / abs(exp)))
                ok_sum &= bool(abs(l5 - exp) / abs(exp) <= 1e-6)
                # args: both calls got the SAME fw object, norm == a.loss_norm, denom == a.max_feeds
                ok_args &= bool(len(c5) == 2 and c5[0][0] == "listwise" and c5[1][0] == "bce" and c5[0][1] is fw5 and c5[1][1] is fw5
                                and c5[0][2] == c5[1][2] == norm and c5[0][3] == c5[1][3] == 512
                                and ((fw5 is None) if (prand_base < 0 or prand is None) else bool(torch.equal(fw5, prand_base + prand[b["fidx"]]))))
                # gradient additivity
                lg = lg0.clone().requires_grad_(True)
                C10.bce_feed_loss(lg, y, m, fw5, norm=norm, denom=512).backward()
                gexp = ref_grad + 5.0 * lg.grad
                worst_grad = max(worst_grad, float((g5 - gexp).abs().max() / gexp.abs().max()))
                ok_grad &= bool((g5 - gexp).abs().max() / gexp.abs().max() <= 1e-5)
                n_c += 1
    check(f"train.py statements, bce_w 0: loss and d loss / d logits bitwise == the frozen w5 statements; only listwise_loss called ({n_c} cases)", ok_bit)
    check("train.py statements, bce_w 5: loss == listwise + 5 * bce_feed_loss (rel <= 1e-6)", ok_sum, f"worst rel {worst_sum:.1e}")
    check("train.py statements, bce_w 5: listwise_loss then bce_feed_loss, BOTH with the same fw object (None at prand_base -1, prand_base + p[fidx] at 0.3), "
          "norm == a.loss_norm, denom == a.max_feeds", ok_args)
    check("train.py statements, bce_w 5: d loss / d logits == d listwise + 5 * d bce (rel <= 1e-5)", ok_grad, f"worst rel {worst_grad:.1e}")
    RES["c1"] = dict(worst_brute={k: float(v) for k, v in worst.items()}, worst_sum=worst_sum, worst_grad=worst_grad, n_cases=n_c)


# ------------------------------------------------------------------------------------------------ C2 CPU e2e
class NormStr(ast.NodeTransformer):
    def visit_Constant(self, n):
        return ast.copy_location(ast.Constant("S"), n) if isinstance(n.value, str) else n


def outputs(base, v):
    fs = [f"{base}/logs/{v}.log", f"{base}/logs/{v}.json", f"{base}/logs/{v}_cliplog.npz", f"{base}/logs/{v}.stdout", f"{base}/logs/{v}_holdout.stdout"]
    for x in (v, f"{v}@last"):
        fs += [f"{base}/ckpt/{x}.pt", f"{base}/preds/nn_{x}_trainonly_val.npy", f"{base}/preds/nn_{x}_trainonly_test.npy", f"{base}/preds/holdout/nn_{x}_holdout.npy"]
    return fs


def clean(base, v):
    for f in outputs(base, v):
        if os.path.exists(f):
            os.remove(f)
    d = f"{base}/logs/queue/claim_{v}"
    if os.path.isdir(d):
        os.rmdir(d)


def e2e_run(code_dir, v, extra):
    rec = f"{SCR}/rec_{v}.npz"
    cmd = [PY, f"{HERE}/e2e_cpu.py", code_dir, rec, "--variant", v] + E2E.split() + extra
    t0 = time.time()
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}  # 1 thread: bitwise-reproducible CPU runs
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=code_dir, env=env)
    open(f"{SCR}/{v}.stdout", "w").write(r.stdout + "\n" + r.stderr)
    P(f"  ran {v} from {code_dir} ({time.time() - t0:.0f}s, rc {r.returncode})")
    assert r.returncode == 0, r.stderr[-3000:]
    return dict(np.load(rec, allow_pickle=True))


def metric_lines(log):
    return [re.sub(r" [\d.]+ min", "", ln) for ln in log.splitlines() if ln.startswith("step ") or ln.startswith("BEST")]


def ck_equal(a, b):
    ca = torch.load(a, map_location="cpu", weights_only=False)["state_dict"]
    cb = torch.load(b, map_location="cpu", weights_only=False)["state_dict"]
    return ca.keys() == cb.keys() and all(torch.equal(ca[k], cb[k]) for k in ca)


def code_copy(src, dst, out, line):
    """Copy the .py modules of `src` to `dst` with common.py's output path redirected to `out` (string change only; same AST
    shape). Returns True iff every other module is byte-identical to `src` and common.py differs only by string constants."""
    shutil.rmtree(dst, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(dst)
    for f in os.listdir(src):
        if f.endswith(".py"):
            shutil.copy(f"{src}/{f}", f"{dst}/{f}")
    c = open(f"{dst}/common.py").read()
    assert c.count(line) == 1, line
    open(f"{dst}/common.py", "w").write(c.replace(line, f'W5 = ROOT / "{out}" / "."'))
    same = all(md5(f"{dst}/{f}") == md5(f"{src}/{f}") for f in os.listdir(dst) if f != "common.py")
    eq = ast.dump(NormStr().visit(ast.parse(open(f"{dst}/common.py").read()))) == ast.dump(NormStr().visit(ast.parse(open(f"{src}/common.py").read())))
    return same and eq


def c2_e2e():
    P("\n== C2 CPU end-to-end (e2e_cpu.py, miniature split): flags OFF vs frozen nn_x/w5 code; --bce_w 5 mechanism")
    ref_dir, ref_out = f"{SCR}/ref_w5_code", f"{SCR}/ref_w5_out"
    new_dir, new_out = f"{SCR}/new_r3_code", f"{SCR}/new_r3_out"
    ok_r = code_copy(W5, ref_dir, ref_out, 'W5 = ROOT / "nn_x" / "w5"')
    ok_n = code_copy(D, new_dir, new_out, 'W5 = ROOT / "nn_x" / "w10/r3"')
    check("code copies for the e2e (outputs redirected to scratch so the real preds/holdout meta is never touched): modules byte-identical to "
          "nn_x/w5 / nn_x/w10/r3, common.py differs only in the output-path string", ok_r and ok_n, (ok_r, ok_n))
    names = ["w5_t10e2e_ref_off", "w5_t10e2e_new_off", "w5_t10e2e_new_on", "w5_t10e2e_new_on_noaux"]
    ra = e2e_run(ref_dir, names[0], [])
    rb = e2e_run(new_dir, names[1], [])
    rc_ = e2e_run(new_dir, names[2], ["--bce_w", DOSE])
    rd = e2e_run(new_dir, names[3], ["--bce_w", DOSE, "--aux_pos", "0"])
    la, lb, lc, ld = (open(f"{o}/logs/{n}.log").read() for o, n in ((ref_out, names[0]), (new_out, names[1]), (new_out, names[2]), (new_out, names[3])))
    nb = len(ra["off"]) - 1
    check("flags OFF vs frozen w5: identical training-batch sequence (incl. cooldown pool draws)",
          np.array_equal(ra["fidx"], rb["fidx"]) and np.array_equal(ra["off"], rb["off"]) and np.array_equal(ra["ci"], rb["ci"]) and nb >= 20,
          f"{nb} training batches, {int(rb['n_eval'])} holdout evals, {int(rb['n_predict'])} predicts")
    check("flags OFF vs frozen w5: identical train-mode model logits and loss-input logits (bitwise); listwise w None, norm feed, denom 8 on every call; no bce call",
          np.array_equal(ra["lg_model"], rb["lg_model"]) and np.array_equal(ra["lg_loss"], rb["lg_loss"]) and int(rb["n_loss"]) == int(rb["n_model"]) == nb
          and int(rb["lw_has_w"].sum()) == 0 and all(x == "feed" for x in rb["lw_norm"]) and (rb["lw_denom"] == 8).all() and int(rb["n_bce"]) == 0
          and np.array_equal(ra["lw_val"], rb["lw_val"]) and np.array_equal(ra["total"], rb["total"]))
    ma, mb = metric_lines(la), metric_lines(lb)
    check("flags OFF vs frozen w5: identical logged loss / val / holdout / BEST lines", ma == mb and len(ma) >= 3, mb[-1] if mb else "")
    same_files = (ck_equal(f"{ref_out}/ckpt/{names[0]}.pt", f"{new_out}/ckpt/{names[1]}.pt") and ck_equal(f"{ref_out}/ckpt/{names[0]}@last.pt", f"{new_out}/ckpt/{names[1]}@last.pt")
                  and all(np.array_equal(np.load(f"{ref_out}/{d_}nn_{names[0]}{s_}"), np.load(f"{new_out}/{d_}nn_{names[1]}{s_}"))
                          for d_, s_ in (("preds/", "_trainonly_val.npy"), ("preds/", "@last_trainonly_val.npy"), ("preds/holdout/", "_holdout.npy"),
                                         ("preds/holdout/", "@last_holdout.npy"))))
    check("flags OFF vs frozen w5: bit-identical selected + @last checkpoints, val preds and holdout preds", same_files)
    ja, jb = json.load(open(f"{ref_out}/logs/{names[0]}.json")), json.load(open(f"{new_out}/logs/{names[1]}.json"))
    d_args = {k: (ja["args"].get(k), jb["args"].get(k)) for k in set(ja["args"]) | set(jb["args"]) if ja["args"].get(k) != jb["args"].get(k)}
    check("flags OFF: stored args == frozen w5 args + {bce_w: 0.0}; same best_step / val_ndcg; no wave-10 log line",
          d_args == {"variant": (names[0], names[1]), "bce_w": (None, 0.0)} and ja["best_step"] == jb["best_step"] and ja["val_ndcg"] == jb["val_ndcg"]
          and "wave-10" not in la and "wave-10" not in lb, d_args)
    # flag ON
    check(f"--bce_w {DOSE}: same training-batch sequence as flags OFF; listwise and bce each called once per training step",
          np.array_equal(rc_["fidx"], rb["fidx"]) and np.array_equal(rc_["ci"], rb["ci"]) and int(rc_["n_loss"]) == int(rc_["n_model"]) == int(rc_["n_bce"]) == nb)
    same_w = all(np.array_equal(rc_["lw_w"][rc_["lw_w_off"][k]:rc_["lw_w_off"][k + 1]], rc_["bce_w"][rc_["bce_w_off"][k]:rc_["bce_w_off"][k + 1]])
                 and rc_["lw_has_w"][k] == rc_["bce_has_w"][k] for k in range(nb))
    check(f"--bce_w {DOSE}: on EVERY step bce_feed_loss received the same w (bitwise; None here), norm 'feed' and denom 8 as listwise_loss",
          same_w and all(x == "feed" for x in rc_["bce_norm"]) and (rc_["bce_denom"] == 8).all() and int(rc_["bce_has_w"].sum()) == 0)
    s0 = int(rc_["shapes"][0].prod())
    check(f"--bce_w {DOSE}: step-0 logits identical to flags OFF (same init / batch); later train-mode logits and the checkpoint differ (mechanism live)",
          np.array_equal(rc_["lg_model"][:s0], rb["lg_model"][:s0]) and not np.array_equal(rc_["lg_model"], rb["lg_model"])
          and not ck_equal(f"{new_out}/ckpt/{names[2]}.pt", f"{new_out}/ckpt/{names[1]}.pt"))
    ex = rc_["lw_val"] + float(DOSE) * rc_["bce_val"]
    aux = rc_["total"] - ex
    check(f"--bce_w {DOSE} (with the K9 aux): total - (listwise + {DOSE} * bce) = the aux term > 0 on every step; step-0 listwise == flags OFF step-0 listwise",
          (aux > 0).all() and len(rc_["total"]) == nb and rc_["lw_val"][0] == rb["lw_val"][0], f"aux range {aux.min():.4f}..{aux.max():.4f}")
    exd = rd["lw_val"] + float(DOSE) * rd["bce_val"]
    rel = np.abs(rd["total"] - exd) / np.abs(exd)
    check(f"--bce_w {DOSE} --aux_pos 0: the scalar sent to .backward() == listwise + {DOSE} * bce on every step (rel <= 1e-6)",
          len(rd["total"]) == nb and rel.max() <= 1e-6, f"worst rel {rel.max():.1e}; step-0 listwise {rd['lw_val'][0]:.4f} bce {rd['bce_val'][0]:.4f}")
    jc = json.load(open(f"{new_out}/logs/{names[2]}.json"))
    check(f"--bce_w {DOSE}: args bce_w {DOSE}, wave-10 log line, @last / holdout files written, checkpoint cfg == flags-off cfg (no model change)",
          jc["args"]["bce_w"] == float(DOSE) and f"wave-10 R3: --bce_w {float(DOSE)}" in lc
          and torch.load(f"{new_out}/ckpt/{names[2]}.pt", map_location="cpu", weights_only=False)["cfg"] == torch.load(f"{new_out}/ckpt/{names[1]}.pt", map_location="cpu", weights_only=False)["cfg"]
          and all(os.path.exists(f"{new_out}/{f}") for f in (f"ckpt/{names[2]}@last.pt", f"preds/nn_{names[2]}@last_trainonly_val.npy",
                                                             f"preds/holdout/nn_{names[2]}_holdout.npy", f"preds/holdout/nn_{names[2]}@last_holdout.npy")),
          f"best_step {jc['best_step']} val {jc['val_ndcg']:.5f}")
    RES["e2e"] = dict(off_lines=mb, on_lines=metric_lines(lc), on_wave10=[ln for ln in lc.splitlines() if "wave-10" in ln],
                      lw0=float(rb["lw_val"][0]), bce0=float(rc_["bce_val"][0]), aux_range=[float(aux.min()), float(aux.max())])
    for d_ in (ref_dir, ref_out, new_dir, new_out):
        shutil.rmtree(d_, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ C3 scale (information)
def c3_scale():
    P("\n== C3 scale (information): per-feed listwise CE vs mean-per-feed BCE and their logit-gradient norms, trained base ckpt and fresh init")
    sys.path.insert(0, D)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from mini_split import MiniSplit, pick_train_feeds
    import model as Mo
    import cd_arms10 as C10
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ck = torch.load(f"{W5}/ckpt/w5_e10_base_s951.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    ms = MiniSplit("train", pick_train_feeds(n_each=40, seed=5), xt_names=cfg["xt_names"], with_domain=True)
    rng = np.random.default_rng(1)
    batches = ms.batches(np.arange(ms.n), 32768, 512, shuffle=True, rng=rng)
    out = {}
    for tag in ("trained", "init"):
        torch.manual_seed(0)
        m = Mo.Ranker(**cfg)
        if tag == "trained":
            m.load_state_dict(ck["state_dict"])
        else:  # fresh init with the trained run's input normalisation
            m.set_norm(ck["state_dict"]["f_mean"], ck["state_dict"]["f_std"])
        m.eval()
        acc = collections.defaultdict(list)
        for fb in batches[:4]:
            b = ms.gather(fb, with_pos=True)
            y, mk = b["y"], b["cmask"]
            with torch.no_grad():
                X = m.dense(b)
                lg = m(b, X, use_pos=True).float()
            lg = lg.detach().requires_grad_(True)
            l_lw = Mo.listwise_loss(lg, y, mk, None, norm="feed", denom=512)
            g_lw = torch.autograd.grad(l_lw, lg)[0]
            l_b = C10.bce_feed_loss(lg, y, mk, None, norm="feed", denom=512)
            g_b = torch.autograd.grad(l_b, lg)[0]
            nf = float((y.sum(1) > 0).sum())
            acc["ce_f"].append(float(l_lw) * 512 / nf)
            acc["bce_f"].append(float(l_b) * 512 / nf)
            acc["g_lw"].append(float(g_lw.norm()))
            acc["g_b"].append(float(g_b.norm()))
            acc["cos"].append(float((g_lw * g_b).sum() / (g_lw.norm() * g_b.norm())))
            acc["lg_mean"].append(float(lg[mk].mean()))
            acc["sig_mean"].append(float(torch.sigmoid(lg[mk]).mean()))
            acc["pos_rate"].append(float(y[mk].mean()))
        o = {k: float(np.mean(v)) for k, v in acc.items()}
        o.update({f"share_w{w}": w * o["g_b"] / o["g_lw"] for w in (1, 5, 20)})
        out[tag] = o
        P(f"  {tag}: per-feed CE {o['ce_f']:.3f} | per-feed mean BCE {o['bce_f']:.4f} | grad norms listwise {o['g_lw']:.4f} bce {o['g_b']:.4f} "
          f"(cos {o['cos']:.3f}) | bce/listwise gradient ratio at w=1 {o['share_w1']:.3f}, w=5 {o['share_w5']:.3f}, w=20 {o['share_w20']:.3f} | "
          f"mean logit {o['lg_mean']:.2f} sigmoid {o['sig_mean']:.4f} vs positive rate {o['pos_rate']:.4f}")
    check("scale numbers finite (information only)", all(np.isfinite(v) for o in out.values() for v in o.values()))
    RES["c3"] = out
    json.dump(out, open(f"{D}/logs/test_r3_scale.json", "w"), indent=1)


# ================================================================================================= GPU part
def g1_smoke():
    gpu = _CVD
    v = "w10_smoke_r3_s23"
    P(f"\n== G1 GPU plumbing smoke: W5_SMOKE=1 lane.sh {gpu} A:r3:23 -> {v} (20K train feeds, 1 epoch, evals every 25 steps)")
    clean(D, v)
    t0 = time.time()
    r = subprocess.run(["bash", f"{D}/lane.sh", gpu, "A:r3:23"], capture_output=True, text=True, cwd=D,
                       env={**os.environ, "W5_SMOKE": "1", "OMP_NUM_THREADS": str(NT), "WAIT_MAX_H": "1"})
    P(f"  lane rc {r.returncode} ({time.time() - t0:.0f}s); tail: " + " | ".join(r.stdout.strip().splitlines()[-3:]))
    ok_files = all(os.path.exists(f"{D}/{f}") for f in (f"logs/{v}.json", f"logs/{v}.log", f"ckpt/{v}.pt", f"ckpt/{v}@last.pt",
                                                       f"preds/nn_{v}_trainonly_val.npy", f"preds/nn_{v}@last_trainonly_val.npy",
                                                       f"preds/holdout/nn_{v}_holdout.npy", f"preds/holdout/nn_{v}@last_holdout.npy"))
    check("smoke run completed through the lane; json / log / selected + @last checkpoints, val and holdout preds written", r.returncode == 0 and ok_files)
    if ok_files:
        j = json.load(open(f"{D}/logs/{v}.json"))
        pv = np.load(f"{D}/preds/nn_{v}_trainonly_val.npy")
        ph = np.load(f"{D}/preds/holdout/nn_{v}_holdout.npy")
        lg = open(f"{D}/logs/{v}.log").read()
        hist = j["history"]
        check("smoke: val preds finite with within-feed variation, holdout preds finite, val NDCG > 0.30 after 1 mini epoch, history has holdout evals",
              np.isfinite(pv).all() and np.isfinite(ph).all() and pv.std() > 0 and j["val_ndcg"] > 0.30 and len(hist) >= 2
              and all(np.isfinite(x["holdout_ndcg"]) for x in hist), f"val {j['val_ndcg']:.5f} best_step {j['best_step']} minutes {j['train_minutes']:.1f} evals {len(hist)}")
        check("smoke: args bce_w 5, wave-10 log line, checkpoint cfg has no new keys (no model change)",
              j["args"]["bce_w"] == 5.0 and "wave-10 R3: --bce_w 5.0" in lg
              and set(torch.load(f"{D}/ckpt/{v}.pt", map_location="cpu", weights_only=False)["cfg"]) == set(torch.load(f"{W5}/ckpt/w5_e10_base_s951.pt", map_location="cpu", weights_only=False)["cfg"]))
        RES["smoke"] = dict(val=j["val_ndcg"], best_step=j["best_step"], minutes=j["train_minutes"], lines=metric_lines(lg)[-3:])
        shutil.copy(f"{D}/logs/{v}.log", f"{D}/logs/smoke_r3.log")
    clean(D, v)
    for f in (f"{D}/preds/holdout/holdout_meta.npy", f"{D}/preds/holdout/holdout_y.npy"):
        pass  # the smoke's holdout meta / y (written by save_holdout) are the real 30K holdout's: kept (== nn_x/w5's, checked by s_readout.py)


if __name__ == "__main__":
    t0 = time.time()
    P(f"test_r3 {PART} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} in {D}")
    if PART == "cpu":
        c0_code()
        c1_unit()
        c2_e2e()
        c3_scale()
    else:
        g1_smoke()
    P(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed in {time.time() - t0:.0f}s" + (f"; FAILED: {FAILS}" if FAILS else ""))
    json.dump(dict(part=PART, passed=N[0] - len(FAILS), total=N[0], fails=FAILS, res=RES), open(f"{D}/logs/test_r3_{PART}.json", "w"), indent=1, default=str)
    sys.exit(1 if FAILS else 0)
