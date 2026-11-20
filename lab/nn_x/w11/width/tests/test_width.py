"""Tests of the nn_x/w11/width WIDTH-GATE code copy (--mup_base per-matrix muP Muon multiplier, cd_arms11.py; OFF by default) and
the lane_wg.sh arms. Written 2026-11-14 ~00:15 UTC (pattern: nn_x/w10/r3/tests/test_r3.py).

  cpu part:  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w11/width/tests/test_width.py cpu
  gpu part:  cd <repo> && CUDA_VISIBLE_DEVICES=<0-3> OMP_NUM_THREADS=12 .venv/bin/python nn_x/w11/width/tests/test_width.py gpu
             (one W5_SMOKE=1 lane_wg.sh run of arm (c) on the d192 root 711: 20K train feeds, 1 epoch, ~3-5 min)

CPU part
  W0 code. (a) module copies model / data / cd_arms* / finetune / holdout_eval / stale_pred / blend byte-identical to nn_x/w5
     (md5 recorded at copy time in tests/w5_source_md5.txt, re-checked against the live files); common.py == nn_x/w5/common.py up
     to the output-root string (AST); lane_wg.sh's B_BASE / HYG strings == nn_x/w5/lane.sh's.  (b) AST proof: train.py minus the
     WIDTH-GATE statements (tests/strip_w11.py, exact set asserted) minus the variant-prefix assert == the frozen nn_x/w5/train.py
     (tests/ref_w5_train.py.txt; md5 == the live nn_x/w5/train.py) minus its assert, docstrings stripped; the new assert accepts
     exactly ('w5_', 'w11_').  (c) parse(): --mup_base 0 by default, every other default == the frozen parser; lane DRY: d192:m02:711
     == nn_x/w5 B:m02ref:101 with init / variant / seed substituted (CDSEED 900 + root); drop02 == m02 + exactly '--drop_cd 0.2';
     mup == m02 + exactly '--mup_base 128'; d128:m02:901 likewise from the v30 snapshot; refusals (GPU N, root 710, width d256, arm foo).
     (d) call sites: setup_mup once behind `if a.mup_base > 0` right after the Muon-optimizer block in main(); mup.apply once behind
     `if mup is not None` right after the per-step lr-reset loop inside the step loop; mup.summary once inside the eval block; no
     other module imports cd_arms11; cd_arms11 imports only math and torch.  (e) every guard is False at the defaults.
  W1 unit (CPU, real d192 snapshot v29_d192_m02_s711@77500 + a real MiniSplit train batch): classify -> 39 scaled matrices at
     sqrt(128/192) = 0.8165 and exactly the 7 fixed-fan-in names at 1.0 (1.843M / 0.179M params); the rebuilt Muon holds the same
     46 parameter objects with the same lr / wd / momentum / nesterov / adjust_lr_fn and 2 groups; apply(lr) sets lr * 0.8165 and lr.
     ONE optimiser step from identical copies (same batch, same dropout RNG, the harness lr 2e-3 / wd 0.02, grad clip 1.0): the
     loss is bitwise identical; AdamW-managed parameters and the 7 fixed-fan-in matrices are bitwise identical after the step;
     every scaled matrix moves by 0.8165 x the flag-off delta (max |dB - m dA| / max |dA| <= 1e-4, i.e. the update AND the decoupled
     decay scale together). The same at d128 (v17_pre2s_s101@77500, --mup_base 128 -> multiplier 1.0): bitwise identical step.
  W2 CPU end-to-end (tests/e2e_cpu.py, miniature split, the CD-harness command with --init, cooldown_mix 0.3, ~24 steps, evals every
     10, save_last / save_holdout / cliplog): frozen nn_x/w5 code vs the new code flag OFF, on the d128 root (as the harness) AND
     the d192 root: identical batch sequence, train-mode logits, loss-input logits, logged lines, bit-identical selected + @last
     checkpoints, val and holdout preds, stored args + {mup_base: 0}. Flag ON (--mup_base 128, d192): same batch sequence; step-0
     logits identical; later logits and the checkpoint differ; the wave-11 setup lines (39 / 7 matrices, 0.8165) and one summary
     line per eval whose hidden / input effective-lr ratio is 0.8165 with input == fixed == lr_at(step); args mup_base 128; cfg
     unchanged. Arm (b) --drop_cd 0.2 (d192): the drop_cd log line, checkpoint differs (plumbing).
GPU part
  G1 plumbing smoke: W5_SMOKE=1 lane_wg.sh <gpu> d192:mup:711 -> w11_smoke_wg_d192_mup_r711 (20K train feeds, 1 epoch, evals every
     25 steps): the lane completes; json / log / selected + @last checkpoints, val + holdout preds written; args mup_base 128; the
     wave-11 lines; val NDCG > 0.30; finite preds. Outputs deleted afterwards except logs/smoke_wg_d192_mup.log (a copy of the run
     log; no number of it enters any rule).
Writes nn_x/w11/width/logs/test_width_<part>.log (+ .json).
"""
import ast
import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

PART = sys.argv[1] if len(sys.argv) > 1 else "cpu"
assert PART in ("cpu", "gpu"), "usage: test_width.py cpu|gpu"
_CVD = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
if PART == "cpu":
    assert _CVD == "", "cpu part: run with CUDA_VISIBLE_DEVICES=''"
else:
    assert _CVD in ("0", "1", "2", "3"), "gpu part: CUDA_VISIBLE_DEVICES=<0-3, the WIDTH-GATE grant>"
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)  # nn_x/w11/width
ROOT = "<repo>"
W5 = f"{ROOT}/nn_x/w5"
PY = f"{ROOT}/.venv/bin/python"
SCR = os.environ.get("WG_SCRATCH", "/tmp/scratchpad/widthgate/tests")
os.makedirs(SCR, exist_ok=True)
os.makedirs(f"{D}/logs", exist_ok=True)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(min(12, int(os.environ.get("OMP_NUM_THREADS", "12") or 12)))
from strip_w11 import EXPECT, W11_IFS, strip_w11  # noqa: E402

LOG = open(f"{D}/logs/test_width_{PART}.log", "w")
RES, FAILS, N = {}, [], [0]
D192 = f"{ROOT}/nn/ckpt/v29_d192_m02_s711@77500.pt"
D128 = f"{ROOT}/nn/ckpt/v17_pre2s_s101@77500.pt"
MULT = (128 / 192) ** 0.5
FIXED = ["c_lin.weight", "c_mlp.0.weight", "retr.wk.weight", "retr.t1.weight", "retr.t2.weight", "u_lin.weight", "h_lin.weight"]
E2E = ("--epochs 2 --lr 2e-3 --warmup 0 --sched wsd --cooldown_frac 1.0 --muon --aux_pos 0.5 --aux_mode listwise --aux_tau 10 "
       "--aux_anneal 0.1 --cooldown_mix 0.3 --loss_norm feed --holdout 30000 --select_on holdout --eval_every 10 --save_last "
       "--save_holdout --select_min_step 10 --cliplog 5 --seed 1611 --no_test --cand_budget 4096 --max_feeds 8")


def P(s):
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def check(name, ok, info=""):
    N[0] += 1
    P(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {info}" if info != "" else ""))
    if not ok:
        FAILS.append(name)


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


class NormStr(ast.NodeTransformer):
    def visit_Constant(self, n):
        return ast.copy_location(ast.Constant("S"), n) if isinstance(n.value, str) else n


def lane_dry(script, gpu, jobs, env=None):
    r = subprocess.run(["bash", script, gpu, jobs], capture_output=True, text=True, env={**os.environ, "DRY": "1", **(env or {})})
    return r.returncode, [ln.split(": ", 1)[1] for ln in r.stdout.strip().splitlines() if ln.startswith("GPU ")], r.stderr


def make_parse_stub(train_py, name):
    """A stub module holding train.py's parse() only (no torch / common import): the parser lines are copied by AST."""
    tree = ast.parse(open(train_py).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    cd = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("XT_PRESETS", "XT_REL", "XT_RELONLY") for t in n.targets)]
    body = "import argparse\n" + "\n".join(ast.unparse(n) for n in cd) + "\n" + ast.unparse(fn).replace("return p.parse_args()", "return p") \
        + "\n\ndef parse_args(argv):\n    return parse().parse_args(argv)\n"
    body = body.replace("POOL_P_DEFAULT", "0.5")
    open(f"{SCR}/{name}.py", "w").write(body)
    if SCR not in sys.path:
        sys.path.insert(0, SCR)
    import importlib
    return importlib.import_module(name)


def parse_cmd(stub, cmd):
    return stub.parse_args(cmd.split("train.py ", 1)[1].split())


# ================================================================================================= CPU part
def w0_code():
    P("\n== W0 code")
    rec = dict(ln.split()[::-1] for ln in open(f"{HERE}/w5_source_md5.txt"))
    same = ["model.py", "data.py", "cd_arms.py", "cd_arms4.py", "cd_arms5.py", "cd_arms6.py", "cd_arms7.py", "cd_arms8.py",
            "finetune.py", "holdout_eval.py", "stale_pred.py", "blend.py"]
    bad = [f for f in same if not (md5(f"{D}/{f}") == rec[f] == md5(f"{W5}/{f}"))]
    check("(a) byte-identical module copies (md5 == recorded == live nn_x/w5)", not bad, f"{len(same)} files" + (f"; differ: {bad}" if bad else ""))
    check("(a) frozen reference tests/ref_w5_train.py.txt == live nn_x/w5/train.py (md5)",
          md5(f"{HERE}/ref_w5_train.py.txt") == rec["train.py"] == md5(f"{W5}/train.py"), rec["train.py"])
    c_new = open(f"{D}/common.py").read().replace('ROOT / "nn_x" / "w11/width"', 'ROOT / "nn_x" / "w5"')
    check("(a) common.py == nn_x/w5/common.py up to the output root (AST, docstrings ignored)",
          ast.dump(strip_doc(ast.parse(c_new))) == ast.dump(strip_doc(ast.parse(open(f"{W5}/common.py").read()))))
    s5, sw = open(f"{W5}/lane.sh").read(), open(f"{D}/lane_wg.sh").read()
    rx = lambda s_, k: re.search(k + r'="(.*?)"\n', s_, re.S).group(1)  # noqa: E731
    check("(a) lane_wg.sh B_BASE / HYG strings == nn_x/w5/lane.sh's", rx(s5, "B_BASE") == rx(sw, "B_BASE") and rx(s5, "HYG") == rx(sw, "HYG"))
    # (b) AST proof
    new_src, ref_src = open(f"{D}/train.py").read(), open(f"{HERE}/ref_w5_train.py.txt").read()
    t_new, removed = strip_w11(ast.parse(new_src))
    check("(b) strip_w11 removed exactly the expected WIDTH-GATE statement set", removed == EXPECT, dict(removed))
    dv_new, dv_ref = DropVariantAssert(), DropVariantAssert()
    t_new, t_ref = strip_doc(dv_new.visit(t_new)), strip_doc(dv_ref.visit(ast.parse(ref_src)))
    same_ast = ast.dump(t_new) == ast.dump(t_ref)
    check("(b) train.py minus WIDTH-GATE statements minus the variant assert == frozen nn_x/w5/train.py (AST, docstrings stripped)", same_ast)
    if not same_ast:
        for i, (x, y) in enumerate(zip(ast.unparse(t_new).splitlines(), ast.unparse(t_ref).splitlines())):
            if x != y:
                P(f"     first AST difference at unparsed line {i}: NEW {x[:100]!r} | REF {y[:100]!r}")
                break
    check("(b) exactly one variant assert each; new accepts exactly ('w5_', 'w11_'), ref 'w5_'",
          len(dv_new.dropped) == 1 == len(dv_ref.dropped) and "a.variant.startswith(('w5_', 'w11_'))" in dv_new.dropped[0]
          and "a.variant.startswith('w5_')" in dv_ref.dropped[0], dv_new.dropped[0][:70])
    # (c) parser defaults + lane DRY commands
    st_new, st_ref = make_parse_stub(f"{D}/train.py", "stub_new"), make_parse_stub(f"{HERE}/ref_w5_train.py.txt", "stub_ref")
    a_new, a_ref = vars(st_new.parse_args(["--variant", "w11_x"])), vars(st_ref.parse_args(["--variant", "w11_x"]))
    check("(c) parse(): --mup_base 0 by default; every other default == the frozen parser",
          a_new.get("mup_base") == 0 and {k: v for k, v in a_new.items() if k != "mup_base"} == a_ref)
    lane = f"{D}/lane_wg.sh"
    rc, cmds, err = lane_dry(lane, "0", "d192:m02:711 d192:drop02:711 d192:mup:711 d128:m02:901 d192:m02:715 d128:m02:905")
    rc5, ref5, _ = lane_dry(f"{W5}/lane.sh", "0", "B:m02ref:101", env={"W5_GPUS": "0"})
    ok = rc == 0 and rc5 == 0 and len(cmds) == 6 and len(ref5) == 1
    if ok:
        exp = ref5[0].replace("w5_cd_m02ref_r101", "w11_wg_d192_m02_r711").replace("v17_pre2s_s101@77500", "v29_d192_m02_s711@77500").replace("--seed 1001", "--seed 1611")
        ok &= cmds[0].split() == exp.split()
        ok &= cmds[1].split() == cmds[0].replace("w11_wg_d192_m02_r711", "w11_wg_d192_drop02_r711").split() + ["--drop_cd", "0.2"]
        ok &= cmds[2].split() == cmds[0].replace("w11_wg_d192_m02_r711", "w11_wg_d192_mup_r711").split() + ["--mup_base", "128"]
        exp128 = ref5[0].replace("w5_cd_m02ref_r101", "w11_wg_d128_m02_r901").replace("v17_pre2s_s101@77500", "v30_f7_m02_s901@77500").replace("--seed 1001", "--seed 1801")
        ok &= cmds[3].split() == exp128.split()
        ok &= "--seed 1615" in cmds[4] and "v29_d192_m02_s715@77500" in cmds[4] and "--seed 1805" in cmds[5] and "v30_f7_m02_s905@77500" in cmds[5]
        pa = [vars(parse_cmd(st_new, c)) for c in cmds]
        ok &= all(p["seed"] == 900 + int(p["variant"].rsplit("_r", 1)[1]) for p in pa)
        ok &= pa[0]["mup_base"] == 0 and pa[0]["drop_cd"] == -1.0 and pa[1]["drop_cd"] == 0.2 and pa[1]["mup_base"] == 0 and pa[2]["mup_base"] == 128 and pa[2]["drop_cd"] == -1.0
        ok &= all(p["init"].endswith("@77500.pt") and os.path.exists(p["init"]) for p in pa)
    check("(c) lane DRY: d192:m02:711 == nn_x/w5 B:m02ref:101 with init / variant / seed (900 + root) substituted; drop02 == m02 + '--drop_cd 0.2' last; "
          "mup == m02 + '--mup_base 128' last; d128:m02:901 from the v30 snapshot, seed 1801; snapshots exist", ok, err[-200:] if err else "")
    RES["dry"] = cmds
    for c in cmds[:4]:
        P("     " + c.split("train.py ", 1)[1][:160] + " ...")
    bad = {j: lane_dry(lane, g, j)[0] for g, j in (("4", "d192:m02:711"), ("0", "d192:m02:710"), ("0", "d256:m02:711"), ("0", "d192:foo:711"), ("0", "d128:m02:907"))}
    check("(c) lane refuses GPU N, root 710, width d256, arm foo, d128 root 907", all(r_ != 0 for r_ in bad.values()), bad)
    check("(c) bash -n lane_wg.sh", subprocess.run(["bash", "-n", lane], capture_output=True).returncode == 0)
    # (d) call sites
    tree = ast.parse(new_src)
    parents = {ch: p_ for p_ in ast.walk(tree) for ch in ast.iter_child_nodes(p_)}

    def chain(n):
        out = []
        while n in parents:
            n = parents[n]
            if isinstance(n, ast.If):
                out.append("if " + ast.unparse(n.test))
            elif isinstance(n, ast.For):
                out.append("for " + ast.unparse(n.target).strip("()"))
            elif isinstance(n, ast.FunctionDef):
                out.append("def " + n.name)
        return out

    def calls(pred):
        return [chain(n) for n in ast.walk(tree) if isinstance(n, ast.Call) and pred(n.func)]

    su = calls(lambda f: isinstance(f, ast.Name) and f.id == "setup_mup")
    ap = calls(lambda f: isinstance(f, ast.Attribute) and f.attr == "apply" and isinstance(f.value, ast.Name) and f.value.id == "mup")
    sm = calls(lambda f: isinstance(f, ast.Attribute) and f.attr == "summary" and isinstance(f.value, ast.Name) and f.value.id == "mup")
    check("(d) setup_mup once behind `if a.mup_base > 0` in main(); mup.apply once behind `if mup is not None` in the step loop; mup.summary once "
          "behind `if mup is not None` inside the eval block",
          su == [["if a.mup_base > 0", "def main"]] and ap == [["if mup is not None", "for step, fb", "def main"]]
          and len(sm) == 1 and sm[0][0] == "if mup is not None" and sm[0][1].startswith("if (step + 1) % a.eval_every == 0 or last") and sm[0][2] == "for step, fb",
          f"{su} {ap} {sm}")
    main_fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    mb = main_fn.body
    j_muon = [i for i, s in enumerate(mb) if isinstance(s, ast.If) and ast.unparse(s.test) == "a.muon" and "Muon" in ast.unparse(s)]
    j_init = [i for i, s in enumerate(mb) if isinstance(s, ast.Assign) and ast.unparse(s) == "mup = None"]
    j_set = [i for i, s in enumerate(mb) if isinstance(s, ast.If) and ast.unparse(s.test) == "a.mup_base > 0"]
    loops = [s for s in ast.walk(main_fn) if isinstance(s, ast.For) and ast.unparse(s.target).strip("()") == "step, fb"]
    body = loops[0].body
    i_reset = [i for i, s in enumerate(body) if isinstance(s, ast.For) and ast.unparse(s.target) == "opt" and 'g[\'lr\'] = lr_at(step)' in ast.unparse(s)]
    i_ap = [i for i, s in enumerate(body) if isinstance(s, ast.If) and ast.unparse(s.test) == "mup is not None"]
    check("(d) positions: `mup = None` + `if a.mup_base > 0` right after the `if a.muon:` optimizer block; the apply guard right after the per-step "
          "lr-reset loop (before any batch work)",
          len(loops) == 1 and j_muon and j_init == [j_muon[0] + 1] and j_set == [j_muon[0] + 2] and len(i_reset) == 1 and i_ap == [i_reset[0] + 1],
          f"muon {j_muon} init {j_init} set {j_set} | reset {i_reset} apply {i_ap}")
    users = sorted(f for f in os.listdir(D) if f.endswith(".py") and f not in ("cd_arms11.py", "train.py") and re.search(r"\bcd_arms11\b", open(f"{D}/{f}").read()))
    imps = sorted({(n.module if isinstance(n, ast.ImportFrom) else n.names[0].name)
                   for n in ast.walk(ast.parse(open(f"{D}/cd_arms11.py").read())) if isinstance(n, (ast.Import, ast.ImportFrom))})
    check("(d) no other module imports cd_arms11; cd_arms11 imports only math and torch", users == [] and imps == ["math", "torch"], f"users {users} imports {imps}")
    # (e) guards
    import argparse
    env = dict(a=argparse.Namespace(**a_new), mup=None)
    vals = {g: bool(eval(g, {}, env)) for g in W11_IFS}
    check("(e) every WIDTH-GATE guard is False at the defaults", not any(vals.values()), vals)


def one_step(model, batch, opts, seed):
    """One harness-like training step (train mode, the main listwise loss only; dropout RNG seeded) on `model`; returns the loss."""
    sys.path.insert(0, D)
    from model import listwise_loss
    model.train()
    torch.manual_seed(seed)
    X = model.dense(batch)
    lg = model(batch, X, use_pos=True)
    loss = listwise_loss(lg, batch["y"], batch["cmask"], None, norm="feed", denom=512)
    for o in opts:
        o.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for o in opts:
        o.step()
    return loss.detach().clone()


def build(ckpt):
    sys.path.insert(0, D)
    from model import Ranker
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = Ranker(**ck["cfg"])
    m.load_state_dict(ck["state_dict"])
    muon_names = {f"{mn}.weight" for mn, mod in m.named_modules() if isinstance(mod, torch.nn.Linear) and min(mod.weight.shape) > 1}
    rest = [(n, p) for n, p in m.named_parameters() if n not in muon_names]
    decay = [p for n, p in rest if p.ndim >= 2 and "emb" not in n]
    no_decay = [p for n, p in rest if not (p.ndim >= 2 and "emb" not in n)]
    opts = [torch.optim.AdamW([{"params": decay, "weight_decay": 0.02}, {"params": no_decay, "weight_decay": 0.0}], lr=2e-3, betas=(0.9, 0.98))]
    P_muon = [p for n, p in m.named_parameters() if n in muon_names]
    opts.append(torch.optim.Muon(P_muon, lr=2e-3, weight_decay=0.02, momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw"))
    return m, muon_names, opts, ck["cfg"]


def w1_unit():
    P("\n== W1 unit: classification, optimizer rebuild, one-step delta ratio (CPU, real d192 snapshot + real MiniSplit batch)")
    import argparse
    sys.path.insert(0, D)
    import cd_arms11 as C
    from mini_split import MiniSplit, pick_train_feeds
    m, muon_names, opts, cfg = build(D192)
    mult = C.classify(m, muon_names, 128)
    scaled = sorted(n for n, x in mult.items() if x != 1.0)
    fixed = sorted(n for n, x in mult.items() if x == 1.0)
    shapes = {n: tuple(p.shape) for n, p in m.named_parameters() if n in muon_names}
    n_sc = sum(np.prod(shapes[n]) for n in scaled)
    n_fx = sum(np.prod(shapes[n]) for n in fixed)
    check("classify(d192, base 128): 46 Muon matrices -> 39 scaled at sqrt(128/192) = 0.8165 (1.843M params) + exactly the 7 fixed-fan-in names (0.179M)",
          len(mult) == 46 and len(scaled) == 39 and fixed == sorted(FIXED) and all(abs(mult[n] - MULT) < 1e-12 for n in scaled)
          and abs(n_sc / 1e6 - 1.843) < 0.002 and abs(n_fx / 1e6 - 0.179) < 0.002, f"scaled {n_sc / 1e6:.3f}M fixed {n_fx / 1e6:.3f}M")
    check("classify: every scaled matrix has fan_in in {d, 2d}; every fixed one has fan_in in {256, 104, 128, 64} (not a multiple of 192)",
          all(shapes[n][1] in (192, 384) for n in scaled) and all(shapes[n][1] in (256, 104, 128, 64) for n in fixed))
    old = opts[1]
    ids_old = {id(p) for p in old.param_groups[0]["params"]}
    lines = []
    mup = C.setup_mup(argparse.Namespace(muon=True, mup_base=128), m, opts, muon_names, lines.append)
    new = opts[1]
    ids_new = {id(p) for g in new.param_groups for p in g["params"]}
    g_sc = [g for g in new.param_groups if g["lr_mult"] != 1.0]
    g_fx = [g for g in new.param_groups if g["lr_mult"] == 1.0]
    same_hp = all(g[k] == old.param_groups[0][k] for g in new.param_groups for k in ("lr", "weight_decay", "momentum", "nesterov", "adjust_lr_fn"))
    check("setup_mup: the Muon optimizer in opts is replaced in place by a 2-group Muon over the SAME 46 parameter objects (39 at 0.8165, 7 at 1.0), same "
          "lr / wd / momentum / nesterov / adjust_lr_fn; AdamW untouched; 1 + 3 log lines",
          isinstance(new, torch.optim.Muon) and new is not old and ids_new == ids_old and len(new.param_groups) == 2 and len(g_sc) == 1 and len(g_fx) == 1
          and len(g_sc[0]["params"]) == 39 and len(g_fx[0]["params"]) == 7 and same_hp and isinstance(opts[0], torch.optim.AdamW) and len(lines) == 4
          and lines[0].startswith("wave-11 WIDTH-GATE muP: --mup_base 128 at d 192") and "0.8165 on 39" in lines[0] and "1.0 on 7" in lines[0]
          and "weight decay" in lines[0], lines[0][:120])
    mup.apply(1.234e-3)
    lrs = {g["lr_mult"]: g["lr"] for g in new.param_groups}
    check("apply(lr): scaled group lr == lr * 0.8165, fixed group lr == lr; summary line names one hidden / input / fixed matrix with those lrs",
          abs(lrs[MULT] - 1.234e-3 * MULT) < 1e-15 and lrs[1.0] == 1.234e-3 and "cross.0.attn.q.weight" in mup.summary(3) and "c_lin.weight" in mup.summary(3)
          and "retr.wk.weight" in mup.summary(3) and f"{1.234e-3 * MULT:.3e}" in mup.summary(3), mup.summary(3)[:150])
    RES["w1_lines"] = lines
    # one step: flag off (A) vs flag on (B) from identical copies on the same real batch
    torch.set_num_threads(1)  # bitwise-reproducible CPU arithmetic for the A/B comparison (multi-threaded reductions may reorder)
    ms = MiniSplit("train", pick_train_feeds(n_each=6, seed=4), xt_names=cfg["xt_names"], with_domain=True)
    fb = ms.batches(np.arange(ms.n), 4096, 8, shuffle=True, rng=np.random.default_rng(0))[0]
    b = ms.gather(fb, with_pos=True)
    for ckpt, tag, exp_mult in ((D192, "d192", MULT), (D128, "d128", 1.0)):
        mA, namesA, optsA, _ = build(ckpt)
        mB, namesB, optsB, _ = build(ckpt)
        C.setup_mup(argparse.Namespace(muon=True, mup_base=128), mB, optsB, namesB, lambda *_: None)
        lr = 2e-3
        for o in optsA:
            for g in o.param_groups:
                g["lr"] = lr
        for o in optsB:
            for g in o.param_groups:
                g["lr"] = lr
        for g in optsB[1].param_groups:  # the per-step hook of train.py
            g["lr"] = lr * g["lr_mult"]
        p0 = {n: p.detach().clone() for n, p in mA.named_parameters()}
        lossA = one_step(mA, b, optsA, 7)
        lossB = one_step(mB, b, optsB, 7)
        pa = {n: p.detach() for n, p in mA.named_parameters()}
        pb = {n: p.detach() for n, p in mB.named_parameters()}
        same_loss = torch.equal(lossA, lossB)
        adam_same = all(torch.equal(pa[n], pb[n]) for n in pa if n not in namesA)
        fixed_same = all(torch.equal(pa[n], pb[n]) for n in FIXED)
        worst, moved = 0.0, 0.0
        for n in namesA:
            if n in FIXED:
                continue
            dA, dB = pa[n] - p0[n], pb[n] - p0[n]
            worst = max(worst, ((dB - exp_mult * dA).abs().max() / dA.abs().max().clamp_min(1e-30)).item())
            moved = max(moved, dA.abs().max().item())
        exact_same = all(torch.equal(pa[n], pb[n]) for n in namesA)
        P(f"     {tag}: same_loss {same_loss} adam_same {adam_same} fixed_same {fixed_same} exact_same {exact_same} worst rel {worst:.2e}")
        if tag == "d192":
            check(f"{tag} one step (lr 2e-3, wd 0.02, clip 1.0): loss bitwise identical; AdamW params and the 7 fixed-fan-in matrices bitwise identical; every "
                  f"scaled matrix moved by 0.8165 x the flag-off delta (max |dB - m dA| / max|dA| <= 1e-4; update and decoupled decay scale together)",
                  same_loss and adam_same and fixed_same and worst <= 1e-4 and moved > 0, f"loss {lossA.item():.5f}; worst rel {worst:.2e}; max |dA| {moved:.2e}")
        else:
            check(f"{tag} one step with --mup_base 128 at d128 (multiplier 1.0): bitwise identical to the flag-off step (loss, every parameter)",
                  same_loss and exact_same and adam_same, f"loss {lossA.item():.5f}; worst rel {worst:.2e}")
        RES[f"w1_step_{tag}"] = dict(loss=float(lossA), worst_rel=float(worst), max_dA=float(moved))


def code_copy(src, dst, out, line):
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


def e2e_run(code_dir, v, init, extra):
    rec = f"{SCR}/rec_{v}.npz"
    cmd = [PY, f"{HERE}/e2e_cpu.py", code_dir, rec, "--variant", v, "--init", init] + E2E.split() + extra
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


def w2_e2e():
    P("\n== W2 CPU end-to-end (e2e_cpu.py, miniature split, CD-harness command with --init): flag OFF vs frozen nn_x/w5 code (d128 + d192 roots); --mup_base 128; --drop_cd 0.2")
    ref_dir, ref_out = f"{SCR}/ref_w5_code", f"{SCR}/ref_w5_out"
    new_dir, new_out = f"{SCR}/new_wg_code", f"{SCR}/new_wg_out"
    ok_r = code_copy(W5, ref_dir, ref_out, 'W5 = ROOT / "nn_x" / "w5"')
    ok_n = code_copy(D, new_dir, new_out, 'W5 = ROOT / "nn_x" / "w11/width"')
    check("code copies for the e2e (outputs redirected to scratch): modules byte-identical to nn_x/w5 / nn_x/w11/width, common.py differs only in the path string", ok_r and ok_n, (ok_r, ok_n))
    runs = {"ref128": (ref_dir, D128, []), "new128": (new_dir, D128, []), "ref192": (ref_dir, D192, []), "new192": (new_dir, D192, []),
            "mup192": (new_dir, D192, ["--mup_base", "128"]), "drop192": (new_dir, D192, ["--drop_cd", "0.2"])}
    res, err = {}, {}

    def worker(k):
        try:
            res[k] = e2e_run(runs[k][0], f"w5_twg_{k}", runs[k][1], runs[k][2])
        except Exception as e:  # noqa: BLE001
            err[k] = repr(e)[-2500:]
            P(f"  ERROR {k}: {err[k][-400:]}")
    ths = [threading.Thread(target=worker, args=(k,)) for k in runs]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    check("all 6 e2e runs completed", not err, list(err))
    if err:
        return
    outs = {"ref128": ref_out, "ref192": ref_out}
    out_of = lambda k: outs.get(k, new_out)  # noqa: E731
    logs = {k: open(f"{out_of(k)}/logs/w5_twg_{k}.log").read() for k in runs}
    for a_, b_ in (("ref128", "new128"), ("ref192", "new192")):
        ra, rb = res[a_], res[b_]
        nb = len(ra["off"]) - 1
        va, vb = f"w5_twg_{a_}", f"w5_twg_{b_}"
        same_seq = np.array_equal(ra["fidx"], rb["fidx"]) and np.array_equal(ra["off"], rb["off"]) and np.array_equal(ra["ci"], rb["ci"]) and nb >= 20
        same_lg = (np.array_equal(ra["lg_model"], rb["lg_model"]) and np.array_equal(ra["lg_loss"], rb["lg_loss"]) and int(rb["n_loss"]) == int(rb["n_model"]) == nb
                   and np.array_equal(ra["lw_val"], rb["lw_val"]) and np.array_equal(ra["total"], rb["total"]))
        ma, mb = metric_lines(logs[a_]), metric_lines(logs[b_])
        files = (ck_equal(f"{out_of(a_)}/ckpt/{va}.pt", f"{out_of(b_)}/ckpt/{vb}.pt") and ck_equal(f"{out_of(a_)}/ckpt/{va}@last.pt", f"{out_of(b_)}/ckpt/{vb}@last.pt")
                 and all(np.array_equal(np.load(f"{out_of(a_)}/{d_}nn_{va}{s_}"), np.load(f"{out_of(b_)}/{d_}nn_{vb}{s_}"))
                         for d_, s_ in (("preds/", "_trainonly_val.npy"), ("preds/", "@last_trainonly_val.npy"), ("preds/holdout/", "_holdout.npy"), ("preds/holdout/", "@last_holdout.npy"))))
        ja, jb = json.load(open(f"{out_of(a_)}/logs/{va}.json")), json.load(open(f"{out_of(b_)}/logs/{vb}.json"))
        d_args = {k: (ja["args"].get(k), jb["args"].get(k)) for k in set(ja["args"]) | set(jb["args"]) if ja["args"].get(k) != jb["args"].get(k)}
        check(f"flag OFF vs frozen w5 ({a_[3:]} root): identical training-batch sequence ({nb} steps incl. pool draws), train-mode / loss-input logits, per-step losses "
              f"and totals (bitwise); identical logged step / BEST lines; bit-identical selected + @last checkpoints, val + holdout preds; args + {{mup_base: 0}}; no wave-11 line",
              same_seq and same_lg and ma == mb and len(ma) >= 3 and files and d_args == {"variant": (va, vb), "mup_base": (None, 0)}
              and ja["best_step"] == jb["best_step"] and "wave-11" not in logs[a_] and "wave-11" not in logs[b_],
              f"{nb} steps; last line {mb[-1] if mb else ''}; d_args {d_args}")
        RES[f"e2e_{b_}"] = dict(steps=int(nb), lines=mb)
    # flag ON
    rb, rc_ = res["new192"], res["mup192"]
    nb = len(rb["off"]) - 1
    s0 = int(rc_["shapes"][0].prod())
    lc = logs["mup192"]
    setup_ok = "wave-11 WIDTH-GATE muP: --mup_base 128 at d 192: Muon lr multiplier sqrt(128/192) = 0.8165 on 39 width-growing matrices" in lc and "1.0 on 7 fixed-fan-in" in lc
    summ = re.findall(r"wave-11 muP step (\d+): effective Muon lr hidden cross\.0\.attn\.q\.weight ([\d.e+-]+) \(mult 0\.8165, decay factor [\d.]+\), input c_lin\.weight ([\d.e+-]+) \(1\.0\), fixed retr\.wk\.weight ([\d.e+-]+) \(1\.0\)", lc)
    ratios = [float(h) / float(i) for _, h, i, f in summ]
    steps_lines = [int(s) for s, *_ in summ]
    eval_steps = [int(re.match(r"step (\d+)/", ln).group(1)) for ln in lc.splitlines() if ln.startswith("step ")]
    lr_at = lambda s: 2e-3 * (0.03 + 0.97 * (1 - ((s - 1) / nb) ** 0.5))  # noqa: E731  (wsd, cooldown_frac 1.0, warmup 0; summary logged at step+1)
    lr_ok = all(abs(float(i) - lr_at(int(s))) <= 1e-3 * lr_at(int(s)) and float(i) == float(f) for s, h, i, f in summ)  # %.3e print
    check("--mup_base 128 (d192): same batch sequence as flag OFF; step-0 train logits identical (same init / batch); later logits and the checkpoint differ (mechanism live)",
          np.array_equal(rc_["fidx"], rb["fidx"]) and np.array_equal(rc_["ci"], rb["ci"]) and np.array_equal(rc_["lg_model"][:s0], rb["lg_model"][:s0])
          and not np.array_equal(rc_["lg_model"], rb["lg_model"]) and not ck_equal(f"{new_out}/ckpt/w5_twg_mup192.pt", f"{new_out}/ckpt/w5_twg_new192.pt")
          and rc_["lw_val"][0] == rb["lw_val"][0] and rc_["lw_val"][1] != rb["lw_val"][1], f"step-0 loss {rc_['lw_val'][0]:.5f}; step-1 loss on {rc_['lw_val'][1]:.5f} vs off {rb['lw_val'][1]:.5f}")
    check("--mup_base 128: the wave-11 setup lines (39 scaled at 0.8165 / 7 fixed) and one summary line per eval with hidden / input effective-lr ratio 0.8165, "
          "input == fixed == lr_at(step) (wsd cooldown from 2e-3)",
          setup_ok and len(summ) == len(eval_steps) >= 3 and steps_lines == eval_steps and all(abs(r_ - MULT) < 1e-3 for r_ in ratios) and lr_ok,
          f"{len(summ)} summary lines at steps {steps_lines}; ratios {[round(r_, 4) for r_ in ratios]}; input lrs {[i for _, _, i, _ in summ]}")
    jc = json.load(open(f"{new_out}/logs/w5_twg_mup192.json"))
    check("--mup_base 128: args mup_base 128, checkpoint cfg == flag-off cfg (no model change), @last / holdout files written",
          jc["args"]["mup_base"] == 128 and torch.load(f"{new_out}/ckpt/w5_twg_mup192.pt", map_location="cpu", weights_only=False)["cfg"]
          == torch.load(f"{new_out}/ckpt/w5_twg_new192.pt", map_location="cpu", weights_only=False)["cfg"]
          and all(os.path.exists(f"{new_out}/{f}") for f in ("ckpt/w5_twg_mup192@last.pt", "preds/nn_w5_twg_mup192@last_trainonly_val.npy",
                                                             "preds/holdout/nn_w5_twg_mup192_holdout.npy", "preds/holdout/nn_w5_twg_mup192@last_holdout.npy")),
          f"best_step {jc['best_step']} val {jc['val_ndcg']:.5f}")
    ld = logs["drop192"]
    check("--drop_cd 0.2 (d192, arm b plumbing): drop_cd log line (p=0.2 from step 0), same batch sequence, checkpoint differs from flag OFF",
          "drop_cd: dropout p=0.2 from step 0" in ld and np.array_equal(res["drop192"]["fidx"], rb["fidx"])
          and not ck_equal(f"{new_out}/ckpt/w5_twg_drop192.pt", f"{new_out}/ckpt/w5_twg_new192.pt"))
    RES["e2e_mup"] = dict(summary=summ, on_lines=metric_lines(lc), setup=[ln for ln in lc.splitlines() if "wave-11" in ln][:4])
    for d_ in (ref_dir, ref_out, new_dir, new_out):
        shutil.rmtree(d_, ignore_errors=True)


# ================================================================================================= GPU part
def outputs(v):
    fs = [f"{D}/logs/{v}.log", f"{D}/logs/{v}.json", f"{D}/logs/{v}_cliplog.npz", f"{D}/logs/{v}.stdout", f"{D}/logs/{v}_holdout.stdout"]
    for x in (v, f"{v}@last"):
        fs += [f"{D}/ckpt/{x}.pt", f"{D}/preds/nn_{x}_trainonly_val.npy", f"{D}/preds/nn_{x}_trainonly_test.npy", f"{D}/preds/holdout/nn_{x}_holdout.npy"]
    return fs


def g1_smoke():
    P(f"\n== G1 plumbing smoke on GPU {_CVD}: W5_SMOKE=1 lane_wg.sh {_CVD} d192:mup:711")
    v = "w11_smoke_wg_d192_mup_r711"
    for f in outputs(v):
        if os.path.exists(f):
            os.remove(f)
    shutil.rmtree(f"{D}/logs/queue/claim_{v}", ignore_errors=True)
    t0 = time.time()
    r = subprocess.run(["bash", f"{D}/lane_wg.sh", _CVD, "d192:mup:711"], capture_output=True, text=True, env={**os.environ, "W5_SMOKE": "1"})
    P(f"  lane rc {r.returncode} in {time.time() - t0:.0f}s; tail: {r.stdout.strip().splitlines()[-1][:160] if r.stdout.strip() else ''}")
    ok_files = all(os.path.exists(f) for f in outputs(v) if not f.endswith("_holdout.stdout"))
    check("lane completed; json / log / cliplog / selected + @last checkpoints / val + test + holdout preds written", r.returncode == 0 and ok_files,
          [f for f in outputs(v) if not os.path.exists(f)])
    if ok_files:
        j = json.load(open(f"{D}/logs/{v}.json"))
        lg = open(f"{D}/logs/{v}.log").read()
        pv = np.load(f"{D}/preds/nn_{v}_trainonly_val.npy")
        ph = np.load(f"{D}/preds/holdout/nn_{v}_holdout.npy")
        summ = re.findall(r"wave-11 muP step (\d+): effective Muon lr hidden cross\.0\.attn\.q\.weight ([\d.e+-]+) \(mult 0\.8165", lg)
        check("args mup_base 128 + init v29_d192_m02_s711@77500; wave-11 setup line (39 / 7) and a summary line per eval; val NDCG > 0.30; finite preds",
              j["args"]["mup_base"] == 128 and j["args"]["init"].endswith("v29_d192_m02_s711@77500.pt") and "0.8165 on 39 width-growing" in lg and len(summ) >= 1
              and j["val_ndcg"] > 0.30 and np.isfinite(pv).all() and np.isfinite(ph).all(),
              f"val {j['val_ndcg']:.5f} best_step {j['best_step']} minutes {j['train_minutes']:.1f}; summaries {len(summ)}")
        RES["smoke"] = dict(val=j["val_ndcg"], best_step=j["best_step"], minutes=j["train_minutes"], history=j["history"])
        shutil.copy(f"{D}/logs/{v}.log", f"{D}/logs/smoke_wg_d192_mup.log")
    for f in outputs(v):
        if os.path.exists(f):
            os.remove(f)
    shutil.rmtree(f"{D}/logs/queue/claim_{v}", ignore_errors=True)


if __name__ == "__main__":
    t0 = time.time()
    P(f"test_width {PART} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} in {D}")
    if PART == "cpu":
        w0_code()
        w1_unit()
        w2_e2e()
    else:
        g1_smoke()
    P(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed in {time.time() - t0:.0f}s" + (f"; FAILED: {FAILS}" if FAILS else ""))
    json.dump(dict(part=PART, passed=N[0] - len(FAILS), total=N[0], fails=FAILS, res=RES), open(f"{D}/logs/test_width_{PART}.json", "w"), indent=1, default=str)
    sys.exit(1 if FAILS else 0)
