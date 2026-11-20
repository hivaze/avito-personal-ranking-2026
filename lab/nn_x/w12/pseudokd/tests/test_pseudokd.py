"""Tests of the pseudokd port (lab/nn_x/w12/pseudokd, 2026-11-17): the wave-11 PSEUDO-DIRECT pseudo-feed co-training
(lab/nn_x/w11/pseudo: cd_arms11.py + the pseudo flag block of its train.py) ported into a COPY of the current student trainer
lab/nn_x/w5/train.py (md5 7c1e3588, round L1117; frozen ref/train_w5.py.txt). Pattern: lab/nn_x/w11/pseudo/tests/test_pseudo.py.

  cpu part:  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/nn_x/w12/pseudokd/tests/test_pseudokd.py cpu
  gpu part:  cd <repo> && CUDA_VISIBLE_DEVICES=<6|7> OMP_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/nn_x/w12/pseudokd/tests/test_pseudokd.py gpu
             (one plumbing smoke through lane_pk.sh PK_SMOKE=1; outputs in lab/nn_x/w12/pseudokd/smoke_out)

CPU part
  C0 static.
    (a) frozen copies == live files (md5): ref/train_w5.py.txt == lab/nn_x/w5/train.py, ref/lane_w5.sh.txt == lab/nn_x/w5/lane.sh,
        ref/cd_arms11_w11.py.txt == lab/nn_x/w11/pseudo/cd_arms11.py; the w5 modules imported at run time are logged (md5).
    (b) AST proof: train.py minus the PSEUDO statements (tests/strip_pseudo.py, exact removed set asserted) == ref/train_w5.py.txt
        (docstrings stripped); the variant-prefix assert is unchanged (w5_).
    (c) TEXTUAL diff ref/train_w5.py.txt -> train.py: the only removed line is the old first docstring line (it moves below the new
        header); every other change is an insertion that lies inside the module docstring or parses into statements that each name
        'pseudo'; the inserted code is exactly the 7 flags, the import, `pseudo = None` + the setup block, the loop block, the summary
        and save blocks (= the w11 pseudo train.py block, helper renamed).
    (d) parse(): the 7 pseudo flags at their OFF defaults; every other default == the frozen w5 parse() defaults (same key set + 7).
    (e) call sites: setup_pseudo / combine / summary / save once each; the setup block follows `pseudo = None`, which follows
        `kdc = setup_kdc(a, log)`; the loop block follows the R-Drop block, precedes the zero_grad loop, holds exactly
        `loss = pseudo.combine(model, a, b, loss, prand, cliplog)` (b = this step's gathered REAL training batch); nothing else in the
        folder imports cd_pseudokd.
    (f) helper: cd_pseudokd.py == cd_arms11.py (AST, docstrings stripped) after removing exactly the 5 new asserts + 1 new log call of
        setup_pseudo; eligible_targets / draw_sources / build_pseudo_batch / PseudoHead / PseudoTrainer / holdout_user_mask identical.
    (g) launcher: lane_pk.sh DRY E:kd05w192ps == nn_x/w5/lane.sh DRY E:kd05w192 token for token (trainer path and variant renamed) +
        exactly the 12 pseudo tokens; the parsed args vs the STORED w5_e10_kd05w192_s951/952 args differ only in variant + flags added
        to w5 later (at their defaults) + the pseudo flags at the arm's values; PK_SMOKE DRY = the same + lane.sh's W5_SMOKE tokens,
        a 'smoke' variant and the output-redirect runner; unknown arms / GPUs refused.
  C1 unit (MiniSplit = real train feeds, CPU): builder at R = 1 / 2 over 8 batches vs the verbatim cd_arms11 brute-force checker AND
     the port's locality / padding / key-set checker; t in [0, 63], g in [16, 64]; determinism; blocked (holdout-user) rows never
     source; holdout_user_mask blocks exactly the holdout users' feeds, never user 0. LOCALITY (no other feed, no later snapshot):
     every row of the batch that is NOT a source of a checked pseudo-feed is overwritten with garbage (history, candidates, clicks,
     categories, user embedding, f1-f5, xt) and the pseudo batch is rebuilt with the same generator: every pseudo-feed whose source
     was kept is bitwise identical. Loss combine with a small Ranker (+ head): loss + 0.3 x listwise(pseudo) (rel 1e-6) with the real
     loss's norm / denom, cliplog.main advanced by the same term, zero-init head a no-op at step 0, --pseudo_noset restores use_set.
     Guards: setup_pseudo refuses --hist_ext / --aug_all / --wcl / --ccl / --fsw / --kdgap / --robust / --casc / --xfold_unlab.
  C2 CPU end-to-end (tests/e2e_cpu.py, miniature split, the E recipe at mini scale WITH the KD teacher (kd_w 0.5, d 192), 2 mini
     epochs, cooldown + pool batches, evals every 10 steps, save_last / save_holdout / save_steps / cliplog):
     OFF: live lab/nn_x/w5/train.py vs this train.py flags off: identical batch sequence, train-mode logits, loss-input logits, loss
     values, totals, logged lines; bit-identical selected / @last / @10 / @20 checkpoints, val and holdout preds, cliplog arrays;
     stored args == w5 args + the 7 pseudo defaults; no pseudo line, no head file.
     ON (the arm's pseudo flags): same batch sequence and step count; combine on EVERY plateau step, never in the cooldown; every
     pseudo-feed passes both checkers; no pseudo source is a holdout user's feed; step-0 real logits identical to OFF; real listwise
     targets are the KD mixture (soft) and pseudo listwise targets one-hot; total == real + aux + 0.3 x pseudo; with --aux_pos 0:
     total == real + 0.3 x pseudo on every plateau step (rel 1e-6) and == real on cooldown steps; checkpoint cfg == OFF cfg; head
     file with cumulative counts; the pseudo log lines.
GPU part
  G1 plumbing smoke (ONE process; GPU N or 7 under the lane guard): PK_SMOKE=1 lane_pk.sh <gpu> E:kd05w192ps:951 ->
     w5_smoke_e10_kd05w192ps_s951 (20K train feeds, 1 epoch, evals every 25 steps, outputs in smoke_out/): completes; json / log /
     selected + @last checkpoints / val, holdout and test preds; pseudo-feeds built on every plateau step and none after t_cool; KD
     line; finite preds; nothing of the smoke in nn_x/w5/{logs,ckpt,preds}. Large outputs deleted afterwards (log / json kept).
Writes lab/nn_x/w12/pseudokd/logs/test_pseudokd_<part>.log (+ .json).
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
assert PART in ("cpu", "gpu"), "usage: test_pseudokd.py cpu|gpu"
_CVD = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
if PART == "cpu":
    assert _CVD == "", "cpu part: run with CUDA_VISIBLE_DEVICES=''"
else:
    assert _CVD in ("6", "7"), "gpu part: CUDA_VISIBLE_DEVICES=N or 7 (this agent's smoke grant)"
NT = min(12, int(os.environ.get("OMP_NUM_THREADS", "12")))
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(NT)
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)  # lab/nn_x/w12/pseudokd
ROOT = "<repo>"
W5 = f"{ROOT}/lab/nn_x/w5"
W11 = f"{ROOT}/lab/nn_x/w11/pseudo"
PY = f"{ROOT}/.venv/bin/python"
SCR = "/tmp/scratchpad/pk_test"
os.makedirs(SCR, exist_ok=True)
os.makedirs(f"{D}/logs", exist_ok=True)
sys.path.insert(0, HERE)
from strip_pseudo import EXPECT, strip_pseudo  # noqa: E402

LOG = open(f"{D}/logs/test_pseudokd_{PART}.log", "w")
RES, FAILS, N = {}, [], [0]
KD = "<repo>/nn_x/w12/kd/out/teacher_mean11.npy"
E2E = ("--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout "
       "--loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --lr 2e-3 --sched wsd --cooldown_frac 0.5 --cooldown_mix 0.3 "
       "--aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 2 --eval_every 10 --no_test --save_last --save_holdout "
       f"--select_min_step 10 --cliplog 5 --warmup 2 --cand_budget 4096 --max_feeds 8 --seed 951 --save_steps 10,20 --kd_file {KD} "
       "--kd_w 0.5 --d 192")
ARM_PSEUDO = ["--pseudo_ratio", "1.0", "--pseudo_w", "0.3", "--pseudo_head", "1", "--pseudo_gap_min", "16", "--pseudo_gap_max", "64",
              "--pseudo_tmax", "63"]
PSEUDO_DEFAULTS = dict(pseudo_ratio=0.0, pseudo_gap_min=16, pseudo_gap_max=64, pseudo_w=0.3, pseudo_tmax=63, pseudo_head=1, pseudo_noset=0)
PSEUDO_ARM = dict(PSEUDO_DEFAULTS, pseudo_ratio=1.0)


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


def lane_dry(script, gpu, jobs, env=None):
    r = subprocess.run(["bash", script, gpu, jobs], capture_output=True, text=True, env={**os.environ, "DRY": "1", **(env or {})})
    assert r.returncode == 0, r.stderr
    return [ln.split(": ", 1)[1] for ln in r.stdout.strip().splitlines() if ln.startswith("GPU ")]


def _make_parse_stub(src_path, name):
    tree = ast.parse(open(src_path).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    cd = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("XT_PRESETS", "XT_REL", "XT_RELONLY") for t in n.targets)]
    body = ("import argparse\nHIST_EXT_DIR = '<repo>/artifacts/data/feats_new/histext_1117'\n" + "\n".join(ast.unparse(n) for n in cd)
            + "\n" + ast.unparse(fn).replace("return p.parse_args()", "return p") + "\n\ndef parse_args(argv):\n    return parse().parse_args(argv)\n")
    body = body.replace("POOL_P_DEFAULT", "0.5")
    open(f"{SCR}/{name}.py", "w").write(body)
    if SCR not in sys.path:
        sys.path.insert(0, SCR)
    import importlib
    return importlib.import_module(name)


def parse_cmd(stub, cmd):
    toks = cmd.split()
    i = [k for k, t in enumerate(toks) if t.endswith("train.py")][-1]
    return stub.parse_args(toks[i + 1:])


# ------------------------------------------------------------------------------------------------ CPU part
def c0_static():
    P("\n== C0 static")
    rec = {}
    for ln in open(f"{D}/ref/md5.txt"):
        if ln.startswith("frozen"):
            continue
        h, f = ln.split()
        rec[f] = h
    pairs = [("ref/train_w5.py.txt", "lab/nn_x/w5/train.py"), ("ref/lane_w5.sh.txt", "lab/nn_x/w5/lane.sh"),
             ("ref/cd_arms11_w11.py.txt", "lab/nn_x/w11/pseudo/cd_arms11.py")]
    bad = [(a, b) for a, b in pairs if not (md5(f"{D}/{a}") == rec[b] == md5(f"{ROOT}/{b}"))]
    check("(a) frozen copies == recorded md5 == live files (w5 train.py, w5 lane.sh, w11 cd_arms11.py)", not bad, str(bad) if bad else rec["lab/nn_x/w5/train.py"])
    mods = ["model.py", "data.py", "common.py"] + sorted(f for f in os.listdir(W5) if re.fullmatch(r"cd_arms\d*[a-z]?\.py", f))
    drift = [m_ for m_ in mods if rec.get(f"lab/nn_x/w5/{m_}") and md5(f"{W5}/{m_}") != rec[f"lab/nn_x/w5/{m_}"]]
    check("(a) imported w5 modules unchanged since the port was frozen (md5 vs ref/md5.txt)", not drift, f"{len(mods)} modules" + (f"; drifted {drift}" if drift else ""))
    new_src, ref_src = open(f"{D}/train.py").read(), open(f"{D}/ref/train_w5.py.txt").read()
    t_new, removed = strip_pseudo(ast.parse(new_src))
    check("(b) strip_pseudo removed exactly the expected PSEUDO statement set", removed == EXPECT, dict(removed))
    same_ast = ast.dump(strip_doc(t_new)) == ast.dump(strip_doc(ast.parse(ref_src)))
    check("(b) train.py minus the PSEUDO statements == frozen nn_x/w5/train.py (AST, docstrings stripped)", same_ast)
    if not same_ast:
        for i, (x, y) in enumerate(zip(ast.unparse(strip_doc(t_new)).splitlines(), ast.unparse(strip_doc(ast.parse(ref_src))).splitlines())):
            if x != y:
                P(f"     first AST difference at unparsed line {i}: NEW {x[:100]!r} | REF {y[:100]!r}")
                break
    va = [ast.unparse(n) for n in ast.walk(ast.parse(new_src)) if isinstance(n, ast.Assert) and "variant.startswith" in ast.unparse(n)]
    check("(b) the variant-prefix assert is unchanged (w5_ only)", va == ["assert a.variant.startswith('w5_'), 'nn_x/w5 copy: variant names must start with w5_'"], va)
    # (c) textual diff
    a_l, b_l = ref_src.splitlines(), new_src.splitlines()
    doc_end = next(i for i, ln in enumerate(b_l) if i > 0 and ln.rstrip().endswith('"""'))  # last line of the new module docstring
    sm = difflib.SequenceMatcher(a=a_l, b=b_l, autojunk=False)
    removed_l, bad_ins, ins_blocks = [], [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("replace", "delete"):
            removed_l += a_l[i1:i2]
        if op in ("replace", "insert"):
            if j2 - 1 <= doc_end:
                continue
            blk = "\n".join(b_l[j1:j2])
            ins_blocks.append(blk)
            try:
                body = ast.parse(__import__("textwrap").dedent(blk)).body
                ok_ = all("pseudo" in ast.unparse(s_) for s_ in body) and len(body) >= 1
            except SyntaxError:
                ok_ = False
            if not ok_:
                bad_ins.append(blk[:120])
    check("(c) textual diff: the only removed line is the old first docstring line (moved below the new header)",
          removed_l == ['"""Train the neural ranker on the (position-biased) train split, select by full NDCG on val.'], removed_l[:3])
    check("(c) textual diff: every code insertion parses into statements that each name 'pseudo' (6 inserted code blocks: import, flags, setup, loop, summary, save)",
          not bad_ins and len(ins_blocks) == 6, f"{len(ins_blocks)} code blocks" + (f"; bad {bad_ins}" if bad_ins else ""))
    w11 = open(f"{W11}/train.py").read()
    w11_lines = {ln.strip() for ln in w11.splitlines()}
    own = [ln.strip() for blk in ins_blocks for ln in blk.splitlines() if not ln.strip().startswith("#") and ln.strip() not in w11_lines]
    exp_own = ["from cd_pseudokd import setup_pseudo  # pseudokd: wave-11 PSEUDO-DIRECT pseudo-feed co-training on the KD student (OFF by default)",
               "pseudo = None  # pseudokd (cd_pseudokd.py): stays None unless --pseudo_ratio > 0",
               "if pseudo is not None and step < pseudo.t_end:  # pseudokd (cd_pseudokd.py): plateau steps only; loss += w * pseudo CE",
               "if pseudo is not None:  # pseudokd: pseudo-feed statistics since the last eval",
               "if pseudo is not None:  # pseudokd: the context-free head (training-only) and the cumulative statistics"]
    check("(c) every inserted code line is a line of the w11 pseudo train.py except the renamed import and 4 re-commented lines",
          own == exp_own, own)
    # (d) parse defaults
    st_new = _make_parse_stub(f"{D}/train.py", "pk_parse_new")
    st_ref = _make_parse_stub(f"{D}/ref/train_w5.py.txt", "pk_parse_ref")
    dn, dr = vars(st_new.parse_args(["--variant", "w5_x"])), vars(st_ref.parse_args(["--variant", "w5_x"]))
    check("(d) parse(): the 7 pseudo flags at their OFF defaults (ratio 0, gap 16/64, w 0.3, tmax 63, head 1, noset 0)",
          all(dn[k] == v for k, v in PSEUDO_DEFAULTS.items()), {k: dn[k] for k in PSEUDO_DEFAULTS})
    check("(d) parse(): every other default == the frozen w5 parse() defaults (key set = w5 + exactly the 7 pseudo flags)",
          set(dn) == set(dr) | set(PSEUDO_DEFAULTS) and all(dn[k] == dr[k] for k in dr))
    # (e) call sites
    tree = ast.parse(new_src)
    calls = collections.Counter(ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)
                                and ast.unparse(n.func) in ("setup_pseudo", "pseudo.combine", "pseudo.summary", "pseudo.save"))
    check("(e) setup_pseudo / combine / summary / save called exactly once each", all(calls[k] == 1 for k in ("setup_pseudo", "pseudo.combine", "pseudo.summary", "pseudo.save")), dict(calls))
    main = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    mb = main.body
    i_set = [i for i, s in enumerate(mb) if isinstance(s, ast.If) and ast.unparse(s.test) == "a.pseudo_ratio > 0"]
    ok_set = (len(i_set) == 1 and ast.unparse(mb[i_set[0] - 1]) == "pseudo = None" and ast.unparse(mb[i_set[0] - 2]) == "kdc = setup_kdc(a, log)"
              and len(mb[i_set[0]].body) == 1 and ast.unparse(mb[i_set[0]].body[0]) == "pseudo = setup_pseudo(a, model, tr, feeds, hold, opts, t_cool, total, log)")
    check("(e) `kdc = setup_kdc(a, log)` -> `pseudo = None` -> `if a.pseudo_ratio > 0: pseudo = setup_pseudo(...)` at the top level of main()", ok_set)
    loop = [n for n in ast.walk(main) if isinstance(n, ast.For) and ast.unparse(n.target).strip("()") == "step, fb"][0]
    body = loop.body
    i_ps = [i for i, s in enumerate(body) if isinstance(s, ast.If) and ast.unparse(s.test) == "pseudo is not None and step < pseudo.t_end"]
    ok_place = False
    if len(i_ps) == 1:
        i = i_ps[0]
        prev, blk, nxt = body[i - 1], body[i], body[i + 1]
        ok_place = (isinstance(prev, ast.If) and ast.unparse(prev.test) == "rdrop is not None and step >= t_cool" and len(blk.body) == 1
                    and ast.unparse(blk.body[0]) == "loss = pseudo.combine(model, a, b, loss, prand, cliplog)"
                    and isinstance(nxt, ast.For) and "opt.zero_grad" in ast.unparse(nxt))
    gat = [ast.unparse(s) for s in body if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "b"]
    check("(e) the loop block follows the R-Drop block, precedes the zero_grad loop and holds the one combine statement on `b` "
          "(the step's gathered training batch: `b = (tr_hz ... tr).gather(fb, with_pos=True)`; valcd's val batch only in the cooldown)",
          ok_place and len(gat) == 1 and "gather(fb, with_pos=True)" in gat[0], gat)
    others = [f for f in os.listdir(D) if f.endswith((".py", ".sh")) and f not in ("train.py", "cd_pseudokd.py") and "cd_pseudokd" in open(f"{D}/{f}").read()]
    check("(e) nothing else in the folder (outside tests/) imports cd_pseudokd", not others, str(others))
    # (f) helper vs cd_arms11
    h_new, h_old = ast.parse(open(f"{D}/cd_pseudokd.py").read()), ast.parse(open(f"{W11}/cd_arms11.py").read())
    sp = [n for n in h_new.body if isinstance(n, ast.FunctionDef) and n.name == "setup_pseudo"][0]
    new_stmts = [s for s in sp.body if (isinstance(s, ast.Assert) and any(k in ast.unparse(s) for k in ("hist_ext", "aug_all", "wcl", "kdgap", "xfold_unlab")))
                 or (isinstance(s, ast.Expr) and "pseudokd port" in ast.unparse(s))]
    sp.body = [s for s in sp.body if s not in new_stmts]
    check("(f) cd_pseudokd.py == cd_arms11.py (AST, docstrings stripped) after removing exactly 5 new asserts + 1 new log call in setup_pseudo",
          len(new_stmts) == 6 and ast.dump(strip_doc(h_new)) == ast.dump(strip_doc(h_old)), f"{len(new_stmts)} new statements")
    # (g) launcher
    pk, l5 = f"{D}/lane_pk.sh", f"{W5}/lane.sh"
    cmd_pk = lane_dry(pk, "6", "E:kd05w192ps:951 E:kd05w192ps:952")
    cmd_5 = lane_dry(l5, "0", "E:kd05w192:951 E:kd05w192:952", env={"W5_GPUS": "0"})
    ok_tok = True
    for s_, c_pk, c_5 in zip((951, 952), cmd_pk, cmd_5):
        t_pk = c_pk.split(") ", 1)[1].split()
        t_5 = c_5.split()
        exp = [t.replace("train.py", f"{D}/train.py") if t == "train.py" else t.replace(f"w5_e10_kd05w192_s{s_}", f"w5_e10_kd05w192ps_s{s_}") for t in t_5] + ARM_PSEUDO
        ok_tok &= t_pk == exp and c_pk.startswith(f"(cd <repo>/nn_x/w5 && PYTHONPATH=<repo>/nn_x/w5) ")
    check("(g) lane_pk.sh DRY E:kd05w192ps == nn_x/w5 lane.sh DRY E:kd05w192 token for token (trainer, variant) + exactly the 12 pseudo tokens; cwd / PYTHONPATH nn_x/w5", ok_tok)
    exp_later = dict(hist_ext=0, hist_ext_dir="<repo>/artifacts/data/feats_new/histext_1117", hist_ext_var="", hist_ext_uncapped=False,
                     hist_ext_init="row255", xt_drop_init=-1.0, aug_all=False, xfold=-1, xfold_unlab=False, wcl=0.0, ccl=0.0, fsw=0.0,
                     fsw_norm="batch", kdgap=0.0, kdgap_clip="0.25,4", robust=0.0, casc=-1, **PSEUDO_ARM)
    for s_, c_pk in zip((951, 952), cmd_pk):
        e_args = vars(parse_cmd(st_new, c_pk))
        stored = json.load(open(f"{W5}/logs/w5_e10_kd05w192_s{s_}.json"))["args"]
        dif = {k: (e_args.get(k), stored.get(k)) for k in stored if e_args.get(k) != stored.get(k)}
        later = {k: e_args[k] for k in set(e_args) - set(stored)}
        check(f"(g) seed {s_}: parsed launcher args vs the STORED w5_e10_kd05w192_s{s_} args: only the variant differs; flags added to w5 later at their defaults; pseudo flags at the arm's values",
              set(dif) == {"variant"} and later == exp_later, f"{dif} | later-added {sorted(later)}")
    cmd_sm = lane_dry(pk, "7", "E:kd05w192ps:951", env={"PK_SMOKE": "1"})[0]
    t_sm = cmd_sm.split(") ", 1)[1].split()
    exp_sm = ([PY, f"{D}/tests/run_redirect.py", f"{D}/smoke_out", f"{D}/train.py"] + cmd_pk[0].split(") ", 1)[1].split()[2:]
              + "--train_feeds 20000 --epochs 1 --eval_every 25 --select_min_step 25 --cliplog 10".split())
    exp_sm = [t.replace("w5_e10_kd05w192ps_s951", "w5_smoke_e10_kd05w192ps_s951") for t in exp_sm]
    check("(g) PK_SMOKE DRY = the arm command + lane.sh's W5_SMOKE tokens, variant w5_smoke_..., through tests/run_redirect.py into smoke_out/", t_sm == exp_sm, cmd_sm[:160])
    ref_rc = [subprocess.run(["bash", pk, g, j], capture_output=True, text=True, env={**os.environ, "DRY": "1", **e}).returncode
              for g, j, e in (("6", "E:kd05w192:951", {}), ("6", "E:kd05w192ps:950", {}), ("3", "E:kd05w192ps:951", {"PK_SMOKE": "1"}), ("8", "E:kd05w192ps:951", {}))]
    check("(g) lane_pk.sh refuses an unknown arm (E:kd05w192), a seed outside 951-962, a smoke off GPUs N and GPU N", all(r_ != 0 for r_ in ref_rc), ref_rc)
    RES["c0"] = dict(removed=dict(removed), own_lines=own, launch=cmd_pk)


def _mini(n_each, seed, xt=None, with_domain=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # common.py import-time guard only; nothing touches CUDA here
    from mini_split import MiniSplit, pick_train_feeds
    import polars as pl
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ms = MiniSplit("train", pick_train_feeds(n_each=n_each, seed=seed), xt_names=xt, with_domain=with_domain)
    ms.user_id = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["user_id"])["user_id"].to_numpy()[ms.src_feeds]
    return ms


def _key(bp, p):
    return (int(bp["pseudo_src"][p]), int(bp["pseudo_t"][p]), int(bp["pseudo_g"][p]))


def c1_unit():
    P("\n== C1 unit: builder vs the brute-force checkers on real mini batches; locality; loss combine; guards")
    sys.path.insert(0, D)
    if W5 not in sys.path:
        sys.path.insert(1, W5)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import model as Mo
    import cd_pseudokd as C11
    from common import feature_stats  # the import-time CUDA_VISIBLE_DEVICES guard passes here; nothing touches CUDA
    from pseudo_check import check_pseudo_batch, check_pseudo_extra
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    assert os.path.dirname(Mo.__file__) == W5 and os.path.dirname(C11.__file__) == D, (Mo.__file__, C11.__file__)
    XT = ["seq_cv_invd20", "it_hist_all_luo", "has:pre2_fe"]
    ms = _mini(24, 3, xt=XT)
    rng = np.random.default_rng(0)
    batches = ms.batches(np.arange(ms.n), 32768, 512, shuffle=True, rng=rng)
    ok_all = torch.ones(ms.n, dtype=torch.bool)
    stats = collections.defaultdict(list)
    n_viol, n_viol2, n_ps, n_b, n_draw, n_tgt_drop, n_neg_drop = 0, 0, 0, 0, 0, 0, 0
    r2_ok = True
    for R in (1.0, 2.0):
        gen = torch.Generator()
        gen.manual_seed(5)
        for fb in batches[:8]:
            b = ms.gather(fb, with_pos=True)
            cemb = ms.c_emb[b["ci"]].long()
            bp, info = C11.build_pseudo_batch(b, cemb, ok_all[b["fidx"]], R, 16, 64, 63, 256, gen)
            n_b += 1
            n_draw += int(np.ceil(R * len(fb)))
            n_tgt_drop += int(info["n_drop_tgt"])
            n_neg_drop += int(info["n_drop_neg"])
            if bp is None:
                continue
            viol = check_pseudo_batch(b, bp, cemb, ok_all[b["fidx"]], 16, 64, 63)
            viol2 = check_pseudo_extra(b, bp, 256)
            n_viol += len(viol)
            n_viol2 += len(viol2)
            if viol or viol2:
                P(f"     violations R {R} batch: {(viol + viol2)[:4]}")
            vm = bp["pseudo_valid"].numpy()
            n_ps += int(vm.sum())
            r2_ok &= bp["y"].shape[0] == int(np.ceil(R * len(fb)))
            for k_, kk in (("M", "M"), ("H", "H"), ("t", "pseudo_t"), ("g", "pseudo_g")):
                stats[k_].append(bp[kk].numpy()[vm])
            if R == 2.0:
                cnt = np.bincount(bp["pseudo_src"].numpy()[vm], minlength=len(fb))
                r2_ok &= bool((cnt <= 2).all())
    t_all, g_all = np.concatenate(stats["t"]), np.concatenate(stats["g"])
    check(f"builder output passes the verbatim cd_arms11 checker on every pseudo-feed ({n_ps} pseudo-feeds over {n_b} batches at R 1 and 2; "
          f"{n_draw} draws, {n_tgt_drop} dropped for no eligible target, {n_neg_drop} for no negative)", n_viol == 0 and n_ps >= 30, f"{n_viol} violations")
    check("builder output passes the port's locality / padding / key-set / shape checker (hidx' = own history at index >= t + g, gather() "
          "padding and shape conventions, same keys and dtypes as a real batch minus ci / pos)", n_viol2 == 0, f"{n_viol2} violations")
    check("t in [0, 63], g in [16, 64], H' >= 1; R = 2: no source row drawn more than twice; P = ceil(R B) on every batch",
          t_all.min() >= 0 and t_all.max() <= 63 and g_all.min() >= 16 and g_all.max() <= 64 and np.concatenate(stats["H"]).min() >= 1 and r2_ok,
          f"t mean {t_all.mean():.1f}, g mean {g_all.mean():.1f}, M' mean {np.concatenate(stats['M']).mean():.1f}, H' mean {np.concatenate(stats['H']).mean():.1f}")
    # determinism
    b = ms.gather(batches[0], with_pos=True)
    cemb = ms.c_emb[b["ci"]].long()
    g1, g2 = torch.Generator(), torch.Generator()
    g1.manual_seed(9)
    g2.manual_seed(9)
    bp1, _ = C11.build_pseudo_batch(b, cemb, ok_all[b["fidx"]], 1.0, 16, 64, 63, 256, g1)
    bp2, _ = C11.build_pseudo_batch(b, cemb, ok_all[b["fidx"]], 1.0, 16, 64, 63, 256, g2)
    check("the same generator seed gives identical draws and identical pseudo tensors",
          bp1 is not None and all(torch.equal(bp1[k], bp2[k]) for k in bp1))
    # LOCALITY: garbage in every row that is not a kept source -> every pseudo-feed of a kept source is bitwise unchanged
    n_loc, n_cmp, loc_ok = 0, 0, True
    jobs_loc = []
    for bi, fb in enumerate(batches):
        half = torch.zeros(len(fb), dtype=torch.bool)
        half[torch.randperm(len(fb), generator=torch.Generator().manual_seed(bi))[: len(fb) // 2]] = True
        jobs_loc += [(bi, fb, half), (bi, fb, ~half)]  # both complementary halves: every source row is checked once
    for bi, fb, keep_rows in jobs_loc:
        b = ms.gather(fb, with_pos=True)
        cemb = ms.c_emb[b["ci"]].long()
        B = b["cmask"].shape[0]
        gg = torch.Generator()
        gg.manual_seed(100 + bi)
        bpA, _ = C11.build_pseudo_batch(b, cemb, ok_all[b["fidx"]], 1.0, 16, 64, 63, 256, gg)
        if bpA is None or B < 2 or not bool(keep_rows.any()) or bool(keep_rows.all()):
            continue
        b2 = {k: v.clone() if torch.is_tensor(v) else v for k, v in b.items()}
        cemb2 = cemb.clone()
        o = ~keep_rows
        gz = torch.Generator().manual_seed(7 + bi)
        b2["he_raw"][o] = torch.randn(b2["he_raw"][o].shape, generator=gz) * 50
        b2["ce_raw"][o] = torch.randn(b2["ce_raw"][o].shape, generator=gz) * 50
        b2["hidx"][o] = torch.randint(0, 10 ** 6, b2["hidx"][o].shape, generator=gz).to(b2["hidx"].dtype)
        cemb2[o] = torch.randint(0, 10 ** 6, cemb2[o].shape, generator=gz)
        for k_, hi_ in (("hcat", 50), ("ccat", 50), ("het", 5), ("hes", 5)):
            b2[k_][o] = torch.randint(0, hi_, b2[k_][o].shape, generator=gz).to(b2[k_].dtype)
        b2["y"][o] = (torch.rand(b2["y"][o].shape, generator=gz) < 0.5).to(b2["y"].dtype) * b2["cmask"][o]
        b2["ue_raw"][o] = torch.randn(b2["ue_raw"][o].shape, generator=gz) * 50
        b2["cf"][o] = torch.randn(b2["cf"][o].shape, generator=gz).to(b2["cf"].dtype)
        b2["xt"][o] = torch.randn(b2["xt"][o].shape, generator=gz).to(b2["xt"].dtype)
        b2["user0"][o] = ~b2["user0"][o]
        b2["fidx"][o] = b2["fidx"][o] + 10 ** 6
        gg = torch.Generator()
        gg.manual_seed(100 + bi)
        bpB, _ = C11.build_pseudo_batch(b2, cemb2, ok_all[b["fidx"]], 1.0, 16, 64, 63, 256, gg)
        rowsA = collections.defaultdict(list)
        for p in range(bpA["y"].shape[0]):
            if bool(bpA["pseudo_valid"][p]) and bool(keep_rows[int(bpA["pseudo_src"][p])]):
                rowsA[_key(bpA, p)].append(p)
        rowsB = collections.defaultdict(list)
        for p in range(bpB["y"].shape[0]):
            if bool(bpB["pseudo_valid"][p]) and bool(keep_rows[int(bpB["pseudo_src"][p])]):
                rowsB[_key(bpB, p)].append(p)
        loc_ok &= set(rowsA) == set(rowsB)
        for k_ in rowsA:
            for pa, pb in zip(rowsA[k_], rowsB.get(k_, [])):
                n_cmp += 1
                Mp, Hp = int(bpA["M"][pa]), int(bpA["H"][pa])
                same = int(bpB["M"][pb]) == Mp and int(bpB["H"][pb]) == Hp
                for kk in ("ce_raw", "cmiss", "ccat", "cf", "y", "xt"):
                    same &= torch.equal(bpA[kk][pa, :Mp], bpB[kk][pb, :Mp])
                for kk in ("he_raw", "hemb_ok", "hcat", "het", "hes", "hidx"):
                    same &= torch.equal(bpA[kk][pa, :Hp], bpB[kk][pb, :Hp])
                for kk in ("ue_raw", "user0", "fidx", "dom"):
                    same &= torch.equal(bpA[kk][pa], bpB[kk][pb])
                loc_ok &= bool(same)
        n_loc += 1
    check(f"LOCALITY: overwriting every non-source row (history, candidates, clicks, categories, user emb, f1-f5, xt, user0, fidx) leaves "
          f"every pseudo-feed of the kept sources bitwise unchanged ({n_cmp} pseudo-feeds compared over {n_loc} (batch, half) cases; both halves of "
          f"every mini batch): a pseudo-feed "
          f"reads its own source row only (no other feed, no later snapshot of the user)", loc_ok and n_cmp >= 30)
    # blocked rows / holdout users / user 0
    b = ms.gather(batches[0], with_pos=True)
    cemb = ms.c_emb[b["ci"]].long()
    blk = torch.ones(ms.n, dtype=torch.bool)
    blk[b["fidx"][: max(1, len(batches[0]) // 2)]] = False
    g3 = torch.Generator()
    g3.manual_seed(9)
    bp3, info3 = C11.build_pseudo_batch(b, cemb, blk[b["fidx"]], 1.0, 16, 64, 63, 256, g3)
    blocked_rows = set(torch.where(~blk[b["fidx"]])[0].tolist())
    check("blocked source rows (holdout users) yield no pseudo-feed; the blocked count is reported",
          (bp3 is None or not (set(bp3["pseudo_src"][bp3["pseudo_valid"]].tolist()) & blocked_rows)) and int(info3["n_blocked"]) == len(blocked_rows), f"{len(blocked_rows)} blocked rows")
    uid = ms.user_id
    nz = np.where(uid != 0)[0]
    hold = np.array([nz[0], nz[1]]) if len(nz) >= 2 else nz[:1]
    hold = np.concatenate([hold, np.where(uid == 0)[0][:1]]) if (uid == 0).any() else hold
    ok_np = C11.holdout_user_mask(ms, hold)
    hu = set(uid[hold].tolist()) - {0}
    exp_blocked = np.isin(uid, list(hu)) & (uid != 0)
    check("holdout_user_mask blocks exactly the feeds of the holdout feeds' users and never user 0 (user 0 is never linked)",
          np.array_equal(~ok_np, exp_blocked) and ok_np[uid == 0].all() and (uid == 0).any() and (uid[hold] == 0).any(),
          f"{int((~ok_np).sum())} feeds blocked for {len(hu)} holdout users; {int((uid == 0).sum())} user-0 feeds never blocked")
    # loss combine with a small model
    torch.manual_seed(0)
    xt_rel, xt_idcols = [0, 1], [0, 1]
    m = Mo.Ranker(d=32, heads=2, hist_layers=1, blocks=1, use_pos_tower=False, aux_pos=True, sim_bias=1,
                  n_dense=Mo.N_DENSE + 2 + len(XT) + 2 * len(xt_rel), domain=True, xt_names=XT, xt_rel=xt_rel, xt_idcols=xt_idcols,
                  attn_gate=True, split_kv=True, retr_m=16)
    m.eval()
    mean, std = feature_stats(ms, np.arange(ms.n), n_batches=4, dense_fn=m.dense)
    m.set_norm(mean, std)
    m.train()
    opts = [torch.optim.AdamW([{"params": list(m.parameters()), "weight_decay": 0.02}], lr=1e-3)]
    a = types.SimpleNamespace(pseudo_ratio=1.0, pseudo_w=0.3, pseudo_gap_min=16, pseudo_gap_max=64, pseudo_tmax=63, max_hist=256, pseudo_noset=0,
                              pseudo_head=1, seed=23, loss="softmax", loss_norm="feed", max_feeds=512, prand_base=-1.0)
    n_groups0 = len(opts[0].param_groups)
    pt = C11.PseudoTrainer(a, m, ms, torch.ones(ms.n, dtype=torch.bool), opts, t_end=10, head_seed=1)
    check("the head's parameters join opts[0] as one extra no-decay param group (3 zero tensors)",
          len(opts[0].param_groups) == n_groups0 + 1 and opts[0].param_groups[-1]["weight_decay"] == 0.0 and len(opts[0].param_groups[-1]["params"]) == 3
          and all(float(p.detach().abs().sum()) == 0 for p in pt.head.parameters()))
    calls = []
    _ll = C11.listwise_loss

    def ll(logit, y, mask, w=None, norm="batch", denom=None):
        calls.append((w, norm, denom))
        return _ll(logit, y, mask, w, norm=norm, denom=denom)

    C11.listwise_loss = ll
    worst, ok_args, ok_cl, n_c = 0.0, True, True, 0
    for fb in batches[:4]:
        b = ms.gather(fb, with_pos=True)
        for norm in ("feed", "batch"):
            for prand_base, prand in ((-1.0, ms.prand), (0.3, ms.prand), (-1.0, None)):
                a.loss_norm, a.prand_base = norm, prand_base
                real = torch.tensor(1.2345)
                cl = types.SimpleNamespace(main=torch.tensor(0.5))
                calls.clear()
                torch.manual_seed(1)
                pt.gen.manual_seed(77)
                tot = pt.combine(m, a, b, real, prand, cl)
                if int(pt.last_P) == 0:
                    continue
                torch.manual_seed(1)
                pt.gen.manual_seed(77)
                bp, _ = pt.build(b)
                lp, _ = pt.pseudo_loss(m, a, bp, prand)
                exp = real + 0.3 * lp
                worst = max(worst, float(abs(tot - exp) / abs(exp)))
                w_, n_, d_ = calls[0]
                ok_args &= (n_ == norm and d_ == 512 and ((w_ is None) if (prand_base < 0 or prand is None) else bool(torch.equal(w_, torch.full((bp["y"].shape[0],), 1.3)))))
                ok_cl &= bool(abs(float(cl.main) - (0.5 + 0.3 * float(lp))) <= 1e-6 * abs(0.5 + 0.3 * float(lp)))
                n_c += 1
    C11.listwise_loss = _ll
    check(f"combine returns loss + 0.3 x listwise_loss(pseudo) (rel <= 1e-6) over {n_c} cases (norms feed / batch; prand_base -1 / 0.3 / no prand)", worst <= 1e-6 and n_c >= 12, f"worst rel {worst:.1e}")
    check("listwise_loss(pseudo) gets norm == a.loss_norm, denom == a.max_feeds, fw None (prand_base -1 / no prand) or constant prand_base + 1", ok_args)
    check("cliplog.main is advanced by the same 0.3 x pseudo term", ok_cl)
    b = ms.gather(batches[0], with_pos=True)
    a.loss_norm, a.prand_base = "feed", -1.0
    m.eval()
    pt.gen.manual_seed(3)
    bp, _ = pt.build(b)
    _, lg_h = pt.pseudo_loss(m, a, bp, None)
    pt.head, head = None, pt.head
    _, lg_nh = pt.pseudo_loss(m, a, bp, None)
    pt.head = head
    check("the zero-init head leaves the pseudo logits bitwise unchanged at step 0", torch.equal(lg_h, lg_nh))
    pt.noset = True
    _, lg_ns = pt.pseudo_loss(m, a, bp, None)
    pt.noset = False
    check("--pseudo_noset: model.use_set restored after the pseudo forward; the pseudo logits differ", m.use_set and not torch.equal(lg_ns, lg_nh))
    m.train()
    # guards of setup_pseudo (the new w5 options)
    st_new = _make_parse_stub(f"{D}/train.py", "pk_parse_new")
    base = ["--variant", "w5_x", "--pseudo_ratio", "1", "--sched", "wsd"]
    refused = []
    for extra in (["--hist_ext", "128"], ["--aug_all"], ["--wcl", "0.5"], ["--ccl", "0.5"], ["--fsw", "0.25"], ["--kdgap", "0.5"], ["--robust", "0.5"],
                  ["--casc", "5"], ["--xfold", "0", "--xfold_unlab", "--kd_file", "x"]):
        aa = st_new.parse_args(base + extra)
        try:
            C11.setup_pseudo(aa, m, ms, np.arange(ms.n), None, [torch.optim.AdamW(m.parameters())], 10, 20, lambda *x: None)
            refused.append(False)
        except AssertionError:
            refused.append(True)
    aa = st_new.parse_args(base + ["--kd_file", KD, "--kd_w", "0.5"])
    msgs = []
    pt2 = C11.setup_pseudo(aa, m, ms, np.arange(ms.n), None, [torch.optim.AdamW(m.parameters())], 10, 20, lambda *x: msgs.append(" ".join(map(str, x))))
    check("setup_pseudo refuses --hist_ext / --aug_all / --wcl / --ccl / --fsw / --kdgap / --robust / --casc / --xfold_unlab; accepts the KD student "
          "and logs the KD line", all(refused) and pt2 is not None and any("KD mixture, kd_w 0.5" in s_ for s_ in msgs), refused)
    RES["c1"] = dict(n_pseudo=n_ps, n_batches=n_b, n_draws=n_draw, drop_tgt=n_tgt_drop, drop_neg=n_neg_drop, t_mean=float(t_all.mean()),
                     g_mean=float(g_all.mean()), locality_compared=n_cmp, worst_rel=worst)


# ------------------------------------------------------------------------------------------------ C2 CPU e2e
def e2e_run(train_file, out, v, extra):
    rec = f"{SCR}/rec_{v}.npz"
    shutil.rmtree(out, ignore_errors=True)
    cmd = [PY, f"{HERE}/e2e_cpu.py", train_file, out, rec, "--variant", v] + E2E.split() + extra
    t0 = time.time()
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": ""}  # 1 thread: bitwise-reproducible CPU runs
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=SCR, env=env)
    open(f"{SCR}/{v}.stdout", "w").write(r.stdout + "\n" + r.stderr)
    P(f"  ran {v} ({os.path.relpath(train_file, ROOT)}; {time.time() - t0:.0f}s, rc {r.returncode})")
    assert r.returncode == 0, r.stderr[-3000:]
    return dict(np.load(rec, allow_pickle=True))


def metric_lines(log):
    return [re.sub(r" [\d.]+ min", "", ln) for ln in log.splitlines() if ln.startswith("step ") or ln.startswith("BEST")]


def ck_equal(a, b):
    ca = torch.load(a, map_location="cpu", weights_only=False)["state_dict"]
    cb = torch.load(b, map_location="cpu", weights_only=False)["state_dict"]
    return ca.keys() == cb.keys() and all(torch.equal(ca[k], cb[k]) for k in ca)


def c2_e2e():
    P("\n== C2 CPU end-to-end (e2e_cpu.py, miniature split, E recipe at mini scale with the KD teacher, kd_w 0.5, d 192)")
    out = {k: f"{SCR}/out_{k}" for k in ("ref", "off", "on", "on0")}
    names = dict(ref="w5_pk_e2e_ref", off="w5_pk_e2e_off", on="w5_pk_e2e_on", on0="w5_pk_e2e_on_noaux")
    ra = e2e_run(f"{W5}/train.py", out["ref"], names["ref"], [])
    rb = e2e_run(f"{D}/train.py", out["off"], names["off"], [])
    rc_ = e2e_run(f"{D}/train.py", out["on"], names["on"], ARM_PSEUDO)
    rd = e2e_run(f"{D}/train.py", out["on0"], names["on0"], ARM_PSEUDO + ["--aux_pos", "0"])
    lg = {k: open(f"{out[k]}/logs/{names[k]}.log").read() for k in out}
    nb = len(ra["off"]) - 1
    check("teacher logits really used: the --kd_file slice was loaded in every run", all(len(r_["kd_file"]) == 1 for r_ in (ra, rb, rc_, rd)))
    check("OFF vs live w5: identical training-batch sequence (incl. cooldown pool draws)",
          np.array_equal(ra["fidx"], rb["fidx"]) and np.array_equal(ra["off"], rb["off"]) and np.array_equal(ra["ci"], rb["ci"]) and nb >= 20,
          f"{nb} training batches, {int(rb['n_eval'])} holdout evals, {int(rb['n_predict'])} predicts")
    check("OFF vs live w5: bitwise identical train-mode logits, loss-input logits, listwise values / weights / norms, totals; no pseudo forward",
          np.array_equal(ra["lg_model"], rb["lg_model"]) and np.array_equal(ra["lg_loss"], rb["lg_loss"]) and np.array_equal(ra["lw_val"], rb["lw_val"])
          and np.array_equal(ra["lw_w"], rb["lw_w"]) and np.array_equal(ra["lw_norm"], rb["lw_norm"]) and np.array_equal(ra["total"], rb["total"])
          and int(rb["lg_tag"].sum()) == 0 and len(rb["ps_step"]) == 0 and int(rb["n_loss"]) == nb)
    ma, mb = metric_lines(lg["ref"]), metric_lines(lg["off"])
    check("OFF vs live w5: identical logged loss / val / holdout / BEST lines", ma == mb and len(ma) >= 5, mb[-1] if mb else "")
    files = [("ckpt/", ".pt"), ("ckpt/", "@last.pt"), ("ckpt/", "@10.pt"), ("ckpt/", "@20.pt")]
    same_ck = all(ck_equal(f"{out['ref']}/{d_}{names['ref']}{s_}", f"{out['off']}/{d_}{names['off']}{s_}") for d_, s_ in files)
    same_pr = all(np.array_equal(np.load(f"{out['ref']}/{d_}nn_{names['ref']}{s_}"), np.load(f"{out['off']}/{d_}nn_{names['off']}{s_}"))
                  for d_, s_ in (("preds/", "_trainonly_val.npy"), ("preds/", "@last_trainonly_val.npy"), ("preds/holdout/", "_holdout.npy"),
                                 ("preds/holdout/", "@last_holdout.npy")))
    cla, clb = np.load(f"{out['ref']}/logs/{names['ref']}_cliplog.npz"), np.load(f"{out['off']}/logs/{names['off']}_cliplog.npz")
    same_cl = set(cla.files) == set(clb.files) and all(np.array_equal(cla[k], clb[k]) for k in cla.files)
    check("OFF vs live w5: bit-identical selected / @last / @10 / @20 checkpoints, val + holdout preds (selected and @last), cliplog arrays",
          same_ck and same_pr and same_cl, (same_ck, same_pr, same_cl))
    ja, jb = json.load(open(f"{out['ref']}/logs/{names['ref']}.json")), json.load(open(f"{out['off']}/logs/{names['off']}.json"))
    d_args = {k: (ja["args"].get(k), jb["args"].get(k)) for k in set(ja["args"]) | set(jb["args"]) if ja["args"].get(k) != jb["args"].get(k)}
    exp_d = {"variant": (names["ref"], names["off"]), **{k: (None, v) for k, v in PSEUDO_DEFAULTS.items()}}
    check("OFF: stored args == w5 args + the 7 pseudo defaults; same best_step / val_ndcg / history; no pseudo log line; no head file",
          d_args == exp_d and ja["best_step"] == jb["best_step"] and ja["val_ndcg"] == jb["val_ndcg"]
          and [h["val_ndcg"] for h in ja["history"]] == [h["val_ndcg"] for h in jb["history"]]
          and "PSEUDO" not in lg["off"] and "pseudokd" not in lg["off"] and not os.path.exists(f"{out['off']}/ckpt/{names['off']}_pseudo_head.pt"), d_args)
    # ---- ON
    t_cool = int(round(0.5 * nb))
    check("ON: same training-batch sequence and number of steps as OFF; one real forward per step",
          np.array_equal(rc_["fidx"], rb["fidx"]) and np.array_equal(rc_["ci"], rb["ci"]) and len(rc_["total"]) == nb and int((rc_["lg_tag"] == 0).sum()) == nb)
    ps_steps = rc_["ps_step"]
    check(f"ON: combine on EVERY plateau step (0..{t_cool - 1}) and on no cooldown step",
          np.array_equal(ps_steps, np.arange(t_cool)) and int(rc_["lg_tag"].sum()) == int((rc_["ps_P"] > 0).sum()) == int(rc_["lw_tag"].sum()),
          f"{len(ps_steps)} combine calls, {int((rc_['ps_P'] > 0).sum())} with pseudo-feeds, {int(rc_['ps_P'].sum())} pseudo-feeds")
    check("ON: every pseudo-feed of every step passes the verbatim cd_arms11 checker AND the port's locality / padding / key-set checker",
          int(rc_["ps_viol"].sum()) == 0 and int(rc_["ps_viol2"].sum()) == 0 and int(rc_["ps_P"].sum()) >= 20,
          f"M' mean {rc_['ps_M'].mean():.1f}, H' mean {rc_['ps_H'].mean():.1f}, t mean {rc_['ps_t'].mean():.1f}, g mean {rc_['ps_g'].mean():.1f}")
    # holdout users never sources (recompute the holdout exactly as train.py does on the mini split)
    import polars as pl
    sf = rc_["train_src_feeds"]
    Mfull = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["M"])["M"].to_numpy()[sf]
    pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)[sf]
    pr[Mfull < 50] = 0.0
    cand = np.where((pr > 0.9) & (Mfull >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
    uid = rc_["train_user_id"]
    hu = set(uid[hold].tolist()) - {0}
    src_uid = uid[rc_["ps_srcfidx"]]
    trained = set(rc_["fidx"].tolist())
    check("ON: no pseudo source is a holdout feed or a feed of a holdout user (user 0 excepted); every source was a training feed of its step",
          not (set(rc_["ps_srcfidx"].tolist()) & set(hold.tolist())) and not (set(src_uid.tolist()) & hu)
          and set(rc_["ps_srcfidx"].tolist()) <= trained and len(hold) > 0,
          f"{len(hold)} holdout feeds, {len(hu)} holdout users, {len(set(rc_['ps_srcfidx'].tolist()))} distinct sources")
    s0 = int(rc_["lg_model_off"][1])
    check("ON: step-0 real logits identical to OFF; the trained checkpoint differs (mechanism live)",
          np.array_equal(rc_["lg_model"][:s0], rb["lg_model"][:s0]) and rc_["lg_tag"][0] == 0 and not ck_equal(f"{out['on']}/ckpt/{names['on']}.pt", f"{out['off']}/ckpt/{names['off']}.pt"))
    real_calls = rc_["lw_tag"] == 0
    check("ON: the real listwise targets are the KD mixture (soft) on every step, the pseudo listwise targets one-hot clicks",
          int(rc_["lw_onehot"][real_calls].sum()) == 0 and bool((rc_["lw_onehot"][~real_calls] == 1).all()) and (~real_calls).sum() >= 1)

    def split_losses(r):
        real, ps = [], []
        k = 0
        for st in range(len(r["total"])):
            real.append(r["lw_val"][k])
            k += 1
            if st < len(r["ps_P"]) and r["ps_P"][st] > 0:
                ps.append(r["lw_val"][k])
                k += 1
            else:
                ps.append(0.0)
        assert k == len(r["lw_val"]), (k, len(r["lw_val"]))
        return np.array(real), np.array(ps)

    real_c, ps_c = split_losses(rc_)
    aux = rc_["total"] - real_c - 0.3 * ps_c
    check("ON (K9 aux on): total - (real listwise + 0.3 x pseudo listwise) = the aux term > 0 on every step; step-0 real listwise == OFF",
          (aux > 0).all() and real_c[0] == rb["lw_val"][0], f"aux range {aux.min():.4f}..{aux.max():.4f}")
    real_d, ps_d = split_losses(rd)
    exd = real_d + 0.3 * ps_d
    rel = np.abs(rd["total"] - exd) / np.abs(exd)
    check("ON --aux_pos 0: the scalar sent to .backward() == real (KD) listwise + 0.3 x pseudo listwise on every step (rel <= 1e-6); == real alone in the cooldown",
          rel.max() <= 1e-6 and (ps_d[t_cool:] == 0).all() and (ps_d[:t_cool] > 0).sum() >= 1, f"worst rel {rel.max():.1e}; plateau steps with pseudo {(ps_d[:t_cool] > 0).sum()} of {t_cool}")
    jc = json.load(open(f"{out['on']}/logs/{names['on']}.json"))
    hp = f"{out['on']}/ckpt/{names['on']}_pseudo_head.pt"
    check("ON: args carry the arm's pseudo flags; setup / KD / per-eval / final log lines; head file (cum feeds == built); checkpoint cfg == OFF cfg",
          all(jc["args"][k] == v for k, v in PSEUDO_ARM.items()) and "wave-11 PSEUDO-DIRECT: --pseudo_ratio 1.0" in lg["on"]
          and "real feeds keep their target (KD mixture, kd_w 0.5" in lg["on"] and "wave-11 PSEUDO step" in lg["on"] and "wave-11 PSEUDO final" in lg["on"]
          and os.path.exists(hp) and torch.load(hp, map_location="cpu", weights_only=False)["cum"]["feeds"] == int(rc_["ps_P"].sum())
          and torch.load(f"{out['on']}/ckpt/{names['on']}.pt", map_location="cpu", weights_only=False)["cfg"]
          == torch.load(f"{out['off']}/ckpt/{names['off']}.pt", map_location="cpu", weights_only=False)["cfg"],
          f"best_step {jc['best_step']} val {jc['val_ndcg']:.5f}")
    RES["e2e"] = dict(off_lines=mb, on_lines=metric_lines(lg["on"]), on_pseudo=[ln for ln in lg["on"].splitlines() if "PSEUDO" in ln or "pseudokd" in ln][:4],
                      n_steps=nb, t_cool=t_cool, n_pseudo=int(rc_["ps_P"].sum()))
    for d_ in out.values():
        shutil.rmtree(d_, ignore_errors=True)


# ================================================================================================= GPU part
def g1_smoke():
    gpu = _CVD
    v = "w5_smoke_e10_kd05w192ps_s951"
    O = f"{D}/smoke_out"
    P(f"\n== G1 GPU plumbing smoke: PK_SMOKE=1 lane_pk.sh {gpu} E:kd05w192ps:951 -> {v} (20K train feeds, 1 epoch, evals every 25 steps; outputs {O})")
    w5_before = {p_: set(os.listdir(f"{W5}/{p_}")) for p_ in ("logs", "ckpt", "preds", "preds/holdout")}
    t0 = time.time()
    r = subprocess.run(["bash", f"{D}/lane_pk.sh", gpu, "E:kd05w192ps:951"], capture_output=True, text=True,
                       env={**os.environ, "PK_SMOKE": "1", "OMP_NUM_THREADS": str(NT), "WAIT_MAX_H": "1"})
    open(f"{D}/logs/smoke_lane.out", "w").write(r.stdout + "\n" + r.stderr)
    P(f"  lane rc {r.returncode} ({time.time() - t0:.0f}s); tail: " + " | ".join(r.stdout.strip().splitlines()[-3:]))
    fs = [f"logs/{v}.json", f"logs/{v}.log", f"ckpt/{v}.pt", f"ckpt/{v}@last.pt", f"ckpt/{v}_pseudo_head.pt", f"preds/nn_{v}_trainonly_val.npy",
          f"preds/nn_{v}@last_trainonly_val.npy", f"preds/nn_{v}_trainonly_test.npy", f"preds/holdout/nn_{v}_holdout.npy", f"preds/holdout/nn_{v}@last_holdout.npy"]
    ok_files = all(os.path.exists(f"{O}/{f}") for f in fs)
    check("smoke: completed through the lane (guard, lock, /proc-verified PID); json / log / selected + @last + head checkpoints, val / test / holdout preds in smoke_out/",
          r.returncode == 0 and ok_files and "DONE" in r.stdout, [f for f in fs if not os.path.exists(f"{O}/{f}")])
    w5_after = {p_: set(os.listdir(f"{W5}/{p_}")) for p_ in w5_before}
    leaked = [f for p_ in w5_before for f in (w5_after[p_] - w5_before[p_]) if "smoke_e10_kd05w192ps" in f]
    check("smoke: nothing of the smoke written into nn_x/w5/{logs,ckpt,preds}", not leaked, leaked)
    if not ok_files:
        return
    j = json.load(open(f"{O}/logs/{v}.json"))
    lgt = open(f"{O}/logs/{v}.log").read()
    pv = np.load(f"{O}/preds/nn_{v}_trainonly_val.npy")
    total = int(re.search(r"total steps (\d+)", lgt).group(1))
    t_cool = int(round(0.8 * total))
    per = [ln for ln in lgt.splitlines() if "wave-11 PSEUDO step" in ln]
    n_pl = [tuple(int(x) for x in re.search(r": (\d+) steps with pseudo-feeds of (\d+) plateau steps", ln).groups()) for ln in per]
    fin = [ln for ln in lgt.splitlines() if "wave-11 PSEUDO final" in ln]
    n_fin = int(re.search(r"final: (\d+) steps with pseudo-feeds", fin[0]).group(1)) if fin else -1
    check("smoke: pseudo-feeds on every plateau step, none after t_cool (per-eval plateau totals == t_cool == final count); KD + port lines",
          len(per) >= 2 and all(a_ == b_ for a_, b_ in n_pl) and sum(b_ for _, b_ in n_pl) == t_cool and n_fin == t_cool
          and "w12 KD: teacher" in lgt and "real feeds keep their target (KD mixture, kd_w 0.5" in lgt and f"pseudo until step {t_cool}" in per[0],
          f"(pseudo, plateau) per eval {n_pl}; t_cool {t_cool}; final {n_fin}")
    check("smoke: val preds finite with variation; val NDCG > 0.30; holdout evals in the history",
          np.isfinite(pv).all() and pv.std() > 0 and j["val_ndcg"] > 0.30 and len(j["history"]) >= 2,
          f"val {j['val_ndcg']:.5f} best_step {j['best_step']} train minutes {j['train_minutes']:.1f} total steps {total}")
    RES["smoke"] = dict(val=j["val_ndcg"], best_step=j["best_step"], minutes=j["train_minutes"], total=total, pseudo_lines=per[:2] + fin,
                        lines=metric_lines(lgt))
    shutil.copy(f"{O}/logs/{v}.log", f"{D}/logs/smoke_{v}.log")
    shutil.copy(f"{O}/logs/{v}.json", f"{D}/logs/smoke_{v}.json")
    for f in fs + [f"logs/{v}_cliplog.npz", f"preds/nn_{v}@last_trainonly_test.npy"]:
        if os.path.exists(f"{O}/{f}") and not f.endswith((".log", ".json")):
            os.remove(f"{O}/{f}")


if __name__ == "__main__":
    t0 = time.time()
    P(f"test_pseudokd {PART} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} in {D}")
    if PART == "cpu":
        c0_static()
        c1_unit()
        c2_e2e()
    else:
        g1_smoke()
    P(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed in {time.time() - t0:.0f}s" + (f"; FAILED: {FAILS}" if FAILS else ""))
    json.dump(dict(part=PART, passed=N[0] - len(FAILS), total=N[0], fails=FAILS, res=RES), open(f"{D}/logs/test_pseudokd_{PART}.json", "w"), indent=1, default=str)
    sys.exit(1 if FAILS else 0)
