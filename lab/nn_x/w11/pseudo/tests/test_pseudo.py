"""Tests of nn_x/w11/pseudo PSEUDO-DIRECT (pseudo-feed co-training; cd_arms11.py; --pseudo_ratio OFF by default).
Written 2026-11-14 ~00:30 UTC (pattern: nn_x/w10/r3/tests/test_r3.py).

  cpu part:  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w11/pseudo/tests/test_pseudo.py cpu
  gpu part:  cd <repo> && CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=12 .venv/bin/python nn_x/w11/pseudo/tests/test_pseudo.py gpu
             (GPU N = this agent's smoke grant; two W5_SMOKE=1 lane runs (flag OFF / ON, ~3-5 min each) + the step-timing script)

CPU part
  C0 code. (a) module copies byte-identical to nn_x/w5 (md5s recorded at copy time in tests/w5_source_md5.txt, re-checked against the
     live nn_x/w5 files); common.py == nn_x/w5/common.py up to the output-root constant; lane.sh differs only in the documented lines.
     (b) AST proof: train.py minus the PSEUDO statements (tests/strip_pseudo.py, exact set asserted) and minus the variant-prefix assert
     == the frozen nn_x/w5/train.py (tests/ref_w5_train.py.txt; md5 == the live nn_x/w5/train.py) minus its assert, docstrings stripped;
     the new assert accepts exactly the prefixes w5_ / w11_. (c) parse() defaults: pseudo_ratio 0, gap 16 / 64, w 0.3, tmax 63, head 1,
     noset 0; lane DRY: A:pseudo / E:pseudo == A:base / E:base + exactly '--pseudo_ratio 1'; E:pseudo_xpre2 == E:base + '--pseudo_ratio 1
     --xt cvnd'; E:base == nn_x/w5 lane.sh E:base token for token; E:pseudo args vs the stored w5_e10_base_s951 args differ only in
     variant (+ later-added w5 flags at their defaults + the pseudo flags). (d) call sites: setup_pseudo once behind `if a.pseudo_ratio > 0`,
     combine once behind `if pseudo is not None and step < pseudo.t_end` placed right after the R-Drop block and before the zero_grad loop;
     summary / save once each behind `if pseudo is not None`; no other module imports cd_arms11.
  C1 unit (MiniSplit, real train batches, CPU): every pseudo-feed built by cd_arms11.build_pseudo_batch at R = 1 and R = 2 over >= 8
     batches satisfies the brute-force checker tests/pseudo_check.py (target = valid type-1/source-1 event with an embedding at t <= 63,
     new, not a candidate of the source feed; g in [16, 64]; H' = H_cap - t - g >= 1 and the history == the source events at index >= t + g
     re-indexed; slot 0 == the target; negatives == the source feed's non-clicked candidates minus items seen at index < t + g, in order;
     no clicked item of the source feed among the negatives; y one-hot; user_emb / f1-f5 / xt zero; domain row 2); R = 2 gives every
     source row exactly 2 draws; blocked (holdout-user) rows yield no pseudo-feed; holdout_user_mask blocks exactly the holdout users'
     feeds and never user 0; the generator makes the draws reproducible; t within [0, 63] and g within [16, 64] over all draws.
     Loss: with a small Ranker (+ head), PseudoTrainer.combine returns loss + 0.3 * listwise_loss(pseudo) with norm == a.loss_norm and
     denom == a.max_feeds (rel 1e-6), fw None at prand_base -1 and a constant prand_base + 1 vector otherwise; the zero-init head leaves
     the step-0 pseudo logits bitwise unchanged; the head's parameters sit in opts[0] as a no-decay group; --pseudo_noset toggles
     model.use_set only inside the pseudo forward (restored) and changes the pseudo logits; cliplog.main is advanced by the same term.
  C2 CPU end-to-end (tests/e2e_cpu.py, miniature split, 2 mini epochs with cooldown + pool batches, evals every 10 steps, save_last /
     save_holdout / cliplog): frozen nn_x/w5 code vs the new code flag OFF: identical batch sequence, train-mode logits, loss-input
     logits, logged lines, bit-identical selected + @last checkpoints, val and holdout preds, stored args + the 7 pseudo defaults. Flag
     ON (--pseudo_ratio 1): same batch sequence and the same number of training steps; step-0 real logits identical; combine called on
     EVERY plateau step (step < t_cool) and never on a cooldown step; every pseudo-feed of every step passes the brute-force checker;
     the scalar sent to .backward() == real listwise + aux + 0.3 x pseudo listwise; a 4th run (--pseudo_ratio 1 --aux_pos 0): ==
     real + 0.3 x pseudo on every plateau step (rel 1e-6) and == real on cooldown steps; checkpoint cfg identical (no model change);
     ckpt/<v>_pseudo_head.pt written; the wave-11 log lines.
GPU part (GPU N)
  G1 plumbing smoke: W5_SMOKE=1 lane.sh 7 A:base:23 then A:pseudo:23 -> w11_smoke_base_s23 / w11_smoke_pseudo_s23 (20K train feeds,
     1 epoch, evals every 25 steps): both lanes complete; same 'total steps' (the step count is unchanged at R = 1); json / log /
     checkpoints / preds written; the pseudo run logs the wave-11 lines with pseudo-feeds built on the plateau steps and none after
     t_cool; val NDCG > 0.30; the head file exists. Outputs deleted afterwards except logs/smoke_*.log copies.
  G2 step timing (tests/time_steps.py): the real train split on GPU N, the E recipe's model; 40 identical batches timed with
     --pseudo_ratio 0 and 1 (forward + backward + optimiser step, cuda.synchronize): median ms/step, the ratio, the pseudo-feed
     statistics per batch (count, mean M', positives, mean H', t, g, drop shares, eligible share) -> logs/time_steps.json.
Writes nn_x/w11/pseudo/logs/test_pseudo_<part>.log (+ .json).
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
assert PART in ("cpu", "gpu"), "usage: test_pseudo.py cpu|gpu"
_CVD = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
if PART == "cpu":
    assert _CVD == "", "cpu part: run with CUDA_VISIBLE_DEVICES=''"
else:
    assert _CVD == "7", "gpu part: CUDA_VISIBLE_DEVICES=N (this agent's smoke grant)"
NT = min(12, int(os.environ.get("OMP_NUM_THREADS", "12")))
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(NT)
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)  # nn_x/w11/pseudo
ROOT = "<repo>"
W5 = f"{ROOT}/nn_x/w5"
PY = f"{ROOT}/.venv/bin/python"
SCR = f"{D}/logs/test_scratch"
os.makedirs(SCR, exist_ok=True)
sys.path.insert(0, HERE)
from strip_pseudo import EXPECT, FLAGS, strip_pseudo  # noqa: E402

LOG = open(f"{D}/logs/test_pseudo_{PART}.log", "w")
RES, FAILS, N = {}, [], [0]
E2E = ("--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout "
       "--loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --lr 1e-3 --sched wsd --cooldown_frac 0.5 --cooldown_mix 0.3 "
       "--aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 2 --eval_every 10 --no_test --save_last --save_holdout "
       "--select_min_step 10 --cliplog 5 --warmup 2 --cand_budget 4096 --max_feeds 8 --seed 23")
PSEUDO_DEFAULTS = dict(pseudo_ratio=0.0, pseudo_gap_min=16, pseudo_gap_max=64, pseudo_w=0.3, pseudo_tmax=63, pseudo_head=1, pseudo_noset=0)


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
    tree = ast.parse(open(f"{D}/train.py").read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    cd = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("XT_PRESETS", "XT_REL", "XT_RELONLY") for t in n.targets)]
    body = "import argparse\n" + "\n".join(ast.unparse(n) for n in cd) + "\n" + ast.unparse(fn).replace("return p.parse_args()", "return p") \
        + "\n\ndef parse_args(argv):\n    return parse().parse_args(argv)\n"
    body = body.replace("POOL_P_DEFAULT", "0.5")
    open(f"{SCR}/train_parse_stub.py", "w").write(body)


def parse_cmd(cmd):
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
    check("(a) frozen reference tests/ref_w5_train.py.txt == live nn_x/w5/train.py (md5)",
          md5(f"{HERE}/ref_w5_train.py.txt") == rec["train.py"] == md5(f"{W5}/train.py"), rec["train.py"])
    c_new = open(f"{D}/common.py").read().replace('ROOT / "nn_x" / "w11/pseudo"', 'ROOT / "nn_x" / "w5"')
    check("(a) common.py == nn_x/w5/common.py up to the output root (AST, docstrings ignored)",
          ast.dump(strip_doc(ast.parse(c_new))) == ast.dump(strip_doc(ast.parse(open(f"{W5}/common.py").read()))))
    a_ = [ln.replace("<repo>/nn_x/w5", "@W") for ln in open(f"{W5}/lane.sh").read().splitlines()]
    b_ = [ln.replace("<repo>/nn_x/w11/pseudo", "@W") for ln in open(f"{D}/lane.sh").read().splitlines()]
    diff = [ln for ln in difflib.unified_diff(a_, b_, lineterm="", n=0) if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
    exp = {"-ALLOWED=${W5_GPUS:-\"0 1 2 4\"}", "+ALLOWED=${W5_GPUS:-\"7\"}", "-RUN=logs/RUNNING_w5.txt", "-TL=@W/trunc_lifts_mbucket.json",
           "+TL=<repo>/nn_x/w5/trunc_lifts_mbucket.json"}
    allowed = ("A:pseudo|E:pseudo)", "A:pseudo_xpre2|E:pseudo_xpre2)", "A:pseudo_noset|E:pseudo_noset)", "A:pseudo4|E:pseudo4)",
               "RUN=RUNNING.txt", "v=w11_${arm}_s${x}", 'v=w5_${arm}_s${x}', "w5_smoke_")
    unexpected = [ln for ln in diff if ln not in exp and not (ln.startswith("+#") or any(k in ln for k in allowed))]
    check("(a) lane.sh differs from nn_x/w5/lane.sh only in the documented lines", not unexpected and len(diff) >= 10,
          f"{len(diff)} diff lines" + (f"; unexpected {unexpected}" if unexpected else ""))
    new_src, ref_src = open(f"{D}/train.py").read(), open(f"{HERE}/ref_w5_train.py.txt").read()
    t_new, removed = strip_pseudo(ast.parse(new_src))
    check("(b) strip_pseudo removed exactly the expected PSEUDO statement set", removed == EXPECT, dict(removed))
    dv_new, dv_ref = DropVariantAssert(), DropVariantAssert()
    t_new, t_ref = strip_doc(dv_new.visit(t_new)), strip_doc(dv_ref.visit(ast.parse(ref_src)))
    check("(b) train.py minus PSEUDO statements minus the variant assert == frozen nn_x/w5/train.py (AST)", ast.dump(t_new) == ast.dump(t_ref))
    check("(b) exactly one variant assert each; new accepts exactly ('w5_', 'w11_'), ref 'w5_'",
          len(dv_new.dropped) == 1 == len(dv_ref.dropped) and "a.variant.startswith(('w5_', 'w11_'))" in dv_new.dropped[0]
          and "a.variant.startswith('w5_')" in dv_ref.dropped[0], dv_new.dropped[0][:60])
    if ast.dump(t_new) != ast.dump(t_ref):
        for i, (x, y) in enumerate(zip(ast.unparse(t_new).splitlines(), ast.unparse(t_ref).splitlines())):
            if x != y:
                P(f"     first AST difference at unparsed line {i}: NEW {x[:100]!r} | REF {y[:100]!r}")
                break
    _make_parse_stub()
    a_def = vars(parse_cmd("train.py --variant w11_x"))
    check("(c) parse(): pseudo flags at their OFF defaults (ratio 0, gap 16/64, w 0.3, tmax 63, head 1, noset 0)",
          all(a_def[k] == v for k, v in PSEUDO_DEFAULTS.items()), {k: a_def[k] for k in PSEUDO_DEFAULTS})
    lane, lane5 = f"{D}/lane.sh", f"{W5}/lane.sh"
    names = ["A:base", "A:pseudo", "E:base", "E:pseudo", "E:pseudo_xpre2", "E:pseudo_noset"]
    cmds = dict(zip(names, lane_dry(lane, "7", "A:base:23 A:pseudo:23 E:base:951 E:pseudo:951 E:pseudo_xpre2:951 E:pseudo_noset:951")))
    ref5 = dict(zip(["A:base", "E:base"], lane_dry(lane5, "0", "A:base:23 E:base:951", env={"W5_GPUS": "0"})))
    for st in ("A", "E"):
        base_tok = [t.replace("_base_s", "_pseudo_s") if t.startswith(("w11_base_s", "w5_e10_base_s")) else t for t in cmds[f"{st}:base"].split()]
        check(f"(c) lane {st}:pseudo == {st}:base (variant renamed) + exactly '--pseudo_ratio 1'", cmds[f"{st}:pseudo"].split() == base_tok + ["--pseudo_ratio", "1"])
    base_tok = [t.replace("_base_s", "_pseudo_xpre2_s") if t.startswith("w5_e10_base_s") else t for t in cmds["E:base"].split()]
    check("(c) lane E:pseudo_xpre2 == E:base + '--pseudo_ratio 1 --xt cvnd' (the pre2 columns dropped: cvnd_pre2s minus the 6 pre2 columns == cvnd)",
          cmds["E:pseudo_xpre2"].split() == base_tok + ["--pseudo_ratio", "1", "--xt", "cvnd"] and parse_cmd(cmds["E:pseudo_xpre2"]).xt == "cvnd")
    base_tok = [t.replace("_base_s", "_pseudo_noset_s") if t.startswith("w5_e10_base_s") else t for t in cmds["E:base"].split()]
    check("(c) lane E:pseudo_noset == E:base + '--pseudo_ratio 1 --pseudo_noset 1'", cmds["E:pseudo_noset"].split() == base_tok + ["--pseudo_ratio", "1", "--pseudo_noset", "1"])
    check("(c) lane E:base == nn_x/w5 lane.sh E:base token for token", cmds["E:base"].split() == ref5["E:base"].split())
    check("(c) lane A:base == nn_x/w5 lane.sh A:base except the variant name w11_base_s23 vs w5_base_s23",
          cmds["A:base"].split() == [t.replace("w5_base_s23", "w11_base_s23") for t in ref5["A:base"].split()] and "--variant w11_base_s23" in cmds["A:base"])
    e_args = vars(parse_cmd(cmds["E:pseudo"]))
    e_base = json.load(open(f"{W5}/logs/w5_e10_base_s951.json"))["args"]
    dif = {k: (e_args.get(k), e_base.get(k)) for k in e_base if e_args.get(k) != e_base.get(k)}
    later = {k: e_args[k] for k in set(e_args) - set(e_base)}
    exp_later = dict(hzcd=False, val_mix=0.0, val_fold=-1, rdrop_cd=0.0, sam_cd=0.0, trunc_lifts="", **{**PSEUDO_DEFAULTS, "pseudo_ratio": 1.0})
    check("(c) E:pseudo args vs the live w5_e10_base_s951 stored args: differ only in variant; later-added w5 flags at their defaults; pseudo_ratio 1",
          set(dif) == {"variant"} and later == exp_later, f"{dif} | later-added {later}")
    # (d) call sites
    tree = ast.parse(new_src)
    calls = collections.Counter()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            s = ast.unparse(n.func)
            if s in ("setup_pseudo", "pseudo.combine", "pseudo.summary", "pseudo.save"):
                calls[s] += 1
    main = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
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
    check("(d) setup_pseudo / combine / summary / save called exactly once each", all(calls[k] == 1 for k in ("setup_pseudo", "pseudo.combine", "pseudo.summary", "pseudo.save")), dict(calls))
    check("(d) the loop block `if pseudo is not None and step < pseudo.t_end:` follows the R-Drop block, precedes the zero_grad loop, holds the one combine statement", ok_place)
    setup = [n for n in ast.walk(main) if isinstance(n, ast.If) and ast.unparse(n.test) == "a.pseudo_ratio > 0"]
    check("(d) the setup block `if a.pseudo_ratio > 0:` holds exactly `pseudo = setup_pseudo(...)` and follows `pseudo = None`",
          len(setup) == 1 and len(setup[0].body) == 1 and ast.unparse(setup[0].body[0]) == "pseudo = setup_pseudo(a, model, tr, feeds, hold, opts, t_cool, total, log)")
    others = [f for f in os.listdir(D) if f.endswith(".py") and f not in ("train.py", "cd_arms11.py") and "cd_arms11" in open(f"{D}/{f}").read()]
    check("(d) no other module imports cd_arms11", not others, str(others))
    RES["c0"] = dict(removed=dict(removed), lane_diff=diff)


def _mini(n_each, seed, xt=None, with_domain=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # common.py import-time guard only; nothing touches CUDA here
    sys.path.insert(0, D)
    from mini_split import MiniSplit, pick_train_feeds
    import polars as pl
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ms = MiniSplit("train", pick_train_feeds(n_each=n_each, seed=seed), xt_names=xt, with_domain=with_domain)
    ms.user_id = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["user_id"])["user_id"].to_numpy()[ms.src_feeds]
    return ms


def c1_unit():
    P("\n== C1 unit: cd_arms11 builder vs the brute-force checker on real mini batches; the loss combine")
    sys.path.insert(0, D)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import model as Mo
    import cd_arms11 as C11
    from common import feature_stats  # the import-time CUDA_VISIBLE_DEVICES guard passes here; nothing touches CUDA
    from pseudo_check import check_pseudo_batch
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    XT = ["seq_cv_invd20", "it_hist_all_luo", "has:pre2_fe"]
    ms = _mini(24, 3, xt=XT)
    rng = np.random.default_rng(0)
    batches = ms.batches(np.arange(ms.n), 32768, 512, shuffle=True, rng=rng)
    ok_all = torch.ones(ms.n, dtype=torch.bool)
    stats = collections.defaultdict(list)
    n_viol, n_ps, n_b, n_draw, n_tgt_drop, n_neg_drop = 0, 0, 0, 0, 0, 0
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
            n_viol += len(viol)
            if viol:
                P(f"     violations R {R} batch: {viol[:4]}")
            vm = bp["pseudo_valid"].numpy()
            Pn = int(vm.sum())
            n_ps += Pn
            shape_ok = bp["y"].shape[0] == int(np.ceil(R * len(fb))) and bp["cmask"].shape[1] in (b["cmask"].shape[1], b["cmask"].shape[1] + 16) and bp["hmask"].shape[1] % 64 == 0
            r2_ok &= shape_ok
            stats["M"].append(bp["M"].numpy()[vm])
            stats["H"].append(bp["H"].numpy()[vm])
            stats["t"].append(bp["pseudo_t"].numpy()[vm])
            stats["g"].append(bp["pseudo_g"].numpy()[vm])
            if R == 2.0:  # every source row drawn exactly twice (eligible rows may lose draws only through the drops)
                cnt = np.bincount(bp["pseudo_src"].numpy()[vm], minlength=len(fb))
                r2_ok &= bool((cnt <= 2).all())
    t_all, g_all = np.concatenate(stats["t"]), np.concatenate(stats["g"])
    check(f"builder output passes the brute-force checker on every pseudo-feed ({n_ps} pseudo-feeds over {n_b} batches at R 1 and 2; "
          f"{n_draw} draws, {n_tgt_drop} dropped for no eligible target, {n_neg_drop} for no negative)", n_viol == 0 and n_ps >= 30, f"{n_viol} violations")
    check("t in [0, 63] and g in [16, 64] over all draws; H' >= 1; exactly one positive per pseudo-feed",
          t_all.min() >= 0 and t_all.max() <= 63 and g_all.min() >= 16 and g_all.max() <= 64 and np.concatenate(stats["H"]).min() >= 1,
          f"t mean {t_all.mean():.1f} max {t_all.max()}, g mean {g_all.mean():.1f} range {g_all.min()}-{g_all.max()}, M' mean {np.concatenate(stats['M']).mean():.1f}, H' mean {np.concatenate(stats['H']).mean():.1f}")
    check("R = 2: no source row is drawn more than twice; pseudo batch shapes = (ceil(R B), source Mx or Mx + 16, multiple of 64) on every batch", r2_ok)
    # determinism
    gen1, gen2 = torch.Generator(), torch.Generator()
    gen1.manual_seed(9)
    gen2.manual_seed(9)
    b = ms.gather(batches[0], with_pos=True)
    cemb = ms.c_emb[b["ci"]].long()
    bp1, _ = C11.build_pseudo_batch(b, cemb, ok_all[b["fidx"]], 1.0, 16, 64, 63, 256, gen1)
    bp2, _ = C11.build_pseudo_batch(b, cemb, ok_all[b["fidx"]], 1.0, 16, 64, 63, 256, gen2)
    check("the same generator seed gives identical draws (src, t, g) and identical pseudo tensors",
          bp1 is not None and all(torch.equal(bp1[k], bp2[k]) for k in ("pseudo_src", "pseudo_t", "pseudo_g", "pseudo_valid", "ce_raw", "he_raw", "y", "cmask", "hmask")))
    # blocked rows
    blk = torch.ones(ms.n, dtype=torch.bool)
    blk[b["fidx"][: max(1, len(batches[0]) // 2)]] = False
    gen3 = torch.Generator()
    gen3.manual_seed(9)
    bp3, info3 = C11.build_pseudo_batch(b, cemb, blk[b["fidx"]], 1.0, 16, 64, 63, 256, gen3)
    blocked_rows = set(torch.where(~blk[b["fidx"]])[0].tolist())
    check("blocked source rows (holdout users) yield no pseudo-feed; the blocked count is reported",
          (bp3 is None or not (set(bp3["pseudo_src"][bp3["pseudo_valid"]].tolist()) & blocked_rows)) and int(info3["n_blocked"]) == len(blocked_rows), f"{len(blocked_rows)} blocked rows")
    # holdout_user_mask
    uid = ms.user_id
    nz = np.where(uid != 0)[0]
    hold = np.array([nz[0], nz[1]]) if len(nz) >= 2 else nz[:1]
    hold = np.concatenate([hold, np.where(uid == 0)[0][:1]]) if (uid == 0).any() else hold
    ok_np = C11.holdout_user_mask(ms, hold)
    hu = set(uid[hold].tolist()) - {0}
    exp_blocked = np.isin(uid, list(hu)) & (uid != 0)
    check("holdout_user_mask blocks exactly the feeds of the holdout feeds' users and never user 0",
          np.array_equal(~ok_np, exp_blocked) and ok_np[uid == 0].all() and (~ok_np).sum() >= len(hu), f"{int((~ok_np).sum())} feeds blocked for {len(hu)} holdout users")
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
    check("the head's parameters join opts[0] as one extra no-decay param group (3 tensors)",
          len(opts[0].param_groups) == n_groups0 + 1 and opts[0].param_groups[-1]["weight_decay"] == 0.0 and len(opts[0].param_groups[-1]["params"]) == 3
          and all(float(p.abs().sum()) == 0 for p in pt.head.parameters()))
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
    check("listwise_loss(pseudo) receives norm == a.loss_norm, denom == a.max_feeds and fw None (prand_base -1 or no prand) / constant prand_base + 1 (0.3 -> 1.3)", ok_args)
    check("cliplog.main is advanced by the same 0.3 x pseudo term", ok_cl)
    # head zero-init / noset
    b = ms.gather(batches[0], with_pos=True)
    a.loss_norm, a.prand_base = "feed", -1.0
    m.eval()  # no dropout: the two forwards are comparable
    pt.gen.manual_seed(3)
    bp, _ = pt.build(b)
    _, lg_h = pt.pseudo_loss(m, a, bp, None)
    pt.head, head = None, pt.head
    _, lg_nh = pt.pseudo_loss(m, a, bp, None)
    pt.head = head
    check("the zero-init head leaves the pseudo logits bitwise unchanged at step 0", torch.equal(lg_h, lg_nh))
    pt.noset = True
    assert m.use_set
    _, lg_ns = pt.pseudo_loss(m, a, bp, None)
    pt.noset = False
    check("--pseudo_noset: model.use_set restored after the pseudo forward; the pseudo logits differ (set attention skipped)", m.use_set and not torch.equal(lg_ns, lg_nh))
    m.train()
    RES["c1"] = dict(n_pseudo=n_ps, n_batches=n_b, n_draws=n_draw, drop_tgt=n_tgt_drop, drop_neg=n_neg_drop, t_mean=float(t_all.mean()), g_mean=float(g_all.mean()),
                     M_mean=float(np.concatenate(stats["M"]).mean()), H_mean=float(np.concatenate(stats["H"]).mean()), worst_rel=worst)


# ------------------------------------------------------------------------------------------------ C2 CPU e2e
class NormStr(ast.NodeTransformer):
    def visit_Constant(self, n):
        return ast.copy_location(ast.Constant("S"), n) if isinstance(n.value, str) else n


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
    P("\n== C2 CPU end-to-end (e2e_cpu.py, miniature split): flags OFF vs frozen nn_x/w5 code; --pseudo_ratio 1 mechanism")
    ref_dir, ref_out = f"{SCR}/ref_w5_code", f"{SCR}/ref_w5_out"
    new_dir, new_out = f"{SCR}/new_ps_code", f"{SCR}/new_ps_out"
    ok_r = code_copy(W5, ref_dir, ref_out, 'W5 = ROOT / "nn_x" / "w5"')
    ok_n = code_copy(D, new_dir, new_out, 'W5 = ROOT / "nn_x" / "w11/pseudo"')
    check("code copies for the e2e (outputs redirected to scratch): modules byte-identical to nn_x/w5 / nn_x/w11/pseudo, common.py differs only in the output-path string", ok_r and ok_n, (ok_r, ok_n))
    names = ["w5_t11e2e_ref_off", "w5_t11e2e_new_off", "w5_t11e2e_new_on", "w5_t11e2e_new_on_noaux"]
    ra = e2e_run(ref_dir, names[0], [])
    rb = e2e_run(new_dir, names[1], [])
    rc_ = e2e_run(new_dir, names[2], ["--pseudo_ratio", "1"])
    rd = e2e_run(new_dir, names[3], ["--pseudo_ratio", "1", "--aux_pos", "0"])
    la, lb, lc, ld = (open(f"{o}/logs/{n}.log").read() for o, n in ((ref_out, names[0]), (new_out, names[1]), (new_out, names[2]), (new_out, names[3])))
    nb = len(ra["off"]) - 1
    check("flags OFF vs frozen w5: identical training-batch sequence (incl. cooldown pool draws)",
          np.array_equal(ra["fidx"], rb["fidx"]) and np.array_equal(ra["off"], rb["off"]) and np.array_equal(ra["ci"], rb["ci"]) and nb >= 20,
          f"{nb} training batches, {int(rb['n_eval'])} holdout evals, {int(rb['n_predict'])} predicts")
    check("flags OFF vs frozen w5: identical train-mode model logits and loss-input logits (bitwise); listwise w None, norm feed, denom 8 on every call; no pseudo forward",
          np.array_equal(ra["lg_model"], rb["lg_model"]) and np.array_equal(ra["lg_loss"], rb["lg_loss"]) and int(rb["n_loss"]) == int(rb["n_model"]) == nb
          and int(rb["lw_has_w"].sum()) == 0 and all(x == "feed" for x in rb["lw_norm"]) and (rb["lw_denom"] == 8).all() and int(rb["lg_tag"].sum()) == 0
          and len(rb["ps_step"]) == 0 and np.array_equal(ra["lw_val"], rb["lw_val"]) and np.array_equal(ra["total"], rb["total"]))
    ma, mb = metric_lines(la), metric_lines(lb)
    check("flags OFF vs frozen w5: identical logged loss / val / holdout / BEST lines", ma == mb and len(ma) >= 3, mb[-1] if mb else "")
    same_files = (ck_equal(f"{ref_out}/ckpt/{names[0]}.pt", f"{new_out}/ckpt/{names[1]}.pt") and ck_equal(f"{ref_out}/ckpt/{names[0]}@last.pt", f"{new_out}/ckpt/{names[1]}@last.pt")
                  and all(np.array_equal(np.load(f"{ref_out}/{d_}nn_{names[0]}{s_}"), np.load(f"{new_out}/{d_}nn_{names[1]}{s_}"))
                          for d_, s_ in (("preds/", "_trainonly_val.npy"), ("preds/", "@last_trainonly_val.npy"), ("preds/holdout/", "_holdout.npy"),
                                         ("preds/holdout/", "@last_holdout.npy"))))
    check("flags OFF vs frozen w5: bit-identical selected + @last checkpoints, val preds and holdout preds", same_files)
    ja, jb = json.load(open(f"{ref_out}/logs/{names[0]}.json")), json.load(open(f"{new_out}/logs/{names[1]}.json"))
    d_args = {k: (ja["args"].get(k), jb["args"].get(k)) for k in set(ja["args"]) | set(jb["args"]) if ja["args"].get(k) != jb["args"].get(k)}
    exp_d = {"variant": (names[0], names[1]), **{k: (None, v) for k, v in PSEUDO_DEFAULTS.items()}}
    check("flags OFF: stored args == frozen w5 args + the 7 pseudo defaults; same best_step / val_ndcg; no wave-11 log line; no head file",
          d_args == exp_d and ja["best_step"] == jb["best_step"] and ja["val_ndcg"] == jb["val_ndcg"] and "wave-11" not in la and "wave-11" not in lb
          and not os.path.exists(f"{new_out}/ckpt/{names[1]}_pseudo_head.pt"), d_args)
    # flag ON
    t_cool = int(round(0.5 * nb))  # E2E: --cooldown_frac 0.5 (train.py: t_cool = round((1 - frac) * total))
    real_fwd = rc_["lg_tag"] == 0
    check("--pseudo_ratio 1: same training-batch sequence and the same number of training steps as flags OFF; one real forward per step",
          np.array_equal(rc_["fidx"], rb["fidx"]) and np.array_equal(rc_["ci"], rb["ci"]) and len(rc_["total"]) == nb and int(real_fwd.sum()) == nb)
    ps_steps = rc_["ps_step"]
    check(f"--pseudo_ratio 1: combine called on EVERY plateau step (steps 0..{t_cool - 1}) and on no cooldown step",
          np.array_equal(ps_steps, np.arange(t_cool)) and int(rc_["lg_tag"].sum()) == int((rc_["ps_P"] > 0).sum()) and int(rc_["lw_tag"].sum()) == int((rc_["ps_P"] > 0).sum()),
          f"{len(ps_steps)} combine calls, {int((rc_['ps_P'] > 0).sum())} with pseudo-feeds, {int(rc_['ps_P'].sum())} pseudo-feeds in total")
    check("--pseudo_ratio 1: every pseudo-feed of every step passes the brute-force checker (construction invariants)", int(rc_["ps_viol"].sum()) == 0 and int(rc_["ps_P"].sum()) >= 20,
          f"M' mean {rc_['ps_M'].mean():.1f}, H' mean {rc_['ps_H'].mean():.1f}, t mean {rc_['ps_t'].mean():.1f}, g mean {rc_['ps_g'].mean():.1f}")
    s0 = int(rc_["lg_model_off"][1])
    check("--pseudo_ratio 1: step-0 real logits identical to flags OFF (same init / batch); later real logits and the checkpoint differ (mechanism live)",
          np.array_equal(rc_["lg_model"][:s0], rb["lg_model"][:s0]) and rc_["lg_tag"][0] == 0 and not ck_equal(f"{new_out}/ckpt/{names[2]}.pt", f"{new_out}/ckpt/{names[1]}.pt"))

    def split_losses(r):
        """per training step: (real listwise, pseudo listwise or 0) from the ordered listwise_loss calls (real first, then pseudo if built)."""
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
    check("--pseudo_ratio 1 (with the K9 aux): total - (real listwise + 0.3 x pseudo listwise) = the aux term > 0 on every step; step-0 real listwise == flags OFF",
          (aux > 0).all() and real_c[0] == rb["lw_val"][0], f"aux range {aux.min():.4f}..{aux.max():.4f}")
    real_d, ps_d = split_losses(rd)
    exd = real_d + 0.3 * ps_d
    rel = np.abs(rd["total"] - exd) / np.abs(exd)
    check("--pseudo_ratio 1 --aux_pos 0: the scalar sent to .backward() == real listwise + 0.3 x pseudo listwise on every step (rel <= 1e-6); == real alone on cooldown steps",
          rel.max() <= 1e-6 and (ps_d[t_cool:] == 0).all() and (ps_d[:t_cool] > 0).sum() >= 1, f"worst rel {rel.max():.1e}; plateau steps with pseudo {(ps_d[:t_cool] > 0).sum()} of {t_cool}")
    jc = json.load(open(f"{new_out}/logs/{names[2]}.json"))
    hp = f"{new_out}/ckpt/{names[2]}_pseudo_head.pt"
    check("--pseudo_ratio 1: args pseudo_ratio 1, wave-11 setup + per-eval + final log lines, head file written (cum stats), checkpoint cfg == flags-off cfg (no model change)",
          jc["args"]["pseudo_ratio"] == 1.0 and "wave-11 PSEUDO-DIRECT: --pseudo_ratio 1.0" in lc and "wave-11 PSEUDO step" in lc and "wave-11 PSEUDO final" in lc
          and os.path.exists(hp) and torch.load(hp, map_location="cpu", weights_only=False)["cum"]["feeds"] == int(rc_["ps_P"].sum())
          and torch.load(f"{new_out}/ckpt/{names[2]}.pt", map_location="cpu", weights_only=False)["cfg"] == torch.load(f"{new_out}/ckpt/{names[1]}.pt", map_location="cpu", weights_only=False)["cfg"],
          f"best_step {jc['best_step']} val {jc['val_ndcg']:.5f}")
    RES["e2e"] = dict(off_lines=mb, on_lines=metric_lines(lc), on_wave11=[ln for ln in lc.splitlines() if "wave-11" in ln][:3], n_steps=nb, t_cool=t_cool,
                      n_pseudo=int(rc_["ps_P"].sum()), M_mean=float(rc_["ps_M"].mean()), H_mean=float(rc_["ps_H"].mean()))
    for d_ in (ref_dir, ref_out, new_dir, new_out):
        shutil.rmtree(d_, ignore_errors=True)


# ================================================================================================= GPU part
def outputs(base, v):
    fs = [f"{base}/logs/{v}.log", f"{base}/logs/{v}.json", f"{base}/logs/{v}_cliplog.npz", f"{base}/logs/{v}.stdout", f"{base}/logs/{v}_holdout.stdout",
          f"{base}/ckpt/{v}_pseudo_head.pt"]
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


def g1_smoke():
    gpu = _CVD
    out = {}
    for arm in ("base", "pseudo"):
        v = f"w11_smoke_{arm}_s23"
        P(f"\n== G1 GPU plumbing smoke: W5_SMOKE=1 lane.sh {gpu} A:{arm}:23 -> {v} (20K train feeds, 1 epoch, evals every 25 steps)")
        clean(D, v)
        t0 = time.time()
        r = subprocess.run(["bash", f"{D}/lane.sh", gpu, f"A:{arm}:23"], capture_output=True, text=True, cwd=D,
                           env={**os.environ, "W5_SMOKE": "1", "OMP_NUM_THREADS": str(NT), "WAIT_MAX_H": "1"})
        P(f"  lane rc {r.returncode} ({time.time() - t0:.0f}s); tail: " + " | ".join(r.stdout.strip().splitlines()[-3:]))
        ok_files = all(os.path.exists(f"{D}/{f}") for f in (f"logs/{v}.json", f"logs/{v}.log", f"ckpt/{v}.pt", f"ckpt/{v}@last.pt",
                                                           f"preds/nn_{v}_trainonly_val.npy", f"preds/nn_{v}@last_trainonly_val.npy",
                                                           f"preds/holdout/nn_{v}_holdout.npy", f"preds/holdout/nn_{v}@last_holdout.npy"))
        check(f"smoke {arm}: run completed through the lane; json / log / selected + @last checkpoints, val and holdout preds written", r.returncode == 0 and ok_files)
        if not ok_files:
            continue
        j = json.load(open(f"{D}/logs/{v}.json"))
        lg = open(f"{D}/logs/{v}.log").read()
        pv = np.load(f"{D}/preds/nn_{v}_trainonly_val.npy")
        total = int(re.search(r"total steps (\d+)", lg).group(1))
        out[arm] = dict(val=j["val_ndcg"], best_step=j["best_step"], minutes=j["train_minutes"], total=total, lines=metric_lines(lg)[-2:],
                        wave11=[ln for ln in lg.splitlines() if "wave-11" in ln])
        check(f"smoke {arm}: val preds finite with variation, val NDCG > 0.30 after 1 mini epoch, history has holdout evals",
              np.isfinite(pv).all() and pv.std() > 0 and j["val_ndcg"] > 0.30 and len(j["history"]) >= 2, f"val {j['val_ndcg']:.5f} best_step {j['best_step']} minutes {j['train_minutes']:.1f} total steps {total}")
        if arm == "pseudo":
            w11 = out[arm]["wave11"]
            per_eval = [ln for ln in w11 if "wave-11 PSEUDO step" in ln]
            n_feeds = [int(re.search(r"; (\d+) pseudo-feeds", ln).group(1)) for ln in per_eval]
            n_pl = [tuple(int(x) for x in re.search(r": (\d+) steps with pseudo-feeds of (\d+) plateau steps", ln).groups()) for ln in per_eval]
            fin = [ln for ln in w11 if "wave-11 PSEUDO final" in ln]
            n_fin = int(re.search(r"final: (\d+) steps with pseudo-feeds", fin[0]).group(1)) if fin else -1
            t_cool = int(round(0.8 * total))  # the smoke's --cooldown_frac 0.2 (pseudo on steps < t_cool only)
            check("smoke pseudo: args pseudo_ratio 1; wave-11 setup line; per-eval lines: pseudo-feeds built on every plateau step, the plateau-step total "
                  "== t_cool (none in the cooldown); final line == t_cool steps; head file exists",
                  j["args"]["pseudo_ratio"] == 1.0 and any("wave-11 PSEUDO-DIRECT: --pseudo_ratio 1.0" in ln for ln in w11) and len(per_eval) >= 2 and n_feeds[0] > 0
                  and all(a_ == b_ for a_, b_ in n_pl) and sum(b_ for _, b_ in n_pl) == t_cool and n_fin == t_cool
                  and f"pseudo until step {t_cool}" in per_eval[0] and os.path.exists(f"{D}/ckpt/{v}_pseudo_head.pt"),
                  f"pseudo-feeds per eval {n_feeds}; (pseudo steps, plateau steps) per eval {n_pl}; t_cool {t_cool}; final {n_fin}")
            check("smoke pseudo: checkpoint cfg keys == the trained nn_x/w5 base's (no model change)",
                  set(torch.load(f"{D}/ckpt/{v}.pt", map_location="cpu", weights_only=False)["cfg"]) == set(torch.load(f"{W5}/ckpt/w5_e10_base_s951.pt", map_location="cpu", weights_only=False)["cfg"]))
        shutil.copy(f"{D}/logs/{v}.log", f"{D}/logs/smoke_{arm}.log")
        clean(D, v)
    if "base" in out and "pseudo" in out:
        check("smoke: the step count is unchanged at R = 1 (same 'total steps' for base and pseudo)", out["base"]["total"] == out["pseudo"]["total"],
              f"{out['base']['total']} vs {out['pseudo']['total']}; wall {out['base']['minutes']:.1f} vs {out['pseudo']['minutes']:.1f} min (incl. 25-step evals)")
    RES["smoke"] = out


def g2_timing():
    P("\n== G2 step timing (tests/time_steps.py on the real train split): --pseudo_ratio 0 vs 1 on identical batches")
    t0 = time.time()
    r = subprocess.run([PY, f"{HERE}/time_steps.py", "40"], capture_output=True, text=True, cwd=D, env={**os.environ, "OMP_NUM_THREADS": str(NT)})
    P(f"  rc {r.returncode} ({time.time() - t0:.0f}s)")
    P("  " + "\n  ".join(r.stdout.strip().splitlines()[-12:]))
    if r.returncode != 0:
        P(r.stderr[-3000:])
    ok = r.returncode == 0 and os.path.exists(f"{D}/logs/time_steps.json")
    check("timing script completed and wrote logs/time_steps.json", ok)
    if ok:
        tj = json.load(open(f"{D}/logs/time_steps.json"))
        check("timing: both modes finite; ratio > 1 (the pseudo pass costs time) and < 3", np.isfinite(tj["ratio"]) and 1.0 < tj["ratio"] < 3.0,
              f"ms/step off {tj['off_ms_median']:.1f} on {tj['on_ms_median']:.1f} ratio {tj['ratio']:.2f}; pseudo-feeds per batch {tj['pseudo']['per_batch_mean']:.1f} of {tj['pseudo']['B_mean']:.1f} feeds")
        RES["timing"] = tj


if __name__ == "__main__":
    t0 = time.time()
    P(f"test_pseudo {PART} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} in {D}")
    if PART == "cpu":
        c0_code()
        c1_unit()
        c2_e2e()
    else:
        g1_smoke()
        g2_timing()
    P(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed in {time.time() - t0:.0f}s" + (f"; FAILED: {FAILS}" if FAILS else ""))
    json.dump(dict(part=PART, passed=N[0] - len(FAILS), total=N[0], fails=FAILS, res=RES), open(f"{D}/logs/test_pseudo_{PART}.json", "w"), indent=1, default=str)
    sys.exit(1 if FAILS else 0)
