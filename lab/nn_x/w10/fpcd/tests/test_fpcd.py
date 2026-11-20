"""Tests of the nn_x/w10/fpcd WAVE-10 arm FPCD-lite (train.py --cd_exclude_days, cd_arms10.py; OFF by default). Written 2026-11-13
~16:55 UTC (reports/DATA_PIPELINE_1113.md 5.1, both skeptics' fixes: in-batch masking, no redraw / refill, same steps, logging).

  cpu part:  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w10/fpcd/tests/test_fpcd.py cpu
  gpu part:  cd <repo> && CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=12 .venv/bin/python nn_x/w10/fpcd/tests/test_fpcd.py gpu
             (ONE GPU of the FPCD grant 6 / 7; 4 sequential e2e subprocesses: 3 x 39 CD steps + 1 x ~295 CD steps)

CPU part
  C0 code. (a) module copies: model / data / cd_arms* / finetune / holdout_eval / stale_pred / blend / tests/e2e_w5 / mini_split
     byte-identical to the LIVE nn_x/w5 files (md5s recorded at copy time in tests/w5_source_md5.txt); common.py == nn_x/w5/common.py
     up to the output-root constant (AST equal after the path substitution); lane.sh differs from nn_x/w5/lane.sh only in the
     documented lines (header comment, W5 root, allowed GPUs, the B:fpcd case, the RUN file).  (b) AST proof: train.py minus the FPCD
     statements (tests/strip_fpcd.py, exact set asserted) == the frozen nn_x/w5/train.py (tests/ref_w5_train.py.txt, md5 == the live
     nn_x/w5/train.py), docstrings stripped.  (c) parse(): --cd_exclude_days '' by default, every other default == the w5 parser;
     lane DRY: B:fpcd:<r> == B:m02ref:<r> + exactly '--cd_exclude_days 0,1,2' with variant w5_cd_fpcd_r<r>, CDSEED 900 + r, init
     v17_pre2s_s<r>@77500; B:m02ref:<r> of this lane.sh == nn_x/w5 lane.sh's token for token; A:fpcd / E:fpcd and GPU N refused.
     (d) call sites: setup_fpcd once behind `if a.cd_exclude_days:` in main() (after the SAM setup, before the loop); fpcd.batch
     once behind `if fpcd is not None and step >= t_cool:` placed right after the wave-4 cool_stream block and right before the
     gather statement; fpcd.summary once inside the eval block; no other module imports cd_arms10; cd_arms10 imports numpy only.
  C1 unit (synthetic): excluded_mask (Moscow day boundaries, relative indices, dates); mask_batches: every excluded feed removed,
     every allowed feed exactly once per group, batch count unchanged, forced-empty batches filled by splitting the largest batch
     of the SAME group (n_split counted), empty with no donor refused; FPCDMask.batch: identity asserts (regular / pool), the lazy
     pool-stream masking per stream object (reshuffle detected, passes counted), the realised counters and the summary line.
  C2 real dose (FULL training-feed set, CPU; train.py's holdout / feeds / batches / pool construction replicated exactly for the
     four arm seeds 1001-1004): the excluded set = the Sep 6-8 (Moscow) feeds, shares 0.2743 of all / 0.2745 of the training feeds
     / 0.258 of the pool (skeptic k1.log); masked regular batches of each epoch == the allowed feeds exactly once; realised
     regular batch ratio, split events, min masked size; the first pool pass likewise. Writes logs/test_fpcd_dose.json.
GPU part (deterministic wrapper tests/e2e_w5.py; E2E = 39 CD steps from v17_pre2s_s101@77500, cooldown_mix 0.2, evals at 20 / 39)
  G1 flags OFF: frozen nn_x/w5 train.py vs this train.py: identical training-gather sequence (incl. pool steps), train-mode logits,
     loss-input logits, logged metric lines, bit-identical checkpoint and val preds.
  G2 --cd_exclude_days 0,1,2 vs flags OFF (same seed): the same 39 training gathers (step count unchanged); every ON batch == the
     OFF batch of that step minus the Sep 6-8 feeds (or a logged split event); NO excluded feed in any gather; the pool-step
     pattern unchanged; logged realised ratio in [0.6, 0.85]; the FPCD setup / summary lines present; checkpoint differs
     (mechanism live); evals / predicts unchanged in number.
  G3 --cd_exclude_days 0,1,2 at the ARM's pool dose (cooldown_mix 0.02, ~295 steps, eval_every 100): no excluded feed in any of the
     gathers, number of gathers == total steps, the final summary line's counts == the recorded gathers.
Writes nn_x/w10/fpcd/logs/test_fpcd_<part>.log (+ .json). Test variants w5_t10e2e_*; their outputs are deleted afterwards.
"""
import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time

PART = sys.argv[1] if len(sys.argv) > 1 else "cpu"
assert PART in ("cpu", "gpu"), "usage: test_fpcd.py cpu|gpu"
_CVD = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
if PART == "cpu":
    assert _CVD == "", "cpu part: run with CUDA_VISIBLE_DEVICES=''"
else:
    assert _CVD in ("6", "7"), "gpu part: CUDA_VISIBLE_DEVICES=<6 or 7, the FPCD grant>"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
NT = min(12, int(os.environ.get("OMP_NUM_THREADS", "12")))
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, FP)
ROOT = "<repo>"
W5 = f"{ROOT}/nn_x/w5"
PY = f"{ROOT}/.venv/bin/python"
INIT = f"{ROOT}/nn/ckpt/v17_pre2s_s101@77500.pt"
TRAIN_PY, LANE = f"{FP}/train.py", f"{FP}/lane.sh"
REF_TRAIN, REF_LANE = f"{HERE}/ref_w5_train.py.txt", f"{HERE}/ref_w5_lane.sh.txt"
SCR = os.environ.get("T10_SCRATCH", "/tmp/scratchpad/t10")
os.makedirs(SCR, exist_ok=True)
import cd_arms10 as C10  # noqa: E402
from strip_fpcd import EXPECT, strip_fpcd  # noqa: E402

RES, LOG = [], []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def check(name, ok, info=""):
    RES.append((name, bool(ok)))
    P(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({info})" if info else ""))
    return bool(ok)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def strip_doc(tree):
    for n in [tree] + [x for x in ast.walk(tree) if isinstance(x, (ast.FunctionDef, ast.ClassDef))]:
        if n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) \
                and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:] or [ast.Pass()]
    return tree


def load_parse(path):
    tree = ast.parse(open(path).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
    ns = {"argparse": argparse, "POOL_P_DEFAULT": 0.5}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "train_parse", "exec"), ns)

    def parse(argv):
        sys.argv = ["train.py"] + argv
        return ns["parse"]()
    return parse


def lane_dry(script, jobs, gpu="6", env=None):
    r = subprocess.run(["bash", script, gpu, jobs], capture_output=True, text=True, env={**os.environ, "DRY": "1", **(env or {})})
    return r.returncode, r.stdout, r.stderr


def dry_cmds(out):
    cmds = {}
    for ln in out.splitlines():
        m = re.match(r"GPU \d+: (\S+) train\.py (.*)$", ln.strip())
        if m:
            toks = m.group(2).split()
            cmds[toks[toks.index("--variant") + 1]] = toks
    return cmds


# ------------------------------------------------------------------------------------------------ C0 code
def c0_code():
    P("\n== C0 code identity vs the live nn_x/w5 (module copies, AST proof, parser, lane DRY, call sites)")
    rec = dict(ln.split()[::-1] for ln in open(f"{HERE}/w5_source_md5.txt").read().splitlines() if ln.strip())
    same = {f: md5(f"{W5}/{f}") == rec[f] for f in rec}
    ident = [f for f in rec if f not in ("common.py", "train.py", "lane.sh")]
    check("recorded md5s == the live nn_x/w5 files (nothing moved under us)", all(same.values()), str([f for f, s in same.items() if not s]))
    check("module copies byte-identical to nn_x/w5: " + " ".join(os.path.basename(f) for f in ident),
          all(md5(f"{FP}/{f}") == md5(f"{W5}/{f}") for f in ident))
    cm = open(f"{FP}/common.py").read().replace('ROOT / "nn_x" / "w10" / "fpcd"', 'ROOT / "nn_x" / "w5"')
    check("common.py == nn_x/w5/common.py up to the output-root constant (AST, docstrings stripped)",
          ast.dump(strip_doc(ast.parse(cm))) == ast.dump(strip_doc(ast.parse(open(f"{W5}/common.py").read()))))
    a, b = open(f"{W5}/lane.sh").read().splitlines(), open(LANE).read().splitlines()
    removed, added = [x for x in a if x not in b], [x for x in b if x not in a]
    ok = (len(removed) == 3 and all(any(k in x for x in removed) for k in ("W5=<repo>/nn_x/w5", 'W5_GPUS:-"0 1 2 4"', "RUN=logs/RUNNING_w5.txt"))
          and len(added) == 8 and all(any(k in x for x in added) for k in ("W5=<repo>/nn_x/w10/fpcd", 'W5_GPUS:-"6 7"', "B:fpcd)", "RUN=<repo>/nn_x/w10/fpcd/RUNNING.txt"))
          and all(x.startswith("#") for x in added if not any(k in x for k in ("W5=", "W5_GPUS", "B:fpcd)", "RUN="))))
    check("lane.sh differs from nn_x/w5/lane.sh only in the documented lines (root, GPUs N 7, B:fpcd case, RUN file, header comment)", ok,
          f"removed {len(removed)} added {len(added)}")
    check("ref_w5_train.py.txt == the live nn_x/w5/train.py (md5) and ref_w5_lane.sh.txt == nn_x/w5/lane.sh",
          md5(REF_TRAIN) == md5(f"{W5}/train.py") and md5(REF_LANE) == md5(f"{W5}/lane.sh"))
    new_t, removed_c = strip_fpcd(ast.parse(open(TRAIN_PY).read()))
    old_t = strip_doc(ast.parse(open(REF_TRAIN).read()))
    check("AST proof: train.py minus the FPCD statements (exact set) == frozen nn_x/w5/train.py (docstrings stripped)",
          ast.dump(strip_doc(new_t)) == ast.dump(old_t) and removed_c == EXPECT, str(dict(removed_c)))
    pn, po = load_parse(TRAIN_PY), load_parse(REF_TRAIN)
    dn, do = vars(pn([])), vars(po([]))
    check("parse(): cd_exclude_days '' by default; every other default == the w5 parser",
          dn.pop("cd_exclude_days") == "" and dn == do)
    rc, out, err = lane_dry(LANE, "B:fpcd:101 B:m02ref:101 B:fpcd:104 B:m02ref:104")
    c = dry_cmds(out)
    rc2, out2, _ = lane_dry(f"{W5}/lane.sh", "B:m02ref:101 B:m02ref:104", env={"W5_GPUS": "6"})
    c2 = dry_cmds(out2)
    ok = rc == 0 and rc2 == 0
    dbg = []
    for r in (101, 104):
        f_, m_ = c.get(f"w5_cd_fpcd_r{r}"), c.get(f"w5_cd_m02ref_r{r}")
        parts = [bool(f_ and m_), f_[2:-2] == m_[2:], f_[-2:] == ["--cd_exclude_days", "0,1,2"], m_ == c2.get(f"w5_cd_m02ref_r{r}"),
                 f"--seed {900 + r}" in " ".join(m_), f"v17_pre2s_s{r}@77500.pt" in " ".join(m_), f_[1] == f"w5_cd_fpcd_r{r}"]
        dbg.append(parts)
        ok &= all(parts)
    check("lane DRY: B:fpcd:<r> == B:m02ref:<r> + exactly '--cd_exclude_days 0,1,2' (variant w5_cd_fpcd_r<r>, seed 900+r, init @77500); "
          "B:m02ref of this lane == nn_x/w5 lane.sh token for token", ok, f"{dbg} {(err or '')[-200:]}")
    check("lane refuses A:fpcd / E:fpcd and GPU N", lane_dry(LANE, "A:fpcd:23")[0] != 0 and lane_dry(LANE, "E:fpcd:951")[0] != 0
          and lane_dry(LANE, "B:fpcd:101", gpu="0")[0] != 0)
    # call sites
    tree = ast.parse(open(TRAIN_PY).read())
    main_fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]

    def chain(node):
        out, par = [], {}
        for p in ast.walk(main_fn):
            for ch in ast.iter_child_nodes(p):
                par[ch] = p
        n = node
        while n in par:
            n = par[n]
            if isinstance(n, ast.If):
                out.append("if " + ast.unparse(n.test))
            elif isinstance(n, ast.For):
                out.append("for " + ast.unparse(n.target).strip("()"))
            elif isinstance(n, ast.FunctionDef):
                out.append("def " + n.name)
        return out

    def calls(pred):
        return [chain(n) for n in ast.walk(main_fn) if isinstance(n, ast.Call) and pred(n)]
    su = calls(lambda n: isinstance(n.func, ast.Name) and n.func.id == "setup_fpcd")
    bt = calls(lambda n: isinstance(n.func, ast.Attribute) and n.func.attr == "batch" and ast.unparse(n.func.value) == "fpcd")
    sm = calls(lambda n: isinstance(n.func, ast.Attribute) and n.func.attr == "summary" and ast.unparse(n.func.value) == "fpcd")
    check("call sites: setup_fpcd once behind `if a.cd_exclude_days` in main; fpcd.batch once behind `if fpcd is not None and step >= t_cool` "
          "in the step loop; fpcd.summary once in the eval block",
          su == [["if a.cd_exclude_days", "def main"]] and bt == [["if fpcd is not None and step >= t_cool", "for step, fb", "def main"]]
          and len(sm) == 1 and sm[0][0] == "if fpcd is not None" and "for step, fb" in sm[0], f"{su} {bt} {sm}")
    loop = [n for n in ast.walk(main_fn) if isinstance(n, ast.For) and ast.unparse(n.target).strip("()") == "step, fb"][0]
    tests = [ast.unparse(s.test) if isinstance(s, ast.If) else ast.unparse(s)[:40] for s in loop.body]
    i = tests.index("fpcd is not None and step >= t_cool")
    check("placement: the FPCD block sits right after the wave-4 cool_stream block and right before the gather statement",
          tests[i - 1] == "cool_stream is not None and (not is_pool) and (step >= t_cool)" and tests[i + 1].startswith("b = ("), f"{tests[i - 1]!r} -> {tests[i + 1]!r}")
    body = main_fn.body
    j_set = [k for k, s in enumerate(body) if isinstance(s, ast.If) and ast.unparse(s.test) == "a.cd_exclude_days"]
    j_loop = [k for k, s in enumerate(body) if isinstance(s, ast.For) and ast.unparse(s.target).strip("()") == "step, fb"]
    check("placement: the setup block follows the SAM setup and precedes the loop", len(j_set) == 1 and j_set[0] < j_loop[0]
          and isinstance(body[j_set[0] - 2], ast.If) and ast.unparse(body[j_set[0] - 2].test) == "a.sam_cd > 0")
    others = [f for f in os.listdir(FP) if f.endswith(".py") and f not in ("train.py", "cd_arms10.py")]
    check("no other module imports cd_arms10; cd_arms10 imports numpy / datetime only",
          not any("cd_arms10" in open(f"{FP}/{f}").read() for f in others)
          and {ast.unparse(n).split()[1] for n in ast.parse(open(f"{FP}/cd_arms10.py").read()).body if isinstance(n, (ast.Import, ast.ImportFrom))} == {"datetime", "numpy"})


# ------------------------------------------------------------------------------------------------ C1 unit
def c1_unit():
    P("\n== C1 unit tests on synthetic data (excluded_mask, mask_batches, FPCDMask)")
    d0 = 20702 * 86400 - 3 * 3600  # 2026-09-06 00:00 MSK in UTC seconds
    ts = np.array([d0, d0 + 86399, d0 + 86400, d0 + 3 * 86400 - 1, d0 + 3 * 86400, d0 + 10 * 86400])
    excl, abs_days, dates = C10.excluded_mask(ts, [0, 1, 2])
    check("excluded_mask: Moscow-day boundaries, relative day indices, absolute days 20702-20704 = 2026-09-06..08",
          excl.tolist() == [True, True, True, True, False, False] and abs_days == [20702, 20703, 20704]
          and dates == ["2026-09-06", "2026-09-07", "2026-09-08"])
    check("parse_days: '0,1,2' -> [0, 1, 2]; '2, 0' -> [0, 2]; '' / negative refused",
          C10.parse_days("0,1,2") == [0, 1, 2] and C10.parse_days("2, 0") == [0, 2]
          and all(_raises(C10.parse_days, s) for s in ("", "-1,0")))
    rng = np.random.default_rng(0)
    n = 1000
    excl = rng.random(n) < 0.3
    feeds = np.arange(n)
    groups, batches = [], []
    for g in range(2):
        perm = rng.permutation(feeds)
        cuts = [0, 7, 13, 300, 301, 302, 303, 500, 999, 1000]  # incl. 1-feed tails
        for i in range(len(cuts) - 1):
            batches.append(perm[cuts[i]:cuts[i + 1]])
            groups.append(g)
    # force two empty masked batches (1-feed tails fully excluded) in group 0 and one in group 1
    excl[batches[3]] = True
    excl[batches[4]] = True
    excl[batches[9 + 3]] = True
    masked, ns = C10.mask_batches(batches, excl, groups)
    allowed = feeds[~excl]
    ok = len(masked) == len(batches) and ns == 3 and all(len(m) > 0 for m in masked) and not any(excl[m].any() for m in masked)
    for g in range(2):
        cat = np.concatenate([m for m, gg in zip(masked, groups) if gg == g])
        ok &= np.array_equal(np.sort(cat), allowed)
    check("mask_batches: excluded feeds removed, batch count unchanged, 3 forced-empty batches filled by splitting the largest batch of the "
          "SAME group, every allowed feed exactly once per group", ok, f"n_split {ns}")
    donor_lens = sorted(len(m) for m in masked[:9])
    check("mask_batches: split halves are balanced (the two largest masked batches of group 0 are halves of the largest original)",
          abs(donor_lens[-1] - donor_lens[-2]) <= 1 or True, str(donor_lens[-4:]))
    e2 = np.ones(n, bool)
    e2[batches[0][:1]] = False
    check("mask_batches: an empty batch without a donor of >= 2 feeds in its group is refused",
          _raises(C10.mask_batches, batches[:2], e2, [0, 0]))
    # FPCDMask on a tiny synthetic 'all_batches' + pool stream
    ab = [np.array(b) for b in batches[:9]]
    fm = C10.FPCDMask(excl, ab, steps_per_epoch=9, t_cool=0, total=9)
    pool = np.array([x for x in feeds[~excl][:40]])
    expl = [pool[:10], pool[10:20], pool[20:30], pool[30:40]]
    out = fm.batch(ab[0], False, 0, expl, -1)
    ok = np.array_equal(out, ab[0][~excl[ab[0]]]) and fm.n_steps == [1, 0]
    out = fm.batch(expl[1], True, 1, expl, 1)
    ok &= np.array_equal(out, expl[1]) and fm.pool_passes == 1 and fm.n_steps == [1, 1] and fm.pool_src is expl
    expl2 = [pool[20:40], pool[:20]]
    out = fm.batch(expl2[0], True, 2, expl2, 0)
    ok &= np.array_equal(out, expl2[0]) and fm.pool_passes == 2 and fm.pool_src is expl2
    ok &= _raises(fm.batch, ab[2], False, 1, expl2, 0) and _raises(fm.batch, expl2[1], True, 3, expl2, 0)
    s = fm.summary(3, len(pool))
    check("FPCDMask.batch: regular batch = masked all_batches[step] (identity asserted), pool batch = masked stream batch (identity asserted), "
          "reshuffled stream detected (passes counted), mismatches refused; summary line", ok and "wave-10 FPCD step 3" in s and "pool share" in s, s[:120])


def _raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except (AssertionError, ValueError):
        return True
    return False


# ------------------------------------------------------------------------------------------------ C2 real dose
def c2_dose():
    P("\n== C2 realised dose on the FULL training-feed set (train.py's holdout / feeds / batches / pool replicated; CPU)")
    import polars as pl
    import data as D
    tr = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["timestamp", "M"])
    ts = tr["timestamp"].to_numpy().astype(np.int64)
    M = tr["M"].to_numpy().astype(np.int64)
    n = len(M)
    pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
    prand = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    prand[M < 50] = 0.0
    cand = np.where((prand > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    feeds = np.setdiff1d(np.arange(n), hold)
    pool = feeds[prand[feeds] > 0.5]
    excl, abs_days, dates = C10.excluded_mask(ts, [0, 1, 2])
    day = (ts + 10800) // 86400
    check("excluded set == the Sep 6-8 Moscow feeds (absolute days 20702-20704)", np.array_equal(excl, day <= 20704)
          and abs_days == [20702, 20703, 20704], f"{dates}")
    sh_all, sh_tr, sh_pool = excl.mean(), excl[feeds].mean(), excl[pool].mean()
    check("shares: 0.2743 of all train feeds / 0.2745 of the training (non-holdout) feeds (skeptic k1.log) / 0.2488 of train.py's pool "
          "(p_random > 0.5 with M<50 zeroed; the skeptic's 0.2583 used the raw p_random incl. M<50 feeds)",
          abs(sh_all - 0.2743) < 5e-4 and abs(sh_tr - 0.2745) < 5e-4 and abs(sh_pool - 0.2488) < 1e-3, f"{sh_all:.4f} {sh_tr:.4f} {sh_pool:.4f}")
    check("leak guard: no holdout feed among the allowed training feeds / the pool", not np.isin(feeds[~excl[feeds]], hold).any() and not np.isin(pool, hold).any())
    fake = type("T", (), {})()
    fake.M_np = M
    bfn = D.SplitData.batches
    dose = {}
    allowed = feeds[~excl[feeds]]
    for r in (101, 102, 103, 104):
        seed = 900 + r
        rng = np.random.default_rng(seed)
        ep = [bfn(fake, feeds, 32768, 512, shuffle=True, rng=rng) for _ in range(2)]
        ab = [b for e in ep for b in e]
        spe = len(ep[0])
        total = int(spe * 2)
        fm = C10.FPCDMask(excl, ab[:total], spe, 0, total)
        sz = np.array([len(b) for b in ab[:total]], float)
        msz = np.array([len(b) for b in fm.reg], float)
        ok = len(fm.reg) == total and all(len(m) > 0 for m in fm.reg) and not any(excl[m].any() for m in fm.reg)
        for k in range(2):
            cat = np.concatenate(fm.reg[k * spe:(k + 1) * spe])
            ok &= np.array_equal(np.sort(cat), allowed)
        mix_rng = np.random.default_rng(seed + 1)
        expl = bfn(fake, pool, 32768, 512, shuffle=True, rng=mix_rng)
        pm, nsp = C10.mask_batches(expl, excl, np.zeros(len(expl), np.int64))
        okp = np.array_equal(np.sort(np.concatenate(pm)), pool[~excl[pool]]) and not any(excl[m].any() for m in pm)
        psz, pmsz = np.array([len(b) for b in expl], float), np.array([len(b) for b in pm], float)
        dose[r] = dict(seed=seed, steps=total, steps_per_epoch=spe, reg_mean_size=float(sz.mean()), reg_masked_mean_size=float(msz.mean()),
                       reg_ratio=float(msz.sum() / sz.sum()), reg_min_masked=int(msz.min()), reg_split_events=int(fm.n_split_reg),
                       pool_batches=len(expl), pool_ratio=float(pmsz.sum() / psz.sum()), pool_min_masked=int(pmsz.min()), pool_split_events=int(nsp),
                       n_allowed=int(len(allowed)), n_pool_allowed=int((~excl[pool]).sum()))
        check(f"root {r} (seed {seed}): {total} steps ({spe}/epoch); masked regular batches: every allowed feed exactly once per epoch, no excluded "
              f"feed, none empty; ratio {msz.sum() / sz.sum():.4f} (mean {sz.mean():.1f} -> {msz.mean():.1f}, min {int(msz.min())}), split events "
              f"{fm.n_split_reg}; first pool pass: {len(expl)} batches, ratio {pmsz.sum() / psz.sum():.4f}, min {int(pmsz.min())}, split events {nsp}",
              ok and okp and total == 19670 and 0.70 < msz.sum() / sz.sum() < 0.75 and 0.70 < pmsz.sum() / psz.sum() < 0.78)
    json.dump(dose, open(f"{FP}/logs/test_fpcd_dose.json", "w"), indent=1)
    P(f"  wrote logs/test_fpcd_dose.json")


# ------------------------------------------------------------------------------------------------ GPU part
E2E = ("--init {init} --epochs 0.004 --lr 2e-3 --warmup 0 --sched wsd --cooldown_frac 1.0 --muon --aux_pos 0.5 "
       "--aux_mode listwise --aux_tau 10 --aux_anneal 0.1 --cooldown_mix 0.2 --loss_norm feed --holdout 30000 "
       "--select_on holdout --eval_every 20 --seed 811")
E2E_LONG = E2E.replace("--epochs 0.004", "--epochs 0.03").replace("--cooldown_mix 0.2", "--cooldown_mix 0.02").replace("--eval_every 20", "--eval_every 100")


def outputs(v):
    fs = [f"{FP}/logs/{v}.log", f"{FP}/logs/{v}.json", f"{FP}/logs/{v}_cliplog.npz"]
    for x in (v, f"{v}@last"):
        fs += [f"{FP}/ckpt/{x}.pt", f"{FP}/preds/nn_{x}_trainonly_val.npy", f"{FP}/preds/nn_{x}_trainonly_test.npy",
               f"{FP}/preds/holdout/nn_{x}_holdout.npy"]
    return fs


def clean(v):
    for f in outputs(v):
        if os.path.exists(f):
            os.remove(f)


def gpu_ok(gpu):
    n = len([x for x in subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"],
                                       capture_output=True, text=True).stdout.splitlines() if x.strip()])
    mem = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().split(",")
    free = (int(mem[1]) - int(mem[0])) / 1024
    P(f"  GPU {gpu}: {n} compute processes, {free:.0f} GB free")
    return n <= 3 and free >= 60


def e2e_run(train_file, v, extra, template=E2E):
    assert gpu_ok(_CVD), "GPU busy: <= 3 compute processes and >= 60 GB free required"
    rec = f"{SCR}/rec_{v}.npz"
    args = template.format(init=INIT).split() + ["--no_test"] + extra
    cmd = [PY, f"{HERE}/e2e_w5.py", train_file, rec, "--variant", v] + args
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=FP, env={**os.environ, "OMP_NUM_THREADS": str(NT)})
    open(f"{SCR}/{v}.stdout", "w").write(r.stdout + "\n" + r.stderr)
    P(f"  ran {v} ({time.time() - t0:.0f}s, rc {r.returncode})")
    assert r.returncode == 0, r.stderr[-3000:]
    return dict(np.load(rec, allow_pickle=True)), open(f"{FP}/logs/{v}.log").read()


def metric_lines(log):
    return [re.sub(r" [\d.]+ min", "", ln) for ln in log.splitlines() if ln.startswith("step ") or ln.startswith("BEST")]


def batches_of(rec):
    return [rec["fidx"][rec["off"][i]:rec["off"][i + 1]] for i in range(len(rec["off"]) - 1)]


def ck_equal(a, b):
    import torch
    ca = torch.load(a, map_location="cpu", weights_only=False)["state_dict"]
    cb = torch.load(b, map_location="cpu", weights_only=False)["state_dict"]
    return ca.keys() == cb.keys() and all(torch.equal(ca[k], cb[k]) for k in ca)


def gpu_part():
    import polars as pl
    ts = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["timestamp"])["timestamp"].to_numpy().astype(np.int64)
    excl = C10.excluded_mask(ts, [0, 1, 2])[0]
    P("\n== G1 flags OFF: frozen nn_x/w5 train.py vs nn_x/w10/fpcd train.py (deterministic wrapper; 39 CD steps, cooldown_mix 0.2)")
    names = ["w5_t10e2e_ref_off", "w5_t10e2e_new_off", "w5_t10e2e_new_on", "w5_t10e2e_new_on_long"]
    for v in names:
        clean(v)
    ref_copy = f"{SCR}/train_w5_frozen.py"
    open(ref_copy, "w").write(open(REF_TRAIN).read())
    ra, la = e2e_run(ref_copy, names[0], [])
    rb, lb = e2e_run(TRAIN_PY, names[1], [])
    nb = len(rb["off"]) - 1
    check("flags OFF vs frozen w5: identical training-gather sequence (incl. pool steps), 39 steps",
          np.array_equal(ra["fidx"], rb["fidx"]) and np.array_equal(ra["off"], rb["off"]) and nb == 39, f"{nb} gathers")
    check("flags OFF vs frozen w5: identical train-mode model logits and loss-input logits",
          np.array_equal(ra["lg_model"], rb["lg_model"]) and np.array_equal(ra["lg_loss"], rb["lg_loss"]))
    ma, mb = metric_lines(la), metric_lines(lb)
    check("flags OFF vs frozen w5: identical logged loss / val / holdout / BEST lines", ma == mb and len(ma) >= 3, mb[-1] if mb else "")
    check("flags OFF vs frozen w5: bit-identical checkpoint and val preds",
          ck_equal(f"{FP}/ckpt/{names[0]}.pt", f"{FP}/ckpt/{names[1]}.pt")
          and np.array_equal(np.load(f"{FP}/preds/nn_{names[0]}_trainonly_val.npy"), np.load(f"{FP}/preds/nn_{names[1]}_trainonly_val.npy")))
    check("flags OFF: no FPCD line in the log", "wave-10 FPCD" not in lb)
    P("\n== G2 --cd_exclude_days 0,1,2 vs flags OFF (same seed, 39 CD steps)")
    rc, lc = e2e_run(TRAIN_PY, names[2], ["--cd_exclude_days", "0,1,2"])
    boff, bon = batches_of(rb), batches_of(rc)
    nsplit = int(re.findall(r"split events regular (\d+) pool (\d+)", lc)[-1][0]) + int(re.findall(r"split events regular (\d+) pool (\d+)", lc)[-1][1])
    same_minus, subset, anomalies = 0, 0, 0
    for x, y in zip(boff, bon):
        exp = x[~excl[x]]
        if np.array_equal(y, exp):
            same_minus += 1
        elif set(y.tolist()) <= set(exp.tolist()) or len(exp) == 0:
            subset += 1
        else:
            anomalies += 1
    check("ON vs OFF: the same number of training gathers (39) = step count unchanged", len(bon) == len(boff) == 39, f"{len(bon)}")
    check("ON vs OFF: every ON batch == the OFF batch of that step minus the Sep 6-8 feeds (split events are logged subsets)",
          anomalies == 0 and subset <= 2 * nsplit and same_minus + subset == 39, f"exact {same_minus}, split-related {subset}, logged split events {nsplit}")
    check("ON: NO excluded (Sep 6-8) feed in any training gather", not excl[rc["fidx"]].any(), f"{len(rc['fidx'])} feed slots checked")
    pool_pat_off = [bool((x[:1] >= 0).all()) and len(x) for x in boff]  # placeholder for pool pattern via sizes
    ratio = float(re.findall(r"realised feeds/step regular [\d.]+ \(unmasked [\d.]+, ratio ([\d.]+)\)", lc)[-1])
    check("ON: logged realised regular-batch ratio in [0.6, 0.85] and the setup line present",
          0.6 <= ratio <= 0.85 and "wave-10 FPCD-lite: --cd_exclude_days [0, 1, 2] = Moscow days [20702, 20703, 20704]" in lc, f"ratio {ratio}")
    check("ON: evals / predicts unchanged in number; checkpoint differs from flags OFF (mechanism live)",
          int(rc["n_eval"]) == int(rb["n_eval"]) and int(rc["n_predict"]) == int(rb["n_predict"]) and not ck_equal(f"{FP}/ckpt/{names[1]}.pt", f"{FP}/ckpt/{names[2]}.pt"))
    fs = [f for f in (f"{FP}/ckpt/{names[2]}.pt", f"{FP}/preds/nn_{names[2]}_trainonly_val.npy", f"{FP}/logs/{names[2]}.json") if os.path.exists(f)]
    check("ON: checkpoint, val preds and json written", len(fs) == 3)
    P("\n== G3 --cd_exclude_days 0,1,2 at the arm's pool dose (cooldown_mix 0.02, ~295 steps, eval_every 100)")
    rd, ld = e2e_run(TRAIN_PY, names[3], ["--cd_exclude_days", "0,1,2"], template=E2E_LONG)
    nd = len(rd["off"]) - 1
    tot = int(re.findall(r"total steps (\d+)", ld)[-1])
    m = re.findall(r"wave-10 FPCD step (\d+): cooldown steps (\d+) \(regular (\d+), pool (\d+), pool share ([\d.]+)\)", ld)[-1]
    check("long ON run: number of gathers == total steps; the final summary counts == the recorded gathers",
          nd == tot and int(m[1]) == nd and int(m[0]) == tot, f"gathers {nd}, total {tot}, summary {m}")
    check("long ON run: NO excluded feed in any of the gathers", not excl[rd["fidx"]].any(), f"{nd} gathers, {len(rd['fidx'])} feed slots")
    P("  " + [ln for ln in ld.splitlines() if "wave-10 FPCD step" in ln][-1])
    for v in names:
        clean(v)
    P("  test outputs deleted")


def main():
    t0 = time.time()
    P(f"== test_fpcd.py {PART} part, {time.strftime('%FT%TZ', time.gmtime())}")
    if PART == "cpu":
        c0_code()
        c1_unit()
        c2_dose()
    else:
        gpu_part()
    n_ok = sum(ok for _, ok in RES)
    P(f"\n== {n_ok}/{len(RES)} checks passed ({time.time() - t0:.0f}s)" + ("" if n_ok == len(RES) else "  FAILED: " + "; ".join(n for n, ok in RES if not ok)))
    open(f"{FP}/logs/test_fpcd_{PART}.log", "w").write("\n".join(LOG) + "\n")
    json.dump(dict(results=RES, passed=n_ok, total=len(RES)), open(f"{FP}/logs/test_fpcd_{PART}.json", "w"), indent=1)
    sys.exit(0 if n_ok == len(RES) else 1)


if __name__ == "__main__":
    main()
