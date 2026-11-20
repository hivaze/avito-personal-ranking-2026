"""LATE-DROPOUT flag probe (CPU only; TRICKS_SWEEP_1113 section 3 item 1). Read-only on nn_x/abl: nothing there is edited,
no bytecode is written, nothing is saved to nn_x/abl/{ckpt,logs,preds}.

Proves, from the very nn_x/abl code that the 10-epoch pair will run, that
  LD = E_BASE + HYG + `--drop 0 --attn_drop 0.1 --drop_cd 0.1`
gives: stable phase (warmup + plateau) residual / FFN / retrieval-value dropout 0 and attention-probability dropout 0.1;
cooldown (from step t_cool) dropout 0.1 on every site (= the base's 0.1 everywhere).

How (no re-implementation of the code under test):
  1. train.py's own `parse` and XT_* preset assignments are exec'd from its AST (as tests/test_abl.py does), so the LD
     and base commands are parsed by the real parser; the namespace diff LD vs base must be exactly drop / attn_drop /
     drop_cd (+ variant), and the base namespace must equal the args of the reference run w5_e10_base_s951 (its log).
  2. The model is built by eval'ing train.py's own `Ranker(...)` call expression (AST) with the parsed args and the xt
     layout derived exactly as train.py derives it.
  3. The training loop's own `t_cool = ...` statement and its `if a.drop_cd >= 0 and step == t_cool: set_dropout(...)`
     block (cd_arms.set_dropout, the function train.py imports) are exec'd from train.py's AST at every step of a short
     CPU run (total 10 steps -> t_cool 8 by train.py's formula) on real MiniSplit train batches with forward, backward and
     an optimizer step.
  4. At every step the effective p of every nn.Dropout module and of every attention-probability site is printed, and
     both are also MEASURED: forward hooks record whether each nn.Dropout output equals its input (p = 0) or the zeroed
     share (p > 0), and a wrapper around F.scaled_dot_product_attention records the dropout_p actually passed.
Usage: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/tricks/late_dropout/probe_dropout_schedule.py
"""
import argparse
import ast
import collections
import json
import os
import sys

sys.dont_write_bytecode = True
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU probe: run with CUDA_VISIBLE_DEVICES=''"
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
warnings.filterwarnings("ignore", message=".*device_type of 'cuda'.*")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(12)
ROOT = "<repo>"
ABL = f"{ROOT}/nn_x/abl"
OUT = f"{ROOT}/nn_x/tricks/late_dropout"
sys.path.insert(0, f"{ABL}/tests")
sys.path.insert(1, ABL)
import abl_arms as AA  # noqa: E402
import cd_arms as CA  # noqa: E402
import model as MO  # noqa: E402
from mini_split import MiniSplit, pick_train_feeds  # noqa: E402

TRAIN_PY = f"{ABL}/train.py"
SRC = open(TRAIN_PY).read()
TREE = ast.parse(SRC)
E_BASE = ("--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 "
          "--select_on holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --epochs 10 --eval_every 2500 --muon "
          "--sched wsd --cooldown_frac 0.2 --aux_anneal 0.1 --retr_m 16 --attn_gate --cooldown_mix 0.02 --lr 2e-3 "
          "--save_steps 70000,77500").split()
HYG = "--save_last --save_holdout --select_min_step 12000 --cliplog 50".split()
LD = "--drop 0 --attn_drop 0.1 --drop_cd 0.1".split()
LINES = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


# ---- 1. train.py's own parser and presets (AST exec, as tests/test_abl.py load_parse / presets)
fn = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "parse"][0]
ns_parse = {"argparse": argparse, "POOL_P_DEFAULT": CA.POOL_P_DEFAULT}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "train_parse", "exec"), ns_parse)


def parse(argv):
    sys.argv = ["train.py"] + argv
    return ns_parse["parse"]()


body = [n for n in TREE.body if isinstance(n, ast.Assign)
        and any(ast.unparse(t).startswith("XT_") for t in n.targets)]
ns_xt = {}
exec(compile(ast.Module(body=body, type_ignores=[]), "train_presets", "exec"), ns_xt)
XT_PRESETS, XT_REL, XT_RELONLY = ns_xt["XT_PRESETS"], ns_xt["XT_REL"], ns_xt["XT_RELONLY"]

cmd_ld = ["--variant", "abl_e10_latedrop_s951"] + E_BASE + HYG + ["--seed", "951"] + LD
cmd_base = ["--variant", "abl_e10_base_s951"] + E_BASE + HYG + ["--seed", "951"]
a, b0 = parse(cmd_ld), parse(cmd_base)
diff = {k: (getattr(b0, k), getattr(a, k)) for k in vars(a) if getattr(a, k) != getattr(b0, k)}
P("== 1. parser (train.py parse via AST)")
P("LD command:", "train.py " + " ".join(cmd_ld))
P("namespace diff LD vs E_BASE+HYG (base value, LD value):", diff)
ok1 = diff == {"variant": ("abl_e10_base_s951", "abl_e10_latedrop_s951"), "drop": (0.1, 0.0), "attn_drop": (-1.0, 0.1),
               "drop_cd": (-1.0, 0.1)}
P("CHECK exact diff = {drop 0.1->0, attn_drop -1->0.1, drop_cd -1->0.1}:", "PASS" if ok1 else "FAIL")
l0 = open(f"{ROOT}/nn_x/w5/logs/w5_e10_base_s951.log").readline()
ref = ast.literal_eval(l0[l0.index("args: ") + 6:].strip())
rdiff = {k: (ref.get(k, "<absent in ref>"), v) for k, v in vars(b0).items() if ref.get(k, "<absent in ref>") != v}
abl_only_defaults = {"trunc_lifts": "", "hzcd": False, "val_mix": 0.0, "val_fold": -1, "xt_exclude": "", "xt_rel_extra": "",
                     "retr_cat": 0}
rdiff_x = {k: v for k, v in rdiff.items() if k != "variant" and not (k in abl_only_defaults and v[0] == "<absent in ref>"
                                                                      and v[1] == abl_only_defaults[k])}
missing = [k for k in ref if k not in vars(b0)]
P("E_BASE+HYG namespace vs the reference run w5_e10_base_s951 args (log line 1): diffs other than variant and the "
  "abl-only flags at their OFF default:", rdiff_x, "| ref keys missing in abl parser:", missing)
ok1b = not rdiff_x and not missing
P("CHECK E_BASE+HYG (seed 951) == w5_e10_base_s951 recipe:", "PASS" if ok1b else "FAIL")

# ---- 2. the model, built by train.py's own Ranker(...) call (AST eval), xt layout as train.py derives it
xt_names = list(XT_PRESETS[a.xt])
if a.xt_exclude:
    xt_names = AA.xt_exclude_names(xt_names, a.xt_exclude, lambda *x: None)
xt_rel = [xt_names.index(n) for n in XT_REL.get(a.xt, []) if xt_names and n in xt_names] or None
if a.xt_rel_extra:
    xt_rel = AA.xt_rel_extend(xt_rel, xt_names, a.xt_rel_extra, lambda *x: None)
xt_idcols = [i for i, n in enumerate(xt_names or []) if n.startswith(("seq_cv", "it_hist"))] or None
xt_relonly = [xt_names.index(n) for n in XT_RELONLY.get(a.xt, []) if xt_names and n in xt_names] or None
calls = [n for n in ast.walk(TREE) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "Ranker"
         and any(k.arg == "drop" for k in n.keywords)]
assert len(calls) == 1, f"expected exactly one Ranker(..., drop=...) call in train.py, found {len(calls)}"
ranker_src = ast.unparse(calls[0])
assert "drop=a.drop" in ranker_src and "attn_drop=a.attn_drop if a.attn_drop >= 0 else None" in ranker_src
main_fn = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
tcool_stmt = [n for n in ast.walk(main_fn) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "t_cool"]
loop = [n for n in ast.walk(main_fn) if isinstance(n, ast.For) and ast.unparse(n.target).strip("()") == "step, fb"]
assert len(tcool_stmt) == 1 and len(loop) == 1
dropcd_if = loop[0].body[0]
assert isinstance(dropcd_if, ast.If) and ast.unparse(dropcd_if.test) == "a.drop_cd >= 0 and step == t_cool", \
    ast.unparse(dropcd_if.test)
P("\n== 2. code under test (verbatim from train.py)")
P("Ranker call:", ranker_src)
P("t_cool:", ast.unparse(tcool_stmt[0]))
P("first statement of the training loop:\n" + ast.unparse(dropcd_if))
P("cd_arms.set_dropout / dropout_sites: from", CA.__file__)
code_ranker = compile(ast.Expression(body=calls[0]), "train_Ranker", "eval")
code_tcool = compile(ast.Module(body=[tcool_stmt[0]], type_ignores=[]), "train_tcool", "exec")
code_dropcd = compile(ast.Module(body=[dropcd_if], type_ignores=[]), "train_dropcd", "exec")


def build(args):
    torch.manual_seed(args.seed)
    env = dict(Ranker=MO.Ranker, a=args, N_DENSE=MO.N_DENSE, xt_names=xt_names, xt_rel=xt_rel, xt_idcols=xt_idcols,
               xt_relonly=xt_relonly, id_buckets=args.id_buckets)
    return eval(code_ranker, env)


# ---- measurement hooks
SDPA_P = []
_orig_sdpa = F.scaled_dot_product_attention


def _sdpa_rec(*x, **k):
    SDPA_P.append(float(k.get("dropout_p", 0.0)))
    return _orig_sdpa(*x, **k)


F.scaled_dot_product_attention = _sdpa_rec  # model.py calls F.scaled_dot_product_attention (same module object)
assert MO.F is F
DROP_OBS = collections.defaultdict(list)


def hook(name):
    def h(mod, inp, out):
        x = inp[0]
        if torch.equal(out, x):
            DROP_OBS[name].append(("identity", 0.0))
        else:  # share of entries zeroed by this call among the nonzero inputs
            nz = x != 0
            DROP_OBS[name].append(("dropped", float(((out == 0) & nz).sum() / nz.sum().clamp_min(1))))
    return h


def site_table(model):
    mods = {n: m for n, m in model.named_modules() if isinstance(m, nn.modules.dropout._DropoutNd)}
    attns = {n: m for n, m in model.named_modules() if type(m).__name__ == "Attn"}
    d = {f"Dropout {n}": m.p for n, m in mods.items()}
    d.update({f"Attn-prob {n}": (m.drop if m.attn_drop is None else m.attn_drop) for n, m in attns.items()})
    return d


def run(args, label, total=10, n_batches=10):
    P(f"\n== 3. CPU smoke run [{label}]: {total} train steps on real MiniSplit train batches (forward + backward + "
      f"AdamW step), train.py's own t_cool formula and drop_cd block")
    model = build(args)
    feeds = pick_train_feeds(n_each=12, seed=7)
    ms = MiniSplit("train", feeds, xt_names=xt_names)
    fbs = ms.batches(np.arange(ms.n), 6000, 32, shuffle=True, rng=np.random.default_rng(0))[:n_batches]
    bb = ms.gather(fbs[0], with_pos=True)
    model.eval()
    Xd = model.dense(bb)
    model.set_norm(Xd[bb["cmask"]].mean(0), Xd[bb["cmask"]].std(0).clamp_min(1e-3))
    model.train()
    for n, m in model.named_modules():
        if isinstance(m, nn.modules.dropout._DropoutNd):
            m.register_forward_hook(hook(n))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    env = dict(a=args, total=total, round=round, int=int, set_dropout=CA.set_dropout, model=model, log=P)
    exec(code_tcool, env)
    t_cool = env["t_cool"]
    P(f"total {total} -> t_cool {t_cool} (train.py formula; the 10-epoch run: total 98350 -> t_cool "
      f"{int(round((1 - args.cooldown_frac) * 98350))})")
    P(f"model.cfg drop {model.cfg['drop']}, cfg attn_drop {model.cfg.get('attn_drop', '<absent = None>')}; "
      f"{sum(p.numel() for p in model.parameters()) / 1e6:.3f}M params")
    rec = {}
    for step in range(total):
        env["step"] = step
        exec(code_dropcd, env)  # train.py: `if a.drop_cd >= 0 and step == t_cool: set_dropout(model, a.drop_cd) ...`
        SDPA_P.clear()
        DROP_OBS.clear()
        b = ms.gather(fbs[step % len(fbs)], with_pos=True)
        lg = model(b, model.dense(b), use_pos=True)
        loss = MO.listwise_loss(lg, b["y"], b["cmask"], None, norm=args.loss_norm, denom=args.max_feeds)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tab = site_table(model)
        meas_attn = sorted(set(SDPA_P))
        meas_drop = {k: (sorted(set(v[0] for v in vals)), float(np.mean([v[1] for v in vals]))) for k, vals in DROP_OBS.items()}
        rec[step] = dict(set_p=tab, sdpa_dropout_p_passed=meas_attn, n_sdpa_calls=len(SDPA_P), dropout_obs=meas_drop,
                         loss=float(loss))
        phase = "cooldown" if step >= t_cool else ("warmup/plateau")
        if step in (0, t_cool - 1, t_cool, total - 1):
            P(f"-- step {step} ({phase}{', FIRST COOLDOWN STEP' if step == t_cool else ''}) loss {float(loss):.4f}")
            for k, v in tab.items():
                if k.startswith("Dropout"):
                    o = meas_drop.get(k[len("Dropout "):], (["not called"], 0.0))
                    P(f"   {k:<34s} p={v:<4}  measured: {'/'.join(o[0])}" + (f", zeroed share {o[1]:.3f}"
                                                                              if "dropped" in o[0] else ""))
                else:
                    P(f"   {k:<34s} p={v}")
            P(f"   F.scaled_dot_product_attention dropout_p actually passed: {meas_attn} over {len(SDPA_P)} calls")
    model.eval()
    SDPA_P.clear()
    DROP_OBS.clear()
    with torch.no_grad():
        b = ms.gather(fbs[0], with_pos=True)
        model(b, model.dense(b), use_pos=False)
    ev = dict(sdpa=sorted(set(SDPA_P)), drop=sorted(set(o for v in DROP_OBS.values() for o, _ in v)))
    P(f"-- eval mode after training: SDPA dropout_p {ev['sdpa']}, nn.Dropout outputs {ev['drop']} (inference has no dropout)")
    return rec, t_cool, ev, model


rec_ld, tc, ev_ld, m_ld = run(a, "LD = E_BASE + HYG + --drop 0 --attn_drop 0.1 --drop_cd 0.1")
rec_b, _, ev_b, m_b = run(b0, "reference: E_BASE + HYG (base, flags off)")

# ---- 4. verdict on the schedule
P("\n== 4. schedule checks")
res = []


def chk(name, ok):
    res.append((name, bool(ok)))
    P(f"CHECK {name}: {'PASS' if ok else 'FAIL'}")


def sites(r, kind):
    return {k: v for k, v in r["set_p"].items() if k.startswith(kind)}


nd = len(sites(rec_ld[0], "Dropout"))
na = len(sites(rec_ld[0], "Attn-prob"))
P(f"dropout sites: {nd} nn.Dropout modules ({', '.join(k[8:] for k in sites(rec_ld[0], 'Dropout'))}); {na} attention-"
  f"probability sites ({', '.join(k[10:] for k in sites(rec_ld[0], 'Attn-prob'))})")
stable = [s for s in rec_ld if s < tc]
cool = [s for s in rec_ld if s >= tc]
chk("LD stable phase: every nn.Dropout (Block FFN-hidden ff.2, Block residual dr, HistRetrieval value dp) p = 0 (set) "
    "and identity (measured) at every step < t_cool",
    all(all(v == 0.0 for v in sites(rec_ld[s], "Dropout").values()) for s in stable)
    and all(all(o[0] == ["identity"] for o in rec_ld[s]["dropout_obs"].values()) and len(rec_ld[s]["dropout_obs"]) == nd
            for s in stable))
chk("LD stable phase: every attention-probability site p = 0.1 (set) and SDPA receives dropout_p == [0.1] (measured)",
    all(all(abs(v - 0.1) < 1e-12 for v in sites(rec_ld[s], "Attn-prob").values()) for s in stable)
    and all(rec_ld[s]["sdpa_dropout_p_passed"] == [0.1] for s in stable))
chk("LD cooldown (step >= t_cool, from the FIRST cooldown step): every nn.Dropout p = 0.1 (set) and drops ~10% (measured), "
    "every attention site p = 0.1 and SDPA receives [0.1]",
    all(all(abs(v - 0.1) < 1e-12 for v in rec_ld[s]["set_p"].values()) for s in cool)
    and all(all(o[0] == ["dropped"] and 0.05 < o[1] < 0.15 for o in rec_ld[s]["dropout_obs"].values())
            and len(rec_ld[s]["dropout_obs"]) == nd for s in cool)
    and all(rec_ld[s]["sdpa_dropout_p_passed"] == [0.1] for s in cool))
chk("base reference: every site 0.1 at every step (so the LD cooldown == the base's dropout everywhere)",
    all(all(abs(v - 0.1) < 1e-12 for v in rec_b[s]["set_p"].values()) for s in rec_b)
    and all(rec_b[s]["sdpa_dropout_p_passed"] == [0.1] for s in rec_b))
chk("LD and base have the same dropout sites (names) and the same parameters at init (dropout has no parameters)",
    rec_ld[0]["set_p"].keys() == rec_b[0]["set_p"].keys()
    and all(torch.equal(x, y) for x, y in zip(build(a).state_dict().values(), build(b0).state_dict().values())))
chk("eval mode: no dropout anywhere (SDPA dropout_p 0, nn.Dropout identity) for LD and base",
    ev_ld == {"sdpa": [0.0], "drop": ["identity"]} and ev_b == {"sdpa": [0.0], "drop": ["identity"]})
chk("train.py t_cool formula at the 10-epoch length (98350 steps) = 78680 (= the base run's logged cooldown start)",
    int(round((1 - a.cooldown_frac) * 98350)) == 78680)
chk("parser: LD differs from E_BASE+HYG only by drop / attn_drop / drop_cd", ok1)
chk("parser: E_BASE+HYG == the w5_e10_base_s951 reference recipe", ok1b)
P("NOTE (not a check): the LD checkpoint cfg stores drop 0.0 and attn_drop 0.1 (the drop_cd switch leaves cfg unchanged), "
  "so any later fine-tune that rebuilds the model from this cfg trains with drop 0 / attn_drop 0.1, not the base FT's 0.1 "
  f"everywhere. cfg here: drop {m_ld.cfg['drop']}, attn_drop {m_ld.cfg.get('attn_drop')}. Not used by the pre-declared "
  "readout (train-only and holdout preds only).")
allok = all(o for _, o in res)
P(f"\nPROBE VERDICT: {'PASS: the flags produce exactly the LATE-DROPOUT schedule' if allok else 'FAIL'} "
  f"({sum(o for _, o in res)}/{len(res)} checks)")
json.dump(dict(verdict="PASS" if allok else "FAIL", checks=res, t_cool_probe=tc, n_dropout_modules=nd, n_attn_sites=na,
               ld_steps={str(k): v for k, v in rec_ld.items()}, base_steps={str(k): v for k, v in rec_b.items()},
               eval_ld=ev_ld, eval_base=ev_b, namespace_diff={k: list(v) for k, v in diff.items()},
               ref_diff={k: list(map(str, v)) for k, v in rdiff_x.items()}),
          open(f"{OUT}/probe_dropout_schedule.json", "w"), indent=1, default=str)
open(f"{OUT}/probe_dropout_schedule.txt", "w").write("\n".join(LINES) + "\n")
sys.exit(0 if allok else 1)
