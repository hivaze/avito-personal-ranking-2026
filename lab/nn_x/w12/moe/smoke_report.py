"""w12/moe GPU smoke report (CPU, read-only): flag-off parity of the copy against the w5 trainer (same args, GPU N) and the per-arm smoke numbers.
  cd <repo> && CUDA_VISIBLE_DEVICES="" .venv/bin/python lab/nn_x/w12/moe/smoke_report.py > lab/nn_x/w12/moe/logs/smoke_report.txt
Parity: smoke_out/parity_w5 (lab/nn_x/w5/train.py) vs smoke_out/parity_copy (lab/nn_x/w12/moe/train.py, flags off), variant
  w5_smoke_e10_kd05w192_s951: logged lines (timing fields and the args dict removed), json history, selected / @last checkpoints, val / holdout /
  test preds. Arms: smoke_out/w5_smoke_e10_<arm>_s951 + <v>_diag.json (run_redirect.py MOE_DIAG=1): steps/s net of evals, peak memory, losses,
  the last 'w12/moe step' diagnostics line; reference = parity_copy (or ref_diag when present: same recipe, newer diag fields).
"""
import json
import re
from pathlib import Path

import numpy as np
import torch

D = Path("<repo>/lab/nn_x/w12/moe")
S = D / "smoke_out"
V = "w5_smoke_e10_kd05w192_s951"


def lines(p):
    out = []
    for ln in open(p):
        if ln.startswith("==== "):
            continue  # time + args dict (the copy has the 9 extra moe defaults)
        ln = re.sub(r" [\d.]+ min", " <min>", ln)
        ln = re.sub(r"train time [\d.]+ min", "train time <min>", ln)
        out.append(ln.rstrip("\n"))
    return out


def preds(d, v):
    fs = {"val": f"preds/nn_{v}_trainonly_val.npy", "val@last": f"preds/nn_{v}@last_trainonly_val.npy",
          "hold": f"preds/holdout/nn_{v}_holdout.npy", "hold@last": f"preds/holdout/nn_{v}@last_holdout.npy",
          "test": f"preds/nn_{v}_trainonly_test.npy", "test@last": f"preds/nn_{v}@last_trainonly_test.npy"}
    return {k: np.load(d / f) for k, f in fs.items() if (d / f).exists()}


def parity(a, b, label):
    print(f"== {label}: {a.name} vs {b.name}, variant {V} (smoke recipe, seed 951, GPU N)")
    if not ((a / f"logs/{V}.json").exists() and (b / f"logs/{V}.json").exists()):
        print("  incomplete")
        return
    la, lb = lines(a / f"logs/{V}.log"), lines(b / f"logs/{V}.log")
    diff = [(x, y) for x, y in zip(la, lb) if x != y]
    print(f"  log lines: {len(la)} vs {len(lb)}; differing lines {len(diff) + abs(len(la) - len(lb))}")
    for x, y in diff[:2]:
        print(f"    {a.name:11s}: {x[:160]}\n    {b.name:11s}: {y[:160]}")
    ja, jb = json.load(open(a / f"logs/{V}.json")), json.load(open(b / f"logs/{V}.json"))
    ha = [(h["step"], h["loss"], h["val_ndcg"], h["holdout_ndcg"]) for h in ja["history"]]
    hb = [(h["step"], h["loss"], h["val_ndcg"], h["holdout_ndcg"]) for h in jb["history"]]
    print(f"  history (step, sampled loss, val, holdout): {ha}\n                                          vs {hb}")
    if not (a / f"ckpt/{V}.pt").exists() or not (b / f"ckpt/{V}.pt").exists():
        print("  checkpoints / preds removed after the comparison: see logs/smoke_parity_recorded.txt")
        return
    pa, pb = preds(a, V), preds(b, V)
    print("  preds max |diff|: " + ", ".join(f"{k} {np.abs(pa[k] - pb[k]).max():.2e}" for k in pa if k in pb)
          + f" | bit-identical: {all(np.array_equal(pa[k], pb[k]) for k in pa)}")
    for suf in ("", "@last"):
        ca = torch.load(a / f"ckpt/{V}{suf}.pt", map_location="cpu", weights_only=False)
        cb = torch.load(b / f"ckpt/{V}{suf}.pt", map_location="cpu", weights_only=False)
        sa, sb = ca["state_dict"], cb["state_dict"]
        same = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
        print(f"  checkpoint{suf or ' (selected)'}: same keys {sa.keys() == sb.keys()}, cfg identical {ca['cfg'] == cb['cfg']}, tensors bit-identical {same}"
              + ("" if same else f" (max |diff| {max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa):.2e})"))


parity(S / "parity_w5", S / "parity_copy", "flag-off parity, lab/nn_x/w5/train.py vs the copy (flags off)")
parity(S / "parity_w5", S / "parity_w5b", "GPU run-to-run control, lab/nn_x/w5/train.py twice (same args)")
parity(S / "parity_copy", S / "ref_diag", "GPU run-to-run control, the copy twice (flags off)")

print("\n== warm step rate (segments after the first eval: steps 25-72, CUDA-synchronised, evals excluded; tag 'rate', run back to back)")
wr = {}
for name, v in (("ref", V), ("A moe-ffn", "w5_smoke_e10_kd05w192moeffn_s951"), ("B moe-vo", "w5_smoke_e10_kd05w192moevo_s951"),
                ("C b3ff1", "w5_smoke_e10_kd05w192b3ff1_s951")):
    p = S / f"rate/logs/{v}_diag.json"
    if p.exists():
        g = json.load(open(p))
        wr[name] = g.get("warm_steps_per_s")
        print(f"  {name:10s}: {g.get('warm_steps')} warm steps, {wr[name]} steps/s, step-time ratio vs ref "
              f"{(wr['ref'] / wr[name]) if wr.get('ref') and wr[name] else float('nan'):.3f}; peak step alloc {g['peak_step_alloc_gb']} GB "
              f"(resident {g['resident_gb']}, overhead {g['step_overhead_gb']}), peak reserved {g['peak_train_reserved_gb']} GB; "
              f"loss last10 {g['loss_last10']:.4f}")
print("\n== smokes (20K train feeds, 1 epoch = 72 steps, evals every 25 steps; GPU N alone; run_redirect.py MOE_DIAG=1)")
rows = [("ref (copy, flags off; ref_diag)", S / "ref_diag", V), ("ref (copy, flags off; parity_copy)", S / "parity_copy", V),
        ("ref (w5 train.py; parity_w5b)", S / "parity_w5b", V)]
for arm, tag in (("A moe-ffn", "kd05w192moeffn"), ("B moe-vo", "kd05w192moevo"), ("C b3ff1", "kd05w192b3ff1")):
    rows.append((arm, S, f"w5_smoke_e10_{tag}_s951"))
ref = None
print("| arm | steps | train s net of evals | steps/s | x ref | peak step alloc GB (resident) | step overhead GB | loss first10 | loss last10 | "
      "loss last 25% | val@72 | holdout@72 | params |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for name, d, v in rows:
    p = d / f"logs/{v}_diag.json"
    if not p.exists():
        print(f"| {name} | (missing) |")
        continue
    g = json.load(open(p))
    js = json.load(open(d / f"logs/{v}.json"))
    log = open(d / f"logs/{v}.log").read()
    par = re.search(r"params: ([\d.]+M)", log).group(1)
    h = js["history"][-1]
    ref = ref or g
    print(f"| {name} | {g['steps']} | {g['train_s_net_of_evals']} | {g['steps_per_s']} | {g['steps_per_s'] / ref['steps_per_s']:.2f} | "
          f"{g.get('peak_step_alloc_gb', g['peak_train_alloc_gb'])} ({g.get('resident_gb', 'n/a')}) | {g.get('step_overhead_gb', 'n/a')} | "
          f"{g['loss_first10']:.4f} | {g['loss_last10']:.4f} | {g['loss_last25pct']:.4f} | {h['val_ndcg']:.5f} | {h['holdout_ndcg']:.5f} | {par} |")
    ml = [ln for ln in log.splitlines() if ln.startswith("w12/moe")]
    for ln in ml[:1] + ml[-1:]:
        print(f"    {ln[:400]}")
    if not g.get("finite", False):
        print("    NON-FINITE LOSS")
