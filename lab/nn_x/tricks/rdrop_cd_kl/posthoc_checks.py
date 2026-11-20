"""Post-hoc instrument checks (POSTHOC_CHECKS.txt; info only) on w5_cd_m02ref_r101.pt. Reuses probe_kl.py helpers.

  CUDA_VISIBLE_DEVICES=N PYTHONDONTWRITEBYTECODE=1 python posthoc_checks.py
"""
import json
import time

import numpy as np
import torch

import probe_kl as P


def run(model, batches, mode, autocast, seed):
    torch.manual_seed(seed)
    out = []
    with torch.no_grad():
        for b in batches:
            model.train() if mode == "train" else model.eval()
            X = model.dense(b)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                l1 = model(b, X, use_pos=True)
                l2 = model(b, X, use_pos=True)
            sym, _, _, sp, t1 = P.pair_stats(l1, l2, b["cmask"])
            out.append((sym.cpu().numpy(), sp.cpu().numpy(), t1.cpu().numpy()))
    sym = np.concatenate([o[0] for o in out]); sp = np.concatenate([o[1] for o in out])
    t1 = np.concatenate([o[2] for o in out])
    return dict(symKL=P.summ(sym), spearman=P.summ(sp), top1_change=float(t1.mean()),
                max_abs_symKL=float(np.abs(sym).max()))


def main():
    t0 = time.time()
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    tr = P.SplitData("train", dev, max_hist=256)
    model, cfg = P.load(P.DECISION["a_m02ref_r101_sel"], dev)
    tr.attach_xt(cfg["xt_names"])
    pr = np.nan_to_num(np.load(P.ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
    pr[tr.M_np < 50] = 0.0
    prt = torch.from_numpy(pr).to(dev)
    tr.dom = torch.stack([1 - prt, prt, torch.zeros_like(prt)], 1)
    cand = np.where((pr > 0.9) & (tr.M_np >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    feeds = np.setdiff1d(np.arange(tr.n), hold)
    res = {}
    bl = tr.batches(feeds, 32768, 512, shuffle=True, rng=np.random.default_rng(P.BATCH_SEED))[:P.N_BATCHES]
    batches = [tr.gather(fb, with_pos=True) for fb in bl]
    res["1_eval_vs_eval_bf16"] = run(model, batches, "eval", True, P.TORCH_SEED)
    res["2_train_vs_train_fp32"] = run(model, batches, "train", False, P.TORCH_SEED)
    res["0_train_vs_train_bf16_rerun"] = run(model, batches, "train", True, P.TORCH_SEED)
    bl2 = tr.batches(feeds, 32768, 512, shuffle=True, rng=np.random.default_rng(2026))[:P.N_BATCHES]
    batches2 = [tr.gather(fb, with_pos=True) for fb in bl2]
    res["3_replicate_bs2026_ts7_bf16"] = run(model, batches2, "train", True, 7)
    res["minutes"] = (time.time() - t0) / 60
    (P.OUT / "posthoc_checks.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(k, json.dumps(v), flush=True)


if __name__ == "__main__":
    main()
