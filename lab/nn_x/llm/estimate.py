"""Exact token counts / steps / FLOP-time-memory estimates for the matched-data test (pure numpy, no torch, no GPU).

  ../../.venv_llm/bin/python estimate.py [--token_budget 56000] [--mfu 0.15] [--n_pre 2] [--val_log]

Amendments 2026-11-12: sequences carry the S1 sink token (n_pre = 2 context tokens before the history; 1 = old layout),
and under V0 the periodic evaluations score the holdout only (--val_log restores holdout + val per evaluation).

Token counts come from data/flat/*/feeds.parquet with the exact batching of FeedStore.token_batches (bucket = sequence
length rounded up to 32; B = budget // bucket), so padding efficiency is measured, not assumed.
FLOP model (reports/llm_research.md 3): training 6*N*T_padded with LoRA + activation checkpointing (2N fwd + 2N bwd to
activations + 2N recompute), 8N for the fully trained random-init control (+2N weight grads), inference 2N*T;
+5% for attention/LoRA matmuls. N = non-embedding body params. MFU is an ASSUMPTION (not measured: no GPU was used).
"""
import argparse
import json

import numpy as np
import polars as pl

ROOT = "<repo>"
N_BODY = {"qwen3-0.6b": 440.47e6, "qwen3-4b-instruct": 3633.5e6, "qwen3.5-2b": 1373.3e6, "qwen3.5-4b": 3570.1e6}
D_MODEL = {"qwen3-0.6b": 1024, "qwen3-4b-instruct": 2560, "qwen3.5-2b": 2048, "qwen3.5-4b": 2560}
LAYERS = {"qwen3-0.6b": 28, "qwen3-4b-instruct": 36, "qwen3.5-2b": 24, "qwen3.5-4b": 32}
FFN = {"qwen3-0.6b": 3072, "qwen3-4b-instruct": 9728, "qwen3.5-2b": 6144, "qwen3.5-4b": 9216}
PEAK = 2250e12  # GPU dense BF16 (spec sheet, [S] in llm_research.md)


def seqlens(split, ids=None, max_hist=256, n_pre=2):
    f = pl.read_parquet(f"{ROOT}/data/flat/{split}/feeds.parquet", columns=["M", "H"])
    M, H = f["M"].to_numpy().astype(np.int64), f["H"].to_numpy().astype(np.int64)
    T = n_pre + np.minimum(H, max_hist) + M
    return T if ids is None else T[ids]


def batch_stats(T, budget, tpad=32, max_feeds=1024):
    Tp = (T + tpad - 1) // tpad * tpad
    nb, padded = 0, 0
    for tp in np.unique(Tp):
        n = int((Tp == tp).sum())
        B = int(max(1, min(max_feeds, budget // tp)))
        k = -(-n // B)
        nb += k
        padded += n * int(tp) + 0  # rows padded to the bucket length; last partial batch has fewer rows, same length
    # padded counts every feed at its bucket length; the true batch max can be < bucket -> this is a mild overestimate
    return nb, int(T.sum()), padded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token_budget", type=int, default=56000)
    ap.add_argument("--mfu", type=float, default=0.15)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--n_pre", type=int, default=2, help="2 = sink + user (S1), 1 = user only")
    ap.add_argument("--val_log", action="store_true", help="periodic evals also score val (V0 forbids it in Stage A)")
    a = ap.parse_args()
    import sys
    sys.path.insert(0, f"{ROOT}/nn_x/llm")
    fM = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["M"])["M"].to_numpy().astype(np.int64)
    pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    p[fM < 50] = 0.0
    cand = np.where((p > 0.9) & (fM >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    out = {}
    T_tr_all = seqlens("train", n_pre=a.n_pre)
    for seed in (23, 24):
        rng = np.random.default_rng(seed)
        feeds = np.setdiff1d(np.sort(rng.choice(len(fM), 150000, replace=False)), hold)
        T = T_tr_all[feeds]
        nb, real, padded = batch_stats(T, a.token_budget)
        out[f"train_s{seed}"] = dict(feeds=len(feeds), mean_T=float(T.mean()), p50=float(np.median(T)),
                                     p90=float(np.quantile(T, 0.9)), max=int(T.max()), steps_per_epoch=nb,
                                     real_tokens=real, padded_tokens=padded, pad_eff=real / padded)
    for name, T in (("holdout", T_tr_all[hold]), ("val", seqlens("val", n_pre=a.n_pre)),
                    ("test", seqlens("test", n_pre=a.n_pre)),
                    ("train_full", T_tr_all)):
        nb, real, padded = batch_stats(T, 96000 if name != "train_full" else a.token_budget)
        out[name] = dict(feeds=len(T), mean_T=float(T.mean()), max=int(T.max()), real_tokens=real,
                         padded_tokens=padded, pad_eff=real / padded, batches=nb)
    tr = out["train_s23"]
    steps = int(tr["steps_per_epoch"] * a.epochs)
    n_eval = -(-steps // a.eval_every)
    rows = []
    for bb in ("qwen3-0.6b", "qwen3-4b-instruct", "qwen3.5-2b"):
        N = N_BODY[bb]
        mfu = a.mfu if bb == "qwen3-0.6b" else 0.25
        eff = PEAK * mfu
        train_flops = 6 * N * tr["padded_tokens"] * a.epochs * 1.05
        ctl_flops = 8 * N * tr["padded_tokens"] * a.epochs * 1.05
        ev = 2 * N * (out["holdout"]["padded_tokens"] + (out["val"]["padded_tokens"] if a.val_log else 0)) * 1.05
        fin = 2 * N * (out["holdout"]["padded_tokens"] + out["val"]["padded_tokens"] + out["test"]["padded_tokens"]) * 1.05
        full_ep = 6 * N * out["train_full"]["padded_tokens"] * 1.05
        # memory (bf16 frozen weights; fp32 residual stream because trainable RMSNorm gains are fp32)
        D, L, F = D_MODEL[bb], LAYERS[bb], FFN[bb]
        tok = a.token_budget
        stored = L * tok * D * 4  # checkpointed layer inputs (fp32 residual)
        work = tok * (4 * D + 3 * F) * 2 * 2  # one layer's recompute + grad working set (bf16), x2 slack
        weights = N * 2 + 30e6 * 4 * 4  # bf16 trunk + ~30M fp32 trainable with AdamW state and grads
        rows.append(dict(backbone=bb, N_body=N, mfu_assumed=mfu, steps=steps, n_evals=n_eval,
                         train_PFLOP=train_flops / 1e15, train_min=train_flops / eff / 60,
                         train_min_pess=train_flops / (eff / 2) / 60,
                         evals_min=n_eval * ev / eff / 60, final_pred_min=fin / eff / 60,
                         control_min=ctl_flops / eff / 60 if bb == "qwen3-0.6b" else None,
                         full_epoch_h=full_ep / eff / 3600, full_epoch_h_pess=full_ep / (eff / 2) / 3600,
                         mem_GB_train_est=(stored + work + weights) / 2**30 + 5.0,
                         mem_GB_control_est=((stored + work) / 2**30 + N * 16 / 2**30 + 5.0) if bb == "qwen3-0.6b" else None))
    out["arms"] = rows
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
