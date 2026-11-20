"""Train the soft-token LLM ranker on the matched train subset; select on the 30K random-like holdout.

GPU (decisive test, see run_matched.sh; some GPUs only, never 0-3):
  CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=12 ../../.venv_llm/bin/python train_llm.py --variant llm06_L1_s23 --seed 23 \
      --device cuda --dtype bf16 --grad_ckpt
CPU smoke (tiny):  CUDA_VISIBLE_DEVICES="" ... train_llm.py --variant smoke --device cpu --smoke 32 --max_steps 3

Data = exactly nn/train.py --train_feeds 150000 --holdout 30000 for the same --seed (148,600 feeds for seed 23).
Inputs = nn/ ranker inputs with --user0 --domain --xt cvnd_pre2s. Loss = listwise softmax per feed with equal feed
weight + 0.5 * (1 - p_random) * K9 listwise production-order distillation (tau 10), as the F7 recipe.
Pre-launch amendments (reports/LLM_FEASIBILITY.md 5.2, 2026-11-12):
  S1  frozen sink token at position 0 (model_llm; on by default, row <|im_start|>; --no_sink = old layout, ablation
      only; --sink_token endoftext = the row originally declared, which fails the S1 CPU acceptance check).
  V0  --no_val_log: val is never scored or logged by the trainer (neither at evaluation steps nor at the end); val
      predictions are still written, and val is first scored by compare.py. Stage A passes it for every L/R arm.
  I3  --precision_check (probe): score the holdout with the run's inference path (bf16 frozen weights + autocast) and
      with an fp32-trunk copy of the same trained model; write <out>/i3_precision.json (final_fp32 = |dNDCG| > 1e-4).
      --final_fp32: all final holdout/val/test predictions come from the fp32-trunk copy (TF32 off).
  E1  history[] in logs/<variant>.json carries step + holdout_ndcg only (read by compare.py --e1_decide).
Outputs (under --out, default nn_x/llm):
  preds/<variant>_trainonly_{val,test}.npy   flat candidate order of data/flat/<split>/ (fp32)
  preds/holdout/<variant>_holdout.npy        holdout candidates, holdout feeds in sorted order (= preds/holdout layout)
  ckpt/<variant>.pt                           trainable tensors only (LoRA, norms, projectors, heads) + cfg
  logs/<variant>.log / .json
"""
import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

import data_llm as D
from model_llm import INPUT_KEYS, LLMRanker, count_params, fp32_from_state, lr_lambda_cosine

MEAN_T_TRAIN = 264.5  # measured mean soft-token length per train feed (reports/llm_research.md)


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--backbone", default="qwen3-0.6b")
    p.add_argument("--layout", default="L1", choices=["L1", "L2"])
    p.add_argument("--random_init", action="store_true", help="control R: same architecture, random init, all trained")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--train_feeds", type=int, default=150000)
    p.add_argument("--holdout", type=int, default=30000)
    p.add_argument("--xt", default="cvnd_pre2s")
    p.add_argument("--max_hist", type=int, default=256)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--token_budget", type=int, default=56000, help="padded tokens per training batch (48-64K)")
    p.add_argument("--eval_token_budget", type=int, default=96000)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lr_lora", type=float, default=2e-4, help="LoRA + RMSNorm gains")
    p.add_argument("--lr_proj", type=float, default=1e-3, help="projectors + heads")
    p.add_argument("--lr_full", type=float, default=5e-4, help="trunk lr of the random-init control")
    p.add_argument("--wd", type=float, default=0.02)
    p.add_argument("--warmup", type=int, default=-1, help="-1: 100 (LoRA) / 200 (random-init control)")
    p.add_argument("--floor", type=float, default=0.1, help="cosine floor (fraction of peak lr)")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--aux_pos", type=float, default=0.5)
    p.add_argument("--aux_tau", type=float, default=10.0)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--no_val_log", action="store_true",
                   help="V0: never score/log val in the trainer (eval steps AND end); val preds are still written")
    p.add_argument("--no_sink", action="store_true", help="drop the S1 sink token (old layout; ablation only)")
    p.add_argument("--sink_token", default="im_start",
                   help="S1 sink row, a special token without <| |> (default im_start; 'endoftext' = the row declared in "
                        "LLM_FEASIBILITY 5.2, which fails the S1 acceptance check on CPU, see reports/llm_build.md)")
    p.add_argument("--precision_check", action="store_true",
                   help="I3 (probe): holdout NDCG with bf16 vs fp32 trunk weights -> <out>/i3_precision.json")
    p.add_argument("--i3_threshold", type=float, default=1e-4, help="|dNDCG| above which final preds must be fp32")
    p.add_argument("--final_fp32", action="store_true", help="I3 decision: final predictions with an fp32 trunk")
    p.add_argument("--cpu_bf16", action="store_true",
                   help="smoke only: emulate the GPU bf16 path on CPU (bf16 frozen weights + CPU bf16 autocast)")
    p.add_argument("--no_test", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"], help="frozen trunk weights / autocast")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--time_limit", type=float, default=0, help="minutes")
    p.add_argument("--smoke", type=int, default=0, help="CPU smoke: use this many train / holdout / val feeds")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent))
    # ---- large-model arms (reports/llm_large.md, 2026-11-12): exact gradient accumulation + the size probe
    p.add_argument("--grad_accum", type=int, default=1,
                   help="split every token-budget batch into this many feed micro-batches, one optimiser step per "
                        "batch (exact: the K9 term is normalised by the FULL batch's feed weight); only if memory forces")
    p.add_argument("--probe_json", default="", help="size probe (P4/P9): after --max_steps steps + one holdout eval, "
                                                    "time fp32 inference, write tok/s, memory and the estimated arm "
                                                    "time to this JSON and exit (no predictions)")
    p.add_argument("--probe_fp32_feeds", type=int, default=2000, help="holdout feeds timed in fp32 by the size probe")
    p.add_argument("--probe_gate_mem_gb", type=float, default=200.0)
    p.add_argument("--probe_gate_hours", type=float, default=10.0)
    p.add_argument("--arm_eval_every", type=int, default=200,
                   help="size probe: eval_every of the FULL arm (the probe itself runs --eval_every 40)")
    return p.parse_args(argv)


class Log:
    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def inputs_only(b):
    """The model and the dense features see only INPUT_KEYS (no labels, no production positions)."""
    return {k: v for k, v in b.items() if k in INPUT_KEYS}


@contextlib.contextmanager
def full_fp32():
    """True fp32 matmuls/convolutions (TF32 off) for the I3 fp32 inference path; restores the flags afterwards."""
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def make_batches(model, store, local_ids, token_budget, shuffle=True, rng=None):
    """Token-budget batches: Qwen3 rows -> FeedStore.token_batches (unchanged); left-padded Qwen3.5 branch rows ->
    FeedStore.token_batches_left (same budget, counted in the branch layout's own padded rows; reports/llm_large.md)."""
    if getattr(model, "left_padded", False):
        return store.token_batches_left(local_ids, token_budget, shuffle=shuffle, rng=rng, n_pre=model.n_pre)
    return store.token_batches(local_ids, token_budget, shuffle=shuffle, rng=rng, n_pre=model.n_pre)


@torch.no_grad()
def predict_store(model, store, local_ids, token_budget, amp):
    was = model.training
    model.eval()
    out = torch.zeros(store.n_cand, dtype=torch.float32, device=store.device)
    for fb in make_batches(model, store, local_ids, token_budget, shuffle=False):
        b = store.gather(fb)
        with torch.autocast(store.device.type, dtype=torch.bfloat16, enabled=amp):
            lg = model(inputs_only(b))
        store.scatter(out, b, lg)
    model.train(was)
    return out.cpu().numpy()


def subset_ndcg(scores, store, local_ids):
    rows = store.rows_of(local_ids)
    M = store.M_np[local_ids]
    offs = np.concatenate([[0], np.cumsum(M)[:-1]])
    return D.ndcg(scores[rows], store.target_np[rows], offs, M), rows


def within_feed_ties(s, M):
    """Number of candidates whose score equals another candidate's score in the same feed (flat s, feed sizes M)."""
    fid = np.repeat(np.arange(len(M)), M)
    o = np.lexsort((s, fid))
    same = (fid[o][1:] == fid[o][:-1]) & (s[o][1:] == s[o][:-1])
    return int(same.sum())


def precision_check(model, store, local_ids, token_budget, amp, threshold, log):
    """I3: holdout NDCG of the run's inference path (bf16 frozen weights + autocast when amp) vs the same trained model
    with fp32 trunk weights and TF32 off. Returns the result dict (final_fp32 = |dNDCG| > threshold)."""
    t0 = time.time()
    lo = predict_store(model, store, local_ids, token_budget, amp)
    m32 = model.fp32_copy().to(store.device)
    with full_fp32():
        hi = predict_store(m32, store, local_ids, token_budget, False)
    del m32
    nd_lo, rows = subset_ndcg(lo, store, local_ids)
    nd_hi = subset_ndcg(hi, store, local_ids)[0]
    M = store.M_np[local_ids]
    offs = np.concatenate([[0], np.cumsum(M)[:-1]])
    y = store.target_np[rows]
    d = D.ndcg_per_feed(lo[rows], y, offs, M) - D.ndcg_per_feed(hi[rows], y, offs, M)
    a, b = lo[rows].astype(np.float64), hi[rows].astype(np.float64)
    res = dict(inference_path="bf16 weights + bf16 autocast" if amp else "fp32 (amp off: trivially equal)",
               feeds=int(len(local_ids)), ndcg_inference_path=float(nd_lo), ndcg_fp32=float(nd_hi),
               delta=float(nd_lo - nd_hi), delta_se=float(d.std() / np.sqrt(len(d))), threshold=threshold,
               final_fp32=bool(abs(nd_lo - nd_hi) > threshold), score_corr=float(np.corrcoef(a, b)[0, 1]),
               max_abs_score_diff=float(np.abs(a - b).max()), score_std=float(b.std()),
               tied_cands_inference_path=within_feed_ties(lo[rows], M), tied_cands_fp32=within_feed_ties(hi[rows], M),
               n_cands=int(len(rows)), seconds=round(time.time() - t0, 1))
    log(f"I3 precision check: holdout NDCG {res['ndcg_inference_path']:.6f} ({res['inference_path']}) vs "
        f"{res['ndcg_fp32']:.6f} (fp32 trunk) -> delta {res['delta']:+.6f} +- {res['delta_se']:.6f}; "
        f"threshold {threshold:g} -> final_fp32 = {res['final_fp32']}; score corr {res['score_corr']:.5f}, "
        f"max|d| {res['max_abs_score_diff']:.4f} (score std {res['score_std']:.3f}); within-feed ties "
        f"{res['tied_cands_inference_path']} vs {res['tied_cands_fp32']} of {res['n_cands']}")
    return res


def main(argv=None):
    a = parse(argv)
    t_start = time.time()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    for sub in ("preds/holdout", "ckpt", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    log = Log(out / "logs" / f"{a.variant}.log")
    log(f"==== {time.ctime()} args: {vars(a)}")
    dev = torch.device(a.device)
    if dev.type == "cuda":
        assert os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(str(i) for i in range(8)), "exactly one GPU id 0-7 per process (some GPUs authorised by the user 2026-11-12)"
        torch.backends.cuda.matmul.allow_tf32 = True
    assert not a.cpu_bf16 or (a.smoke and dev.type == "cpu"), "--cpu_bf16 is a CPU smoke-test option only"
    use_bf16 = a.dtype == "bf16" and (dev.type == "cuda" or a.cpu_bf16)
    amp = use_bf16  # bf16 frozen trunk weights + bf16 autocast (fp32 LoRA/norms/heads)
    names, rel, idcols, relonly, n_dense = D.xt_config(a.xt)
    # ---------------------------------------------------------------- data (exactly nn/train.py's subset)
    M_all, ts_all = D.train_meta()
    prand = D.load_prandom(M_all)
    feeds, hold, rng = D.matched_subset(a.seed, a.train_feeds, prand, M_all, a.holdout)
    log(f"matched subset: seed {a.seed}: {len(feeds)} train feeds ({int((prand[feeds] > 0.5).sum())} with p_random>0.5), "
        f"holdout {len(hold)}")
    val_ids = None
    if a.smoke:
        sm = np.random.default_rng(a.seed + 99)
        feeds = np.sort(sm.choice(feeds, a.smoke, replace=False))
        hold = np.sort(sm.choice(hold, min(a.smoke, len(hold)), replace=False)) if len(hold) else hold
        val_ids = np.sort(sm.choice(14423, a.smoke, replace=False))
        log(f"SMOKE: {len(feeds)} train / {len(hold)} holdout / {len(val_ids)} val feeds")
    tr = D.FeedStore("train", np.union1d(feeds, hold), dev, a.max_hist, names)
    tr.set_domain(prand)
    tr_loc, ho_loc = tr.local(feeds), tr.local(hold)
    va = D.FeedStore("val", val_ids, dev, a.max_hist, names)
    va_all = np.arange(va.n)
    # ---------------------------------------------------------------- model
    Model = LLMRanker
    if a.backbone.startswith("qwen3.5"):  # hybrid DeltaNet trunk: only through the exact branch layout
        from qwen35_branch import Qwen35Ranker as Model
    model = Model(a.backbone, n_dense, rel, idcols, relonly, layout=a.layout, lora_r=a.lora_r,
                  random_init=a.random_init, dtype="bf16" if use_bf16 else "fp32",
                  attn_impl=a.attn, grad_ckpt=a.grad_ckpt, sink=not a.no_sink, sink_token=a.sink_token).to(dev)
    model.eval()
    mean, std = D.feature_stats(tr, tr_loc, dense_fn=model.dense)
    model.set_norm(mean, std)
    model.train()
    pc = count_params(model)
    log(f"backbone {a.backbone} ({'RANDOM INIT' if a.random_init else 'pretrained'}), emb_rms {model.emb_rms:.5f}, "
        f"params {pc}; S1 sink {'token id ' + str(model.sink_id) if model.sink else 'OFF'} "
        f"(rms {float(model.sink_emb.pow(2).mean().sqrt()):.5f}); V0 no_val_log {a.no_val_log}")
    trunk, proj = model.param_groups()
    lr_trunk = a.lr_full if a.random_init else a.lr_lora
    emb_like = ("cat.", "et.", "es.", "rec.", "dom.")
    groups = [  # LoRA / norms: no decay; random-init trunk: decay on matrices; projectors/heads: decay on matrices
        {"params": [p for n, p in trunk if a.random_init and p.ndim >= 2], "lr": lr_trunk, "weight_decay": a.wd},
        {"params": [p for n, p in trunk if not (a.random_init and p.ndim >= 2)], "lr": lr_trunk, "weight_decay": 0.0},
        {"params": [p for n, p in proj if p.ndim >= 2 and not n.startswith(emb_like)], "lr": a.lr_proj,
         "weight_decay": a.wd},
        {"params": [p for n, p in proj if not (p.ndim >= 2 and not n.startswith(emb_like))], "lr": a.lr_proj,
         "weight_decay": 0.0}]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8, fused=dev.type == "cuda")
    n_ep = math.ceil(a.epochs)
    ep_batches = [make_batches(model, tr, tr_loc, a.token_budget, shuffle=True, rng=rng) for _ in range(n_ep)]
    batches = [bb for ep in ep_batches for bb in ep][: int(len(ep_batches[0]) * a.epochs)]
    full_steps = len(batches)  # steps of the full arm (the size probe truncates with --max_steps)
    if a.max_steps:
        batches = batches[: a.max_steps]
    total = len(batches)
    warm = a.warmup if a.warmup >= 0 else (200 if a.random_init else 100)
    if a.smoke:
        warm = min(warm, max(1, total // 5))
    lam = lr_lambda_cosine(warm, total, a.floor)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lam] * len(groups))
    denom = max(1.0, a.token_budget / MEAN_T_TRAIN)  # constant: every feed has the same loss weight (--loss_norm feed)
    log(f"steps/epoch {len(ep_batches[0])}, total steps {total}, warmup {warm}, loss denom {denom:.1f}, "
        f"trainable {pc['trainable'] / 1e6:.2f}M (trunk group {sum(p.numel() for _, p in trunk) / 1e6:.2f}M)"
        + (f", feeds/step {len(tr_loc) / len(ep_batches[0]):.1f}, batching "
           f"{'token_batches_left (branch rows)' if getattr(model, 'left_padded', False) else 'token_batches'}, "
           f"grad_accum {a.grad_accum}" if (a.grad_accum > 1 or getattr(model, "left_padded", False)) else ""))
    t_setup = time.time() - t_start
    step_secs, eval_secs = [], []
    # ---------------------------------------------------------------- train
    best = (-1.0, None, -1)
    hist = []
    t0 = time.time()
    tok_real = tok_pad = 0
    run_loss, run_n = 0.0, 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_train = 0.0  # pure training seconds (evaluation excluded)
    peak_mem = peak_res = float("nan")
    for step, fb in enumerate(batches):
        ts = time.time()
        if a.grad_accum > 1:  # exact accumulation: per-feed listwise terms are additive (constant denom); the K9 term
            parts = [p_ for p_ in np.array_split(fb, a.grad_accum) if len(p_)]  # uses the FULL batch's weight sum
            fw_full = (1 - tr.prand[torch.from_numpy(np.asarray(fb, np.int64)).to(dev)]).sum().clamp_min(1e-6)
            opt.zero_grad(set_to_none=True)
        else:
            parts, fw_full = [fb], None
        loss_val = 0.0
        for part in parts:
            b = tr.gather(part, with_pos=True)
            bi = inputs_only(b)
            X = model.dense(bi)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                lg = model(bi, X)
            m = b["cmask"]
            loss = D.listwise_loss(lg, b["y"], m, None, norm="feed", denom=denom)
            if a.aux_pos > 0:  # K9: listwise CE toward softmax(-pos/tau), feed weight 1 - p_random
                tgt = torch.softmax((-b["pos"].float() / a.aux_tau).masked_fill(~m, -1e4), 1) * m
                lp = torch.log_softmax(model.aux_out.masked_fill(~m, -1e4), 1)
                fw = 1 - tr.prand[b["fidx"]]
                loss = loss + a.aux_pos * (-(tgt * lp).sum(1) * fw).sum() / (
                    fw.sum().clamp_min(1e-6) if fw_full is None else fw_full)
            if fw_full is None:
                opt.zero_grad(set_to_none=True)
            loss.backward()
            loss_val += loss.item()
            del b, bi, X, lg, loss
        gn = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], a.clip)
        opt.step()
        sched.step()
        Tl = tr.seq_len(fb, model.n_pre)
        tok_real += int(Tl.sum())
        tok_pad += tr.row_tokens(fb, model.n_pre) if getattr(model, "left_padded", False) else int(len(fb) * Tl.max())
        run_loss += loss_val
        run_n += 1
        if dev.type == "cuda" and step < 30:
            torch.cuda.synchronize()
        t_train += time.time() - ts
        step_secs.append(time.time() - ts)
        last = step == total - 1
        if a.smoke:
            log(f"step {step + 1}/{total} loss {loss_val:.5f} grad_norm {float(gn):.3f} "
                f"{(time.time() - t0) / (step + 1):.1f} s/step")
        if (step + 1) % a.eval_every == 0 or last:
            el = (time.time() - t0) / 60
            te = time.time()
            ps = predict_store(model, tr, ho_loc, a.eval_token_budget, amp) if len(ho_loc) else None
            ndh = subset_ndcg(ps, tr, ho_loc)[0] if ps is not None else float("nan")
            vtxt, ent = "", {}
            if not a.no_val_log:  # V0: Stage A never takes this branch
                pv = predict_store(model, va, va_all, a.eval_token_budget, amp)
                nd = D.ndcg(pv, va.target_np, va.cand_off_np, va.M_np)
                vtxt, ent = f" val_ndcg {nd:.5f}", {"val_ndcg": nd}
            mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else float("nan")
            peak_mem = mem  # max_memory_allocated is monotone (stats reset once, before step 1)
            peak_res = torch.cuda.max_memory_reserved() / 2**30 if dev.type == "cuda" else float("nan")
            eval_secs.append(time.time() - te)
            log(f"step {step + 1}/{total} loss {run_loss / max(run_n, 1):.4f} holdout_ndcg {ndh:.5f}{vtxt} "
                f"lr {sched.get_last_lr()[0]:.2e} {el:.1f} min, eval {time.time() - te:.0f}s, "
                f"train tok/s {tok_real / max(1e-9, t_train):.0f} real / {tok_pad / max(1e-9, t_train):.0f} padded "
                f"(pad eff {tok_real / max(tok_pad, 1):.3f}), "
                f"peak mem {mem:.1f} GB (reserved {peak_res:.1f})")
            hist.append(dict(step=step + 1, loss=run_loss / max(run_n, 1), holdout_ndcg=ndh, minutes=el, **ent))
            run_loss, run_n = 0.0, 0
            if ndh > best[0] or best[1] is None:
                best = (ndh, model.trainable_state(), step + 1)
            if a.time_limit and el > a.time_limit:
                log("time limit reached")
                break
    train_min = (time.time() - t0) / 60
    if a.probe_json:  # size probe: estimate the full arm on this lane, then exit without predictions
        cls, cfg, sd, n_pre = type(model), model.cfg, model.trainable_state(), model.n_pre
        del opt, sched, model, best
        return size_probe(a, cls, cfg, sd, n_pre, tr, ho_loc, va, va_all, dev, full_steps, step_secs, eval_secs,
                          t_setup, peak_mem, peak_res, tok_real, tok_pad, t_train, hist, log)
    # ---------------------------------------------------------------- holdout-best checkpoint -> predictions
    sd = best[1]
    missing = model.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    torch.save({"state": sd, "cfg": model.cfg, "args": vars(a), "step": best[2], "holdout_ndcg": best[0]},
               out / "ckpt" / f"{a.variant}.pt")
    i3 = None
    if a.precision_check and len(ho_loc):  # I3 (probe stage): bf16 vs fp32 trunk on the holdout
        i3 = precision_check(model, tr, ho_loc, a.eval_token_budget, amp, a.i3_threshold, log)
        (out / "i3_precision.json").write_text(json.dumps(dict(variant=a.variant, best_step=best[2], **i3), indent=2))
    amp_f, prec = amp, contextlib.nullcontext
    if a.final_fp32:  # I3 decision applied: every final prediction from the fp32-trunk copy, TF32 off
        cls, cfg = type(model), model.cfg
        del opt, sched, model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        model = fp32_from_state(cls, cfg, sd).to(dev)
        amp_f, prec = False, full_fp32
        log("I3: final predictions with fp32 trunk weights (TF32 off)")
    with prec():
        pv = predict_store(model, va, va_all, a.eval_token_budget, amp_f)
    nd = float("nan")
    if not a.no_val_log:  # V0: the trainer never scores val in Stage A (compare.py does, after all arms finish)
        nd = D.ndcg(pv, va.target_np, va.cand_off_np, va.M_np)
    if not a.smoke:
        np.save(out / "preds" / f"{a.variant}_trainonly_val.npy", pv.astype(np.float32))
    if len(ho_loc):
        with prec():
            ps = predict_store(model, tr, ho_loc, a.eval_token_budget, amp_f)
        ndh, rows = subset_ndcg(ps, tr, ho_loc)
        np.save(out / "preds" / "holdout" / f"{a.variant}_holdout.npy", ps[rows].astype(np.float32))
        if not a.smoke:
            M = tr.M_np[ho_loc]
            np.save(out / "preds" / "holdout" / "holdout_meta.npy", np.stack([M, np.concatenate([[0], np.cumsum(M)[:-1]])]))
            np.save(out / "preds" / "holdout" / "holdout_y.npy", tr.target_np[rows])
    else:
        ndh = float("nan")
    vtxt = "" if a.no_val_log else f" val_ndcg {nd:.5f}"
    log(f"BEST step {best[2]} holdout_ndcg {ndh:.5f}{vtxt}; train time {train_min:.1f} min; "
        f"pure train {t_train / 60:.1f} min, real tokens/s {tok_real / max(1e-9, t_train):.0f}, padded tokens/s "
        f"{tok_pad / max(1e-9, t_train):.0f}")
    if not a.no_test and not a.smoke:
        del tr
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        te = D.FeedStore("test", None, dev, a.max_hist, names)
        with prec():
            pt = predict_store(model, te, np.arange(te.n), a.eval_token_budget, amp_f)
        np.save(out / "preds" / f"{a.variant}_trainonly_test.npy", pt.astype(np.float32))
        log(f"saved test preds {pt.shape}")
    (out / "logs" / f"{a.variant}.json").write_text(json.dumps(dict(
        args=vars(a), best_step=best[2], total_steps=total, last_eval_step=hist[-1]["step"] if hist else None,
        holdout_ndcg=ndh, val_ndcg=None if a.no_val_log else nd, train_minutes=train_min,
        pure_train_minutes=t_train / 60, real_tokens=tok_real, padded_tokens=tok_pad,
        padded_tokens_per_s=tok_pad / max(1e-9, t_train), peak_mem_gb=peak_mem, params=pc,
        sink=bool(model.sink), final_fp32=bool(a.final_fp32), i3=i3, history=hist), indent=2))
    return dict(best_step=best[2], holdout_ndcg=ndh, val_ndcg=nd, hist=hist, i3=i3)


def size_probe(a, cls, cfg, sd, n_pre, tr, ho_loc, va, va_all, dev, full_steps, step_secs, eval_secs, t_setup,
               peak_mem, peak_res, tok_real, tok_pad, t_train, hist, log):
    """P4/P9 size probe (reports/llm_large.md): measured training step time, holdout-eval time and fp32-inference rate
    on this lane -> estimated wall-clock of the full arm (setup + all steps + all holdout evals + fp32 model load +
    fp32 final predictions on holdout/val/test). Gate: peak reserved memory <= --probe_gate_mem_gb AND estimated arm
    time <= --probe_gate_hours."""
    import polars as pl
    steady = step_secs[5:] if len(step_secs) > 10 else step_secs
    s_step = float(np.mean(steady))
    ev = a.arm_eval_every  # fix 04:20: was a.eval_every (= the probe's 40 -> 38 evals instead of the arm's 8)
    n_evals = full_steps // ev + (1 if full_steps % ev else 0)
    s_eval = float(eval_secs[-1]) if eval_secs else float("nan")
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    m32 = fp32_from_state(cls, cfg, sd).to(dev)
    s_load32 = time.time() - t1
    ids = np.sort(np.random.default_rng(a.seed).choice(ho_loc, min(a.probe_fp32_feeds, len(ho_loc)), replace=False))
    with full_fp32():
        predict_store(m32, tr, ids[:64], a.eval_token_budget, False)  # warm-up (kernels, allocator)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        predict_store(m32, tr, ids, a.eval_token_budget, False)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        s32 = time.time() - t1
    tok32 = int(tr.seq_len(ids, m32.n_pre).sum())
    mem32 = torch.cuda.max_memory_reserved() / 2**30 if dev.type == "cuda" else float("nan")
    ft = pl.read_parquet(D.FLAT / "test" / "feeds.parquet", columns=["M", "H"])
    real = dict(holdout=int(tr.seq_len(ho_loc, m32.n_pre).sum()), val=int(va.seq_len(va_all, m32.n_pre).sum()),
                test=int((m32.n_pre + np.minimum(ft["H"].to_numpy(), a.max_hist) + ft["M"].to_numpy()).sum()))
    s_fp32_all = s32 / tok32 * sum(real.values())
    est_s = 2 * t_setup + full_steps * s_step + n_evals * s_eval + s_load32 + s_fp32_all
    gate_mem = max(peak_res, mem32) <= a.probe_gate_mem_gb
    gate_time = est_s / 3600 <= a.probe_gate_hours
    res = dict(variant=a.variant, backbone=a.backbone, gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
               grad_accum=a.grad_accum, token_budget=a.token_budget, probe_steps=len(step_secs),
               full_steps=full_steps, n_evals=n_evals, setup_s=round(t_setup, 1), s_per_step=round(s_step, 3),
               train_real_tok_s=tok_real / max(t_train, 1e-9), train_padded_tok_s=tok_pad / max(t_train, 1e-9),
               eval_s=round(s_eval, 1), fp32_load_s=round(s_load32, 1), fp32_real_tok_s=tok32 / max(s32, 1e-9),
               fp32_tokens_final=real, fp32_final_s=round(s_fp32_all, 1),
               est_hours=dict(train=full_steps * s_step / 3600, evals=n_evals * s_eval / 3600,
                              fp32_final=(s_load32 + s_fp32_all) / 3600, setup=2 * t_setup / 3600,
                              total=est_s / 3600),
               peak_alloc_train_gb=peak_mem, peak_reserved_train_gb=peak_res, peak_reserved_fp32_gb=mem32,
               holdout_ndcg_probe=hist[-1]["holdout_ndcg"] if hist else None,
               gate=dict(mem_gb=a.probe_gate_mem_gb, hours=a.probe_gate_hours, mem_ok=bool(gate_mem),
                         time_ok=bool(gate_time), PASS=bool(gate_mem and gate_time)),
               utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    Path(a.probe_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.probe_json).write_text(json.dumps(res, indent=2))
    log(f"SIZE PROBE {a.backbone}: {s_step:.2f} s/step ({res['train_padded_tok_s']:.0f} padded / "
        f"{res['train_real_tok_s']:.0f} real tok/s) x {full_steps} steps = {res['est_hours']['train']:.2f} h; "
        f"{n_evals} holdout evals x {s_eval:.0f} s = {res['est_hours']['evals']:.2f} h; fp32 final "
        f"{res['fp32_real_tok_s']:.0f} real tok/s -> {res['est_hours']['fp32_final']:.2f} h; setup "
        f"{res['est_hours']['setup']:.2f} h; TOTAL {res['est_hours']['total']:.2f} h (gate <= {a.probe_gate_hours:g}); "
        f"peak reserved train {peak_res:.1f} GB / fp32 {mem32:.1f} GB (gate <= {a.probe_gate_mem_gb:g}) -> "
        f"SIZE PROBE GATE {'PASS' if res['gate']['PASS'] else 'FAIL'}")
    return res


if __name__ == "__main__":
    main()
