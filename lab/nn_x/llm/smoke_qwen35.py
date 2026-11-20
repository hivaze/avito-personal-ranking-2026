"""CPU smoke test of the Qwen3.5 branch trunk (qwen35_branch.py) with the local Qwen3.5-2B. NO GPU.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 ../../.venv_llm/bin/python smoke_qwen35.py --out <dir>

Q1 naive layout (candidates appended, the L1 4D mask given to the attention layers, DeltaNet layers causal as HF runs
   them): candidate scores CHANGE under a candidate permutation -> a custom mask alone cannot make Qwen3.5 order-free.
Q2 branch layout: permutation-equivariant.
Q3 branch == HF: a candidate's final hidden state equals HF Qwen3_5TextModel's own forward over the unpadded sequence
   [sink, user, h_old..h_new, c] (last token), for feeds of different history lengths inside one left-padded batch.
   (S1 amendment 2026-11-12: the frozen sink = the checkpoint's own <|im_start|> row, id 248045 for Qwen3.5.)
Q4 one AdamW step lowers the loss (LoRA on attention, MLP and DeltaNet projections); CPU time per step.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import data_llm as D
from model_llm import INPUT_KEYS, build_layout, count_params
from qwen35_branch import Qwen35Ranker
from smoke_test import clone_b, inputs_only, permute_cands

RES = {}


def rec(name, ok, **kw):
    RES[name] = dict(ok=bool(ok), **kw)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(kw, default=str)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="qwen3.5-2b")
    a = ap.parse_args()
    torch.set_num_threads(12)
    torch.manual_seed(0)
    assert not torch.cuda.is_available()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names, rel, idc, relonly, nd = D.xt_config("cvnd_pre2s")
    va = D.FeedStore("val", np.sort(np.random.default_rng(5).choice(14423, 12, replace=False)), "cpu",
                     xt_names=names, verbose=False)
    t = time.time()
    model = Qwen35Ranker(a.backbone, nd, rel, idc, relonly, layout="L1")
    model.eval()
    mean, std = D.feature_stats(va, np.arange(va.n), model.dense, n_batches=2)
    model.set_norm(mean, std)
    pc = count_params(model)
    n_lora = sum(1 for n, _ in model.named_parameters() if "lora_A" in n)
    print(f"loaded {a.backbone} in {time.time() - t:.0f}s, emb_rms {model.emb_rms:.5f}, params {pc}, lora modules {n_lora}",
          flush=True)
    RES["model"] = dict(params=pc, lora_modules=n_lora, emb_rms=model.emb_rms)
    lm = model.lm
    rng = np.random.default_rng(6)
    lo = np.arange(3)
    b = va.gather(lo)

    # ---------------- Q1 naive causal append (what plain HF does with a custom mask)
    @torch.no_grad()
    def naive(bb):
        bi = inputs_only(clone_b(bb))
        X = model.dense(bi)
        u, h, c, Xn = model.tokens(bi, X)
        seq, Tx = model.assemble(u, h, c, bi["H"], bi["M"])
        pos, mask, ct, _, _ = build_layout(bi["H"], bi["M"], Tx, "L1", None, c.shape[1], model.n_pre)
        o = lm(inputs_embeds=seq, attention_mask={"full_attention": mask, "linear_attention": None},
               position_ids=pos).last_hidden_state
        hc = o.gather(1, ct[..., None].expand(-1, -1, o.shape[-1])).float()
        return model.head(hc).squeeze(-1) + model.wide(Xn).squeeze(-1)

    def perm_err(fn):
        s = fn(b)
        b2, perms = permute_cands(b, rng)
        s2 = fn(b2)
        e = []
        for i, p in enumerate(perms):
            m = int(b["M"][i])
            e.append(float((s2[i, :m][torch.from_numpy(np.argsort(p[:m]))] - s[i, :m]).abs().max()))
        return max(e)
    t = time.time()
    e_naive = perm_err(naive)
    rec("Q1_naive_append_is_order_dependent", e_naive > 1e-3, max_abs_score_change_under_permutation=e_naive,
        seconds=round(time.time() - t, 1))
    # ---------------- Q2 branch permutation equivariance
    t = time.time()
    with torch.no_grad():
        e_branch = perm_err(lambda bb: model(inputs_only(clone_b(bb))))
    rec("Q2_branch_permutation_equivariant", e_branch < 1e-3, max_abs_err=e_branch, seconds=round(time.time() - t, 1))
    # ---------------- Q3 branch == HF forward over [user, history, c]
    with torch.no_grad():
        bi = inputs_only(clone_b(b))
        X = model.dense(bi)
        u, h, c, Xn = model.tokens(bi, X)
        from qwen35_branch import left_layout
        P, T, pos, mask, _, _ = left_layout(bi["H"], bi["M"], c.shape[1], "L1", model.n_pre)
        seq = model.assemble_left(u, h, c, bi["H"], bi["M"])
        hb = model.trunk(seq, pos, mask, P)[:, P:]
        errs, Hs = [], []
        for fi in range(len(lo)):
            H = int(bi["H"][fi])
            Hs.append(H)
            hist = h[fi, :H].flip(0)  # newest-first -> old -> new
            for m in (0, int(bi["M"][fi]) // 2, int(bi["M"][fi]) - 1):
                s1 = torch.cat(([model.sink_emb[None]] if model.sink else []) + [u[fi:fi + 1], hist, c[fi, m:m + 1]])[None]
                ref = lm(inputs_embeds=s1).last_hidden_state[0, -1]
                errs.append(float((ref - hb[fi, m]).abs().max() / ref.abs().max()))
    rec("Q3_branch_equals_HF_forward", max(errs) < 1e-4, max_rel_err=max(errs), history_lengths=Hs,
        sink_id=model.sink_id, n_pre=model.n_pre,
        note="HF reference = unpadded [sink, user, history, c] through Qwen3_5TextModel.forward (chunked delta rule)")
    # ---------------- Q4 optimiser step
    trunk, proj = model.param_groups()
    opt = torch.optim.AdamW([{"params": [p for _, p in trunk], "lr": 2e-4}, {"params": [p for _, p in proj], "lr": 1e-3}])

    def loss_of(bb, train):
        model.train(train)
        bi = inputs_only(clone_b(bb))
        lg = model(bi, model.dense(bi))
        return D.listwise_loss(lg, bb["y"], bb["cmask"], None, norm="feed", denom=len(bb["M"]))
    b6 = va.gather(np.arange(6))
    with torch.no_grad():
        l0 = float(loss_of(b6, False))
    t = time.time()
    loss = loss_of(b6, True)
    opt.zero_grad()
    loss.backward()
    opt.step()
    st = time.time() - t
    with torch.no_grad():
        l1 = float(loss_of(b6, False))
    T_real = int(va.seq_len(np.arange(6)).sum())
    rec("Q4_optimiser_step", l1 < l0, eval_loss_before=round(l0, 5), eval_loss_after=round(l1, 5),
        step_seconds_6_feeds=round(st, 1), real_tokens=T_real,
        train_TFLOPs_per_s_4N=round(4 * pc["body"] * T_real / st / 1e12, 3))
    (out / "smoke_qwen35_results.json").write_text(json.dumps(RES, indent=2, default=str))
    print("ALL PASS" if all(v.get("ok", True) for k, v in RES.items() if k != "model") else "SOME FAILED")


if __name__ == "__main__":
    main()
