"""Reviewer GPU smoke test of the Qwen3.5 branch trunk (nn_x/llm/qwen35_branch.py) with the TRAINED Q-arm checkpoints.
Tiny batches (6 val feeds + 4 holdout feeds). Read-only w.r.t. the project (writes only to argv[2] in the scratchpad).

  CUDA_VISIBLE_DEVICES=<gpu> python gpu_leak_test.py <qwen3.5-4b|qwen3.5-9b> <out.json>

G1 permutation equivariance within a feed (fp32 trunk, TF32 off; and bf16 trunk + bf16 autocast = the training path)
G2 labels / production positions never read: keys accessed by dense()+forward(); flipped y + reversed pos -> same scores
G3 cross-feed isolation: perturb every input of one feed in the batch -> other feeds' scores unchanged; feed alone vs in batch
G4 other-candidate isolation (L1): perturb candidate j's token inputs (with X fixed) -> the other candidates unchanged;
   positive control: the same perturbation under the L2 mask DOES move the other candidates (test can detect leakage)
G5 branch == HF Qwen3_5TextModel forward over the unpadded [sink, user, history, c] (last token), left-padded batch
G6 the saved prediction files are reproduced by the saved holdout-selected checkpoint (val + holdout feeds, fp32 path)
"""
import contextlib
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "<repo>/nn_x/llm")
import data_llm as D  # noqa: E402
from model_llm import INPUT_KEYS, fp32_from_state  # noqa: E402
from qwen35_branch import Qwen35Ranker, left_layout  # noqa: E402

BB = sys.argv[1]
OUT = sys.argv[2]
TAG = {"qwen3.5-4b": "llmq4_L1_s23", "qwen3.5-9b": "llmq9_L1_s23"}[BB]
LLM = "<repo>/nn_x/llm"
torch.set_num_threads(12)
dev = torch.device("cuda")
RES = {"backbone": BB, "ckpt": TAG}


def rec(name, ok, **kw):
    RES[name] = dict(ok=bool(ok), **kw)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(kw, default=str)}", flush=True)


@contextlib.contextmanager
def fp32_exact():
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def clone(b):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}


def inp(b):
    return {k: v for k, v in b.items() if k in INPUT_KEYS}


class Spy(dict):
    seen = set()

    def __getitem__(self, k):
        Spy.seen.add(k)
        return super().__getitem__(k)

    def get(self, k, d=None):
        Spy.seen.add(k)
        return super().get(k, d)


CK = torch.load(f"{LLM}/ckpt/{TAG}.pt", map_location="cpu", weights_only=False)
cfg, sd = CK["cfg"], CK["state"]
names, rel, idc, relonly, nd = D.xt_config("cvnd_pre2s")
assert cfg["xt_rel"] == rel and cfg["n_dense"] == nd
VAL_IDS = np.array([137, 107, 12, 1, 2, 319])  # H = 0, 1, 40, >=256, 168, 0 (left padding exercised)
va = D.FeedStore("val", VAL_IDS, dev, 256, names, verbose=False)
t0 = time.time()
m32 = fp32_from_state(Qwen35Ranker, cfg, sd).to(dev).eval()
mbf = Qwen35Ranker(**cfg).to(dev)
r = mbf.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=False)
assert not r.unexpected_keys
mbf.eval()
RES["load_s"] = round(time.time() - t0, 1)
lora_B_abs = float(np.mean([v.abs().mean().item() for k, v in sd.items() if "lora_B" in k]))
print(f"loaded {BB} fp32 + bf16 in {RES['load_s']} s; ckpt step {CK['step']}, mean|lora_B| {lora_B_abs:.2e}", flush=True)
RES["ckpt_step"], RES["mean_abs_lora_B"] = CK["step"], lora_B_abs
local = np.arange(va.n)
b = va.gather(local, with_pos=True)
if "pos" not in b:  # val has no production positions on disk: add a synthetic one (the model must never read it)
    b["pos"] = torch.arange(b["cmask"].shape[1], device=dev)[None].expand_as(b["cmask"]).long().clone()
assert "y" in b and "pos" in b


def run(model, bb, X=None, amp=False):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        return model(bb, X).float()


PATHS = {"fp32": (m32, False, fp32_exact), "bf16": (mbf, True, contextlib.nullcontext)}
for pname, (model, amp, ctx) in PATHS.items():
    with ctx():
        s0 = run(model, inp(clone(b)), amp=amp)
        sstd = float(s0[b["cmask"]].std())
        # ---- G1 permutation equivariance
        rng = np.random.default_rng(1)
        b2 = clone(b)
        perms = []
        for i in range(va.n):
            m = int(b["M"][i])
            p = torch.from_numpy(np.concatenate([rng.permutation(m), np.arange(m, b["cmask"].shape[1])])).to(dev)
            perms.append(p)
            for k in ("cmask", "ci", "ce_raw", "cmiss", "ccat", "cf", "y", "xt", "pos"):
                b2[k][i] = b[k][i][p]
        s2 = run(model, inp(b2), amp=amp)
        e1 = 0.0
        for i, p in enumerate(perms):
            m = int(b["M"][i])
            inv = torch.argsort(p[:m])
            e1 = max(e1, float((s2[i, :m][inv] - s0[i, :m]).abs().max()))
        tol = 1e-3 if pname == "fp32" else 0.05 * sstd
        rec(f"G1_perm_equivariance_{pname}", e1 < tol, max_abs_err=e1, score_std=sstd, tol=tol)
        # ---- G2 labels / positions never read
        Spy.seen = set()
        sp = Spy(clone(b))
        Xs = model.dense(sp)
        run(model, sp, Xs, amp=amp)
        read = sorted(Spy.seen)
        b3 = clone(b)
        b3["y"] = (1 - b3["y"]) * b3["cmask"]
        b3["pos"] = b3["pos"].flip(1)
        s3 = run(model, b3, amp=amp)  # full dict, labels + positions perturbed, NOT filtered by inputs_only
        s3b = run(model, clone(b), amp=amp)
        nf = float((run(model, clone(b), amp=amp) - s3b).abs().max())  # repeat-call noise floor (determinism)
        d2 = float((s3 - s3b).abs().max())
        RES[f"noise_floor_{pname}"] = nf
        rec(f"G2_labels_positions_not_read_{pname}", ("y" not in read and "pos" not in read and d2 <= nf),
            keys_read=read, max_diff_flipped_labels_reversed_pos=d2, repeat_noise_floor=nf)
        # ---- G3 cross-feed isolation: perturb all inputs of feed k
        worst = 0.0
        for k in range(va.n):
            b4 = clone(b)
            g = torch.Generator(device=dev).manual_seed(k)
            for key in ("ce_raw", "he_raw", "cf", "xt"):
                b4[key][k] = torch.randn(b4[key][k].shape, device=dev, generator=g).to(b4[key].dtype) * 3
            b4["ue_raw"][k] = torch.randn(b4["ue_raw"][k].shape, device=dev, generator=g)
            b4["ccat"][k] = (b4["ccat"][k] + 7) % 54
            b4["hcat"][k] = (b4["hcat"][k] + 5) % 54
            s4 = run(model, inp(b4), amp=amp)
            others = torch.ones(va.n, dtype=torch.bool, device=dev)
            others[k] = False
            mk = b["cmask"] & others[:, None]
            worst = max(worst, float((s4 - s0)[mk].abs().max()))
            assert float((s4 - s0)[k][b["cmask"][k]].abs().max()) > 1e-3  # the perturbed feed itself must change
        alone = 0.0
        for k in range(va.n):
            sa = run(model, inp(va.gather([k])), amp=amp)
            m = int(b["M"][k])
            alone = max(alone, float((sa[0, :m] - s0[k, :m]).abs().max()))
        tol3 = 1e-3 if pname == "fp32" else 0.05 * sstd
        rec(f"G3_cross_feed_isolation_{pname}", worst <= 1e-5 * max(1.0, sstd) and alone < tol3,
            max_abs_change_other_feeds=worst, max_abs_feed_alone_vs_in_batch=alone, tol_alone=tol3)
        # ---- G4 other-candidate isolation (L1) + L2 positive control
        bi = inp(clone(b))
        X = model.dense(bi)
        s_ref = run(model, bi, X, amp=amp)
        worst4, self4, l2_other = 0.0, 0.0, 0.0
        for f in range(va.n):
            m = int(b["M"][f])
            j = m // 2
            bj, Xj = clone(bi), X.clone()
            g = torch.Generator(device=dev).manual_seed(100 + f)
            v = torch.randn(bj["ce"].shape[-1], device=dev, generator=g)
            bj["ce"][f, j] = v / v.norm()
            bj["ccat"][f, j] = (bj["ccat"][f, j] + 11) % 54
            Xj[f, j] = torch.randn(Xj.shape[-1], device=dev, generator=g) * 3
            sj = run(model, bj, Xj, amp=amp)
            mask_o = b["cmask"].clone()
            mask_o[f, j] = False
            worst4 = max(worst4, float((sj - s_ref)[mask_o].abs().max()))
            self4 = max(self4, float((sj - s_ref)[f, j].abs()))
            if pname == "fp32":
                model.cfg["layout"] = "L2"
                try:
                    r2 = run(model, bi, X, amp=amp)
                    r2j = run(model, bj, Xj, amp=amp)
                finally:
                    model.cfg["layout"] = "L1"
                mo = b["cmask"][f].clone()
                mo[j] = False
                l2_other = max(l2_other, float((r2j - r2)[f][mo].abs().max()))
        ok4 = worst4 <= 1e-5 * max(1.0, sstd) and self4 > 1e-3 and (pname != "fp32" or l2_other > 1e-3)
        rec(f"G4_other_candidate_isolation_L1_{pname}", ok4, max_abs_change_other_candidates=worst4,
            max_abs_change_perturbed_candidate=self4,
            **({"L2_positive_control_max_change_other_candidates": l2_other} if pname == "fp32" else {}))

# ---- G5 branch == HF forward (fp32, TF32 off), trained LoRA active
with fp32_exact(), torch.no_grad():
    bi = inp(clone(b))
    X = m32.dense(bi)
    u, h, c, Xn = m32.tokens(bi, X)
    P, T, pos, mask, _, _ = left_layout(bi["H"], bi["M"], c.shape[1], "L1", m32.n_pre)
    seq = m32.assemble_left(u, h, c, bi["H"], bi["M"])
    hb = m32.trunk(seq, pos, mask, P)[:, P:]
    errs, Hs = [], []
    for fi in range(va.n):
        H = int(bi["H"][fi])
        Hs.append(H)
        hist = h[fi, :H].flip(0)
        for m in sorted({0, int(bi["M"][fi]) // 2, int(bi["M"][fi]) - 1}):
            s1 = torch.cat([m32.sink_emb[None].to(u.dtype), u[fi:fi + 1], hist, c[fi, m:m + 1]])[None]
            ref = m32.lm(inputs_embeds=s1).last_hidden_state[0, -1]
            errs.append(float((ref - hb[fi, m]).abs().max() / ref.abs().max()))
rec("G5_branch_equals_HF_forward_fp32", max(errs) < 1e-4, max_rel_err=max(errs), n_checks=len(errs), history_lengths=Hs)

# ---- G6 saved predictions reproduced by the saved checkpoint (fp32 path = I3 final predictions)
with fp32_exact(), torch.no_grad():
    sv = run(m32, inp(clone(b)))
    pv = np.load(f"{LLM}/preds/{TAG}_trainonly_val.npy")
    e_val, rel_v = 0.0, []
    for i in range(va.n):
        m = int(b["M"][i])
        rows = b["ci"][i, :m].cpu().numpy()
        orig = va.orig_rows[rows]
        e_val = max(e_val, float(np.abs(pv[orig] - sv[i, :m].cpu().numpy()).max()))
    # holdout: 4 holdout feeds, train store with the domain mixture as train_llm (tr.set_domain(prand))
    M_all, _ = D.train_meta()
    pr = D.load_prandom(M_all)
    hold = D.holdout_feeds(pr, M_all)
    pick = np.sort(np.random.default_rng(3).choice(len(hold), 4, replace=False))
    st = D.FeedStore("train", hold[pick], dev, 256, names, verbose=False)
    st.set_domain(pr)
    bh = st.gather(np.arange(st.n))
    sh = run(m32, inp(bh))
    ph = np.load(f"{LLM}/preds/holdout/{TAG}_holdout.npy")
    Mh = M_all[hold]
    offh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
    e_ho = 0.0
    for i, hi in enumerate(pick):
        m = int(bh["M"][i])
        e_ho = max(e_ho, float(np.abs(ph[offh[hi]:offh[hi] + m] - sh[i, :m].cpu().numpy()).max()))
    sstd = float(sv[b["cmask"]].std())
rec("G6_saved_preds_reproduced_by_ckpt", e_val < 1e-3 and e_ho < 1e-3, max_abs_diff_val=e_val,
    max_abs_diff_holdout=e_ho, score_std=sstd)
RES["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
RES["all_pass"] = all(v["ok"] for v in RES.values() if isinstance(v, dict) and "ok" in v)
print("ALL PASS" if RES["all_pass"] else "SOME FAILED", f"peak {RES['peak_mem_gb']:.1f} GB", flush=True)
json.dump(RES, open(OUT, "w"), indent=1, default=str)
