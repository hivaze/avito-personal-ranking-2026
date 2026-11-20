"""CPU smoke tests of the soft-token LLM ranker (Qwen3-0.6B). NO GPU: run with CUDA_VISIBLE_DEVICES="".

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 ../../.venv_llm/bin/python smoke_test.py --out <dir>

Checks (all on a few dozen real feeds):
  T1 preset parity with nn/train.py, matched-subset sizes (seed 23/24)
  T2 FeedStore == nn/data.SplitData (gather tensors and dense features bit-identical, val)
  T3 model build: LoRA B == 0 at init, trainable/frozen split, emb RMS, fp32 heads
  T4 permutation equivariance over candidates (L1 and L2)
  T5 batched == single: (a) a candidate scored in the full L1 pass == scored alone as [user, history, c];
     (b) a feed scored inside a padded multi-feed batch == scored alone
  T6 no leakage: labels/positions never read (full dict == inputs-only dict; perturbed y/pos -> same scores);
     feed A unaffected by feed B; in L1 candidate i's hidden state unaffected by candidate j's token
  T7 one AdamW step lowers the loss on the batch (eval mode before/after); frozen base weights unchanged;
     tiny overfit set: loss falls over a few steps
  T8 CPU throughput (forward, forward+backward with/without checkpointing) vs the 2N/4N/6N FLOP model
  T9 random-init control: builds, one step runs; its sink is its own init row
Amendment checks (2026-11-12; T3-T7 above all run WITH the S1 sink token):
  T10 S1 sink: frozen buffer == the checkpoint's <|im_start|> row (id 151644), not trainable, unchanged by the
      optimiser (T7); user-token cosine across users at layer 13 < 0.99 with the sink, on 24 fresh val feeds (276
      pairs); contrast rows: no sink, and the originally declared <|endoftext|> row; massive-token counts
  T11 HF forward parity (review R1): batched L1 score == HF NATIVE causal forward (default mask and positions) over the
      unpadded [sink, user, h_old..h_new, c], feeds with H=0 / H=256 / mixed, non-zero LoRA B
  T12 I3 path: fp32_copy() reproduces the trained fp32 model exactly (trainable state + sink carried); bf16 frozen
      weights + CPU bf16 autocast vs the fp32-trunk copy through train_llm.precision_check (CPU stand-in for the GPU
      probe; the real decision is made on the GPU probe)
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

import data_llm as D
from model_llm import INPUT_KEYS, LLMRanker, count_params

RES = {}


def rec(name, ok, **kw):
    RES[name] = dict(ok=bool(ok), **kw)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(kw, default=str)}", flush=True)


def inputs_only(b):
    return {k: v for k, v in b.items() if k in INPUT_KEYS}


def clone_b(b):
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in b.items()}


CAND_KEYS = ("cmask", "ci", "ce_raw", "cmiss", "ccat", "cf", "y", "xt", "pos")


def permute_cands(b, rng):
    """Random permutation of the valid candidate slots of every feed (padding slots stay in place)."""
    b2 = clone_b(b)
    B, Mx = b["cmask"].shape
    perms = []
    for i in range(B):
        m = int(b["M"][i])
        p = np.concatenate([rng.permutation(m), np.arange(m, Mx)])
        perms.append(p)
        pt = torch.from_numpy(p)
        for k in CAND_KEYS:
            if k in b2:
                b2[k][i] = b[k][i][pt]
    return b2, perms


@torch.no_grad()
def scores(model, b, X=None):
    model.eval()
    return model(inputs_only(clone_b(b)) if X is None else inputs_only(b), X)


def single_candidate_scores(model, b, X, fi, cand_ids):
    """Score candidates of feed fi one at a time as [user, history, c] (M=1), reusing the full-feed tokens/X."""
    model.eval()
    bi = inputs_only(clone_b(b))
    with torch.no_grad():
        Xf = model.dense(bi)  # also fills bi["ce"/"he"/"un"]
        assert X is None or torch.equal(Xf, X)
        u, h, c, Xn = model.tokens(bi, Xf)
        out = []
        for m in cand_ids:
            Hc, M1 = bi["H"][fi:fi + 1], torch.ones(1, dtype=torch.long)
            seq, Tx = model.assemble(u[fi:fi + 1], h[fi:fi + 1], c[fi:fi + 1, m:m + 1], Hc, M1)
            pos, mask, ct, _, _ = model.layout_of(Hc, M1, Tx, 1, layout="L1")
            o = model.lm(inputs_embeds=seq, attention_mask={"full_attention": mask}, position_ids=pos).last_hidden_state
            hc = o[0, int(ct[0, 0])].float()
            out.append(float(model.head(hc[None]).squeeze() + model.wide(Xn[fi:fi + 1, m]).squeeze()))
    return np.array(out)


def checkpoint_row(repo, row):
    """One row of the checkpoint's embedding matrix read straight from the safetensors file (independent of HF)."""
    import glob
    from safetensors import safe_open
    name = repo.replace("/", "--")
    for f in sorted(glob.glob(f"~/.cache/huggingface/hub/models--{name}/snapshots/*/*.safetensors")):
        with safe_open(f, "pt") as g:
            for k in g.keys():
                if k.endswith("embed_tokens.weight"):
                    return g.get_slice(k)[row:row + 1][0].float()
    raise FileNotFoundError(repo)


@torch.no_grad()
def user_token_states(model, b, layers=(2, 6, 13, 27)):
    """Hidden states of the user token (and the sink) per layer for a batch of feeds, current sink setting."""
    model.eval()
    bi = inputs_only(clone_b(b))
    X = model.dense(bi)
    u, h, c, _ = model.tokens(bi, X)
    seq, Tx = model.assemble(u, h, c, bi["H"], bi["M"])
    pos, mask, _, _, _ = model.layout_of(bi["H"], bi["M"], Tx, c.shape[1])
    o = model.lm(inputs_embeds=seq, attention_mask={"full_attention": mask}, position_ids=pos,
                 output_hidden_states=True)
    ut = model.n_pre - 1
    res = {}
    for li in layers:
        hs = o.hidden_states[li]
        U = hs[:, ut].float()
        Un = torch.nn.functional.normalize(U, dim=-1)
        cos = Un @ Un.T
        iu = torch.triu_indices(len(U), len(U), 1)
        hist_max = [float(hs[f, model.n_pre:model.n_pre + int(bi["H"][f])].abs().amax(-1).median())
                    for f in range(len(U)) if int(bi["H"][f]) > 0]
        n_other = [int((hs[f, model.n_pre:model.n_pre + int(bi["H"][f]) + int(bi["M"][f])].abs().amax(-1) > 300).sum())
                   for f in range(len(U))]
        res[li] = dict(user_cos_max=float(cos[iu[0], iu[1]].max()), user_cos_mean=float(cos[iu[0], iu[1]].mean()),
                       user_max_abs_h=float(U.abs().amax(-1).max()),
                       users_massive=f"{int((U.abs().amax(-1) > 300).sum())}/{len(U)}",
                       other_tokens_massive=f"{sum(n_other)}/{int((bi['H'] + bi['M']).sum())}",
                       sink_max_abs_h=float(hs[:, 0].abs().amax(-1).max()) if model.sink else None,
                       history_median_max_abs_h=float(np.median(hist_max)))
    return res


def t10(model, va):
    """S1: sink row identity + the user token no longer content-independent (layer-13 cosine across users), on fresh
    feeds (the sink row was chosen on other feeds); contrasts: no sink, and the declared <|endoftext|> row."""
    ref = checkpoint_row("Qwen/Qwen3-0.6B", model.sink_id)
    same_row = torch.equal(model.sink_emb.cpu(), ref)
    fresh = D.FeedStore("val", np.sort(np.random.default_rng(1234).choice(14423, 24, replace=False)), "cpu",
                        xt_names=va.xt_names, verbose=False)
    b = fresh.gather(np.arange(fresh.n))
    rnd = lambda r: {k: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()}  # noqa: E731
                     for k, v in r.items()}
    with_sink = user_token_states(model, b)
    keep = model.sink_emb.clone()
    try:
        model.sink_emb.copy_(checkpoint_row("Qwen/Qwen3-0.6B", 151643))  # the row declared in LLM_FEASIBILITY 5.2
        eot_sink = user_token_states(model, b)
        model.sink_emb.copy_(keep)
        model.sink, model.n_pre = False, 1  # same weights, old layout (user at position 0)
        no_sink = user_token_states(model, b)
    finally:
        model.sink, model.n_pre = True, 2
        model.sink_emb.copy_(keep)
    cos13 = with_sink[13]["user_cos_max"]
    rec("T10_sink_token", same_row and model.sink_id == 151644 and cos13 < 0.99,
        sink_token_id=model.sink_id, sink_equals_checkpoint_row=same_row,
        sink_rms=round(float(model.sink_emb.pow(2).mean().sqrt()), 5),
        layer13_user_cos_max={"im_start_sink (default)": round(cos13, 6),
                              "endoftext_sink (declared row)": round(eot_sink[13]["user_cos_max"], 6),
                              "no_sink (old layout)": round(no_sink[13]["user_cos_max"], 6)},
        with_im_start_sink=rnd(with_sink), with_endoftext_sink=rnd(eot_sink), without_sink=rnd(no_sink),
        note="24 fresh val feeds (rng 1234), 276 user pairs; acceptance: max pairwise user-token cosine at layer 13 < "
             "0.99; massive = max|h| > 300 (ordinary tokens ~5-300)")


def t11(model, va):
    """Review R1 with the sink: batched L1 score == HF native causal forward over [sink, user, history, c]."""
    import polars as pl
    fv = pl.read_parquet(D.FLAT / "val" / "feeds.parquet", columns=["M", "H"])
    Hv, Mv = fv["H"].to_numpy(), fv["M"].to_numpy()
    rng = np.random.default_rng(7)
    pick = []
    z = np.where(Hv == 0)[0]
    if len(z):
        pick.append(int(rng.choice(z)))
    pick.append(int(rng.choice(np.where(Hv >= 256)[0])))
    pick += [int(x) for x in rng.choice(np.where((Hv > 20) & (Hv < 120) & (Mv > 60))[0], 2, replace=False)]
    pick = np.sort(np.array(pick))
    st = D.FeedStore("val", pick, "cpu", xt_names=va.xt_names, verbose=False)
    saved = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n}
    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.01)
    model.eval()
    b = st.gather(np.arange(st.n))
    bi = inputs_only(clone_b(b))
    errs = []
    with torch.no_grad():
        X = model.dense(bi)
        s_b = model(bi, X)
        u, h, c, Xn = model.tokens(bi, X)
        for f in range(st.n):
            H, M = int(bi["H"][f]), int(bi["M"][f])
            for m in sorted({0, M // 2, M - 1}):
                seq = torch.cat([model.sink_emb[None], u[f:f + 1], h[f, :H].flip(0), c[f, m:m + 1]], 0)[None]
                o = model.lm(inputs_embeds=seq).last_hidden_state[0, -1]  # HF default causal mask, positions 0..H+2
                ref = float(model.head(o[None].float()).squeeze() + model.wide(Xn[f, m][None].float()).squeeze())
                errs.append(abs(ref - float(s_b[f, m])))
        for n, p in model.named_parameters():
            if n in saved:
                p.copy_(saved[n])
    err = max(errs)
    rec("T11_hf_native_forward_parity", err < 1e-3, max_abs_err=err, n_checks=len(errs),
        history_lengths=[int(x) for x in b["H"]], score_std=float(s_b[b["cmask"]].std()),
        note="reference = model.lm(inputs_embeds=[sink, user, h_old..h_new, c]) with HF's own causal mask and default "
             "position ids; LoRA B set to N(0, 0.01^2) for the test")


def t12(model, tr, nd, rel, idc, relonly):
    """I3: fp32_copy() exactness, then the bf16-vs-fp32 precision check on CPU (bf16 weights + CPU autocast)."""
    import train_llm as TL
    ids = np.arange(tr.n)
    model.eval()
    p_orig = TL.predict_store(model, tr, ids, 20000, False)
    m32 = model.fp32_copy()
    p_copy = TL.predict_store(m32, tr, ids, 20000, False)
    copy_err = float(np.abs(p_orig - p_copy).max())
    m16 = LLMRanker("qwen3-0.6b", nd, rel, idc, relonly, layout="L1", dtype="bf16")
    r = m16.load_state_dict(model.trainable_state(), strict=False)
    assert not r.unexpected_keys
    frozen16 = sorted({str(p.dtype) for n, p in m16.lm.named_parameters() if not p.requires_grad and "embed" not in n})
    lines = []
    res = TL.precision_check(m16, tr, ids, 20000, True, 1e-4, lines.append)
    print(lines[0], flush=True)
    rec("T12_i3_precision_path", copy_err < 1e-5 and res["inference_path"].startswith("bf16") and
        np.isfinite(res["delta"]), fp32_copy_max_abs_err=copy_err, bf16_frozen_dtypes=frozen16,
        **{k: v for k, v in res.items() if k != "inference_path"},
        note="CPU stand-in on 32 train feeds (bf16 matmuls on CPU differ from GPU kernels); the binding I3 decision is "
             "made by the GPU probe over the 30K holdout")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--n_feeds", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import peft
    import transformers
    RES["env"] = dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
                      torch_file=torch.__file__, cuda_available=torch.cuda.is_available(), threads=torch.get_num_threads())
    print(RES["env"], flush=True)
    assert not torch.cuda.is_available(), "smoke tests are CPU-only"
    names, rel, idc, relonly, nd = D.xt_config("cvnd_pre2s")

    # ---------------------------------------------------------------- T1
    P, R, RO = D.nn_train_presets()
    same = P["cvnd_pre2s"] == D.XT_PRESETS["cvnd_pre2s"] and R["cvnd_pre2s"] == D.XT_REL["cvnd_pre2s"] \
        and RO["cvnd_pre2s"] == D.XT_RELONLY["cvnd_pre2s"]
    M_all, _ = D.train_meta()
    pr = D.load_prandom(M_all)
    sizes = {s: len(D.matched_subset(s, 150000, pr, M_all)[0]) for s in (23, 24)}
    rec("T1_presets_and_subset", same and sizes == {23: 148600, 24: 148557}, n_dense=nd, n_xt=len(names),
        subset_sizes=sizes)

    # ---------------------------------------------------------------- T2 (nn/data.SplitData on CPU, val)
    import importlib.util
    cs = torch.cuda.synchronize
    torch.cuda.synchronize = lambda *x, **k: None  # SplitData's only CUDA call; patched in this process only
    spec = importlib.util.spec_from_file_location("nn_data_ro", str(D.ROOT / "nn" / "data.py"))
    nnd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nnd)
    ref = nnd.SplitData("val", torch.device("cpu"), verbose=False)
    # nn/data.feat_path hard-links missing features into nn/feat_cache: only call attach_xt if nothing is missing
    assert all((D.FEAT_CACHE / "val" / f"{n.removeprefix('has:')}.npy").exists() for n in names)
    ref.attach_xt(names)
    torch.cuda.synchronize = cs
    sub = np.sort(np.random.default_rng(1).choice(ref.n, 200, replace=False))
    st = D.FeedStore("val", sub, "cpu", xt_names=names, verbose=False)
    ok, filler = True, 0
    for lo in (np.arange(0, 40), np.arange(40, 80)):
        ga, gb = ref.gather(sub[lo]), st.gather(lo)
        for k in ga:
            if k in ("hcat", "het", "hes"):  # masked slots hold filler values (for H=0 feeds: the NEXT stored feed's
                hm = ga["hmask"]             # first event, which differs between a subset and the full split)
                ok &= torch.equal(ga[k][hm], gb[k][hm])
                filler += int(((ga[k] != gb[k]) & ~hm).sum())
            elif k not in ("fidx", "ci"):
                ok &= torch.equal(ga[k], gb[k])
        ok &= torch.equal(D.dense_features(ga, nd, rel, 0.0, False, idc, relonly),
                          D.dense_features(gb, nd, rel, 0.0, False, idc, relonly))
    del ref
    rec("T2_feedstore_equals_splitdata", ok, feeds=80, masked_filler_slots_differing=filler,
        note="all real values and the dense features are bit-identical; only never-read filler at masked history "
             "slots of H=0 feeds differs")

    # ---------------------------------------------------------------- data for the model tests
    va = D.FeedStore("val", np.sort(np.random.default_rng(2).choice(14423, a.n_feeds, replace=False)), "cpu",
                     xt_names=names, verbose=False)
    feeds, hold, _ = D.matched_subset(23, 150000, pr, M_all)
    trf = np.sort(np.random.default_rng(3).choice(feeds, a.n_feeds, replace=False))
    tr = D.FeedStore("train", trf, "cpu", xt_names=names, verbose=False)
    tr.set_domain(pr)
    # ---------------------------------------------------------------- T3
    t = time.time()
    model = LLMRanker("qwen3-0.6b", nd, rel, idc, relonly, layout="L1")
    load_s = time.time() - t
    model.eval()
    mean, std = D.feature_stats(tr, np.arange(tr.n), model.dense, n_batches=4)
    model.set_norm(mean, std)
    pc = count_params(model)
    lora_b_zero = all(float(p.detach().abs().max()) == 0 for n, p in model.named_parameters() if "lora_B" in n)
    n_lora_mod = sum(1 for n, _ in model.named_parameters() if "lora_A" in n)
    frozen_ok = all((not p.requires_grad) for n, p in model.lm.named_parameters()
                    if "lora_" not in n and not n.endswith("norm.weight"))
    b = va.gather(np.arange(4))
    s0 = scores(model, b)
    sink_frozen = model.sink and model.n_pre == 2 and "sink_emb" not in dict(model.named_parameters()) \
        and "sink_emb" in dict(model.named_buffers())
    rec("T3_model_build", lora_b_zero and frozen_ok and s0.dtype == torch.float32 and n_lora_mod == 28 * 7
        and sink_frozen, load_s=round(load_s, 1), emb_rms=round(model.emb_rms, 5), params=pc, lora_modules=n_lora_mod,
        lora_B_zero=lora_b_zero, score_dtype=str(s0.dtype), sink_is_frozen_buffer=sink_frozen, n_pre=model.n_pre)
    sink0 = model.sink_emb.detach().clone()

    # ---------------------------------------------------------------- T4 permutation equivariance (L1, L2)
    rng = np.random.default_rng(4)
    res4 = {}
    for layout in ("L1", "L2"):
        model.cfg["layout"] = layout
        errs = []
        for lo in (np.arange(0, 4), np.arange(4, 8)):
            b = va.gather(lo)
            s = scores(model, b)
            b2, perms = permute_cands(b, rng)
            s2 = scores(model, b2)
            for i, p in enumerate(perms):
                m = int(b["M"][i])
                inv = np.argsort(p[:m])
                errs.append(float((s2[i, :m][torch.from_numpy(inv)] - s[i, :m]).abs().max()))
        res4[layout] = max(errs)
    model.cfg["layout"] = "L1"
    rec("T4_permutation_equivariance", max(res4.values()) < 1e-3, max_abs_err=res4,
        note="fp32; differences are summation-order noise")

    # ---------------------------------------------------------------- T5 batched == single
    b = va.gather(np.arange(4))
    bi = inputs_only(clone_b(b))
    with torch.no_grad():
        X = model.dense(bi)
    s = scores(model, b)
    fi = 1
    cids = list(range(0, int(b["M"][fi]), max(1, int(b["M"][fi]) // 6)))[:6]
    single = single_candidate_scores(model, b, X, fi, cids)
    err_a = float(np.abs(single - s[fi, cids].numpy()).max())
    b1 = va.gather(np.array([fi]))
    s1 = scores(model, b1)
    m1 = int(b1["M"][0])
    err_b = float((s1[0, :m1] - s[fi, :m1]).abs().max())
    rec("T5_batched_equals_single", err_a < 1e-3 and err_b < 1e-3, cand_alone_vs_batched=err_a,
        feed_alone_vs_in_batch=err_b, scores_range=[float(s[fi, :m1].min()), float(s[fi, :m1].max())])

    # ---------------------------------------------------------------- T6 no leakage
    b = tr.gather(np.arange(4), with_pos=True)
    model.eval()
    with torch.no_grad():
        s_in = model(inputs_only(clone_b(b)))
    bf = clone_b(b)
    bf["y"] = 1 - bf["y"]
    bf["pos"] = torch.flip(bf["pos"], [1])
    # the training loop never gives y/pos to the model; show that even a dict that CONTAINS perturbed y/pos gives the
    # same scores and that the model reads neither key
    class Spy(dict):
        def __init__(self, *x, **k):
            super().__init__(*x, **k)
            self.read = set()

        def __getitem__(self, k):
            self.read.add(k)
            return super().__getitem__(k)

        def get(self, k, d=None):
            self.read.add(k)
            return super().get(k, d)
    spy = Spy(clone_b(bf))
    with torch.no_grad():
        s_spy = model(spy)
    read = sorted(spy.read)
    eq_labels = float((s_spy - s_in).abs().max())
    # feed isolation: perturb feed 3 entirely -> feeds 0..2 unchanged
    bp = clone_b(b)
    bp["ce_raw"][3] = torch.randn_like(bp["ce_raw"][3]) * 50
    bp["he_raw"][3] = torch.randn_like(bp["he_raw"][3]) * 50
    bp["xt"][3] = torch.randn_like(bp["xt"][3])
    bp["ue_raw"][3] = -bp["ue_raw"][3]
    with torch.no_grad():
        s_p = model(inputs_only(bp))
    iso = max(float((s_p[i, :int(b["M"][i])] - s_in[i, :int(b["M"][i])]).abs().max()) for i in range(3))
    # candidate isolation in L1 at the token level: change candidate j's token -> other candidates' hidden unchanged
    bi = inputs_only(clone_b(b))
    with torch.no_grad():
        Xb = model.dense(bi)
        u, h, c, _ = model.tokens(bi, Xb)

        def hid(cc, layout):
            seq, Tx = model.assemble(u, h, cc, bi["H"], bi["M"])
            pos, mask, ct, _, _ = model.layout_of(bi["H"], bi["M"], Tx, cc.shape[1], layout=layout)
            o = model.lm(inputs_embeds=seq, attention_mask={"full_attention": mask}, position_ids=pos).last_hidden_state
            return o.gather(1, ct[..., None].expand(-1, -1, o.shape[-1]))
        c2 = c.clone()
        c2[:, 0] = c2[:, 0] * -3.0
        d1 = (hid(c2, "L1") - hid(c, "L1"))[:, 1:].abs().max().item()
        d2 = (hid(c2, "L2") - hid(c, "L2"))[:, 1:].abs().max().item()
    ok6 = "y" not in read and "pos" not in read and eq_labels < 1e-5 and iso < 1e-3 and d1 < 1e-3 and d2 > 1e-3
    rec("T6_no_leakage", ok6, keys_read_by_model=read, max_diff_perturbed_labels=eq_labels,
        max_diff_other_feed_perturbed=iso, L1_other_cand_hidden_diff=d1, L2_other_cand_hidden_diff=d2,
        note="X (dense features) is feed-relative by design (within-feed z/rank, as in the ranker); attention isolation "
             "is tested at the token level")

    # ---------------------------------------------------------------- T7 optimiser step / overfit
    trunk, proj = model.param_groups()
    base_before = {n: p.detach().clone() for n, p in model.lm.named_parameters() if "q_proj.base_layer" in n}
    opt = torch.optim.AdamW([{"params": [p for _, p in trunk], "lr": 2e-4, "weight_decay": 0.0},
                             {"params": [p for _, p in proj], "lr": 1e-3, "weight_decay": 0.02}], betas=(0.9, 0.98))

    def batch_loss(bb, train):
        model.train(train)
        bi = inputs_only(clone_b(bb))
        X = model.dense(bi)
        lg = model(bi, X)
        m = bb["cmask"]
        loss = D.listwise_loss(lg, bb["y"], m, None, norm="feed", denom=len(bb["M"]))
        if train:
            tgt = torch.softmax((-bb["pos"].float() / 10).masked_fill(~m, -1e4), 1) * m
            lp = torch.log_softmax(model.aux_out.masked_fill(~m, -1e4), 1)
            fw = 1 - tr.prand[bb["fidx"]]
            loss = loss + 0.5 * (-(tgt * lp).sum(1) * fw).sum() / fw.sum().clamp_min(1e-6)
        return loss
    b16 = tr.gather(np.arange(16), with_pos=True)
    with torch.no_grad():
        l_before = float(batch_loss(b16, False))
    t = time.time()
    loss = batch_loss(b16, True)
    opt.zero_grad()
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0))
    opt.step()
    step_s = time.time() - t
    with torch.no_grad():
        l_after = float(batch_loss(b16, False))
    base_same = all(torch.equal(base_before[n], p) for n, p in model.lm.named_parameters() if n in base_before)
    lora_moved = any(float(p.detach().abs().max()) > 0 for n, p in model.named_parameters() if "lora_B" in n)
    ov = tr.gather(np.arange(16, 24), with_pos=True)
    with torch.no_grad():
        losses = [round(float(batch_loss(ov, False)), 4)]
    for _ in range(8):
        loss = batch_loss(ov, True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        with torch.no_grad():
            losses.append(round(float(batch_loss(ov, False)), 4))
    sink_unchanged = torch.equal(sink0, model.sink_emb)
    rec("T7_optimiser", l_after < l_before and base_same and lora_moved and losses[-1] < losses[0]
        and min(losses[-3:]) < losses[0] - 0.1 and sink_unchanged,
        eval_loss_before=round(l_before, 5), eval_loss_after_1_step=round(l_after, 5), grad_norm=round(gn, 3),
        step_seconds_16_feeds=round(step_s, 1), frozen_base_unchanged=base_same, lora_B_moved=lora_moved,
        sink_unchanged_by_optimiser=sink_unchanged,
        overfit_8_feeds_eval_loss_step0_to_8=losses)

    # ---------------------------------------------------------------- T8 CPU throughput vs FLOP model
    N = pc["body"]
    b8 = tr.gather(np.arange(8), with_pos=True)
    T_real = int(tr.seq_len(np.arange(8), model.n_pre).sum())  # n_pre: the S1 sink token counts (review fix)
    T_pad = int(8 * tr.seq_len(np.arange(8), model.n_pre).max())
    model.eval()
    with torch.no_grad():
        t = time.time()
        model(inputs_only(clone_b(b8)))
        fwd = time.time() - t
    thr = {"fwd_s": round(fwd, 2)}
    for ck in (False, True):
        if ck:
            model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        t = time.time()
        loss = batch_loss(b8, True)
        loss.backward()
        thr[f"fwd_bwd_ckpt{int(ck)}_s"] = round(time.time() - t, 2)
        opt.zero_grad()
    model.lm.gradient_checkpointing_disable()
    thr.update(real_tokens=T_real, padded_tokens=T_pad,
               fwd_TFLOPs_per_s_2N=round(2 * N * T_pad / fwd / 1e12, 3),
               fwd_bwd_TFLOPs_per_s_4N=round(4 * N * T_pad / thr["fwd_bwd_ckpt0_s"] / 1e12, 3),
               fwd_bwd_ckpt_TFLOPs_per_s_6N=round(6 * N * T_pad / thr["fwd_bwd_ckpt1_s"] / 1e12, 3),
               fwd_padded_tok_per_s=round(T_pad / fwd), train_padded_tok_per_s_ckpt=round(T_pad / thr["fwd_bwd_ckpt1_s"]),
               ratio_fwdbwd_ckpt_over_fwd=round(thr["fwd_bwd_ckpt1_s"] / fwd, 2))
    rec("T8_cpu_throughput", True, **thr)
    t10(model, va)
    t11(model, va)
    t12(model, tr, nd, rel, idc, relonly)
    del model, opt

    # ---------------------------------------------------------------- T9 random-init control
    t = time.time()
    ctl = LLMRanker("qwen3-0.6b", nd, rel, idc, relonly, random_init=True)
    ctl.set_norm(mean, std)
    pcc = count_params(ctl)
    tr_c, pr_c = ctl.param_groups()
    opt = torch.optim.AdamW([{"params": [p for _, p in tr_c], "lr": 5e-4}, {"params": [p for _, p in pr_c], "lr": 1e-3}])
    model = ctl
    with torch.no_grad():
        lb = float(batch_loss(b16, False))
    loss = batch_loss(b16, True)
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        la = float(batch_loss(b16, False))
    sink_rms = float(ctl.sink_emb.pow(2).mean().sqrt())
    sink_own = ctl.sink and ctl.sink_id == 151644 and 0.01 < sink_rms < 0.03 and \
        not torch.allclose(ctl.sink_emb, sink0) and "sink_emb" not in dict(ctl.named_parameters())
    ctl.eval()
    bq = tr.gather(np.arange(4))
    with torch.no_grad():
        s_c = ctl(inputs_only(clone_b(bq)))
        s_cc = ctl.fp32_copy()(inputs_only(clone_b(bq)))
    copy_err = float((s_c - s_cc)[bq["cmask"]].abs().max())
    rec("T9_random_init_control", pcc["trainable"] > 4e8 and np.isfinite(la) and sink_own and copy_err < 1e-5,
        params=pcc, emb_rms=round(ctl.emb_rms, 5), sink_rms_own_init=round(sink_rms, 5), sink_is_own_init_row=sink_own,
        fp32_copy_max_abs_err=copy_err,
        eval_loss_before=round(lb, 5), eval_loss_after_1_step=round(la, 5), seconds=round(time.time() - t, 1))
    (out / "smoke_results.json").write_text(json.dumps(RES, indent=2, default=str))
    print("ALL PASS" if all(v.get("ok", True) for k, v in RES.items() if k != "env") else "SOME FAILED", flush=True)


if __name__ == "__main__":
    main()
