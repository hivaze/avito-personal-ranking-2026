"""HDROP-ALL diagnostic (reports/TRICKS_SWEEP_1113.md section 3 #3): history-event dropout TTA replay on the 10-epoch base
members (no training). Pre-declaration: nn_x/tricks/HDROP-ALL/PREDECLARED.txt.

REPLAY-ONLY. Imports the nn_x/abl model / data / common code read-only and wraps data.SplitData.gather IN THIS PROCESS
ONLY (no file outside nn_x/tricks/HDROP-ALL/ is written or modified). Checkpoints are read from nn_x/w5/ckpt.

  GPU step (one GPU id):
    cd <repo>/nn_x/abl && CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=12 ../../.venv/bin/python \
        ../tricks/HDROP-ALL/hdrop_replay.py predict w5_e10_base_s951 w5_e10_base_s952
      -> nn_x/tricks/HDROP-ALL/preds/{val,holdout}/<v>~<mode>.npy, logs/predict.log
  unit test (GPU, small, no NDCG): ... hdrop_replay.py unit w5_e10_base_s951

Modes: ref (full history); d10_k{0..3}, d20_k{0..3}: each valid event among the first min(H,256) except the newest
(index 0) dropped with prob p; survivors compacted newest-first in place (batch width unchanged); every in-model
history-derived input recomputed (same tensors as solvers/r6/tta.py compact_hist, plus hidx). xt inputs unchanged.
"""
import os
import sys
import time

import numpy as np

ROOT = "<repo>"
ABL = f"{ROOT}/nn_x/abl"
W5CK = f"{ROOT}/nn_x/w5/ckpt"
W5P = f"{ROOT}/nn_x/w5/preds"
HERE = f"{ROOT}/nn_x/tricks/HDROP-ALL"
MEMBERS = ["w5_e10_base_s951", "w5_e10_base_s952"]
RATES = {"d10": (1, 0.1), "d20": (2, 0.2)}
K_MASKS = 4
SEED0 = 20261113
MODES = ["ref"] + [f"{r}_k{k}" for r in RATES for k in range(K_MASKS)]

sys.path.insert(0, ABL)


def compact_hist(b, keep):
    """== solvers/r6/tta.py compact_hist (device-agnostic) + hidx: keep only events with keep=True (subset of hmask),
    moved in their original newest-first order to a contiguous prefix; masks and H updated; width unchanged."""
    import torch
    order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)
    Kx = keep.shape[1]
    newH = keep.sum(1)
    hmask = torch.arange(Kx, device=keep.device)[None, :] < newH[:, None]
    g2 = lambda x: torch.gather(x, 1, order)  # noqa: E731
    he_raw = torch.gather(b["he_raw"], 1, order[..., None].expand(-1, -1, b["he_raw"].shape[-1]))
    hemb_ok = g2(b["hemb_ok"]) & hmask
    he_raw = he_raw * hemb_ok[..., None].float()  # data.gather zeroes he_raw of invalid / padded events
    b.update(he_raw=he_raw, hemb_ok=hemb_ok, hmask=hmask, hcat=g2(b["hcat"]), het=g2(b["het"]), hes=g2(b["hes"]),
             H=newH.to(b["H"].dtype))
    if "hidx" in b:
        b["hidx"] = g2(b["hidx"])
    if "hitem" in b:
        b["hitem"] = g2(b["hitem"])
    return b


def drop_keep(hmask, p, gen):
    """keep mask: valid events survive with prob 1-p; the newest event (index 0) is always kept."""
    import torch
    u = torch.rand(hmask.shape, generator=gen, device=hmask.device)
    first = torch.zeros_like(hmask)
    first[:, 0] = True
    return hmask & ((u >= p) | first)


def seed_of(mode, midx, split):
    r, _ = RATES[mode.split("_")[0]]
    k = int(mode.split("_k")[1])
    return SEED0 + 1000 * r + 10 * midx + k + (500000 if split == "holdout" else 0)


def setup():
    import torch
    import data as Dm  # nn_x/abl/data.py (read-only)
    from common import predict  # nn_x/abl/common.py: GPU guard (one CUDA_VISIBLE_DEVICES id), predict()
    from abl_retr import load_ranker
    from cd_arms5 import holdout_preds  # TF32 off, exactly the stored holdout preds' code path
    MODE = {"p": None, "gen": None}
    _gather = Dm.SplitData.gather

    def gather(self, fidx_np, with_pos=False):  # replay hook: history dropout + compaction, nothing else
        b = _gather(self, fidx_np, with_pos)
        if MODE["p"] is not None:
            b = compact_hist(b, drop_keep(b["hmask"], MODE["p"], MODE["gen"]))
        return b

    Dm.SplitData.gather = gather
    return torch, Dm, predict, load_ranker, holdout_preds, MODE


def load_ck(torch, load_ranker, v, dev):
    ck = torch.load(f"{W5CK}/{v}.pt", map_location=dev, weights_only=False)
    assert ck["args"]["max_hist"] == 256 and not ck["args"]["retr_uniq"] and not ck["cfg"].get("id_buckets")
    m = load_ranker(ck, dev)
    m.eval()
    return ck, m


def cmd_predict(variants):
    torch, Dm, predict, load_ranker, holdout_preds, MODE = setup()
    from metrics import ndcg  # src/metrics via common's sys.path insert
    os.makedirs(f"{HERE}/preds/val", exist_ok=True)
    os.makedirs(f"{HERE}/preds/holdout", exist_ok=True)
    os.makedirs(f"{HERE}/logs", exist_ok=True)
    logf = open(f"{HERE}/logs/predict.log", "a")

    def L(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True  # train.py main(): val predict runs with TF32 on
    t0 = time.time()
    va = Dm.SplitData("val", dev)
    tr = Dm.SplitData("train", dev)
    pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
    assert (pr["fidx"] == np.arange(tr.n)).all()
    prand_np = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    prand_np[tr.M_np < 50] = 0.0
    prand = torch.from_numpy(prand_np).to(dev)
    cand = np.where((prand_np > 0.9) & (tr.M_np >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    hold_rows = np.concatenate([np.arange(o, o + m) for o, m in zip(tr.cand_off_np[hold], tr.M_np[hold])])
    meta = np.load(f"{ROOT}/preds/holdout/holdout_meta.npy")
    assert (meta[0] == tr.M_np[hold]).all(), "holdout definition differs from preds/holdout/holdout_meta.npy"
    yh = tr.target_np[hold_rows]
    assert (yh == np.load(f"{ROOT}/preds/holdout/holdout_y.npy")).all()
    np.save(f"{HERE}/preds/holdout/hold_H.npy", np.minimum(tr.H_np[hold], 256))
    L(f"{time.strftime('%FT%TZ', time.gmtime())} predict: data loaded in {time.time() - t0:.0f}s; modes {MODES}")
    xt_cur = None
    for v in variants:
        midx = MEMBERS.index(v)
        ck, m = load_ck(torch, load_ranker, v, dev)
        xt = ck["cfg"].get("xt_names") or []
        if xt != xt_cur:
            for sd in (va, tr):
                sd.xt = None
                torch.cuda.empty_cache()
                sd.attach_xt(xt)
            xt_cur = xt
        tr.dom = torch.stack([1 - prand, prand, torch.zeros_like(prand)], 1) if ck["cfg"].get("domain") else None
        for mode in MODES:
            fv, fh = f"{HERE}/preds/val/{v}~{mode}.npy", f"{HERE}/preds/holdout/{v}~{mode}.npy"
            if os.path.exists(fv) and os.path.exists(fh):
                L(f"  {v} {mode}: exists")
                continue
            t1 = time.time()
            if mode == "ref":
                MODE.update(p=None, gen=None)
            else:
                MODE.update(p=RATES[mode.split("_")[0]][1],
                            gen=torch.Generator(device=dev).manual_seed(seed_of(mode, midx, "val")))
            pv = predict(m, va)  # TF32 on, bf16 autocast
            if mode != "ref":
                MODE.update(gen=torch.Generator(device=dev).manual_seed(seed_of(mode, midx, "holdout")))
            sh = holdout_preds(predict, m, tr, hold, hold_rows)  # TF32 off
            MODE.update(p=None, gen=None)
            assert np.isfinite(pv).all() and np.isfinite(sh).all()
            msg = ""
            if mode == "ref":
                r_v = np.load(f"{W5P}/nn_{v}_trainonly_val.npy")
                r_h = np.load(f"{W5P}/holdout/nn_{v}_holdout.npy")
                msg = (f"; REPLAY vs stored: val max|d| {np.abs(r_v - pv).max():.2e} corr {np.corrcoef(r_v, pv)[0, 1]:.8f}, "
                       f"holdout max|d| {np.abs(r_h - sh).max():.2e} corr {np.corrcoef(r_h, sh)[0, 1]:.8f}")
            np.save(fv + ".tmp.npy", pv.astype(np.float32))
            os.replace(fv + ".tmp.npy", fv)
            np.save(fh + ".tmp.npy", sh.astype(np.float32))
            os.replace(fh + ".tmp.npy", fh)
            L(f"  {time.strftime('%FT%TZ', time.gmtime())} {v} {mode}: done ({time.time() - t1:.0f}s){msg}")
        del m
        torch.cuda.empty_cache()
    L(f"{time.strftime('%FT%TZ', time.gmtime())} predict done ({(time.time() - t0) / 60:.1f} min)")


def cmd_unit(variants):
    """Mechanics only (no labels, no NDCG): (a) p=0 dropout == identity (logits bit-equal); (b) compaction == physically
    deleting the dropped events from the flat history arrays (same batch width); (c) newest event always kept; (d) the
    realised drop rate on events 1..H-1 matches p."""
    torch, Dm, predict, load_ranker, holdout_preds, MODE = setup()
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    va = Dm.SplitData("val", dev)
    ck, m = load_ck(torch, load_ranker, variants[0], dev)
    va.attach_xt(ck["cfg"]["xt_names"])
    rng = np.random.default_rng(0)
    fb = np.sort(rng.choice(va.n, 300, replace=False))
    MODE.update(p=None)
    b0 = va.gather(fb)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        l0 = m(dict(b0), use_pos=False)
    MODE.update(p=0.0, gen=torch.Generator(device=dev).manual_seed(1))
    b1 = va.gather(fb)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        l1 = m(dict(b1), use_pos=False)
    print("(a) p=0 identity: max|dlogit|", float((l0 - l1).abs()[b0["cmask"]].max()))
    # (b) physical deletion
    MODE.update(p=None)
    b = va.gather(fb)
    keep = drop_keep(b["hmask"], 0.2, torch.Generator(device=dev).manual_seed(7))
    bc = compact_hist({k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}, keep)
    print("(c) newest kept on all feeds with H>0:", bool((keep[:, 0] == b["hmask"][:, 0]).all()))
    dr = (b["hmask"][:, 1:] & ~keep[:, 1:]).sum().item() / max(b["hmask"][:, 1:].sum().item(), 1)
    print(f"(d) realised drop rate on events 1..: {dr:.4f} (p 0.2)")
    # physical: rebuild he/hcat/het/hes from the flat arrays with dropped events removed
    ok = True
    for i, f in enumerate(fb):
        H = int(min(va.H_np[f], 256))
        ho = int(va.hist_off[f].item())
        idx = np.arange(ho, ho + H)[keep[i, :H].cpu().numpy()]
        n = len(idx)
        ok &= int(bc["H"][i].item()) == n and bool(bc["hmask"][i, :n].all()) and not bool(bc["hmask"][i, n:].any())
        ok &= bool((bc["hcat"][i, :n].cpu().numpy() == va.h_cat[idx].long().cpu().numpy()).all())
        ok &= bool((bc["het"][i, :n].cpu().numpy() == va.h_et[idx].long().cpu().numpy()).all())
        ok &= bool((bc["hes"][i, :n].cpu().numpy() == va.h_es[idx].long().cpu().numpy()).all())
        ei = va.h_emb[idx].long()
        ref = va.table[ei.clamp_min(0)].float() * (ei >= 0)[:, None].float()
        ok &= bool((bc["he_raw"][i, :n] == ref).all()) and bool((bc["hemb_ok"][i, :n] == (ei >= 0)).all())
        ok &= bool((bc["he_raw"][i, n:] == 0).all()) and not bool(bc["hemb_ok"][i, n:].any())
        ok &= bool((bc["hidx"][i, :n] == va.h_emb[idx]).all())
    print("(b) compaction == physical deletion on 300 feeds:", ok, "| width", b["hmask"].shape[1], "->", bc["hmask"].shape[1])


if __name__ == "__main__":
    cmd, variants = sys.argv[1], sys.argv[2:]
    assert cmd in ("predict", "unit") and variants and all(v in MEMBERS for v in variants)
    cmd_predict(variants) if cmd == "predict" else cmd_unit(variants)
