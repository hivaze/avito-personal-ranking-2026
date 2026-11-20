"""CPU unit test of the round-5 agedrop flag (agedrop.py + the train.py hook). No GPU, read-only on project data.

  CUDA_VISIBLE_DEVICES="" python nn_x/solvers/test_agedrop.py

Uses the real baseline checkpoint sx_base_s23 (cfg + weights; cvnd_pre2s inputs) and real val batches on the CPU.
  T1 flag off == baseline: train.py only calls agedrop when --agedrop is non-empty (checked on the source); with a
     p == 0 preset the dense features, logits and the full training loss (main listwise + K9 aux, train-mode dropout) are
     BITWISE identical to not calling it, i.e. the dedicated generator leaves the model's RNG stream untouched.
  T2 blank == genuinely missing: with p == 1 the dense features (incl. the in-model rank/z channels) equal those of a
     batch whose 8 raw ID inputs are NaN (and, separately, 0) on disk, i.e. the attach_xt NaN->0 / log1p(0) path.
  T3 only the ID columns of valid candidate rows change; all other xt columns and padded rows are bitwise unchanged.
  T4 torch age buckets == numpy reference (feats_v2/life_/post_time.py arithmetic) on all val rows.
  T5 realised blanking rate per bucket matches p (|diff| < 4 SE + 1e-3) for preset prv_rl13.
  T6 the hook never writes into model.cfg (checkpoints stay loadable by the stock model.py).
"""
import copy
import os
import re
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "16"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(16)
torch.cuda.synchronize = lambda *a, **k: None  # data.SplitData syncs the GPU after loading; no-op on the CPU
import agedrop as AD  # noqa: E402
import data as D  # noqa: E402
from model import Ranker, listwise_loss  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
dev = torch.device("cpu")


def clone(b):
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in b.items()}


def train_loss(model, b, seed):
    """Same loss as train.py (loss=softmax, loss_norm=feed, denom=max_feeds=512, aux K9 listwise tau 10, aux weight
    0.5, fw_aux = 1 - p_random of the feed), in train mode with a fixed torch seed for the model's dropout."""
    torch.manual_seed(seed)
    X = model.dense(b)
    lg = model(b, X, use_pos=True)
    y, m = b["y"], b["cmask"]
    loss = listwise_loss(lg, y, m, None, norm="feed", denom=512)
    tgt = torch.softmax((-b["pos"].float() / 10.0).masked_fill(~m, -1e4), 1) * m
    lp = torch.log_softmax(model.aux_out.masked_fill(~m, -1e4), 1)
    fw_aux = 1 - b["prand"]
    loss = loss + 0.5 * (-(tgt * lp).sum(1) * fw_aux).sum() / fw_aux.sum().clamp_min(1e-6)
    return X, lg, loss


def main():
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)

    # T1a: the train.py hook is guarded by the flag (source check)
    src = open(f"{HERE}/train.py").read()
    check("T1a train.py: AgeDrop built only if --agedrop is set, called only if built",
          re.search(r"agedrop = None\n\s+if a\.agedrop:", src) is not None
          and src.count("agedrop(b)") == 1 and "if agedrop is not None:\n            agedrop(b)" in src
          and 'p.add_argument("--agedrop", default=""' in src)

    ck = torch.load("<repo>/nn_x/solvers/ckpt/sx_base_s23.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    names = cfg["xt_names"]
    model = Ranker(**cfg)
    model.load_state_dict(ck["state_dict"])
    model.train()
    cfg_before = copy.deepcopy(model.cfg)
    idc = AD.id_cols(names)
    check("T0 ID columns = xt_idcols of the checkpoint", idc == cfg["xt_idcols"],
          f"{[names[i] for i in idc]}")

    d = D.SplitData("val", dev, max_hist=ck["args"]["max_hist"], verbose=False)
    d.attach_xt(names)
    rng = np.random.default_rng(0)
    bl = d.batches(np.arange(d.n), 32768, 512, shuffle=True, rng=rng)
    test_batches = bl[:3]

    def batch(fb, data=d):
        b = data.gather(fb)
        g = torch.Generator().manual_seed(int(fb[0]))
        r = torch.rand(b["cmask"].shape, generator=g).masked_fill(~b["cmask"], 2.0)
        b["pos"] = r.argsort(1).argsort(1)  # synthetic production order (val has no pos)
        b["prand"] = torch.rand(b["cmask"].shape[0], generator=g)
        b["dom"] = torch.stack([1 - b["prand"], b["prand"], torch.zeros_like(b["prand"])], 1)
        return b

    AD.PRESETS["_zero"] = dict(edges_h=AD.EDGES_H, p=[0.0] * 13)
    AD.PRESETS["_one"] = dict(edges_h=AD.EDGES_H, p=[1.0] * 13)
    for i, fb in enumerate(test_batches):
        b0 = batch(fb)
        b1 = clone(b0)
        X0, lg0, l0 = train_loss(model, b0, seed=100 + i)
        ad = AD.AgeDrop("_zero", names, dev, seed=7)
        drop = ad(b1)
        X1, lg1, l1 = train_loss(model, b1, seed=100 + i)
        check(f"T1b batch {i}: p=0 -> no row dropped, X / logits / loss bitwise equal", int(drop.sum()) == 0
              and torch.equal(X0, X1) and torch.equal(lg0, lg1) and torch.equal(l0, l1),
              f"loss {l0.item():.6f} vs {l1.item():.6f}")
        # T1c: a second baseline call reproduces itself (the test itself is deterministic)
        _, _, l0b = train_loss(model, clone(b0), seed=100 + i)
        check(f"T1c batch {i}: baseline loss reproducible", torch.equal(l0, l0b))

    # T2: p = 1 blanking == a batch whose raw ID inputs are NaN (resp. 0) on disk
    orig_fp = D.feat_path
    tmp = tempfile.mkdtemp(dir="/tmp/scratchpad/round5/agedrop")
    for fill, tag in ((np.nan, "nan"), (0.0, "zero")):
        over = {}
        for n in [names[i] for i in idc]:
            p = f"{tmp}/{n}_{tag}.npy"
            np.save(p, np.full(d.n_cand, fill, np.float32))
            over[n] = p
        D.feat_path = lambda split, name, over=over: over.get(name) or orig_fp(split, name)
        dm = D.SplitData("val", dev, max_hist=ck["args"]["max_hist"], verbose=False)
        dm.attach_xt(names)
        D.feat_path = orig_fp
        for i, fb in enumerate(test_batches[:2]):
            bb = batch(fb)
            ad = AD.AgeDrop("_one", names, dev, seed=7)
            ad(bb)
            bm = batch(fb, dm)
            model.eval()
            Xb, Xm = model.dense(bb), model.dense(bm)
            model.train()
            cm = bb["cmask"]  # padded rows are masked everywhere in the model; compare the valid candidate rows
            check(f"T2 ({tag} on disk) batch {i}: blanked dense features == genuinely-missing dense features",
                  torch.equal(Xb[cm], Xm[cm]) and torch.equal(bb["xt"][cm], bm["xt"][cm]),
                  f"max|diff| {float((Xb[cm] - Xm[cm]).abs().max()):.2e}")
        del dm

    # T3: only ID columns of valid rows change
    ad = AD.AgeDrop("prv_rl13", names, dev, seed=11)
    b0 = batch(test_batches[0])
    b1 = clone(b0)
    drop = ad(b1)
    non = [j for j in range(len(names)) if j not in idc]
    pad = ~b0["cmask"]
    changed_rows = (b1["xt"] != b0["xt"]).any(-1)
    check("T3 non-ID columns bitwise unchanged", torch.equal(b1["xt"][..., non], b0["xt"][..., non]))
    check("T3 padded rows unchanged; changed rows subset of dropped rows",
          torch.equal(b1["xt"][pad], b0["xt"][pad]) and bool((changed_rows <= drop).all()),
          f"dropped {int(drop.sum())} changed {int(changed_rows.sum())}")
    check("T3 dropped rows have all ID columns == 0.0", bool((b1["xt"][drop][:, idc] == 0).all()))
    other = {k for k in b0 if k != "xt"}
    check("T3 other batch tensors untouched", all(torch.equal(b0[k], b1[k]) for k in other if torch.is_tensor(b0[k])))

    # T4: bucket assignment vs numpy reference on all val rows
    sys.path.insert(0, "/tmp/scratchpad/round5/agedrop")
    import measure as MS
    f4 = np.load("<repo>/data/flat/val/cand_f4.npy")
    bt = AD.age_bucket(torch.from_numpy(f4), ad.knots, ad.edges).numpy()
    bn = MS.bucket(f4).astype(np.int64)
    check("T4 torch age buckets == numpy reference (all val rows)", (bt == bn).all(),
          f"mismatch {(bt != bn).sum()} / {len(bt)}; f4 min {f4.min():.4f}")
    bq = d.gather(test_batches[0])
    rows = bq["ci"][bq["cmask"]].numpy()
    check("T4b batch column cf[..., F4_COL] is cand_f4 (not another f)",
          np.array_equal(bq["cf"][..., AD.F4_COL][bq["cmask"]].numpy(), f4[rows]))
    cnt_all = np.bincount(bn, minlength=13)
    check("T4c every bucket populated on val", (cnt_all > 0).all(), f"{cnt_all.tolist()}")

    # T5: realised blanking rate per bucket ~ p (fresh rows only: dropping is independent of the current value)
    ad = AD.AgeDrop("prv_rl13", names, dev, seed=13)
    cnt = np.zeros(13)
    hit = np.zeros(13)
    for fb in bl[:200]:
        b = d.gather(fb)
        g = AD.age_bucket(b["cf"][..., AD.F4_COL], ad.knots, ad.edges)[b["cmask"]].numpy()
        dr = ad(b)[b["cmask"]].numpy()
        cnt += np.bincount(g, minlength=13)
        hit += np.bincount(g, weights=dr, minlength=13)
    p = np.array(AD.PRESETS["prv_rl13"]["p"])
    r = hit / np.maximum(cnt, 1)
    se = np.sqrt(p * (1 - p) / np.maximum(cnt, 1))
    good = (np.abs(r - p) < 4 * se + 1e-3) & (cnt > 1000)
    check("T5 realised drop rate per bucket == p (every bucket with > 1000 rows)", good.all(),
          " ".join(f"{lab}:{x:.3f}/{q:.3f}" for lab, x, q in zip(AD.LABELS, r, p)))
    check("T6 model.cfg unchanged by the hook", model.cfg == cfg_before and "agedrop" not in str(model.cfg))
    print("ALL PASS" if ok else "SOME TESTS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
