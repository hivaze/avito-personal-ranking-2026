"""r6 recap (v2: larger sample, split-half + jackknife instead of bootstrap): gradient-noise-scale (B_simple, McCandlish et al. 2018) of live checkpoints on train feeds. CPU ONLY.

Question: is the fixed batch (cand_budget 32768 / max_feeds 512 -> ~317 feeds per step on average) far below or above
the critical batch size? Never varied in 285 training segments.

Method: n disjoint random batches of b train feeds (holdout excluded, mixed M, chunked by M like data.batches);
per-feed loss L_f = listwise CE_f + aux_w * (1 - p_random_f) * K9_f (the training objective at the plateau,
aux_w = 0.5); model.train() (dropout on, as in training). G_i = mean_f grad L_f over batch i.
  tr(Sigma) = b * sum_i |G_i - Gbar|^2 / (n - 1);  |G|^2 = |Gbar|^2 - tr(Sigma) / (n b);  B_simple = tr(Sigma) / |G|^2
Reported for all parameters and for the Muon matrices only, with a bootstrap over batches.

PRE-DECLARED reading (written before running):
  * B_simple(plateau ckpts) >= 2 x 317 feeds  -> a 2x batch (cand_budget 65536, max_feeds 1024) costs <= ~1/3 extra
    examples per unit progress (E/E_min = 1 + B/B_noise) -> rank "2x batch" as a throughput lever worth a screen.
  * B_simple <= 317 feeds -> the current batch is already at/above the critical size; larger batches waste data;
    rank "smaller batch (0.5x)" as the regularisation-side screen instead.
  * in between -> ambiguous; report only.
  Information only: nothing here changes any live recipe.

Usage: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 python nn_x/r6/recap_gns.py ckpt1 [ckpt2 ...]
"""
import os
import sys
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only: run with CUDA_VISIBLE_DEVICES=''"
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "12")))
torch.cuda.synchronize = lambda *a, **k: None  # data.py calls it unconditionally
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D  # noqa: E402  (local copy of nn/data.py)
from model import Ranker  # noqa: E402  (local copy of nn/model.py)

ROOT = "<repo>"


def _feat_path_ro(split, name):  # read-only: never create links inside nn/feat_cache
    c = D.FEAT_CACHE / split / f"{name}.npy"
    if not c.exists():
        raise FileNotFoundError(c)
    return c


D.feat_path = _feat_path_ro


def per_feed_loss(model, b, prand_b, aux_w=0.5, tau=10.0):
    X = model.dense(b)
    lg = model(b, X, use_pos=True)
    y, m = b["y"], b["cmask"]
    lgm = lg.masked_fill(~m, -1e4)
    logp = torch.log_softmax(lgm, 1)
    ys = y.sum(1)
    l = -(y * logp).sum(1) / ys.clamp_min(1)
    l = l * (ys > 0).float()
    tgt = torch.softmax((-b["pos"].float() / tau).masked_fill(~m, -1e4), 1) * m
    lpa = torch.log_softmax(model.aux_out.masked_fill(~m, -1e4), 1)
    aux = -(tgt * lpa).sum(1)
    return l + aux_w * (1 - prand_b) * aux


def main(names):
    t0 = time.time()
    log = open(f"{HERE}/logs/recap_gns2.log", "a")

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    dev = torch.device("cpu")
    tr = D.SplitData("train", dev, verbose=False)
    P(f"loaded train in {time.time() - t0:.0f}s")
    pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
    prand_np = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    prand_np[tr.M_np < 50] = 0.0
    cand = np.where((prand_np > 0.9) & (tr.M_np >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    feeds = np.setdiff1d(np.arange(tr.n), hold)
    prand = torch.from_numpy(prand_np)
    tr.dom = torch.stack([1 - prand, prand, torch.zeros_like(prand)], 1)
    n_b, bsz = int(os.environ.get("GNS_N", "48")), int(os.environ.get("GNS_B", "32"))
    cur_xt = None
    for name in names:
        ck = torch.load(f"{ROOT}/nn/ckpt/{name}.pt", map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        if cfg.get("xt_names") != cur_xt:
            tr.xt = None
            tr.attach_xt(cfg["xt_names"])
            cur_xt = cfg.get("xt_names")
        model = Ranker(**cfg)
        model.load_state_dict(ck["state_dict"])
        model.train()
        muon = {f"{mn}.weight" for mn, mod in model.named_modules()
                if isinstance(mod, torch.nn.Linear) and min(mod.weight.shape) > 1}
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        is_mu = np.concatenate([np.full(p.numel(), n in muon) for n, p in params])
        rng = np.random.default_rng(777)
        perm = rng.permutation(feeds)[: n_b * bsz]
        G = np.zeros((n_b, len(is_mu)), np.float32)
        Ms = []
        t1 = time.time()
        for i in range(n_b):
            fb = np.sort(perm[i * bsz:(i + 1) * bsz])
            Ms.append(tr.M_np[fb].mean())
            model.zero_grad(set_to_none=True)
            # chunk by M (same padding policy as data.batches) to bound memory; gradient of the sum is additive
            order = fb[np.argsort(tr.M_np[fb])]
            chunks, cur, budget = [], [], 0
            for f in order:
                mp = int(-(-tr.M_np[f] // 16) * 16)
                if cur and (len(cur) + 1) * mp > 32768:
                    chunks.append(np.array(cur)); cur = []
                cur.append(f)
            chunks.append(np.array(cur))
            for ch in chunks:
                b = tr.gather(ch, with_pos=True)
                L = per_feed_loss(model, b, prand[torch.from_numpy(ch)])
                (L.sum() / bsz).backward()
            G[i] = torch.cat([p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel())
                              for _, p in params]).numpy()
            if i % 8 == 0:
                P(f"  {name} batch {i}/{n_b} {time.time() - t1:.0f}s")

        def bsimple(Gs):
            n = Gs.shape[0]
            gbar = Gs.mean(0, dtype=np.float64)
            trS = bsz * float(((Gs - gbar) ** 2).sum()) / (n - 1)
            g2 = float((gbar ** 2).sum()) - trS / (n * bsz)
            return trS / g2 if g2 > 0 else float("inf"), trS, g2

        res = {}
        for lab, mask in (("all", np.ones_like(is_mu)), ("muon", is_mu)):
            Gm = G[:, mask]
            bs, trS, g2 = bsimple(Gm)
            h1, h2 = bsimple(Gm[: n_b // 2])[0], bsimple(Gm[n_b // 2:])[0]
            k = 8
            grp = np.array_split(np.arange(n_b), k)
            jk = []
            for g in grp:
                keep = np.setdiff1d(np.arange(n_b), g)
                _, t_, g_ = bsimple(Gm[keep])
                jk.append((t_, g_))
            jt, jg = np.array(jk).T
            # jackknife SE of trSigma and |G|^2, and of B_simple = trS/g2 (delta on the ratio of jackknife means)
            se_t = np.sqrt((k - 1) / k * ((jt - jt.mean()) ** 2).sum())
            se_g = np.sqrt((k - 1) / k * ((jg - jg.mean()) ** 2).sum())
            jb = jt / np.where(jg > 0, jg, np.nan)
            se_b = np.sqrt((k - 1) / k * np.nansum((jb - np.nanmean(jb)) ** 2))
            res[lab] = bs
            P(f"{name} [{lab}] B_simple {bs:.0f} feeds (jackknife SE {se_b:.0f}; split halves {h1:.0f} / {h2:.0f}); "
              f"trSigma {trS:.4g} +- {se_t:.2g}; |G|^2 {g2:.4g} +- {se_g:.2g}; noise share of |Gbar|^2 "
              f"{trS / (n_b * bsz) / (g2 + trS / (n_b * bsz)):.2f}; feeds {n_b * bsz}; mean M {np.mean(Ms):.1f}; "
              f"step {ck.get('step', 'sel')}; {time.time() - t1:.0f}s")
        del model, G
    P(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main(sys.argv[1:])
