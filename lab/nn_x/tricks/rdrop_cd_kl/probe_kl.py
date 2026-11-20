"""RDROP-CD KL probe (TRICKS_SWEEP_1113 section 3 item 4). Rule and measurement: PREDECLARED.txt in this directory.

Two train-mode (dropout-on) no-grad forwards of each decision checkpoint on the same 64 regular training batches, plus one
eval-mode forward; per-feed listwise softmax over the valid candidates; symmetric KL, within-feed Spearman, top-1 change.
Read-only use of nn_x/abl/{model.py,data.py} (run with PYTHONDONTWRITEBYTECODE=1). Outputs only in this directory.

  CUDA_VISIBLE_DEVICES=N PYTHONDONTWRITEBYTECODE=1 python probe_kl.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4", "this probe runs on GPU N only"
torch.set_num_threads(12)
ROOT = Path("<repo>")
sys.path.insert(0, str(ROOT / "nn_x" / "abl"))
from data import SplitData  # noqa: E402
from model import Ranker  # noqa: E402

OUT = Path(__file__).resolve().parent
N_BATCHES = 64
BATCH_SEED = 1113
TORCH_SEED = 1113
DECISION = {
    "a_m02ref_r101_sel": ROOT / "nn_x/w5/ckpt/w5_cd_m02ref_r101.pt",
    "b_m02ref_r101_last": ROOT / "nn_x/w5/ckpt/w5_cd_m02ref_r101@last.pt",
    "c_v17_s101_at77500_cdstart": ROOT / "nn/ckpt/v17_pre2s_s101@77500.pt",
    "d_e10_base_s951": ROOT / "nn_x/w5/ckpt/w5_e10_base_s951.pt",
}
SCALE_REF = ROOT / "nn_x/w5/ckpt/w5_cd_m02ref_r102.pt"  # info: eval-mode inter-seed KL vs (a)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(OUT / "probe_kl.log", "a") as f:
        f.write(s + "\n")


def midrank(x, m):
    """Within-row mid-ranks (ties averaged) of x [B,M] over valid entries m [B,M]; invalid entries -> 0."""
    xi, xj = x[:, :, None], x[:, None, :]
    mj = m[:, None, :]
    less = ((xj < xi) & mj).sum(-1).double()
    eq = ((xj == xi) & mj).sum(-1).double() - 1.0
    return torch.where(m, less + 0.5 * eq, torch.zeros_like(less))


def pair_stats(l1, l2, m):
    """Per-feed symmetric KL (nats), Spearman, top-1 change between logits l1, l2 [B,M] (valid mask m)."""
    a = l1.double().masked_fill(~m, -torch.inf)
    b = l2.double().masked_fill(~m, -torch.inf)
    lp1, lp2 = torch.log_softmax(a, 1), torch.log_softmax(b, 1)
    p1, p2 = lp1.exp(), lp2.exp()
    d = torch.where(m, lp1 - lp2, torch.zeros_like(lp1))
    kl12 = (p1 * d).sum(1)
    kl21 = (p2 * (-d)).sum(1)
    sym = 0.5 * (kl12 + kl21)
    cnt = m.sum(1).double()
    r1, r2 = midrank(l1, m), midrank(l2, m)
    mu1 = r1.sum(1, keepdim=True) / cnt[:, None]
    mu2 = r2.sum(1, keepdim=True) / cnt[:, None]
    c1 = torch.where(m, r1 - mu1, torch.zeros_like(r1))
    c2 = torch.where(m, r2 - mu2, torch.zeros_like(r2))
    den = (c1.pow(2).sum(1) * c2.pow(2).sum(1)).sqrt()
    sp = torch.where(den > 0, (c1 * c2).sum(1) / den.clamp_min(1e-300), torch.full_like(den, torch.nan))
    t1 = a.argmax(1) != b.argmax(1)
    return sym, kl12, kl21, sp, t1


def feed_ce(lg, y, m):
    lp = torch.log_softmax(lg.double().masked_fill(~m, -torch.inf), 1)
    ys = y.double().sum(1)
    ce = -(torch.where(m, y.double() * lp, torch.zeros_like(lp))).sum(1) / ys.clamp_min(1)
    return torch.where(ys > 0, ce, torch.full_like(ce, torch.nan))


def load(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = Ranker(**ck["cfg"]).to(dev)
    m.load_state_dict(ck["state_dict"], strict=True)
    return m, ck["cfg"]


def summ(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)), p90=float(np.quantile(v, 0.9)),
                p99=float(np.quantile(v, 0.99)), max=float(v.max()), p10=float(np.quantile(v, 0.1)))


def cluster_se(vals, bid):
    """SE of the feed-mean with batches as clusters (ratio estimator)."""
    ok = np.isfinite(vals)
    vals, bid = vals[ok], bid[ok]
    ub = np.unique(bid)
    s = np.array([vals[bid == k].sum() for k in ub])
    n = np.array([(bid == k).sum() for k in ub], dtype=np.float64)
    mu = s.sum() / n.sum()
    G = len(ub)
    return float(np.sqrt(G / (G - 1) * (((s - mu * n) ** 2).sum()) / n.sum() ** 2))


def main():
    t0 = time.time()
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    log(f"==== {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC probe_kl.py (rule: PREDECLARED.txt)")
    cfgs = {k: torch.load(p, map_location="cpu", weights_only=False)["cfg"] for k, p in DECISION.items()}
    xt0 = cfgs["a_m02ref_r101_sel"]["xt_names"]
    for k, c in cfgs.items():
        assert c["xt_names"] == xt0 and c.get("domain") == cfgs["a_m02ref_r101_sel"].get("domain"), k
        assert not c.get("use_pos_tower") and c.get("xt_drop", 0) == 0 and not c.get("id_buckets"), k
        log(f"cfg {k}: drop {c['drop']} attn_drop {c.get('attn_drop')} retr_m {c.get('retr_m')} arch {c['arch']}")
    tr = SplitData("train", dev, max_hist=256)
    tr.attach_xt(xt0)
    pr = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
    pr[tr.M_np < 50] = 0.0
    if cfgs["a_m02ref_r101_sel"].get("domain"):
        prt = torch.from_numpy(pr).to(dev)
        tr.dom = torch.stack([1 - prt, prt, torch.zeros_like(prt)], 1)
    cand = np.where((pr > 0.9) & (tr.M_np >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    feeds = np.setdiff1d(np.arange(tr.n), hold)
    bl = tr.batches(feeds, 32768, 512, shuffle=True, rng=np.random.default_rng(BATCH_SEED))[:N_BATCHES]
    nf = sum(len(x) for x in bl)
    Ms = np.concatenate([tr.M_np[x] for x in bl])
    log(f"train feeds {len(feeds)} (holdout {len(hold)} excluded); {len(bl)} batches, {nf} feeds, M range {Ms.min()}-{Ms.max()}, "
        f"M>=2 {int((Ms >= 2).sum())}, load {time.time() - t0:.0f}s")
    batches = [tr.gather(fb, with_pos=True) for fb in bl]

    results, ref_eval = {}, None
    for name, path in list(DECISION.items()) + [("info_m02ref_r102_sel", SCALE_REF)]:
        model, cfg = load(path, dev)
        torch.manual_seed(TORCH_SEED)
        rec = {k: [] for k in ("M", "haspos", "bid", "sym_tt", "kl12", "kl21", "sp_tt", "t1_tt", "sym_et", "sp_et",
                               "t1_et", "ce_eval", "ce_train1", "sym_seed", "sp_seed", "t1_seed")}
        evals = []
        with torch.no_grad():
            for bi, b in enumerate(batches):
                model.train()
                X = model.dense(b)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l1 = model(b, X, use_pos=True)
                    l2 = model(b, X, use_pos=True)
                model.eval()
                Xe = model.dense(b)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    le = model(b, Xe, use_pos=True)
                m = b["cmask"]
                sym, k12, k21, sp, t1 = pair_stats(l1, l2, m)
                syme, _, _, spe, t1e = pair_stats(le, l1, m)
                rec["M"].append(b["M"]); rec["haspos"].append(b["y"].sum(1) > 0)
                rec["bid"].append(torch.full_like(b["M"], bi))
                rec["sym_tt"].append(sym); rec["kl12"].append(k12); rec["kl21"].append(k21)
                rec["sp_tt"].append(sp); rec["t1_tt"].append(t1)
                rec["sym_et"].append(syme); rec["sp_et"].append(spe); rec["t1_et"].append(t1e)
                rec["ce_eval"].append(feed_ce(le, b["y"], m)); rec["ce_train1"].append(feed_ce(l1, b["y"], m))
                evals.append(le.detach().clone())
                if ref_eval is not None:
                    ss, _, _, sps, t1s = pair_stats(ref_eval[bi], le, m)
                else:
                    ss = sps = torch.full_like(sym, torch.nan); t1s = torch.zeros_like(t1)
                rec["sym_seed"].append(ss); rec["sp_seed"].append(sps); rec["t1_seed"].append(t1s)
        if name == "a_m02ref_r101_sel":
            ref_eval = evals
        R = {k: torch.cat(v).double().cpu().numpy() if k not in ("haspos", "t1_tt", "t1_et", "t1_seed")
             else torch.cat(v).cpu().numpy() for k, v in rec.items()}
        np.savez_compressed(OUT / f"perfeed_{name}.npz", **R)
        M, hp = R["M"], R["haspos"].astype(bool)
        g2 = M >= 2
        res = dict(ckpt=str(path), n_feeds=int(len(M)), n_feeds_M2=int(g2.sum()))
        res["S_primary_mean_symKL_tt_M2"] = float(R["sym_tt"][g2].mean())
        res["S_cluster_se"] = cluster_se(R["sym_tt"][g2], R["bid"][g2])
        res["symKL_tt_M2"] = summ(R["sym_tt"][g2])
        res["symKL_tt_allfeeds_mean"] = float(R["sym_tt"].mean())
        res["symKL_tt_M2_lt50_mean"] = float(R["sym_tt"][g2 & (M < 50)].mean())
        res["symKL_tt_Mge50_mean"] = float(R["sym_tt"][M >= 50].mean())
        res["symKL_tt_haspos_M2_mean"] = float(R["sym_tt"][g2 & hp].mean())
        res["KL12_mean_M2"] = float(R["kl12"][g2].mean()); res["KL21_mean_M2"] = float(R["kl21"][g2].mean())
        res["spearman_tt_M2"] = summ(R["sp_tt"][g2])
        res["top1_change_tt_M2"] = float(R["t1_tt"][g2].mean())
        res["symKL_et_M2"] = summ(R["sym_et"][g2])
        res["spearman_et_M2"] = summ(R["sp_et"][g2])
        res["top1_change_et_M2"] = float(R["t1_et"][g2].mean())
        res["ce_eval_mean_haspos"] = float(np.nanmean(R["ce_eval"]))
        res["ce_train1_mean_haspos"] = float(np.nanmean(R["ce_train1"]))
        if name.startswith("info"):
            res["interseed_eval_symKL_vs_a_M2"] = summ(R["sym_seed"][g2])
            res["interseed_eval_spearman_vs_a_M2"] = summ(R["sp_seed"][g2])
            res["interseed_eval_top1_change_vs_a_M2"] = float(R["t1_seed"][g2].mean())
        results[name] = res
        log(f"{name}: S = {res['S_primary_mean_symKL_tt_M2']:.6g} nats (cluster SE {res['S_cluster_se']:.2g}); "
            f"median {res['symKL_tt_M2']['median']:.3g} p90 {res['symKL_tt_M2']['p90']:.3g}; "
            f"Spearman mean {res['spearman_tt_M2']['mean']:.5f}; top1 change {res['top1_change_tt_M2']:.4f}; "
            f"eval-vs-train symKL {res['symKL_et_M2']['mean']:.6g}")
        del model
        torch.cuda.empty_cache()
    Sd = {k: results[k]["S_primary_mean_symKL_tt_M2"] for k in DECISION}
    closed = all(v < 1e-3 for v in Sd.values())
    verdict = ("CLOSED as inert: S < 1e-3 nats on all four decision checkpoints" if closed else
               "QUEUED: S >= 1e-3 nats on " + ", ".join(k for k, v in Sd.items() if v >= 1e-3))
    results["_decision"] = dict(S=Sd, S_max=max(Sd.values()), threshold=1e-3, closed=closed, verdict=verdict,
                                n_batches=N_BATCHES, batch_seed=BATCH_SEED, torch_seed=TORCH_SEED,
                                minutes=(time.time() - t0) / 60)
    (OUT / "probe_kl.json").write_text(json.dumps(results, indent=2))
    log(f"VERDICT: {verdict}; S = {Sd}; {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
