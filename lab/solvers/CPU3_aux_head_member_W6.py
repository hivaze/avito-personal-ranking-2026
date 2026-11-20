"""CPU3 / W6: the K9 production-order aux head of the train-only NN members as a signal (CPU only, read-only on the project).

  replay <split> <member> [...]   CPU re-inference of train-only members (solvers/auxexp copy of nn/model.py with the aux head
                                  computed in eval mode + a domain-vector override). Two passes over the same batches:
                                  rl   = default val/test domain (random-like one-hot [0,1,0]) -> main_rl, aux_rl
                                  prod = production one-hot [1,0,0] (the domain the aux head was trained on) -> main_prod, aux_prod
                                  writes solvers/out/auxexp/<member>_<split>_<kind>.npy (float32, aligned with data/flat/<split>/cand rows)
  stage1                          replay check, standalone NDCG of the 12-member z-mean of aux_prod (primary) / aux_rl (secondary),
                                  rho with the 12-member main train-only z-mean, frontier gate gap <= 0.049 (1 - rho)
  stage2                          (only if stage 1 passes) nested GroupKFold(5)-by-user choice of w in {0.1, 0.25}

Pre-declared (task spec): primary = aux_prod; STOP if gap > 0.049 (1 - rho) (sol_ensemble.md 2.5 break-even, val).
"""
import json
import os
import sys
import time

for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = os.environ.get("AUX_THREADS", "16")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import polars as pl

ROOT = "<repo>"
OUTR = f"{ROOT}/solvers/out/auxexp"  # replay outputs
OUT = f"{ROOT}/solvers/out/CPU3_aux_head_member_W6"  # analysis outputs
os.makedirs(OUTR, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f"{ROOT}/src")
from metrics import ndcg_per_feed  # noqa: E402

F5 = ["v12_all_e10", "v12_all_s81", "v12_all_s82", "v12_allprew_s83", "v12_allprew_s84", "v12_allprew_s85"]
F6 = ["v13_lr1_mix1", "v14_s92", "v14_s93", "v14_s94", "v14_s95", "v14_s96"]
MEMBERS = F5 + F6
KINDS = ["main_rl", "aux_rl", "main_prod", "aux_prod"]


def rpath(m, split, kind):
    return f"{OUTR}/{m}_{split}_{kind}.npy"


# ----------------------------------------------------------------------------------------------------------- replay
def replay(split, members):
    import torch
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    sys.path.insert(0, f"{ROOT}/solvers/auxexp")
    import data as D
    from model import Ranker
    d = D.SplitData(split, torch.device("cpu"), max_hist=256)
    xt_cur = None
    prod = torch.tensor([1.0, 0.0, 0.0])
    for m in members:
        if all(os.path.exists(rpath(m, split, k)) for k in KINDS):
            print(m, "done already", flush=True)
            continue
        t0 = time.time()
        ck = torch.load(f"{ROOT}/nn/ckpt/{m}.pt", map_location="cpu", weights_only=False)
        assert ck["args"]["max_hist"] == 256
        cfg = ck["cfg"]
        assert cfg.get("aux_pos") and cfg.get("domain") and not cfg.get("id_buckets"), m
        model = Ranker(**cfg)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        if cfg.get("xt_names") != xt_cur:
            d.xt = None
            if cfg.get("xt_names"):
                d.attach_xt(cfg["xt_names"])
            xt_cur = cfg.get("xt_names")
        outs = {k: torch.zeros(d.n_cand, dtype=torch.float32) for k in KINDS}
        with torch.no_grad():
            for fb in d.batches(np.arange(d.n), 32768, 512, shuffle=False):
                b = d.gather(fb)
                X = model.dense(b)
                model.dom_override = None  # infer_domain = 1 (random-like), as nn/predict.py
                lg = model(b, X=X, use_pos=False)
                d.scatter(outs["main_rl"], b, lg)
                d.scatter(outs["aux_rl"], b, model.aux_out)
                model.dom_override = prod
                lg = model(b, X=X, use_pos=False)
                d.scatter(outs["main_prod"], b, lg)
                d.scatter(outs["aux_prod"], b, model.aux_out)
                model.dom_override = None
        for k, v in outs.items():
            np.save(rpath(m, split, k), v.numpy().astype(np.float32))
        ref = np.load(f"{ROOT}/preds/nn_{m}_trainonly_{split}.npy")
        c = float(np.corrcoef(ref.astype(np.float64), outs["main_rl"].numpy().astype(np.float64))[0, 1])
        print(f"{m} {split}: main_rl corr with stored trainonly {c:.7f}, max abs diff "
              f"{np.abs(ref - outs['main_rl'].numpy()).max():.4f}, {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------------------------------------- analysis
class Val:
    def __init__(self, u0_singleton=True):
        from sklearn.model_selection import GroupKFold
        f = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet")
        self.M = f["M"].to_numpy().astype(np.int64)
        self.off = f["cand_off"].to_numpy().astype(np.int64)
        self.user = f["user_id"].to_numpy().astype(np.int64)
        self.ts = f["timestamp"].to_numpy().astype(np.int64)
        self.n = len(self.M)
        self.fid = np.repeat(np.arange(self.n), self.M)
        self.y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
        day = (self.ts + 3 * 3600) // 86400
        self.day = np.where(day <= day.min() + 1, 0, 1)  # 0 = Tue, 1 = Thu (as nn/pcmp.py)
        g = self.user.copy()
        if u0_singleton:  # nn/finetune.py --u0_singleton (used by the adopted n2cpv3 fine-tune)
            u0 = g == 0
            g = np.where(u0, -1 - np.arange(self.n), g)
        self.fold = np.full(self.n, -1)
        for k, (_, ho) in enumerate(GroupKFold(n_splits=5).split(np.arange(self.n), groups=g)):
            self.fold[ho] = k
        luo = np.load(f"{ROOT}/data/feats/val/it_hist_all_luo.npy", mmap_mode="r")
        cold = (np.asarray(luo) == 0).astype(np.float64)
        pos_cold = np.bincount(self.fid, cold * self.y, self.n)
        npos = np.bincount(self.fid, self.y, self.n)
        self.allcold = (npos > 0) & (pos_cold == npos)  # every positive is a cold (zero LUO popularity) candidate

    def z(self, s):
        s = np.asarray(s, dtype=np.float64)
        mu = np.bincount(self.fid, s, self.n) / self.M
        sd = np.sqrt(np.maximum(np.bincount(self.fid, s * s, self.n) / self.M - mu ** 2, 0)) + 1e-6
        return (s - mu[self.fid]) / sd[self.fid]

    def pf(self, s):
        return ndcg_per_feed(s, self.y, self.off, self.M)

    def rho(self, za, zb):  # mean within-feed Pearson corr of two per-feed z-scored scores (as solvers/ensemble/exp7_pairs.py)
        return float((np.bincount(self.fid, za * zb, self.n) / self.M).mean())

    def slices(self, d):
        masks = {"all": np.ones(self.n, bool), "M<50": self.M < 50, "M>=50": self.M >= 50, "Tue": self.day == 0,
                 "Thu": self.day == 1, "allcold": self.allcold}
        return {k: (float(d[m].mean()), float(d[m].std() / np.sqrt(m.sum()))) for k, m in masks.items()}


def fmt(sl):
    return "  ".join(f"{k} {m:+.5f}+-{s:.5f}" for k, (m, s) in sl.items())


def zmean(V, arrs):
    return V.z(np.mean([V.z(a) for a in arrs], 0))


def stage1():
    V = Val()
    rep = {"members": MEMBERS}
    # replay check
    chk = {}
    for m in MEMBERS:
        ref = np.load(f"{ROOT}/preds/nn_{m}_trainonly_val.npy").astype(np.float64)
        mr = np.load(rpath(m, "val", "main_rl")).astype(np.float64)
        chk[m] = float(np.corrcoef(ref, mr)[0, 1])
    rep["replay_corr"] = chk
    rep["replay_ok"] = bool(min(chk.values()) >= 0.99999)
    print("replay corr main_rl vs stored trainonly val: min %.7f  ok=%s" % (min(chk.values()), rep["replay_ok"]))
    main = zmean(V, [np.load(f"{ROOT}/preds/nn_{m}_trainonly_val.npy") for m in MEMBERS])
    sig = {k: zmean(V, [np.load(rpath(m, "val", k)) for m in MEMBERS]) for k in KINDS}
    nd_main = V.pf(main)
    rep["main_TO_zmean_ndcg"] = float(nd_main.mean())
    print(f"main train-only 12-member z-mean (stored preds): val NDCG {nd_main.mean():.5f}")
    rows = {}
    for k in ["aux_prod", "aux_rl", "main_prod", "main_rl"]:
        nd = V.pf(sig[k])
        rho = V.rho(main, sig[k])
        gap = float(nd_main.mean() - nd.mean())
        be = 0.049 * (1 - rho)
        # per-member single-model standalone NDCG of this signal
        per_m = [float(V.pf(np.load(rpath(m, "val", k))).mean()) for m in MEMBERS]
        rows[k] = {"ndcg": float(nd.mean()), "ndcg_se": float(nd.std() / np.sqrt(V.n)), "rho": rho, "gap": gap,
                   "break_even": be, "gate_pass": bool(gap <= be), "member_ndcg_mean": float(np.mean(per_m)),
                   "member_ndcg_min": float(np.min(per_m)), "member_ndcg_max": float(np.max(per_m)),
                   "slices_vs_main": V.slices(nd - nd_main)}
        print(f"{k:9s}: z-mean NDCG {nd.mean():.5f} (members {np.min(per_m):.5f}..{np.max(per_m):.5f}), rho {rho:.4f}, "
              f"gap {gap:+.5f}, break-even 0.049(1-rho) = {be:.5f} -> gate {'PASS' if gap <= be else 'FAIL'}", flush=True)
        print(f"           standalone minus main by slice: {fmt(rows[k]['slices_vs_main'])}")
    rep["signals"] = rows
    # within-member diagnostics: corr of aux with the member's own main head, and aux_prod vs aux_rl
    diag = {}
    for m in MEMBERS:
        a_p, a_r = V.z(np.load(rpath(m, "val", "aux_prod"))), V.z(np.load(rpath(m, "val", "aux_rl")))
        mm = V.z(np.load(rpath(m, "val", "main_rl")))
        diag[m] = {"rho_auxprod_main": V.rho(a_p, mm), "rho_auxrl_main": V.rho(a_r, mm), "rho_auxprod_auxrl": V.rho(a_p, a_r)}
    rep["member_diag"] = diag
    print("member mean rho(aux_prod, own main) %.4f, rho(aux_rl, own main) %.4f, rho(aux_prod, aux_rl) %.4f" % tuple(
        np.mean([diag[m][k] for m in MEMBERS]) for k in ["rho_auxprod_main", "rho_auxrl_main", "rho_auxprod_auxrl"]))
    # no blend of any kind is computed in stage 1 (pre-declared: if the gate fails, STOP with no blend attempted)
    g = rows["aux_prod"]
    rep["decision"] = "PASS -> stage 2" if g["gate_pass"] else "STOP (frontier gate failed; W6 closed, no blend attempted)"
    print("DECISION (primary aux_prod):", rep["decision"])
    json.dump(rep, open(f"{OUT}/stage1.json", "w"), indent=1)


def stage2():
    """Nested GroupKFold(5)-by-user choice of w in {0.1, 0.25} for TO' = z(main) + w z(aux_prod) per member;
    final = 0.5 z(FT n2cpv3 12-member mean) + 0.5 z(TO' 12-member mean) vs the same mix at w = 0."""
    s1 = json.load(open(f"{OUT}/stage1.json"))
    assert s1["signals"]["aux_prod"]["gate_pass"], "stage 1 gate failed: stage 2 must not run"
    V = Val()
    ft = zmean(V, [np.load(f"{ROOT}/preds/nn_{m}_n2cpv3_val_oof.npy") for m in MEMBERS])
    zm = [V.z(np.load(f"{ROOT}/preds/nn_{m}_trainonly_val.npy")) for m in MEMBERS]
    za = [V.z(np.load(rpath(m, "val", "aux_prod"))) for m in MEMBERS]
    pf = {}
    for w in (0.0, 0.1, 0.25):
        to = zmean(V, [a + w * b for a, b in zip(zm, za)])
        pf[w] = V.pf(V.z(ft) + V.z(to))
    nested = np.empty(V.n)
    choice = {}
    for k in range(5):
        tr = V.fold != k
        wk = max((0.1, 0.25), key=lambda w: pf[w][tr].mean())
        choice[k] = wk
        nested[~tr] = pf[wk][~tr]
    d = nested - pf[0.0]
    sl = V.slices(d)
    z = sl["all"][0] / sl["all"][1]
    ok = sl["all"][0] > 2 * sl["all"][1] and z >= 2.5 and all(sl[s][0] >= -2 * sl[s][1] for s in ["M<50", "M>=50", "Tue", "Thu"])
    rep = {"base_w0": float(pf[0.0].mean()), "w0.1": float(pf[0.1].mean()), "w0.25": float(pf[0.25].mean()),
           "fold_choice": choice, "nested": float(nested.mean()), "delta_slices": sl, "z": float(z), "accept": bool(ok)}
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(f"{OUT}/stage2.json", "w"), indent=1)


def diag_val():
    """Context for the stage-1 numbers (no blend): random baseline, single-feature NDCG of f1..f5 and their within-feed
    corr with the aux_prod / main z-means."""
    V = Val()
    main = zmean(V, [np.load(f"{ROOT}/preds/nn_{m}_trainonly_val.npy") for m in MEMBERS])
    aux = zmean(V, [np.load(rpath(m, "val", "aux_prod")) for m in MEMBERS])
    rng = np.random.default_rng(0)
    rnd = [float(V.pf(rng.random(len(V.y))).mean()) for _ in range(5)]
    rep = {"random_ndcg": float(np.mean(rnd)), "random_ndcg_sd": float(np.std(rnd))}
    print(f"random order NDCG {np.mean(rnd):.5f} (sd over 5 draws {np.std(rnd):.5f})")
    for i in range(1, 6):
        f = np.load(f"{ROOT}/data/flat/val/cand_f{i}.npy").astype(np.float64)
        zf = V.z(f)
        r = {"ndcg_pos": float(V.pf(f).mean()), "ndcg_neg": float(V.pf(-f).mean()),
             "rho_aux_prod": V.rho(aux, zf), "rho_main": V.rho(main, zf)}
        rep[f"f{i}"] = r
        print(f"f{i}: NDCG +f {r['ndcg_pos']:.5f} / -f {r['ndcg_neg']:.5f}; within-feed corr with aux_prod z-mean "
              f"{r['rho_aux_prod']:+.4f}, with main TO z-mean {r['rho_main']:+.4f}")
    # coldness: within-feed corr of the signals with the zero-LUO-popularity indicator
    luo = np.asarray(np.load(f"{ROOT}/data/feats/val/it_hist_all_luo.npy", mmap_mode="r")).astype(np.float64)
    zl = V.z(np.log1p(np.maximum(luo, 0)))
    rep["rho_aux_prod_logluo"] = V.rho(aux, zl)
    rep["rho_main_logluo"] = V.rho(main, zl)
    print(f"within-feed corr with log1p(it_hist_all_luo): aux_prod {rep['rho_aux_prod_logluo']:+.4f}, main {rep['rho_main_logluo']:+.4f}")
    json.dump(rep, open(f"{OUT}/diag_val.json", "w"), indent=1)


def fidelity(members, n_feeds=3000):
    """Does the aux head imitate the production order? In-sample check on production train feeds (p_random <= 0.05,
    M >= 20, not holdout; fixed seed): within-feed Spearman of aux_prod / aux_rl / main_rl with -pos, top-1 agreement,
    and click NDCG of each score on these (position-biased) feeds. Diagnostic only."""
    import torch
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    sys.path.insert(0, f"{ROOT}/solvers/auxexp")
    import data as D
    from model import Ranker
    from scipy.stats import rankdata
    d = D.SplitData("train", torch.device("cpu"), max_hist=256)
    pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
    pr[d.M_np < 50] = 0.0
    cand = np.where((pr <= 0.05) & (d.M_np >= 20) & (d.M_np <= 400))[0]
    feeds = np.sort(np.random.default_rng(777).choice(cand, n_feeds, replace=False))
    M, off = d.M_np[feeds], d.cand_off_np[feeds]
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off, M)])
    pos = d.c_pos.numpy()[rows].astype(np.float64)
    y = d.target_np[rows].astype(np.float64)
    offs = np.concatenate([[0], np.cumsum(M)[:-1]])
    fid = np.repeat(np.arange(len(M)), M)
    prod = torch.tensor([1.0, 0.0, 0.0])
    rep = {"n_feeds": int(len(feeds)), "mean_M": float(M.mean())}
    xt_cur = None

    def wf_spearman(s, t):
        out = []
        for i in range(len(M)):
            a, b = s[offs[i]:offs[i] + M[i]], t[offs[i]:offs[i] + M[i]]
            ra, rb = rankdata(a), rankdata(b)
            if ra.std() > 0 and rb.std() > 0:
                out.append(np.corrcoef(ra, rb)[0, 1])
        return float(np.mean(out))

    for m in members:
        ck = torch.load(f"{ROOT}/nn/ckpt/{m}.pt", map_location="cpu", weights_only=False)
        model = Ranker(**ck["cfg"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        if ck["cfg"].get("xt_names") != xt_cur:
            d.xt = None
            d.attach_xt(ck["cfg"]["xt_names"])
            xt_cur = ck["cfg"].get("xt_names")
        outs = {k: torch.zeros(d.n_cand, dtype=torch.float32) for k in ["main_rl", "aux_rl", "aux_prod", "main_prod"]}
        with torch.no_grad():
            for fb in d.batches(feeds, 32768, 512, shuffle=False):
                b = d.gather(fb)
                X = model.dense(b)
                model.dom_override = None
                lg = model(b, X=X, use_pos=False)
                d.scatter(outs["main_rl"], b, lg)
                d.scatter(outs["aux_rl"], b, model.aux_out)
                model.dom_override = prod
                lg = model(b, X=X, use_pos=False)
                d.scatter(outs["main_prod"], b, lg)
                d.scatter(outs["aux_prod"], b, model.aux_out)
                model.dom_override = None
        r = {}
        for k, v in outs.items():
            s = v.numpy()[rows].astype(np.float64)
            top1 = np.mean([np.argmax(s[o:o + mm]) == np.argmin(pos[o:o + mm]) for o, mm in zip(offs, M)])
            r[k] = {"spearman_neg_pos": wf_spearman(s, -pos), "top1_is_pos0": float(top1),
                    "click_ndcg": float(ndcg_per_feed(s, y, offs, M).mean())}
        r["neg_pos_click_ndcg"] = float(ndcg_per_feed(-pos, y, offs, M).mean())
        rep[m] = r
        print(m, json.dumps(r), flush=True)
    json.dump(rep, open(f"{OUT}/fidelity_train_prod.json", "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "replay":
        replay(sys.argv[2], sys.argv[3:])
    elif cmd == "stage1":
        stage1()
    elif cmd == "stage2":
        stage2()
    elif cmd == "diag_val":
        diag_val()
    elif cmd == "fidelity":
        fidelity(sys.argv[2:])
