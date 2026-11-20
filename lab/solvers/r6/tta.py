"""R6: inference-time tricks (TTA) on existing train-only checkpoints, CPU replay (no training, read-only on project).

Pre-declaration: reports/r6_tta.md section 0. Arms (K passes per member, mean of raw logits):
  det  deterministic eval replay (baseline; also the replay check vs preds/nn_<v>_trainonly_val.npy)
  T1   MC-dropout: model.eval(), then .train() only on nn.Dropout + Attn modules (attention / residual / FF / retrieval
       dropout, p = drop = 0.1); Ranker itself stays eval (no xt-drop / aux branch). K = 8.
  T2   history-event dropout: each valid history event dropped with p = 0.1, survivors compacted to a contiguous
       newest-first prefix (recency indices shift, H decreases), all in-model history-derived inputs recomputed. K = 8.
  T3   history crop: keep the H' most recent events, H' in {256 (all), 192, 128, 64}; 4 passes averaged.
  W256 (diagnostic, non-decisional) det with every batch's history padded to width 256: HistRetrieval's top-16 over
       history is ambiguous among duplicate history items (equal cosine) and torch.topk breaks ties differently by
       tensor width, so this measures the size of that implementation-defined noise. T2/T3 keep the det batch width.
Precomputed xt inputs (ctx_*, pre2_hmax32, seq_cv_*) are NOT recomputed under T2/T3 (cannot be; documented).

  python solvers/r6/tta.py run --members v17_pre2s_s101,v17_pre2s_s102 --arms det,T3,T1,T2
  python solvers/r6/tta.py check            # replay check (per-feed Spearman vs stored GPU preds)
  python solvers/r6/tta.py eval             # deltas, slices, acceptance -> solvers/r6/out/tta_eval.json
  python solvers/r6/tta.py write <arm>      # pass-averaged member preds -> solvers/r6/out/ (only if the arm passed)
Per-pass predictions (float32, val cand-aligned) are cached in the scratchpad (PASSES); `run` skips existing files.
"""
import argparse
import json
import os
import sys
import time

THREADS = 12
for _k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[_k] = str(THREADS)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

torch.set_num_threads(THREADS)
import data as D  # noqa: E402
from model import Attn, Ranker  # noqa: E402

ROOT = "<repo>"
CKDIR = f"{ROOT}/nn/ckpt"
SCR = "/tmp/scratchpad/r6/tta"
PASSES = f"{SCR}/passes"
LOGS = f"{SCR}/logs"
OUT = f"{ROOT}/solvers/r6/out"
MEMBERS = ["v17_pre2s_s101", "v17_pre2s_s102", "v17_pre2s_s103", "v17_pre2s_s104", "v14_s93", "v14_s95"]
K_PASSES = 8
CROPS = [256, 192, 128, 64]
P_HDROP = 0.1
SEED0 = 20261112
ARM_ID = {"T1": 1, "T2": 2}


def seed_of(arm, midx, k):
    return SEED0 + 1000 * ARM_ID[arm] + 10 * midx + k


# ----------------------------------------------------------------------------------------------- model / data
def load(variant):
    ck = torch.load(f"{CKDIR}/{variant}.pt", map_location="cpu", weights_only=False)
    m = Ranker(**ck["cfg"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def set_mc(model, on):
    """T1: dropout-only stochasticity. Ranker (and every other module) stays in eval mode."""
    model.eval()
    n = 0
    if on:
        for mod in model.modules():
            if isinstance(mod, (nn.Dropout, Attn)):
                mod.train()
                n += 1
    return n


def compact_hist(b, keep):
    """Keep only history events with keep=True ([B,K] bool, subset of hmask) and move them, in their original
    newest-first order, to a contiguous prefix; update every history tensor of the batch, the masks and H."""
    order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)
    Kx = keep.shape[1]
    newH = keep.sum(1)
    hmask = torch.arange(Kx)[None, :] < newH[:, None]
    g2 = lambda x: torch.gather(x, 1, order)  # noqa: E731
    he_raw = torch.gather(b["he_raw"], 1, order[..., None].expand(-1, -1, b["he_raw"].shape[-1]))
    hemb_ok = g2(b["hemb_ok"]) & hmask
    he_raw = he_raw * hemb_ok[..., None].float()  # data.gather zeroes he_raw of invalid / padded events
    b.update(he_raw=he_raw, hemb_ok=hemb_ok, hmask=hmask, hcat=g2(b["hcat"]), het=g2(b["het"]), hes=g2(b["hes"]),
             H=newH.to(b["H"].dtype))
    return b


def hist_drop_fn(p, gen):
    def f(b):
        u = torch.rand(b["hmask"].shape, generator=gen)
        return compact_hist(b, b["hmask"] & (u >= p))
    return f


def pad_hist_fn(width):
    """Diagnostic W256: same history, padded to a fixed width (only changes topk tie-breaking in HistRetrieval)."""
    def f(b):
        k0 = b["hmask"].shape[1]
        if k0 >= width:
            return b
        z = lambda x: torch.cat([x, torch.zeros((x.shape[0], width - k0) + tuple(x.shape[2:]), dtype=x.dtype)], 1)  # noqa: E731
        b.update({k: z(b[k]) for k in ("hmask", "hemb_ok", "hcat", "het", "hes", "he_raw")})
        return b
    return f


def hist_crop_fn(hp):
    def f(b):
        keep = b["hmask"] & (torch.arange(b["hmask"].shape[1])[None, :] < hp)
        return compact_hist(b, keep)
    return f


@torch.no_grad()
def predict(model, d, hist_fn=None, cand_budget=32768, max_feeds=512):
    out = torch.zeros(d.n_cand, dtype=torch.float32)
    for fb in d.batches(np.arange(d.n), cand_budget, max_feeds, shuffle=False):
        b = d.gather(fb)
        if hist_fn is not None:
            b = hist_fn(b)
        lg = model(b, use_pos=False)
        d.scatter(out, b, lg)
    return out.numpy()


def pass_path(v, arm, k):
    return f"{PASSES}/{v}_{arm}_{k}.npy"


def arm_passes(arm):
    """(pass key k, description) for each pass of an arm."""
    if arm in ("det", "W256"):
        return [(0, "eval")]
    if arm in ("T1", "T2"):
        return [(k, f"seed {k}") for k in range(K_PASSES)]
    if arm == "T3":  # crop 256 == det (max_hist 256), so the arm reuses det_0 for H'=all
        return [(hp, f"crop {hp}") for hp in CROPS if hp != 256]
    raise ValueError(arm)


def cmd_run(a):
    os.makedirs(PASSES, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    members = a.members.split(",")
    arms = a.arms.split(",")
    d, d_xt = None, None
    for v in members:
        midx = MEMBERS.index(v)
        model, ck = load(v)
        assert ck["args"]["max_hist"] == 256, ck["args"]["max_hist"]
        if d is None:
            d = D.SplitData("val", torch.device("cpu"), max_hist=256)
        xt = tuple(ck["cfg"].get("xt_names") or [])
        if xt != d_xt:
            d.xt = None
            if xt:
                d.attach_xt(list(xt))
            d_xt = xt
        for arm in arms:
            for k, desc in arm_passes(arm):
                pp = pass_path(v, arm, k)
                if os.path.exists(pp):
                    continue
                hist_fn = None
                if arm == "T1":
                    torch.manual_seed(seed_of(arm, midx, k))
                    nmod = set_mc(model, True)
                else:
                    set_mc(model, False)
                    nmod = 0
                if arm == "T2":
                    g = torch.Generator().manual_seed(seed_of(arm, midx, k))
                    hist_fn = hist_drop_fn(P_HDROP, g)
                elif arm == "T3":
                    hist_fn = hist_crop_fn(k)
                elif arm == "W256":
                    hist_fn = pad_hist_fn(256)
                t0 = time.time()
                pr = predict(model, d, hist_fn)
                dt = time.time() - t0
                set_mc(model, False)
                assert np.isfinite(pr).all()
                np.save(pp + ".tmp.npy", pr.astype(np.float32))
                os.replace(pp + ".tmp.npy", pp)
                rec = dict(member=v, arm=arm, k=k, desc=desc, seconds=round(dt, 1), threads=THREADS, mc_modules=nmod,
                           load=os.getloadavg()[0], t=time.strftime("%H:%M:%S"))
                with open(f"{LOGS}/passes.jsonl", "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(json.dumps(rec), flush=True)


# ----------------------------------------------------------------------------------------------- evaluation
def val_meta():
    import polars as pl
    f = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64)
    off = f["cand_off"].to_numpy().astype(np.int64)
    y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
    day = (f["timestamp"].to_numpy() + 3 * 3600) // 86400
    day = np.where(day <= day.min() + 1, 0, 1)  # same Tue/Thu split as nn/pcmp.py
    fid = np.repeat(np.arange(len(M)), M)
    return M, off, y, day, fid


def fz(s, fid, M):  # == nn/zmean.py fz
    s = s.astype(np.float64)
    mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-6
    return (s - mu[fid]) / sd[fid]


def per_feed_spearman(a, b, off, M):
    from scipy.stats import rankdata
    out = np.empty(len(M))
    for i in range(len(M)):
        sl = slice(off[i], off[i] + M[i])
        ra, rb = rankdata(a[sl]), rankdata(b[sl])
        sa, sb = ra.std(), rb.std()
        out[i] = 1.0 if (sa == 0 and sb == 0) else (np.corrcoef(ra, rb)[0, 1] if sa > 0 and sb > 0 else 0.0)
    return out


def cmd_check(a):
    sys.path.insert(0, f"{ROOT}/src")
    from metrics import ndcg_per_feed
    M, off, y, day, fid = val_meta()
    res = {}
    for v in MEMBERS:
        c = np.load(pass_path(v, "det", 0))
        r = np.load(f"{ROOT}/preds/nn_{v}_trainonly_val.npy")
        sp = per_feed_spearman(c, r, off, M)
        nc, nr = ndcg_per_feed(c, y, off, M), ndcg_per_feed(r, y, off, M)
        res[v] = dict(spearman_mean=float(sp.mean()), spearman_min=float(sp.min()),
                      spearman_p01=float(np.quantile(sp, 0.01)), share_ge_09999=float((sp >= 0.9999).mean()),
                      pearson=float(np.corrcoef(c, r)[0, 1]), max_abs_diff=float(np.abs(c - r).max()),
                      ndcg_cpu=float(nc.mean()), ndcg_gpu=float(nr.mean()),
                      d_ndcg=float((nc - nr).mean()), d_ndcg_se=float((nc - nr).std() / np.sqrt(len(M))),
                      passed=bool(sp.mean() >= 0.9999))
        print(v, json.dumps(res[v]), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/tta_replay_check.json", "w"), indent=1)


def member_pred(v, arm, kmax=None):
    """Pass-averaged raw logits of a member for an arm (T3 includes det_0 as the H'=all pass)."""
    if arm == "det":
        return np.load(pass_path(v, "det", 0)).astype(np.float64)
    if arm == "T3":
        ps = [pass_path(v, "det", 0)] + [pass_path(v, "T3", hp) for hp in CROPS if hp != 256]
    elif arm == "T1+T2":
        ps = [pass_path(v, x, k) for x in ("T1", "T2") for k in range(kmax or K_PASSES)]
    elif arm == "W256":
        ps = [pass_path(v, "W256", 0)]
    elif arm.startswith("crop"):
        hp = int(arm[4:])
        ps = [pass_path(v, "det", 0) if hp == 256 else pass_path(v, "T3", hp)]
    else:
        ps = [pass_path(v, arm, k) for k in range(kmax or K_PASSES)]
    return np.mean([np.load(p).astype(np.float64) for p in ps], 0)


def summarize(d, slices):
    n = len(d)
    out = dict(delta=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(n)))
    out["z"] = out["delta"] / out["se"] if out["se"] > 0 else 0.0
    for nm, msk in slices.items():
        dd = d[msk]
        out[nm] = dict(delta=float(dd.mean()), se=float(dd.std(ddof=1) / np.sqrt(msk.sum())), n=int(msk.sum()))
    return out


def cmd_eval(a):
    sys.path.insert(0, f"{ROOT}/src")
    from metrics import ndcg_per_feed
    M, off, y, day, fid = val_meta()
    slices = {"M<50": M < 50, "M>=50": M >= 50, "Tue": day == 0, "Thu": day == 1}
    nd = lambda s: ndcg_per_feed(s, y, off, M)  # noqa: E731
    det = {v: member_pred(v, "det") for v in MEMBERS}
    det_nd = {v: nd(det[v]) for v in MEMBERS}
    ens_det = np.mean([fz(det[v], fid, M) for v in MEMBERS], 0)
    ens_det_nd = nd(ens_det)
    res = dict(n_feeds=int(len(M)), det=dict(members={v: float(det_nd[v].mean()) for v in MEMBERS},
                                             ensemble=float(ens_det_nd.mean())), arms={}, diag={})
    # GPU-stored reference ensemble (for orientation only)
    gpu = np.mean([fz(np.load(f"{ROOT}/preds/nn_{v}_trainonly_val.npy"), fid, M) for v in MEMBERS], 0)
    res["det"]["ensemble_gpu_stored"] = float(nd(gpu).mean())
    # diagnostic: CPU fp32 replay vs GPU bf16-autocast stored preds (same checkpoints), paired per feed
    res["diag"]["fp32_cpu_minus_bf16_gpu"] = dict(
        ensemble=summarize(ens_det_nd - nd(gpu), slices),
        members={v: summarize(det_nd[v] - nd(np.load(f"{ROOT}/preds/nn_{v}_trainonly_val.npy")), {})
                 for v in MEMBERS})

    def arm_eval(arm, kmax=None):
        mem = {v: member_pred(v, arm, kmax) for v in MEMBERS}
        mres = {}
        for v in MEMBERS:
            dv = nd(mem[v]) - det_nd[v]
            mres[v] = dict(ndcg=float(det_nd[v].mean() + dv.mean()), **summarize(dv, slices))
        ens = np.mean([fz(mem[v], fid, M) for v in MEMBERS], 0)
        de = nd(ens) - ens_det_nd
        er = dict(ndcg=float(ens_det_nd.mean() + de.mean()), **summarize(de, slices))
        npos = sum(mres[v]["delta"] > 0 for v in MEMBERS)
        return dict(members=mres, ensemble=er, members_positive=int(npos),
                    mean_member_delta=float(np.mean([mres[v]["delta"] for v in MEMBERS])),
                    spearman_vs_det_ens=float(np.corrcoef(ens, ens_det)[0, 1]))

    for arm in ("T1", "T2", "T3"):
        r = arm_eval(arm)
        r["passed"] = bool(r["ensemble"]["delta"] > 2 * r["ensemble"]["se"] and r["members_positive"] >= 5)
        res["arms"][arm] = r
        print(arm, "ens", f"{r['ensemble']['delta']:+.5f} +- {r['ensemble']['se']:.5f}", "members+",
              r["members_positive"], "PASS" if r["passed"] else "FAIL", flush=True)
    # non-decisional diagnostics
    for kk in (1, 2, 4):
        for arm in ("T1", "T2"):
            r = arm_eval(arm, kk)
            res["diag"][f"{arm}_K{kk}"] = dict(ensemble=r["ensemble"], mean_member_delta=r["mean_member_delta"],
                                                members_positive=r["members_positive"])
    r = arm_eval("T1+T2")
    res["diag"]["T1+T2_K8each"] = dict(ensemble=r["ensemble"], mean_member_delta=r["mean_member_delta"],
                                       members_positive=r["members_positive"])
    for hp in CROPS[1:]:
        r = arm_eval(f"crop{hp}")
        res["diag"][f"crop{hp}_single"] = dict(ensemble=r["ensemble"], mean_member_delta=r["mean_member_delta"],
                                               members_positive=r["members_positive"],
                                               members={v: r["members"][v]["delta"] for v in MEMBERS})
    if all(os.path.exists(pass_path(v, "W256", 0)) for v in MEMBERS):
        r = arm_eval("W256")
        res["diag"]["W256_tiebreak"] = dict(ensemble=r["ensemble"], mean_member_delta=r["mean_member_delta"],
                                            members_positive=r["members_positive"],
                                            members={v: r["members"][v]["delta"] for v in MEMBERS})
    # timing
    tim = {}
    if os.path.exists(f"{LOGS}/passes.jsonl"):
        for line in open(f"{LOGS}/passes.jsonl"):
            rec = json.loads(line)
            tim.setdefault(rec["arm"], []).append(rec["seconds"])
    res["timing_seconds_per_pass"] = {k: dict(mean=float(np.mean(x)), n=len(x)) for k, x in tim.items()}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/tta_eval.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("det", "timing_seconds_per_pass")}, indent=1))


def cmd_write(a):
    ev = json.load(open(f"{OUT}/tta_eval.json"))
    assert ev["arms"][a.arm]["passed"], f"{a.arm} did not pass its pre-declared acceptance; nothing written"
    for v in MEMBERS:
        np.save(f"{OUT}/nn_{v}_tta{a.arm}_trainonly_val.npy", member_pred(v, a.arm).astype(np.float32))
    print("written", OUT)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--members", required=True)
    r.add_argument("--arms", default="det,T3,T1,T2")
    sp.add_parser("check")
    sp.add_parser("eval")
    w = sp.add_parser("write")
    w.add_argument("arm")
    a = p.parse_args()
    dict(run=cmd_run, check=cmd_check, eval=cmd_eval, write=cmd_write)[a.cmd](a)
