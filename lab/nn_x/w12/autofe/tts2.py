"""Representation check that a residual model cannot make: test-time standardisation (TTS).
The ranker standardises its 104 dense inputs with TRAIN feature_stats; several inputs sit 0.2-0.75 train-SD lower
on val / test. Here the checkpoint's f_mean / f_std are swapped for label-free statistics of the evaluated split
(val) and the member is re-scored on CPU (audit harness nn_x/audit/dpa_1113/pipeline/s01_extract.py, parity-checked).
Variants: full (mean+std), mean only, half-way. 4 train-only W5 members; paired per-feed NDCG."""
import os, sys, time, json
sys.path.insert(0, "<repo>/nn_x/audit/dpa_1113/pipeline")
from s01_extract import *  # noqa  (CPU only, 12 threads; read-only use of the audit harness)
from model import Ranker
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

MY = Path("<repo>/nn_x/w12/autofe")
sd, fe, f, raw, y = build("val")
yv = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
off = sd.cand_off_np; M = sd.M_np
batches = sd.batches(np.arange(sd.n), 65536, 1024, shuffle=False)


def forward(m):
    out = np.zeros(sd.n_cand, np.float32)
    with torch.no_grad():
        for fb in batches:
            b = sd.gather(fb); lg = m(b, use_pos=False); mm = b["cmask"]
            out[b["ci"][mm].numpy()] = lg[mm].float().numpy()
    return out


res = {}; pool = {}; stored = []
vstats = None
for s in (951, 952, 953, 954):
    name = f"w5_e10_kd05w192_s{s}"
    ck = torch.load(ROOT / f"nn_x/w5/ckpt/{name}.pt", map_location="cpu", weights_only=False)
    m = Ranker(**ck["cfg"]); m.load_state_dict(ck["state_dict"]); m.eval()
    assert ck["cfg"]["xt_names"] == XT
    if vstats is None:  # label-free statistics of the val split, same estimator as feature_stats (all candidates)
        s1 = s2 = 0; n = 0
        with torch.no_grad():
            for fb in batches:
                b = sd.gather(fb); X = m.dense(b)[b["cmask"]].double()
                s1 = s1 + X.sum(0); s2 = s2 + (X ** 2).sum(0); n += X.shape[0]
        vmean = (s1 / n).float(); vstd = (s2 / n - (s1 / n) ** 2).clamp_min(0).sqrt().float().clamp_min(1e-3)
        vstats = (vmean, vstd)
        sh = ((vmean - m.f_mean) / m.f_std).numpy(); rt = (vstd / m.f_std).numpy()
        print("val-vs-train stats: |shift| > 0.15 SD on", int((np.abs(sh) > 0.15).sum()), "of", len(sh), "inputs; max |shift|", float(np.abs(sh).max()),
              "; std ratio range", float(rt.min()), float(rt.max()), flush=True)
        np.save(MY / "cache/tts_shift.npy", np.stack([sh, rt]))
    tm, ts = m.f_mean.clone(), m.f_std.clone()
    t0 = time.time()
    base = forward(m)
    st = np.load(ROOT / f"preds/nn_{name}_trainonly_val.npy")
    nb_ = ndcg_per_feed(base, yv, off, M)
    print(f"{name}: parity corr {np.corrcoef(base, st)[0, 1]:.6f} cpu {nb_.mean():.5f} stored {ndcg_per_feed(st, yv, off, M).mean():.5f} {time.time() - t0:.0f}s", flush=True)
    pool.setdefault("base", []).append(feed_rank_z(off, M, base)[:, 1])
    stored.append(feed_rank_z(off, M, st)[:, 1])
    res.setdefault("fp32_minus_stored", {})[name] = float(nb_.mean() - ndcg_per_feed(st, yv, off, M).mean())
    np.save(MY / f"cache/val_cpu_{name}.npy", base)
    sh_ = (vstats[0] - tm) / ts; rt_ = vstats[1] / ts
    ok = (rt_ >= 0.6) & (rt_ <= 1.6)  # leaves out the pre2-blank flag (val std 0) and 3 columns whose scale differs > 1.6x
    big = ok & (sh_.abs() > 0.15)     # only the inputs that actually sit at a different level on val
    VAR = {"safe_full": (torch.where(ok, vstats[0], tm), torch.where(ok, vstats[1], ts)),
           "safe_mean": (torch.where(ok, vstats[0], tm), ts),
           "shifted_mean_half": (torch.where(big, 0.5 * (vstats[0] + tm), tm), ts)} if s == 951 else {}
    if s == 951:
        print("columns swapped: safe", int(ok.sum()), "of", len(ok), "; shifted", int(big.sum()), flush=True)
    for var, (mu, sg) in VAR.items():
        m.f_mean.copy_(mu); m.f_std.copy_(sg)
        p = forward(m)
        d = ndcg_per_feed(p, yv, off, M) - nb_
        res.setdefault(var, {})[name] = (float(d.mean()), float(d.std() / np.sqrt(len(d))))
        print(f"  TTS {var}: delta {d.mean():+.5f} ± {d.std() / np.sqrt(len(d)):.5f}  {time.time() - t0:.0f}s", flush=True)
    m.f_mean.copy_(tm); m.f_std.copy_(ts)
pb = ndcg_per_feed(np.mean(pool["base"], 0), yv, off, M)
ps = ndcg_per_feed(np.mean(stored, 0), yv, off, M)
d = pb - ps
res["fp32_minus_stored"]["pool4"] = (float(d.mean()), float(d.std() / np.sqrt(len(d))))
print(f"pool of 4 (mean rank-z): CPU fp32 {pb.mean():.5f} stored preds {ps.mean():.5f} delta {d.mean():+.5f} +- {d.std() / np.sqrt(len(d)):.5f}", flush=True)
json.dump(res, open(MY / "tts_result.json", "w"), indent=1)
