"""Synthesiser check = the P4 (FPCD-lite) pre-gate the skeptic proposed: on the m02 roots the arm would re-cool
(r6_cd_m02_r101..r104), how different is the trained model's 'pre2-blank' mode (Sep 6-8 input format: 6 pre2 xt
columns -> 0, has:pre2_fe = 0) from its normal mode on val? Within-feed Pearson / Spearman of the logits, top-1
agreement, NDCG delta. Skeptic's pre-declared gate (dpa skeptic verdict on P4): corr >= 0.99 -> near copy, CLOSE;
corr < 0.98 -> run the arm. Inference only, CPU, fp32. Re-uses the parity-checked CPU rebuild (imported read-only)."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time
sys.path.insert(0, "/tmp/scratchpad/dpa/skeptic_p4")
from s01_extract_copy import *  # noqa
sys.path.insert(0, "<repo>/nn_x/w5")
from model import Ranker  # noqa: E402
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(12)

sd, fe, f, raw, y = build("val")
yv = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
off = sd.cand_off_np.astype(np.int64); M = sd.M_np.astype(np.int64)
nf = len(M); fid = np.repeat(np.arange(nf), M)
P2 = [i for i, n in enumerate(XT) if "pre2" in n]
print("pre2 cols:", [XT[i] for i in P2], flush=True)
xt0 = sd.xt.clone()


def within_corr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    ma = np.bincount(fid, a, minlength=nf) / M; mb = np.bincount(fid, b, minlength=nf) / M
    da = a - ma[fid]; db = b - mb[fid]
    num = np.bincount(fid, da * db, minlength=nf)
    den = np.sqrt(np.bincount(fid, da * da, minlength=nf) * np.bincount(fid, db * db, minlength=nf))
    ok = den > 0
    return float(np.mean(num[ok] / den[ok])), float(num[ok].sum() / den[ok].sum())


def frank(x):
    r = np.empty(len(x))
    for o, m in zip(off, M):
        r[o:o + m] = np.argsort(np.argsort(x[o:o + m], kind="mergesort"), kind="mergesort")
    return r


def top1(x):
    return np.array([np.argmax(x[o:o + m]) for o, m in zip(off, M)])


for r in (101, 102, 103, 104):
    name = f"r6_cd_m02_r{r}"
    ck = torch.load(ROOT / f"nn_x/r6/ckpt/{name}.pt", map_location="cpu", weights_only=False)
    assert ck["cfg"]["xt_names"] == XT, "xt layout differs"
    m = Ranker(**ck["cfg"]); m.load_state_dict(ck["state_dict"]); m.eval()
    out = {}
    for mode in ("asis", "blank"):
        sd.xt = xt0.clone()
        if mode == "blank":
            sd.xt[:, P2] = 0
        o = np.zeros(sd.n_cand, np.float32); t0 = time.time()
        with torch.no_grad():
            for fb in sd.batches(np.arange(sd.n), 65536, 1024, shuffle=False):
                b = sd.gather(fb); lg = m(b, use_pos=False); mm = b["cmask"]
                o[b["ci"][mm].numpy()] = lg[mm].float().numpy()
        out[mode] = o
        print(f"  {name} {mode} done {time.time() - t0:.0f}s", flush=True)
    na = ndcg_per_feed(out["asis"], yv, off, M); nb_ = ndcg_per_feed(out["blank"], yv, off, M)
    d = nb_ - na
    pear_mean, pear_pool = within_corr(out["asis"], out["blank"])
    sp_mean, _ = within_corr(frank(out["asis"]), frank(out["blank"]))
    t1 = float(np.mean(top1(out["asis"]) == top1(out["blank"])))
    print(f"{name}: val asis {na.mean():.5f} blank {nb_.mean():.5f} delta {d.mean():+.5f} ({d.std(ddof=1) / np.sqrt(nf):.5f}) | "
          f"within-feed Pearson mean {pear_mean:.4f} pooled {pear_pool:.4f} | Spearman mean {sp_mean:.4f} | top-1 agree {t1:.3f}",
          flush=True)
