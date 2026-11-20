"""WAVE-10 FPCD-lite PRE-GATE (reports/DATA_PIPELINE_1113.md 5.1, skeptic fix 4): inference only, ONE GPU. On the m02 roots the
arm would re-cool (nn_x/r6/ckpt/r6_cd_m02_r101-104.pt, read-only), re-predict val with the pre2 block in the Sep 6-8 TRAIN
format (the 6 pre2 xt columns pre2_fe_z / pre2_new_z / pre2_fe_s / pre2_new_s / pre2_hmax32 / has:pre2_fe set to 0 = exactly
what attach_xt produces for the all-NaN Sep 6-8 rows; the 3 within-feed rel channels of pre2_fe_s / pre2_new_s / pre2_hmax32
follow inside the model) and compare with the normal preds: within-feed Pearson (mean over feeds and pooled), within-feed
Spearman (mean), top-1 agreement, NDCG delta (+- SE). Inference exactly as train.py's val predict (common.predict, bf16 autocast,
tf32); the normal preds are checked against the stored nn_x/r6/preds/nn_r6_cd_m02_r<r>_trainonly_val.npy.
Gate (pre-declared by the skeptic, DATA_PIPELINE 5.1): corr >= 0.99 -> CLOSE; corr < 0.98 -> run; 0.98-0.99 grey zone -> run as
an information arm on idle GPUs (lead's standing instruction).

  cd <repo>/nn_x/w10/fpcd && CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=12 ../../../.venv/bin/python pregate_pre2blank.py
      -> logs/pregate_pre2blank.txt (+ preds/pregate/nn_r6_cd_m02_r<r>_val_{asis,blank}.npy)
"""
import sys
import time

import numpy as np
import torch

from common import ROOT, W5, predict
from data import SplitData
from model import Ranker

sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed  # noqa: E402

OUT = W5 / "logs" / "pregate_pre2blank.txt"
PD = W5 / "preds" / "pregate"
PD.mkdir(parents=True, exist_ok=True)
ROOTS = (101, 102, 103, 104)
lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)
    open(OUT, "w").write("\n".join(lines) + "\n")


def within_corr(a, b, fid, M):
    a, b = a.astype(np.float64), b.astype(np.float64)
    nf = len(M)
    ma, mb = np.bincount(fid, a, nf) / M, np.bincount(fid, b, nf) / M
    da, db = a - ma[fid], b - mb[fid]
    num = np.bincount(fid, da * db, nf)
    den = np.sqrt(np.bincount(fid, da * da, nf) * np.bincount(fid, db * db, nf))
    ok = den > 0
    return float(np.mean(num[ok] / den[ok])), float(num[ok].sum() / den[ok].sum()), int((~ok).sum())


def frank(x, fid, off):
    """within-feed rank of every candidate (0 = lowest), vectorised (ties broken by row order: float logits, ~no ties)."""
    o = np.lexsort((x, fid))
    r = np.empty(len(x), np.float64)
    r[o] = np.arange(len(x)) - off[fid[o]]
    return r


def top1(x, fid, off, M):
    o = np.lexsort((-x, fid))  # within feed, descending score
    return o[off] - off  # position of the max within each feed


def main():
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True  # as train.py (its final val predict wrote the stored preds)
    va = SplitData("val", dev)
    y = va.target_np.astype(np.float64)
    off, M = va.cand_off_np, va.M_np
    fid = np.repeat(np.arange(va.n), M)
    P(f"{time.strftime('%FT%TZ', time.gmtime())} FPCD-lite pre-gate: pre2-blank (Sep 6-8 train format) vs normal val preds of "
      f"r6_cd_m02_r{ROOTS[0]}-{ROOTS[-1]} (nn_x/r6/ckpt, read-only); GPU inference as train.py's val predict (bf16 autocast)")
    xt_ref, keep = None, None
    res = {}
    for r in ROOTS:
        name = f"r6_cd_m02_r{r}"
        t0 = time.time()
        ck = torch.load(ROOT / "nn_x/r6/ckpt" / f"{name}.pt", map_location=dev, weights_only=False)
        m = Ranker(**ck["cfg"]).to(dev)
        m.load_state_dict(ck["state_dict"])
        xt = ck["cfg"]["xt_names"]
        if xt_ref is None:
            xt_ref = xt
            va.attach_xt(xt)
            p2 = [i for i, n in enumerate(xt) if "pre2" in n]
            keep = va.xt[:, p2].clone()
            P(f"xt ({len(xt)}): pre2 block columns {[xt[i] for i in p2]} (indices {p2}); n_dense {ck['cfg']['n_dense']}; "
              f"xt_rel {ck['cfg'].get('xt_rel')} xt_relonly {ck['cfg'].get('xt_relonly')}")
        assert xt == xt_ref, f"{name}: xt layout differs"
        va.xt[:, p2] = keep
        pa = predict(m, va)
        ref = ROOT / "nn_x/r6/preds" / f"nn_{name}_trainonly_val.npy"
        chk = ""
        if ref.exists():
            rr = np.load(ref)
            chk = f"; normal-mode check vs stored preds: corr {np.corrcoef(rr, pa)[0, 1]:.7f} max|d| {np.abs(rr - pa).max():.2e}"
        va.xt[:, p2] = 0
        pb = predict(m, va)
        va.xt[:, p2] = keep
        np.save(PD / f"nn_{name}_val_asis.npy", pa.astype(np.float32))
        np.save(PD / f"nn_{name}_val_blank.npy", pb.astype(np.float32))
        na, nb = ndcg_per_feed(pa, y, off, M), ndcg_per_feed(pb, y, off, M)
        d = nb - na
        pm, pp, nconst = within_corr(pa, pb, fid, M)
        sm, _, _ = within_corr(frank(pa, fid, off), frank(pb, fid, off), fid, M)
        t1 = float(np.mean(top1(pa, fid, off, M) == top1(pb, fid, off, M)))
        res[r] = dict(pearson_mean=pm, pearson_pooled=pp, spearman_mean=sm, top1=t1, ndcg_asis=float(na.mean()),
                      ndcg_blank=float(nb.mean()), delta=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))))
        P(f"{name}: val normal {na.mean():.5f} blank {nb.mean():.5f} delta {d.mean():+.5f} +- {d.std(ddof=1) / np.sqrt(len(d)):.5f} | "
          f"within-feed Pearson mean {pm:.4f} pooled {pp:.4f} | Spearman mean {sm:.4f} | top-1 agree {t1:.3f} | "
          f"({time.time() - t0:.0f}s{chk})")
        del m
        torch.cuda.empty_cache()
    pm = np.array([res[r]["pearson_mean"] for r in ROOTS])
    sm = np.array([res[r]["spearman_mean"] for r in ROOTS])
    verdict = ("CLOSE (corr >= 0.99)" if pm.min() >= 0.99 else "RUN (corr < 0.98)" if pm.max() < 0.98
               else "GREY ZONE 0.98-0.99 -> run as a 4-root INFORMATION arm on the idle GPUs (lead's standing instruction)")
    P(f"SUMMARY: within-feed Pearson mean {pm.min():.4f}-{pm.max():.4f} (mean {pm.mean():.4f}); Spearman mean {sm.min():.4f}-{sm.max():.4f}; "
      f"top-1 {min(res[r]['top1'] for r in ROOTS):.3f}-{max(res[r]['top1'] for r in ROOTS):.3f}; NDCG delta "
      f"{min(res[r]['delta'] for r in ROOTS):+.5f}..{max(res[r]['delta'] for r in ROOTS):+.5f} -> gate: {verdict}")
    P(f"{time.strftime('%FT%TZ', time.gmtime())} done")


if __name__ == "__main__":
    main()
