"""C15 PLE / PCGrad pre-gate (PREDECLARED_ccb.txt C15), CPU, scratch copies of nn_x/w5/{model,data}.py (unchanged).
Checkpoints w5_e10_base_s951 / s952 at @70000 (plateau; @60000 not on disk) and @last. 64 training batches drawn with the
recipe's loader (SplitData.batches over the training feeds = all train minus the 30K holdout, cand_budget 32768, max_feeds
512, shuffle with np.random.default_rng(0)); per batch one train-mode forward (torch seed 0 + batch index) and the gradients
of the MAIN listwise loss (loss_norm feed, denom max_feeds, no feed weight) and of the K9 aux loss (listwise CE toward
softmax(-pos/tau), feed weight 1 - p_random, aux_norm batch) taken separately; cosine per shared 2-D trunk matrix (every 2-D
parameter that receives a non-zero gradient from BOTH losses). PROCEED iff median cos < 0 on >= 1/3 of the matrices at @70000
for both seeds. Information only (batch verdict: p = 1, CLOSED)."""
import os
import sys
sys.dont_write_bytecode = True
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch.set_num_threads(12)
from data import SplitData  # noqa: E402  (scratch copy)
from model import Ranker, listwise_loss  # noqa: E402  (scratch copy)

ROOT = Path("<repo>")
CK = ROOT / "nn_x/w5/ckpt"
T0 = time.time()
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 64


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


ck0 = torch.load(CK / "w5_e10_base_s951@70000.pt", map_location="cpu", weights_only=False)
A = ck0["args"]
xt_names = ck0["cfg"].get("xt_names")
log(f"args: loss_norm {A['loss_norm']} max_feeds {A['max_feeds']} cand_budget {A['cand_budget']} aux {A['aux_mode']} tau {A['aux_tau']} "
    f"aux_norm {A['aux_norm']} prand_base {A['prand_base']} domain {A['domain']} xt {A['xt']} ({len(xt_names)} cols)")
tr = SplitData("train", torch.device("cpu"), max_hist=A["max_hist"])
tr.attach_xt(xt_names)
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
assert (pr["fidx"] == np.arange(tr.n)).all()
prand_np = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
prand_np[tr.M_np < 50] = 0.0
prand = torch.from_numpy(prand_np)
tr.dom = torch.stack([1 - prand, prand, torch.zeros_like(prand)], 1)
cand = np.where((prand_np > 0.9) & (tr.M_np >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, min(A["holdout"], len(cand)), replace=False))
feeds = np.setdiff1d(np.arange(tr.n), hold)
batches = tr.batches(feeds, A["cand_budget"], A["max_feeds"], shuffle=True, rng=np.random.default_rng(0))[:NB]
log(f"data loaded; {len(feeds)} training feeds; {len(batches)} batches (feeds per batch {[len(x) for x in batches[:8]]}...)")


def run(ckname):
    ck = torch.load(CK / f"{ckname}.pt", map_location="cpu", weights_only=False)
    model = Ranker(**ck["cfg"])
    model.load_state_dict(ck["state_dict"])
    model.train()
    names = [n for n, p in model.named_parameters() if p.ndim == 2 and p.requires_grad]
    params = [dict(model.named_parameters())[n] for n in names]
    cos = {n: [] for n in names}
    nm_main, nm_aux = {n: [] for n in names}, {n: [] for n in names}
    for i, fb in enumerate(batches):
        torch.manual_seed(i)
        b = tr.gather(fb, with_pos=True)
        X = model.dense(b)
        lg = model(b, X, use_pos=True)
        y, m = b["y"], b["cmask"]
        main = listwise_loss(lg, y, m, None, norm=A["loss_norm"], denom=A["max_feeds"])
        tgt = torch.softmax((-b["pos"].float() / A["aux_tau"]).masked_fill(~m, -1e4), 1) * m
        lp = torch.log_softmax(model.aux_out.masked_fill(~m, -1e4), 1)
        fw_aux = 1 - prand[b["fidx"]]
        aux = (-(tgt * lp).sum(1) * fw_aux).sum() / fw_aux.sum().clamp_min(1e-6)
        gm = torch.autograd.grad(main, params, retain_graph=True, allow_unused=True)
        ga = torch.autograd.grad(aux, params, allow_unused=True)
        for n, a_, c_ in zip(names, gm, ga):
            if a_ is None or c_ is None:
                continue
            na, nc = float(a_.norm()), float(c_.norm())
            nm_main[n].append(na)
            nm_aux[n].append(nc)
            if na > 0 and nc > 0:
                cos[n].append(float((a_ * c_).sum() / (na * nc)))
        if i % 16 == 0:
            log(f"  {ckname} batch {i}: main {float(main):.4f} aux {float(aux):.4f}")
    shared = [n for n in names if len(cos[n]) == len(batches)]
    med = {n: float(np.median(cos[n])) for n in shared}
    neg = [n for n in shared if med[n] < 0]
    res = dict(ckpt=ckname, n_2d=len(names), n_shared=len(shared), n_neg_median=len(neg), share_neg=len(neg) / max(len(shared), 1),
               median_of_medians=float(np.median(list(med.values()))) if med else None,
               per_matrix={n: dict(median_cos=med[n], q10=float(np.quantile(cos[n], 0.1)), q90=float(np.quantile(cos[n], 0.9)),
                                   main_norm=float(np.median(nm_main[n])), aux_norm=float(np.median(nm_aux[n]))) for n in shared},
               not_shared=[n for n in names if n not in shared])
    log(f"{ckname}: 2-D params {len(names)}, shared {len(shared)}, median cos < 0 on {len(neg)} ({res['share_neg']:.3f}); "
        f"median of medians {res['median_of_medians']:.4f}")
    for n in shared:
        log(f"    {n:50s} med {med[n]:+.4f} q10 {res['per_matrix'][n]['q10']:+.4f} q90 {res['per_matrix'][n]['q90']:+.4f}")
    return res


R = {}
for c in ("w5_e10_base_s951@70000", "w5_e10_base_s952@70000", "w5_e10_base_s951@last", "w5_e10_base_s952@last"):
    R[c] = run(c)
    json.dump(R, open(HERE / "c15_gradcos.json", "w"), indent=1)
proceed = all(R[c]["share_neg"] >= 1 / 3 for c in ("w5_e10_base_s951@70000", "w5_e10_base_s952@70000"))
log(f"PROCEED flag (median cos < 0 on >= 1/3 of shared matrices at @70000, both seeds): {proceed}  "
    + " | ".join(f"{c} {R[c]['n_neg_median']}/{R[c]['n_shared']}" for c in R))
R["proceed"] = proceed
json.dump(R, open(HERE / "c15_gradcos.json", "w"), indent=1)
