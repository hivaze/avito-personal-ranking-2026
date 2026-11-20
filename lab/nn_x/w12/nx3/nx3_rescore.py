"""nx3 STAGE A re-scoring on the GPU (ledger nn_x/w5/logs/E_kd_predeclared.txt, block "TRAINED-INPUT TEST nx3", stale-val clause and amendment A3 VALUE).
WRITE-NEW copy of the mechanism of nn_x/abl/stale_pred.py (val inputs swapped through a patched data.feat_path) and of nn_x/abl/holdout_eval.py
(holdout rows = rows of the train split, inputs attached by the checkpoint cfg). Read-only on every checkpoint, feature and prediction file of the
project; writes NEW files under <outdir> only (existing files there are skipped, never overwritten).

  cd <repo>/nn_x/abl && CUDA_VISIBLE_DEVICES=<g> OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 \
     POLARS_MAX_THREADS=12 PYTHONDONTWRITEBYTECODE=1 ../../.venv/bin/python ../w12/nx3/nx3_rescore.py <mode> <outdir> <variant> [<variant> ...]
  variant = a checkpoint name of nn_x/w5/ckpt (w5_*) or nn_x/abl/ckpt (abl_*), optionally with @last.
  The modules data / common / model are those of nn_x/abl (model.py is byte-identical to nn_x/w5/model.py; data.py / common.py differ in paths only),
  so one process scores members of both harnesses; every member's natural re-score is compared with its stored prediction (printed, logged).

  mode svl   STALE-VAL, project scenario SVL = 'both_pre2lt3' (docs/05_METHODOLOGY.md 8.3): the override table of nn_x/w5/stale_pred.py with the six
             entries of nn_x/w12/stale/hz2n_review/stale_pred_lt3.py, PLUS nx_rp3_z / nx_rp3_rk -> their stale-val twins nn_x/w12/nx3/staleval/
             (used only by checkpoints whose cfg lists them; the twin has no NaN, so has:nx_rp3_z stays 1). Val only, TF32 on as train.py's val predict.
             -> <outdir>/svl/nn_<v>_staleval.npy
  mode perm  VALUE clause: the two NX value columns permuted within the feed (files of nx3_make_perm.py: one permutation per feed for both columns,
             NaN pattern unchanged so has:nx_rp3_z is unchanged = 1 on every val row and on every lag-view holdout row).
             val (TF32 on) and the 30,000 holdout feeds (TF32 off, domain mixture as holdout_eval.py), each NATURAL and PERMUTED by the same code path
             -> <outdir>/perm/nn_<v>_{val,hold}_{nat,perm}.npy     (hold: scores of the holdout candidates, holdout feeds in sorted order)
  Every line is appended to <outdir>/rescore.log. GPU guard: the caller (read_stage_a.sh) checks the GPU under the lane lock; this script only
  refuses to start without exactly one CUDA id (nn_x/abl/common.py).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "<repo>/nn_x/abl")
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import torch  # noqa: E402

import data as Dm  # noqa: E402
from common import ROOT, eval_ndcg, ndcg, predict  # noqa: E402
from model import Ranker  # noqa: E402

NX3 = ROOT / "nn_x/w12/nx3"
NXV = ("nx_rp3_z", "nx_rp3_rk")
# ---- SVL override table: nn_x/w5/stale_pred.py OVR (durable counterfactual copy) + stale_pred_lt3.py update + the NX twins
CF_BOTH = ROOT / "nn_x/r6/stale_cf"
CF = ROOT / "solvers/out/stale/cf"
LT3 = ROOT / "nn_x/w12/stale/hz2n_review/cf_lt3"
OVR_SVL = {**{n: CF_BOTH / f"both_{n}.npy" for n in ("it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo",
                                                    "it_hist_et4_luo", "seq_cv_invd20", "seq_cv_norm", "seq_cv_pop")},
           **{n: CF / f"pre2both_{n}.npy" for n in ("pre2_fe", "pre2_fe_z", "pre2_fe_s")}}
OVR_SVL.update({n: LT3 / f"lt3_{n}.npy" for n in ("pre2_fe", "pre2_fe_z", "pre2_fe_s", "pre2_new", "pre2_new_z", "pre2_new_s")})
OVR_TWIN = {n: NX3 / "staleval" / f"{n}.npy" for n in NXV}
OVR_PERM = {sp: {n: NX3 / "perm" / sp / f"{n}.npy" for n in NXV} for sp in ("val", "train")}

_feat_path = Dm.feat_path
ACTIVE = {"ovr": None}  # None, or {split: {name: path}}


def feat_path(split, name):
    o = ACTIVE["ovr"]
    if o is not None and name in o.get(split, {}):
        return o[split][name]
    return _feat_path(split, name)


Dm.feat_path = feat_path  # SplitData.attach_xt resolves feature files through this module-level name (as nn_x/abl/stale_pred.py)


def harness(v):
    if v.startswith("w5_"):
        return ROOT / "nn_x/w5"
    assert v.startswith("abl_"), f"{v}: only w5_* and abl_* members"
    return ROOT / "nn_x/abl"


def load_model(v, dev):
    ck = torch.load(harness(v) / "ckpt" / f"{v}.pt", map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    assert not cfg.get("id_buckets", 0), "item-ID members are not handled here"
    m = Ranker(**cfg).to(dev)
    m.load_state_dict(ck["state_dict"])
    return m, cfg, (cfg.get("xt_names") or [])


def cmp(a, b):
    return f"corr {np.corrcoef(a, b)[0, 1]:.7f} max|d| {np.abs(a - b).max():.2e}"


def save_new(path, arr):
    """Exclusive create: an existing file is kept as it is."""
    if not path.exists():
        with open(path, "xb") as fh:
            np.save(fh, arr)


def main():
    mode, out, variants = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    assert mode in ("svl", "perm") and variants
    O = out / mode
    O.mkdir(parents=True, exist_ok=True)
    logf = open(out / "rescore.log", "a")

    def P(*a):
        s = f"{time.strftime('%FT%TZ', time.gmtime())} [{mode}] " + " ".join(str(x) for x in a)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    for p in list(OVR_SVL.values()) + list(OVR_TWIN.values()) + [q for d in OVR_PERM.values() for q in d.values()]:
        assert p.exists(), f"missing override file {p}"
    dev = torch.device("cuda:0")
    ovr_val = {"val": {**OVR_SVL, **OVR_TWIN}} if mode == "svl" else {"val": OVR_PERM["val"]}
    files = (lambda v: [O / f"nn_{v}_staleval.npy"]) if mode == "svl" else \
        (lambda v: [O / f"nn_{v}_{sp}_{k}.npy" for sp in ("val", "hold") for k in ("nat", "perm")])
    todo = [v for v in variants if not all(f.exists() for f in files(v))]
    P(f"{len(variants)} variants, {len(todo)} to do")
    if not todo:
        return
    # ------------------------------------------------------------------ val (TF32 on, as train.py's final val predict)
    torch.backends.cuda.matmul.allow_tf32 = True
    va = Dm.SplitData("val", dev)
    for v in todo:
        t0 = time.time()
        m, cfg, xt = load_model(v, dev)
        used = sorted(n for n in ovr_val["val"] if n in xt or f"has:{n}" in xt)
        if mode == "perm":
            assert all(n in xt for n in NXV) and "has:nx_rp3_z" in xt, f"{v}: the VALUE clause needs a member that reads {NXV} and has:nx_rp3_z"
        ACTIVE["ovr"] = None
        va.xt = None
        va.attach_xt(xt)
        pn = predict(m, va)
        ref = harness(v) / "preds" / f"nn_{v}_trainonly_val.npy"
        msg = f"natural vs stored val preds: {cmp(np.load(ref), pn)}" if ref.exists() else "no stored val preds"
        ACTIVE["ovr"] = ovr_val
        va.xt = None
        va.attach_xt(xt)
        if "has:nx_rp3_z" in xt:  # 'has kept at 1': the flag column must be 1 on every val row under the override
            assert float(va.xt[:, xt.index("has:nx_rp3_z")].float().min()) == 1.0
        po = predict(m, va)
        ACTIVE["ovr"] = None
        if mode == "svl":
            save_new(O / f"nn_{v}_staleval.npy", po.astype(np.float32))
            P(f"{v}: natural val ndcg {eval_ndcg(pn, va):.5f}, stale-val (SVL{' + NX twins' if any(n in used for n in NXV) else ''}) ndcg "
              f"{eval_ndcg(po, va):.5f}; overridden inputs used {used}; {msg}; {time.time() - t0:.0f}s")
        else:
            save_new(O / f"nn_{v}_val_nat.npy", pn.astype(np.float32))
            save_new(O / f"nn_{v}_val_perm.npy", po.astype(np.float32))
            P(f"{v}: val natural ndcg {eval_ndcg(pn, va):.5f}, permuted ndcg {eval_ndcg(po, va):.5f}; overridden inputs used {used}; {msg}; "
              f"{time.time() - t0:.0f}s")
        del m
        torch.cuda.empty_cache()
    if mode == "svl":
        P(f"peak GPU memory of this process: allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB")
        return
    # ------------------------------------------------------------------ holdout rows (copy of holdout_eval.py; TF32 off as cd_arms5.holdout_preds)
    del va
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    tr = Dm.SplitData("train", dev)
    pr = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
    pr[tr.M_np < 50] = 0.0
    cand = np.where((pr > 0.9) & (tr.M_np >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    M, off = tr.M_np[hold], tr.cand_off_np[hold]
    rows = np.concatenate([np.arange(o, o + m_) for o, m_ in zip(off, M)])
    offs = np.concatenate([[0], np.cumsum(M)[:-1]])
    y = tr.target_np[rows]
    meta = np.load(ROOT / "nn_x/w5/preds/holdout/holdout_meta.npy")
    assert (meta == np.stack([M, offs])).all(), "holdout meta mismatch with nn_x/w5/preds/holdout/holdout_meta.npy"
    prt = torch.from_numpy(pr).to(dev)
    res = {}
    for kind, ovr in (("nat", None), ("perm", {"train": OVR_PERM["train"]})):
        cur = None
        for v in todo:
            t0 = time.time()
            m, cfg, xt = load_model(v, dev)
            if xt != cur:  # reuse the loaded inputs when unchanged (as holdout_eval.py)
                ACTIVE["ovr"] = ovr
                tr.xt = None
                torch.cuda.empty_cache()
                tr.attach_xt(xt)
                ACTIVE["ovr"] = None
                cur = xt
                hcol = tr.xt[torch.from_numpy(rows).to(dev), xt.index("has:nx_rp3_z")].float().cpu().numpy()
                P(f"holdout inputs attached ({kind}; {len(xt)} columns); has:nx_rp3_z = 1 on {int(hcol.sum()):,} of {len(rows):,} holdout rows")
                res[kind, "has"] = hcol
            tr.dom = torch.stack([1 - prt, prt, torch.zeros_like(prt)], 1) if cfg.get("domain") else None
            s = predict(m, tr, hold)[rows].astype(np.float32)
            save_new(O / f"nn_{v}_hold_{kind}.npy", s)
            msg = ""
            if kind == "nat":
                ref = harness(v) / "preds" / "holdout" / f"nn_{v}_holdout.npy"
                msg = f"; natural vs stored holdout preds: {cmp(np.load(ref), s)}" if ref.exists() else "; no stored holdout preds"
            P(f"{v}: holdout {kind} ndcg {ndcg(s, y, offs, M):.5f} (30,000 feeds){msg}; {time.time() - t0:.0f}s")
            del m
            torch.cuda.empty_cache()
    assert (res["nat", "has"] == res["perm", "has"]).all(), "the permutation changed the has flag on holdout rows"
    P("has:nx_rp3_z identical under natural and permuted inputs on all holdout rows")
    P(f"peak GPU memory of this process: allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
