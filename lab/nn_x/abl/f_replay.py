"""f1..f5 REPLAY on existing nn_x/abl checkpoints (reports/ABLATION_PLAN_1113.md section 3.3 'always' row; pre-declared in
nn_x/abl/ledger/ledger.md section 6): what a trained member has learned from the organisers' production scores f1..f5
(15 dense inputs = raw, within-feed z and mid-rank pct of each f_k, computed in model.dense_features from b['cf'], and the
same 15 columns in the linear wide term). NO training. REPLAY-ONLY: this script patches data.SplitData.gather at run time
and is imported by nothing on the training path (train.py never imports it; tests/test_abl.py checks that only train.py
and this file import abl_retr), so the harness's bit-identity proofs are untouched.

  GPU step (one GPU id; ~1 min per mode):
    cd <repo>/nn_x/abl && CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=12 ../../.venv/bin/python f_replay.py predict \
        abl_base_s23@last abl_base_s24@last
      -> preds/replay/nn_<v>~<mode>_trainonly_val.npy, preds/replay/holdout/nn_<v>~<mode>_holdout.npy, logs/f_replay_predict.log
  CPU read-out (paired per feed, r6_eval slices):
    cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/abl/f_replay.py readout \
        abl_base_s23@last abl_base_s24@last          -> nn_x/abl/logs/f_replay.txt (+ .json)

Modes (applied to b['cf'] [B,M,5] right after SplitData.gather, so raw, z, pct and the wide term all follow):
  ref        unchanged: must reproduce the stored train-only val preds (TF32 on, as train.py's val predict) and the stored
             holdout preds (TF32 off, as cd_arms5.holdout_preds); the max |difference| is printed and logged
  zero_all   f1..f5 := 0 for every candidate (raw becomes a constant, z = 0, pct = 0.5 for every candidate: the inputs
             carry no information; the model sees a constant it never saw in training -> an upper bound on the reliance)
  perm_all   the f1..f5 rows are permuted WITHIN FEED among the valid candidates (one permutation per feed, fixed seed):
             the per-feed marginal distribution is kept, the candidate <-> score association is broken (the standard
             permutation-importance read)
  zero_f<k>, perm_f<k>   the same for one column k in 1..5 (the other four columns unchanged)
Read-out: delta = mode - ref, paired per feed on the SAME checkpoint, pooled over the given checkpoints; val slices all /
  M<50 / M>=50 / Tue / Thu / coldpos / warmpos, holdout all / late20%; SE = sd / sqrt(n_feeds) (same checkpoint: no
  run-to-run term applies). RELIANCE READ ONLY (the plan: 'replay != retrain'): no gate, no recipe action.
"""
import json
import os
import sys
import time

import numpy as np

ROOT = "<repo>"
ABL = f"{ROOT}/nn_x/abl"
MODES = ["ref", "zero_all", "perm_all"] + [f"{k}_f{i}" for i in range(1, 6) for k in ("zero", "perm")]
PERM_SEED = 2026


def mode_spec(mode):
    """(kind, cols) of a mode name: kind None (ref) | 'zero' | 'perm'; cols = list of f columns (0-based)."""
    if mode == "ref":
        return None, []
    kind, what = mode.split("_", 1)
    assert kind in ("zero", "perm") and (what == "all" or (what[0] == "f" and what[1:] in "12345")), mode
    return kind, list(range(5)) if what == "all" else [int(what[1:]) - 1]


def rep_name(v, mode):
    return f"{v}~{mode}"


# ------------------------------------------------------------------------------------------------ GPU step
def predict_modes(variants):
    import torch  # noqa: E402

    import data as Dm  # noqa: E402
    from common import CKPT, PREDS, eval_ndcg, predict  # noqa: E402  (GPU guard: one CUDA_VISIBLE_DEVICES id)
    from abl_retr import load_ranker  # noqa: E402  (plain Ranker(**cfg) for every checkpoint without cfg['retr_cat'])
    from cd_arms5 import holdout_preds  # noqa: E402  (exactly the stored holdout preds' code path: TF32 off)
    from metrics import ndcg  # noqa: E402  (src/metrics via common's sys.path insert)

    out_dir = PREDS / "replay"
    (out_dir / "holdout").mkdir(parents=True, exist_ok=True)
    log = open(f"{ABL}/logs/f_replay_predict.log", "a")

    def L(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    MODE = {"kind": None, "cols": None, "gen": None}
    _gather = Dm.SplitData.gather

    def gather(self, fidx_np, with_pos=False):  # replay hook: modifies ONLY b['cf'] (f1..f5) of the gathered batch
        b = _gather(self, fidx_np, with_pos)
        if MODE["kind"] is not None:
            cf = b["cf"].clone()
            cols = MODE["cols"]
            if MODE["kind"] == "zero":
                cf[..., cols] = 0.0
            else:  # within-feed permutation among the valid candidates; padded rows keep padded values
                cm = b["cmask"]
                r = torch.rand(cm.shape, generator=MODE["gen"], device=cm.device).masked_fill(~cm, 2.0)
                perm = r.argsort(1)
                src = b["cf"].gather(1, perm[..., None].expand(-1, -1, cf.shape[-1]))
                cf[..., cols] = src[..., cols]
            b["cf"] = cf
        return b

    Dm.SplitData.gather = gather
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    va = Dm.SplitData("val", dev)
    tr = Dm.SplitData("train", dev)
    pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
    assert (pr["fidx"] == np.arange(tr.n)).all()
    prand_np = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    prand_np[tr.M_np < 50] = 0.0
    prand = torch.from_numpy(prand_np).to(dev)
    cand = np.where((prand_np > 0.9) & (tr.M_np >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    hold_rows = np.concatenate([np.arange(o, o + m) for o, m in zip(tr.cand_off_np[hold], tr.M_np[hold])])
    meta = np.load(PREDS / "holdout" / "holdout_meta.npy")
    assert (meta[0] == tr.M_np[hold]).all(), "holdout definition differs from the stored holdout_meta"
    L(f"{time.strftime('%FT%TZ', time.gmtime())} f_replay predict: data loaded in {time.time() - t0:.0f}s; modes {MODES}")
    xt_cur = None
    for v in variants:
        ck = torch.load(CKPT / f"{v}.pt", map_location=dev, weights_only=False)
        m = load_ranker(ck, dev)
        m.eval()
        xt = ck["cfg"].get("xt_names") or []
        if xt != xt_cur:
            for sd in (va, tr):
                sd.xt = None
                torch.cuda.empty_cache()
                sd.attach_xt(xt)
            xt_cur = xt
        tr.dom = torch.stack([1 - prand, prand, torch.zeros_like(prand)], 1) if ck["cfg"].get("domain") else None
        for k, mode in enumerate(MODES):
            fv = out_dir / f"nn_{rep_name(v, mode)}_trainonly_val.npy"
            fh = out_dir / "holdout" / f"nn_{rep_name(v, mode)}_holdout.npy"
            if fv.exists() and fh.exists():
                L(f"  {v} {mode}: exists")
                continue
            t1 = time.time()
            kind, cols = mode_spec(mode)
            MODE.update(kind=kind, cols=cols, gen=torch.Generator(device=dev).manual_seed(PERM_SEED + 101 * k))
            pv = predict(m, va)  # TF32 on, bf16 autocast: train.py's val predict
            nd_v = eval_ndcg(pv, va)
            sh = holdout_preds(predict, m, tr, hold, hold_rows)  # TF32 off: cd_arms5 / holdout_eval numerics
            nd_h = ndcg(sh, tr.target_np[hold_rows], meta[1], meta[0])
            msg = ""
            if mode == "ref":  # reproduction check against the stored preds of this checkpoint
                r_v = np.load(PREDS / f"nn_{v}_trainonly_val.npy")
                r_h = np.load(PREDS / "holdout" / f"nn_{v}_holdout.npy")
                msg = (f"; REPRODUCTION vs stored: val max|d| {np.abs(r_v - pv).max():.2e} (corr {np.corrcoef(r_v, pv)[0, 1]:.7f}), "
                       f"holdout max|d| {np.abs(r_h - sh).max():.2e}")
            np.save(fv, pv.astype(np.float32))
            np.save(fh, sh.astype(np.float32))
            L(f"  {time.strftime('%FT%TZ', time.gmtime())} {v} {mode:9s}: val {nd_v:.5f} holdout {nd_h:.5f} ({time.time() - t1:.0f}s){msg}")
        MODE.update(kind=None, cols=None, gen=None)
        del m
        torch.cuda.empty_cache()
    L(f"{time.strftime('%FT%TZ', time.gmtime())} f_replay predict done ({(time.time() - t0) / 60:.1f} min)")


# ------------------------------------------------------------------------------------------------ CPU read-out
def readout(variants):
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "readout: CPU only (CUDA_VISIBLE_DEVICES='')"
    sys.path.insert(0, f"{ROOT}/nn_x/r6")
    import r6_eval as R  # noqa: E402  (read-only paired per-feed machinery + slices)

    R.DIRS["abl"] = (f"{ABL}/preds", f"{ABL}/logs")  # the stored holdout_meta / holdout_y live here
    R.DIRS["ablrep"] = (f"{ABL}/preds/replay", f"{ABL}/logs")
    P = R.P
    v, h = R.setup("val"), R.setup("hold")
    P(f"== f1..f5 REPLAY read-out (replay-only hook on b['cf']; checkpoints {variants}; delta = mode - ref, paired per feed "
      f"on the same checkpoint, pooled over checkpoints; SE_feed only; reliance read, not a gate)")
    res = {"variants": variants, "modes": {}}
    refs = {s: f"ablrep:{rep_name(s, 'ref')}" for s in variants}
    for s in variants:
        pv, ph = R.per_feed(v, refs[s]), R.per_feed(h, refs[s])
        P(f"  ref {s}: val {pv.mean():.5f} holdout {ph.mean():.5f}")
        res[f"ref_{s}"] = dict(val=float(pv.mean()), hold=float(ph.mean()))
    P("  mode       val delta (z)          M<50      M>=50     Tue       Thu       coldpos   warmpos  | hold delta (z)        late20%")
    for mode in MODES[1:]:
        pairs = [(refs[s], f"ablrep:{rep_name(s, mode)}") for s in variants]
        if not all(os.path.exists(c["path"](n)) for c in (v, h) for p in pairs for n in p):
            P(f"  {mode}: incomplete")
            continue
        rv, rh = R.compare(v, pairs), R.compare(h, pairs)
        pv, ph = rv["pooled"], rh["pooled"]
        res["modes"][mode] = dict(val=pv, hold=ph, per_ckpt_val=[p["slices"]["all"]["delta"] for p in rv["pairs"]],
                                  per_ckpt_hold=[p["slices"]["all"]["delta"] for p in rh["pairs"]])
        P(f"  {mode:9s}  {pv['all']['delta']:+.5f}+-{pv['all']['se']:.5f} ({pv['all']['z']:+.1f})  {pv['M<50']['delta']:+.5f}  "
          f"{pv['M>=50']['delta']:+.5f}  {pv['Tue']['delta']:+.5f}  {pv['Thu']['delta']:+.5f}  {pv['coldpos']['delta']:+.5f}  "
          f"{pv['warmpos']['delta']:+.5f} | {ph['all']['delta']:+.5f}+-{ph['all']['se']:.5f} ({ph['all']['z']:+.1f})  "
          f"{ph['late20%']['delta']:+.5f}   per-ckpt val {' '.join(f'{x:+.5f}' for x in res['modes'][mode]['per_ckpt_val'])}")
    out = f"{ABL}/logs/f_replay.txt"
    open(out, "w").write("\n".join(R.OUT) + "\n")
    json.dump(res, open(out.replace(".txt", ".json"), "w"), indent=1, default=float)
    print(f"wrote {out}")


if __name__ == "__main__":
    mode, variants = sys.argv[1], sys.argv[2:]
    assert mode in ("predict", "readout") and variants, "usage: f_replay.py predict|readout <variant@last ...>"
    predict_modes(variants) if mode == "predict" else readout(variants)
