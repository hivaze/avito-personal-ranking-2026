"""Derive the wave-4 hcap caps (P5, review_integrity INTEGRITY-1) from data. CPU only, read-only (data/flat mmaps).

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/r6/hcap_caps.py
      -> nn_x/r6/logs/hcap_caps.txt (+ .json)

Per-feed history event counts by event type (1-10), val + test (all feeds) vs train (200K random feeds, seed 0).
Rule (written before the arm ran):
  (1) a type is CAPPED iff the train p99.9 exceeds max(val p99.9, test p99.9) by more than 10%;
  (2) cap = max(val p99.9, test p99.9) when val/test have a hard ceiling there (max - p99.9 <= 6);
      otherwise (val/test have a plateau-then-cliff count histogram with a thin long tail, i.e. type 1) cap = the end
      of the plateau = the first count c >= the val/test p95 whose histogram value falls below half of the mean over
      the 10 counts before it.
Expected (INTEGRITY-1): type 1 ~110, type 4 ~117, types 6/8/10 ~41-44; types 2/3/5/7/9 uncapped.
"""
import json
import os

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = "<repo>"
OUT = f"{ROOT}/nn_x/r6/logs/hcap_caps"


def counts(sp, feeds=None):
    f = pl.read_parquet(f"{ROOT}/data/flat/{sp}/feeds.parquet", columns=["H", "hist_off"])
    H = f["H"].to_numpy().astype(np.int64)
    off = f["hist_off"].to_numpy().astype(np.int64)
    et = np.load(f"{ROOT}/data/flat/{sp}/hist_event_type.npy", mmap_mode="r")
    feeds = np.arange(len(H)) if feeds is None else feeds
    rows = np.concatenate([np.arange(o, o + h) for o, h in zip(off[feeds], H[feeds])])
    fid = np.repeat(np.arange(len(feeds)), H[feeds])
    e = np.asarray(et[rows]).astype(np.int64)
    return np.bincount(fid * 11 + e, minlength=len(feeds) * 11).reshape(len(feeds), 11)


def main():
    n_tr = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["H"]).height
    Ctr = counts("train", np.sort(np.random.default_rng(0).choice(n_tr, 200000, replace=False)))
    Cva, Cte = counts("val"), counts("test")
    Cvt = np.concatenate([Cva, Cte])
    lines, caps, res = [], {}, {}
    q = lambda C, t, p: float(np.quantile(C[:, t], p))  # noqa: E731
    lines.append("type | p99.9 train / val / test | max val / test | capped? | cap | rule | val+test feeds > cap | "
                 "train feeds > cap | train events cut")
    for t in range(1, 11):
        p_tr, p_va, p_te = q(Ctr, t, .999), q(Cva, t, .999), q(Cte, t, .999)
        pvt = max(p_va, p_te)
        mx = int(max(Cva[:, t].max(), Cte[:, t].max()))
        capped = p_tr > 1.1 * pvt
        cap, rule = None, "-"
        if capped:
            if mx - pvt <= 6:
                cap, rule = int(round(pvt)), "hard ceiling: val/test p99.9"
            else:
                h = np.bincount(Cvt[:, t], minlength=Cvt[:, t].max() + 2).astype(float)
                c = int(np.quantile(Cvt[:, t], 0.95))
                while not (h[c] < 0.5 * h[c - 10:c].mean()):
                    c += 1
                cap, rule = c, f"plateau end (h[{c}]={int(h[c])} < 0.5 x mean h[{c - 10}..{c - 1}]={h[c - 10:c].mean():.0f})"
            caps[t] = cap
        r = dict(p999_train=p_tr, p999_val=p_va, p999_test=p_te, max_valtest=mx, capped=bool(capped), cap=cap, rule=rule)
        if cap:
            r.update(vt_over=float((Cvt[:, t] > cap).mean()), tr_over=float((Ctr[:, t] > cap).mean()),
                     tr_cut=float(np.maximum(Ctr[:, t] - cap, 0).sum() / Ctr[:, t].sum()))
        res[t] = r
        lines.append(f"{t:>4} | {p_tr:6.0f} / {p_va:4.0f} / {p_te:4.0f} | {mx:4d} | {'yes' if capped else 'no ':3s} | "
                     f"{cap if cap else '-':>4} | {rule} | " + (f"{r['vt_over']:.4f} | {r['tr_over']:.4f} | {r['tr_cut']:.4f}"
                                                              if cap else "- | - | -"))
    spec = ",".join(f"{t}:{c}" for t, c in sorted(caps.items()))
    tot_cut = sum(np.maximum(Ctr[:, t] - c, 0).sum() for t, c in caps.items()) / Ctr.sum()
    lines.append(f"\n--hcap \"{spec}\"   (train events dropped: {tot_cut:.4f} of all train history events; "
                 f"train feeds changed: {float(np.any([Ctr[:, t] > c for t, c in caps.items()], 0).mean()):.4f})")
    txt = "\n".join(lines)
    print(txt)
    open(OUT + ".txt", "w").write(txt + "\n")
    json.dump(dict(caps=caps, spec=spec, per_type=res), open(OUT + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
