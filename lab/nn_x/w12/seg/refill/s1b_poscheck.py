"""Step 1b (label-free): decomposition of the positive check. Pretend the cap is 192 (stored history and every earlier
snapshot cut to their newest 192); recovered = first min(n, 64) events older than the stored 192nd; truth = stored 193-256.
Each recovered event is classed: in the true events 193-256 | stored among the newest 192 (a placement error) |
not stored anywhere in the 256 (a real earlier event the later snapshot does not hold: per-type caps)."""
import numpy as np
from align_lib import OUT, SPL, Flat, union_timeline

F = Flat()
for name, split in (("holdout", "train"), ("val", "val")):
    z = np.load(OUT / f"align_{name}.npz")
    q = z["fid"]
    tot = np.zeros(4)
    rec_truth = []
    et_in, et_out = np.zeros(12), np.zeros(12)
    for f in q:
        U, st = union_timeline(F, split, f, cap=192)
        if st["n_merged"] < 1 or st["deep_anchor"] < 184 or st["n_T"] < 1:
            continue
        iL = st["iL"]
        rec = U["key"][iL + 1:][:64]
        full, _ = F.seq(SPL.index(split), f)
        new, truth = full[:192], full[192:256]
        tl = truth.tolist()
        nl = new.tolist()
        c = [0, 0, 0]
        for k in rec.tolist():
            if k in tl:
                tl.remove(k); c[0] += 1; et_in[(k // 64) % 64] += 1
            elif k in nl:
                nl.remove(k); c[1] += 1
            else:
                c[2] += 1; et_out[(k // 64) % 64] += 1
        tot[:3] += c
        tot[3] += len(rec)
        if len(rec) == 64:
            rec_truth.append(c[0] / 64)
    print(f"{name}: recovered events {tot[3]:.0f}: in the true events 193-256 {100 * tot[0] / tot[3]:.1f}% | stored among the newest 192 (placement error) {100 * tot[1] / tot[3]:.1f}% | "
          f"not stored in the 256 at all {100 * tot[2] / tot[3]:.1f}% || feeds with 64 recovered: {len(rec_truth)}, share of the 64 true events recovered: mean {100 * np.mean(rec_truth):.1f}%")
    print(f"   event types 1..10 of recovered-and-true: {np.round(et_in[1:11] / et_in.sum(), 3)} | of recovered-not-stored: {np.round(et_out[1:11] / max(et_out.sum(), 1), 3)}")
