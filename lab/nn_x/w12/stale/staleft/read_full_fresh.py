"""staleft: FRESH-val read of a FULL STALE-FT run against the STORED standard fine-tune (n2cpv3s3), for the lead.
  CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/stale/staleft/read_full_fresh.py <tag, e.g. p10> [w16|w8|all24]
FT branch = member-equal z-mean; final = 0.5 z(FT) + 0.5 z(TO). The stale-val read needs a P = 0 control from finetune_stale.py
(gen_full_jobs.sh 0 p0 for the same members, then STALEFT_SUF=sft3 STALEFT_MEMBERS="..." pilot_read.py <tag>)."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "<repo>/nn_x/w12/stale/staleft")
import numpy as np
from pathlib import Path
import pilot_read as R
tag = sys.argv[1]; which = sys.argv[2] if len(sys.argv) > 2 else "w16"
AB = R.ROOT / "nn_x/abl/preds"
w = [f"w5_e10_{a}_s{s}" for s in range(951, 959 if which != "w8" else 955) for a in ("kd05w192", "kd05t25w192")]
p = [f"abl_e10_{a}_s{s}" for s in range(951, 955) for a in ("pop1kd05t25", "pop1kd05w192")] if which == "all24" else []
src = lambda m: (AB if m.startswith("abl_") else R.W5)  # noqa: E731
def pool(ms_, kind):
    return np.mean([R.zv(np.load((R.PR / f"nn_{m}_sft3_{tag}_val_oof.npy") if kind == "new" else (src(m) / f"nn_{m}_n2cpv3s3_val_oof.npy"))) for m in ms_], 0)
to = lambda ms_: np.mean([R.zv(np.load(src(m) / f"nn_{m}_trainonly_val.npy")) for m in ms_], 0)  # noqa: E731
fin = lambda a, b: 0.5 * R.zv(a) + 0.5 * R.zv(b)  # noqa: E731
done = [m for m in w + p if (R.PR / f"nn_{m}_sft3_{tag}_val_oof.npy").exists()]
print(f"{len(done)} of {len(w + p)} members finished: reading those")
ww = [m for m in done if m in w]
d = R.nd(fin(pool(ww, "new"), to(ww))) - R.nd(fin(pool(ww, "old"), to(ww)))
print(f"W pool ({len(ww)} members): final fresh val, STALE-FT {tag} - stored standard FT: {R.row(d)}   [all | M>=50 | M<50 | pmix-public | pmix-private]")
pp = [m for m in done if m in p]
if pp:
    f1 = lambda k: 0.8 * R.zv(pool(ww, k)) + 0.2 * R.zv(pool(pp, k))  # noqa: E731
    t1 = 0.8 * R.zv(to(ww)) + 0.2 * R.zv(to(pp))
    d = R.nd(fin(f1("new"), t1)) - R.nd(fin(f1("old"), t1))
    print(f"P20 form (0.8 z(W) + 0.2 z(pop-light x{len(pp)}) per branch): final fresh val delta {R.row(d)}")
