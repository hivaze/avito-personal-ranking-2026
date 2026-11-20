"""staleft information read (DECLARATION 4): fresh-val final-file delta of the scored forms W8 / W8P20 when the W8 fine-tune branch is the 1-seed
STALE-FT P = 1.0 branch instead of the stored 3-seed standard branch (and against the 1-seed P = 0 control). CPU; predictions on disk only."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "<repo>/nn_x/w12/stale/staleft")
import numpy as np
import pilot_read as R
AB = R.ROOT / "nn_x/abl/preds"
W8 = [f"w5_e10_{a}_s{s}" for s in range(951, 955) for a in ("kd05w192", "kd05t25w192")]
P1 = [f"abl_e10_pop1kd05t25_s{s}" for s in range(951, 955)]
zm = lambda fs: np.mean([R.zv(np.load(f)) for f in fs], 0)  # noqa: E731
to8 = zm([R.W5 / f"nn_{m}_trainonly_val.npy" for m in W8]); top = zm([AB / f"nn_{m}_trainonly_val.npy" for m in P1])
ftp = zm([AB / f"nn_{m}_n2cpv3s3_val_oof.npy" for m in P1])
ft = {"stored 3-seed": zm([R.W5 / f"nn_{m}_n2cpv3s3_val_oof.npy" for m in W8]), "1-seed P=0": zm([R.PR / f"nn_{m}_sft_p0_val_oof.npy" for m in W8]),
      "1-seed P=1.0": zm([R.PR / f"nn_{m}_sft_p10_val_oof.npy" for m in W8])}
if all((R.PR / f"nn_{m}_sft_p05_val_oof.npy").exists() for m in W8):
    ft["1-seed P=0.5"] = zm([R.PR / f"nn_{m}_sft_p05_val_oof.npy" for m in W8])
fin = lambda a, b: 0.5 * R.zv(a) + 0.5 * R.zv(b)  # noqa: E731
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
w8 = {k: R.nd(fin(v, to8)) for k, v in ft.items()}
p20 = {k: R.nd(fin(0.8 * R.zv(v) + 0.2 * R.zv(ftp), 0.8 * R.zv(to8) + 0.2 * R.zv(top))) for k, v in ft.items()}
P("fresh val, 14,423 feeds; columns: all | M>=50 | M<50 | pmix-public | pmix-private")
for name, d in (("W8 form", w8), ("W8P20 form", p20)):
    P(f"{name}: levels " + ", ".join(f"{k} {v.mean():.5f}" for k, v in d.items()))
    for k in d:
        if k != "stored 3-seed":
            P(f"  {name}: {k} - stored 3-seed: {R.row(d[k] - d['stored 3-seed'])}")
    for k in d:
        if k.startswith("1-seed P=") and k != "1-seed P=0":
            P(f"  {name}: {k} - 1-seed P=0:    {R.row(d[k] - d['1-seed P=0'])}")
P(f"W8P20 form - W8 form with the stored branch (the scored pair): {R.row(p20['stored 3-seed'] - w8['stored 3-seed'])}")
open(R.HERE / "out/p20_read.txt", "w").write("\n".join(out) + "\n")
