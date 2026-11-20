"""Four component pools of the current best file (FD16P20@m0.4_0.2) per split: z-mean of member z for
FT base (16), FT pop-light (4), TO base (16), TO pop-light (4). Also per-member z stacks for member-level reads (val only).
Checks that the rule recombination reproduces the stored ens_FD16P20_m04_02kd_{ft,to} files."""
import ph, numpy as np, sys
P = "<repo>/preds"
FD16 = [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in range(951, 959)]
FDP = [f'abl_fd10_pop1kd05w192_s{s}' for s in range(951, 955)]
for split in sys.argv[1:]:
    D = ph.Data(split)
    suf = {'val': ('n2cpv3s3_val_oof', 'trainonly_val'), 'test': ('n2cpv3s3_test', 'trainonly_test')}[split]
    out = {}
    for br, sf in zip(("ft", "to"), suf):
        for nm, mem in (("base", FD16), ("pl", FDP)):
            Zs = np.stack([ph.zf(np.load(f"{P}/nn_{m}_{sf}.npy").astype(np.float32), D.off, D.M).astype(np.float32) for m in mem])
            out[f"{br}_{nm}"] = Zs.mean(0)
            if split == "val": np.save(f"../out/memz_{br}_{nm}_val.npy", Zs.astype(np.float16))
    Mr = np.repeat(D.M, D.M); w = np.where(Mr >= 50, 0.2, 0.4)
    for br in ("ft", "to"):
        rec = (1 - w) * ph.zf(out[f"{br}_base"], D.off, D.M) + w * ph.zf(out[f"{br}_pl"], D.off, D.M)
        stored = D.ft if br == "ft" else D.to
        print(split, br, "max abs diff vs stored", np.abs(rec - stored).max(), "corr", np.corrcoef(rec, stored)[0, 1])
    np.savez(f"../out/comp_{split}.npz", **out)
