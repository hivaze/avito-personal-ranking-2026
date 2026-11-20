"""hz2 ADDENDUM 6 (HZ11): hold_infer.py with a free member list and the extra view name "stored" (every xt column as stored = CPU lag-0 replay through the same code path).
Usage: CUDA_VISIBLE_DEVICES="" python hold_infer2.py <feats | feats_L | stored> <member,member,...> <late | early | full>
  -> cache/hpred_<view>_<sub>_<member>.npy (rows of the feeds outside the subset are NaN). Label-free: writes predictions only."""
import sys
import hold_infer as HI
from hold_infer import *  # noqa: F401,F403


def attach2(s, names, vdir):
    cols = []
    for n in names:
        base = n[4:] if n.startswith('has:') else n
        x = np.asarray(np.load(FC / 'train' / f'{base}.npy', mmap_mode='r')[s.crow])
        x = (~np.isnan(x)).astype(np.float32) if n.startswith('has:') else xt_transform(n, x)
        cols.append(torch.from_numpy(np.nan_to_num(x, nan=0.0)).to(torch.float16))
    s.xt = torch.stack(cols, 1)
    return []


if __name__ == '__main__':
    if sys.argv[1] == 'stored':
        HI.attach = attach2
    HI.main()
