"""e01: are the anonymised codes (event_type, event_source, category) assigned by frequency rank?
Full-split bincounts (train all 583.6M history events / 246.7M candidates; val; test)."""
import numpy as np
ROOT = '<repo>/data/flat'
np.set_printoptions(linewidth=200, suppress=True)


def bc(split, name, n):
    x = np.load(f'{ROOT}/{split}/{name}.npy', mmap_mode='r')
    out = np.zeros(n, np.int64)
    step = 50_000_000
    for s in range(0, len(x), step):
        out += np.bincount(np.asarray(x[s:s + step]).astype(np.int64), minlength=n)[:n]
    return out


for name, n in [('hist_event_type', 11), ('hist_event_source', 31), ('hist_category_id', 54), ('cand_category_id', 54)]:
    print(f'\n===== {name}')
    for split in ['train', 'val', 'test']:
        c = bc(split, name, n)
        sh = c / c.sum()
        nz = c[1:]
        order = np.argsort(-nz) + 1
        # is code order == frequency rank (ignoring 0)?
        present = np.where(nz > 0)[0] + 1
        rank_of_code = np.empty(n, int); rank_of_code[order] = np.arange(1, len(order) + 1)
        mono = np.all(np.diff(nz[present - 1]) <= 0)
        inv = sum(1 for i in range(len(present)) for j in range(i + 1, len(present)) if nz[present[i] - 1] < nz[present[j] - 1])
        npairs = len(present) * (len(present) - 1) // 2
        print(f'{split:5s} total={c.sum():,} code0 share={sh[0]:.4f}  monotone-decreasing-by-code={mono}  '
              f'inversions={inv}/{npairs}  (Kendall tau={1 - 2 * inv / max(npairs, 1):.3f})')
        print('   shares by code 1..:', ' '.join(f'{i}:{sh[i]*100:.2f}' for i in range(1, n) if c[i] > 0))
        print('   codes sorted by freq:', order[:40].tolist())
