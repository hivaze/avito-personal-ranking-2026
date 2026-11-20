"""witness: benchmark of ANCHORED sub-batching (label-free): the reference batches of common.predict (65536 / 1024) are cut into sub-batches that
keep the reference padding (same M bucket; every sub-batch holds a feed whose history pads to the reference batch's Hx).
  python bench2.py <member> <n feeds>"""
from wt_common import *
import torch
torch.set_num_threads(int(NT))
import lagged_score as LS
v = sys.argv[1]; n = int(sys.argv[2])
sub = LS.SubSplit(); s = sub.s
m, names = LS.load_model(v); sub.attach(names, 'lag0')
rng = np.random.default_rng(0); feeds = np.sort(rng.choice(s.n, n, replace=False))
rows = np.concatenate([np.arange(s.cand_off_np[f], s.cand_off_np[f] + s.M_np[f]) for f in feeds])


def anchored(s, feed_ids, sub_budget, sub_maxf):
    out = []
    for fb in s.batches(feed_ids, 65536, 1024, shuffle=False):
        fb = np.asarray(fb)
        hx = np.minimum(np.maximum(-(-np.minimum(s.H_np[fb], s.max_hist) // 64) * 64, 64), s.max_hist)
        anc = fb[hx == hx.max()]; rest = fb[hx != hx.max()]
        Mp = int(-(-s.M_np[fb].max() // 16) * 16)
        B = max(1, min(sub_maxf, sub_budget // Mp))
        nsub = min(max(1, -(-len(fb) // B)), len(anc))
        order = np.concatenate([anc, rest])
        out += [order[i::nsub] for i in range(nsub)]
    return out


print(f'{v}: {n} feeds, {len(rows)} rows; torch threads {torch.get_num_threads()}', flush=True)
ref = None
for budget, maxf in ((65536, 1024), (4096, 64), (2048, 32), (1024, 16), (512, 8), (256, 4)):
    bl = s.batches(feeds, 65536, 1024, shuffle=False) if budget == 65536 else anchored(s, feeds, budget, maxf)
    out = torch.zeros(s.n_cand, dtype=torch.float32); t0 = time.time()
    with torch.inference_mode():
        for fb in bl:
            b = s.gather(fb); s.scatter(out, b, m(b, use_pos=False))
    x = out.numpy()[rows]; dt = time.time() - t0
    if ref is None: ref = x
    print(f' budget {budget:6d} max_feeds {maxf:5d}: {len(bl):4d} batches, {dt:6.1f}s, {len(rows) / dt:8.0f} rows/s, max |diff| vs reference {np.abs(x - ref).max():.2e}', flush=True)
