"""witness: batch-size benchmark of the CPU inference path (label-free). Same model, same feeds; only the batching of SplitData.batches changes.
  python bench.py <member> <n feeds>"""
from wt_common import *
import torch
torch.set_num_threads(int(NT))
import lagged_score as LS
v = sys.argv[1]; n = int(sys.argv[2])
sub = LS.SubSplit(); s = sub.s
m, names = LS.load_model(v); sub.attach(names, 'lag0')
rng = np.random.default_rng(0); feeds = np.sort(rng.choice(s.n, n, replace=False))
rows = np.concatenate([np.arange(s.cand_off_np[f], s.cand_off_np[f] + s.M_np[f]) for f in feeds])
print(f'{v}: {n} feeds, {len(rows)} rows; torch threads {torch.get_num_threads()} interop {torch.get_num_interop_threads()}', flush=True)
ref = None
for budget, maxf in ((65536, 1024), (16384, 256), (4096, 64), (1024, 16), (65536, 1024)):
    out = torch.zeros(s.n_cand, dtype=torch.float32); t0 = time.time(); nb_ = 0
    with torch.inference_mode():
        for fb in s.batches(feeds, budget, maxf, shuffle=False):
            b = s.gather(fb); s.scatter(out, b, m(b, use_pos=False)); nb_ += 1
    x = out.numpy()[rows]; dt = time.time() - t0
    if ref is None: ref = x
    print(f' budget {budget:6d} max_feeds {maxf:5d}: {nb_:4d} batches, {dt:6.1f}s, {len(rows) / dt:8.0f} rows/s, max |diff| vs first {np.abs(x - ref).max():.2e}', flush=True)
