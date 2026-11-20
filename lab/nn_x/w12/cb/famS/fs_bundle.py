"""Member-equal z-mean bundle of famS arms, saved as prediction files (so that the bundle is a tag for fs_read.py / fs_m50.py and usable by the lead).
Per-feed z = the convention of nats.py (pc_common.Meta.z = stack_val.feed_rank_z[:, 1]); no label is used.
usage: python fs_bundle.py <bundle_tag> <tag> <tag> [...]   -> preds/<bundle_tag>_{val,hold,test}.npy + preds/<bundle_tag>.members.json"""
import sys, json
import numpy as np
sys.path.insert(0, '<repo>/nn_x/w12/cb')
from nats import _meta
from pathlib import Path
R = Path('<repo>/nn_x/w12/cb')
name, tags = sys.argv[1], sys.argv[2:]
m = _meta()


def src(tag, sp):
    for d in (R / 'famS/preds', R / 'infra/out'):
        if (d / f'{tag}_{sp}.npy').exists():
            return d / f'{tag}_{sp}.npy'
    raise FileNotFoundError((tag, sp))


for sp in ('val', 'hold', 'test'):
    z = np.mean([m.z(sp, np.load(src(t, sp)).astype(np.float32)) for t in tags], 0).astype(np.float32)
    assert len(z) == m.n[sp] and np.isfinite(z).all()
    np.save(R / f'famS/preds/{name}_{sp}.npy', z)
    print(f'{name} {sp}: {len(z)} rows, members {len(tags)}')
(R / f'famS/preds/{name}.members.json').write_text(json.dumps(dict(bundle=name, members=tags, rule='member-equal mean of per-feed z')))
