"""Label-free probe: do UNCAPPED feeds (H < 256) have events in their most recent strictly earlier snapshot that are
older than their own oldest stored event (T events)? Sample of 3,000 feeds per split. No label is read."""
import os, sys
sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_k] = "12"
sys.path.insert(0, "<repo>/lab/nn_x/w12/seg/refill")
import numpy as np
from align_lib import Flat, union_timeline, SPL, lcs

F = Flat()
rng = np.random.default_rng(1117)
for s in SPL:
    H, u = F.H[s], F.uid[s]
    n = len(H)
    for nm, m in (("uncapped", H < 256), ("capped", H == 256)):
        ids = np.where(m & (u != 0) & (H > 0))[0]
        ids = ids[[len(F.earlier(s, f)) > 0 for f in rng.choice(ids, min(len(ids), 12000), replace=False)][:0]] if False else ids
        cand = rng.permutation(ids)
        out = []
        for f in cand:
            sn = F.earlier(s, f)
            if not sn:
                continue
            U, st = union_timeline(F, s, f, snaps=sn[:1])
            ek, _ = F.seq(sn[0][0], sn[0][1])
            sk, _ = F.seq(SPL.index(s), f)
            # shared keys (multiset) as a crude overlap measure
            out.append([H[f], len(ek), st["n_merged"], st["first_anchor"], st["deep_anchor"], st["n_T"], st["n_U"],
                        (F.ts[s][f] - F.ts[SPL[sn[0][0]]][sn[0][1]]) / 86400])
            if len(out) >= 3000:
                break
        a = np.array(out, float)
        A = a[:, 2] >= 1
        Bm = A & (a[:, 4] >= a[:, 0] - 8)
        print(f"{s} {nm}: sampled {len(a)}; H median {np.median(a[:, 0]):.0f}; earlier-snapshot len median {np.median(a[:, 1]):.0f}; "
              f"A (>=16 anchors, run>=8) {A.mean():.3f}; end-anchored (deepest anchor >= H-8) {Bm.mean():.3f}; "
              f"T>=1 among A&B {(a[Bm, 5] >= 1).mean():.3f}, T mean {a[Bm, 5].mean():.1f}, T median {np.median(a[Bm, 5]):.0f}; "
              f"U>=1 among A {(a[A, 6] >= 1).mean():.3f}, U mean {a[A, 6].mean():.1f}; gap median {np.median(a[:, 7]):.2f} d", flush=True)
        if nm == "uncapped":
            # uncapped and NOT aligned: how many share no key at all with the stored history?
            na = ~A
            print(f"   uncapped not-A: {na.mean():.3f}; of them anchors==0: {(a[na, 3] == 0).mean():.3f}; H<16 share among not-A {(a[na, 0] < 16).mean():.3f}")
