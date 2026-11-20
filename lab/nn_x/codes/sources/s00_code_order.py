"""s00: are source / event-type / category codes frequency-ranked? Full-split bincounts (train all 583.6M events)."""
import numpy as np
F = "<repo>/data/flat"
for col, K in [("event_source", 31), ("event_type", 11), ("category_id", 54)]:
    print(f"== hist_{col}")
    for sp in ["train", "val", "test"]:
        a = np.load(f"{F}/{sp}/hist_{col}.npy", mmap_mode="r")
        c = np.zeros(K, np.int64)
        for i in range(0, a.shape[0], 50_000_000):
            c += np.bincount(np.asarray(a[i:i + 50_000_000]).astype(np.int64), minlength=K)[:K]
        nz = c[1:]
        order = np.argsort(-nz) + 1
        inv = sum(1 for i in range(1, K) for j in range(i + 1, K) if nz[i - 1] < nz[j - 1] and nz[j - 1] > 0)
        print(sp, "counts:", c.tolist())
        print(sp, "codes by descending count:", order[nz[order - 1] > 0].tolist(), "inversions among nonzero codes:", inv)
    if col == "category_id":
        for sp in ["train", "val"]:
            a = np.load(f"{F}/{sp}/cand_category_id.npy", mmap_mode="r")
            c = np.bincount(np.asarray(a).astype(np.int64), minlength=K)
            print(sp, "cand cat codes by desc count:", (np.argsort(-c[1:]) + 1).tolist())
