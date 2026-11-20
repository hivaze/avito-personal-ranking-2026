"""ENCODER-DOSE (C) step 2b: feed the exported 64-d states to the ranker's batches without editing nn_x/abl/data.py.

attach_kv64(sd, path_dir) loads data/feats/<split>/kv64_states.npy (fp16 [n_rows, 64], compact, newest-first per exported feed) and
kv64_off.npy ([n_feeds] int64, -1 = not exported) onto sd.device and wraps sd.gather so that every batch carries
    b["kv"]    [B, Hx, 64] fp16   the state of history slot k (= raw row hist_off + k), 0 where the slot is padded or the feed has no
                                  states (the adapter replaces those by its learned null token through kv_ok / hmask)
    b["kv_ok"] [B] bool           the feed has exported states
The gather is the same index arithmetic as SplitData.gather's history gather (hist_off + k, k < min(H, max_hist)), applied to the
compact offsets kv64_off. Flag off = this function is never called = the batch dict is the base's.
"""
import numpy as np
import torch


def attach_kv64(sd, feats_dir):
    """sd: nn_x/abl/data.SplitData (or tests' MiniSplit). feats_dir: directory with kv64_states.npy / kv64_off.npy of sd.split."""
    assert not hasattr(sd, "kv64_states"), "kv64 already attached"
    st = np.load(f"{feats_dir}/kv64_states.npy", mmap_mode="r")
    off = np.load(f"{feats_dir}/kv64_off.npy")
    assert st.dtype == np.float16 and st.shape[1] == 64 and off.dtype == np.int64, (st.dtype, st.shape, off.dtype)
    dev = sd.device
    # one zero row appended at the end: padded / unexported slots gather it (index n_rows); copied in chunks straight from the
    # memmap (train: ~35 GB fp16 -> no host / device double copy)
    n = st.shape[0]
    sd.kv64_states = torch.zeros((n + 1, 64), dtype=torch.float16, device=dev)
    for a0 in range(0, n, 8_000_000):
        sd.kv64_states[a0:min(a0 + 8_000_000, n)] = torch.from_numpy(np.ascontiguousarray(st[a0:a0 + 8_000_000])).to(dev)
    sd.kv64_zero = n
    feeds = getattr(sd, "src_feeds", None)  # tests' MiniSplit: a subset of the split's feeds, in sorted order
    sd.kv64_off = torch.from_numpy(off if feeds is None else off[np.asarray(feeds)]).to(dev)
    base_gather = sd.gather

    def gather(fidx_np, with_pos=False):
        b = base_gather(fidx_np, with_pos)
        fidx = b["fidx"]
        o = sd.kv64_off[fidx]  # [B]
        ok = o >= 0
        K = b["hmask"].shape[1]
        ar = torch.arange(K, device=dev)
        rows = o[:, None] + ar[None, :]
        rows = torch.where(ok[:, None] & b["hmask"], rows, torch.full_like(rows, sd.kv64_zero))
        b["kv"] = sd.kv64_states[rows]
        b["kv_ok"] = ok
        return b

    sd.gather = gather
    return sd
