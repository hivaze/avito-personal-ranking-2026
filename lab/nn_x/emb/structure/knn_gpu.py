import torch, numpy as np


def knn_ip(base, query, k, dev="cuda", qchunk=4096, bchunk=1_500_000):
    """exact top-k inner-product search on GPU (fp32). base/query: float32 numpy arrays. returns (D, I)."""
    B = torch.tensor(base, device=dev, dtype=torch.float32)
    Dout = np.zeros((len(query), k), np.float32); Iout = np.zeros((len(query), k), np.int64)
    for s in range(0, len(query), qchunk):
        q = torch.tensor(query[s:s + qchunk], device=dev, dtype=torch.float32)
        bestD = None
        for b in range(0, len(B), bchunk):
            S = q @ B[b:b + bchunk].T
            d, i = S.topk(min(k, S.shape[1]), 1)
            i = i + b
            if bestD is None:
                bestD, bestI = d, i
            else:
                d2 = torch.cat([bestD, d], 1); i2 = torch.cat([bestI, i], 1)
                o = d2.topk(k, 1).indices
                bestD, bestI = d2.gather(1, o), i2.gather(1, o)
        Dout[s:s + qchunk] = bestD.cpu().numpy(); Iout[s:s + qchunk] = bestI.cpu().numpy()
    del B
    torch.cuda.empty_cache()
    return Dout, Iout
