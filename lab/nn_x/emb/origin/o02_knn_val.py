"""O2a: exact top-5 neighbours (int dot, fp32 no-TF32) of every val-table item within the val table (2.52M rows), GPU.
Outputs out/knn_val_idx.npy (N,5) int32, out/knn_val_dot.npy (N,5) int32 (dot of int8 vectors; cos = dot / (|a||b|))."""
import time
import numpy as np
import torch

T0 = time.time()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
OUT = "<repo>/nn_x/emb/origin/out"
N = 2516711
x = np.array(np.memmap("<repo>/data/competition/val/item_embeddings/data.memmap", dtype=np.int8,
                       mode="r", shape=(N, 128)))
X = torch.from_numpy(x.astype(np.float32)).cuda()
K = 6
I = np.zeros((N, K - 1), np.int32)
D = np.zeros((N, K - 1), np.int32)
B = 4096
with torch.no_grad():
    for s in range(0, N, B):
        q = X[s:s + B]
        S = q @ X.T
        ar = torch.arange(len(q), device="cuda")
        S[ar, ar + s] = -1e9  # exclude self
        v, i = torch.topk(S, K - 1, dim=1)
        I[s:s + len(q)] = i.cpu().numpy()
        D[s:s + len(q)] = torch.round(v).cpu().numpy().astype(np.int32)
        if s % (B * 100) == 0:
            print(f"[{time.time()-T0:6.1f}s] {s}", flush=True)
np.save(f"{OUT}/knn_val_idx.npy", I)
np.save(f"{OUT}/knn_val_dot.npy", D)
print(f"[{time.time()-T0:6.1f}s] done", flush=True)
