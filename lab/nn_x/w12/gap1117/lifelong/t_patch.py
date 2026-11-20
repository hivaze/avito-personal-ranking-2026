"""Step 3 checks of model_patch.diff (GPU N only; label-free: no target array is read, no metric is computed).
The diff is applied to a temporary copy of patch_base/ (the lab/nn_x/w5 files it was written against, sha256 in
patch_base/SHA256_BASE.txt); lab/nn_x/w5 itself is never touched and no bytecode is written there.
  CUDA_VISIBLE_DEVICES=N python t_patch.py [--train_smoke] > out_t_patch.txt
Checks
 T1 flags off: patched == base code, fp32, every val feed (max |diff| must be 0) and bf16 against the stored val predictions
 T2 flags on, extension emptied (ext_n = 0): identical to flags off (max |diff| 0)
 T3 flags on (k0, capped only, rec_x = row 255; = the refill's untrained d1 convention): non-extended feeds unchanged within
    fp32 noise; how much extended feeds move (label-free)
 T4 (--train_smoke) train split with the extension: 3 optimiser steps on batches holding extended feeds; finite loss, rec_x
    receives gradient, the stored-only dense inputs are unchanged by the extension
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_k] = "12"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "GPU N only"
ap = argparse.ArgumentParser()
ap.add_argument("--train_smoke", action="store_true")
ap.add_argument("--member", default="w5_e10_kd05w192_s951")
ARGS = ap.parse_args()
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(12)
HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.environ.get("TMPDIR_PATCH", "/tmp/scratchpad")
tmp = tempfile.mkdtemp(prefix="w5patch_", dir=SCR)
for side in ("a", "b"):
    os.makedirs(f"{tmp}/{side}/lab/nn_x/w5")
    for f in ("data.py", "model.py", "train.py", "finetune.py", "holdout_eval.py"):
        shutil.copy(f"{HERE}/patch_base/{f}", f"{tmp}/{side}/lab/nn_x/w5/{f}")
r = subprocess.run(["patch", "-p1", "-d", f"{tmp}/b", "-i", f"{HERE}/model_patch.diff"], capture_output=True, text=True)
assert r.returncode == 0, r.stdout + r.stderr
print(f"diff applied cleanly to patch_base (tmp {tmp})", flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


A_model = load("model_a", f"{tmp}/a/lab/nn_x/w5/model.py")
A_data = load("data_a", f"{tmp}/a/lab/nn_x/w5/data.py")
B_model = load("model_b", f"{tmp}/b/lab/nn_x/w5/model.py")
B_data = load("data_b", f"{tmp}/b/lab/nn_x/w5/data.py")
dev = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = False  # fp32 identity checks
t0 = time.time()
CK = f"<repo>/lab/nn_x/w5/ckpt/{ARGS.member}.pt"
ck = torch.load(CK, map_location=dev, weights_only=False)
cfg = ck["cfg"]


@torch.no_grad()
def predict(model, data, bf16=False, feed_ids=None):
    """mirror of nn_x/w5/common.predict (same batching), fp32 unless bf16."""
    model.eval()
    feed_ids = np.arange(data.n) if feed_ids is None else feed_ids
    out = torch.zeros(data.n_cand, dtype=torch.float32, device=dev)
    for fb in data.batches(feed_ids, 65536, 1024, shuffle=False):
        b = data.gather(fb)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            lg = model(b, use_pos=False)
        data.scatter(out, b, lg)
    return out.cpu().numpy()


def split(mod, name, **kw):
    s = mod.SplitData(name, dev, **kw)
    s.attach_xt(cfg["xt_names"]) if hasattr(s, "attach_xt") else None
    return s


# --- the feat_cache path of data.py is relative to ROOT/nn_x/w5 (legacy layout ON); both copies read it read-only
va_a = split(A_data, "val")
m_a = A_model.Ranker(**cfg).to(dev)
m_a.load_state_dict(ck["state_dict"])
m_b0 = B_model.Ranker(**cfg).to(dev)
m_b0.load_state_dict(ck["state_dict"])
assert set(m_a.state_dict()) == set(m_b0.state_dict()) and m_b0.cfg == m_a.cfg, "flags-off cfg / state dict differ"
va_b0 = split(B_data, "val")
pa = predict(m_a, va_a)
pb = predict(m_b0, va_b0)
print(f"T1 flags off, fp32, all {va_a.n} val feeds: max |patched - base| = {np.abs(pa - pb).max():.3e} (must be 0)", flush=True)
assert np.abs(pa - pb).max() == 0
stored = np.load(f"<repo>/lab/nn_x/w5/preds/nn_{ARGS.member}_trainonly_val.npy")
torch.backends.cuda.matmul.allow_tf32 = True
pbf = predict(m_b0, va_b0, bf16=True)
print(f"T1b flags off, bf16 (as common.predict) vs the STORED val predictions of {ARGS.member}: corr {np.corrcoef(pbf, stored)[0, 1]:.7f}, "
      f"max |diff| {np.abs(pbf - stored).max():.3e}", flush=True)
torch.backends.cuda.matmul.allow_tf32 = False
del va_a
torch.cuda.empty_cache()

# --- flags on
va_b = split(B_data, "val", hist_ext=128)
m_b = B_model.Ranker(**{**cfg, "hist_ext": 128, "hist_ext_src": dict(dir=B_data.HIST_EXT_DIR, var="", uncapped=False)}).to(dev)
res = m_b.load_state_dict(ck["state_dict"], strict=False)
assert list(res.missing_keys) == ["rec_x.weight"] and not res.unexpected_keys, res
m_b.init_hist_ext("row255")
assert torch.equal(m_b.rec_x.weight[0], m_b.rec.weight[255]) and torch.equal(m_b.rec_x.weight[-1], m_b.rec.weight[255])
# T2: extension emptied
keep_n, keep_np = va_b.x_n.clone(), va_b.x_n_np.copy()
va_b.x_n.zero_()
va_b.x_n_np[:] = 0
p2 = predict(m_b, va_b)
print(f"T2 flags on, extension emptied: max |diff| vs flags off = {np.abs(p2 - pb).max():.3e} (must be 0)", flush=True)
assert np.abs(p2 - pb).max() == 0
va_b.x_n.copy_(keep_n)
va_b.x_n_np[:] = keep_np
# T3: extension on
p3 = predict(m_b, va_b)
ext = va_b.x_n_np > 0
M, off = va_b.M_np, va_b.cand_off_np
rows_ext = np.concatenate([np.arange(o, o + m) for o, m in zip(off[ext], M[ext])])
rows_no = np.setdiff1d(np.arange(va_b.n_cand), rows_ext)
d = np.abs(p3 - pb)
print(f"T3 flags on (k0, capped only, rec_x = row 255): {ext.sum()} of {va_b.n} val feeds extended; "
      f"non-extended feeds max |diff| {d[rows_no].max():.2e} (fp32 noise of other kernel shapes; batches now 384 wide), "
      f"extended feeds mean |diff| {d[rows_ext].mean():.4f} (score sd {pb[rows_ext].std():.3f}), corr {np.corrcoef(p3[rows_ext], pb[rows_ext])[0, 1]:.5f}", flush=True)
top_change = 0
for f in np.where(ext)[0]:
    s0, s1 = pb[off[f]:off[f] + M[f]], p3[off[f]:off[f] + M[f]]
    top_change += int(np.argmax(s0) != np.argmax(s1))
print(f"   extended feeds whose top-scored candidate changes: {top_change} of {ext.sum()} ({top_change / ext.sum():.3f})", flush=True)
# T3b: time of one val inference pass with / without the extension (bf16, as common.predict)
torch.backends.cuda.matmul.allow_tf32 = True
for nm, dd, mm in (("off", va_b0, m_b0), ("on", va_b, m_b)):
    predict(mm, dd, bf16=True)
    torch.cuda.synchronize()
    tt = time.time()
    for _ in range(3):
        predict(mm, dd, bf16=True)
    torch.cuda.synchronize()
    print(f"T3b val inference pass, bf16, extension {nm}: {(time.time() - tt) / 3:.2f} s", flush=True)
del va_b, va_b0
torch.cuda.empty_cache()

if ARGS.train_smoke:
    torch.backends.cuda.matmul.allow_tf32 = True
    tr = B_data.SplitData("train", dev, hist_ext=128)
    tr.xt = torch.zeros(tr.n_cand, len(cfg["xt_names"]), dtype=torch.float16, device=dev)  # placeholder inputs (smoke only)
    p_r = torch.zeros(tr.n, device=dev)
    tr.dom = torch.stack([1 - p_r, p_r, torch.zeros_like(p_r)], 1)
    m_t = B_model.Ranker(**{**cfg, "hist_ext": 128, "hist_ext_src": {}}).to(dev)
    m_t.load_state_dict(ck["state_dict"], strict=False)
    m_t.init_hist_ext("row255")
    m_t.train()
    opt = torch.optim.AdamW(m_t.parameters(), lr=1e-4)
    rng = np.random.default_rng(0)
    bs = [x for x in tr.batches(np.arange(tr.n), 32768, 512, shuffle=True, rng=rng)[:40] if tr.x_n_np[x].max() > 0][:3]
    for i, fb in enumerate(bs):
        b = tr.gather(fb)
        X = m_t.dense(b)
        b_st = tr.gather(fb)
        b_st = {k: v for k, v in b_st.items() if not k.startswith("x") or k == "xt"}
        X_st = m_t.dense(b_st)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m_t(b, X)
        loss = B_model.listwise_loss(lg, b["y"], b["cmask"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g = m_t.rec_x.weight.grad
        print(f"T4 train step {i}: batch {len(fb)} feeds (extended {int((tr.x_n_np[fb] > 0).sum())}, history width "
              f"{b['he_raw'].shape[1]} + {b['xe_raw'].shape[1]}), loss {float(loss):.4f} finite {bool(torch.isfinite(loss))}, "
              f"rec_x grad norm {float(g.norm()):.3e} (rows with grad {int((g.abs().sum(1) > 0).sum())}), "
              f"dense inputs identical with / without the x* keys: {bool(torch.equal(X, X_st))}", flush=True)
        opt.step()
print(f"done in {time.time() - t0:.0f}s", flush=True)
shutil.rmtree(tmp)
