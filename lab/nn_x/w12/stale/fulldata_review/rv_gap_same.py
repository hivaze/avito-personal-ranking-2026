"""Reviewer: in-sample lift of a student trained with labels (0.5) + an OUT-OF-SAMPLE teacher (0.5) on its training feeds
= the situation of the 30,000 holdout feeds in a full-data twin. Member w5_e10_kx05_s951 (teacher_oof, d192, kd_w 0.5), @last.
Prediction path = nn_x/w12/kd/dump_teacher.py (the project's own teacher dump), on the CPU.
usage: CUDA_VISIBLE_DEVICES="" python rv_gap.py <variant> <n_insample_feeds>"""
import os, sys, time, json
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == ""
sys.dont_write_bytecode = True
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
torch.set_num_threads(12)
def _boom(*a, **k): raise RuntimeError("CUDA attempted")
torch.cuda.is_available = lambda: False; torch.cuda._lazy_init = _boom
_RealDev = torch.device
class _Meta(type):
    def __instancecheck__(cls, o): return isinstance(o, _RealDev)
    def __call__(cls, *a, **k):
        d = _RealDev(*a, **k); return _RealDev("cpu") if d.type == "cuda" else d
class FakeDevice(metaclass=_Meta): pass
torch.device = FakeDevice
R = "<repo>"; CD = f"{R}/nn_x/w5"; sys.path.insert(0, CD)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
try: import common
finally: os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(12)
from data import SplitData
from model import Ranker
v, n_in = sys.argv[1], int(sys.argv[2]); t0 = time.time()
def P(*a): print(*a, flush=True)
dev = torch.device("cpu"); tr = SplitData("train", dev)
Mt, offt = tr.M_np, tr.cand_off_np
p = np.nan_to_num(np.load(f"{R}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0); p[Mt < 50] = 0.0
cand = np.where((p > 0.9) & (Mt >= 50))[0]; hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
ins30 = None
ins = np.sort(np.random.default_rng(777).choice(np.setdiff1d(cand, hold), n_in, replace=False)); hsub = np.sort(np.random.default_rng(98).choice(hold, 2000, replace=False))
ck = torch.load(f"{CD}/ckpt/{v}@last.pt", map_location=dev, weights_only=False); cfg = ck["cfg"]
assert ck["args"]["holdout"] == 30000
if cfg.get("xt_names"): tr.attach_xt(cfg["xt_names"])
prt = torch.from_numpy(p)
tr.dom = torch.stack([1 - prt, prt, torch.zeros_like(prt)], 1) if cfg.get("domain") else None
m = Ranker(**cfg).to(dev); m.load_state_dict(ck["state_dict"])
P(f"{v}@last loaded (kd_file {ck['args'].get('kd_file')}), {time.time() - t0:.0f}s")
s = common.predict(m, tr, np.sort(np.concatenate([ins, hsub])))
rows = lambda ids: np.concatenate([np.arange(o, o + k) for o, k in zip(offt[ids], Mt[ids])])
ir, hr_all = rows(ins), rows(hold)
np.savez(f"{R}/nn_x/w12/stale/fulldata_review/logs/gapsame_{v}.npz", ins=ins, hsub=hsub, s_ins=s[ir], s_hsub=s[rows(hsub)])
def nd(sc, t, M):
    n = len(M); fid = np.repeat(np.arange(n), M); off = np.concatenate([[0], np.cumsum(M)[:-1]])
    o = np.lexsort((t, -sc.astype(np.float64), fid)); rk = np.arange(len(sc)) - np.repeat(off, M); d = 1.0 / np.log2(rk + 2.0)
    dcg = np.bincount(fid, weights=t[o] * d, minlength=n); npos = np.bincount(fid, weights=t, minlength=n)
    return dcg / np.bincount(fid, weights=(rk < np.repeat(npos, M)) * d, minlength=n)
se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
y = tr.target_np.astype(np.float64)
T = np.load(ck["args"]["kd_file"], mmap_mode="r")
own_h = np.load(f"{CD}/preds/holdout/nn_{v}@last_holdout.npy").astype(np.float64)
a_in, t_in = nd(s[ir].astype(np.float64), y[ir], Mt[ins]), nd(np.asarray(T[ir], np.float64), y[ir], Mt[ins])
a_h, t_h = nd(own_h, y[hr_all], Mt[hold]), nd(np.asarray(T[hr_all], np.float64), y[hr_all], Mt[hold])
# path check on 2,000 holdout feeds: this CPU prediction vs the saved GPU file
pos = np.searchsorted(hold, hsub); offh = np.concatenate([[0], np.cumsum(Mt[hold])]); rr = np.concatenate([np.arange(offh[i], offh[i + 1]) for i in pos])
c_cpu, c_gpu = nd(s[rows(hsub)].astype(np.float64), y[rows(hsub)], Mt[hsub]), nd(own_h[rr], y[rows(hsub)], Mt[hsub])
P(f"path check (2,000 holdout feeds): CPU {c_cpu.mean():.5f} vs saved GPU file {c_gpu.mean():.5f} ({(c_cpu - c_gpu).mean():+.5f} +- {se(c_cpu - c_gpu):.5f})")
P(f"student: in-sample feeds (n {n_in}) {a_in.mean():.5f} +- {se(a_in):.5f} | holdout (n 30000) {a_h.mean():.5f} +- {se(a_h):.5f} | unpaired gap {a_in.mean() - a_h.mean():+.5f} +- {np.sqrt(se(a_in) ** 2 + se(a_h) ** 2):.5f}")
P(f"its teacher file: same in-sample feeds {t_in.mean():.5f} | holdout {t_h.mean():.5f} | unpaired gap {t_in.mean() - t_h.mean():+.5f}")
di, dh = a_in - t_in, a_h - t_h
P(f"student - teacher: in-sample feeds {di.mean():+.5f} +- {se(di):.5f} | holdout {dh.mean():+.5f} +- {se(dh):.5f} | DIFF-IN-DIFF (label memorisation with an OOS teacher) {di.mean() - dh.mean():+.5f} +- {np.sqrt(se(di) ** 2 + se(dh) ** 2):.5f}")
json.dump(dict(variant=v, n_in=n_in, ins=float(a_in.mean()), hold=float(a_h.mean()), t_ins=float(t_in.mean()), t_hold=float(t_h.mean()), did=float(di.mean() - dh.mean()), did_se=float(np.sqrt(se(di) ** 2 + se(dh) ** 2))),
          open(f"{R}/nn_x/w12/stale/fulldata_review/logs/gapsame_{v}.json", "w"))
P(f"done {(time.time() - t0) / 60:.1f} min")
