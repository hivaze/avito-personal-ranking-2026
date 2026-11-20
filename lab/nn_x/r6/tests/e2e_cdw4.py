"""Deterministic end-to-end wrapper for tests/test_cdw4.py G3 (GPU). Runs a train.py FILE as __main__ with the given
arguments after (1) switching torch to deterministic kernels (as test_cdw2 / test_r6 GPU mode) and (2) wrapping, without
changing their results:
  data.SplitData.gather   -> records every TRAINING batch (calls with_pos=True; eval / predict / feature_stats gather
                             without positions): feed ids, whether it came from the hcap view, per-type max counts;
  common.eval_subset      -> records whether each selection-holdout eval read the hcap view;
  common.predict          -> records whether each val predict read an hcap view (must never);
  cd_arms4.CoolStream     -> records the source of every non-pool cooldown batch (recent / rho / regular).
  usage: python tests/e2e_cdw4.py <train file> <record.npz> --variant r6_... <train.py args>
"""
import os
import runpy
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = os.path.dirname(os.path.abspath(__file__))
R6 = os.path.dirname(HERE)
sys.path.insert(0, R6)
train_file, rec_path, args = sys.argv[1], sys.argv[2], sys.argv[3:]
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)
import cd_arms4 as C4  # noqa: E402
import common as Cm  # noqa: E402
import data as D  # noqa: E402

REC = dict(fidx=[], capped=[], tcnt=[], src=[], eval_capped=[], predict_capped=[])
STATE = {"pending": None, "stream": False}
_gather, _eval, _predict = D.SplitData.gather, Cm.eval_subset, Cm.predict
_init, _next = C4.CoolStream.__init__, C4.CoolStream.next


def gather(self, fidx_np, with_pos=False):
    b = _gather(self, fidx_np, with_pos)
    if with_pos:
        REC["fidx"].append(np.asarray(fidx_np, dtype=np.int64).copy())
        REC["capped"].append(hasattr(self, "hcap"))
        het, hm = b["het"], b["hmask"]
        REC["tcnt"].append(torch.stack([((het == t) & hm).sum(1) for t in range(11)], 1).max(0).values.cpu().numpy())
        REC["src"].append(STATE["pending"] or ("pool" if STATE["stream"] else "nostream"))
        STATE["pending"] = None
    return b


def eval_subset(model, data, feed_ids, rows=None):
    REC["eval_capped"].append(hasattr(data, "hcap"))
    return _eval(model, data, feed_ids, rows)


def predict(model, data, *a, **k):
    REC["predict_capped"].append(hasattr(data, "hcap"))
    return _predict(model, data, *a, **k)


def init(self, *a, **k):
    STATE["stream"] = True
    return _init(self, *a, **k)


def nxt(self, fb):
    before = dict(self.n)
    out = _next(self, fb)
    STATE["pending"] = [k for k in self.n if self.n[k] != before[k]][0]
    return out


D.SplitData.gather, Cm.eval_subset, Cm.predict = gather, eval_subset, predict
C4.CoolStream.__init__, C4.CoolStream.next = init, nxt
sys.argv = ["train.py"] + args
try:
    runpy.run_path(train_file, run_name="__main__")
finally:
    f = REC["fidx"]
    np.savez(rec_path, fidx=np.concatenate(f) if f else np.zeros(0, np.int64),
             off=np.concatenate([[0], np.cumsum([len(x) for x in f])]).astype(np.int64),
             capped=np.array(REC["capped"], bool), tcnt=np.array(REC["tcnt"]).reshape(-1, 11),
             src=np.array(REC["src"]), eval_capped=np.array(REC["eval_capped"], bool),
             predict_capped=np.array(REC["predict_capped"], bool))
