"""CPU smoke test of the WHOLE finetune_stale.main() path (no GPU is touched: CUDA_VISIBLE_DEVICES="" for the life of the process).
The module sources are executed with two textual patches (common.py: the GPU-id guard; finetune_stale.py: device cpu) and with
truncated work (3 training batches per epoch, predictions on the first 48 feeds of each request). Outputs go to out/cpu_smoke/.
It checks that every code path runs (stale view, feed mix, extra outputs, saving); numbers are meaningless."""
import os, sys, types, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["STALEFT_OUT"] = "<repo>/nn_x/w12/stale/staleft/out/cpu_smoke"
os.environ["NUMBA_CACHE_DIR"] = "<repo>/nn_x/w12/stale/staleft/out/numba_cache"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, torch
torch.set_num_threads(12)


def load_patched(name, reps):
    src = open(os.path.join(HERE, f"{name}.py")).read()
    for a, b in reps:
        assert src.count(a) == 1, (name, a)
        src = src.replace(a, b)
    m = types.ModuleType(name); m.__file__ = os.path.join(HERE, f"{name}.py")
    sys.modules[name] = m
    exec(compile(src, m.__file__, "exec"), m.__dict__)
    return m


common = load_patched("common", [('    raise RuntimeError("nn_x/w5: set CUDA_VISIBLE_DEVICES to exactly one GPU id 0-7")', "    pass")])
_predict = common.predict


def predict_small(model, data, feed_ids=None, **kw):
    feed_ids = np.arange(data.n) if feed_ids is None else np.asarray(feed_ids)
    return _predict(model, data, feed_ids[:48], **kw)


common.predict = predict_small
import data as Dm
_batches = Dm.SplitData.batches


def batches_small(self, feed_ids, cand_budget=32768, max_feeds=512, shuffle=True, rng=None, mpad=16):
    out = _batches(self, feed_ids, cand_budget, max_feeds, shuffle, rng, mpad)
    return out[:3] if shuffle else out


Dm.SplitData.batches = batches_small
F = load_patched("finetune_stale", [('dev = torch.device("cuda:0")', 'dev = torch.device("cpu")')])
t0 = time.time()
for P in sys.argv[1:] or ["0.5"]:
    F.main(["--variant", "w5_e10_kd05w192_s951", "--out", f"smoke_p{P}", "--extra", "n2c_pre_v3", "--lr", "5e-5", "--decay", "cos", "--ema", "0.99",
            "--ema_start", "0.5", "--epochs", "2", "--u0_singleton", "--seeds", "1", "--stale_train", P, "--warmup", "2"] + os.environ.get("SMOKE_EXTRA", "").split())
    print(f"smoke P={P} done {time.time() - t0:.0f}s", flush=True)
