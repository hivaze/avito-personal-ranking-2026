"""(a/c) Re-run the n2cpv3 fine-tune of one checkpoint through ONE code path's finetune.py (nn/ or nn_x/r6/), with
every output redirected to this directory (CKPT read-only from the given dir; LOGS/PREDS -> out_ft/).
Usage: CUDA_VISIBLE_DEVICES=<id> python e_ft_rerun.py {nn|r6} <ckpt_dir> <variant>
Args = the n2cpv3 recipe from the logs (identical for nn/ F7 and r6 members, c_ftlogs.out)."""
import sys
from pathlib import Path

path, ckdir, variant = sys.argv[1:4]
ROOT = Path("<repo>")
code = {"nn": ROOT / "nn", "r6": ROOT / "nn_x" / "r6"}[path]
sys.path.insert(0, str(code))
OUT = Path(__file__).resolve().parent / "out_ft"
OUT.mkdir(exist_ok=True)

import torch  # noqa: E402

import data  # noqa: E402
import finetune  # noqa: E402

torch.set_num_threads(12)
assert Path(finetune.__file__).parent == code and Path(data.__file__).parent == code
sys.path.insert(0, str(Path(__file__).resolve().parent))
from names import EXTRA  # noqa: E402

for sp in ("val", "test"):  # nn/data.feat_path would hard-link a missing file into nn/feat_cache: refuse instead
    for n in EXTRA:
        assert (ROOT / "nn" / "feat_cache" / sp / f"{n}.npy").exists(), (sp, n)
finetune.CKPT = Path(ckdir)
finetune.LOGS = OUT
finetune.PREDS = OUT
out = f"{path}path__{variant}_n2cpv3"
sys.argv = ["finetune.py", "--variant", variant, "--out", out, "--epochs", "4", "--lr", "5e-05", "--wd", "0.02",
            "--warmup", "30", "--cand_budget", "8192", "--max_feeds", "128", "--loss", "softmax", "--loss_norm", "batch",
            "--folds", "5", "--seeds", "1", "--extra", "n2c_pre_v3", "--extra_lr_mult", "10.0", "--decay", "cos",
            "--ema", "0.99", "--ema_start", "0.5", "--u0_singleton", "--select", "snapshot"]
print("code path", path, finetune.__file__, "ckpt", finetune.CKPT / f"{variant}.pt", "out", OUT / out, flush=True)
finetune.main()
