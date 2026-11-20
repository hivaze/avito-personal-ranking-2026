"""Parity reference for nn_x/abl/popfree/finetune.py (2026-11-13): executes nn_x/w5/finetune.py VERBATIM (read from
disk, never copied or edited) with the nn_x/abl harness modules first on sys.path, so that its `from common import
CKPT, LOGS, PREDS` resolves to nn_x/abl/{ckpt,logs,preds} and a checkpoint in nn_x/abl/ckpt is fine-tuned by the
unchanged w5 code path (no symlink / copy under nn_x/w5/ckpt). Used once for the parity proof of popfree/finetune.py.

Usage: CUDA_VISIBLE_DEVICES=<0|1> python ref_w5_finetune.py --variant abl_base_s23 --out abl_base_s23_pftest_ref ...
"""
import sys

sys.path.insert(0, "<repo>/nn_x/abl")
SRC = "<repo>/nn_x/w5/finetune.py"
code = compile(open(SRC).read(), SRC, "exec")
sys.argv[0] = SRC
exec(code, {"__name__": "__main__", "__file__": SRC})
