"""CPU check (2026-11-13 ~07:15 UTC) of the pop1 extension of nn_x/abl/popfree/finetune.py honour_xt_exclude.
NEW = the candidate file (argv[1]), OLD = the pre-extension copy (argv[2]). On real checkpoints:
  abl_pop1_s23 (27 xt, n_dense 93): NEW accepts and returns the cfg xt_names (xp_imp_c50 last); OLD refuses (the reason for
  the extension). abl_xdrop_pop_s23, abl_base_s23, abl_xrel_more_s23, abl_e10_xdrop_pop_s951 (no --xt_add): NEW == OLD
  (same return value, identical log line). Tampered pop1 cfgs (last xt name dropped; n_dense - 1; xt_add removed from the
  args) are refused by NEW. CPU only; read-only on the checkpoints. nn_x/abl/common.py insists on CUDA_VISIBLE_DEVICES = one
  id at import, so the env var is set, but the script asserts at the end that CUDA was never initialised (no GPU use)."""
import importlib.util
import os
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("3", "4", "5", "7")
import torch

CK = "<repo>/nn_x/abl/ckpt"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


NEW, OLD = load(sys.argv[1], "ft_new"), load(sys.argv[2], "ft_old")
fails = []


def run(mod, ck):
    lines = []
    try:
        return mod.honour_xt_exclude(ck, lambda s: lines.append(s)), lines, None
    except AssertionError as e:
        return None, lines, str(e)[:160]


def check(name, ok, info=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}")
    if not ok:
        fails.append(name)


ck = torch.load(f"{CK}/abl_pop1_s23.pt", map_location="cpu", weights_only=False)
cfg, args = ck["cfg"], ck["args"]
rn, ln, en = run(NEW, ck)
ro, lo, eo = run(OLD, ck)
check("abl_pop1_s23: NEW accepts; returns the cfg xt_names (27, xp_imp_c50 last); n_dense 93; log names the xt_add",
      en is None and rn == list(cfg["xt_names"]) and len(rn) == 27 and rn[-1] == "xp_imp_c50" and cfg["n_dense"] == 93
      and "abl xt_add honoured: ['xp_imp_c50'] appended last" in ln[0], (len(rn or []), cfg["n_dense"], ln[0][-120:] if ln else en))
check("abl_pop1_s23: OLD refuses (why the extension is needed)", eo is not None, eo)
for v in ("abl_xdrop_pop_s23", "abl_base_s23", "abl_xrel_more_s23", "abl_e10_xdrop_pop_s951"):
    c = torch.load(f"{CK}/{v}.pt", map_location="cpu", weights_only=False)
    a_, b_ = run(NEW, c), run(OLD, c)
    check(f"{v} (no --xt_add): NEW == OLD (return value, log line, no error)", a_ == b_ and a_[2] is None,
          f"{len(a_[0] or [])} xt, xt_add arg {c['args'].get('xt_add', '<absent>')!r}")
for name, mut in (("last xt name dropped", lambda k: k["cfg"].__setitem__("xt_names", list(k["cfg"]["xt_names"])[:-1])),
                  ("n_dense - 1", lambda k: k["cfg"].__setitem__("n_dense", k["cfg"]["n_dense"] - 1)),
                  ("--xt_add removed from the args", lambda k: k["args"].__setitem__("xt_add", ""))):
    k2 = {"cfg": dict(cfg), "args": dict(args)}
    mut(k2)
    check(f"tampered abl_pop1_s23 ({name}): NEW refuses", run(NEW, k2)[2] is not None, run(NEW, k2)[2])
check("CUDA never initialised (CPU-only check)", not torch.cuda.is_initialized())
print(f"== {'ALL PASS' if not fails else 'FAILED: ' + str(fails)}")
sys.exit(1 if fails else 0)
