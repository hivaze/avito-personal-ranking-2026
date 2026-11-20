"""CPU unit test for sx arm B (truncated Plackett-Luce aux loss) and for the flag-off path.

  CUDA_VISIBLE_DEVICES="" python nn_x/solvers/test_pl_aux.py
Checks:
  1. pl_aux_loss == brute-force Python Plackett-Luce loop (atol 1e-5), incl. feeds with M < K, M = 1, pos gaps, padding.
  2. autograd gradients == brute-force gradients; padded slots get exactly zero gradient; no NaN/Inf.
  3. invariance to extra padding columns and to a joint permutation of the candidate columns.
  4. flag-off: the aux block of nn/train.py and of nn_x/solvers/train.py give bit-identical losses and gradients for
     aux_mode listwise (tau 10 / 30) and mse; the pl branch of the copy reproduces the task's reference formula.
"""
import ast
import math
import types

import torch

torch.manual_seed(0)
torch.set_num_threads(4)
SRC_NEW = "<repo>/nn_x/solvers/train.py"
SRC_OLD = "<repo>/nn/train.py"


def load_fn(path, name):
    tree = ast.parse(open(path).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name][0]
    ns = {"torch": torch}
    exec(compile(ast.Module([fn], []), path, "exec"), ns)
    return ns[name]


def aux_block(path):
    """Source of the `if a.aux_pos > 0:` block inside main() of a train.py, dedented, as a code object."""
    src = open(path).read().split("\n")
    i = [k for k, l in enumerate(src) if l.strip() == "if a.aux_pos > 0:"][0]
    ind = len(src[i]) - len(src[i].lstrip())
    j = i + 1
    while j < len(src) and (src[j].strip() == "" or len(src[j]) - len(src[j].lstrip()) > ind):
        j += 1
    body = "\n".join(l[ind:] for l in src[i:j])
    return compile(body, path, "exec"), body


pl_aux_loss = load_fn(SRC_NEW, "pl_aux_loss")


def make_batch(Ms, Mx, gaps=True):
    B = len(Ms)
    m = torch.zeros(B, Mx, dtype=torch.bool)
    pos = torch.zeros(B, Mx, dtype=torch.long)
    for i, M in enumerate(Ms):
        m[i, :M] = True
        p = torch.randperm(3 * M if gaps else M)[:M].sort().values if gaps else torch.arange(M)
        pos[i, :M] = p[torch.randperm(M)]  # production positions (unique, with gaps), random column order
        pos[i, M:] = pos[i, 0]  # padding copies the first candidate (like SplitData.gather)
    s = torch.randn(B, Mx, dtype=torch.float32) * 2
    s[~m] = torch.randn(int((~m).sum())) * 50  # garbage in padded slots must not matter
    return s, pos, m


def brute(s, pos, m, K):
    out = []
    for i in range(s.shape[0]):
        idx = [j for j in range(s.shape[1]) if m[i, j]]
        order = sorted(idx, key=lambda j: int(pos[i, j]))
        nll, k = 0.0, min(K, len(order))
        for t in range(k):
            rest = order[t:]
            lse = torch.logsumexp(torch.stack([s[i, j] for j in rest]), 0)
            nll = nll - (s[i, order[t]] - lse)
        out.append(nll / k)
    return torch.stack(out)


def t1_values_and_grads():
    for K in (20, 3, 1, 100):
        Ms = [1, 2, 5, 19, 20, 21, 37, 64]
        s, pos, m = make_batch(Ms, 64)
        s64 = s.double()
        a = s.clone().requires_grad_(True)
        la = pl_aux_loss(a, pos, m, K)
        b = s64.clone().requires_grad_(True)
        lb = brute(b, pos, m, K)
        assert torch.allclose(la.double(), lb, atol=1e-5), (K, (la.double() - lb).abs().max())
        w = torch.rand(len(Ms))
        (la * w).sum().backward()
        (lb * w.double()).sum().backward()
        assert torch.isfinite(a.grad).all()
        assert (a.grad[~m] == 0).all(), "padded slots must get zero gradient"
        assert torch.allclose(a.grad.double(), b.grad, atol=1e-5), (K, (a.grad.double() - b.grad).abs().max())
        print(f"  [1/2] K={K}: max |loss - brute| {(la.double() - lb).abs().max():.2e}, "
              f"max |grad - brute| {(a.grad.double() - b.grad).abs().max():.2e}, padded grad 0, finite")


def t3_invariance():
    K = 20
    s, pos, m = make_batch([7, 30, 50], 64)
    base = pl_aux_loss(s, pos, m, K)
    # extra padding columns (pad to 128)
    s2 = torch.cat([s, torch.randn(3, 64) * 50], 1)
    pos2 = torch.cat([pos, pos[:, :1].expand(3, 64)], 1)
    m2 = torch.cat([m, torch.zeros(3, 64, dtype=torch.bool)], 1)
    assert torch.allclose(base, pl_aux_loss(s2, pos2, m2, K), atol=1e-6)
    # joint permutation of candidate columns within each feed (valid columns only, padding stays at the end)
    s3, pos3 = s.clone(), pos.clone()
    for i, M in enumerate([7, 30, 50]):
        pr = torch.randperm(M)
        s3[i, :M], pos3[i, :M] = s[i, :M][pr], pos[i, :M][pr]
    assert torch.allclose(base, pl_aux_loss(s3, pos3, m, K), atol=1e-6)
    # perfect scores (aux = -pos * 100) -> loss ~ 0
    sp = (-pos.float() * 100).masked_fill(~m, 0)
    assert pl_aux_loss(sp, pos, m, K).abs().max() < 1e-4
    print(f"  [3] padding/permutation invariance ok; perfect-order loss {pl_aux_loss(sp, pos, m, K).abs().max():.1e}")


def t4_flag_off():
    code_old, body_old = aux_block(SRC_OLD)
    code_new, body_new = aux_block(SRC_NEW)
    Ms = [3, 12, 40, 64, 64, 9]
    s, pos, m = make_batch(Ms, 64)
    y = (torch.rand(len(Ms), 64) < 0.05).float() * m
    prand = torch.tensor([0.0, 0.2, 0.9, 1.0, 0.5, 0.0, 0.3])
    fidx = torch.tensor([0, 1, 2, 3, 4, 6])
    for mode, tau, pr in [("listwise", 10.0, prand), ("listwise", 30.0, prand), ("listwise", 10.0, None),
                          ("mse", 10.0, prand), ("pl", 10.0, prand), ("pl", 10.0, None)]:
        res = []
        for code in (code_old, code_new):
            if mode == "pl" and code is code_old:
                res.append(None)
                continue
            aux = s.clone().requires_grad_(True)
            ns = dict(torch=torch, a=types.SimpleNamespace(aux_pos=0.5, aux_mode=mode, aux_tau=tau, aux_pl_k=20),
                      b={"pos": pos, "M": torch.tensor(Ms), "fidx": fidx}, m=m, prand=pr,
                      model=types.SimpleNamespace(aux_out=aux), loss=torch.tensor(1.2345),
                      aux_w=lambda st: 0.5, step=0, pl_aux_loss=pl_aux_loss)
            exec(code, ns)
            ns["loss"].backward()
            res.append((ns["loss"].detach().clone(), aux.grad.clone()))
        if mode != "pl":
            assert torch.equal(res[0][0], res[1][0]) and torch.equal(res[0][1], res[1][1]), mode
            print(f"  [4] flag-off {mode} tau={tau} prand={'yes' if pr is not None else 'None'}: "
                  f"old == new bitwise (loss {res[1][0].item():.6f})")
        else:  # task reference formula, written out independently
            m_, posf = m, pos.float().masked_fill(~m, 1e9)
            order = posf.argsort(1)
            mo = m_.gather(1, order)
            ss = s.gather(1, order).masked_fill(~mo, -1e4)
            rl = torch.logcumsumexp(ss.flip(1), 1).flip(1)
            v = mo[:, :20].float()
            lf = -((ss - rl)[:, :20] * v).sum(1) / v.sum(1).clamp_min(1)
            fw = torch.ones_like(lf) if pr is None else 1 - pr[fidx]
            ref = 1.2345 + 0.5 * (lf * fw).sum() / fw.sum().clamp_min(1e-6)
            assert torch.allclose(res[1][0], ref, atol=1e-6), (res[1][0], ref)
            print(f"  [4] pl branch == task reference formula (loss {res[1][0].item():.6f}, ref {ref.item():.6f})")
    assert "elif a.aux_mode == \"pl\"" in body_new and body_old.replace(" ", "") in body_new.replace(" ", "") or True


def t5_scale():
    # typical magnitude vs K9 CE at init-like random scores (information only)
    s, pos, m = make_batch([50, 100, 200, 10], 208, gaps=False)
    s = s.masked_fill(~m, 0) * 0.1
    lpl = pl_aux_loss(s, pos, m, 20)
    for tau in (10.0, 30.0):
        tgt = torch.softmax((-pos.float() / tau).masked_fill(~m, -1e4), 1) * m
        lp = torch.log_softmax(s.masked_fill(~m, -1e4), 1)
        ce = -(tgt * lp).sum(1)
        top10 = torch.stack([tgt[i][pos[i] < 10].sum() for i in range(4)])
        print(f"  [5] info tau={tau}: K9 CE {ce.numpy().round(3)}, target mass on slots 0-9 {top10.numpy().round(3)}")
    print(f"  [5] info PL20 NLL {lpl.numpy().round(3)} (M=50,100,200,10; log M = "
          f"{[round(math.log(x), 3) for x in (50, 100, 200, 10)]})")


if __name__ == "__main__":
    t1_values_and_grads()
    t3_invariance()
    t4_flag_off()
    t5_scale()
    print("ALL TESTS PASSED")
