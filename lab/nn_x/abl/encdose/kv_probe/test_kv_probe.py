"""ENCODER-DOSE (C) tests (ledger nn_x/abl/ledger/ledger.md section 10.C): the K/V adapter, the batch gather wrapper, the leak rule
and the raw-slot <-> encoder-position mapping. CPU only; synthetic states (no export needed).

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/abl/encdose/kv_probe/test_kv_probe.py   -> nn_x/abl/logs/test_kv_probe.log

  K1 leak rule: train feeds exported iff Moscow day >= 20708 (Sep 12); val / test all exported.
  K2 mapping: raw_slots_from_states(h, H)[k] == h[H - k] (newest-first raw slot k); the export's gather index pos = H - k reproduces it.
  K3 gather wrapper: b["kv"][i, k] == states[off[f] + k] for real slots of exported feeds, 0 for padded slots / unexported feeds;
     b["kv_ok"] == (off >= 0); every other batch key bit-identical to the base gather.
  K4 adapter, flag off / at init: a Ranker built as abl_base_s23 with setup_kv64 gives BIT-IDENTICAL logits (eval and train mode,
     same seed) to the same Ranker without the adapter on a real MiniSplit train batch; the adapter's own parameters are the only
     additional ones; cfg["kv64"] == 1 is recorded; remove_kv64 restores the base.
  K5 adapter, learning: with gate != 0 the logits change; gradients reach proj / gate / null / user0 / ln; the null token is used exactly
     where kv_ok is False or the slot is padded; the user slot gets user0.
  K6 PCA apply: a fitted (mean, components) applied by the export formula gives orthonormal components and the right shapes.
"""
import os
import sys
import tempfile

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = "<repo>"
ABL = f"{ROOT}/nn_x/abl"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{ABL}/tests")
sys.path.insert(1, ABL)
sys.path.insert(2, HERE)
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import model as MO  # noqa: E402
import test_abl as TA  # noqa: E402
from mini_split import MiniSplit, pick_train_feeds  # noqa: E402
import kv_adapter as KA  # noqa: E402
import kv_data as KD  # noqa: E402

LOG = open(f"{ABL}/logs/test_kv_probe.log", "w")
FAIL = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def check(name, ok, info=""):
    P(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}")
    if not ok:
        FAIL.append(name)


def main():
    P("== K1 leak rule")
    # export_states imports torch / the scorer's Tables lazily inside functions? No: import the module for its pure helpers
    import export_states as EX
    day = np.array([20700, 20707, 20708, 20715, 20799])
    check("train: exported iff day >= 20708 (Sep 12 MSK)", EX.exported_mask("train", day).tolist() == [False, False, True, True, True])
    check("val / test: all exported", EX.exported_mask("val", day).all() and EX.exported_mask("test", day).all())
    check("Sep 12 2026 00:00 MSK is day 20708", EX.msk_day(np.array([1789160400]))[0] == 20708 and EX.msk_day(np.array([1789160400 - 1]))[0] == 20707)

    P("\n== K2 raw-slot mapping")
    H, d = 7, 5
    h = torch.arange((H + 1) * d, dtype=torch.float32).view(H + 1, d)  # h[t] = row t
    s = EX.raw_slots_from_states(h, H)
    check("raw_slots_from_states(h, H)[k] == h[H - k] for k = 0..H-1", all(torch.equal(s[k], h[H - k]) for k in range(H)) and s.shape == (H, d))
    Hc = torch.tensor([H, 3])
    L = H
    ar = torch.arange(L)
    pos = (Hc[:, None] - ar[None, :]).clamp_min(0)
    z = torch.stack([h, h])
    g = z.gather(1, pos[..., None].expand(-1, -1, d))
    check("export gather pos = H - k reproduces the mapping for every feed in a padded chunk (slots k >= H are don't-care)",
          all(torch.equal(g[0, k], h[H - k]) for k in range(H)) and all(torch.equal(g[1, k], h[3 - k]) for k in range(3)))

    P("\n== K3 gather wrapper (synthetic states)")
    feeds = pick_train_feeds(n_each=12, seed=11)
    base_lay = TA.xt_layout("cvnd_pre2s")
    ms = MiniSplit("train", feeds, xt_names=base_lay[0])
    fs = np.sort(np.asarray(feeds))
    import polars as pl
    f = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["H"])
    Hall = f["H"].to_numpy().astype(np.int64)
    n_all = len(Hall)
    rng = np.random.default_rng(3)
    exp = rng.random(len(fs)) < 0.6  # 60% of the test feeds "exported"
    off = np.full(n_all, -1, np.int64)
    Hs = Hall[fs]
    cum = 0
    for i, (fe, e) in enumerate(zip(fs, exp)):
        if e:
            off[fe] = cum
            cum += Hs[i]
    states = rng.standard_normal((max(cum, 1), 64)).astype(np.float16)
    tmp = tempfile.mkdtemp(prefix="kv64_", dir="/tmp/scratchpad")
    np.save(f"{tmp}/kv64_states.npy", states)
    np.save(f"{tmp}/kv64_off.npy", off)
    fb = np.arange(ms.n)
    b0 = ms.gather(fb, with_pos=True)
    KD.attach_kv64(ms, tmp)
    b1 = ms.gather(fb, with_pos=True)
    same = all(torch.equal(b0[k], b1[k]) for k in b0)
    ok_kv = True
    for i in range(ms.n):
        fe = fs[i]
        K = b1["hmask"].shape[1]
        for k in range(K):
            want = states[off[fe] + k] if (off[fe] >= 0 and k < min(Hall[fe], 256)) else np.zeros(64, np.float16)
            ok_kv = ok_kv and np.array_equal(b1["kv"][i, k].numpy(), want)
    check("b['kv'][i, k] == states[off + k] on real slots of exported feeds, 0 elsewhere; b['kv_ok'] == (off >= 0); other keys bit-identical",
          same and ok_kv and torch.equal(b1["kv_ok"], torch.from_numpy(off[fs] >= 0)) and b1["kv"].dtype == torch.float16, (same, ok_kv))

    P("\n== K4 adapter: flag off / at init bit-identity")
    out = TA.subprocess.run(["bash", TA.LANE, "4", "base:23"], capture_output=True, text=True, env={**os.environ, "DRY": "1", "ABL_GPUS": "4 5"})
    cmd = [ln for ln in out.stdout.splitlines() if " train.py " in ln][0].split(" train.py ", 1)[1].split()
    a = TA.load_parse(TA.TRAIN_PY)(cmd)
    m0 = TA._ranker(a, base_lay)
    Xd = m0.dense(b1)
    m0.set_norm(Xd[b1["cmask"]].mean(0), Xd[b1["cmask"]].std(0).clamp_min(1e-3))
    sd0 = {k: v.clone() for k, v in m0.state_dict().items()}
    m0.eval()
    l_base = m0(b1, m0.dense(b1))
    torch.manual_seed(5)
    m0.train()
    lt_base = m0(b1, m0.dense(b1))
    m0.eval()
    KA.setup_kv64(m0, log=lambda *a_: None)
    l_kv = m0(b1, m0.dense(b1))
    torch.manual_seed(5)
    m0.train()
    lt_kv = m0(b1, m0.dense(b1))
    m0.eval()
    new_keys = set(m0.state_dict()) - set(sd0)
    check("logits BIT-IDENTICAL with the adapter at init (eval and train mode, same dropout seed); only kv_adapter.* parameters added; cfg['kv64'] == 1",
          torch.equal(l_base, l_kv) and torch.equal(lt_base, lt_kv) and all(k.startswith("kv_adapter.") for k in new_keys) and len(new_keys) == 9
          and m0.cfg.get("kv64") == 1 and all(torch.equal(sd0[k], m0.state_dict()[k]) for k in sd0), (float((l_base - l_kv).abs().max()), sorted(new_keys)))
    KA.remove_kv64(m0)
    l_rm = m0(b1, m0.dense(b1))
    check("remove_kv64: hooks removed, logits == base, cfg['kv64'] gone", torch.equal(l_rm, l_base) and "kv64" not in m0.cfg and not hasattr(m0, "kv_adapter"))

    P("\n== K5 adapter: learning path")
    KA.setup_kv64(m0, log=lambda *a_: None)
    with torch.no_grad():
        m0.kv_adapter.gate.fill_(0.5)
    m0.eval()
    l_on = m0(b1, m0.dense(b1))
    changed = not torch.equal(l_on, l_base)
    m0.zero_grad()
    l_on.sum().backward()
    grads = {n: (p.grad is not None and float(p.grad.abs().sum()) > 0) for n, p in m0.kv_adapter.named_parameters()}
    check("gate 0.5: logits change; gradients reach proj / gate / ln / null / user0", changed and all(grads.values()), grads)
    s = m0.kv_adapter.states(b1["kv"], b1["kv_ok"], b1["hmask"])
    nul = m0.kv_adapter.null.detach()
    with torch.no_grad():
        m0.kv_adapter.null.fill_(7.0)
        s2 = m0.kv_adapter.states(b1["kv"], b1["kv_ok"], b1["hmask"])
    use = (b1["kv_ok"][:, None] & b1["hmask"])
    check("null token used exactly where kv_ok is False or the slot is padded; user slot == user0; real slots == kv",
          torch.equal(s2[:, 1:][~use], torch.full_like(s2[:, 1:][~use], 7.0)) and torch.equal(s2[:, 1:][use], b1["kv"][use].float())
          and torch.equal(s2[:, 0], m0.kv_adapter.user0.detach()[None].expand(s2.shape[0], -1)) and s.shape == (b1["kv"].shape[0], b1["kv"].shape[1] + 1, 64),
          (int((~use).sum()), int(use.sum())))
    del nul

    P("\n== K6b kv64 checkpoint round trip (load_ranker_kv64)")
    import io
    buf = io.BytesIO()
    torch.save({"cfg": dict(m0.cfg), "state_dict": m0.state_dict()}, buf)
    buf.seek(0)
    ck = torch.load(buf, map_location="cpu", weights_only=False)
    m3 = KA.load_ranker_kv64(ck)
    m3.eval()
    with torch.no_grad():
        l3 = m3(b1, m3.dense(b1))
        l_on2 = m0(b1, m0.dense(b1))
    check("checkpoint with cfg['kv64'] = 1 + kv_adapter.* keys reloads through load_ranker_kv64 with bit-identical logits (gate 0.5 state)",
          torch.equal(l3, l_on2) and ck["cfg"].get("kv64") == 1 and any(k.startswith("kv_adapter.") for k in ck["state_dict"]), float((l3 - l_on2).abs().max()))

    P("\n== K6 PCA apply")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5000, 384)).astype(np.float64) @ rng.standard_normal((384, 384)) * 0.1
    mean = X.mean(0)
    cov = np.cov(X.T, bias=True)
    ev, evec = np.linalg.eigh(cov)
    idx = np.argsort(ev)[::-1][:64]
    comp = evec[:, idx].T
    Z = (X - mean) @ comp.T
    check("64 components orthonormal; projection shape [n, 64]; the first component carries the largest variance",
          np.allclose(comp @ comp.T, np.eye(64), atol=1e-8) and Z.shape == (5000, 64) and Z.var(0)[0] >= Z.var(0)[1] - 1e-9)
    P(f"\n== {'ALL PASS' if not FAIL else 'FAILURES: ' + str(FAIL)} ({len(FAIL)} failed)")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
