"""STALE-VAL on the CPU (task B of KEY = witness): CPU fp32 copy of nn_x/w5/stale_pred.py (= nn_x/abl/stale_pred.py, byte-identical).

The originals are untouched; this file re-uses their scenario definition verbatim and replaces only the device / numerics:
  scenario 'both_pre2both' (val inputs only, private-like coverage): it_hist_{all,et1..et4}_luo and seq_cv_{invd20,norm,pop} from
  nn_x/r6/stale_cf/both_<name>.npy; pre2_fe / pre2_fe_z / pre2_fe_s from solvers/out/stale/cf/pre2both_<name>.npy (has:pre2_fe follows
  the overridden pre2_fe). Every other input, the val histories and the model are unchanged.
  Inference = SplitData('val') of nn_x/w5/data.py on the CPU + Ranker from the checkpoint cfg, the batches of common.predict (65536 / 1024,
  no shuffle) cut into anchored sub-batches with identical padding (wt_common.anchored_batches), fp32, no autocast (the originals: bf16 autocast on a GPU). The val split carries no domain mix (as train.py's val predict).
Outputs (my directory only): cache/staleval/nn_<member>_staleval.npy  (same content as the GPU file nn_x/<fam>/preds/stale/nn_<v>_staleval.npy)
                             cache/staleval/nn_<member>_valbase.npy   (mode base: no overrides = CPU twin of nn_<v>_trainonly_val.npy)
A member is any name of nn_x/w5/ckpt, nn_x/abl/ckpt, nn_x/r6/ckpt, nn/ckpt (prefix w5_ / abl_ / r6_ / other), '@last' included.

  cd <repo> && CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python nn_x/w12/stale/witness/stale_val_cpu.py \
      <stale|base|both> <member,member,... | group name>
"""
from wt_common import *

_CF_DUR = ROOT / 'nn_x/r6/stale_cf'
CF = ROOT / 'solvers/out/stale/cf'
OVR = {**{n: _CF_DUR / f'both_{n}.npy' for n in ('it_hist_all_luo', 'it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo', 'it_hist_et4_luo',
                                                 'seq_cv_invd20', 'seq_cv_norm', 'seq_cv_pop')},
       **{n: CF / f'pre2both_{n}.npy' for n in ('pre2_fe', 'pre2_fe_z', 'pre2_fe_s')}}


def sv_path(v, mode='stale'):
    return SV / f"nn_{v}_{'staleval' if mode == 'stale' else 'valbase'}.npy"


class ValSplit:
    """SplitData('val') on the CPU with the stale-val override switch of stale_pred.py (module-level feat_path patch)."""

    def __init__(self):
        import torch
        sys.path.insert(0, str(ROOT / 'nn_x/w5'))
        import data as Dm
        for p in OVR.values():
            assert p.exists(), f'missing counterfactual file {p}'
        self.torch = torch; self.Dm = Dm; self.active = False
        _fp = Dm.feat_path

        def feat_path(split, name):
            if self.active and split == 'val' and name in OVR:
                return OVR[name]
            return _fp(split, name)

        Dm.feat_path = feat_path          # SplitData.attach_xt resolves feature files through this module-level name
        self.va = Dm.SplitData('val', torch.device('cpu'), verbose=False)
        self.key = None
        self.bl = None if os.environ.get('WT_REFBATCH') else anchored_batches(self.va)   # same padding as the reference batches, faster on CPU

    def attach(self, names, stale):
        key = (tuple(names), bool(stale))
        if key != self.key:
            self.active = bool(stale)
            self.va.xt = None
            self.va.attach_xt(list(names))
            self.active = False
            self.key = key
        return sorted(n for n in OVR if n in names or f'has:{n}' in names) if stale else []

    def score(self, m):
        torch = self.torch; va = self.va
        out = torch.zeros(va.n_cand, dtype=torch.float32)
        with torch.inference_mode():
            for fb in (self.bl if self.bl is not None else va.batches(np.arange(va.n), 65536, 1024, shuffle=False)):
                b = va.gather(fb)
                va.scatter(out, b, m(b, use_pos=False))
        return out.numpy().astype(np.float32)


def score_member(vs, v, modes=('stale',), log=print):
    """Val predictions of one member under the requested modes ('stale', 'base'); cached. Returns seconds per mode scored."""
    from lagged_score import load_model
    todo = [md for md in modes if not sv_path(v, md).exists()]
    if not todo:
        return {}
    m, names = load_model(v)
    sec = {}
    for md in todo:
        t1 = time.time()
        used = vs.attach(names, md == 'stale')
        save_atomic(sv_path(v, md), vs.score(m))
        sec[md] = time.time() - t1
        log(f'{now()} val-{md} {v}: {sec[md]:.0f}s (overridden inputs used: {used})')
    return sec


def main():
    import torch
    torch.set_num_threads(int(NT))
    modes = ('stale', 'base') if sys.argv[1] == 'both' else (sys.argv[1],)
    mem = []
    for tok in sys.argv[2].split(','):
        mem += G[tok] if tok in G else [tok]
    vs = ValSplit()
    for v in mem:
        if not ckpt_path(v).exists():
            print(f'{v}: no checkpoint, skipped', flush=True); continue
        score_member(vs, v, modes)


if __name__ == '__main__':
    main()
