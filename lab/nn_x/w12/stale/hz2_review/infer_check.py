"""hz2_review: path control of the STUDY'S CPU reader (hold_infer.py: load_mini + attach + CPU fp32 forward) against the TRAINER'S GPU view path.
A finished hz2 run wrote preds/holdout/nn_<v>_holdout_hz.npy = its selected checkpoint scored through cd_arms6b's view on the GPU (bf16). The reader must give
the same scores for the same checkpoint on holdout feeds of days >= Sep 13 (no HZ2_KEEP row there). Output goes to MY directory only (the study's main() would
write into its cache). Label-free here; the comparison (z correlation; NDCG as a path control) is in infer_check_read.py.
Usage: CUDA_VISIBLE_DEVICES="" python infer_check.py <variant>"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/nn_x/w12/stale/hz2')
import hold_infer as HI          # the reader under test
from hold_infer import *         # noqa
v = sys.argv[1]
out_p = ROOT / 'nn_x/w12/stale/hz2_review' / f'chk_hpred_feats_late_{v}.npy'
s = HI.load_mini(); lg('mini split loaded', s.n, s.n_cand)
ck = torch.load(HI.ckpt_path(v), map_location='cpu', weights_only=False)
used = HI.attach(s, ck['cfg']['xt_names'], HERE / 'feats'); lg('hz2 columns used', used)
m = Ranker(**ck['cfg']); m.load_state_dict(ck['state_dict']); m.eval()
out = torch.full((s.n_cand,), float('nan'), dtype=torch.float32)
slot = day_of(np.load(HI.MINI / 'ts.npy')) - DAY0
fsel = np.flatnonzero(slot >= 7)
with torch.inference_mode():
    for fb in s.batches(fsel, 65536, 1024, shuffle=False):
        b = s.gather(fb); s.scatter(out, b, m(b, use_pos=False))
np.save(out_p, out.numpy().astype(np.float32)); lg('saved', out_p)
