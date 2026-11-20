"""Candidate table of reports/FREEZE_PROTOCOL_1207.md (version 2) and helpers shared by the audit scripts.
READ-ONLY with respect to preds/, subs/, data/ and every script outside nn_x/w12/freeze/. CPU only.

Member lists are NOT typed here: they are taken from the builder sources by executing the definition block of
  nn_x/w12/kd/build_r4.py  (lines 'S8 = range' .. before 'fv, Mv, ov = feed_meta'; the same extraction as nn_x/w12/kd/fd/fd_finish.sh)
  nn_x/w12/kd/build_r3.py  (lines 'W = lambda' .. 'P1n = ...'; the builder of W16 and W8P20)
The numbers 'val_rec', 'public', 'sha_rec' are copied from records (source in 'src'); nothing here is measured.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
from pathlib import Path

ROOT = Path('<repo>')
PREDS = ROOT / 'preds'
SUBS = ROOT / 'subs'
OUT = ROOT / 'nn_x/w12/freeze/out'
LEDGER = 'nn_x/w5/logs/E_kd_predeclared.txt'
DOC04 = 'docs/04_SUBMISSIONS.md section 5a'
SCORED_REF = 'subs/cand_S16kd_r2.parquet'          # scored file outside the candidate table (public 0.3940782, CURRENT_STATUS section 1)
SAMPLE = 'data/raw/sample_submission.parquet'

# id, file, tag of the stored branch files, read-out key, expectation "trained on all data", records
# A new row needs: id, file, tag, key, builder ('build_r4' or 'build_r3'), fulldata, val_rec, public, gate, sha_rec, src; optional 'gate_ref' = tag of the
# reference of its gate output on record when that is not the default of gate_rerun.sh (full-data rows: W8P20kd). b_readout.sh, run_all.sh and gate_rerun.sh
# read this list; no other script holds a list of candidates.
CANDS = [
    dict(id='FD16P20', file='subs/cand_FD16P20kd_r4.parquet', tag='FD16P20kd', key='FD16P20', builder='build_r4', fulldata=True,
         val_rec='0.40271', public='0.394892', gate='nn_x/w12/kd/out/gate_FD16P20kd.txt',
         sha_rec='14adc19ac794e3977cc0275a0c15d2c76e55c5dcdeec911dc47a8193128626a8', src=f'{LEDGER} block 2026-11-15T12:53:32Z; public block 14:34:22Z'),
    dict(id='FD8P20', file='subs/cand_FD8P20kd_r4.parquet', tag='FD8P20kd', key='FD8P20', builder='build_r4', fulldata=True,
         val_rec='0.40240', public='unscored', gate='nn_x/w12/kd/out/gate_FD8P20kd.txt',
         sha_rec='e5be275a94ff2f7801ecce19ebb1d9ae5011538f7fb8eb09e0b7d5af08869135', src=f'{LEDGER} block 2026-11-15T11:19:36Z'),
    dict(id='FD16P20vf', file='subs/cand_FD16P20vfkd_r4.parquet', tag='FD16P20vfkd', key='FD16P20vf', builder='build_r4', fulldata=True,
         val_rec='0.40286', public='unscored', gate='nn_x/w12/kd/out/gate_FD16P20vfkd.txt',
         sha_rec='4a826d6c03717b8bacd729e52024d168e5bee6078f8527894d37f932965bfbbb', src=f'{LEDGER} block 2026-11-15T16:46:18Z; sha256 {DOC04}'),
    dict(id='FD16P20@m0.2_0.4', file='subs/cand_FD16P20_m02_04kd_r4.parquet', tag='FD16P20_m02_04kd', key='FD16P20@m0.2_0.4', builder='build_r4', fulldata=True,
         val_rec='0.40298', public='0.3947465', gate='nn_x/w12/kd/out/gate_FD16P20_m02_04kd.txt',
         sha_rec='a0ebcaf4ad9946cfa92dec753dcc1d0ee612e6739663aab17fc59143d8d0ce8a', src=f'{LEDGER} block 2026-11-15T12:53:32Z; public block 14:34:22Z'),
    dict(id='W8P20', file='subs/cand_W8P20kd_r3.parquet', tag='W8P20kd', key='W8P20', builder='build_r3', fulldata=False,
         val_rec='0.40214', public='0.3950094', gate='nn_x/w12/kd/out/gate_W8P20kd.txt',
         sha_rec='406e5705cef75d264118eeb20102ab963cedf6d00b0b7c6be7c2f1b1361a4d78', src=f'{LEDGER} block 2026-11-14T21:24:41Z; public block 2026-11-15T02:15:30Z'),
    dict(id='W16', file='subs/cand_W16kd_r3.parquet', tag='W16kd', key='W16', builder='build_r3', fulldata=False,
         val_rec='0.40158', public='0.39448337', gate='nn_x/w12/kd/out/gate_W16kd.txt',
         sha_rec='34579efeaea55322a2d2d8a077a4c24d0752bb04a88194abdbf3fd75f3c3a79e', src=f'{LEDGER} block 2026-11-14T21:30:25Z; public block 2026-11-15T02:15:30Z'),
    dict(id='W8', file='subs/cand_W8kd_r2.parquet', tag='W8kd', key='W8w', builder='build_r4', fulldata=False,
         val_rec='0.40165', public='0.394486', gate='nn_x/w12/kd/out/gate_W8kd.txt',
         sha_rec='a00d31e618da48ccb694bce58375f7a236e9cb3685a3177ce4bb81c25b40744d', src=f'{DOC04}; reports/CURRENT_STATUS.md Log 2026-11-14 11:09Z'),
]
BY_ID = {c['id']: c for c in CANDS}
SUFFIXES = ('n2cpv3s3_val_oof', 'n2cpv3s3_test', 'trainonly_val', 'trainonly_test')   # the four prediction files of a member (build_r4.py SUF)


def _r4_namespace():
    src = (ROOT / 'nn_x/w12/kd/build_r4.py').read_text()
    head = src[src.index('S8 = range'):src.index('fv, Mv, ov = feed_meta')]
    ns = {}
    exec('from pathlib import Path\n' + head, ns)
    return ns


def _r3_namespace():
    src = (ROOT / 'nn_x/w12/kd/build_r3.py').read_text()
    head = src[src.index('W = lambda seeds'):src.index('fv, Mv, ov = feed_meta')]
    ns = {}
    exec(head, ns)
    return ns


def members(cid):
    """(base members, popularity-light members, weight of the popularity-light pool or (w_short, w_long)) from the builder sources."""
    r4, r3 = _r4_namespace(), _r3_namespace()
    if cid == 'W8P20':                      # built by build_r3.py: 0.8 z(W8 pool) + 0.2 z(pop1kd pool)
        base, pop, w = list(r3['W8n']), list(r3['P1n']), 0.2
        assert base == r4['wide'](r4['S4']) and pop == r4['P1d'], 'build_r3 / build_r4 member blocks disagree for W8P20'
    elif cid == 'W16':                      # built by build_r3.py: 16 wide students
        base, pop, w = list(r3['W16n']), [], 0.0
        assert base == r4['wide'](r4['S8']), 'build_r3 / build_r4 member blocks disagree for W16'
    elif cid == 'W8':                       # built by build_w.py; the same member list is build_r4 key W8w and build_r3 W8n
        base, pop, w = r4['spec']('W8w')
        assert list(base) == list(r3['W8n']), 'build_r3 / build_r4 member blocks disagree for W8'
    else:
        key = BY_ID[cid]['key']
        kk, mr = (key.split('@m') + [None])[:2]
        base, pop, w = r4['spec'](kk)
        if mr is not None:
            w = (float(mr.split('_')[0]), float(mr.split('_')[1]))
    return list(base), list(pop), w


def harness(member):
    """(code dir of the member's training harness)."""
    if member.startswith('w5_'):
        return ROOT / 'nn_x/w5'
    if member.startswith('abl_'):
        return ROOT / 'nn_x/abl'
    raise ValueError(member)
