"""Derive cd_arms6c.py from nn_x/w5/cd_arms6b.py by exact, single-occurrence text replacements (so the diff is small and auditable).
  python make_6c.py   -> cd_arms6c.py, cd_arms6c_vs_cd_arms6b.diff"""
import subprocess
import sys
sys.dont_write_bytecode = True
SRC = '<repo>/nn_x/w5/cd_arms6b.py'
src = open(SRC).read()


def rep(a, b):
    global src
    assert src.count(a) == 1, (src.count(a), a)
    src = src.replace(a, b)


rep('"""cd_arms6b = cd_arms6 + hz2 (',
    '"""cd_arms6c = cd_arms6b + hz2n (nn_x/w12/stale/hz2n, reports/STALE_1115/hz2n.md): the SAME module with ONE optional extension of the column map\n'
    '(lines marked "# 6c"): the encoder-score inputs pre2_fe_z / pre2_fe_s (fe head; also pre2_fe and has:pre2_fe when the model reads them) and pre2_new_z /\n'
    'pre2_new_s (new head; also pre2_new) are swapped too IF their view file exists in <HZCD_DIR>/train/ (pre2_hz_fe_z.npy, pre2_hz_fe_s.npy, pre2_hz_fe.npy,\n'
    'pre2_hz_fe_has.npy; pre2_hz_new_z.npy, pre2_hz_new_s.npy, pre2_hz_new.npy). Those files hold, on the rows the view makes cold (view it_hist_all == 0,\n'
    'stored it_hist_all_luo > 0, not a HZ2_KEEP row), the NO-ID encoder score and the inputs rebuilt from it exactly as the stale-val witness does\n'
    '(solvers/CPU2_stale_replay_pre2s_and_arms.py::build_pre2both for the fe head, ::build_pre2full for the new head); every other row holds the stored value\n'
    '(the within-feed z is re-standardised over the whole feed, as in the witness). They are read exactly like the other swapped columns: same attach_xt\n'
    'transform branch (hz names are chosen so that xt_transform takes the branch of the stored name), same fp16 cast, same HZ2_KEEP semantics (keep rows stay\n'
    'on stored values). If none of these files exists in HZCD_DIR (data/feats_new, nn_x/w12/stale/hz2/feats, feats_L), this module is cd_arms6b: same columns,\n'
    'same tensors, same batches, same log line apart from the tag "6c" (proof: nn_x/w12/stale/hz2n/test_cd_arms6c.py). The text below is cd_arms6b.\n\n'
    'cd_arms6b = cd_arms6 + hz2 (')
rep('KEEP_PATH = os.environ.get("HZ2_KEEP", "")',
    '# 6c: encoder-score inputs, swapped only when <HZ_DIR>/train/<file>.npy exists (hz2n view). xt_transform(hz name) == xt_transform(stored name) by branch:\n'
    '# *_z -> clip +-5; pre2_hz_fe / pre2_hz_fe_s / pre2_hz_new / pre2_hz_new_s -> clip +-10 (as the stored names); pre2_hz_fe_has is a stored 0/1 float file\n'
    '# (clip +-10 is the identity on it), value-identical to attach_xt\'s "has:" branch of the overridden score.\n'
    'ENC_TO_HZ = {"pre2_fe": "pre2_hz_fe", "pre2_fe_z": "pre2_hz_fe_z", "pre2_fe_s": "pre2_hz_fe_s", "has:pre2_fe": "pre2_hz_fe_has",\n'
    '             "pre2_new": "pre2_hz_new", "pre2_new_z": "pre2_hz_new_z", "pre2_new_s": "pre2_hz_new_s"}\n'
    '# 6c: each head is all-or-nothing among the inputs the model reads (a partial head is refused); the has: flag is optional\n'
    'ENC_GROUPS = (("pre2_fe", "pre2_fe_z", "pre2_fe_s"), ("pre2_new", "pre2_new_z", "pre2_new_s"))\n'
    'KEEP_PATH = os.environ.get("HZ2_KEEP", "")')
rep('    cols.sort()  # 6b: xt column order (a no-op without extra files)\n',
    '    enc = [(i, n, ENC_TO_HZ[n]) for i, n in enumerate(xt_names or [])  # 6c\n'
    '           if n in ENC_TO_HZ and os.path.exists(hz_path("train", ENC_TO_HZ[n]))]  # 6c\n'
    '    if enc:  # 6c: refuse a partial head (every score-derived input of a swapped head that the model reads must have its file) and the raw-ID preset\n'
    '        got = {n for _, n, _ in enc}  # 6c\n'
    '        miss = [n for g in ENC_GROUPS if got & set(g) for n in (xt_names or []) if n in g and n not in got]  # 6c\n'
    '        assert not miss, f"--hzcd: partial encoder-score view in {HZ_DIR}/train: no file for {miss}"  # 6c\n'
    '        assert "pre2_id_contrib" not in (xt_names or []), "--hzcd encoder-score view: pre2_id_contrib as a model input is not supported"  # 6c\n'
    '    cols += enc  # 6c\n'
    '    cols.sort()  # 6b: xt column order (a no-op without extra files)\n')
rep('    msg = "; ".join(', '    enc = [c for c in cols if c[1] in ENC_TO_HZ]  # 6c\n'
    '    enc_msg = (f"; ENCODER-SCORE VIEW ON: {len(enc)} of them are encoder-score columns {[c[1] for c in enc]} <- {[c[2] for c in enc]} "  # 6c\n'
    '               f"(no-ID score on rows made cold by the view)" if enc else "")  # 6c\n'
    '    msg = "; ".join(')
rep('f"[6b: {len(cols)} swapped columns {[c[1] for c in cols]}; HZ2_KEEP rows on stored values: {n_keep}] "',
    'f"[6c: {len(cols)} swapped columns {[c[1] for c in cols]}; HZ2_KEEP rows on stored values: {n_keep}{enc_msg}] "')
open('cd_arms6c.py', 'w').write(src)
d = subprocess.run(['diff', SRC, 'cd_arms6c.py'], capture_output=True, text=True).stdout
open('cd_arms6c_vs_cd_arms6b.diff', 'w').write(d)
print(d)
