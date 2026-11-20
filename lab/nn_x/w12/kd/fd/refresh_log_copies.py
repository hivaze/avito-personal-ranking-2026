"""Refresh the nn/logs copies that src/presub_gate.py audits (found by the freeze-protocol review 2026-11-15: fd_finish.sh copied logs with cp --update=none while runs were still
training, so 21 + 24 copies are byte PREFIXES of the complete logs). For every member of the given build_r4.py keys: if the copy is a strict byte prefix of the source log, replace it by
the complete source (write-new + rename); if it is identical, leave it; if it differs in any other way, report and do NOT touch it.
  .venv/bin/python nn_x/w12/kd/fd/refresh_log_copies.py <KEY>[,<KEY>..] [apply]      (without 'apply' it only reports)"""
import sys, os
from pathlib import Path
R = Path('<repo>'); src_txt = (R / 'nn_x/w12/kd/build_r4.py').read_text()
head = src_txt[src_txt.index('S8 = range'):src_txt.index('fv, Mv, ov = feed_meta')]; ns = {}; exec('from pathlib import Path\n' + head, ns)
apply = 'apply' in sys.argv[2:]; mem = []
for k in sys.argv[1].split(','):
    b, p, _ = ns['spec'](k.split('@m')[0]); mem += [v for v in b + p if v not in mem]
n = dict(same=0, prefix=0, differ=0, nosrc=0, nocopy=0)
for v in mem:
    H = R / ('nn_x/w5' if v.startswith('w5_') else 'nn_x/abl')
    for l in (f'{v}.log', f'{v}_n2cpv3s3_ft.log'):
        s, c = H / 'logs' / l, R / 'nn/logs' / l
        if not s.exists(): n['nosrc'] += 1; print('no source:', l); continue
        if not c.exists(): n['nocopy'] += 1; continue
        a, b_ = s.read_bytes(), c.read_bytes()
        if a == b_: n['same'] += 1
        elif a.startswith(b_):
            n['prefix'] += 1
            if apply:
                t = c.with_suffix(c.suffix + '.new'); t.write_bytes(a); os.replace(t, c)
        else: n['differ'] += 1; print('DIFFERS (not touched):', l, len(a), len(b_))
print(sys.argv[1], '| members', len(mem), '|', n, '| applied' if apply else '| report only')
