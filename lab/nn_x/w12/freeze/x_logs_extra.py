"""Side checks on the member logs (protocol section 5.3 c and finding 2 of section 5.5). CPU, read-only.
Outputs: out/x_gatecopy_ft.txt (are the lines that src/presub_gate.py audits in the nn/logs copies equal to those of the complete logs?)
         out/x_member_width.txt (width, teacher, teacher weight, seeds, epochs and input view of every member, from the args line of its training log;
                                 counts of the prediction files of check c)."""
import sys, json, re
sys.path.insert(0, '<repo>/nn_x/w12/freeze')
from cands import ROOT, OUT, CANDS, members, harness

ms = []
for c in CANDS:
    b, p, w = members(c['id'])
    ms += [m for m in b + p if m not in ms]

# ---- 1. the fine-tune feature line and the args line of the nn/logs copies
L = []; seen = {}
for m in ms:
    src = (harness(m) / 'logs' / f'{m}_n2cpv3s3_ft.log').read_text(errors='replace').splitlines(); cp = (ROOT / 'nn/logs' / f'{m}_n2cpv3s3_ft.log').read_text(errors='replace').splitlines()
    fs = [l for l in src if l.startswith('extra features')]; fc = [l for l in cp if l.startswith('extra features')]
    seen[m] = (len(fs) >= 1 and len(fc) >= 1 and fs[0] == fc[0], src[0] == cp[0])
L.append(f'members {len(seen)} | fine-tune feature line of the nn/logs copy equals the harness log: {sum(v[0] for v in seen.values())} | args line equal: {sum(v[1] for v in seen.values())}')
L.append(f'not equal: {[m for m, v in seen.items() if not (v[0] and v[1])]}')
d = json.load(open(OUT / 'd_logs.json'))['members']
fd = {m: r for m, r in d.items() if '_fd10_' in m}; ho = {m: r for m, r in d.items() if '_e10_' in m}
L.append(f"full-data members {len(fd)} | training-log copy identical {sum(r['gatecopy_train'] == 'identical' for r in fd.values())} prefix {sum(r['gatecopy_train'].startswith('prefix') for r in fd.values())}"
         f" | fine-tune-log copy identical {sum(r['gatecopy_ft'] == 'identical' for r in fd.values())} prefix {sum(r['gatecopy_ft'].startswith('prefix') for r in fd.values())}"
         f" | training feature line of the copy equals the complete log {sum(bool(r.get('gatecopy_feature_line_equal')) for r in fd.values())}")
L.append(f"held-out members {len(ho)} | training-log copy identical {sum(r['gatecopy_train'] == 'identical' for r in ho.values())} | fine-tune-log copy identical {sum(r['gatecopy_ft'] == 'identical' for r in ho.values())}")
L.append(f"kept step == total steps (full-data): {sum(r['kept_step_last'] == r['total_steps'] for r in fd.values())} of {len(fd)}")
(OUT / 'x_gatecopy_ft.txt').write_text('\n'.join(L) + '\n'); print('\n'.join(L))

# ---- 2. width, teacher, view of every member
W = []
b = json.load(open(OUT / 'b_branches.json'))
W.append(f"rows {b['rows']} feeds {b['feeds']} | member prediction files {len(b['member_files'])}, all exist {all(v['exists'] for v in b['member_files'].values())},"
         f" identical to harness copy {sum(v.get('identical_to_harness_copy') is True for v in b['member_files'].values())}, harness copy exists {sum(v.get('harness_copy_exists') is True for v in b['member_files'].values())}")
fam = {}
for m in ms:
    a = (harness(m) / 'logs' / f'{m}.log').read_text(errors='replace').splitlines()[0]
    g = lambda k: (re.search(rf"'{k}': ([^,}}]+)", a) or [None, None])[1]
    fam.setdefault(m.rsplit('_s', 1)[0], []).append((g('d'), g('kd_w'), re.search(r'teacher_\w+', a).group(0) if 'teacher_' in a else None, g('seed'), g('epochs'), 'xp_imp_c50' in a, "'xt_exclude': 'seq_cv_invd20" in a,
                                                    d[m]['n_train_inputs']))
for f, vs in fam.items():
    W.append(f"{f}: n {len(vs)} | d {sorted(set(v[0] for v in vs))} | kd_w {sorted(set(v[1] for v in vs))} | teacher {sorted(set(v[2] for v in vs))} | seeds {sorted(v[3] for v in vs)} | epochs {sorted(set(v[4] for v in vs))}"
             f" | xp_imp_c50 added {sorted(set(v[5] for v in vs))} | 8 popularity inputs excluded {sorted(set(v[6] for v in vs))} | train-stage inputs {sorted(set(v[7] for v in vs))}")
(OUT / 'x_member_width.txt').write_text('\n'.join(W) + '\n'); print('\n'.join(W))
