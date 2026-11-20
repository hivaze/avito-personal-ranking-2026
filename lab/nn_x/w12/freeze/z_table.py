"""Collects out/a_files.json, out/b_branches.json, out/d_logs.json, out/f_gate.json and the builder read-outs into one table:
out/AUDIT_TABLE.txt (pass / fail per check and file) and out/AUDIT_TABLE.json. CPU, reads only files of nn_x/w12/freeze/out/."""
import sys, json, re, datetime
sys.path.insert(0, '<repo>/nn_x/w12/freeze')
from cands import OUT, CANDS

A = json.load(open(OUT / 'a_files.json')); B = json.load(open(OUT / 'b_branches.json')); D = json.load(open(OUT / 'd_logs.json')); F = json.load(open(OUT / 'f_gate.json'))
r4 = (OUT / 'readout_build_r4.txt').read_text().splitlines(); r3 = (OUT / 'readout_build_r3.txt').read_text().splitlines()
guard = (OUT / 'readout_guard.txt').read_text()
ro = {}
for l in r4:
    m = re.match(r'(\S+): (\d+) base \+ (\d+) pop-light members at w \S+ \| final val (\S+) ', l)
    if m: ro[m.group(1)] = (m.group(4), 'build_r4.py key ' + m.group(1))
    m = re.match(r'references \(final val\): W8 (\S+) \| W16 (\S+) \| W8P20 (\S+)', l)
    if m: ro['ref:W8'], ro['ref:W16'], ro['ref:W8P20'] = m.group(1), m.group(2), m.group(3)
for l in r3:
    m = re.match(r'(W16|W8P20)\s+final val (\S+) ', l)
    if m: ro['r3:' + m.group(1)] = m.group(2)
    m = re.match(r'W8 final val (\S+) ', l)
    if m: ro['r3:W8'] = m.group(1)
pf = lambda x: 'pass' if x else 'FAIL'
rows = []; out = {}
for c in CANDS:
    i = c['id']; a = A['files'][i]; b = B['cands'][i]; d = D['cands'][i]; f = F['cands'][i]
    if c['builder'] == 'build_r4' and i != 'W8':
        rv = ro.get(c['key'], (None, ''))[0]; how = f'build_r4.py {c["key"]}'
    elif i == 'W8':
        rv = ro.get('W8w', (None, ''))[0]; how = 'build_r4.py W8w (+ references line ' + str(ro.get('ref:W8')) + ')'
    else:
        rv = ro.get('r3:' + i); how = f'build_r3.py {i} (build_r4.py has no key; its references line reads {ro.get("ref:" + i)})'
        if ro.get('ref:' + i) != c['val_rec']: rv = None
    chk = {
        'a': a['a1_exists'] and a['a2_sha_equals_record'] and a['a3_schema_equals_scored'] and a['a3_schema_equals_sample'] and a['a4_rows_equal'] and a['a5_no_null_nan_inf']
             and a['a6_keys_equal_sample'] and a['a6_keys_equal_scored'] and a['a6_no_dup_key'] and a['a7_every_feed_once_per_candidate'] and a['a8_scores_are_ranks_0_to_M_minus_1'],
        'b': b['b1_branch_files_ok'] and b['b2_final_val_equals_record'] and b['b3_file_equals_stored_branches'] and rv == c['val_rec'] and 'GUARD: unchanged' in guard,
        'c': b['c1_member_files_ok'] and b['c2_members_reproduce_stored_branches'] and not b['member_files_differ_from_harness_copy'],
        'd': d['d_pass'],
        'f': f['f1_exists'] and f['f1_one_verdict'] and f['f2_pass'] and f['f3_members_equal_spec'] and f['f4_val_equals_record'] and f['f5_gate_not_older_than_file'],
    }
    out[i] = dict(checks={k: pf(v) for k, v in chk.items()}, sha256=a['sha256'], rows=a['rows'], feeds=a['feeds_in_file'], n_base=len(b['members_base']), n_pop=len(b['members_pop']),
                  final_val_stored=b['final_val_from_stored_5d'], final_val_members=b['final_val_from_members_5d'], final_val_readout=rv, readout=how, val_rec=c['val_rec'],
                  member_files=f"{b['member_files_expected'] - len(b['member_files_missing'])}/{b['member_files_expected']}", trained=d['expected'], n_flag=d['n_flag'], train_feeds=d['train_feeds'],
                  gate=f['verdict_lines'][0] if f.get('verdict_lines') else None, gate_file=c['gate'])
    rows.append(f"{i:18s} | (a) {pf(chk['a'])} | (b) {pf(chk['b'])} | (c) {pf(chk['c'])} | (d) {pf(chk['d'])} | (f) {pf(chk['f'])} | val: record {c['val_rec']} stored {b['final_val_from_stored_5d']} members {b['final_val_from_members_5d']} read-out {rv} [{how}]"
                f" | members {len(b['members_base'])}+{len(b['members_pop'])}, files {out[i]['member_files']} | {d['expected']}: flag in {d['n_flag']} of {d['n_members']} logs, train feeds {d['train_feeds']} | {out[i]['gate']}")
E = B['e_test_corr']
txt = [f'AUDIT TABLE  written {datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}  (inputs: a_files {A["stamp_utc"]}, b_branches {B["stamp_utc"]}, d_logs {D["stamp_utc"]}, f_gate {F["stamp_utc"]})',
       'check (e) has no pass / fail: it is a measurement.'] + rows + ['(e) test-score correlation, stored branch files, final = 0.5 z(FT) + 0.5 z(TO):']
for k in ('FD16P20|W8P20', 'FD16P20|FD8P20'):
    v = E[k]; txt.append(f"   {k}: pooled Pearson {v['pooled_pearson']:.4f} | feed-mean Pearson {v['feed_mean_pearson']:.4f} | feed-mean Spearman {v['feed_mean_spearman']:.4f} | same top-1 item in {100 * v['same_top1_share']:.1f}% of test feeds")
txt.append('   ' + guard.strip().splitlines()[-1] + ' (directory listings of subs/ and preds/ens_* before and after the builder read-outs)')
(OUT / 'AUDIT_TABLE.txt').write_text('\n'.join(txt) + '\n'); json.dump(out, open(OUT / 'AUDIT_TABLE.json', 'w'), indent=1)
print('\n'.join(txt))
sys.exit(0 if all(v == 'pass' for o in out.values() for v in o['checks'].values()) else 1)
