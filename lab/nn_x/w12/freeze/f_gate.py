"""Audit check (f): the gate output on record of every candidate file (src/presub_gate.py output stored by the build scripts). CPU, read-only.
Output: out/f_gate.json. Nothing is re-run here: the gate is re-run on freeze day (protocol section 4).
  f1 the gate file exists and holds exactly one verdict line "GATE: ..."                 f2 the verdict is "GATE: PASS  (FAIL 0, WARN 0)", no [FAIL] / [WARN] line
  f3 the member lines of the gate output == the member list of the builder sources      f4 "D val new" of the gate output == final val on record
  f5 the gate file is not older than the submission file it judges (mtime, information; the gate output carries no sha256 of the file)
"""
import sys, json, re, datetime
sys.path.insert(0, '<repo>/nn_x/w12/freeze')
from cands import ROOT, OUT, CANDS, members

ts = lambda p: datetime.datetime.fromtimestamp(p.stat().st_mtime, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
res = {'stamp_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'cands': {}}
for c in CANDS:
    gp = ROOT / c['gate']; r = {'gate_file': c['gate'], 'f1_exists': gp.exists()}
    if gp.exists():
        L = gp.read_text(errors='replace').splitlines()
        v = [l for l in L if l.startswith('GATE:')]; r['verdict_lines'] = v; r['f1_one_verdict'] = len(v) == 1
        r['n_fail_lines'] = sum(l.startswith('[FAIL]') for l in L); r['n_warn_lines'] = sum(l.startswith('[WARN]') for l in L)
        r['f2_pass'] = len(v) == 1 and v[0].strip() == 'GATE: PASS  (FAIL 0, WARN 0)' and r['n_fail_lines'] == 0 and r['n_warn_lines'] == 0
        gm = [re.match(r'\s+(\S+): \d+ train-stage inputs audited, fine-tune n2cpv3s3 audited', l) for l in L]; gm = [x.group(1) for x in gm if x]
        base, pop, w = members(c['id']); r['gate_members'] = len(gm); r['f3_members_equal_spec'] = gm == base + pop
        dv = [re.match(r'D val new (\S+) ref (\S+) delta (\S+) \+- (\S+)', l) for l in L]; dv = [x for x in dv if x]
        r['gate_val_new'] = dv[0].group(1) if dv else None; r['gate_val_ref'] = dv[0].group(2) if dv else None; r['f4_val_equals_record'] = bool(dv) and dv[0].group(1) == c['val_rec']
        r['line_B'] = next((l for l in L if l.startswith('B reproduction')), None); r['line_leak'] = next((l.strip() for l in L if 'LEAK PROBE' in l), None)
        r['line_E'] = next((l for l in L if l.startswith('E predicted')), None); r['line_F'] = next((l for l in L if l.startswith('F new-vs-ref')), None)
        r['gate_mtime_utc'] = ts(gp); r['file_mtime_utc'] = ts(ROOT / c['file']); r['f5_gate_not_older_than_file'] = gp.stat().st_mtime >= (ROOT / c['file']).stat().st_mtime
    res['cands'][c['id']] = r
    print(f"{c['id']:18s} {c['gate']} | {r.get('verdict_lines')} | f2 {r.get('f2_pass')} | members {r.get('gate_members')} == spec {r.get('f3_members_equal_spec')} | D val new {r.get('gate_val_new')} (ref {r.get('gate_val_ref')}) == record {r.get('f4_val_equals_record')}"
          f" | gate written {r.get('gate_mtime_utc')} file {r.get('file_mtime_utc')} f5 {r.get('f5_gate_not_older_than_file')} | {r.get('line_B')}")
json.dump(res, open(OUT / 'f_gate.json', 'w'), indent=1)
print('written', OUT / 'f_gate.json')
