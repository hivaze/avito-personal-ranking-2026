"""LATE-DROPOUT plateau kill gate (PREDECLARED.txt, 04:50:54 UTC + addendum 04:51:13 UTC). Evaluated ONCE, when both arm logs
contain their 'step 60000/98350 ...' eval line. Reads h_LD,s(60000) from nn_x/abl/logs/abl_e10_latedrop_s<s>.log,
h_base,s(60000) from nn_x/w5/logs/w5_e10_base_s<s>.json (history), SE_gate from se_gate.json (base runs only).
KILL iff for BOTH seeds: h_LD - h_base < -2 * SE_gate. Writes gate.json; prints 'GATE KILL' or 'GATE CONTINUE' last.
Exit 0 = decided, 2 = a step-60000 line is missing (not decided)."""
import json
import re
import sys
import time

D = '<repo>/nn_x/tricks/late_dropout'
SEEDS = [951, 952]
STEP = 60000
G = json.load(open(f'{D}/se_gate.json'))
se_gate = G['se_gate']
assert abs(se_gate - 0.00076575) < 5e-9, se_gate  # the constant written into PREDECLARED.txt (addendum)
pat = re.compile(rf'^step {STEP}/98350 loss (\S+) val_ndcg (\S+) holdout_ndcg (\S+) ')
per = {}
for s in SEEDS:
    try:
        hit = [pat.match(line) for line in open(f'<repo>/nn_x/abl/logs/abl_e10_latedrop_s{s}.log')]
    except FileNotFoundError:
        hit = []
    hit = [m for m in hit if m]
    if not hit:
        print(f'seed {s}: no step-{STEP} line yet')
        sys.exit(2)
    m = hit[-1]
    jb = json.load(open(f'<repo>/nn_x/w5/logs/w5_e10_base_s{s}.json'))
    hb = [h for h in jb['history'] if h['step'] == STEP][0]
    d = float(m.group(3)) - hb['holdout_ndcg']
    per[str(s)] = dict(h_ld=float(m.group(3)), h_base=hb['holdout_ndcg'], d=d, below=bool(d < -2 * se_gate),
                       val_ld=float(m.group(2)), val_base=hb['val_ndcg'], loss_ld=float(m.group(1)), loss_base=hb['loss'])
decision = 'KILL' if all(per[str(s)]['below'] for s in SEEDS) else 'CONTINUE'
out = dict(step=STEP, se_gate=se_gate, threshold=-2 * se_gate, per_seed=per, decision=decision,
           evaluated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
json.dump(out, open(f'{D}/gate.json', 'w'), indent=1)
for s in SEEDS:
    v = per[str(s)]
    print(f'seed {s}: step {STEP} holdout LD {v["h_ld"]:.5f} base {v["h_base"]:.5f} d {v["d"]:+.5f} ({v["d"] / se_gate:+.2f} SE_gate)'
          f'{" BELOW -2 SE_gate" if v["below"] else ""} | val LD {v["val_ld"]:.5f} base {v["val_base"]:.5f}')
print(f'GATE {decision}')
