"""Assembles reports/MIX_1115/rulesA.md from report_body.txt (prose with @@PLACEHOLDERS@@) + tables.py + log excerpts. Writes only that one file."""
import subprocess, sys, re
from pathlib import Path
import tables as T
from lib import *
HERE = Path('<repo>/nn_x/w12/mix/rulesA'); OUT = Path('<repo>/reports/MIX_1115/rulesA.md')
def excerpt(fn, start=None, end=None, maxlen=400, grep=None, head=None):
    L = (HERE / fn).read_text().splitlines()
    if start is not None:
        i0 = next(i for i, l in enumerate(L) if start in l); L = L[i0:]
        if end is not None:
            i1 = next((i for i, l in enumerate(L[1:], 1) if end in l), len(L)); L = L[:i1]
    if grep is not None: L = [l for l in L if re.search(grep, l)]
    if head is not None: L = L[:head]
    return '```\n' + '\n'.join(l[:maxlen] for l in L if l.strip()) + '\n```'
names = T.names; elig = T.elig
top = [names[i] for i in np.argsort(-np.where(elig, T.R['C1']['nested'], -9))[:8]]
keys = top + [k for k in ('C(M50)', 'C(Mq)', 'G(M3q)', 'G(M3r)', 'D(M50)', 'I(M50)', 'A', 'B', 'E(CS45)', 'E(CLq)', 'E(Hq)', 'L(CL3q)', 'C(DAY)') if k not in top]
rep = {
 '@@NESTED@@': T.nested_table(keys), '@@VARIANTS@@': T.variant_table(), '@@FIXED@@': T.fixed_table(),
 '@@RULES_CM50@@': T.chosen_rules('C(M50)'), '@@RULES_A@@': T.chosen_rules('A'), '@@RULES_G@@': T.chosen_rules('G(M3r)'),
 '@@REPRO@@': excerpt('logs_repro.txt', maxlen=330),
 '@@PLREAD_C1@@': excerpt('logs_pl_read.txt', start='observed = C1', end='family           par', maxlen=330),
 '@@PLREAD_C3@@': excerpt('logs_pl_read.txt', start='observed = C3', end='family           par', maxlen=330),
 '@@T2@@': excerpt('logs_t2.txt', head=10, maxlen=330),
 '@@CRIT@@': excerpt('logs_criteria.txt', head=13, maxlen=330) + '\n' + excerpt('logs_criteria.txt', grep='FALLBACK', maxlen=400),
 '@@XMODEL@@': excerpt('logs_xmodel.txt', grep='=== SRC|^C\\(M50\\)|^F\\(Hq\\)|^G\\(M3q\\)|^K\\(Mq,Hq\\)|^A  ', maxlen=420),
 '@@HYP@@': excerpt('logs_hyp.txt', maxlen=330),
 '@@ALL8@@': excerpt('logs_all8.txt', maxlen=330),
 '@@SEARCH8@@': excerpt('logs_search8.txt', grep='===|^family|^C\\(M50\\)|^G\\(M3q\\)|^F\\(Hq\\)|^K\\(Mq,Hq\\)|^I\\(M50\\)|^A  |VARIANT B-3', maxlen=260),
 '@@ROBUST_VAL@@': excerpt('logs_robust.txt', start='pmix weights', end='=== LABEL-FREE gate coverage', grep='===|^rule|^fixed MRULE|^C\\(M50\\)|^G\\(M3q\\)|^F\\(Hq\\)|^K\\(Mq,Hq\\)|minus fixed', maxlen=420),
 '@@ROBUST_TEST@@': excerpt('logs_robust.txt', start='=== LABEL-FREE gate coverage', maxlen=260),
 '@@DOSE@@': excerpt('logs_dose.txt', maxlen=330), '@@CDOSE@@': excerpt('logs_cdose.txt', grep='===|^c |^0\\.[0567]|^1\\.0', maxlen=200),
 '@@CROSS@@': excerpt('logs_crossfd.txt', maxlen=330) + '\n' + excerpt('logs_fd_vs_twin.txt', maxlen=330),
 '@@LEAK@@': excerpt('logs_leak_L1.txt', grep='L1|variant|per-family', maxlen=330) + '\n' + excerpt('logs_leak_L1b.txt', grep='L1|variant|per-family', maxlen=330) + '\n' + excerpt('logs_leak_L2.txt', maxlen=420),
 '@@UNIT@@': excerpt('logs_test_search.txt', maxlen=330),
 '@@DECL@@': '```\n' + (HERE / 'DECLARATION.txt').read_text() + '\n```',
}
body = (HERE / 'report_body.txt').read_text()
for k, v in rep.items():
    assert k in body, k; body = body.replace(k, v)
assert '@@' not in body, re.findall('@@[A-Z0-9_]+@@', body)
OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(body); print('written', OUT, len(body), 'chars', body.count('\n'), 'lines')
