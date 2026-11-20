"""premortem make_report: assembles reports/STALE_1115/premortem.md from report_head.txt (narrative with {{KEY}} placeholders) plus the text blocks blk_<KEY>.txt and the tables in out/*.md.
Usage: make_report.py"""
import re
from pathlib import Path
H = Path('<repo>/nn_x/w12/stale/premortem'); O = H / 'out'
def sect(fn, start, end=None):
    """Lines of out/<fn> from the heading that starts with `start` up to (not including) the heading that starts with `end`."""
    p = O / fn
    if not p.exists(): return f'(pending: {fn} not written yet)'
    L = p.read_text().splitlines(); i0 = next((i for i, l in enumerate(L) if l.startswith(start)), None)
    if i0 is None: return f'(section {start!r} not found in {fn})'
    i1 = len(L)
    if end is not None:
        i1 = next((i for i in range(i0 + 1, len(L)) if L[i].startswith(end)), len(L))
    body = L[i0 + 1:i1]
    return '\n'.join(body).strip('\n')
def rows(fn, start, pats, end=None, head=2):
    """Table header (first `head` table lines after `start`) + the rows whose first cell matches one of pats."""
    s = sect(fn, start, end).splitlines(); tl = [l for l in s if l.startswith('|')]
    keep = tl[:head] + [l for l in tl[head:] if any(re.match(r'\|\s*' + p + r'\s*\|', l) for p in pats)]
    return '\n'.join(keep)
K = {
    'T1A': sect('t1_table.md', '## T1a', '## T1b'),
    'T1B': sect('t1_table.md', '## T1b', '## T1c'),
    'T1D': sect('t1_table.md', '## T1d', '## T1e'),
    'T1E': sect('t1_table.md', '## T1e', '## T1f'),
    'T1F': sect('t1_table.md', '## T1f'),
    'T2A': sect('t2_forecast.md', '## T2a', '## T2b'),
    'T2B': sect('t2_forecast.md', '## T2b', '## T2c'),
    'T2C': sect('t2_forecast.md', '## T2c', '## T2d'),
    'T2D': sect('t2_forecast.md', '## T2d', '## T2e'),
    'T2E': sect('t2_forecast.md', '## T2e', '## T2f'),
    'T2F': sect('t2_forecast.md', '## T2f', '## T2g'),
    'T2G': sect('t2_forecast.md', '## T2g', '## T2h'),
    'T2H_A': sect('t2_forecast.md', '### W8P20 against W8', '### W16 against W8'),
    'T2H_B': sect('t2_forecast.md', '### W16P40x8 against W8P20', '### S16 against W8'),
    'T2H_C': sect('t2_forecast.md', '### W8TO against W8', '### RULEM2040 against W16P20x8'),
    'T3A': sect('t3_weight.md', '## T3a', '## T3b'),
    'T3B': sect('t3_weight.md', '## T3b', '## T3c'),
    'T3C': sect('t3_weight.md', '## T3c'),
    'T4WFT': sect('t4_wft.md', '## T4a'),
    'T4FOLD': sect('t4_fold.md', '## T4b'),
    'T4SIZE': sect('t4_size.md', '## T4d'),
    'T5': (O / 't5_misc.md').read_text().strip() if (O / 't5_misc.md').exists() else '(pending)',
    'T6': sect('t6_lag.md', '## T6'),
    'T7': (O / 't7_robust.md').read_text().strip() if (O / 't7_robust.md').exists() else '(pending)',
    'T8': sect('t8_sameview.md', '## T8'),
    'T10': (O / 't10_branch.md').read_text().strip() if (O / 't10_branch.md').exists() else '(pending)',
    'T9': (O / 't9_softft.md').read_text().strip() if (O / 't9_softft.md').exists() else '(pending)',
}
def demote(v):
    out = []
    for l in v.splitlines():
        if l.startswith('### ') or l.startswith('## '): l = '**' + l.lstrip('#').strip() + '**'
        out.append(l)
    return '\n'.join(out)
K = {k: demote(v) for k, v in K.items()}
txt = (H / 'report_head.txt').read_text()
for f in sorted(H.glob('blk_*.txt')):
    K[f.stem[4:]] = f.read_text().strip('\n')
for k, v in K.items(): txt = txt.replace('{{' + k + '}}', v)
left = re.findall(r'\{\{[A-Z0-9_]+\}\}', txt)
assert not left, left
Path('<repo>/reports/STALE_1115/premortem.md').write_text(txt)
print('written', len(txt.splitlines()), 'lines')
