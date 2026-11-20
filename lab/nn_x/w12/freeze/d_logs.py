"""Audit check (d): what each member was trained on, read from its training log. CPU, read-only. Output: out/d_logs.json.
For every member of every candidate row (member lists from the builder sources, cands.members):
  training log  <harness>/logs/<member>.log   (harness = nn_x/w5 for w5_*, nn_x/abl for abl_*)
  d1 FULL-DATA members (FD rows): the args line holds 'holdout_in_train': True, the line "FULLDATA --holdout_in_train" is present,
     "training feeds 3121208 -> 3151208", "train feeds 3151208", the cooldown pool holds the 30000 holdout feeds, the kept step is the last step,
     and <member>_fulldata.json says holdout_in_train true.
  d2 HELD-OUT members (W8P20 / W16 / W8 rows): no 'holdout_in_train': True in the args line, no FULLDATA line, "train feeds 3121208",
     "30000 random-like train feeds excluded from training", no <member>_fulldata.json.
  d3 the run finished ("saved test preds" in the log) and the fine-tune log <member>_n2cpv3s3_ft.log exists with its final line.
  d4 (information) the copies in nn/logs/ that src/presub_gate.py audits: identical to the harness log, a byte prefix of it, or different.
"""
import sys, json, re, datetime
sys.path.insert(0, '<repo>/nn_x/w12/freeze')
from cands import ROOT, OUT, CANDS, members, harness

N_HELD, N_FULL = 3121208, 3151208


def rd(p):
    return p.read_bytes() if p.exists() else None


def one(m):
    H = harness(m); lp = H / 'logs' / f'{m}.log'; r = {'log': str(lp.relative_to(ROOT)), 'log_exists': lp.exists()}
    if not lp.exists():
        return r
    raw = lp.read_bytes(); txt = raw.decode('utf-8', 'replace'); lines = txt.splitlines()
    args = lines[0] if lines else ''
    r['args_variant_ok'] = f"'variant': '{m}'" in args
    r['args_holdout_in_train_true'] = "'holdout_in_train': True" in args
    r['fulldata_flag_line'] = any(l.startswith('FULLDATA --holdout_in_train') for l in lines)
    r['n_fulldata_lines'] = sum(l.startswith('FULLDATA') for l in lines)
    mt = [re.match(r'train feeds (\d+), steps/epoch (\d+), total steps (\d+)', l) for l in lines]; mt = [x for x in mt if x]
    r['train_feeds'] = int(mt[0].group(1)) if mt else None; r['steps_per_epoch'] = int(mt[0].group(2)) if mt else None; r['total_steps'] = int(mt[0].group(3)) if mt else None
    r['excluded_line'] = any(l.startswith('holdout: 30000 random-like train feeds excluded from training') for l in lines)
    r['put_back_line'] = any('put back: training feeds 3121208 -> 3151208' in l for l in lines)
    mp = [re.search(r'exploration pool (\d+) feeds', l) for l in lines]; mp = [x for x in mp if x]
    r['exploration_pool'] = int(mp[-1].group(1)) if mp else None
    r['pool_holds_holdout_line'] = any('exploration pool 164118 feeds, of which 30000 holdout feeds (IN the pool)' in l for l in lines)
    mk = [re.search(r'kept step (\d+) = last', l) for l in lines]; mk = [x for x in mk if x]
    r['kept_step_last'] = int(mk[-1].group(1)) if mk else None
    r['finished'] = any(l.startswith('saved test preds') for l in lines)
    fj = H / 'logs' / f'{m}_fulldata.json'; r['fulldata_json_exists'] = fj.exists()
    if fj.exists():
        j = json.load(open(fj)); r['fulldata_json'] = {k: j.get(k) for k in ('holdout_in_train', 'kept_step', 'total_steps', 'n_training_feeds', 'n_holdout_feeds_in_training', 'pool_holdout_feeds')}
    ftp = H / 'logs' / f'{m}_n2cpv3s3_ft.log'; r['ft_log_exists'] = ftp.exists()
    if ftp.exists():
        ft = ftp.read_text(errors='replace'); r['ft_log_finished'] = 'final EMA weights' in ft or 'OOF ndcg' in ft.splitlines()[-1]
        r['ft_args_variant_ok'] = f"'variant': '{m}'" in ft.splitlines()[0]
    for nm, src in ((f'{m}.log', raw), (f'{m}_n2cpv3s3_ft.log', rd(ftp))):
        cp = rd(ROOT / 'nn/logs' / nm); k = 'gatecopy_train' if nm == f'{m}.log' else 'gatecopy_ft'
        if cp is None: r[k] = 'missing'
        elif src is None: r[k] = 'no source'
        elif cp == src: r[k] = 'identical'
        elif src.startswith(cp): r[k] = f'prefix ({len(cp)} of {len(src)} bytes)'
        else: r[k] = 'DIFFERENT'
    # the header the gate audits: the feature-list line that follows the args line
    fl = [l for l in lines if l.startswith('train-stage extra features')]; r['n_train_inputs'] = int(re.search(r'\((\d+)\)', fl[0]).group(1)) if fl else None
    cpt = rd(ROOT / 'nn/logs' / f'{m}.log')
    if cpt is not None:
        cfl = [l for l in cpt.decode('utf-8', 'replace').splitlines() if l.startswith('train-stage extra features')]
        r['gatecopy_feature_line_equal'] = bool(fl) and bool(cfl) and cfl[0] == fl[0]
    return r


res = {'stamp_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'members': {}, 'cands': {}}
for c in CANDS:
    base, pop, w = members(c['id']); fails = []
    for m in base + pop:
        if m not in res['members']: res['members'][m] = one(m)
        r = res['members'][m]
        if not r['log_exists']: fails.append(f'{m}: no training log'); continue
        if c['fulldata']:
            ok = (r['args_holdout_in_train_true'] and r['fulldata_flag_line'] and r['put_back_line'] and r['train_feeds'] == N_FULL and r['pool_holds_holdout_line']
                  and r['kept_step_last'] == r['total_steps'] and r['fulldata_json_exists'] and r['fulldata_json']['holdout_in_train'] is True and r['fulldata_json']['n_training_feeds'] == N_FULL)
            if not ok: fails.append(f'{m}: full-data evidence incomplete')
        else:
            ok = (not r['args_holdout_in_train_true'] and not r['fulldata_flag_line'] and r['n_fulldata_lines'] == 0 and r['train_feeds'] == N_HELD and r['excluded_line']
                  and not r['fulldata_json_exists'] and r['exploration_pool'] == 134118)
            if not ok: fails.append(f'{m}: held-out evidence incomplete')
        if not (r['args_variant_ok'] and r['finished'] and r.get('ft_log_exists') and r.get('ft_log_finished') and r.get('ft_args_variant_ok')):
            fails.append(f'{m}: run or fine-tune log incomplete')
    ms = [res['members'][m] for m in base + pop]
    res['cands'][c['id']] = dict(expected='full data' if c['fulldata'] else 'held-out', n_members=len(ms), fails=fails, d_pass=not fails,
                                 n_flag=sum(bool(x.get('args_holdout_in_train_true')) for x in ms), n_fulldata_line=sum(bool(x.get('fulldata_flag_line')) for x in ms),
                                 train_feeds=sorted(set(x.get('train_feeds') for x in ms)), pools=sorted(set(x.get('exploration_pool') for x in ms)),
                                 total_steps=sorted(set(x.get('total_steps') for x in ms)),
                                 gatecopy_train=sorted(set(re.sub(r' \(.*', '', x.get('gatecopy_train', '?')) for x in ms)), gatecopy_ft=sorted(set(re.sub(r' \(.*', '', x.get('gatecopy_ft', '?')) for x in ms)),
                                 gatecopy_feature_line_equal=all(x.get('gatecopy_feature_line_equal') for x in ms))
    v = res['cands'][c['id']]
    print(f"{c['id']:18s} expected {v['expected']:9s} | members {v['n_members']} | 'holdout_in_train': True in args: {v['n_flag']} | FULLDATA flag line: {v['n_fulldata_line']} | train feeds {v['train_feeds']} | cooldown pool {v['pools']} | total steps {v['total_steps']}"
          f" | (d) {'PASS' if v['d_pass'] else 'FAIL ' + '; '.join(fails)} | nn/logs copies: train {v['gatecopy_train']} ft {v['gatecopy_ft']} feature line equal {v['gatecopy_feature_line_equal']}")
json.dump(res, open(OUT / 'd_logs.json', 'w'), indent=1)
pre = [(m, r['gatecopy_train'], r['gatecopy_ft']) for m, r in res['members'].items() if r.get('gatecopy_train') != 'identical' or r.get('gatecopy_ft') != 'identical']
print(f'members: {len(res["members"])} distinct; nn/logs copies that are not identical to the harness log: {len(pre)}')
for x in pre: print('   ', *x)
print('written', OUT / 'd_logs.json')
