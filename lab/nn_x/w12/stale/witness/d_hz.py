"""witness task D: READ of the re-opened arm hz (popularity-parity cooldown under KD at d192), rule "ARM hz" of
nn_x/w5/logs/E_kd_predeclared.txt (2026-11-15T02:15:30Z), applied as written; implementation choices in DECLARATION.txt block 1 (4) and block 4 (9).

  RULE hz (train-only pools, SELECTED checkpoints, paired per feed, hz re-cools vs ref re-cools of the same snapshots):
     PASS iff val delta >= 0 AND stale-val delta >= +2 SE AND val M<50 delta >= -1 SE.
     closed again iff val < 0 OR stale-val < +1 SE.   Reported, no decision: holdout, HOLD-SHORT, pmix, cold / warm positives, lagged holdout.
  Only snapshot pairs that are complete on both sides (logs/<variant>.json) AND have CPU stale-val for both enter the decision table (the same pairs
  in every column); lagged and @last columns use the pairs that also have those files (counts printed per table).

  cd <repo>/nn_x/w12/stale/witness && WT_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 ../../../../.venv/bin/python d_hz.py [arm = hz] [ref = ref]
Output: logs/read_<arm>_vs_<ref>.txt (+ a time-stamped copy), cache/readout/<arm>_vs_<ref>.json.
"""
import os
os.environ.setdefault('WT_THREADS', '4')
from wt_common import *
import lagged_score as LS
import stale_val_cpu as SVC
import wt_read as WR

ARM = sys.argv[1] if len(sys.argv) > 1 else 'hz'
REF = sys.argv[2] if len(sys.argv) > 2 else 'ref'
OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)


F = {'val': stored_val, 'hold': stored_holdout, 'stale': lambda x: SVC.sv_path(x, 'stale'), 'lag0': lambda x: LS.pred_path('lag0', x),
     'priv': lambda x: LS.pred_path('priv', x), 'pub': lambda x: LS.pred_path('pub', x)}
snaps = [(st, s) for st in 'kl' for s in S8]
name = lambda a, st, s: f'w5_{st}cd_{a}_s{s}'
if ARM == 'hzcd':      # the 11-12 arm (hard labels, d128, 4 roots): `d_hz.py hzcd m02ref` = code test of this script + the old arm in the same format
    snaps = [('', r) for r in (101, 102, 103, 104)]
    name = lambda a, st, s: f'w5_cd_{a}_r{s}'
tagp = lambda ps: [f'{st}{s}' for st, s in ps]


def both(ps, suf, kinds, arms=None):
    return [(st, s) for st, s in ps if all(F[k](name(a, st, s) + suf).exists() for a in (arms or (ARM, REF)) for k in kinds)]


complete = [(st, s) for st, s in snaps if done_json(name(ARM, st, s)).exists() and done_json(name(REF, st, s)).exists()]
RES = {}
P(f'=== witness READ of arm {ARM} vs {REF} re-cools, {now()} ===')
P(f'complete ({ARM}, {REF}) snapshot pairs (both json): {len(complete)} of 16: {tagp(complete)}')
P(f'not complete: {tagp([p for p in snaps if p not in complete])}')
sel = both(both(complete, '', ('val', 'hold')), '', ('stale',))
P(f'DECISION PAIRS (complete + stored val / holdout + CPU stale-val of the SELECTED checkpoints on both sides): {len(sel)}: {tagp(sel)}')
if len(sel) < 1:
    P('no pair to read: tools and command only (see the report)')
    (LOGS / f'read_{ARM}_vs_{REF}.txt').write_text('\n'.join(OUT) + '\n'); sys.exit(0)
wt = WR.Witness()
pk = G['pop1kd']
LAGK = lambda k: 'lag' in k or k.startswith(('D_', 'drop', '_pf'))


def table(ps, suf, title, short=True):
    mk = lambda a, q: [(1.0, [name(a, st, s) + suf for st, s in q])]
    r = wt.read(mk(ARM, ps), mk(REF, ps), lag_views=(), short=short)
    P('\n' + WR.show(r, f'{title}: {ARM} {len(ps)}-pool vs {REF} {len(ps)}-pool, paired by snapshot'))
    lagp = both(ps, suf, ('lag0', 'priv'))
    if lagp:
        pubp = both(lagp, suf, ('pub',))
        rl = wt.read(mk(ARM, lagp), mk(REF, lagp), lag_views=('lag0', 'priv') + (('pub',) if len(pubp) == len(lagp) else ()), short=False)
        ref_cols = {k: rl[k] for k in ('val', 'stale', 'SV2', 'hold', 'hold_subset') if k in rl}
        P(WR.show({k: v for k, v in rl.items() if LAGK(k)} | ref_cols,
                  f'   LAGGED HOLDOUT (subset 13,424 feeds; CPU fp32 both sides), {len(lagp)} pairs {tagp(lagp)} (val / stale-val / stored holdout of the same pairs for reference)'))
        for cn, (x, y) in (('ivw(val, lag_priv)', ('val', 'lag_priv')), ('ivw(stale, lag_priv)', ('stale', 'lag_priv')), ('ivw(SV2, lag_priv)', ('SV2', 'lag_priv'))):
            if x in rl and y in rl:
                m, s_ = WR.ivw(rl[x], rl[y]); P(f'   {cn:22s} {m:+.5f} +- {s_:.5f}  (z {m / s_:+.2f}; same {len(lagp)} pairs)')
        r['lag'] = WR.dump(rl); r['lag_pairs'] = tagp(lagp)
    else:
        P(f'   LAGGED HOLDOUT: no pair of this table has lagged predictions on both sides (not scored)')
    return r, mk


def per_snapshot(ps, suf, mk):
    kinds = ('val', 'stale', 'hold') if both(ps, suf, ('stale',)) == ps else ('val', 'hold')
    P(f'\n   per snapshot (single re-cool member, {ARM} - {REF}): ' + ' | '.join(kinds))
    pos = {k: 0 for k in kinds}
    for st, s in ps:
        rm = wt.read(mk(ARM, [(st, s)]), mk(REF, [(st, s)]), lag_views=(), short=False)
        for k in kinds:
            pos[k] += rm[k][0] > 0
        P(f'      {st}cd s{s}: ' + ' | '.join(WR.cell(rm[k]) for k in kinds))
    P('      positive in ' + ' / '.join(f'{pos[k]} ({k})' for k in kinds) + f' of {len(ps)} snapshots')
    for st in 'kl':
        q = [p for p in ps if p[0] == st]
        if len(q) >= 2 and len(q) < len(ps):
            rh = wt.read(mk(ARM, q), mk(REF, q), lag_views=(), short=False)
            P(f'      half {st}cd ({len(q)} snapshots; teacher {"mean11" if st == "k" else "mean25"}): ' + ' | '.join(f'{k} {WR.cell(rh[k])}' for k in kinds))


def mix_table(ps, suf, r):
    if not all(F[k](v).exists() for v in pk for k in ('val', 'hold', 'stale')):
        P('\n   admixture table: pop1kd predictions missing'); return None
    mk = lambda a, q: [(1.0, [name(a, st, s) + suf for st, s in q])]
    mix = lambda a, q: [(0.8, [name(a, st, s) + suf for st, s in q]), (0.2, pk)]
    r2 = wt.read(mix(ARM, ps), mix(REF, ps), lag_views=())
    P('\n' + WR.show(r2, f'[0.8 z({ARM} {len(ps)}-pool) + 0.2 z(pop1kd x4)] vs [0.8 z({REF} {len(ps)}-pool) + 0.2 z(pop1kd x4)]  (pop1kd members: selected checkpoints)'))
    lagp = both(ps, suf, ('lag0', 'priv'))
    if lagp and all(F[k](v).exists() for v in pk for k in ('lag0', 'priv')):
        r2l = wt.read(mix(ARM, lagp), mix(REF, lagp), lag_views=('lag0', 'priv'), short=False)
        P(WR.show({k: v for k, v in r2l.items() if LAGK(k)}, f'   same, LAGGED HOLDOUT subset, {len(lagp)} pairs'))
        r2['lag'] = WR.dump(r2l)
    r3 = wt.read(mix(ARM, ps), mk(ARM, ps), lag_views=(), short=False); r4 = wt.read(mix(REF, ps), mk(REF, ps), lag_views=(), short=False)
    g = lambda rr, k: WR.cell(rr[k]) if k in rr else 'n/a'
    P(f'   value of the pop1kd admixture (w 0.2) on the {ARM} pool ({len(ps)} members): val {g(r3, "val")}, stale-val {g(r3, "stale")}, SV2 {g(r3, "SV2")}, holdout {g(r3, "hold")}')
    P(f'   value of the pop1kd admixture (w 0.2) on the {REF} pool ({len(ps)} members): val {g(r4, "val")}, stale-val {g(r4, "stale")}, SV2 {g(r4, "SV2")}, holdout {g(r4, "hold")}')
    if lagp and all(F[k](v).exists() for v in pk for k in ('lag0', 'priv')):
        for a in (ARM, REF):
            rr = wt.read(mix(a, lagp), mk(a, lagp), lag_views=('lag0', 'priv'), short=False)
            P(f'   same on the lagged subset, {a} pool ({len(lagp)} members): lag0 {g(rr, "lag_lag0")}, private-like {g(rr, "lag_priv")}, D {g(rr, "D_priv")}')
    if 'stale' in r and 'stale' in r2:
        hv, hs = r['val'][0], r['stale'][0]
        if hv > 0 and hs > 0:
            fv, fs = r2['val'][0] / hv, r2['stale'][0] / hs
            kind = 'ADD' if (fv >= 0.5 and fs >= 0.5) else ('SUBSTITUTE' if (fv < 0.5 and fs < 0.5) else 'MIXED')
            P(f'   add or substitute (DECLARATION block 1 (4)): share of the plain {ARM} - {REF} delta kept on top of the admixture: val {fv:.2f}, stale-val {fs:.2f} -> {kind}')
        else:
            P(f'   add or substitute: the plain {ARM} - {REF} delta is not positive on both val ({hv:+.5f}) and stale-val ({hs:+.5f}); shares not defined')
    return r2


# ---------------------------------------------------------------- SELECTED checkpoints (decision)
P('\n\n################ SELECTED checkpoints (decision) ################')
r, mk = table(sel, '', 'SELECTED')
v, sv, sm = r['val'], r['stale'], r['val_M<50']
c1, c2, c3 = v[0] >= 0, sv[0] >= 2 * sv[1], sm[0] >= -sm[1]
closed = (v[0] < 0) or (sv[0] < sv[1])
verdict = 'PASS' if (c1 and c2 and c3) else ('CLOSED AGAIN under KD' if closed else 'no pass (neither PASS nor the closing clause)')
P(f'\nRULE hz on {len(sel)} pairs: val delta >= 0 [{c1}: {v[0]:+.5f} +- {v[1]:.5f}]; stale-val delta >= +2 SE [{c2}: {sv[0]:+.5f} +- {sv[1]:.5f} = {sv[0] / sv[1]:+.2f} SE]; '
  f'val M<50 >= -1 SE [{c3}: {sm[0]:+.5f} +- {sm[1]:.5f} = {sm[0] / sm[1]:+.2f} SE]')
P(f'VERDICT ({len(sel)} of 16 pairs{"; fewer than 8 pairs: NOT a decision read" if len(sel) < 8 else ""}): {verdict}')
r['verdict'] = verdict; r['n_pairs'] = len(sel); r['pairs'] = tagp(sel)
per_snapshot(sel, '', mk)
RES['plain'] = WR.dump(r)
r2 = mix_table(sel, '', r)
if r2 is not None:
    RES['mix'] = WR.dump(r2)

# ---------------------------------------------------------------- @last checkpoints (reported)
P('\n\n################ @last checkpoints (reported, no decision) ################')
lastfree = both(sel, '@last', ('val', 'hold'))
laststale = both(lastfree, '@last', ('stale',))
P(f'of the {len(sel)} decision pairs: {len(lastfree)} have stored @last val / holdout files, {len(laststale)} have CPU @last stale-val on both sides {tagp(laststale)}')
if lastfree:
    if len(laststale) == len(lastfree):
        rl_, mkl = table(lastfree, '@last', '@last'); per_snapshot(lastfree, '@last', mkl); RES['plain@last'] = WR.dump(rl_)
        r2 = mix_table(lastfree, '@last', rl_)
        if r2 is not None:
            RES['mix@last'] = WR.dump(r2)
    else:
        rl_, mkl = table(lastfree, '@last', '@last, stored-file witnesses only (no CPU stale-val for some pairs)'); RES['free@last'] = WR.dump(rl_)
        if laststale:
            rs_, mks = table(laststale, '@last', f'@last, the {len(laststale)} pairs with CPU stale-val', short=False); per_snapshot(laststale, '@last', mks)
            RES['plain@last'] = WR.dump(rs_)
            r2 = mix_table(laststale, '@last', rs_)
            if r2 is not None:
                RES['mix@last'] = WR.dump(r2)

# ---------------------------------------------------------------- nulls of the harness (information)
P('\n\n################ harness nulls on the same snapshots (information) ################')
orig = lambda st, s: f'w5_e10_{"kd05w192" if st == "k" else "kd05t25w192"}_s{s}'
q = [(st, s) for st, s in sel if SVC.sv_path(orig(st, s)).exists()]
if q:
    rh = wt.read([(1.0, [name(REF, st, s) for st, s in q])], [(1.0, [orig(st, s) for st, s in q])], lag_views=('lag0', 'priv'), short=False)
    P(WR.show(rh, f'{REF} re-cools vs the ORIGINAL runs of the same {len(q)} snapshots (harness control, expected 0)')); RES['ref_vs_orig'] = WR.dump(rh)
    ro = wt.read([(1.0, [name(ARM, st, s) for st, s in q])], [(1.0, [orig(st, s) for st, s in q])], lag_views=('lag0', 'priv'), short=False)
    P(WR.show(ro, f'{ARM} re-cools vs the ORIGINAL runs of the same {len(q)} snapshots (information)')); RES['arm_vs_orig'] = WR.dump(ro)
q2 = [(st, s) for st, s in snaps if done_json(name('ref2', st, s)).exists() and done_json(name('ref', st, s)).exists()
      and all(F[k](name(a, st, s)).exists() for a in ('ref2', 'ref') for k in ('val', 'hold'))]
if q2 and REF == 'ref':
    rn = wt.read([(1.0, [name('ref2', st, s) for st, s in q2])], [(1.0, [name('ref', st, s) for st, s in q2])], lag_views=(), short=False)
    P(WR.show(rn, f'ref2 vs ref (second cooldown seed, {len(q2)} snapshot pairs {tagp(q2)}): the A/A null of a paired re-cool contrast')); RES['ref2_vs_ref'] = WR.dump(rn)

(CACHE / 'readout').mkdir(exist_ok=True)
json.dump(RES, open(CACHE / f'readout/{ARM}_vs_{REF}.json', 'w'), indent=1)
(LOGS / f'read_{ARM}_vs_{REF}.txt').write_text('\n'.join(OUT) + '\n')
(LOGS / f'read_{ARM}_vs_{REF}_{time.strftime("%H%MZ", time.gmtime())}.txt').write_text('\n'.join(OUT) + '\n')
