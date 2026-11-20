"""hz2 GATE (DECLARATION.txt, label-free): per input, age-weighted zero-share gap dZ and mean gap dM (in reference SD) of three views of the train rows
(as built / half-dose hz / hz2) against private and public test rows (feeds with M >= 50), by listing-age bucket; populations trPool (decision), trHold, trAll;
subsets all days / late (>= Sep 13) / four day groups.   Usage: python gate.py [tag]   -> logs/gate_<tag>.txt, cache/gate_<tag>.json
"""
import json
from hz2_common import *

TAG = sys.argv[1] if len(sys.argv) > 1 else 'final'
HAS_L = (HERE / 'feats_L' / 'train' / 'seq_cv_hz_norm.npy').exists()
VIEWS = ['asbuilt', 'hz', 'hz2', 'hz2k'] + (['hz2L', 'hz2Lk'] if HAS_L else [])
DG_NAMES = ['Sep 6-8', 'Sep 9-12', 'Sep 13-14', 'Sep 17-19']
PRIM = ['it_hist_all_luo', 'seq_cv_pop', 'xp_imp_c50']
SEC = ['it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo', 'it_hist_et4_luo', 'seq_cv_invd20', 'seq_cv_norm']
out_lines = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    out_lines.append(s)
    print(s, flush=True)


def view_path(view, col):
    if view in ('hz2', 'hz2k'):
        return OUT / 'train' / f'{HZ2_NAME[col]}.npy'
    if view in ('hz2L', 'hz2Lk'):
        return HERE / 'feats_L' / 'train' / f'{HZ2_NAME[col]}.npy'
    if view == 'hz' and col in LUO:
        return FNEW / 'train' / f'{HZ2_NAME[col]}.npy'
    return FC / 'train' / f'{col}.npy'


def main():
    ftr = feeds('train'); nF = len(ftr)
    M = ftr['M'].to_numpy().astype(np.int64)
    hold, p = holdout_feeds(ftr)
    popc = np.full(nF, 2, np.int64)                       # 0 pool, 1 holdout, 2 other
    pool = (p > 0.5) & (M >= 50); pool[hold] = False
    popc[pool] = 0; popc[hold] = 1
    part_pub = np.load(OUT / 'train' / 'hz2_part_pub.npy')
    L = np.load(OUT / 'train' / 'hz2_lag.npy').astype(np.int64)
    d = (day_of(ftr['timestamp'].to_numpy()) - DAY0).astype(np.int64)
    dg = np.where(d <= 2, 0, np.where(d <= 6, 1, np.where(d <= 9, 2, 3)))
    gfeed = (popc * 2 + part_pub.astype(np.int64)) * 4 + dg       # 24 feed groups
    kn = knots()
    N = int(M.sum())
    f4 = np.load(FLAT / 'train/cand_f4.npy', mmap_mode='r')
    cell = np.empty(N, np.int16)
    off = np.concatenate([[0], np.cumsum(M)])
    CH = 400000
    for s in range(0, nF, CH):
        e = min(nF, s + CH)
        b = age_bucket(age_hours(np.asarray(f4[off[s]:off[e]]), kn))
        cell[off[s]:off[e]] = (np.repeat(gfeed[s:e], M[s:e]) * 7 + b).astype(np.int16)
    NC = 24 * 7
    cnt = np.bincount(cell, minlength=NC).astype(np.float64)
    lg(f'cells ready; rows {N}; pool feeds {int(pool.sum())}, holdout {len(hold)}')
    stats = {}
    keep = np.load(OUT / 'train' / 'hz2_keep_rows.npy')
    lg(f'keep-mask rows {float(keep.mean()):.4f}')
    for col in COLS9:
        x0 = None
        for view in VIEWS:
            if view == 'hz' and col not in LUO:
                stats[(view, col)] = stats[('asbuilt', col)]
                continue
            x = xt_transform(col, np.load(view_path(view, col)))
            if view == 'asbuilt':
                x0 = x
            if view.endswith('k'):
                x = np.where(keep, x0, x)
            z = np.bincount(cell, weights=(x == 0), minlength=NC)
            sm = np.bincount(cell, weights=x.astype(np.float64), minlength=NC)
            stats[(view, col)] = (z, sm)
        del x, x0
        lg(f'  {col} done')
    cnt3 = cnt.reshape(3, 2, 4, 7)
    ref = {}
    fte = feeds('test'); tts = fte['timestamp'].to_numpy().astype(np.int64); tM = fte['M'].to_numpy().astype(np.int64)
    toff = fte['cand_off'].to_numpy().astype(np.int64)
    for pi, part in enumerate(('priv', 'pub')):
        sel = np.flatnonzero(((tts < PUB_TS) if part == 'pub' else (tts >= PUB_TS)) & (tM >= 50))
        crow = np.repeat(toff[sel] - np.concatenate([[0], np.cumsum(tM[sel])[:-1]]), tM[sel]) + np.arange(int(tM[sel].sum()))
        b = age_bucket(age_hours(np.load(FLAT / 'test/cand_f4.npy')[crow], kn))
        ref[part] = dict(n=len(crow), feeds=len(sel), w=(np.bincount(b, minlength=7) / len(b)).tolist())
        for col in COLS9:
            x = xt_transform(col, np.load(FC / 'test' / f'{col}.npy')[crow])
            ref[part][col] = dict(sd=float(x.std()), z=[float((x[b == k] == 0).mean()) for k in range(7)], m=[float(x[b == k].mean()) for k in range(7)],
                                  zero=float((x == 0).mean()))
        P(f'reference {part}: {len(sel)} test feeds with M >= 50, {len(crow)} rows; row share by age bucket ' + ' '.join(f'{w:.3f}' for w in ref[part]['w']))
    res = {}
    POPS = {'trPool': [0], 'trHold': [1], 'trAll': [0, 1, 2]}
    SUBS = {'all': [0, 1, 2, 3], 'late': [2, 3], 'Sep 6-8': [0], 'Sep 9-12': [1], 'Sep 13-14': [2], 'Sep 17-19': [3]}

    def g(view, col, pop, part, sub):
        pi = 0 if part == 'priv' else 1
        z, sm = stats[(view, col)]
        z = z.reshape(3, 2, 4, 7)[POPS[pop]][:, pi][:, SUBS[sub]].sum((0, 1))
        s = sm.reshape(3, 2, 4, 7)[POPS[pop]][:, pi][:, SUBS[sub]].sum((0, 1))
        c = cnt3[POPS[pop]][:, pi][:, SUBS[sub]].sum((0, 1))
        w = c / c.sum()
        zv = z / np.maximum(c, 1); mv = s / np.maximum(c, 1)
        r = ref[part][col]
        gz = np.abs(zv - np.array(r['z'])); gm = np.abs(mv - np.array(r['m'])) / r['sd']
        return dict(dZ=float((w * gz).sum()), dM=float((w * gm).sum()), big=float(np.max(np.where(w >= 0.05, gz, 0.0))),
                    bigk=int(np.argmax(np.where(w >= 0.05, gz, 0.0))), w=w.tolist(), zv=zv.tolist(), mv=mv.tolist(), rows=float(c.sum()),
                    zero=float(z.sum() / c.sum()))

    for pop in POPS:
        for part in ('priv', 'pub'):
            for sub in SUBS:
                key = f'{pop}|{part}|{sub}'
                res[key] = {view: {col: g(view, col, pop, part, sub) for col in COLS9} for view in VIEWS}
                r = res[key]
                P(f'\n=== {pop} | {part}-like feeds vs {"private" if part == "priv" else "public"} test rows | days: {sub} | rows {r["hz2"][PRIM[0]]["rows"]:.0f}')
                P('column             | view    |   dZ    | largest bucket (>=5% rows) |   dM   | pooled zero share view / ref')
                for col in COLS9:
                    for view in VIEWS:
                        if view == 'hz' and col not in LUO:
                            continue
                        x = r[view][col]
                        P(f'{col:18s} | {view:7s} | {x["dZ"]:.4f}  | {x["big"]:.4f} ({AGE_NAMES[x["bigk"]]:>10s})        | {x["dM"]:.4f} | {x["zero"]:.4f} / {ref[part][col]["zero"]:.4f}')
                # verdicts
                v = {}
                for view in VIEWS[1:]:
                    x = r[view]
                    g1 = all(x[c]['dZ'] <= 0.05 for c in PRIM); g2 = all(x[c]['big'] <= 0.10 for c in PRIM); g3 = all(x[c]['dM'] <= 0.20 for c in PRIM)
                    g4 = x['it_hist_all_luo']['dZ'] <= r['asbuilt']['it_hist_all_luo']['dZ'] / 3
                    g5 = all(x[c]['dZ'] <= 0.10 and x[c]['dM'] <= 0.30 for c in SEC)
                    v[view] = dict(G1=g1, G2=g2, G3=g3, G4=g4, G5=g5, PASS=bool(g1 and g2 and g3 and g4 and g5))
                    P(f'  {view}: G1 {"ok" if g1 else "FAIL"}  G2 {"ok" if g2 else "FAIL"}  G3 {"ok" if g3 else "FAIL"}  G4 {"ok" if g4 else "FAIL"}  G5 {"ok" if g5 else "FAIL"}'
                      f'  -> {"PASS" if v[view]["PASS"] else "FAIL"}')
                res[key]['verdict'] = v
                if sub in ('all', 'late') and pop in ('trPool', 'trHold'):
                    P('  zero share by age bucket (' + ' / '.join(VIEWS) + ' / reference), then mean on the model scale; row share of the view:')
                    for col in PRIM:
                        for k in range(7):
                            P(f'    {col:16s} {AGE_NAMES[k]:>10s} rows {r["hz2"][col]["w"][k]:.3f} : ' + ' / '.join(f'{r[v][col]["zv"][k]:.3f}' for v in VIEWS)
                              + f' / {ref[part][col]["z"][k]:.3f}    mean ' + ' / '.join(f'{r[v][col]["mv"][k]:.3f}' for v in VIEWS) + f' / {ref[part][col]["m"][k]:.3f}')
    P('\n=== DECISION (trPool) ===')
    for part in ('priv', 'pub'):
        for sub in ('all', 'late'):
            vd = res[f'trPool|{part}|{sub}']['verdict']
            P(f'  {part}-like, days {sub}: ' + '   '.join(f'{v} {"PASS" if vd[v]["PASS"] else "FAIL"} (' + ''.join(g_ if vd[v][g_] else g_.lower() + '!' for g_ in ('G1', 'G2', 'G3', 'G4', 'G5')) + ')' for v in VIEWS[1:]))
    json.dump(dict(res=res, ref=ref), open(CACHE / f'gate_{TAG}.json', 'w'))
    open(LOGS / f'gate_{TAG}.txt', 'w').write('\n'.join(out_lines) + '\n')
    lg('gate done')


if __name__ == '__main__':
    main()
