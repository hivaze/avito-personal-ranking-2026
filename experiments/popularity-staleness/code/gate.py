# Label-free gate of the lagged holdout. It compares the rebuilt columns with the columns of real test
# rows by listing-age bucket: share of zeros and mean on the model's input scale. The five criteria
# G1-G5 were fixed in DECLARATION.txt before any column was built.
# Run as: CUDA_VISIBLE_DEVICES="" python gate.py [tag view view ...]
# Produced logs/gate<tag>.txt and cache/gate<tag>.json. For the final construction one criterion failed
# (exposure count, listings aged 10-24 days: zero share 0.246 against 0.130 on private rows), so no
# labelled read was made on the full holdout.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/next/poplag/gate.py
#
"""poplag GATE (label-free): parity of the lagged holdout columns against the test rows' columns by item age x coverage.
Criterion = DECLARATION.txt (G1-G5), fixed before any column was built. No target file is opened.
Usage: CUDA_VISIBLE_DEVICES="" python gate.py [tag view view ...]   -> logs/gate<tag>.txt, cache/gate<tag>.json
       view = pub | priv (v1 files) or <pub|priv>_g<g> (v2 files, band g); the test part is the prefix. Default: tag '' and views pub priv.
"""
import sys, time
from pl_common import *

PRIM = ['it_hist_all_luo', 'seq_cv_pop', 'xp_imp_c50']
SEC = ['it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo', 'it_hist_et4_luo', 'seq_cv_invd20', 'seq_cv_norm']
out_lines = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out_lines.append(s)


def stats(raw, name, bk):
    x = xt_transform(name, raw).astype(np.float64); z = (np.asarray(raw) == 0)
    n = np.bincount(bk, minlength=8)[:7].astype(np.float64); k = bk < 7
    zs = np.bincount(bk, weights=z, minlength=8)[:7] / np.maximum(n, 1)
    ms = np.bincount(bk, weights=x, minlength=8)[:7] / np.maximum(n, 1)
    return dict(n=n, zero=zs, mean=ms, zero_all=float(z[k].mean()), mean_all=float(x[k].mean()), sd_all=float(x[k].std()))


def main():
    kn = knots(); mini = CACHE / 'mini'
    crow = np.load(mini / 'crow.npy'); f4h = np.load(mini / 'c_f.npy')[:, 3]
    bkh = age_bucket(age_hours(f4h, kn))
    tag = sys.argv[1] if len(sys.argv) > 1 else ''
    if 'late' in tag:   # information only: the same statistics restricted to holdout days >= Sep 13 (rows of other days get bucket 7 = ignored)
        late = np.repeat(day_of(np.load(mini / 'ts.npy')) >= 20709, np.load(mini / 'M.npy'))
        bkh = np.where(late, bkh, 7)
    wh = np.bincount(bkh, minlength=8)[:7] / max(int((bkh < 7).sum()), 1)
    fte = feeds('test'); Mt = fte['M'].to_numpy().astype(np.int64); tts = fte['timestamp'].to_numpy().astype(np.int64)
    fi = np.load(FLAT / 'test/cand_fidx.npy'); bkt = age_bucket(age_hours(np.load(FLAT / 'test/cand_f4.npy'), kn))
    part = {'pub': (tts < PUB_TS)[fi] & (Mt[fi] >= 50), 'priv': (tts >= PUB_TS)[fi] & (Mt[fi] >= 50)}
    partall = {'pub': (tts < PUB_TS)[fi], 'priv': (tts >= PUB_TS)[fi]}
    P(f'=== poplag GATE {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} (label-free) ===')
    P(f'holdout rows {int((bkh < 7).sum())} ({"holdout days >= Sep 13 only; INFORMATION, not the declared gate" if "late" in tag else "30,000 feeds"}, all M >= 50); test rows M >= 50: public {int(part["pub"].sum())} ({int(((tts < PUB_TS) & (Mt >= 50)).sum())} feeds), '
      f'private {int(part["priv"].sum())} ({int(((tts >= PUB_TS) & (Mt >= 50)).sum())} feeds)')
    P('age-bucket share of rows: ' + ' | '.join(f'{AGE_NAMES[b]} hold {wh[b]:.3f} pub {np.mean(bkt[part["pub"]] == b):.3f} priv {np.mean(bkt[part["priv"]] == b):.3f}' for b in range(7)))
    res = {}; verdict = {}; summ = []
    tag = sys.argv[1] if len(sys.argv) > 1 else ''
    for vname in (sys.argv[2:] if len(sys.argv) > 2 else ['pub', 'priv']):
        view = vname.split('_')[0]
        P(f'\n##### view {vname}: lagged holdout vs test {view} rows (M >= 50) #####')
        fails = []; res[vname] = {}
        for name in PRIM + SEC:
            st0 = stats(np.asarray(np.load(FC / 'train' / f'{name}.npy', mmap_mode='r')[crow]), name, bkh)
            lagp = CACHE / f'lag_{vname}_{name}.npy'
            stl = stats(np.load(lagp), name, bkh)
            traw = np.load(FC / 'test' / f'{name}.npy')
            stt = stats(traw[part[view]], name, bkt[part[view]])
            stta = stats(traw[partall[view]], name, bkt[partall[view]])
            sd = stt['sd_all']
            dZ0 = float((wh * np.abs(st0['zero'] - stt['zero'])).sum()); dZ = float((wh * np.abs(stl['zero'] - stt['zero'])).sum())
            dM0 = float((wh * np.abs(st0['mean'] - stt['mean'])).sum() / sd); dM = float((wh * np.abs(stl['mean'] - stt['mean'])).sum() / sd)
            sgnZ = float((wh * (stl['zero'] - stt['zero'])).sum()); sgnM = float((wh * (stl['mean'] - stt['mean'])).sum() / sd)
            maxb = float(np.max(np.where(wh >= 0.05, np.abs(stl['zero'] - stt['zero']), 0)))
            res[vname][name] = dict(dZ0=dZ0, dZ=dZ, dM0=dM0, dM=dM, maxbucket=maxb, signedZ=sgnZ, signedM=sgnM, sd=sd,
                                   zero=dict(lag0=st0['zero'].tolist(), lag=stl['zero'].tolist(), test=stt['zero'].tolist()),
                                   mean=dict(lag0=st0['mean'].tolist(), lag=stl['mean'].tolist(), test=stt['mean'].tolist()),
                                   pooled=dict(zero_lag0=st0['zero_all'], zero_lag=stl['zero_all'], zero_test=stt['zero_all'], zero_test_allM=stta['zero_all'],
                                               mean_lag0=st0['mean_all'], mean_lag=stl['mean_all'], mean_test=stt['mean_all']))
            P(f'\n{name}  [{"PRIMARY" if name in PRIM else "secondary"}]  SD(test {view}) {sd:.3f}')
            P('  bucket      rows_hold | zero: lag0   lagged   test | mean(model scale): lag0   lagged   test | d zero (lag-test)  d mean/SD')
            for b in range(7):
                P(f'  {AGE_NAMES[b]:11s} {wh[b]:8.3f}  |      {st0["zero"][b]:.3f}   {stl["zero"][b]:.3f}   {stt["zero"][b]:.3f} |'
                  f'                    {st0["mean"][b]:.3f}   {stl["mean"][b]:.3f}   {stt["mean"][b]:.3f} |   {stl["zero"][b] - stt["zero"][b]:+.3f}           {(stl["mean"][b] - stt["mean"][b]) / sd:+.3f}')
            P(f'  pooled: zero lag0 {st0["zero_all"]:.3f} lagged {stl["zero_all"]:.3f} test(M>=50) {stt["zero_all"]:.3f} test(all M) {stta["zero_all"]:.3f} | '
              f'mean lag0 {st0["mean_all"]:.3f} lagged {stl["mean_all"]:.3f} test {stt["mean_all"]:.3f}')
            P(f'  dZ lag0 {dZ0:.4f} -> lagged {dZ:.4f} (signed {sgnZ:+.4f}; max bucket >= 5% rows {maxb:.3f}) | dM lag0 {dM0:.4f} -> lagged {dM:.4f} (signed {sgnM:+.4f})')
            if name in PRIM:
                if dZ > 0.05: fails.append(f'G1 {name} dZ {dZ:.4f} > 0.05')
                if maxb > 0.10: fails.append(f'G2 {name} max bucket {maxb:.3f} > 0.10')
                if dM > 0.20: fails.append(f'G3 {name} dM {dM:.4f} > 0.20')
                if name == 'it_hist_all_luo' and dZ > dZ0 / 3: fails.append(f'G4 it_hist_all_luo dZ {dZ:.4f} > lag-0 dZ / 3 = {dZ0 / 3:.4f}')
            else:
                if dZ > 0.10: fails.append(f'G5 {name} dZ {dZ:.4f} > 0.10')
                if dM > 0.30: fails.append(f'G5 {name} dM {dM:.4f} > 0.30')
        verdict[vname] = 'PASS' if not fails else 'FAIL'
        J = sum(res[vname][n]['dZ'] for n in PRIM)
        P(f'\n>>> GATE view {vname}: {verdict[vname]}' + ('' if not fails else ' | ' + '; '.join(fails)))
        summ.append(f'{vname:10s} {verdict[vname]}  J = sum dZ primaries {J:.4f} | ' + ' '.join(f'{n} dZ {res[vname][n]["dZ"]:.4f} maxb {res[vname][n]["maxbucket"]:.3f} dM {res[vname][n]["dM"]:.3f}' for n in PRIM)
                    + ' | sec max dZ ' + f'{max(res[vname][n]["dZ"] for n in SEC):.4f} max dM {max(res[vname][n]["dM"] for n in SEC):.3f}' + ' | fails: ' + ('; '.join(fails) or '-'))
        res[vname]['fails'] = fails; res[vname]['J'] = J
    P('\n=== SUMMARY ===')
    for l in summ: P(l)
    res['verdict'] = verdict
    json.dump(res, open(CACHE / f'gate{tag}.json', 'w'), indent=1)
    (LOGS / f'gate{tag}.txt').write_text('\n'.join(out_lines) + '\n')


if __name__ == '__main__':
    main()
