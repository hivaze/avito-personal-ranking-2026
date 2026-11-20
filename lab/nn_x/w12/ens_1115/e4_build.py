"""e4: build the staged candidate files (nn_x/w12/ens_1115/stage/ only; nothing is written to subs/ or preds/; the lead gates and moves them).
Form = build_r4.py: per branch 0.8 z(base pool) + 0.2 z(popularity-light pool), pools member-equal means of per-feed z, final = 0.5 z(FT n2cpv3s3) + 0.5 z(train-only).
  ENSA = structure A: 32 wide students (16 held-out + 16 full-data) + 36 popularity-light students (8 held-out + 28 full-data)        [NOT all-data: 24 of 68 members exclude the holdout feeds]
  ENSB = structure B: 32 all-data base members (16 wide + 16 hz re-cools) + 28 all-data popularity-light students                   [all 60 members trained on all data]
Per file: stage/cand_<TAG>kd_stage.parquet (src/make_sub.make_sub), stage/ens_<TAG>kd_{ft,to}_{val,test}.npy, stage/<TAG>_members.txt, stage/<TAG>_manifest.json (sha256, read-out)."""
from ens_common import *
import hashlib, time, pmix
from make_sub import make_sub
import polars as pl
m = Meta(); st = Store(m); WTS = pmix.weights(); big = m.Mv >= 50
SPEC = {'ENSA': (W16 + FD16, P1d + P1w + FDP12 + FDPd + FDP2 + FDPF, 0.2), 'ENSB': (FD16 + FDXHZ, FDP12 + FDPd + FDP2 + FDPF, 0.2)}
want = sys.argv[1].split(',') if len(sys.argv) > 1 else list(SPEC)
inv = json.load(open(OUT / 'inventory.json'))['members']
fin = lambda sp, ft, to: 0.5 * m.z(sp, ft) + 0.5 * m.z(sp, to)
stored = lambda tag, sp: (np.load(P / f'ens_{tag}_ft_{sp}.npy'), np.load(P / f'ens_{tag}_to_{sp}.npy'))
REF = {k: {sp: fin(sp, *stored(t, sp)) for sp in ('val', 'test')} for k, t in (('FD16P20', 'FD16P20kd'), ('W8P20', 'W8P20kd'))}; eR = {k: m.ev(REF[k]['val']) for k in REF}
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
def raw_pool(names, kind):
    """Independent path: straight from the .npy files in preds/ exactly as build_r4.pool does (not from the cache)."""
    sp = SPLIT[kind]; return np.mean([m.z(sp, np.load(P / f'nn_{n_}_{SUF[kind]}.npy')) for n_ in names], 0)
for tag in want:
    base, pop, w = SPEC[tag]; B = {}
    for sp in ('val', 'test'):
        br = []
        for kind in (f'ft_{sp}', f'to_{sp}'):
            b = raw_pool(base, kind); p = raw_pool(pop, kind); x = (1 - w) * m.z(sp, b) + w * m.z(sp, p)
            xc = (1 - w) * m.z(sp, st.pool(kind, base)) + w * m.z(sp, st.pool(kind, pop)); assert np.abs(x - xc).max() < 1e-4, (tag, kind, float(np.abs(x - xc).max()))   # cache path == file path
            br.append(x.astype(np.float32))
        B[sp] = br
    e = m.ev(fin('val', *B['val'])); tfin = fin('test', *B['test']).astype(np.float32); assert np.isfinite(tfin).all()
    out = STAGE / f'cand_{tag}kd_stage.parquet'
    for sp in ('val', 'test'):
        np.save(STAGE / f'ens_{tag}kd_ft_{sp}.npy', B[sp][0]); np.save(STAGE / f'ens_{tag}kd_to_{sp}.npy', B[sp][1])
    if out.exists(): out.unlink()
    make_sub(tfin, str(out)); sub = pl.read_parquet(out)
    full = {n_: inv[n_]['full_data'] for n_ in base + pop}
    man = dict(tag=tag, file=str(out.relative_to(ROOT)), built=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), sha256=sha(out), rows=len(sub), feeds=int(sub['feed_id'].n_unique()),
               n_base=len(base), n_pop=len(pop), w_pop=w, base=base, pop=pop, members_all_data=int(sum(bool(v) for v in full.values())), members_total=len(full),
               final_val=float(e.mean()), ft_branch_val=float(m.ev(B['val'][0]).mean()), to_branch_val=float(m.ev(B['val'][1]).mean()),
               branch_files={f'{b_}_{sp}': dict(path=str((STAGE / f'ens_{tag}kd_{b_}_{sp}.npy').relative_to(ROOT)), sha256=sha(STAGE / f'ens_{tag}kd_{b_}_{sp}.npy')) for b_ in ('ft', 'to') for sp in ('val', 'test')}, vs={})
    for r in REF:
        d = e - eR[r]; pp, sp_ = pmix.wmean_se(d, WTS['public']); pr, sr = pmix.wmean_se(d, WTS['private'])
        man['vs'][r] = dict(val=float(d.mean()), val_se=se(d), pmix_public=float(pp), pmix_public_se=float(sp_), pmix_private=float(pr), pmix_private_se=float(sr), short=float(d[~big].mean()), long=float(d[big].mean()),
                            test_corr=float(np.corrcoef(tfin, REF[r]['test'])[0, 1]), paired_public_se=float(d.std() / np.sqrt(8562)))
    json.dump(man, open(STAGE / f'{tag}_manifest.json', 'w'), indent=1)
    open(STAGE / f'{tag}_members.txt', 'w').write(f'# {tag}: per branch (1 - {w}) z(base pool) + {w} z(popularity-light pool); final 0.5 z(FT n2cpv3s3) + 0.5 z(train-only)\n# base ({len(base)}):\n' + '\n'.join(base) +
                                                  f'\n# popularity-light ({len(pop)}):\n' + '\n'.join(pop) + '\n')
    print(f'{tag}: {len(base)} base + {len(pop)} pop at w {w} | all-data members {man["members_all_data"]} of {man["members_total"]} | final val {e.mean():.5f} FT {man["ft_branch_val"]:.5f} TO {man["to_branch_val"]:.5f} | sha256 {man["sha256"]} | rows {man["rows"]} feeds {man["feeds"]}')
    for r in REF:
        v = man['vs'][r]; print(f'   vs {r:8s}: val {v["val"]:+.5f}+-{v["val_se"]:.5f} | pmix public {v["pmix_public"]:+.5f}+-{v["pmix_public_se"]:.5f} private {v["pmix_private"]:+.5f}+-{v["pmix_private_se"]:.5f} | M<50 {v["short"]:+.5f} M>=50 {v["long"]:+.5f} | test corr {v["test_corr"]:.5f} | paired public SE {v["paired_public_se"]:.5f}')
