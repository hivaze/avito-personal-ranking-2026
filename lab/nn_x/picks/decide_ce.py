import sys; sys.path.insert(0, '<repo>/src')
src = open('/tmp/scratchpad/decide.py').read()
src = src.replace("'F7p': [f'v17_pre2s_s{s}@77500' for s in range(101,107)],", "'F7p': [f'v17_pre2s_s{s}@77500' for s in range(101,107)],\n 'F8': [f'v20_pre2s_e2_s{s}' for s in range(301,313)],")
src = src.split("C = {'Ca {F5}'")[0] + '''
C = {'Ca {F5}': fam_equal(['F5']), 'Cd F5cd+F6+F7': mem_equal(['F5cd','F6','F7']), 'Ce F5cd+F6+F7+F8': mem_equal(['F5cd','F6','F7','F8'])}
ref_v = ev(C['Ca {F5}'][0]); ref_h = eh(C['Ca {F5}'][1])
for k, (sv, sh) in C.items():
    pv, ph = ev(sv), eh(sh); d = pv-ref_v
    print(f'  {k:22s} hold {ph.mean():.5f} | val {pv.mean():.5f} d {d.mean():+.5f}+-{se(d):.5f}')
f8h = eh(fam_h(FAM['F8'][:6])).mean(); bests = {f: eh(fam_h(FAM[f][:6])).mean() for f in ['F5cd','F6','F7']}
f8v = ev(fam_mix_v(FAM['F8'][:6])).mean(); bestv = {f: ev(fam_mix_v(FAM[f][:6])).mean() for f in ['F5cd','F6','F7']}
print('gate (b) family-ensemble (first 6 seeds) holdout:', round(f8h,5), {k: round(v,5) for k,v in bests.items()}, ' val mix:', round(f8v,5), {k: round(v,5) for k,v in bestv.items()})
'''
exec(src)
