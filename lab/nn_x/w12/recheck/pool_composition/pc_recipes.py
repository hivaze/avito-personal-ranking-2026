"""Final table of the <= 5 recipes (ordered by out/PREDECLARED_rule.txt). TO branch = the recipe's train-only pool; FT branch = the existing
W8 fine-tunes (the extra members have no n2cpv3s3 files). Nothing is written to preds/ or subs/."""
from pc_common import *
m = Meta(); st = Store(m); G = groups()
ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
zv = lambda x: m.z('val', x); zt = lambda x: m.z('test', x); big = m.Mv >= 50
U = lambda *gs: [v for g in gs for v in G[g]]
W8 = U('w192', 't25w192'); NEW = U('w192new', 't25w192new')
R = {'ALLWIDE38': W8 + NEW + U('g2k05', 'g2k03', 'kx05', 'kx08', 'kw_drop0', 'kw_e20', 'kw_hl2', 'kw_ff4', 't25w192b3', 'kd08w192'),
     'W16k = W8 + kw(drop0,e20,hl2,ff4) x8': W8 + U('kw_drop0', 'kw_e20', 'kw_hl2', 'kw_ff4'),
     'W12b = W8 + kd03g2w192 x4': W8 + U('g2k03'), 'W12a = W8 + kd05g2w192 x4': W8 + U('g2k05'), 'W10b = W8 + kd05t25w192b3 x2': W8 + U('t25w192b3'),
     'W8 (reference)': W8, 'W11 = W8 + 3 new wide (R3 veto)': W8 + NEW, 'S16 (public -0.00041)': W8 + U('kd05', 'kd05t25')}
ftv = np.mean([zv(np.load(P/f'nn_{v}_n2cpv3s3_val_oof.npy')) for v in W8], 0); ftt = np.mean([zt(np.load(P/f'nn_{v}_n2cpv3s3_test.npy')) for v in W8], 0)
w = {sp: st.pool(sp, W8) for sp in ('val', 'hold', 'test')}; wL = st.pool('hold', [v + '@last' for v in W8])
ref = 0.5 * zv(ftv) + 0.5 * zv(w['val']); reft = 0.5 * zt(ftt) + 0.5 * zt(w['test']); e0 = ev(ref); t0 = ev(w['val']); h0 = eh(w['hold']); hL = eh(wL)
out = {}
for nm, ms in R.items():
    tv = st.pool('val', ms); th = st.pool('hold', ms); tt = st.pool('test', ms); tL = st.pool('hold', [v + '@last' for v in ms])
    dT = ev(tv) - t0; dH = eh(th) - h0; dL = eh(tL) - hL; s = 0.5 * zv(ftv) + 0.5 * zv(tv); dF = ev(s) - e0; stt = 0.5 * zt(ftt) + 0.5 * zt(tt)
    out[nm] = dict(n=len(ms), hold=float(eh(th).mean()), d_hold=float(dH.mean()), d_hold_se=se(dH), d_hold_last=float(dL.mean()), d_hold_last_se=se(dL), to_val=float(ev(tv).mean()), d_to_val=float(dT.mean()), d_to_val_se=se(dT),
                   final_val=float(ev(s).mean()), d_final=float(dF.mean()), d_final_se=se(dF), corr_to_test=float(np.corrcoef(tt, w['test'])[0, 1]), corr_final_test=float(np.corrcoef(stt, reft)[0, 1]), members=ms)
    print(f'{nm:40s} N {len(ms):2d} | hold {eh(th).mean():.5f} ({fmt(dH)}; @last {fmt(dL)}) | TO val {ev(tv).mean():.5f} ({fmt(dT)}) | final val (FT W8) {ev(s).mean():.5f} ({fmt(dF)}) | test corr TO {out[nm]["corr_to_test"]:.4f} final {out[nm]["corr_final_test"]:.4f}')
json.dump(out, open(OUT/'recipes.json', 'w'), indent=1)
