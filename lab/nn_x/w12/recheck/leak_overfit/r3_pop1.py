"""R3: pop1-view admixture to W8 (kd_reprice#5 / pool_composition#10 / lead's W8P20).
 (a) weight curve at TO level, val + holdout; (b) CONTROL: same admixture with non-view families (is the val gain specific to pop1?);
 (c) nested weight choice on user halves; (d) slices: val day, M; holdout weekday / private-like days; (e) final level; (f) paired public SE."""
from lo_common import *
rng = np.random.default_rng(7)
w8v, w8h = poolv(W8), poolh(W8); e0v, e0h = ev(w8v), eh(w8h)
zw8v, zw8h = zv(w8v), zh(w8h)
fam = {'pop1kd x4 (students, pop1 view)': POP1KD, 'pop1 x8 (hard, pop1 view)': POP1, 'base x4 (hard, same view)': E('base'), 'kx05 x4 (students, xfit teacher)': E('kx05'),
       'kd05 x4 (narrow students)': E('kd05'), 'kd05g2w192 x4 (gen-2)': E('kd05g2w192'), 'F10 GRU x6 (hard)': F10, 'F7x x7 (hard m02)': F7x, 'kd10 x4 (pure teacher)': E('kd10')}
WS = (0.1, 0.15, 0.2, 0.3, 0.5)
print('== (a)+(b) TO level: (1-w) z(W8) + w z(family pool) minus W8; val | holdout')
PV, PH = {}, {}
for k, ms in fam.items():
    PV[k], PH[k] = zv(poolv(ms)), zh(poolh(ms))
    line = f' {k:34s} own val {ev(PV[k]).mean():.5f} hold {eh(PH[k]).mean():.5f} corrW8 {np.corrcoef(PV[k], zw8v)[0,1]:.4f} |'
    for w in WS:
        dv = ev((1-w)*zw8v+w*PV[k])-e0v; dh = eh((1-w)*zw8h+w*PH[k])-e0h
        line += f' w{w}: {dv.mean():+.5f}({dv.mean()/se(dv):+.1f}) / {dh.mean():+.5f}({dh.mean()/se(dh):+.1f}) |'
    print(line)
k = 'pop1kd x4 (students, pop1 view)'
print('== (c) nested weight choice for pop1kd (grid 0..0.5 step 0.05), 40 user-half splits on val')
uid = fv['user_id'].to_numpy(); uu = np.unique(uid); grid = np.arange(0, 0.51, 0.05)
D = np.stack([ev((1-w)*zw8v+w*PV[k])-e0v for w in grid])
picks, gains = [], []
for r in range(20):
    a = np.isin(uid, rng.permutation(uu)[:len(uu)//2])
    for A, B in ((a, ~a), (~a, a)):
        j = D[:, A].mean(1).argmax(); picks.append(grid[j]); gains.append(D[j, B].mean())
print(f' picked w mean {np.mean(picks):.3f} sd {np.std(picks):.3f}; out-of-half gain mean {np.mean(gains):+.5f} sd over splits {np.std(gains):.5f}; full-val curve: ' + ' '.join(f'{w:.2f}:{d.mean():+.5f}' for w, d in zip(grid, D)))
print('== (d) slices of the fixed w=0.2 pop1kd admixture (the lead\'s W8P20 weight), TO level')
dv = ev(0.8*zw8v+0.2*PV[k])-e0v; dh = eh(0.8*zw8h+0.2*PH[k])-e0h
day = (fv['timestamp'].to_numpy().astype(np.int64)+10800)//86400
print(f' val all {f(dv)} | M<50 {f(dv[Mv<50])} M>=50 {f(dv[Mv>=50])} | ' + ' '.join(f'day{d} (n {int((day==d).sum())}) {f(dv[day==d])}' for d in np.unique(day)))
ftr = pl.read_parquet(FLAT/'train/feeds.parquet', columns=['timestamp', 'M', 'user_id'])
hidx = np.load(R/'nn_x/w12/recheck/kd_audit/hold_idx.npy'); assert (ftr['M'].to_numpy()[hidx] == Mh).all()
hts = ftr['timestamp'].to_numpy().astype(np.int64)[hidx]; hday = (hts+10800)//86400; hdow = (hday+3) % 7; DN = 'Mon Tue Wed Thu Fri Sat Sun'.split()
print(f' holdout all {f(dh)} | ' + ' '.join(f'{DN[d]} (n {int((hdow==d).sum())}) {f(dh[hdow==d])}' for d in range(7)))
wk = hdow >= 5
print(f' holdout weekend (public-like) {f(dh[wk])} | Mon+Tue (private weekdays) {f(dh[(hdow==0)|(hdow==1)])} | all weekdays {f(dh[~wk])}')
ud = np.unique(hday); late = hday >= np.quantile(hday, 0.8)
print(f' holdout days {ud.min()}..{ud.max()}; last 20% of days {f(dh[late])} first 80% {f(dh[~late])}')
# same slices for hard pop1 x8 in the G3 pool (C3 TO - G3 TO), the public-confirmed analogue
g3h, c3h = eh(poolh(G3M)), eh(poolh(G3M+POP1)); dc = c3h-g3h
print(f' [analogue] C3 TO - G3 TO holdout all {f(dc)} | weekend {f(dc[wk])} | Mon+Tue {f(dc[(hdow==0)|(hdow==1)])} | weekdays {f(dc[~wk])} | late20 {f(dc[late])}')
g3v = zv(np.load(P/'ens_G3_ft_val.npy'))*0.5+0.5*zv(np.load(P/'ens_G3_to_val.npy')); c3v = zv(np.load(P/'ens_Q2a8s3_ft_val.npy'))*0.5+0.5*zv(np.load(P/'ens_Q2a8s3_to_val.npy'))
dcv = ev(c3v)-ev(g3v)
print(f' [analogue] C3 - G3 final val {f(dcv)} | ' + ' '.join(f'day{d} {f(dcv[day==d])}' for d in np.unique(day)) + f' | M<50 {f(dcv[Mv<50])} M>=50 {f(dcv[Mv>=50])}')
print('== (e) final level val: per branch (1-w) z(W8 branch) + w z(pop1kd branch); FT branch = n2cpv3s3 OOF of the pop1kd seeds that exist')
ftv = zv(np.load(P/'ens_W8kd_ft_val.npy')); tov = zv(np.load(P/'ens_W8kd_to_val.npy')); fin0 = ev(0.5*ftv+0.5*tov)
have = [m for m in POP1KD if any((D_/f'nn_{m}_n2cpv3s3_val_oof.npy').exists() for D_ in (P, W, AB))]
pft = zv(np.mean([zv(np.load(ftpath(m, 'val'))) for m in have], 0)); pto = PV[k]
print(f' pop1kd FT members available: {len(have)} of 4; pop1kd FT pool val {ev(pft).mean():.5f}, TO pool {ev(pto).mean():.5f}')
for w in (0.1, 0.15, 0.2, 0.3):
    d = ev(0.5*zv((1-w)*ftv+w*pft)+0.5*zv((1-w)*tov+w*pto))-fin0
    d2 = ev(0.5*ftv+0.5*zv((1-w)*tov+w*pto))-fin0
    print(f'  w {w}: both branches {f(d)} (z {d.mean()/se(d):+.2f}; per-feed sd {d.std():.4f} -> SE of the paired PUBLIC delta {d.std()/np.sqrt(8562):.5f}) | TO branch only {f(d2)}')
d = ev(0.5*zv(0.8*ftv+0.2*pft)+0.5*zv(0.8*tov+0.2*pto))-fin0
print(f'  w 0.2 slices: M<50 {f(d[Mv<50])} M>=50 {f(d[Mv>=50])} | ' + ' '.join(f'day{dd} {f(d[day==dd])}' for dd in np.unique(day)))
# control at the final level: same recipe with narrow kd05 students (have FT) instead of pop1kd
cft = zv(np.mean([zv(np.load(ftpath(m, 'val'))) for m in E('kd05')], 0)); cto = PV['kd05 x4 (narrow students)']
d = ev(0.5*zv(0.8*ftv+0.2*cft)+0.5*zv(0.8*tov+0.2*cto))-fin0; print(f'  CONTROL w 0.2 with kd05 x4 (same view, both branches): {f(d)}')
c3ft = zv(np.load(P/'ens_Q2a8s3_ft_val.npy')); c3to = zv(np.load(P/'ens_Q2a8s3_to_val.npy'))
d = ev(0.5*zv(0.8*ftv+0.2*c3ft)+0.5*zv(0.8*tov+0.2*c3to))-fin0; print(f'  CONTROL w 0.2 with the C3 pool (28 hard members, both branches): {f(d)}')
g3ft = zv(np.load(P/'ens_G3_ft_val.npy')); g3to = zv(np.load(P/'ens_G3_to_val.npy'))
d = ev(0.5*zv(0.8*ftv+0.2*g3ft)+0.5*zv(0.8*tov+0.2*g3to))-fin0; print(f'  CONTROL w 0.2 with the G3 pool (20 hard members, no pop1): {f(d)}')
