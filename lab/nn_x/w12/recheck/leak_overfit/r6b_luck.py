"""R6b: is the realised W8's above-average reading a transferable property of its 8 members or witness-specific noise?
 (i) across random 8-subsets of the finished wide students: corr(subset val, subset holdout); and val-half A vs val-half B (disjoint users)
 (ii) member level over many 4-seed arms: corr of within-arm deviations val vs holdout, val half A vs half B, val vs label-free... """
from lo_common import *
rng = np.random.default_rng(11)
ext = [m for m in E('kd05w192', (955, 956, 957, 958))+E('kd05t25w192', (955, 956, 957, 958)) if (W/f'nn_{m}_trainonly_val.npy').exists() and (W/f'holdout/nn_{m}_holdout.npy').exists()]
allm = W8+ext; n = len(allm); print('wide students available:', n)
ZV = np.stack([LV(m) for m in allm]); ZH = np.stack([LH(m) for m in allm])
uid = fv['user_id'].to_numpy(); uu = np.unique(uid); A = np.isin(uid, rng.permutation(uu)[:len(uu)//2])
hA = rng.random(len(Mh)) < 0.5
subs = [tuple(sorted(rng.choice(n, 8, replace=False))) for _ in range(200)]; subs = list(dict.fromkeys(subs))
rv, rh, rva, rvb, rha, rhb = [], [], [], [], [], []
for s in subs:
    e1 = ev(ZV[list(s)].mean(0)); e2 = eh(ZH[list(s)].mean(0))
    rv.append(e1.mean()); rh.append(e2.mean()); rva.append(e1[A].mean()); rvb.append(e1[~A].mean()); rha.append(e2[hA].mean()); rhb.append(e2[~hA].mean())
c = lambda a, b: np.corrcoef(a, b)[0, 1]
print(f' {len(subs)} distinct 8-subsets of {n}: sd val {np.std(rv):.5f} hold {np.std(rh):.5f} | corr(val, holdout) {c(rv, rh):+.2f} | corr(val half A, val half B) {c(rva, rvb):+.2f} | corr(hold half A, hold half B) {c(rha, rhb):+.2f}')
w8 = tuple(range(8)); iv = ev(ZV[:8].mean(0)); ih = eh(ZH[:8].mean(0))
print(f' W8 percentile among the subsets: val {np.mean(np.array(rv) < iv.mean()):.2f} hold {np.mean(np.array(rh) < ih.mean()):.2f}')
print(f' realised W8: val {iv.mean():.5f} (half A {iv[A].mean():.5f}, half B {iv[~A].mean():.5f}) hold {ih.mean():.5f}; subset means val {np.mean(rv):.5f} hold {np.mean(rh):.5f}')
# regression of holdout on val across subsets -> expected "true" excess of W8 given its val excess
b = np.polyfit(np.array(rv)-np.mean(rv), np.array(rh)-np.mean(rh), 1)[0]
print(f' slope of subset holdout excess on subset val excess {b:+.2f}; W8 val excess {iv.mean()-np.mean(rv):+.5f} -> predicted holdout excess {b*(iv.mean()-np.mean(rv)):+.5f} (observed {ih.mean()-np.mean(rh):+.5f})')
print('== member level, within-arm deviations (arms with >= 4 seeds on both witnesses)')
arms = ['base', 'kd03', 'kd05', 'kd08', 'kd10', 'kd05t25', 'kd05w192', 'kd05t25w192', 'kd05g2w192', 'kd03g2w192', 'kx05', 'kx08']
dv, dh, dva, dvb = [], [], [], []
for a in arms:
    ms = [m for m in E(a, (951, 952, 953, 954, 955, 956, 957, 958)) if (W/f'nn_{m}_trainonly_val.npy').exists() and (W/f'holdout/nn_{m}_holdout.npy').exists()]
    if len(ms) < 4: continue
    v = [ev(LV(m)) for m in ms]; h = np.array([eh(LH(m)).mean() for m in ms])
    vm = np.array([x.mean() for x in v]); va = np.array([x[A].mean() for x in v]); vb = np.array([x[~A].mean() for x in v])
    dv += list(vm-vm.mean()); dh += list(h-h.mean()); dva += list(va-va.mean()); dvb += list(vb-vb.mean())
print(f' {len(dv)} members in {len(arms)} arms: sd of member deviation val {np.std(dv):.5f} hold {np.std(dh):.5f} | corr(val dev, holdout dev) {c(dv, dh):+.2f} | corr(val half A dev, val half B dev) {c(dva, dvb):+.2f}')
