"""What can the W8P20 public probe (E_kd_predeclared 20:16Z item A) resolve?  Read-only, builds no file.
W8P20 = per branch 0.8 z(W8 pool) + 0.2 z(pop1kd pool), final = 0.5 z(FT) + 0.5 z(TO). The FT branch uses the pop1kd fine-tunes that
exist at run time (2-4 of 4), so the val number here is INTERIM for the FT branch; the paired SE is what this script is for.
SE of the paired public delta = sd(per-feed NDCG delta on val) / sqrt(8562); private / sqrt(20033). (lb_stats uses the public M mix; the
plain val sd is within a few % of it.)  Declared thresholds: adopt at public >= W8 + 0.0003, close at <= W8 - 0.0003.
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/recheck/feasibility_value/s6_w8p20_probe.py"""
from math import erf, sqrt
from fv_common import *

P = Path('<repo>/preds'); A = Path('<repo>/nn_x/abl/preds')
ft, Mt, ot = feed_meta('test'); ot = ot.astype(np.int64)
zt = lambda x: feed_rank_z(ot, Mt, np.asarray(x, np.float32))[:, 1]
w_to_v, w_ft_v = np.load(P / 'ens_W8kd_to_val.npy'), np.load(P / 'ens_W8kd_ft_val.npy')
w_to_t, w_ft_t = np.load(P / 'ens_W8kd_to_test.npy'), np.load(P / 'ens_W8kd_ft_test.npy')
S = (951, 952, 953, 954); nm = lambda s: f'nn_abl_e10_pop1kd05t25_s{s}'
ftS = [s for s in S if (A / f'{nm(s)}_n2cpv3s3_val_oof.npy').exists() and (A / f'{nm(s)}_n2cpv3s3_test.npy').exists()]
print('pop1kd members with a finished n2cpv3s3 fine-tune:', ftS)
p_to_v = np.mean([zv(np.load(A / f'{nm(s)}_trainonly_val.npy')) for s in S], 0)
p_to_t = np.mean([zt(np.load(A / f'{nm(s)}_trainonly_test.npy')) for s in S], 0)
p_ft_v = np.mean([zv(np.load(A / f'{nm(s)}_n2cpv3s3_val_oof.npy')) for s in ftS], 0)
p_ft_t = np.mean([zt(np.load(A / f'{nm(s)}_n2cpv3s3_test.npy')) for s in ftS], 0)
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
w8_v = 0.5 * zv(w_ft_v) + 0.5 * zv(w_to_v); w8_t = 0.5 * zt(w_ft_t) + 0.5 * zt(w_to_t)
e8 = ev(w8_v); print(f'W8 final val {e8.mean():.5f} (log: 0.40165)')
ts = fv['timestamp'].to_numpy() if 'timestamp' in fv.columns else None
for w in (0.15, 0.2, 0.3):
    fv_ = 0.5 * zv((1 - w) * zv(w_ft_v) + w * zv(p_ft_v)) + 0.5 * zv((1 - w) * zv(w_to_v) + w * zv(p_to_v))
    ft_ = 0.5 * zt((1 - w) * zt(w_ft_t) + w * zt(p_ft_t)) + 0.5 * zt((1 - w) * zt(w_to_t) + w * zt(p_to_t))
    d = ev(fv_) - e8; sd = d.std(); sp, sr = sd / sqrt(8562), sd / sqrt(20033)
    big = Mv >= 50
    print(f'\nweight {w}: final val delta vs W8 {fmt(d)} (M>=50 {fmt(d[big])}, M<50 {fmt(d[~big])}); per-feed sd {sd:.4f}; '
          f'test corr to W8 final {np.corrcoef(ft_, w8_t)[0, 1]:.5f}')
    print(f'  SE of the paired public delta {sp:.5f}, private {sr:.5f}; the +-0.0003 thresholds are {0.0003 / sp:.2f} public SE')
    for true in (0.0, 0.0003, 0.0005):
        print(f'  true delta {true:+.4f}: P(public >= W8 + 0.0003) {1 - Phi((0.0003 - true) / sp):.2f} | P(public <= W8 - 0.0003) {Phi((-0.0003 - true) / sp):.2f} '
              f'| P(private file better than W8 on the private sample) {1 - Phi(-true / sr):.2f}')
    if ts is not None and w == 0.2:
        import datetime
        day = np.array([datetime.datetime.utcfromtimestamp(int(t) + 3 * 3600).weekday() for t in ts])  # MSK weekday
        for k in np.unique(day):
            print(f'  val weekday {k} (0 = Monday, MSK): n {int((day == k).sum())}, delta {fmt(d[day == k])}')
