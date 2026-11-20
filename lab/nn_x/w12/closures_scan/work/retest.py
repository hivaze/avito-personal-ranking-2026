# Re-test track record: ideas whose first read was positive and that were later read again in a stronger
# or final form on validation (same quantity). Numbers copied from docs/07-what-failed.md (line given).
import numpy as np, json
W = '<repo>/lab/nn_x/w12/closures_scan/'
# (idea, first val, first se, first form, later val, later se, later form, docs07 line, same_regime)
R = [
 ('Unique retrieval slots', 0.00117, 0.00047, '2-epoch screen (z 2.48)', 0.00036, 0.00068, '10-epoch pair', 387, 1),
 ('Popularity-free second view', 0.00036, 0.00019, 'first 4 seeds, file', -0.00024, 0.00018, '4 fresh seeds, file', 531, 1),
 ('Two-step walk input (nx3)', 0.00037, 0.00056, '8-seed click-label pool', 0.00011, 0.00038, 'all-data 16-student file', 487, 0),
 ('Per-feed weight M^0.5', 0.00036, 0.00034, '16-pool cooldown re-runs', -0.00002, 0.00027, 'all-data file vs matched control', 458, 1),
 ('Stale-view cooldown', 0.00024, 0.00038, '16-pool', -0.00047, 0.00035, 'all-data 16-member file', 534, 1),
 ('Width 224 students', 0.00026, 0.00058, '4-pool (seed-paired)', 0.00011, 0.00035, 'all-data files (+0.00005..+0.00017, midpoint)', 380, 1),
 ('Feed-size rule, 0.4 on long feeds', 0.00051, 0.00021, 'in-sample on validation', 0.00002, 0.00027, 'nested by user', 524, 1),
 ('Forward echo of f5', 0.00038, 0.00024, 'nested', -0.00015, 0.00038, 'nested re-read on pick 1', 507, 1),
 ('Fine-tune HPO (35 trials)', 0.00055, 0.00041, 'confirm half', -0.00009, None, 'second member set', 559, 1),
 ('12,000 random-order feeds in fine-tune', 0.0005, None, 'first generation (7 of 8 pairs)', -0.00017, None, 'second generation (mean of 3)', 562, 1),
 ('Pseudo-feeds inside students', 0.00092, 0.00060, '2-seed member screen', 0.00054, 0.00047, '8-seed pool', 588, 1),
 ('Encoder scores, quick leak-free build', 0.0011, 0.0007, '3 epochs', -0.0003, 0.0008, 'final members after fine-tune', 640, 0),
 ('Aux-loss weight 1.0', 0.00032, 0.00020, 'cooldown members (click labels)', -0.00023, 0.00055, 'students 4-pool', 418, 0),
 ('Mixture of experts', 0.00027, 0.00047, 'click-label members, 2 epochs', 0.00006, 0.00047, 'students 8-pool (19 Nov)', 370, 0),
 ('Self-distillation -> ensemble distillation', 0.00064, 0.00041, 'one 2-epoch teacher', 0.00257, 0.00047, '11-member teacher, members', 574, 0),
 ('Width 192: click labels -> students', 0.00021, 0.00048, 'click-label re-test arm', 0.00049, None, 'students, 8 wide (holdout +0.00107)', 378, 0),
]
same = [r for r in R if r[8] == 1]
allr = R
def summ(rs, tag):
    f = np.array([r[1] for r in rs]); l = np.array([r[4] for r in rs])
    print(f'{tag}: n={len(rs)} mean first {f.mean():+.5f}  mean later {l.mean():+.5f}  ratio {l.mean()/f.mean():.2f}  later>0: {(l>0).sum()}  later>=+0.0003: {(l>=0.0003).sum()}  later>=half of first: {(l>=0.5*f).sum()}')
    return dict(n=len(rs), mean_first=float(f.mean()), mean_later=float(l.mean()), pos=int((l > 0).sum()), ge3=int((l >= 0.0003).sum()), half=int((l >= 0.5 * f).sum()))
out = dict(same=summ(same, 'same regime'), all=summ(allr, 'all re-tests'))
# SE of the mean of later reads (rows with SE; others assumed 0.0004)
ses = np.array([r[5] if r[5] else 0.0004 for r in same])
print('SE of mean later read (same regime):', np.sqrt((ses**2).sum())/len(ses))
out['se_mean_later_same'] = float(np.sqrt((ses**2).sum())/len(ses))
out['rows'] = R
json.dump(out, open(W + 'work/retest.json', 'w'), indent=1)
