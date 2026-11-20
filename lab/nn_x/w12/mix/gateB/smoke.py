from gate import *
y = labels().astype(np.float32); f, M, off = meta('val'); g = groups()
q = pools4('val', None, None, tag='FD16P4'); E = tensor(q, y, tag='FD16P4'); X = features('val', q); ctx = Ctx(E, X, q, y)
tr = np.arange(len(M))[folds(g, 0, 5) != 0]
t = time.time(); print('const2', ctx.const2(tr)[2], 'constp', ctx.constp(tr)[2], 'mstep', ctx.mstep(tr)[2], f'{time.time() - t:.2f}s')
t = time.time(); print('stump1', ctx.stump1(tr)[2], f'{time.time() - t:.2f}s')
t = time.time(); print('pstump', ctx.pstump(tr)[2], f'{time.time() - t:.2f}s')
for kind in ('LINM', 'LINp', 'LIN2'):
    ctx.lin(tr, kind, 1e-3)
    for lam in (3e-5, 3e-3):
        t = time.time(); wp, wf, d = ctx.lin(tr, kind, lam); dt = time.time() - t
        print(kind, lam, f'{dt:.2f}s', 'wp', wp.min().round(3), wp.mean().round(3), wp.max().round(3), 'wf', wf.min().round(3), wf.mean().round(3), wf.max().round(3), 'train exact', ctx.exact(wp, wf)[tr].mean().round(6), 'a', np.round(d['a'], 2), 'b', np.round(d['b'], 2))
# gradient check of lin_obj against finite differences of the interpolated objective
Xs, _ = ctx.prep(tr); rng = np.random.default_rng(0); th = rng.normal(size=18) * 0.3; d = 8
def obj(th): return lin_obj(ctx.Es, tr, Xs, th[0], th[1:d + 1], th[d + 1], th[d + 2:])[0].mean()
val, gp, gf = lin_obj(ctx.Es, tr, Xs, th[0], th[1:d + 1], th[d + 1], th[d + 2:]); an = np.concatenate([[gp.mean()], Xs[tr].T @ gp / len(tr), [gf.mean()], Xs[tr].T @ gf / len(tr)])
num = np.array([(obj(th + 1e-5 * np.eye(18)[k]) - obj(th - 1e-5 * np.eye(18)[k])) / 2e-5 for k in range(18)]); print('grad check max rel err', np.abs(an - num).max() / np.abs(num).max())
t = time.time(); x = ctx.exact(np.full(len(M), 0.2), np.full(len(M), 0.5)); print('exact pass', f'{time.time() - t:.3f}s', x.mean(), 'tensor', E[:, IP[0.2], J5].mean())
