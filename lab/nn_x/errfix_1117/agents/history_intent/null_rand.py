import hi, numpy as np
D = hi.Data()
rng = np.random.default_rng(123)
grid = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
oofs = []; zs = []
for i in range(40):
    g = hi.zfeed(rng.standard_normal(len(D.s)), D.off, D.sz)
    res = D.crossfit(lambda w: D.s + w * g, grid)
    oofs.append(res['oof']); zs.append(res['z'])
oofs = np.array(oofs); zs = np.array(zs)
print('null oof: mean %+.5f sd %.5f max %+.5f; z>1.96: %d/40; oof>=0.0003: %d/40' % (oofs.mean(), oofs.std(), oofs.max(), (zs > 1.96).sum(), (oofs >= 0.0003).sum()))
print(np.round(np.sort(zs), 2))
