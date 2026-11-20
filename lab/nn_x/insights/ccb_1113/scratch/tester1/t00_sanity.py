from h import *
H = Harness()
rng = np.random.default_rng(0)
xv = rng.standard_normal(len(H.y)); xh = rng.standard_normal(len(H.yh))
H.run("noise0_sanity", [xv], [xh], primary=False)
