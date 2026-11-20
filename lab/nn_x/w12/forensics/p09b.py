from fx import *
f5=np.load(FLAT/"val/cand_f5.npy"); y=np.load(FLAT/"val/cand_target.npy")
v,c=np.unique(f5,return_counts=True); o=np.argsort(-c)[:8]
print("top atoms",[(float(v[i]),int(c[i]),round(float(y[f5==v[i]].mean()),4)) for i in o]); print("min/max",f5.min(),f5.max(),"quantiles",np.quantile(f5,[.01,.1,.25,.5,.75,.9,.99]))
exec(open("p09_f5echo.py").read().split("print(\"f5 value stats")[0].split("from fx import *")[1])
exec("def later_feats"+open("p09_f5echo.py").read().split("def later_feats")[1].split("print(\"f5 value stats")[0])
g=later_feats(D["val"],pl.concat([D["val"],D["test"]]),"x")
r=g["row"].to_numpy(); now=g["f5"].to_numpy(); nxt=g["f5_next"].to_numpy(); gap=(g["ts_next"]-g["ts"]).to_numpy()/3600; yy=y[r]
atom=v[o[0]]; print("main atom",atom)
qs=np.quantile(f5,[0,.1,.3,.5,.7,.9,1.0])
print("f5_now bin | n | rate if next>now | rate if next<=now | lift")
for a,b in zip(qs[:-1],qs[1:]):
    m=(now>=a)&(now<=b)
    up=m&(nxt>now); dn=m&(nxt<=now)
    print("[%.3f,%.3f] %7d  up %.4f (n %6d)  notup %.4f (n %6d)  lift %.2f"%(a,b,m.sum(),yy[up].mean(),up.sum(),yy[dn].mean(),dn.sum(),yy[up].mean()/max(yy[dn].mean(),1e-9)))
# biggest jumps
d=nxt-now
for t in (0.05,0.2,0.5,1.0):
    m=d>t; print("delta>%.2f n %6d rate %.4f ; within 24h n %d rate %.4f"%(t,m.sum(),yy[m].mean(),(m&(gap<24)).sum(),yy[m&(gap<24)].mean() if (m&(gap<24)).any() else np.nan))
