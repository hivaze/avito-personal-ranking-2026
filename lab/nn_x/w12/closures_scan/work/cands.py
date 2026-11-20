import math
from scipy import stats
C = [ # line, label, val, se, hold, hold_se
 (585,'teacher-labelled held-out user fold, 4-pool',0.00069,0.00057,-0.00017,0.00032),
 (534,'pop-light members inside the teacher',0.00042,0.00032,None,None),
 (444,'Plackett-Luce aux target',0.00059,0.00045,0.00021,None),
 (444.1,'aux weight 1.0 (same row)',0.00051,0.00051,0.00026,None),
 (535,'fine-tune on stale inputs (val)',0.00022,0.00026,None,None),
 (535.1,'fine-tune on stale inputs (stale val)',0.00054,0.00028,None,None),
 (386,'category retrieval',0.00046,0.00049,0.0,None),
 (526,'A1 (val)',0.00017,0.00014,None,None),
 (474,'user-vector drift',0.00027,0.00028,-0.00008,0.00020),
 (482,'item prior union (residual)',0.00042,0.0003,None,None),
 (530,'input dropout by age (stale val)',0.00089,0.00062,-0.00189,0.00034),
 (378,'width 224',0.00026,0.00058,0.00016,0.00032),
 (557,'FT HPO confirm half',0.00055,0.00041,None,None),
]
for line,lab,v,s,h,hs in C:
    for D in (0.0003,):
        lr = stats.norm.pdf(v,D,s)/stats.norm.pdf(v,0,s)
        lrh = stats.norm.pdf(h,D,hs)/stats.norm.pdf(h,0,hs) if (h is not None and hs) else 1.0
        out=[]
        for pi in (0.05,0.14):
            o = pi/(1-pi)*lr*lrh; out.append(o/(1+o))
        print(f'{line}\t{lab}\tz={v/s:.2f}\tLR_val={lr:.2f}\tLR_hold={lrh:.2f}\tP(real|prior .05)={out[0]:.2f}\tP(real|prior .14)={out[1]:.2f}\tshrunk(x0.09)={v*0.09:+.5f}')
