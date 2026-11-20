from scipy import stats
import math, json
D=0.0003
def lr(v,s): return 1.0 if (v is None or not s) else stats.norm.pdf(v,D,s)/stats.norm.pdf(v,0,s)
S=[ # key, label, reads (list of (v,s)) used for the pool/file-level posterior; member-level reads listed separately
 ('428','R-Drop cooldown: 4-member pool (selected ckpt)',[(-0.00024,0.00046),(0.00036,0.00025)]),
 ('428m','R-Drop cooldown: member @last',[(0.00057,0.00030),(0.00052,0.00017)]),
 ('428s','SAM cooldown: 4-member pool (selected ckpt)',[(0.00033,0.00045),(0.00025,0.00023)]),
 ('587','U20 teacher-labelled fold: 4-pool',[(0.00069,0.00057),(-0.00017,0.00032)]),
 ('536','pop-light members in teacher: 16-pool val',[(0.00042,0.00032)]),
 ('387','unique slots: 10-ep pair',[(0.00036,0.00068),(0.00018,0.00041)]),
 ('388','category retrieval: 10-ep pair val',[(0.00046,0.00049)]),
 ('580','mixed teacher: seed-paired',[(-0.00030,0.00057),(0.00013,0.00032)]),
 ('580c','mixed teacher: seed-paired +0.0006 luck correction (upper)',[(0.00030,0.00057),(0.00013,0.00032)]),
 ('418','aux 1.0: students 4-pool (from scratch)',[(-0.00023,0.00055),(0.00028,0.00031)]),
 ('418m','aux 1.0 cooldown: click members, 7 roots',[(0.00038,0.00017),(0.00019,0.00009)]),
 ('446','Plackett-Luce target: members 2 seeds 2 ep (val only)',[(0.00059,0.00045)]),
 ('380','width 224: 4-pool + all-data file',[(0.00026,0.00058),(0.00016,0.00032),(0.00017,0.00025)]),
 ('537','fine-tune stale inputs: 8-member pool val',[(0.00022,0.00026)]),
]
out={}
for k,lab,rs in S:
    L=1.0
    for v,s in rs: L*=lr(v,s)
    p=[ (pi/(1-pi)*L)/(1+pi/(1-pi)*L) for pi in (0.05,0.14)]
    out[k]=dict(label=lab,LR=L,p05=p[0],p14=p[1])
    print(f'{k}\t{lab}\tLR {L:.2f}\tP(real | prior 0.05) {p[0]:.2f}\tP(real | prior 0.14) {p[1]:.2f}')
json.dump(out,open('short.json','w'),indent=1)
