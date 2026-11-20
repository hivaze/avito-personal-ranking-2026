import ast, re, itertools
R="<repo>/nn_x/w12/kd/"
def load(path):
    src=open(path).read(); tree=ast.parse(src)
    ns={}
    for n in tree.body:
        seg=ast.get_source_segment(src,n)
        if isinstance(n, ast.FunctionDef) and n.name=="spec": exec(seg, ns)
        elif isinstance(n, ast.Assign) and any(getattr(t,'id',None) in ("S8","S4","wide","rc","P1d","P1w","FD16","FDP","FDPd","FD8","fdx") for t in n.targets): exec(seg, ns)
        elif isinstance(n, ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("S8","P1d","FD16","FDPd","FD8"): exec(seg, ns)
    return ns, src
new, snew = load(R+"build_r4.py"); old, sold = load(R+"build_r4_pre_20261116T054417Z.py")
# both files: statements on one line separated by ';' -> exec the first lines directly
def load2(path):
    src=open(path).read().split("\n")
    i0=[i for i,l in enumerate(src) if l.startswith("S8 = ")][0]; i1=[i for i,l in enumerate(src) if l.startswith("fv, Mv, ov")][0]
    ns={}; exec("\n".join(src[i0:i1]), ns); return ns
new=load2(R+"build_r4.py"); old=load2(R+"build_r4_pre_20261116T054417Z.py")
keys=sorted(set(re.findall(r"k == '([^']+)'", snew)))
pref=sorted(set(re.findall(r"k\.startswith\('([^']+)'\)", snew)))
print("exact keys in new:", len(keys), "| prefix keys:", pref)
okeys=sorted(set(re.findall(r"k == '([^']+)'", sold)))
print("new exact keys not in old:", sorted(set(keys)-set(okeys)), "| old not in new:", sorted(set(okeys)-set(keys)))
probe=list(okeys)+[p+t for p in pref for t in ("hz","hz2","xd30","th")]
same=0
for k in probe:
    a=new["spec"](k); b=old["spec"](k)
    assert a==b, k
    same+=1
print("pre-existing keys with identical spec() old/new:", same, "of", len(probe))
try:
    old["spec"]("FDN16P20"); print("OLD KNOWS FDN16P20 ?!")
except SystemExit as e: print("old: FDN16P20 ->", e)
a=new["spec"]("FDN16P20"); b=new["spec"]("FD16P20")
print("FDN16P20 base", len(a[0]), "pop", a[1], "w", a[2])
print("pop equal:", a[1]==b[1], "| w equal:", a[2]==b[2], "| base differs in all 16:", all(x!=y for x,y in zip(a[0],b[0])), "| mapping w5_fd10_ -> abl_fd10_nx3 exact:", [x.replace("w5_fd10_","abl_fd10_nx3") for x in b[0]]==a[0])
print(a[0])
# everything outside the spec line identical
import difflib
d=[l for l in difflib.unified_diff(sold.split("\n"), snew.split("\n"), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++","---")]
print("diff lines:", len(d), [x[:40] for x in d])
