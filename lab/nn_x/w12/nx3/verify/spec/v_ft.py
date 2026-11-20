import ast, re, sys, subprocess, shlex, os
R="<repo>"
def parser_of(path):
    src=open(path).read(); tree=ast.parse(src)
    fn=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=="parse"][0]
    ns={"argparse":__import__("argparse")}; exec(ast.get_source_segment(src, fn), ns); return ns["parse"]
def parse_with(path, argv):
    old=sys.argv; sys.argv=["x"]+argv
    try: return vars(parser_of(path)())
    finally: sys.argv=old
def logged(log):
    out=[]
    for ln in open(log, errors="replace"):
        m=re.match(r"^==== (.*?) args: (\{.*\})\s*$", ln)
        if m: out.append((m.group(1), ast.literal_eval(m.group(2))))
    return out
bad=0
for arm in ("nx3kd05w192","nx3kd05t25w192"):
    for seed in range(951,959):
        v=f"abl_fd10_{arm}_s{seed}"
        o=subprocess.run(["bash", f"{R}/nn_x/abl/ftbag/ft_one_s3_r6.sh", v], capture_output=True, text=True, env={**os.environ,"DRY":"1","MODE":"abl","CUDA_VISIBLE_DEVICES":"0"})
        line=o.stdout.strip(); m=re.match(r"^GPU N: \(cd (\S+) &&\) (\S+) finetune\.py (.*)$", line); assert m, (line, o.stderr)
        assert m.group(1)==f"{R}/nn_x/abl/popfree", m.group(1)
        new=parse_with(f"{R}/nn_x/abl/popfree/finetune.py", shlex.split(m.group(3)))
        refv=f"w5_fd10_{arm.replace('nx3','',1)}_s{seed}"
        la=logged(f"{R}/nn_x/w5/logs/{refv}_n2cpv3s3_ft.log"); assert len(la)==1, (refv, len(la))
        ref=la[0][1]
        d={k:(new.get(k,'<absent>'),ref.get(k,'<absent>')) for k in sorted(set(new)|set(ref)) if new.get(k,'<absent>')!=ref.get(k,'<absent>')}
        exp={"variant":(v,refv),"out":(v+"_n2cpv3s3",refv+"_n2cpv3s3"),"xt_check":("on","<absent>")}
        ok = d==exp
        bad += (not ok)
        if seed==951 or not ok: print(v,"vs",refv,"FT args:", "OK (only variant, out, xt_check)" if ok else f"BAD {d}", "| keys", len(new), len(ref)); 
        if seed==951: print("   DRY:", line)
# same-harness precedent
la=logged(f"{R}/nn_x/abl/logs/abl_fd10_pop1kd05w192_s951_n2cpv3s3_ft.log")
print("precedent abl_fd10_pop1kd05w192_s951 FT args lines", len(la), {k:v for k,v in la[0][1].items() if k in ("xt_check","epochs","seeds","folds","extra","lr","ema","decay","ema_start","u0_singleton","select")})
print("bad", bad)
