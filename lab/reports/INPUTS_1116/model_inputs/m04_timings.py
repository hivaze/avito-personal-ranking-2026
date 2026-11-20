"""INVENTORY A, measurement M4 (CPU, read-only): wall-clock minutes per run from the run json / FT json on record
(field train_minutes / minutes; processes ran on GPUs shared by up to 4 jobs, so these are process-hours, upper bounds
of exclusive GPU time). Writes out/m04_timings.txt."""
import json, glob, os, statistics as st
R = "<repo>/"; W = R + "nn_x/w5/logs/"; A = R + "nn_x/abl/logs/"
out = []
def show(tag, pat, key="train_minutes"):
    v = []
    for f in sorted(glob.glob(pat)):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        if j.get(key) is not None:
            v.append(j[key])
    if not v:
        out.append(f"{tag}: none [{pat.replace(R, '')}]"); return
    out.append(f"{tag}: n={len(v)} min {min(v):.1f} median {st.median(v):.1f} max {max(v):.1f} minutes; sum {sum(v) / 60:.1f} process-h  [{pat.replace(R, '')}]")
show("held-out wide student, teacher_mean11 (w5 harness, d192, 10 ep)", W + "w5_e10_kd05w192_s95[1-8].json")
show("held-out wide student, teacher_mean25 (w5 harness, d192, 10 ep)", W + "w5_e10_kd05t25w192_s95[1-8].json")
show("held-out wide pop1 student (abl harness, d192, 10 ep)", A + "abl_e10_pop1kd05w192_s95[1-4].json")
show("held-out narrow student d128 (w5)", W + "w5_e10_kd05_s95[1-8].json")
show("held-out narrow pop1 student d128 (abl)", A + "abl_e10_pop1kd05t25_s95[1-4].json")
show("full-data wide student (stage F, mean11)", W + "w5_fd10_kd05w192_s95[1-8].json")
show("full-data wide student (stage F, mean25)", W + "w5_fd10_kd05t25w192_s95[1-8].json")
show("full-data wide pop1 student (stage P)", A + "abl_fd10_pop1kd05w192_s9[5-6][0-9].json")
show("full-data wide popularity-free student", A + "abl_fd10_pfkd05w192_s95[1-8].json")
show("hard-label base d128 10 ep (w5_e10_base)", W + "w5_e10_base_s95[1-6].json")
show("hard-label xdrop_pop d128 10 ep (abl)", A + "abl_e10_xdrop_pop_s95[1-4].json")
show("hard-label xdrop_pre2 d128 10 ep (abl)", A + "abl_e10_xdrop_pre2_s95[1-2].json")
show("hard-label pop1 d128 10 ep (abl)", A + "abl_e10_pop1_s95[1-8].json")
show("hard-label pop2 d128 10 ep (abl)", A + "abl_e10_pop2_s95[1-4].json")
show("hard-label xrel_more d128 10 ep (abl)", A + "abl_e10_xrel_more_s95[1-2].json")
show("2-epoch hard-label screen, base (abl)", A + "abl_base_s2[34].json")
show("2-epoch hard-label screen, pop1 (abl)", A + "abl_pop1_s2[34].json")
show("2-epoch hard-label screen, time (abl)", A + "abl_time_s2[34].json")
show("re-cool harness K:ref (cooldown only, d192)", W + "w5_kcd_ref_s95[1-8].json")
show("re-cool harness L:ref", W + "w5_lcd_ref_s95[1-8].json")
show("smoke run, w5 full-data lane", W + "w5_fdsmoke*_s951.json")
show("smoke run, abl full-data lane", A + "abl_fdsmoke*_s951.json")
show("fine-tune n2cpv3s3, held-out wide students (w5)", W + "w5_e10_kd05*w192_s95[1-8]_n2cpv3s3_ft.json", "minutes")
show("fine-tune n2cpv3s3, held-out wide pop1 students (abl)", A + "abl_e10_pop1kd05w192_s95[1-4]_n2cpv3s3_ft.json", "minutes")
show("fine-tune n2cpv3s3, full-data wide students (w5)", W + "w5_fd10_kd05*w192_s95[1-8]_n2cpv3s3_ft.json", "minutes")
for v in ("nn_x/abl/logs/abl_fdsmoke_10_pop1kd05w192_s951", "nn_x/w5/logs/w5_fdsmoke_10_kd05w192_s951"):
    t0 = open(R + v + ".log").readline()[5:29]; t1 = os.path.getmtime(R + v + ".json")
    import time, calendar
    s0 = calendar.timegm(time.strptime(t0.strip(), "%a %b %d %H:%M:%S %Y"))
    out.append(f"smoke wall-clock {v}: log start {t0.strip()} UTC -> json written {time.strftime('%H:%M:%S', time.gmtime(t1))} UTC = {(t1 - s0) / 60:.1f} min from the args line (logged before the data load) to the json: whole smoke job incl. data load, 73 steps, val and test prediction")
open(R + "reports/INPUTS_1116/model_inputs/out/m04_timings.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
