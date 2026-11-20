"""Feature groups of the seq_ family (for evaluations)."""
from feat_hist import E_NAMES, C_NAMES
TRANS = [f"seq_tr_{g}_{s}" for g in ("cat", "c1024", "c4096") for s in ("lift_all", "lift", "mix", "last", "xlift")]
CV = ["seq_cv_invd20", "seq_cv_w5", "seq_cv_fwd5", "seq_cv_t1", "seq_cv_seqs", "seq_cv_pop", "seq_cv_norm"]
ICV = ["seq_icv_share4096", "seq_icv_lift4096", "seq_icv_lift1024", "seq_icv_tot"]
NXT = ["seq_nxt_cos"]
REL = [f"{c}_{k}" for c in ["seq_max_all", "seq_dcos_hl2", "seq_dcos_t1_hl5", "seq_cv_invd20", "seq_icv_lift4096",
                            "seq_tr_c4096_lift"] for k in ("rk", "z")]
GROUPS = dict(E=E_NAMES, C=C_NAMES, T=TRANS, CV=CV, ICV=ICV, N=NXT, R=REL)
if __name__ == "__main__":
    import sys
    gs = sys.argv[1].split("+")
    excl = sys.argv[2].split("+") if len(sys.argv) > 2 else []
    names = [n for g in gs for n in (GROUPS[g] if g in GROUPS else [g])]
    names = [n for n in names if not any(n in GROUPS.get(e, [e]) for e in excl)]
    print(",".join(dict.fromkeys(names)))
