# The two input flags of the ablation screen. --xt_exclude removes named columns from the preset of 34
# pre-computed inputs (the popularity-free view removes the 8 popularity columns); --xt_rel_extra adds
# within-feed rank and z channels for named columns. A later edit added --xt_add, which appends one
# column (the exposure count of the popularity-light view).
# Imported by lab/nn_x/abl/train.py; with the flags off the trainer runs the previous code path
# (proved by lab/nn_x/abl/tests/test_abl.py). It is not run on its own.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/abl/abl_arms.py
#
"""nn_x/abl ablation-screen input flags (nn_x/abl/ledger/ledger.md section 4, arms xdrop_pop / xdrop_ctxnd / xrel_more),
used by nn_x/abl/train.py. Written 2026-11-13 ~00:05 UTC. Both flags are OFF at the train.py default '' and train.py
calls these helpers only behind `if a.xt_exclude:` / `if a.xt_rel_extra:`, so a flag-off run executes exactly the
nn_x/w5 code path (tests/test_abl.py: AST proof against tests/ref_train_w5.py.txt + end-to-end batch / logit / log /
checkpoint identity). Imports nothing but the standard library, so the unit tests call the very functions train.py uses.

  --xt_exclude n1,n2,...  (xdrop_* arms)  train.py builds xt_names = XT_PRESETS[--xt] and then, with this flag, keeps
        only the columns NOT named here. Everything downstream is derived from the filtered list exactly as before:
        XT_REL / XT_RELONLY entries are kept only if `n in xt_names` (so an excluded column loses its rank / z channels
        too), xt_idcols enumerates the filtered list, n_dense = N_DENSE (+2 user0) + len(xt_names) + 2 len(xt_rel)
        - len(xt_relonly), SplitData.attach_xt loads only the kept columns (same per-column transform) and the model
        cfg stores the filtered xt_names / indices, so holdout_eval.py and stale_pred.py reproduce the inputs from the
        checkpoint. Unknown or duplicated names, or a list that empties the preset, are errors (no silent no-op).
  --xt_rel_extra n1,...   (xrel_more arm)  appends the preset indices of the named columns to xt_rel (after XT_REL's
        own entries, in the given order); model.dense_features then adds [pct-rank, z-score] within the feed for each
        of them, after the ID-input dropout and before the xt_relonly removal, exactly as for the XT_REL columns.
        Unknown names, duplicates, and names already in XT_REL[preset] are errors.
  --xt_add n1,n2,...      (pop1 arm; added 2026-11-13 ~06:10 UTC, ledger section 7)  train.py appends the named feature
        columns LAST to xt_names, right after --xt_exclude and before the feat_exists miss check (so a missing file fails
        loudly). The XT_REL / XT_RELONLY lookups stay keyed on the preset names and xt_idcols on the seq_cv / it_hist
        prefixes, so the indices of every existing column and of its rank / z channels are unchanged and n_dense grows by
        exactly len(names). The appended columns get attach_xt's fixed per-name transform (e.g. the default clip +-10).
        Errors (no silent no-op): no --xt preset / list, empty or duplicated names, a name already in the (filtered) list,
        a seq_cv* / it_hist* name (would join xt_idcols / the count transform), or a name of XT_REL / XT_RELONLY[preset]
        (passed as `reserved`; would re-enter a rank / z channel). Train.py calls it only behind `if a.xt_add:`.
"""


def _names(spec):
    out = [n.strip() for n in spec.split(",") if n.strip()]
    assert out, "empty name list"
    assert len(set(out)) == len(out), f"duplicated names in {spec!r}"
    return out


def xt_exclude_names(xt_names, spec, log):
    """xt_names minus the columns named in `spec` (comma list); every name must be in xt_names."""
    assert xt_names, "--xt_exclude needs an --xt preset / list"
    drop = _names(spec)
    unknown = [n for n in drop if n not in xt_names]
    assert not unknown, f"--xt_exclude: not in the --xt list: {unknown}"
    kept = [n for n in xt_names if n not in drop]
    assert kept, "--xt_exclude would drop every xt column (use --xt '' instead)"
    log(f"abl xt_exclude: dropped {len(drop)} of {len(xt_names)} xt columns {drop}; {len(kept)} kept")
    return kept


def xt_rel_extend(xt_rel, xt_names, spec, log):
    """xt_rel (list of preset indices or None) + the indices of the columns named in `spec`, in order."""
    assert xt_names, "--xt_rel_extra needs an --xt preset / list"
    extra = _names(spec)
    unknown = [n for n in extra if n not in xt_names]
    assert not unknown, f"--xt_rel_extra: not in the --xt list: {unknown}"
    cur = list(xt_rel or [])
    idx = [xt_names.index(n) for n in extra]
    dup = [n for n, i in zip(extra, idx) if i in cur]
    assert not dup, f"--xt_rel_extra: already rank/z columns of this preset: {dup}"
    log(f"abl xt_rel_extra: within-feed rank/z also for {extra} (preset indices {idx}; xt_rel {cur} -> {cur + idx}; "
        f"+{2 * len(idx)} dense inputs)")
    return cur + idx


def xt_add_names(xt_names, spec, log, reserved=()):
    """xt_names + the columns named in `spec` (comma list), appended LAST in the given order (pop1 arm)."""
    assert xt_names, "--xt_add needs an --xt preset / list"
    add = _names(spec)
    dup = [n for n in add if n in xt_names]
    assert not dup, f"--xt_add: already in the --xt list: {dup}"
    idc = [n for n in add if n.startswith(("seq_cv", "it_hist"))]
    assert not idc, f"--xt_add: seq_cv* / it_hist* names would join xt_idcols and the count transform: {idc}"
    res = [n for n in add if n in set(reserved)]
    assert not res, f"--xt_add: XT_REL / XT_RELONLY names of the preset (would re-enter a rank / z channel): {res}"
    out = list(xt_names) + add
    log(f"abl xt_add: appended {len(add)} xt columns {add} after the {len(xt_names)} kept (indices "
        f"{list(range(len(xt_names), len(out)))}; +{len(add)} dense inputs)")
    return out


def xt_add_relonly_idx(xt_rel, xt_relonly, xt_names, add_spec, spec, log):
    """ENCODER-DOSE pre2noid arm (ledger section 10.A; added 2026-11-14 ~00:10 UTC): (xt_rel + idx, xt_relonly + idx) for the
    --xt_add columns named in `spec`, i.e. the added columns get within-feed rank / z channels (appended after the kept XT_REL
    entries, in the given order) and their RAW column is removed, exactly the XT_RELONLY treatment of the preset's own
    pre2_fe_s / pre2_new_s / pre2_hmax32. train.py calls it only behind `if a.xt_add_relonly:`, after the XT_REL / XT_RELONLY
    lookups (and after --xt_rel_extra). Errors (no silent no-op): no --xt_add, a name not in --xt_add, duplicates, an index
    already in xt_rel or xt_relonly."""
    assert add_spec, "--xt_add_relonly needs --xt_add"
    names = _names(spec)
    added = _names(add_spec)
    bad = [n for n in names if n not in added]
    assert not bad, f"--xt_add_relonly: not in --xt_add: {bad}"
    idx = [xt_names.index(n) for n in names]
    cur_rel, cur_ro = list(xt_rel or []), list(xt_relonly or [])
    dup = [n for n, i in zip(names, idx) if i in cur_rel or i in cur_ro]
    assert not dup, f"--xt_add_relonly: already rank/z or rel-only columns: {dup}"
    log(f"abl xt_add_relonly: rank/z-only for the added {names} (indices {idx}; xt_rel {cur_rel} -> {cur_rel + idx}; "
        f"xt_relonly {cur_ro} -> {cur_ro + idx}; net dense inputs +{len(idx)})")
    return cur_rel + idx, cur_ro + idx
