"""The validation tools: calibration, the private-mix view, the gate, the freeze-day check."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from avito_ranker.ensemble.build import build_ensemble, missing_member_files, write_ensemble
from avito_ranker.ensemble.config import BRANCHES, SCORED_SPLITS, load_ensemble_config, pool_file, submission_file
from avito_ranker.ensemble.feeds import FeedIndex, load_feed_index
from avito_ranker.paths import path
from avito_ranker.validation import private_mix
from avito_ranker.validation.calibration import SIZE_BIN_EDGES, predict_board, public_mask, size_weights
from avito_ranker.validation.freeze import candidate_table, check_candidate, file_sha256, recorded_gate_output
from avito_ranker.validation.gate import (
    GateReport,
    MemberLogs,
    audit_inputs,
    check_file,
    check_parity,
    check_reproduction,
    compare_on_val,
    per_feed_spearman,
    run_gate,
    users_with_later_feed,
)
from support import ENSEMBLE_DIR


def make_feeds(sizes, users=None, timestamps=None) -> FeedIndex:
    """A feed index for synthetic feeds; users and timestamps default to distinct values."""
    sizes = np.asarray(sizes, np.int64)
    n = len(sizes)
    return FeedIndex.from_table(pl.DataFrame({
        "feed_id": np.arange(n),
        "user_id": np.arange(1, n + 1) if users is None else np.asarray(users),
        "timestamp": np.arange(n) if timestamps is None else np.asarray(timestamps),
        "M": sizes,
        "cand_off": np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64),
    }))


def built_tiny(tiny_repo):
    """Build and store the synthetic ensemble; return its recipe."""
    config = load_ensemble_config(tiny_repo.config_path)
    write_ensemble(config, build_ensemble(config))
    return config


# ---------------------------------------------------------------- calibration

def test_size_weights_move_val_to_the_feed_size_mix_of_the_part(tiny_repo):
    val, test = load_feed_index("val"), load_feed_index("test")
    public = public_mask(test.table["timestamp"].to_numpy())
    assert 0 < public.sum() < len(public)
    n_bins = len(SIZE_BIN_EDGES) + 1
    val_bin = np.searchsorted(SIZE_BIN_EDGES, val.sizes, side="right")
    for part, in_part in (("public", public), ("private", ~public)):
        weights = size_weights(part)
        assert weights.mean() == pytest.approx(1.0)
        reweighted = np.bincount(val_bin, weights=weights, minlength=n_bins) / weights.sum()
        target = np.bincount(np.searchsorted(SIZE_BIN_EDGES, test.sizes[in_part], side="right"), minlength=n_bins)
        np.testing.assert_allclose(reweighted, target / in_part.sum(), atol=1e-12)


def test_forecast_of_a_constant_is_the_constant(tiny_repo):
    forecast = predict_board(np.full(len(tiny_repo.val_sizes), 0.4))
    assert forecast == {"public": pytest.approx(0.4), "private": pytest.approx(0.4)}


# ---------------------------------------------------------------- private-mix view

def test_cells_are_closed_on_the_right():
    sizes = np.array([5, 10, 11, 25, 26, 50, 51, 100, 101, 200, 201, 300])
    cells = private_mix._cell(sizes, np.zeros(len(sizes)))
    np.testing.assert_array_equal(cells // 4, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    cold = np.array([0.0, 0.15, 0.16, 0.30, 0.31, 0.45, 0.46, 1.0])
    np.testing.assert_array_equal(private_mix._cell(np.full(8, 5), cold) % 4, [0, 0, 1, 1, 2, 2, 3, 3])


def test_cold_share_counts_items_absent_from_train(tiny_repo):
    np.testing.assert_allclose(private_mix.cold_share("val"), tiny_repo.val_cold_share)
    np.testing.assert_allclose(private_mix.cold_share("test"), tiny_repo.test_cold_share)


def test_mix_weights_move_val_to_the_cell_mix_of_the_part(tiny_repo):
    val, test = load_feed_index("val"), load_feed_index("test")
    val_cell = private_mix._cell(val.sizes, tiny_repo.val_cold_share)
    test_cell = private_mix._cell(test.sizes, tiny_repo.test_cold_share)
    public = public_mask(test.table["timestamp"].to_numpy())
    weights = private_mix.mix_weights()
    for part, in_part in (("public", public), ("private", ~public)):
        assert weights[part].mean() == pytest.approx(1.0)
        reweighted = np.bincount(val_cell, weights=weights[part], minlength=24) / weights[part].sum()
        np.testing.assert_allclose(reweighted, np.bincount(test_cell[in_part], minlength=24) / in_part.sum(),
                                   atol=1e-12)


def test_mix_weights_refuse_a_test_cell_without_val_support(tiny_repo):
    # Make every val item warm: val then has no feed in the cold cells that test occupies.
    val_items = np.load(tiny_repo.flat / "val/cand_item_id.npy")
    np.save(tiny_repo.flat / "train/cand_item_id.npy", val_items)
    with pytest.raises(ValueError, match="without val support"):
        private_mix.mix_weights()


def test_weighted_mean_and_standard_error():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    plain = private_mix.weighted_mean_se(x)
    assert plain.mean == pytest.approx(x.mean()) and plain.se == pytest.approx(x.std() / np.sqrt(500))
    uniform = private_mix.weighted_mean_se(x, np.ones(500))
    assert uniform.mean == pytest.approx(plain.mean) and uniform.se == pytest.approx(plain.se)
    # Weight on half of the feeds only: the estimate is that of the half.
    half = private_mix.weighted_mean_se(x, np.r_[np.ones(250), np.zeros(250)])
    assert half.mean == pytest.approx(x[:250].mean())
    assert half.se == pytest.approx(x[:250].std() / np.sqrt(250))


def test_compare_reads_the_same_difference_three_ways(tiny_repo):
    rng = np.random.default_rng(1)
    reference = rng.random(len(tiny_repo.val_sizes))
    comparison = private_mix.compare(reference + 0.01, reference)
    for estimate in (comparison.val, comparison.public, comparison.private):
        assert estimate.mean == pytest.approx(0.01)
        assert estimate.se == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- gate: pieces

def test_report_counts_findings_and_formats_the_verdict():
    report = GateReport(echo=False)
    report.say("A file validity checked")
    assert report.verdict == "PASS" and report.passed
    report.warn("something to justify")
    assert report.verdict == "PASS with WARN" and report.passed
    report.fail("something wrong")
    assert report.verdict == "FAIL" and not report.passed
    assert report.text == "A file validity checked\n[WARN] something to justify\n[FAIL] something wrong\n"


def test_per_feed_spearman():
    feeds = make_feeds([4, 3, 5])
    a = np.array([1.0, 2.0, 3.0, 4.0, 9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    b = np.array([10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    rho = per_feed_spearman(feeds, a, b)
    assert rho[0] == pytest.approx(1.0) and rho[1] == pytest.approx(-1.0)
    # Feed 3 is constant in ``a``: its values keep their row order, the reverse of ``b``.
    assert rho[2] == pytest.approx(-1.0)


def test_users_with_later_feed_ignores_the_guest_user_and_looks_into_test():
    val = make_feeds([2, 2, 2, 2, 2], users=[5, 5, 0, 0, 7], timestamps=[10, 20, 10, 20, 30])
    test = make_feeds([2, 2], users=[7, 9], timestamps=[40, 50])
    np.testing.assert_array_equal(users_with_later_feed(val, test), [True, False, False, False, True])


def test_member_logs_find_the_inputs_of_the_right_run(tiny_repo):
    # A lane log holding two runs, and a per-run log of a third.
    tiny_repo.write_member_log("run_one", ("a", "b"), file_name="lane.out")
    tiny_repo.write_member_log("run_two", ("c", "has:c"), file_name="lane.out")
    tiny_repo.write_member_log("run_three", ("d",))
    logs = MemberLogs(tiny_repo.member_logs)
    assert logs.training_inputs("run_one") == ["a", "b"]
    assert logs.training_inputs("run_two") == ["c", "has:c"]
    assert logs.training_inputs("run_three") == ["d"]
    assert logs.training_inputs("run_") is None            # a name must match whole, not as a prefix
    assert logs.finetune_inputs("full_a", "ft1") == ["centrality", "seq_top3"]
    assert logs.finetune_inputs("run_one", "ft1") is None


def test_leak_probe_fails_when_the_gain_sits_on_feeds_with_a_later_feed():
    rng = np.random.default_rng(2)
    n = 400
    # Users 1..100 appear twice in val (first feed has a later one), the other 200 feeds are single users.
    users = np.r_[np.arange(1, 101), np.arange(1, 101), np.arange(1000, 1200)]
    times = np.r_[np.full(100, 20718 * 86400), np.full(300, 20720 * 86400)]
    val = make_feeds(np.full(n, 60), users=users, timestamps=times)
    test = make_feeds([5], users=[999_999], timestamps=[20730 * 86400])
    reference = rng.random(n) * 0.1 + 0.4
    noise = rng.normal(scale=0.01, size=n)

    leaking = GateReport(echo=False)
    compare_on_val(leaking, reference + noise + np.where(np.arange(n) < 100, 0.05, 0.0), reference, val, test)
    assert leaking.n_fail == 1 and "leak probe" in "".join(leaking.lines)

    clean = GateReport(echo=False)
    compare_on_val(clean, reference + noise + 0.05, reference, val, test)
    assert clean.n_fail == 0
    assert any(line.startswith("   has_later n=   100") for line in clean.lines)
    assert not any("user0" in line for line in clean.lines)   # fewer than 50 feeds: slice not printed


def test_slice_far_below_the_reference_warns():
    n = 200
    # Every user has two val feeds, a short one on Tuesday and a long one on Thursday.
    val = make_feeds(np.r_[np.full(100, 20), np.full(100, 80)], users=np.r_[np.arange(1, 101), np.arange(1, 101)],
                     timestamps=np.r_[np.full(100, 20718 * 86400), np.full(100, 20720 * 86400)])
    test = make_feeds([5], users=[999_999], timestamps=[20730 * 86400])
    rng = np.random.default_rng(3)
    reference = rng.random(n) * 0.1 + 0.4
    worse_on_long_feeds = reference + rng.normal(scale=0.001, size=n) - np.where(val.sizes >= 50, 0.02, 0.0)
    report = GateReport(echo=False)
    compare_on_val(report, worse_on_long_feeds, reference, val, test)
    assert "[WARN] D slice M>=50 below -2 SE" in report.lines
    assert "[WARN] D slice M<50 below -2 SE" not in report.lines


def test_parity_check_warns_when_test_rankings_move_more_than_val_rankings():
    rng = np.random.default_rng(4)
    val, test = make_feeds(np.full(50, 30)), make_feeds(np.full(40, 30))
    val_new = rng.normal(size=val.n_rows)
    test_new = rng.normal(size=test.n_rows)

    same = GateReport(echo=False)
    check_parity(same, val, test, (val_new, 2.0 * val_new + 1.0), (test_new, 2.0 * test_new + 1.0))
    assert same.n_warn == 0
    assert same.lines == ["F new-vs-ref per-feed Spearman: val mean 1.0000 p5 1.0000 | test mean 1.0000 p5 1.0000"]

    broken = GateReport(echo=False)
    check_parity(broken, val, test, (val_new, val_new), (test_new, test_new + rng.normal(size=test.n_rows)))
    assert broken.n_warn == 1 and "test-side construction break" in broken.lines[-1]


# ---------------------------------------------------------------- gate: on the synthetic tree

def test_gate_passes_a_clean_build(tiny_repo):
    config = built_tiny(tiny_repo)
    report = run_gate(config, echo=False)
    assert report.n_fail == 0, report.text
    assert report.lines[0] == "A file validity checked"
    assert report.lines[1] == "B reproduction: min per-feed Spearman 1.000000"
    assert report.lines[2:7] == [f"  {m}: 4 train-stage inputs audited, fine-tune ft1 audited" for m in config.members]
    assert report.lines[7].startswith("D val new ")
    assert any(line.startswith("E predicted public ") for line in report.lines)
    assert any(line.startswith("F new-vs-ref per-feed Spearman: val mean ") for line in report.lines)
    assert report.lines[-1] == f"\nGATE: {report.verdict}  (FAIL 0, WARN {report.n_warn})"


def test_gate_reads_a_rebuild_from_its_own_folder(tiny_repo):
    config = load_ensemble_config(tiny_repo.config_path)
    rebuild = tiny_repo.root / "rebuild"
    rebuild.mkdir()
    write_ensemble(config, build_ensemble(config), rebuild)
    assert not submission_file(config).exists()
    assert run_gate(config, folder=rebuild, echo=False).n_fail == 0


def _stored_submission(config) -> pl.DataFrame:
    return pl.read_parquet(submission_file(config))


def test_file_check_catches_ties_missing_rows_and_bad_scores(tiny_repo):
    config = built_tiny(tiny_repo)
    good = _stored_submission(config)

    def findings(frame: pl.DataFrame) -> list[str]:
        report = GateReport(echo=False)
        check_file(report, frame)
        return [line for line in report.lines if line.startswith("[FAIL]")]

    assert findings(good) == []
    scores = good["score"].to_numpy().copy()
    scores[1] = scores[0]
    assert findings(good.with_columns(pl.Series("score", scores))) == ["[FAIL] A: 1 feeds with tied scores"]
    assert findings(good.head(len(good) - 3)) == [
        f"[FAIL] A: key set differs from sample_submission ({len(good) - 3} rows, 3 missing)"]
    assert "[FAIL] A: duplicate keys" in findings(pl.concat([good.head(len(good) - 1), good.head(1)]))
    scores = good["score"].to_numpy().copy()
    scores[5] = np.nan
    assert "[FAIL] A: non-finite scores" in findings(good.with_columns(pl.Series("score", scores)))


def test_reproduction_check_catches_a_reordered_feed(tiny_repo):
    config = built_tiny(tiny_repo)
    test = load_feed_index("test")
    good = _stored_submission(config)
    file_scores = good["score"].to_numpy().astype(np.float64)

    ok = GateReport(echo=False)
    check_reproduction(ok, good, file_scores, test)
    assert ok.lines == ["B reproduction: min per-feed Spearman 1.000000"]

    reordered = file_scores.copy()
    reordered[: test.sizes[0]] = (test.sizes[0] - 1) - reordered[: test.sizes[0]]   # the first feed upside down
    bad = GateReport(echo=False)
    check_reproduction(bad, good, reordered, test)
    assert bad.n_fail == 1 and bad.lines[-1] == "B reproduction: min per-feed Spearman -1.000000"


def test_input_audit_applies_the_banned_and_warn_lists(tiny_repo):
    config = load_ensemble_config(tiny_repo.config_path)
    (tiny_repo.member_logs / "full_a.log").unlink()
    tiny_repo.write_member_log("full_a", ("seq_cv_pop", "it_hist_all", "pre_fe_z", "has:pre_fe", "lh_cnt",
                                          "it_hist_all_luo", "pre2_fe", "has:pre2_fe", "als_u", "hour"))
    tiny_repo.write_finetune_log("full_b", ("centrality", "it_hist_et3_rk", "it_tr_ctr", "pre_fe"))
    report = GateReport(echo=False)
    audit_inputs(report, config)
    findings = [line for line in report.lines if line.startswith("[")]
    assert findings == [
        "[FAIL] full_a: banned train-stage input it_hist_all",       # counts that include the user's later snapshots
        "[FAIL] full_a: banned train-stage input pre_fe_z",          # encoder score with leaking train rows
        "[FAIL] full_a: banned train-stage input has:pre_fe",        # its presence flag is banned with it
        "[FAIL] full_a: banned train-stage input lh_cnt",
        "[WARN] full_a: train-stage input pre2_fe is on the warn list",   # the raw score; has:pre2_fe is fine
        "[WARN] full_a: train-stage input als_u is on the warn list",
        "[WARN] full_a: train-stage input hour is on the warn list",
        "[FAIL] full_b: banned fine-tune input it_hist_et3_rk",
        "[WARN] full_b: fine-tune input it_tr_ctr is on the warn list",   # pre_fe is allowed in the fine-tune
    ]
    assert "  full_a: 10 train-stage inputs audited, fine-tune ft1 audited" in report.lines


def test_input_audit_fails_on_missing_logs(tiny_repo):
    config = load_ensemble_config(tiny_repo.config_path)
    (tiny_repo.member_logs / "full_c.log").unlink()
    (tiny_repo.member_logs / "light_a_ft1_ft.log").unlink()
    report = GateReport(echo=False)
    audit_inputs(report, config)
    shown = MemberLogs(tiny_repo.member_logs).shown_folder
    assert [line for line in report.lines if line.startswith("[")] == [
        f"[FAIL] C: full_c: no train-stage input list in {shown}; the inputs cannot be audited",
        f"[FAIL] C: light_a: no fine-tune input list in {shown}/light_a_ft1_ft.log; the inputs cannot be audited",
    ]
    assert not any(line.startswith(("  light_a:", "  full_c:")) for line in report.lines)
    assert report.verdict == "FAIL"


def test_gate_fails_when_the_logs_are_in_another_folder(tiny_repo):
    """A rebuild audited against an empty logs folder fails instead of passing with nothing audited."""
    config = built_tiny(tiny_repo)
    elsewhere = tiny_repo.root / "runs" / "logs"
    elsewhere.mkdir(parents=True)
    report = run_gate(config, echo=False, logs_dir=elsewhere)
    assert report.verdict == "FAIL"
    assert sum(line.startswith("[FAIL] C: ") for line in report.lines) == 2 * len(config.members)
    assert run_gate(config, echo=False, logs_dir=tiny_repo.member_logs).n_fail == 0


def test_member_logs_read_the_last_run_of_a_name(tiny_repo):
    """A log is appended to: a crashed run and its re-run under the same name leave two runs in one file."""
    tiny_repo.write_member_log("rerun", ("old_input",))
    tiny_repo.write_member_log("rerun", ("new_input", "has:new_input"))
    with open(tiny_repo.member_logs / "full_a_ft1_ft.log", "a") as log:
        log.write("==== second fine-tune\nextra features (1): ['later_extra']\n")
    logs = MemberLogs(tiny_repo.member_logs)
    assert logs.training_inputs("rerun") == ["new_input", "has:new_input"]
    assert logs.finetune_inputs("full_a", "ft1") == ["later_extra"]
    # A run whose args line matches but that never logged its inputs does not hide the earlier one.
    with open(tiny_repo.member_logs / "rerun.log", "a") as log:
        log.write("==== run args: {'variant': 'rerun', 'seed': 2}\nTraceback (most recent call last):\n")
    assert MemberLogs(tiny_repo.member_logs).training_inputs("rerun") == ["new_input", "has:new_input"]


def test_input_audit_catches_a_hard_link_to_a_quarantined_file(tiny_repo):
    config = load_ensemble_config(tiny_repo.config_path)
    quarantined = tiny_repo.root / "quarantine" / "val" / "old_name.npy"
    quarantined.parent.mkdir()
    np.save(quarantined, np.zeros(3, np.float32))
    try:
        os.link(quarantined, tiny_repo.root / "feature_cache" / "val" / "ctx_nd_max.npy")
    except OSError:
        pytest.skip("the temporary folder does not support hard links")
    np.save(tiny_repo.root / "features" / "val" / "seq_cv_pop.npy", np.zeros(3, np.float32))   # an ordinary file
    report = GateReport(echo=False)
    audit_inputs(report, config)
    fails = [line for line in report.lines if line.startswith("[FAIL]")]
    assert fails == [f"[FAIL] {m}: input ctx_nd_max (val) is hard-linked to a quarantined file" for m in config.members]


def test_gate_fails_on_non_finite_pool_scores(tiny_repo):
    config = built_tiny(tiny_repo)
    file = pool_file(config.pool_tag, "finetuned", "val")
    scores = np.load(file)
    scores[0] = np.inf
    np.save(file, scores)
    report = run_gate(config, echo=False)
    assert "[FAIL] ens_TINY_ft_val: non-finite predictions" in report.lines
    assert report.verdict == "FAIL"


# ---------------------------------------------------------------- freeze-day check

def test_file_sha256(tmp_path: Path):
    file = tmp_path / "blob"
    file.write_bytes(b"abc" * 3_000_000)   # longer than one read block
    assert file_sha256(file) == hashlib.sha256(b"abc" * 3_000_000).hexdigest()


def test_candidate_table_lists_the_records():
    configs = [load_ensemble_config(ENSEMBLE_DIR / f"{name}.toml") for name in ("fd16p20", "w8p20")]
    table = candidate_table(configs)
    assert table[0] == ("FD16P20    | cand_FD16P20kd_r4.parquet | 16 full-view members + 4 in the pool at weight 0.2 | "
                        "val 0.40271 | public 0.394892 | "
                        "sha256 14adc19ac794e3977cc0275a0c15d2c76e55c5dcdeec911dc47a8193128626a8")
    assert table[1].startswith("W8P20      | cand_W8P20kd_r3.parquet | 8 full-view members + 4 in the pool at weight 0.2 | "
                               "val 0.40214 | public 0.3950094 | sha256 406e5705")


def test_candidate_table_states_the_feed_size_rule():
    configs = [load_ensemble_config(ENSEMBLE_DIR / f"{name}.toml") for name in ("fd16p20_m04_02", "w8p20")]
    table = candidate_table(configs)
    assert table[0] == ("FD16P20@m0.4_0.2 | cand_FD16P20_m04_02kd_r4.parquet | 16 full-view members + 4 in the pool "
                        "at weight 0.4 on feeds of fewer than 50 candidates, 0.2 on the others | "
                        "val 0.40293 | public 0.3952092 | "
                        "sha256 3cf6faccbad1357ae1a1b11999fef8168835d77407684832c7578c832c6e71b3")
    assert table[1].startswith("W8P20            | cand_W8P20kd_r3.parquet | ")   # names padded to one width


def test_freeze_check_compares_hash_and_gate_text_with_the_record(tiny_repo):
    built_tiny(tiny_repo)
    sha256 = file_sha256(tiny_repo.submissions / "tiny.parquet")
    tiny_repo.config_path.write_text(tiny_repo.config_path.read_text().replace('"unset"', f'"{sha256}"'))
    config = load_ensemble_config(tiny_repo.config_path)
    recorded = recorded_gate_output(config)
    recorded.write_text(run_gate(config, echo=False).text)

    same = check_candidate(config, tiny_repo.root / "runs" / "freeze")
    assert same.ok and same.sha256_matches and same.gate_matches_record and same.gate_diff == ()
    assert same.gate_output.read_text() == recorded.read_text()

    recorded.write_text(recorded.read_text().replace("A file validity checked", "A file validity checked long ago"))
    changed = check_candidate(config, tiny_repo.root / "runs" / "freeze")
    assert changed.ok and not changed.gate_matches_record   # a text difference is reported, it does not fail
    assert "-A file validity checked long ago" in changed.gate_diff
    assert "+A file validity checked" in changed.gate_diff


def test_freeze_check_fails_on_another_file_under_the_recorded_name(tiny_repo):
    config = built_tiny(tiny_repo)   # the recipe's record still says sha256 = "unset"
    recorded_gate_output(config).write_text(run_gate(config, echo=False).text)
    result = check_candidate(config, tiny_repo.root / "runs" / "freeze")
    assert result.gate_passed and result.gate_matches_record
    assert not result.sha256_matches and not result.ok


# ---------------------------------------------------------------- stored artefacts

@pytest.mark.artefacts
@pytest.mark.parametrize("recipe", ["fd16p20_m04_02", "fd16p20", "fd16p20vf", "w8p20"])
def test_stored_submissions_pass_the_freeze_check(recipe: str, tmp_path: Path, need):
    """sha256 on record, gate PASS, gate text equal to the output stored at build time."""
    config = load_ensemble_config(ENSEMBLE_DIR / f"{recipe}.toml")
    pools = [pool_file(tag, branch, split) for tag in (config.pool_tag, config.reference_pool_tag)
             for branch in BRANCHES for split in SCORED_SPLITS]
    logs = [path("outputs.member_logs") / f"{member}.log" for member in config.members]
    need(submission_file(config), recorded_gate_output(config), *pools, *logs,
         path("data.flat") / "val" / "cand_target.npy", path("data.raw") / "sample_submission.parquet")
    result = check_candidate(config, tmp_path)
    assert result.sha256_matches
    assert result.gate_verdict == "PASS"
    assert result.gate_matches_record, "\n".join(result.gate_diff)


@pytest.mark.artefacts
def test_private_mix_on_the_competition_data(need):
    """The numbers the working records quote for the view: cold shares and effective sample sizes."""
    need(path("data.flat") / "train" / "hist_item_id.npy", path("data.flat") / "val" / "feeds.parquet",
         path("data.flat") / "test" / "cand_item_id.npy")
    test = load_feed_index("test")
    public = public_mask(test.table["timestamp"].to_numpy())
    assert (int(public.sum()), int((~public).sum())) == (8562, 20033)
    cold = private_mix.cold_share("test")
    means = (private_mix.cold_share("val").mean(), cold[public].mean(), cold[~public].mean())
    assert [round(float(m), 3) for m in means] == [0.290, 0.371, 0.406]
    weights = private_mix.mix_weights()
    effective = {part: w.sum() ** 2 / (w ** 2).sum() for part, w in weights.items()}
    assert (round(effective["public"]), round(effective["private"])) == (9321, 6389)


@pytest.mark.artefacts
def test_recipes_have_all_member_files(need):
    """Every member of the shipped recipes has its four prediction files."""
    need(path("outputs.predictions"))
    for name in ("fd16p20_m04_02", "fd16p20", "fd16p20vf", "w8p20"):
        config = load_ensemble_config(ENSEMBLE_DIR / f"{name}.toml")
        missing = missing_member_files(config)
        if len(missing) == 4 * len(config.members):
            pytest.skip(f"no member predictions of {config.name} in {path('outputs.predictions')}")
        assert missing == []
