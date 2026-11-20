"""The ranker, its dense block, the loader and predict.

Most tests run on the CPU against a small synthetic split written to a
temporary folder. The last one loads stored member checkpoints and compares
their predictions with the stored arrays; it needs the artefacts and a GPU.
"""
from __future__ import annotations

import json
import tomllib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from avito_ranker import paths
from avito_ranker.data.loader import SplitTensors, load_split
from avito_ranker.model import (
    InputColumn,
    Ranker,
    RankerConfig,
    build_ranker,
    dense_features,
    dense_statistics,
    load_recipe,
    read_checkpoint,
    save_checkpoint,
)
from avito_ranker.model.checkpoint import rename_stored_key
from avito_ranker.training.predict import predict
from support import CONFIGS_TABLE, RECIPE_DIR

RECIPES = ("full", "popularity_light", "popularity_light_narrow", "popularity_free")
POPULARITY_INPUTS = {"seq_cv_invd20", "seq_cv_pop", "seq_cv_norm", "it_hist_all_luo", "it_hist_et1_luo",
                     "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo"}
N_FEEDS, N_ITEMS = 48, 400


# ----------------------------------------------------------------------------- a synthetic split
def _write_tiny_split(root: Path, split: str, rng: np.random.Generator) -> None:
    """Write flat arrays, an item table and every input of both views for a split of 48 small feeds."""
    sizes = rng.integers(5, 70, N_FEEDS)
    lengths = rng.integers(0, 300, N_FEEDS)   # some above 256, so the loader has to cut
    lengths[:3] = (0, 256, 299)
    n_cand, n_hist = int(sizes.sum()), int(lengths.sum())
    users = rng.integers(1, 30, N_FEEDS)
    users[5] = 0                               # one not-logged-in session
    flat = root / "flat" / split
    flat.mkdir(parents=True)
    pl.DataFrame({
        "fidx": np.arange(N_FEEDS, dtype=np.int32),
        "feed_id": (100 + np.arange(N_FEEDS)).astype(np.int32),
        "timestamp": np.arange(N_FEEDS, dtype=np.int32),
        "user_id": users.astype(np.int32),
        "M": sizes.astype(np.int32),
        "H": lengths.astype(np.int32),
        "cand_off": np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64),
        "hist_off": np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64),
    }).write_parquet(flat / "feeds.parquet")

    def item_rows(n: int) -> np.ndarray:
        rows = rng.integers(0, N_ITEMS, n).astype(np.int32)
        rows[rng.random(n) < 0.05] = -1        # items without a content embedding
        return rows

    targets = np.zeros(n_cand, np.int8)
    targets[np.r_[0, np.cumsum(sizes)[:-1]] + rng.integers(0, sizes)] = 1
    arrays = {
        "user_emb": rng.integers(-60, 60, (N_FEEDS, 128)).astype(np.int8),
        "cand_emb_idx": item_rows(n_cand),
        "cand_category_id": rng.integers(2, 54, n_cand).astype(np.int8),
        "cand_target": targets,
        "cand_pos": rng.integers(0, 300, n_cand).astype(np.int16),
        "hist_emb_idx": item_rows(n_hist),
        "hist_category_id": rng.integers(2, 54, n_hist).astype(np.int8),
        "hist_event_type": rng.integers(1, 11, n_hist).astype(np.int8),
        "hist_event_source": rng.integers(0, 31, n_hist).astype(np.int8),
        **{f"cand_f{k}": rng.random(n_cand).astype(np.float32) for k in range(1, 6)},
    }
    for name, values in arrays.items():
        np.save(flat / f"{name}.npy", values)

    table = root / "competition" / split / "item_embeddings"
    table.mkdir(parents=True)
    rng.integers(-60, 60, (N_ITEMS, 128)).astype(np.int8).tofile(table / "data.memmap")
    (table / "data.meta").write_text(json.dumps({"dtype": "int8", "shape": [N_ITEMS, 128], "order": "C"}))

    cache = root / "feature_cache" / split
    cache.mkdir(parents=True)
    files = {c.file for name in ("full", "popularity_light") for c in load_recipe(name).view.columns}
    for file in sorted(files):
        values = (rng.gamma(1.0, 30.0, n_cand) if file.startswith(("seq_cv", "it_hist")) else rng.normal(0, 4, n_cand))
        values = values.astype(np.float32)
        if file.startswith("pre2"):
            values[rng.random(n_cand) < 0.2] = np.nan   # rows without an encoder score
        np.save(cache / f"{file}.npy", values)


@pytest.fixture
def tiny_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A synthetic ``val`` split on disk, with ``avito_ranker.paths`` pointed at it."""
    root = tmp_path / "data"
    _write_tiny_split(root, "val", np.random.default_rng(3))
    (root / "paths.toml").write_text(
        f'[data]\nflat = "{root / "flat"}"\ncompetition = "{root / "competition"}"\n'
        f'feature_cache = "{root / "feature_cache"}"\n' + CONFIGS_TABLE)
    monkeypatch.setenv("AVITO_PATHS", str(root / "paths.toml"))
    paths._table.cache_clear()
    yield root
    paths._table.cache_clear()


@pytest.fixture
def tiny_split(tiny_data: Path) -> SplitTensors:
    return load_split("val", "full", device="cpu")


def _tiny_model(seed: int = 0, n_finetune_extras: int = 0) -> Ranker:
    """A full-view ranker of width 32 with non-trivial dense scaling, in evaluation mode."""
    config = RankerConfig(view=load_recipe("full").view, d_model=32, n_finetune_extras=n_finetune_extras)
    torch.manual_seed(seed)
    return Ranker(config).eval()


# ----------------------------------------------------------------------------- recipes and sizes
def test_views_list_the_documented_inputs():
    full, light = load_recipe("full"), load_recipe("popularity_light")
    assert (full.view.n_inputs, full.n_dense) == (34, 104)
    assert (light.view.n_inputs, light.n_dense) == (27, 93)
    full_files = [c.file for c in full.view.columns]
    light_files = [c.file for c in light.view.columns]
    assert set(full_files) - set(light_files) == POPULARITY_INPUTS
    assert set(light_files) - set(full_files) == {"xp_imp_c50"}
    # five inputs also enter relative to their feed; three of them only that way
    assert [full_files[i] for i in full.view.ranked] == ["seq_cv_invd20", "it_hist_all_luo", "pre2_fe_s",
                                                         "pre2_new_s", "pre2_hmax32"]
    assert [c.file for c in full.view.columns if not c.keep_raw] == ["pre2_fe_s", "pre2_new_s", "pre2_hmax32"]
    assert load_recipe("popularity_light_narrow").view == light.view
    free = load_recipe("popularity_free")
    assert (free.view.n_inputs, free.n_dense) == (26, 92)
    assert [c.file for c in free.view.columns] == [f for f in light_files if f != "xp_imp_c50"]


@pytest.mark.parametrize("recipe, expected", list(zip(RECIPES, (2_118_848, 2_114_613, 1_005_685, 2_114_228))))
def test_parameter_counts(recipe: str, expected: int):
    model = Ranker(load_recipe(recipe))
    assert sum(p.numel() for p in model.parameters()) == expected
    # the hidden matrices, which the optimiser treats separately: every linear weight with two real dimensions
    hidden = [m for m in model.modules() if isinstance(m, torch.nn.Linear) and min(m.weight.shape) > 1]
    assert len(hidden) == 46


def test_config_round_trip():
    config = load_recipe("popularity_light_narrow")
    assert RankerConfig.from_dict(config.to_dict()) == config
    assert config.with_finetune_extras(42).n_finetune_extras == 42


def test_input_scaling_rules():
    stored = np.array([-20.0, -1.0, 0.0, 3.0, 20.0, np.nan], np.float32)
    assert InputColumn("a", "clip_10").scale(stored).tolist() == [-10, -1, 0, 3, 10, 0]
    assert InputColumn("a", "clip_5").scale(stored).tolist() == [-5, -1, 0, 3, 5, 0]
    assert InputColumn("a", "presence").scale(stored).tolist() == [1, 1, 1, 1, 1, 0]
    np.testing.assert_allclose(InputColumn("a", "log_count").scale(stored), [0, 0, 0, np.log1p(3), np.log1p(20), 0],
                               rtol=1e-6)
    with pytest.raises(ValueError):
        InputColumn("a", "sqrt")
    with pytest.raises(ValueError):
        InputColumn("a", "clip_10", rank_and_z=False, keep_raw=False)


# ----------------------------------------------------------------------------- initialisation
def test_same_seed_gives_the_same_weights_and_leaves_the_same_random_state():
    def build() -> tuple[Ranker, torch.Tensor]:
        torch.manual_seed(951)
        model = Ranker(load_recipe("full"))
        return model, torch.rand(4)

    (first, draw_1), (second, draw_2) = build(), build()
    assert all(torch.equal(a, b) for a, b in zip(first.state_dict().values(), second.state_dict().values()))
    assert torch.equal(draw_1, draw_2)


# Sum and absolute sum of the 2,118,848 initial weights of the full-view model at seed 951, recorded from the
# research model class (whose seeded weights the new class reproduces bit for bit).
SEED_951_SUM = 3784.5321255168897
SEED_951_ABS_SUM = 134338.87157719847


@pytest.mark.skipif(not torch.__version__.startswith("2.11."), reason="initial values recorded with torch 2.11")
def test_seeded_initial_weights_match_the_trained_members():
    """Seed 951 must give the weights the research model class gave: the sums below were recorded from it."""
    torch.manual_seed(951)
    model = Ranker(load_recipe("full"))
    total = sum(float(p.detach().double().sum()) for p in model.parameters())
    magnitude = sum(float(p.detach().double().abs().sum()) for p in model.parameters())
    assert total == pytest.approx(SEED_951_SUM, rel=1e-9)
    assert magnitude == pytest.approx(SEED_951_ABS_SUM, rel=1e-9)
    # blocks that start as no-ops, and the attention prior scales
    assert not model.retrieval.output.weight.any() and not model.retrieval.type_bias.weight.any()
    assert model.cross_blocks[0].attention.similarity_scale.tolist() == pytest.approx([0, 8 / 3, 16 / 3, 8])
    assert not hasattr(model.set_blocks[0].attention, "similarity_scale")




def test_finetune_branch_changes_neither_the_other_weights_nor_the_scores(tiny_split: SplitTensors):
    plain, extended = _tiny_model(seed=5), _tiny_model(seed=5, n_finetune_extras=7)
    shared = plain.state_dict()
    assert all(torch.equal(shared[k], v) for k, v in extended.state_dict().items() if k in shared)
    branch = {f"finetune_mlp.{i}.{p}" for i in (0, 2) for p in ("weight", "bias")}
    assert set(extended.state_dict()) - set(shared) == branch

    tiny_split.finetune_extras = torch.randn(tiny_split.n_candidates, 7)
    extended.use_validation_domain()
    batch = tiny_split.gather(np.arange(6))
    assert batch.finetune_extras.shape == (*batch.candidate_mask.shape, 7)
    with torch.no_grad():
        assert torch.equal(plain(batch), extended(batch))   # the branch's last layer starts at zero
        extended.finetune_mlp[2].weight.normal_()
        assert not torch.equal(plain(batch), extended(batch))
        batch.finetune_extras = None
        with pytest.raises(ValueError, match="fine-tune branch"):   # a split without the extras is refused
            extended(batch)


# ----------------------------------------------------------------------------- loader
def test_batch_plan_covers_every_feed_once_within_the_budget(tiny_split: SplitTensors):
    feeds = np.arange(tiny_split.n_feeds)
    plan = tiny_split.plan_batches(feeds, max_candidates=128, max_feeds=4, shuffle=True, rng=np.random.default_rng(1))
    assert sorted(np.concatenate(plan).tolist()) == feeds.tolist()
    for batch in plan:
        padded = -(-tiny_split.feed_sizes[batch] // 16) * 16
        assert len(set(padded.tolist())) == 1                       # one size bucket per batch
        assert len(batch) <= 4 and len(batch) * padded[0] <= 128
    again = tiny_split.plan_batches(feeds, max_candidates=128, max_feeds=4, shuffle=True, rng=np.random.default_rng(1))
    assert all(np.array_equal(a, b) for a, b in zip(plan, again))
    in_order = tiny_split.plan_batches(feeds, max_candidates=128, max_feeds=4)
    assert all(np.all(np.diff(b) > 0) for b in in_order)
    assert len(in_order) == len(plan)                               # the shuffle does not change the count
    with pytest.raises(ValueError, match="rng"):
        tiny_split.plan_batches(feeds, shuffle=True)


def test_gather_pads_and_masks(tiny_split: SplitTensors):
    feeds = np.array([0, 1, 2, 5, 9])
    batch = tiny_split.gather(feeds, with_positions=True)
    sizes, lengths = tiny_split.feed_sizes[feeds], np.minimum(tiny_split.history_lengths[feeds], 256)
    n_slots, n_events = batch.candidate_mask.shape[1], batch.history_mask.shape[1]
    assert n_slots % 16 == 0 and n_slots - 16 < sizes.max() <= n_slots
    assert n_events == 256                                   # feed 2 has 299 events: cut to the newest 256
    assert batch.candidate_mask.sum(1).tolist() == sizes.tolist()
    assert batch.history_mask.sum(1).tolist() == lengths.tolist()
    assert batch.history_length.tolist() == lengths.tolist()
    assert batch.is_anonymous.tolist() == [False, False, False, True, False]

    # real slots point at the feed's rows, padded slots repeat its first row
    offsets = tiny_split.candidate_offsets[feeds]
    for i, (offset, size) in enumerate(zip(offsets, sizes)):
        assert batch.candidate_row[i, :size].tolist() == list(range(offset, offset + size))
        assert (batch.candidate_row[i, size:] == offset).all()
    rows = batch.candidate_row[batch.candidate_mask].numpy()
    np.testing.assert_array_equal(batch.target[batch.candidate_mask].numpy(), tiny_split.targets[rows])
    assert batch.target.sum(1).tolist() == [1.0] * 5 and batch.position is not None
    assert tiny_split.gather(feeds).position is None

    # unit embeddings: length 1, or 0 where the item has no embedding or the slot is padding
    norms = batch.candidate_unit.norm(dim=-1)
    assert torch.allclose(norms[~batch.candidate_no_embedding], torch.ones(()), atol=1e-5)
    assert (norms[batch.candidate_no_embedding] == 0).all()
    history_norms = batch.history_unit.norm(dim=-1)
    assert torch.allclose(history_norms[batch.history_has_embedding], torch.ones(()), atol=1e-5)
    assert (history_norms[~batch.history_has_embedding] == 0).all()
    assert not (batch.history_has_embedding & ~batch.history_mask).any()
    assert batch.inputs.shape == (5, n_slots, 34) and batch.inputs.dtype == torch.float32


def test_domain_mix_and_iteration(tiny_split: SplitTensors):
    p = np.linspace(0, 1, tiny_split.n_feeds).astype(np.float32)
    tiny_split.set_domain_mix(p)
    seen = 0
    for feeds in tiny_split.plan_batches(np.arange(tiny_split.n_feeds), max_candidates=256, max_feeds=8):
        batch = tiny_split.gather(feeds)
        assert torch.allclose(batch.domain_mix.sum(1), torch.ones(len(batch.feed_index)))
        np.testing.assert_allclose(batch.domain_mix[:, 1].numpy(), p[batch.feed_index.numpy()])
        assert (batch.domain_mix[:, 2] == 0).all()
        seen += len(batch.feed_index)
    assert seen == tiny_split.n_feeds


def test_finetune_extras_use_the_scaling_of_the_first_split(tiny_data: Path, tiny_split: SplitTensors):
    files = ["ctx_nd_max", "seq_cv_pop", "pre2_fe"]
    scaling = tiny_split.attach_finetune_extras(files)
    extras = tiny_split.finetune_extras
    assert extras.shape == (tiny_split.n_candidates, 3) and extras.abs().max() <= 6
    assert scaling.log_columns.tolist() == [False, True, False]     # the count went through log1p
    stored = np.load(tiny_data / "feature_cache" / "val" / "pre2_fe.npy")
    assert (extras[np.isnan(stored), 2] == 0).all()                 # missing values sit at the mean
    other = load_split("val", "full", device="cpu")
    assert other.attach_finetune_extras(files, scaling) is scaling
    assert torch.equal(other.finetune_extras, extras)


# ----------------------------------------------------------------------------- dense block
def test_dense_block_columns(tiny_split: SplitTensors):
    feeds = np.array([1, 5, 7])
    batch = tiny_split.gather(feeds)
    view = tiny_split.view
    block = dense_features(batch, view)
    assert block.shape == (3, batch.candidate_mask.shape[1], 104) and torch.isfinite(block).all()

    i, size = 0, int(tiny_split.feed_sizes[feeds[0]])
    real = block[i, :size].double().numpy()
    f1 = batch.candidate_numeric[i, :size, 0].double().numpy()
    np.testing.assert_allclose(real[:, 0], f1)
    np.testing.assert_allclose(real[:, 1], (f1 - f1.mean()) / (f1.std() + 1e-4), atol=1e-4)
    rank = np.array([(f1 < x).sum() + 0.5 * ((f1 == x).sum() - 1) for x in f1]) / (size - 1)
    np.testing.assert_allclose(real[:, 2], rank, atol=1e-6)
    np.testing.assert_array_equal(real[:, 15], batch.candidate_no_embedding[i, :size].numpy())

    cand = batch.candidate_unit[i, :size].double().numpy()
    hist = batch.history_unit[i].double().numpy()
    has = batch.history_has_embedding[i].numpy()
    np.testing.assert_allclose(real[:, 17], cand @ batch.user_unit[i].double().numpy(), atol=1e-5)
    cosine = cand @ hist.T
    np.testing.assert_allclose(real[:, 19], cosine[:, has].max(1), atol=1e-5)          # best match in the history
    np.testing.assert_allclose(real[:, 20], np.where(has[:8], cosine[:, :8], -1.5).max(1), atol=1e-5)  # newest 8
    np.testing.assert_allclose(real[:, 22], cosine[:, has].mean(1), atol=1e-5)

    category = batch.candidate_category[i, :size].numpy()
    history_category = batch.history_category[i].numpy()[batch.history_mask[i].numpy()]
    same = (category[:, None] == history_category[None, :]).sum(1)
    np.testing.assert_allclose(real[:, 35], np.log1p(same), atol=1e-6)
    np.testing.assert_allclose(real[:, 48], (category[:, None] == category[None, :]).mean(1), atol=1e-6)
    np.testing.assert_allclose(real[:, 49], np.log(size), atol=1e-6)
    assert (block[1, :, 61] == 1).all() and (block[0, :, 61] == 0).all()               # feed 5 is anonymous

    # the view's columns: 31 raw inputs in view order, then (rank, z-score) of the 5 ranked ones
    inputs = batch.inputs[i, :size].double().numpy()
    np.testing.assert_allclose(real[:, 63:94], inputs[:, view.raw], atol=1e-6)
    first_ranked = inputs[:, view.ranked[0]]
    expected = np.array([(first_ranked < x).sum() + 0.5 * ((first_ranked == x).sum() - 1) for x in first_ranked])
    np.testing.assert_allclose(real[:, 94], expected / (size - 1), atol=1e-6)


def test_dense_statistics_feed_the_model(tiny_split: SplitTensors):
    mean, std = dense_statistics(tiny_split, np.arange(tiny_split.n_feeds), tiny_split.view, n_batches=4, seed=0)
    assert mean.shape == std.shape == (104,) and torch.isfinite(mean).all() and (std >= 0).all()
    model = _tiny_model()
    model.set_dense_scaling(mean, std)
    assert torch.equal(model.dense_mean, mean) and model.dense_std.min() >= 1e-3


# ----------------------------------------------------------------------------- forward pass and predict
def test_scores_ignore_padding_and_batch_mates(tiny_split: SplitTensors):
    model = _tiny_model()
    together = np.array([3, 8, 11, 20])
    with torch.no_grad():
        joint = model(tiny_split.gather(together))
        for row, feed in enumerate(together):
            alone = model(tiny_split.gather(np.array([feed])))
            size = int(tiny_split.feed_sizes[feed])
            assert torch.allclose(joint[row, :size], alone[0, :size], atol=2e-5)


def test_position_scores_come_only_on_request(tiny_split: SplitTensors):
    model = _tiny_model()
    batch = tiny_split.gather(np.arange(4))
    with torch.no_grad():
        scores = model(batch)
        both = model(batch, model.dense_features(batch), with_position_scores=True)
    assert isinstance(scores, torch.Tensor) and scores.dtype == torch.float32
    assert len(both) == 2 and both[1].shape == scores.shape and torch.equal(both[0], scores)


def test_domain_rows(tiny_split: SplitTensors):
    model = _tiny_model()
    batch = tiny_split.gather(np.arange(4))
    with torch.no_grad():
        default = model(batch)
        batch.domain_mix = torch.tensor([[0.0, 1.0, 0.0]]).repeat(4, 1)
        assert torch.allclose(model(batch), default, atol=1e-6)      # the default is the random-order row
        batch.domain_mix = torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1)
        assert not torch.allclose(model(batch), default, atol=1e-6)
        batch.domain_mix = None
        model.use_validation_domain()                                 # the third row starts as a copy of the second
        assert torch.equal(model(batch), default)


def test_predict_is_aligned_with_candidate_rows(tiny_split: SplitTensors):
    model = _tiny_model().train()
    chosen = np.arange(0, tiny_split.n_feeds, 2)
    scores = predict(model, tiny_split, chosen, max_candidates=256, max_feeds=8)
    assert model.training                                             # the mode is restored
    assert scores.shape == (tiny_split.n_candidates,) and scores.dtype == np.float32
    skipped = np.setdiff1d(np.arange(tiny_split.n_feeds), chosen)
    for feed in skipped:
        offset, size = tiny_split.candidate_offsets[feed], tiny_split.feed_sizes[feed]
        assert not scores[offset:offset + size].any()
    model.eval()
    feed = int(chosen[3])
    offset, size = tiny_split.candidate_offsets[feed], tiny_split.feed_sizes[feed]
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        direct = model(tiny_split.gather(np.array([feed])))[0, :size].numpy()
    np.testing.assert_allclose(scores[offset:offset + size], direct, atol=0.05)   # bfloat16 with other batch mates


# ----------------------------------------------------------------------------- checkpoints
def test_checkpoint_round_trip(tmp_path: Path, tiny_split: SplitTensors):
    model = _tiny_model(seed=2)
    model.set_dense_scaling(torch.randn(104), torch.rand(104) + 0.5)
    model.use_validation_domain()
    save_checkpoint(tmp_path / "model.pt", model, step=7)
    checkpoint = read_checkpoint(tmp_path / "model.pt")
    assert checkpoint.config == model.config and checkpoint.info == {"step": 7}
    restored = build_ranker(checkpoint).eval()
    assert restored.inference_domain == model.inference_domain
    batch = tiny_split.gather(np.arange(5))
    with torch.no_grad():
        assert torch.equal(restored(batch), model(batch))

    with_branch = build_ranker(checkpoint, n_finetune_extras=9)
    assert with_branch.finetune_mlp[0].in_features == 9
    assert torch.equal(with_branch.head[1].weight, model.head[1].weight)
    del checkpoint.state["wide.bias"]
    with pytest.raises(ValueError):
        build_ranker(checkpoint)


@pytest.mark.parametrize("stored, ours", [
    ("f_mean", "dense_mean"),
    ("c_miss", "candidate_missing_vector"),
    ("cat.weight", "category.weight"),
    ("c_mlp.2.bias", "dense_mlp.2.bias"),
    ("retr.a", "retrieval.similarity_scale"),
    ("retr.e_src.weight", "retrieval.event_source.weight"),
    ("retr.Z.bias", "retrieval.output.bias"),
    ("dom.weight", "domain.weight"),
    ("hist_enc.0.lnc.weight", "history_encoder.0.norm_context.weight"),
    ("cross.1.attn.bias_scale", "cross_blocks.1.attention.similarity_scale"),
    ("cross.0.attn.q.weight", "cross_blocks.0.attention.query.weight"),
    ("sets.1.attn.g.bias", "set_blocks.1.attention.gate.bias"),
    ("sets.0.ff.3.weight", "set_blocks.0.ffn.3.weight"),
    ("aux_head.3.weight", "position_head.3.weight"),
    ("extra.0.weight", "finetune_mlp.0.weight"),
])
def test_stored_parameter_names_are_translated(stored: str, ours: str):
    assert rename_stored_key(stored) == ours
    assert ours in Ranker(RankerConfig(view=load_recipe("full").view, d_model=16, n_finetune_extras=3)).state_dict()


def _stored_cfg() -> dict:
    """The ``cfg`` entry of a stored member checkpoint of the popularity-free view, as the research code wrote it."""
    columns = load_recipe("popularity_free").view.columns
    names = [c.file if c.scaling != "presence" else f"has:{c.file}" for c in columns]
    return {
        "d": 192, "heads": 4, "hist_layers": 1, "blocks": 2, "drop": 0.1, "use_hist": True, "use_set": True,
        "use_pos_tower": False, "n_dense": 92, "aux_pos": True, "ff": 2, "sim_bias": 1, "pos_mode": "both",
        "domain": True, "intent": False, "xt_names": names, "id_buckets": 0, "id_dim": 32, "id_drop": 0.5,
        "arch": "tf", "id_hist_only": False, "xt_rel": [22, 23, 24], "xt_drop": 0.0, "xt_idcols": None,
        "xt_relonly": [22, 23, 24], "attn_gate": True, "split_kv": True, "retr_m": 16,
    }


def test_stored_member_config_is_translated(tmp_path: Path):
    torch.save({"cfg": _stored_cfg(), "state_dict": {"cross.0.attn.k.bias": torch.zeros(192)}, "args": {"seed": 1}},
               tmp_path / "member.pt")
    checkpoint = read_checkpoint(tmp_path / "member.pt")
    assert checkpoint.config == load_recipe("popularity_free")       # scalings follow from the stored file names
    assert list(checkpoint.state) == ["cross_blocks.0.attention.key.bias"]
    assert checkpoint.info == {"args": {"seed": 1}}


@pytest.mark.parametrize("change", [{"arch": "gru"}, {"use_pos_tower": True}, {"retr_uniq": True}, {"n_dense": 93}])
def test_stored_member_with_other_options_is_refused(tmp_path: Path, change: dict):
    torch.save({"cfg": {**_stored_cfg(), **change}, "state_dict": {}, "args": {}}, tmp_path / "member.pt")
    with pytest.raises(ValueError):
        read_checkpoint(tmp_path / "member.pt")


# ----------------------------------------------------------------------------- stored members
# One training recipe per model recipe, the folder of its stored checkpoints, and the parameter count.
STORED_MEMBERS = [
    ("student_wide", "members.checkpoints", "full", 2_118_848),
    ("student_popularity_light", "members.checkpoints", "popularity_light", 2_114_613),
    ("student_popularity_light_narrow_t25_holdout", "members.checkpoints",
     "popularity_light_narrow", 1_005_685),
    ("student_popularity_free", "members.checkpoints", "popularity_free", 2_114_228),
]


@pytest.mark.artefacts
@pytest.mark.gpu
@pytest.mark.parametrize("training_recipe, folder, recipe, n_parameters", STORED_MEMBERS)
def test_stored_member_predictions_are_reproduced(need, training_recipe: str, folder: str, recipe: str,
                                                  n_parameters: int):
    """The first stored run of a training recipe, loaded into the new model, gives its stored validation predictions.

    The tolerance is 2e-3 in score units. On the GPU model that trained the
    members the difference is exactly 0.
    """
    with (RECIPE_DIR / f"{training_recipe}.toml").open("rb") as handle:
        member = tomllib.load(handle)["record"]["members"][0]
    checkpoint_file = paths.path(folder) / f"{member}.pt"
    stored_file = paths.path("outputs.predictions") / f"nn_{member}_trainonly_val.npy"
    need(checkpoint_file, stored_file, paths.path("data.flat") / "val" / "feeds.parquet",
         paths.path("data.feature_cache") / "val" / "pre2_fe_s.npy")
    checkpoint = read_checkpoint(checkpoint_file)
    assert checkpoint.config == load_recipe(recipe)                   # the recipe describes the stored member
    model = build_ranker(checkpoint).to("cuda")
    assert sum(p.numel() for p in model.parameters()) == n_parameters
    split = load_split("val", checkpoint.config.view, "cuda")
    scores = predict(model, split)
    stored = np.load(stored_file)
    assert np.abs(scores - stored).max() <= 2e-3
