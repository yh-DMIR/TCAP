"""Behavioral checks for task adaptation and checkpoint selection."""

import numpy as np
import pytest

from tcap.layout import tabicl_module, tabpfn_module
from tcap.scoring import rank_candidates, target_weights, tcap_score
from tcap.selection import select_candidate
from tcap.splits import make_task_split


def test_four_disjoint_stratified_sets_and_contexts():
    y = np.tile(np.arange(4), 250)
    split = make_task_split(y, seed=42)
    groups = [split.train, split.early_stop, split.rerank, split.test]
    assert [len(g) for g in groups] == [640, 80, 80, 200]
    assert len(set(np.concatenate(groups))) == len(y)
    for group in groups:
        assert np.array_equal(np.bincount(y[group]), np.full(4, len(group) // 4))
    assert len(split.rerank_context) == 720
    assert len(split.final_context) == 800
    assert not set(split.rerank_context) & set(split.rerank)
    assert not set(split.final_context) & set(split.test)
    assert split.to_dict() == make_task_split(y, seed=42).to_dict()
    assert split.to_dict() != make_task_split(y, seed=43).to_dict()


def test_small_classes_fail_without_unstratified_fallback():
    with pytest.raises(ValueError, match="stratified"):
        make_task_split([0] * 99 + [1])


def test_projection_uses_mean_module_energy_and_preserves_source_scale():
    target = np.array([2., 2., 1.])
    source = np.array([1., -1., 3.])
    spans = {"first": [(0, 2)], "second": [(2, 3)]}
    # Mean squared energies are 4 and 1; smooth weights are 0.77 and 0.23.
    assert target_weights(target, spans) == pytest.approx({"first": .77, "second": .23})
    expected = .69 / (np.sqrt(.77 * 8 + .23) + 1e-12)
    assert tcap_score(source, target, spans) == pytest.approx(expected)
    assert tcap_score(3 * source, target, spans) == pytest.approx(3 * expected)
    assert tcap_score(target, source, spans) != pytest.approx(expected)


def test_tiny_and_zero_displacements_are_deterministic():
    spans = {"a": [(0, 1)], "b": [(1, 2)]}
    assert tcap_score(np.ones(2), np.zeros(2), spans, tiny_energy_fallback=True) == 0
    assert np.isfinite(tcap_score(np.ones(2), np.full(2, 1e-14), spans))
    ranks = rank_candidates(np.zeros(2), [("z", np.ones(2)), ("a", np.ones(2))],
                            spans, target_key="target", tiny_energy_fallback=True)
    assert [r.key for r in ranks] == ["a", "z"]


@pytest.mark.parametrize("scale", [1e-3, 1e-7, 1e-14])
def test_default_normalizes_positive_energy_even_below_threshold(scale):
    target = scale * np.array([1., .1])
    sources = [("A", np.array([0., 20.])), ("B", np.array([1., 0.]))]
    spans = {"a": [(0, 1)], "b": [(1, 2)]}
    energy = target ** 2
    weights = .9 * energy / energy.sum() + .05
    assert list(target_weights(target, spans).values()) == pytest.approx(weights)
    ranking = rank_candidates(target, sources, spans, target_key="target")
    assert [row.key for row in ranking] == ["B", "A"]
    for key, source in sources:
        expected = np.dot(weights * target, source) / (np.sqrt(np.dot(weights, target ** 2)) + 1e-12)
        assert tcap_score(source, target, spans) == pytest.approx(expected)
        assert next(row.score for row in ranking if row.key == key) == pytest.approx(expected)


def test_tiny_energy_uniform_fallback_changes_ranking_only_when_enabled():
    target = np.array([1e-7, 1e-8])
    sources = [("A", np.array([0., 20.])), ("B", np.array([1., 0.]))]
    spans = {"a": [(0, 1)], "b": [(1, 2)]}
    assert target_weights(target, spans, tiny_energy_fallback=True) == {"a": .5, "b": .5}
    ranking = rank_candidates(target, sources, spans, target_key="target", tiny_energy_fallback=True)
    assert [row.key for row in ranking] == ["A", "B"]
    for row in ranking:
        source = dict(sources)[row.key]
        expected = np.dot(.5 * target, source) / (np.sqrt(.5 * np.dot(target, target)) + 1e-12)
        assert row.score == pytest.approx(expected)
        assert tcap_score(source, target, spans, tiny_energy_fallback=True) == pytest.approx(expected)
    larger = target * 1e4
    assert target_weights(larger, spans, tiny_energy_fallback=True) == target_weights(larger, spans)


def test_zero_energy_requires_explicit_fallback():
    target, source = np.zeros(2), np.ones(2)
    spans = {"a": [(0, 1)], "b": [(1, 2)]}
    with pytest.raises(ValueError, match="zero total module energy"):
        target_weights(target, spans)
    with pytest.raises(ValueError, match="zero total module energy"):
        tcap_score(source, target, spans)
    with pytest.raises(ValueError, match="zero total module energy"):
        rank_candidates(target, [("source", source)], spans, target_key="target")


def test_offdiag_excludes_self_before_retrieval_and_direct_calls_no_validation():
    ranks = rank_candidates(np.ones(2), [("target", np.full(2, 100.)),
                            ("external", np.ones(2))], {"a": [(0, 2)]}, target_key="target")
    assert [r.key for r in ranks] == ["external"]
    assert select_candidate(ranks).selected.key == "external"
    with pytest.raises(ValueError, match="requires validation"):
        select_candidate(ranks, top_k=2)


def test_rerank_uses_only_retrieved_candidates_and_log_loss():
    ranking = rank_candidates(np.ones(2), [("a", np.full(2, 3.)),
                              ("b", np.full(2, 2.)), ("c", np.ones(2))],
                              {"a": [(0, 2)]}, target_key="target")
    called = []

    def evaluate(key):
        called.append(key)
        # Both have perfect ACC, but b has lower log loss.
        p = .6 if key == "a" else .9
        return np.array([[p, 1 - p], [1 - p, p]]), np.array([0, 1]), np.array([0, 1])

    result = select_candidate(ranking, top_k=2, evaluate_validation=evaluate)
    assert result.selected.key == "b"
    assert called == ["a", "b"]
    assert len(result.validation) == 2
    with pytest.raises(ValueError, match="only 3"):
        select_candidate(ranking, top_k=5, evaluate_validation=evaluate)


def test_layout_rejects_unknown_parameters_and_matches_native_names():
    assert tabicl_module("icl_predictor.tf_icl.blocks.11.attn.weight") == "icl_block_11"
    assert tabicl_module("icl_predictor.y_encoder.weight") == "icl_output"
    assert tabpfn_module("feature_distribution_embedder.layers.3.weight") == "feature_distribution_block_3"
    assert tabpfn_module("column_aggregator.cls_token") == "column_aggregator_support"
    assert tabpfn_module("icl_blocks.23.weight") == "icl_block_23"
    assert tabpfn_module("icl_y_encoder.weight") == "icl_target_encoder"
    with pytest.raises(ValueError):
        tabicl_module("icl_predictor.tf_icl.blocks.12.weight")
    with pytest.raises(ValueError):
        tabpfn_module("unknown.weight")


@pytest.mark.parametrize("spans", [
    {"a": [(0, 2)], "b": [(1, 3)]},  # overlap
    {"a": [(0, 1)], "b": [(2, 3)]},  # missing parameter
    {"a": [(0, 3)], "empty": []},
])
def test_invalid_parameter_partitions_fail(spans):
    with pytest.raises(ValueError):
        tcap_score(np.ones(3), np.ones(3), spans)
