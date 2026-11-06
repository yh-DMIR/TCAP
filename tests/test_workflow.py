"""Spy backend verifies inference budgets and exact context membership."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcap.artifacts import load_artifact, load_bank, sha256_file, write_json
from tcap.backends import AdaptationConfig
from tcap.layout import Layout, Parameter
from tcap.splits import make_task_split
from tcap import workflow


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    data = tmp_path / "target.csv"
    pd.DataFrame({"row_id": np.arange(1000), "y": np.tile([0, 1], 500)}).to_csv(data, index=False)
    split = make_task_split(np.tile([0, 1], 500))
    layout = Layout((Parameter("dummy", (2,), 0, 2, "module"),))
    config = AdaptationConfig()
    protocol = {"backend": "spy", "backend_package_version": "test", "base_sha256": "shared",
                "layout_fingerprint": layout.fingerprint,
                "split": "stratified-64-8-8-20-v1", "adaptation": asdict(config)}

    def artifact(key, delta, data_sha=None):
        out = tmp_path / key
        out.mkdir()
        checkpoint = out / "best.ckpt"
        checkpoint.write_text(key)
        np.save(out / "delta.npy", np.array(delta, dtype=np.float32))
        write_json(out / "task.json", {
            "schema_version": 1, "task_key": key, "data_sha256": data_sha or key,
            "label_column": "y", "checkpoint": "best.ckpt", "delta": "delta.npy",
            "checkpoint_sha256": sha256_file(checkpoint),
            "delta_sha256": sha256_file(out / "delta.npy"),
            "protocol": protocol, "layout": layout.to_dict(),
            "split_indices": split.to_dict(),
        })
        return out / "task.json"

    target = artifact("target", [1, 1], sha256_file(data))
    sources = [artifact(f"source{i}", [6-i, 6-i]) for i in range(5)]
    bank = tmp_path / "bank.json"
    write_json(bank, [str(p.relative_to(tmp_path)) for p in sources])
    calls = []

    class SpyBackend:
        def predict_proba(self, checkpoint, X_context, y_context, X_query, **kwargs):
            key = Path(checkpoint).read_text()
            calls.append((key, X_context.row_id.to_numpy(), X_query.row_id.to_numpy()))
            p = .95 if key == "source2" else .6
            # Row id is deliberately predictive for this synthetic spy dataset.
            prob = np.where(X_query.row_id.to_numpy() % 2 == 0, p, 1-p)
            return np.column_stack((prob, 1-prob)), np.array([0, 1])

    monkeypatch.setattr(workflow, "get_backend", lambda name: SpyBackend())
    monkeypatch.setattr(workflow, "cleanup", lambda: None)
    return data, target, bank, split, calls, artifact


def test_direct_workflow_predicts_only_top1_once(fixture):
    data, target, bank, split, calls, _ = fixture
    result = workflow.predict_task(target=target, bank=bank, data=data, device="cpu")
    assert result["selection"]["selected_source_key"] == "source0"
    assert result["selection"]["include_self"] is False
    assert result["selection"]["tiny_energy_fallback"] is False
    assert len(calls) == 1
    assert set(calls[0][1]) == set(split.final_context)
    assert set(calls[0][2]) == set(split.test)
    assert "test_metrics" not in result


@pytest.mark.parametrize("rerank", [False, True])
def test_zero_target_workflow_requires_opt_in_before_inference(fixture, rerank):
    data, target, bank, split, calls, _ = fixture
    delta = target.parent / "delta.npy"
    np.save(delta, np.zeros(2, dtype=np.float32))
    payload = json.loads(target.read_text())
    payload["delta_sha256"] = sha256_file(delta)
    write_json(target, payload)
    with pytest.raises(ValueError, match="zero total module energy"):
        workflow.predict_task(target=target, bank=bank, data=data, rerank=rerank, device="cpu")
    assert not calls
    result = workflow.predict_task(target=target, bank=bank, data=data, rerank=rerank,
                                   device="cpu", tiny_energy_fallback=True)
    assert result["selection"]["tiny_energy_fallback"] is True
    assert all(row["score"] == 0 for row in result["selection"]["ranking"])
    assert len(calls) == (6 if rerank else 1)
    assert set(calls[-1][1]) == set(split.final_context)


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_records_explicit_fallback_setting(fixture, monkeypatch, enabled):
    from tcap import cli
    data, target, bank, _, calls, _ = fixture
    output = target.parent / "predictions"
    argv = ["tcap", "predict", "--target", str(target), "--bank", str(bank),
            "--data", str(data), "--output", str(output), "--device", "cpu"]
    if enabled:
        argv.append("--tiny-energy-fallback")
    monkeypatch.setattr("sys.argv", argv)
    cli.main()
    selection = json.loads((output / "selection.json").read_text())
    assert selection["tiny_energy_fallback"] is enabled
    assert len(calls) == 1
    with np.load(output / "predictions.npz", allow_pickle=False) as predictions:
        assert predictions["probabilities"].shape == (200, 2)


def test_rerank_makes_five_validation_calls_then_one_final_prediction(fixture):
    data, target, bank, split, calls, _ = fixture
    result = workflow.predict_task(target=target, bank=bank, data=data, rerank=True, device="cpu")
    assert result["selection"]["selected_source_key"] == "source2"
    assert result["selection"]["include_self"] is True
    assert len(calls) == 6
    for _, context, query in calls[:5]:
        assert set(context) == set(split.rerank_context)
        assert set(query) == set(split.rerank)
        assert not set(context) & set(query)
    assert calls[-1][0] == "source2"
    assert set(calls[-1][1]) == set(split.final_context)
    assert set(calls[-1][2]) == set(split.test)
    # Self is ranked normally; source0..4 outrank self, so self is not forced into top-5.
    assert "target" in [row["key"] for row in result["selection"]["ranking"]]
    assert "target" not in [row["key"] for row in result["selection"]["validation"]]


def test_bank_rejects_mixed_bases_or_adaptation_protocols(fixture):
    data, target, bank, _, _, _ = fixture
    source = bank.parent / "source0/task.json"
    payload = json.loads(source.read_text())
    payload["protocol"]["base_sha256"] = "different-base"
    write_json(source, payload)
    with pytest.raises(ValueError, match="Incompatible"):
        load_bank(bank, load_artifact(target), include_self=False)


def test_changed_target_data_or_checkpoint_is_rejected(fixture):
    data, target, bank, _, _, _ = fixture
    data.write_text(data.read_text() + "1000,0\n")
    with pytest.raises(ValueError, match="CSV differs"):
        workflow.predict_task(target=target, bank=bank, data=data)
    (target.parent / "best.ckpt").write_text("different")
    with pytest.raises(ValueError, match="checkpoint content changed"):
        load_artifact(target)


def test_same_dataset_under_alias_is_not_external(fixture):
    data, target, bank, _, _, artifact = fixture
    alias = artifact("alias", [2, 2], sha256_file(data))
    write_json(bank, [str(alias.relative_to(bank.parent))])
    with pytest.raises(ValueError, match="different task key"):
        load_bank(bank, load_artifact(target), include_self=False)


def test_adaptation_receives_only_train_and_explicit_early_stopping_data(tmp_path, monkeypatch):
    data, base = tmp_path / "input.csv", tmp_path / "base.ckpt"
    y = np.tile([0, 1], 500)
    pd.DataFrame({"row_id": np.arange(1000), "y": y}).to_csv(data, index=False)
    base.write_text("base")
    split = make_task_split(y)
    layout = Layout((Parameter("dummy", (2,), 0, 2, "module"),))

    class SpyBackend:
        def layout(self, base):
            return layout

    def fit(backend, base, X_train, y_train, X_val, y_val, config, output, device):
        assert set(X_train.row_id) == set(split.train)
        assert set(X_val.row_id) == set(split.early_stop)
        assert not set(X_train.row_id) & set(split.rerank)
        assert not set(X_val.row_id) & set(split.test)
        checkpoint = output / "best.ckpt"
        checkpoint.write_text("adapted")
        return checkpoint, [{"batch_size": 4, "status": "success"}]

    monkeypatch.setattr(workflow, "get_backend", lambda name: SpyBackend())
    monkeypatch.setattr(workflow, "fit_with_oom_fallback", fit)
    monkeypatch.setattr(workflow, "cleanup", lambda: None)
    monkeypatch.setattr(workflow, "version", lambda name: "test")
    monkeypatch.setattr(workflow, "_write_delta",
                        lambda backend, base, checkpoint, layout, output:
                        np.save(output, np.array([1, 1], dtype=np.float32)))
    task = workflow.adapt_task(backend_name="spy", base=base, data=data,
                               task_key="target", output=tmp_path / "artifact")
    assert load_artifact(task).metadata["split_indices"] == split.to_dict()
