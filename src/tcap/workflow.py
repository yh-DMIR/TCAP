"""Method workflow: adapt once, score, select, predict with target context."""

from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import SCHEMA_VERSION, load_artifact, load_bank, sha256_file, write_json
from .backends import AdaptationConfig, cleanup, fit_with_oom_fallback, get_backend
from .scoring import rank_candidates
from .selection import select_candidate
from .splits import make_task_split


def read_csv(path: str | Path, *, label: str | None = None):
    frame = pd.read_csv(path)
    target = label if label is not None else ("y" if "y" in frame.columns else frame.columns[-1])
    if target not in frame.columns or frame.shape[1] < 2 or frame[target].isna().any():
        raise ValueError("Expected a CSV with feature columns and a complete label column")
    X, y = frame.drop(columns=[target]), frame[target].to_numpy()
    return X, y, target


def _take(X, indices):
    return X.iloc[indices].reset_index(drop=True)


def _write_delta(backend, base, checkpoint, layout, output):
    import torch
    base_state = backend.load_state(base)
    task_state = backend.load_state(checkpoint)
    delta = np.lib.format.open_memmap(output, mode="w+", dtype="float32", shape=(layout.size,))
    for parameter in layout.parameters:
        for state in (base_state, task_state):
            if parameter.name not in state or tuple(state[parameter.name].shape) != parameter.shape:
                raise ValueError(f"Missing or incompatible parameter: {parameter.name}")
        value = (task_state[parameter.name].detach().cpu().to(torch.float32)
                 - base_state[parameter.name].detach().cpu().to(torch.float32))
        delta[parameter.start:parameter.end] = value.reshape(-1).numpy()
    delta.flush()
    del delta, base_state, task_state


def adapt_task(*, backend_name: str, base: str | Path, data: str | Path,
               task_key: str, output: str | Path, label: str | None = None,
               config: AdaptationConfig | None = None, device: str = "cuda") -> Path:
    """Create one source/target artifact. Never evaluate its test predictions."""
    config = config or AdaptationConfig()
    base, data, output = Path(base).resolve(), Path(data).resolve(), Path(output).resolve()
    if not base.is_file():
        raise FileNotFoundError(base)
    if not task_key.strip():
        raise ValueError("A nonempty task_key is required")
    X, y, label = read_csv(data, label=label)
    if len(np.unique(y)) > 10:
        raise ValueError("Task adaptation supports at most 10 classes")
    split = make_task_split(y, seed=config.seed)
    backend = get_backend(backend_name)
    layout = backend.layout(base)
    base_sha256, data_sha256 = sha256_file(base), sha256_file(data)
    # Each adaptation writes to a fresh output directory.
    output.mkdir(parents=True, exist_ok=False)
    print(f"Adapting {task_key}: train={len(split.train)}, early_stop={len(split.early_stop)}, "
          f"reserved_rerank={len(split.rerank)}, reserved_test={len(split.test)}", flush=True)
    try:
        checkpoint, attempts = fit_with_oom_fallback(
            backend, base, _take(X, split.train), y[split.train],
            _take(X, split.early_stop), y[split.early_stop], config, output, device,
        )
        cleanup()
        _write_delta(backend, base, checkpoint, layout, output / "delta.npy")
    finally:
        cleanup()
    write_json(output / "task.json", {
        "schema_version": SCHEMA_VERSION, "task_key": task_key,
        "data_sha256": data_sha256, "label_column": label,
        "checkpoint": checkpoint.relative_to(output).as_posix(), "delta": "delta.npy",
        "checkpoint_sha256": sha256_file(checkpoint),
        "delta_sha256": sha256_file(output / "delta.npy"),
        "protocol": {"backend": backend_name, "backend_package_version": version(backend_name),
                     "base_sha256": base_sha256, "layout_fingerprint": layout.fingerprint,
                     "split": "stratified-64-8-8-20-v1", "adaptation": asdict(config)},
        "layout": layout.to_dict(), "split_indices": split.to_dict(), "oom_attempts": attempts,
    })
    return output / "task.json"


def predict_task(*, target: str | Path, bank: str | Path, data: str | Path,
                 rerank: bool = False, top_k: int | None = None,
                 include_self: bool | None = None, mu: float = 0.1,
                 device: str = "cuda", n_estimators: int = 8,
                 tiny_energy_fallback: bool = False):
    """Return predictions on the prescribed held-out rows, without test metrics.

    Direct: external top-1. Rerank: include-self top-5.
    Explicit include_self overrides the corresponding mode default.
    Tiny-energy uniform weighting is disabled unless explicitly requested.
    """
    if top_k is None:
        top_k = 5 if rerank else 1
    if top_k < 1 or (not rerank and top_k != 1) or n_estimators < 1:
        raise ValueError("Use top_k=1 for direct prediction; top_k>1 requires rerank=True")
    include_self = rerank if include_self is None else include_self
    task = load_artifact(target)
    data = Path(data).resolve()
    if sha256_file(data) != task.metadata["data_sha256"]:
        raise ValueError("Target CSV differs from the data used for target adaptation")
    X, y, _ = read_csv(data, label=task.metadata["label_column"])
    seed = task.metadata["protocol"]["adaptation"]["seed"]
    split = make_task_split(y, seed=seed)
    if split.to_dict() != task.metadata["split_indices"]:
        raise ValueError("Split implementation/environment differs from target adaptation")
    sources = load_bank(bank, task, include_self=include_self)
    ranking = rank_candidates(task.delta(), ((key, source.delta()) for key, source in sources.items()),
                              task.layout.spans, target_key=task.key, include_self=include_self, mu=mu,
                              tiny_energy_fallback=tiny_energy_fallback)
    backend = get_backend(task.metadata["protocol"]["backend"])

    def infer(key, context_indices, query_indices):
        try:
            return backend.predict_proba(
                sources[key].checkpoint, _take(X, context_indices), y[context_indices],
                _take(X, query_indices), device=device, seed=seed, n_estimators=n_estimators,
            )
        finally:
            cleanup()

    def validation(key):
        proba, classes = infer(key, split.rerank_context, split.rerank)
        return proba, classes, y[split.rerank]

    selection = select_candidate(ranking, top_k=top_k,
                                 evaluate_validation=validation if rerank else None)
    # Selection is fixed before the only test-query prediction. No test labels passed.
    proba, classes = infer(selection.selected.key, split.final_context, split.test)
    return {
        "probabilities": proba, "classes": classes,
        "predictions": classes[proba.argmax(axis=1)], "row_indices": split.test,
        "selection": {"target_key": task.key, "selected_source_key": selection.selected.key,
                      "mode": "topk+val" if rerank else "top1",
                      "include_self": include_self, "top_k": top_k, "mu": mu,
                      "n_estimators": n_estimators,
                      "tiny_energy_fallback": tiny_energy_fallback,
                      "ranking": [asdict(row) for row in ranking],
                      "validation": list(selection.validation),
                      "context_sizes": {"rerank": len(split.rerank_context),
                                        "final": len(split.final_context)}},
    }
