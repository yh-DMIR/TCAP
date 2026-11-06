"""Direct top-1 or optional validation reranking; test data is never accepted."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from .scoring import RankedCandidate


@dataclass(frozen=True)
class Selection:
    selected: RankedCandidate
    validation: tuple[dict, ...] = ()


def select_candidate(ranking: list[RankedCandidate], *, top_k: int = 1,
                     evaluate_validation: Callable | None = None) -> Selection:
    """Without a validation callback, return top-1 with no model inference.

    Callback receives a source key and returns (probabilities, classes, y_val).
    Ties: lower log loss, higher ACC, earlier TCAP rank, source key.
    Evaluation errors propagate, rather than silently shrinking the top-k pool.
    """
    if not ranking or top_k < 1:
        raise ValueError("Expected a nonempty ranking and positive top_k")
    if evaluate_validation is None:
        if top_k != 1:
            raise ValueError("top_k > 1 requires validation reranking")
        return Selection(ranking[0])
    if len(ranking) < top_k:
        raise ValueError(f"Requested top-{top_k}, but only {len(ranking)} candidates are available")
    rows = []
    for candidate in ranking[:top_k]:
        proba, classes, y_val = evaluate_validation(candidate.key)
        proba, classes, y_val = np.asarray(proba), np.asarray(classes), np.asarray(y_val)
        if (proba.shape != (len(y_val), len(classes)) or not np.isfinite(proba).all()
                or (proba < 0).any() or (proba > 1).any()
                or not np.allclose(proba.sum(axis=1), 1, rtol=1e-5, atol=1e-6)):
            raise ValueError(f"Invalid validation probabilities from {candidate.key}")
        rows.append({"key": candidate.key, "rank": candidate.rank,
                     "log_loss": float(log_loss(y_val, proba, labels=classes)),
                     "accuracy": float(accuracy_score(y_val, classes[proba.argmax(axis=1)]))})
    best = min(rows, key=lambda row: (row["log_loss"], -row["accuracy"], row["rank"], row["key"]))
    chosen = next(candidate for candidate in ranking if candidate.key == best["key"])
    return Selection(chosen, tuple(rows))
