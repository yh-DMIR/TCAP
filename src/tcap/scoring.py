"""Mean module energy, target geometry, and one-sided projection scoring."""

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

EPSILON = 1e-12
Spans = Mapping[str, list[tuple[int, int]]]


def _validate_vector(vector, spans: Spans) -> np.ndarray:
    vector = np.asarray(vector)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("Displacements must be finite one-dimensional vectors")
    ordered = sorted((a, b) for ranges in spans.values() for a, b in ranges)
    offset = 0
    if not spans or any(not ranges for ranges in spans.values()):
        raise ValueError("Every module must contain parameters")
    for start, end in ordered:
        if start != offset or end <= start:
            raise ValueError("Module spans must partition the vector exactly once")
        offset = end
    if offset != len(vector):
        raise ValueError("Layout and displacement dimensions differ")
    return vector


def _module_dot(left, right, ranges) -> float:
    # Accumulate module inner products in float64.
    return sum(float(np.dot(np.asarray(left[a:b], dtype=np.float64),
                            np.asarray(right[a:b], dtype=np.float64)))
               for a, b in ranges)


def target_weights(target, spans: Spans, *, mu: float = 0.1,
                   eps: float = EPSILON,
                   tiny_energy_fallback: bool = False) -> dict[str, float]:
    """Normalize positive energy as in the paper; uniform fallback is opt-in.

    With tiny_energy_fallback=True, total energy <= eps uses uniform weights.
    Zero total energy is undefined by the paper formula and otherwise raises.
    """
    if not 0 <= mu <= 1 or not np.isfinite(mu) or not np.isfinite(eps) or eps <= 0:
        raise ValueError("Expected mu in [0,1] and eps > 0")
    target = _validate_vector(target, spans)
    energy = {name: _module_dot(target, target, ranges)
              / sum(b - a for a, b in ranges) for name, ranges in spans.items()}
    total = sum(energy.values())
    if tiny_energy_fallback and total <= eps:
        raw = {name: 1 / len(spans) for name in spans}
    elif total > 0:
        raw = {name: value / total for name, value in energy.items()}
    else:
        raise ValueError(
            "Target displacement has zero total module energy; the paper's "
            "weight normalization is undefined. Set tiny_energy_fallback=True "
            "(CLI: --tiny-energy-fallback) to use uniform weights."
        )
    return {name: (1 - mu) * value + mu / len(spans) for name, value in raw.items()}


def tcap_score(source, target, spans: Spans, *, mu: float = 0.1,
               eps: float = EPSILON, tiny_energy_fallback: bool = False) -> float:
    source = _validate_vector(source, spans)
    target = _validate_vector(target, spans)
    weights = target_weights(target, spans, mu=mu, eps=eps,
                             tiny_energy_fallback=tiny_energy_fallback)
    numerator = sum(weights[name] * _module_dot(target, source, ranges)
                    for name, ranges in spans.items())
    norm_sq = sum(weights[name] * _module_dot(target, target, ranges)
                  for name, ranges in spans.items())
    # No source-side norm: retaining candidate scale is essential to TCAP.
    return float(numerator / (np.sqrt(max(norm_sq, 0.0)) + eps))


@dataclass(frozen=True)
class RankedCandidate:
    key: str
    score: float
    rank: int


def rank_candidates(target, candidates: Iterable[tuple[str, np.ndarray]], spans: Spans,
                    *, target_key: str, include_self: bool = False,
                    mu: float = 0.1,
                    tiny_energy_fallback: bool = False) -> list[RankedCandidate]:
    """Score only parameters. This function accepts no features or labels."""
    target = _validate_vector(target, spans)
    weights = target_weights(target, spans, mu=mu,
                             tiny_energy_fallback=tiny_energy_fallback)
    norm_sq = sum(weights[name] * _module_dot(target, target, ranges)
                  for name, ranges in spans.items())
    denominator = np.sqrt(max(norm_sq, 0.0)) + EPSILON
    rows = []
    seen = set()
    for key, source in candidates:
        if key in seen:
            raise ValueError(f"Duplicate candidate key: {key}")
        seen.add(key)
        if not include_self and key == target_key:
            continue
        source = _validate_vector(source, spans)
        numerator = sum(weights[name] * _module_dot(target, source, ranges)
                        for name, ranges in spans.items())
        rows.append((key, float(numerator / denominator)))
    rows.sort(key=lambda row: (-row[1], row[0]))
    if not rows:
        raise ValueError("No eligible candidates remain after self exclusion")
    return [RankedCandidate(key, score, rank)
            for rank, (key, score) in enumerate(rows, 1)]
