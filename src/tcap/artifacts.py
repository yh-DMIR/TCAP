"""Portable task artifacts with shared-base and adaptation-protocol checks."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .layout import Layout

SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class TaskArtifact:
    path: Path
    metadata: dict
    layout: Layout

    @property
    def key(self) -> str:
        return self.metadata["task_key"]

    @property
    def checkpoint(self) -> Path:
        return (self.path.parent / self.metadata["checkpoint"]).resolve()

    def delta(self):
        vector = np.load(self.path.parent / self.metadata["delta"], mmap_mode="r",
                         allow_pickle=False)
        if vector.shape != (self.layout.size,):
            raise ValueError(f"Wrong displacement dimensions for {self.key}")
        return vector


def load_artifact(path: str | Path) -> TaskArtifact:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported task artifact schema: {path}")
    layout = Layout.from_dict(payload["layout"])
    if layout.fingerprint != payload["protocol"]["layout_fingerprint"]:
        raise ValueError(f"Parameter layout fingerprint mismatch: {path}")
    result = TaskArtifact(path, payload, layout)
    for field in ("checkpoint", "delta"):
        file = path.parent / payload[field]
        if not file.is_file():
            raise FileNotFoundError(file)
        if sha256_file(file) != payload[field + "_sha256"]:
            raise ValueError(f"Artifact {field} content changed: {file}")
    return result


def check_compatible(target: TaskArtifact, source: TaskArtifact):
    if target.metadata["protocol"] != source.metadata["protocol"]:
        raise ValueError(
            f"Incompatible task {source.key}: backend, shared base, parameter layout, "
            "package version, split protocol and adaptation settings must match target"
        )
    if source.key == target.key and source.metadata["data_sha256"] != target.metadata["data_sha256"]:
        raise ValueError(f"Task key {source.key} refers to different input data")


def load_bank(path: str | Path, target: TaskArtifact, *, include_self: bool):
    """JSON list of task.json paths, resolved relative to the bank manifest."""
    path = Path(path).expanduser().resolve()
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not all(isinstance(item, str) for item in entries):
        raise ValueError("A bank must be a JSON list of task artifact paths")
    sources = {}
    for entry in entries:
        source = load_artifact(path.parent / entry)
        check_compatible(target, source)
        if source.key in sources:
            raise ValueError(f"Duplicate bank task key: {source.key}")
        # Same data under an alias is still self, not an external source.
        if source.key != target.key and source.metadata["data_sha256"] == target.metadata["data_sha256"]:
            raise ValueError(f"Source {source.key} is target data under a different task key")
        if include_self or source.key != target.key:
            sources[source.key] = source
    if include_self:
        # The target's own checkpoint competes by score; it is not forced into top-k.
        sources[target.key] = target
    if not sources:
        raise ValueError("No eligible candidates remain in the bank")
    return sources
