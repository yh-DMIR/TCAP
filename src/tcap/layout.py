"""Strict model-specific parameter partitions: 15 (TabICL), 34 (TabPFN)."""

from dataclasses import asdict, dataclass
import hashlib
import json
import re


def tabicl_module(name: str) -> str:
    if name.startswith("col_embedder."):
        return "col_embedder"
    if name.startswith("row_interactor."):
        return "row_interactor"
    match = re.match(r"^icl_predictor\.tf_icl\.blocks\.(\d+)\.", name)
    if match and 0 <= int(match[1]) < 12:
        return f"icl_block_{int(match[1])}"
    if name.startswith(("icl_predictor.y_encoder.", "icl_predictor.ln.",
                        "icl_predictor.decoder.")):
        return "icl_output"
    raise ValueError(f"Unassigned TabICL-v2 parameter: {name}")


def tabpfn_module(name: str) -> str:
    if name.startswith(("x_embed.", "col_y_encoder.")):
        return "input_target_embedding"
    for pattern, prefix in (
        (r"^feature_distribution_embedder\.layers\.(\d+)\.", "feature_distribution_block"),
        (r"^column_aggregator\.blocks\.(\d+)\.", "column_aggregator_block"),
        (r"^icl_blocks\.(\d+)\.", "icl_block"),
    ):
        match = re.match(pattern, name)
        if match:
            return f"{prefix}_{int(match[1])}"
    if name.startswith("column_aggregator."):
        return "column_aggregator_support"
    if name.startswith("icl_y_encoder."):
        return "icl_target_encoder"
    if name.startswith(("output_norm.", "many_class_decoder.", "output_projection.")):
        return "output_decoder"
    raise ValueError(f"Unassigned TabPFN-v3 parameter: {name}")


@dataclass(frozen=True)
class Parameter:
    name: str
    shape: tuple[int, ...]
    start: int
    end: int
    module: str


@dataclass(frozen=True)
class Layout:
    parameters: tuple[Parameter, ...]

    @property
    def spans(self) -> dict[str, list[tuple[int, int]]]:
        result = {}
        for p in self.parameters:
            result.setdefault(p.module, []).append((p.start, p.end))
        return result

    @property
    def size(self) -> int:
        return self.parameters[-1].end

    def to_dict(self) -> dict:
        return {"parameters": [asdict(p) for p in self.parameters]}

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict):
        parameters = tuple(Parameter(p["name"], tuple(p["shape"]), p["start"],
                                     p["end"], p["module"]) for p in payload["parameters"])
        if not parameters:
            raise ValueError("Empty parameter layout")
        offset = 0
        seen = set()
        for p in parameters:
            size = 1
            for dim in p.shape:
                size *= dim
            if p.name in seen or p.start != offset or p.end - p.start != size or size <= 0:
                raise ValueError(f"Invalid parameter layout entry: {p.name}")
            seen.add(p.name)
            offset = p.end
        return cls(parameters)


def build_layout(named_parameters, backend: str) -> Layout:
    mapper = {"tabicl": tabicl_module, "tabpfn": tabpfn_module}[backend]
    parameters, offset = [], 0
    # TabICL includes fixed RoPE frequencies with zero displacement;
    # TabPFN uses trainable parameters only.
    for name, tensor in sorted(named_parameters, key=lambda item: item[0]):
        if not tensor.requires_grad and backend == "tabpfn":
            continue
        end = offset + tensor.numel()
        parameters.append(Parameter(name, tuple(tensor.shape), offset, end, mapper(name)))
        offset = end
    layout = Layout.from_dict({"parameters": [asdict(p) for p in parameters]})
    expected = 15 if backend == "tabicl" else 34
    if len(layout.spans) != expected:
        raise ValueError(f"Expected {expected} modules for {backend}, got {len(layout.spans)}")
    expected_blocks = {f"icl_block_{i}" for i in range(12 if backend == "tabicl" else 24)}
    if not expected_blocks <= layout.spans.keys():
        raise ValueError(f"{backend} checkpoint has an incompatible ICL block layout")
    return layout
