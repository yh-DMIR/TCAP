"""Small method-only command line: adapt, then predict."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import write_json
from .backends import AdaptationConfig
from .workflow import adapt_task, predict_task


def parser():
    p = argparse.ArgumentParser(description="TCAP checkpoint retrieval for TabICL-v2 / TabPFN-v3")
    commands = p.add_subparsers(dest="command", required=True)
    adapt = commands.add_parser("adapt", help="Adapt one task from the shared base; no test inference")
    adapt.add_argument("--backend", required=True, choices=("tabicl", "tabpfn"))
    adapt.add_argument("--base", required=True, type=Path)
    adapt.add_argument("--data", required=True, type=Path)
    adapt.add_argument("--task-key", required=True)
    adapt.add_argument("--label", default=None)
    adapt.add_argument("--output", required=True, type=Path)
    adapt.add_argument("--device", default="cuda")
    adapt.add_argument("--seed", type=int, default=42)
    adapt.add_argument("--epochs", type=int, default=30)
    adapt.add_argument("--learning-rate", type=float, default=1e-5)
    adapt.add_argument("--weight-decay", type=float, default=0.01)
    adapt.add_argument("--batch-size", type=int, default=4)
    predict = commands.add_parser("predict", help="TCAP top-1 prediction; optional --rerank")
    predict.add_argument("--target", required=True, type=Path, help="Target adaptation task.json")
    predict.add_argument("--bank", required=True, type=Path, help="JSON list of source task.json paths")
    predict.add_argument("--data", required=True, type=Path, help="Same target CSV as adaptation")
    predict.add_argument("--output", required=True, type=Path)
    predict.add_argument("--rerank", action="store_true", help="Enable top-k + validation log-loss selection")
    predict.add_argument("--top-k", type=int, default=None, help="Default 1; default 5 with --rerank")
    predict.add_argument("--include-self", action=argparse.BooleanOptionalAction, default=None,
                         help="Default off for direct top-1, on for validation reranking")
    predict.add_argument("--mu", type=float, default=0.1)
    predict.add_argument("--tiny-energy-fallback", action="store_true",
                         help="Opt in to uniform module weights when total target energy <= 1e-12")
    predict.add_argument("--n-estimators", type=int, default=8)
    predict.add_argument("--device", default="cuda")
    return p


def main():
    args = parser().parse_args()
    if args.command == "adapt":
        config = AdaptationConfig(seed=args.seed, epochs=args.epochs,
                                  learning_rate=args.learning_rate,
                                  weight_decay=args.weight_decay, batch_size=args.batch_size)
        path = adapt_task(backend_name=args.backend, base=args.base, data=args.data,
                          task_key=args.task_key, output=args.output, label=args.label,
                          config=config, device=args.device)
        print(f"Saved task artifact: {path}")
    else:
        if args.output.exists():
            raise FileExistsError(f"Use a fresh prediction output directory: {args.output}")
        result = predict_task(target=args.target, bank=args.bank, data=args.data,
                              rerank=args.rerank, top_k=args.top_k, include_self=args.include_self,
                              mu=args.mu, device=args.device, n_estimators=args.n_estimators,
                              tiny_energy_fallback=args.tiny_energy_fallback)
        args.output.mkdir(parents=True, exist_ok=False)
        arrays = {key: result[key] for key in ("probabilities", "classes", "predictions", "row_indices")}
        for key in ("classes", "predictions"):
            if arrays[key].dtype.kind == "O":
                arrays[key] = arrays[key].astype(str)
        np.savez_compressed(args.output / "predictions.npz",
                            **arrays)
        write_json(args.output / "selection.json", result["selection"])
        print(f"Selected {result['selection']['selected_source_key']}; predictions saved to {args.output}")
