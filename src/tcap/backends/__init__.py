"""Lazy model adapters; inference always uses an explicitly selected checkpoint."""

from dataclasses import dataclass
import gc
from pathlib import Path
import random

import numpy as np


@dataclass(frozen=True)
class AdaptationConfig:
    seed: int = 42
    epochs: int = 30
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    batch_size: int = 4
    patience: int = 8
    n_estimators_finetune: int = 32
    n_estimators_validation: int = 32
    n_estimators_inference: int = 8
    max_data_size: int = 10_000

    def __post_init__(self):
        if any(value < 1 for value in (
            self.epochs, self.batch_size, self.patience, self.n_estimators_finetune,
            self.n_estimators_validation, self.n_estimators_inference, self.max_data_size,
        )) or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid fine-tuning configuration")


def cleanup():
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def seed_training(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_backend(name: str):
    if name == "tabicl":
        from .tabicl import TabICLBackend
        return TabICLBackend()
    if name == "tabpfn":
        from .tabpfn import TabPFNBackend
        return TabPFNBackend()
    raise ValueError(f"Unknown backend: {name}")


def fit_with_oom_fallback(backend, base: Path, X_train, y_train, X_val, y_val,
                          config: AdaptationConfig, output: Path, device: str):
    """Retry from base/epoch zero at 4 -> 2 -> 1, with separate attempt dirs."""
    import torch
    batch = config.batch_size
    attempts = []
    while True:
        seed_training(config.seed)
        attempt = output / f"batch_{batch}"
        attempt.mkdir(parents=True, exist_ok=False)
        oom = False
        try:
            checkpoint = backend.finetune(
                base, X_train, y_train, X_val, y_val, config, attempt, device, batch)
        except RuntimeError as exc:
            if not (isinstance(exc, torch.cuda.OutOfMemoryError)
                    or "cuda out of memory" in str(exc).lower()
                    or "cuda error: out of memory" in str(exc).lower()):
                raise
            if batch == 1:
                raise
            attempts.append({"batch_size": batch, "status": "cuda_oom"})
            oom = True
        # Leave the except block before GC, releasing the failed traceback/model.
        if not oom:
            attempts.append({"batch_size": batch, "status": "success"})
            return checkpoint, attempts
        cleanup()
        batch = max(1, batch // 2)
        print(f"CUDA OOM: restart adaptation from the shared base at batch {batch}", flush=True)
