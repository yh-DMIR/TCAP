"""TabICL-v2 adaptation and checkpoint inference."""

from pathlib import Path

import numpy as np

from ..layout import build_layout


class TabICLBackend:
    name = "tabicl"

    def layout(self, base: Path):
        from tabicl import TabICLClassifier
        classifier = TabICLClassifier(model_path=base, device="cpu", n_estimators=1,
                                      allow_auto_download=False)
        classifier._resolve_device()
        classifier._load_model()
        return build_layout(classifier.model_.named_parameters(), self.name)

    def load_state(self, path: Path):
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError(f"Missing state_dict in {path}")
        return state

    def finetune(self, base, X_train, y_train, X_val, y_val, config, output, device, batch):
        from ._tabicl_finetune.classifier import FinetunedTabICLClassifier
        classifier = FinetunedTabICLClassifier(
            model_path=base, allow_auto_download=False, device=device,
            epochs=config.epochs, learning_rate=config.learning_rate,
            weight_decay=config.weight_decay, finetune_batch_size=batch,
            n_estimators_finetune=config.n_estimators_finetune,
            n_estimators_validation=config.n_estimators_validation,
            n_estimators_inference=config.n_estimators_inference,
            max_data_size=config.max_data_size, finetune_ctx_query_ratio=0.2,
            eval_metric="roc_auc", early_stopping=True, patience=config.patience,
            freeze_col=False, freeze_row=False, freeze_icl=False,
            save_interval=0, random_state=config.seed,
        )
        classifier.fit(X_train, y_train, X_val=X_val, y_val=y_val, output_dir=output)
        checkpoint = output / "best.ckpt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        return checkpoint

    def predict_proba(self, checkpoint, X_context, y_context, X_query, *, device,
                      seed, n_estimators):
        from tabicl import TabICLClassifier
        classifier = TabICLClassifier(
            model_path=checkpoint, allow_auto_download=False, device=device,
            n_estimators=n_estimators, random_state=seed,
        )
        classifier.fit(X_context, y_context)
        return np.asarray(classifier.predict_proba(X_query)), np.asarray(classifier.classes_)
