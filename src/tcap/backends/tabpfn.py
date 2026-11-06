"""TabPFN-v3 adapter with 32-view mean-loss microbatch training."""

from pathlib import Path

import numpy as np

from ..layout import build_layout


class TabPFNBackend:
    name = "tabpfn"

    def layout(self, base: Path):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        classifier = TabPFNClassifier.create_default_for_version(
            ModelVersion.V3, model_path=base, n_estimators=1,
            device="cpu", ignore_pretraining_limits=True,
        )
        classifier._initialize_model_variables()
        return build_layout(classifier.model_.named_parameters(), self.name)

    def load_state(self, path: Path):
        from tabpfn.checkpoint import Checkpoint
        state = Checkpoint(path).load().get("state_dict")
        if not isinstance(state, dict):
            raise ValueError(f"Missing state_dict in {path}")
        return state

    def finetune(self, base, X_train, y_train, X_val, y_val, config, output, device, batch):
        from tabpfn.constants import ModelVersion
        from tabpfn.model_loading import save_tabpfn_model
        from ._tabpfn_finetune.finetuned_classifier import FinetunedTabPFNClassifier
        classifier = FinetunedTabPFNClassifier(
            device=device, epochs=config.epochs, learning_rate=config.learning_rate,
            weight_decay=config.weight_decay, validation_split_ratio=None,
            n_finetune_ctx_plus_query_samples=config.max_data_size,
            finetune_ctx_query_split_ratio=0.2, random_state=config.seed,
            early_stopping=True, early_stopping_patience=config.patience,
            grad_clip_value=1.0, use_lr_scheduler=True, lr_warmup_only=False,
            n_estimators_finetune=config.n_estimators_finetune,
            n_estimators_validation=config.n_estimators_validation,
            n_estimators_final_inference=config.n_estimators_inference,
            finetune_batch_size=batch, use_activation_checkpointing=True,
            save_checkpoint_interval=None, use_fixed_preprocessing_seed=False,
            eval_metric="roc_auc", model_version=ModelVersion.V3,
            extra_classifier_kwargs={"model_path": base},
        )
        classifier.fit(X_train, y_train, X_val=X_val, y_val=y_val, output_dir=output)
        checkpoint = output / "TabPFN-best.ckpt"
        # fit() restores best early-stopping weights before this export.
        save_tabpfn_model(classifier.finetuned_estimator_, checkpoint)
        return checkpoint

    def predict_proba(self, checkpoint, X_context, y_context, X_query, *, device,
                      seed, n_estimators):
        from tabpfn import TabPFNClassifier
        classifier = TabPFNClassifier(
            model_path=checkpoint, device=device, n_estimators=n_estimators,
            random_state=seed, ignore_pretraining_limits=True,
        )
        classifier.fit(X_context, y_context)
        return np.asarray(classifier.predict_proba(X_query)), np.asarray(classifier.classes_)
