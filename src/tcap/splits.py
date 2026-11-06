"""One stratified, disjoint split shared by adaptation and selection."""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class TaskSplit:
    train: np.ndarray
    early_stop: np.ndarray
    rerank: np.ndarray
    test: np.ndarray

    @property
    def rerank_context(self) -> np.ndarray:
        return np.concatenate((self.train, self.early_stop))

    @property
    def final_context(self) -> np.ndarray:
        return np.concatenate((self.train, self.early_stop, self.rerank))

    def to_dict(self) -> dict:
        return {name: getattr(self, name).tolist()
                for name in ("train", "early_stop", "rerank", "test")}


def make_task_split(y, *, seed: int = 42) -> TaskSplit:
    """64/8/8/20 percent; finite sample sizes follow sklearn rounding.

    Test labels are used here only to stratify the prescribed outer split.
    No unstratified fallback is allowed for small/rare classes.
    """
    y = np.asarray(y)
    if y.ndim != 1 or len(np.unique(y)) < 2:
        raise ValueError("Expected one-dimensional classification labels with >=2 classes")
    indices = np.arange(len(y))

    def divide(idx, size):
        return train_test_split(idx, test_size=size, stratify=y[idx], random_state=seed)

    try:
        non_test, test = divide(indices, 0.2)
        context, rerank = divide(non_test, 0.1)  # 8% of all rows
        train, early_stop = divide(context, 1 / 9)  # 8% of all rows
    except ValueError as exc:
        raise ValueError(
            "Cannot create four stratified subsets; provide enough examples per class. "
            "TCAP does not merge validation sets or fall back to unstratified splitting."
        ) from exc
    expected = set(np.unique(y))
    for name, subset in (("train", train), ("early_stop", early_stop),
                         ("rerank", rerank), ("test", test)):
        if set(np.unique(y[subset])) != expected:
            raise ValueError(f"The {name} split does not contain every class")
    return TaskSplit(train, early_stop, rerank, test)
