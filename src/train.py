# src/train.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from .config import RANDOM_STATE, TEST_SIZE, N_SPLITS
from .data_loader import load_wdbc


@dataclass(frozen=True)
class DataSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def make_split() -> DataSplit:
    """
    Load the raw WDBC data, then create a stratified train/test split.
    """
    X, y = load_wdbc()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Convert to numpy for sklearn consistency (optional; pandas works too)
    return DataSplit(
        X_train=X_train.to_numpy(),
        X_test=X_test.to_numpy(),
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy(),
    )


def make_cv() -> StratifiedKFold:
    """
    Cross-validation strategy to be used later for GridSearchCV.
    """
    return StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def build_pipelines() -> Dict[str, Pipeline]:
    """
    Build model pipelines with StandardScaler.
    """
    pipelines: Dict[str, Pipeline] = {
        "logreg_l1": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                )),
            ]
        ),
        "svm_rbf": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVC(
                    kernel="rbf",
                    probability=True,  # for ROC/AUC later
                    random_state=RANDOM_STATE,
                )),
            ]
        ),
        "mlp": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                )),
            ]
        ),
    }
    return pipelines


def main() -> None:
    split = make_split()
    cv = make_cv()
    pipelines = build_pipelines()

    # print values
    print("Train shape:", split.X_train.shape, "Test shape:", split.X_test.shape)
    print("Train class counts:", dict(zip(*np.unique(split.y_train, return_counts=True))))
    print("Test class counts:", dict(zip(*np.unique(split.y_test, return_counts=True))))
    print("Pipelines:", list(pipelines.keys()))
    print("CV folds:", cv.get_n_splits())

    # training goes here


if __name__ == "__main__":
    main()