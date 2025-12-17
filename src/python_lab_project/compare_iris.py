from __future__ import annotations

from statistics import mean
from typing import Callable, List

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from python_lab_project.core import SimpleDecisionTree


def to_rows(X: np.ndarray, y: np.ndarray) -> List[list]:
    # format jak w Twojej implementacji: [f1, f2, ..., label]
    return [list(map(float, X[i])) + [int(y[i])] for i in range(len(y))]


def predict_custom(tree: SimpleDecisionTree, X: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(len(X)):
        row = list(map(float, X[i])) + [None]  # etykieta nieużywana w predict
        preds.append(int(tree.predict_one(row)))
    return np.array(preds, dtype=int)


def cv_accuracy_custom(
    X: np.ndarray,
    y: np.ndarray,
    max_depth=None,
    min_samples_split: int = 2,
    n_splits: int = 5,
    random_state: int = 0,
) -> float:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        train_rows = to_rows(X[train_idx], y[train_idx])
        tree = SimpleDecisionTree(max_depth=max_depth, min_samples_split=min_samples_split)
        tree.fit(train_rows)

        y_pred = predict_custom(tree, X[test_idx])
        scores.append(accuracy_score(y[test_idx], y_pred))
    return float(mean(scores))


def cv_accuracy_sklearn(
    X: np.ndarray,
    y: np.ndarray,
    max_depth=None,
    min_samples_split: int = 2,
    n_splits: int = 5,
    random_state: int = 0,
) -> float:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        clf = DecisionTreeClassifier(
            criterion="gini",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        scores.append(accuracy_score(y[test_idx], y_pred))
    return float(mean(scores))


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    # ustawienia porównania (możesz zmieniać max_depth żeby zobaczyć wpływ overfitu)
    max_depth = None
    min_samples_split = 2

    acc_custom = cv_accuracy_custom(X, y, max_depth=max_depth, min_samples_split=min_samples_split)
    acc_sklearn = cv_accuracy_sklearn(X, y, max_depth=max_depth, min_samples_split=min_samples_split)

    print("Dataset: Iris (150 próbek, 4 cechy, 3 klasy)")
    print(f"Ustawienia: max_depth={max_depth}, min_samples_split={min_samples_split}, CV=5-fold\n")
    print(f"Twoje drzewo:      mean accuracy = {acc_custom:.3f}")
    print(f"sklearn DecisionTree: mean accuracy = {acc_sklearn:.3f}")


if __name__ == "__main__":
    main()
