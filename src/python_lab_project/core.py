from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Sequence, Tuple, Union


Row = Sequence[Any]  # np. ['Green', 3, 'Apple']


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float))


def count_labels(rows: List[Row]) -> Dict[Any, int]:
    """Policz ile razy pojawia się każda etykieta (ostatnia kolumna)."""
    counts: Dict[Any, int] = {}
    for r in rows:
        lbl = r[-1]
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def gini_impurity(rows: List[Row]) -> float:
    """Gini impurity dla danego zbioru przykładów."""
    total = len(rows)
    if total == 0:
        return 0.0
    counts = count_labels(rows)
    impurity = 1.0
    for c in counts.values():
        p = c / float(total)
        impurity -= p * p
    return impurity


def split_rows(
    rows: List[Row],
    col_index: int,
    pivot: Any,
) -> Tuple[List[Row], List[Row]]:
    """Podziel wiersze na dwie grupy według warunku w kolumnie."""
    left: List[Row] = []
    right: List[Row] = []
    for r in rows:
        val = r[col_index]
        if is_number(val):
            condition = val >= pivot
        else:
            condition = val == pivot

        if condition:
            left.append(r)
        else:
            right.append(r)
    return left, right


@dataclass
class Node:
    """Węzeł drzewa.

    Jeśli to liść:
      - feature_index = None
      - threshold = None
      - left/right = None
      - prediction != None (słownik etykieta -> liczba wystąpień)

    Jeśli to węzeł decyzyjny:
      - feature_index, threshold ustawione
      - left/right to kolejne węzły
      - prediction = None
    """
    feature_index: Optional[int] = None
    threshold: Optional[Any] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    prediction: Optional[Dict[Any, int]] = None

    @property
    def is_leaf(self) -> bool:
        return self.prediction is not None


class SimpleDecisionTree:
    """Proste drzewo decyzyjne do klasyfikacji."""

    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._root: Optional[Node] = None

    # --- API zewnętrzne ---

    def fit(self, rows: List[Row]) -> None:
        """Zbuduj drzewo na podstawie danych treningowych."""
        self._root = self._build(rows, depth=0)

    def predict_proba_one(self, row: Row) -> Dict[Any, float]:
        """Zwróć rozkład prawdopodobieństwa etykiet dla pojedynczego przykładu."""
        if self._root is None:
            raise RuntimeError("Tree is not fitted yet.")
        leaf = self._traverse(row, self._root)
        counts = leaf.prediction or {}
        total = sum(counts.values()) or 1
        return {lbl: c / float(total) for lbl, c in counts.items()}

    def predict_one(self, row: Row) -> Any:
        """Zwróć jedną etykietę – tę o największej liczbie w liściu."""
        proba = self.predict_proba_one(row)
        # max po prawdopodobieństwie
        return max(proba.items(), key=lambda kv: kv[1])[0]

    def print_tree(self) -> None:
        if self._root is None:
            print("Tree is empty.")
        else:
            self._print_node(self._root, indent="")

    # --- Implementacja wewnętrzna ---

    def _build(self, rows: List[Row], depth: int) -> Node:
        """Rekurencyjna budowa drzewa."""
        # warunki stopu
        if len(rows) < self.min_samples_split:
            return Node(prediction=count_labels(rows))
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(prediction=count_labels(rows))

        best_gain, best_feature, best_threshold = self._best_split(rows)

        # jeśli nie ma sensownego podziału → liść
        if best_gain <= 0 or best_feature is None:
            return Node(prediction=count_labels(rows))

        left_rows, right_rows = split_rows(rows, best_feature, best_threshold)

        left_child = self._build(left_rows, depth + 1)
        right_child = self._build(right_rows, depth + 1)

        return Node(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
        )

    def _best_split(
        self, rows: List[Row]
    ) -> Tuple[float, Optional[int], Optional[Any]]:
        """Znajdź najlepszy podział: kolumna + próg + gain."""
        current_impurity = gini_impurity(rows)
        n_features = len(rows[0]) - 1  # ostatnia kolumna = etykieta

        best_gain = 0.0
        best_feature: Optional[int] = None
        best_threshold: Optional[Any] = None

        for col in range(n_features):
            values = {r[col] for r in rows}  # unikalne wartości w kolumnie
            for v in values:
                left, right = split_rows(rows, col, v)
                if not left or not right:
                    continue

                p_left = len(left) / float(len(rows))
                gain = current_impurity - (
                    p_left * gini_impurity(left)
                    + (1 - p_left) * gini_impurity(right)
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = col
                    best_threshold = v

        return best_gain, best_feature, best_threshold

    def _traverse(self, row: Row, node: Node) -> Node:
        """Zejdź po drzewie do liścia dla danego przykładu."""
        if node.is_leaf:
            return node

        assert node.feature_index is not None  # dla mypy/IDE
        val = row[node.feature_index]

        if is_number(val):
            condition = val >= node.threshold  # type: ignore[arg-type]
        else:
            condition = val == node.threshold

        if condition:
            return self._traverse(row, node.left)  # type: ignore[arg-type]
        else:
            return self._traverse(row, node.right)  # type: ignore[arg-type]

    def _print_node(self, node: Node, indent: str) -> None:
        """Rekurencyjny wypis drzewa."""
        if node.is_leaf:
            total = sum(node.prediction.values()) if node.prediction else 0
            pretty = {
                lbl: f"{int(c / total * 100)}%"
                for lbl, c in (node.prediction or {}).items()
            }
            print(indent + "Leaf:", pretty)
            return

        print(
            indent
            + f"[x[{node.feature_index}] {'>=' if is_number(node.threshold) else '=='} {node.threshold}]"
        )
        print(indent + "  -> True:")
        self._print_node(node.left, indent + "    ")
        print(indent + "  -> False:")
        self._print_node(node.right, indent + "    ")


def main():
    # Ten sam prosty zbiór z owocami
    training_data = [
        ["Green", 3, "Apple"],
        ["Yellow", 3, "Apple"],
        ["Red", 1, "Grape"],
        ["Red", 1, "Grape"],
        ["Yellow", 3, "Lemon"],
    ]

    tree = SimpleDecisionTree()
    tree.fit(training_data)

    print("== Zbudowane drzewo ==")
    tree.print_tree()

    test_rows = [
        ["Green", 3, "Apple"],
        ["Yellow", 4, "Apple"],
        ["Red", 2, "Grape"],
        ["Red", 1, "Grape"],
        ["Yellow", 3, "Lemon"],
    ]

    print("\n== Predykcje ==")
    for r in test_rows:
        predicted = tree.predict_proba_one(r)
        print(f"Actual: {r[-1]:6} | Predicted: {predicted}")

if __name__ == "__main__":
    main()