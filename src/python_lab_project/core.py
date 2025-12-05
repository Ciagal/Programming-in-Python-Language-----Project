from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Sequence, Tuple, Union


Row = Sequence[Any]  # e.g. ['Green', 3, 'Apple']


"""
core
====

This module contains a minimal decision tree classifier implemented from scratch.
The goal is to show, in a transparent way, how a tree can be built and used
without relying on high-level libraries such as scikit-learn.

Data representation
-------------------

Each row is represented as a sequence (e.g. list or tuple) where:

* all columns except the last one are features,
* the last column is the label/class.

Example:

    ["Green", 3, "Apple"]

Here:

* feature 0: color = "Green"
* feature 1: size  = 3
* label         = "Apple"

The tree supports both numerical and categorical features:

* numerical features are split using a threshold with condition ``>=``,
* categorical features are split using equality ``==``.

Main components
---------------

* :func:`gini_impurity` – computes the Gini impurity of a set of rows,
* :func:`split_rows` – splits rows into two groups using a feature and a pivot,
* :class:`Node` – a single node in the decision tree (internal node or leaf),
* :class:`SimpleDecisionTree` – a basic decision tree classifier with a
  scikit-learn-like interface (:meth:`fit`, :meth:`predict_one`,
  :meth:`predict_proba_one`).

The :func:`main` function at the bottom demonstrates a simple usage example
on a tiny "fruit" dataset.
"""


def is_number(x: Any) -> bool:
    """Return ``True`` if ``x`` is an integer or a float."""
    return isinstance(x, (int, float))


def count_labels(rows: List[Row]) -> Dict[Any, int]:
    """Count how many times each label (last column) appears.

    Args:
        rows: List of rows. The last element of each row is treated as the label.

    Returns:
        A dictionary mapping each label to the number of its occurrences
        in the provided rows.
    """
    counts: Dict[Any, int] = {}
    for r in rows:
        lbl = r[-1]
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def gini_impurity(rows: List[Row]) -> float:
    """Compute the Gini impurity for a given set of examples.

    Gini impurity is defined as:

    .. math::

        G = 1 - \\sum_k p_k^2

    where :math:`p_k` is the proportion of class :math:`k` in the set.

    Args:
        rows: List of rows. The last element of each row is treated as the label.

    Returns:
        The Gini impurity as a float in the range ``[0, 1]``.
    """
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
    """Split rows into two groups based on a condition in a given column.

    For numerical values the condition is::

        value >= pivot

    For non-numerical (categorical) values the condition is::

        value == pivot

    Rows that satisfy the condition go to the ``left`` group, the rest go to
    the ``right`` group.

    Args:
        rows: List of rows.
        col_index: Index of the feature column used to split the rows.
        pivot: Threshold or category used as the split value.

    Returns:
        A tuple ``(left, right)`` where:

        * ``left`` contains rows where the condition is ``True``,
        * ``right`` contains the remaining rows.
    """
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
    """A single node in the decision tree.

    A node can be either:

    * a **leaf** node:

      - ``feature_index = None``
      - ``threshold = None``
      - ``left = None``
      - ``right = None``
      - ``prediction`` is a dictionary mapping labels to their counts

    * or a **decision** (internal) node:

      - ``feature_index`` and ``threshold`` are set
      - ``left`` and ``right`` point to child nodes
      - ``prediction = None``

    Attributes:
        feature_index: Index of the feature used for splitting at this node.
        threshold: Threshold or category used for the split.
        left: Left child node (rows that satisfied the split condition).
        right: Right child node (rows that did not satisfy the split condition).
        prediction: For leaf nodes, dictionary mapping labels to counts.
            For internal nodes this is ``None``.
    """

    feature_index: Optional[int] = None
    threshold: Optional[Any] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    prediction: Optional[Dict[Any, int]] = None

    @property
    def is_leaf(self) -> bool:
        """Return ``True`` if this node is a leaf node."""
        return self.prediction is not None


class SimpleDecisionTree:
    """A simple decision tree classifier.

    This is a minimal implementation of a decision tree for educational
    purposes. It operates on small tabular datasets where each row is a
    sequence of values and the last element is the label.

    The tree uses Gini impurity to choose the best splits and supports
    both numerical and categorical features.

    Args:
        max_depth: Maximum depth of the tree. If ``None``, the tree is allowed
            to grow until it hits other stopping criteria (e.g. minimum number
            of samples in a node).
        min_samples_split: Minimum number of samples required to split a node.

    Attributes:
        max_depth: Configured maximum depth of the tree.
        min_samples_split: Configured minimum number of samples to split.
        _root: The root :class:`Node` of the trained tree, or ``None`` if
            the tree has not been fitted yet.
    """

    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._root: Optional[Node] = None

    # --- Public API ---

    def fit(self, rows: List[Row]) -> None:
        """Build the decision tree from the training data.

        Args:
            rows: Training data, a list of rows. Each row is a sequence where
                all elements except the last one are features and the last
                element is the label.
        """
        self._root = self._build(rows, depth=0)

    def predict_proba_one(self, row: Row) -> Dict[Any, float]:
        """Return the label probability distribution for a single example.

        The probabilities are computed from the label counts stored in the leaf
        node reached by the given example.

        Args:
            row: A single input row (features and label). The label in the last
                position is not used for prediction, but typical examples still
                include it for convenience.

        Returns:
            A dictionary mapping labels to their estimated probabilities.

        Raises:
            RuntimeError: If the tree has not been fitted yet.
        """
        if self._root is None:
            raise RuntimeError("Tree is not fitted yet.")
        leaf = self._traverse(row, self._root)
        counts = leaf.prediction or {}
        total = sum(counts.values()) or 1
        return {lbl: c / float(total) for lbl, c in counts.items()}

    def predict_one(self, row: Row) -> Any:
        """Return the most likely label for a single example.

        Args:
            row: A single input row.

        Returns:
            The label with the highest estimated probability.
        """
        proba = self.predict_proba_one(row)
        # take the key with the highest probability
        return max(proba.items(), key=lambda kv: kv[1])[0]

    def print_tree(self) -> None:
        """Print a human-readable representation of the tree structure."""
        if self._root is None:
            print("Tree is empty.")
        else:
            self._print_node(self._root, indent="")

    # --- Internal implementation ---

    def _build(self, rows: List[Row], depth: int) -> Node:
        """Recursively build the tree.

        This method creates either a leaf node (if a stopping condition is met)
        or an internal decision node with left and right children.

        Stopping conditions:

        * number of rows in the node is smaller than ``min_samples_split``,
        * current depth exceeds or reaches ``max_depth`` (if defined),
        * no split can produce a positive gain.

        Args:
            rows: Rows that reach the current node.
            depth: Current depth of the node in the tree (root has depth 0).

        Returns:
            A :class:`Node` representing the (sub)tree built from ``rows``.
        """
        # stopping conditions
        if len(rows) < self.min_samples_split:
            return Node(prediction=count_labels(rows))
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(prediction=count_labels(rows))

        best_gain, best_feature, best_threshold = self._best_split(rows)

        # if no useful split is found -> leaf
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
        """Find the best split: feature index, threshold and information gain.

        The method iterates over all features and their unique values, tries
        splitting the data using each value as a pivot, and keeps the split
        that yields the highest decrease in impurity (i.e. highest gain).

        Args:
            rows: Rows belonging to the current node.

        Returns:
            A tuple ``(best_gain, best_feature, best_threshold)`` where:

            * ``best_gain`` is the highest information gain found,
            * ``best_feature`` is the index of the feature used for the split,
            * ``best_threshold`` is the threshold/category used for the split.

            If no valid split is found, ``best_gain`` is ``0.0`` and
            ``best_feature`` as well as ``best_threshold`` are ``None``.
        """
        current_impurity = gini_impurity(rows)
        n_features = len(rows[0]) - 1  # last column is the label

        best_gain = 0.0
        best_feature: Optional[int] = None
        best_threshold: Optional[Any] = None

        for col in range(n_features):
            values = {r[col] for r in rows}  # unique values in the column
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
        """Traverse the tree down to a leaf for a given example.

        Args:
            row: A single input row.
            node: The current node from which we continue the traversal.

        Returns:
            The leaf :class:`Node` that the row ends up in.
        """
        if node.is_leaf:
            return node

        assert node.feature_index is not None  # for mypy/IDE
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
        """Recursively print a single node (and its children).

        Leaf nodes print the label distribution in percentages.
        Internal nodes print the splitting condition and then recursively
        print the ``True`` and ``False`` branches with increased indentation.

        Args:
            node: Node to print.
            indent: Prefix used to visually indent the output.
        """
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
    """Small demo showcasing how to train and inspect the tree.

    The example uses a tiny "fruit" dataset with two features:

    * color (categorical),
    * size (numerical),

    and a label describing the type of fruit.
    """
    # Same simple fruit dataset as in many decision tree tutorials
    training_data = [
        ["Green", 3, "Apple"],
        ["Yellow", 3, "Apple"],
        ["Red", 1, "Grape"],
        ["Red", 1, "Grape"],
        ["Yellow", 3, "Lemon"],
    ]

    tree = SimpleDecisionTree()
    tree.fit(training_data)

    print("== Built tree ==")
    tree.print_tree()

    test_rows = [
        ["Green", 3, "Apple"],
        ["Yellow", 4, "Apple"],
        ["Red", 2, "Grape"],
        ["Red", 1, "Grape"],
        ["Yellow", 3, "Lemon"],
    ]

    print("\n== Predictions ==")
    for r in test_rows:
        predicted = tree.predict_proba_one(r)
        print(f"Actual: {r[-1]:6} | Predicted: {predicted}")


if __name__ == "__main__":
    main()
