import pytest
from python_lab_project.core import (
    SimpleDecisionTree,
    Node,
    split_rows,
    gini_impurity,
)


# Test data

TRAIN_DATA = [
    ["Green", 3, "Apple"],
    ["Yellow", 3, "Apple"],
    ["Red", 1, "Grape"],
    ["Red", 1, "Grape"],
]


# Basic tree functionality

def test_fit_creates_tree():
    tree = SimpleDecisionTree()
    tree.fit(TRAIN_DATA)
    assert tree._root is not None
    assert isinstance(tree._root, Node)


def test_predict_one_simple():
    tree = SimpleDecisionTree()
    tree.fit(TRAIN_DATA)

    assert tree.predict_one(["Green", 3, "?"]) == "Apple"
    assert tree.predict_one(["Red", 1, "?"]) == "Grape"


def test_predict_proba_sums_to_one():
    tree = SimpleDecisionTree()
    tree.fit(TRAIN_DATA)

    proba = tree.predict_proba_one(["Green", 3, "?"])
    assert abs(sum(proba.values()) - 1.0) < 1e-9


def test_predict_proba_without_fit_raises():
    tree = SimpleDecisionTree()
    with pytest.raises(RuntimeError):
        tree.predict_proba_one(["Red", 1, "?"])


# Node behavior

def test_leaf_detection():
    leaf = Node(prediction={"Apple": 3})
    non_leaf = Node(feature_index=0, threshold="Green")

    assert leaf.is_leaf
    assert not non_leaf.is_leaf


# Utility functions

def test_split_rows_numeric():
    rows = [
        ["Green", 3, "Apple"],
        ["Green", 1, "Apple"],
    ]
    left, right = split_rows(rows, 1, 2)

    assert len(left) == 1
    assert len(right) == 1
    assert left[0][1] == 3
    assert right[0][1] == 1


def test_gini_impurity_basic():
    uniform = [
        ["Green", 3, "Apple"],
        ["Yellow", 3, "Apple"],
    ]
    assert gini_impurity(uniform) == 0.0

    mixed = [
        ["Green", 3, "Apple"],
        ["Red", 1, "Grape"],
    ]
    g = gini_impurity(mixed)
    assert 0 < g <= 1


def test_gini_impurity_empty():
    assert gini_impurity([]) == 0.0


# Tree growth parameters

def test_min_samples_split_creates_leaf():
    small = [
        ["Green", 3, "Apple"],
        ["Yellow", 3, "Apple"],
    ]
    tree = SimpleDecisionTree(min_samples_split=10)
    tree.fit(small)

    assert tree._root.is_leaf


def test_max_depth_limit():
    tree = SimpleDecisionTree(max_depth=0)
    tree.fit(TRAIN_DATA)
    assert tree._root.is_leaf


# Tree traversal


def test_traversal_returns_leaf():
    tree = SimpleDecisionTree()
    tree.fit(TRAIN_DATA)

    leaf = tree._traverse(["Green", 3, "?"], tree._root)
    assert leaf.is_leaf
    assert isinstance(leaf.prediction, dict)


def test_traverse_numeric_split():
    rows = [
        ["Green", 3, "Apple"],
        ["Red", 1, "Grape"],
    ]
    tree = SimpleDecisionTree()
    tree.fit(rows)

    leaf = tree._traverse(["Green", 5, "?"], tree._root)
    assert leaf.is_leaf


def test_traverse_categorical_split():
    rows = [
        ["Green", 3, "Apple"],
        ["Yellow", 3, "Apple"],
    ]
    tree = SimpleDecisionTree()
    tree.fit(rows)

    leaf = tree._traverse(["Green", 3, "?"], tree._root)
    assert leaf.is_leaf


def test_traverse_false_branch():
    rows = [
        ["Green", 3, "Apple"],
        ["Yellow", 3, "Lemon"],
    ]
    tree = SimpleDecisionTree()
    tree.fit(rows)

    leaf = tree._traverse(["Yellow", 99, "?"], tree._root)
    assert leaf.is_leaf



# Tree printing

def test_print_tree_empty(capsys):
    tree = SimpleDecisionTree()
    tree.print_tree()
    out = capsys.readouterr().out
    assert "Tree is empty." in out


def test_print_tree_non_empty(capsys):
    tree = SimpleDecisionTree()
    tree.fit([["Green", 3, "Apple"]])
    tree.print_tree()
    out = capsys.readouterr().out
    assert "Leaf:" in out


def test_print_node_leaf(capsys):
    leaf = Node(prediction={"Apple": 3})
    tree = SimpleDecisionTree()
    tree._print_node(leaf, "")
    out = capsys.readouterr().out
    assert "Leaf:" in out


def test_print_node_decision(capsys):
    left = Node(prediction={"Apple": 2})
    right = Node(prediction={"Grape": 1})
    node = Node(feature_index=0, threshold="Green", left=left, right=right)

    tree = SimpleDecisionTree()
    tree._print_node(node, "")
    out = capsys.readouterr().out

    assert "x[0]" in out
    assert "Green" in out
    assert "True:" in out
    assert "False:" in out


def test_print_node_full_tree(capsys):
    rows = [
        ["Green", 3, "Apple"],
        ["Yellow", 3, "Lemon"],
    ]
    tree = SimpleDecisionTree()
    tree.fit(rows)
    tree.print_tree()
    out = capsys.readouterr().out

    assert "Leaf:" in out
    assert "True:" in out
    assert "False:" in out



# Comparison with scikit-learn

def test_sklearn_comparison():
    """Compare our implementation with sklearn for consistency."""
    try:
        from sklearn.tree import DecisionTreeClassifier
        import numpy as np
    except ImportError:
        pytest.skip("sklearn not installed")

    X = [["Green", 3], ["Yellow", 3], ["Red", 1], ["Red", 1]]
    y = ["Apple", "Apple", "Grape", "Grape"]

    color_map = {"Green": 0, "Yellow": 1, "Red": 2}
    X_enc = np.array([[color_map[c], v] for c, v in X])

    clf = DecisionTreeClassifier()
    clf.fit(X_enc, y)

    tree = SimpleDecisionTree()
    tree.fit([row + [label] for row, label in zip(X, y)])

    assert tree.predict_one(["Green", 3, "?"]) == clf.predict([[0, 3]])[0]


