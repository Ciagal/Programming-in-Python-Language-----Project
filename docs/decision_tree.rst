Decision tree from scratch
==========================

This project implements a tiny decision tree classifier in pure Python.
The goal is not performance, but **clarity** – to understand how a tree
can be built from scratch.

Data format
-----------

The classifier works on a very simple data representation:

* each **row** is a sequence (e.g. list or tuple),
* **all columns except the last** are features,
* the **last column** is the label.

Example::

    ["Green", 3, "Apple"]

* feature 0 → color = ``"Green"``
* feature 1 → size  = ``3``
* label       → ``"Apple"``

Both numerical and categorical features are supported:

* numerical features are split with ``>= threshold``,
* categorical features are split with ``== value``.

Training the tree
-----------------

The main entry point is :class:`python_lab_project.core.SimpleDecisionTree`.

Typical usage looks like this::

    from python_lab_project.core import SimpleDecisionTree

    training_data = [
        ["Green", 3, "Apple"],
        ["Yellow", 3, "Apple"],
        ["Red", 1, "Grape"],
        ["Red", 1, "Grape"],
        ["Yellow", 3, "Lemon"],
    ]

    tree = SimpleDecisionTree(max_depth=3, min_samples_split=2)
    tree.fit(training_data)

After calling :meth:`SimpleDecisionTree.fit`, the internal structure of the
tree is stored in a hierarchy of :class:`python_lab_project.core.Node` objects.

Making predictions
------------------

There are two main prediction methods::

    from python_lab_project.core import SimpleDecisionTree

    row = ["Green", 3, "Apple"]  # label is not required for prediction

    # 1) full probability distribution
    proba = tree.predict_proba_one(row)
    # e.g. {"Apple": 0.8, "Lemon": 0.2}

    # 2) single most likely label
    label = tree.predict_one(row)
    # e.g. "Apple"

The probabilities are computed from the label counts stored in the leaf node
reached by the given example.

Inspecting the tree
-------------------

To understand how the model splits the data, you can print the tree::

    tree.print_tree()

Example output::

    [x[0] == Yellow]
      -> True:
        Leaf: {'Apple': '50%', 'Lemon': '50%'}
      -> False:
        [x[1] >= 2]
          -> True:
            Leaf: {'Apple': '100%'}
          -> False:
            Leaf: {'Grape': '100%'}

Reading this:

* if ``x[0] == "Yellow"`` → go to the left branch,
* otherwise check ``x[1] >= 2`` for the remaining rows,
* leaf nodes show the class distribution in percentages.

How splitting works
-------------------

The tree searches for the *best split* using **Gini impurity**.

For a set of labels with class proportions :math:`p_k`, Gini impurity is:

.. math::

    G = 1 - \sum_k p_k^2

A "pure" node (all samples of the same class) has ``G = 0``.
A more mixed node has higher Gini impurity.

The function :func:`python_lab_project.core.gini_impurity` computes this
value for a list of rows.

During training, the method :meth:`python_lab_project.core.SimpleDecisionTree._best_split`
tries all possible splits defined by:

* feature index,
* unique value in that feature,

and chooses the one that provides the largest impurity decrease (information
gain).

Demo entry point
----------------

The module also exposes a small demo via :func:`python_lab_project.core.main`.
You can run it directly::

    python -m python_lab_project.core

It will:

* train the tree on the tiny fruit dataset,
* print the built tree,
* show predictions for a few test rows.
