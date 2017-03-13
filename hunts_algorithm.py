import time

from criterion import generate_best_split_of_all_features
from entropy import Entropy
from gini_index import Gini
from prediction_node import PredictionNode
from record_subject import Subject, group_has_same_label, most_common_class_label


def start_hunts(data_features, data_class_labels):
    subjects = [Subject(row, label) for row, label in zip(data_features, data_class_labels)]

    model = PredictionNode()

    start = time.time()

    max_depth = None
    min_samples_leaf = 1

    hunts(model, subjects, 0, min_samples_leaf, max_depth)
    end = time.time()

    print("\nTime elapsed: " + str(end - start) + "s")
    print("\tNumber of records: " + str(len(subjects)) + "\tNumber of features: " + str(len(subjects[0].features)))
    print("\tMax depth: " + str(max_depth) + "\tMin samples leaf: " + str(min_samples_leaf) + "\n")

    return model


def hunts(parent_node, subjects, depth, min_samples_leaf=10, max_depth=100):
    # Base-Case: If there are no features left, then return.

    """
        Note that this is a recursive algorithm.
    :param parent_node: Current Node (Part of recursion)
    :param subjects: Subjects related to current node (Part of recursion)
    :param depth: Current recursive depth
    :param min_samples_leaf: Minimum number of subjects before the tree should be closed.
    :param max_depth: The maximum recursive depth before the tree should be closed.
    :return: Nothing. The end result is stored in the original Parent Node, all through the recursion.
    """

    # If a max_depth is defined and is smaller or equal to the current depth, end the recursion.
    # If the number of subjects is less or equal to the minimum number of subjects, end the recursion.
    if max_depth is not None and depth >= max_depth or len(subjects) <= min_samples_leaf:
        parent_node.label = most_common_class_label(subjects)
    # If all subjects relate to the same class label, end the recursion.
    elif group_has_same_label(subjects):
        parent_node.label = subjects[0].class_label
    else:
        """Constructs a test with the mean value of subjects
        current feature (feature_index), so we know where to split."""

        # Uses gini to generate the best split out of all features.
        best_gini_split = generate_best_split_of_all_features(subjects, Entropy)

        # Stores the test of the "best split" in the parent node, for future prediction with that node.
        parent_node.split_test = best_gini_split.test

        """
            If the number of nodes in the new split is one or less,
            then there are no more useful features to be split.
            The most common class label amongst the subjects is set
            at the parent_node.
        """
        if len(best_gini_split.split) <= 1:
            parent_node.label = most_common_class_label(subjects)
        else:
            # For all nodes in the chosen split (best_gini_split.split) run it through the hunts algorithm.
            for split_node in best_gini_split.split:
                parent_node.child_nodes.append(split_node.node)
                hunts(split_node.node, split_node.subjects, depth + 1)
