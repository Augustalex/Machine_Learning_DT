import time

from criterion import Gini, Entropy
from node_and_record import Subject, group_has_same_label, most_common_class_label, PredictionNode


def start_hunts(data_features, data_class_labels, max_features):
    subjects = [Subject(row, label) for row, label in zip(data_features, data_class_labels)]

    model = PredictionNode()

    max_depth = None
    min_samples_leaf = 1

    hunts(model, subjects, 0, min_samples_leaf, max_depth)

    #print("\nNumber of records: " + str(len(subjects)) + "\tNumber of features: " + str(len(subjects[0].class_features)))
    #print("Max depth: " + str(max_depth) + "\tMin samples leaf: " + str(min_samples_leaf) + "\n")

    return model


def hunts(parent_node, subjects, depth, min_samples_leaf, max_depth, criterion, max_features):
    # Base-Case: If there are no features left, then return.

    """
        Note that this is a recursive algorithm.
    - Criterion ar Gini eller Entropy
    - Check labb spec for other parameters we might have missed
    - Make a Class "DecisionTreeClassifier"
    - Class should have class variable "node" or "tree"
    - The class should have two methods at least "fit" and "predict"
    - Predict should use the "tree" in the class created by "fit"
    - Check out "predictProb", mentioned in the lab spec. What does it do..? Should not be
    too different from the regular "prediction" method.

    - Random forrest should have several instances of this DecisionTreeClassifier class we
    are going to make

    :param parent_node: Current Node (Part of recursion)
    :param subjects: Subjects related to current node (Part of recursion)
    :param depth: Current recursive depth
    :param min_samples_leaf: Minimum number of subjects before the tree should be closed.
    :param max_depth: The maximum recursive depth before the tree should be closed.
    :param criterion: The algorithm for use in rating splits. Choices are entropy and gini.
    :param max_features: The percentage of features rated by the criterion at each split.
    :return: Nothing. The end result is stored in the original Parent Node, all through the recursion.
    """

    # If a max_depth is defined and is smaller or equal to the current depth, end the recursion.
    # If the number of subjects is less or equal to the minimum number of subjects, end the recursion.
    if max_depth is not None and depth >= max_depth or len(subjects) <= min_samples_leaf:
        create_leaf_node(parent_node, subjects)
    # If all subjects relate to the same class label, end the recursion.
    elif group_has_same_label(subjects):
        create_leaf_node(parent_node, subjects)
    else:
        """Constructs a test with the mean value of subjects
        current feature (feature_index), so we know where to split."""

        # Uses gini to generate the best split out of all features.
        criterion_executor = Gini(max_features) if criterion == 'gini' else Entropy(max_features)
        best_gini_split = criterion_executor.generate_best_split_of_all_features(subjects)

        # Stores the test of the "best split" in the parent node, for future prediction with that node.
        parent_node.split_test = best_gini_split.test

        """
            If the number of nodes in the new split is one or less,
            then there are no more useful features to be split.
            The most common class label amongst the subjects is set
            at the parent_node.
        """
        if len(best_gini_split.split) <= 1:
            create_leaf_node(parent_node, subjects)
        else:
            # For all nodes in the chosen split (best_gini_split.split) run it through the hunts algorithm.
            for split_node in best_gini_split.split:
                if len(split_node.subjects) == 0:
                    pass
                else:
                    parent_node.child_nodes.append(split_node.node)
                    hunts(split_node.node, split_node.subjects, depth + 1, min_samples_leaf, max_depth, criterion, max_features)


def create_leaf_node(node, subjects):
    node.label = most_common_class_label(subjects)
    node.subjects = subjects
