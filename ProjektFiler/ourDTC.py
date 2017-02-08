from collections import defaultdict

import pandas
import sklearn.model_selection

from specialNode import Node
import numpy as np


class Subject:
    def __init__(self, features, class_label):
        self.class_label = class_label
        self.features = features


def new_mean_value_test(subjects, featureIndex):
    mean = 0
    for s in subjects:
        mean += int(s.features[featureIndex])

    mean /= len(subjects[0].features)

    def test(subject):
        return subject.features[featureIndex] < mean

    return test


def hunts(parent_node, subjects, feature_index):
    # Base-Case: If there are no features left, then return.
    if feature_index == len(subjects[0].features):
        print("Most common: " + most_common_class_label(subjects))
        parent_node.label = most_common_class_label(subjects)

    if group_has_same_label(subjects):
        parent_node.label = subjects[0].class_label
    else:
        """Constructs a test with the mean value of subjects
                current feature (feature_index), so we know where to split."""
        test = new_mean_value_test(subjects, feature_index)

        # Splits the node with the new feature test
        for split_node in parent_node.split(subjects, test):
            parent_node.child_nodes.append(split_node.node)
            hunts(split_node.node, split_node.subjects, feature_index + 1)


def most_common_class_label(subjects):
    result_set = defaultdict(int)
    for subject in subjects:
        result_set[subject.class_label] += 1

    return max(result_set, key=result_set.get)


def group_has_same_label(subjects):
    first_label = subjects[0]
    for subject in subjects:
        if subject.class_label != first_label:
            # if the subjects class label is not the same as the first label
            # then this group does not all have the same label
            return False

    # They all have the same class label
    return True

df = pandas.read_csv(r"C:\Users\August\Documents\ILS Projekt\ILS Projekt Dataset\csv_binary\binary\labor.csv",
                     header=None)

# print(df[:2])

arr = pandas.np.array(df)
X_arr = arr[1:, :-1]
y_arr = arr[1:, -1:]

X_arr.astype(np.double)

# Bosco Talkshow
# Training set, test set, train klass label, test klass label. We split
# into sets
X, X_test, y, y_test = sklearn.model_selection\
    .train_test_split(X_arr, y_arr, test_size=0.33, random_state=42)

X.astype(np.double)

X = [[float(n) for n in row] for row in X]


def test_hunts():
    subjects = [Subject(row, label) for row, label in zip(X, y)]

    node = Node()
    hunts(node, subjects, 0)
    print_node(node)


def print_node(node):
    if not node.label:
        print(node.test)
    else:
        print("class: " + node.label)
    for child in node.child_nodes:
        print_node(child)

test_hunts()
