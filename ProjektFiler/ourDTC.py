from collections import defaultdict

import pandas
import sklearn.model_selection
import time
from sklearn.tree import DecisionTreeClassifier

from ProjektFiler.gini_index import generate_best_split, generate_class_frequency_map, \
    generate_best_split_of_all_features
from specialNode import Node, split, Subject
import numpy as np


def new_mean_value_test(subjects, featureIndex):
    mean = 0
    for s in subjects:
        mean += int(s.features[featureIndex])

    mean /= len(subjects[0].features)

    def test(subject):
        return subject.features[featureIndex] < mean

    return test


def start_hunts(data_features, data_class_labels):
    subjects = [Subject(row, label) for row, label in zip(data_features, data_class_labels)]

    model = Node()

    start = time.time()

    max_depth = None
    min_samples_leaf = 1

    _hunts(model, subjects, 0, min_samples_leaf, max_depth)
    end = time.time()

    print("\nTime elapsed: " + str(end - start) + "s")
    print("\tNumber of records: " + str(len(subjects)) + "\tNumber of features: " + str(len(subjects[0].features)))
    print("\tMax depth: " + str(max_depth) + "\tMin samples leaf: " + str(min_samples_leaf) + "\n")

    return model



def _hunts(parent_node, subjects, depth, min_samples_leaf=10, max_depth=100):
    # print("\n\n\t\tDEPTH " + str(depth) + "\n\n")
    # Base-Case: If there are no features left, then return.
    if max_depth is not None and depth >= max_depth or len(subjects) <= min_samples_leaf:
        parent_node.label = most_common_class_label(subjects)
    elif group_has_same_label(subjects):
        parent_node.label = subjects[0].class_label
    else:
        """Constructs a test with the mean value of subjects
        current feature (feature_index), so we know where to split."""

        # Splits the node with the new feature test
        best_gini_split = generate_best_split_of_all_features(subjects)
        parent_node.split_test = best_gini_split.test

        split_nodes = best_gini_split.split
        # split_nodes = split(subjects, test)
        if len(split_nodes) <= 1:
            _hunts(parent_node, subjects, depth + 1)
        else:
            for split_node in split_nodes:
                parent_node.child_nodes.append(split_node.node)
                _hunts(split_node.node, split_node.subjects, depth + 1)


def __hunts(parent_node, subjects, feature_index, min_samples_leaf=1,max_depth=100):
    # Base-Case: If there are no features left, then return.
    if feature_index == len(subjects[0].features) or feature_index >= max_depth:
        parent_node.label = most_common_class_label(subjects)
    elif group_has_same_label(subjects):
        parent_node.label = subjects[0].class_label
    else:
        """Constructs a test with the mean value of subjects
        current feature (feature_index), so we know where to split."""

        # Splits the node with the new feature test
        best_gini_split = generate_best_split(subjects, feature_index)
        parent_node.split_test = best_gini_split.test

        split_nodes = best_gini_split.split
        # split_nodes = split(subjects, test)
        if len(split_nodes) <= 1:
            __hunts(parent_node, subjects, feature_index + 1)
        else:
            for split_node in split_nodes:
                parent_node.child_nodes.append(split_node.node)
                __hunts(split_node.node, split_node.subjects, feature_index + 1)


def most_common_class_label(subjects):
    result_set = defaultdict(int)
    for subject in subjects:
        result_set[subject.class_label[0]] += 1

    return max(result_set, key=result_set.get)


def group_has_same_label(subjects):
    first_label = subjects[0].class_label
    for subject in subjects:
        if subject.class_label != first_label:
            # if the subjects class label is not the same as the first label
            # then this group does not all have the same label
            return False

    # They all have the same class label
    return True


df = pandas.read_csv(r"C:\Users\August\Documents\ILS Projekt\ILS Projekt Dataset\csv_binary\binary\diabetes.csv",
                     header=None)

# print(df[:2])

arr = pandas.np.array(df)
X_arr = arr[1:, :-1]
y_arr = arr[1:, -1:]

X_arr.astype(np.double)

# Bosco Talkshow
# Training set, test set, train klass label, test klass label. We split
# into sets
print(int(round(time.time())))
X, X_test, y, y_test = sklearn.model_selection \
    .train_test_split(X_arr, y_arr, test_size=0.33, random_state=int(round(time.time())))

X = [[float(n) for n in row] for row in X]
X_test = [[float(n) for n in row] for row in X_test]


def test_hunts():
    their_model = DecisionTreeClassifier()
    their_model.fit(X, y)

    model = start_hunts(X, y)

    y_test_predict = predict(model, X_test)
    compare_results(y_test_predict, y_test)

    y_their_predict = their_model.predict(X_test)
    compare_results(y_their_predict, y_test)


def compare_results(prediction, correct_result):
    correct = 0
    for i in range(len(prediction)):
        if prediction[i] == correct_result[i]:
            correct += 1

    percentage = correct / len(prediction)
    print("Correct to " + str(percentage * 100) + "%")


def predict(node, test_subjects):
    subjects = [Subject(row, None) for row in test_subjects]
    return [get_class_for_subject(node, subject) for subject in subjects]


def get_class_for_subject(node, subject):
    if not node.label:
        return get_class_for_subject(
            node.get_child_from_test(subject), subject
        )
    else:
        return node.label


test_hunts()
