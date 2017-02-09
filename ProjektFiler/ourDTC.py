from collections import defaultdict

import pandas
import sklearn.model_selection

from specialNode import Node, split
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

def new_gini_test(subjects, feature_index):
    #children = split(subject, )

    class_frequency = defaultdict(int)

    for subject in subjects:
        class_frequency[subject.class_label] += 1

    n_classes = len(subjects)
    gini_index = 1
    for class_key in class_frequency.keys():
        gini_index -= (class_frequency[class_key] / n_classes) ** 2


def child_gini_index(child, subjects):

    


def start_hunts(data_features, data_class_labels):
    subjects = [Subject(row, label) for row, label in zip(data_features, data_class_labels)]

    model = Node()
    hunts(model, subjects, 0)

    return model


def hunts(parent_node, subjects, feature_index):
    # Base-Case: If there are no features left, then return.
    if feature_index == len(subjects[0].features):
        parent_node.label = most_common_class_label(subjects)
    elif group_has_same_label(subjects):
        parent_node.label = subjects[0].class_label
    else:
        """Constructs a test with the mean value of subjects
        current feature (feature_index), so we know where to split."""
        test = new_mean_value_test(subjects, feature_index)
        parent_node.split_test = test

        # Splits the node with the new feature test
        split_nodes = split(subjects, test)
        if len(split_nodes) <= 1:
            hunts(parent_node, subjects, feature_index + 1)
        else:
            for split_node in split_nodes:
                parent_node.child_nodes.append(split_node.node)
                hunts(split_node.node, split_node.subjects, feature_index + 1)


def most_common_class_label(subjects):
    result_set = defaultdict(int)
    for subject in subjects:
        result_set[subject.class_label[0]] += 1

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
X, X_test, y, y_test = sklearn.model_selection \
    .train_test_split(X_arr, y_arr, test_size=0.33, random_state=42)

X = [[float(n) for n in row] for row in X]
X_test = [[float(n) for n in row] for row in X_test]


def test_hunts():
    model = start_hunts(X, y)

    y_test_predict = predict(model, X_test)
    compare_results(y_test_predict, y_test)


def compare_results(prediction, correct_result):
    correct = 0
    for i in range(len(prediction)):
        if prediction[i] == correct_result[i]:
            correct += 1

    percentage = correct / len(prediction)
    print("Correct to " + str(percentage*100) + "%")


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
