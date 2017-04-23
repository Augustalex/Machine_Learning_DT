import numpy
import pandas
import time

from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ProjektFiler.OurDecisionTreeClassifier import OurDecisionTreeClassifier, OurRandomForrestClassifier, \
    unzip_features_and_labels, flatten_num_py_arrays
from Tree import print_tree_vertical
from excelifyer import Excelifyer


def get_labels(data):
    values = []
    for item in data:
        if not (values.__contains__(item)):
            values.append(item)
    return values


def accuracy_test(prediction, correct_result):
    correct = 0
    for i in range(len(prediction)):
        if prediction[i] == correct_result[i]:
            correct += 1

    percentage = correct / len(prediction)
    return percentage * 100


def our_recall_score(true_class_labels, prediction, feature):
    true_positives = 0
    false_negatives = 0
    for i in range(len(prediction)):
        if prediction[i] == feature:
            if prediction[i] == true_class_labels[i]:
                true_positives += 1
        elif prediction[i] != true_class_labels[i]:
            false_negatives += 1
    if (true_positives + false_negatives) == 0:
        return 0
    recall = true_positives / float(true_positives + false_negatives)
    return recall


def our_precision_score(true_class_lables, prediction, feature):
    true_positives = 0
    false_positives = 0
    for i in range(len(prediction)):
        if prediction[i] == feature:
            if prediction[i] == true_class_lables[i]:
                true_positives += 1
            else:
                false_positives += 1
    if (true_positives + false_positives) == 0:
        return 0
    precision = true_positives / float(true_positives + false_positives)
    return precision


def our_auc_score(precision, recall):
    if recall == 0:
        return 0
    return (2 * ((precision * recall) / (precision + recall)))


def compare(prediction1, labels, train_labels):
    precisions1 = []
    recalls1 = []
    auc1 = []
    train_labels = get_labels(train_labels)
    accuracy1 = accuracy_test(prediction1, labels)
    for label in train_labels:
        p = our_precision_score(labels, prediction1, label)
        precisions1.append(p)
        r = our_recall_score(labels, prediction1, label)
        recalls1.append(r)
        aucONE = our_auc_score(p, r)
        auc1.append(aucONE)
    precisions1_ = numpy.array(precisions1).mean()
    recalls1_ = numpy.array(recalls1).mean()
    auc1_ = numpy.array(auc1).mean()
    return accuracy1, precisions1_, recalls1_, auc1_


def start(data_set):
    accuracies1 = []
    precisions1 = []
    recalls1 = []
    aucs1 = []
    our_training_time = []
    our_testing_time = []

    odtc = OurDecisionTreeClassifier()

    data_set = pandas.np.array(data_set)
    features_, labels_ = unzip_features_and_labels(data_set)

    train_features, test_features, train_labels, test_labels = \
        train_test_split(
            features_, labels_,
            test_size=0.1,
            random_state=int(round(time.time()))
        )
    # un-numpy the arrays before predicting
    train_features, test_features, train_labels, test_labels = flatten_num_py_arrays(
        [train_features, test_features, train_labels, test_labels])
    # train and test our tree
    start_our_fit = time.time()
    odtc.fit(train_features, train_labels)
    end_our_fit = time.time()
    our_fit = end_our_fit - start_our_fit
    ops = time.time()
    prediction1 = odtc.predict(test_features)
    ope = time.time()
    our_predict = ope - ops

    a1, p1, r1, auc1 = compare(prediction1, test_labels, labels_)

    accuracies1.append(a1)
    precisions1.append(p1)
    recalls1.append(r1)
    aucs1.append(auc1)
    our_training_time.append(our_fit)
    our_testing_time.append(our_predict)

    print("\n----------------------------------------")
    print("\nFor our Decision Tree Classifier:")
    print("Our average accuracy:", numpy.array(accuracies1).mean())
    print("Our average precision:", numpy.array(precisions1).mean())
    print("Our average recall:", numpy.array(recalls1).mean())
    print("Our average AUC:", numpy.array(aucs1).mean())
    print("Our average training time:", numpy.array(our_training_time).mean())
    print("Our average testing time:", numpy.array(our_testing_time).mean())
    print("\n----------------------------------------\n\n")

    print_tree_vertical(odtc.model)


start(pandas.read_csv(r"..\ILS Projekt Dataset\csv_binary\binary\diabetes.csv", header=None))
