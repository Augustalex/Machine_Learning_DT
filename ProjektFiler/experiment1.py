import pandas
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ProjektFiler.OurDecisionTreeClassifier import OurDecisionTreeClassifier, OurRandomForrestClassifier, \
    unzip_features_and_labels, undress_num_py_arrays
from prediction_node import compare_results

def get_class_feaures(data):
    classValues = []
    for item in data:
        if not (classValues.__contains__(item)):
            classValues.append(item)
    return classValues

def accuracy_test(prediction1, prediction2, true_class_labels):
    """returns the accuracy of both predictions vs the actual class_labels of the test data"""
    print("Our accuracy:")
    compare_results(prediction1, true_class_labels)
    print("Their accuracy:")
    compare_results(prediction2, true_class_labels)

def our_recall_score(true_class_labels, prediction, feature):
    tp = 0
    fn = 0
    for i in range(len(prediction)):
        if prediction[i] == feature:
            if prediction[i] == true_class_labels[i]:
                tp += 1
            elif prediction[i] != true_class_labels[i]:
                fn += 1
    if (tp + fn) == 0:
        return 0
    recall = tp / float(tp+fn)
    return recall


def our_precision_score(true_class_lables, prediction, feature):
    tp = 0
    fp = 0
    for i in range(len(prediction)):
        if prediction[i] == feature:
            if prediction[i] == true_class_lables[i]:
                tp += 1
            else:
                fp += 1
    if (tp + fp) == 0:
        return 0

    precision = tp / float(tp + fp)
    return precision


def our_f1_score(true_class_labels, prediction):
    print("august must solve this")
    pass


def our_auc_score(precision, recall):
    if recall == 0:
        return 0
    return (2*((precision*recall)/(precision + recall)))

def compare(prediction1, prediction2, labels, train_labels):
    precisions1 = []
    precisions2 = []
    recalls1 = []
    recalls2 = []
    features = get_class_feaures(train_labels)
    accuracy_test(prediction1, prediction2, labels)
    for feature in features:
        precisions1.append(our_precision_score(labels, prediction1, feature))
        precisions2.append(our_precision_score(labels, prediction2, feature))
        recalls1.append(our_recall_score(labels,prediction1, feature))
        recalls2.append(our_recall_score(labels,prediction2,feature))
    print("\nOur precision:", precisions1)
    print("\nTheir precision:", precisions2)
    print("\nOur recall:", recalls1)
    print("\nTheir recall:", recalls2)
    for p in precisions1:
        for r in recalls1:
            print("\nOur auc score:" ,our_auc_score(p, r))

    for p in precisions2:
        for r in recalls2:
            print("\nTheir auc score:" ,our_auc_score(p, r))


def start(data_set):
    odtc = OurDecisionTreeClassifier()
    dtc = DecisionTreeClassifier()
    #orf = OurRandomForestClassifier(sample_size=0.3, n_estimators=11)
    #rf = RandomForrestClassifier(n_estimators=11)

    data_set = pandas.np.array(data_set)
    features_, labels_ = unzip_features_and_labels(data_set)
    train_features, test_features, train_labels, test_labels = \
        train_test_split(
            features_, labels_,
            test_size=0.33,
            random_state=int(round(time.time()))
        )

    #un-numpy the arrays before predicting
    train_features, test_features, train_labels, test_labels = undress_num_py_arrays(
        [train_features, test_features, train_labels, test_labels])
    #fit our tree
    odtc.fit(train_features, train_labels)
    p1 = odtc.predict(test_features)
    #fit their tree
    dtc.fit(train_features, train_labels)
    p2 = dtc.predict(test_features)

    compare(p1, p2, test_labels, labels_)

    #orf.fit(train_features,train_labels)
    #rf.fit(train_features,train_labels)
    #compare(orf, rf, test_features, test_labels)

start(pandas.read_csv(r"..\ILS Projekt Dataset\csv_binary\binary\hepatitis.csv", header=None))