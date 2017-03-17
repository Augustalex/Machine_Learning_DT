import numpy
import pandas
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ProjektFiler.OurDecisionTreeClassifier import OurDecisionTreeClassifier, OurRandomForrestClassifier, \
    unzip_features_and_labels, undress_num_py_arrays


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
    recall = tp / float(tp + fn)
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


def our_auc_score(precision, recall):
    if recall == 0:
        return 0
    return (2 * ((precision * recall) / (precision + recall)))


def compare(prediction1, prediction2, labels, train_labels):
    precisions1 = []
    precisions2 = []
    recalls1 = []
    recalls2 = []
    train_labels = get_labels(train_labels)
    accuracy1 = accuracy_test(prediction1,labels)
    accuracy2 = accuracy_test(prediction2,labels)
    for label in train_labels:
        precisions1.append(our_precision_score(labels, prediction1, label))
        precisions2.append(our_precision_score(labels, prediction2, label))
        recalls1.append(our_recall_score(labels, prediction1, label))
        recalls2.append(our_recall_score(labels, prediction2, label))
    precisions1_ = numpy.array(precisions1).mean()
    precisions2_ = numpy.array(precisions2).mean()
    recalls1_ = numpy.array(recalls1).mean()
    recalls2_ = numpy.array(recalls2).mean()
    auc1_ = our_auc_score(precisions1_, recalls1_)
    auc2_ = our_auc_score(precisions2_, recalls2_)
    return (accuracy1, accuracy2, precisions1_, precisions2_, recalls1_, recalls2_, auc1_, auc2_)


def start(data_set, rf_flag=False, max_features=None):
    accuracies1 = []
    accuracies2 = []
    predictions1 = []
    predictions2 = []
    recalls1 = []
    recalls2 = []
    aucs1 = []
    aucs2 = []
    our_training_time = []
    our_testing_time = []
    their_training_time = []
    their_testing_time = []

    accuracies3 = []
    accuracies4 = []
    predictions3 = []
    predictions4 = []
    recalls3 = []
    recalls4 = []
    aucs3 = []
    aucs4 = []
    our_training_time1 = []
    our_testing_time1 = []
    their_training_time1 = []
    their_testing_time1 = []

    odtc = OurDecisionTreeClassifier()
    dtc = DecisionTreeClassifier()
    orf = OurRandomForrestClassifier(sample_size=0.3, n_estimators=11)
    rf = RandomForestClassifier(n_estimators=11)

    data_set = pandas.np.array(data_set)
    features_, labels_ = unzip_features_and_labels(data_set)

    for i in range(1):
        train_features, test_features, train_labels, test_labels = \
            train_test_split(
                features_, labels_,
                test_size=0.1,
                random_state=int(round(time.time()))
            )
        # un-numpy the arrays before predicting
        train_features, test_features, train_labels, test_labels = undress_num_py_arrays(
            [train_features, test_features, train_labels, test_labels])
        # fit our tree
        start_our_fit = time.time()
        odtc.fit(train_features, train_labels)
        end_our_fit = time.time()
        our_fit = end_our_fit - start_our_fit
        ops = time.time()
        prediction1 = odtc.predict(test_features)
        ope = time.time()
        our_predict = ope - ops
        # fit their tree
        start_their_fit = time.time()
        dtc.fit(train_features, train_labels)
        end_their_fit = time.time()
        their_fit = end_their_fit - start_their_fit
        tps = time.time()
        prediction2 = dtc.predict(test_features)
        tpe = time.time()
        their_predict = tpe - tps

        a1, a2, p1, p2, r1, r2, auc1, auc2 = compare(prediction1, prediction2, test_labels, labels_)
        accuracies1.append(a1)
        accuracies2.append(a2)
        predictions1.append(p1)
        predictions2.append(p2)
        recalls1.append(r1)
        recalls2.append(r2)
        aucs1.append(auc1)
        aucs2.append(auc2)
        our_training_time.append(our_fit)
        our_testing_time.append(our_predict)
        their_testing_time.append(their_predict)
        their_training_time.append(their_fit)

        if rf_flag:
            start_fitRF = time.time()
            orf.fit(train_features, train_labels)
            end_fitRF = time.time()
            our_fit1 = end_fitRF - start_fitRF
            start_predictRF = time.time()
            prediction3 = orf.predict(test_features)
            end_predictRF = time.time()
            our_predict1 = end_predictRF - start_predictRF
            start_their_fitRF = time.time()
            rf.fit(train_features, train_labels)
            end_their_fitRF = time.time()
            their_fit1 = end_their_fitRF - start_their_fitRF
            start_their_predictRF = time.time()
            prediction4 = rf.predict(test_features)
            end_their_predictRF = time.time()
            their_predict1 = end_their_predictRF - start_their_predictRF
            a3, a4, p3, p4, r3, r4, auc3, auc4 = compare(prediction3, prediction4, test_labels, labels_)
            accuracies3.append(a3)
            accuracies4.append(a4)
            predictions3.append(p3)
            predictions4.append(p4)
            recalls3.append(r3)
            recalls4.append(r4)
            aucs3.append(auc3)
            aucs4.append(auc4)
            our_training_time1.append(our_fit1)
            our_testing_time1.append(our_predict1)
            their_testing_time1.append(their_predict1)
            their_training_time1.append(their_fit1)

    print("\n----------------------------------------")
    print("\nFor our Decision Tree Classifier:")
    print("Our average accuracy:", numpy.array(accuracies1).mean())
    print("Our average prediction:", numpy.array(predictions1).mean())
    print("Our average recall:", numpy.array(recalls1).mean())
    print("Our average AUC:", numpy.array(aucs1).mean())
    print("Our average training time:", numpy.array(our_training_time).mean())
    print("Our average testing time:", numpy.array(our_testing_time).mean())
    print("\n----------------------------------------")
    print("\nFor their Decision Tree Classifier:")
    print("Their average accuracy:", numpy.array(accuracies2).mean())
    print("Their average prediction:", numpy.array(predictions2).mean())
    print("Their average recall:", numpy.array(recalls2).mean())
    print("Their average AUC:", numpy.array(aucs2).mean())
    print("Their average training time:", numpy.array(their_training_time).mean())
    print("Their average testing time:", numpy.array(their_testing_time).mean())

    if rf_flag:
        print("\n----------------------------------------")
        print("\nFor our Random Forest Classifier:")
        print("Our average accuracy:", numpy.array(accuracies3).mean())
        print("Our average prediction:", numpy.array(predictions3).mean())
        print("Our average recall:", numpy.array(recalls3).mean())
        print("Our average AUC:", numpy.array(aucs3).mean())
        print("Our average training time:", numpy.array(our_training_time1).mean())
        print("Our average testing time:", numpy.array(our_testing_time1).mean())
        print("\n----------------------------------------")
        print("\nFor their Random Forest Classifier:")
        print("Their average accuracy:", numpy.array(accuracies4).mean())
        print("Their average prediction:", numpy.array(predictions4).mean())
        print("Their average recall:", numpy.array(recalls4).mean())
        print("Their average AUC:", numpy.array(aucs4).mean())
        print("Their average training time:", numpy.array(their_training_time1).mean())
        print("Their average testing time:", numpy.array(their_testing_time1).mean())


start(pandas.read_csv(r"..\ILS Projekt Dataset\csv_binary\binary\balance-scale.csv", header=None), rf_flag=False)
