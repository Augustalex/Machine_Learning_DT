import numpy
import pandas
import time

import xlsxwriter
from scipy.stats import wilcoxon
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

def compare(prediction1, prediction2, labels, train_labels):
    precisions1 = []
    precisions2 = []
    recalls1 = []
    recalls2 = []
    auc1 = []
    auc2 = []
    train_labels = get_labels(train_labels)
    accuracy1 = accuracy_test(prediction1,labels)
    accuracy2 = accuracy_test(prediction2,labels)
    for label in train_labels:
        p = our_precision_score(labels, prediction1, label)
        precisions1.append(p)
        p2 = our_precision_score(labels, prediction2,label)
        precisions2.append(p2)
        r = our_recall_score(labels, prediction1, label)
        recalls1.append(r)
        r2 = our_recall_score(labels, prediction2, label)
        recalls2.append(r2)
        aucONE = our_auc_score(p, r)
        auc1.append(aucONE)
        aucTWO = our_auc_score(p2,r2)
        auc2.append(aucTWO)
    precisions1_ = numpy.array(precisions1).mean()
    precisions2_ = numpy.array(precisions2).mean()
    recalls1_ = numpy.array(recalls1).mean()
    recalls2_ = numpy.array(recalls2).mean()
    auc1_ = numpy.array(auc1).mean()
    auc2_ = numpy.array(auc2).mean()
    return accuracy1, accuracy2, precisions1_, precisions2_, recalls1_, recalls2_, auc1_, auc2_

def start(data_set, rf_flag=False, max_features=None):
    accuracies1 = []
    accuracies2 = []
    precisions1 = []
    precisions2 = []
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
    precisions3 = []
    precisions4 = []
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

    for i in range(10):
        print("Iteration: ", i*10)
        train_features, test_features, train_labels, test_labels = \
            train_test_split(
                features_, labels_,
                test_size=0.1,
                random_state=int(round(time.time()))
            )
        # un-numpy the arrays before predicting
        train_features, test_features, train_labels, test_labels = undress_num_py_arrays(
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
        # train and test their tree
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
        precisions1.append(p1)
        precisions2.append(p2)
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
            precisions3.append(p3)
            precisions4.append(p4)
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
    print("Our average precision:", numpy.array(precisions1).mean())
    print("Our average recall:", numpy.array(recalls1).mean())
    print("Our average AUC:", numpy.array(aucs1).mean())
    print("Our average training time:", numpy.array(our_training_time).mean())
    print("Our average testing time:", numpy.array(our_testing_time).mean())
    print("\n----------------------------------------")
    print("\nFor their Decision Tree Classifier:")
    print("Their average accuracy:", numpy.array(accuracies2).mean())
    print("Their average precision:", numpy.array(precisions2).mean())
    print("Their average recall:", numpy.array(recalls2).mean())
    print("Their average AUC:", numpy.array(aucs2).mean())
    print("Their average training time:", numpy.array(their_training_time).mean())
    print("Their average testing time:", numpy.array(their_testing_time).mean())

    if rf_flag:
        print("\n----------------------------------------")
        print("\nFor our Random Forest Classifier:")
        print("Our average accuracy:", numpy.array(accuracies3).mean())
        print("Our average precision:", numpy.array(precisions3).mean())
        print("Our average recall:", numpy.array(recalls3).mean())
        print("Our average AUC:", numpy.array(aucs3).mean())
        print("Our average training time:", numpy.array(our_training_time1).mean())
        print("Our average testing time:", numpy.array(our_testing_time1).mean())
        print("\n----------------------------------------")
        print("\nFor their Random Forest Classifier:")
        print("Their average accuracy:", numpy.array(accuracies4).mean())
        print("Their average precision:", numpy.array(precisions4).mean())
        print("Their average recall:", numpy.array(recalls4).mean())
        print("Their average AUC:", numpy.array(aucs4).mean())
        print("Their average training time:", numpy.array(their_training_time1).mean())
        print("Their average testing time:", numpy.array(their_testing_time1).mean())

    #w = wilcoxon(aucs2, aucs1)
    #w2 = wilcoxon(aucs1, aucs2)
    #print("\n", w, w2)

    our_decision_tree_data = pandas.DataFrame({'Our average accuracy': numpy.array(accuracies1).mean(),
                           'Our average precision': numpy.array(precisions1).mean(),
                           'Our average recall': numpy.array(recalls1).mean(),
                           'Our average AUC': numpy.array(aucs1).mean(),
                           'Our average training time': numpy.array(our_training_time).mean(),
                           'Our average testing time' : numpy.array(our_testing_time).mean()}, index=[0])
    their_decison_tree_data = pandas.DataFrame({'Their average accuracy': numpy.array(accuracies2).mean(),
                           'Their average precision': numpy.array(precisions2).mean(),
                           'Their average recall': numpy.array(recalls2).mean(),
                           'Their average AUC': numpy.array(aucs2).mean(),
                           'Their average training time': numpy.array(their_training_time).mean(),
                           'Their average testing time' : numpy.array(their_testing_time).mean()}, index=[0])
    our_random_forest_data = pandas.DataFrame({'Our average accuracy': numpy.array(accuracies3).mean(),
                                               'Our average precision': numpy.array(precisions3).mean(),
                                               'Our average recall': numpy.array(recalls3).mean(),
                                               'Our average AUC': numpy.array(aucs3).mean(),
                                               'Our average training time': numpy.array(our_training_time1).mean(),
                                               'Our average testing time': numpy.array(our_testing_time1).mean()},
                                              index=[0])
    their_random_forest_data = pandas.DataFrame({'Their average accuracy': numpy.array(accuracies4).mean(),
                                                'Their average precision': numpy.array(precisions4).mean(),
                                                'Their average recall': numpy.array(recalls4).mean(),
                                                'Their average AUC': numpy.array(aucs4).mean(),
                                                'Their average training time': numpy.array(their_training_time1).mean(),
                                                'Their average testing time': numpy.array(their_testing_time1).mean()},
                                               index=[0])

    writer = pandas.ExcelWriter('breast-w.xlsx', engine='xlsxwriter')

    our_decision_tree_data.to_excel(writer, sheet_name='our dtc')
    their_decison_tree_data.to_excel(writer, sheet_name='their dtc')
    our_random_forest_data.to_excel(writer, sheet_name='our rf')
    their_random_forest_data.to_excel(writer, sheet_name='their rf')

    writer.save()

start(pandas.read_csv(r"..\ILS Projekt Dataset\csv_binary\binary\breast-w.csv", header=None), rf_flag=True)
