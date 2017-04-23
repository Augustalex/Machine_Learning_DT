import numpy
import pandas
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from ProjektFiler.experiment1 import accuracy_test, our_auc_score, our_recall_score, our_precision_score

from ProjektFiler.OurDecisionTreeClassifier import OurDecisionTreeClassifier, OurRandomForrestClassifier, \
    unzip_features_and_labels, flatten_num_py_arrays

from excelifyer import Excelifyer

#max features och min sample leaf
def simple_grid_search(data_set, file_name):
    data_set = pandas.np.array(data_set)
    features_, labels_ = unzip_features_and_labels(data_set)

    train_features, test_features, train_labels, test_labels = \
        train_test_split(
            features_, labels_,
            test_size=0.3,
            random_state=int(round(time.time()))
        )
    # un-numpy the arrays before predicting
    train_features, test_features, train_labels, test_labels = flatten_num_py_arrays(
        [train_features, test_features, train_labels, test_labels])

    algorithmResults = {}
    algorithmResults['ODTC'] = []
    algorithmResults['ORFC'] = []
    algorithmResults['DTC'] = []
    algorithmResults['RFC'] = []
    algorithmResults['KNN'] = []

    max_features_step = 2
    sample_leaf_step = 5


    result_files = {}
    result_files['ODTC'] = Excelifyer(use_column_headers=False)
    result_files['ORFC'] = Excelifyer(use_column_headers=False)
    result_files['DTC'] = Excelifyer(use_column_headers=False)
    result_files['RFC'] = Excelifyer(use_column_headers=False)
    result_files['KNN'] = Excelifyer(use_column_headers=False)

    for x in range(1, 10):
        max_features = x * max_features_step
        algorithmResults['ODTC'].append([])
        algorithmResults['ORFC'].append([])
        algorithmResults['DTC'].append([])
        algorithmResults['RFC'].append([])
        algorithmResults['KNN'].append([])
        if max_features >= len(train_features[0]):
            max_features = len(train_features[0]) - 1
        for y in range(1, 20):
            sample_leaf = y * sample_leaf_step
            odtc = OurDecisionTreeClassifier(max_features=max_features, min_sample_leaf=sample_leaf)
            orfc = OurRandomForrestClassifier(max_features=max_features,min_sample_leaf=sample_leaf, sample_size=0.3, n_estimators=11)
            dtc = DecisionTreeClassifier(max_features=max_features,min_samples_leaf=sample_leaf)
            rfc = RandomForestClassifier(max_features=max_features,min_samples_leaf=sample_leaf, n_estimators=11)
            knn = KNeighborsClassifier(leaf_size=sample_leaf)

            odtc.fit(train_features, train_labels)
            orfc.fit(train_features, train_labels)
            dtc.fit(train_features, train_labels)
            rfc.fit(train_features, train_labels.ravel())
            knn.fit(train_features, train_labels.ravel())

            our_prediction_dtc = odtc.predict(test_features)
            our_prediction_rfc = orfc.predict(test_features)
            their_prediction_dtc = dtc.predict(test_features)
            their_prediction_rfc = rfc.predict(test_features)
            knn_prediction = knn.predict(test_features)

            a_odtc = accuracy_test(our_prediction_dtc, test_labels)
            a_orfc = accuracy_test(our_prediction_rfc, test_labels)
            a_dtc = accuracy_test(their_prediction_dtc, test_labels)
            a_rfc = accuracy_test(their_prediction_rfc, test_labels)
            algorithmResults['ODTC'][x - 1].append(a_odtc)
            algorithmResults['ORFC'][x - 1].append(a_orfc)
            algorithmResults['DTC'][x - 1].append(a_dtc)
            algorithmResults['RFC'][x - 1].append(a_rfc)
            a_knn = accuracy_test(knn_prediction, test_labels)
            algorithmResults['KNN'][x-1].append(a_knn)

            result_files['ODTC'].at_cell(x - 1, y - 1, a_odtc)
            result_files['ORFC'].at_cell(x - 1, y - 1, a_orfc)
            result_files['DTC'].at_cell(x - 1, y - 1, a_dtc)
            result_files['RFC'].at_cell(x - 1, y - 1, a_rfc)
            result_files['KNN'].at_cell(x - 1, y - 1, a_knn)
            print(x, y)

    for algorithm in result_files:
        result_files[algorithm].to_excel('gridSearchResult' + file_name + algorithm + '.xlsx', sheet_name=algorithm)

    optimas = {}
    for algorithm in algorithmResults:
        algorithmOptmia = [0,0]
        for x in range(len(algorithmResults[algorithm])):
            for y in range(len(algorithmResults[algorithm][x])):
                if algorithmResults[algorithm][x][y] > algorithmResults[algorithm][algorithmOptmia[0]][algorithmOptmia[1]]:
                        algorithmOptmia = [x, y]
        optimas[algorithm] = [
            algorithmOptmia[0] * max_features_step, #Max features optmia
            algorithmOptmia[1] * sample_leaf_step, #Min sample leafs
            algorithmResults[algorithm][algorithmOptmia[0]][algorithmOptmia[1]]] #Accury value for the two parameter values
        print(algorithm, optimas[algorithm])

    best_alg = max(optimas.keys(), key=(lambda key: optimas[key][2]))
    print(best_alg)

    doc = Excelifyer(use_column_headers=False)
    doc.at_row(0, ' ', ['Algorithm', 'Max features', 'Min sample leafs', 'Accuracy'])
    rowIndex = 1
    for optima in optimas:
        arr = [optima, optimas[optima][0], optimas[optima][1], optimas[optima][2]]
        print(arr)
        doc.at_row(rowIndex, ' ', arr)
        rowIndex += 1

    doc.to_excel('gridSearchRanking' + file_name + '.xlsx')


simple_grid_search(pandas.read_csv(r"..\ILS Projekt Dataset\csv_binary\binary\hepatitis.csv", header=None), 'hepatitis')

"""
data_set = pandas.np.array(pandas.read_csv(r"..\ILS Projekt Dataset\csv_binary\binary\balance-scale.csv", header=None))

features_, labels_ = unzip_features_and_labels(data_set)

train_features, test_features, train_labels, test_labels = \
    train_test_split(
        features_, labels_,
        test_size=0.2,
        random_state=int(round(time.time()))
    )
# un-numpy the arrays before predicting
train_features, test_features, train_labels, test_labels = flatten_num_py_arrays(
    [train_features, test_features, train_labels, test_labels])

hej = DecisionTreeClassifier(max_features=4)
hej2 = DecisionTreeClassifier(max_features=2)
hej2.fit(train_features, train_labels)
hej.fit(train_features, train_labels)
p = hej.predict(test_features)
p2 = hej2.predict(test_features)
a = accuracy_test(p, test_labels)
a2 = accuracy_test(p2, test_labels)

knn = KNeighborsClassifier()
knn.fit(train_features, train_labels.ravel())
pp = knn.predict(test_features)

a3 = accuracy_test(pp, test_labels)

print("featurea", len(train_features[0]))
print(a, a2, a3)
"""
