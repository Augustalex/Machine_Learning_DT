from random import randint

import numpy
import pandas
import time

from collections import defaultdict
from scipy._lib.six import xrange

from hunts_algorithm import hunts
from Tree import predict, PredictionNode, get_classes_for_subject, compare_results, Subject, print_tree_vertical
from sklearn.model_selection import train_test_split



def parse_integer_table(data):
    return [[float(n) for n in row] for row in data]


def unzip_features_and_labels(data):
    features = data[1:, :-1]
    labels = data[1:, -1:]
    features = parse_integer_table(features)
    return features, labels


class OurDecisionTreeClassifier:
    def __init__(self, criterion='gini', max_features=None, max_depth=None, min_sample_leaf=1):
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.model = PredictionNode()

    def fit(self, features_train, class_labels_train):
        subjects = [Subject(row, label) for row, label in zip(features_train, class_labels_train)]
        hunts(self.model, subjects, 0, self.min_sample_leaf, self.max_depth, self.criterion, self.max_features)
        return self

    def predict(self, test_features):
        test_prediction = predict(self.model, test_features)
        return test_prediction

    def predictProb(self, test_features, roundToDecimal=None):
        class_frequency_maps = [get_classes_for_subject(self.model, Subject(test_feature)) for test_feature in test_features]
        return [from_frequency_to_probability(frequency_map, roundToDecimal) for frequency_map in class_frequency_maps]


class OurRandomForrestClassifier:
    def __init__(self, sample_size, n_estimators, max_features=None, criterion="gini", max_depth=None, min_sample_leaf=1,
                 bagging=True):
        # criterion: gini or entropy
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.bagging = bagging
        self.sample_size = sample_size
        self.n_estimators = n_estimators

        self.estimators = []

    """
        Our random-forest fit function receives training data and if bagging is true: loops through
        all the estimators and inserts random samples until (sample size * total rows) is
        reached. When the sample data for a estimator is filled we create a decision tree
        classifier and fit the tree to that data and add it to our estimators list for predictions
        in the next step.
    """

    def fit(self, features_train, class_labels_train):
        # total number of rows
        n_rows = len(class_labels_train)
        # total sample size to use: by multiplying with total rows.
        size = n_rows * self.sample_size
        if self.bagging:
            for dt in xrange(self.n_estimators):
                samples_x = []
                samples_y = []
                for i in xrange(int(size)):
                    # in each loop we append an random row of data to our samples list.
                    index = randint(0, n_rows - 1)
                    samples_x.append(features_train[index])
                    samples_y.append(class_labels_train[index])
                dt = OurDecisionTreeClassifier()
                model = dt.fit(samples_x, samples_y)
                self.estimators.append(model)
        else:  # the definition of insanity, to do the same thing over and over again hoping for a different outcome
            for dt in xrange(self.n_estimators):
                dt = OurDecisionTreeClassifier()
                model = dt.fit(features_train, class_labels_train)
                self.estimators.append(model)

    def predict(self, test_features):
        """ predict method takes in the test features and for each created estimators model
            it runs the prediction algorithm on the test_features data.
            It then returns the most voted for prediction.
            """
        predictions = [predict(estimator.model, test_features) for estimator in self.estimators]

        nominees = []
        for i in range(0, len(test_features)):
            castVotes = [prediction_set[i] for prediction_set in predictions]
            nominees.append(nominate_index(castVotes))

        return nominees


def nominate_index(votes_by_index):
    count = defaultdict(int)
    for vote in votes_by_index:
        if type(vote) == numpy.ndarray:
            vote = vote[0]

        count[vote] += 1

    return max(count, key=lambda i: count[i])


def from_frequency_to_probability(frequency_map, roundToDecimal=None):
    proba_list = []

    total = sum(frequency_map.values())
    for class_label in frequency_map:
        probability = frequency_map[class_label] / total
        if roundToDecimal is not None:
            round(probability, roundToDecimal)
        proba_list.append((class_label, probability))
    return proba_list


def flatten_num_py_arrays(arrays):
    return tuple([array.tolist() if type(array) == 'numpy.ndarray' else array for array in arrays])


def run():
    data = pandas.read_csv(r"..\ILS Projekt Dataset\csv_binary\binary\diabetes.csv", header=None)
    dtc = OurDecisionTreeClassifier()
    data = pandas.np.array(data)
    features_, labels_ = unzip_features_and_labels(data)
    train_features, test_features, train_labels, test_labels = \
        train_test_split(
            features_, labels_,
            test_size=0.33,
            random_state=int(round(time.time()))
        )

    dtc.fit(train_features, train_labels)
    print_tree_vertical(dtc.model)
    # for l in map(lambda x: x, test_prediction):
    #     print(l)



    # compare_results(test_prediction, test_labels)
    # probability_prediction = dtc.predictProb(test_features)
    #
    # print(probability_prediction)
    # print(test_prediction)


def run_forest_run():
    data = pandas.read_csv(r"..\ILS Projekt Dataset\csv_binary\binary\diabetes.csv", header=None)
    data = pandas.np.array(data)
    features_, labels_ = unzip_features_and_labels(data)
    train_features, test_features, train_labels, test_labels = \
        train_test_split(
            features_, labels_,
            test_size=0.33,
            random_state=int(round(time.time()))
        )

    train_features, test_features, train_labels, test_labels = flatten_num_py_arrays(
        [train_features, test_features, train_labels, test_labels])

    rfc = OurRandomForrestClassifier(sample_size=0.3, n_estimators=11)
    rfc.fit(train_features, train_labels)
    test_prediction = rfc.predict(test_features)
    compare_results(test_prediction, test_labels)

# run()