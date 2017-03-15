import numpy
import pandas
import time

from criterion import generate_class_frequency_map
from hunts_algorithm import start_hunts
from prediction_node import compare_results, predict, PredictionNode, get_classes_for_subject
from gini_index import Gini
from sklearn.model_selection import train_test_split

from record_subject import Subject


def parse_integer_table(data):
    return [[float(n) for n in row] for row in data]


def unzip_features_and_labels(data):
    features = data[1:, :-1]
    labels = data[1:, -1:]
    features = parse_integer_table(features)
    return features, labels


class OurDecisionTreeClassifier:
    def __init__(self, criterion=Gini, max_features=None, max_depth=None, min_sample_leaf=1):
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.model = None

    def fit(self, features_train, class_labels_train):
        self.model = start_hunts(features_train, class_labels_train)
        return self

    def predict(self, test_features):
        test_prediction = predict(self.model, test_features)
        return test_prediction

    def predictProb(self, test_features):
        #leaf_nodes = []
        #for node in test_features:S
        #    if not node.child_nodes:
        #        leaf_nodes.append(node)
        class_frequency_maps = [get_classes_for_subject(self.model, Subject(test_feature)) for test_feature in test_features]
        return [from_frequency_to_probability(frequency_map) for frequency_map in class_frequency_maps]

class OurRandomForrestClassifier:

    def __init__(self, sample_size, max_features, criterion=Gini, max_depth=None, min_sample_leaf=1, bagging=True, n_estimators=10):
        # criterion: gini or entropy
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.bagging = bagging
        self.sample_size = sample_size
        self.n_estimators = n_estimators

        self.estimators = None

    """
        Our random-forest fit function recieves training data and if bagging is true: loops through
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
        if not self.bagging == False:
            for dt in range(self.n_estimators):
                samples_x = []
                samples_y = []
                for i in range(size):
                    # in each loop we append an random row of data to our samples list.
                    index = numpy.random.randint(0, n_rows)
                    samples_x.append(features_train[index])
                    samples_y.append(class_labels_train[index])
                dt = OurDecisionTreeClassifier()
                model = dt.fit(samples_x,samples_y)
                self.estimators.append(model)


    def predict(self, test_features):
        pass

def from_frequency_to_probability(frequency_map):
    proba_list = []

    total = sum(frequency_map)
    for n_of_class in frequency_map:
        proba_list.append((n_of_class/total))
    return proba_list

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
    test_prediction = dtc.predict(test_features)
    compare_results(test_prediction, test_labels)
    probability_prediction = dtc.predictProb(test_features)
    #
    #print(probability_prediction)
    # print(test_prediction)

run()


