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

    # X_arr.astype(pandas.np.double)
    features = parse_integer_table(features)

    # Training set, test set, train klass label, test klass label. We split
    # into sets
    # X_test = parse_integer_table(X_test)

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
