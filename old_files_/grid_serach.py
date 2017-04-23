import time
import numpy as np
import pandas

from collections import namedtuple
from sklearn.model_selection import train_test_split
from ProjektFiler.OurDecisionTreeClassifier import OurDecisionTreeClassifier, unzip_features_and_labels
from ProjektFiler.experiment1 import accuracy_test
from excelifyer import Excelifyer

DataSet = namedtuple('DataSet', ['featureSet', 'labels'])


class ClassifierFactory:
    def __init__(self):
        self.featureSets = None
        self.labels = None
        self.factory = None
        self.parameter_reset = None
        self.parameters = None

    def set_data(self, featureSet, labels):
        self.featureSets = featureSet
        self.labels = labels

    def set_classifier_factory(self, factory):
        self.factory = factory

    def set_parameter_iterator(self, parameter_iterator):
        self.parameters = parameter_iterator

    def make_classifier(self, params):
        return self.factory(params)


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Parameters:
    def __init__(self, n_iterations, start_params, steps):
        self.n_iterations = n_iterations
        self.start_params = start_params
        self.steps = steps

    def __iter__(self):
        params = self.start_params
        steps = self.steps
        for x in range(0, self.n_iterations):
            for y in range(0, self.n_iterations):
                updated_parameters = (params[0] + steps[0] * y, params[1] + steps[1] * x)
                position = Position(round(updated_parameters[0] / steps[0]), round(updated_parameters[1] / steps[1]))
                yield (position, updated_parameters)


def grid_search(dtcFactory):
    data = DataSet(dtcFactory.featureSets, dtcFactory.labels)

    output = Excelifyer(use_column_headers=True, use_row_headers=True)

    for positionParametersTuple in dtcFactory.parameters:
        position = positionParametersTuple[0]
        parameters = positionParametersTuple[1]
        print(position.x, position.y)
        output.set_row_header(position.y, 'MF ' + str(parameters[1]))
        output.set_column_header(position.x, 'MSL ' + str(parameters[0]))
        classifier = dtcFactory.make_classifier(parameters)
        mean_accuracy = mean_test_suite(data, classifier, 10)
        output.at_cell(position.x, position.y, mean_accuracy)

    output.to_excel('grid_search_test_1.xlsx')


def mean_test_suite(data_set, classifier, n_iterations):
    data = np.array([iterate(data_set, classifier) for i in range(0, n_iterations)])
    return np.mean(data, dtype=np.float64)


def iterate(data_set, classifier):
    train_features, test_features, train_labels, test_labels = \
        train_test_split(
            data_set.featureSet,
            data_set.labels,
            test_size=0.33,
            random_state=int(round(time.time()))
        )

    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    return accuracy_test(predictions, test_labels)


def test_case():
    data = pandas.read_csv(r"ILS Projekt Dataset\csv_binary\binary\diabetes.csv", header=None)
    data = pandas.np.array(data)
    features, labels = unzip_features_and_labels(data)

    fac = ClassifierFactory()
    fac.set_data(features, labels)

    max_features = 1
    max_features_step = 10

    min_sample_leafs = 1
    min_sample_leafs_step = 10

    iterations = 2

    parameter_iterator = Parameters(iterations, [max_features, min_sample_leafs],
                                    [max_features_step, min_sample_leafs_step])

    fac.set_parameter_iterator(parameter_iterator)
    fac.set_classifier_factory(
        lambda params: OurDecisionTreeClassifier(max_features=params[0], min_sample_leaf=params[1]))
    grid_search(fac)


test_case()
