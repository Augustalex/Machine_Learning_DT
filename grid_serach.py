import time
from collections import namedtuple

import numpy as np
import pandas
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

    def make_classifier(self):
        return self.factory(self.parameters.currentParameters)


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Parameters:
    def __init__(self, nextIteration, position):
        self.currentParameters = nextIteration()
        self._nextIteration = nextIteration
        self.position = position

    def __iter__(self):
        """
            Returns generator that iterates over the parameters via
            the given 'nextIteration' function. It then yields
            the current position of the parameters in the
            iteration matrix.                                   """
        # return self._nextIteration()

        for permutation in self._nextIteration():
            self.currentParameters = permutation
            yield (self.position(self.currentParameters), self.currentParameters)


def grid_search(dtcFactory):
    data = DataSet(dtcFactory.featureSets, dtcFactory.labels)

    output = Excelifyer(use_column_headers=True, use_row_headers=True)

    for positionParametersTuple in dtcFactory.parameters:
        position = positionParametersTuple[0]
        parameters = positionParametersTuple[1]
        print(position.x, position.y)
        output.set_row_header(position.y, 'MF ' + str(parameters[1]))
        output.set_column_header(position.x, 'MSL ' + str(parameters[0]))
        classifier = dtcFactory.make_classifier()
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
    data = pandas.read_csv(r"ILS Projekt Dataset\csv_binary\binary\tic-tac-toe.csv", header=None)
    data = pandas.np.array(data)
    features, labels = unzip_features_and_labels(data)

    fac = ClassifierFactory()
    fac.set_data(features, labels)

    max_features = 1
    max_features_step = 10

    min_sample_leafs = 1
    min_sample_leafs_step = 10

    iterations = 10

    parameter_iterator = Parameters(
        nextIteration=iterate_function([max_features, min_sample_leafs], [max_features_step, min_sample_leafs_step], iterations),
        position=lambda params: Position(round(params[0] / max_features_step), round(params[1] / min_sample_leafs_step))
    )

    fac.set_parameter_iterator(parameter_iterator)
    fac.set_classifier_factory(lambda params: OurDecisionTreeClassifier(max_features=params[0], min_sample_leaf=params[1]))
    grid_search(fac)


def iterate_function(params, steps, iterations):
    def generator():
        for x in range(0, iterations):
            for y in range(0, iterations):
                yield (params[0] + steps[0] * y, params[1] + steps[1] * x)

    return generator

test_case()
