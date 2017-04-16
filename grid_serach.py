import time

import numpy as np
from sklearn.model_selection import train_test_split

from ProjektFiler.OurDecisionTreeClassifier import OurDecisionTreeClassifier
from ProjektFiler.experiment1 import accuracy_test

class ClassifierFactory:

    def __init__(self):
        self.featureSet = None
        self.labels = None
        self.factory = None

    def set_data(self, featureSet, labels):
        self.featureSet = featureSet
        self.labels = labels

    def set_classifier_factory(self, factory):
        self.factory = factory


def grid_search(dtcFactory):
    return mean_test_suite(featureSets, labels, 10)


def mean_test_suite(featureSets, labels, n_iterations):
    data = np.array([iterate(featureSets, labels) for i in range(0, n_iterations)])
    return np.mean(data, dtype=np.float64)


def iterate(featureSets, labels):
    train_features, test_features, train_labels, test_labels = \
        train_test_split(
            featureSets, labels,
            test_size=0.33,
            random_state=int(round(time.time()))
        )

    dtc = OurDecisionTreeClassifier()
    dtc.fit(train_features, train_labels)
    predictions = dtc.predict(test_features)
    return accuracy_test(predictions, test_labels)

