from specialNode import Node
import numpy as np

class Subject:
    def __init__(self, features, class_label):
        self.class_label = class_label
        self.features = features


def testFactory(subjects, featureIndex):
    mean = np.mean(subject.features[featureIndex] for subject in subjects)
    def test(subject):
        return subject.features[featureIndex] < mean

    return test


def hunts(parentNode, featureIndex):
    label = parentNode.subjects.class_label
    all_labels = filter(lambda x: x.class_label == label, parentNode.subjects)
    if len(all_labels) > 1:
        for node in parentNode.split(testFactory(parentNode.subject, featureIndex)):
            hunts(node, featureIndex + 1)
