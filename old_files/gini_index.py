from collections import namedtuple

from criterion import generate_class_frequency_map, Criterion
from prediction_node import split_to_prediction_nodes

SplitInformation = namedtuple('SplitInformation', ['split', 'index', 'test'])


class Gini(Criterion):

    def __init__(self, max_features=None):
        Criterion.__init__(self, max_features)

    def calculate_node_index(self, subjects, split_test):
        # Perform split of subjects based on given test
        split_pairs = split_to_prediction_nodes(subjects, split_test)

        # Generate Gini Index for split based on Gini Index Algorithm
        n_parent_subjects = len(subjects)
        gini_index = 0
        for child in split_pairs:
            child_gini = child_gini_index(child.subjects)

            gini_index += (len(child.subjects) / n_parent_subjects) * child_gini

        # Return a tuple containing the split, gini_index and a test (which is needed to store in a node for prediction)
        return SplitInformation(split=split_pairs, index=gini_index, test=split_test)

    def select_candidate(self, candidates):
        return min(candidates, key=lambda x: x.index)


def child_gini_index(subjects):
    # A dictionary containing how frequently a class label occurs amongst the subjects.
    class_frequency_map = generate_class_frequency_map(subjects)

    # Generate child node Gini Index based on Gini Index algorithm
    gini_index = 1
    for class_key in class_frequency_map.keys():
        gini_index -= (class_frequency_map[class_key] / len(subjects)) ** 2

    return gini_index
