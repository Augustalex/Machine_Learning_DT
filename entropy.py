import math

from criterion import generate_class_frequency_map, SplitInformation, Criterion
from prediction_node import split_to_prediction_nodes


class Entropy(Criterion):
    @staticmethod
    def calculate_node_index(subjects, split_test):
        split_pairs = split_to_prediction_nodes(subjects, split_test)

        parent_entropy = calculate_node_entropy(subjects)

        children_entropy = 0
        for child in split_pairs:
            subject_dist = (len(child.subjects) / len(subjects))
            child_entropy = calculate_node_entropy(child.subjects)
            children_entropy += subject_dist * child_entropy

        information_gain = parent_entropy - children_entropy

        return SplitInformation(split=split_pairs, index=information_gain, test=split_test)

    @staticmethod
    def select_candidate(candidates):
        return min(candidates, key=lambda x: x.index)


def calculate_node_entropy(subjects):
    frequency_map = generate_class_frequency_map(subjects)

    entropy = 0
    for subject_class in frequency_map.keys():
        prob = frequency_map[subject_class] / len(subjects)
        entropy -= prob / (math.log(prob, 2))

    return entropy
