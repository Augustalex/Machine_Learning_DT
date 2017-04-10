import math
from collections import defaultdict, namedtuple
from random import randint
from node_and_record import split_to_prediction_nodes, generate_class_frequency_map

SplitInformation = namedtuple('SplitInformation', ['split', 'index', 'test'])


class Criterion:
    def __init__(self, max_features=None):
        self.max_features = max_features

    def calculate_node_index(self, subjects, split_test):
        pass

    def select_candidate(self, candidates):
        pass

    def generate_best_split_of_all_features(self, subjects):
        if len(subjects) == 0:
            return SplitInformation([], 0, None)

        n_features = len(subjects[0].class_features)
        chosen_feature_indices = self.select_n_random_feature_indices(n_features)

        # Generate best split of all best splits for each feature
        # print("\n\t\tGenerating splits.")
        candidates = [self.generate_best_split(subjects, feature_index) for feature_index in chosen_feature_indices]

        # Select the candidate with an index closest to 0
        best_candidate = self.select_candidate(candidates)
        # print("\t\tBest was: " + str(best_candidate.index) + "\n")
        return best_candidate

    def generate_best_split(self, subjects, feature_index):
        # Generate all possible binary splits for the given feature
        test_permutations = generate_binary_split_test_permutations(subjects, feature_index)

        # Calculate entropy gain for each generated split
        candidates = [self.calculate_node_index(subjects, test) for test in test_permutations]

        # Select the lowest index
        best_candidate = self.select_candidate(candidates)

        return best_candidate

    def select_n_random_feature_indices(self, n_features):
        if self.max_features:
            n_chosen_features = round(self.max_features * n_features)
        else:
            n_chosen_features = n_features

        features = [i for i in range(0, n_features)]
        # TODO might be problematic with list comprehension (is it correct in syntax?)
        chosen_feature_indices = [features.remove(randint(0, len(features)-1)) in range(0, n_chosen_features)]

        return chosen_feature_indices


def generate_binary_split_test_permutations(subjects, feature_index):
    """
        Generates all possible binary splits from a given set of subjects
        and a single feature of which to split.
    :param subjects: Which node to test against
    :param feature_index: What feature to split
    :return: a list of split tests
    """
    return [
        lambda x: x.class_features[feature_index] <= subject.class_features[feature_index]
        for subject in subjects
        ]

"""
    This is the Entropy class.
"""


class Entropy(Criterion):

    def __init__(self, max_features=None):
        Criterion.__init__(self, max_features)

    def calculate_node_index(self, subjects, split_test):
        split_pairs = split_to_prediction_nodes(subjects, split_test)

        parent_entropy = calculate_node_entropy(subjects)

        children_entropy = 0
        for child in split_pairs:
            subject_dist = (len(child.subjects) / len(subjects))
            child_entropy = calculate_node_entropy(child.subjects)
            children_entropy += subject_dist * child_entropy

        information_gain = parent_entropy - children_entropy

        return SplitInformation(split=split_pairs, index=information_gain, test=split_test)

    def select_candidate(self, candidates):
        return min(candidates, key=lambda x: x.index)


def calculate_node_entropy(subjects):
    frequency_map = generate_class_frequency_map(subjects)

    entropy = 0
    for subject_class in frequency_map.keys():
        prob = frequency_map[subject_class] / len(subjects)
        entropy -= prob / (math.log(prob, 2))

    return entropy


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
