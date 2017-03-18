from collections import defaultdict, namedtuple
from random import randint

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


def generate_class_frequency_map(subjects):
    # A dictionary which entries is automatically set to 0 when first assigned a value
    class_frequency = defaultdict(int)
    # For each class label found in a subject increase its entry in the dictionary by 1.
    for subject in subjects:
        class_frequency[subject.class_label[0]] += 1

    return class_frequency


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
