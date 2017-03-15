from collections import defaultdict, namedtuple

SplitInformation = namedtuple('SplitInformation', ['split', 'index', 'test'])


class Criterion:
    @staticmethod
    def calculate_node_index(subjects, split_test):
        pass

    @staticmethod
    def select_candidate(candidates):
        pass


def generate_best_split_of_all_features(subjects, criterion):
    n_features = len(subjects[0].class_features)

    # Generate best split of all best splits for each feature
    # print("\n\t\tGenerating splits.")
    candidates = [generate_best_split(subjects, feature_index, criterion) for feature_index in range(n_features)]

    # Select the candidate with an index closest to 0
    best_candidate = criterion.select_candidate(candidates)
    # print("\t\tBest was: " + str(best_candidate.index) + "\n")
    return best_candidate


def generate_best_split(subjects, feature_index, criterion):

    # Generate all possible binary splits for the given feature
    test_permutations = generate_binary_split_test_permutations(subjects, feature_index)

    # Calculate entropy gain for each generated split
    candidates = [criterion.calculate_node_index(subjects, test) for test in test_permutations]

    # Select the lowest index
    best_candidate = criterion.select_candidate(candidates)

    return best_candidate


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
