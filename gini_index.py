from collections import defaultdict, namedtuple

from prediction_node import split_to_prediction_nodes

GiniSplit = namedtuple('GiniSplit', ['split', 'gini_index', 'test'])


def generate_best_split_of_all_features(subjects):
    n_features = len(subjects[0].features)

    # Generate best split of all best splits for each feature
    print("\n\t\tGenerating splits.")
    candidates = [generate_best_split(subjects, feature_index) for feature_index in range(n_features)]

    # Select the candidate with a gini_index closest to 0
    best_candidate = min(candidates, key=lambda x: x.gini_index)
    print("\t\tBest was: " + str(best_candidate.gini_index) + "\n")
    return best_candidate


def generate_best_split(subjects, feature_index):
    # split_permutations = generate_binary_split_permutations(subjects, feature_index)

    # Generate all possible binary splits for the given feature
    test_permutations = generate_binary_split_test_permutations(subjects, feature_index)

    # Calculate gini index for each generated split
    gini_candidates = [calculate_split_gini(subjects, test) for test in test_permutations]

    # Select the best gini index
    best_candidate = min(gini_candidates, key=lambda x: x.gini_index)

    #print("Feature " + str(feature_index))
    #print([str(x.gini_index) for x in gini_candidates])
    return best_candidate


def calculate_split_gini(subjects, test):

    # Perform split of subjects based on given test
    split_pairs = split_to_prediction_nodes(subjects, test)

    # Generate Gini Index for split based on Gini Index Algorithm
    n_parent_subjects = len(subjects)
    gini_index = 0
    for child in split_pairs:
        child_gini = child_gini_index(child.subjects)

        gini_index += (len(child.subjects) / n_parent_subjects) * child_gini

    # Return a tuple containing the split, gini_index and a test (which is needed to store in a node for prediction)
    return GiniSplit(split=split_pairs, gini_index=gini_index, test=test)


def child_gini_index(subjects):
    # A dictionary containing how frequently a class label occurs amongst the subjects.
    class_frequency_map = generate_class_frequency_map(subjects)

    # Generate child node Gini Index based on Gini Index algorithm
    gini_index = 1
    for class_key in class_frequency_map.keys():
        gini_index -= (class_frequency_map[class_key] / len(subjects)) ** 2

    return gini_index


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
        lambda x: x.features[feature_index] <= subject.features[feature_index]
        for subject in subjects
        ]

