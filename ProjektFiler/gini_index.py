from collections import defaultdict, namedtuple

from specialNode import split

GiniSplit = namedtuple('GiniSplit', ['split', 'gini_index', 'test'])


def generate_best_split_of_all_features(subjects):
    n_features = len(subjects[0].features)

    #print("\n\t\tGenerating splits.")
    candidates = [generate_best_split(subjects, feature_index) for feature_index in range(n_features)]

    best_candidate = min(candidates, key=lambda x: x.gini_index)
    #print("\t\tBest was: " + str(best_candidate.gini_index) + "\n")
    return best_candidate


def generate_best_split(subjects, feature_index):
    # split_permutations = generate_binary_split_permutations(subjects, feature_index)

    test_permutations = generate_binary_split_test_permutations(subjects, feature_index)
    gini_candidates = [calculate_split_gini(subjects, test) for test in test_permutations]

    best_candidate = gini_candidates[0]
    for candidate in gini_candidates:
        if candidate.gini_index < best_candidate.gini_index:
            best_candidate = candidate

    #print("Feature " + str(feature_index))
    #print([str(x.gini_index) for x in gini_candidates])
    return best_candidate


def calculate_split_gini(subjects, test):
    split_pairs = split(subjects, test)
    n_parent_subjects = len(subjects)
    gini_index = 0
    for child in split_pairs:
        child_gini = child_gini_index(child.subjects)

        gini_index += (len(child.subjects) / n_parent_subjects) * child_gini

    return GiniSplit(split=split_pairs, gini_index=gini_index, test=test)


def child_gini_index(subjects):
    class_frequency_map = generate_class_frequency_map(subjects)

    gini_index = 1
    for class_key in class_frequency_map.keys():
        gini_index -= (class_frequency_map[class_key] / len(subjects)) ** 2

    return gini_index


def generate_class_frequency_map(subjects):
    class_frequency = defaultdict(int)

    for subject in subjects:
        class_frequency[subject.class_label[0]] += 1

    return class_frequency


def generate_binary_split_test_permutations(subjects, feature_index):
    return [
        lambda x: x.features[feature_index] <= subject.features[feature_index]
        for subject in subjects
        ]


def generate_binary_split_permutations(subjects, feature_index):
    return [
        split(subjects, lambda x: x.features[feature_index] <= subject.features[feature_index])
        for subject in subjects
        ]
