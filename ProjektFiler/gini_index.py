from collections import defaultdict

from specialNode import split


def generate_best_split(parent, subjects, feature_index):
    split_permutations = generate_binary_split_permutations(subjects, feature_index)
    gini_candidates = [calculate_split_gini(len(subjects), split_pairs) for split_pairs in split_permutations]

    best_candidate = gini_candidates[0]
    for candidate in gini_candidates:
        if candidate[1] < best_candidate[1]:
            best_candidate = candidate

    print("Feature " + str(feature_index))
    print([str(x[1]) for x in gini_candidates])
    print("Best was: " + str(best_candidate[1]))
    return best_candidate[0]


def calculate_split_gini(n_parent_subjects, split_pairs):
    gini_index = 0

    for child in split_pairs:
        child_gini = child_gini_index(child.subjects)

        gini_index += (len(child.subjects) / n_parent_subjects) * child_gini

    return split_pairs, gini_index


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


def generate_binary_split_permutations(subjects, feature_index):
    return [
        split(subjects, lambda x: x.features[feature_index] <= subject.features[feature_index])
        for subject in subjects
        ]
