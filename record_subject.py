from collections import defaultdict


class Subject:
    def __init__(self, features, class_label):
        self.class_label = class_label
        self.features = features

    def print(self):
        print("Subject [ class: " + str(self.class_label) + " ]")


def group_has_same_label(subjects):
    first_label = subjects[0].class_label
    for subject in subjects:
        if subject.class_label != first_label:
            # if the subjects class label is not the same as the first label
            # then this group does not all have the same label
            return False

    # They all have the same class label
    return True


def most_common_class_label(subjects):
    result_set = defaultdict(int)
    for subject in subjects:
        result_set[subject.class_label[0]] += 1

    return max(result_set, key=result_set.get)
