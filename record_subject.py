from collections import defaultdict

class Subject:
    """
        Represents a Row in a matrix of training data.
        It contains several features (represents the columns of the data)
        and a class label.
    """
    def __init__(self, features, class_label):
        self.class_label = class_label
        self.class_features = features

    def print(self):
        print("Subject [ class: " + str(self.class_label) + " ]")


def group_has_same_label(subjects):
    """
        Returns true if all subjects contains the same class label.
    :param subjects:
    :return: boolean
    """
    first_label = subjects[0].class_label
    for subject in subjects:
        if subject.class_label != first_label:
            # if the subjects class label is not the same as the first label
            # then this group does not all have the same label
            return False

    # They all have the same class label
    return True


def most_common_class_label(subjects):
    """
    Picks the class label which is most common amongst the given set of subjects.
    :param subjects:
    :return: class label (Any type)
    """
    result_set = defaultdict(int)
    for subject in subjects:
        result_set[subject.class_label[0]] += 1

    return max(result_set, key=result_set.get)
