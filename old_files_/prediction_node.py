from collections import defaultdict, namedtuple

from criterion import generate_class_frequency_map
from record_subject import Subject

SplitNodeSubjects = namedtuple('SplitNode', ['node', 'subjects'])

"""
    PredictionNode becomes the model for the test data.
    We create the node class to hold the label, test: for when we use our model for the
    test data. Ofcourse we also need a reference to the child nodes.
"""


class PredictionNode:
    """
        This Node is used for Prediction of data.
        First the node must be fitting to reflect a model
        for which the prediction is made on. The fitting is done
        by testing splits and choosing the best split and its test.
        This test is stored in the PredictionNode and is thus used for predictions
        of new data.
    """

    def __init__(self, split_value=None):
        self.split_value = split_value
        self.split_test = None
        self.label = None
        self.subjects = None
        self.child_nodes = []

    def set_class_label(self, label):
        self.label = label

    ''' Here we get the right child node from the test, if
        we don't have a test or else: we throw a exception. '''

    def get_child_from_test(self, subject):
        """
            From the test stored in the Node a given subject
            will translate to a different child node based
            on the outcome of the test.
        :param subject:
        :return:
        """

        # No test equals no outcome, so we raise an exception
        if self.split_test is None:
            raise Exception('Node has no split test.')

        # Else we do an split test on the subjects
        result = self.split_test(subject)

        # While looping through the child, we return the child that matches the split value.
        for child in self.child_nodes:
            if child.split_value == result:
                return child

        raise Exception('No suitable path for subject in Node.')


def predict(model, test_subjects):
    subjects = [Subject(row, None) for row in test_subjects]
    res = [get_class_for_subject(model, subject) for subject in subjects]
    # print(res)
    return res


def get_class_for_subject(model, subject):
    if model.label is None:
        return get_class_for_subject(
            model.get_child_from_test(subject),
            subject
        )
    else:
        return model.label


"""

    Class label: n subjects of that class
    A: 2
    B: 3
    C: 1


"""


def get_classes_for_subject(model, subject):
    if not model.label:
        return get_classes_for_subject(
            model.get_child_from_test(subject), subject
        )
    else:
        # print('hej', model.label, model.subjects)
        if model.subjects is not None:
            return generate_class_frequency_map(model.subjects)


def compare_results(prediction, correct_result):
    correct = 0
    for i in range(len(prediction)):
        if prediction[i] == correct_result[i]:
            correct += 1

    percentage = correct / len(prediction)
    print("Correct to " + str(percentage * 100) + "%")


def split_to_prediction_nodes(subjects, split_test):
    """ here we take care of the splitting of the
        nodes and group them correctly"""

    # Creates a dictionary where all empty keys are lists
    splits = defaultdict(list)
    for subject in subjects:
        test_result = split_test(subject)
        # Groups the subject with the result of the split_test
        splits[test_result].append(subject)

    # Puts the group of subjects into new child nodes
    return [
        SplitNodeSubjects(node=PredictionNode(test_result), subjects=splits[test_result])
        for test_result in splits.keys()
        ]
