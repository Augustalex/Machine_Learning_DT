from collections import defaultdict, namedtuple


class Subject:
    """
        Represents a Row in a matrix of training data.
        It contains several features (represents the columns of the data)
        and a class label.
    """

    def __init__(self, features, class_label=None):
        self.class_label = class_label
        self.class_features = features

    def print(self):
        print("Subject [ class: " + str(self.class_label) + " ]")

    def get_as_string(self):
        return '[ ' + str((self.class_label[0] if len(self.class_label[0]) > 1 else self.class_label)) + ' ]'


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


def generate_class_frequency_map(subjects):
    # A dictionary which entries are automatically set to 0 when first assigned a value
    class_frequency = defaultdict(int)
    # For each class label found in a subject increase its entry in the dictionary by 1.
    for subject in subjects:
        class_frequency[subject.class_label[0]] += 1

    return class_frequency


def print_tree(tree):
    output = print_tree_recursive(tree, '')
    print(output)


def print_tree_recursive(tree, acc):
    output = ''
    if len(tree.child_nodes) == 0:
        output += '( LEAF: '
        for subject in tree.subjects:
            output += subject.get_as_string() + ', '
        output += ')'

    for node in tree.child_nodes:
        output += '( ' + str(node.split_value) + ': '
        output += print_tree_recursive(node, '')
        output += ')'

    return acc + output

def print_extreme(tree, acc):
    acc += '└── ' + str(tree.split_value)
    if len(tree.child_nodes) == 0:
        acc += '└── '
        for subject in tree.subjects:
            acc += '├── ' + subject.get_as_string() + ', '
        acc += ')'

    for node in tree.child_nodes:
        acc += '├── ' + str(node.split_value)
        acc += print_tree_recursive(node, acc)
        acc += ')'

class PrintTree:

    def __init__(self, depth, node):
        self.underlings = PrintTree.count_underlings(node)
        self.depth = 0

    @staticmethod
    def count_underlings(node):
        count = len(node.child_nodes)

        for child in node.child_nodes:
            count += PrintTree.count_underlings(child)

        return count