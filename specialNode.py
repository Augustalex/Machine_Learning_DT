from collections import defaultdict, namedtuple

SplitNodeSubjects = namedtuple('SplitNode', ['node', 'subjects'])

''' We create the node class to hold the label, test: for when we use our model for the
    test data. Ofcourse we also need a reference to the child nodes.'''


class Node:
    def __init__(self, split_value=None):
        self.split_value = split_value
        self.split_test = None
        self.label = None
        self.child_nodes = []

    def set_class_label(self, label):
        self.label = label

    ''' Here we get the right child node from the test, if
        we don't have a test or else: we throw a exception. '''

    def get_child_from_test(self, subject):
        if self.split_test is None:
            raise Exception('Node has no split test.')

        result = self.split_test(subject)
        for child in self.child_nodes:
            if child.split_value == result:
                return child

        raise Exception('No suitable path for subject in Node.')


def split(subjects, split_test):
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
        SplitNodeSubjects(
            node=Node(test_result),
            subjects=splits[test_result])
        for test_result in splits.keys()
        ]


class Subject:
    def __init__(self, features, class_label):
        self.class_label = class_label
        self.features = features
