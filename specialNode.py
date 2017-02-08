from collections import defaultdict, namedtuple

SplitNode = namedtuple('SplitNode', ['node', 'subjects'])

class Node:

    def __init__(self):
        self.label = None
        self.test = None
        self.child_nodes = []

    def split(self, subjects, test):
        self.test = test
        splits = defaultdict(list)
        for subject in subjects:
            result_option = test(subject)
            splits[result_option].append(subject)

        return [SplitNode(node=Node(self), subjects=splits[key]) for key in splits.keys()]

    def set_class_label(self, label):
        self.label = label


