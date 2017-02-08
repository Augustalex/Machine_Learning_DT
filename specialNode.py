from collections import defaultdict


class Node:

    def __init__(self, subjects):
        self.subjects = subjects

    def split(self, test):
        splits = defaultdict(list)
        for subject in self.subjects:
            result_option = test(subject)
            splits[result_option].append(subject)

        return [
            Node(splits[key]) for key in splits.keys()
        ]


