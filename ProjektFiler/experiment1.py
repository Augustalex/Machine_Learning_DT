from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from ProjektFiler.OurDecisionTreeClassifier import OurDecisionTreeClassifier, OurRandomForrestClassifier
from prediction_node import compare_results

odtc = OurDecisionTreeClassifier()
dtc = DecisionTreeClassifier()
orf = OurRandomForrestClassifier()
rf = RandomForestClassifier()

def accuracy_test(prediction1, prediction2, true_class_labels):
    """returns the accuracy of both predictions vs the actual class_labels of the test data"""
    compare_results(prediction1, true_class_labels)
    compare_results(prediction2, true_class_labels)

def recall_test(prediction1, prediction2, true_class_labels):
    recall_score(true_class_labels, prediction1)
    recall_score(true_class_labels, prediction2)

def precision_test(prediction1, prediction2, true_class_lables):
    precision_score(true_class_lables, prediction1)
    precision_score(true_class_lables, prediction2)

def f1_test(prediction1, prediction2, true_class_labels):
    f1_score(true_class_labels, prediction1)
    f1_score(true_class_labels, prediction2)

def auc_test(prediction1, prediction2, true_class_labels):
    roc_auc_score(true_class_labels, prediction1)
    roc_auc_score(true_class_labels, prediction2)

def compare_trees(tree1, tree2, data_set):
    # accuracy 1
    # accuracy 1
    # recall 1
    # recall 2
    # precicion 1
    # precision 2
    pass