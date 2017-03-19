from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ProjektFiler.OurDecisionTreeClassifier import OurDecisionTreeClassifier, OurRandomForrestClassifier, \
    unzip_features_and_labels, undress_num_py_arrays


def conduct_experiment_2:

    # K-nearest neighbour
    KNeighborsClassifier()
    # Their random forest
    RandomForestClassifier()
    # Their Decision tree
    DecisionTreeClassifier()

    # Our Decision tree
    OurDecisionTreeClassifier()
    # Our random forest
    OurRandomForrestClassifier()

