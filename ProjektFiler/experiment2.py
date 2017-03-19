from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from ProjektFiler.experiment1 import accuracy_test, our_auc_score, our_recall_score, our_precision_score

from ProjektFiler.OurDecisionTreeClassifier import OurDecisionTreeClassifier, OurRandomForrestClassifier, \
    unzip_features_and_labels, undress_num_py_arrays


def conduct_experiment_2():

    # K-nearest neighbour
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                         metric_params=None, n_jobs=1)

    # Their random forest
    rfc = RandomForestClassifier(n_estimators=10, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                           min_weight_fraction_leaf=0, max_features="auto", max_leaf_nodes=None, bootstrap=True)
    # Their Decision tree
    dtc = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                           min_weight_fraction_leaf=0, max_features="auto", max_leaf_nodes=None)

    # Our Decision tree
    odtc = OurDecisionTreeClassifier(max_features=None, max_depth=None, min_sample_leaf=1)
    # Our random forest
    orfc = OurRandomForrestClassifier(sample_size=0.2, n_estimators=11, max_features=None, max_depth=None,
                               min_sample_leaf=1, bagging=True)

