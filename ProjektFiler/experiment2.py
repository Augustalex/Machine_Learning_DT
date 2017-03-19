import numpy
import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from ProjektFiler.experiment1 import accuracy_test, our_auc_score, our_recall_score, our_precision_score

from ProjektFiler.OurDecisionTreeClassifier import OurDecisionTreeClassifier, OurRandomForrestClassifier, \
    unzip_features_and_labels, undress_num_py_arrays

# Maybe have de optimized classifiers as parameters?
"""
    This function is used to compare the optimized classifiers and write their result to an excel document.
    It loops through all the classifiers and conducts N predictions and averages them to be able to give an
    overall value. The metrics tested are: Accuracy, precision, recalls and AUC.
"""

def test_optimized_classifiers():
    all_classifiers = []
    # K-nearest neighbour
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                               metric='minkowski',
                               metric_params=None, n_jobs=1)

    # Their random forest
    rfc = RandomForestClassifier(n_estimators=10, criterion="gini", max_depth=None, min_samples_split=2,
                                 min_samples_leaf=1,
                                 min_weight_fraction_leaf=0, max_features="auto", max_leaf_nodes=None, bootstrap=True)
    # Their Decision tree
    dtc = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None, min_samples_split=2,
                                 min_samples_leaf=1,
                                 min_weight_fraction_leaf=0, max_features="auto", max_leaf_nodes=None)

    # Our Decision tree
    odtc = OurDecisionTreeClassifier(max_features=None, max_depth=None, min_sample_leaf=1)
    # Our random forest
    orfc = OurRandomForrestClassifier(sample_size=0.2, n_estimators=11, max_features=None, max_depth=None,
                                      min_sample_leaf=1, bagging=True)

    all_classifiers.append([knn, rfc, dtc, odtc, orfc])

    # Excel file
    writer = pandas.ExcelWriter('experiment2.xlsx', engine='xlsxwriter')

    # Parameter optimalixation here? process here?

    # After fitting
    for classifier in all_classifiers:
        recalls = []
        precisions = []
        aucscores = []
        accuracies = []
        for i in range(10):
            # Do 10 predictions
            prediction = i.predict()

            # use prediction here
            rs = our_recall_score()
            ps = our_precision_score()
            auc = our_auc_score()
            acc = accuracy_test()

            recalls.append(rs)
            precisions.append(ps)
            aucscores.append(auc)
            accuracies.append(acc)

        # For debugging
        print("Average accuracy:", numpy.array(accuracies).mean())
        print("Average precision:", numpy.array(precisions).mean())
        print("Average recall:", numpy.array(recalls).mean())
        print("Average AUC:", numpy.array(aucscores).mean())
        # time is irrelevant?

        # Convert to dataframe for excel
        classifier_data = pandas.DataFrame({"Average accuracy:", numpy.array(accuracies).mean(),
                                                   "Average precision:", numpy.array(precisions).mean(),
                                                   "Average recall:", numpy.array(recalls).mean(),
                                                   "Average AUC:", numpy.array(aucscores).mean()},
                                                  index=[0])

        # Add sheet
        classifier_data.to_excel(writer, sheet_name=str(classifier))

    writer.save()
