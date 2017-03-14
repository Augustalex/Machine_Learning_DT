import numpy
import pandas
import sklearn
import time
from sklearn.svm.libsvm import predict_proba
from sklearn.tree import DecisionTreeClassifier

from hunts_algorithm import start_hunts
from prediction_node import compare_results, predict
from gini_index import Gini


class OurDecisionTreeClassifier:
    data = None
    def __init__(self, data, criterion = Gini, max_features = None, max_depth = None, min_sample_leaf = 1):
        self.data = data
        self.arr = pandas.np.array(data)
        self.X_arr = self.arr[1:, :-1]
        self.y_arr = self.arr[1:, -1:]

        self.X_arr.astype(pandas.np.double)

        # Training set, test set, train klass label, test klass label. We split
        # into sets
        self.X_training, self.X_test, self.y_training, self.y_test = sklearn.model_selection \
            .train_test_split(self.X_arr, self.y_arr, test_size=0.33, random_state=int(round(time.time())))

        self.X_training = [[float(n) for n in row] for row in self.X_training]
        self.X_test = [[float(n) for n in row] for row in self.X_test]
        self.tuple = [self.X_test, self.y_test, self.X_training, self.y_training, self.X_arr, self.y_arr]

    def fit(self):
        model = start_hunts(self.X_training, self.y_training)
        return model

    def predict(self, model):
        test_prediction = predict(model, self.X_test)
        return test_prediction

    def predictProb(self, test_prediction):
        prob = test_prediction
        #gå genom alla subjects
        #om det är en lövnod -> räkna ditrubutionen. (frekvensen/antal)




dtc = OurDecisionTreeClassifier(pandas.read_csv(r"..\ILS Projekt Dataset\csv_binary\binary\diabetes.csv", header=None))
tree = dtc.fit()
test_prediction = dtc.predict(tree)
probability_prediction = dtc.predictProb()

print(probability_prediction)
#print(test_prediction)