import numpy
import pandas
import time

import sklearn
from sklearn.tree import DecisionTreeClassifier

from hunts_algorithm import start_hunts
from prediction_node import predict, compare_results

df = pandas.read_csv(r".\ILS Projekt Dataset\csv_binary\binary\diabetes.csv",
                     header=None)

arr = pandas.np.array(df)
X_arr = arr[1:, :-1]
y_arr = arr[1:, -1:]

# X_arr.astype(numpy.double)

# Training set, test set, train klass label, test klass label. We split
# into sets
# print(int(round(time.time())))
# x_train, x_test, y_train, y_test = sklearn.model_selection \
#     .train_test_split(X_arr, y_arr, test_size=0.33, random_state=int(round(time.time())))
#
# x_train = [[float(n) for n in row] for row in x_train]
# x_test = [[float(n) for n in row] for row in x_test]


def test_hunts():

    model = start_hunts(x_train, y_train)
    y_test_predict = predict(model, x_train)
    compare_results(y_test_predict, y_train)


# test_hunts()

#numpy.set_printoptions(threshold=numpy.inf)
# dtc = DecisionTreeClassifier()
# dtc.fit(x_train, y_train)
# dtc.predict(x_test)
# hej = dtc.predict_proba(x_train)

# print(hej)
