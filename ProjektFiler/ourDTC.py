import time

import numpy as np
import pandas
import sklearn.model_selection
from sklearn.tree import DecisionTreeClassifier

from hunts_algorithm import start_hunts
from prediction_node import compare_results, predict

df = pandas.read_csv(r"..\ILS Projekt Dataset\csv_binary\binary\diabetes.csv",
                     header=None)

# print(df[:2])

arr = pandas.np.array(df)
X_arr = arr[1:, :-1]
y_arr = arr[1:, -1:]

X_arr.astype(np.double)

# Bosco Talkshow
# Training set, test set, train klass label, test klass label. We split
# into sets
print(int(round(time.time())))
X, X_test, y, y_test = sklearn.model_selection \
    .train_test_split(X_arr, y_arr, test_size=0.33, random_state=int(round(time.time())))

X = [[float(n) for n in row] for row in X]
X_test = [[float(n) for n in row] for row in X_test]


def test_hunts():
    their_model = DecisionTreeClassifier()
    their_model.fit(X, y)

    model = start_hunts(X, y)

    y_test_predict = predict(model, X_test)
    compare_results(y_test_predict, y_test)

    y_their_predict = their_model.predict(X_test)
    compare_results(y_their_predict, y_test)


test_hunts()



