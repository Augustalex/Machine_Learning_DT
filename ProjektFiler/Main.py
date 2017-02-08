import sys
import pandas
import scipy
import sklearn.model_selection
from specialNode import Node
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import numpy as np

df = pandas.read_csv(r"C:\Users\August\Documents\ILS Projekt\ILS Projekt Dataset\csv_binary\binary\labor.csv",
                     header=None)

# print(df[:2])

arr = pandas.np.array(df)
X_arr = arr[1:, :-1]
y_arr = arr[1:, -1:]

X_arr.astype(np.float)

# Bosco Talkshow
# Training set, test set, train klass label, test klass label. We split
# into sets
X, X_test, y, y_test = sklearn.model_selection\
    .train_test_split(X_arr, y_arr, test_size=0.33, random_state=42)

bdt = DecisionTreeClassifier(max_depth=10)
bdt.fit(X, y)

Z = bdt.predict(scipy.ndarray.astype(X_test, dtype=scipy.float32), False)
print(Z)

i = 0
end = len(Z)
correct = 0
error = 0
while i < end:
    if y_test[i] == Z[i]:
        correct += 1
    else:
        error += 1

    i += 1

print("Percentile:", correct/end)
"""
node = Node(X_train)
result = node.split(
    lambda subject:
    int(subject[0]) < 3
)

for x in result:
    print([row[0] for row in x.subjects])

print("", "")

"""

def binary_test(subject):
    return int(subject[0]) < 2

def multi_way_test(subject):
    if int(subject[0]) == 1:
        return "A"
    elif int(subject[0]) == 2:
        return "B"
    elif int(subject[0]) == 3:
        return "C"
    else:
        return "D"
