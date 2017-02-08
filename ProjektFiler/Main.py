import sys
import pandas
import scipy
import sklearn.model_selection
from specialNode import Node

df = pandas.read_csv(r"C:\Users\August\Documents\ILS Projekt\ILS Projekt Dataset\csv_binary\binary\labor.csv",
                     header=None)

# print(df[:2])

arr = pandas.np.array(df)
X = arr[1:, :-1]
y = arr[1:, -1:]

# Bosco Talkshow
# Training set, test set, train klass label, test klass label. We split
# into sets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33, random_state=42)

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

node1 = Node(X_train)
result1 = node1.split(multi_way_test)

for x in result1:
    print([row[0] for row in x.subjects])
