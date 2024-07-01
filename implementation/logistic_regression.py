from __future__ import print_function
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from supervised.logistic_regression import LogisticRegression
# from deeplearning.activation import Sigmoid

def normalize(X):
    l2 = np.atleast_1d(np.linalg.norm(X, 2, -1))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, -1)

def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def main():
    data = datasets.load_iris()
    X = data.data[data.target != 0]
    y = data.target[data.target!=0]
    X = normalize(X)
    y[y==1] = 0
    y[y==2] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    model  = LogisticRegression(gradient_decent=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

if "__main__":
    main()