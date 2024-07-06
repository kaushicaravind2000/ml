from sklearn.model_selection import train_test_split
from sklearn import datasets
import sys
import os
from logistic_regression import accuracy_score, normalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from supervised.k_nearest_neighbours import KNN


def main():
    dataset =  datasets.load_iris()
    #l2 regularize
    X = normalize(dataset.data)
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    knn = KNN()
    y_pred = knn.predict(X_test,X_train, y_train)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    
if "__main__":
    main()