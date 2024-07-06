import numpy as np
from collections import Counter
import math


class KNN():
    """
    KNN is a simple algorithm used for classification and regression. 
    It doesn't require traditional training of a model and predicting the tests with the updated weighted matrix.
    Instead, KNN completely depends on the data corpus that we have. 

    The algorithm simply works by finding the k nearest data points in the training set to a given test point and 
    makes predictions based on the majority class (for classification) or the average value (for regression) 
    of those nearest neighbors.
    """
    def __init__(self, k=6):
        self.k = k

    def voting(self, nearest_neighbours):
        majority_class = Counter(nearest_neighbours).most_common(1)[0][0]
        return majority_class
    
    def euclidean_distance(self, x1, x2):
        #the cute math formula which made no sense during school root(sum((xn-yn)^2)
        distances = 0
        for i in range(len(x1)):
            distances += math.pow(x1[i]-x2[i], 2)
        return math.sqrt(distances)


    def predict(self, X_test, X_train, y_train):
        #create an empty array equal to the length of samples in the test data
        y_pred = np.empty(X_test.shape[0])

        for i, sample in enumerate(X_test):

            #get the index of top k elements that are filtered based on the euc distance
            idx = np.argsort([self.euclidean_distance(sample, x) for x in X_train])[:self.k]
            #get the class based on the index from the train labels
            classes = np.array([y_train[i] for i in idx])
            #a voting mechanism to get the majority class from the top k fetched
            y_pred[i] = self.voting(classes)
        
        return y_pred