#implementation of logistic regression from scratch
import numpy as np
import math
from deeplearning.activation import Sigmoid
from tqdm import tqdm

class LogisticRegression():
    def __init__(self, learning_rate = .01, gradient_decent = False) -> None:
        self.learning_rate = learning_rate 
        self.gradient_decent = gradient_decent
        self.weights = None
        self.sigmoid = Sigmoid()

    def initialize_weights(self, X):
        weights_shape = np.shape(X)[1]
        limit = 1/math.sqrt(weights_shape)
        self.weights = np.random.uniform(-limit, limit, weights_shape)
        
    def fit(self, X, y, epoch=5000):
        self.initialize_weights(X)

        for i in tqdm(range(epoch)):
            y_pred = self.sigmoid(X.dot(self.weights))
            
            if self.gradient_decent:
                self.weights -= self.learning_rate * -(y - y_pred).dot(X)

    def predict(self, X):
        y_pred = np.round(self.sigmoid(X.dot(self.weights))).astype(int)
        return y_pred