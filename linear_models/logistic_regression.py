"""
Logistic regression is used for binary classification problems like:
spam detection
fraud detection
market direction prediction

Idea: y = mx + c but we need probabilities btwn 0 and 1 so we pass sidmoid function where z = mx + c

if p >= 0.5 → class = 1
else → class = 0

updating gradients:
dw = (1/n) * Xᵀ(p - y)
db = (1/n) * Σ(p - y)

"""

import numpy as np
from utils.performance_metrics import *

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.m = None
        self.c = None

    #sigmoid function
    def sigmoid(self,z):
        return 1 / (1 + np.exp(-1))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        #initializing m and c parameters
        self.m = np.zeros(n_features)
        self.c = 0

        for _ in range(self.n_iters):
            #best fit line of linear model
            z = np.dot(X, self.m) + self.c

            #probability prediction using sigmoid
            y_pred = self.sigmoid(z)

            #gradients
            dm = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            dc = (1 / n_samples) * np.sum(y_pred - y)

            #updatig the paramaters
            self.m -= self.lr * dm
            self.c -= self.lr * dc

    def predict_proba(self, X):
        z = np.dot(X, self.m) + self.c
        return self.sigmoid(z)
    
    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= 0.5,1,0)
        
if __name__ == "__main__":

    #input values
    X = np.array([
        [25],
        [30],
        [35],
        [40],
        [45],
        [50]
    ])

    #y label
    y = np.array([0,0,0,1,1,1])
    model = LogisticRegression(lr=0.01, n_iters=5000)
    model.fit(X, y)

    preds = model.predict(X)

    print("Weights: ", model.m)
    print("Intercept: ", model.c)
    print("Predictions: ",preds)
    
