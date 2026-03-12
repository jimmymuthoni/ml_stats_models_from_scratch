import numpy as np
from utils.performance_metrics import *

class LassoRegression:
    """
    L1 it introduces fetaure selection:
    costfunction: dw = (1/n) * Xᵀ(y_pred - y) + λ * sign(w)
    """
    def __init__(self,lr=0.01,epochs=1000,lam=0.1):
        self.lr = lr
        self.epochs = epochs
        self.lam = lam
        self.m = None
        self.c = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.m = np.zeros(n_features)
        self.c = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.m) + self.c

            #gradients for co-effiecints
            dm = (1/n_samples) * np.dot(X.T, (y_pred - y)) + self.lam * np.sign(self.m)
            #gradient for bias
            dc = (1/n_samples) * np.sum(y_pred - y)

            #updating gradiensts and bias
            self.m -= self.lr + dm
            self.c -= self.lr + dc

    def predict(self, X):
        return np.dot(X, self.m) + self.c
    
    
if __name__ == "__main__":

    #input features
    X = np.array([
        [1400,3,10],
        [1600,3,5],
        [1700,4,2],
        [1875,3,1],
        [1100,2,20]
    ])

    #ouput feature
    y = np.array([245000,312000,279000,308000,199000])
    #feaure scaling
    X = (X - np.mean(X, axis=0)) // np.std(X, axis=0)

    model = LassoRegression(lr=0.00000001,epochs=2000,lam=0.5)
    model.fit(X, y)

    preds = model.predict(X)

    print("MSE:", mse(y, preds))
    print("R2:", r2_score(y, preds))