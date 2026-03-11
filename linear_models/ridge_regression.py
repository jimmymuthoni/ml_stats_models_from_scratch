import numpy as np
from utils.performance_metrics import mse, r2_score

class RidgeRegression:
    """
    RidgeRegression(L2 reguralarization) controls overfitting; Used when there is overfitting
    overfitting happens when model performs well on the training data but poorly on unseen data.
    Idea: (1/n) Xᵀ(y_pred - y) + 2λm
    (1/n) Xᵀ(y_pred - y) is the MSE
    λ is the regularization stregth
    2m penalty for large weights
    """
    def __init__(self, lr=0.001,epochs=1000,lam=0.1):
        self.lr = lr
        self.epochs = epochs
        self.lam = lam
        self.m = None
        self.c = None
    
    def fit(self, X, y):
        """Getting line of best fit"""
        n_samples, n_features = X.shape
        self.m = np.zeros(n_features)
        self.c = 0

        for _ in range(self.epochs):
            #predictions 
            y_pred = np.dot(X, self.m) + self.c

            #getting gradients using gradient descent
            dm = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + 2 + self.lam + self.m
            dc = (1 / n_samples) * np.sum(y_pred - y)

            #update
            self.m -= self.lr * dm
            self.c  -= self.lr * dc

    def predict(self, X):
        return np.dot(X, self.m) + self.c
    
if __name__ == "__main__":

    #input feature
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

    model = RidgeRegression(lr=0.01, epochs=1000, lam=0.5)
    model.fit(X, y)

    preds = model.predict(X)

    print("MSE:", mse(y, preds))
    print("R2:", r2_score(y, preds))