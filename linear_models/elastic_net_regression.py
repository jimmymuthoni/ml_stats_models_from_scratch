import numpy as np
from utils.performance_metrics import *

class ElasticNetRegression:
    """
    This combines both L2 and L1
    """
    def __init__(self, lr=0.001, epochs=1000, l1=0.1, l2=0.1):
        self.lr = lr
        self.n_iters = epochs
        self.l1 = l1
        self.l2 = l2
        self.m = None
        self.c = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.m = np.zeros(n_features)
        self.c = 0
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.m) + self.c
            # gradients
            dm = (1/n_samples) * np.dot(X.T, (y_pred - y)) + self.l1 * np.sign(self.m) + 2 * self.l2 * self.m
            dc = (1/n_samples) * np.sum(y_pred - y)

            # update
            self.m -= self.lr * dm
            self.c -= self.lr * dc

    def predict(self, X):
        return np.dot(X, self.m) + self.c
    
if __name__ == "__main__":
    X = np.array([
        [1400,3,10],
        [1600,3,5],
        [1700,4,2],
        [1875,3,1],
        [1100,2,20]
    ])

    y = np.array([245000,312000,279000,308000,199000])

    # scale features
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    model = ElasticNetRegression(
        lr=0.01,
        epochs=5000,
        l1=0.1,
        l2=0.1
    )

    model.fit(X, y)
    preds = model.predict(X)

    print("MSE:", mse(y, preds))
    print("R2:", r2_score(y, preds))
