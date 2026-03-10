import numpy as np
from utils.performance_metrics import r2_score, adjusted_r2_score,mse,rmse

class MultipleLinearRegression:
    """
    This algorithm solves supervised classification problem statements
    """
    def __init__(self, lr=0.01, epochs = 1000):
        self.lr = lr
        self.epochs = epochs
        self.m = None
        self.c = None
    
    def fit(self, X, y):
        """Training """
        n_samples, n_features = X.shape
        self.m = np.zeros(n_features)
        self.c = 0

        for _ in range(self.epochs):
            #predictions
            y_pred = np.dot(X, self.m) + self.c

            #getting the gradients using gradient descent apgorith and updating them
            dm = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            dc = (1 / n_samples) * np.sum(y_pred - y)

            #updating the parameters
            self.m = self.m - self.lr * dm
            self.c = self.c - self.lr * dc

    def predict(self, X):
        return np.dot(X, self.m) + self.c


#main entry point
if __name__ == "__main__":
    X = np.array([
        [1400,3,10],
        [1600,3,5],
        [1700,4,2],
        [1875,3,1],
        [1100,2,20]
    ])

    #feature scaling
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    y = np.array([245000,312000,279000,308000,199000])

    model = MultipleLinearRegression(lr=0.0000001, epochs=2000)
    model.fit(X, y)

    preds = model.predict(X)
    print("Weights: ", model.m)
    print("Intercept: ", model.c)

    print("MSE:", mse(y, preds))
    print("RMSE:", rmse(y, preds))
    print("R2:", r2_score(y, preds))
    print("Adjusted R2:", adjusted_r2_score(y, preds, X.shape[1]))
