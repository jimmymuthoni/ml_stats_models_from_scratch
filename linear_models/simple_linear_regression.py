import pandas as pd
import numpy as np
from utils.performance_metrics import rmse, r2_score

class SimpleLinearRegression:
    """
    Uses equation of a straight line (y = mX + c)to make predions by getting line of best fit.
    Gradient descent is then used to get to global minimum ---> 
    """
    def __init__(self,lr=0.01,epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.m = 0 #slope
        self.c = 0 #intercept

    def fit(self, X, y):
        n = len(X)

        for epoch in range(self.epochs):
            #predictions
            y_pred = self.m*X + self.c

            #gradient descent algo to update values of slope and intercept to minimize loss
            dm = (1/n) * np.sum((X * y_pred - y))
            dc = (1/n) * np.sum(y_pred - y)

            #updating the parameters to make line of best fit
            self.m = self.m - self.lr * dm
            self.c = self.c - self.lr * dc

    def predict(self, X):
        return self.m * X + self.c
    
#program entry
if __name__ == "__main__":
    data = {
        "hours": [1,2,3,4,5,6,7,8],
        "score": [50,55,65,75,80,83,88,92]
    }
    df = pd.DataFrame(data)

    X = df['hours'].values
    y = df['score'].values

    model = SimpleLinearRegression(lr=0.01,epochs=1000)
    model.fit(X, y)
    predictions = model.predict(X)

    print("\nPredictions")
    print(predictions)

    print("\nModel parameters")
    print("Weight (w):", model.m)
    print("Bias (b):", model.c)

    print("\nPerformance Metrics")

    print("RMSE:", rmse(y, predictions))
    print("R2 Score:", r2_score(y, predictions))




