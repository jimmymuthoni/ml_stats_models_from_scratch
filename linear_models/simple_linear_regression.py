import pandas as pd
import numpy as np

class SimpleLinearRegression:
    def __init__(self,rl=0.01,epochs=1000):
        self.rl = rl
        self.epochs = epochs
        self.m = 0 #slope
        self.c = 0 #intercept

    def fit(self, X, y):
        n = len(X)

        for epoch in range(self.epochs):
            #predictions
            y_pred = self.m*X + self.c

            #gradients ---> gradient descent algo
            dm = 

