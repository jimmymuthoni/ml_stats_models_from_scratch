import numpy as np
import pandas as pd


class LinearRegression:
    def __init__(self, learning_rate = 0.01, epochs = 1000, normalize = False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.normalize = normalize
        self.theta = None
        self.cost_hostory = []
        self.mean = None
        self.std = None

        #feature scaling
        def _normalize(self, X):
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=1)
            return (X - self.mean) / self.std

        #add bias term
        def add_bias(self, X):
            m = X.shape[0]
            return 
