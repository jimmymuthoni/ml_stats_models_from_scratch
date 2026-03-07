import numpy as np

#mse -----> mean squared error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
    

#rmse(root mean squared error)
def rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)

#r2 score
def r2_score(y_true, y_pred):
    ss_residual = np.sum((y_true - y_pred) ** 2)
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2