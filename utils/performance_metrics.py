import numpy as np

#mse -----> mean squared error = (sum(y-y^)**2) / n n---> number of data points
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
    

#rmse(root mean squared error = sqr((sum(y-y^)**2) / n)
def rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)

#r2 score ----> 1 - (ss_residual/ss_total)
#ss_residual ---> sum of square residual, ss_total ----> sum of square total
def r2_score(y_true, y_pred):
    ss_residual = np.sum((y_true - y_pred) ** 2)
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

#adjusted r2_score ---> 1 - ((1 - r2) * (n -1) / (n - p -1))
#n ---> number of datapoints
#p-----> Number of independent features
def adjusted_r2_score(y_true, y_pred, p):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p -1))
    return adj_r2

## Ridge regression loss function:(1/n) Xᵀ(y_pred - y) + 2λw
def ridge_regression(y_true, y_pred, m, lam):
    mse = np.mean((y_true - y_pred) **2)
    ridge = lam * np.sum(m **2)
    return mse + ridge

##Classification perfomance metrics
#accuracy score: --> proportion of the correct predictions
def accuracy_score(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

#precision: ---> Out of predicted positive how many are correct
def precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp)

#Recall: --> Out of actuall positive how many did we detect
def recall_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn)

#F1 Score: ---> balances precision and recall
def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)

#confusion matrix implementation
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tp, fp],
                     [fn, tn]])