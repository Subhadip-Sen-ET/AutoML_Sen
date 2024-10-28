import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_roc_auc(Y_actual: np.array, Y_predicted_proba: np.array,
                      average: str = None):
    
    return roc_auc_score(y_true = Y_actual, 
                         y_score = Y_predicted_proba,
                         average = average)