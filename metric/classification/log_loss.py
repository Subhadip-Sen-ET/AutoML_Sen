import numpy as np
from sklearn.metrics import log_loss

def calculate_log_loss(Y_actual: np.array, Y_predicted_proba: np.array, 
                       sample_weight=None):
    
    return log_loss(y_true = Y_actual, 
                    y_pred = Y_predicted_proba,
                    sample_weight = sample_weight)