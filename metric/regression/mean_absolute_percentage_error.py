import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

def calculate_mean_absolute_percentage_error(Y_actual: np.array,
                                             Y_predicted: np.array,
                                             sample_weight = None):
    
    return mean_absolute_percentage_error(Y_true = Y_actual,
                                          Y_pred = Y_predicted,
                                          sample_weight = sample_weight)