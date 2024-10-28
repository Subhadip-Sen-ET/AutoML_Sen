import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_mean_squared_error(Y_actual: np.array, 
                                 Y_predicted: np.array,
                                 sample_weight = None):
    
    return mean_squared_error(y_true = Y_actual, 
                              y_pred = Y_predicted,
                              sample_weight = sample_weight)