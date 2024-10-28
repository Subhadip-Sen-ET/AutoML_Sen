import numpy as np
from sklearn.metrics import recall_score

def calculate_recall(Y_actual: np.array, Y_predicted: np.array, 
                     average: str = 'binary', sample_weight: str = None):
    
    return recall_score(y_true = Y_actual, 
                        y_pred = Y_predicted,
                        average = average,
                        sample_weight = sample_weight)