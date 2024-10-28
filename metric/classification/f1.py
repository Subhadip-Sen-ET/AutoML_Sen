import numpy as np
from sklearn.metrics import f1_score

def calculate_f1(Y_actual: np.array, Y_predicted: np.array, 
                 average: str = 'binary', sample_weight: str = None):

    return f1_score(y_true = Y_actual, 
                    y_pred = Y_predicted,
                    average = average,
                    sample_weight = sample_weight)