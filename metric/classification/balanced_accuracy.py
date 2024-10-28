import numpy as np
from sklearn.metrics import balanced_accuracy_score

def calculate_balanced_accuracy(Y_actual: np.array, Y_predicted: np.array, 
                                sample_weight: str = None):

    return balanced_accuracy_score(y_true = Y_actual, 
                                   y_pred = Y_predicted,
                                   sample_weight = sample_weight)

