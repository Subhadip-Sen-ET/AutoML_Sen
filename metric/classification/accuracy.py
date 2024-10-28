import numpy as np
from sklearn.metrics import accuracy_score

def calculate_accuracy(Y_actual: np.array, Y_predicted: np.array, 
                       sample_weight: str = None):

    return accuracy_score(y_true = Y_actual, 
                          y_pred = Y_predicted,
                          sample_weight = sample_weight)