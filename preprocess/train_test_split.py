from sklearn.model_selection import train_test_split
import numpy as np
RANDOM_STATE = 42

class train_test_split:
    
    def __init__(self, X: np.array, Y: np.array, test_size:float=0.2, stratify=None):
        
        self.X = X
        self.Y = Y
        self.test_size = test_size
        self.stratify = stratify
        self.shuffle = True
        
    def split(self):
        
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, 
                                                            stratify=self.stratify,
                                                            test_size=self.test_size,
                                                            random_state=RANDOM_STATE,
                                                            shuffle=self.shuffle)
        
        return X_train, X_test, Y_train, Y_test
            
        
        