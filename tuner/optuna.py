import optuna
import numpy as np
from crossvalidation import kfold, stratified_kfold
from metric.classification import f1
from metric.regression import mean_squared_error

class OptunaTuner:
    def __init__(self, n_trials: int = 100, n_splits: int = 5, 
                 study_name: str = 'unknown_study_1', model_type: str = 'classifier'):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.study_name = study_name
        self.model_type = model_type
        self.best_training_score = None
        self.best_validation_score = None

    def tune(self, X: np.array, Y: np.array, model, direction: str = 'maximize'):
        def objective(trial):
            
            self.model = model(trial)
            
            if self.model_type == 'classifier':
                self.model = self.model.classifier(trial)
                splitter = stratified_kfold.StratifiedKFoldSplit(X = X, 
                                                                 Y = Y, 
                                                                 n_splits = self.n_splits)
            if self.model_type == 'regressor':
                self.model = self.model.regressor(trial)
                splitter = kfold.StratifiedKFoldSplit(X = X, 
                                                      Y = Y, 
                                                      n_splits = self.n_splits)
            k_folds = splitter.k_splits()    
            train_X, val_X = k_folds['X Train'], k_folds['X Val']
            train_Y, val_Y = k_folds['Y Train'], k_folds['Y Val']
            
            list_train_score = []
            list_val_score = []
            
            for i in range(len(train_X)):
                self.model.fit(train_X[i], train_Y[i])
                Y_pred_train = self.model.predict(train_X[i])
                Y_pred_val = self.model.predict(val_X[i])
            
                ############ Change metric accordingly ##########    
            
                if self.model_type == 'classifier':
                    unq_cat = len(np.unique(Y))
                    if unq_cat >= 2:
                        average = 'weighted'
                    if unq_cat == 2:
                        average = 'binary'
                    train_score = f1.calculate_f1(Y_actual = train_Y[i], 
                                                  Y_predicted = Y_pred_train,
                                                  average = average)
                    val_score = f1.calculate_f1(Y_actual = val_Y[i],
                                                Y_predicted = Y_pred_val,
                                                average = average)
                    
                if self.model_type == 'regressor':
                    train_score = mean_squared_error.calculate_mean_squared_error(Y_actual = train_Y[i], 
                                                                                  Y_predicted = Y_pred_train)
                    val_score = mean_squared_error.calculate_mean_squared_error(Y_actual = val_Y[i], 
                                                                                Y_predicted = Y_pred_val)
                    
                    
                list_train_score.append(train_score)
                list_val_score.append(val_score)
                    
            mean_training_score = np.mean(list_train_score)
            mean_validation_score = np.mean(list_val_score)
             
            ################### Change till here #####################
            
            if self.best_validation_score is None or mean_validation_score > self.best_validation_score:
                self.best_training_score = mean_training_score
                self.best_validation_score = mean_validation_score
                
            return mean_validation_score

        self.study = optuna.create_study(study_name = self.study_name, 
                                         direction='maximize')
        self.study.optimize(objective, n_trials = self.n_trials)

    def get_best_params(self):
        if self.study is None:
            raise ValueError("Tuning has not been run yet.")
        return self.study.best_params
    
    def get_best_score(self):
        if self.study is None:
            raise ValueError("Tuning has not been run yet.")
        return self.study.best_value
    
    def get_best_training_score(self):
        if self.best_training_score is None:
            raise ValueError("Tuning has not been run yet.")
        return self.best_training_score
    
    def get_best_validation_score(self):
        if self.best_validation_score is None:
            raise ValueError("Tuning has not been run yet.")
        return self.best_validation_score

