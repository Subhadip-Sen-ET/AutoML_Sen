from model import randomforest
from sklearn.datasets import load_iris
import pandas as pd
from tuner.optuna import OptunaTuner
import warnings
warnings.filterwarnings('ignore')

iris = load_iris()
data = pd.DataFrame(iris.data, columns = iris.feature_names)
targ = pd.DataFrame(iris.target, columns = ['target'])
df = pd.concat([data, targ], axis=1)
df = df.sample(frac=1)
X = df[df.columns[:-1]].values
Y = df[df.columns[-1]].values

tuner = OptunaTuner(study_name = 'subhadip_study_1', model_type = 'classifier')
tuner.tune(X = X, Y = Y, model = randomforest.RandomForest)






