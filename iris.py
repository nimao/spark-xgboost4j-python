import xgboost as xgb
import pandas as pd
import numpy as np

from sklearn import preprocessing

iris = pd.read_csv('iris.data',
        names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

le = preprocessing.LabelEncoder()

iris['class'] = le.fit_transform(iris['class'])

label = iris['class']
features = iris.drop('class', axis=1)

data = xgb.DMatrix(features, label=label)

param = {'eta': 0.1,
        'max_depth': 2,
        'objective': 'multi:softprob',
        'num_class': 3,
        'num_round': 100,
        'num_workers': 2}

bst = xgb.train(param, data)


bst.save_model('python_xgb.model')
