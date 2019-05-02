from sklearn import datasets, metrics, preprocessing
from stacked_generalization.lib.stacking import FWLSRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np


boston = datasets.load_boston()
X = preprocessing.StandardScaler().fit_transform(boston.data)
Y = boston.target

X_train = X[:200]
Y_train = Y[:200]
X_test = X[200:]
Y_test = Y[200:]

breg = LinearRegression()
regs = [RandomForestRegressor(n_estimators=50, random_state=1),
        GradientBoostingRegressor(n_estimators=25, random_state=1),
        Ridge(),
        ExtraTreesRegressor(n_estimators=50),
        ]
feature_func = lambda x: np.c_[np.ones((x.shape[0], 1)),
                               x[:, 1].reshape((x.shape[0], 1)),
                               x[:, 6].reshape((x.shape[0], 1)),]

sr = FWLSRegressor(breg,
                   regs,
                   feature_func,
                   n_folds=3,
                   verbose=0,
                   oob_score_flag=False)

sr.fit(X_train, Y_train)
score = metrics.mean_squared_error(sr.predict(X_test), Y_test)
print ("MSE of stacked regressor: %f" % score)
