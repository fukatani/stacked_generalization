import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from sklearn import datasets, metrics, preprocessing
from stacked_generalization.lib.stacking import StackedRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression


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
        GradientBoostingRegressor(n_estimators=30, random_state=2),
        ExtraTreesRegressor()
        ]

sr = StackedRegressor(breg,
                      regs,
                      n_folds=3,
                      verbose=0)
sr.fit(X_train, Y_train)
score = metrics.mean_squared_error(sr.predict(X_test), Y_test)
print ("MSE of stacked regressor: %f" % score)

gb = GradientBoostingRegressor(n_estimators=25, random_state=1)
gb.fit(X_train, Y_train)
score = metrics.mean_squared_error(gb.predict(X_test), Y_test)
print ("MSE of gradient boosting regressor: %f" % score)