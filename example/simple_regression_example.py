from sklearn import datasets, metrics, preprocessing
from stacked_generalization.lib.stacking import StackedRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression


boston = datasets.load_boston()
X = preprocessing.StandardScaler().fit_transform(boston.data)
breg = LinearRegression()
regs = [RandomForestRegressor(n_estimators=50, random_state=1),
        GradientBoostingRegressor(n_estimators=25, random_state=1),
        ExtraTreesRegressor(),
        Ridge(random_state=1)]
sr = StackedRegressor(breg,
                      regs,
                      n_folds=3,
                      verbose=0)
sr.fit(X, boston.target)
score = metrics.mean_squared_error(sr.predict(X), boston.target)
print ("MSE: %f" % score)