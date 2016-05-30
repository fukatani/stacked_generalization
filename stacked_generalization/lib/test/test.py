import unittest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


from sklearn import datasets
from sklearn.utils.validation import check_random_state
from stacked_generalization.lib.stacking import StackedClassifier
from stacked_generalization.lib.stacking import StackedRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils.testing import assert_less
import numpy as np
from stacked_generalization.lib.util import numpy_c_concatenate
from stacked_generalization.lib.util import saving_predict_proba
from stacked_generalization.lib.util import get_model_id


class TestStackedClassfier(unittest.TestCase):
    def setUp(self):
        iris = datasets.load_iris()
        rng = check_random_state(0)
        perm = rng.permutation(iris.target.size)
        iris.data = iris.data[perm]
        iris.target = iris.target[perm]
        self.iris = iris

    def test_stacked_classfier(self):
        bclf = LogisticRegression(random_state=1)
        clfs = [RandomForestClassifier(n_estimators=40, criterion = 'gini', random_state=1),
                ExtraTreesClassifier(n_estimators=30, criterion = 'gini', random_state=3),
                GradientBoostingClassifier(n_estimators=25, random_state=1),
                RidgeClassifier(random_state=1),
                ]
        for n_folds, stack_by_proba in self.iter_for_stack_param():
            sl = StackedClassifier(bclf,
                                   clfs,
                                   n_folds=n_folds,
                                   verbose=0,
                                   stack_by_proba=stack_by_proba,
                                   oob_score_flag=True)
            sl.fit(self.iris.data, self.iris.target)
            score = sl.score(self.iris.data, self.iris.target)
            self.assertGreater(score, 0.8, "Failed with score = {0}".format(score))
            self.assertGreater(score, 0.8, "Failed with score = {0}".format(sl.oob_score_))
            print('oob_score: {0} @n_folds={1}, stack_by_proba={2}'
                  .format(sl.oob_score_, sl.n_folds, sl.stack_by_proba))

    def iter_for_stack_param(self):
        yield 2, True
        yield 4, True
        yield 2, False
        yield 3, False

    def test_stacked_regressor(self):
        bclf = LinearRegression()
        clfs = [RandomForestRegressor(n_estimators=50, random_state=1),
                GradientBoostingRegressor(n_estimators=25, random_state=1),
                Ridge(random_state=1)]

        # Friedman1
        X, y = datasets.make_friedman1(n_samples=1200,
                                       random_state=1,
                                       noise=1.0)
        X_train, y_train = X[:200], y[:200]
        X_test, y_test = X[200:], y[200:]

        sr = StackedRegressor(bclf,
                              clfs,
                              n_folds=3,
                              verbose=0,
                              oob_score_flag=True)
        sr.fit(X_train, y_train)
        mse = mean_squared_error(y_test, sr.predict(X_test))
        assert_less(mse, 6.0)

    def test_concatenate(self):
        A = None
        B = np.array([[1,2],[3,4]])
        np.testing.assert_equal(numpy_c_concatenate(A, B), B)
        A = np.array([[0], [1]])
        np.testing.assert_equal(numpy_c_concatenate(A, B), [[0,1,2], [1,3,4]])

    def test_save_prediction(self):
        model = RandomForestClassifier()
        model.id = get_model_id(model)
        model.fit(self.iris.data, self.iris.target)
        indexes = np.fromfunction(lambda x: x, (self.iris.data.shape[0], ), dtype=np.int32)
        saving_predict_proba(model, self.iris.data, indexes)
        os.remove('RandomForestClassifier_0_149.csv')

if __name__ == '__main__':
    unittest.main()
