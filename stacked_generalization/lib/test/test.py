import unittest
import os

from sklearn import datasets
from sklearn.utils.validation import check_random_state
from stacked_generalization.lib.stacking import StackedClassifier, FWLSClassifier
from stacked_generalization.lib.stacking import StackedRegressor, FWLSRegressor
from stacked_generalization.lib.joblibed import JoblibedClassifier, JoblibedRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score
from sklearn.utils.testing import assert_less
import numpy as np
from stacked_generalization.lib.util import numpy_c_concatenate
from stacked_generalization.lib.util import saving_predict_proba
from stacked_generalization.lib.util import get_model_id
from stacked_generalization.lib.util import multiple_feature_weight
from sklearn.cross_validation import StratifiedKFold
from numpy.testing import assert_allclose
import glob


class TestStackedClassfier(unittest.TestCase):
    def setUp(self):
        iris = datasets.load_iris()
        rng = check_random_state(0)
        perm = rng.permutation(iris.target.size)
        iris.data = iris.data[perm]
        iris.target = iris.target[perm]
        self.iris = iris

    def test_stacked_classfier_extkfold(self):
        bclf = LogisticRegression(random_state=1)
        clfs = [RandomForestClassifier(n_estimators=40, criterion = 'gini', random_state=1),
                RidgeClassifier(random_state=1),
                ]
        sl = StackedClassifier(bclf,
                               clfs,
                               n_folds=3,
                               verbose=0,
                               Kfold=StratifiedKFold(self.iris.target, 3),
                               stack_by_proba=False,
                               oob_score_flag=True,
                               oob_metrics=log_loss)
        sl.fit(self.iris.data, self.iris.target)
        score = sl.score(self.iris.data, self.iris.target)
        self.assertGreater(score, 0.9, "Failed with score = {0}".format(score))

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

        for csv_file in glob.glob("*.csv"):
            os.remove(csv_file)
        for csv_file in glob.glob("*.pkl"):
            os.remove(csv_file)

        sl = StackedClassifier(bclf,
                               clfs,
                               oob_score_flag=True,
                               save_stage0=True)
        sl.fit(self.iris.data, self.iris.target)
        sl.score(self.iris.data, self.iris.target)
        self.assertGreater(score, 0.8, "Failed with score = {0}".format(score))
        sl.fit(self.iris.data, self.iris.target)
        sl.score(self.iris.data, self.iris.target)
        self.assertGreater(score, 0.8, "Failed with score = {0}".format(score))

        self.assertTrue(glob.glob('ExtraTreesClassifier_*.csv'))
        for csv_file in glob.glob("*.csv"):
            os.remove(csv_file)
        for csv_file in glob.glob("*.pkl"):
            os.remove(csv_file)

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
        any_file_removed = False
        for filename in os.listdir('.'):
            if filename.startswith('RandomForestClassifier'):
                os.remove(filename)
                any_file_removed = True
        self.assertTrue(any_file_removed)


    def test_fwls_classfier(self):
        feature_func = lambda x: np.ones(x.shape)
        bclf = LogisticRegression(random_state=1)
        clfs = [RandomForestClassifier(n_estimators=40, criterion = 'gini', random_state=1),
                RidgeClassifier(random_state=1),
                ]
        sl = FWLSClassifier(bclf,
                            clfs,
                            feature_func=feature_func,
                            n_folds=3,
                            verbose=0,
                            Kfold=StratifiedKFold(self.iris.target, 3),
                            stack_by_proba=False)
        sl.fit(self.iris.data, self.iris.target)
        score = sl.score(self.iris.data, self.iris.target)
        self.assertGreater(score, 0.9, "Failed with score = {0}".format(score))

    def test_fwls_regressor(self):
        feature_func = lambda x: np.ones(x.shape)
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

        sr = FWLSRegressor(bclf,
                              clfs,
                              feature_func,
                              n_folds=3,
                              verbose=0,
                              oob_score_flag=True)
        sr.fit(X_train, y_train)
        mse = mean_squared_error(y_test, sr.predict(X_test))
        assert_less(mse, 6.0)

    def test_multiple_feature_weight(self):
        A = np.array([[1,2],[3,4],[5,6]])
        B = np.array([[1],[1],[1]])
        C = multiple_feature_weight(A, B)
        np.testing.assert_equal(C, A)
        B = np.array([[2],[2],[2]])
        C = multiple_feature_weight(A, B)
        np.testing.assert_equal(C, np.array([[2,4],[6,8],[10,12]]))
        B = np.array([[1,2],[2,1],[1,2]])
        C = multiple_feature_weight(A, B)
        np.testing.assert_equal(C, np.array([[ 1,  2,  2,  4],
                                             [ 6,  3,  8,  4],
                                             [ 5, 10,  6, 12]]))

class TestJoblibedClassfier(unittest.TestCase):
    def setUp(self):
        iris = datasets.load_iris()
        rng = check_random_state(0)
        iris.data = iris.data
        iris.target = iris.target
        self.iris = iris
        for csv_file in glob.glob("*.csv"):
            os.remove(csv_file)

    def test_classifier(self):
        index = [i for i in range(len(self.iris.data))]

        rf = RandomForestClassifier()
        jrf = JoblibedClassifier(rf, "rf", cache_dir='')
        jrf.fit(self.iris.data, self.iris.target, index)
        prediction = jrf.predict(self.iris.data, index)
        score = accuracy_score(self.iris.target, prediction)
        self.assertGreater(score, 0.9, "Failed with score = {0}".format(score))

        rf = RandomForestClassifier(n_estimators=20)
        jrf = JoblibedClassifier(rf, "rf", cache_dir='')
        jrf.fit(self.iris.data, self.iris.target)
        index = [i for i in range(len(self.iris.data))]
        prediction2 = jrf.predict(self.iris.data, index)
        self.assertTrue((prediction == prediction2).all())

    def test_regressor(self):
        X, y = datasets.make_friedman1(n_samples=1200,
                                       random_state=1,
                                       noise=1.0)
        X_train, y_train = X[:200], y[:200]
        index = [i for i in range(200)]

        rf = RandomForestRegressor()
        jrf = JoblibedRegressor(rf, "rfr", cache_dir='')
        jrf.fit(X_train, y_train, index)
        prediction = jrf.predict(X_train, index)
        mse = mean_squared_error(y_train, prediction)
        assert_less(mse, 6.0)

        rf = RandomForestRegressor(n_estimators=20)
        jrf = JoblibedRegressor(rf, "rfr", cache_dir='')
        jrf.fit(X_train, y_train, index)
        prediction2 = jrf.predict(X_train, index)
        assert_allclose(prediction, prediction2)


if __name__ == '__main__':
    unittest.main()
