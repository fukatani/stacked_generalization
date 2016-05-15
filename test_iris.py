import unittest
from sklearn import datasets
from sklearn.utils.validation import check_random_state
from stacked_generalization import StackedClassifier, FWLSClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from sklearn.tree.tree import SPARSE_SPLITTERS

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
        clfs = [RandomForestClassifier(n_estimators=50, criterion = 'gini', random_state=1),
                ExtraTreesClassifier(n_estimators=50, criterion = 'gini', random_state=2),
                ExtraTreesClassifier(n_estimators=40, criterion = 'gini', random_state=3),
                GradientBoostingClassifier(n_estimators=25, random_state=1),
                GradientBoostingClassifier(n_estimators=40, random_state=1),
                RidgeClassifier(random_state=1)]
        sl = StackedClassifier(bclf, clfs, n_folds=3, verbose=0, stack_by_proba=True, oob_score_flag=True)
        sl.fit(self.iris.data, self.iris.target)
        score = sl.score(self.iris.data, self.iris.target)
        self.assertGreater(score, 0.8, "Failed with score = {0}".format(score))
        self.assertGreater(score, 0.8, "Failed with score = {0}".format(sl.oob_score_))


if __name__ == '__main__':
    unittest.main()
