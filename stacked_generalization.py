import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from util import TwoStageKFold


class StackedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 bclf,
                 clfs,
                 n_folds=3,
                 stack_by_proba=True,
                 oob_score_flag=False,
                 verbose=0):
        self.n_folds = n_folds
        self.clfs = clfs
        self.bclf = bclf
        self.stack_by_proba = stack_by_proba
        self.all_learner = {}
        self.oob_score_flag = oob_score_flag
        self.verbose = verbose

    def _iter_for_kfold(self, skf):
        if isinstance(skf, StratifiedKFold):
            for i, (train_index, cv_index) in enumerate(skf):
                yield i, (train_index, cv_index, None)
        elif isinstance(skf, TwoStageKFold):
            for i, (train_index, blend_index, test_index) in enumerate(skf):
                yield i, (train_index, blend_index, test_index)

    def _fit_child(self, skf, xs_train, y_train):
        blend_train = np.zeros((xs_train.shape[0], len(self.clfs)))
        blend_test = np.zeros((xs_train.shape[0], len(self.clfs)))
        for j, clf in enumerate(self.clfs):
            self._out_to_console('Training classifier [{0}]'.format(j), 0)
            all_learner_key = str(type(clf)) + str(j)
            self.all_learner[all_learner_key] = []
            for i, (train_index, cv_index, test_index) in self._iter_for_kfold(skf):
                now_learner = clone(clf)
                self._out_to_console('Fold [{0}]'.format(i), 0)

                xs_now_train = xs_train[train_index]
                y_now_train = y_train[train_index]
                xs_cv = xs_train[cv_index]
                y_cv = y_train[cv_index]

                now_learner.fit(xs_now_train, y_now_train)
                self.all_learner[all_learner_key].append(now_learner)
                # This output will be the basis for our blended classifier to train against,
                # which is also the output of our classifiers
                if self.stack_by_proba and hasattr(now_learner, 'predict_proba'):
                    blend_train[cv_index, j] = now_learner.predict_proba(xs_cv)[:, 1]
                else:
                    blend_train[cv_index, j] = now_learner.predict(xs_cv)
                if test_index is not None:
                    xs_test = xs_train[test_index]
                    if self.stack_by_proba and hasattr(now_learner, 'predict_proba'):
                        blend_test[test_index, j] = now_learner.predict_proba(xs_test)[:, 1]
                    else:
                        blend_test[test_index, j] = now_learner.predict(xs_test)
        return blend_train, blend_test

    def fit(self, xs_train, y_train):
        # Ready for cross validation
        skf = StratifiedKFold(y_train, self.n_folds)
        self._out_to_console('xs_train.shape = {0}'.format(xs_train.shape), 1)

        blend_train, _ = self._fit_child(skf, xs_train, y_train)

        #calc out of bugs score
        if self.oob_score_flag:
            self.calc_oob_score(blend_train, y_train, skf)

        # blending
        self._out_to_csv('blend_train', blend_train, 2)
        self._out_to_csv('y_train', y_train, 2)
        blend_train = self._pre_propcess(blend_train)
        self.bclf.fit(blend_train, y_train)

        self._out_to_console('xs_train.shape = {0}'.format(xs_train.shape), 1)
        self._out_to_console('blend_train.shape = {0}'.format(blend_train.shape), 1)

    def predict_proba(self, xs_test):
        blend_test = self._make_blendX(xs_test)
        blend_test = self._pre_propcess(blend_test)
        return self.bclf.predict_proba(blend_test)

    def _make_blendX(self, xs_test):
        blend_test = np.zeros((xs_test.shape[0], len(self.all_learner)))
        for j, clfs in enumerate(self.all_learner.values()):
            blend_test_j = np.zeros((xs_test.shape[0], self.n_folds))
            for i, clf in enumerate(clfs):
                if self.stack_by_proba and hasattr(clf, 'predict_proba'):
                    blend_test_j[:, i] = clf.predict_proba(xs_test)[:, 1]
                else:
                    blend_test_j[:, i] = clf.predict(xs_test)
            # Take the mean of the predictions of the cross validation set
            blend_test[:, j] = blend_test_j.mean(1)
        return blend_test

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is a vote by the StackedClassifier.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def score(self, xs_test, y_test):
        y_test_predict = self.predict(xs_test)
        self._out_to_csv('xs_test', xs_test, 2)
        self._out_to_csv('y_test', y_test, 2)
        self._out_to_csv('y_test_predict', y_test_predict, 2)
        return metrics.accuracy_score(y_test, y_test_predict)

    def calc_oob_score(self, blend_train, y_train, skf):
        scores = []
        for train_index, cv_index in skf:
            self.bclf.fit(blend_train[train_index], y_train[train_index])
            scores.append(self.bclf.score(blend_train[cv_index], y_train[cv_index]))
        self.oob_score = sum(scores) / len(scores)
        self._out_to_console('oob_score: {0}'.format(self.oob_score), 0)

    def two_stage_cv(self, xs_train, y_train):
        from util import TwoStageKFold
        tkf = TwoStageKFold(y_train.size, self.n_folds)
        blend_train, blend_test = self._fit_child(tkf, xs_train, y_train)
        self._out_to_console('blend_train.shape = {0}'.format(blend_train.shape), 1)
        self._out_to_console('blend_test.shape = {0}'.format(blend_test.shape), 1)

        score = 0
        for i, (_, blend_index, test_index) in self._iter_for_kfold(tkf):
            self.bclf.fit(blend_train[blend_index], y_train[blend_index])
            score += self.score(xs_train[test_index], y_train[test_index])
        score /= len(tkf)
        return score

    def _out_to_console(self, message, limit_verbose):
        if self.verbose > limit_verbose:
            print(message)

    def _out_to_csv(self, file_name, data, limit_verbose):
        import os
        file_name = 'data/{0}.csv'.format(file_name)
        if self.verbose > limit_verbose:
            while True:
                if os.path.isfile(file_name):
                    file_name.replace('.csv', '_.csv')
                else:
                    break
            np.savetxt(file_name, data, delimiter=",")

    def _pre_propcess(self, X):
        return X

class FWLSClassifier(StackedClassifier):
    def __init__(self,
                 bclf,
                 clfs,
                 n_folds=3,
                 stack_by_proba=True,
                 verbose=0,
                 feature=None):
        super(FWLSClassifier, self).__init__(bclf,
                                            clfs,
                                            n_folds,
                                            stack_by_proba,
                                            verbose)
        self.feature = feature

    def _pre_propcess(self, X):
        X = util.multiple_feature_weight(X, self.feature)
        return X

