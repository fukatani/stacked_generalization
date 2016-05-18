import numpy as np
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from stacked_generalization.lib.util import numpy_c_concatenate
from stacked_generalization.lib.util import multiple_feature_weight
from sklearn.metrics import r2_score
from collections import OrderedDict


class StackedClassifier(BaseEstimator, ClassifierMixin):
    """A stacking classifier.

    Parameters
    ----------
    bclf : stage1 model for stacking.

    clfs : list of stage0 machine learning models.

    n_folds : integer
     Number of folds at stage0 blending.

    stack_by_proba : boolean
        If True and stage0 machine learning model has 'predict_proba',
        result of it is used in blending.
        If not, result of 'predict' is used in blending.

    oob_score_flag : boolean
        If True, stacked clssfier calc out-of-bugs score after fitting.
        You can evaluate model by this score (with out CV).

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    .. [1] L. Breiman, "Stacked Regressions", Machine Learning, 24, 49-64 (1996).
    """
    def __init__(self,
                 bclf,
                 clfs,
                 n_folds=3,
                 stack_by_proba=True,
                 oob_score_flag=False,
                 Kfold=StratifiedKFold,
                 verbose=0):
        self.n_folds = n_folds
        self.clfs = clfs
        self.bclf = bclf
        self.stack_by_proba = stack_by_proba
        self.all_learner = OrderedDict()
        self.oob_score_flag = oob_score_flag
        self.verbose = verbose
        self.MyKfold = Kfold

    def _fit_child(self, skf, xs_train, y_train):
        """Build stage0 models from the training set (xs_train, y_train).

        Parameters
        ----------
        skf: StratifiedKFold-like iterator
            Use for cross validation blending.

        xs_train : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.

        y_train : array-like, shape = [n_samples]
            The target values (class labels in classification).

        Returns
        -------
        blend_train : array-like, shape = [n_samples]
            For stage1 model training.
        blend_test : array-like, shape = [n_samples]
            If you use TwoStageKFold, blended sample for test will be prepared.
        """
        blend_train = None
        blend_test = None
        for j, clf in enumerate(self.clfs):
            self._out_to_console('Training classifier [{0}]'.format(j), 0)
            all_learner_key = str(type(clf)) + str(j)
            self.all_learner[all_learner_key] = []
            blend_train_j = None
            for i, (train_index, cv_index) in enumerate(skf):
                now_learner = clone(clf)
                self._out_to_console('Fold [{0}]'.format(i), 0)

                xs_now_train = xs_train[train_index]
                y_now_train = y_train[train_index]
                xs_cv = xs_train[cv_index]
                #y_cv = y_train[cv_index] no use

                now_learner.fit(xs_now_train, y_now_train)
                self.all_learner[all_learner_key].append(now_learner)
                # This output will be the basis for our blended classifier to train against,
                # which is also the output of our classifiers
                if blend_train_j is None:
                    blend_train_j = self._get_blend_init(y_train, now_learner)
                blend_train_j[cv_index] = self._get_child_predict(now_learner, xs_cv)
            blend_train = numpy_c_concatenate(blend_train, blend_train_j)
        return blend_train, blend_test

    def fit(self, xs_train, y_train):
        """Build a stacked classfier from the training set (xs_train, y_train).

        Parameters
        ----------
        xs_train : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.

        y_train : array-like, shape = [n_samples]
            The target values (class labels in classification).

        Returns
        -------
        self : object
            Returns self.
        """
        self.n_classes_ = np.unique(y_train).shape[0]

        # Ready for cross validation
        skf = self._make_kfold(y_train)
        self._out_to_console('xs_train.shape = {0}'.format(xs_train.shape), 1)

        #fit stage0 models.
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

        return self

    def predict_proba(self, xs_test):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes].
            The class probabilities of the input samples.
        """
        blend_test = self._make_blend_test(xs_test)
        blend_test = self._pre_propcess(blend_test)
        return self.bclf.predict_proba(blend_test)

    def _get_blend_init(self, y_train, clf):
        if self.stack_by_proba and hasattr(clf, 'predict_proba'):
            width = self.n_classes_ - 1
        else:
            width = 1
        return np.zeros((y_train.size, width))

    def _make_blend_test(self, xs_test):
        """Make blend sample for test.

        Parameters
        ----------
        xs_test : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        blend_test : array of shape = [n_samples, n_stage0_models].
            Calc as the mean of the predictions of the cross validation set.
        """
        blend_test = None
        for j, clfs in enumerate(self.all_learner.values()):
            blend_test_j = None
            for i, clf in enumerate(clfs):
                blend_test_j_temp = self._get_child_predict(clf, xs_test)
                if blend_test_j is None:
                    blend_test_j = blend_test_j_temp
                else:
                    blend_test_j += blend_test_j_temp
            blend_test_j = blend_test_j / len(clfs) #convert to mean
            blend_test = numpy_c_concatenate(blend_test, blend_test_j)
        return blend_test

    def _get_child_predict(self, clf, X):
        if self.stack_by_proba and hasattr(clf, 'predict_proba'):
            return clf.predict_proba(X)[:, 1:]
        else:
            predict_result = clf.predict(X)
            return predict_result.reshape(predict_result.size, 1)

    def _make_kfold(self, Y):
        return self.MyKfold(Y, self.n_folds)

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

    def calc_oob_score(self, blend_train, y_train, skf):
        """Compute out-of-bag score"""
        scores = []
        for train_index, cv_index in skf:
            self.bclf.fit(blend_train[train_index], y_train[train_index])
            scores.append(self.bclf.score(blend_train[cv_index], y_train[cv_index]))
        self.oob_score_ = sum(scores) / len(scores)
        self._out_to_console('oob_score: {0}'.format(self.oob_score_), 0)

    def _out_to_console(self, message, limit_verbose):
        if self.verbose > limit_verbose:
            print(message)

    def _out_to_csv(self, file_name, data, limit_verbose):
        """write_out numpy array to csv"""
        import os
        file_name = 'data/{0}.csv'.format(file_name)
        if self.verbose > limit_verbose:
            while True:
                if os.path.isfile(file_name):
                    file_name = file_name.replace('.csv', '_.csv')
                else:
                    break
            np.savetxt(file_name, data, delimiter=",")

    def _pre_propcess(self, X):
        return X


class StackedRegressor(StackedClassifier):
    def __init__(self,
                 bclf,
                 clfs,
                 n_folds=3,
                 oob_score_flag=False,
                 verbose=0):
        self.n_folds = n_folds
        self.clfs = clfs
        self.bclf = bclf
        self.all_learner = OrderedDict()
        self.oob_score_flag = oob_score_flag
        self.verbose = verbose
        self.stack_by_proba = False

    def predict(self, X):
        """
        The predicted value of an input sample is a vote by the StackedRegressor.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted values.
        """
        blend_test = self._make_blend_test(X)
        blend_test = self._pre_propcess(blend_test)
        return self.bclf.predict(blend_test)

    def _make_kfold(self, Y):
        return KFold(Y.size, self.n_folds)

    def calc_oob_score(self, blend_train, y_train, skf):
        """Compute out-of-bag score"""
        y_predict = np.zeros(y_train.shape)
        for train_index, cv_index in skf:
            self.bclf.fit(blend_train[train_index], y_train[train_index])
            y_predict[cv_index] = self.bclf.predict(blend_train[cv_index])
        self.oob_score_ = r2_score(y_predict, y_train)
        self._out_to_console('oob_score: {0}'.format(self.oob_score_), 0)


class FWLSClassifier(StackedClassifier):
    """
    Feature Weighted Linear Stacking Classfier.
    References
    ----------

    .. [1] J. Sill1 et al, "Feature Weighted Linear Stacking", https://arxiv.org/abs/0911.0460, 2009.
    """
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
        X = multiple_feature_weight(X, self.feature)
        return X
