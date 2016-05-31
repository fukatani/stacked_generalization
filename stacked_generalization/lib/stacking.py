import numpy as np
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from stacked_generalization.lib.util import numpy_c_concatenate
from stacked_generalization.lib.util import multiple_feature_weight
from sklearn.metrics import mean_squared_error
from collections import OrderedDict
from sklearn.preprocessing import LabelBinarizer
from sklearn.externals import joblib
import util
import os


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
                 verbose=0,
                 save_stage0=False,
                 save_dir=''):
        self.n_folds = n_folds
        self.clfs = clfs
        self.bclf = bclf
        self.stack_by_proba = stack_by_proba
        self.all_learner = OrderedDict()
        self.oob_score_flag = oob_score_flag
        self.verbose = verbose
        self.MyKfold = Kfold
        self.save_stage0 = save_stage0
        self.save_dir = save_dir
        for clf in clfs:
            if not hasattr(clf, 'id'):
                clf.id = self.save_dir + util.get_model_id(clf)

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
                self.all_learner[all_learner_key].append(now_learner)
                xs_now_train = xs_train[train_index]
                y_now_train = y_train[train_index]
                xs_cv = xs_train[cv_index]
                #y_cv = y_train[cv_index] no use

                if not hasattr(now_learner, 'id'):
                    now_learner.id = self.get_stage0_id(now_learner)

                if self.save_stage0 and self._is_saved(now_learner, cv_index):
                    print('Prediction cache exists: skip fitting.')
                    now_learner = joblib.load(self.save_dir + now_learner.id + '.pkl')
                    self.all_learner[all_learner_key][-1] = now_learner
                else:
                    self._out_to_console('Fold [{0}]'.format(i), 0)
                    now_learner.fit(xs_now_train, y_now_train)
                    if self.save_stage0:
                        joblib.dump(now_learner, self.save_dir + now_learner.id + '.pkl', compress=True)

                if blend_train_j is None:
                    blend_train_j = self._get_blend_init(y_train, now_learner)
                blend_train_j[cv_index] = self._get_child_predict(now_learner, xs_cv, cv_index)
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

    def predict_proba(self, xs_test, index=None):
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
        blend_test = self._make_blend_test(xs_test, index)
        blend_test = self._pre_propcess(blend_test)
        return self.bclf.predict_proba(blend_test)

    def _get_blend_init(self, y_train, clf):
        if self.stack_by_proba and hasattr(clf, 'predict_proba'):
            width = self.n_classes_ - 1
        elif hasattr(clf, 'predict'):
            width = self.n_classes_
        elif hasattr(clf, 'n_components'):
            width = clf.n_components
        else:
            raise Exception('Unimplemented for {0}'.format(type(clf)))
        return np.zeros((y_train.size, width))

    def _is_saved(self, model, index):
        model_id = self.get_stage0_id(model)
        return os.path.isfile("{0}_{1}_{2}.csv".format(model_id, min(index), max(index)))

    def _make_blend_test(self, xs_test, index=None):
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
                blend_test_j_temp = self._get_child_predict(clf, xs_test, index)
                if blend_test_j is None:
                    blend_test_j = blend_test_j_temp
                else:
                    blend_test_j += blend_test_j_temp
            blend_test_j = blend_test_j / len(clfs) #convert to mean
            blend_test = numpy_c_concatenate(blend_test, blend_test_j)
        return blend_test

    def _get_child_predict(self, clf, X, index=None):
        if self.stack_by_proba and hasattr(clf, 'predict_proba'):
            if self.save_stage0 and index is not None:
                proba = util.saving_predict_proba(clf, X, index)
            else:
                proba = clf.predict_proba(X)
            return proba[:, 1:]
        elif hasattr(clf, 'predict'):
            predict_result = clf.predict(X)
            lb = LabelBinarizer()
            lb.fit(predict_result)
            return lb.fit_transform(predict_result)
        else:
            return clf.fit_transform(X)

    def _make_kfold(self, Y):
        return self.MyKfold(Y, self.n_folds)

    def predict(self, X, index=None):
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
        proba = self.predict_proba(X, index)
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

    def get_stage0_id(self, model):
        return self.save_dir + util.get_model_id(model)

class StackedRegressor(StackedClassifier):
    def __init__(self,
                 bclf,
                 clfs,
                 n_folds=3,
                 oob_score_flag=False,
                 verbose=0,
                 save_stage0=False,
                 save_dir=''):
        self.n_folds = n_folds
        self.clfs = clfs
        self.bclf = bclf
        self.all_learner = OrderedDict()
        self.oob_score_flag = oob_score_flag
        self.verbose = verbose
        self.stack_by_proba = False
        self.save_stage0 = save_stage0
        self.save_dir = save_dir

    def predict(self, X, index=None):
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
        blend_test = self._make_blend_test(X, index)
        blend_test = self._pre_propcess(blend_test)
        return self.bclf.predict(blend_test)

    def _make_kfold(self, Y):
        return KFold(Y.size, self.n_folds)

    def calc_oob_score(self, blend_train, y_train, skf, metrics=mean_squared_error):
        """Compute out-of-bag score"""
        y_predict = np.zeros(y_train.shape)
        for train_index, cv_index in skf:
            self.bclf.fit(blend_train[train_index], y_train[train_index])
            y_predict[cv_index] = self.bclf.predict(blend_train[cv_index])
        self.oob_score_ = metrics(y_predict, y_train)
        self._out_to_console('oob_score: {0}'.format(self.oob_score_), 0)

    def _get_blend_init(self, y_train, clf):
        if hasattr(clf, 'predict'):
            width = 1
        elif hasattr(clf, 'n_components'):
            width = clf.n_components
        return np.zeros((y_train.size, width))

    def _get_child_predict(self, clf, X, index=None):
        if hasattr(clf, 'predict'):
            if self.save_stage0 and index is not None:
                predict_result = util.saving_predict(clf, X, index)
            else:
                predict_result = clf.predict(X)
            return predict_result.reshape(predict_result.size, 1)
        else:
            return clf.fit_transform(X)


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
                 feature=None,
                 save_stage0=False):
        super(FWLSClassifier, self).__init__(bclf,
                                            clfs,
                                            n_folds,
                                            stack_by_proba,
                                            verbose,
                                            save_stage0)
        self.feature = feature

    def _pre_propcess(self, X):
        X = multiple_feature_weight(X, self.feature)
        return X
