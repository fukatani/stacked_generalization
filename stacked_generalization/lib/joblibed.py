import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone, RegressorMixin
from sklearn.externals import joblib
import stacked_generalization.lib.util as util
import os


class BaseJoblibed(BaseEstimator):
    """Base class for joblibed learner.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    def __init__(self,
                 estimator,
                 prefix,
                 skip_refit=True,
                 cache_dir='temp/'):
        self.estimator = estimator
        self.prefix = prefix
        self.estimator.id = 'j' + prefix
        self.skip_refit = skip_refit
        self.cache_dir = cache_dir
        if self.cache_dir and not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

    def fit(self, xs_train, y_train, index=None):
        dump_file = ""
        if index is not None:
            dump_file = util.get_cache_file(self.estimator.id,
                                            index,
                                            cache_dir=self.cache_dir,
                                            suffix='pkl')
        if self.skip_refit and os.path.isfile(dump_file):
            if index is not None:
                self.estimator = joblib.load(dump_file)
        else:
            self.estimator.fit(xs_train, y_train)
            if index is not None:
                joblib.dump(self.estimator, dump_file, compress=True)
        return self


class JoblibedClassifier(BaseJoblibed, ClassifierMixin):
    """A joblibed classifier.

    Parameters
    ----------
    estimator : cache target model.
    prefix : file prefix.

    """
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
        return util.saving_predict_proba(self.estimator,
                                         xs_test,
                                         index,
                                         self.cache_dir)

    def predict(self, X, index=None):
        """Predict class for X.

        The predicted class of an input sample is a vote by the JoblibedClassifier.

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

    def score(self, X, y, index=None, sample_weight=None):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y,
                              self.predict(X, index),
                              sample_weight=sample_weight)


class JoblibedRegressor(BaseJoblibed, RegressorMixin):
    """A joblibed regressor.

    Parameters
    ----------
    estimator : cache target model.
    prefix : file prefix.

    """
    def predict(self, xs_test, index=None):
        return util.saving_predict(self.estimator,
                                   xs_test,
                                   index,
                                   self.cache_dir)
