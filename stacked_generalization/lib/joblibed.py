import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.externals import joblib
import util
import os


class JoblibedClassifier(BaseEstimator, ClassifierMixin):
    """A joblibed classifier.

    Parameters
    ----------
    estimator : cache target model.
    prefix : file prefix.

    """
    def __init__(self,
                 estimator,
                 prefix,
                 skip_refit=True):
        self.estimator = estimator
        self.prefix = prefix
        self.estimator.id = 'j' + prefix
        self.skip_refit = skip_refit

    def fit(self, xs_train, y_train, index=None):
        dump_file = ""
        if index is not None:
            dump_file = "{0}_{1}_{2}.pkl".format(self.estimator.id, min(index), max(index))
        if self.skip_refit and os.path.isfile(dump_file):
            if index is not None:
                self.estimator = joblib.load(dump_file)
        else:
            self.estimator.fit(xs_train, y_train)
            if index is not None:
                joblib.dump(self.estimator, dump_file, compress=True)
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
        return util.saving_predict_proba(self.estimator, xs_test, index)

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
