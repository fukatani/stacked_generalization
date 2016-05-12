from sklearn.cross_validation import KFold
import numpy as np

def write_result(pid, output, suffix=''):
    import csv
    import datetime
    suffix += datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    with open("predict_result_data_{0}.csv".format(suffix), "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["PassengerId", "Survived"])
        for pid, survived in zip(pid.astype(int), output.astype(int)):
            writer.writerow([pid, survived])

def multiple_feature_weight(self, blend, feature):
    assert blend.shape[0] == feature.shape[0]
    result = None
    for a_vec in blend.T:
        for b_vec in feature.T:
            if result is None:
                result = a_vec * b_vec
            else:
                result = np.c_[result, a_vec * b_vec]
    return result


class TwoStageKFold(KFold):

    def __init__(self, n, n_folds=3, shuffle=False,
                 random_state=None):
        super(KFold, self).__init__(n, n_folds, shuffle, random_state)
        self.idxs = np.arange(n)
        if shuffle:
            rng = check_random_state(self.random_state)
            rng.shuffle(self.idxs)

    def __iter__(self):
        ind = np.arange(self.n)
        for blend_mask, test_mask in self._iter_masks():
            train_mask = np.logical_not(np.logical_or(blend_mask, test_mask))
            train_index = ind[train_mask]
            blend_index = ind[blend_mask]
            test_index = ind[test_mask]
            yield train_index, blend_index, test_index

    # Since subclasses must implement either _iter_test_masks or
    # _iter_test_indices, neither can be abstract.
    def _iter_masks(self):
        """Generates boolean masks corresponding to test sets.

        By default, delegates to _iter_test_indices()
        """
        indices = list(self._iter_indices())
        for i in range(len(indices)):
            blend_mask = self._empty_mask()
            blend_mask[indices[i]] = True
            test_mask = self._empty_mask()
            test_mask[indices[i-1]] = True
            yield blend_mask, test_mask

    def _iter_indices(self):
        n = self.n
        n_folds = self.n_folds
        fold_sizes = (n // n_folds) * np.ones(n_folds, dtype=np.int)
        fold_sizes[:n % n_folds] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield self.idxs[start:stop]
            current = stop

    def __repr__(self):
        return '%s.%s(n=%i, n_folds=%i, shuffle=%s, random_state=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
            self.n_folds,
            self.shuffle,
            self.random_state,
        )

    def __len__(self):
        return self.n_folds
