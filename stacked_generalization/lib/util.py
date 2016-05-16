from sklearn.cross_validation import StratifiedKFold
import numpy as np

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

def numpy_c_concatenate(A, B):
    if A is None:
        return B
    else:
        return np.c_[A, B]
