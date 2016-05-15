import unittest
from stacked_generalization.lib.util import TwoStageKFold, numpy_c_concatenate
import numpy as np


class TestUtil(unittest.TestCase):

    def test_twostage_Kfold(self):
        tsk = TwoStageKFold(8, n_folds=3)
        tsk_list = list(tsk)

        train_index, blend_index, test_index = tsk_list[0]
        np.testing.assert_equal(train_index, [3, 4, 5])
        np.testing.assert_equal(blend_index, [0, 1, 2])
        np.testing.assert_equal(test_index, [6, 7])

        train_index, blend_index, test_index = tsk_list[1]
        np.testing.assert_equal(train_index, [6, 7])
        np.testing.assert_equal(blend_index, [3, 4, 5])
        np.testing.assert_equal(test_index, [0, 1, 2])

        train_index, blend_index, test_index = tsk_list[2]
        np.testing.assert_equal(train_index, [0, 1, 2])
        np.testing.assert_equal(blend_index, [6, 7])
        np.testing.assert_equal(test_index, [3, 4, 5])

    def test_concatenate(self):
        A = None
        B = np.array([[1,2],[3,4]])
        np.testing.assert_equal(numpy_c_concatenate(A, B), B)
        A = np.array([[0], [1]])
        np.testing.assert_equal(numpy_c_concatenate(A, B), [[0,1,2], [1,3,4]])

if __name__ == '__main__':
    unittest.main()

