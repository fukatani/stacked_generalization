import unittest
from kaggle_titanic_util import DataReader
from util import TwoStageKFold
import numpy as np


class TestTitanic(unittest.TestCase):
    """
    For execute this unittest, you have to download data from kaggle.
    """

    def setUp(self):
        self.dr = DataReader('train.csv')

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

if __name__ == '__main__':
    unittest.main()

