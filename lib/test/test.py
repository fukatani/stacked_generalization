import unittest
from stacked_generalization.lib.util import numpy_c_concatenate
import numpy as np


class TestUtil(unittest.TestCase):

    def test_concatenate(self):
        A = None
        B = np.array([[1,2],[3,4]])
        np.testing.assert_equal(numpy_c_concatenate(A, B), B)
        A = np.array([[0], [1]])
        np.testing.assert_equal(numpy_c_concatenate(A, B), [[0,1,2], [1,3,4]])

if __name__ == '__main__':
    unittest.main()

