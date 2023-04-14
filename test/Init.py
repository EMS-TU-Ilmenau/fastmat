import unittest
import fastmat as fm
import numpy as np

class TestInit(unittest.TestCase):

    dims = (11, 12)

    def test_get_tenC(self):
        self.assertEqual(
            fm.Circulant(np.random.rand(*self.dims), verbose=True).tenC.shape,
            self.dims
        )
