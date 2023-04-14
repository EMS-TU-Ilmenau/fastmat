import unittest

import numpy as np
import fastmat as fm

class TestClass(unittest.TestCase):
    def test_Partial(self):
        arr = np.random.randn(25, 35)
        M = fm.Matrix(arr)
        idxRow = np.random.choice(arr.shape[0], 5)
        idxCol = np.random.choice(arr.shape[1], 5)
        
        # Check both the new and the deprecated interfaces
        P = fm.Partial(M, rows=idxRow, cols=idxCol)
        Pd = fm.Partial(M, N=idxRow, M=idxCol)
        np.testing.assert_array_equal(P[...], Pd[...])
        str(P), repr(P)

        # Test validity of returned index vectors
        np.testing.assert_array_equal(idxRow, P.rowSelection)
        np.testing.assert_array_equal(idxCol, P.colSelection)
        np.testing.assert_array_equal(idxRow, P.indicesN)
        np.testing.assert_array_equal(idxCol, P.indicesM)

        # Assert Exceptions
        self.assertRaises(
            TypeError, lambda: fm.Partial(None, rows=None, cols=None)
        )
        self.assertRaises(
            TypeError, lambda: fm.Partial(arr, rows=idxRow, cols=idxCol)
        )
        self.assertRaises(
            TypeError, lambda: fm.Partial(M, rows=2, cols=idxCol)
        )
        self.assertRaises(
            TypeError, lambda: fm.Partial(M, rows=idxRow, cols=4)
        )
        self.assertRaises(
            TypeError,
            lambda: fm.Partial(M, rows=idxRow.astype(float), cols=idxCol)
        )
        self.assertRaises(
            TypeError,
            lambda: fm.Partial(M, rows=idxRow, cols=idxCol.astype(float))
        )
        self.assertRaises(
            ValueError, lambda: fm.Partial(M, rows=idxRow - 15, cols=idxCol)
        )
        self.assertRaises(
            ValueError, lambda: fm.Partial(M, rows=idxRow + 15, cols=idxCol)
        )
        self.assertRaises(
            ValueError, lambda: fm.Partial(M, rows=idxRow, cols=idxCol - 15)
        )
        self.assertRaises(
            ValueError, lambda: fm.Partial(M, rows=idxRow, cols=idxCol + 15)
        )
        
        # Check retrieving row and column norms for partial Partials
        Pr = fm.Partial(M, rows=idxRow, cols=None)
        Pc = fm.Partial(M, rows=None, cols=idxCol)
        Pt = fm.Partial(M, rows=None, cols=None)
        Pr.colNorms, Pc.rowNorms, Pt.colNorms, Pt.rowNorms
        np.testing.assert_array_equal(Pt[...], M[...])
