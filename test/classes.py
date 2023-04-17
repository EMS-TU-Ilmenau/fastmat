import unittest

import numpy as np
import fastmat as fm
import scipy.sparse as sps

class TestClass(unittest.TestCase):

    def testBlocks(self):
        self.assertRaises(ValueError, lambda: fm.Blocks(None))
        self.assertRaises(ValueError, lambda: fm.Blocks([]))
        self.assertRaises(TypeError, lambda: fm.Blocks([None, 1]))
        self.assertRaises(TypeError, lambda: fm.Blocks([[1, 2], [2, 3, 4]]))
        self.assertRaises(
            ValueError, lambda: fm.Blocks([
                [fm.Eye(10), fm.Eye(10), fm.Eye(10)],
                [fm.Eye(10), fm.Eye(10)]
            ])
        )
        self.assertRaises(
            ValueError, lambda: fm.Blocks([
                [fm.Eye(10), fm.Eye(10)],
                [fm.Eye(11), fm.Eye(11)]
            ])
        )
        self.assertRaises(
            ValueError, lambda: fm.Blocks([
                [fm.Eye(10), fm.Eye(10)],
                [fm.Eye(10), fm.Eye(11)]
            ])
        )

    def test_LFSRCirculant(self):
        self.assertRaises(ValueError, lambda: fm.LFSRCirculant(0, 1))
        self.assertRaises(ValueError, lambda: fm.LFSRCirculant(0x1891, 0))
        self.assertRaises(ValueError, lambda: fm.LFSRCirculant(0x1892, 1))

        L = fm.LFSRCirculant(0x1891, 0xFFF)
        L.size, L.taps, L.states


    def test_Matrix(self):
        self.assertRaises(
            NotImplementedError, lambda: fm.Matrix(np.zeros(()))
        )
        self.assertRaises(
            NotImplementedError, lambda: fm.Matrix(np.zeros((10, )))
        )
        self.assertRaises(
            TypeError, lambda: fm.Matrix(None)
        )
        self.assertRaises(
            TypeError, lambda: fm.Matrix(sps.diags([1, 2, 3]))
        )

        arrM = np.random.randn(25, 35)
        M = fm.Matrix(arrM)
        self.assertRaises(TypeError, lambda: M + 1)
        self.assertRaises(TypeError, lambda: 1 + M)
        self.assertRaises(TypeError, lambda: M - 1)
        self.assertRaises(TypeError, lambda: 1 - M)
        self.assertRaises(TypeError, lambda: M * 'a')
        self.assertRaises(TypeError, lambda: 'a' * M)
        self.assertRaises(ZeroDivisionError, lambda: M / 0)
        self.assertRaises(ZeroDivisionError, lambda: M / 0.)
        self.assertRaises(TypeError, lambda: M / M)
        self.assertRaises(NotImplementedError, lambda: M // 2)

        self.assertRaises(ValueError, lambda: M * np.zeros(()))
        self.assertRaises(ValueError, lambda: M * np.random.randn(2, 3, 4))
        self.assertRaises(ValueError, lambda: M * np.random.randn(30, ))
        self.assertRaises(ValueError, lambda: M * np.random.randn(30, 30))

        # Test that the C-API interface is locked for pyhton-style instances
        self.assertRaises(
            NotImplementedError, lambda: M._forwardC(None, None, 0, 0)
        )
        self.assertRaises(
            NotImplementedError, lambda: M._backwardC(None, None, 0, 0)
        )

        # Test exceptions when creating operators from non-fastmat instances
        for cc in [
            fm.Hermitian, fm.Conjugate, fm.Transpose,
            fm.Inverse, fm.PseudoInverse
        ]:
            self.assertRaises(TypeError, lambda: cc(None))
        
        self.assertRaises(ValueError, lambda: fm.Inverse(M))

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
            ValueError, lambda: fm.Partial(M, rows=idxRow - 45, cols=idxCol)
        )
        self.assertRaises(
            ValueError, lambda: fm.Partial(M, rows=idxRow + 45, cols=idxCol)
        )
        self.assertRaises(
            ValueError, lambda: fm.Partial(M, rows=idxRow, cols=idxCol - 45)
        )
        self.assertRaises(
            ValueError, lambda: fm.Partial(M, rows=idxRow, cols=idxCol + 45)
        )
        
        # Check retrieving row and column norms for partial Partials
        Pr = fm.Partial(M, rows=idxRow, cols=None)
        Pc = fm.Partial(M, rows=None, cols=idxCol)
        Pt = fm.Partial(M, rows=None, cols=None)
        Pr.colNorms, Pc.rowNorms, Pt.colNorms, Pt.rowNorms
        np.testing.assert_array_equal(Pt[...], M[...])
