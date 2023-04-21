import unittest

import numpy as np
import fastmat as fm
import scipy.sparse as sps

class TestClass(unittest.TestCase):

    def testBlocks(self):
        self.assertRaises(ValueError, lambda: fm.Blocks(None))
        self.assertRaises(ValueError, lambda: fm.Blocks([]))
        self.assertRaises(TypeError, lambda: fm.Blocks([None, 1]))
        self.assertRaises(TypeError, lambda: fm.Blocks([fm.Eye(100)]))
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

    def test_BlockDiag(self):
        self.assertRaises(ValueError, lambda: fm.BlockDiag())
        self.assertRaises(
            ValueError, lambda: fm.BlockDiag(fm.Eye(10), None, fm.Eye(100))
        )

    def test_Circulant(self):
        dims = (11, 12)
        self.assertEqual(
            fm.Circulant(np.random.rand(*dims), verbose=True).tenC.shape, dims
        )
        self.assertRaises(ValueError, lambda: fm.Circulant(np.zeros(())))

    def test_Diag(self):
        self.assertRaises(ValueError, lambda: fm.Diag(np.zeros(())))
        self.assertRaises(ValueError, lambda: fm.Diag(np.zeros((2, 3))))
        self.assertRaises(ValueError, lambda: fm.Diag(np.zeros((2, 3, 4))))

    def test_Fourier(self):
        self.assertRaises(ValueError, lambda: fm.Fourier(0))

    def test_Hadamard(self):
        self.assertRaises(ValueError, lambda: fm.Hadamard(0))
        self.assertRaises(
            ValueError, lambda: (fm.Hadamard(31), fm.Hadamard(63))
        )

    def test_Kron(self):
        arrM = np.random.randn(25, 35)
        M = fm.Matrix(arrM)
        self.assertRaises(ValueError, lambda: fm.Kron(fm.Eye(10), M))
        self.assertRaises(ValueError, lambda: fm.Kron(fm.Eye(10)))

    def test_LFSRCirculant(self):
        self.assertRaises(ValueError, lambda: fm.LFSRCirculant(0, 1))
        self.assertRaises(ValueError, lambda: fm.LFSRCirculant(0x1891, 0))
        self.assertRaises(ValueError, lambda: fm.LFSRCirculant(0x1892, 1))
        L = fm.LFSRCirculant(0x1891, 0xFFF)
        L.size, L.taps, L.states

    def test_LowRank(self):
        self.assertRaises(ValueError, lambda: fm.LowRank(
            None, np.ones((10, 2)), np.ones((10, 2))
        ))
        self.assertRaises(ValueError, lambda: fm.LowRank(
            np.arange(2), None, np.ones((10, 2))
        ))
        self.assertRaises(ValueError, lambda: fm.LowRank(
            np.arange(2), np.ones((10, 2)), None
        ))
        self.assertRaises(ValueError, lambda: fm.LowRank(
            np.arange(1), np.ones((10, )), np.ones((10, ))
        ))
        self.assertRaises(ValueError, lambda: fm.LowRank(
            np.arange(4), np.ones((2, 3, 4)), np.zeros((4, 3))
        ))
        self.assertRaises(ValueError, lambda: fm.LowRank(
            np.arange(3), np.ones((3, 4)), np.zeros((3, 4))
        ))
        self.assertRaises(ValueError, lambda: fm.LowRank(
            np.arange(4), np.ones((3, 4)), np.zeros((4, 3))
        ))
        L = fm.LowRank(np.ones((2, )), np.ones((10, 2)), np.ones((10, 2)))
        L.vecS, L.arrU, L.arrV

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

        arrM = np.random.randn(25, 35) + 1j * np.random.randn(25, 35)
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
            repr(cc)
            self.assertRaises(TypeError, lambda: cc(None))
        
        self.assertRaises(ValueError, lambda: fm.Inverse(M))

    def test_Outer(self):
        self.assertRaises(
            ValueError, lambda: fm.Outer(np.arange(10), None)
        )
        self.assertRaises(
            ValueError, lambda: fm.Outer(None, np.arange(10))
        )
        self.assertRaises(
            ValueError, lambda: fm.Outer(np.arange(10), np.zeros((20, 30)))
        )
        self.assertRaises(
            ValueError, lambda: fm.Outer(np.zeros((20, 30)), np.arange(10))
        )

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
        for P in [
            fm.Partial(M, rows=idxRow, cols=None),
            fm.Partial(M, rows=None, cols=idxCol),
            fm.Partial(M, rows=None, cols=None)
        ]:
            P.rowNorms, P.colNorms

        # This test will be applied to the last iterated value, which should
        # have rows=None and cols=None
        np.testing.assert_array_equal(P[...], M[...])

    def test_Parametric(self):
        P = fm.Parametric(np.arange(10), np.arange(10), lambda x, y: x + y)
        P.vecX, P.vecY, P.fun
    
    def test_Permutation(self):
        for vecSigma in [np.arange(10) - 1, [1, 1, 3, 0, 4, 5, 6, 7, 8, 9]]:
            self.assertRaises(ValueError, lambda: fm.Permutation(vecSigma))

    def test_Polynomial(self):
        self.assertRaises(ValueError, lambda: fm.Polynomial(
            fm.Matrix(np.random.randn(25, 35)), [1, 2, 3]
        ))
        fm.Polynomial(fm.Fourier(10), [1, 2, 3]).coeff

    def test_Product(self):
        self.assertRaises(
            TypeError, lambda: fm.Product(fm.Eye(10), None, fm.Eye(10))
        )
        self.assertRaises(ValueError, lambda: fm.Product())
        self.assertRaises(
            ValueError, lambda: fm.Product(fm.Eye(10), fm.Eye(11), fm.Eye(10))
        )
        fm.Product(fm.Eye(10), fm.Eye(10), debug=True)

    def test_Sparse(self):
        self.assertRaises(TypeError, lambda: fm.Sparse(np.zeros((10, 11))))
        fm.Sparse(sps.diags(np.arange(10))).spArray

    def test_Sum(self):
        self.assertRaises(ValueError, lambda: fm.Sum())
        self.assertRaises(TypeError, lambda: fm.Sum(fm.Eye(10), 10))
        self.assertRaises(ValueError, lambda: fm.Sum(fm.Eye(10), fm.Eye(11)))
