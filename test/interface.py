import unittest

import numpy as np
import fastmat as fm

class TestInterface(unittest.TestCase):

    def test_operators(self):
        arrA = np.random.randn(25, 35)
        arrB = np.random.randn(25, 35)
        A, B  = fm.Matrix(arrA), fm.Matrix(arrB)

        np.testing.assert_array_equal((A + B)[...], arrA + arrB)
        np.testing.assert_array_equal((A.__add__(B))[...], arrA + arrB)
        np.testing.assert_array_equal((A.__radd__(B))[...], arrB + arrA)
        np.testing.assert_array_equal((A - B)[...], arrA - arrB)
        np.testing.assert_array_equal((A.__sub__(B))[...], arrA - arrB)
        np.testing.assert_array_equal((A.__rsub__(B))[...], arrB - arrA)

        np.testing.assert_array_equal((A * B.T)[...], arrA @ arrB.T)
        np.testing.assert_array_equal((A.__mul__(B.T))[...], arrA @ arrB.T)
        np.testing.assert_array_equal((A.T.__rmul__(B))[...], arrB @ arrA.T)
        np.testing.assert_array_equal((A * 2.)[...], arrA * 2.)
        np.testing.assert_array_equal((A.__mul__(2.))[...], arrA * 2.)
        np.testing.assert_array_equal((A.__rmul__(2.))[...], 2. * arrA)

        np.testing.assert_array_equal((A / 2.)[...], arrA / 2.)

        self.assertEqual(len(A), 0)
        self.assertEqual(len(A + B), 2)

        self.assertRaises(TypeError, lambda: A.__add__(2))
        self.assertRaises(TypeError, lambda: A.__radd__(2))
        self.assertRaises(TypeError, lambda: A.__sub__(2))
        self.assertRaises(TypeError, lambda: A.__rsub__(2))
        self.assertRaises(TypeError, lambda: A.__mul__('ab'))
        self.assertRaises(TypeError, lambda: A.__rmul__('cd'))

    def test_properties(self):
        # Prepare some low-rank matrix with known values/vectors,
        # yet full of randomness
        s = np.random.randn(7)
        u, v = np.random.randn(25, 7), np.random.randn(7, 25)
        u /= np.linalg.norm(u, axis=0)[np.newaxis, :]
        v /= np.linalg.norm(v, axis=1)[:, np.newaxis]
        instance = fm.Matrix(np.einsum('ij,j,jk -> ik', u, s, v))

        # Test for correctness of the largest SV/SVec/EV
        idx_largest_sv = np.argmax(np.abs(s))
#        self.assertEqual(instance.largestSV, s[idx_largest_sv])
        u1, v1 = instance.largestSingularVectors
        u2, v2 = instance.largestSingularVectors
        np.testing.assert_array_equal(u1, u2)
        np.testing.assert_array_equal(v1, v2)
#        np.testing.assert_array_equal(np.squeeze(u1), u[:, idx_largest_sv])
#        np.testing.assert_array_equal(np.squeeze(v1), v[idx_largest_sv, :])
        self.assertEqual(instance.largestEV, instance.largestEigenValue)
        self.assertEqual(instance.H.largestEV, instance.largestEigenValue)
        self.assertEqual(instance.T.largestEV, instance.largestEigenValue)
        self.assertEqual(instance.conj.largestEV, instance.largestEigenValue)
        np.testing.assert_array_equal(
            instance.largestEigenVec, instance.largestEigenVec
        )

        # Test the general Matrix class interface
        self.assertEqual(instance.gram, instance.gram)
        np.testing.assert_array_equal(instance.colNorms, instance.colNorms)
        np.testing.assert_array_equal(instance.rowNorms, instance.rowNorms)
        self.assertEqual(instance.colNormalized, instance.colNormalized)
        self.assertEqual(instance.normalized, instance.colNormalized)
        self.assertEqual(instance.rowNormalized, instance.rowNormalized)
        self.assertEqual(instance.inverse, instance.inverse)
        self.assertEqual(instance.pseudoInverse, instance.pseudoInverse)

        self.assertEqual(instance.complexity, instance.getComplexity())

        # Test the deprecated properties
        self.assertEqual(instance.numN, instance.numRows)
        self.assertEqual(instance.numN, instance.numRows)
        self.assertEqual(instance.numM, instance.numCols)
        self.assertEqual(instance.numM, instance.numCols)

        # Test each one of the deprecated properties twice (due to caching)
        self.assertEqual(instance.largestEV, instance.largestEigenValue)
        self.assertEqual(instance.largestEV, instance.largestEigenValue)
        self.assertEqual(instance.largestSV, instance.largestSingularValue)
        self.assertEqual(instance.largestSV, instance.largestSingularValue)
    
    def test_representation(self):
        arr = np.random.randn(20, 20)
        instance = fm.Matrix(arr)

        for ii in [
            instance, instance.H, instance.T, instance.conj,
            instance.inverse, instance.pseudoInverse,
            instance.profileForward, instance.profileBackward
        ]:
            self.assertTrue(isinstance(str(ii), str))
            self.assertTrue(isinstance(repr(ii), str))

    def test_slicing(self):
        d = 5
        n, m = 5 * d, 7 * d
        arrH, arrV = np.random.randn(n), np.random.randn(m)
        arr = np.outer(arrH, arrV)
        np.set_printoptions(edgeitems=100, linewidth=250)
        for inst in [fm.Matrix(arr), fm.Outer(arrH, arrV)]:
            idxR = np.arange(arr.shape[0])
            idxC = np.arange(arr.shape[1])

            # Test access methods for single- and batch rows and columns
            np.testing.assert_array_equal(
                np.hstack([inst[:, cc].reshape((-1, 1)) for cc in idxC]), arr
            )
            np.testing.assert_array_equal(
                np.vstack([inst[rr, :].reshape((1, -1)) for rr in idxR]), arr
            )
            np.testing.assert_array_equal(
                np.hstack([inst[:, cc:cc + d] for cc in idxC[::d]]), arr
            )
            np.testing.assert_array_equal(
                np.vstack([inst[rr:rr + d, :] for rr in idxR[::d]]), arr
            )
            np.testing.assert_array_equal(
                np.hstack([inst[..., cc:cc + d] for cc in idxC[::d]]), arr
            )
            np.testing.assert_array_equal(
                np.vstack([inst[rr:rr + d, ...] for rr in idxR[::d]]), arr
            )
            np.testing.assert_array_equal(
                np.hstack([
                    np.vstack([
                        inst[rr:rr + d, cc:cc + d] for rr in idxR[::d]
                    ]) for cc in idxC[::d]
                ]), arr
            )
            np.testing.assert_array_equal(
                np.hstack([inst[:, cc] for cc in np.split(idxC, 5)]), arr
            )
            np.testing.assert_array_equal(
                np.vstack([inst[rr, :] for rr in np.split(idxR, 5)]), arr
            )
            np.testing.assert_array_equal(
                np.hstack([inst[..., cc] for cc in np.split(idxC, 5)]), arr
            )
            np.testing.assert_array_equal(
                np.vstack([inst[rr, ...] for rr in np.split(idxR, 5)]), arr
            )

            # Now fetch the whole array at once
            np.testing.assert_array_equal(inst[...], arr
                                          )
            # And repeat from above, but this time with cache in place
            np.testing.assert_array_equal(
                np.hstack([inst[:, cc].reshape((-1, 1)) for cc in idxC]), arr
            )
            np.testing.assert_array_equal(
                np.vstack([inst[rr, :].reshape((1, -1)) for rr in idxR]), arr
            )
            np.testing.assert_array_equal(
                np.hstack([inst[:, cc] for cc in np.split(idxC, 5)]), arr
            )
            np.testing.assert_array_equal(
                np.vstack([inst[rr, :] for rr in np.split(idxR, 5)]), arr
            )

            self.assertRaises(IndexError, lambda: inst[:, :, :])
            self.assertRaises(IndexError, lambda: inst[:])
            self.assertRaises(IndexError, lambda: inst[0])
            self.assertRaises(IndexError, lambda: inst.getRow(-1))
            self.assertRaises(IndexError, lambda: inst.getRow(n))
            self.assertRaises(IndexError, lambda: inst.getCol(-1))
            self.assertRaises(IndexError, lambda: inst.getCol(m))

            self.assertRaises(TypeError, lambda: inst['a', 'b'])
            self.assertRaises(TypeError, lambda: inst['a', :])
            self.assertRaises(TypeError, lambda: inst[:, 'b'])
            self.assertRaises(TypeError, lambda: inst.getCol(np.zeros((2, 2))))
            self.assertRaises(TypeError, lambda: inst.getRow(np.zeros((2, 2))))
            for exception, dtype in [(IndexError, int), (TypeError, float)]:
                self.assertRaises(
                    exception, lambda: inst[:, np.zeros((2, 2), dtype=dtype)]
                )
                self.assertRaises(
                    exception, lambda: inst[np.zeros((2, 2), dtype=dtype), :]
                )

    def test_utilities(self):
        arr = np.random.randn(25, 35)
        instance = fm.Matrix(arr)

        # Test querying the memory footprint of objects
        numBytesReference = instance.nbytesReference
        numBytes = instance.nbytes

        # Test if auto-reload of getMemoryFootprint works
        # (needed if fastmat is not imported as a whole)
        try:
            global getMemoryFootprint
            _getMemoryFootprint = getMemoryFootprint
            getMemoryFootprint = None
            numBytesReference2 = instance.nbytesReference
            getMemoryFootprint = _getMemoryFootprint
            self.assert_equal(numBytesReference, numBytesReference2)
        except NameError:
            pass

