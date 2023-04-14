import unittest

import numpy as np
import fastmat as fm

class TestInterface(unittest.TestCase):

    def test_slicing(self):
        arr = np.random.randn(25, 35)
        instance = fm.Matrix(arr)

        idxR = np.arange(arr.shape[0])
        idxC = np.arange(arr.shape[1])
        # Test the individual single- and batch row- and column access methods
        np.testing.assert_array_equal(
            np.hstack([instance[:, cc].reshape((-1, 1)) for cc in idxC]), arr
        )
        np.testing.assert_array_equal(
            np.vstack([instance[rr, :].reshape((1, -1)) for rr in idxR]), arr
        )
        np.testing.assert_array_equal(
            np.hstack([instance[:, cc] for cc in np.split(idxC, 5)]), arr
        )
        np.testing.assert_array_equal(
            np.vstack([instance[rr, :] for rr in np.split(idxR, 5)]), arr
        )
        # Now fetch the whole array at once
        np.testing.assert_array_equal(
            instance[...], arr
        )
        # And repeat the same from above, but this time with cache in place
        np.testing.assert_array_equal(
            np.hstack([instance[:, cc].reshape((-1, 1)) for cc in idxC]), arr
        )
        np.testing.assert_array_equal(
            np.vstack([instance[rr, :].reshape((1, -1)) for rr in idxR]), arr
        )
        np.testing.assert_array_equal(
            np.hstack([instance[:, cc] for cc in np.split(idxC, 5)]), arr
        )
        np.testing.assert_array_equal(
            np.vstack([instance[rr, :] for rr in np.split(idxR, 5)]), arr
        )


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
        arr = np.random.randn(25, 35)
        instance = fm.Matrix(arr)

        instance, str(instance), repr(instance)


#    def test_deprecation_warnings(self):
#        self.