import unittest

import numpy as np
import fastmat as fm
import fastmat.algorithms as fma

class TestAlgorithm(unittest.TestCase):
    def test_Baseclass(self):
        # Construct a simple CS problem to test for correct support retrieval
        numN, numM, numK = 100, 40, 6
        A = fm.Matrix(np.random.randn(numM, numN))
        alg = fma.OMP(A, numMaxSteps=numK)
        idx_support = np.random.choice(numN, numK, replace=False)
        x = np.zeros((numN, ))
        x[idx_support] = np.random.randn(numK)
        b = A * x

        # Setup tracing the steps of the algorithm
        def printResult(instance):
            print(instance)

        alg.cbTrace = fma.Algorithm.snapshot
        alg.cbResult = printResult

        # Run the setting and query the collected trace, once before and after
        alg.trace
        y = alg.process(b)
        alg.trace

        # Try clearing the trace
        self.assertRaises(TypeError, lambda: setattr(alg, 'trace', None))
        self.assertRaises(TypeError, lambda: setattr(alg, 'trace', ()))
        self.assertRaises(TypeError, lambda: setattr(alg, 'trace', {}))
        self.assertRaises(TypeError, lambda: setattr(alg, 'trace', 1))
        alg.trace = []

        # Try setting some gibberish and some legit parameters
        self.assertRaises(
            AttributeError, lambda: alg.updateParameters(numBla=123)
        )
        alg.updateParameters(numMaxSteps=numK)

        np.testing.assert_array_equal(
            np.sort(np.squeeze(np.nonzero(np.squeeze(y)))),
            np.sort(np.squeeze(idx_support))
        )
        np.testing.assert_allclose(
            np.squeeze(y), np.squeeze(x), rtol=1e-12, atol=1e-15
        )

        # Check that the abstract base class is protected
        self.assertRaises(NotImplementedError, fma.Algorithm)

        # retrieve the memory footprint of the algorithm
        alg.nbytes


    def test_OMP(self):
        numN, numM, numB = 100, 40, 7
        arrM = np.random.randn(numM, numN)
        self.assertRaises(TypeError, lambda: fma.OMP(arrM))
        self.assertRaises(TypeError, lambda: fma.OMP(None))

        alg = fma.OMP(fm.Matrix(arrM), numMaxSteps=7)
        arrB = np.random.randn(numM, )
        alg.process(arrB)
        alg.process(np.random.randn(numM, numB))
        self.assertRaises(
            ValueError, lambda: alg.process(np.random.randn(numM, numM, numB))
        )
        self.assertRaises(ValueError, lambda: alg.process(np.zeros(())))

        alg.updateParameters(numMaxSteps=0)
        self.assertRaises(ValueError, lambda: alg.process(arrB))
        alg.updateParameters(numMaxSteps=-123)
        self.assertRaises(ValueError, lambda: alg.process(arrB))
