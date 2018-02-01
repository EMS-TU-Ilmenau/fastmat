# -*- coding: utf-8 -*-

# Copyright 2016 Sebastian Semper, Christoph Wagner
#     https://www.tu-ilmenau.de/it-ems/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from ..base import Algorithm


def _softThreshold(arrX, numAlpha):
    '''
    Do a soft Thresholding step.
      arrM         - positive part of arrX - numAlpha
      arrX         - vector to be thresholded
      numAlpha     - thresholding threshold
    '''
    arrM = np.maximum(np.abs(arrX) - numAlpha, 0)
    return np.multiply((arrM / (arrM + numAlpha)), arrX)


def FISTA(
        fmatA,
        arrB,
        numLambda=0.1,
        numMaxSteps=100
):
    r"""Fast Iterative Shrinking-Thresholding Algorithm (FISTA)

    **Definition and Interface**:
    For a given matrix :math:`A \in \mathbb{C}^{m \times N}` with
    :math:`m \ll N` and a vector :math:`b \in \mathbb{C}^m` we approximately
    solve

    .. math::
        \min\limits_{ x \in \mathbb{C}^N}\Vert{ A \cdot  x -  b}\Vert^2_2 +
        \lambda \cdot \Vert x \Vert_1,

    where :math:`\lambda > 0` is a regularization parameter to steer the
    trade-off between data fidelity and sparsity of the solution.

    >>> # import the packages
    >>> import numpy.linalg as npl
    >>> import numpy as np
    >>> import fastmat as fm
    >>> import fastmat.algs as fma
    >>> # define the dimensions and the sparsity
    >>> n, k = 512, 3
    >>> # define the sampling positions
    >>> t = np.linspace(0, 20 * np.pi, n)
    >>> # construct the convolution matrix
    >>> c = np.cos(2 * t)
    >>> C = fm.Circulant(c)
    >>> # create the ground truth
    >>> x = np.zeros(n)
    >>> x[npr.choice(range(n), k, replace=0)] = 1
    >>> b = C * x
    >>> # reconstruct it
    >>> y = fma.FISTA(C, b, 0.005, 1000)
    >>> # test if they are close in the
    >>> # domain of C
    >>> print(npl.norm(C * y - b))

    We solve a sparse deconvolution problem, where the atoms are harmonics
    windowed by a gaussian envelope. The ground truth :math:`x` is build out
    of three pulses at arbitrary locations.

    .. note::
        The proper choice of :math:`\lambda` is crucial for good perfomance
        of this algorithm, but this is not an easy task. Unfortunately we are
        not in the place here to give you a rule of thumb what to do, since it
        highly depends on the application at hand. Again, consult [1]_ for any
        further considerations of this matter.

    .. todo::
        - Todos for ISTA
        - Check if its working

    Parameters
    ----------
    fmatA : fm.Matrix
        the system matrix
    arrB : np.ndarray
        the measurement vector
    numLambda : float, optional
        the thresholding parameter; default is 0.1
    numMaxSteps : int, optional
        maximum number of steps; default is 100

    Returns
    -------
    np.ndarray
        solution array
    """

    # Wrapper around the FISTA algrithm to allow processing of arrays of signals
    #     fmatA         - input system matrix
    #     arrB          - input data vector (measurements)
    #     numLambda     - balancing parameter in optimization problem
    #                     between data fidelity and sparsity
    #     numMaxSteps   - maximum number of steps to run
    #     numL          - step size during the conjugate gradient step

    if len(arrB.shape) > 2:
        raise ValueError("Only n x m arrays are supported for FISTA")

    # calculate the largest singular value to get the right step size
    numL = 1.0 / (fmatA.largestSV ** 2)
    t = 1
    arrX = np.zeros(
        (fmatA.numM, arrB.shape[1]),
        dtype=np.promote_types(np.float32, arrB.dtype)
    )
    # initial arrY
    arrY = np.copy(arrX)
    # start iterating
    for numStep in range(numMaxSteps):
        arrXold = np.copy(arrX)
        # do the gradient step and threshold
        arrStep = arrY - numL * fmatA.backward(fmatA.forward(arrY) - arrB)

        arrX = _softThreshold(arrStep, numL * numLambda * 0.5)

        # update t
        tOld =t
        t = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        # update arrY
        arrY = arrX + ((tOld - 1) / t) * (arrX - arrXold)
    # return the unthresholded values for all non-zero support elements
    return np.where(arrX != 0, arrStep, arrX)


################################################################################
###  Maintenance and Documentation
################################################################################

################################################## inspection interface
class FISTAinspect(Algorithm):
    @staticmethod
    def _getTest():
        from ..inspect import TEST, dynFormat, arrSparseTestDist
        from ..core.types import _getTypeEps
        from ..Product import Product
        from ..Hadamard import Hadamard
        from ..Matrix import Matrix

        def testFISTA(test):
            # prepare vectors
            numM = test[TEST.NUM_M]
            test[TEST.RESULT_REF] = np.hstack(
                [arrSparseTestDist((numM, 1), dtype=test[TEST.DATATYPE],
                                   density=1. * test['numK'] / numM).todense()
                 for nn in range(test[TEST.DATACOLS])])
            test[TEST.RESULT_INPUT] = (test[TEST.INSTANCE] *
                                       test[TEST.RESULT_REF])
            test[TEST.RESULT_OUTPUT] = FISTA(
                test[TEST.INSTANCE], test[TEST.RESULT_INPUT],
                numLambda=test['lambda'], numMaxSteps=test['maxSteps'])

        return {
            TEST.ALGORITHM: {
                'order': 6,
                TEST.NUM_N: (lambda param: 3 * param['order']),
                TEST.NUM_M: (lambda param: 2 ** param['order']),
                'numK': 'order',
                'lambda': 10.,
                'maxSteps': 1000,
                'typeA': TEST.Permutation(TEST.ALLTYPES),

                TEST.OBJECT: Matrix,
                TEST.INITARGS: (lambda param: [
                    Product(Matrix(np.random.uniform(
                        -100, 100, (getattr(param, TEST.NUM_M),
                                    getattr(param, TEST.NUM_M))).astype(
                        param['typeA'])),
                        Hadamard(param.order),
                        typeExpansion=param['typeA']).array]),

                TEST.DATAALIGN: TEST.ALIGNMENT.DONTCARE,
                TEST.INIT_VARIANT: TEST.IgnoreFunc(testFISTA),

                'strTypeA': (lambda param: TEST.TYPENAME[param['typeA']]),
                TEST.NAMINGARGS: dynFormat("(%dx%d)*Hadamard(%s)[%s]",
                                           TEST.NUM_N, TEST.NUM_M,
                                           'order', 'strTypeA'),

                # matrix inversion always expands data type to floating-point
                TEST.TYPE_PROMOTION: np.float32,
                TEST.TOL_MINEPS: _getTypeEps(np.float32),
                TEST.TOL_POWER: 5.
                # TEST.CHECK_PROXIMITY    : False
            },
        }

    @staticmethod
    def _getBenchmark():
        from ..inspect import BENCH, arrTestDist
        from ..Matrix import Matrix
        from ..Product import Product
        from ..Fourier import Fourier
        from scipy import sparse as sps

        def createTarget(M, datatype):
            '''Create test target for algorithm performance evaluation.'''

            if M < 10:
                raise ValueError("Problem size too small for FISTA benchmark")

            # assume a 1:5 ratio of measurements and problem size
            # assume a sparsity of half the number of measurements
            N = int(np.round(M / 5.0))
            K = int(N / 2)

            # generate matA (random measurement matrix, Fourier dictionary)
            matA = Product(Matrix(arrTestDist((N, M), datatype)), Fourier(M))

            # generate arrB from random baseline support (RHS)
            arrB = matA * sps.rand(M, 1, 1.0 * K / M).todense().astype(datatype)

            return (FISTA, [matA, arrB])

        return {
            BENCH.COMMON: {
                BENCH.NAME: 'FISTA Algorithm',
                BENCH.FUNC_GEN: (lambda c: createTarget(10 * c, np.float64)),
                BENCH.FUNC_SIZE: (lambda c: 10 * c)
            },
            BENCH.PERFORMANCE: {
                BENCH.CAPTION: 'FISTA performance'
            },
            BENCH.DTYPES: {
                BENCH.FUNC_GEN: (lambda c, dt: createTarget(10 * c, dt)),
                BENCH.FUNC_SIZE: (lambda c: 10 * c),
                BENCH.FUNC_STEP: (lambda c: c * 10 ** (1. / 12)),
            }
        }

    @staticmethod
    def _getDocumentation():
        from ..inspect import DOC
        return ""
