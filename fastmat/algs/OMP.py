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
import numpy.linalg as npl

from ..base import Algorithm


def OMP(fmatA, arrY, numK):
    r"""Orthogonal Matching Pursuit

    **Definition and Interface**:
    For a given matrix :math:`A \in \mathbb{C}^{m \times N}` with
    :math:`m \ll N` and a vector :math:`b \in \mathbb{C}^m` we approximately
    solve

    .. math::
        \min\limits_{ x \in \mathbb{C}^N} \Vert x \Vert_0
        \quad\mathrm{s.t.}\quad A \cdot x =  x.

    If it holds that :math:`b =  A \cdot  x_0` for some :math:`k`-sparse
    :math:`x_0` and :math:`k` is low enough, we can recover :math:`x_0` via
    OMP [2]_.

    This type of problem as the one described above occurs in Compressed
    Sensing and Sparse Signal Recovery, where signals are approximated by
    sparse representations.

    >>> # import the packages
    >>> import numpy.linalg as npl
    >>> import numpy as np
    >>> import fastmat as fm
    >>> import fastmat.algs as fma
    >>> # define the dimensions
    >>> # and the sparsity
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
    >>> y = fma.OMP(C, b, k)
    >>> # test if they are close in the
    >>> # domain of C
    >>> print(npl.norm(C * y - b))

    We describe a sparse deconvolution problem, where the signal is in
    :math:`\mathbb{R}^{512}` and consists of :math:`3` windowed cosine pulses
    of the form :math:`c` with circulant displacement. Then we take the
    convolution and try to recover the location of the pulses using the OMP
    algorithm.

    .. note::
     The algorithm exploits two mathematical shortcuts. First it obviously
     uses the fast transform of the involved system matrix during the
     correlation step and second it uses a method to calculate the pseudo
     inverse after a rank-:math:`1` update of the matrix.

    .. todo::
      - optimize einsum-stuff

    Parameters
    ----------
    fmatA : fm.Matrix
        the system matrix
    arrB : np.ndarray
        the measurement vector
    numK : int
        the desired sparsity order

    Returns
    -------
    np.ndarray
        solution array
    """
    # Wrapper around the ISTA algrithm to allow processing of arrays of signals
    #     fmatA           - input system matrix
    #     arrY            - input data vector (measurements)
    #     numK            - specified sparsity order, i.e. number of iterations
    #                       to run
    #     numN,numM       - number of rows / columns of the system matrix
    #     numL            - number of problems to solve
    #     arrDiag         - array that contains column norms of fmatA
    #     numStrideSize   - size of strides during norm calculation
    #     numStrides      - number of whole strides to go through
    #     numPreSteps     - number of entries that do not fit in whoel strides
    #
    if len(arrY.shape) > 2:
        raise ValueError("Only n x m arrays are supported for OMP")

    # get the number of vectors to operate on
    numN, numM, numL = fmatA.numN, fmatA.numM, arrY.shape[1]

    fmatC = fmatA.normalized

    # determine return value data type
    returnType = np.promote_types(
        np.promote_types(fmatC.dtype, arrY.dtype), np.float32)

    # temporary array to store only support entries in
    arrXtmp     = np.zeros((numK, numL), dtype=returnType)

    # initital residual is the measurement
    arrResidual = arrY.astype(returnType, copy=True)

    # list containing the support
    arrSupport  = np.empty((numK, numL),       dtype='int')

    # matrix B that contains the pseudo inverse of A restricted to the support
    arrB        = np.zeros((numK, numN, numL), dtype=returnType)

    # A restricted to the support
    arrA        = np.zeros((numN, numK, numL), dtype=returnType)

    # different helper variables
    v2       = np.empty((numN, numL), dtype=returnType)
    v2n      = np.empty((numN, numL), dtype=returnType)
    v2y      = np.empty((numL),       dtype=returnType)
    newCols  = np.empty((numN, numL), dtype=returnType)
    arrC     = np.empty((numM, numL), dtype=returnType)
    newIndex = np.empty(numL,         dtype='int')

    # iterativly build up the solution
    for ii in range(numK):

        # do the normalized correlation step
        arrC = np.abs(fmatC.backward(arrResidual))

        # pick the maximum index in each correlation array
        newIndex = np.apply_along_axis(np.argmax, 0, arrC)

        # add these to the support
        arrSupport[ii, :] = newIndex

        # get the newly picked columns of A
        newCols = fmatA.getCols(newIndex)

        # store them into the submatrix
        arrA[:, ii, :] = newCols

        # in the first step everything is simple
        if ii == 0:
            v2  = newCols
            v2n = (v2 / npl.norm(v2, axis=0) ** 2).conj()

            v2y = np.einsum('ji,ji->i', v2n, arrY)

            arrXtmp[0, :] = v2y
            arrB[0, :, :] = v2n
        else:
            v1 = np.einsum('ijk,jk->ik', arrB[:ii, :, :], newCols)

            v2 = newCols - np.einsum('ijk,jk->ik', arrA[: , :ii, :], v1)
            v2n = (v2 / npl.norm(v2, axis=0) ** 2).conj()

            v2y = np.einsum('ji,ji->i', v2n, arrY)

            arrXtmp[:ii, :] -= v2y * v1
            arrXtmp[ii , :] += v2y

            arrB[:ii, :, :] -= np.einsum('ik,jk->jik', v2n, v1)
            arrB[ii, :, :] = v2n

        # update the residual
        arrResidual -= v2y * v2

    # return the computed vector
    arrX = np.zeros((numM, numL), dtype=returnType)
    arrX[arrSupport, np.arange(numL)] = arrXtmp

    return arrX


################################################################################
###  Maintenance and Documentation
################################################################################

################################################## inspection interface
class OMPinspect(Algorithm):

    @staticmethod
    def _getTest():
        from ..inspect import TEST, dynFormat, arrSparseTestDist
        from ..Hadamard import Hadamard
        from ..Matrix import Matrix

        def testOmp(test):
            # prepare vectors
            numM = test[TEST.NUM_M]
            test[TEST.RESULT_REF]       = np.hstack(
                [arrSparseTestDist((numM, 1), dtype=test[TEST.DATATYPE],
                                   density=1. * test['numK'] / numM).todense()
                 for nn in range(test[TEST.DATACOLS])])
            test[TEST.RESULT_INPUT]     = (test[TEST.INSTANCE] *
                                           test[TEST.RESULT_REF])
            test[TEST.RESULT_OUTPUT]    = OMP(test[TEST.INSTANCE],
                                              test[TEST.RESULT_INPUT],
                                              test['numK'])

        return {
            TEST.ALGORITHM: {
                'order'         : 5,
                TEST.NUM_N      : (lambda param: 2 ** param['order']),
                TEST.NUM_M      : TEST.NUM_N,
                'numK'          : 5,
                'typeA'         : TEST.Permutation(TEST.ALLTYPES),

                TEST.OBJECT     : Matrix,
                TEST.DATAALIGN  : TEST.ALIGNMENT.DONTCARE,
                TEST.INITARGS   : (lambda param: [
                    Hadamard(param.order).array.astype(param['typeA'])]),

                TEST.INIT_VARIANT : TEST.IgnoreFunc(testOmp),

                'strTypeA'      : (lambda param: TEST.TYPENAME[param['typeA']]),
                TEST.NAMINGARGS : dynFormat("Hadamard(%s,%s)",
                                            'order', 'strTypeA'),

                # matrix inversion always expands data type to floating-point
                TEST.TYPE_PROMOTION : np.float32
            },
        }

    @staticmethod
    def _getBenchmark():
        from ..inspect import BENCH, arrTestDist
        from ..Matrix import Matrix
        from ..Fourier import Fourier
        from ..Product import Product
        from scipy import sparse as sps

        def createTarget(M, datatype):
            '''Create test target for algorithm performance evaluation.'''

            if M < 10:
                raise ValueError("Problem size too small for OMP benchmark")

            # assume a 1:5 ratio of measurements and problem size
            # assume a sparsity of half the number of measurements
            N = int(np.round(M / 5.0))
            K = int(N / 2)

            # generate matA = [random measurement matrix] * [Fourier dictionary]
            matA = Product(Matrix(arrTestDist((N, M), datatype)), Fourier(M))

            # generate attb from random baseline support (RHS)
            arrB = matA.forward(
                sps.rand(M, 1, 1.0 * K / M).todense().astype(datatype))

            return (OMP, [matA, arrB, K])

        return {
            BENCH.COMMON: {
                BENCH.NAME      : 'OMP Algorithm',
                BENCH.DOCU      : r"""""",
                BENCH.FUNC_GEN  : (lambda c: createTarget(10 * c, np.float64)),
                BENCH.FUNC_SIZE : (lambda c: 10 * c)
            },
            BENCH.PERFORMANCE: {
                BENCH.CAPTION   : 'OMP performance'
            },
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, dt: createTarget(10 * c, dt)),
                BENCH.FUNC_SIZE : (lambda c: 10 * c),
                BENCH.FUNC_STEP : (lambda c: c * 10 ** (1. / 12)),
            }
        }

    @staticmethod
    def _getDocumentation():
        from ..inspect import DOC
        return ""
