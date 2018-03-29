# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False

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
cimport numpy as np

from ..core.cmath cimport _arrEmpty, _arrZero, _arrForceTypeAlignment, _norm
from ..core.types cimport *
from ..base import Algorithm

from ..Matrix cimport Matrix


################################################################################
###  CG:
################################################################################
cpdef np.ndarray CG(
    Matrix fmatA,
    np.ndarray arrB,
    float eps=0
):
    r"""Conjugate Gradient Method

    **Definition and Interface**:
    For a given full rank Hermitian matrix :math:`A \in \mathbb{C}^n` and a
    vector :math:`b \in \mathbb{C}^n`, we solve

    .. math::
        A \cdot x = b

    for :math:`x \in \mathbb{C}^n`, i.e. :math:`x = A^{-1} \cdot b`. If
    :math:`A` is not Hermitian, we solve

    .. math::
        A^\mathrm{H} \cdot A \cdot x = A^\mathrm{H} \cdot b

    instead. In this case it should be noted, that the condition number of
    :math:`A^\mathrm{H} \cdot A` might be a lot larger than the one of
    :math:`A` an thus we might run into stability problems for large and already
    ill-conditioned systems.

    This algorithm was originally described in [3]_ and is applicable here,
    because it only uses the backward and forward projection of a matrix.

    >>> # import the packages
    >>> import numpy.random as npr
    >>> import numpy as np
    >>> import fastmat as fm
    >>> import fastmat.algs as fma
    >>> # construct the matrix
    >>> n = 26
    >>> H = fm.Hadamard(n)
    >>> # define the right hand side
    >>> b = npr.randn(2 ** n)
    >>> # solve the system
    >>> y = fma.CG(H, b)
    >>> # check if solution is correct
    >>> print(np.allclose(b, H.forward(y)))

    We construct a Hadamard matrix of order :math:`26`, which would consume
    \SI{4.5}{\peta\byte} of memory if we used \SI{1}{byte} integers to
    represent it and solve above system of linear equations.

    Parameters
    ----------
    fmatA : fm.Matrix
        the system matrix
    arrB : np.ndarray
        the right hand side of the system of equations
    eps : float, optional
        threshold for stopping the iteration; default is 0

    Returns
    -------
    np.ndarray
        solution array
    """

    if arrB.ndim > 2:
        raise ValueError("Only N x M arrays are supported for CG")

    cdef np.dtype typeOut = np.promote_types(
        np.float32, np.promote_types(fmatA.dtype, arrB.dtype))
    cdef ntype npTypeOut = typeOut.type_num
    arrIn = _arrForceTypeAlignment(arrB, npTypeOut, np.NPY_FORCECAST)
    if eps == 0:
        eps = getTypeEps(typeOut)

    # dispatch specialization of core routine according tOptor
    if typeOut == np.float32:
        return _CGcore[np.float32_t](fmatA, arrIn, npTypeOut, 0., eps)
    elif typeOut == np.float64:
        return _CGcore[np.float64_t](fmatA, arrIn, npTypeOut, 0., eps)
    elif typeOut == np.complex64:
        return _CGcore[np.complex64_t](fmatA, arrIn, npTypeOut, 0., eps)
    elif typeOut == np.complex128:
        return \
            _CGcore[np.complex128_t](fmatA, arrIn, npTypeOut, 0., eps)
    else:
        raise NotImplementedError("Output type %d not supported." % (typeOut))


cdef np.ndarray _CGcore(
    Matrix fmatA,
    np.ndarray arrB,
    ntype npTypeOut,
    TYPE_FLOAT typeTag,
    float eps
):

    # Solve linear equation system 'fmatA * x = arrB' for x.
    # The following variables are used:
    # arrB         - input data array
    # fmatA        - input system matrix
    # arrR         - residual vector (TYPE_FLOAT, CONT)
    # vecR         - residual vector (TYPE_FLOAT[:] in arrR)
    # arrP         - next krylov subspace vector (TYPE_FLOAT[:N], CONT)
    # vecP         - next krylov subspace vector (TYPE_FLOAT[:] in arrP)
    # vecQ         - projection of krylov subspace vector onto cols of matA
    # arrIn        - symmetrized right-hand side [TYPE_FLOAT, F, CONT]
    # arrOut       - current solution [TYPE_FLOAT, F, CONT]
    # vecOut       - vector of current solution (TYPE_FLOAT[:] in arrOut)
    # numAlpha     - optimal step with
    # numRNormNew  - new residual norm
    # numRNormOld  - old residual norm
    # eps          - stopping condition to the projected residual

    # NOTE: typeTag is used for telling the compiler the used specialization
    # fetch dimensions of arrB
    cdef intsize mm, nn, numStep
    cdef intsize N = arrB.shape[0]
    cdef intsize M = arrB.shape[1] if arrB.ndim > 1 else 1

    # allocate temporary variables used in vector loop
    cdef np.float64_t numRNormOld, numRNormNew, ratio
    cdef TYPE_FLOAT numValue, numCorr, numAlpha

    # allocate temporary buffers in memory and create memoryviews into them
    cdef np.ndarray arrOut, arrP = _arrEmpty(1, N, 1, npTypeOut)
    cdef TYPE_FLOAT * vecP = <TYPE_FLOAT * > arrP.data
    cdef np.ndarray arrQ
    cdef TYPE_FLOAT * vecQ

    # change right hand side of equation system according to symmetrization
    # force to be F-contiguous and of consistent data type (no ints here)
    cdef np.ndarray arrR = _arrForceTypeAlignment(
        fmatA.backward(arrB), npTypeOut, np.NPY_FORCECAST)

    # as projections may just return the input vector unchanged, force arrR to
    # be an independent of the input (as we intent to change it)
    if id(arrR) == id(arrB):
        arrR = arrR.copy()

    cdef TYPE_FLOAT * pArrR = <TYPE_FLOAT * > arrR.data
    cdef TYPE_FLOAT * vecR
    arrOut = _arrZero(2, fmatA.numM, M, npTypeOut)
    cdef TYPE_FLOAT * pArrOut = <TYPE_FLOAT * > arrOut.data
    cdef TYPE_FLOAT * vecOut

    # iterate over vectors
    for mm in range(M):

        vecOut = &(pArrOut[mm * N])
        vecR   = &(pArrR[mm * N])

        # norm of residual
        numRNormOld = _norm(vecR, N)

        # projected residual (initialized with RHS-reformulation),
        # use to bootstrap krylov subspace vector by copy
        for nn in range(N):
            vecP[nn] = vecR[nn]

        # step counter
        numStep = 0

        # iterate until stopping criterion is met
        while numRNormOld > eps:
            arrQ = fmatA.gram.forward(arrP)
            vecQ = <TYPE_FLOAT * > arrQ.data

            # calculate next optimal step width according to
            # current residual and correlation between old and
            # new projection: correlate np.conj(arrP) (*) arrQ
            numCorr = 0
            for nn in range(N):
                numValue = vecP[nn]
                # conjugate elements of P on-the-fly
                if (TYPE_FLOAT == np.complex64_t) or \
                        (TYPE_FLOAT == np.complex128_t):
                    numValue.imag = -numValue.imag

                numCorr += numValue * vecQ[nn]

            numAlpha = numRNormOld / numCorr

            # update current solution and residual
            # here we use the orthogonality of arrP and arrQ
            # with respect to the inner product induced by A
            # while iterating, calculate norm of vecR on-the-fly
            for nn in range(N):
                vecOut[nn] = vecOut[nn] + numAlpha * vecP[nn]
                vecR[nn] = vecR[nn] - numAlpha * vecQ[nn]

            # new residual norm
            numRNormNew = _norm(vecR, N)

            # next krylov subspace vector
            value = numRNormNew / numRNormOld
            for nn in range(N):
                vecP[nn] = vecP[nn] * value + vecR[nn]

            numRNormOld = numRNormNew

            # step increment
            numStep += 1

    return arrOut


################################################################################
###  Maintenance and Documentation
################################################################################

################################################## inspection interface
class CGinspect(Algorithm):

    def _getTest(self):
        from ..inspect import TEST, dynFormat, arrTestDist
        from ..Eye import Eye

        def testCG(test):

            # prepare vectors
            test[TEST.RESULT_REF]    = arrTestDist((test[TEST.NUM_M],
                                                    test[TEST.DATACOLS]),
                                                   dtype=test[TEST.DATATYPE])
            test[TEST.RESULT_INPUT]  = (test[TEST.INSTANCE] *
                                        test[TEST.RESULT_REF])
            test[TEST.RESULT_OUTPUT] = CG(test[TEST.INSTANCE],
                                          test[TEST.RESULT_INPUT])

        return {
            TEST.ALGORITHM: {
                TEST.NUM_N      : 27,
                TEST.NUM_M      : TEST.NUM_N,

                'typeA'         : TEST.Permutation(TEST.ALLTYPES),
                'arrA'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'typeA',
                    TEST.SHAPE  : (TEST.NUM_N, TEST.NUM_N)
                }),

                TEST.OBJECT     : TEST.Permutation([Matrix, Eye]),
                TEST.INITARGS   : (lambda param:
                    [param.arrA()] if param[TEST.OBJECT] is Matrix
                    else [param[TEST.NUM_N]]
                ),
                TEST.DATAALIGN  : TEST.ALIGNMENT.DONTCARE,
                TEST.INIT_VARIANT : TEST.IgnoreFunc(testCG),

                'strType'       : (lambda param: param[TEST.OBJECT].__name__),
                'strTypeA'      : (lambda param: TEST.TYPENAME[param['typeA']]),
                TEST.NAMINGARGS : dynFormat("%s,%s", 'strType', 'strTypeA'),

                # matrix inversion always expands data type to floating-point
                TEST.TYPE_PROMOTION : np.float32,
                #TEST.CHECK_PROXIMITY : False,
                TEST.TOL_POWER  : 7.
            },
        }

    def _getBenchmark(self):
        from ..inspect import BENCH, arrTestDist
        from ..Diag import Diag

        def createTarget(M, datatype):
            '''Create test target for algorithm performance evaluation.'''

            # generate matA (random diagonal matrix)
            matA = Diag(arrTestDist((M, 1), datatype))

            # generate arrb from random baseline support and matrix (RHS)
            arrB = matA.forward(arrTestDist((M, 1), datatype))

            return (CG, [matA, arrB])

        return {
            BENCH.COMMON: {
                BENCH.NAME      : 'Method of Conjugate Gradients',
                BENCH.DOCU      : r'`A = \diag(\{1,\dots,n\})`',
                BENCH.FUNC_GEN  : (lambda c: createTarget(c, np.double))
            },
            BENCH.PERFORMANCE: {
                BENCH.CAPTION   : 'CG performance'
            },
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : createTarget,
                BENCH.FUNC_SIZE : (lambda c: c),
                BENCH.FUNC_STEP : (lambda c: c * 10 ** (1. / 12)),
            }
        }

    def _getDocumentation(self):
        from ..inspect import DOC
        return ""
