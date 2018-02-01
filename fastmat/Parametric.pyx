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

from .Matrix cimport Matrix
from .core.types cimport *
from .core.cmath cimport _dotSingleRow, _conjugateInplace, _arrEmpty

cdef class Parametric(Matrix):
    r"""

    Let :math:`f \D \mathbb{C}^2 \rightarrow \mathbb{C}` be any function and
    two vectors :math:`x \in \mathbb{C}^m` and :math:`y \in \mathbb{C}^n` such
    that :math:`(x_j,y_i) \in \Dom(f)` for :math:`i \in [n]` and
    :math:`j    \in [m]`. Then the matrix :math:`F \in \mathbb{C}^{n \times m}`
    is defined as

    .. math::
        F_{i,j} = f(x_j,y_i).

    This class is not designed to be super fast, but memory efficient. This
    means, that everytime the forward or backward projections are called, the
    elements are generated according to the specified function on the fly.

    .. note::
        For small dimensions, where the matrix fits into memory, it is
        definately more efficient to cast the matrix to a regular Matrix
        object.

    >>> # import the package
    >>> import fastmat as fm
    >>>
    >>> # define parameter
    >>> # function for the elements
    >>> def f(x, y):
    >>>     return x ** 2 - y ** 2
    >>>
    >>> # define the input array
    >>> # for the function f
    >>> x = np.linspace(1, 4, 4)
    >>>
    >>> # construct the transform
    >>> F = fm.Parametric(x, x, f)

    This yields

    .. math::
        f : \mathbb{C} \rightarrow \mathbb{C}

    .. math::
        (x_1,x_2)^\mathrm{T} \mapsto x_1^2 - x_2^2

    .. math::
        x = (1,2,3,4)^\mathrm{T}

    .. math::
        F = \begin{bmatrix}
        1 &   3 &  8 & 15 \\
        -3 &   0 &  5 & 12 \\
        -8 &  -5 &  0 &  7 \\
        -15 & -12 & -7 &  0
        \end{bmatrix}

    We used Cython [3]_ to get an efficient implementation in order to reduce
    computation time. Moreover, it is generally assumed the the defined
    function is able to use row and column broadcasting during evaluation.
    If this is not the case, one has to set the flag ``rangeAccess`` to
    ``False``.
    """

    property vecY:
        r"""Return the support vector in Y dimension.

        *(read only)*
        """

        def __get__(self):
            return self._vecY

    property vecX:
        r"""Return the support vector in X dimension.

        *(read only)*
        """

        def __get__(self):
            return self._vecX

    property fun:
        r"""Return the parameterizing function

        *(read only)*
        """

        def __get__(self):
            return self._fun

    def __init__(
        self,
        vecX,
        vecY,
        funF,
        funDtype=None,
        rangeAccess=True
    ):
        '''Initialize Matrix instance'''

        # store flags
        self._rangeAccess = rangeAccess

        # store support vectors, determine dtypes, cast to common dtype
        vecDtype = np.promote_types(vecX.dtype, vecY.dtype)
        self._vecY = vecY.astype(vecDtype)
        self._vecX = vecX.astype(vecDtype)

        # store element function and determine its data type
        # if funDataType is not passed by user, data type is determined
        # upon the function output with the first entries
        self._fun = funF
        self._funDtype = type(funF(self._vecX[0], self._vecY[0])) \
            if funDtype is None else funDtype

        # set properties of matrix
        self._initProperties(
            len(self._vecY),            # numN
            len(self._vecX),            # numM
            self._funDtype,             # data type of matrix
            cythonCall=True,
            forceInputAlignment=True,
            bypassAutoArray=False       # deactivate automatic generation of
                                        # array for transformation bypass. As
                                        # Parametric is always slower than dot
                                        # product with the dense array this
                                        # would otherwise happen always for all
                                        # sizes
        )

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        cdef intsize nn, N = self.numN
        cdef np.ndarray arrRes

        if self._rangeAccess:
            arrRes = self._fun(self._vecX[idx], self._vecY)
        else:
            arrRes = _arrEmpty(1, N, 1, self._info.dtype[0].typeNum)
            for nn in range(N):
                arrRes[nn] = self._fun(self._vecX[idx], self._vecY[nn])

        return arrRes

    cpdef np.ndarray _getRow(self, intsize idx):
        cdef intsize mm, M = self.numM
        cdef np.ndarray arrRes

        if self._rangeAccess:
            arrRes = self._fun(self._vecX, self._vecY[idx])
        else:
            arrRes = _arrEmpty(1, M, 1, self._info.dtype[0].typeNum)
            for mm in range(M):
                arrRes[mm] = self._fun(self._vecX[mm], self._vecY[idx])

        return arrRes

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        return self._fun(self._vecX[idxM], self._vecY[idxN])

    ############################################## class core methods
    cdef void _core(
        self,
        np.ndarray arrIn,
        np.ndarray arrOut,
        ftype typeIn,
        ftype typeFun,
        ftype typeOut,
        bint backward
    ):

        # determine size of matrices
        cdef intsize mm, nn, numN = self.numN, numM = self.numM

        cdef np.ndarray vecSuppN = self._vecY
        cdef np.ndarray vecSuppM = self._vecX

        # if backward, N and M indexing is swapped and each element must
        # be conjugated
        if backward:
            numN, numM = numM, numN
            vecSuppN, vecSuppM = vecSuppM, vecSuppN

        cdef object val, numSuppN
        cdef np.ndarray vecVal = _arrEmpty(
            1, vecSuppM.shape[0], 1, self._info.dtype[0].typeNum)

        for nn in range(numN):
            # when rangeAccess is allowed, _fun may be called for multiple
            # elements at once
            numSuppN = vecSuppN[nn]
            if self._rangeAccess:
                val = self._fun(
                    numSuppN if backward else vecSuppM,
                    vecSuppM if backward else numSuppN
                )

                if isinstance(val, np.ndarray):
                    # if returned object is an ndarray, assume it to contain
                    # requested data
                    vecVal[:] = val
                else:
                    # otherwise, interpret as scalar value and assign it to all
                    # elements of the requested range
                    for mm in range(numM):
                        vecVal[mm] = val
            elif backward:
                # backward, element-wise
                for mm in range(numM):
                    vecVal[mm] = self._fun(numSuppN, vecSuppM[mm])
            else:
                # forward, element-wirde
                for mm in range(numM):
                    vecVal[mm] = self._fun(vecSuppM[mm], numSuppN)

            # conjugate in backward case
            if backward:
                _conjugateInplace(vecVal)

            # multiply determined vector and accumulate across N. Do a regular
            # scalar product for the first iteration, then accumulate products
            # NOTE: N is actually M in backward case
            _dotSingleRow(arrIn, vecVal, arrOut, typeIn, typeFun, typeOut, nn)

    ############################################## class forward / backward
    cpdef _forwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        '''Calculate the forward transform of this matrix.'''
        self._core(
            arrX, arrRes, typeX, self._info.dtype[0].fusedType, typeRes, False)

    cpdef _backwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        '''Calculate the backward transform of this matrix.'''
        self._core(
            arrX, arrRes, typeX, self._info.dtype[0].fusedType, typeRes, True)

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        arrRes = np.zeros(
            (self.numN, self.numM), dtype=self.dtype)

        for nn in range(self.numN):
            if self._rangeAccess:
                arrRes[nn, :] = self._fun(self._vecX, self._vecY[nn])
            else:
                for mm in range(self.numM):
                    arrRes[nn, mm] = self._fun(self._vecX[mm], self._vecY[nn])

        return arrRes

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                # define parameters for test
                TEST.NUM_N      : 4,
                TEST.NUM_M      : TEST.Permutation([6, TEST.NUM_N]),
                'typeY'         : TEST.Permutation(TEST.LARGETYPES),
                'typeX'         : TEST.Permutation(TEST.FEWTYPES),
                'rangeAccess'   : TEST.Permutation([False, True]),
                'complexFun'    : TEST.Permutation([False, True]),

                # define arguments for test
                'vecY'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'typeY',
                    TEST.SHAPE  : (TEST.NUM_N, )
                }),
                'vecX'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'typeX',
                    TEST.SHAPE  : (TEST.NUM_M, )
                }),

                # define constructor for test instances and naming of test
                TEST.OBJECT     : Parametric,
                TEST.INITARGS   : (lambda param: [
                    param['vecX'](),
                    param['vecY'](),
                    (TEST.IgnoreFunc(lambda x, y:
                                     x * np.int8(2) - np.complex64(1j) * y)
                     if param['complexFun']
                     else TEST.IgnoreFunc(lambda x, y :
                                          x * np.int8(2) - np.int8(3) * y))
                ]),
                TEST.INITKWARGS : {'rangeAccess': 'rangeAccess'},

                # name the test instances individually to reflect test scenario
                'strC'          : (lambda param:
                                   ('complex' if param['complexFun']
                                    else 'real')),
                'strV'          : (lambda param:
                                   ('vector' if param['rangeAccess']
                                    else 'single')),
                TEST.NAMINGARGS : dynFormat("x:%s,y:%s,%s,%s",
                                            'vecX', 'vecY', 'strV', 'strC'),
                TEST.TOL_POWER  : 4.
            },
            TEST.CLASS: {
                # ignore int8 datatype as there will be overflows
                TEST.IGNORE     : TEST.IgnoreFunc(lambda param: (
                    param['typeX'] == param['typeY'] == np.int8))
            },
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c: Parametric(
                    np.arange(c).astype(np.double) / c,
                    np.arange(c).astype(np.double) / c,
                    (lambda x, y: np.sin(2 * np.pi * x ** y)))),
                BENCH.FUNC_SIZE : (lambda c: c),
                BENCH.FUNC_STEP : (lambda c: c * 10 ** (1. / 12))
            },
            BENCH.FORWARD: {},
            BENCH.SOLVE: {},
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c: Parametric(
                    np.arange(2 ** c).astype(np.double) / (2 ** c),
                    np.arange(2 ** c).astype(np.double) / (2 ** c),
                    (lambda x, y: 1.0))),
                BENCH.FUNC_SIZE : (lambda c: 2 ** c),
                BENCH.FUNC_STEP : (lambda c: c + 1)
            },
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, datatype: Parametric(
                    np.arange(2 ** c).astype(datatype) / (2 ** c),
                    np.arange(2 ** c).astype(datatype) / (2 ** c),
                    (lambda x, y: datatype(1)))),
                BENCH.FUNC_SIZE : (lambda c: 2 ** c)
            }
        }

    def _getDocumentation(self):
        return ""
