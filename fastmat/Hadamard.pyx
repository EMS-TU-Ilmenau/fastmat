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

#from libc.stdlib cimport malloc, free
#from libc.string cimport memcpy

import numpy as np
cimport numpy as np

from .core.types cimport nptype, _getNpType
from .core.strides cimport *
from .core.cmath cimport _arrEmpty
from .Matrix cimport Matrix
from .Eye cimport Eye

# have a very lazy import to avoid initialization of scipy.linalg during import
# of main module
spHadamard = None


cdef void _hadamardCore(STRIDE_s *strA, STRIDE_s *strB, TYPE_IN dummy):
    cdef intsize vv, ee

    cdef char *ptrVectorA
    cdef char *ptrElementA
    cdef char *ptrVectorB
    cdef char *ptrElementB

    cdef TYPE_IN a, b

    ptrVectorA = strA.base
    ptrVectorB = strB.base
    for vv in range(strA[0].numVectors):
        ptrElementA = ptrVectorA
        ptrElementB = ptrVectorB
        for ee in range(strA[0].numElements):
            a = (<TYPE_IN *> ptrElementA)[0]
            b = (<TYPE_IN *> ptrElementB)[0]
            (<TYPE_IN *> ptrElementA)[0] = a + b
            (<TYPE_IN *> ptrElementB)[0] = a - b

            ptrElementA += strA.strideElement
            ptrElementB += strB.strideElement

        ptrVectorA += strA.strideVector
        ptrVectorB += strB.strideVector

cdef class Hadamard(Matrix):
    r"""

    A Hadamard Matrix is recursively defined as

    .. math::
        H_n =  H_1 \otimes  H_{n-1},

    where

    .. math::
        H_1 = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}

    and :math:`H_0 = (1)`. Obviously the dimension of :math:`H_n` is
    :math:`2^n`. The transform is realized with the Fast Hadamard Transform
    (FHT).

    >>> # import the package
    >>> import fastmat as fm
    >>>
    >>> # define the parameter
    >>> n = 4
    >>>
    >>> # construct the matrix
    >>> H = fm.Hadamard(n)

    This yields a Hadamard matrix :math:`{\mathcal{H}}_4` of order :math:`4`,
    i.e. with :math:`16` rows and columns.

    The algorithm we used is described in [2]_ and was implemented in
    Cython [3]_.
    """

    property order:
        r"""Return the order of the hadamard matrix."""

        def __get__(self):
            return self._order

    def __init__(self, order):
        '''Initialize Matrix instance'''
        if order < 1:
            raise ValueError("Hadamard: Order must be larger than 0.")

        cdef int maxOrder = sizeof(self._numN) * 8 - 2
        if order > maxOrder:
            raise ValueError(
                "Hadamard: Order exceeds maximum for this platform: %d" %(
                    maxOrder))

        self._order = order

        # set properties of matrix
        numN = 2 ** self._order
        self._initProperties(numN, numN, np.int8, cythonCall=True,
                             forceInputAlignment=True)

    cpdef np.ndarray _getArray(self):
        '''
        Return an explicit representation of the matrix as numpy-array.
        '''
        return self._reference()

    ############################################## class property override
    cpdef object _getLargestEV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return np.sqrt(self.numN)

    cpdef object _getLargestSV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return np.sqrt(self.numN)

    cpdef Matrix _getNormalized(self):
        return Hadamard(self.order) * np.float32(1. / np.sqrt(self.numN))

    cpdef Matrix _getGram(self):
        return Eye(self.numN) * np.float32(self.numN)

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        cdef float complexity = self.numN * self.order
        return (complexity, complexity + 1)

    ############################################## class forward / backward
    cpdef _forwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        '''
        Calculate the forward transform of this matrix.
        '''
        cdef nptype dtype = typeInfo[typeRes].typeNum
        cdef intsize N = arrX.shape[0], M = arrX.shape[1], order = self._order
        cdef intsize mm, oo
        cdef STRIDE_s strInput, strOutput, strA, strB
        cdef intsize butterflyDistance, butterflyCount

        strideInit(&strInput, arrX, 0)
        strideInit(&strOutput, arrRes, 0)

        for mm in range(M):
            opCopyVector(&strOutput, mm, &strInput, mm)

            butterflyDistance = 1
            butterflyCount = N / 2
            for oo in range(order):
                strideCopy(&strA, &strOutput)
                strideCopy(&strB, &strOutput)

                # Butterfly map over iterations of oo
                #        oo |0 |1  |2
                # element 0 |A |A  |A
                # element 1 |A | B | B
                # element 2 |B |A  |  C
                # element 3 |B | B |   D
                # element 4 |C |C  |A
                # element 5 |C | D | B
                # element 6 |D |C  |  C
                # element 7 |D |D |   D
                strideSubgridVector(&strA, mm, 0,
                                    1, butterflyDistance,
                                    2 * butterflyDistance, butterflyCount)
                strideSubgridVector(&strB, mm, butterflyDistance,
                                    1, butterflyDistance,
                                    2 * butterflyDistance, butterflyCount)

                if typeX == TYPE_FLOAT32:
                    _hadamardCore[np.float32_t](&strA, &strB, 0)
                elif typeX == TYPE_FLOAT64:
                    _hadamardCore[np.float64_t](&strA, &strB, 0)
                elif typeX == TYPE_COMPLEX64:
                    _hadamardCore[np.complex64_t](&strA, &strB, 0)
                elif typeX == TYPE_COMPLEX128:
                    _hadamardCore[np.complex128_t](&strA, &strB, 0)
                elif typeX == TYPE_INT64:
                    _hadamardCore[np.int64_t](&strA, &strB, 0)
                elif typeX == TYPE_INT32:
                    _hadamardCore[np.int32_t](&strA, &strB, 0)
                elif typeX == TYPE_INT8:
                    _hadamardCore[np.int8_t](&strA, &strB, 0)
                else:
                    raise NotImplementedError(
                        "Hadamard: %d not supported." %(typeX))

                butterflyDistance <<= 1
                butterflyCount >>= 1

    cpdef _backwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        '''
        Calculate the backward transform of this matrix.
        '''
        return self._forwardC(arrX, arrRes, typeX, typeRes)

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        global spHadamard
        if spHadamard is None:
            spHadamard = __import__('scipy.linalg', globals(), locals(),
                                    ['hadamard']).hadamard

        return spHadamard(self.numN, dtype=self.dtype)

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                # define matrix sizes and parameters
                'order'         : TEST.Permutation([4, 6]),
                TEST.NUM_N      : (lambda param : 2 ** param['order']),
                TEST.NUM_M      : TEST.NUM_N,

                # define constructor for test instances and naming of test
                TEST.OBJECT     : Hadamard,
                TEST.INITARGS   : ['order'],
                TEST.NAMINGARGS : dynFormat("%d", 'order')
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c: Hadamard(c)),
                BENCH.FUNC_SIZE : (lambda c: 2 ** c),
                BENCH.FUNC_STEP : (lambda c: c + 1),
            },
            BENCH.FORWARD: {},
            BENCH.SOLVE: {},
            BENCH.OVERHEAD: {},
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, datatype: Hadamard(c))
            }
        }

    def _getDocumentation(self):
        return ""
