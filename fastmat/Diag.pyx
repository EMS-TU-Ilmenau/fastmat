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
cimport numpy as np

from .Matrix cimport Matrix
from .Eye cimport Eye
from .core.cmath cimport _conjugate, _multiply, _arrZero
from .core.types cimport *


################################################################################
################################################## class Diag
cdef class Diag(Matrix):
    r"""


    .. math::
        x \mapsto {\mathrm{diag}}(d_1,\dots,d_n) \cdot  x

    A diagonal matrix is uniquely defined by the entries of its diagonal.

    >>> # import the package
    >>> import fastmat as fm
    >>> import numpy as np
    >>>
    >>> # build the parameters
    >>> n = 4
    >>> d = np.array([1, 0, 3, 6])
    >>>
    >>> # construct the matrix
    >>> D = fm.Diag(d)

    This yields

    .. math::
        d = (1, 0, 3, 6)^\mathrm{T}

    .. math::
        D = \begin{bmatrix}
            1 & & & \\
            & 0 & & \\
            & & 3 & \\
            & & & 6
        \end{bmatrix}
    """

    property vecD:
        r"""Return the matrix-defining vector of diagonal entries.

        *(read-only)*
        """

        def __get__(self):
            return self._vecD

    ############################################## class methods
    def __init__(self, vecD):
        '''Initialize Matrix instance with a list of child matrices'''
        # numN is size of matrix (and of diagonal vector)
        numN = len(vecD)

        # store diagonal entry vector as copy of vecD and complain if
        # dimension does not match
        self._vecD = np.atleast_1d(np.squeeze(np.copy(vecD)))
        if self._vecD.ndim != 1:
            raise ValueError(
                "Diag: Definition vector must have exactly one dimension.")

        # set properties of matrix
        self._initProperties(
            numN, numN, self._vecD.dtype,
            cythonCall=True,
            forceInputAlignment=True,
            fortranStyle=True
        )

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        cdef np.ndarray arrRes

        arrRes = _arrZero(1, self.numN, 1, self._info.dtype[0].typeNum)
        arrRes[idx] = self._vecD[idx]

        return arrRes

    cpdef object _getLargestEV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return np.abs(self._vecD).max().astype(np.float64)

    cpdef object _getLargestSV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return np.abs(self._vecD).max().astype(np.float64)

    cpdef np.ndarray _getRow(self, intsize idx):
        return self._getCol(idx)

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        return self._vecD[idxN] if idxN == idxM else self.dtype(0)

    cpdef Matrix _getGram(self):
        return Diag(np.abs(self._vecD) ** 2)

    cpdef Matrix _getNormalized(self):
        if self._info.dtype.isComplex:
            return Diag((self._vecD / abs(self._vecD)).astype(self.dtype))
        else:
            return Diag(np.sign(self._vecD).astype(self.dtype))

    cpdef Matrix _getT(self):
        return self

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        return (2. * self.numN, 3. * self.numN)

    ############################################## class forward / backward
    cpdef _forwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        ''' Calculate the forward transform of this matrix.'''
        _multiply(
            arrX, self._vecD, arrRes,
            typeX, self._info.dtype[0].fusedType, typeRes)

    cpdef _backwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        ''' Calculate the backward transform of this matrix.'''
        _multiply(
            arrX, _conjugate(self._vecD), arrRes,
            typeX, self._info.dtype[0].fusedType, typeRes)

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        cdef intsize ii, N = self.numN
        cdef np.ndarray d = self.vecD

        return np.diag(d)

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                TEST.NUM_N      : 35,
                TEST.NUM_M      : TEST.NUM_N,
                'mTypeD'        : TEST.Permutation(TEST.ALLTYPES),
                TEST.PARAMALIGN : TEST.Permutation(TEST.ALLALIGNMENTS),
                'vecD'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeD',
                    TEST.SHAPE  : (TEST.NUM_N, ),
                    TEST.ALIGN  : TEST.PARAMALIGN
                }),
                TEST.INITARGS   : (lambda param: [param['vecD']()]),
                TEST.OBJECT     : Diag,
                TEST.NAMINGARGS : dynFormat("%s", 'vecD')
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c: Diag(np.random.uniform(2, 3, c)))
            },
            BENCH.FORWARD: {},
            BENCH.SOLVE: {},
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c:
                                   Diag(np.random.uniform(2, 3, 2 ** c)))
            },
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, dt: Diag(
                    np.random.uniform(2, 3, 2 ** c).astype(dt)))
            }
        }

    def _getDocumentation(self):
        return ""
