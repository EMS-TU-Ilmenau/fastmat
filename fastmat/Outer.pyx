# -*- coding: utf-8 -*-

# Copyright 2018 Sebastian Semper, Christoph Wagner
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
from .core.cmath cimport _conjugate, _multiply, _arrSqueezedCopy

cdef class Outer(Matrix):
    r"""

    The outer product is a special case of the Kronecker product of
    one-dimensional vectors. For given :math:`a \in \mathbb{C}^n` and
    :math:`b \in \mathbb{C}^m` it is defined as

    .. math::
        x \mapsto  a \cdot  b^\mathrm{T} \cdot  x.

    It is clear, that this matrix has at most rank :math:`1` and as such
    has a fast transformation.

    >>> # import the package
    >>> import fastmat as fm
    >>> import numpy as np
    >>>
    >>> # define parameter
    >>> n, m = 4, 5
    >>> v = np.arange(n)
    >>> h = np.arange(m)
    >>>
    >>> # construct the matrix
    >>> M = fm.Outer(v, h)

    This yields

    .. math::
        v = (0,1,2,3,4)^\mathrm{T}

    .. math::
        h = (0,1,2,3,4,5)^\mathrm{T}

    .. math::
        M = \begin{bmatrix}
        0 & 0 & 0 & 0 & 0  \\
        0 & 1 & 2 & 3 & 4  \\
        0 & 2 & 4 & 6 & 8  \\
        0 & 3 & 6 & 9 & 12
        \end{bmatrix}
    """

    property vecV:
        r"""Return the matrix-defining vector of vertical defining entries.

        *(read only)*
        """

        def __get__(self):
            return self._vecV

    property vecH:
        r"""Return the matrix-defining vector of horizontal defining entries.

        *(read only)*
        """

        def __get__(self):
            return self._vecH

    def __init__(self, vecV, vecH, **options):
        '''
        Initialize a Outer product matrix instance.

        Parameters
        ----------
        arrV : :py:class:`numpy.ndarray`
            A 1d vector defining the column factors of the resulting matrix.

        arrH : :py:class:`numpy.ndarray`
            A 1d vector defining the row factors of the resulting matrix.

        **options : optional
            Additional keyworded arguments. Supports all optional arguments
            supported by :py:class:`fastmat.Matrix`.
        '''

        # check dimensions
        vecV = _arrSqueezedCopy(vecV)
        vecH = _arrSqueezedCopy(vecH)

        if vecV.ndim != 1 or vecH.ndim != 1:
            raise ValueError("Outer parameters must be one-dimensional.")

        # height and width of matrix is defined by length of input vectors
        cdef intsize numRows = len(vecV)
        cdef intsize numCols = len(vecH)

        # determine joint data type of operation
        datatype = np.promote_types(vecV.dtype, vecH.dtype)

        # store H/V entry vectors as copies of vecH/vecV
        self._vecH = vecH.astype(datatype, copy=True).reshape((1, numCols))
        self._vecV = vecV.astype(datatype, copy=True).reshape((numRows, 1))

        # store ravelled pointers of vecH/vecV
        self._vecHRav = self._vecH.ravel()
        self._vecVRav = self._vecV.ravel()

        # hermitian transpose of vecH/vecV for backward
        self._vecHConj = _conjugate(self._vecH).reshape((numCols, 1))
        self._vecVHerm = _conjugate(self._vecV).reshape((1, numRows))

        # set properties of matrix
        self._initProperties(numRows, numCols, datatype, **options)

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        return self._vecHRav[idx] * self._vecVRav

    cpdef np.ndarray _getRow(self, intsize idx):
        return self._vecVRav[idx] * self._vecHRav

    cpdef object _getItem(self, intsize idxRow, intsize idxCol):
        return self._vecVRav[idxRow] * self._vecHRav[idxCol]

    cpdef np.ndarray _getColNorms(self):
        return (np.abs(self.vecH.reshape((-1, ))).astype(np.float64) *
                np.linalg.norm(self.vecV))

    cpdef np.ndarray _getRowNorms(self):
        return (np.abs(self.vecV.reshape((-1, ))).astype(np.float64) *
                np.linalg.norm(self.vecH))

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        cdef float complexity = 2 * (self.numRows + self.numCols)
        return (complexity, complexity)

    ############################################## class forward / backward
    cpdef np.ndarray _forward(self, np.ndarray arrX):
        return self._vecV.dot(self._vecH.dot(arrX))

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        return self._vecHConj.dot(self._vecVHerm.dot(arrX))

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        dtype = np.promote_types(self.dtype, np.float64)
        return self._vecV.dot(self._vecH.astype(dtype))

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                TEST.NUM_ROWS   : 4,
                TEST.NUM_COLS   : TEST.Permutation([6, TEST.NUM_ROWS]),
                'mTypeH'        : TEST.Permutation(TEST.FEWTYPES),
                'mTypeV'        : TEST.Permutation(TEST.ALLTYPES),
                'vecH'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeH',
                    TEST.SHAPE  : (TEST.NUM_ROWS, 1)
                }),
                'vecV'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeV',
                    TEST.SHAPE  : (TEST.NUM_COLS, 1)
                }),
                TEST.INITARGS   : (lambda param : [param['vecH'](),
                                                   param['vecV']()]),
                TEST.OBJECT     : Outer,
                TEST.NAMINGARGS : dynFormat("%so%s", 'vecH', 'vecV'),
                TEST.TOL_MINEPS : (lambda param:
                                   max(getTypeEps(param['mTypeH']),
                                       getTypeEps(param['mTypeV']))),
                TEST.TOL_POWER  : 4.
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {
                # ignore int8 datatype as there will be overflows
                TEST.IGNORE     : TEST.IgnoreFunc(lambda param: (
                    param['mTypeH'] == param['mTypeV'] ==
                    param[TEST.DATATYPE] == np.int8))
            }
        }

    def _getBenchmark(self):
        from .inspect import BENCH, arrTestDist
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c: Outer(
                    arrTestDist((c, ), dtype=np.float32),
                    arrTestDist((c, ), dtype=np.float32)))
            },
            BENCH.FORWARD: {},
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c: Outer(
                    arrTestDist((2 ** c, ), dtype=np.float32),
                    arrTestDist((2 ** c, ), dtype=np.float32)))
            },
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, datatype: Outer(
                    arrTestDist((2 ** c, ), dtype=datatype),
                    arrTestDist((2 ** c, ), dtype=datatype)))
            }
        }
