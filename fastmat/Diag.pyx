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
from .Eye cimport Eye
from .core.cmath cimport _conjugate, _multiply, _arrZero, _arrSqueezedCopy
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
    def __init__(self, vecD, **options):
        '''
        Initialize a Diag matrix instance.

        Parameters
        ----------
        vecD : :py:class:`numpy.ndarray`
            The generating vector of the diagonal entries of this matrix.

        **options : optional
            Additional keyworded arguments. Supports all optional arguments
            supported by :py:class:`fastmat.Matrix`.
        '''
        # numRows is size of matrix (and of diagonal vector)
        numRows = len(vecD)

        # store diagonal entry vector as copy of vecD and complain if
        # dimension does not match
        self._vecD = _arrSqueezedCopy(vecD)
        if self._vecD.ndim != 1:
            raise ValueError(
                "Diag: Definition vector must have exactly one dimension.")

        # set properties of matrix
        self._cythonCall = True
        self._initProperties(numRows, numRows, self._vecD.dtype, **options)
        self._forceContiguousInput = True
        self._fortranStyle = True

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        cdef np.ndarray arrRes

        arrRes = _arrZero(1, self.numRows, 1, self.numpyType)
        arrRes[idx] = self._vecD[idx]

        return arrRes

    cpdef object _getLargestEigenValue(self):
        return np.abs(self._vecD).max().astype(np.float64)

    cpdef object _getLargestSingularValue(self):
        return np.abs(self._vecD).max().astype(np.float64)

    cpdef np.ndarray _getRow(self, intsize idx):
        return self._getCol(idx)

    cpdef object _getItem(self, intsize idxRow, intsize idxCol):
        return self._vecD[idxRow] if idxRow == idxCol else self.dtype(0)

    cpdef Matrix _getGram(self):
        return Diag(np.abs(self._vecD) ** 2)

    cpdef np.ndarray _getColNorms(self):
        return np.abs(self._vecD)

    cpdef np.ndarray _getRowNorms(self):
        return np.abs(self._vecD)

    cpdef Matrix _getColNormalized(self):
        if typeInfo[self.fusedType].isComplex:
            return Diag((self._vecD / np.abs(self._vecD)).astype(self.dtype))
        else:
            return Diag(np.sign(self._vecD).astype(self.dtype))

    cpdef Matrix _getRowNormalized(self):
        return self.colNormalized

    cpdef Matrix _getT(self):
        return self

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        return (2. * self.numRows, 3. * self.numRows)

    ############################################## class forward / backward
    cpdef _forwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        _multiply(arrX, self._vecD, arrRes,
                  typeX, self.fusedType, typeRes)

    cpdef _backwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        _multiply(arrX, _conjugate(self._vecD), arrRes,
                  typeX, self.fusedType, typeRes)

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        cdef intsize ii
        cdef np.ndarray d = self.vecD

        return np.diag(d)

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                TEST.NUM_ROWS   : 35,
                TEST.NUM_COLS   : TEST.NUM_ROWS,
                'mTypeD'        : TEST.Permutation(TEST.ALLTYPES),
                TEST.PARAMALIGN : TEST.Permutation(TEST.ALLALIGNMENTS),
                'vecD'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeD',
                    TEST.SHAPE  : (TEST.NUM_ROWS, ),
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
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c:
                                   Diag(np.random.uniform(2, 3, 2 ** c)))
            },
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, dt: Diag(
                    np.random.uniform(2, 3, 2 ** c).astype(dt)))
            }
        }
