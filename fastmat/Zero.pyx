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

cimport numpy as np
import numpy as np

from .Matrix cimport Matrix
from .core.cmath cimport _arrZero
from .core.types cimport *


cdef class Zero(Matrix):

    r"""


    .. math::
        x \mapsto  0

    The zero matrix only needs the dimension :math:`n` of the vectors it acts
    on. It is very fast and very good!

    >>> import fastmat as fm
    >>>
    >>> # define the parameter
    >>> n = 10
    >>>
    >>> # construct the matrix
    >>> O = fm.Zero(n)
    """

    def __init__(self, numRows, numCols, **options):
        '''
        Initialize Zero matrix instance.

        Parameters
        ----------
        numRows : int
            Height (row count) of the desired zero matrix.

        numCols : int
            Width (column count) of the desired zero matrix.

        **options : optional
            Additional keyworded arguments. Supports all optional arguments
            supported by :py:class:`fastmat.Matrix`.
        '''
        # set properties of matrix
        self._initProperties(numRows, numCols, np.int8, **options)

    ############################################## class property override
    cpdef np.ndarray _getArray(self):
        return _arrZero(2, self.numRows, self.numCols, self.numpyType)

    cpdef np.ndarray _getCol(self, intsize idx):
        return _arrZero(1, self.numRows, 1, self.numpyType)

    cpdef np.ndarray _getRow(self, intsize idx):
        return _arrZero(1, self.numCols, 1, self.numpyType)

    cpdef object _getItem(self, intsize idxRow, intsize idxCol):
        return 0

    cpdef object _getLargestEigenValue(self):
        return 0.

    cpdef object _getLargestSingularValue(self):
        return 0.

    cpdef Matrix _getT(self):
        return Zero(self.numCols, self.numRows)

    cpdef Matrix _getH(self):
        return Zero(self.numCols, self.numRows)

    cpdef Matrix _getConj(self):
        return self

    cpdef Matrix _getGram(self):
        return Zero(self.numCols, self.numCols)

    cpdef np.ndarray _getColNorms(self):
        return np.zeros((self.numCols, ), dtype=self.dtype)

    cpdef np.ndarray _getRowNorms(self):
        return np.zeros((self.numRows, ), dtype=self.dtype)

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        return (self.numRows, self.numCols)

    ############################################## class forward / backward
    cpdef np.ndarray _forward(self, np.ndarray arrX):
        return _arrZero(
            arrX.ndim, self.numRows, arrX.shape[1] if arrX.ndim > 1 else 1,
            getNumpyType(arrX))

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        return _arrZero(
            arrX.ndim, self.numCols, arrX.shape[1] if arrX.ndim > 1 else 1,
            getNumpyType(arrX))

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        return np.zeros((self.numRows, self.numCols), dtype=self.dtype)

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                TEST.NUM_ROWS   : 35,
                TEST.NUM_COLS   : TEST.Permutation([30, TEST.NUM_ROWS]),
                TEST.OBJECT     : Zero,
                TEST.INITARGS   : [TEST.NUM_ROWS, TEST.NUM_COLS],
                TEST.NAMINGARGS : dynFormat(
                    "%dx%d", TEST.NUM_ROWS, TEST.NUM_COLS
                )
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN      : (lambda c: Zero(c, c)),
            },
            BENCH.FORWARD: {},
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN      : (lambda c: Zero(2 ** c, 2 ** c)),
            }
        }
