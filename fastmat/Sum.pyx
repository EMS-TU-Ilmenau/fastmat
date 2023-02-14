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

from .core.types cimport *
from .Matrix cimport Matrix

cdef class Sum(Matrix):
    r"""

    For matrices :math:`A_k \in \mathbb{C}^{n \times m}` with
    :math:`k = 1,\dots,N` we define a new mapping :math:`M` as the sum

    .. math::
        M = \sum\limits_{k = 1}^{N} A_k,

    which then also is a mapping in :math:`\mathbb{C}^{n \times m}`.

    >>> # import the package
    >>> import fastmat as fm
    >>>
    >>> # define the components
    >>> A = fm.Circulant(x_A)
    >>> B = fm.Circulant(x_B)
    >>> C = fm.Fourier(n)
    >>> D = fm.Diag(x_D)
    >>>
    >>> # construct the sum of transformations
    >>> M = fm.Sum(A, B, C, D)

    Assume we have two circulant matrices :math:`A` and :math:`B`, an
    :math:`N`-dimensional Fourier matrix :math:`C` and a diagonal matrix
    :math:`D`. Then we define

    .. math::
        M =  A +  B +  C +  D.
    """

    def __init__(self, *matrices, **options):
        '''
        Initialize a Sum matrix instance.

        Parameters
        ----------
        *matrices : :py:class:`fastmat.Matrix`
            The matrix instances to be summed.

        **options : optional
            Additional optional keyworded arguments. Supports all optional
            arguments supported by :py:class:`fastmat.Matrix`.
        '''
        cdef Matrix mat

        # Fold multiple levels of sums
        lstTerms = []

        def __addTerms(matrices):
            for mat in matrices:
                if not isinstance(mat, Matrix):
                    raise TypeError("Sum: Term is not a fastmat Matrix.")

                # flatten nested instances of Sum
                if isinstance(mat, Sum):
                    __addTerms(mat.content)
                else:
                    lstTerms.append(mat)

        __addTerms(matrices)
        self._content = tuple(lstTerms)

        # Have at least one term
        cdef int ii, cntTerms = len(self._content)
        if cntTerms < 1:
            raise ValueError("Sum: No terms given.")

        # check for matching transform dimensions
        cdef intsize numRows = self._content[0].numRows
        cdef intsize numCols = self._content[0].numCols
        for ii in range(1, cntTerms):
            mat = self._content[ii]
            if (mat.numRows != numRows) or (mat.numCols != numCols):
                raise ValueError("Sum: Term dimension mismatch: " + repr(mat))

        # determine data type of sum result
        dataType = np.int8
        for ii in range(0, cntTerms):
            dataType = np.promote_types(dataType, self._content[ii].dtype)

        # set properties of matrix
        self._cythonCall = True
        self._initProperties(numRows, numCols, dataType, **options)
        self._widenInputDatatype = True

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        cdef int cc, cnt = len(self._content)

        result = self._content[0]._getCol(idx).astype(self.dtype)
        for cc in range(1, cnt):
            result += self._content[cc]._getCol(idx)

        return result

    cpdef np.ndarray _getRow(self, intsize idx):
        cdef int cc, cnt = len(self._content)

        result = self._content[0]._getRow(idx).astype(self.dtype)
        for cc in range(1, cnt):
            result += self._content[cc]._getRow(idx)

        return result

    cpdef object _getItem(self, intsize idxRow, intsize idxCol):
        cdef int cc, cnt = len(self._content)

        result = self._content[0]._getItem(idxRow, idxCol).astype(self.dtype)
        for cc in range(1, cnt):
            result += self._content[cc]._getItem(idxRow, idxCol)

        return result

    # TODO: Implement column and row norms

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        cdef float complexity = 0.
        cdef Matrix item

        for item in self._content:
            complexity += item.numRows + item.numCols

        return (complexity, complexity)

    ############################################## class forward / backward
    cpdef _forwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        cdef int cc, cnt = len(self._content)

        arrRes[:] = self._content[0].forward(arrX)
        for cc in range(1, cnt):
            arrRes += self._content[cc].forward(arrX)

    cpdef _backwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        cdef int cc, cnt = len(self._content)

        arrRes[:] = self._content[0].backward(arrX)
        for cc in range(1, cnt):
            arrRes += self._content[cc].backward(arrX)

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        cdef np.ndarray arrRes
        cdef int cc, cnt = len(self._content)

        arrRes = np.zeros((self.numRows, self.numCols), dtype=self.dtype)
        for cc in range(cnt):
            arrRes += self._content[cc].reference()

        return arrRes

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                TEST.NUM_ROWS   : 25,
                TEST.NUM_COLS   : TEST.Permutation([17, TEST.NUM_ROWS]),
                'mType1'        : TEST.Permutation(TEST.ALLTYPES),
                'mType2'        : TEST.Permutation(TEST.FEWTYPES),
                'arrM1'         : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mType1',
                    TEST.SHAPE  : (TEST.NUM_ROWS, TEST.NUM_COLS)
                }),
                'arrM2'         : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mType2',
                    TEST.SHAPE  : (TEST.NUM_ROWS, TEST.NUM_COLS)
                }),
                'arrM3'         : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mType1',
                    TEST.SHAPE  : (TEST.NUM_ROWS, TEST.NUM_COLS)
                }),
                TEST.INITARGS   : [lambda param: Matrix(param['arrM1']()),
                                   lambda param: Matrix(param['arrM2']()),
                                   lambda param: Matrix(param['arrM3']())],
                TEST.OBJECT     : Sum,
                TEST.NAMINGARGS : dynFormat("%s+%s+%s",
                                            'arrM1', 'arrM2', 'arrM3'),
                TEST.TOL_POWER  : 3.
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH
        from .Eye import Eye
        from .Fourier import Fourier
        from .Circulant import Circulant
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c: Sum(
                    Fourier(c), Eye(c), Circulant(np.random.randn(c)))),
                BENCH.FUNC_SIZE : (lambda c: c)
            },
            BENCH.FORWARD: {},
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c: Sum(*([Eye(2 ** c)] * c))),
                BENCH.FUNC_SIZE : (lambda c: 2 ** c)
            }
        }
