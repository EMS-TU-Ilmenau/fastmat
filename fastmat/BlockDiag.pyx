# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False

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
from .core.types cimport *

################################################################################
################################################## class BlockDiag
cdef class BlockDiag(Matrix):
    r"""
    .. math::
        M = \mathrm{diag}\left\{\left(  A_{i}\right)_{i}\right\},

    where the :math:`A_{i}` can be fast transforms of \*any\* type.

    >>> # import the package
    >>> import fastmat as fm
    >>>
    >>> # define the blocks
    >>> A = fm.Circulant(x_A)
    >>> B = fm.Circulant(x_B)
    >>> C = fm.Fourier(n)
    >>> D = fm.Diag(x_D)
    >>>
    >>> # define the block
    >>> # diagonal matrix
    >>> M = fm.BlockDiag(A, B, C, D)

    Assume we have two circulant matrices :math:`A` and :math:`B`, an
    :math:`N`-dimensional Fourier matrix :math:`C` and a diagonal matrix
    :math:`D`. Then we define

    .. math::
        M = \begin{bmatrix}
        A & & & \\
        &  B & & \\
        & &  C & \\
        & & &  D
        \end{bmatrix}.

    Meta types can also be nested, so that a block diagonal matrix can contain
    products of block matrices as its entries. Note that the efficiency of the
    fast transforms decreases the more building blocks they have.

    >>> import fastmat as fm
    >>> # import the package
    >>>
    >>> # define the blocks
    >>> A = fm.Circulant(x_A)
    >>> B = fm.Circulant(x_B)
    >>> F = fm.Fourier(n)
    >>> D = fm.Diag(x_D)
    >>>
    >>> # define a product
    >>> P = fm.Product(A.H, B)
    >>>
    >>> # define the block
    >>> # diagonal matrix
    >>> M = fm.BlockDiag(P, F, D)

    Assume we have a product :math:`P` of two matrices :math:`A^\mathrm{H}` and
    :math:`B`, an :math:`N`-dimensional Fourier matrix :math:`{\mathcal{F}}`
    and a diagonal matrix :math:`D`. Then we define

    .. math::
        M = \begin{bmatrix}
        A^\mathrm{H} \cdot  B &                  &        \\
                              & {\mathcal{F}}    &        \\
                              &                  &  D
        \end{bmatrix}.

    .. todo::
        - BlockDiag should simply skip all Zero Matrices (flag them as "None")?
    """

    ############################################## class methods
    def __init__(self, *matrices, **options):
        '''
        Initialize a BlockDiag matrix instance.

        Parameters
        ----------
        *matrices : :py:class:`fastmat.Matrix`
            The matrix instances to be put along the main diagonal of the block
            diagonal matrix, beginning at index (0, 0) with the first matrix.

        **options : optional
            Additional keyworded arguments. Supports all optional arguments
            supported by :py:class:`fastmat.Matrix`.
        '''
        cdef intsize numRows = 0, numCols = 0
        cdef Matrix term

        self._content = matrices

        if len(matrices) < 1:
            raise ValueError("You must specify at least one matrix.")

        # determine total size and data type of matrix
        dataType = np.int8
        for term in self._content:
            if not isinstance(term, Matrix):
                raise ValueError(
                    "Only fastmat matrices supported, %s given." %(str(term)))

            numRows += term.numRows
            numCols += term.numCols
            dataType = np.promote_types(dataType, term.dtype)

        # set properties of matrix
        self._cythonCall = True
        self._initProperties(numRows, numCols, dataType, **options)
        self._widenInputDatatype = True

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        cdef float complexityFwd = self.numRows
        cdef float complexityBwd = self.numCols
        cdef Matrix item
        for item in self:
            complexityFwd += item.numRows + item.numCols
            complexityBwd += item.numCols + item.numRows

        return (complexityFwd, complexityBwd)

    ############################################## class forward / backward
    cpdef _forwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        cdef Matrix term
        cdef intsize idxN = 0, idxM = 0, ii, cnt = len(self._content)

        for ii in range(0, cnt):
            term = self._content[ii]
            arrRes[idxN:(idxN + term.numRows), :] \
                = term.forward(arrX[idxM:(idxM + term.numCols)])

            idxN += term.numRows
            idxM += term.numCols

    cpdef _backwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        cdef Matrix term
        cdef intsize idxN = 0, idxM = 0, cnt = len(self._content)

        for ii in range(0, cnt):
            term = self._content[ii]

            arrRes[idxM:(idxM + term.numCols), :] \
                = term.backward(arrX[idxN:(idxN + term.numRows)])

            idxN += term.numRows
            idxM += term.numCols

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        cdef np.ndarray arrRes
        cdef Matrix term
        cdef intsize idxN = 0, idxM = 0

        arrRes = np.zeros((self.numRows, self.numCols), dtype=self.dtype)

        for term in self._content:
            arrRes[
                idxN:(idxN + term.numRows), :
            ][
                :, idxM:(idxM + term.numCols)
            ] = term._getArray()

            idxN += term.numRows
            idxM += term.numCols

        return arrRes

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                'size'          : 5,
                TEST.NUM_ROWS   : (lambda param: param['size'] * 3),
                TEST.NUM_COLS   : TEST.NUM_ROWS,
                'mType1'        : TEST.Permutation(TEST.ALLTYPES),
                'mType2'        : TEST.Permutation(TEST.FEWTYPES),
                'arr1'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mType1',
                    TEST.SHAPE  : ('size', 'size')
                }),
                'arr2'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mType2',
                    TEST.SHAPE  : ('size', 'size')
                }),
                'arr3'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mType1',
                    TEST.SHAPE  : ('size', 'size')
                }),
                TEST.INITARGS   : (lambda param: [Matrix(param['arr1']()),
                                                  -2. * Matrix(param['arr2']()),
                                                  2. * Matrix(param['arr3']())
                                                  ]),
                TEST.OBJECT: BlockDiag,
                'strType1'      : (lambda param:
                                   TEST.TYPENAME[param['mType1']]),
                'strType2'      : (lambda param:
                                   TEST.TYPENAME[param['mType2']]),
                TEST.NAMINGARGS : dynFormat("%s,%s,%s:(%dx%d) each",
                                            'strType1', 'strType2', 'strType1',
                                            'size', 'size'),
                TEST.TOL_POWER  : 3.
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH, arrTestDist
        from .Circulant import Circulant
        from .Diag import Diag
        from .Eye import Eye
        from .Fourier import Fourier
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c: BlockDiag(
                    Circulant(np.random.randn(c)),
                    Circulant(np.random.randn(c)),
                    Fourier(c), Diag(np.random.randn(c))
                )),
                BENCH.FUNC_SIZE : (lambda c: 4 * c)
            },
            BENCH.FORWARD: {},
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c: BlockDiag(*([Eye(2 ** c)] * 16))),
                BENCH.FUNC_SIZE : (lambda c: 2 ** c * 16)
            }
        }
