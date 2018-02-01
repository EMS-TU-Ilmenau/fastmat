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


################################################################################
################################################## class Blocks
cdef class Blocks(Matrix):
    r"""


    .. math::
        M = \left(  A_{i,j}\right)_{i,j},

    where the :math:`A_{i,j}` can be a fast transforms of *any* type.

    >>> # import the package
    >>> import fastmat as fm
    >>>
    >>> # define the blocks
    >>> A = fm.Circulant(x_A)
    >>> B = fm.Circulant(x_B)
    >>> C = fm.Fourier(n)
    >>> D = fm.Diag(x_D
    >>>
    >>> # define the block
    >>> # matrix row-wise
    >>> M = fm.Blocks([[A,B],[C,D]])

    Assume we have two circulant matrices :math:`A` and :math:`B`, an
    :math:`N`-dimensional Fourier matrix :math:`C` and a diagonal matrix
    :math:`D`. Then we define

    .. math::
        M =
        \begin{bmatrix}
            A &  B \\
            C &  D
        \end{bmatrix}.

    .. todo::
        - Blocks should simply skip all Zero Matrices (flag them as "None")?
    """

    ############################################## class methods
    def __init__(self, arrMatrices):
        '''Initialize Matrix instance with a list of child matrices'''
        if not isinstance(arrMatrices, list):
            raise ValueError("Blocks: Not a nested list of fastmat matrices.")

        if len(arrMatrices) < 1:
            raise ValueError("Blocks: Contains no matrices.")

        cdef intsize numN = 0, numM = 0
        cdef intsize ii, rr, cc
        cdef tuple row, firstRow
        cdef Matrix term

        # initialize sizes and number of rows / cols
        self._numRows = len(arrMatrices)
        self._numCols = len(arrMatrices[0])
        dataType = np.int8

        # generate transposed block structure, force tuples for backward()
        cdef list lst = [tuple(arrMatrices[rr]) for rr in range(self._numRows)]
        self._rows = tuple(lst)
        lst = [[] for _ in range(self._numCols)]
        firstRow = self._rows[0]

        # extract list of row heights and column widths
        self._rowN = tuple([row[0].numN for row in self._rows])
        self._colM = tuple([term.numM for term in firstRow])

        # enumerate rows
        for rr in range(self._numRows):
            row = self._rows[rr]

            # get number of rows from first elements in columns
            numN += row[0].numN

            # check for presence of enough blocks
            if len(row) != self._numCols:
                raise ValueError(
                    "Blocks.row(%d) has incompatible number of entries" %(rr))

            # enumerate columns
            for cc in range(self._numCols):
                term = row[cc]

                # check for matching column height and width
                if term.numN != row[0].numN:
                    raise ValueError(
                        ("Blocks[%d,%d] with shape %s is incompatible with " +
                         "row height (%d,:)") %(
                            rr, cc, str(term.shape), row[0].numN))

                if term.numM != firstRow[cc].numM:
                    raise ValueError(
                        ("Blocks[%d,%d] with shape %s is incompatible with " +
                         "column width (:,%d)") %(
                            rr, cc, str(term.shape), firstRow[cc].numM))

                # first run: get stats and update dimension of
                # Blocks from first column's entries
                if rr == 0:
                    numM += term.numM

                # build transposed copy of blocks to work on
                # in backward
                lst[cc].append(term)

                # determine necessary output data type by
                # applying type promotion
                dataType = np.promote_types(dataType, term.dtype)

        # convert generated transposed array to tuple of tuples
        for cc in range(self._numCols):
            lst[cc] = tuple(lst[cc])

        self._cols = tuple(lst)

        # build a flat list of all nested matrices in _content
        self._content = tuple([item for row in self._rows for item in row])

        # set properties of matrix
        self._initProperties(
            numN, numM, dataType,
            cythonCall=True,
            widenInputDatatype=True
        )

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        cdef float complexityFwd, complexityBwd
        cdef Matrix item

        complexityFwd = complexityBwd = sum(self._colM) + sum(self._rowN)
        for item in self:
            complexityFwd += item.numN + item.numM
            complexityBwd += item.numM + item.numN

        return (complexityFwd, complexityBwd)

    ############################################## class forward / backward
    cpdef _forwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        '''Calculate the forward transform of this matrix'''
        cdef np.ndarray viewOut, arrOut
        cdef Matrix term
        cdef tuple row, viewRows, viewCols
        cdef intsize idxN, idxM, rr, cc
        cdef list lst

        # generate views into output array
        lst = [None] * self._numRows
        idxN = 0
        for rr in range(self._numRows):
            lst[rr] = arrRes[idxN:(idxN + self._rowN[rr])]
            idxN += self._rowN[rr]

        viewRows = tuple(lst)

        # generate views into input array
        lst = [None] * self._numCols
        idxM = 0
        for cc in range(self._numCols):
            lst[cc] = arrX[idxM:(idxM + self._colM[cc])]
            idxM += self._colM[cc]

        viewCols = tuple(lst)

        # do the trick
        for rr in range(self._numRows):
            row = self._rows[rr]
            viewOut = viewRows[rr]

            viewOut[:] = row[0].forward(viewCols[0])
            for cc in range(1, self._numCols):
                viewOut += row[cc].forward(viewCols[cc])

    cpdef _backwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        '''Calculate the backward transform of this matrix'''
        cdef np.ndarray viewOut
        cdef Matrix term
        cdef tuple col, viewRows, viewCols
        cdef intsize idxN, idxM, rr, cc
        cdef list lst

        # generate views into output array
        lst = [None] * self._numCols
        idxM = 0
        for cc in range(self._numCols):
            lst[cc] = arrRes[idxM:(idxM + self._colM[cc])]
            idxM += self._colM[cc]

        viewCols = tuple(lst)

        # generate views into input array
        lst = [None] * self._numRows
        idxN = 0
        for rr in range(self._numRows):
            lst[rr] = arrX[idxN:(idxN + self._rowN[rr])]
            idxN += self._rowN[rr]

        viewRows = tuple(lst)

        # do the trick
        for cc in range(self._numCols):
            col = self._cols[cc]
            viewOut = viewCols[cc]
            viewOut[:] = col[0].backward(viewRows[0])

            for rr in range(1, self._numRows):
                viewOut += col[rr].backward(viewRows[rr])

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        cdef np.ndarray arrRes
        cdef Matrix term
        cdef tuple row
        cdef intsize idxN = 0, idxM = 0

        arrRes = np.empty((self.numN, self.numM), dtype=self.dtype)

        cdef intsize rr, tt
        for rr in range(self._numRows):
            row = self._rows[rr]
            idxM = 0
            for tt in range(self._numCols):
                term = row[tt]
                arrRes[idxN:(idxN + term.numN), idxM:(idxM + term.numM)] = \
                    term._reference()
                idxM += term.numM

            idxN += term.numN

        return arrRes

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                'size'          : 4,
                TEST.NUM_N      : (lambda param: param['size'] * 2),
                TEST.NUM_M      : TEST.NUM_N,
                'mType1'        : TEST.Permutation(TEST.ALLTYPES),
                'mType2'        : TEST.Permutation(TEST.ALLTYPES),
                'arr1'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mType1',
                    TEST.SHAPE  : ('size', 'size')
                }),
                'arr2'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mType2',
                    TEST.SHAPE  : ('size', 'size')
                }),
                'arr3'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mType2',
                    TEST.SHAPE  : ('size', 'size')
                }),
                'arr4'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mType1',
                    TEST.SHAPE  : ('size', 'size')
                }),
                TEST.INITARGS   : (lambda param : [[[Matrix(param['arr1']()),
                                                     Matrix(param['arr2']())],
                                                    [Matrix(param['arr3']()),
                                                     Matrix(param['arr4']())]]
                                                   ]),
                TEST.OBJECT     : Blocks,
                'strType1'      : (lambda param:
                                   TEST.TYPENAME[param['mType1']]),
                'strType2'      : (lambda param:
                                   TEST.TYPENAME[param['mType2']]),
                TEST.NAMINGARGS : dynFormat("[%s,%s],[%s,%s]:(%dx%d) each",
                                            'strType1', 'strType2',
                                            'strType2', 'strType1',
                                            'size', 'size')
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH
        from .Circulant import Circulant
        from .Diag import Diag
        from .Eye import Eye
        from .Fourier import Fourier
        return {
            BENCH.FORWARD: {
                BENCH.FUNC_GEN  : (lambda c: Blocks(
                    [[Circulant(np.random.randn(c)),
                      Circulant(np.random.randn(c))],
                     [Fourier(c), Diag(np.random.randn(c))]])),
                BENCH.FUNC_SIZE : (lambda c: 2 * c)
            },
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c: Blocks([[Eye(2 ** c)] * 4] * 4)),
                BENCH.FUNC_SIZE : (lambda c: 2 ** c * 4)
            }
        }

    def _getDocumentation(self):
        return ""
