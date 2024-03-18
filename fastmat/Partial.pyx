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
from .core.types cimport *
from .core.cmath cimport _arrZero, _arrSqueeze

cdef class Partial(Matrix):
    r"""

    Let :math:`I \subset \{1,\dots,n\}` and :math:`J \subset \{1,\dots,m\}`
    index sets and :math:`M \in \mathbb{C}^{n \times m}` a linear transform.
    Then the partial transform :math:`M_{I,J}` is defined as

    .. math::
        x \in \mathbb{C}^m \mapsto ( M_J \cdot  x_J)_{i \in I}.

    In other words, we select the rows :math:`I` of :math:`M` and columns J
    of :math:`M` and rows :math:`J` of :math:`x`.

    >>> # import the package
    >>> import fastmat as fm
    >>> import numpy as np
    >>>
    >>> # define the index set
    >>> a = np.arange(n)
    >>> am = np.mod(a, 2)
    >>> b = np.array(am, dtype='bool')
    >>> I = a[b]
    >>>
    >>> # construct the partial transform
    >>> M = fm.Partial(F, I)

    Let :math:`{\mathcal{F}}` be the :math:`n`-dimensional Fourier matrix.
    And let :math:`I` be the set of odd integers. Then we define a partial
    transform as

    .. math::
        M = {\mathcal{F}}_I
    """

    property rowSelection:
        r"""Return the support of the base matrix which defines the partial

        Subselected rows

        *(read only)*
        """

        def __get__(self):
            return (self._rowSelection if self._rowSelection is not None
                    else np.arange(self.numRows))

    property colSelection:
        r"""Return the support of the base matrix which defines the partial

        Subselected columns

        *(read only)*
        """

        def __get__(self):
            return (self._colSelection if self._colSelection is not None
                    else np.arange(self.numCols))

    property indicesN:
        '''Deprecated. See .rowSelection'''
        def __get__(self):
            import warnings
            warnings.warn('indicesN is deprecated. Use rowSelection.',
                          FutureWarning)
            return self.rowSelection

    property indicesM:
        '''Deprecated. See .colSelection'''
        def __get__(self):
            import warnings
            warnings.warn('indicesM is deprecated. Use colSelection.',
                          FutureWarning)
            return self.colSelection

    def __init__(self, mat, **options):
        '''
        Initialize a Partial matrix instance.

        Parameters
        ----------
        mat : :py:class:`fastmat.Matrix`
            A fastmat matrix instance subject to partial access.

        rows : :py:class:`numpy.ndarray`, optional
            A 1d vector selecting rows of mat.

            If `N` is of type bool it's size must match the height of mat and
            the values of `N` corresponds to taking/dumping the corresponding
            row.

            If `N` is of type int it's values correspond to the indices of the
            rows of mat to select. The size of `N` then matches the height of
            the partialed matrix.

            Defaults to selecting all rows.

        cols : :py:class:`numpy.ndarray`, optional
            A 1d vector selecting columns of mat. The behaviour is identical to
            `N`.

            Defaults to selecting all columns.

        **options : optional
            Additional optional keyworded arguments. Supports all optional
            arguments supported by :py:class:`fastmat.Matrix`.
        '''

        # initialize matrix for full support (used anyway for checking)
        if not isinstance(mat, Matrix):
            raise TypeError("Partial: fastmat Matrix required.")

        self._content = (mat, )

        # check if anything needs to be done in the row- or column dimensions
        # store support indices in row and column dimension if needed

        def checkSelection(vecSelection, sizeDim, nameDim):
            if vecSelection is not None:
                vecSelection = np.array(vecSelection)
                if vecSelection.dtype == bool:
                    # convert specification of boolean decisions to indices
                    vecSelection = np.arange(sizeDim)[vecSelection]
                elif isInteger(vecSelection):
                    pass
                else:
                    raise TypeError(
                        "Partial: Type of %s indices must be int or bool." %(
                            nameDim, ))

                bounded = (
                    ((len(vecSelection) != sizeDim) or
                     (np.sum(vecSelection - np.arange(sizeDim)) != 0)) and
                    (np.any((vecSelection >= sizeDim) | (vecSelection < 0)))
                )
                if bounded:
                    raise ValueError(
                        "Partial: A %s index exceeds matrix dimensions." %(
                            nameDim, ))

                return vecSelection
            else:
                return None

        if 'N' in options:
            import warnings
            warnings.warn(
                'N=~ is deprecated in Partial.__init__(). Use rows=~.',
                FutureWarning
            )
            options['rows'] = options['N']

        if 'M' in options:
            import warnings
            warnings.warn(
                'M=~ is deprecated in Partial.__init__() Use cols=~.',
                FutureWarning
            )
            options['cols'] = options['M']

        self._rowSelection = checkSelection(
            options.get('rows', None), mat.numRows, 'row'
        )
        self._colSelection = checkSelection(
            options.get('cols', None), mat.numCols, 'col'
        )

        # set properties of matrix.
        self._initProperties(
            (len(self._rowSelection) if self._rowSelection is not None
             else mat.numRows),
            (len(self._colSelection) if self._colSelection is not None
             else mat.numCols),
            mat.dtype,
            **options
        )

    def __repr__(self):
        '''
        Return a string representing this very class instance.
        The __repr__() of the nested matrix is extended by an info about
        the applied reduction if the Partial itself is not subclassed
        further.
        '''
        # determine if subclassed
        if type(self) == Partial:
            return "<%s[%dx%d](%s[%dx%d]):0x%12x>" %(
                self.__class__.__name__,
                self.numRows, self.numCols,
                self._content[0].__class__.__name__,
                self._content[0].numRows, self._content[0].numCols,
                id(self))
        else:
            return super(Partial, self).__repr__()

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        cdef intsize idxCol = (self._colSelection[idx]
                               if self._colSelection is not None else idx)
        return _arrSqueeze(self._content[0].getCol(idxCol)[self._rowSelection]
                           if self._rowSelection is not None
                           else self._content[0].getCol(idxCol))

    cpdef np.ndarray _getRow(self, intsize idx):
        cdef intsize idxRow = (self._rowSelection[idx]
                               if self._rowSelection is not None else idx)
        return _arrSqueeze(self._content[0].getRow(idxRow)[self._colSelection]
                           if self._colSelection is not None
                           else self._content[0].getRow(idxRow))

    cpdef np.ndarray _getColNorms(self):
        cdef np.ndarray arrNorms
        if self._rowSelection is not None:
            return super(Partial, self)._getColNorms()
        else:
            arrNorms = self._content[0].colNorms
            return (arrNorms if self._colSelection is None
                    else arrNorms[self._colSelection])

    cpdef np.ndarray _getRowNorms(self):
        cdef np.ndarray arrNorms
        if self._colSelection is not None:
            return super(Partial, self)._getRowNorms()
        else:
            arrNorms = self._content[0].rowNorms
            return (arrNorms if self._rowSelection is None
                    else arrNorms[self._rowSelection])

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        cdef Matrix M = self._content[0]
        cdef float complexityFwd = 0.
        cdef float complexityBwd = 0.
        if self._colSelection is not None:
            complexityFwd += M.numCols + self.numCols
            complexityBwd += self.numRows

        if self._rowSelection is not None:
            complexityBwd += M.numRows + self.numRows
            complexityFwd += self.numCols

        return (complexityFwd, complexityBwd)

    ############################################## class forward / backward
    cpdef np.ndarray _forward(self, np.ndarray arrX):
        cdef np.ndarray arrInput

        if self._colSelection is not None:
            arrInput = _arrZero(
                2, self.content[0].numCols, arrX.shape[1], getNumpyType(arrX))
            arrInput[self._colSelection, :] = arrX
        else:
            arrInput = arrX

        return (self.content[0].forward(arrInput)[self._rowSelection, :]
                if self._rowSelection is not None
                else self._content[0].forward(arrInput))

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        cdef np.ndarray arrInput

        if self._rowSelection is not None:
            arrInput = _arrZero(
                2, self.content[0].numRows, arrX.shape[1], getNumpyType(arrX))
            arrInput[self._rowSelection, :] = arrX
        else:
            arrInput = arrX

        return (self.content[0].backward(arrInput)[self._colSelection, :]
                if self._colSelection is not None
                else self.content[0].backward(arrInput))

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        cdef np.ndarray arrFull = self.content[0].reference()
        return arrFull[
            self._rowSelection if self._rowSelection is not None else np.s_, :
        ][
            :, self._colSelection if self._colSelection is not None else np.s_
        ]

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        from .Hadamard import Hadamard
        return {
            TEST.COMMON: {
                'order'         : 4,
                'num_rows'      : (lambda param: 2 ** param['order']),
                'num_cols'      : 'num_rows',
                TEST.NUM_ROWS   : (lambda param: (
                    np.count_nonzero(param['subRows'])
                    if param['subRows'].dtype == bool
                    else len(param['subRows'])
                )),
                TEST.NUM_COLS   : (lambda param: (
                    np.count_nonzero(param['subCols'])
                    if param['subCols'].dtype == bool
                    else len(param['subCols'])
                )),
                'subCols'       : TEST.Permutation([
                    np.array([1, 2, 3, 11, 12]),
                    np.array([6]),
                    np.array([
                        True, True, False, False, True, False, False, True,
                        False, True, True, False, False, True, False, True
                    ])
                ]),
                'subRows'       : TEST.Permutation([
                    np.array([7, 8, 9, 13]),
                    np.array([10]),
                    np.array([
                        False, True, True, False, False, True, False, True,
                        True, True, False, False, True, False, False, True
                    ])
                ]),
                TEST.INITARGS   : (lambda param: [
                    Hadamard(param['order']),
                ]),
                TEST.INITKWARGS : (lambda param: {
                    'rows': param['subRows'],
                    'cols': param['subCols']
                }),
                'strIndexTypeM' : (lambda param: (
                    'B' if param['subCols'].dtype == bool else 'I'
                )),
                'strIndexTypeN' : (lambda param: (
                    'B' if param['subRows'].dtype == bool else 'I'
                )),
                TEST.OBJECT     : Partial,
                TEST.NAMINGARGS : dynFormat(
                    "Hadamard(%d)->%dx%d,%s%s",
                    'order', TEST.NUM_ROWS, TEST.NUM_COLS,
                    'strIndexTypeN', 'strIndexTypeM'
                )
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH
        from .Eye import Eye
        return {
            BENCH.FORWARD: {
                BENCH.FUNC_GEN  : (lambda c: Partial(
                    Eye(2 * c), rows=np.arange(c), cols=np.arange(c)
                ))
            },
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c: Partial(
                    Eye(2 ** c), rows=np.arange(2 ** c), cols=np.arange(2 ** c)
                ))
            }
        }
