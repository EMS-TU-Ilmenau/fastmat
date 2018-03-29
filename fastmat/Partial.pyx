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
from .core.types cimport *
from .core.cmath cimport _arrZero

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

    property indicesN:
        r"""Return the support of the base matrix which defines the partial

        Subselected rows

        *(read only)*
        """

        def __get__(self):
            return self._indicesN if self._pruneN else np.arange(self.numN)

    property indicesM:
        r"""Return the support of the base matrix which defines the partial

        Subselected columns

        *(read only)*
        """

        def __get__(self):
            return self._indicesM if self._pruneM else np.arange(self.numM)

    def __init__(
        self,
        mat,
        N=None,
        M=None
    ):
        '''Initialize Matrix instance'''

        # initialize matrix for full support (used anyway for checking)
        if not isinstance(mat, Matrix):
            raise ValueError("Partial: fastmat Matrix required.")

        self._indicesM = None
        self._indicesN = None
        self._content = (mat, )

        # check if anything needs to be done in N- or M-dimension
        # store support indices in N- and M- dimension if needed
        self._pruneN = False
        if N is not None:
            N = np.array(N)
            if N.dtype == np.bool:
                # convert specification of boolean decisions to indices
                N = np.arange(mat.numN)[N]
            elif isInteger(N):
                pass
            else:
                raise TypeError(
                    "Partial: Type of row indices must be integer or bool.")

            if (len(N) != mat.numN) or (np.sum(N - np.arange(mat.numN)) != 0):
                if np.any((N >= mat.numN) | (N < 0)):
                    raise ValueError(
                        "Partial: A row index exceed matrix dimensions.")

                self._indicesN = N
                self._pruneN = True

        self._pruneM = False
        if M is not None:
            M = np.array(M)
            if M.dtype == np.bool:
                # convert specification of boolean decisions to indices
                M = np.arange(mat.numM)[M]
            elif isInteger(M):
                pass
            else:
                raise TypeError(
                    "Partial: Type of column indices must be integer or bool.")

            if (len(M) != mat.numM) or (np.sum(M - np.arange(mat.numM)) != 0):
                if np.any((M >= mat.numM) | (M < 0)):
                    raise ValueError(
                        "Partial: A column index exceeds matrix dimensions.")

            self._indicesM = M
            self._pruneM = True

        # set properties of matrix.
        self._initProperties(
            len(self._indicesN) if self._pruneN else mat.numN,
            len(self._indicesM) if self._pruneM else mat.numM,
            mat.dtype)

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
                self.numN, self.numM,
                self._content[0].__class__.__name__,
                self._content[0].numN, self._content[0].numM,
                id(self))
        else:
            return super(Partial, self).__repr__()

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        cdef intsize idxM = self._indicesM[idx] if self._pruneM else idx
        return (self._content[0].getCol(idxM)[self._indicesN] if self._pruneN
                else self._content[0].getCol(idxM))

    cpdef np.ndarray _getRow(self, intsize idx):
        cdef intsize idxN = self._indicesN[idx] if self._pruneN else idx
        return (self._content[0].getRow(idxN)[self._indicesM] if self._pruneM
                else self._content[0].getRow(idxN))

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        cdef Matrix M = self._content[0]
        cdef float complexityFwd = 0.
        cdef float complexityBwd = 0.
        if self._pruneM:
            complexityFwd += M.numM + self.numM
            complexityBwd += self.numN

        if self._pruneN:
            complexityBwd += M.numN + self.numN
            complexityFwd += self.numM

        return (complexityFwd, complexityBwd)

    ############################################## class forward / backward
    cpdef np.ndarray _forward(self, np.ndarray arrX):
        '''Calculate the forward transform of this matrix'''

        cdef np.ndarray arrInput

        if self._pruneM:
            arrInput = _arrZero(
                2, self.content[0].numM, arrX.shape[1], getNumpyType(arrX))
            arrInput[self._indicesM, :] = arrX
        else:
            arrInput = arrX

        return (self.content[0].forward(arrInput)[self._indicesN, :]
                if self._pruneN else self._content[0].forward(arrInput))

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        '''Calculate the backward transform of this matrix'''

        cdef np.ndarray arrInput

        if self._pruneN:
            arrInput = _arrZero(
                2, self.content[0].numN, arrX.shape[1], getNumpyType(arrX))
            arrInput[self._indicesN, :] = arrX
        else:
            arrInput = arrX

        return (self.content[0].backward(arrInput)[self._indicesM, :]
                if self._pruneM else self.content[0].backward(arrInput))

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        cdef np.ndarray arrFull = self.content[0].reference()
        return arrFull[
            self._indicesN if self._pruneN else np.s_, :][
            :, self._indicesM if self._pruneM else np.s_]

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        from .Hadamard import Hadamard
        return {
            TEST.COMMON: {
                'order'         : 4,
                'num_N'         : (lambda param: 2 ** param['order']),
                'num_M'         : 'num_N',
                TEST.NUM_N      : (lambda param: (
                    np.count_nonzero(param['subRows'])
                    if param['subRows'].dtype == np.bool
                    else len(param['subRows'])
                )),
                TEST.NUM_M      : (lambda param: (
                    np.count_nonzero(param['subCols'])
                    if param['subCols'].dtype == np.bool
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
                    param['subRows'],
                    param['subCols']
                ]),
                'strIndexTypeM' : (lambda param: (
                    'B' if param['subCols'].dtype == np.bool else 'I'
                )),
                'strIndexTypeN' : (lambda param: (
                    'B' if param['subRows'].dtype == np.bool else 'I'
                )),
                TEST.OBJECT     : Partial,
                TEST.NAMINGARGS : dynFormat("Hadamard(%d)->%dx%d,%s%s",
                                            'order', TEST.NUM_N, TEST.NUM_M,
                                            'strIndexTypeM', 'strIndexTypeN')
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
                    Eye(2 * c), N=np.arange(c), M=np.arange(c)))
            },
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c: Partial(
                    Eye(2 ** c), np.arange(2 ** c)))
            }
        }

    def _getDocumentation(self):
        return ""
