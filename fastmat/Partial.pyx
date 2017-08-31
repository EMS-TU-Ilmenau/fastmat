# -*- coding: utf-8 -*-
'''
  fastmat/Partial.py
 -------------------------------------------------- part of the fastmat package

  Partial matrices.


  Author      : wcw, sempersn
  Introduced  : 2016-04-08
 ------------------------------------------------------------------------------

   Copyright 2016 Sebastian Semper, Christoph Wagner
       https://www.tu-ilmenau.de/ems/

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

 ------------------------------------------------------------------------------
'''
import numpy as np
cimport numpy as np

from .Matrix cimport Matrix
from .core.types cimport *
from .core.cmath cimport _arrZero

################################################################################
################################################## class Partial
cdef class Partial(Matrix):

    ############################################## class properties
    # indicesN, indicesM - Property (read-only)
    # Define the support of the base matrix which defines the partial.
    property indicesN:
        def __get__(self):
            return self._indicesN

    property indicesM:
        def __get__(self):
            return self._indicesM

    ############################################## class methods
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
        self._indicesM = np.arange(mat.numM)
        self._indicesN = np.arange(mat.numN)
        self._content = (mat, )

        # check if anything needs to be done in N- or M-dimension
        # store support indices in N- and M- dimension if needed
        self._pruneN = False
        if N is not None:
            N = np.array(N)
            if (len(N) != mat.numN) or (np.sum(N - self._indicesN) != 0):
                if np.any((N >= mat.numN) | (N < 0)):
                    raise ValueError(
                        "Partial: A row index exceed matrix dimensions.")

                self._indicesN = N
                self._pruneN = True

        self._pruneM = False
        if M is not None:
            M = np.array(M)
            if (len(M) != mat.numM) or (np.sum(M - self._indicesM) != 0):
                if np.any((M >= mat.numM) | (M < 0)):
                    raise ValueError(
                        "Partial: A column index exceeds matrix dimensions.")

            self._indicesM = M
            self._pruneM = True

        # set properties of matrix
        self._initProperties(
            len(self._indicesN), len(self._indicesM), mat.dtype)

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
                2, self._content[0].numM, arrX.shape[1], _getNpType(arrX))
            arrInput[self._indicesM, :] = arrX
        else:
            arrInput = arrX

        return (self._content[0].forward(arrInput)[self._indicesN, :]
                if self._pruneN else self._content[0].forward(arrInput))

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        '''Calculate the backward transform of this matrix'''

        cdef np.ndarray arrInput

        if self._pruneN:
            arrInput = _arrZero(
                2, self._content[0].numN, arrX.shape[1], _getNpType(arrX))
            arrInput[self._indicesN, :] = arrX
        else:
            arrInput = arrX

        return (self._content[0].backward(arrInput)[self._indicesM, :]
                if self._pruneM else self._content[0].backward(arrInput))

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        cdef np.ndarray arrFull = self._content[0].reference()
        return arrFull[self._indicesN, :][:, self._indicesM]

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                'num_N'         : 15,
                'num_M'         : TEST.Permutation([20, 'num_N']),
                TEST.NUM_N      : (lambda param: len(param['subRows'])),
                TEST.NUM_M      : (lambda param: len(param['subCols'])),

                'mType'         : TEST.Permutation(TEST.ALLTYPES),
                'arrM'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mType',
                    TEST.SHAPE  : ('num_N', 'num_M')
                }),

                'subCols'       : TEST.Permutation([np.array([1, 2, 3, 11, 12]),
                                                    np.array([6])]),
                'subRows'       : TEST.Permutation([np.array([7, 8, 9, 13]),
                                                    np.array([10])]),
                TEST.INITARGS   : (lambda param: [Matrix(param['arrM']()),
                                                  param['subRows'],
                                                  param['subCols']]),
                TEST.OBJECT     : Partial,
                TEST.NAMINGARGS : dynFormat("%s,%dx%d",
                                            'arrM', TEST.NUM_N, TEST.NUM_M)
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
                BENCH.FUNC_GEN  : (lambda c:
                                   Partial(Eye(2 ** c), np.arange(2 ** c)))
            }
        }

    def _getDocumentation(self):
        from .inspect import DOC
        return DOC.SUBSECTION(
            r'Partial Transform (\texttt{fastmat.Partial})',
            DOC.SUBSUBSECTION(
                'Definition and Interface', r"""
Let $I \subset \{1,\dots,n\}$ and $J \subset \{1,\dots,m\}$ index sets and
$\bm M \in \C^{n \times m}$ a linear transform. Then the partial transform
$\bm M_{I,J}$ is defined as
\[\bm x \in \C^m \mapsto (\bm M_J \cdot \bm x_J)_{i \in I}.\]
In other words, we select the rows $I$ of $\bm M$ and columns J of $\bm M$ and
rows $J$ of $\bm x$.""",
                DOC.SNIPPET('# import the package',
                            'import fastmat as fm',
                            'import numpy as np',
                            '',
                            '# define the index set',
                            'a = np.arange(n)',
                            'am = np.mod(a, 2)',
                            "b = np.array(am, dtype='bool')",
                            'I = a[b]',
                            '',
                            '# construct the partial transform',
                            'M = fm.Partial(F, I)',
                            caption=r"""
Let $\bm{\mathcal{F}}$ be the $n$-dimensional Fourier matrix. And let $I$ be the
set of odd integers. Then we define a partial transform as
\[\bm M = \bm{\mathcal{F}}_I\]""")
            ),
            DOC.SUBSUBSECTION(
                'Performance Benchmarks', r"""
The forward projection benchmarks were performed on a partial Hadamard matrix
$\bm P = \bm \Hs_{2^n,\{1,\dots,n\}}$ while the runtime performance benchmark
was performed on a partial Identity matrix.""",
                DOC.PLOTFORWARD(),
                DOC.PLOTFORWARDMEMORY(),
                DOC.PLOTOVERHEAD()
            )
        )
