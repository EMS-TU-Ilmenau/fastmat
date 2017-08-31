# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False
'''
  fastmat/BlockDiag.py
 -------------------------------------------------- part of the fastmat package

  Block diagonal matrix.


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

  TODO:
    - BlockDiag should simply skip all Zero Matrices (flag them as "None")?
'''
import numpy as np
cimport numpy as np

from .Matrix cimport Matrix
from .core.types cimport *

################################################################################
################################################## class BlockDiag
cdef class BlockDiag(Matrix):

    ############################################## class methods
    def __init__(self, *matrices):
        '''Initialize Matrix instance with a list of child matrices'''
        cdef intsize numN = 0, numM = 0
        cdef Matrix term

        self._content = matrices

        # determine total size and data type of matrix
        dataType = np.int8
        for term in self._content:
            if not isinstance(term, Matrix):
                raise ValueError(
                    "Only fastmat matrices supported, %s given." %(str(term)))

            numN += term.numN
            numM += term.numM
            dataType = np.promote_types(dataType, term.dtype)

        # set properties of matrix
        self._initProperties(
            numN, numM, dataType,
            cythonCall=True,
            widenInputDatatype=True
        )

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        cdef float complexityFwd = self.numN
        cdef float complexityBwd = self.numM
        cdef Matrix item
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
        cdef Matrix term
        cdef intsize idxN = 0, idxM = 0, ii, cnt = len(self._content)

        for ii in range(0, cnt):
            term = self._content[ii]
            arrRes[idxN:(idxN + term.numN), :] \
                = term.forward(arrX[idxM:(idxM + term.numM)])

            idxN += term.numN
            idxM += term.numM

    cpdef _backwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        '''Calculate the backward transform of this matrix'''
        cdef Matrix term
        cdef intsize idxN = 0, idxM = 0, cnt = len(self._content)

        for ii in range(0, cnt):
            term = self._content[ii]

            arrRes[idxM:(idxM + term.numM), :] \
                = term.backward(arrX[idxN:(idxN + term.numN)])

            idxN += term.numN
            idxM += term.numM

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        cdef np.ndarray arrRes
        cdef Matrix term
        cdef intsize idxN = 0, idxM = 0

        arrRes = np.zeros((self.numN, self.numM), dtype=self.dtype)

        for term in self._content:
            arrRes[idxN:(idxN + term.numN), :][:, idxM:(idxM + term.numM)] = \
                term._getArray()

            idxN += term.numN
            idxM += term.numM

        return arrRes

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                'size'          : 5,
                TEST.NUM_N      : (lambda param: param['size'] * 3),
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

    def _getDocumentation(self):
        from .inspect import DOC
        return DOC.SUBSECTION(
            r'Block Diagonal Matrix (\texttt{fastmat.BlockDiag})',
            DOC.SUBSUBSECTION(
                'Definition and Interface',
                r"""
\[\bm M = \mathrm{diag}\left\{\left( \bm A_{i}\right)_{i}\right\},\]
where the $\bm A_{i}$ can be fast transforms of \emph{any} type.""",
                DOC.SNIPPET('# import the package',
                            'import fastmat as fm',
                            '',
                            '# define the blocks',
                            'A = fm.Circulant(x_A)',
                            'B = fm.Circulant(x_B)',
                            'C = fm.Fourier(n)',
                            'D = fm.Diag(x_D)',
                            '',
                            '# define the block',
                            '# diagonal matrix',
                            'M = fm.BlockDiag(A, B, C, D)',
                            caption=r"""
Assume we have two circulant matrices $\bm A$ and $\bm B$, an $N$-dimensional
Fourier matrix $\bm C$ and a diagonal matrix $\bm D$. Then we define
\[\bm M = \left(\begin{array}{cccc}
    \bm A & & & \\
    & \bm B & & \\
    & & \bm C & \\
    & & & \bm D
\end{array}\right).\]"""),
                r"""
Meta types can also be nested, so that a block diagonal matrix can contain
products of block matrices as its entries. Note that the efficiency of the fast
transforms decreases the more building blocks they have.""",
                DOC.SNIPPET('# import the package',
                            'import fastmat as fm',
                            '',
                            '# define the blocks',
                            'A = fm.Circulant(x_A)',
                            'B = fm.Circulant(x_B)',
                            'F = fm.Fourier(n)',
                            'D = fm.Diag(x_D)',
                            '',
                            '# define a product',
                            'P = fm.Product(A.H, B)',
                            '',
                            '# define the block',
                            '# diagonal matrix',
                            'M = fm.BlockDiag(P, F, D)',
                            caption=r"""
Assume we have a product $\bm P$ of two matrices $\bm A^\herm$ and $\bm B$, an
$N$-dimensional Fourier matrix $\bm{\mathcal{F}}$ and a diagonal matrix
$\bm D$. Then we define
\[\bm M = \left(\begin{array}{cccc}
    \bm A^\herm \cdot \bm B &                  &        \\
                            & \bm{\mathcal{F}} &        \\
                            &                  & \bm D
\end{array}\right).\]""")
            ),
            DOC.SUBSUBSECTION(
                'Performance Benchmarks',
                DOC.PLOTFORWARD(doc=r'(Matrix as in snippet)'),
                DOC.PLOTFORWARDMEMORY(doc=r'(same as in snippet)'),
                DOC.PLOTOVERHEAD(doc=r"""
$\bm B = \begin{pmatrix}
    \bm I_{2^k} & \dots     & 0             \\
                & \ddots    &               \\
    0           & \dots     & \bm I_{2^k}
\end{pmatrix}$"""),
            )
        )
