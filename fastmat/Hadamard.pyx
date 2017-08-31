# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False
'''
  fastmat/Hadamard.pyx
 -------------------------------------------------- part of the fastmat package

  Hadamard matrix.


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
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

import numpy as np
cimport numpy as np

from .Matrix cimport Matrix
from .Eye cimport Eye
from .core.types cimport *

# have a very lazy import to avoid initialization of scipy.linalg during import
# of main module
spHadamard = None


################################################################################
################################################## class Hadamard
cdef class Hadamard(Matrix):

    ############################################## class properties
    # order - Property (read-only)
    # Return the order of the hadamard matrix.
    property order:
        def __get__(self):
            return self._order

    ############################################## class methods
    def __init__(self, order):
        '''Initialize Matrix instance'''
        if order < 1:
            raise ValueError("Hadamrd: Order must be larger than 0.")

        self._order = order

        # set properties of matrix
        numN = 2 ** self._order
        self._initProperties(
            numN, numN, np.int8,
            cythonCall=True,
            forceInputAlignment=True,
            fortranStyle=True
        )

    cpdef np.ndarray _getArray(self):
        '''
        Return an explicit representation of the matrix as numpy-array.
        '''
        return self._reference()

    ############################################## class property override
    cpdef object _getLargestEV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return np.sqrt(self.numN)

    cpdef object _getLargestSV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return np.sqrt(self.numN)

    cpdef Matrix _getNormalized(self):
        return Hadamard(self.order) * np.float32(1. / np.sqrt(self.numN))

    cpdef Matrix _getGram(self):
        return Eye(self.numN) * np.float32(self.numN)

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        cdef float complexity = self.numN * self.order
        return (complexity, complexity + 1)

    ############################################## class core methods
    cdef void _coreLoop(
        self,
        TYPE_IN *pIn,
        TYPE_IN *pOut,
        intsize N,
        intsize M
    ):
        cdef intsize jj, mm, halfN = N / 2
        cdef TYPE_IN a, b
        # pointers to a single column (vector)
        cdef TYPE_IN *vecIn
        cdef TYPE_IN *vecOut

        for mm in range(M):

            # pointers to a single column (vector)
            vecIn = &(pIn[mm * N])
            vecOut = &(pOut[mm * N])

            # process one step for a single column-vector
            for jj in range(halfN):
                a = vecIn[jj * 2]
                b = vecIn[jj * 2 + 1]
                vecOut[jj] = a + b
                vecOut[jj + halfN] = a - b

    cdef void _core(
        self,
        np.ndarray arrIn,
        np.ndarray arrOut,
        TYPE_IN typeArr
    ):
        cdef intsize N = arrOut.shape[0], M = arrOut.shape[1]
        cdef intsize mm, nn
        cdef int steps = self._order

        # extract memory pointers from ndarrays passed
        cdef TYPE_IN *pIn = <TYPE_IN *> arrIn.data
        cdef TYPE_IN *pOut = <TYPE_IN *> arrOut.data

        # allocate temporary buffers in memory
        cdef TYPE_IN *pA = <TYPE_IN *> malloc(sizeof(TYPE_IN) * N * M)
        cdef TYPE_IN *pB = <TYPE_IN *> malloc(sizeof(TYPE_IN) * N * M)

        # perform calculation
        if steps == 1:
            self._coreLoop[TYPE_IN](pIn, pOut, N, M)
        else:
            self._coreLoop[TYPE_IN](pIn, pA, N, M)

        steps -= 1

        while (steps >= 2):
            steps -= 2
            self._coreLoop[TYPE_IN](pA, pB, N, M)
            if steps == 0:
                self._coreLoop[TYPE_IN](pB, pOut, N, M)
            else:
                self._coreLoop[TYPE_IN](pB, pA, N, M)

        if steps == 1:
            self._coreLoop[TYPE_IN](pA, pOut, N, M)

        # release memory of temporary buffers
        free(pA)
        free(pB)

    ############################################## class forward / backward
    cpdef _forwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        '''
        Calculate the forward transform of this matrix.
        '''
        # dispatch input ndarray to type specialization
        if typeX == TYPE_FLOAT32:
            self._core[np.float32_t](arrX, arrRes, typeX)
        elif typeX == TYPE_FLOAT64:
            self._core[np.float64_t](arrX, arrRes, typeX)
        elif typeX == TYPE_COMPLEX64:
            self._core[np.complex64_t](arrX, arrRes, typeX)
        elif typeX == TYPE_COMPLEX128:
            self._core[np.complex128_t](arrX, arrRes, typeX)
        elif typeX == TYPE_INT64:
            self._core[np.int64_t](arrX, arrRes, typeX)
        elif typeX == TYPE_INT32:
            self._core[np.int32_t](arrX, arrRes, typeX)
        elif typeX == TYPE_INT8:
            self._core[np.int8_t](arrX, arrRes, typeX)
        else:
            raise NotImplementedError("Hadamard: %d not supported." %(typeX))

    cpdef _backwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        '''
        Calculate the backward transform of this matrix.
        '''
        self._forwardC(arrX, arrRes, typeX, typeRes)

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        global spHadamard
        if spHadamard is None:
            spHadamard = __import__('scipy.linalg', globals(), locals(),
                                    ['hadamard']).hadamard

        return spHadamard(self.numN, dtype=self.dtype)

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                # define matrix sizes and parameters
                'order'         : TEST.Permutation([4, 6]),
                TEST.NUM_N      : (lambda param : 2 ** param['order']),
                TEST.NUM_M      : TEST.NUM_N,

                # define constructor for test instances and naming of test
                TEST.OBJECT     : Hadamard,
                TEST.INITARGS   : ['order'],
                TEST.NAMINGARGS : dynFormat("%d", 'order')
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c: Hadamard(c)),
                BENCH.FUNC_SIZE : (lambda c: 2 ** c),
                BENCH.FUNC_STEP : (lambda c: c + 1),
            },
            BENCH.FORWARD: {},
            BENCH.SOLVE: {},
            BENCH.OVERHEAD: {},
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, datatype: Hadamard(c))
            }
        }

    def _getDocumentation(self):
        from .inspect import DOC
        return DOC.SUBSECTION(
            r'Hadamard Matrix (\texttt{fastmat.Hadamard})',
            DOC.SUBSUBSECTION(
                'Definition and Interface', r"""
A Hadamard Matrix is recursively defined as
    \[\bm H_n = \bm H_1 \otimes \bm H_{n-1},\]
where
    \[\bm H_1 = \left(\begin{array}{cc} 1 & 1 \\ 1 & -1 \end{array}\right)\]
and $\bm H_0 = (1)$. Obviously the dimension of $\bm H_n$ is $2^n$. The
transform is realized with the Fast Hadamard Transform (FHT).""",
                DOC.SNIPPET('# import the package',
                            'import fastmat as fm',
                            '',
                            '# define the parameter',
                            'n = 4',
                            '',
                            '# construct the matrix',
                            'H = fm.Hadamard(n)',
                            caption=r"""
This yields a Hadamard matrix $\bm{\mathcal{H}}_4$ of order $4$,
 i.e. with $16$ rows and columns."""),
                r"""
The algorithm we used is described in \cite{hada_hershey1997hadamard} and was
implemented in Cython \cite{hada_smith2011cython}."""
            ),
            DOC.SUBSUBSECTION(
                'Performance Benchmarks', r"""
All benchmarks were performed on a matrix
$\bm \Hs_k$ and $n = 2^k$ with $n, k \in \N$""",
                DOC.PLOTFORWARD(),
                DOC.PLOTFORWARDMEMORY(),
                DOC.PLOTSOLVE(),
                DOC.PLOTOVERHEAD(),
                DOC.PLOTTYPESPEED(),
                DOC.PLOTTYPEMEMORY()
            ),
            DOC.BIBLIO(
                hada_hershey1997hadamard=DOC.BIBITEM(
                    r'Rao K. Yarlagadda, John E. Hershey',
                    r"""
Hadamard Matrix Analysis and Synthesis, With Applications to Communications
and Signal/Image Processing""",
                    r"""
The Springer International Series in Engineering and Computer Science,
Volume 383, 1997"""),
                hada_smith2011cython=DOC.BIBITEM(
                    r"""
Stefan Behnel, Robert Bradshaw, Craig Citro, Lisandro Dalcin,
Dag Sverre Seljebotn and Kurt Smith""",
                    r'Cython: The Best of Both Worlds',
                    r'Computing in Science and Engineering, Volume 13,2011.')
            )
        )
