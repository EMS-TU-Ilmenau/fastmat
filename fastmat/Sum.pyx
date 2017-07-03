# -*- coding: utf-8 -*-
'''
  fastmat/Sum.py
 -------------------------------------------------- part of the fastmat package

  Linear combination of matrices.


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

from .helpers.types cimport *
from .Matrix cimport Matrix

################################################################################
################################################## class Sum
cdef class Sum(Matrix):

    ############################################## class properties
    # content - Property (read-only)
    # List the content in the linear combination as tuples
    property content:
        def __get__(self):
            return list(self._content)

    ############################################## class methods
    def __init__(self, *matrices):
        '''
        Initialize Matrix instance with a list of other matrices to be summed.
        If another Sum is seen, add its content instead of adding the Sum.
        '''
        cpdef Matrix mat

        # Fold multiple levels of sums
        lstTerms = []

        def __addTerms(matrices):
            for mat in matrices:
                if not isinstance(mat, Matrix):
                    raise TypeError("Sum contains non-fastmat-matrix terms")

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
            raise ValueError("No sum terms specified")

        # check for matching transform dimensions
        cdef intsize numN = self._content[0].numN
        cdef intsize numM = self._content[0].numM
        for ii in range(1, cntTerms):
            mat = self._content[ii]
            if (mat.numN != numN) or (mat.numM != numM):
                raise ValueError("Mismatch in term dimensions: " + repr(mat))

        # determine data type of sum result
        dataType = np.int8
        for ii in range(0, cntTerms):
            dataType = np.promote_types(dataType, self._content[ii].dtype)

        # set properties of matrix
        self._initProperties(
            numN, numM, dataType,
            cythonCall=True,
            widenInputDatatype=True
        )

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

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        cdef int cc, cnt = len(self._content)

        result = self._content[0]._getItem(idxN, idxM).astype(self.dtype)
        for cc in range(1, cnt):
            result += self._content[cc]._getItem(idxN, idxM)

        return result

    ############################################## class forward / backward
    cpdef _forwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        '''Calculate the forward transform of this matrix'''
        cdef int cc, cnt = len(self._content)

        arrRes[:] = self._content[0].forward(arrX)
        for cc in range(1, cnt):
            arrRes += self._content[cc].forward(arrX)

        return arrRes

    cpdef _backwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        '''Calculate the backward transform of this matrix'''
        cdef int cc, cnt = len(self._content)

        arrRes[:] = self._content[0].backward(arrX)
        for cc in range(1, cnt):
            arrRes += self._content[cc].backward(arrX)

        return arrRes

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        cdef np.ndarray arrRes
        cdef int cc, cnt = len(self._content)

        arrRes = np.zeros((self.numN, self.numM), dtype=self.dtype)
        for cc in range(cnt):
            arrRes += self._content[cc].reference()

        return arrRes


################################################################################
################################################################################
from .helpers.unitInterface import *

test = {
    NAME_COMMON: {
        TEST_NUM_N: 25,
        TEST_NUM_M: Permutation([17, TEST_NUM_N]),
        'mType1': Permutation(typesAll),
        'mType2': Permutation(typesSmallIFC),
        'arrM1': ArrayGenerator({
            NAME_DTYPE  : 'mType1',
            NAME_SHAPE  : (TEST_NUM_N, TEST_NUM_M)
            #            NAME_CENTER : 2,
        }),
        'arrM2': ArrayGenerator({
            NAME_DTYPE  : 'mType2',
            NAME_SHAPE  : (TEST_NUM_N, TEST_NUM_M)
            #            NAME_CENTER : 2,
        }),
        'arrM3': ArrayGenerator({
            NAME_DTYPE  : 'mType1',
            NAME_SHAPE  : (TEST_NUM_N, TEST_NUM_M)
            #            NAME_CENTER : 2,
        }),
        TEST_INITARGS: [
            lambda param : Matrix(param['arrM1']()),
            lambda param : Matrix(param['arrM2']()),
            lambda param : Matrix(param['arrM3']())
        ],
        TEST_OBJECT: Sum,
        TEST_NAMINGARGS: dynFormatString("%s+%s+%s", 'arrM1', 'arrM2', 'arrM3'),
        TEST_TOL_POWER: 3.
    },
    TEST_CLASS: {
        # test basic class methods
    }, TEST_TRANSFORMS: {
        # test forward and backward transforms

    }
}

################################################## Benchmarks
from .Eye import Eye
from .Fourier import Fourier
from .Circulant import Circulant


benchmark = {
    BENCH_FORWARD: {
        BENCH_FUNC_GEN  : (lambda c : Sum(
            Fourier(c), Eye(c), Circulant(np.random.randn(c))
        )),
        BENCH_FUNC_SIZE : (lambda c : c),
        NAME_DOCU       : r'$\bm L = \bm \Fs_{k} + \bm I_{k} + \bm \C_{k}$'
    },
    BENCH_SOLVE: {
        BENCH_FUNC_GEN  : (lambda c : Sum(
            Fourier(c), Eye(c), Circulant(np.random.randn(c))
        )),
        BENCH_FUNC_SIZE : (lambda c : c),
        NAME_DOCU       : r'$\bm L = \bm \Fs_{k} + \bm I_{k} + \bm \C_{k}$'
    },
    BENCH_OVERHEAD: {
        BENCH_FUNC_GEN  : (lambda c : Sum(*([Eye(2 ** c)] * 100))),
        BENCH_FUNC_SIZE : (lambda c : 2 ** c),
        NAME_DOCU       : r'''$\bm L = \sum\limits_{i = 1}^{100}
                 \bm I_{2^k}$; so $n = 2^k$ for $k \in \N$'''
    }
}

################################################## Documentation
docLaTeX = r"""
\subsection{Sum of Matrices (\texttt{fastmat.Sum})}
\subsubsection{Definition and Interface}
For matrices $\bm A_k \in \C^{n \times m}$ with $k = 1,\dots,N$ we define a new
mapping $\bm M$ as the sum \[\bm M = \Sum{k = 1}{N}{\bm A_k},\] which then also
is a mapping in $\C^{n \times m}$.

\begin{snippet}
\begin{lstlisting}[language=Python]
# import the package
import fastmat as fm

# define the components
A = fm.Circulant(x_A)
B = fm.Circulant(x_B)
C = fm.Fourier(n)
D = fm.Diag(x_D)

# construct the sum of transformations
M = fm.Sum(A, B, C, D)
\end{lstlisting}

Assume we have two circulant matrices $\bm A$ and $\bm B$, an $N$-dimensional
Fourier matrix $\bm C$ and a diagonal matrix $\bm D$. Then we define
\[\bm M = \bm A + \bm B + \bm C + \bm D.\]
\end{snippet}
"""
