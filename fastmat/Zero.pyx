# -*- coding: utf-8 -*-
'''
  fastmat/Zero.pyx
 -------------------------------------------------- part of the fastmat package

  All-Zero matrix.


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
cimport numpy as np

from .Matrix cimport Matrix
from .helpers.cmath cimport _arrZero
from .helpers.types cimport *

################################################################################
################################################## class Zero
cdef class Zero(Matrix):

    ############################################## class methods
    def __init__(self, numN, numM):
        '''Initialize Matrix instance with its dimensions'''
        # set properties of matrix
        self._initProperties(numN, numM, np.int8)

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        return _arrZero(1, self.numN, 1, self._info.dtype[0].typeNum)

    cpdef np.ndarray _getRow(self, intsize idx):
        return _arrZero(1, self.numM, 1, self._info.dtype[0].typeNum)

    cpdef object _getLargestEV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return 0.

    cpdef object _getLargestSV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return 0.

    cpdef Matrix _getT(self):
        return Zero(self.numM, self.numN)

    cpdef Matrix _getH(self):
        return Zero(self.numM, self.numN)

    cpdef Matrix _getConj(self):
        return self

    cpdef Matrix _getGram(self):
        return Zero(self.numM, self.numM)

    cpdef Matrix _getNormalized(self):
        raise ValueError("A Zero Matrix cannot be normalized.")

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        return 0

    cpdef np.ndarray toarray(self):
        '''
        Return an explicit representation of the matrix as numpy-array.
        '''
        return _arrZero(2, self.numN, self.numM, self._info.dtype[0].typeNum)

    ############################################## class forward / backward
    cpdef np.ndarray _forward(self, np.ndarray arrX):
        '''Calculate the forward transform of this matrix'''
        return _arrZero(
            arrX.ndim, self.numN, arrX.shape[1] if arrX.ndim > 1 else 1,
            _getNpType(arrX))

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        '''Calculate the backward transform of this matrix'''
        return _arrZero(
            arrX.ndim, self.numM, arrX.shape[1] if arrX.ndim > 1 else 1,
            _getNpType(arrX))

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        return np.zeros((self.numN, self.numM), dtype=self.dtype)


################################################################################
################################################################################
from .helpers.unitInterface import *

################################################## Testing
test = {
    NAME_COMMON: {
        TEST_NUM_N: 35,
        TEST_NUM_M: Permutation([30, TEST_NUM_N]),
        TEST_OBJECT: Zero,
        TEST_INITARGS: [TEST_NUM_N, TEST_NUM_M],
        TEST_NAMINGARGS: dynFormatString("%dx%d", TEST_NUM_N, TEST_NUM_M)
    },
    TEST_CLASS: {
        # test basic class methods
    },
    TEST_TRANSFORMS: {
        # test forward and backward transforms
    }
}

################################################## Benchmarks
benchmark = {
    NAME_COMMON: {
        NAME_DOCU       : r'''$\bm M = \bm 0_{2^k \times 2^k}$;
            so $n = 2^k$ for $k \in \N$''',
        BENCH_FUNC_GEN  : (lambda c : Zero(c, c)),
    },
    BENCH_FORWARD: {
    },
    BENCH_OVERHEAD: {
        BENCH_FUNC_GEN  : (lambda c : Zero(2 ** c, 2 ** c)),
    }
}

################################################## Documentation
docLaTeX = r"""
\subsection{Zero Transform (\texttt{fastmat.Zero})}
\subsubsection{Definition and Interface}
    \[\bm x \mapsto \bm 0\]
The zero matrix only needs the dimension $n$ of the vectors it acts on. It is
very fast and very good!

\begin{snippet}
\begin{lstlisting}[language=Python]
# import the package
import fastmat as fm

# define the parameter
n = 10

# construct the matrix
O = fm.Zero(n)
\end{lstlisting}

This yields something trivial.
\end{snippet}
"""
