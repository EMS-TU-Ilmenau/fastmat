# -*- coding: utf-8 -*-
'''
  fastmat/Eye.pyx
 -------------------------------------------------- part of the fastmat package

  Eye matrix.
ii

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
from .helpers.types cimport *
from .helpers.cmath cimport _arrZero

################################################################################
################################################## class Eye
cdef class Eye(Matrix):

    ############################################## class methods
    def __init__(self, numN):
        '''
        Initialize Matrix instance with its dimensions.

        Generated a [numN x numM] matrix of small integers
        '''
        self._initProperties(numN, numN, np.int8)

    cpdef np.ndarray toarray(Eye self):
        '''
        Return an explicit representation of the matrix as numpy-array.
        '''
        return np.eye(self.numN, dtype=self.dtype)

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        cdef np.ndarray arrRes

        arrRes = _arrZero(1, self.numN, 1, self._info.dtype[0].typeNum)
        arrRes[idx] = 1

        return arrRes

    cpdef np.ndarray _getRow(self, intsize idx):
        return self._getCol(idx)

    cpdef object _getLargestSV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return 1.

    cpdef object _getLargestEV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return 1.

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        return 1 if (idxN == idxM) else 0

    cpdef Matrix _getNormalized(self):
        return self

    cpdef Matrix _getGram(self):
        return self

    cpdef Matrix _getT(self):
        return self

    cpdef Matrix _getH(self):
        return self

    cpdef Matrix _getConj(self):
        return self

    ############################################## class forward / backward
    cpdef np.ndarray _forward(Eye, np.ndarray arrX):
        '''Calculate the forward transform of this matrix'''
        return arrX

    cpdef np.ndarray _backward(Eye self, np.ndarray arrX):
        '''Calculate the backward transform of this matrix'''
        return arrX

    ############################################## class reference
    cpdef np.ndarray _reference(Eye self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        return np.eye(self.numN, dtype=self.dtype)


################################################################################
################################################################################
from .helpers.unitInterface import *

################################################## Testing
test = {
    NAME_COMMON: {
        TEST_NUM_N      : 35,
        TEST_NUM_M      : TEST_NUM_N,
        TEST_OBJECT     : Eye,
        TEST_INITARGS   : [TEST_NUM_N]
    },
    TEST_CLASS: {
        # test basic class functions
    },
    TEST_TRANSFORMS: {
        # test
    }
}


################################################## Benchmarks
benchmark = {
    NAME_COMMON: {
        BENCH_FUNC_GEN  : (lambda c : Eye(c)),
        NAME_DOCU       : r'$\bm I_n$'
    },
    BENCH_FORWARD: {
    },
    BENCH_SOLVE: {
        BENCH_FUNC_GEN  : (lambda c : Eye(c))
    },
    BENCH_OVERHEAD: {
        BENCH_FUNC_GEN  : (lambda c : Eye(2 ** c)),
        NAME_DOCU       : r'$\bm I_n$ with $n = 2^k$ for $k \in \N$'
    }
}


################################################## Documentation
docLaTeX = r"""
\subsection{Identity Transform (\texttt{fastmat.Eye})}
\subsubsection{Definition and Interface}
For $\bm x \in \C^n$ we map
\[\bm x \mapsto \bm x.\]
The identity matrix only needs the dimension $n$ of the vectors it acts on.

\begin{snippet}
\begin{lstlisting}[language=Python]
# import the package
import fastmat

# set the parameter
n = 10

# construct the
# identity
I = fastmat.Eye(n)
\end{lstlisting}

This yields the identity matrix $\bm I_{10}$ with dimension $10$.
\end{snippet}
"""
