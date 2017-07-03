# -*- coding: utf-8 -*-
'''
  fastmat/Diag.pyx
 -------------------------------------------------- part of the fastmat package

  Diagonal matrix.


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
from .Eye cimport Eye
from .helpers.cmath cimport _conjugate, _multiply, _arrZero
from .helpers.types cimport *


################################################################################
################################################## class Diag
cdef class Diag(Matrix):

    ############################################## class properties
    # vecD - Property (read-only)
    # Return the matrix-defining vector of diagonal entries.
    property vecD:
        def __get__(self):
            return self._vecD

    ############################################## class methods
    def __init__(self, vecD):
        '''Initialize Matrix instance with a list of child matrices'''
        # numN is size of matrix (and of diagonal vector)
        numN = len(vecD)

        # store diagonal entry vector as copy of vecD and complain if
        # dimension does not match
        self._vecD = np.atleast_1d(np.squeeze(np.copy(vecD)))
        if self._vecD.ndim != 1:
            raise ValueError(
                "Diag-defining vector must have exactly one active dimension.")

        # set properties of matrix
        self._initProperties(
            numN, numN, self._vecD.dtype,
            cythonCall=True,
            forceInputAlignment=True
        )

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        cdef np.ndarray arrRes

        arrRes = _arrZero(1, self.numN, 1, self._info.dtype[0].typeNum)
        arrRes[idx] = self._vecD[idx]

        return arrRes

    cpdef object _getLargestEV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return np.abs(self._vecD).max().astype(np.float64)

    cpdef object _getLargestSV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return np.abs(self._vecD).max().astype(np.float64)

    cpdef np.ndarray _getRow(self, intsize idx):
        return self._getCol(idx)

    cpdef Matrix _getGram(self):
        return Diag(np.abs(self._vecD) ** 2)

    cpdef Matrix _getNormalized(self):
        if self._info.dtype.isComplex:
            return Diag((self._vecD / abs(self._vecD)).astype(self.dtype))
        else:
            return Diag(np.sign(self._vecD).astype(self.dtype))

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        return self._vecD[idxN] if idxN == idxM else self.dtype(0)

    ############################################## class forward / backward
    cpdef _forwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        ''' Calculate the forward transform of this matrix.'''
        _multiply(
            arrX, self._vecD, arrRes,
            typeX, self._info.dtype[0].fusedType, typeRes)

    cpdef _backwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        ''' Calculate the backward transform of this matrix.'''
        _multiply(
            arrX, _conjugate(self._vecD), arrRes,
            typeX, self._info.dtype[0].fusedType, typeRes)

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        cdef intsize ii, N = self.numN
        cdef np.ndarray d = self.vecD

        return np.diag(d)


################################################################################
################################################################################
from .helpers.unitInterface import *
################################################### Testing
test = {
    NAME_COMMON: {
        TEST_NUM_N: 35,
        TEST_NUM_M: TEST_NUM_N,
        'mTypeD': Permutation(typesAll),
        TEST_PARAMALIGN : Permutation(alignmentsAll),
        'vecD': ArrayGenerator({
            NAME_DTYPE  : 'mTypeD',
            NAME_SHAPE  : (TEST_NUM_N, ),
            NAME_ALIGN  : TEST_PARAMALIGN
            #            NAME_CENTER : 2,
        }),
        TEST_INITARGS: (lambda param : [
            param['vecD']()
        ]),
        TEST_OBJECT: Diag,
        TEST_NAMINGARGS: dynFormatString("%s", 'vecD')
    },
    TEST_CLASS: {
        # test basic class methods
    }, TEST_TRANSFORMS: {
        # test forward and backward transforms
    }
}


################################################## Benchmarks
benchmark = {
    NAME_COMMON: {
        NAME_DOCU       : r'''$\bm D \in \R^{n \times n}$ with diagonal
            entries drawn from a uniform distribution on $[2,3]$''',
        BENCH_FUNC_GEN  : (lambda c : Diag(np.random.uniform(2, 3, c)))
    },
    BENCH_FORWARD: {
    },
    BENCH_SOLVE: {
    },
    BENCH_OVERHEAD: {
        BENCH_FUNC_GEN  :
            (lambda c : Diag(np.random.uniform(2, 3, 2 ** c))),
        NAME_DOCU       :
            r'''$\bm D \in \R^{n \times n}$ with $n = 2^k$, $k \in \N$ and
            diagonal entries drawn from a uniform distribution on $[2,3]$'''
    },
    BENCH_DTYPES: {
        BENCH_FUNC_GEN:
            (lambda c, dt : Diag(np.random.uniform(2, 3, 2 ** c).astype(dt))),
        NAME_DOCU       :
            r'''$\bm D \in \R^{n \times n}$ with $n = 2^k$, $k \in \N$ and
            diagonal entries drawn from a uniform distribution on $[2,3]$'''
    }
}


################################################## Documentation
docLaTeX = r"""
\subsection{Diagonal Matrix (\texttt{fastmat.Diagonal})}
\subsubsection{Definition and Interface}
\[x \mapsto \bm{\mathrm{diag}}(d_1,\dots,d_n) \cdot \bm x\]
A diagonal matrix is uniquely defined by the entries of its diagonal.

\begin{snippet}
\begin{lstlisting}[language=Python]
# import the package
import fastmat as fm
import numpy as np

# build the parameters
n = 4
d = np.array([1, 0, 3, 6])

# construct the matrix
D = fm.Diagonal(d)
\end{lstlisting}

This yields
\[\bm d = (1, 0, 3, 6)^T\]
\[\bm D = \left(\begin{array}{cccc}
    1 & & & \\
    & 0 & & \\
    & & 3 & \\
    & & & 6
\end{array}\right)\]
\end{snippet}
"""
