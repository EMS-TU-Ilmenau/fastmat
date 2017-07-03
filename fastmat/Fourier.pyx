# -*- coding: utf-8 -*-
'''
  fastmat/Fourier.pyx
 -------------------------------------------------- part of the fastmat package

  Fourier matrix.


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
from .helpers.cmath cimport _conjugate, _conjugateInplace

################################################################################
################################################## class Fourier
cdef class Fourier(Matrix):

    ############################################## class properties
    # num_order - Property (read-only)
    # Return the Order of the fourier matrix.
    property order:
        def __get__(self):
            return self._order

    ############################################## class methods
    def __init__(self, order):
        '''Initialize Matrix instance with a list of child matrices'''

        self._order = order

        # set properties of matrix
        numN = self._order
        self._initProperties(numN, numN, np.complex128)

    cpdef np.ndarray toarray(self):
        '''
        Return an explicit representation of the matrix as numpy-array.
        '''
        return np.fft.fft(np.eye(self._order, dtype=self.dtype), axis=0)

    ############################################## class property override
    cpdef np.ndarray _getRow(self, intsize idx):
        return self._getCol(idx)

    cpdef object _getLargestSV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return np.sqrt(self._order)

    cpdef object _getLargestEV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return np.sqrt(self._order)

    cpdef Matrix _getNormalized(self):
        return Fourier(self._order) * (1. / np.sqrt(self._order))

    cpdef Matrix _getGram(self):
        return Eye(self._order) * self.dtype(self._order)

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        return np.exp(idxN * idxM * -2j * np.pi / self.order).astype(self.dtype)

    ############################################## class forward / backward
    cpdef np.ndarray _forward(self, np.ndarray arrX):
        '''Calculate the forward transform of this matrix'''
        cdef np.ndarray arrRes = np.fft.fft(arrX, axis=0)
        return arrRes

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        '''Calculate the backward transform of this matrix'''
        cdef np.ndarray arrRes = np.fft.fft(_conjugate(arrX), axis=0)
        _conjugateInplace(arrRes)
        return arrRes

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        return np.exp(
            np.multiply(
                *np.meshgrid(np.arange(self._order), np.arange(self._order)
                             )) * -2j * np.pi / self._order
        ).astype(self.dtype)


################################################################################
################################################################################
from .helpers.unitInterface import *

################################################### Testing
test = {
    NAME_COMMON: {
        # define matrix sizes and parameters
        TEST_TOL_POWER: 3.,
        TEST_NUM_N: 35,
        TEST_NUM_M: TEST_NUM_N,

        # define constructor for test instances and naming of test
        TEST_OBJECT: Fourier,
        TEST_INITARGS: [TEST_NUM_N],
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
        BENCH_FUNC_GEN  : (lambda c : Fourier(c)),
        NAME_DOCU       : r'$\bm \Fs_n$'
    },
    BENCH_FORWARD: {
    },
    BENCH_SOLVE: {
    },
    BENCH_OVERHEAD: {
        BENCH_FUNC_GEN  : (lambda c : Fourier(2 ** c)),
        NAME_DOCU       : r'$\bm \Fs_n$ with $n = 2^k$ for $k \in \N$'
    },
    BENCH_DTYPES: {
        BENCH_FUNC_GEN  : (lambda c, datatype : Fourier(2 ** c)),
        NAME_DOCU       : r'$\bm \Fs_n$ with $n = 2^k$ for $k \in \N$'
    }
}


################################################## Documentation
docLaTeX = r"""
\subsection{Fourier Transform (\texttt{fastmat.Fourier})}
\subsubsection{Definition and Interface}
The Fourier Transform realizes the mapping
\[\bm x \mapsto \bm F_n \cdot \bm x,\]
where the Fourier matrix $\bm F_n$ is uniquely defined by the size of the
vectors it acts on.

\begin{snippet}
\begin{lstlisting}[language=Python]
# import the package
import fastmat as fm

# define parameter
n = 4

# construct the matrix
F = fm.Fourier(n)
\end{lstlisting}

This yields a Fourier $\bm{\mathcal{F}}_4$ matrix of size $4$.
\end{snippet}

As a library to provide the Fast Fourier Transform we used the one provided by
NumPy \cite{four_walt2011numpy}.

\begin{thebibliography}{9}
\bibitem{four_walt2011numpy}
St\'efan van der Walt, S. Chris Colbert and Ga\"el Varoquaux
\emph{The NumPy Array: A Structure for Efficient Numerical Computation},
Computing in Science and Engineering,
Volume 13,
2011.
\end{thebibliography}
"""
