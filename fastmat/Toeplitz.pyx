# -*- coding: utf-8 -*-
'''
  fastmat/Toeplitz.py
 -------------------------------------------------- part of the fastmat package

  Toeplitz matrix. The first column is vecC1, the first row beginning at the
  second element is vecC2. We simply use a special circulant matrix to simulate
  a Toeplitz matrix. Toeplitz matrices can be embedded into larger circulant
  matrices. These are diagonal in the fourier domain and hence their multipli-
  cation with a vector can be carried out efficiently.


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
from .helpers.cmath cimport _arrEmpty
from .Partial cimport Partial
from .Product cimport Product
from .Fourier cimport Fourier
from .Diag cimport Diag


################################################################################
################################################## class Toeplitz
cdef class Toeplitz(Partial):

    ############################################## class properties
    # vecC - Property (read-only)
    # Return the column-defining vector of Toeplitz matrix.
    property vecC:
        def __get__(self):
            return self._vecC

    # vecR - Property (read-only)
    # Return the row-defining vector of Toeplitz matrix.
    property vecR:
        def __get__(self):
            return self._vecR

    ############################################## class methods
    def __init__(self, vecC, vecR, **options):
        '''
        Initialize Toeplitz Matrix instance.

        The toeplitz matrix is embedded into a circulant matrix. See class
        Circulant for further details.

        Toeplitz([ c C C C C ], [ r R R R ]) represents this matrix:
          [ c r R R R ]
          [ C c r R R ]
          [ C C c r R ]
          [ C C C c r ]
          [ C C C C c ]

        The Circulant it is derived from has the following generator:
          [ c C C C C R R R r ] [9 x 9]

        ...and may be zero-padded to a [16 x 16] Matrix for efficiency:
          [ c C C C C 0 0 0 0 0 0 0 R R R r]

        Valid options
        -------------
         'pad'=[false TRUE]
            perform zero-padding of Circulant for efficiency

        All options specified will also be passed on to the geneation of the
        underlying Product instance initialization.
        '''

        self._vecC = vecC
        self._vecR = vecR

        # evaluate options passed to class
        pad = options.get('pad', True)

        # perform padding (if enabled) and generate vector
        cdef intsize size = len(vecC) + len(vecR)
        cdef intsize N = 2 ** np.ceil(np.log2(size))
        if pad and (N > size):
            dataType = np.promote_types(self._vecC.dtype, self._vecR.dtype)
            vec = np.concatenate([
                self._vecC,
                np.zeros((N - size,), dtype=dataType),
                np.flipud(self._vecR)
            ])
        else:
            N = size
            vec = np.concatenate([self._vecC, np.flipud(self._vecR)])

        # Describe as circulant matrix with product of data and vector
        # in fourier domain. Both fourier matrices cause scaling of the
        # data vector by N, which will be compensated in Diag().

        # Create inner product
        cdef Fourier FN = Fourier(N)
        cdef Product P = Product(FN.H, Diag(np.fft.fft(vec) / N), FN, **options)

        # initialize Partial of Product
        super(Toeplitz, self).__init__(
            P, N=np.arange(len(self._vecC)), M=np.arange(len(self._vecR) + 1))

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        cdef intsize N = self.numN
        cdef np.ndarray arrRes

        if idx == 0:
            return self._vecC
        elif idx >= N:
            # double slicing needed, otherwise fail when M = N + 1
            return self._vecR[idx - N:idx][::-1]
        else:
            arrRes = _arrEmpty(1, N, 0, self._info.dtype[0].typeNum)
            arrRes[:idx] = self._vecR[idx - 1::-1]
            arrRes[idx:] = self._vecC[:N - idx]
            return arrRes

    cpdef np.ndarray _getRow(self, intsize idx):
        cdef intsize M = self.numM
        cdef np.ndarray arrRes

        if idx >= M - 1:
            # double slicing needed, otherwise fail when N = M + 1
            return self._vecC[idx - M + 1:idx + 1][::-1]
        else:
            arrRes = _arrEmpty(1, M, 0, self._info.dtype[0].typeNum)
            arrRes[:idx + 1] = self._vecC[idx::-1]
            arrRes[idx + 1:] = self._vecR[:M - 1 - idx]
            return arrRes

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        cdef intsize distance = idxN - idxM
        return (self._vecR[-distance - 1] if distance < 0
                else self._vecC[distance])

    cpdef np.ndarray toarray(self):
        '''
        Return an explicit representation of the matrix as numpy-array.
        '''
        return self._reference()

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        _reference borrowing from Partial is too slow. Therefore,
        construct a reference directly from the vectors.
        '''
        cdef intsize ii, N = self.numN, M = self.numM
        cdef np.ndarray arrRes = np.empty((N, M), dtype=self.dtype)

        # put columns in lower-triangular part of matrix
        for ii in range(0, min(N, M)):
            arrRes[ii:N, ii] = self._vecC[0:(N - ii)]

        # put rows in upper-triangular part of matrix
        for ii in range(0, min(N, M - 1)):
            arrRes[ii, (ii + 1):M] = self._vecR[0:(M - ii - 1)]

        return arrRes


##########################################################################
##########################################################################
from .helpers.unitInterface import *
################################################### Testing
test = {
    NAME_COMMON: {
        TEST_NUM_N: 5,
        'num_M' : Permutation([4, 6]),
        TEST_NUM_M: (lambda param : param['num_M'] + 1),
        'mTypeH': Permutation(typesAll),
        'mTypeV': Permutation(typesSmallIFC),
        'padding' : False,
        TEST_PARAMALIGN : ALIGN_DONTCARE,
        'vecH': ArrayGenerator({
            NAME_DTYPE  : 'mTypeH',
            NAME_SHAPE  : (TEST_NUM_N, ),
            NAME_ALIGN  : TEST_PARAMALIGN
            #            NAME_CENTER : 2,
        }),
        'vecV': ArrayGenerator({
            NAME_DTYPE  : 'mTypeV',
            NAME_SHAPE  : ('num_M', ),
            NAME_ALIGN  : TEST_PARAMALIGN
            #            NAME_CENTER : 2,
        }),
        TEST_INITARGS: (lambda param : [
            param['vecH'](),
            param['vecV']()
        ]),
        TEST_INITKWARGS: {'pad' : 'padding'},
        TEST_OBJECT: Toeplitz,
        TEST_NAMINGARGS: dynFormatString(
            "%s,%s,pad=%s",
            'vecH', 'vecV', str('padding')),
        TEST_TOL_POWER: 2
    },
    TEST_CLASS: {
        # test basic class methods

        # perform thorough testing of slicing cases in array construction
        # therefore, aside the symmetric shape case also test shapes that
        # differ by +1/-1 and +x/-x in row and column size
        TEST_NUM_N: 4,
        'num_M' : Permutation([2, 3, 4, 5, 6]),
    }, TEST_TRANSFORMS: {
        # test forward and backward transforms

        # test differences in padding only for the transforms
        'padding' : Permutation([True, False]),
    }
}

################################################## Benchmarks
benchmark = {
    NAME_COMMON: {
        NAME_DOCU       : r'''$\bm T \in \R^{n \times n}$ with generating
            entries drawn from a standard Gaussian distribution''',
        BENCH_FUNC_GEN  :
            (lambda c : Toeplitz(np.random.randn(c), np.random.randn(c - 1)))
    },
    BENCH_FORWARD: {
    },
    BENCH_SOLVE: {
    },
    BENCH_OVERHEAD: {
        BENCH_FUNC_GEN  :
            (lambda c : Toeplitz(
                np.random.randn(2 ** c),
                np.random.randn(2 ** c - 1))),
        NAME_DOCU       : r'''$\bm T \in \R^{2^k \times 2^k}$ with
            generating entries drawn from a standard Gaussian
            distribution; so $n = 2^k$ for $k \in \N$'''
    },
    BENCH_DTYPES: {
        BENCH_FUNC_GEN  :
            (lambda c, datatype : Toeplitz(
                np.random.randn(2 ** c).astype(datatype),
                np.random.randn(2 ** c - 1).astype(datatype))),
        NAME_DOCU       : r'''$\bm T \in \R^{2^k \times 2^k}$ with
            generating entries drawn from a standard Gaussian
            distribution; so $n = 2^k$ for $k \in \N$'''
    }
}

################################################## Documentation
docLaTeX = r"""
\subsection{Toeplitz Matrix (\texttt{fastmat.Toeplitz})}
\subsubsection{Definition and Interface}
A Toeplitz matrix $\bm T \in \C^{n \times m}$ realizes the mapping
\[\bm x \mapsto \bm T \cdot \bm x,\]
where $\bm x \in C^n$ and
\[\bm T = \left(\begin{array}{cccc}
    t_1 & t_{-1} & \dots & t_{-(m-1)} \\
    t_2 & t_1 & \ddots & t_{-(n-2)} \\
    \vdots & \vdots & \ddots & \vdots \\
    t_n & t_{n-1} & \dots & t_1
\end{array}\right).\]
This means that a Toeplitz matrix is uniquely defined by the $n + m - 1$ values
that are on the diagonals.

\begin{snippet}
\begin{lstlisting}[language=Python]
# import the package
import fastmat as fm
import numpy as np

# define the parameters
d1 = np.array([1,0,3,6])
d2 = np.array([5,7,9])

# construct the transform
T = fm.Toeplitz(d1,d2)
\end{lstlisting}

This yields
\[\bm d_1 = (1,0,3,6)^T\]
\[\bm d_2 = (5,7,9)^T\]
\[\bm T = \left(\begin{array}{cccc}
    1 & 5 & 7 & 9 \\
    0 & 1 & 5 & 7 \\
    3 & 0 & 1 & 5 \\
    6 & 3 & 0 & 1
\end{array}\right)\]
\end{snippet}

\begin{snippet}
Since the multiplication with a Toeplitz matrix makes use of the FFT, it can
be very slow, if the sum of the dimensions of $\bm d_1$ and $\bm d_2$ are far
away from a power of $2$, $3$ or $4$. This can be alleviated if one applies
smart zeropadding during the transformation. This can be activated as follows.

\begin{lstlisting}[language=Python]
# import the package
import fastmat as fm
import numpy as np

# define the parameters
d1 = np.array([1,0,3,6])
d2 = np.array([5,7,9])

# construct the transform
T = fm.Toeplitz(d1,d2,pad='true')
\end{lstlisting}

This yields the same matrix and transformation as above, but it might be faster
depending on the dimensions involved in the problem.
\end{snippet}

\textbf{Depends:}
    \texttt{Fourier}, \texttt{Diag}, \texttt{Product}, \texttt{Partial}
"""
