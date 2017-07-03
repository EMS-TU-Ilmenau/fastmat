# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False, nonecheck=False
'''
  fastmat/Circulant.py
 -------------------------------------------------- part of the fastmat package

  Circulant matrix.


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
    - sort out when optimizations are possible (real/complex, such stuff)
    - specify when this transform was introduced
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
################################################## class Circulant
cdef class Circulant(Partial):

    ############################################## class properties
    # vecC - Property (read-only)
    # Return the matrix-defining column vector of the circulant matrix
    property vecC:
        def __get__(self):
            return self._vecC

    ############################################## class methods
    def __init__(self, vecC, **options):
        '''
        Initialize Circulant Matrix instance.

        Circulant([ d c C C C]) represents this matrix:
          [ d C C C c ]
          [ c d C C C ]
          [ C c d C C ]
          [ C C c d C ]
          [ C C C c d ]

        the generating vector passed as vecC may be zero-padded to increase
        computation efficiency of involved fft operations. However, the padded
        zeros must be appended to the center of the generating column. The
        generating vector must head the column vector to define the resulting
        lower triangular matrix. The upper triangular is defined by a reversed
        copy of the vector at the end of column vector. This is short by one
        entry, which is already placed at the very first position. The minimal
        column vector length is one element short of twice the generating vector
        length. Thus, padding is only efficient for really bad behaved fourier
        transforms.
          [ d c C C C 0 0 0 0 0 0 0 C C C c]

        Valid options
        -------------
         'pad'=[FALSE true]
            perform zero-padding of Circulant for efficiency. By default set to
            False as padding introduces significant overhead to non-padded for
            most but some sizes.

        All options specified will also be passed on to the geneation of the
        underlying Product instance initialization.
        '''

        # save generating vector. Matrix sizes will be set by Product
        # during its intitalization (invoked by 'super' below)
        self._vecC = np.atleast_1d(np.squeeze(np.copy(vecC)))
        if self._vecC.ndim != 1:
            raise ValueError("Column-definition vector must be 1D.")

        # evaluate options passed to class
        pad = options.get('pad', False)

        # perform padding (if enabled) and generate vector
        # N will be the next larger power of two for deciding whether
        # padding is actually necessary
        cdef intsize size = len(vecC)
        cdef intsize needed = 2 * size - 1
        cdef intsize N = 2 ** np.ceil(np.log2(size))
        if size < 1:
            raise ValueError("Vector must have at least one entry")

        if pad and (N > size):
            # N must have at least 2 * N - 1 elements
            N = 2 ** np.ceil(np.log2(needed))
            vec = np.concatenate([
                self._vecC,
                np.zeros((N - needed,), dtype=self._vecC.dtype),
                self._vecC[1:]
            ])
        else:
            N = size
            vec = self._vecC

        # Describe circulant matrix as product of data and vector in
        # fourier domain. Both fourier matrices cause scaling of the
        # data vector by N, which will be compensated in Diag().

        # Create inner product
        cdef Fourier FN = Fourier(N)
        cdef Product P = Product(FN.H, Diag(np.fft.fft(vec) / N), FN, **options)

        # initialize Partial of Product
        super(Circulant, self).__init__(P, N=np.arange(size), M=np.arange(size))

    cpdef np.ndarray toarray(self):
        '''
        Return an explicit representation of the matrix as numpy-array.
        '''
        return self._reference()

    ############################################## class property override
    cpdef object _getItem(self, intsize idxN, intsize idxM):
        return self._vecC[(idxN - idxM) % self.numN]

    cpdef np.ndarray _getCol(self, intsize idx):
        '''Return selected columns of self.toarray()'''
        cdef np.ndarray arrRes = _arrEmpty(
            1, self.numN, 0, self._info.dtype[0].typeNum)
        self._roll(arrRes, idx)
        return arrRes

    cpdef np.ndarray _getRow(self, intsize idx):
        '''Return selected rows of self.toarray()'''
        cdef np.ndarray arrRes = _arrEmpty(
            1, self.numN, 0, self._info.dtype[0].typeNum)
        self._roll(arrRes[::-1], self.numN - idx - 1)
        return arrRes

    ############################################## internal roll core
    cdef void _roll(self, np.ndarray vecOut, intsize shift):
        '''Return self.vecC rolled by 'shift' elements.'''
        if shift == 0:
            vecOut[:] = self._vecC
        else:
            vecOut[:shift] = self._vecC[self.numN - shift:]
            vecOut[shift:] = self._vecC[:self.numN - shift]

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        cdef np.ndarray arrRes
        cdef intsize ii, N = self.numN, M = self.numM

        arrRes = np.empty((N, M), dtype=self.dtype)
        arrRes[:, 0] = self._vecC
        for ii in range(N):
            self._roll(arrRes[:, ii], ii)

        return arrRes


################################################################################
################################################################################
from .helpers.unitInterface import *
################################################### Testing
test = {
    NAME_COMMON: {
        TEST_NUM_N: 31,
        TEST_NUM_M: TEST_NUM_N,
        'mTypeC': Permutation(typesAll),
        'padding' : Permutation([True, False]),
        TEST_PARAMALIGN : Permutation(alignmentsAll),
        'vecC': ArrayGenerator({
            NAME_DTYPE  : 'mTypeC',
            NAME_SHAPE  : (TEST_NUM_N, ),
            NAME_ALIGN  : TEST_PARAMALIGN
            #            NAME_CENTER : 2,
        }),
        TEST_INITARGS: (lambda param : [
            param['vecC']()
        ]),
        TEST_INITKWARGS: {'pad' : 'padding'},
        TEST_OBJECT: Circulant,
        TEST_NAMINGARGS: dynFormatString("%s,pad=%s", 'vecC', str('padding')),

        TEST_TOL_POWER          : 2.,
        TEST_TOL_MINEPS         : _getTypeEps(np.float64)
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
        BENCH_FUNC_GEN  :
            (lambda c : Circulant(np.random.randn(2 ** c))),
        BENCH_FUNC_SIZE : (lambda c : 2 ** c),
        BENCH_FUNC_STEP : (lambda c : c + 1),
        NAME_DOCU       : r'''$\bm{\mathcal{C}} \in \R^{n \times n}$
            with $n = 2^k$, $k \in \N$ and first columns entries
            drawn from a  standard Gaussian distribution.'''
    },
    BENCH_FORWARD: {
    },
    'padding': {
        NAME_TEMPLATE   : BENCH_FORWARD,
        BENCH_FUNC_GEN  :
            (lambda c : Circulant(np.random.randn(2 * c), pad=True)),
        BENCH_FUNC_SIZE : (lambda c : 2 * c),
        BENCH_FUNC_STEP : (lambda c : c * 10 ** (1. / 12))
    },
    BENCH_SOLVE: {
    },
    BENCH_OVERHEAD: {
    },
    BENCH_DTYPES: {
        BENCH_FUNC_GEN  :
            (lambda c, dt : Circulant(np.random.randn(2 ** c).astype(dt)))
    }
}


################################################## Documentation
docLaTeX = r"""
\subsection{Circulant Matrix (\texttt{fastmat.Circulant})}
\subsubsection{Definition and Interface}
Circulant matrices realize the following mapping
\[\bm x \mapsto \bm C \cdot \bm x = \bm c * \bm x,\]
with $\bm x \in \C^n$ and
\[\bm C = \left(\begin{array}{cccc} c_1 & c_n & \dots & c_2 \\ c_2 & c_1 &
    \ddots & c_3 \\ \vdots & \vdots & \ddots & \vdots \\ c_n & c_{n-1} &
    \dots & c_1 \end{array}\right).\]
This means that $\bm C$ is completely defined by its first column and realizes
the convolution with the vector $\bm c$.

\begin{snippet}
\begin{lstlisting}[language=Python]
# import the package
import fastmat as fm
import numpy as np

# construct the
# parameter
n = 4
c = np.array([1, 0, 3, 6])

# construct the matrix
C = fm.Circulant(c)
\end{lstlisting}

This yields
\[\bm c = (1,0,3,6)^T\]
\[\bm C = \left(\begin{array}{cccc}
    1 & 6 & 3 & 0 \\
    0 & 1 & 6 & 3 \\
    3 & 0 & 1 & 6 \\
    6 & 3 & 0 & 1
\end{array}\right)\]
\end{snippet}

\textbf{Depends:}
    \texttt{Fourier}, \texttt{Diag}, \texttt{Product}, \texttt{Partial}
"""
