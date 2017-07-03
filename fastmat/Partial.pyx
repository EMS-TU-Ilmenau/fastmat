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
from .helpers.types cimport *
from .helpers.cmath cimport _arrZero

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

    # content - Property (read-only)
    # Return the transform the partial is derived from.
    property content:
        def __get__(self):
            return self._content

    ############################################## class methods
    def __init__(
        self,
        mat,
        N=None,
        M=None
    ):
        '''Initialize Matrix instance'''

        # initialize matrix for full support (used anyway for checking)
        self._indicesM = np.arange(mat.numM)
        self._indicesN = np.arange(mat.numN)
        self._content = mat

        # check if anything needs to be done in N- or M-dimension
        # store support indices in N- and M- dimension if needed
        self._pruneN = (N is not None) \
            and ((len(N) != mat.numN) or (np.sum(N - self._indicesN) != 0))
        self._pruneM = (M is not None) \
            and ((len(M) != mat.numM) or (np.sum(M - self._indicesM) != 0))

        if self._pruneN:
            self._indicesN = N

        if self._pruneM:
            self._indicesM = M

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
                self._content.__class__.__name__,
                self._content.numN, self._content.numM,
                id(self))
        else:
            return super(Partial, self).__repr__()

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        cdef intsize idxM = self._indicesM[idx] if self._pruneM else idx
        return (self._content.getCol(idxM)[self._indicesN] if self._pruneN
                else self._content.getCol(idxM))

    cpdef np.ndarray _getRow(self, intsize idx):
        cdef intsize idxN = self._indicesN[idx] if self._pruneN else idx
        return (self._content.getRow(idxN)[self._indicesM] if self._pruneM
                else self._content.getRow(idxN))

    ############################################## class forward / backward
    cpdef np.ndarray _forward(self, np.ndarray arrX):
        '''Calculate the forward transform of this matrix'''

        cdef np.ndarray arrInput

        if self._pruneM:
            arrInput = _arrZero(
                2, self._content.numM, arrX.shape[1], _getNpType(arrX))
            arrInput[self._indicesM, :] = arrX
        else:
            arrInput = arrX

        return (self._content.forward(arrInput)[self._indicesN, :]
                if self._pruneN else self._content.forward(arrInput))

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        '''Calculate the backward transform of this matrix'''

        cdef np.ndarray arrInput

        if self._pruneN:
            arrInput = _arrZero(
                2, self._content.numN, arrX.shape[1], _getNpType(arrX))
            arrInput[self._indicesN, :] = arrX
        else:
            arrInput = arrX

        return (self._content.backward(arrInput)[self._indicesM, :]
                if self._pruneM else self._content.backward(arrInput))

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        return self._content.reference()[self._indicesN, :][:, self._indicesM]


################################################################################
################################################################################
from .helpers.unitInterface import *

################################################### Testing
test = {
    NAME_COMMON: {
        'num_N' : 15,
        'num_M' : Permutation([20, 'num_N']),
        TEST_NUM_N : (lambda param : len(param['subRows'])),
        TEST_NUM_M : (lambda param : len(param['subCols'])),
        'mType' : Permutation(typesAll),
        'arrM' : ArrayGenerator({
            NAME_DTYPE  : 'mType',
            NAME_SHAPE  : ('num_N', 'num_M')
            #            NAME_CENTER : 2,
        }),
        'subCols' : Permutation([np.array([1, 2, 3, 11, 12]), np.array([6])]),
        'subRows' : Permutation([np.array([7, 8, 9, 13]), np.array([10])]),
        TEST_INITARGS: (lambda param : [
            Matrix(param['arrM']()),
            param['subRows'],
            param['subCols']
        ]),
        TEST_OBJECT: Partial,
        TEST_NAMINGARGS: dynFormatString(
            "%s,%dx%d", 'arrM', TEST_NUM_N, TEST_NUM_M)
    },
    TEST_CLASS: {
        # test basic class methods
    }, TEST_TRANSFORMS: {
        # test forward and backward transforms
    }
}

################################################## Benchmarks
from .Eye import Eye

benchmark = {
    BENCH_FORWARD: {
        BENCH_FUNC_GEN  :
            (lambda c: Partial(Eye(2 * c), N=np.arange(c), M=np.arange(c))),
        NAME_DOCU       :
            r'Partial Hadamard matrix $\bm P = \bm \Hs_{2^n,\{1,\dots,n\}}$'
    },
    BENCH_OVERHEAD: {
        BENCH_FUNC_GEN  : (lambda c: Partial(Eye(2 ** c), np.arange(2 ** c))),
        NAME_DOCU       :
            r'Partial Matrix, which is a not so partial identity matrix.'
    }
}


################################################## Documentation
docLaTeX = r"""
\subsection{Partial Transform (\texttt{fastmat.Partial})}
\subsubsection{Definition and Interface}
Let $I \subset \{1,\dots,n\}$ be an index set and $\bm M \in \C^{n \times m}$ a
linear transform. Then the partial transform $\bm M_I$ is defined as
\[\bm x \in \C^m \mapsto (\bm M \cdot \bm x)_{i \in I}.\]
In other words, we select rows of $\bm M$.

\begin{snippet}
\begin{lstlisting}[language=Python]
# import the package
import fastmat as fm
import numpy as np

# define the fourier transform
F = fm.Fourier(n)

# define the index set
a = np.arange(n)
am = np.mod(a, 2)
b = np.array(am, dtype='bool')
I = a[b]

# construct the partial transform
M = fm.Partial(F, I)
\end{lstlisting}

Let $\bm{\mathcal{F}}$ be the $n$-dimensional Fourier matrix. And let $I$ be the
set of odd integers. Then we define a partial transform as
\[\bm M = \bm{\mathcal{F}}_I\]
\end{snippet}
"""
