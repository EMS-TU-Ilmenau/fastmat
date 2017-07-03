# -*- coding: utf-8 -*-
'''
  fastmat/Sparse.py
 -------------------------------------------------- part of the fastmat package

  Sparse matrix.


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
import scipy.sparse as sps

from .Matrix cimport Matrix
from .helpers.types cimport *
from .helpers.cmath cimport _conjugate

################################################################################
################################################## class Sparse
cdef class Sparse(Matrix):

    ############################################## class properties
    # content, contentH - Property (read-only)
    # Return the scipy sparse matrix or its hermitian transpose.
    property content:
        def __get__(self):
            return self._content

    property contentH:
        def __get__(self):
            if self._contentH is None:
                self._contentH = self._content.T.conj().tocsr()
            return self._contentH

    ############################################## class methods
    def __init__(self, value):
        '''Initialize Matrix instance'''
        self._content = value.tocsr()

        # set properties of matrix
        self._initProperties(
            self._content.shape[0], self._content.shape[1], self._content.dtype)

    cpdef np.ndarray toarray(self):
        '''
        Return an explicit representation of the matrix as numpy-array.
        '''
        return self._content.todense()

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        return np.squeeze(self._content.getcol(idx).todense())

    cpdef np.ndarray _getRow(self, intsize idx):
        return np.squeeze(_conjugate(self.contentH.getcol(idx).todense()))

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        return self._content[idxN, idxM]

    ############################################## class forward / backward
    cpdef np.ndarray _forward(self, np.ndarray arrX):
        '''Calculate the forward transform of this matrix'''
        return self._content.dot(arrX)

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        '''Calculate the backward transform of this matrix'''
        return self.contentH.dot(arrX)

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        return self._content.todense()


################################################################################
################################################################################
import numpy as np
from .helpers.unitInterface import *

################################################### Testing
test = {
    NAME_COMMON: {
        TEST_NUM_N: 25,
        TEST_NUM_M: Permutation([30, TEST_NUM_N]),
        'mType': Permutation(typesAll),
        'density': .1,
        TEST_INITARGS: (lambda param : [
            arrSparseTestDist(
                (param[TEST_NUM_N], param[TEST_NUM_M]),
                param['mType'],
                density=param['density'],
                compactFullyOccupied=True
            )
        ]),
        TEST_OBJECT: Sparse,
        'strType': (lambda param: NAME_TYPES[param['mType']]),
        TEST_NAMINGARGS: dynFormatString("%s,%s", 'strType', TEST_NUM_M)
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
        NAME_DOCU       : r'''Sparse matrix $\bm S \in \R^{n \times n}$
            with $10$\% of non-zero entries'''
    },
    BENCH_FORWARD: {
        BENCH_FUNC_GEN  :
            (lambda c : Sparse(sps.rand(c, c, 0.1, format='csr')))
    },
    BENCH_OVERHEAD: {
        BENCH_FUNC_GEN  :
            (lambda c : Sparse(sps.rand(2 ** c, 2 ** c, 0.1, format='csr')))
    },
    BENCH_DTYPES: {
        BENCH_FUNC_GEN  :
            (lambda c, dt :
             Sparse(sps.rand(2 ** c, 2 ** c, 0.1, format='csr', dtype=dt))
                if np.issubdtype(dt, np.float) else None)
    }
}


################################################## Documentation
docLaTeX = r"""

\subsection{Sparse Matrix (\texttt{fastmat.Sparse})}
\subsubsection{Definition and Interface}
\[x \mapsto \bm S x,\]
where $\bm S$ is a \texttt{scipy.sparse} matrix.
To provide a high level of generality, the user has to make use of the standard
\texttt{scipy.sparse} matrix constructors and pass them to \fm{} during
construction. After that a \texttt{Sparse} matrix can be used like every other
type in \fm{}.

\begin{snippet}
\begin{lstlisting}[language=Python]
# import the package
import fastmat as fm

# import scipy to get
# the constructor
import scipy.sparse.rand as r

# set the matrix size
n = 100

# construct the sparse matrix
S = fm.Sparse(
        r(
            n,
            n,
            0.01,
            format='csr'
        ))
\end{lstlisting}

This yields a random sparse matrix with 1\% of its entries occupied drawn from a
random distribution.
\end{snippet}

It is also possible to directly cast SciPy sparse matrices into the \fm{} sparse
matrix format as follows.

\begin{snippet}
\begin{lstlisting}[language=Python]
# import the package
import fastmat as fm

# import scipy to get
# the constructor
import scipy.sparse as ss

# construct the SciPy sparse matrix
S_scipy = ss.csr_matrix(
        [
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1]
        ]
    )

# construct the fastmat sparse matrix
S = fm.Sparse(S_scipy)
\end{lstlisting}
\end{snippet}

\textit{Hint:} The \texttt{format} specifier drastically influences performance
during multiplication of these matrices. From our experience \texttt{'csr'}
works best in these cases.

For this matrix class we used the already tried and tested routines of SciPy
\cite{srs_walt2011numpy}, so we merely provide a convenient wrapper to integrate
nicely into \fm{}.

\begin{thebibliography}{9}
\bibitem{srs_walt2011numpy}
St\'efan van der Walt, S. Chris Colbert and Ga\"el Varoquaux
\emph{The NumPy Array: A Structure for Efficient Numerical Computation},
Computing in Science and Engineering,
Volume 13,
2011.
\end{thebibliography}
"""
