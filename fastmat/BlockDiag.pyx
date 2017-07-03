# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False, nonecheck=False
'''
  fastmat/BlockDiag.py
 -------------------------------------------------- part of the fastmat package

  Block diagonal matrix.


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
    - BlockDiag should simply skip all Zero Matrices (flag them as "None")?
'''
import numpy as np
cimport numpy as np

from .Matrix cimport Matrix
from .helpers.types cimport *
from .helpers.cmath cimport _arrZero

################################################################################
################################################## class BlockDiag
cdef class BlockDiag(Matrix):

    ############################################## class properties
    # content - Property (read-only)
    # Return a list of child matrices on the diagonal of matrix
    property content:
        def __get__(self):
            return self._content

    ############################################## class methods
    def __init__(self, *matrices):
        '''Initialize Matrix instance with a list of child matrices'''
        cdef intsize numN = 0, numM = 0
        cdef Matrix term

        self._content = matrices

        # determine total size and data type of matrix
        dataType = np.int8
        for term in self._content:
            if not isinstance(term, Matrix):
                raise ValueError(
                    "Only fastmat matrices supported, %s given." %(str(term)))

            numN += term.numN
            numM += term.numM
            dataType = np.promote_types(dataType, term.dtype)

        # set properties of matrix
        self._initProperties(
            numN, numM, dataType,
            cythonCall=True,
            widenInputDatatype=True
        )

    ############################################## class forward / backward
    cpdef _forwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        '''Calculate the forward transform of this matrix'''
        cdef Matrix term
        cdef intsize idxN = 0, idxM = 0, ii, cnt = len(self._content)

        for ii in range(0, cnt):
            term = self._content[ii]
            arrRes[idxN:(idxN + term.numN), :] \
                = term.forward(arrX[idxM:(idxM + term.numM)])

            idxN += term.numN
            idxM += term.numM

        return arrRes

    cpdef _backwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        '''Calculate the backward transform of this matrix'''
        cdef Matrix term
        cdef intsize idxN = 0, idxM = 0, cnt = len(self._content)

        for ii in range(0, cnt):
            term = self._content[ii]

            arrRes[idxM:(idxM + term.numM), :] \
                = term.backward(arrX[idxN:(idxN + term.numN)])

            idxN += term.numN
            idxM += term.numM

        return arrRes

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        cdef np.ndarray arrRes
        cdef Matrix term
        cdef intsize idxN = 0, idxM = 0

        arrRes = np.zeros((self.numN, self.numM), dtype=self.dtype)

        for term in self._content:
            arrRes[idxN:(idxN + term.numN), :][:, idxM:(idxM + term.numM)] = \
                term.toarray()

            idxN += term.numN
            idxM += term.numM

        return arrRes


################################################################################
################################################################################
from .helpers.unitInterface import *

################################################### Testing
test = {
    NAME_COMMON: {
        'size': 5,
        TEST_NUM_N: (lambda param: param['size'] * 3),
        TEST_NUM_M: TEST_NUM_N,
        'mType1': Permutation(typesAll),
        'mType2': Permutation(typesAll),
        'arr1': ArrayGenerator({
            NAME_DTYPE  : 'mType1',
            NAME_SHAPE  : ('size', 'size')
            #            NAME_CENTER : 2,
        }),
        'arr2': ArrayGenerator({
            NAME_DTYPE  : 'mType2',
            NAME_SHAPE  : ('size', 'size')
            #            NAME_CENTER : 2,
        }),
        'arr3': ArrayGenerator({
            NAME_DTYPE  : 'mType1',
            NAME_SHAPE  : ('size', 'size')
            #            NAME_CENTER : 2,
        }),
        TEST_INITARGS: (lambda param : [
            Matrix(param['arr1']()),
            -2. * Matrix(param['arr2']()),
            2. * Matrix(param['arr3']())
        ]),
        TEST_OBJECT: BlockDiag,
        'strType1': (lambda param: NAME_TYPES[param['mType1']]),
        'strType2': (lambda param: NAME_TYPES[param['mType2']]),
        TEST_NAMINGARGS: dynFormatString(
            "%s,%s,%s:(%dx%d) each",
            'strType1', 'strType2', 'strType1', 'size', 'size'),
        TEST_TOL_POWER : 3.
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
    NAME_COMMON: {
        NAME_DOCU       : r'''$\bm B = \begin{pmatrix}
            \bm M_{k \times k} & 0 \\ 0 & \bm M_M_{k \times k}
            \end{pmatrix}$, $n = 2k$''',
        BENCH_FUNC_GEN  : (lambda c : BlockDiag(
            Matrix(arrTestDist((c, c), np.float64)),
            Matrix(arrTestDist((c, c), np.float64))
        )),
        BENCH_FUNC_SIZE : (lambda c : 2 * c)
    },
    BENCH_FORWARD: {
    },
    BENCH_OVERHEAD: {
        NAME_DOCU       : r'''$\bm B =
            \begin{pmatrix}
                \bm I_{2^k} & \dots     & 0             \\
                            & \ddots    &               \\
                0           & \dots     & \bm I_{2^k}
            \end{pmatrix}$''',
        BENCH_FUNC_GEN  :
            (lambda c : BlockDiag(*([Eye(2 ** c)] * 16))),
        BENCH_FUNC_SIZE : (lambda c : 2 ** c * 16)
    }
}


################################################## Documentation
docLaTeX = r"""
\subsection{Block Diagonal Matrix (\texttt{fastmat.BlockDiag})}
\subsubsection{Definition and Interface}
\[\bm M = \mathrm{diag}\left\{\left( \bm A_{i}\right)_{i}\right\},\]
where the $\bm A_{i}$ can be fast transforms of \emph{any} type.

\begin{snippet}
\begin{lstlisting}[language=Python]
# import the package
import fastmat as fm

# define the blocks
A = fm.Circulant(x_A)
B = fm.Circulant(x_B)
C = fm.Fourier(n)
D = fm.Diag(x_D)

# define the block
# diagonal matrix
M = fm.BlockDiag(A, B, C, D)
\end{lstlisting}

Assume we have two circulant matrices $\bm A$ and $\bm B$, an $N$-dimensional
Fourier matrix $\bm C$ and a diagonal matrix $\bm D$. Then we define
\[\bm M = \left(\begin{array}{cccc}
    \bm A & & & \\
    & \bm B & & \\
    & & \bm C & \\
    & & & \bm D
\end{array}\right).\]
\end{snippet}

Meta types can also be nested, so that a block diagonal matrix can contain
products of block matrices as its entries. Note that the efficiency of the fast
transforms decreases the more building blocks they have.

\begin{snippet}
\begin{lstlisting}[language=Python]
# import the package
import fastmat as fm

# define the blocks
A = fm.Circulant(x_A)
B = fm.Circulant(x_B)
F = fm.Fourier(n)
D = fm.Diag(x_D)

# define a product
P = fm.Product(A.H, B)

# define the block
# diagonal matrix
M = fm.BlockDiag(P, F, D)
\end{lstlisting}

Assume we have a product $\bm P$ of two matrices $\bm A^\herm$ and $\bm B$,
an $N$-dimensional Fourier matrix $\bm{\mathcal{F}}$ and a diagonal matrix
$\bm D$. Then we define
\[\bm M = \left(\begin{array}{cccc}
    \bm A^\herm \cdot B & &  \\
    & \bm{\mathcal{F}} & \\
    & & \bm D
\end{array}\right).\]
\end{snippet}
"""
