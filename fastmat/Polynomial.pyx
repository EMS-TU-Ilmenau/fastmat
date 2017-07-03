# -*- coding: utf-8 -*-
'''
  fastmat/Polynomial.py
 -------------------------------------------------- part of the fastmat package

  Determine the Polynomial of a fastmat matrix.


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
################################################## class Polynomial
cdef class Polynomial(Matrix):

    ############################################## class properties
    # content - Property (read-only)
    # Return the base transformation matrix.
    property content:
        def __get__(self):
            return self._content

    # coeff - Property (read-only)
    # Return the polynomial coefficient vector.
    property coeff:
        def __get__(self):
            return self._coeff

    ############################################## class methods
    def __init__(self, mat, coeff, **options):
        '''Initialize Matrix instance'''

        if mat.numN != mat.numM:
            raise ValueError("Matrix in polynomial must be square.")

        dtype = np.promote_types(mat.dtype, coeff.dtype)

        # handle type expansion with default depending on matrix type
        # default: expand small types due to accumulation during transforms
        # skip by specifying `typeExpansion=None` or override with `~=...`
        typeExpansion = options.get('typeExpansion', safeTypeExpansion(dtype))
        dtype = (dtype if typeExpansion is None
                 else np.promote_types(dtype, typeExpansion))

        self._content = mat
        self._coeff = np.flipud(coeff.astype(dtype))
        self._coeffConj = self._coeff.conj()

        # set properties of matrix
        self._initProperties(self._content.numN, self._content.numM, dtype)

    ############################################## class forward / backward
    cpdef np.ndarray _forward(self, np.ndarray arrX):
        '''Apply the Horner scheme to a polynomial of matrices.'''
        cdef cc, cnt = self._coeff.shape[0]
        cdef np.ndarray arrRes, arrIn = arrX

        arrRes = np.inner(arrX, self._coeff[0])

        # use inner for element-wise scalar mul as inner does type promotion
        for cc in range(1, cnt):
            arrRes  = self._content.forward(arrRes) + np.inner(
                arrX, self._coeff[cc])

        return arrRes

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        '''Apply the Horner scheme to a polynomial of matrices.'''
        cdef cc, cnt = self._coeffConj.shape[0]
        cdef np.ndarray arrRes

        arrRes = np.inner(arrX, self._coeffConj[0])

        # use inner for element-wise scalar mul as inner does type promotion
        for cc in range(1, cnt):
            arrRes  = self._content.backward(arrRes) + np.inner(
                arrX, self._coeffConj[cc])

        return arrRes

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        cdef intsize ii, ind = 0
        cdef np.ndarray arrRes, tmp

        dtype = np.promote_types(self.dtype, np.float64)
        arrRes = np.zeros((self.numN, self.numN), dtype=dtype)

        for cc in np.flipud(self._coeff):
            arrTrafo = self._content.reference()
            tmp = np.eye(self._content.numN, dtype=dtype)
            for ii in range(ind):
                tmp = arrTrafo.dot(tmp)

            arrRes = arrRes + np.inner(tmp, cc)
            ind += 1

        return arrRes


################################################################################
################################################################################
from .helpers.unitInterface import *
################################################### Testing
test = {
    NAME_COMMON: {
        'order': 5,
        TEST_TOL_POWER: 'order',
        TEST_NUM_N: 7,
        TEST_NUM_M: 7,
        'mTypeC': Permutation(typesAll),
        'mTypeM': Permutation(typesAll),
        TEST_PARAMALIGN : Permutation(alignmentsAll),
        'vecC': ArrayGenerator({
            NAME_DTYPE  : 'mTypeC',
            NAME_SHAPE  : ('order', ),
            NAME_ALIGN  : TEST_PARAMALIGN
            #            NAME_CENTER : 2
        }),
        'arrM': ArrayGenerator({
            NAME_DTYPE  : 'mTypeM',
            NAME_SHAPE  : (TEST_NUM_N, TEST_NUM_M)
            #            NAME_CENTER : 2
        }),
        TEST_OBJECT: Polynomial,
        TEST_INITARGS: (lambda param : [
            Matrix(param['arrM']()),
            param['vecC']()
        ]),
        TEST_NAMINGARGS: dynFormatString("%s,%s", 'vecC', 'arrM'),
        TEST_TOL_POWER: 'order'
    },
    TEST_CLASS: {
        # test basic class methods
    }, TEST_TRANSFORMS: {
        # test forward and backward transforms
    }
}


################################################## Benchmarks
from .Eye import Eye
from .Hadamard import Hadamard

benchmark = {
    NAME_COMMON: {
        NAME_DOCU       : r'''$\bm P = a_2 \cdot \bm \Hs_k^2 + a_1 \cdot
            \bm \Hs_k^1 + a_0 \cdot \bm I_{2^k}$''',
        BENCH_FUNC_GEN  :
            (lambda c : Polynomial(Hadamard(c), np.random.uniform(1, 2, 6))),
        BENCH_FUNC_SIZE : (lambda c : 2 ** c),
        BENCH_FUNC_STEP : (lambda c : c + 1)
    },
    BENCH_FORWARD: {
    },
    BENCH_SOLVE: {
    },
    BENCH_OVERHEAD: {
        BENCH_FUNC_GEN  :
            (lambda c : Polynomial(Eye(2 ** c), np.random.uniform(1, 2, 10))),
        NAME_DOCU       : r'''Polynomial of Identity $\bm I_{2^k}$
            matrices with degree $10$'''
    }
}


################################################## Documentation
docLaTeX = r"""
\subsection{Polynomial (\texttt{fastmat.Polynomial})}
\subsubsection{Definition and Interface}
For given coefficients $a_k,\dots,a_0 \in \C$ and a linear mapping $\bm A \in
\C^{n \times n}$, we define
\[\bm M = a_n \bm A^n + a_{n-1} \bm A^{n-1} + a_1 \bm A + a_0 \bm I.\]
The transform $\bm M \cdot \bm x$ can be calculated efficiently with Horner's
method.

\begin{snippet}
\begin{lstlisting}[language=Python]
# import the package
import fastmat as fm

# define the transforms
H = fm.Hadamard(n)

# define the coefficient array
arr_a = [1, 2 + 1j, -3.0, 0.0]

# define the polynomial
M = fm.Polynomial(H, arr_a)
\end{lstlisting}

Let $\bm H_n$ be the Hadamard matrix of order $n$. And let
$\bm a = (1, 2 + i, -3, 0) \in \C^{4}$ be a coefficient vector,
then the polynomial is defined as
\[\bm M = \bm H_n^3 + (2+i) \bm H_n^2 - 3 \bm H_n.\]
\end{snippet}
"""
