# -*- coding: utf-8 -*-
'''
  fastmat/Outer.pyx
 -------------------------------------------------- part of the fastmat package

  Outer product (rank one) matrix.


  Author      : sempersn
  Introduced  : 2017-31-01
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
from .helpers.cmath cimport _conjugate, _multiply

################################################################################
################################################## class Outer
cdef class Outer(Matrix):

    ############################################## class properties
    # vecV - Property (read-only)
    # Return the matrix-defining vector of vertical defining entries.
    property vecV:
        def __get__(self):
            return self._vecV

    # vecH - Property (read-only)
    # Return the matrix-defining vector of horizontal defining entries.
    property vecH:
        def __get__(self):
            return self._vecH

    ############################################## class methods
    def __init__(self, vecV, vecH):

        # check dimensions
        vecV = np.atleast_1d(np.squeeze(vecV))
        vecH = np.atleast_1d(np.squeeze(vecH))

        if vecV.ndim != 1 or vecH.ndim != 1:
            raise ValueError("Outer parameters must be one-dimensional.")

        # height and width of matrix is defined by length of input vectors
        cdef intsize numN = len(vecV)
        cdef intsize numM = len(vecH)

        # determine joint data type of operation
        datatype = np.promote_types(vecV.dtype, vecH.dtype)

        # store H/V entry vectors as copies of vecH/vecV
        self._vecH = vecH.astype(datatype, copy=True).reshape((1, numM))
        self._vecV = vecV.astype(datatype, copy=True).reshape((numN, 1))

        # store ravelled pointers of vecH/vecV
        self._vecHRav = self._vecH.ravel()
        self._vecVRav = self._vecV.ravel()

        # hermitian transpose of vecH/vecV for backward
        self._vecHConj = _conjugate(self._vecH).reshape((numM, 1))
        self._vecVHerm = _conjugate(self._vecV).reshape((1, numN))

        # set properties of matrix
        self._initProperties(numN, numM, datatype)

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        return self._vecHRav[idx] * self._vecVRav

    cpdef np.ndarray _getRow(self, intsize idx):
        return self._vecVRav[idx] * self._vecHRav

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        return self._vecVRav[idxN] * self._vecHRav[idxM]

    ############################################## class forward / backward
    cpdef np.ndarray _forward(self, np.ndarray arrX):
        '''Calculate the forward transform of this matrix.'''

        return self._vecV.dot(self._vecH.dot(arrX))

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        '''Calculate the backward transform of this matrix.'''

        return self._vecHConj.dot(self._vecVHerm.dot(arrX))

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        dtype = np.promote_types(self.dtype, np.float64)
        return self._vecV.dot(self._vecH.astype(dtype))


################################################################################
################################################################################
from .helpers.unitInterface import *

################################################## Testing
test = {
    NAME_COMMON: {
        TEST_NUM_N: 4,
        TEST_NUM_M: Permutation([6, TEST_NUM_N]),
        'mTypeH': Permutation(typesSmallIFC),
        'mTypeV': Permutation(typesAll),
        TEST_PARAMALIGN: ALIGN_DONTCARE,
        'vecH': ArrayGenerator({
            NAME_DTYPE  : 'mTypeH',
            NAME_SHAPE  : (TEST_NUM_N, 1),
            NAME_ALIGN  : TEST_PARAMALIGN
        }),
        'vecV': ArrayGenerator({
            NAME_DTYPE  : 'mTypeV',
            NAME_SHAPE  : (TEST_NUM_M, 1),
            NAME_ALIGN  : TEST_PARAMALIGN
        }),
        TEST_INITARGS: (lambda param : [
            param['vecH'](),
            param['vecV']()
        ]),
        TEST_OBJECT: Outer,
        TEST_NAMINGARGS: dynFormatString("%so%s", 'vecH', 'vecV'),
        TEST_TOL_MINEPS: lambda param:
            max(_getTypeEps(param['mTypeH']), _getTypeEps(param['mTypeV'])),
        TEST_TOL_POWER: 4.
    },
    TEST_CLASS: {
        # test basic class methods
    }, TEST_TRANSFORMS: {
        # test forward and backward transforms
        TEST_IGNORE:
            IgnoreFunc(lambda param : param['mTypeH'] == param['mTypeV'] == \
                       param[TEST_DATATYPE] == np.int8)
    }
}


################################################## Benchmarks
benchmark = {
    NAME_COMMON: {
        BENCH_FUNC_GEN  :
            (lambda c : Outer(
                np.random.uniform(2, 3, c),
                np.random.uniform(2, 3, c)
            )),
        NAME_DOCU       : r'''$\bm M \in \R^{n \times n}$ with defining vectors'
            entries drawn from a uniform distribution on $[2,3]$'''
    },
    BENCH_FORWARD: {
    },
    BENCH_OVERHEAD: {
        BENCH_FUNC_GEN  :
            (lambda c : Outer(
                np.random.uniform(2, 3, 2 ** c),
                np.random.uniform(2, 3, 2 ** c)
            )),
        NAME_DOCU       : r'''$\bm M \in \R^{n \times n}$ with $n = 2^k$,
            $k \in \N$ and diagonal entries drawn from a uniform
            distribution on $[2,3]$'''
    },
    BENCH_DTYPES: {
        BENCH_FUNC_GEN  :
            (lambda c, datatype : Outer(
                np.random.uniform(2, 3, 2 ** c).astype(datatype),
                np.random.uniform(2, 3, 2 ** c).astype(datatype)
            )),
        NAME_DOCU       : r'''$\bm M \in \R^{n \times n}$ with $n = 2^k$,
            $k \in \N$ and defining vectors' entries drawn from a uniform
            distribution on $[2,3]$'''
    }
}


################################################## Documentation
docLaTeX = r"""
\subsection{Outer Product Matrix (\texttt{fastmat.Outer})}
\subsubsection{Definition and Interface}
The outer product is a special case of the Kronecker product of one-dimensional
vectors. For given $\bm a \in \C^n$ and $\bm b \in \C^m$ it is defined as
    \[\bm x \mapsto \bm a \cdot \bm b^\trans \cdot \bm x.\]
It is clear, that this matrix has at most rank $1$ and as such has a fast
transformation.

\begin{snippet}
\begin{lstlisting}[language=Python]
# import the package
import fastmat as fm
import numpy as np

# build the parameters
n,m = 4,5
v = np.arange(n)
h = np.arange(m)

# construct the matrix
M = fm.Outer(v,h)
\end{lstlisting}

This yields
\[\bm v = (0,1,2,3,4)^T\]
\[\bm h = (0,1,2,3,4,5)^T\]
\[\bm M = \left(\begin{array}{ccccc}
    0 & 0 & 0 & 0 & 0 \\
    0 & 1 & 2 & 3 & 4 \\
    0 & 2 & 4 & 6 & 8 \\
    0 & 3 & 6 & 9 & 12
\end{array}\right)\]
\end{snippet}
"""
