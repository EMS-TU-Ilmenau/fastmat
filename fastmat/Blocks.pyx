# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False, nonecheck=False
'''
  fastmat/Blocks.py
 -------------------------------------------------- part of the fastmat package

  Block matrix.


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
    - Blocks should simply skip all Zero Matrices (flag them as "None")?
'''
import numpy as np
cimport numpy as np

from .Matrix cimport Matrix


################################################################################
################################################## class Blocks
cdef class Blocks(Matrix):

    ############################################## class properties
    # content - Property (read-only)
    # Return a matrix of child matrices
    property content:
        def __get__(self):
            return self._content

    ############################################## class methods
    def __init__(self, arrMatrices):
        '''Initialize Matrix instance with a list of child matrices'''
        if not isinstance(arrMatrices, list):
            raise ValueError("A list of list (2D-list) of fastmat matrices " +
                             "must be passed to Blocks")

        if len(arrMatrices) < 1:
            raise ValueError("'Blocks' must contain at least one matrix")

        cdef intsize numN = 0, numM = 0
        cdef intsize ii, rr, cc
        cdef tuple row, firstRow
        cdef Matrix term

        # initialize sizes and number of rows / cols
        self._numRows = len(arrMatrices)
        self._numCols = len(arrMatrices[0])
        dataType = np.int8

        # generate transposed block structure, force tuples for backward()
        cdef list lst = [tuple(arrMatrices[rr]) for rr in range(self._numRows)]
        self._content = tuple(lst)
        lst = [[] for _ in range(self._numCols)]
        firstRow = self._content[0]

        # extract list of row heights and column widths
        self._rowN = tuple([row[0].numN for row in self._content])
        self._colM = tuple([term.numM for term in firstRow])

        # enumerate rows
        for rr in range(self._numRows):
            row = self._content[rr]

            # get number of rows from first elements in columns
            numN += row[0].numN

            # check for presence of enough blocks
            if len(row) != self._numCols:
                raise ValueError(
                    "Blocks.row(%d) has incompatible number of entries" %(rr))

            # enumerate columns
            for cc in range(self._numCols):
                term = row[cc]

                # check for matching column height and width
                if term.numN != row[0].numN:
                    raise ValueError(
                        ("Blocks[%d,%d] with shape %s is incompatible to " +
                         "row height (%d,:)") %(
                            rr, cc, str(term.shape), row[0].numN))

                if term.numM != firstRow[cc].numM:
                    raise ValueError(
                        ("Blocks[%d,%d] with shape %s is incompatible to " +
                         "column width (:,%d)") %(
                            rr, cc, str(term.shape), firstRow[cc].numM))

                # first run: get stats and update dimension of
                # Blocks from first column's entries
                if rr == 0:
                    numM += term.numM

                # build transposed copy of blocks to work on
                # in backward
                lst[cc].append(term)

                # determine necessary output data type by
                # applying type promotion
                dataType = np.promote_types(dataType, term.dtype)

        # convert generated transposed array to tuple of tuples
        for cc in range(self._numCols):
            lst[cc] = tuple(lst[cc])

        self._contentT = tuple(lst)

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
        cdef np.ndarray viewOut, arrOut
        cdef Matrix term
        cdef tuple row, viewRows, viewCols
        cdef intsize idxN, idxM, rr, cc
        cdef list lst

        # generate views into output array
        lst = [None] * self._numRows
        idxN = 0
        for rr in range(self._numRows):
            lst[rr] = arrRes[idxN:(idxN + self._rowN[rr])]
            idxN += self._rowN[rr]

        viewRows = tuple(lst)

        # generate views into input array
        lst = [None] * self._numCols
        idxM = 0
        for cc in range(self._numCols):
            lst[cc] = arrX[idxM:(idxM + self._colM[cc])]
            idxM += self._colM[cc]

        viewCols = tuple(lst)

        # do the trick
        for rr in range(self._numRows):
            row = self._content[rr]
            viewOut = viewRows[rr]

            viewOut[:] = row[0].forward(viewCols[0])
            for cc in range(1, self._numCols):
                viewOut += row[cc].forward(viewCols[cc])

        return arrRes

    cpdef _backwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        '''Calculate the backward transform of this matrix'''
        cdef np.ndarray viewOut
        cdef Matrix term
        cdef tuple col, viewRows, viewCols
        cdef intsize idxN, idxM, rr, cc
        cdef list lst

        # generate views into output array
        lst = [None] * self._numCols
        idxM = 0
        for cc in range(self._numCols):
            lst[cc] = arrRes[idxM:(idxM + self._colM[cc])]
            idxM += self._colM[cc]

        viewCols = tuple(lst)

        # generate views into input array
        lst = [None] * self._numRows
        idxN = 0
        for rr in range(self._numRows):
            lst[rr] = arrX[idxN:(idxN + self._rowN[rr])]
            idxN += self._rowN[rr]

        viewRows = tuple(lst)

        # do the trick
        for cc in range(self._numCols):
            col = self._contentT[cc]
            viewOut = viewCols[cc]
            viewOut[:] = col[0].backward(viewRows[0])

            for rr in range(1, self._numRows):
                viewOut += col[rr].backward(viewRows[rr])

        return arrRes

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        cdef np.ndarray arrRes
        cdef Matrix term
        cdef tuple row
        cdef intsize idxN = 0, idxM = 0

        arrRes = np.empty((self.numN, self.numM), dtype=self.dtype)

        cdef intsize rr, tt
        for rr in range(self._numRows):
            row = self._content[rr]
            idxM = 0
            for tt in range(self._numCols):
                term = row[tt]
                arrRes[idxN:(idxN + term.numN), idxM:(idxM + term.numM)] = \
                    term._reference()
                idxM += term.numM

            idxN += term.numN

        return arrRes


################################################################################
################################################################################
from .helpers.unitInterface import *

################################################### Testing
test = {
    NAME_COMMON: {
        'size': 4,
        TEST_NUM_N: (lambda param: param['size'] * 2),
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
            NAME_DTYPE  : 'mType2',
            NAME_SHAPE  : ('size', 'size')
            #            NAME_CENTER : 2,
        }),
        'arr4': ArrayGenerator({
            NAME_DTYPE  : 'mType1',
            NAME_SHAPE  : ('size', 'size')
            #            NAME_CENTER : 2,
        }),
        TEST_INITARGS: (lambda param : [
            [[Matrix(param['arr1']()), Matrix(param['arr2']())],
             [Matrix(param['arr3']()), Matrix(param['arr4']())]]
        ]),
        TEST_OBJECT: Blocks,
        'strType1': (lambda param: NAME_TYPES[param['mType1']]),
        'strType2': (lambda param: NAME_TYPES[param['mType2']]),
        TEST_NAMINGARGS: dynFormatString(
            "[%s,%s],[%s,%s]:(%dx%d) each",
            'strType1', 'strType2', 'strType2', 'strType1', 'size', 'size')
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
    BENCH_OVERHEAD: {
        BENCH_FUNC_GEN  :
            (lambda c : Blocks([[Eye(2 ** c)] * 4] * 4)),
        NAME_DOCU       : r'''$\bm B = \begin{pmatrix}
            \bm I_{2^k} & \bm I_{2^k} & \bm I_{2^k} & \bm I_{2^k} \\
            \bm I_{2^k} & \bm I_{2^k} & \bm I_{2^k} & \bm I_{2^k} \\
            \bm I_{2^k} & \bm I_{2^k} & \bm I_{2^k} & \bm I_{2^k} \\
            \bm I_{2^k} & \bm I_{2^k} & \bm I_{2^k} & \bm I_{2^k}
            \end{pmatrix}$, $n = 2^{k+2}$ for $k \in \N$''' ,
        BENCH_FUNC_SIZE : (lambda c : 2 ** c * 4)
    }
}


################################################## Documentation
docLaTeX = r"""
\subsection{Block Matrix (\texttt{fastmat.Blocks})}
\subsubsection{Definition and Interface}
\[\bm M = \left( \bm A_{i,j}\right)_{i,j},\]
where the $\bm A_{i,j}$ can be fast transforms of \emph{any} type.

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
# matrix row-wise
M = fm.Blocks([[A,B],[C,D]])
\end{lstlisting}

Assume we have two circulant matrices $\bm A$ and $\bm B$, an $N$-dimensional
Fourier matrix $\bm C$ and a diagonal matrix $\bm D$. Then we define
\[M = \left(\begin{array}{cc} A & B \\ C & D \end{array}\right).\]
\end{snippet}
"""
