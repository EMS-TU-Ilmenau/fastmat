# -*- coding: utf-8 -*-
#cython: boundscheck=False
'''
  fastmat/MLUltraSound.py
 -------------------------------------------------- part of the fastmat package

  MLUltraSound matrix.


  Author      : sempersn
  Introduced  : 2017-10-06
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

from .core.types cimport *
from .core.cmath cimport *
from .Matrix cimport Matrix
from .Partial cimport Partial
from .Product cimport Product
from .Toeplitz cimport Toeplitz
from .Eye cimport Eye
from .Diag cimport Diag
from .Kron cimport Kron
from .MLToeplitz cimport MLToeplitz
from .DiagBlocks cimport DiagBlocks


################################################################################
################################################## class Toeplitz
cdef class MLUltraSound(Partial):

    ############################################## class properties
    # vecC - Property (read-only)
    # Return the matrix-defining column vector of the circulant matrix
    property tenT:
        def __get__(self):
            return self._tenT

    ############################################## class methods
    def __init__(self, tenT, **options):
        '''
        Initialize MLUltraSound Matrix instance.
        '''

        # store the defining elements and extract the needed dimensions
        self._tenT = np.copy(tenT)
        self._numBlocksN = tenT.shape[0]
        self._numSize1 = int((tenT.shape[2] + 1) /2)
        self._numSize2 = int((tenT.shape[3] + 1) /2)
        self._arrN = np.array([self._numSize1, self._numSize2], dtype='int')

        # construct one sample MLToeplitz instance to get optimal fourier
        # sizes
        _T = MLToeplitz(np.copy(tenT[0, 0, :, :]))

        # subselection index of the embedded partials
        indK = _T.indicesN.reshape((-1, 1))

        # the diagonalizing matrices of each embedded block
        F = _T._content[0]._content[-1]

        # build up the whole diagonalizing matrix
        # TODO checkout if this can be speeded up by doing FFTs
        K = Kron(
            Eye(self._numBlocksN),
            *F._content
        )

        # allocate memory for the diagonal matrix
        diags = np.empty((
            self._numBlocksN,
            self._numBlocksN,
            F.numN
        ), dtype='complex')

        # calculate the large subselection index array for the whole matrix
        rgn = np.arange(self._numBlocksN) *F.numN
        arrIndicesN = (rgn + np.repeat(
            indK, self._numBlocksN, 1
        )).reshape(-1, order='F')

        # extract the diagonalizing stuff from the nested toeplitz matrices
        for ii in range(self._numBlocksN):
            for jj in range(self._numBlocksN):
                T = MLToeplitz(np.copy(tenT[ii, jj, :, :]))
                diags[ii, jj, :] = (T._content[0]._content[1].vecD)[:]

        # construct the composing matrices
        B = DiagBlocks(diags)
        P = Product(K.H, B, K)

        # call the parent constructor
        super(MLUltraSound, self).__init__(P, N=arrIndicesN, M=arrIndicesN)

    cpdef np.ndarray _getArray(self):
        '''Return an explicit representation of the matrix as numpy-array.'''
        return self._reference()

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        return (0., 0.)

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        arrRes = np.zeros((self.numN, self.numM), dtype=self.dtype)

        # go through all blocks and construct the corresponding
        # MLToeplitz instance by calling its reference
        for ii in range(self._numBlocksN):
            for jj in range(self._numBlocksN):
                arrRes[
                    ii * self._numSize1 * self._numSize2:
                    (ii + 1) * self._numSize1 * self._numSize2,
                    jj * self._numSize1 * self._numSize2:
                    (jj + 1) * self._numSize1 * self._numSize2
                ] = MLToeplitz(
                    np.copy(self._tenT[ii, jj, :, :])
                )._reference()

        return arrRes

    cpdef Matrix _getNormalized(self):
        cdef intsize ii, jj

        cdef intsize stride = np.prod(self._arrN)

        cdef np.ndarray arrNorms = np.zeros(self.numM)

        for ii in range(self._numBlocksN):
            for jj in range(self._numBlocksN):
                arrNorms[jj *stride:(jj +1) *stride] += self._normalizeCore(
                    self._tenT[ii, jj, :, :]
                )

        #print(np.sqrt(arrNorms))

        return self * Diag(1 / np.sqrt(arrNorms))

    def _normalizeCore(self, tenT):
        cdef intsize ii, numS1, numS2, numS3

        cdef intsize numL = int((tenT.shape[0] + 1) /2)

        cdef intsize numD = np.array(tenT.shape).shape[0]

        cdef np.ndarray arrT, arrNorms
        if numD == 1:
            # if we are deep enough we do the normal toeplitz stuff
            arrT = tenT

            arrNorms = np.zeros(numL)

            arrNorms[0] = np.linalg.norm(arrT[:numL]) **2

            for ii in range(numL - 1):
                arrNorms[ii + 1] = arrNorms[ii] \
                    + np.abs(arrT[2 * numL - 2 - ii]) ** 2 \
                    - np.abs(arrT[numL - ii - 1]) ** 2

        else:
            numS1 = np.prod(self._arrN[-numD :])
            numS2 = np.prod(self._arrN[-(numD - 1) :])
            arrNorms = np.zeros(numS1)
            arrT = np.zeros((tenT.shape[0], numS2))

            # go deeper in recursion and get norms of blocks
            for ii in range(tenT.shape[0]):
                arrT[ii, :] = self._normalizeCore(tenT[ii])

            numS3 = arrT.shape[1]
            arrNorms[:numS3] = np.sum(arrT[:numL, :], axis=0)

            # now do blockwise subtraction and addition
            for ii in range(numL - 1):
                arrNorms[
                    (ii +1) *numS2 : (ii +2) *numS2
                ] = arrNorms[ii *numS2 : (ii +1) *numS2] + \
                    + arrT[2 * numL - 2 - ii] \
                    - arrT[numL - ii - 1]
        return arrNorms

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                # 35 is just any number that causes no padding
                # 41 is the first size for which bluestein is faster
                TEST.NUM_N      : 40,
                TEST.NUM_M      : TEST.NUM_N,
                'mTypeC'        : TEST.Permutation(TEST.ALLTYPES),
                'vecC'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeC',
                    TEST.SHAPE  : (2, 2, 7, 9)
                }),
                TEST.INITARGS   : (lambda param : [param['vecC']()]),
                TEST.OBJECT     : MLUltraSound,
                TEST.TOL_POWER  : 2.,
                TEST.TOL_MINEPS : _getTypeEps(np.float64)
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c:
                                   MLUltraSound(np.random.randn(
                                       c , c, 2 * c - 1, 2 * c - 1)
                                   )),
                BENCH.FUNC_SIZE : (lambda c: c ** 3),
                BENCH.FUNC_STEP : (lambda c: c + 2)
            },
            BENCH.FORWARD: {},
            BENCH.SOLVE: {},
            BENCH.OVERHEAD: {},
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, dt: MLUltraSound(
                    np.random.randn(c, c, 2 * c - 1, 2 * c - 1).astype(dt)))
            }
        }

    def _getDocumentation(self):
        from .inspect import DOC
        return DOC.SUBSECTION(
            r'Multilevel Ultrasound Matrix (\texttt{fastmat.MLUltraSound})',
            DOC.SUBSUBSECTION(
                'Definition and Interface',
                r"""
                This class is an implementation of matrices, which have a block 
structure, where each block itself is a $2$-level Toeplitz matrix, as 
implemented in \texttt{MLToeplitz}. The name originated from the fact, that 
these matrices pop up, when modeling a ultrasonic pulse echo as a superposition 
of hyperbolas.

As such we can 
collect the unique defining elements of $\bm H$ in $\bm h \in \bm \R^{M \times 
M 
\times 2 N_x - 1 \times 2 N_y - 1}$ and then set 
%
\[
    \bm H_{i,j} = \bm T_{([N_x, N_y], 2)}(\bm h_{i,j}).
\]
%
Moreover, we can diagonalize each $\bm H_{i,j}$.
To this end we embed each $\bm H_{i,j}$ into a $2$-level circulant matrix
%
\begin{align}\label{eff_mlultra}
    \begingroup
    \setlength\arraycolsep{3pt}
    \begin{bmatrix}
    \mathfrak{T}_2(\bm H_{0,0})   & \dots  & \mathfrak{T}_2(\bm H_{0,M-1}) \\
    \vdots        & \ddots & \vdots \\
    \mathfrak{T}_2(\bm H_{M-1,0}) & \dots  & \mathfrak{T}_2(\bm H_{M-1,M-1}) \\
    \end{bmatrix} \endgroup = 
\bm K^\herm \cdot \bm D \cdot \bm K
\end{align}
%
for 
$\bm F = \bm \Fs_{2 N_x - 1} \otimes \bm \Fs_{2 N_y - 1} $,
$\bm K = \bm I_{M} \otimes \bm F$
%
and 
\[\bm D = 
\begingroup
\setlength\arraycolsep{3pt}
\begin{bmatrix}
\diag(\bm F \Vectorize \tilde{\bm h}_{0,0}) & \dots & 
\diag(\bm F \Vectorize \tilde{\bm h}_{0,M-1}) \\
\vdots & \ddots & \vdots \\
\diag(\bm F \Vectorize \tilde{\bm h}_{M-1,0}) & \dots & 
\diag(\bm F \Vectorize \tilde{\bm h}_{M-1,M-1}) \\
\end{bmatrix},
\endgroup
\]
which shows, that $\bm H$ simply is a subselection of the above block 
diagonalizated matrix. This is exactly, how it is implemented in \fm{}.
                """,
                DOC.SNIPPET('# import the package',
                            'import fastmat as fm',
                            'import numpy as np',
                            '',
                            '# construct the',
                            '# parameters',
                            'n = 2',
                            'l = 2',
                            't = np.arange(',
                            '2 *2 *(2 *n - 1) ** l',
                            ').reshape(',
                            '(n, n, 2 *n - 1, 2 *n - 1)',
                            ')',
                            '',
                            '# construct the matrix',
                            'T = fm.MLUltraSound(t)',
                            caption=r"""
This yields
\[\bm t = \begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}\]
\[\bm T = \left(\begin{array}{cccc}
    1 & 3 & 7 & 9 \\
    2 & 1 & 8 & 7 \\
    4 & 5 & 1 & 3 \\
    5 & 4 & 2 & 1
\end{array}\right)\]"""),
                r"""
This class depends on \texttt{Fourier}, \texttt{DiagBlocks}, \texttt{Kron},
\texttt{Product} and \texttt{Partial}."""
            ),
            DOC.SUBSUBSECTION(
                'Performance Benchmarks', r"""
All benchmarks were performed on a matrix
$\bm{\mathcal{C}} \in \R^{n \times n}$ with $n \in \N$ and all
entries drawn from a  standard Gaussian distribution.""",
                DOC.PLOTFORWARD(),
                DOC.PLOTFORWARDMEMORY(),
                DOC.PLOTSOLVE(),
                DOC.PLOTOVERHEAD(),
                DOC.PLOTTYPESPEED(),
                DOC.PLOTTYPEMEMORY()
            )
        )
