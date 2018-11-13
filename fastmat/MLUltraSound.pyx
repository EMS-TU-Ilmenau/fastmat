# -*- coding: utf-8 -*-
#cython: boundscheck=False

# Copyright 2016 Sebastian Semper, Christoph Wagner
#     https://www.tu-ilmenau.de/it-ems/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

cdef class MLUltraSound(Partial):
    r"""
    Multilevel Ultrasound Matrix ``fastmat.MLUltraSound``


    This class is an implementation of matrices, which have a block
    structure, where each block itself is a :math:`2`-level Toeplitz matrix, as
    implemented in ``MLToeplitz``. The name originated from the fact, that
    these matrices pop up, when modeling a ultrasonic pulse echo as a
    superposition of hyperbolas.

    As such we can collect the unique defining elements of :math:`H` in
    :math:`h \in \mathbb{R}^{M \times M \times 2 N_x - 1 \times 2 N_y - 1}` and
    then set

    .. math::
        H_{i,j} =  T_{([N_x, N_y], 2)}( h_{i,j}).

    Moreover, we can diagonalize each :math:`H_{i,j}`.
    To this end we embed each :math:`H_{i,j}` into a :math:`2`-level circulant
    matrix

    .. math::
        \begin{bmatrix}
        \mathfrak{T}_2( H_{0,0})   & \dots  & \mathfrak{T}_2( H_{0,M-1}) \\
        \vdots        & \ddots & \vdots \\
        \mathfrak{T}_2( H_{M-1,0}) & \dots  & \mathfrak{T}_2( H_{M-1,M-1}) \\
        \end{bmatrix}  =
        K^\mathrm{H} \cdot  D \cdot  K

    for :math:`F =  \mathcal{F}_{2 N_x - 1} \otimes  \mathcal{F}_{2 N_y - 1} `,
    :math:`K =  I_{M} \otimes  F` and

    .. math::
        D =
        \begin{bmatrix}
        \mathrm{diag}( F \mathrm{vec} \tilde{ h}_{0,0}) & \dots &
        \mathrm{diag}( F \mathrm{vec} \tilde{ h}_{0,M-1}) \\
        \vdots & \ddots & \vdots \\
        \mathrm{diag}( F \mathrm{vec} \tilde{ h}_{M-1,0}) & \dots &
        \mathrm{diag}( F \mathrm{vec} \tilde{ h}_{M-1,M-1}) \\
        \end{bmatrix},

    which shows, that :math:`H` simply is a subselection of the above block
    diagonalizated matrix. This is exactly, how it is implemented in
    ``fastmat``.

    >>> # import the package
    >>> import fastmat as fm
    >>> import numpy as np
    >>>
    >>> # construct the
    >>> # parameters
    >>> n = 2
    >>> l = 2
    >>> t = np.arange(
    >>> 2 *2 *(2 *n - 1) ** l
    >>> ).reshape(
    >>> (n, n, 2 *n - 1, 2 *n - 1)
    >>> )
    >>>
    >>> # construct the matrix
    >>> T = fm.MLUltraSound(t)

    This yields

    .. math::
        t = \begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}

    .. math::
        T = \begin{bmatrix}
        1 & 3 & 7 & 9 \\
        2 & 1 & 8 & 7 \\
        4 & 5 & 1 & 3 \\
        5 & 4 & 2 & 1
        \end{bmatrix}

    This class depends on ``Fourier``, ``DiagBlocks``, ``Kron``,
    ``Product`` and ``Partial``.
    """

    property tenT:
        r"""Return the matrix-defining tensor of the matrix"""

        def __get__(self):
            return self._tenT

    def __init__(self, tenT, **options):
        '''
        Initialize MLUltraSound Matrix instance.
        '''

        # store the defining elements and extract the needed dimensions
        self._tenT = np.copy(tenT)
        cdef intsize numDim0 = tenT.shape[0]
        self._arrDim = np.array(
            [numDim0, (tenT.shape[2] + 1) // 2, (tenT.shape[3] + 1) // 2],
            dtype='int'
        )

        # construct one sample MLToeplitz instance to get optimal fourier
        # sizes
        _T = MLToeplitz(np.copy(tenT[0, 0, :, :]))

        # subselection index of the embedded partials
        indK = _T.rowSelection.reshape((-1, 1))

        # the diagonalizing matrices of each embedded block
        F = _T._content[0]._content[-1]

        # build up the whole diagonalizing matrix
        # TODO checkout if this can be speeded up by doing FFTs
        K = Kron(Eye(self._arrDim[0]), *F._content)

        # allocate memory for the diagonal matrix
        diags = np.empty((numDim0, numDim0, F.numRows), dtype='complex')

        # calculate the large subselection index array for the whole matrix
        rgn = np.arange(numDim0) * F.numRows
        arrIndicesN = (rgn + np.repeat(
            indK, numDim0, 1
        )).reshape(-1, order='F')

        # extract the diagonalizing stuff from the nested toeplitz matrices
        for ii in range(numDim0):
            for jj in range(numDim0):
                T = MLToeplitz(np.copy(tenT[ii, jj, :, :]))
                diags[ii, jj, :] = (T._content[0]._content[1].vecD)[:]

        # construct the composing matrices
        B = DiagBlocks(diags)
        P = Product(K.H, B, K)

        # call the parent constructor
        cdef dict kwargs = options.copy()
        kwargs['rows'] = arrIndicesN
        kwargs['cols'] = arrIndicesN
        super(MLUltraSound, self).__init__(P, **kwargs)

        # Currently Fourier matrices bloat everything up to complex double
        # precision, therefore make sure tenT matches the precision of the
        # matrix itself
        if self.dtype != self._tenT.dtype:
            self._tenT = self._tenT.astype(self.dtype)

    cpdef np.ndarray _getArray(self):
        return self._reference()

    cpdef Matrix _getNormalized(self):
        cdef intsize ii, jj, numDim0 = self._arrDim[0]

        cdef intsize stride = np.prod(self._arrDim[1:])

        cdef np.ndarray arrNorms = np.zeros(self.numCols)

        for ii in range(numDim0):
            for jj in range(numDim0):
                arrNorms[jj * stride:(jj +1) * stride] += self._normalizeCore(
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
            numS1 = np.prod(self._arrDim[-numD :])
            numS2 = np.prod(self._arrDim[-(numD - 1) :])
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

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        return (0., 0.)

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        arrRes = np.zeros((self.numRows, self.numCols), dtype=self.dtype)

        cdef intsize numDim0 = self._arrDim[0]
        cdef intsize numDim1 = self._arrDim[1]
        cdef intsize numDim2 = self._arrDim[2]

        # go through all blocks and construct the corresponding
        # MLToeplitz instance by calling its reference
        for ii in range(numDim0):
            for jj in range(numDim0):
                arrRes[
                    ii * numDim1 * numDim2:
                    (ii + 1) * numDim1 * numDim2,
                    jj * numDim1 * numDim2:
                    (jj + 1) * numDim1 * numDim2
                ] = MLToeplitz(
                    np.copy(self._tenT[ii, jj, :, :])
                )._reference()

        return arrRes

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                # 35 is just any number that causes no padding
                # 41 is the first size for which bluestein is faster
                TEST.NUM_ROWS   : 40,
                TEST.NUM_COLS   : TEST.NUM_ROWS,
                'mTypeC'        : TEST.Permutation(TEST.FEWTYPES),
                TEST.PARAMALIGN : TEST.ALIGNMENT.DONTCARE,
                TEST.DATAALIGN  : TEST.ALIGNMENT.DONTCARE,
                'vecC'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeC',
                    TEST.SHAPE  : (2, 2, 7, 9)
                }),
                TEST.INITARGS   : (lambda param : [param['vecC']()]),
                TEST.OBJECT     : MLUltraSound,
                TEST.TOL_POWER  : 2.,
                TEST.TOL_MINEPS : getTypeEps(np.float64)
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
