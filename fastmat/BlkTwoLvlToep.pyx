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
from .NDFourier cimport NDFourier

cdef class BlkTwoLvlToep(Partial):
    r"""
    Multilevel Ultrasound Matrix ``fastmat.BlkTwoLvlToep``


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
    >>> T = fm.BlkTwoLvlToep(t)

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

    This class depends on ``NDFourier``, ``DiagBlocks``, ``Kron``,
    ``Product`` and ``Partial``.
    """

    property tenT:
        r"""Return the matrix-defining tensor of the matrix"""

        def __get__(self):
            return self._tenT

    def __init__(self, **options):
        '''
        Initialize BlkTwoLvlToep Matrix instance.
        '''

        try:
            from BlkTwoLvlToepWrp import BlkTwoLvlToepGPU
            self._cudaAvailable = True
        except:
            self._cudaAvailable = False

        if self._cudaAvailable == False:
            # store the defining elements and extract the needed dimensions
            self._tenT = options['tenT']
            self._numBlocks = self._tenT.shape[0]
            if self._tenT.shape[0] != self._tenT.shape[1]:
                raise ValueError("First two dimensions must be of same size!")

            self._numSizeLvl1 = int((self._tenT.shape[2] + 1) /2)
            self._numSizeLvl2 = int((self._tenT.shape[3] + 1) /2)
            self._arrSizeLvls = np.array([self._numSizeLvl1, self._numSizeLvl2])

            # construct one sample MLToeplitz instance to get optimal fourier
            # sizes
            # we do not use optimization, since we have not implemented that
            # for the NDFourier matrix yet
            _T = MLToeplitz(np.copy(self._tenT[0, 0, :, :]), optimize=False)

            # subselection index of the embedded partials
            # they come from the embedding of a multilevel toeplitz
            # matrix into a multilevel circulant matrix
            # from this array, we only need to produce shifted versions
            # since every block is two level toeplitz
            indK = _T.indicesN.reshape((-1, 1))

            # the diagonalizing matrices of each embedded block
            F = _T._content[0]._content[-1]

            # calculate the large subselection index array for the whole matrix
            rgn = np.arange(self._numBlocks) *F.numN
            arrIndicesN = (rgn + np.repeat(
                    indK, self._numBlocks, 1
                )).reshape(-1, order='F')

            # build up the whole diagonalizing matrix
            # because each block itself is 2level toeplitz we need the identity
            # matrix to account for the non existing structure in the first
            # dimension
            K = Kron(
                Eye(self._numBlocks),
                NDFourier(self._tenT.shape[2], self._tenT.shape[3])
            )

            # we simply take 2DFFTs over the last two axes of the defining
            # tensor broadcasting over the first two axes and then we reshape
            # all of the 2D fouriertransformed arrays to a vector, since these
            # are the diagonals in fourier domain of each diagonal block
            diags = np.fft.fftn(
                self._tenT,
                axes=(2,3)
            ).reshape((self._numBlocks, self._numBlocks, -1)) / (
                self._tenT.shape[2] * self._tenT.shape[3]
            )

            # construct the composing matrices
            B = DiagBlocks(diags)
            P = Product(K.H, B, K)

            numN = np.prod(self._arrSizeLvls)

            # call the parent constructor
            super(BlkTwoLvlToep, self).__init__(
                P,
                N=arrIndicesN,
                M=arrIndicesN
            )

        else:
            print("doing Cuda!")
            self._numBlocks = options['numZ']
            self._numSizeLvl1 = options['numX']
            self._numSizeLvl2 = options['numY']

            numN = options['numZ'] * options['numZ'] * options['numZ']

            # set properties of matrix
            self._initProperties(
                numN, numN, np.complex128,
                cythonCall=True,
                forceInputAlignment=True,
                fortranStyle=True
            )


    cpdef _forwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        ''' Calculate the forward transform of this matrix.'''
        from BlkTwoLvlToepWrp import BlkTwoLvlToepGPU
        if self._cudaAvailable == True:
            F = BlkTwoLvlToepGPU(
                self._numBlocks,
                self._numSizeLvl1,
                self._numSizeLvl2
            )
            F.forward(arrX[0], arrRes[0])
        else:
            super(BlkTwoLvlToep, self)._forwardC(
                arrX,
                arrRes,
                typeX,
                typeRes
            )

    cpdef _backwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        ''' Calculate the backward transform of this matrix.'''
        self._forwardCu(arrX, arrRes)

    cpdef np.ndarray _getArray(self):
        '''Return an explicit representation of the matrix as numpy-array.'''
        return self._reference()

    cpdef Matrix _getNormalized(self):
        cdef intsize ii, jj

        cdef intsize stride = np.prod(self._arrSizeLvls)

        cdef np.ndarray arrNorms = np.zeros(self.numM)

        for ii in range(self._numBlocks):
            for jj in range(self._numBlocks):
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
            numS1 = np.prod(self._arrSizeLvls[-numD :])
            numS2 = np.prod(self._arrSizeLvls[-(numD - 1) :])
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
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        arrRes = np.zeros((self.numN, self.numM), dtype=self.dtype)

        # go through all blocks and construct the corresponding
        # MLToeplitz instance by calling its reference
        for ii in range(self._numBlocks):
            for jj in range(self._numBlocks):
                arrRes[
                    ii * self._numSizeLvl1 * self._numSizeLvl2:
                    (ii + 1) * self._numSizeLvl1 * self._numSizeLvl2,
                    jj * self._numSizeLvl1 * self._numSizeLvl2:
                    (jj + 1) * self._numSizeLvl1 * self._numSizeLvl2
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
                TEST.NUM_N      : 40,
                TEST.NUM_M      : TEST.NUM_N,
                'mTypeC'        : TEST.Permutation(TEST.ALLTYPES),
                'vecC'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeC',
                    TEST.SHAPE  : (2, 2, 7, 9)
                }),
                TEST.INITARGS   : (lambda param : [param['vecC']()]),
                TEST.OBJECT     : BlkTwoLvlToep,
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
                                   BlkTwoLvlToep(np.random.randn(
                                       c , c, 2 * c - 1, 2 * c - 1)
                                   )),
                BENCH.FUNC_SIZE : (lambda c: c ** 3),
                BENCH.FUNC_STEP : (lambda c: c + 2)
            },
            BENCH.FORWARD: {},
            BENCH.SOLVE: {},
            BENCH.OVERHEAD: {},
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, dt: BlkTwoLvlToep(
                    np.random.randn(c, c, 2 * c - 1, 2 * c - 1).astype(dt)))
            }
        }

    def _getDocumentation(self):
        return ""
