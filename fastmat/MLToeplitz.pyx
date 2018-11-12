# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=True

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
from .Fourier cimport Fourier
from .Diag cimport Diag
from .Kron cimport Kron
from .Toeplitz cimport Toeplitz


################################################################################
################################################## class Toeplitz
cdef class MLToeplitz(Partial):
    r"""

    Let :math:`d \geqslant 2`. Then, given a :math:`d`-dimensional complex
    sequence :math:`{t} = [t_{{k}}]` for :math:`{k} \in \mathbb{N}^d` a
    :math:`d`-level Toeplitz matrix :math:`{T}_{{n},d}` is recursively defined
    as

    .. math::
        {T}_{({n},d)} =
        \begin{bmatrix}
        {T}_{(1,{m}),\ell}        & {T}_{(2 n_1 - 1,{m}),\ell}
        & \dots     & {T}_{(n_1 + 1,{m}),\ell}    \\
        {T}_{(2,{m}),\ell}        & {T}_{(1,{m}),\ell}
        & \dots     & {T}_{(n_1 + 2,{m}),\ell}    \\
        \vdots                          & \vdots
        & \ddots    & \vdots                            \\
        {T}_{(n_1,{m}),\ell}      & {T}_{(n_1 - 1,{m}),\ell}
        & \dots     & {T}_{(1,{m}),\ell}          \\
        \end{bmatrix},

    where :math:`m =  n_{-1}` and :math:`\ell = d-1`. So for :math:`n = [2,2]`
    and :math:`t \in \mathbb{C}^{3 \times 3}` we get

    .. math::
        T_{[2,2],2} =
        \begin{bmatrix}
         T_{[1,2],1} &  T_{[3,2],1} \\
         T_{[2,2],1} &  T_{[1,2],1}
        \end{bmatrix}
        =
        \begin{bmatrix}
        t_{1,1} & t_{1,3} & t_{3,1} & t_{3,3} \\
        t_{1,2} & t_{1,1} & t_{3,2} & t_{3,1} \\
        t_{2,1} & t_{2,2} & t_{1,1} & t_{1,3} \\
        t_{2,2} & t_{2,1} & t_{1,2} & t_{1,1}
        \end{bmatrix}.

    >>> # import the package
    >>> import fastmat as fm
    >>> import numpy as np
    >>> # construct the
    >>> # parameters
    >>> n = 2
    >>> l = 2
    >>> t = np.arange((2 * n - 1) ** l).reshape(
    >>>     (2 * n - 1, 2 * n - 1)
    >>> )
    >>> # construct the matrix
    >>> T = fm.MLToeplitz(t)

    This yields

    .. math::
        t = \begin{bmatrix}
                1 & 2 & 3 \\
                4 & 5 & 6 \\
                7 & 8 & 9
            \end{bmatrix}

    .. math::
        T = \begin{bmatrix}
                1 & 3 & 7 & 9 \\
                2 & 1 & 8 & 7 \\
                4 & 5 & 1 & 3 \\
                5 & 4 & 2 & 1
            \end{bmatrix}

    This class depends on ``Fourier``, ``Diag``, ``Kron``,
    ``Product`` and ``Partial``.
    """

    property tenT:
        r"""Return the matrix-defining tensor of the circulant"""

        def __get__(self):
            return self._tenT

    ############################################## class methods
    def __init__(self, tenT, **options):
        '''
        Initialize Multilevel Toeplitz matrix instance.

        Parameters
        ----------
        tenC : :py:class:`numpy.ndarray`
            The generating nd-array defining the toeplitz tensor. The matrix
            data type is determined by the data type of this array.

        **options:
            See the special options of :py:class:`fastmat.Fourier`, which are
            also supported by this matrix and the general options offered by
            :py:meth:`fastmat.Matrix.__init__`.
        '''
        cdef intsize inn, nn

        self._tenT = _arrSqueezedCopy(tenT)

        if self._tenT.ndim < 1:
            raise ValueError("Column-definition tensor must be at least 1D")

        # extract the level dimensions from the defining tensor
        self._arrDim = (
            (np.array((<object> self._tenT).shape) + 1) / 2).astype('int')

        # get the size of the matrix
        cdef intsize numN = np.prod(self._arrDim)

        # stages during optimization
        cdef int maxStage = options.get('maxStage', 4)

        # minimum number to pad to during optimization and helper arrays
        cdef np.ndarray arrDimpad = 2 * self._arrDim - 1
        cdef np.ndarray arrOptSize = np.zeros_like(self._arrDim)
        cdef np.ndarray arrDoOpt = np.zeros_like(self._arrDim)

        cdef bint optimize = options.get('optimize', True)
        if optimize:
            # go through all level dimensions and get optimal FFT size
            for inn, nn in enumerate(arrDimpad):
                arrOptSize[inn] = _findOptimalFFTSize(nn, maxStage)

                # use this size, if we get better in that level
                if _getFFTComplexity(arrOptSize[inn]) < _getFFTComplexity(nn):
                    arrDoOpt[inn] = 1

        arrDoOpt = arrDoOpt == 1
        cdef np.ndarray arrDimopt = np.copy(2 * self._arrDim - 1)

        # set the optimization size to the calculated one
        arrDimopt[arrDoOpt] = arrOptSize[arrDoOpt]

        # get the size of the zero padded matrix
        cdef intsize numNopt = np.prod(arrDimopt)

        # allocate memory for the tensor in MD fourier domain
        cdef np.ndarray tenThat = np.empty_like(self._tenT, dtype='complex')
        tenThat[:] = self._tenT[:]

        # go through the array and apply the preprocessing in direction
        # of each axis. this cannot be done without the for loop, since
        # manipulations always influence the data for the next dimension
        for ii in range(len(self._arrDim)):
            tenThat = np.apply_along_axis(
                self._preProcSlice,
                ii,
                tenThat,
                ii,
                arrDimopt,
                self._arrDim
            )

        # after correct zeropadding, go into fourier domain
        tenThat = np.fft.fftn(tenThat).reshape(numNopt) / numNopt

        # subselection array
        cdef np.ndarray arrIndices = np.arange(numNopt)[
            self._genArrS(self._arrDim, arrDimopt)
        ]

        # create the decomposing kronecker product
        cdef Kron KN = Kron(*list(map(
            lambda ii : Fourier(ii, optimize=False), arrDimopt
        )), **options)

        # now decompose the ML matrix as a product
        cdef Product P = Product(KN.H, Diag(tenThat, **options), KN, **options)

        # initialize Partial of Product. Only use Partial when padding size
        cdef dict kwargs = options.copy()
        cdef bint truncate = not np.allclose(self._arrDim, arrDimopt)
        kwargs['rows'] = (arrIndices if truncate else None)
        kwargs['cols'] = (arrIndices if truncate else None)

        super(MLToeplitz, self).__init__(P, **kwargs)

        # Currently Fourier matrices bloat everything up to complex double
        # precision, therefore make sure tenT matches the precision of the
        # matrix itself
        if self.dtype != self._tenT.dtype:
            self._tenT = self._tenT.astype(self.dtype)

    cpdef np.ndarray _getArray(self):
        return self._reference()

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        return (0., 0.)

    cpdef np.ndarray _preProcSlice(
        self,
        np.ndarray theSlice,
        int numSliceInd,
        np.ndarray arrDimopt,
        np.ndarray arrDim
    ):
        '''
        Preprocess one axis of the defining tensor.

        Here we check for one dimension, whether it makes sense to zero-pad or
        not by estimating the FFT-complexity in each dimension.

        Parameters
        ----------
        theSlice : :py:class:`numpy.ndarray`
            ?

        numSliceInd : int
            ?

        arrDimopt : :py:class:`numpy.ndarray`
            ?

        arrDim : :py:class:`numpy.ndarray`
            ?
        '''
        cdef np.ndarray z, arrRes = np.empty(1)

        if arrDimopt[numSliceInd] > 2 *arrDim[numSliceInd] -1:
            z = np.zeros(arrDimopt[numSliceInd] - 2 *arrDim[numSliceInd] +1)
            arrRes = np.concatenate((
                np.copy(theSlice[:arrDim[numSliceInd]]),
                z,
                np.copy(theSlice[arrDim[numSliceInd]:])
            ))
        else:
            arrRes = np.copy(theSlice)

        return arrRes

    cpdef np.ndarray _genArrS(
        self,
        np.ndarray arrDim,
        np.ndarray arrDimout,
        bint verbose=False
    ):
        '''
        Filter out the non-zero elements in the padded version of X iteratively.

        One can achieve this from a zero-padded version of the tensor Xpadten,
        but the procedure itself is very helpful for understanding how the
        nested levels have an impact on the padding structure.

        Parameters
        ----------
        arrDim : :py:class:`numpy.ndarray`
            The original sizes of the defining tensor.

        arrDimout : :py:class:`numpy.ndarray`
            The desired sizes of the tensor.

        arrDimopt : :py:class:`numpy.ndarray`
            The size we should optimize to.

        arrDim : :py:class:`numpy.ndarray`
            The size the dimension originally had.

        verbose : bool
            Output verbose information.

            Defaults to False.

        Returns
        -------
        arrS : :py:class:`numpy.ndarray`
            The output array which does the selection.
        '''
        cdef intsize ii, n = arrDim.shape[0]

        # output size of the matrix we embed into
        cdef intsize numNout = np.prod(arrDimout)

        # initialize the result as all ones
        cdef np.ndarray arrS = np.arange(numNout) >= 0

        for ii in range(n):
            if verbose:
                print("state", arrS)
                print("modulus", np.mod(
                    np.arange(numNout),
                    np.prod(arrDimout[:(n -ii)])
                ))
                print("inequ",
                      arrDim[n -1 -ii] * np.prod(arrDimout[:(n -1 -ii)]))
                print("res", np.mod(
                    np.arange(numNout),
                    np.prod(arrDimout[:(n -1 -ii)])
                ) < arrDim[n -1 -ii] * np.prod(arrDimout[:(n -1 -ii)]))
            # iteratively subselect more and more indices in arrS
            np.logical_and(
                arrS,
                np.mod(
                    np.arange(numNout),
                    np.prod(arrDimout[ii:])
                ) < arrDim[ii] * np.prod(arrDimout[ii +1:]),
                arrS
            )

        return arrS

    cpdef Matrix _getNormalized(self):
        arrDimorms = self._normalizeCore(self._tenT)

        return self * Diag(1. / np.sqrt(arrDimorms))

    cpdef np.ndarray _normalizeCore(self, np.ndarray tenT):
        cdef intsize ii, numS1, numS2, numS3

        cdef intsize numL = int((tenT.shape[0] + 1) /2)
        cdef intsize numEll = tenT.shape[0]

        cdef intsize numD = tenT.ndim

        cdef np.ndarray arrT, arrDimorms
        if numD == 1:
            # if we are deep enough we do the normal toeplitz stuff
            arrT = tenT

            arrDimorms = np.zeros(numL)

            arrDimorms[0] = np.linalg.norm(arrT[:numL]) **2

            for ii in range(numL - 1):

                arrDimorms[ii + 1] = arrDimorms[ii] \
                    + np.abs(arrT[2 * numL - 2 - ii]) ** 2 \
                    - np.abs(arrT[numL - ii - 1]) ** 2

        else:
            numS1 = np.prod(self._arrDim[-numD :])
            numS2 = np.prod(self._arrDim[-(numD - 1) :])
            arrDimorms = np.zeros(numS1)
            arrT = np.zeros((numEll, numS2))

            # go deeper in recursion and get norms of blocks
            for ii in range(numEll):
                arrT[ii, :] = self._normalizeCore(tenT[ii, :])

            numS3 = arrT.shape[1]
            arrDimorms[:numS3] = np.sum(arrT[:numL, :], axis=0)

            # now do blockwise subtraction and addition
            for ii in range(numL - 1):

                arrDimorms[
                    (ii +1) *numS2 : (ii +2) *numS2
                ] = arrDimorms[ii *numS2 : (ii +1) *numS2] + \
                    + arrT[2 * numL - 2 - ii] \
                    - arrT[numL - ii - 1]

        return arrDimorms

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        return self._refRecursion(self._arrDim, self._tenT, False)

    def _refRecursion(
        self,
        np.ndarray arrDim,
        np.ndarray tenU,
        bint verbose=False
    ):
        '''
        Build the d-level circulant matrix recursively from a d-dimensional
        tensor.

        Build the (d-1) level matrices first and put them to the correct
        locations for d=1.

        Parameters
        ----------
        arrDim : :py:class:`numpy.ndarray`
            The dimensions in each level.

        tenC : :py:class:`numpy.ndarray`
            The defining elements.

        verbose : bool
            Output verbose information.

            Defaults to False.
        '''
        cdef intsize nn_, mm, countAbs

        # number of dimensions
        cdef intsize numD = arrDim.shape[0]

        # get size of resulting block toeplitz matrix
        cdef intsize numN = np.prod(arrDim)

        # get an array of all partial sequential products
        # starting at the front
        cdef np.ndarray arrDimprod = np.array(
            list(map(lambda ii : np.prod(arrDim[ii:]), range(numN - 1)))
        )

        # permutation array for block placement, since we need to place the
        # blocks in the same fashion, we arrange the elements in the blocks,
        # such that the preprocessing does the right thing
        cdef np.ndarray arrP = np.arange(2 * arrDim[0] - 1)
        arrP = np.concatenate((arrP[:arrDim[0]][::-1], arrP[arrDim[0]:][::-1]))
        arrP = np.argsort(arrP)

        # allocate memory for the result
        cdef np.ndarray T = np.zeros((numN, numN), dtype=self.dtype)
        cdef np.ndarray subT

        # check if we can go a least a level deeper
        if numD > 1:

            # iterate over size of the first dimension
            for nn_ in range(2 * arrDim[0] - 1):

                # select the right block position with the permutation array
                nn       = arrP[nn_]
                tmp      = nn - arrDim[0] + 1
                countAbs = arrDim[0] - abs(tmp)
                countSig = 0 if tmp == 0 else tmp / abs(tmp)

                # now calculate the block recursively
                subT     = self._refRecursion(arrDim[1 :], tenU[nn_])

                # decide whether it is below or above the diagonal or on it
                # and the act accordingly
                if countSig == 0:
                    # we are on the diagonal
                    for mm in range(countAbs):
                        T[
                            mm * arrDimprod[1] : (mm + 1) * arrDimprod[1],
                            mm * arrDimprod[1] : (mm + 1) * arrDimprod[1]
                        ] = subT
                elif countSig < 0:
                    # we are below the diagonal
                    for mm in range(countAbs):
                        T[
                            (mm + abs(tmp))     * arrDimprod[1] :
                            (mm + 1 + abs(tmp)) * arrDimprod[1],
                            mm       * arrDimprod[1]            :
                            (mm + 1) * arrDimprod[1]
                        ] = subT
                else:
                    # we are above the diagonal
                    for mm in range(countAbs):
                        T[
                            mm * arrDimprod[1] : (mm + 1) * arrDimprod[1],
                            (mm + abs(tmp))     * arrDimprod[1] :
                            (mm + 1 + abs(tmp)) * arrDimprod[1],
                        ] = subT

            return T
        else:
            # if we are in a lowest level, we just construct the right
            # single level toeplitz block
            return Toeplitz(tenU[:numN], tenU[numN:][::-1]).array

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                # 35 is just any number that causes no padding
                # 41 is the first size for which bluestein is faster
                TEST.NUM_ROWS   : 27,
                TEST.NUM_COLS   : TEST.NUM_ROWS,
                'mTypeC'        : TEST.Permutation(TEST.FEWTYPES),
                'optimize'      : True,
                TEST.PARAMALIGN : TEST.ALIGNMENT.DONTCARE,
                'vecC'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeC',
                    TEST.SHAPE  : (5, 5, 5),
                    TEST.ALIGN  : TEST.PARAMALIGN
                }),
                TEST.INITARGS   : (lambda param : [param['vecC']()]),
                TEST.INITKWARGS : {'optimize' : 'optimize'},
                TEST.OBJECT     : MLToeplitz,
                TEST.NAMINGARGS : dynFormat("%s,optimize=%s",
                                            'vecC', str('optimize')),
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
                                   MLToeplitz(np.random.randn(
                                       *(2 * [(2 * c - 1) + 2])
                                   ))),
                BENCH.FUNC_SIZE : (lambda c: (c + 1) ** 2),
                BENCH.FUNC_STEP : (lambda c: c * 10 ** (1. / 12))
            },
            BENCH.FORWARD: {},
            BENCH.SOLVE: {},
            BENCH.OVERHEAD: {},
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, dt: MLToeplitz(
                    np.random.randn(*(2 * [(2 * c - 1) + 2])).astype(dt)))
            }
        }
