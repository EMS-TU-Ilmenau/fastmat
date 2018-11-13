# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False

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
from .Circulant cimport Circulant


cdef class MLCirculant(Partial):
    r"""
    Multilevel Circulant Matrices are not circulant by themselves, but consist
    of multiply nested levels of circulant structures. To this end, let
    :math:`d \geqslant 2`, :math:`n = [n_1, \dots, n_d]`,
    :math:`n_{1-} = [n_1,\dots, n_{d-1}]` and :math:`m = [n_2,\dots, n_d]`.
    Then, given a :math:`d`-dimensional complex sequence
    :math:`c = [c_{k}]` for :math:`k \in \mathbb{N}^d`
    a :math:`d`-level circulant matrix :math:`C_{n,d}` is recursively defined as

    .. math::C_{n,d} =
        \begin{bmatrix}
        {C}_{[1,{m}],\ell}        & {C}_{[n_1,{m}],\ell}
        & \dots     & {C}_{[2,{m}],\ell}  \\
        {C}_{[2,{m}],\ell}        & {C}_{[1,{m}],\ell}
        & \dots     & {C}_{[3,{m}],\ell}  \\
        \vdots                              & \vdots
        & \ddots    & \vdots                        \\
        {C}_{[n_1,{m}],\ell}      & {C}_{[n_1 - 1,{m}],\ell}
        & \dots     & {C}_{[1,{m}],\ell}  \\
        \end{bmatrix}.

    So for :math:`n = (2,2)` and :math:`c \in \mathbb{C}^{2 \times 2}` we get

    .. math::
        C_{[2,2],2} =
        \begin{bmatrix}
        C_{[1,2],1} & C_{[2,2],1} \\
        C_{[2,2],1} & C_{[1,2],1}
        \end{bmatrix}
        =
        \begin{bmatrix}
        c_{1,1} & c_{1,2} & c_{2,1} & c_{2,2} \\
        c_{1,2} & c_{1,1} & c_{2,2} & c_{2,1} \\
        c_{2,1} & c_{2,2} & c_{1,1} & c_{1,2} \\
        c_{2,2} & c_{2,1} & c_{1,2} & c_{1,1}
        \end{bmatrix}.

    The approach we follow here is similar to the Circulant matrix case. But
    here, we have matrix, which is defined by a tensor of order d because of
    its d-level nature. The remarkable thing is, that a very analogue thing as
    to the Circulant case holds. We only have to calculate an d-dimensional
    fourier transform on the defining tensor. then its vectorized version is
    the spectrum of the d-level circulant matrix it defines. Another way to
    view this, is that a kronecker product of fourier matrices diagonalizes
    this d-level matrix. So, we have that

            C = (F_n1 kron ... kron F_nd)^H
                 * diag((F_n1 kron ... kron F_nd) * tenC)
                 * (F_n1 kron ... kron F_nd)

    As another performance improvement, we do not simply calculate the fourier
    transform of tenC in every dimension, but instead we make use of the
    bluestein algorithm in each dimension independently. as such the d-level
    circulant matrix gets embedded into a larger d-level circulant matrix,
    which is then more efficient in reducing the bottleneck of the FFTs. this
    is why this matrix class is derived from Partial.

    >>> # import the package
    >>> import fastmat as fm
    >>> import numpy as np
    >>> # construct the
    >>> # parameters
    >>> n = 2
    >>> l = 2
    >>> c = np.arange(n ** l).reshape((n,n))
    >>> # construct the matrix
    >>> C = fm.MLCirculant(c)

    This then yields

    .. math::
        c = \begin{bmatrix}
            0 & 1 \\
            2 & 3
            \end{bmatrix}

    and thus

    .. math::
        C = \begin{bmatrix}
            0 & 1 & 2 & 3 \\
            1 & 0 & 3 & 2 \\
            2 & 3 & 0 & 1 \\
            3 & 2 & 1 & 0
            \end{bmatrix},

    This class depends on ``fm.Fourier``, ``fm.Diag``, ``fm.Kron``,
    ``fm.Product`` and ``fm.Partial``.

    .. todo::
        - save memory by not storing tenC but only its fourier transform

    """

    property tenC:
        r"""Return the matrix-defining column vector of the circulant matrix"""

        def __get__(self):
            return self._tenC

    def __init__(self, tenC, **options):
        '''
        Initialize Multilevel Circulant matrix instance.

        Parameters
        ----------
        tenC : :py:class:`numpy.ndarray`
            The generating nd-array defining the circulant tensor. The matrix
            data type is determined by the data type of this array.

        **options:
            See the special options of :py:class:`fastmat.Fourier`, which are
            also supported by this matrix and the general options offered by
            :py:meth:`fastmat.Matrix.__init__`.
        '''
        cdef intsize iin, nn

        # copy tensor in class itself
        self._tenC = _arrSqueezedCopy(tenC)

        if self._tenC.ndim < 1:
            raise ValueError("Column-definition tensor must be at least 1D.")

        # extract the level dimensions from the defining tensor
        self._arrDim = np.array((<object> self._tenC).shape)

        # get the size of the matrix, which must be the product of the
        # defining tensor
        cdef intsize numRows = np.prod(self._arrDim)

        # stages during optimization to get the best FFT size
        cdef int maxStage = options.get('maxStage', 4)

        # minimum numbers to pad to during FFT optimization
        cdef np.ndarray arrNpad = 2 * self._arrDim - 1

        # array that will hold the calculated optimal FFT sizes
        cdef np.ndarray arrOptSize = np.zeros_like(self._arrDim)

        # array that will hold 0 if we don't do an expansion of the
        # FFT size into this dimension and 1 if we do.
        # the dimensions are sorted like in arrDim
        cdef np.ndarray arrDoOpt = np.zeros_like(self._arrDim)

        # check if the optimization flag was set
        cdef bint optimize = options.get('optimize', True)

        if optimize:
            # go through all level dimensions and get optimal FFT size
            for inn, nn in enumerate(arrNpad):
                arrOptSize[inn] = _findOptimalFFTSize(nn, maxStage)

                # use this size, if we get better in that level
                if (_getFFTComplexity(arrOptSize[inn]) <
                        _getFFTComplexity(self._arrDim[inn])):
                    arrDoOpt[inn] = 1

        # convert the array to a boolean array
        arrDoOpt = arrDoOpt == 1
        cdef np.ndarray arrNopt = np.copy(self._arrDim)

        # set the optimization size to the calculated one by replacing
        # the original sizes by the calculated better FFT sizes
        arrNopt[arrDoOpt] = arrOptSize[arrDoOpt]

        # get the size of the inflated matrix
        cdef intsize numRowsopt = np.prod(arrNopt)

        # allocate memory for the tensor in d-dimensional fourier domain
        # and save the memory
        cdef np.ndarray tenChat = np.empty_like(self._tenC, dtype='complex')
        tenChat[:] = self._tenC[:]

        # go through the array and apply the preprocessing in direction
        # of each axis. where we apply the preprocessing for the bluestein
        # algorithm in every direction
        # this cannot be done without the for loop, since
        # manipulations always influence the data for the next dimension
        for ii in range(len(self._arrDim)):
            tenChat = np.apply_along_axis(
                self._preProcSlice,
                ii,
                tenChat,
                ii,
                arrNopt,
                self._arrDim
            )

        # after correct zeropadding, go into fourier domain by calculating the
        # d-dimensional fourier transform on the preprocessed tensor
        tenChat = np.fft.fftn(tenChat).reshape(numRowsopt) / numRowsopt

        # subselection array to remember the parts of the inflated matrix,
        # where the original d-level circulant matrix has its entries
        cdef np.ndarray arrIndices = np.arange(numRowsopt)[
            self._genArrS(self._arrDim, arrNopt)
        ]

        # create the decomposing kronecker product, which realizes
        # the d-dimensional FFT with measures offered by fastmat
        cdef Kron KN = Kron(*list(map(
            lambda ii : Fourier(ii, optimize=False), arrNopt
        )))

        # now describe the d-level matrix as a product as denoted in the
        # introduction of this function
        cdef Product P = Product(KN.H, Diag(tenChat),
                                 KN, **options)

        # initialize Partial of Product. Only use Partial when
        # inflating the size of the matrix
        cdef bint truncate = not np.allclose(self._arrDim, arrNopt)
        cdef dict kwargs = options.copy()
        kwargs['rows'] = (arrIndices if truncate else None)
        kwargs['cols'] = (arrIndices if truncate else None)

        super(MLCirculant, self).__init__(P, **kwargs)

        # Currently Fourier matrices bloat everything up to complex double
        # precision, therefore make sure tenC matches the precision of the
        # matrix itself
        if self.dtype != self._tenC.dtype:
            self._tenC = self._tenC.astype(self.dtype)

    cpdef Matrix _getNormalized(self):
        # As in the single level Circulant case, we simply calculate the
        # frobenius norm of the whole tensor
        norm = np.linalg.norm(self._tenC.reshape((-1)))
        return self * (1. / norm)

    cpdef np.ndarray _getArray(self):
        return self._reference()

    cpdef np.ndarray _preProcSlice(
        self,
        np.ndarray theTensor,
        int numTensorDim,
        np.ndarray arrNopt,
        np.ndarray arrN
    ):
        '''
        Preprocess one axis of the defining tensor.

        Here we check for one dimension, whether it makes sense to zero-pad or
        not by comparing arrNopt and arrN. If arrNopt is larger, it seems like
        it makes sense to do zero padding into this dimension and then we do
        exactly that.

        Parameters
        ----------
        theTensor : :py:class:`numpy.ndarray`
            The tensor we do the proprocessing on.

        numTensorDim : int
            The current dimension we are operating on.

        arrNopt : :py:class:`numpy.ndarray`
            The size we should optimize to.

        arrN : :py:class:`numpy.ndarray`
            The size the dimension originally had.
        '''
        cdef np.ndarray arrRes = np.empty(1)
        cdef np.ndarray z

        if arrNopt[numTensorDim] > arrN[numTensorDim]:
            z = np.zeros(arrNopt[numTensorDim] - 2 * arrN[numTensorDim] + 1)
            return np.concatenate((theTensor, z, theTensor[1:]))
        else:
            return np.copy(theTensor)

    cpdef np.ndarray _genArrS(
        self,
        np.ndarray arrN,
        np.ndarray arrNout,
        bint verbose=False
    ):
        '''
        Filter out the non-zero elements in the padded version of X iteratively.

        One can achieve this from a zero-padded version of the tensor Xpadten,
        but the procedure itself is very helpful for understanding how the
        nested levels have an impact on the padding structure.

        Parameters
        ----------
        arrN : :py:class:`numpy.ndarray`
            The original sizes of the defining tensor.

        arrNout : :py:class:`numpy.ndarray`
            The desired sizes of the tensor.

        arrNopt : :py:class:`numpy.ndarray`
            The size we should optimize to.

        arrN : :py:class:`numpy.ndarray`
            The size the dimension originally had.

        verbose : bool
            Output verbose information.

            Defaults to False.

        Returns
        -------
        arrS : :py:class:`numpy.ndarray`
            The output array which does the selection.
        '''
        cdef intsize ii, n = arrN.shape[0]

        # output size of the matrix we embed into
        cdef intsize numRowsout = np.prod(arrNout)

        # initialize the result as all ones
        cdef np.ndarray arrS = np.arange(numRowsout) >= 0

        # now buckle up!
        # we go through all dimensions first
        for ii in range(n):
            if verbose:
                print("state", arrS)
                print("modulus", np.mod(
                    np.arange(numRowsout),
                    np.prod(arrNout[:(n -ii)])
                ))
                print("inequ", arrN[n -1 -ii] * np.prod(arrNout[:(n -1 -ii)]))
                print("res", np.mod(
                    np.arange(numRowsout),
                    np.prod(arrNout[:(n -1 -ii)])
                ) < arrN[n -1 -ii] * np.prod(arrNout[:(n -1 -ii)]))

            # iteratively subselect more and more indices in arrS
            # we do this the following way first we filter out all ones, that
            # originate from expanding in the first dimension and then in
            # the second and so forth. this filtering is done by logically
            # ANDing the current arrS with another array, which is generated
            # with the modulo operation, which generates a repeating pattern of
            # numbers from 0 .. T_k .. N_k with lower and lower N_k and then we
            # set everyhing to false, which is larger than T_k, where T_k
            # corresponds to a size which is a conglomerate of arrN and arrNopt.

            np.logical_and(
                arrS,
                np.mod(
                    np.arange(numRowsout),
                    np.prod(arrNout[ii:])
                ) < arrN[ii] * np.prod(arrNout[ii +1:]),
                arrS
            )
        return arrS

    cpdef np.ndarray _reference(self):
        return self._refRecursion(self._arrDim, self._tenC, False)

    def _refRecursion(
        self,
        np.ndarray arrN,
        np.ndarray tenC,
        bint verbose=False
    ):
        '''
        Build the d-level circulant matrix recursively from a d-dimensional
        tensor.

        Build the (d-1) level matrices first and put them to the correct
        locations for d=1.

        Parameters
        ----------
        arrN : :py:class:`numpy.ndarray`
            The dimensions in each level.

        tenC : :py:class:`numpy.ndarray`
            The defining elements.

        verbose : bool
            Output verbose information.

            Defaults to False.

        Returns
        -------
        arrC : :py:class:`numpy.ndarray`
            The resulting d-level matrix.
        '''
        cdef intsize nn, ii, NN, MM

        # the submatrix, which is (d-1)-level at the current iteration position
        cdef np.ndarray subC

        # number of dimensions (levels = d)
        cdef intsize numD = arrN.shape[0]

        # get size of resulting block circulant matrix
        cdef intsize numRows = np.prod(arrN)

        # product of all dimensions
        cdef np.ndarray arrNprod = np.array(
            list(map(lambda ii : np.prod(arrN[ii:]), range(len(arrN) + 1)))
        )
        if verbose:
            print(numD, arrN)
            print()
            print(tenC)
            print(arrNprod)

        # The resulting d-level matrix
        cdef np.ndarray arrC = np.zeros((numRows, numRows), dtype=self.dtype)

        if numD > 1:
            # iterate over dimensions
            for nn in range(arrN[0]):

                # calculate the submatrices by going one level deeper into
                # the recursion. here we always trim away the first dimension
                # of the tensor, making it of rank (d-1)
                if verbose:
                    print("Going one level deeper: %d" % (nn))
                subC = self._refRecursion(arrN[1 :], tenC[nn])

                # place them at the correct positions with some modulo g0re
                for ii in range(arrN[0]):
                    NN = (arrNprod[1] * ((nn + ii) % (arrN[0]))) % arrNprod[0]
                    MM = (arrNprod[1] * ii)
                    if verbose:
                        print("nn=%d, ii=%d, NN=%d, MM=%d, \
                                NNto=%d, MMto=%d, CN=%d, CM=%d"
                              % (nn, ii, NN, MM, NN + arrNprod[1],
                                 MM + arrNprod[1], subC.shape[0], subC.shape[1])
                              )
                        print(arrC[NN:NN + arrNprod[1],
                                   MM:MM + arrNprod[1]].shape)
                        print((<object> arrC).shape)
                        print(arrN[0])

                    # do the actual placement by copying in the right memor
                    # region
                    arrC[NN:NN + arrNprod[1], MM:MM + arrNprod[1]] = subC
            return arrC
        else:
            # if we are in the lowest level, we just return the circulant
            # block by calling the normal circulant reference
            if verbose:
                print("Deepest level reached")

            return Circulant(tenC[:numRows])._reference()

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
                TEST.DATAALIGN  : TEST.ALIGNMENT.DONTCARE,
                'tenC'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeC',
                    TEST.SHAPE  : (3, 3, 3),
                    TEST.ALIGN  : TEST.PARAMALIGN
                }),
                TEST.INITARGS   : (lambda param : [param['tenC']()]),
                TEST.INITKWARGS : {'optimize' : 'optimize'},
                TEST.OBJECT     : MLCirculant,
                TEST.NAMINGARGS : dynFormat("%s,optimize=%s",
                                            'tenC', str('optimize')),
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
                                   MLCirculant(np.random.randn(
                                       *(2 * [c + 1])
                                   ))),
                BENCH.FUNC_SIZE : (lambda c: (c + 1) ** 2),
                BENCH.FUNC_STEP : (lambda c: c * 10 ** (1. / 12))
            },
            BENCH.FORWARD: {},
            BENCH.SOLVE: {},
            BENCH.OVERHEAD: {},
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, dt: MLCirculant(
                    np.random.randn(*(2 * [c + 1])).astype(dt)))
            }
        }
