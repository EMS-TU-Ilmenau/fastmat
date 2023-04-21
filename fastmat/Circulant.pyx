# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False

# Copyright 2018 Sebastian Semper, Christoph Wagner
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


cdef class Circulant(Partial):
    r"""
    This class provides a very general implementation of circulant matrices,
    which essentially realize a (possibly multidimensional) circular
    convolution.

    This type of matrix is highly structured. A two-level circulant
    Matrix looks like:

    >>> c_00 c_02 c_01   c_20 c_22 c_21   c_10 c_12 c_11
    >>> c_01 c_00 c_02   c_21 c_20 c_22   c_11 c_10 c_12
    >>> c_02 c_01 c_00   c_22 c_21 c_20   c_12 c_11 c_10
    >>>
    >>> c_10 c_12 c_11   c_00 c_02 c_01   c_20 c_22 c_21
    >>> c_11 c_10 c_12   c_01 c_00 c_02   c_21 c_20 c_22
    >>> c_12 c_11 c_10   c_02 c_01 c_00   c_22 c_21 c_20
    >>>
    >>> c_20 c_22 c_21   c_10 c_12 c_11   c_00 c_02 c_01
    >>> c_21 c_20 c_22   c_11 c_10 c_12   c_01 c_00 c_02
    >>> c_22 c_21 c_20   c_12 c_11 c_10   c_02 c_01 c_00

    This shows that one can define an L-level Circulant matrix by a tensor
    of order L. By design circulant matrices are always square matrices.
    """

    property tenC:
        r"""Return the matrix-defining column vector of the circulant matrix"""

        def __get__(self):
            return self._tenC

    def __init__(self, tenC, **options):
        '''Initialize Multilevel Circulant matrix instance.

        Also see the special options of ``fastmat.Fourier``, which are
        also supported by this matrix and the general options offered by
        ``fastmat.Matrix.__init__``.

        Parameters
        ----------
        tenC : :py:class:`numpy.ndarray`
            The generating nd-array tensor defining the circulant matrix. The
            matrix data type is determined by the data type of this array.

        **options : optional
            Additional keyworded arguments. Supports all optional arguments
            supported by :py:class:`fastmat.Matrix` and
            :py:class:`fastmat.Fourier`.
        '''
        cdef intsize ii, nn, numRowsopt, size, minimalSize, paddedSize

        # collect matrix options
        cdef bint optimize = options.get('optimize', True)
        cdef int maxStage = options.get('maxStage', 4)

        # copy tensor in class itself and extract dimensions
        cdef np.ndarray _tenC = _arrSqueezedCopy(tenC)
        self._tenC = _tenC

        # extract the level dimensions from the defining tensor
        cdef np.ndarray arrDim = np.array((<object> self._tenC).shape)

        # distinguish between dimension of circulant
        cdef np.ndarray arrIndices, arrNopt, arrNpad, arrOptSize, arrDoOpt
        cdef np.ndarray tenChat
        cdef Diag D
        cdef Fourier F
        cdef Kron KN
        cdef Product P
        cdef bint truncate
        if (tenC.ndim < 1) or (self._tenC.ndim < 1):
            raise ValueError("Column-definition tensor must be at least 1D.")
        elif self._tenC.ndim == 1:
            # regular circulant matrix

            size = _tenC.size
            minimalSize = size * 2 - 1

            # determine if zero-padding of the convolution to achieve a better
            # FFT size is beneficial or not
            if optimize:
                paddedSize = _findOptimalFFTSize(minimalSize, maxStage)

                assert paddedSize >= size * 2 - 1

                if _getFFTComplexity(size) > _getFFTComplexity(paddedSize):
                    # zero-padding pays off, so do it!
                    tenC = np.concatenate([
                        tenC, np.zeros((paddedSize - minimalSize, ),
                                       dtype=tenC.dtype),
                        tenC[1:]
                    ])
                    size = paddedSize

            # Describe circulant matrix as product of data and vector in
            # fourier domain. Both fourier matrices cause scaling of the data
            # vector by size, which will be compensated in Diag().

            # Create inner product
            D = Diag(np.fft.fft(tenC, axis=0) / size, **options)
            F = Fourier(size, **options)
            P = Product(F.H, D, F, **options)

            # initialize Partial of Product. Only use Partial when padding size
            truncate = size != self._tenC.size
            arrIndices = np.arange(self._tenC.size)
        else:
            # multi-level circulant matrix

            # minimum numbers to pad to during FFT optimization
            arrNpad = 2 * arrDim - 1

            # array that will hold the calculated optimal FFT sizes
            arrOptSize = np.zeros_like(arrDim)

            # array that will hold 0 if we don't do an expansion of the
            # FFT size into this dimension and 1 if we do.
            # the dimensions are sorted like in arrDim
            arrDoOpt = np.zeros_like(arrDim)

            if optimize:
                # go through all level dimensions and get optimal FFT size
                for inn, nn in enumerate(arrNpad):
                    arrOptSize[inn] = _findOptimalFFTSize(nn, maxStage)

                    # use this size, if we get better in that level
                    if (_getFFTComplexity(arrOptSize[inn]) <
                            _getFFTComplexity(arrDim[inn])):
                        arrDoOpt[inn] = 1

            # convert the array to a boolean array
            arrDoOpt = arrDoOpt == 1
            arrNopt = np.copy(arrDim)

            # set the optimization size to the calculated one by replacing
            # the original sizes by the calculated better FFT sizes
            arrNopt[arrDoOpt] = arrOptSize[arrDoOpt]

            # get the size of the inflated matrix
            numRowsopt = np.prod(arrNopt)

            # allocate memory for the tensor in d-dimensional fourier domain
            # and save the memory
            tenChat = np.empty_like(self._tenC, dtype='complex')
            tenChat[:] = self._tenC[:]

            # go through the array and apply the preprocessing in direction
            # of each axis. where we apply the preprocessing for the bluestein
            # algorithm in every direction
            # this cannot be done without the for loop, since
            # manipulations always influence the data for the next dimension
            for ii in range(arrDim.size):
                tenChat = np.apply_along_axis(
                    self._preProcSlice,
                    ii,
                    tenChat,
                    ii,
                    arrNopt,
                    arrDim
                )

            # after correct zeropadding, go into fourier domain by calculating
            # the d-dimensional fourier transform on the preprocessed tensor
            tenChat = np.fft.fftn(tenChat).reshape(numRowsopt) / numRowsopt

            # subselection array to remember the parts of the inflated matrix,
            # where the original d-level circulant matrix has its entries
            arrIndices = np.arange(numRowsopt)[
                self._genArrS(arrDim, arrNopt)
            ]

            # create the decomposing kronecker product, which realizes
            # the d-dimensional FFT with measures offered by fastmat
            KN = Kron(*list(map(
                lambda ii : Fourier(ii, optimize=False), arrNopt
            )))

            # now describe the d-level matrix as a product as denoted in the
            # introduction of this function
            P = Product(KN.H, Diag(tenChat), KN, **options)

            # initialize Partial of Product. Only use Partial when
            # inflating the size of the matrix
            truncate = not np.allclose(arrDim, arrNopt)

        # instantiate the matrix
        cdef dict kwargs = options.copy()
        kwargs['rows'] = (arrIndices if truncate else None)
        kwargs['cols'] = (arrIndices if truncate else None)
        super(Circulant, self).__init__(P, **kwargs)

        # Currently Fourier matrices bloat everything up to complex double
        # precision, therefore make sure tenC matches the precision of the
        # matrix itself
        if self.dtype != self._tenC.dtype:
            self._tenC = self._tenC.astype(self.dtype)

    cpdef np.ndarray _getColNorms(self):
        return np.full((self._tenC.size, ), np.linalg.norm(self._tenC))

    cpdef np.ndarray _getRowNorms(self):
        return np.full((self._tenC.size, ), np.linalg.norm(self._tenC))

    cpdef Matrix _getColNormalized(self):
        return self * (1. / np.linalg.norm(self._tenC))

    cpdef Matrix _getRowNormalized(self):
        return self * (1. / np.linalg.norm(self._tenC))

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
        np.ndarray arrNout
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
        return self._refRecursion(
            np.array((<object> self._tenC).shape), self._tenC
        )

    def _refRecursion(
        self,
        np.ndarray arrN,
        np.ndarray tenC
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

        # The resulting d-level matrix
        cdef np.ndarray arrC, vecC

        if numD > 1:

            arrC = np.zeros((numRows, numRows), dtype=self.dtype)

            # iterate over dimensions
            for nn in range(arrN[0]):

                # calculate the submatrices by going one level deeper into
                # the recursion. here we always trim away the first dimension
                # of the tensor, making it of rank (d-1)
                subC = self._refRecursion(arrN[1 :], tenC[nn])

                # place them at the correct positions with some modulo g0re
                for ii in range(arrN[0]):
                    NN = (arrNprod[1] * ((nn + ii) % (arrN[0]))) % arrNprod[0]
                    MM = (arrNprod[1] * ii)

                    # do the actual placement by copying in the right memor
                    # region
                    arrC[NN:NN + arrNprod[1], MM:MM + arrNprod[1]] = subC
            return arrC
        else:
            # if we are in the lowest level, we just return the circulant
            # block by calling the normal circulant reference

            arrC = np.empty((numRows, numRows), dtype=self.dtype)
            vecC = tenC[:numRows]
            arrC[:, 0] = vecC
            for ii in range(numRows):
                arrC[:ii, ii] = vecC[numRows - ii:]
                arrC[ii:, ii] = vecC[:numRows - ii]

            return arrC

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                'dimshape'      : TEST.Permutation([
                    (7, ),      # 7 is just any number that causes no padding
                    (41, ),     # 41 is the first size bluestein is faster for
                    (3, 4),
                    (3, 4, 5)
                ]),
                TEST.NUM_ROWS   : (
                    lambda param: np.prod(param['dimshape'])
                ),
                TEST.NUM_COLS   : TEST.NUM_ROWS,
                'mTypeC'        : TEST.Permutation(TEST.FEWTYPES),
                'optimize'      : True,
                TEST.PARAMALIGN : TEST.ALIGNMENT.DONTCARE,
                TEST.DATAALIGN  : TEST.ALIGNMENT.DONTCARE,
                'tenC'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeC',
                    TEST.SHAPE  : 'dimshape',
                    TEST.ALIGN  : TEST.PARAMALIGN
                }),
                TEST.INITARGS   : (lambda param : [
                    param['tenC']()
                ]),
                TEST.INITKWARGS : {
                    'optimize'  : 'optimize'
                },
                TEST.OBJECT     : Circulant,
                TEST.NAMINGARGS : dynFormat(
                    "%s,optimize=%s",
                    'tenC', 'optimize'
                ),
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
                                   Circulant(np.random.randn(
                                       *(2 * [c + 1])
                                   ))),
                BENCH.FUNC_SIZE : (lambda c: (c + 1) ** 2),
                BENCH.FUNC_STEP : (lambda c: c * 10 ** (1. / 12))
            },
            BENCH.FORWARD: {},
            BENCH.OVERHEAD: {},
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, dt: Circulant(
                    np.random.randn(*(2 * [c + 1])).astype(dt)))
            }
        }
