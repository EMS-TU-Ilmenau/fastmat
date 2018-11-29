# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=True

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
from .Toeplitz cimport Toeplitz


################################################################################
################################################## class Toeplitz
cdef class MLToeplitz(Partial):
    r"""


    """

    property tenT:
        r"""Return the matrix-defining tensor of the circulant"""

        def __get__(self):
            return self._tenT

    ############################################## class methods
    def __init__(self, tenT, arrBreakingPoints=None, **options):
        '''
        Initialize Multilevel Toeplitz matrix instance.

        Parameters
        ----------
        tenT : :py:class:`numpy.ndarray`
            The generating nd-array defining the toeplitz tensor. The matrix
            data type is determined by the data type of this array.

        arrBreakingPoints : :py:class:`numpy.ndarray`


        **options:
            See the special options of :py:class:`fastmat.Fourier`, which are
            also supported by this matrix and the general options offered by
            :py:meth:`fastmat.Matrix.__init__`.
        '''

        # this class exploits the fact that one can embed a multilevel toeplitz
        # matrix into a multilevel circulant matrix
        # so this initialization takes care of determining the size of the
        # resulting circulant matrix and the pattern how we embed the toeplitz
        # matrix into the circulant one
        # moreover we exploit the fact that some dimension allow zero padding
        # in order to speed up the fft calculations


        self._tenT = _arrSqueezedCopy(tenT)
        if self._tenT.ndim < 1:
            raise ValueError("Column-definition tensor must be at least 1D")

        singleDim = (self._tenT.ndim == 1)

        # extract the level dimensions from the defining tensor
        # and the breaking points
        self._arrDimT = np.array(self._tenT[:].shape)
        self._arrDimRows = arrBreakingPoints
        self._arrDimCols = self._arrDimT - self._arrDimRows + 1

        # array of the shape of the defining elements
        cdef np.ndarray arrTenTSize = np.array(
            self._tenT[:].shape
        ).astype('int')

        # stages during optimization for calculating the optimal fft size
        cdef int maxStage = options.get('maxStage', 4)

        # minimum number to pad to during optimization and helper arrays
        # these will describe the size of the resulting multilevel
        # circulant matrix where we embed everything in to
        cdef np.ndarray arrDimPad = np.copy(arrTenTSize)
        cdef np.ndarray arrOptSize = np.zeros_like(arrTenTSize)
        cdef np.ndarray arrDoOpt = np.zeros_like(arrTenTSize)

        cdef bint optimize = options.get('optimize', True)

        cdef intsize idd, dd
        if optimize:
            # go through all level dimensions and get optimal FFT size
            for idd, dd in enumerate(arrDimPad):
                arrOptSize[idd] = _findOptimalFFTSize(dd, maxStage)

                # use this size, if we get better in that level
                if _getFFTComplexity(arrOptSize[idd]) < _getFFTComplexity(dd):
                    arrDoOpt[idd] = 1

        # register in which level we will ultimately do zero padding
        arrDoOpt = arrDoOpt == 1
        cdef np.ndarray arrDimOpt = np.copy(arrTenTSize)

        # set the optimization size to the calculated one, but only if we
        # decided for that level to do an optimization
        arrDimOpt[arrDoOpt] = arrOptSize[arrDoOpt]

        # get the size of the zero padded circulant matrix
        # it will be square but we still need to figure out, how the
        # defining element look like and how the embedding pattern looks like
        cdef intsize numRowsOpt = np.prod(arrDimOpt)
        cdef intsize numColsOpt = np.prod(arrDimOpt)

        # allocate memory for the tensor in MD fourier domain
        cdef np.ndarray tenThat = np.copy(self._tenT).astype('complex')

        # go through the array and apply the preprocessing in direction
        # of each axis. this cannot be done without the for loop, since
        # manipulations always influence the data for the next dimension
        # this preprocessing takes the defining elements and inserts zeros
        # where we specified the breaking point in the defining elements to
        # be
        for ii in range(self._tenT.ndim):
            tenThat = np.apply_along_axis(
                self._preProcSlice,
                ii,
                tenThat,
                ii,
                arrDimOpt,
                self._arrDimT,
                arrBreakingPoints
            )

        # after correct zeropadding, go into fourier domain
        tenThat = np.fft.fftn(tenThat).reshape(numRowsOpt) / numRowsOpt

        # subselection arrays, which are different for the rows and columns
        tplIndices = self._genArrS(
            self._arrDimRows,
            self._arrDimCols,
            arrBreakingPoints,
            arrDimOpt,
            self._arrDimT,
            False
        )

        # create the decomposing kronecker product, which just is a
        # kronecker profuct of fourier matrices, since we now have a nice
        # and simple circulant matrix

        if singleDim:
            KN = Fourier(arrDimOpt[0], optimize=False)
        else:
            KN = Kron(*list(map(
                lambda ii : Fourier(ii, optimize=False), arrDimOpt
            )), **options)

        # now decompose the circulant matrix as a product
        cdef Product P = Product(KN.H, Diag(tenThat, **options), KN, **options)

        # initialize Partial of Product to only extract the embedded toeplitz
        # matrix
        cdef dict kwargs = options.copy()

        kwargs['rows'] = np.arange(numRowsOpt)[tplIndices[0]]
        kwargs['cols'] = np.arange(numColsOpt)[tplIndices[1]]

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
        np.ndarray arrDimOpt,
        np.ndarray arrTenTSize,
        np.ndarray arrBP
    ):
        '''
        Preprocess one axis of the defining tensor.

        Here we check for one dimension, whether we decided to zero pad
        by comparing arrTenTSize and arrDimOpt.

        Parameters
        ----------
        theSlice : :py:class:`numpy.ndarray`
            ?

        numSliceInd : int
            ?

        arrDimOpt : :py:class:`numpy.ndarray`
            ?

        arrTenTSize : :py:class:`numpy.ndarray`
            the size of the
        '''
        cdef np.ndarray z, arrRes = np.empty(1)

        if arrDimOpt[numSliceInd] > arrTenTSize[numSliceInd]:
            # if the optimal size is larger than the tensor in this dimension
            # we generate the needed amount of zeros and fiddle it into
            # the defining elements at the breaking points
            z = np.zeros(arrDimOpt[numSliceInd] - arrTenTSize[numSliceInd])
            arrRes = np.concatenate((
                np.copy(theSlice[:arrBP[numSliceInd]]),
                z,
                np.copy(theSlice[arrBP[numSliceInd]:])
            ))
        else:
            # we are lucky and we cannot do anything
            arrRes = np.copy(theSlice)

        return arrRes

    cpdef tuple _genArrS(
        self,
        np.ndarray arrDimRows,
        np.ndarray arrDimCols,
        np.ndarray arrBP,
        np.ndarray arrDimOut,
        np.ndarray arrDimT,
        bint verbose=False
    ):
        '''
        Generate the subselection indices for the circulant matrix

        We iteratively reduce the number of TRUE values in the resulting in
        a tuple of two ndarrays, which contain the row and columns
        subselection indices to get from the circulant to the toeplitz
        matrix.

        Parameters
        ----------
        arrDim : :py:class:`numpy.ndarray`
            The original sizes of the defining tensor.

        arrBP : :py:class:`numpy.ndarray`
            Array containing the breaking points of each level

        arrDimOut : :py:class:`numpy.ndarray`
            The dimensions we zero pad to. this is the same for the
            rows and columns of the resulting circulant matrix

        verbose : bool
            Output verbose information. run it in verbosity mode to understand
            this function

            Defaults to False.

        Returns
        -------
        tplInd : tuple
            The output array which does the selection.
        '''
        cdef intsize ii, n = arrDimOut.shape[0]

        # output size of the matrix we embed into
        cdef intsize numRowsOut = np.prod(arrDimOut)
        cdef intsize numColsOut = np.prod(arrDimOut)

        # initialize the result as all ones
        cdef np.ndarray arrSRows = np.arange(numRowsOut) >= 0
        cdef np.ndarray arrSCols = np.arange(numColsOut) >= 0

        for ii in range(n):
            if verbose:
                print("State Rows", arrSRows)
                print("State Cols", arrSCols)
                print("modulus Rows", np.mod(
                    np.arange(numRowsOut),
                    np.prod(arrDimOut[ii:])
                ))
                print("modulus Cols", np.mod(
                    np.arange(numColsOut),
                    np.prod(arrDimOut[ii:])
                ))
                print("inequ Rows",
                      arrBP[ii] * np.prod(arrDimOut[ii + 1:]))
                print("inequ Cols",
                      (arrDimOut[ii] - arrBP[ii] + 1) * np.prod(arrDimOut[ii + 1:]))
                print("res Rows", np.mod(
                    np.arange(numRowsOut),
                    np.prod(arrDimOut[ii:])
                ) < arrBP[ii] * np.prod(arrDimOut[ii + 1:]))
                print("res Cols", np.mod(
                    np.arange(numColsOut),
                    np.prod(arrDimOut[ii:])
                ) < (arrDimOut[ii] - arrBP[ii] + 1) * np.prod(arrDimOut[ii + 1:]))
            # iteratively subselect more and more indices in arrSRows
            np.logical_and(
                arrSRows,
                np.mod(
                    np.arange(numRowsOut),
                    np.prod(arrDimOut[ii:])
                ) < arrBP[ii] * np.prod(arrDimOut[ii + 1:]),
                arrSRows
            )
            # iteratively subselect more and more indices in arrSCols
            np.logical_and(
                arrSCols,
                np.mod(
                    np.arange(numColsOut),
                    np.prod(arrDimOut[ii:])
                ) < (arrDimOut[ii] - arrBP[ii] + 1) * np.prod(arrDimOut[ii + 1:]),
                arrSCols
            )
        return (arrSRows, arrSCols)

    cpdef np.ndarray _getColNorms(self):
        return np.sqrt(self._normalizeColCore(self._tenT))

    cpdef np.ndarray _normalizeColCore(self, np.ndarray tenT):
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

    # TODO: Implement _getRowNorms

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
        Build the d-level toeplitz matrix recursively from a d-dimensional
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
        cdef intsize numRows = np.prod(arrDim)

        # get an array of all partial sequential products
        # starting at the front
        cdef np.ndarray arrDimprod = np.array(
            list(map(lambda ii : np.prod(arrDim[ii:]), range(numRows - 1)))
        )

        # permutation array for block placement, since we need to place the
        # blocks in the same fashion, we arrange the elements in the blocks,
        # such that the preprocessing does the right thing
        cdef np.ndarray arrP = np.arange(2 * arrDim[0] - 1)
        arrP = np.concatenate((arrP[:arrDim[0]][::-1], arrP[arrDim[0]:][::-1]))
        arrP = np.argsort(arrP)

        # allocate memory for the result
        cdef np.ndarray T = np.zeros((numRows, numRows), dtype=self.dtype)
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
            return Toeplitz(tenU[:numRows], tenU[numRows:][::-1]).array

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
            BENCH.OVERHEAD: {},
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, dt: MLToeplitz(
                    np.random.randn(*(2 * [(2 * c - 1) + 2])).astype(dt)))
            }
        }
