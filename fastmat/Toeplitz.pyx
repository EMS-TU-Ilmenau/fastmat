# -*- coding: utf-8 -*-

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
from .Kron cimport Kron
from .Fourier cimport Fourier
from .Diag cimport Diag


cdef class Toeplitz(Partial):
    r"""
    This class provides a very general implementation of Toeplitz matrices,
    which essentially realize a (possibly multidimensional) non-circular
    convolution.

    This type of matrix is highly structured. A two-level Toeplitz
    Matrix looks like:

    >>> t_00 t_05 t_04 t_03   t_40 t_45 t_44 t_43   t_30 t_35 t_34 t_33
    >>> t_01 t_00 t_05 t_04   t_41 t_40 t_45 t_44   t_31 t_30 t_35 t_34
    >>> t_02 t_01 t_00 t_05   t_42 t_41 t_40 t_45   t_32 t_31 t_30 t_35
    >>>
    >>> t_10 t_15 t_14 t_13   t_00 t_05 t_04 t_03   t_40 t_45 t_44 t_43
    >>> t_11 t_10 t_15 t_14   t_01 t_00 t_05 t_04   t_41 t_40 t_45 t_44
    >>> t_12 t_11 t_10 t_15   t_02 t_01 t_00 t_05   t_42 t_41 t_40 t_45
    >>>
    >>> t_20 t_25 t_24 t_23   t_10 t_15 t_14 t_13   t_00 t_05 t_04 t_03
    >>> t_21 t_20 t_25 t_24   t_11 t_10 t_15 t_14   t_01 t_00 t_05 t_04
    >>> t_22 t_21 t_20 t_25   t_12 t_11 t_10 t_15   t_02 t_01 t_00 t_05

    This shows that one can define an L-level Toeplitz matrix by a tensor
    of order L together with means of deciding the sizes ``n_1,...,n_L`` of
    the individual levels.
    """

    property tenT:
        r"""Return the defining Tensor of Toeplitz matrix."""

        def __get__(self):
            return self._tenT

    property vecC:
        r"""Return the column-defining vector of Toeplitz matrix."""

        def __get__(self):
            import warnings
            warnings.warn('vecC is deprecated.', FutureWarning)
            return self._tenT[:self._arrDimRows[0]]

    property vecR:
        r"""Return the row-defining vector of Toeplitz matrix."""

        def __get__(self):
            import warnings
            warnings.warn('vecR is deprecated.', FutureWarning)
            return self._tenT[-self._arrDimCols[0] + 1:]

    def __init__(self, *args, **options):
        '''
        Initialize Toeplitz matrix instance.

        One either has to specify ``(vecC, vecR)`` or ``tenT`` with optinal
        ``split`` argument.

        Parameters
        ----------
        tenT : :py:class:`numpy.ndarray`
            This is the most general way to define a (multilevel) Toeplitz
            Matrix. The number of dimensions (length of .shape) determines
            the number of levels. If `split` is not defined then tenT needs
            to have odd size in each dimension, so that this results in a
            square matrix. The handling of the indexing in direction of columns
            follows the same reversed fashion as in the one-dimensional case
            with `vecR`, but here naturally for each level.

        split : :py:class:`numpy.ndarray`, optional
            This vector needs to have as many elements as the number of elements
            of `tenT.shape`. If it is specified it defines the number
            of elements which are used to determine the number of rows of
            each level. The rest of the elements are indexed in reverse order
            in the same fashion as without split.


        **options : optional
            Additional keyworded arguments. Supports all optional arguments
            supported by :py:class:`fastmat.Matrix` and
            :py:class:`fastmat.Fourier`.

            All optional arguments will be passed on to all
            :py:class:`fastmat.Matrix` instances that are generated during
            initialization.

        Note:
            For backward compatibility reasons it is still possible to
            substitute the `tenT` argument by two 1D :py:class:`numpy.ndarray`
            arrays `vecC` and `vecR` that describe the column- and row-defining
            vectors in the single-level case respectively. The column-defining
            vector describes the forst column of the resulting matrix and the
            row-defining vector the first row except the (0,0) element (which
            is already specified by the column-defining vector). Note that this
            vector is indexed backwards in the sense that its first element is
            the last element in the defined Toeplitz matrix.
        '''

        # The multidimensional implementation of this class exploits the fact
        # that one can embed a multilevel toeplitz matrix into a multilevel
        # circulant matrix so this initialization takes care of determining the
        # size of the resulting circulant matrix and the pattern how we embed
        # the toeplitz matrix into the circulant one moreover we exploit the
        # fact that some dimension allow zero padding in order to speed up the
        # fft calculations

        cdef np.ndarray vecC, vecR, arrDim, arrSplit
        cdef int maxStage
        cdef bint optimize

        # multiplex different parameter variants during initialization
        split = options.pop('split', None)
        arrSplit = np.array([] if split is None else split)

        # pop FFT parameters
        maxStage = options.get('maxStage', 4)    # max butterfly size per stage
        optimize = options.get('optimize', True) # enable bluestein FFT variant

        if len(args) == 1:
            # define the Matrix by a tensor defining its levels over axes
            self._tenT = args[0]
        elif len(args) == 2:
            if not all(isinstance(aa, np.ndarray) for aa in args):
                raise ValueError(
                    "You must specify two 1D-ndarrays containing the " +
                    "column- and row-definition vectors or one ndarray tensor"
                )

            if arrSplit.size != 0:
                raise ValueError(
                    "You must not define split points when supplying " +
                    "column- and row-definition vectors."
                )

            dataType = np.promote_types(args[0].dtype, args[1].dtype)
            vecC = _arrSqueeze(args[0].astype(dataType))
            vecR = _arrSqueeze(args[1].astype(dataType))
            if (vecC.ndim != 1) or (vecR.ndim != 1):
                raise ValueError(
                    "Column- and row-definition vectors must be 1D."
                )

            arrSplit = np.array(vecC.size)
            self._tenT = np.hstack((vecC, vecR))
        else:
            raise ValueError(
                "Invalid number of arguments to Toeplitz: Expecting exactly " +
                "one or two fixed arguments"
            )

        arrDim = np.array((<object> self._tenT).shape)

        # If no splitpoint vector was either given in options or generated from
        # column- and row-definition vectors, assume square levels such that
        # each dimension must obey the axis size relation (2 * n - 1)
        arrSplit = np.atleast_1d(arrSplit)
        if arrSplit.size == 0:
            if not all(((ll + 1) % 2 == 0)
                       for ll in arrDim):
                raise ValueError(
                    "Defining a tensor with non-square levels requires " +
                    "explicit split points."
                )

            arrSplit = (arrDim + 1) // 2

        if arrSplit.size != self._tenT.ndim:
            raise ValueError(
                "The split point vector must have one entry for each " +
                "dimension of the defining tensor"
            )
        elif arrSplit.ndim != 1:
            raise ValueError(
                "The split point vector must be 1D"
            )
        elif any(ll < 1 or ll > arrDim[ii]
                 for ii, ll in enumerate(arrSplit)):
            raise ValueError(
                "Entry in split vector outside of defining tensor bounds"
            )

        # determine row- and column- as well as definition vector size for
        # each level
        self._arrDimRows = arrSplit
        self._arrDimCols = arrDim - self._arrDimRows + 1

        cdef np.ndarray arrDimOpt, tenThat
        cdef intsize size, sizeOpt, ii, dd
        cdef Matrix FN
        cdef Product P
        cdef dict kwargs = options.copy()
        cdef dict Foptions = options.copy()
        Foptions['optimize'] = False            # don't optimize fouriers again

        # minimum number to pad to during optimization and helper arrays
        # these will describe the size of the resulting multilevel
        # circulant matrix where we embed everything in to
        arrDimOpt = np.copy(arrDim)
        if optimize:
            # go through all level dimensions and get optimal FFT size
            for ii, dd in enumerate(arrDim):
                # use optimal size, if we can get better in that level
                sizeOpt = _findOptimalFFTSize(dd, maxStage)
                arrDimOpt[ii] = (sizeOpt if (_getFFTComplexity(sizeOpt) <
                                             _getFFTComplexity(dd))
                                 else dd)

        # size is the size of the original tensor, sizeOpt with optimized sizes
        size = np.prod(arrDim)
        sizeOpt = np.prod(arrDimOpt)

        # allocate memory for the tensor in MD fourier domain
        tenThat = np.copy(self._tenT).astype('complex')

        # go through the array and apply the preprocessing in direction
        # of each axis. this cannot be done without the for loop, since
        # manipulations always influence the data for the next dimension
        # this preprocessing takes the defining elements and inserts zeros
        # where we specified the breaking point in the defining elements to
        # be
        for ii in range(self._tenT.ndim):
            tenThat = np.apply_along_axis(
                self._preProcSlice, ii, tenThat, ii, arrDimOpt, arrDim, arrSplit
            )

        if self._tenT.ndim == 1:
            # after correct zeropadding, go into fourier domain
            tenThat = np.fft.fft(tenThat, axis=0)

            # Describe as circulant matrix with product of data and vector
            # in fourier domain. Both fourier matrices cause scaling of the
            # data vector by N, which will be compensated in Diag().
            F = Fourier(sizeOpt, **Foptions)
        else:
            # get the size of the zero padded circulant matrix
            # it will be square but we still need to figure out, how the
            # defining element look like and how the embedding pattern looks
            # like

            # after correct zeropadding, go into fourier domain
            tenThat = np.fft.fftn(tenThat).reshape(sizeOpt)

            # create the decomposing kronecker product, which just is a
            # kronecker profuct of fourier matrices, since we now have a nice
            # and simple circulant matrix
            F = Kron(
                *list(map(lambda ii : Fourier(ii, **Foptions), arrDimOpt)),
                **options
            )

        # now decompose the toeplitz matrix as a product
        P = Product(F.H, Diag(tenThat / sizeOpt, **options), F, **options)

        # determine selection array of Partial to result in the correct Matrix
        # subselected from the whole multi-level Circulant matrix
        # begin with a basic indexing array of the inner level circulant's size
        cdef np.ndarray arrS = np.arange(np.prod(arrDimOpt))

        # initialize the result as all ones
        cdef np.ndarray arrSRows = arrS >= 0
        cdef np.ndarray arrSCols = arrS >= 0

        cdef np.ndarray modCols, modRows
        cdef intsize limRows, limCols, sizeLevelsBelow

        for ii in range(self._tenT.ndim):
            sizeLevelsBelow = np.prod(arrDimOpt[ii + 1:])
            modRows = np.mod(arrS, arrDimOpt[ii] * sizeLevelsBelow)
            modCols = np.mod(arrS, arrDimOpt[ii] * sizeLevelsBelow)
            limRows = self._arrDimRows[ii] * sizeLevelsBelow
            limCols = self._arrDimCols[ii] * sizeLevelsBelow
            # print("State Rows", arrSRows)
            # print("State Cols", arrSCols)
            # print("modulus Rows", modRows)
            # print("modulus Cols", modCols)
            # print("lim Rows", limRows)
            # print("lim Cols", limCols)
            # print("res Rows", modRows < limRows)
            # print("res Cols", modCols < limCols)

            # iteratively subselect more and more indices in arrSRows
            np.logical_and(arrSRows, modRows < limRows, arrSRows)
            # iteratively subselect more and more indices in arrSCols
            np.logical_and(arrSCols, modCols < limCols, arrSCols)

        kwargs.update({'rows': arrSRows, 'cols': arrSCols})

        # Finally, construct the multilevel Toeplitz matrix!
        super(Toeplitz, self).__init__(P, **kwargs)

        # Currently Fourier matrices bloat everything up to complex double
        # precision, therefore make sure vecC and vecR matches the
        # precision of the matrix itself
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
        np.ndarray arrDim,
        np.ndarray arrBP
    ):
        '''
        Preprocess one axis of the tensor by zero-padding the center of the
        axis according to bluestein's methos.

        Parameters
        ----------
        theSlice : :py:class:`numpy.ndarray`
            ?

        numSliceInd : int
            Axis to process.

        arrDimOpt : :py:class:`numpy.ndarray`
            Target shape of tensor after zero padding.

        arrDim : :py:class:`numpy.ndarray`
            Shape of the tensor befor zero padding.
        '''
        if arrDimOpt[numSliceInd] > arrDim[numSliceInd]:
            # if the optimal size is larger than the tensor in this dimension
            # we generate the needed amount of zeros and fiddle it into
            # the defining elements at the breaking points
            return np.concatenate((
                np.copy(theSlice[:arrBP[numSliceInd]]),
                np.zeros(arrDimOpt[numSliceInd] - arrDim[numSliceInd],
                         dtype=theSlice.dtype),
                np.copy(theSlice[arrBP[numSliceInd]:])
            ))
        else:
            # we are lucky and we cannot do anything
            return theSlice

    cpdef np.ndarray _getColNorms(self):
        return np.sqrt(self._normalizeColCore(
            self._tenT, self._arrDimRows, self._arrDimCols
        ))

    cpdef np.ndarray _getRowNorms(self):
        return np.sqrt(self._normalizeRowCore(
            self._tenT, self._arrDimRows,   self._arrDimCols
        ))

    cpdef np.ndarray _normalizeColCore(
        self,
        np.ndarray tenT,
        np.ndarray arrDimRows,
        np.ndarray arrDimCols
    ):
        cdef intsize ii
        cdef numSizeRows1, numSizeRows2, numSizeCols1, numSizeCols2

        # number of blocks in current level in direction of columns and rows
        cdef intsize numRows = arrDimRows[0]
        cdef intsize numCols = arrDimCols[0]

        # number of defining elements in the current level
        cdef intsize numEll = tenT.shape[0]

        # number of dimensions left in this current level
        cdef intsize numD = tenT.ndim

        # data structures for the defining elements and the array
        # of the norms in the current level
        cdef np.ndarray arrT, arrNorms
        if numD == 1:
            # if we are in the last level we do the normal toeplitz stuff
            arrT = tenT
            arrNorms = np.zeros(numCols)

            # the first element of the norms is the sum over the
            # absolute values squared, where we use the first numRows elements
            # in the (now) vector of defining elements
            arrNorms[0] = np.sum(np.abs(arrT[:numRows]) ** 2)

            # first we go over the leftmost part of the matrix, which in the
            # maximal case is the leftmost square part of the matrix
            # it is only square iff numCols >= numRows
            for ii in range(min(numRows - 1, numCols - 1)):
                addInd = numCols + numRows - 2 - ii
                subInd = numRows - ii - 1

                # here we subtract the element which has left the current
                # column and add the one, which enters it
                arrNorms[ii + 1] = (
                    arrNorms[ii] + np.abs(arrT[addInd]) ** 2 -
                    np.abs(arrT[subInd]) ** 2
                )

            # in case we have more columns than rows, we have to keep on
            # iterating. here we have to subtract different indices and add
            # different ones
            if numCols > numRows:
                for ii in range(numCols - numRows):
                    addInd = numCols - 1 - ii
                    subInd = ((numRows + numCols - 1 - ii) %
                              (numRows + numCols - 1))

                    # here we subtract the element which has left the current
                    # column and add the one, which enters it
                    arrNorms[numRows + ii] = (arrNorms[numRows + ii - 1] +
                                              np.abs(arrT[addInd]) ** 2 -
                                              np.abs(arrT[subInd]) ** 2)
        else:
            numSizeRows1 = np.prod(self._arrDimRows[-numD :])
            numSizeRows2 = np.prod(self._arrDimRows[-(numD - 1) :])
            numSizeCols1 = np.prod(self._arrDimCols[-numD :])
            numSizeCols2 = np.prod(self._arrDimCols[-(numD - 1) :])
            arrNorms = np.zeros(numSizeCols1)
            arrT = np.zeros((numEll, numSizeCols2))

            # go deeper in recursion and get column norms of blocks
            # this will result in a 2D ndarray, where the first index ii
            # corresponds to the block defined by tenT[ii,...]
            # and it contains the squared column norms of these possibly
            # multilevel toeplitz block.
            for ii in range(numEll):
                arrT[ii, :] = self._normalizeColCore(
                    tenT[ii], arrDimRows[1:], arrDimCols[1:],
                )

            # now again the first norm entry is the sum over the norms
            # of the first numRows norms of the blocks a level deeper
            arrNorms[:numSizeCols2] = np.sum(arrT[:numRows, :], axis=0)

            # first we go over the leftmost blocks of the matrix, which in the
            # maximal case is the leftmost part of the matrix
            # here, the matrices are not square anymore, since the subblocks
            # must not be square if numRows = numCols.
            for ii in range(min(numRows - 1, numCols - 1)):
                addInd = numCols + numRows - 2 - ii
                subInd = numRows - ii - 1

                # here we subtract the element which has left the current
                # column and add the one, which enters it
                arrNorms[
                    (ii + 1) * numSizeCols2:(ii + 2) * numSizeCols2
                ] = arrNorms[
                    ii * numSizeCols2:(ii + 1) * numSizeCols2
                ] + arrT[addInd] - arrT[subInd]

            # in case we have more blocks in column direction than in row
            # direction, we have to keep on
            # iterating. here we have to subtract different indices and add
            # different ones
            if numCols > numRows:
                for ii in range(numCols - numRows):
                    addInd = numCols - 1 - ii
                    subInd = ((numRows + numCols - 1 - ii) %
                              (numRows + numCols - 1))

                    # here we subtract the element which has left the current
                    # column and add the one, which enters it
                    # it basically is the same as in the single level case
                    arrNorms[
                        (numRows + ii) * numSizeCols2:
                        (numRows + ii + 1) * numSizeCols2
                    ] = arrNorms[
                        (numRows + ii - 1) * numSizeCols2:
                        (numRows + ii) * numSizeCols2
                    ] - arrT[subInd] + arrT[addInd]

        return arrNorms

    cpdef np.ndarray _normalizeRowCore(
        self,
        np.ndarray tenT,
        np.ndarray arrDimRows,
        np.ndarray arrDimCols
    ):
        cdef intsize ii
        cdef numSizeRows1, numSizeRows2, numSizeCols1, numSizeCols2

        # number of blocks in current level in direction of columns and rows
        cdef intsize numRows = arrDimRows[0]
        cdef intsize numCols = arrDimCols[0]

        # number of defining elements in the current level
        cdef intsize numEll = tenT.shape[0]

        # number of dimensions left in this current level
        cdef intsize numD = tenT.ndim

        # data structures for the defining elements and the array
        # of the norms in the current level
        cdef np.ndarray arrT, arrNorms

        if numD == 1:
            # if we are in the last level we do the normal toeplitz stuff
            arrT = tenT
            arrNorms = np.zeros(numRows)

            # the first element of the norms is the sum over the
            # absolute values squared, where we use the first numRows elements
            # in the (now) vector of defining elements
            arrNorms[0] = (
                np.sum(np.abs(arrT[numRows:]) ** 2) + np.abs(arrT[0]) ** 2
            )

            # first we go over the leftmost part of the matrix, which in the
            # maximal case is the leftmost square part of the matrix
            # it is only square iff numCols >= numRows
            for ii in range(min(numRows - 1, numCols - 1)):
                subInd = numRows + ii
                addInd = ii + 1
                # print("#%d - %d + %d" %(ii + 1, subInd, addInd))

                # here we subtract the element which has left the current
                # column and add the one, which enters it
                arrNorms[ii + 1] = (arrNorms[ii] +
                                    np.abs(arrT[addInd]) ** 2 -
                                    np.abs(arrT[subInd]) ** 2)

            # in case we have more columns than rows, we have to keep on
            # iterating. here we have to subtract different indices and add
            # different ones
            if numRows > numCols:
                for ii in range(numRows - numCols):
                    subInd = ii
                    addInd = numCols + ii

                    # here we subtract the element which has left the current
                    # column and add the one, which enters it
                    arrNorms[numCols + ii] = (arrNorms[numCols + ii - 1] +
                                              np.abs(arrT[addInd]) ** 2 -
                                              np.abs(arrT[subInd]) ** 2)
        else:
            numSizeRows1 = np.prod(self._arrDimRows[-numD :])
            numSizeRows2 = np.prod(self._arrDimRows[-(numD - 1) :])
            numSizeCols1 = np.prod(self._arrDimCols[-numD :])
            numSizeCols2 = np.prod(self._arrDimCols[-(numD - 1) :])
            arrNorms = np.zeros(numSizeRows1)
            arrT = np.zeros((numEll, numSizeRows2))

            # go deeper in recursion and get column norms of blocks
            # this will result in a 2D ndarray, where the first index ii
            # corresponds to the block defined by tenT[ii,...]
            # and it contains the squared column norms of these possibly
            # multilevel toeplitz block.
            for ii in range(numEll):
                arrT[ii, :] = self._normalizeRowCore(
                    tenT[ii], arrDimRows[1:], arrDimCols[1:],
                )

            # now again the first norm entry is the sum over the norms
            # of the first numRows norms of the blocks a level deeper
            arrNorms[:numSizeRows2] = (
                np.sum(arrT[numRows:, :], axis=0) + arrT[0, :]
            )

            # first we go over the leftmost blocks of the matrix, which in the
            # maximal case is the leftmost part of the matrix
            # here, the matrices are not square anymore, since the subblocks
            # must not be square if numRows = numCols.
            for ii in range(min(numRows - 1, numCols - 1)):

                addInd = ii + 1
                subInd = numRows + ii

                # here we subtract the element which has left the current
                # column and add the one, which enters it
                arrNorms[
                    (ii + 1) * numSizeRows2:(ii + 2) * numSizeRows2
                ] = arrNorms[
                    ii * numSizeRows2:(ii + 1) * numSizeRows2
                ] + arrT[addInd] - arrT[subInd]

            # in case we have more blocks in column direction than in row
            # direction, we have to keep on
            # iterating. here we have to subtract different indices and add
            # different ones
            if numRows > numCols:
                for ii in range(numRows - numCols):
                    addInd = numCols + ii
                    subInd = ii

                    # here we subtract the element which has left the current
                    # column and add the one, which enters it
                    # it basically is the same as in the single level case
                    arrNorms[
                        (numCols + ii) * numSizeRows2:
                        (numCols + ii + 1) * numSizeRows2
                    ] = arrNorms[
                        (numCols + ii - 1) * numSizeRows2:
                        (numCols + ii) * numSizeRows2
                    ] - arrT[subInd] + arrT[addInd]

        return arrNorms

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        return self._refRecursion(
            np.array((<object> self._tenT).shape),
            self._arrDimRows,
            self._arrDimCols,
            self._tenT,
            False
        )

    def _refRecursion(
        self,
        np.ndarray arrDim,
        np.ndarray arrDimRows,
        np.ndarray arrDimCols,
        np.ndarray tenT,
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

        tenT : :py:class:`numpy.ndarray`
            The defining elements.

        verbose : bool
            Output verbose information.

            Defaults to False.
        '''
        cdef intsize ii, nn_, mm, countAbs

        # number of dimensions
        cdef intsize numD = arrDim.shape[0]

        # get size of resulting block toeplitz matrix
        cdef intsize numRows = np.prod(arrDimRows)
        cdef intsize numCols = np.prod(arrDimCols)

        # get an array of all partial sequential products, starting at the front
        cdef np.ndarray arrDimRowProd = np.array(
            list(map(lambda ii : np.prod(arrDimRows[ii:]), range(numRows - 1)))
        )
        cdef np.ndarray arrDimColProd = np.array(
            list(map(lambda ii : np.prod(arrDimCols[ii:]), range(numCols - 1)))
        )

        # allocate memory for the result
        cdef np.ndarray T = np.zeros((numRows, numCols), dtype=self.dtype)
        cdef np.ndarray subT, vecR

        # check if we can go a least a level deeper
        if numD > 1:
            # iterate over size of the first dimension
            for nn_ in range(arrDimRows[0] + arrDimCols[0] - 1):
                # now calculate the block recursively
                subT = self._refRecursion(
                    arrDim[1:],
                    arrDimRows[1:],
                    arrDimCols[1:],
                    tenT[nn_]
                )
                # check if we are on or below the diagonal
                if nn_ < arrDimRows[0]:
                    # if yes, we have it easy
                    for mm in range(min(arrDimRows[0] - nn_, arrDimCols[0])):
                        mm_ = mm + nn_
                        T[
                            mm_ * arrDimRowProd[1]:(mm_ + 1) * arrDimRowProd[1],
                            mm * arrDimColProd[1]:(mm + 1) * arrDimColProd[1]
                        ] = subT
                else:
                    # if not as well!
                    for mm in range(
                        min(nn_ - arrDimRows[0] + 1, arrDimRows[0])
                    ):
                        rr = mm
                        cc = arrDimCols[0] - nn_ + arrDimRows[0] + mm - 1
                        T[
                            rr * arrDimRowProd[1]:(rr + 1) * arrDimRowProd[1],
                            cc * arrDimColProd[1]:(cc + 1) * arrDimColProd[1]
                        ] = subT

            return T
        else:
            # if we are in a lowest level, we just construct the right
            # single level toeplitz block. As _reference overloading from
            # Partial is too slow we rather construct a reference directly
            # from the defining tensor.

            # put columns in lower-triangular part of matrix
            for ii in range(0, min(numRows, numCols)):
                T[ii:numRows, ii] = tenT[:(numRows - ii)]

            # put rows in upper-triangular part of matrix
            vecR = tenT[numRows:][::-1]
            for ii in range(0, min(numRows, numCols - 1)):
                T[ii, (ii + 1):numCols] = vecR[:(numCols - ii - 1)]

            return T

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, NAME, dynFormat, mergeDicts
        test1D = {
            TEST.NUM_ROWS   : 4,
            TEST.NUM_COLS   : TEST.Permutation([3, 5, 41]),
            'mTypeH'        : TEST.Permutation(TEST.FEWTYPES),
            'mTypeV'        : TEST.Permutation(TEST.FEWTYPES),
            'vecH'          : TEST.ArrayGenerator({
                TEST.DTYPE  : 'mTypeH',
                TEST.SHAPE  : (TEST.NUM_ROWS, ),
                TEST.ALIGN  : TEST.PARAMALIGN
            }),
            'vecV'          : (lambda param: TEST.ArrayGenerator({
                TEST.DTYPE  : param['mTypeV'],
                TEST.SHAPE  : (param[TEST.NUM_COLS] - 1, ),
                TEST.ALIGN  : param[TEST.PARAMALIGN]
            })),
            TEST.INITARGS   : (lambda param : [param.vecH(), param.vecV()]),
            TEST.INITKWARGS : {'optimize': 'optimize'},
            TEST.NAMINGARGS : dynFormat(
                "%s,%s,optimize=%s", 'vecH', 'vecV', 'optimize'
            ),
        }
        testND = {
            'optimize': True,
            'shape'         : np.array([3, 3, 41]),
            'split'         : np.array([2, 1, 4]),
            'mTypeC'        : TEST.Permutation(TEST.FEWTYPES),
            'tenT'          : (lambda param: TEST.ArrayGenerator({
                TEST.DTYPE  : param['mTypeC'],
                TEST.SHAPE  : param['shape'],
                TEST.ALIGN  : param[TEST.PARAMALIGN]
            })),
            TEST.NUM_ROWS   : (lambda param: np.prod(param['split'])),
            TEST.NUM_COLS   : (
                lambda param: np.prod(param['shape'] - param['split'] + 1)
            ),
            TEST.INITARGS   : (lambda param : [param.tenT()]),
            TEST.INITKWARGS : {'optimize': 'optimize', 'split': 'split'},
            TEST.NAMINGARGS : dynFormat(
                "%s,optimize=%s,split=%s",
                'tenT', 'optimize', 'split'
            ),
        }
        return {
            # 41 is the first size for which bluestein is faster
            TEST.COMMON: {
                'optimize'      : TEST.Permutation([False, True]),
                TEST.PARAMALIGN : TEST.ALIGNMENT.DONTCARE,
                TEST.DATAALIGN  : TEST.ALIGNMENT.DONTCARE,
                TEST.OBJECT     : Toeplitz,
                TEST.TOL_POWER  : 2,
                TEST.TOL_MINEPS : getTypeEps(np.float64)
            },
            TEST.CLASS: mergeDicts(test1D, {
                # perform thorough testing of slicing during array construction
                # therefore, aside the symmetric shape case also test shapes
                # that differ by +1/-1 and +x/-x in row and col size
                TEST.NUM_COLS   : TEST.Permutation([2, 3, 4, 5, 6, 41]),
            }),
            TEST.TRANSFORMS: test1D,
            'classML': mergeDicts(testND, {NAME.TEMPLATE: TEST.CLASS}),
            'transformML': mergeDicts(testND, {NAME.TEMPLATE: TEST.TRANSFORMS})
        }

    def _getBenchmark(self):
        from .inspect import BENCH, arrTestDist
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c: Toeplitz(
                    arrTestDist((c, ), dtype=np.float32),
                    arrTestDist((c - 1, ), dtype=np.float32)))
            },
            BENCH.FORWARD: {},
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c: Toeplitz(
                    arrTestDist((2 ** c, ), dtype=np.float32),
                    arrTestDist((2 ** c - 1, ), dtype=np.float32)))
            },
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, datatype: Toeplitz(
                    arrTestDist((2 ** c, ), dtype=datatype),
                    arrTestDist((2 ** c - 1, ), dtype=datatype)))
            }
        }
