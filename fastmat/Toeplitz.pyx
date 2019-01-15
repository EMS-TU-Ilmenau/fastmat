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


################################################################################
################################################## class Toeplitz
cdef class Toeplitz(Partial):
    r"""

    A Toeplitz matrix :math:`T \in \mathbb{C}^{n \times m}` realizes the mapping

    .. math::
        x \mapsto  T \cdot  x,

    where :math:`x \in C^n` and

    .. math::
        T = \begin{bmatrix}
        t_1 & t_{-1} & \dots & t_{-(m-1)} \\
        t_2 & t_1 & \ddots & t_{-(n-2)} \\
        \vdots & \vdots & \ddots & \vdots \\
        t_n & t_{n-1} & \dots & t_1
        \end{bmatrix}.

    This means that a Toeplitz matrix is uniquely defined by the
    :math:`n + m - 1` values that are on the diagonals.

    >>> # import the package
    >>> import fastmat as fm
    >>> import numpy as np
    >>>
    >>> # define the parameters
    >>> d1 = np.array([1,0,3,6])
    >>> d2 = np.array([5,7,9])
    >>>
    >>> # construct the transform
    >>> T = fm.Toeplitz(d1,d2)

    This yields

    .. math::
        d_1 = (1,0,3,6)^\mathrm{T}

    .. math::
        d_2 = (5,7,9)^\mathrm{T}

    .. math::
        T = \begin{bmatrix}
        1 & 5 & 7 & 9 \\
        0 & 1 & 5 & 7 \\
        3 & 0 & 1 & 5 \\
        6 & 3 & 0 & 1
        \end{bmatrix}

    Since the multiplication with a Toeplitz matrix makes use of the FFT, it
    can be very slow, if the sum of the dimensions of :math:`d_1` and
    :math:`d_2` are far away from a power of :math:`2`, :math:`3` or
    :math:`4`. This can be alleviated if one applies smart zeropadding during
    the transformation.
    This can be activated as follows.

    >>> # import the package
    >>> import fastmat as fm
    >>> import numpy as np
    >>>
    >>> # define the parameters
    >>> d1 = np.array([1,0,3,6])
    >>> d2 = np.array([5,7,9])
    >>>
    >>> # construct the transform
    >>> T = fm.Toeplitz(d1,d2,pad='true')

    This yields the same matrix and transformation as above, but it might be
    faster depending on the dimensions involved in the problem.

    This class depends on ``Fourier``, ``Diag``, ``Product`` and
    ``Partial``.
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
            return self._tenT[:self._arrDimCols[0]]

    property vecR:
        r"""Return the row-defining vector of Toeplitz matrix."""

        def __get__(self):
            import warnings
            warnings.warn('vecR is deprecated.', FutureWarning)
            return self._tenT[self._arrDimCols[0]:]

    def __init__(self, *args, **options):
        '''
        Initialize Toeplitz matrix instance.

        Parameter for one-dimensional case
        ----------------------------------
        vecC : :py:class:`numpy.ndarray`
            The generating column vector of the toeplitz matrix describing the
            first column of the matrix.

        vecR : :py:class:`numpy.ndarray`
            The generating row vector of the toeplitz matrix excluding the
            element corresponding to the first column, which is already defined
            in `vecC`.

        **options:
            See below.

        Parameter for one-or-multi-dimensional case
        -------------------------------------------
        tenT : :py:class:`numpy.ndarray`
            The generating nd-array defining the toeplitz tensor. The matrix
            data type is determined by the data type of this array. In this
            parameter variant the column- and row-defining vectors are given
            in one single vector. The intersection point between these two
            vectors is given in the `split` option.

        **options:
            See below.

        Options
        -------
        split : :py:class:`numpy.ndarray`
            A 1d vector specifying the split-point for row/column definition
            of each vector. If this option is not specified each level
            :math:`T \in \mathbb{C}^{d_i \times d_i}` with the corresponding
            :math:`i` of `tenT` is assumed to have a square shape of size
            dimension of `tenT` having :math:`d_i * 2 - 1` entries.

            Defaults to a splitpoint vetor corresponding to all-square levels.


        Also see the special options of :py:class:`fastmat.Fourier`, which are
        also supported by this matrix and the general options offered by
        :py:meth:`fastmat.Matrix.__init__`.
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
        elif any(ll < 1 or ll >= arrDim[ii]
                 for ii, ll in enumerate(arrSplit)):
            raise ValueError(
                "Entry in split vector out of defining tensor bounds"
            )

        # determine row- and column- as well as definition vector size for
        # each level
        self._arrDimRows = arrSplit
        self._arrDimCols = arrDim - self._arrDimRows + 1

        cdef np.ndarray arrDimTPad, arrOptSize, arrDoOpt, arrDimOpt, tenThat
        cdef intsize size, vecSize, idd, dd, numRowsOpt, numColsOpt
        cdef dict kwargs = options.copy()
        cdef Fourier FN
        cdef Product P
        if self._tenT.ndim == 1:
            # save generating vectors. Matrix sizes will be set by Product
            dataType = self._tenT.dtype

            # perform padding (if enabled) and generate vector
            size = self._tenT.size
            vecSize = size

            # determine if zero-padding of the convolution to achieve a better
            # FFT size is beneficial or not
            if optimize:
                vecSize = _findOptimalFFTSize(size, maxStage)

                assert vecSize >= size

                if _getFFTComplexity(size) <= _getFFTComplexity(vecSize):
                    vecSize = size

            if vecSize > size:
                # zero-padding pays off, so do it!
                vec = np.concatenate([
                    self._tenT[:self._arrDimRows[0]],
                    np.zeros((vecSize - size,), dtype=dataType),
                    self._tenT[self._arrDimRows[0]:]
                ])
            else:
                vec = self._tenT

            # Describe as circulant matrix with product of data and vector
            # in fourier domain. Both fourier matrices cause scaling of the
            # data vector by N, which will be compensated in Diag().

            # Create inner product
            FN = Fourier(vecSize, **options)
            P = Product(
                FN.H,
                Diag(np.fft.fft(vec, axis=0) / vecSize, **options),
                FN,
                **options
            )

            # initialize Partial of Product
            kwargs['rows'] = (np.arange(self._arrDimRows[0])
                              if size != self._arrDimRows[0] else None)
            kwargs['cols'] = (np.arange(self._arrDimCols[0])
                              if size != self._arrDimCols[0] else None)

            super(Toeplitz, self).__init__(P, **kwargs)

            # Currently Fourier matrices bloat everything up to complex double
            # precision, therefore make sure vecC and vecR matches the
            # precision of the matrix itself
            if self.dtype != self._tenT.dtype:
                self._tenT = self._tenT.astype(self.dtype)
        else:
            # this class exploits the fact that one can embed a multilevel
            # toeplitz matrix into a multilevel circulant matrix so this
            # initialization takes care of determining the size of the resulting
            # circulant matrix and the pattern how we embed the toeplitz matrix
            # into the circulant one moreover we exploit the fact that some
            # dimension allow zero padding in order to speed up the fft
            # calculations

            # extract the level dimensions from the defining tensor
            # and the breaking points
            arrDim = np.array((<object> self._tenT).shape)
            self._arrDimRows = arrSplit
            self._arrDimCols = arrDim - self._arrDimRows + 1

            # minimum number to pad to during optimization and helper arrays
            # these will describe the size of the resulting multilevel
            # circulant matrix where we embed everything in to
            arrDimTPad = np.copy(arrDim)
            arrOptSize = np.zeros_like(arrDim)
            arrDoOpt = np.zeros_like(arrDim)

            if optimize:
                # go through all level dimensions and get optimal FFT size
                for idd, dd in enumerate(arrDimTPad):
                    arrOptSize[idd] = _findOptimalFFTSize(dd, maxStage)

                    # use this size, if we get better in that level
                    arrDoOpt[idd] = (_getFFTComplexity(arrOptSize[idd]) <
                                     _getFFTComplexity(dd))

            # register in which level we will ultimately do zero padding
            arrDimOpt = np.copy(arrDim)

            # set the optimization size to the calculated one, but only if we
            # decided for that level to do an optimization
            arrDimOpt[arrDoOpt] = arrOptSize[arrDoOpt]

            # get the size of the zero padded circulant matrix
            # it will be square but we still need to figure out, how the
            # defining element look like and how the embedding pattern looks
            # like
            numRowsOpt = np.prod(arrDimOpt)
            numColsOpt = np.prod(arrDimOpt)

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
                    self._preProcSlice,
                    ii,
                    tenThat,
                    ii,
                    arrDimOpt,
                    arrDim,
                    arrSplit
                )

            # after correct zeropadding, go into fourier domain
            tenThat = np.fft.fftn(tenThat).reshape(numRowsOpt) / numRowsOpt

            # subselection arrays, which are different for the rows and columns
            tplIndices = self._genArrS(
                self._arrDimRows,
                self._arrDimCols,
                arrSplit,
                arrDimOpt,
                arrDim,
                False
            )

            # create the decomposing kronecker product, which just is a
            # kronecker profuct of fourier matrices, since we now have a nice
            # and simple circulant matrix

            KN = Kron(*list(map(
                lambda ii : Fourier(ii, optimize=False), arrDimOpt
            )), **options)

            # now decompose the circulant matrix as a product
            P = Product(KN.H, Diag(tenThat, **options), KN, **options)

            kwargs['rows'] = np.arange(numRowsOpt)[tplIndices[0]]
            kwargs['cols'] = np.arange(numColsOpt)[tplIndices[1]]

            super(Toeplitz, self).__init__(P, **kwargs)

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
        np.ndarray arrDim,
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
                print("inequ Cols", (
                    arrDimOut[ii] - arrBP[ii] + 1
                ) * np.prod(arrDimOut[ii + 1:]))
                print("res Rows", np.mod(
                    np.arange(numRowsOut),
                    np.prod(arrDimOut[ii:])
                ) < arrBP[ii] * np.prod(arrDimOut[ii + 1:]))
                print("res Cols", np.mod(
                    np.arange(numColsOut),
                    np.prod(arrDimOut[ii:])
                ) < (
                    arrDimOut[ii] - arrBP[ii] + 1
                ) * np.prod(arrDimOut[ii + 1:]))

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
                ) < (
                    arrDimOut[ii] - arrBP[ii] + 1
                ) * np.prod(arrDimOut[ii + 1:]),
                arrSCols
            )
        return (arrSRows, arrSCols)

    cpdef np.ndarray _getColNorms(self):
        return np.sqrt(
            self._normalizeColCore(
                self._tenT,
                self._arrDimRows,
                self._arrDimCols
            )
        )

    cpdef np.ndarray _getRowNorms(self):
        return np.sqrt(
            self._normalizeRowCore(
                self._tenT,
                self._arrDimRows,
                self._arrDimCols
            )
        )

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
            for ii in range(
                min(numRows - 1, numCols - 1)
            ):
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
                    subInd = (
                        numRows + numCols - 1 - ii
                    ) % (
                        numRows + numCols - 1
                    )

                    # here we subtract the element which has left the current
                    # column and add the one, which enters it
                    arrNorms[numRows + ii] = (
                        arrNorms[numRows + ii - 1] +
                        np.abs(arrT[addInd]) ** 2 -
                        np.abs(arrT[subInd]) ** 2
                    )
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
                    tenT[ii],
                    arrDimRows[1:],
                    arrDimCols[1:],
                )

            # now again the first norm entry is the sum over the norms
            # of the first numRows norms of the blocks a level deeper
            arrNorms[:numSizeCols2] = np.sum(arrT[:numRows, :], axis=0)

            # first we go over the leftmost blocks of the matrix, which in the
            # maximal case is the leftmost part of the matrix
            # here, the matrices are not square anymore, since the subblocks
            # must not be square if numRows = numCols.
            for ii in range(
                min(numRows - 1, numCols - 1)
            ):

                addInd = numCols + numRows - 2 - ii
                subInd = numRows - ii - 1

                # here we subtract the element which has left the current
                # column and add the one, which enters it
                #
                arrNorms[
                    (ii + 1) * numSizeCols2 : (ii + 2) * numSizeCols2
                ] = arrNorms[
                    ii * numSizeCols2 : (ii + 1) * numSizeCols2
                ] + arrT[addInd] - arrT[subInd]

            # in case we have more blocks in column direction than in row
            # direction, we have to keep on
            # iterating. here we have to subtract different indices and add
            # different ones
            if numCols > numRows:

                for ii in range(numCols - numRows):

                    addInd = numCols - 1 - ii
                    subInd = (
                        numRows + numCols - 1 - ii
                    ) % (numRows + numCols - 1)

                    # here we subtract the element which has left the current
                    # column and add the one, which enters it
                    # it basically is the same as in the single level case
                    arrNorms[
                        (
                            numRows + ii
                        ) * numSizeCols2 : (
                            numRows + ii + 1
                        ) * numSizeCols2
                    ] = arrNorms[
                        (numRows + ii - 1) * numSizeCols2 :
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
            for ii in range(
                min(numRows - 1, numCols - 1)
            ):

                subInd = numRows + ii
                addInd = ii + 1

                # print("#", ii + 1)
                # print("-", subInd)
                # print("+", addInd)

                # here we subtract the element which has left the current
                # column and add the one, which enters it
                arrNorms[ii + 1] = (
                    arrNorms[ii] +
                    np.abs(arrT[addInd]) ** 2 -
                    np.abs(arrT[subInd]) ** 2
                )

            # in case we have more columns than rows, we have to keep on
            # iterating. here we have to subtract different indices and add
            # different ones
            if numRows > numCols:

                for ii in range(numRows - numCols):

                    subInd = ii
                    addInd = numCols + ii

                    # here we subtract the element which has left the current
                    # column and add the one, which enters it
                    arrNorms[numCols + ii] = (
                        arrNorms[numCols + ii - 1] +
                        np.abs(arrT[addInd]) ** 2 -
                        np.abs(arrT[subInd]) ** 2
                    )
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
                    tenT[ii],
                    arrDimRows[1:],
                    arrDimCols[1:],
                )

            # now again the first norm entry is the sum over the norms
            # of the first numRows norms of the blocks a level deeper
            arrNorms[:numSizeRows2] = (
                np.sum(arrT[numRows:, :], axis=0) +
                arrT[0, :]
            )

            # first we go over the leftmost blocks of the matrix, which in the
            # maximal case is the leftmost part of the matrix
            # here, the matrices are not square anymore, since the subblocks
            # must not be square if numRows = numCols.
            for ii in range(
                min(numRows - 1, numCols - 1)
            ):

                addInd = ii + 1
                subInd = numRows + ii

                # here we subtract the element which has left the current
                # column and add the one, which enters it
                arrNorms[
                    (ii + 1) * numSizeRows2 : (ii + 2) * numSizeRows2
                ] = arrNorms[
                    ii * numSizeRows2 : (ii + 1) * numSizeRows2
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
                        (
                            numCols + ii
                        ) * numSizeRows2 : (
                            numCols + ii + 1
                        ) * numSizeRows2
                    ] = (
                        arrNorms[
                            (numCols + ii - 1) * numSizeRows2
                            :(numCols + ii) * numSizeRows2
                        ] - arrT[subInd] + arrT[addInd]
                    )

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

        # get an array of all partial sequential products
        # starting at the front
        cdef np.ndarray arrDimRowProd = np.array(
            list(map(lambda ii : np.prod(
                arrDimRows[ii:]
            ), range(numRows - 1)))
        )
        cdef np.ndarray arrDimColProd = np.array(
            list(map(lambda ii : np.prod(
                arrDimCols[ii:]
            ), range(numCols - 1)))
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

                # print(subT[:].shape)

                # check if we are on or below the diagonal
                if nn_ < arrDimRows[0]:
                    # pass
                    # print("bef")
                    # if yes, we have it easy
                    for mm in range(
                        min(arrDimRows[0] - nn_, arrDimCols[0])
                    ):
                        mm_ = mm + nn_
                        # print(nn_, mm, mm_)
                        # print(T[
                        #     mm_ * arrDimRowProd[1] :
                        #         (mm_ + 1) * arrDimRowProd[1],
                        #     mm * arrDimColProd[1] :
                        #         (mm + 1) * arrDimColProd[1]
                        # ].shape)
                        T[
                            mm_ * arrDimRowProd[1] :
                                (mm_ + 1) * arrDimRowProd[1],
                            mm * arrDimColProd[1] :
                                (mm + 1) * arrDimColProd[1]
                        ] = subT
                else:
                    # if not as well!
                    # print("next")
                    for mm in range(
                        min(nn_ - arrDimRows[0] + 1, arrDimRows[0])
                    ):
                        rr = mm
                        cc = arrDimCols[0] - nn_ + arrDimRows[0] + mm - 1
                        # print("Row", rr)
                        # print("Col", cc)
                        # print("Ind", nn_)
                        # print(T[
                        #     rr * arrDimRowProd[1] :
                        #         (rr + 1) * arrDimRowProd[1],
                        #     cc * arrDimColProd[1] :
                        #         (cc + 1) * arrDimColProd[1]
                        # ].shape)
                        T[
                            rr * arrDimRowProd[1] :
                                (rr + 1) * arrDimRowProd[1],
                            cc * arrDimColProd[1] :
                                (cc + 1) * arrDimColProd[1]
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
        from .inspect import TEST, NAME, dynFormat
        return {
            TEST.COMMON: {
                TEST.DATAALIGN  : TEST.ALIGNMENT.DONTCARE,
                # 35 is just any number that causes no padding
                # 41 is the first size for which bluestein is faster
                TEST.NUM_ROWS   : TEST.Permutation([5, 41]),
                TEST.NUM_COLS   : TEST.Permutation([4, 6]),
                'optimize'      : TEST.Permutation([False, True]),
                TEST.PARAMALIGN : TEST.ALIGNMENT.DONTCARE,
                TEST.DATAALIGN  : TEST.ALIGNMENT.DONTCARE,
                TEST.INITKWARGS : {
                    'optimize'      : 'optimize'
                },
                TEST.OBJECT     : Toeplitz,
                TEST.TOL_POWER  : 2,
                TEST.TOL_MINEPS : getTypeEps(np.float64)
            },
            TEST.CLASS: {
                # perform thorough testing of slicing during array construction
                # therefore, aside the symmetric shape case also test shapes
                # that differ by +1/-1 and +x/-x in row and col size
                TEST.NUM_ROWS   : 4,
                TEST.NUM_COLS   : TEST.Permutation([2, 3, 4, 5, 6]),
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
                TEST.INITARGS   : (lambda param : [
                    param.vecH(),
                    param.vecV()
                ]),
                TEST.NAMINGARGS : dynFormat(
                    "%s,%s,optimize=%s", 'vecH', 'vecV', 'optimize'
                ),
            },
            TEST.TRANSFORMS: {
                # during class tests we do not need to verify bluestein again
                TEST.NUM_ROWS   : TEST.Permutation([7]),
                'vecVwidth'     : (lambda param: param[TEST.NUM_COLS] - 1),
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
                TEST.INITARGS   : (lambda param : [
                    param.vecH(),
                    param.vecV()
                ]),
                TEST.NAMINGARGS : dynFormat(
                    "%s,%s,optimize=%s", 'vecH', 'vecV', 'optimize'
                ),
            },
            'classML': {
                NAME.TEMPLATE   : TEST.CLASS,
                # 35 is just any number that causes no padding
                # 41 is the first size for which bluestein is faster
                TEST.NUM_ROWS   : 24,
                TEST.NUM_COLS   : 24,
                'mTypeC'        : TEST.Permutation(TEST.FEWTYPES),
                'tenT'          : (lambda param: TEST.ArrayGenerator({
                    TEST.DTYPE  : param['mTypeC'],
                    TEST.SHAPE  : (5, 5, 5),
                    TEST.ALIGN  : param[TEST.PARAMALIGN]
                })),
                'split'         : np.array([3, 2, 4]),
                TEST.INITARGS   : (lambda param : [
                    param.tenT()
                ]),
                TEST.INITKWARGS : {
                    'optimize'      : 'optimize',
                    'split'         : 'split'
                },
                TEST.NAMINGARGS : dynFormat(
                    "%s,optimize=%s,split=%s",
                    'tenT', 'optimize', 'split'
                ),
            },
            'transformML': {
                NAME.TEMPLATE   : TEST.TRANSFORMS,
                # 35 is just any number that causes no padding
                # 41 is the first size for which bluestein is faster
                TEST.NUM_ROWS   : 24,
                TEST.NUM_COLS   : 24,
                'mTypeC'        : TEST.Permutation(TEST.FEWTYPES),
                'tenT'          : (lambda param: TEST.ArrayGenerator({
                    TEST.DTYPE  : param['mTypeC'],
                    TEST.SHAPE  : (5, 5, 5),
                    TEST.ALIGN  : param[TEST.PARAMALIGN]
                })),
                'split'         : np.array([3, 2, 4]),
                TEST.INITARGS   : (lambda param : [
                    param.tenT()
                ]),
                TEST.INITKWARGS : {
                    'optimize'      : 'optimize',
                    'split'         : 'split'
                },
                TEST.NAMINGARGS : dynFormat(
                    "%s,optimize=%s,split=%s",
                    'tenT', 'optimize', 'split'
                ),
            }
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
