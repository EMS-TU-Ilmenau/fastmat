# -*- coding: utf-8 -*-

# Copyright 2018 Sebastian Semper, Christoph Wagner
#    https://www.tu-ilmenau.de/it-ems/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from libc.string cimport memset
from libc.math cimport isnan

import types
import sys

import numpy as np
cimport numpy as np
from scipy.sparse import spmatrix

# initialize numpy C-API interface for cython extension-type classes.
# This call is required once for every module that uses PyArray_ calls
# WARNING: DO NOT REMOVE THIS LINE OR SEGFAULTS ARE ABOUT TO HAPPEN!
np.import_array()

from .core.types cimport *
from .core.cmath cimport *
from .core.calibration import getMatrixCalibration, CALL_FORWARD, CALL_BACKWARD
from .core.resource import getMemoryFootprint
from .Product cimport Product
from .Sum cimport Sum
from .Diag cimport Diag

################################################################################
################################################## class FastmatFlags
cdef class FastmatFlags(object):
    def __init__(self):
        self.bypassAllow = True
        self.bypassAutoArray = True

flags = FastmatFlags()


################################################################################
################################################## class MatrixCallProfile
cdef class MatrixCallProfile(object):
    r"""MatrixCallProfile Class

    **Description:**

    Manage performance estimates and offer a clean and quick interface for
    make the decision whether to bypass a transform. To interface itself is
    call-oriented and keeps track of two estimates: how much time does a call
    take when using the locally implemented method (the "overridee") in
    comparision to when using the abstract base classes' ("overridden") method.

    The estimates will be represented by a linear runtime estimation model
    based on which the runtime of a call is composed by two factors: A fixed
    duration, named *CallOverhead* represents portions of the computation time
    irrelevant of the actual problem size whereas *PerUnit* represents portions
    that are.

    In order to compensate for runtime models scaling nonlinearly with problem
    size the *PerUnit* estimates are based on a complexity factor which can be
    implemented by overriding *Matrix._getComplexity()*, effectively allowing
    estimation of known-to-be jumpy or irregular proportions of runtime vs.
    problem size.

    The class keeps two models for one call: one for the local-class methods
    and one for the more general baseclass types. The local-class model also
    features tracking of nested calls (i.e. when the algorithm makes use of
    transforms of other fastmat Matrix instances)

    For the successful and complete initialization of the call profile the
    Matrix baseclass, the instanciated fastmat class and all the nested fastmat
    classes it requires must be calibrated prior to the instantiation of the
    class. If any prerequisite cannot be found all derived parameters contain
    *NaN* to signal incomplete calibration. *MatrixCallProfile.isValid()*
    allows checking if all necessary model parameters are available and
    properly set.

    The actual decision can be invoked by calling
    *MatrixCallProfile.isBypassfaster()* returning True if using the general
    methods implemented in the Matrix baseclass are estimated to be more
    efficient than the specific transforms implemented in the instances' class.
    Note that if *MatrixCallProfile.isValid()* returns _False_ the decision is
    always _False_ and the decision is never in favour of the base classes'
    implementation.
    """

    def __init__(self, targetInstance, targetCall, cplxAlg=0., cplxBypass=0.):
        '''
        Initialize A MatrixCallProfile instance.

        Parameters
        ----------
        targetInstance : :py:class:`fastmat.Matrix`
            The fastmat matrix instance a call profile needs to be generated
            for.

        targetCall : callable
            TODO: Needs to be specified.

        cplxAlg : int, optional
            The complexity estimate for the transforms implemented in the
            matrix class of `targetInstance`.

            Defaults to 0.

        cplxBypass : int, optional
            The complexity estimate for the (bypass) transforms implemented in
            the :py:class:`fastmat.Matrix` base class.

            Defaults to 0.
        '''

        # store given complexity estimates in profile
        self.complexityAlg, self.complexityBypass = cplxAlg, cplxBypass

        # fetch calibration data and complexity estimate of base class
        cdef MatrixCalibration calClass = None, calBase = None

        if targetInstance is not None:
            calClass = getMatrixCalibration(targetInstance.__class__)
            calBase = getMatrixCalibration(Matrix)

        # Call the unbound method of MatrixCalibration to get calibration
        # parameters. This way we can pass a None instance of MatrixCalibration
        # and get proper return values (i.e. a tuple containing (nan, nan))
        cdef tuple calParams
        cdef np.float32_t nan = np.nan
        if calClass is None:
            self.timeAlgCallOverhead, self.timeAlgPerUnit = nan, nan
        else:
            calParams = MatrixCalibration.getCall(calClass, targetCall)
            self.timeAlgCallOverhead = calParams[0]
            self.timeAlgPerUnit = calParams[1] * self.complexityAlg

        if calBase is None:
            self.timeBypassCallOverhead, self.timeBypassPerUnit = nan, nan
        else:
            calParams = MatrixCalibration.getCall(calBase, targetCall)
            self.timeBypassCallOverhead = calParams[0]
            self.timeBypassPerUnit = calParams[1] * self.complexityBypass

        # reset tracking of nested classes' estimates
        (self.timeNestedCallOverhead, self.timeNestedPerUnit) = 0., 0.

    def __str__(self):
        return "%.3g + %.3g * M [Nested: %.3g + %.3g * M, cplx %.3g]" %(
            self.timeAlgCallOverhead, self.timeAlgPerUnit,
            self.timeNestedCallOverhead, self.timeNestedPerUnit,
            self.complexityAlg
        )

    def __repr__(self):
        return self.__str__()

    cpdef void addNestedProfile(
            self, intsize numCols, bint allowBypass, MatrixCallProfile nested):
        '''
        Add the runtime estimate of a nested (child) class instance to the
        total estimate for this class instances' profile.

        This is needed if a meta class contains multiple fastmat class
        instances which are equipped with valid runtime estimation models
        themselves.

        Parameters
        ----------
        numCols : int
            Number of computations to be processed. Acts as a scaling factor
            to account for situations where an implemented transforms requires
            multiple invokations of the profiled nested classes' call to
            produce one output element.

            Example:
                If this is set to three, three vectors must be processed in
                the nested classes' transform in order to process one vector
                in this instances' transform.

        allowBypass : bool
            Specify if bypassing the nested class instances is configured to
            allow runtime shortcuts arising from bypassing nested instances'
            transforms. (This must be taken account for in the profiles)

        nested : :py:class:`fastmat.MatrixCallProfile`
            The call profile of the nested class instance to be added to the
            `nested` section of this call profile.
        '''
        cdef bint bypass = allowBypass and nested.isBypassFaster(numCols)
        if bypass:
            self.timeNestedCallOverhead += nested.timeBypassCallOverhead
            self.timeNestedPerUnit += nested.timeBypassPerUnit * numCols
        else:
            self.timeNestedCallOverhead += nested.timeAlgCallOverhead
            self.timeNestedPerUnit += nested.timeAlgPerUnit * numCols

    cpdef bint isValid(self):
        '''
        Return True if valid calibration data and all model parameters are
        available.
        '''
        return (
            np.isfinite(self.timeAlgCallOverhead) and
            np.isfinite(self.timeAlgPerUnit) and
            np.isfinite(self.timeBypassCallOverhead) and
            np.isfinite(self.timeBypassPerUnit) and
            self.timeAlgCallOverhead > 0 and
            self.timeAlgPerUnit > 0 and
            self.timeBypassCallOverhead > 0 and
            self.timeBypassPerUnit > 0
        )

    cpdef bint isBypassFaster(self, intsize numVectors):
        '''
        Return true if the general base class method is estimated to be faster.

        Parameters
        -----------
        numVectors : int
            Number of column vectors to process during one transform call.
        '''
        return (
            self.timeAlgCallOverhead + self.timeNestedCallOverhead -
            self.timeBypassCallOverhead +
            numVectors * (self.timeAlgPerUnit + self.timeNestedPerUnit -
                          self.timeBypassPerUnit) > 0
        )

    cpdef tuple estimateRuntime(self, intsize numVectors):
        '''
        Return a runtime estimate for the algorithm and its bypass runtime.

        Parameters
        ----------
        numVectors : int
            Number of column vectors to process during one transform call.
        '''
        return (
            (self.timeAlgCallOverhead + self.timeNestedCallOverhead +
             (self.timeAlgPerUnit + self.timeNestedPerUnit) * numVectors),
            self.timeBypassCallOverhead + self.timeBypassPerUnit * numVectors
        )


################################################################################
################################################## class MatrixCalibration
cdef class MatrixCalibration(dict):
    '''
    MatrixCalibration class

    A wrapper around :py:class:`dict` offering specialized routines to access
    matrix calibration data.
    '''

    cpdef tuple getCall(self, targetCall):
        '''
        Return the calibration values for a particular call type.

        Supports being called as unbound method.

        Parameters
        ----------
        targetCall : hashable
            Indicator of the call type to be calibrated.

        Returns
        -------
            A tuple containing the calibration data for the given call if is
            exists and valid calibration data for the corresponding Matrix is
            available. Returns (NaN, NaN) if either of that is not.

        '''
        if self is not None:
            return self.get(targetCall, (np.nan, np.nan))


################################################################################
################################################## class Matrix
cdef class Matrix(object):
    r"""Matrix Base Class


    **Description:**
    The baseclass of all matrix classes in fastmat. It also serves as wrapper
    around the standard Numpy Array :ref:`[1]<ref1>`.
    """

    ############################################## basic class properties
    property dtype:
        # r"""Report the matrix elements' data type
        #
        # *(read-only)*
        # """
        def __get__(self):
            return np.PyArray_TypeObjectFromType(self.numpyType)

    property shape:
        # r"""Report the matrix shape as tuple of (rows, columns)
        #
        # *(read-only)*
        # """
        def __get__(self):
            return (self.numRows, self.numCols)

    ############################################## class resource handling
    # nbytes - Property(read)
    # Size of the Matrix object
    property nbytes:
        # r"""Number of bytes in memory used by this instance
        # """
        def __get__(self):
            return getMemoryFootprint(self)

    # _nbytesReference - Property(read)
    # Size of the Matrix object
    property nbytesReference:
        # r"""Number of Bytes Reference
        #
        # """
        def __get__(self):
            if self._forwardReferenceMatrix is None:
                self._forwardReferenceInit()

            global getMemoryFootprint
            if getMemoryFootprint is None:
                getMemoryFootprint = __import__(
                    '.core.resource', globals(), locals(),
                    ['getMemoryFootprint']).getMemoryFootprint

            return getMemoryFootprint(self._forwardReferenceMatrix)

    def __copy__(self):
        # r"""
        #
        # Performs a copy() on this class instance. As fastmat matrix classes
        # are defined immutable nothing must be copied, so retuning self
        # suffices.
        # """
        return self

    def __deepcopy__(self, dict memo):
        # r"""
        #
        # Performs a deepcopy() on this class instance. As fastmat matrix
        # classes are defined immutable nothing must be copied, so retuning
        # self suffices.
        # """

        return self

    ############################################## array, col, row & item access
    property array:
        # r"""Return a dense array representation of the matrix
        # """
        def __get__(self):
            return (self.getArray() if self._array is None else self._array)

    cpdef np.ndarray getArray(self):
        '''
        Return a dense array representation of this matrix.
        '''
        if self.__class__ is not Matrix:
            self._array = self._getArray()

        return self._array

    cpdef np.ndarray _getArray(self):
        '''Internally overloadable method for customizing self.getArray.'''
        # check whether this instance is a Matrix or some child class
        if self.__class__ == Matrix:
            # it is a Matrix, so return our internal data
            return self._array
        else:
            # it is any child class. Use self.forward() to generate a ndarray
            # representation of this matrix. This calculation is very slow but
            # also very general. Therefore it is used also in child classes
            # when no explicit code for self._getArray() is provided.
            return self._forward(np.eye(self.numCols, dtype=self.dtype))

    def getCols(self, indices):
        r"""
        Return a set of columns by index.

        Parameters
        ----------
        indices : int OR :py:class:`numpy.ndarray`
            If an integer is given, this is equal to the output of
            :py:meth:`getCol`(indices).
            If a 1d vector is given, a 2d :py:class:`numpy.ndarray` containing
            the columns, as specified by the indices in `indices`, is returned

        Returns
        -------
            1d or 2d (depending on type of `indices`) :py:class:`numpy.ndarray`
            holding the specified column(s).
        """
        cdef np.ndarray arrResult, arrIdx
        cdef intsize ii, numSize

        if np.isscalar(indices):
            arrResult = self.getCol(indices)
        else:
            arrIdx = np.array(indices)
            if arrIdx.ndim > 1:
                raise ValueError("Index array must have at most one dimension.")

            # if a dense representation already exists, use it!
            if self._array is not None:
                arrResult = self._array[:, indices]
            else:
                numSize = arrIdx.size
                arrResult = _arrEmpty(2, self.numRows, numSize, self.numpyType)

                for ii in range(numSize):
                    arrResult[:, ii] = self.getCol(arrIdx[ii])

        return arrResult

    def getCol(self, idx):
        r"""
        Return a column by index.

        Parameters
        ----------
        idx : int
            Index of the column to return.

        Returns
        -------
            1d-:py:class:`numpy.ndarray` holding the specified column.
        """
        if idx < 0 or idx >= self.numCols:
            raise ValueError("Column index exceeds matrix dimensions.")

        # if a dense representation already exists, use it!
        if self._array is not None:
            return _arrSqueeze(self._array[:, idx])

        return self._getCol(idx)

    cpdef np.ndarray _getCol(self, intsize idx):
        '''Internally overloadable method for customizing self.getCol.'''
        cdef np.ndarray arrData = _arrZero(1, self.numCols, 1, self.numpyType)
        arrData[idx] = 1
        return self.forward(arrData)

    def getRows(self, indices):
        r"""
        Return a set of rows by index.

        Parameters
        ----------
        indices : int OR :py:class:`numpy.ndarray`
            If an integer is given, this is equal to the output of
            :py:meth:`getRow`(indices).
            If a 1d vector is given, a 2d :py:class:`numpy.ndarray` containing
            the rows, as specified by the indices in `indices`, is returned

        Returns
        -------
            1d or 2d (depending on type of `indices`) :py:class:`numpy.ndarray`
            holding the specified row(s).
        """
        cdef np.ndarray arrResult, arrIdx
        cdef intsize ii, numSize

        if np.isscalar(indices):
            arrResult = self.getRow(indices)
        else:
            arrIdx = np.array(indices)
            if arrIdx.ndim > 1:
                raise ValueError("Index array must have at most one dimension.")

            # if a dense representation already exists, use it!
            if self._array is not None:
                arrResult = self._array[indices, :]
            else:
                numSize = arrIdx.size
                arrResult = _arrEmpty(2, numSize, self.numCols, self.numpyType)

                for ii in range(numSize):
                    arrResult[ii, :] = self.getRow(arrIdx[ii])

        return arrResult

    def getRow(self, idx):
        r"""
        Return a row by index.

        Parameters
        ----------
        idx : int
            Index of the row to return.

        Returns
        -------
            1d-:py:class:`numpy.ndarray` holding the specified row.
        """
        if idx < 0 or idx >= self.numRows:
            raise ValueError("Row index exceeds matrix dimensions.")

        # if a dense representation already exists, use it!
        if self._array is not None:
            return _arrSqueeze(self._array[idx, :])

        return self._getRow(idx)

    cpdef np.ndarray _getRow(self, intsize idx):
        '''Internally overloadable method for customizing self.getRow.'''
        cdef np.ndarray arrData = _arrZero(1, self.numRows, 1, self.numpyType)
        arrData[idx] = 1
        return self.backward(arrData).conj()

    def __getitem__(self, tplIdx):
        r"""
        Return the indexed element or slice the matrix.

        Parameters
        ----------
        tplIdx : tuple
            Element index or slice objects. The tuple contains either one
            Ellipsis object or two objects of type `int`, iterable or `slice`.

        Returns
        -------
        If `tplIdx` is of type (int, int)
            Return the single element at the given index

        If `tplIdx` is of type (int, iterable or slice)
            Return the row indexed by `int` as 1d :py:class:`numpy.ndarray`, or
            a selection of it.

        if `tplIdx` is of type (iterable or slice, int)
            Return the column indexed by `int` as 1d :py:class:`numpy.ndarray`,
            or a selection of it.

        if `tplIdx` is of type (iterable or slice, iterable or slice)
            Return a subselection of the matrix' array representation as
            2d :py:class:`numpy.ndarray`.

        if `tplIdx` is of type (ellipsis):
            Return the full matrix' array representation as
            2d :py:class:`numpy.ndarray`.
        """
        if tplIdx is Ellipsis:
            return self.array
        elif not isinstance(tplIdx, tuple) or len(tplIdx) != 2:
            raise ValueError("Applied matrix indexing not supported.")

        cdef bint fullAccessCols, fullAccessRows
        cdef slice slcCol, slcRow
        cdef tuple idxCols, idxRows
        cdef intsize numCols, numRows

        # from here on we know it's a 2D tuple. Access the _array if available
        if self._array is not None:
            return self._array[tplIdx[0], tplIdx[1]]
        elif isinstance(tplIdx[0], slice):
            slcRow = tplIdx[0]
            if isinstance(tplIdx[1], slice):
                # double slice! Check if complete cols or rows may be requested
                slcCol = tplIdx[1]

                # determine slice index ranges, number of requested elements
                # along each axis and whether full axis access is possible
                idxRows = slcRow.indices(self.numRows)
                idxCols = slcCol.indices(self.numCols)
                numRows = (idxRows[1] - idxRows[0]) // idxRows[2]
                numCols = (idxCols[1] - idxCols[0]) // idxCols[2]
                fullAccessCols = (numRows == self.numRows and
                                  idxRows == (0, self.numRows, 1))
                fullAccessRows = (numCols == self.numCols and
                                  idxCols == (0, self.numCols, 1))

                # check if the slices result in a true element access
                if numRows == 1 and numCols == 1:
                    return self._getItem(idxRows[0], idxCols[0])

                # depending on this information access the array in the most
                # efficient way
                if fullAccessCols and fullAccessRows:
                    return self.array
                elif fullAccessCols:
                    return self.getCols(np.arange(*idxRows))
                elif fullAccessRows:
                    return self.getRows(np.arange(*idxCols))
                else:
                    # do what involves less resulting memory
                    if numCols * self.numRows < numRows * self.numCols:
                        return self.getCols(np.arange(*idxCols))[slcRow, ...]
                    else:
                        return self.getRows(np.arange(*idxRows))[..., slcCol]

            else:
                # Column-access!
                return self.getCols(tplIdx[1])[slcRow, ...]
        elif np.isscalar(tplIdx[0]) and np.isscalar(tplIdx[1]):
            # True element access!
            return self._getItem(*tplIdx)
        else:
            # Row access!
            return self.getRows(tplIdx[0])[..., tplIdx[1]]

    cpdef object _getItem(self, intsize idxRow, intsize idxCol):
        '''Internally overloadable method for customizing self.getItem.'''
        if self._array is not None:
            return self._array[idxRow, idxCol]
        else:
            if self.numRows < self.numCols:
                return self._getCol(idxCol)[idxRow]
            else:
                return self._getRow(idxRow)[idxCol]

    ############################################## Matrix content property
    property content:
        # r"""Return the Matrix contents
        #
        # *(read-only)*
        # """
        def __get__(self):
            return self._content

    property largestEV:
        def __get__(self):
            import warnings
            warnings.warn(
                'largestEV is deprecated. Use largestEigenValue.',
                FutureWarning
            )
            return self.largestEigenValue

    ############################################## algorithmic properties
    property largestEigenValue:
        r"""Return the largest eigenvalue for this matrix instance

        *(read-only)*
        """
        def __get__(self):
            if self._largestEigenValue is None:
                return self.getLargestEigenValue()
            else:
                return self._largestEigenValue

    def getLargestEigenValue(self):
        r"""
        Largest Singular Value

        For a given matrix :math:`A \in \mathbb{C}^{n \times n}`, so :math:`A`
        is square, we calculate the absolute value of the largest eigenvalue
        :math:`\lambda \in \mathbb{C}`. The eigenvalues obey the equation

        .. math::
             A \cdot  v= \lambda \cdot  v,

        where :math:`v` is a non-zero vector.

        Input matrix :math:`A`, parameter :math:`0 < \varepsilon \ll 1` as a
        stopping criterion
        Output largest eigenvalue :math:`\sigma_{\rm max}( A)`

        .. note::
            This algorithm performs well if the two largest eigenvalues are
            not very close to each other on a relative scale with respect to
            their absolute value. Otherwise it might get trouble converging
            properly.

        >>> # import the packages
        >>> import numpy.linalg as npl
        >>> import numpy as np
        >>> import fastmat as fm
        >>>
        >>> # define the matrices
        >>> n = 5
        >>> H = fm.Hadamard(n)
        >>> D = fm.Diag(np.linspace
        >>>         1, 2 ** n, 2 ** n))
        >>>
        >>> K1 = fm.Product(H, D)
        >>> K2 = K1.array
        >>>
        >>> # calculate the eigenvalue
        >>> x1 = K1.largestEigenValue
        >>> x2 = npl.eigvals(K2)
        >>> x2 = np.sort(np.abs(x2))[-1]
        >>>
        >>> # check if the solutions match
        >>> print(x1 - x2)

        We define a matrix-matrix product of a Hadamard matrix and a diagonal
        matrix. Then we also cast it into a numpy-array and use the integrated
        EVD. For demonstration, try to increase :math:`n`to :math:`>10`and see
        what happens.
        """

        if self.numRows != self.numCols:
            raise ValueError("largestEigenValue: Matrix must be square.")

        result = self._getLargestEigenValue()
        self._largestEigenValue = self._largestEigenValue if np.isnan(
            result) else result
        return result

    cpdef object _getLargestEigenValue(self):

        self.scipyLinearOperator.dtype = np.promote_types(
            np.float64,
            self.dtype
        )

        # the scipy eigenvalue operations do not work on 1x1 transforms
        if self.numRows > 1:
            from scipy.sparse import linalg
            result = linalg.eigs(
                self.scipyLinearOperator,
                1,
                return_eigenvectors=False
            )[0]
        else:
            from numpy.linalg import eigvals
            result = eigvals(self.array)[0]

        self.scipyLinearOperator.dtype = self.dtype
        return result

    property largestEigenVec:
        r"""Return the vector corresponding to the largest eigen value

        *(read-only)*
        """

        def __get__(self):
            return (self.getLargestEigenVec() if self._largestEigenVec is None
                    else self._largestEigenVec)

    def getLargestEigenVec(self):
        if self.numRows != self.numCols:
            raise ValueError("largestEigenVec: Matrix must be square.")

        result = self._getLargestEigenVec()
        self._largestEigenValue = result[0]
        self._largestEigenVec = result[1]
        return self._largestEigenVec

    cpdef tuple _getLargestEigenVec(self):

        # we temporally promote the operators type to satisfy scipy
        self.scipyLinearOperator.dtype = np.promote_types(
            np.float64,
            self.dtype
        )

        if self.numRows > 2:
            # now we can do the efficient thing using the linear operator
            from scipy.sparse.linalg import eigs

            S, V = eigs(
                self.scipyLinearOperator,
                1,
                return_eigenvectors=True
            )[0]
        else:
            from numpy.linalg import eig
            result = eig(self.array)
            S = (result[0]).astype(self.scipyLinearOperator.dtype)[0]
            V = (result[1][:, 0]).astype(self.scipyLinearOperator.dtype)

        self.scipyLinearOperator.dtype = self.dtype
        return (S, V)

    property largestSV:
        def __get__(self):
            import warnings
            warnings.warn(
                'largestSV is deprecated. Use largestSingularValue.',
                FutureWarning
            )
            return self.largestSingularValue

    property largestSingularValue:
        r"""Return the largestSingularValue for this matrix instance

        *(read-only)*
        """

        def __get__(self):
            if self._largestSingularValue is None:
                return self.getLargestSingularValue()
            else:
                return self._largestSingularValue

    def getLargestSingularValue(self):
        r"""Largest Singular Value

        For a given matrix :math:`A \in \mathbb{C}^{n \times m}`, we calculate
        the largest singular value :math:`\sigma_{\rm max}( A) > 0`, which is
        the largest entry of the diagonal matrix
        :math:`\Sigma \in \mathbb{C}^{n \times m}` in the decomposition

        .. math::
             A =  U  \Sigma  V^{\rm H},

        where :math:`U` and :math:`V` are matrices of the appropriate
        dimensions. This is done via the so called power iteration of
        :math:`A^{\rm H} \cdot  A`.

        - Input matrix :math:`A`, parameter :math:`0 < \varepsilon \ll 1` as \
          a stopping criterion
        - Output largest singular value :math:`\sigma_{\rm max}( A)`

        .. note::
            This algorithm performs well if the two largest singular values
            are not very close to each other on a relative scale. Otherwise
            it might get trouble converging properly.

        >>> # import packages
        >>> import numpy.linalg as npl
        >>> import numpy as np
        >>> import fastmat
        >>>
        >>> # define involved matrices
        >>> n = 5
        >>> H = fm.Hadamard(n)
        >>> F = fm.Fourier(2**n)
        >>> K1 = fm.Kron(H, F)
        >>> K2 = K1
        >>>
        >>> # calculate the largest SV
        >>> # and a reference solution
        >>> x1 = largestSingularValue(K1.largestSingularValue
        >>> x2 = npl.svd(K2,compute_uv
        >>> # check if they match
        >>> print(x1-x2)

        We define a Kronecker product of a Hadamard matrix and a Fourier
        matrix. Then we also cast it into a numpy-array and use the integrated
        SVD. For demonstration, try to increase :math:`n` to `>10` and see what
        happens.

        Returns
        -------
            The largest singular value
        """

        result = self._getLargestSingularValue()
        self._largestSingularValue = self._largestSingularValue if np.isnan(
            result) else result
        return result

    cpdef object _getLargestSingularValue(self):

        from scipy.sparse.linalg import svds

        # we temporally promote the operators type to satisfy scipy
        self.scipyLinearOperator.dtype = np.promote_types(
            np.float64,
            self.dtype
        )

        if (self.numRows > 2) and (self.numCols > 2):
            result = svds(
                self.scipyLinearOperator,
                1,
                return_singular_vectors=False
            )[0]
        else:
            from numpy.linalg import svd
            result = (svd(
                np.atleast_2d(self.array),
                compute_uv=False
            )).astype(np.float64)[0]

        self.scipyLinearOperator.dtype = self.dtype
        return result

    property largestSingularVectors:
        r"""Return the vectors corresponding to the largest singular value

        This property returns a tuple (u, v) of the first
        columns of U and V in the singular value decomposition of

        .. math::
             A =  U  \Sigma  V^{\rm H},

        which means that the tuple contains the leading left and right
        singular vectors of the matrix

        *(read-only)*
        """

        def __get__(self):
            if self._largestSingularVectors is None:
                return self.getLargestSingularVectors()
            else:
                return self._largestSingularVectors

    def getLargestSingularVectors(self):
        result = self._getLargestSingularVectors()
        self._largestSingularValue = (result[1]).astype(np.float64)
        self._largestSingularVectors = (
            result[0].astype(np.float64),
            result[2].astype(np.float64)
        )
        return self._largestSingularVectors

    cpdef tuple _getLargestSingularVectors(self):

        # we temporally promote the operators type to satisfy scipy
        self.scipyLinearOperator.dtype = np.promote_types(
            np.float64,
            self.dtype
        )

        if (self.numRows > 1) and (self.numCols > 1):
            # now we can do the efficient thing using the linear operator
            from scipy.sparse.linalg import svds

            U, S, V = svds(
                self.scipyLinearOperator,
                1,
                return_singular_vectors=True
            )[0]
        else:
            # now we do the stupid thing, but scipy forces us to do so
            from numpy.linalg import svd

            U, S, V = svd(np.atleast_2d(self.array), compute_uv=True)

        self.scipyLinearOperator.dtype = self.dtype
        return (U, S, V)

    property scipyLinearOperator:
        """Return a Representation as scipy's linear Operator

        This property allows to make use of all the powerfull algorithms
        provided by scipy, that allow passing a linear operator to
        them, like optimization routines, system solvers or decomposition
        algorithms.

        *(read-only)*
        """

        def __get__(self):
            if self._scipyLinearOperator is None:
                return self.getScipyLinearOperator()
            else:
                return self._scipyLinearOperator

    def getScipyLinearOperator(self):
        self._scipyLinearOperator = self._getScipyLinearOperator()
        return self._scipyLinearOperator

    cpdef object _getScipyLinearOperator(self):
        from scipy.sparse.linalg import LinearOperator
        return LinearOperator(
            shape=self.shape,
            matvec=self.forward,
            rmatvec=self.backward,
            matmat=self.forward,
            dtype=self.dtype
        )

    ############################################## gram
    property gram:
        r"""Return the gram matrix for this fastmat class

        *(read-only)*
        """

        def __get__(self):
            return (self.getGram() if self._gram is None
                    else self._gram)

    def getGram(self):
        r"""
        Return the gramian of this matrix as fastmat matrix.
        """
        self._gram = self._getGram()
        return self._gram

    cpdef Matrix _getGram(self):
        '''Internally overloadable method for customizing self.getGram.'''
        return Product(self.H, self)

    ############################################## colNorms
    property colNorms:
        r"""Return the column norms for this matrix instance

        *(read-only)*
        """

        def __get__(self):
            return (self.getColNorms() if self._colNorms is None
                    else self._colNorms)

    def getColNorms(self):
        r"""
        Return a column normalized version of this matrix as fastmat matrix.
        """
        self._colNorms = self._getColNorms()
        return self._colNorms

    cpdef np.ndarray _getColNorms(self):
        '''
        Internally overloadable method for customizing self.getColNorms.
        '''
        # choose float64 as type of normalization diagonal matrix to preserve
        # accuracy of norms also when constructing meta classed of larger type
        diagType = np.float64

        # array that contains the norms of each column
        cdef np.ndarray arrNorms = np.empty(self.numCols, dtype=diagType)

        # number of elements we consider at once during normalization
        # Scale chunk size to fit a memory buffer of at most 16M
        cdef intsize numStrideSize = max(
            1, min(
                1024,
                (16777216 // typeInfo[self.fusedType].dsize) // self.numRows
            )
        )

        cdef np.ndarray arrSelector = np.zeros(
            (self.numCols, min(numStrideSize, self.numCols)),
            dtype=diagType)

        cdef intsize ii
        for ii in range(0, self.numCols, numStrideSize):
            vecIndices = np.arange(ii, min(ii + numStrideSize, self.numCols))
            if ii == 0 or vecSlice.size != vecIndices.size:
                vecSlice = np.arange(0, vecIndices.size)

            arrSelector[vecIndices, vecSlice] = 1

            arrNorms[vecIndices] = np.linalg.norm(
                self.forward(
                    arrSelector if vecIndices.size == arrSelector.shape[1]
                    else arrSelector[:, vecSlice]),
                axis=0
            )
            arrSelector[vecIndices, vecSlice] = 0

        return arrNorms

    ############################################## rowNorms
    property rowNorms:
        r"""Return the row norms for this matrix instance

        *(read-only)*
        """

        def __get__(self):
            return (self.getRowNorms() if self._rowNorms is None
                    else self._rowNorms)

    def getRowNorms(self):
        r"""
        Return a row normalized version of this matrix as fastmat matrix.
        """
        self._rowNorms = self._getRowNorms()
        return self._rowNorms

    cpdef np.ndarray _getRowNorms(self):
        '''
        Internally overloadable method for customizing self.getRowNorms.
        '''
        # choose float64 as type of normalization diagonal matrix to preserve
        # accuracy of norms also when constructing meta classed of larger type
        diagType = np.float64

        # array that contains the norms of each column
        cdef np.ndarray arrNorms = np.empty(self.numRows, dtype=diagType)

        # number of elements we consider at once during normalization
        # Scale chunk size to fit a memory buffer of at most 16M
        cdef intsize numStrideSize = max(
            1, min(
                1024,
                (16777216 // typeInfo[self.fusedType].dsize) // self.numCols
            )
        )

        cdef np.ndarray arrSelector = np.zeros(
            (self.numRows, min(numStrideSize, self.numRows)),
            dtype=diagType)

        cdef intsize ii
        for ii in range(0, self.numRows, numStrideSize):
            vecIndices = np.arange(ii, min(ii + numStrideSize, self.numRows))
            if ii == 0 or vecSlice.size != vecIndices.size:
                vecSlice = np.arange(0, vecIndices.size)

            arrSelector[vecIndices, vecSlice] = 1

            arrNorms[vecIndices] = np.linalg.norm(
                self.backward(
                    arrSelector if vecIndices.size == arrSelector.shape[1]
                    else arrSelector[:, vecSlice]
                ),
                axis=0)
            arrSelector[vecIndices, vecSlice] = 0

        return arrNorms

    ############################################## colNormalized
    property colNormalized:
        r"""Return a column normalized matrix for this instance

        *(read-only)*
        """

        def __get__(self):
            return (self.getColNormalized() if self._colNormalized is None
                    else self._colNormalized)

    def getColNormalized(self):
        r"""
        Return a column normalized version of this matrix as fastmat matrix.
        """
        self._colNormalized = self._getColNormalized()
        return self._colNormalized

    cpdef Matrix _getColNormalized(self):

        cpdef np.ndarray arrNorms = self.colNorms

        # check if we've found any zero
        if np.any(arrNorms == 0):
            raise ValueError("Normalization: Matrix has zero-norm column.")

        # finally invert the diagonal and generate normalized matrix
        return self * Diag(1. / arrNorms)

    ############################################## rowNormalized
    property rowNormalized:
        r"""Return a column normalized matrix for this instance

        *(read-only)*
        """

        def __get__(self):
            return (self.getRowNormalized() if self._rowNormalized is None
                    else self._rowNormalized)

    def getRowNormalized(self):
        r"""
        Return a column normalized version of this matrix as fastmat matrix.
        """
        self._rowNormalized = self._getRowNormalized()
        return self._rowNormalized

    cpdef Matrix _getRowNormalized(self):

        cpdef np.ndarray arrNorms = self.rowNorms

        # check if we've found any zero
        if np.any(arrNorms == 0):
            raise ValueError("Normalization: Matrix has zero-norm row.")

        # finally invert the diagonal and generate normalized matrix
        return Diag(1. / arrNorms) * self

    ############################################## Transpose
    property T:
        r"""Return the transpose of the matrix as fastmat class

        *(read-only)*
        """

        def __get__(self):
            return (self.getT() if self._T is None else self._T)

    def getT(self):
        r"""
        Return the transpose of this matrix as fastmat matrix.
        """
        self._T = self._getT()
        return self._T

    cpdef Matrix _getT(self):
        '''Internally overloadable method for customizing self.getT.'''
        return Transpose(self)

    ############################################## Hermitian
    property H:
        r"""Return the hermitian transpose

        *(read-only)*
        """

        def __get__(self):
            return (self.getH() if self._H is None else self._H)

    def getH(self):
        r"""
        Return the hermitian transpose of this matrix as fastmat matrix.
        """
        self._H = self._getH()
        return self._H

    cpdef Matrix _getH(self):
        '''Internally overloadable method for customizing self.getH.'''
        return Hermitian(self)

    ############################################## conjugate
    property conj:
        r"""Return the conjugate of the matrix as fastmat class

        *(read-only)*
        """

        def __get__(self):
            return (self.getConj() if self._conj is None else self._conj)

    def getConj(self):
        r"""
        Return the conjugate of this matrix as fastmat matrix.
        """
        self._conj = self._getConj()
        return self._conj

    cpdef Matrix _getConj(self):
        '''Internally overloadable method for customizing self.getConj.'''
        return getConjugate(self)

    property inverse:
        r"""Return the inverse

        *(read-only)*
        """

        def __get__(self):
            if self._inverse is None:
                return self.getInverse()
            else:
                return self._inverse

    def getInverse(self):
        r"""
        Return the hermitian transpose of this matrix as fastmat matrix.
        """
        self._inverse = self._getInverse()
        return self._inverse

    cpdef Matrix _getInverse(self):
        '''Internally overloadable method for self.inverse.'''
        return Inverse(self)

    property pseudoInverse:
        r"""Return the moore penrose inverse

        *(read-only)*
        """

        def __get__(self):
            if self._pseudoInverse is None:
                return self.getPseudoInverse()
            else:
                return self._pseudoInverse

    def getPseudoInverse(self):
        r"""
        Return the hermitian transpose of this matrix as fastmat matrix.
        """
        self._pseudoInverse = self._getPseudoInverse()
        return self._pseudoInverse

    cpdef Matrix _getPseudoInverse(self):
        '''Internally overloadable method for self.pseudoInverse.'''
        return PseudoInverse(self)

    ############################################## numN deprecation warning
    property numN:
        def __get__(self):
            import warnings
            warnings.warn('numN is deprecated. Use numRows.', FutureWarning)
            return self.numRows

    property numM:
        def __get__(self):
            import warnings
            warnings.warn('numM is deprecated. Use numCols.', FutureWarning)
            return self.numCols

    property normalized:
        def __get__(self):
            import warnings
            warnings.warn('normalized is deprecated. Use colNormalized.',
                          FutureWarning)
            return self.colNormalized

    ############################################## computation complexity
    property complexity:
        r"""Complexity

        *(read-only)*

        Return the computational complexity of all functionality implemented in
        the class itself, not including calls of external code.
        """

        def __get__(self):
            return (self.profileForward.complexityAlg,
                    self.profileBackward.complexityAlg)

    def getComplexity(self):
        r"""
        Return a transform complexity estimate for this matrix instance.

        Returns a tuple containing the complexity estimates for the
        :py:meth:`fastmat.Matrix.forward` and
        :py:meth:`fastmat.Matrix.backward` transforms (in that order).
        """
        return self._getComplexity()

    cpdef tuple _getComplexity(self):
        '''
        Internally overloadable method for customizing self.getComplexity.
        '''
        cdef float complexity = self.numRows * self.numCols
        return (complexity, complexity)

    cdef void _initProfiles(self):
        """
        Generate performance profiles for the transforms of this matrix.

        Generate performance profiles based on intrinsic complexity and
        external dependencies (e.g. nested class calls) and condense results
        in an overall figure of merit based on:
          - computational complexity of the particular class instance
          - calibration parameters of class (map complexity to runtime)
          - nested calls to other fastmat matrix instances

        The resulting figure is divided into a fixed overhead and a variable
        effort per data vector the transform is applied to.

        The overall figures shall become valid (non-zero) if and only if all of
        the following requirements are met:
          - valid calibration parameters could be loaded for this class
          - in case of meta classes all embedded matrix classes must report
            valid (non-zero) values for overall overhead and effort

        Separate profiles will be generated for the
          - forward and backward transforms of this particular class instance
          - a bypass transform based on dot-product complexity
        """
        # determine complexity of class instance transforms and Bypass transform
        cdef tuple cplxA = self.getComplexity()
        cdef tuple cplxB = Matrix._getComplexity(self)

        # initialize profiles
        self.profileForward = MatrixCallProfile(
            self, 'forward', cplxAlg=cplxA[0], cplxBypass=cplxB[0])
        self.profileBackward = MatrixCallProfile(
            self, 'backward', cplxAlg=cplxA[1], cplxBypass=cplxB[0])

        # Explore the performance profiles of nested fastmat classes
        # Fills in the profile fields overheadNested and effortNested
        self._exploreNestedProfiles()

        # disable transform bypass if profile is either missing or incomplete
        if not (self.profileForward.isValid() and
                self.profileBackward.isValid()):
            self.bypassAllow = False

    cpdef _exploreNestedProfiles(self):
        r"""
        Explore the runtime properties of all nested fastmat matrices and
        update this matrix instances' profile information.
        """
        # Use an iterator on self._content by default to sum the profile
        # properties of all nested classes of meta-classes by default.
        # basic-classes either have an empty tuple for _content or need to
        # overwrite this method.
        cdef Matrix item
        cdef bint bypass

        for item in self:
            bypass = (item.bypassAllow and
                      (item._array is not None or item.bypassAutoArray))
            self.profileForward.addNestedProfile(
                1, bypass, item.profileForward)
            self.profileBackward.addNestedProfile(
                1, bypass, item.profileBackward)

    cpdef tuple estimateRuntime(self, intsize numVectors=1):
        r"""
        Estimate the runtime of this matrix instances' transforms.

        Parameters
        ----------
            numVectors : int
                Estimate the runtime for processing this number of vectors.

        Returns
        -------
            A tuple containing float estimates on the runtime of the
            :py:meth:`fastmat.Matrix.forward` and the
            :py:meth:`fastmat.Matrix.backward` transform if valid performance
            profiles are available to this matrix instance. If not, return
            (NaN, NaN)
        """
        cdef np.float32_t estAlgFwd, estBypassFwd, estAlgBwd, estBypassBwd

        estAlgFwd, estBypassFwd = self.profileForward.estimateRuntime(
            numVectors
        )
        estAlgBwd, estBypassBwd = self.profileBackward.estimateRuntime(
            numVectors
        )
        return (
            estBypassFwd
            if ((self._array is not None or self.bypassAutoArray) and
                self.bypassAllow and (estBypassFwd < estAlgFwd))
            else estAlgFwd,
            estBypassBwd
            if ((self._arrayH is not None or self.bypassAutoArray) and
                self.bypassAllow and (estBypassBwd < estAlgBwd))
            else estAlgBwd
        )

    ############################################## class methods
    def __init__(self, arrMatrix, **options):
        '''
        Initialize an instance of a fastmat matrix.

        This is the baseclass for all fastmat matrices and serves as a wrapper
        to define a matrix based on a two dimensional ndarray. Any specialized
        matrix type in fastmat is derived from this base class and defines its
        own `__init__`.

        Every `__init__` routine allows the specification of arbitrary
        keyworded arguments, which are passed in `**options`. Each specialized
        `__init__` routine processes the options it accepts and passes the rest
        on to the initialization routines in the base class to define the basic
        behaviour of the class.

        Parameters
        ----------
        arrMatrix : :py:class:`numpy.ndarray`
            A 2d array representing a dense matrix to be cast as a fastmat
            matrix.

        forceContiguousInput : bool, optional
            If set, the input array is forced to be contiguous in the style as
            specified by `fortranStyle`. If the input array already fulfils the
            requirement nothing is done.

            Defaults to False.

        widenInputDatatype : bool, optional
            If set, the data type of the input array is promoted to at least
            match the output data type of the operation. Just like the
            `minType` option this parameter controls the accumulator width,
            however dynamically according to the output data type in this case.

            Defaults to False.

        fortranStyle : bool, optional
            Control the style of contiguousity to be enforced by
            forceConfiguousInput. If this option is set to True, Fortran-style
            ordering (contiguous along columns) is enforced, if False C-Style
            (contiguous along rows).

            Defaults to True.

        minType : bool, optional
            Specify a minimum data type for the input array to a transform. The
            input array data type will be promoted to at least the data type
            specified in this option before performing the actual transforms.
            Using this option is strongly advised for cases where small data
            types of both input array and matrix could cause range overflows
            otherwise, as the output data type promotion rules do not consider
            avoiding accumulator overflows due to performance reasons.

            Defaults to :py:class:`numpy.int8`.

        bypassAllow : bool, optional
            Allow bypassing the implemented :py:meth:`fastmat.Matrix.forward`
            and :py:meth:`fastmat.Matrix.backward` transforms with dense
            matrix-vector products if runtime estimates suggest this is faster
            than using the implemented transforms. This requires valid
            calibration data to be available for the class of the to-be-created
            instance itself and the :py:class:`fastmat.Matrix` base class at
            the time the new instance is created. If no valid performance
            calibration data exists this parameter is ignored and the
            implemented transforms will be used always.

            Defaults to the value set in the package-wide
            :py:class:`fastmat.flags` options.

        bypassAutoArray : bool, optional
            Prevents the automatic generation of a dense matrix representation
            that would be used for bypassing the implemented transforms in case
            the performance profiles suggest this would be faster, if set to
            True. This is heavily advised if the matrix is unfeasibly large for
            a dense representation and does not feature fast transforms.

            Defaults to the value as set in the package-wide
            :py:class`fastmat.flags` if no nested matrix of this instance has
            set this option to False. If just one has, this parameter defaults
            to False. If the matrix instance would disregard this, a nested
            instances' AutoArray function would be called implicitly through
            this instances' dense array constructur although this is disabled
            for the particular ndested matrix.
        '''
        if not isinstance(arrMatrix, np.ndarray):
            raise TypeError("Matrix: Use Sparse() for scipy spmatrix."
                            if isinstance(arrMatrix, spmatrix)
                            else "Matrix: Matrix is not a numpy ndarray")

        dims = len(arrMatrix.shape)
        if dims == 2:
            self._array = np.copy(arrMatrix)
        elif dims < 2:
            self._array = np.reshape(np.copy(arrMatrix), (len(arrMatrix), 1))
        else:
            raise NotImplementedError("Matrix data array must be 2D")

        # set properties of matrix
        self._initProperties(
            self._array.shape[0],        # numRows
            self._array.shape[1],        # numCols
            self._array.dtype,           # data type of matrix
            **options
        )

    def _initProperties(
        self,
        intsize numRows,
        intsize numCols,
        object dataType,
        **options
    ):
        r"""
        Perform the initialization of basic matrix options for __init__().

        See the description of **options parameters in __init__() for further
        details.
        """

        # assign basic class options (at c-level)
        self.numRows = numRows
        self.numCols = numCols
        self.fusedType = getFusedType(dataType)
        self.numpyType = typeInfo[self.fusedType].numpyType

        # get and assign c-level options
        self._forceContiguousInput = options.get('forceContiguousInput', False)
        self._widenInputDatatype   = options.get('widenInputDatatype', False)
        self._fortranStyle         = options.get('fortranStyle', True)
        self._minFusedType  = getFusedType(options.get('minType', np.int8))
        self.bypassAllow    = options.get('bypassAllow', flags.bypassAutoArray)

        cdef bint autoArray = (flags.bypassAutoArray and
                               all(not item.bypassAutoArray for item in self))
        self.bypassAutoArray = options.get('bypassAutoArray', autoArray)

        # initialize performance profile
        # NOTE: If a valid profile is not available (either no calibration data
        # is found or the complexity model could not be evaluated properly),
        # self.bypassAllow will be disabled explicitly in self._initProfiles()
        # This way bypassing decisions are no longer dependent on (potentially)
        # volatile floating-point comparisions against NaN values, as was the
        # case previously)
        self._initProfiles()

    def _getProperties(self):
        '''
        Return the matrix properties as processed by _initProperties() as dict.
        '''
        return {
            'forceContiguousInput'  : self._forceContiguousInput,
            'widenInputDatatype'    : self._widenInputDatatype,
            'fortranStyle'          : self._fortranStyle,
            'minType'               : np.PyArray_TypeObjectFromType(
                typeInfo[self._minFusedType].numpyType
            ),
            'bypassAllow'           : self.bypassAllow,
            'bypassAutoArray'       : self.bypassAutoArray
        }

    def __repr__(self):
        # Return a string representing this very class instance. For
        # identification module name, class name, instance id and shape will
        # be returned formatted as string.
        return "<%s[%dx%d]:0x%12x>" %(
            self.__class__.__name__, self.numRows, self.numCols, id(self)
        )

    def __str__(self):
        # Return a string representing the classes' contents in a more
        # human-readable and human-interpretable fashion. Currently the
        # call is redirected to self.__repr__().
        return self.getArray().__str__()

    def __len__(self):
        """
        Return count of nested fastmat matrix instances in this matrix.
        """
        return 0 if self._content is None else len(self._content)

    def __iter__(self):
        """
        Iterate all nested fastmat matrix instances of this matrix.
        """
        return self if self._content is None else self._content.__iter__()

    def __next__(self):
        """Stop iteration as __iter__ redirected here. Python3-Style."""
        raise StopIteration

    def next(self):
        """Stop iteration as __iter__ redirected here. Python2-Style."""
        raise StopIteration

    ############################################## class operator overloading

    __array_priority__ = 20.

    def __add__(self, element):
        """Return the sum of this fastmat matrix instance and another."""
        if isinstance(element, Matrix):
            return Sum(self, element)
        else:
            raise TypeError("Not an addition of fastmat matrices.")

    def __radd__(self, element):
        """Return the sum of another fastmat matrix instance and this."""
        if isinstance(element, Matrix):
            return Sum(self, element)
        else:
            raise TypeError("Not an addition of fastmat matrices.")

    def __sub__(self, element):
        """
        Return the difference of this fastmat matrix instance and another.
        """
        if isinstance(element, Matrix):
            return Sum(
                self, Product(element, np.int8(-1), typeExpansion=np.int8))
        else:
            raise TypeError("Not a subtraction of fastmat matrices.")

    def __rsub__(self, element):
        """
        Return the difference of another fastmat matrix instance and this.
        """
        if isinstance(element, Matrix):
            return Sum(
                element, Product(self, np.int8(-1), typeExpansion=np.int8))
        else:
            raise TypeError("Not a subtraction of fastmat matrices.")

    def __mul__(self, factor):
        """
        Return the product of this matrix with eith another fastmat matrix or a
        scalar.
        """
        if isinstance(factor, np.ndarray):
            return self.forward(factor)
        else:
            return Product(self, factor, typeExpansion=np.int8)

    def __rmul__(self, factor):
        """
        Return the product of a scalar and this matrix.
        """
        if np.isscalar(factor) or isinstance(factor, Matrix):
            return Product(factor, self, typeExpansion=np.int8)
        else:
            raise TypeError("Invalid product term for fastmat Matrix.")

    def __div__(self, divisor):
        """
        Return the product of a this matrix by the reciproce of a given scalar.
        """
        if np.isscalar(divisor):
            if divisor != 0:
                return Product(self, 1. / divisor)
            else:
                raise ZeroDivisionError("Division by Zero.")
        else:
            raise TypeError("Only scalars allowed as divisors.")

    def __truediv__(self, divisor):
        return self.__div__(divisor)

    def __floordiv__(self, divisor):
        raise NotImplementedError("Matrix Floor division not allowed.")

    ############################################## class forward / backward
    cdef np.ndarray _prepareInputArray(self, np.ndarray arrInput,
                                       intsize requiredSize,
                                       TRANSFORM * xform):
        '''
        Prepare an input array to a transform

        Performs dimension checks on an input :py:meth:`numpy.ndarray` array
        and determines data types used for internal array representations and
        the returned output array.

        Parameters
        ----------
        arrInput : :py:class:`numpy.ndarray`
            The input array as passed to the transform method, either as 1d or
            2d.

        requiredSize : int
            The data (column) vector size the input data array must proof
            during dimension checks. Bypass this check if equal to zero.

        xform : :py:class:`TRANSFORM`
            extension-class structure holding the data types determined in
            this method.

        Returns
        -------
            An 2d :py:class:`numpy.ndarray` array with adjusted data type and
            memory alignment as defined by the specified internal properties
            of this matrix instance.
        '''
        # check input dimenstions and reshape to 2-dimensions if required
        # allow to bypass this check as in some call cases we can be certain
        # that this is fulfilled already
        if requiredSize > 0:
            # check dimension count and sizes
            if arrInput.ndim > 2:
                raise ValueError("Input data array must be at most 2D")

            # arrInput.N must match self.numCols in forward() and
            # .numRows in backward()
            if arrInput.shape[0] != requiredSize:
                raise ValueError(
                    "Mismatch of vector size %d to relevant matrix axis %d" %(
                        arrInput.shape[0], requiredSize
                    )
                )

            # force array of data to be 2D and determine vector count
            if arrInput.ndim == 1:
                arrInput = _arrReshape(
                    arrInput, 2, requiredSize, 1, np.NPY_ANYORDER)

        # assume 2D input now
        xform.numVectors = arrInput.shape[1]

        # Determine internal and output array data types
        #  * promote internal dtype to be at least self._minFusedType
        #  * promote internal dtype to output dtype if self._widenInputDatatype
        #  * force internal array alignment if self._forceContiguousInput
        # force input data type to fulfill some requirements if needed
        #  * check for data type match
        #  * check for data alignment (contiguousy and segmentation)
        xform[0].fInput = typeSelection[np.PyArray_TYPE(arrInput)]
        xform[0].fInternal = \
            typeInfo[xform[0].fInput].promote[self._minFusedType]
        xform[0].fOutput = \
            typeInfo[xform[0].fInternal].promote[self.fusedType]
        if self._widenInputDatatype:
            xform[0].fInternal = xform[0].fOutput

        xform[0].nInternal = typeInfo[xform[0].fInternal].numpyType
        xform[0].nOutput = typeInfo[xform[0].fOutput].numpyType

        if self._forceContiguousInput:
            return _arrForceTypeAlignment(
                arrInput, xform[0].nInternal, 0, self._fortranStyle
            )
        elif xform[0].fInternal != xform[0].fInput:
            return _arrForceType(arrInput, xform[0].nInternal)
        else:
            return arrInput

    cpdef _forwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        '''
        Internally overloadable cython method for customizing self.forward.
        '''
        raise NotImplementedError("No _forwardC method implemented in class.")

    cpdef np.ndarray _forward(self, np.ndarray arrX):
        '''Internally overloadable method for customizing self.forward.'''
        cdef TRANSFORM xform
        if self._cythonCall:
            # Determine types, prepare input array and create output array
            # bypass dimension check in _prepareInputArray() call (already OK)
            arrX = self._prepareInputArray(arrX, 0, &xform)
            arrOutput = _arrEmpty(2, self.numRows, arrX.shape[1], xform.nOutput)
            self._forwardC(arrX, arrOutput, xform.fInput, xform.fOutput)
            return arrOutput
        else:
            return self._array.dot(arrX)

    cpdef np.ndarray forward(self, np.ndarray arrX):
        """Forward

        Calculate the forward transform A * x. Dimension-checking is performed
        to ensure valid fast transforms as these may succeed even when
        dimensions do not match. To support both single- and multidimensional
        input vectors x, single dimensional input will be reshaped to (n, 1)
        before processing and flattened to (n) after        completion. This
        allows the use of both vectors and arrays. The actual transform code
        gets called by the callbacks specified in funcPython and funcCython,
        depending on the state of self._cythonCall.

        .. warning::
            Do not override this method!

        .. note::
            The returned ndarray object may own its data, may be a view into
            another ndarray and may even be identical to the input array.

        Parameters
        ----------
        arrX : :py:class:`numpy.ndarray`
            The input data array of either 1d or 2d. 1d arrays will be
            reshaped to 2d during internal processing.

        Returns
        -------
            The result of the operation as :py:class:`np.ndarray` with the
            same number of dimensions as `arrX`.
        """
        # local variable holding return array
        cdef np.ndarray arrInput = arrX, arrOutput
        cdef int ndimInput = arrInput.ndim

        # Determine types, prepare input array and create output array
        cdef TRANSFORM xform
        arrInput = self._prepareInputArray(arrInput, self.numCols, &xform)

        # estimate runtimes according profile for Forward
        # if the dot-product bypass strategy leads to smaller runtimes, do it!
        if (self.bypassAllow and
                self.profileForward.isBypassFaster(xform.numVectors) and
                (self._array is not None or self.bypassAutoArray)):
            arrOutput = self.array.dot(arrInput)
        else:
            # call fast transform with either cython or python style
            if self._cythonCall:
                # Create output array
                arrOutput = _arrEmpty(
                    2, self.numRows, xform.numVectors if ndimInput > 1 else 1,
                    xform.nOutput
                )

                # Call calculation routine (fused type dispatch must be
                # be done locally in each class, typeInfo gets passed)
                self._forwardC(
                    arrInput, arrOutput, xform.fInternal, xform.fOutput
                )
            else:
                arrOutput = self._forward(arrInput)

        if ndimInput == 1:
            # reshape back to single-dimensional vector
            arrOutput = _arrReshape(
                arrOutput, 1, self.numRows, 1, np.NPY_ANYORDER)

        return arrOutput

    cpdef _backwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        '''
        Internally overloadable cython method for customizing self.backward.
        '''
        raise NotImplementedError("No _backwardC method implemented in class.")

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        '''Internally overloadable method for customizing self.backward.'''
        cdef TRANSFORM xform
        if self._cythonCall:
            # Determine types, prepare input array and create output array
            # bypass dimension check in _prepareInputArray() call (already OK)
            arrX = self._prepareInputArray(arrX, 0, &xform)
            arrOutput = _arrEmpty(2, self.numCols, arrX.shape[1], xform.nOutput)
            self._backwardC(arrX, arrOutput, xform.fInput, xform.fOutput)
            return arrOutput
        else:
            return _conjugate(self._array.T).dot(arrX)

    cpdef np.ndarray backward(self, np.ndarray arrX):
        r"""Backward Transform

        Calculate the backward transform A^\mathrm{H}*x where H is the
        hermitian transpose. Dimension-checking is performed to ensure valid
        fast transforms as these may succeed even when dimensions do not match.
        To support both single- and multidimensional input vectors x, single
        dimensional input will be reshaped to (n, 1) before processing and
        flattened to (n) after completion. This allows the use of both vectors
        and arrays. The actual transform code gets called by the callbacks
        specified in funcPython and funcCython, depending on the state of
        self._cythonCall.

        .. warning::
            Do not override this method

        .. note::
            The returned ndarray object may own its data, may be a view into
            another ndarray and may even be identical to the input array.

        Parameters
        ----------
        arrX : :py:class:`numpy.ndarray`
            The input data array of either 1d or 2d. 1d arrays will be
            reshaped to 2d during internal processing.

        Returns
        -------
            The result of the operation as :py:class:`np.ndarray` with the
            same number of dimensions as `arrX`.
        """
        # local variable holding return array
        cdef np.ndarray arrInput = arrX, arrOutput
        cdef int ndimInput = arrInput.ndim

        # Determine types, prepare input array and create output array
        cdef TRANSFORM xform
        arrInput = self._prepareInputArray(arrInput, self.numRows, &xform)

        # estimate runtimes according profile for Backward if the dot-product
        # bypass strategy leads to smaller runtimes, do it!
        if (self.bypassAllow and
                self.profileBackward.isBypassFaster(xform.numVectors) and
                (self._arrayH is not None or self.bypassAutoArray)):
            if self._arrayH is None:
                self._arrayH = self.array.T.conj()

            arrOutput = self._arrayH.dot(arrInput)
        else:
            # call fast transform with either cython or python style
            if self._cythonCall:
                # Create output array
                arrOutput = _arrEmpty(
                    2, self.numCols, xform.numVectors if ndimInput > 1 else 1,
                    xform.nOutput
                )

                # Call calculation routine (fused type dispatch must be
                # be done locally in each class, typeInfo gets passed)
                self._backwardC(
                    arrInput, arrOutput, xform.fInternal, xform.fOutput
                )
            else:
                arrOutput = self._backward(arrInput)

        if ndimInput == 1:
            # reshape back to single-dimensional vector
            arrOutput = _arrReshape(
                arrOutput, 1, self.numCols, 1, np.NPY_ANYORDER)

        return arrOutput

    ############################################## class reference
    cpdef np.ndarray reference(self):
        r"""
        Return explicit array reference of this matrix instance.

        Return an explicit representation of the matrix without using any
        fastmat code. Provides type checks and raises errors if the matrix
        type (self.dtype) cannot hold the reference data. This implementation
        is meant to provide a reference version for testing and MUST not use
        any fastmat code for its implementation.

        Returns
        -------
            The array representation of this matrix instance as 2d
            :py:class:`np.ndarray`.
        """
        cdef np.ndarray arrRes

        # self._reference() may be overwritten by a child class with either a
        # reference ndarray or a function returning one.

        # test if cython extension-type class method
        callables = [types.FunctionType, types.BuiltinFunctionType,
                     types.MethodType, types.BuiltinMethodType]

        # python2 offers extra unboundMethod distinction
        if sys.version_info < (3, 0):
            callables.append(types.UnboundMethodType)

        # Retrieve this reference array by checking for a ndarray, python or
        # cython-extension-type function
        objSelf = <object> self
        if isinstance(objSelf._reference, np.ndarray):
            arrRes = <np.ndarray> objSelf._reference
        elif isinstance(objSelf._reference, tuple(callables)):
            arrRes = objSelf._reference()
        else:
            try:
                # test if python method
                getattr(objSelf, '__call__')
                arrRes = objSelf._reference()
            except AttributeError:
                arrRes = <np.ndarray> objSelf._reference

        return arrRes

    cpdef np.ndarray _reference(self):
        '''Internally overloadable method for customizing self.reference.'''
        return self._array

    def _forwardReferenceInit(self):
        self._forwardReferenceMatrix = self.reference()

    def _forwardReference(self, arrX):
        if self._forwardReferenceMatrix is None:
            self._forwardReferenceInit()

        return self._forwardReferenceMatrix.dot(arrX)

    ############################################## class inspection, QM
    def _getTest(self):
        '''Return unit test configuration for this matrix class.'''
        from .inspect import TEST, dynFormat
        if self.__class__ == Matrix:
            # Test code for Matrix base class
            return {
                TEST.COMMON: {
                    TEST.NUM_ROWS   : 35,
                    TEST.NUM_COLS   : TEST.Permutation([30, TEST.NUM_ROWS]),
                    'mType'         : TEST.Permutation(TEST.ALLTYPES),
                    TEST.PARAMALIGN : TEST.Permutation(TEST.ALLALIGNMENTS),
                    'arrM'          : TEST.ArrayGenerator({
                        TEST.DTYPE  : 'mType',
                        TEST.SHAPE  : (TEST.NUM_ROWS, TEST.NUM_COLS),
                        TEST.ALIGN  : TEST.PARAMALIGN
                    }),
                    TEST.OBJECT     : Matrix,
                    TEST.INITARGS   : (lambda param: [param['arrM']()]),
                    TEST.NAMINGARGS : dynFormat("%s", 'arrM')
                },
                TEST.CLASS: {},
                TEST.TRANSFORMS: {}
            }
        elif isinstance(
            self, (Hermitian, Conjugate, Transpose)
        ):
            # Test code for the three Transposition classes that are also
            # defined in this submodule. As the Transpositions are directly
            # derived from the Matrix base class we can put the relevant code
            # directly in here and adapt it to work for all three classes.

            # Note that the resulting matrix shape differs for Hermitian and
            # Transpose so we must reflect this when specifying the shape of the
            # to-be-generated data array. As Permutations of a dimension is not
            # uncommon the corresponding fields must be linked in DATASHAPE
            numRows = 35
            numCols = 30
            swap = isinstance(self, (Conjugate))
            return {
                TEST.COMMON: {
                    TEST.NUM_ROWS   : numRows,
                    TEST.NUM_COLS   : TEST.Permutation(
                        [numCols, TEST.NUM_ROWS]
                    ),
                    'mType'         : TEST.Permutation(TEST.FLOATTYPES),
                    TEST.PARAMALIGN : TEST.Permutation(TEST.ALLALIGNMENTS),
                    'arrM'          : TEST.ArrayGenerator({
                        TEST.DTYPE  : 'mType',
                        TEST.SHAPE  : (TEST.NUM_ROWS, TEST.NUM_COLS),
                        TEST.ALIGN  : TEST.PARAMALIGN
                    }),
                    TEST.DATASHAPE  : (TEST.NUM_COLS if swap else TEST.NUM_ROWS,
                                       TEST.DATACOLS),
                    TEST.DATASHAPE_T: (TEST.NUM_ROWS if swap else TEST.NUM_COLS,
                                       TEST.DATACOLS),
                    TEST.OBJECT     : self.__class__,
                    TEST.INITARGS   : (lambda param: [Matrix(param['arrM']())]),
                    TEST.NAMINGARGS : dynFormat("%s", 'arrM')
                },
                TEST.CLASS: {},
                TEST.TRANSFORMS: {}
            }
        elif isinstance(
            self, (Inverse, PseudoInverse)
        ):
            # Test code for the two Inversion classes that are also
            # defined in this submodule.
            numRows = 10
            numCols1 = 9
            numCols2 = 11
            square = isinstance(self, (Inverse))
            return {
                TEST.COMMON: {
                    TEST.NUM_ROWS   : numRows,
                    TEST.NUM_COLS   : (TEST.NUM_ROWS if square else
                                       TEST.Permutation(
                                           [numCols1, numCols2, TEST.NUM_ROWS]
                                       )),
                    'mType'         : TEST.Permutation(TEST.FLOATTYPES),
                    TEST.PARAMALIGN : TEST.Permutation(TEST.ALLALIGNMENTS),
                    'arrM'          : TEST.ArrayGenerator({
                        TEST.DTYPE  : 'mType',
                        TEST.SHAPE  : (TEST.NUM_ROWS, TEST.NUM_COLS),
                        TEST.ALIGN  : TEST.PARAMALIGN
                    }),
                    TEST.DATASHAPE  : (TEST.NUM_ROWS, TEST.DATACOLS),
                    TEST.DATASHAPE_T: (TEST.NUM_COLS, TEST.DATACOLS),
                    TEST.OBJECT     : self.__class__,
                    TEST.INITARGS   : (lambda param: [Matrix(param['arrM']())]),
                    TEST.NAMINGARGS : dynFormat("%s", 'arrM'),
                    TEST.TOL_POWER  : 10.0,
                    TEST.TOL_MINEPS    : 1e-2,
                    TEST.CHECK_DATATYPE: False
                },
                TEST.CLASS: {},
                TEST.TRANSFORMS: {}
            }
        else:
            # Any other class should go and define its own tests!
            return {}

    def _getBenchmark(self):
        '''Return benchmark configuration for this matrix class.'''
        from .inspect import BENCH
        if self.__class__ == Matrix:
            # Benchmark code for Matrix base class
            return {
                BENCH.FORWARD: {
                    BENCH.FUNC_GEN :
                        (lambda c: Matrix(np.zeros((c, c)))),
                },
                BENCH.OVERHEAD: {
                    BENCH.FUNC_GEN  :
                        (lambda c: Matrix(np.zeros((2 ** c, 2 ** c)))),
                    BENCH.FUNC_SIZE : (lambda c: 2 ** c)
                },
                BENCH.DTYPES: {
                    BENCH.FUNC_GEN  :
                        (lambda c, dt: Matrix(np.zeros((2 ** c, 2 ** c),
                                                       dtype=dt)))
                }
            }
        elif isinstance(
            self, (Hermitian, Conjugate, Transpose, Inverse, PseudoInverse)
        ):
            # Benchmark code for the three Transposition classes that are also
            # defined in this submodule. As the Transpositions are directly
            # derived from the Matrix base class we can put the relevant code
            # directly in here and adapt it to work for all three classes.
            # Also note, that we can use the Eye again as the property override
            # of Eye.{T,conj,H} does not grip if instantiated directly through
            # the class constructor
            from .Eye import Eye
            return {
                BENCH.OVERHEAD: {
                    BENCH.FUNC_GEN  :
                        (lambda c: self.__class__(Eye(2 ** c) * 1j)),
                    BENCH.FUNC_SIZE : (lambda c: 2 ** c)
                }
            }
        else:
            # Any other class should go and define its own benchmarks!
            return {}


################################################################################
cdef class Hermitian(Matrix):
    r""" Hermitian Transpose of a Matrix

    """
    ############################################## class methods

    def __init__(self, Matrix matrix):
        '''
        Initialize an instance of a hermitian transposed matrix.

        Parameters
        ----------
        matrix : :py:class:`fastmat.Matrix`
            The matrix instance to be transposed.
        '''
        if not isinstance(matrix, Matrix):
            raise TypeError("Hermitian: Not a fastmat Matrix")

        self._nested = matrix
        self._content = (matrix, )
        self._cythonCall = matrix._cythonCall
        self._initProperties(
            matrix.shape[1], matrix.shape[0], matrix.dtype,
            **matrix._getProperties()
        )

    def __repr__(self):
        return "<%s.H>" %(self._nested.__repr__())

    ############################################## class property override
    cpdef np.ndarray _getArray(self):
        return self._nested._getArray().T.conj()

    cpdef np.ndarray _getCol(self, intsize idx):
        return _conjugate(self._nested._getRow(idx))

    cpdef np.ndarray _getRow(self, intsize idx):
        return _conjugate(self._nested._getCol(idx))

    cpdef object _getItem(self, intsize idxRow, intsize idxCol):
        return np.conjugate(self._nested._getItem(idxCol, idxRow))

    cpdef object _getLargestEigenValue(self):
        return self._nested.largestEigenValue

    cpdef object _getLargestSingularValue(self):
        return self._nested.largestSingularValue

    cpdef Matrix _getT(self):
        return getConjugate(self._nested)

    cpdef Matrix _getH(self):
        return self._nested

    cpdef Matrix _getConj(self):
        return Transpose(self._nested)

    ############################################## class performance estimation
    cpdef tuple _getComplexity(self):
        return (self.numCols + self.numRows, self.numRows + self.numCols)

    cpdef _forwardC(self, np.ndarray arrX, np.ndarray arrRes,
                    ftype typeX, ftype typeRes):
        self._nested._backwardC(arrX, arrRes, typeX, typeRes)

    cpdef _backwardC(self, np.ndarray arrX, np.ndarray arrRes,
                     ftype typeX, ftype typeRes):
        self._nested._forwardC(arrX, arrRes, typeX, typeRes)

    cpdef np.ndarray _forward(self, np.ndarray arrX):
        return self._nested._backward(arrX)

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        return self._nested._forward(arrX)

    cpdef np.ndarray _reference(self):
        return self._nested.reference().T.conj()


################################################################################
cdef inline Matrix getConjugate(Matrix matrix):
    """
    Return the conjugate of matrix.

    Parameters
    ----------
    matrix : :py:class:`fastmat.Matrix`
        The matrix to return the conjugate of.

    Returns
    -------
        If `matrix` is not a :py:class:`fastmat.Conjuagate`, a
        :py:class:`fastmat.Conjugate` object of `matrix`is returned. Otherwise
        the base matrix of the `Conjugate` object is returned to avoid multiple
        serial conjugations.
    """
    return (Conjugate(matrix)
            if typeInfo[matrix.fusedType].isComplex
            else matrix)


################################################################################
cdef class Conjugate(Matrix):
    r""" Conjugate of a Matrix

    """
    ############################################## class methods
    def __init__(self, Matrix matrix):
        '''
        Initialize an instance of a conjugated matrix.

        Parameters
        ----------
        matrix : :py:class:`fastmat.Matrix`
            The matrix instance to be conjugated.
        '''
        if not isinstance(matrix, Matrix):
            raise TypeError("Conjugate: Not a fastmat Matrix")

        self._nested = matrix
        self._content = (matrix, )
        self._cythonCall = matrix._cythonCall
        self._initProperties(
            matrix.shape[0], matrix.shape[1], matrix.dtype,
            **matrix._getProperties()
        )

    def __repr__(self):
        return "<conj(%s)>" %(self._nested.__repr__())

    ############################################## class property override
    cpdef np.ndarray _getArray(self):
        return self._nested._getArray().conj()

    cpdef np.ndarray _getCol(self, intsize idx):
        return _conjugate(self._nested._getCol(idx))

    cpdef np.ndarray _getRow(self, intsize idx):
        return _conjugate(self._nested._getRow(idx))

    cpdef object _getItem(self, intsize idxRow, intsize idxCol):
        return np.conjugate(self._nested._getItem(idxRow, idxCol))

    cpdef object _getLargestEigenValue(self):
        return self._nested.largestEigenValue

    cpdef object _getLargestSingularValue(self):
        return self._nested.largestSingularValue

    cpdef Matrix _getT(self):
        return Hermitian(self._nested)

    cpdef Matrix _getH(self):
        return Transpose(self._nested)

    cpdef Matrix _getConj(self):
        return self._nested

    ############################################## class performance estimation
    cpdef tuple _getComplexity(self):
        cdef float complexity = self.numRows + self.numCols
        return (complexity, complexity)

    ############################################## class forward / backward
    cpdef _forwardC(self, np.ndarray arrX, np.ndarray arrRes,
                    ftype typeX, ftype typeRes):
        cdef np.ndarray arrInput = _conjugate(arrX)
        self._nested._forwardC(arrInput, arrRes, typeX, typeRes)
        _conjugateInplace(arrRes)

    cpdef _backwardC(self, np.ndarray arrX, np.ndarray arrRes,
                     ftype typeX, ftype typeRes):
        cdef np.ndarray arrInput = _conjugate(arrX)
        self._nested._backwardC(arrInput, arrRes, typeX, typeRes)
        _conjugateInplace(arrRes)

    cpdef np.ndarray _forward(self, np.ndarray arrX):
        cdef np.ndarray arrRes = self._nested._forward(_conjugate(arrX))
        _conjugateInplace(arrRes)
        return arrRes

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        cdef np.ndarray arrRes = self._nested._backward(_conjugate(arrX))
        _conjugateInplace(arrRes)
        return arrRes

    ########################################## references: test / benchmark
    cpdef np.ndarray _reference(self):
        return self._nested._reference().conj()


################################################################################
################################################## Transpose class
cdef class Transpose(Hermitian):
    r"""Transpose of a Matrix

    """
    ############################################## class methods

    def __init__(self, Matrix matrix):
        '''
        Initialize an instance of a transposed matrix.

        Parameters
        ----------
        matrix : :py:class:`fastmat.Matrix`
            The matrix instance to be transposed.
        '''
        if not isinstance(matrix, Matrix):
            raise TypeError("Transpose: Not a fastmat Matrix")

        # NOTE: Transpose is implemented as a Hermitian of a Conjugated Matrix.
        # Thus the _nested field points to the Conjugate of the actual base
        # matrix. For working on the true base matrix, _nestedConj must be
        # accessed instead of _nested like in Hermitian or Conjugated.
        self._nestedConj = matrix
        matrix = getConjugate(matrix)

        self._nested = matrix
        self._content = (matrix, )
        self._initProperties(
            matrix.shape[1], matrix.shape[0], matrix.dtype,
            **matrix._getProperties()
        )

    def __repr__(self):
        return "<%s.T>" %(self._nestedConj.__repr__())

    ############################################## class property override
    cpdef np.ndarray _getArray(self):
        return self._nestedConj._getArray().T

    cpdef np.ndarray _getCol(self, intsize idx):
        return self._nestedConj._getRow(idx)

    cpdef np.ndarray _getRow(self, intsize idx):
        return self._nestedConj._getCol(idx)

    cpdef object _getItem(self, intsize idxRow, intsize idxCol):
        return self._nestedConj._getItem(idxCol, idxRow)

    cpdef object _getLargestEigenValue(self):
        return self._nestedConj.largestEigenValue

    cpdef object _getLargestSingularValue(self):
        return self._nestedConj.largestSingularValue

    cpdef Matrix _getT(self):
        return self._nestedConj

    cpdef Matrix _getH(self):
        return getConjugate(self._nestedConj)

    cpdef Matrix _getConj(self):
        return Hermitian(self._nestedConj)

    ########################################## references: test / benchmark
    cpdef np.ndarray _reference(self):
        return self._nestedConj._reference().T


cdef class Inverse(Matrix):
    r""" Inverse of a Matrix

    This class is implemented by always solving a system of linear equations
    in order to act out the forward transform of a given matrix.
    """
    ############################################## class methods

    def __init__(self, Matrix matrix):
        '''
        Initialize an instance of an inverted matrix.

        Parameters
        ----------
        matrix : :py:class:`fastmat.Matrix`
            The matrix instance to be inverse.
        '''
        from scipy.sparse.linalg import lgmres

        if not isinstance(matrix, Matrix):
            raise TypeError("Inverse: Not a fastmat Matrix")

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Inverse: Matrix not square, so not invertible.")

        self._nested = matrix
        self._content = (matrix, )
        self._cythonCall = False
        self._initProperties(
            matrix.shape[0],
            matrix.shape[0],
            np.promote_types(matrix.dtype, np.float32),
            **matrix._getProperties()
        )
        self._linearOperator = matrix.scipyLinearOperator
        self._solver = lgmres

    cpdef np.ndarray _solveForward(self, np.ndarray arrX):
        return self._solver(self._linearOperator, arrX, atol=1e-12)[0]

    cpdef np.ndarray _solveBackward(self, np.ndarray arrX):
        return self._solver(self._linearOperator.H, arrX, atol=1e-12)[0]

    def __repr__(self):
        return "<%s.(^-1)>" %(self._nested.__repr__())

    cpdef np.ndarray _forward(self, np.ndarray arrX):
        return np.apply_along_axis(
            self._solveForward, 0, arrX
        )

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        return np.apply_along_axis(
            self._solveBackward, 0, arrX
        )

    cpdef np.ndarray _reference(self):
        import numpy.linalg as npl
        return npl.inv(self._nested.reference())


cdef class PseudoInverse(Matrix):
    r""" Inverse of a Matrix

    This class is implemented by always solving a system of linear equations
    in order to act out the forward transform of a given matrix.
    """
    ############################################## class methods

    def __init__(self, Matrix matrix):
        '''
        Initialize an instance of a pseudo inverse matrix.

        Parameters
        ----------
        matrix : :py:class:`fastmat.Matrix`
            The matrix instance that we want the pseudo inverse of.
        '''
        from scipy.sparse.linalg import lsqr

        if not isinstance(matrix, Matrix):
            raise TypeError("Inverse: Not a fastmat Matrix")

        self._nested = matrix
        self._content = (matrix, )
        self._cythonCall = False
        self._initProperties(
            matrix.shape[1],
            matrix.shape[0],
            np.promote_types(matrix.dtype, np.float32),
            **matrix._getProperties()
        )
        self._linearOperator = matrix.scipyLinearOperator
        self._solver = lsqr

    cpdef np.ndarray _solveForward(self, np.ndarray arrX):
        return self._solver(self._linearOperator, arrX, atol=1e-12)[0]

    cpdef np.ndarray _solveBackward(self, np.ndarray arrX):
        return self._solver(self._linearOperator.H, arrX, atol=1e-12)[0]

    def __repr__(self):
        return "<%s.(^+)>" %(self._nested.__repr__())

    cpdef np.ndarray _forward(self, np.ndarray arrX):
        return np.apply_along_axis(
            self._solveForward, 0, arrX
        )

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        return np.apply_along_axis(
            self._solveBackward, 0, arrX
        )

    cpdef np.ndarray _reference(self):
        import numpy.linalg as npl
        return npl.pinv(self._nested.reference())
