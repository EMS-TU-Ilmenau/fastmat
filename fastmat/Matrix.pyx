# -*- coding: utf-8 -*-

# Copyright 2016 Sebastian Semper, Christoph Wagner
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
from .core.calibration import getMatrixCalibration
#_conjugate, _arrEmpty, _arrReshape, _arrForceContType
from .Product cimport Product
from .Sum cimport Sum
from .Diag cimport Diag

# have a very lazy import to avoid various package imports during main init
getMemoryFootprint = None

################################################################################
################################################## class FastmatFlags
cdef class FastmatFlags(object):
    def __init__(self):
        self.bypassAllow = True
        self.bypassAutoArray = True

flags = FastmatFlags()

################################################################################
################################################## class MatrixCalibration
cdef class MatrixCalibration(object):
    def __init__(self, offsetForward, offsetBackward,
                 gainForward, gainBackward):
        self.gainForward = gainForward
        self.gainBackward = gainBackward
        self.offsetForward = offsetForward
        self.offsetBackward = offsetBackward

    def export(self):
        return (self.offsetForward, self.offsetBackward,
                self.gainForward, self.gainBackward)

    def __repr__(self):
        return str(self.export())


cdef tuple profileToTuple(PROFILE_s profile):
    return (profile.overhead,
            profile.effort,
            profile.overheadNested,
            profile.effortNested,
            profile.complexity)

cdef void finishProfile(PROFILE_s *profile, float offset, float gain):
    profile[0].overhead = profile[0].overheadNested + offset
    profile[0].effort   = profile[0].effortNested + gain * profile[0].complexity

cdef bint profileIsValid(PROFILE_s *profile):
    return (np.isfinite(profile[0].overhead) and profile[0].overhead > 0 and
            np.isfinite(profile[0].effort) and profile[0].effort > 0)

cdef bint profileUpdate(PROFILE_s *profile, intsize numM, bint allowBypass,
                        PROFILE_s *profClass, PROFILE_s *profBypass):
    cdef float estimateClass, estimateBypass
    cdef PROFILE_s *profSelected

    estimateClass  = profClass[0].overhead + profClass[0].effort * numM
    estimateBypass = profBypass[0].overhead + profBypass[0].effort * numM

    profSelected = (profBypass if (allowBypass and
                                   estimateBypass < estimateClass)
                    else profClass)
    profile[0].overheadNested += profSelected[0].overhead
    profile[0].effortNested   += profSelected[0].effort * numM

    return profSelected == profClass

################################################################################
################################################## class Matrix
cdef class Matrix(object):
    r"""Matrix Base Class

    **Description:**
    The baseclass of all matrix classes in fastmat. It also serves as wrapper
    around the standard Numpy Array [1]_.

    **Performance Benchmarks**

    .. figure:: _static/bench/Matrix.overhead.png
        :width: 50%

        Overhead Benchmarking of this class

    .. figure:: _static/bench/Matrix.dtypes.png
        :width: 50%

        Dtypes Benchmarking of this class

    .. figure:: _static/bench/Matrix.forward.png
        :width: 50%

        Forward Multiplication numbers

    .. todo::
        - extend docu
        - add caches for cols / rows
        - implement generation of normalized matrix of general matrices

    """

    ############################################## basic class properties
    property dtype:
        # r"""Report the matrix elements' data type
        #
        # *(read-only)*
        # """
        def __get__(self):
            return np.PyArray_TypeObjectFromType(self._info.dtype.typeNum)

    property dtypeNum:
        # r"""Report the matrix elements data type
        #
        # (fastmat fused type number)
        #
        # *(read-only)*
        # """
        def __get__(self):
            return self._info.dtype[0].fusedType

    property numN:
        # r"""Report the matrix row count
        #
        # *(read-only)*
        # """
        def __get__(self):
            return self._info.numN

    property numM:
        # r"""Report the matrix column count
        #
        # *(read-only)*
        # """
        def __get__(self):
            return self._info.numM

    property shape:
        # r"""Report the matrix shape as tuple of (rows, columns)
        #
        # *(read-only)*
        # """
        def __get__(self):
            return (self.numN, self.numM)

    property tag:
        # r"""Description tag of Matrix
        #
        # *(read-write)*
        # """
        def __get__(self):
            return (self._tag)

        def __set__(self, tag):
            self._tag = tag

    property bypassAllow:
        # r"""Enable or disable transform bypassing based on performance
        # estimates
        #
        # *(read-write)*
        # """
        def __get__(self):
            return self._bypassAllow

        def __set__(self, value):
            self._bypassAllow = value

    property bypassAutoArray:
        # r"""Enable or disable automatic dense array generation for
        # transform bypass
        #
        # *(read-write)*
        # """
        def __get__(self):
            return self._bypassAutoArray

        def __set__(self, value):
            self._bypassAutoArray = value

    ############################################## class resource handling
    # nbytes - Property(read)
    # Size of the Matrix object
    property nbytes:
        # r"""Number of bytes
        # """
        def __get__(self):
            global getMemoryFootprint
            if getMemoryFootprint is None:
                getMemoryFootprint = __import__(
                    'fastmat.core.resource', globals(), locals(),
                    ['getMemoryFootprint']).getMemoryFootprint

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
        if self.__class__ is not Matrix:
            self._array = self._getArray()

        return self._array

    cpdef np.ndarray _getArray(self):
        # r"""
        # """
        # check whether this instance is a Matrix or some child class
        if self.__class__ == Matrix:
            # it is a Matrix, so return our internal data
            return self._array
        else:
            # it is any child class. Use self.forward() to generate a ndarray
            # representation of this matrix. This calculation is very slow but
            # also very general. Therefore it is used also in child classes
            # when no explicit code for self._getArray() is provided.
            return self._forward(np.eye(self.numM, dtype=self.dtype))

    def getCols(self, indices):
        r"""

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
                arrResult = _arrEmpty(2, self._info.numN, numSize,
                                      self._info.dtype[0].typeNum)

                for ii in range(numSize):
                    arrResult[:, ii] = self.getCol(arrIdx[ii])

        return arrResult

    def getCol(self, idx):
        r"""

        """
        if idx < 0 or idx >= self.numM:
            raise ValueError("Column index exceeds matrix dimensions.")

        # if a dense representation already exists, use it!
        if self._array is not None:
            return np.squeeze(self._array[:, idx])

        return self._getCol(idx)

    cpdef np.ndarray _getCol(self, intsize idx):
        r"""

        """
        cdef np.ndarray arrData = _arrZero(1, self._info.numM, 1,
                                           self._info.dtype[0].typeNum)
        arrData[idx] = 1
        return self.forward(arrData)

    def getRows(self, indices):
        r"""

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
                arrResult = _arrEmpty(2, numSize, self._info.numM,
                                      self._info.dtype[0].typeNum)

                for ii in range(numSize):
                    arrResult[ii, :] = self.getRow(arrIdx[ii])

        return arrResult

    def getRow(self, idx):
        r"""

        """
        if idx < 0 or idx >= self.numN:
            raise ValueError("Row index exceeds matrix dimensions.")

        # if a dense representation already exists, use it!
        if self._array is not None:
            return np.squeeze(self._array[idx, :])

        return self._getRow(idx)

    cpdef np.ndarray _getRow(self, intsize idx):
        r"""

        """
        cdef np.ndarray arrData = _arrZero(1, self._info.numN, 1,
                                           self._info.dtype[0].typeNum)
        arrData[idx] = 1
        return self.backward(arrData).conj()

    def __getitem__(self, tplIdx):
        r"""

        """
        if len(tplIdx) != 2:
            raise ValueError("Matrix element access requires two indices.")

        cdef intsize idxN = tplIdx[0], idxM = tplIdx[1]
        cdef intsize N = self._info.numN, M = self._info.numM

        if idxN < 0 or idxN >= N or idxM < 0 or idxM >= M:
            raise IndexError("Index %s exceeds matrix dimensions %s." %(
                str(tplIdx), str(self.shape)))

        # if a dense representation already exists, use it!
        if self._array is not None:
            return self._array[idxN, idxM]

        return self._getItem(idxN, idxM)

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        return self._getCol(idxM)[idxN]

    ############################################## Matrix content property
    property content:
        # r"""Return the Matrix contents
        #
        # *(read-only)*
        # """
        def __get__(self):
            return self._content

    ############################################## algorithmic properties
    property largestEV:
        # r"""Return the largest eigenvalue for this matrix instance
        #
        # *(read-only)*
        # """
        def __get__(self):
            return (self.getLargestEV() if self._largestEV is None
                    else self._largestEV)

    def getLargestEV(self, maxSteps=10000,
                     relEps=1., eps=0., alwaysReturn=False):
        # r"""
        # Largest Singular Value
        #
        # For a given matrix :math:`A \in \mathbb{C}^{n \times n}`, so :math:`A`
        # square, we calculate the absolute value of the largest eigenvalue
        # :math:`\lambda \in \mathbb{C}`. The eigenvalues obey the equation
        #
        # .. math::
        #      A \cdot  v= \lambda \cdot  v,
        #
        # where :math:`v` is a non-zero vector.
        #
        # Input matrix :math:`A`, parameter :math:`0 < \varepsilon \ll 1` as a
        # stopping criterion
        # Output largest eigenvalue :math:`\sigma_{\rm max}( A)`
        #
        # .. note::
        #     This algorithm performs well if the two largest eigenvalues are
        #     not very close to each other on a relative scale with respect to
        #     their absolute value. Otherwise it might get trouble converging
        #     properly.
        #
        # >>> # import the packages
        # >>> import numpy.linalg as npl
        # >>> import numpy as np
        # >>> import fastmat as fm
        # >>>
        # >>> # define the matrices
        # >>> n = 5
        # >>> H = fm.Hadamard(n)
        # >>> D = fm.Diag(np.linspace
        # >>>         1, 2 ** n, 2 ** n))
        # >>>
        # >>> K1 = fm.Product(H, D)
        # >>> K2 = K1.array
        # >>>
        # >>> # calculate the eigenvalue
        # >>> x1 = K1.largestEV
        # >>> x2 = npl.eigvals(K2)
        # >>> x2 = np.sort(np.abs(x2))[-1]
        # >>>
        # >>> # check if the solutions match
        # >>> print(x1 - x2)
        #
        # We define a matrix-matrix product of a Hadamard matrix and a diagonal
        # matrix. Then we also cast it into a numpy-array and use the integrated
        # EVD. For demonstration, try to increase :math:`n`to :math:`>10`and see
        # what happens.
        # """

        if self._numN != self._numM:
            raise ValueError("largestEV: Matrix must be square.")

        result = self._getLargestEV(maxSteps, relEps, eps, alwaysReturn)
        self._largestEV = self._largestEV if np.isnan(result) else result
        return result

    cpdef object _getLargestEV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        # r"""Determine largest eigen-value of a fastmat matrix.
        # """

        # The following variables are used:
        # eps            - relative stopping threshold
        # numNormOld     - norm of vector from last iteration
        # numNormNew     - norm of vector in current iteration
        # vecBNew        - normalized current iterate
        # vecBOld        - normalized last iterate

        cdef np.ndarray vecBOld, vecBNew, vecBNewHat

        # determine an convergance threshold if eps is deliberately set to zero
        if eps == 0:
            eps = relEps * _getTypeEps(safeTypeExpansion(self.dtype)) * \
                (self.numN if self.numN >= self.numM else self.numM)

        if self.numN != self.numM:
            raise ValueError("largestEV: Matrix must be square.")

        # sample one point uniformly in space, have a zero-vector reference
        vecBNew = np.random.randn(self.numM, 1).astype(
            np.promote_types(np.float32, self.dtype))
        vecBNew /= np.linalg.norm(vecBNew)
        vecBOld = np.zeros((<object> vecBNew).shape, vecBNew.dtype)

        # now continiously apply the matrix and renormalize until convergence
        for numSteps in range(maxSteps):
            vecBOld = vecBNew
            vecBNew = self.forward(vecBOld)

            normNew = np.linalg.norm(vecBNew)
            if normNew == 0:
                # presumably a zero-matrix
                return vecBNew.dtype.type(0.)

            vecBNew /= normNew

            if np.linalg.norm(vecBNew - vecBOld) < eps:
                vecBNewHat = vecBNew.conj()
                return (np.inner(vecBNewHat, self.forward(vecBNew)) /
                        np.inner(vecBNewHat, vecBNew))

        # did not converge - return NaN
        if alwaysReturn:
            vecBNewHat = vecBNew.conj()
            return (np.inner(vecBNewHat, self.forward(vecBNew)) /
                    np.inner(vecBNewHat, vecBNew))
        else:
            return vecBNew.dtype.type(np.NaN)

    property largestSV:
        # r"""Return the largestSV for this matrix instance
        #
        # *(read-only)*
        # """
        def __get__(self):
            return (self.getLargestSV() if self._largestSV is None
                    else self._largestSV)

    def getLargestSV(self, maxSteps=10000,
                     relEps=1., eps=0., alwaysReturn=False):
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

        - Input matrix :math:`A`, parameter :math:`0 < \varepsilon \ll 1` as a
        stopping criterion
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
        >>> x1 = largestSV(K1.largestSV
        >>> x2 = npl.svd(K2,compute_uv
        >>> # check if they match
        >>> print(x1-x2)

        We define a Kronecker product of a Hadamard matrix and a Fourier
        matrix. Then we also cast it into a numpy-array and use the integrated
        SVD. For demonstration, try to increase :math:`n` to `>10` and see what
        happens.
        """

        result = self._getLargestSV(maxSteps, relEps, eps, alwaysReturn)
        self._largestSV = self._largestSV if np.isnan(result) else result
        return result

    cpdef object _getLargestSV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        # r"""
        #
        # """
        cdef np.ndarray vecBOld, vecBNew
        cdef Matrix matGram = self.gram
        cdef intsize ii

        # determine an convergance threshold if eps is deliberately set to zero
        if eps == 0:
            eps = relEps * _getTypeEps(safeTypeExpansion(self.dtype)) * \
                (self.numN if self.numN >= self.numM else self.numM)

        # sample one initial sample, have a zero-vector reference
        vecB = np.random.randn(self.numM, 1).astype(
            np.promote_types(np.float64, self.dtype))
        normNew = np.linalg.norm(vecB)

        # iterate until changes cool down
        for ii in range(maxSteps):
            vecB = matGram.forward(vecB / normNew)

            normOld = normNew
            normNew = np.linalg.norm(vecB)
            if normNew == 0:
                # presumably a zero-matrix
                return vecB.dtype.type(0.)

            if np.abs(normNew - normOld) < eps:
                # return square root of the current norm after applying the gram
                # matrix
                return np.sqrt(normNew)

        # did not converge - return NaN
        return (np.sqrt(normNew) if alwaysReturn else np.float64(np.NaN))

    ############################################## generic algebraic properties
    property gram:
        r"""Return the gram matrix for this fastmat class

        *(read-only)*
        """

        def __get__(self):
            return (self.getGram() if self._gram is None
                    else self._gram)

    def getGram(self):
        r"""

        """
        self._gram = self._getGram()
        return self._gram

    cpdef Matrix _getGram(self):
        return Product(self.H, self)

    property normalized:
        r"""Return a normalized matrix for this instance

        *(read-only)*
        """

        def __get__(self):
            return (self.getNormalized() if self._normalized is None
                    else self._normalized)

    def getNormalized(self):
        r"""

        """
        self._normalized = self._getNormalized()
        return self._normalized

    cpdef Matrix _getNormalized(self):
        # determine type of normalization diagonal matrix
        diagType = safeTypeExpansion(self.dtype)

        # array that contains the norms of each column
        cdef np.ndarray arrDiag = np.empty(self.numM, dtype=diagType)

        # number of elements we consider at once during normalization
        cdef intsize numStrideSize = 256

        cdef np.ndarray arrSelector = np.zeros(
            (self.numM, min(numStrideSize, self.numM)), dtype=diagType)

        cdef int ii
        for ii in range(0, self.numM, numStrideSize):
            vecIndices = np.arange(ii, min(ii + numStrideSize, self.numM))
            if ii == 0 or vecSlice.size != vecIndices.size:
                vecSlice = np.arange(0, vecIndices.size)

            arrSelector[vecIndices, vecSlice] = 1

            arrDiag[vecIndices] = np.linalg.norm(
                self.forward(
                    arrSelector if vecIndices.size == arrSelector.shape[1]
                    else arrSelector[:, vecSlice]),
                axis=0)
            arrSelector[vecIndices, vecSlice] = 0

        # check if we've found any zero
        if np.any(arrDiag == 0):
            raise ValueError("Normalization: Matrix has zero-norm column.")

        # finally invert the diagonal and generate normalized matrix
        return self * Diag(1. / arrDiag)

    property T:
        r"""Return the transpose of the matrix as fastmat class

        *(read-only)*
        """

        def __get__(self):
            return (self.getT() if self._T is None else self._T)

    def getT(self):
        r"""

        """
        self._T = self._getT()
        return self._T

    cpdef Matrix _getT(self):
        return Transpose(self)

    property H:
        r"""Return the hermitian transpose

        *(read-only)*
        """

        def __get__(self):
            return (self.getH() if self._H is None else self._H)

    def getH(self):
        r"""

        """
        self._H = self._getH()
        return self._H

    cpdef Matrix _getH(self):
        return Hermitian(self)

    property conj:
        r"""Return the conjugate of the matrix as fastmat class

        *(read-only)*
        """

        def __get__(self):
            return (self.getConj() if self._conj is None else self._conj)

    def getConj(self):
        r"""

        """
        self._conj = self._getConj()
        return self._conj

    cpdef Matrix _getConj(self):
        return getConjugate(self)

    ############################################## computation complexity
    property complexity:
        r"""Complexity

        *(read-only)*

        Return the computational complexity of all functionality implemented in
        the class itself, not including calls of external code.
        """

        def __get__(self):
            return (self._profileForward.complexity,
                    self._profileBackward.complexity)

    def getComplexity(self):
        r"""

        """
        cdef tuple complexity = self._getComplexity()
        assert complexity is not None
        self._profileForward.complexity = complexity[0]
        self._profileBackward.complexity = complexity[1]

    cpdef tuple _getComplexity(self):
        cdef float complexity = self.numN * self.numM
        return (complexity, complexity)

    property profile:
        r"""

        *(read-only)*

        Return the timing profiles of this specific class. These are also used
        to determine which computation strategy to use for a perticular input.
        (distinct amount of vectors)
        """

        def __get__(self):
            return (profileToTuple(self._profileForward),
                    profileToTuple(self._profileBackward))

    property profileForward:
        r"""

        """

        def __get__(self):
            return self._profileForward

    property profileBackward:
        r"""

        """

        def __get__(self):
            return self._profileBackward

    property profileBypassFwd:
        r"""

        """

        def __get__(self):
            return self._profileBypassFwd

    property profileBypassBwd:
        r"""

        """

        def __get__(self):
            return self._profileBypassBwd

    def estimateRuntime(self, intsize M=1):
        r"""Estimate Runtime

        """
        cdef float estimateForward  = (self._profileForward.overhead +
                                       self._profileForward.effort * M)
        cdef float estimateBypassFwd = (self._profileBypassFwd.overhead +
                                        self._profileBypassFwd.effort * M)
        cdef float estimateBackward = (self._profileBackward.overhead +
                                       self._profileBackward.effort * M)
        cdef float estimateBypassBwd = (self._profileBypassBwd.overhead +
                                        self._profileBypassBwd.effort * M)
        return (estimateBypassFwd if (self._bypassAllow and
                                      (self._array is not None or
                                       self._bypassAutoArray) and
                                      (estimateBypassFwd < estimateForward))
                else estimateForward,
                estimateBypassBwd if (self._bypassAllow and
                                      (self._arrayH is not None or
                                       self._bypassAutoArray) and
                                      (estimateBypassBwd < estimateBackward))
                else estimateBackward)

    cdef void _initProfiles(self):
        """Initialize Profiles

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
        # initialize profiles
        memset(&(self._profileForward), 0, sizeof(PROFILE_s))
        memset(&(self._profileBackward), 0, sizeof(PROFILE_s))
        memset(&(self._profileBypassFwd), 0, sizeof(PROFILE_s))
        memset(&(self._profileBypassBwd), 0, sizeof(PROFILE_s))

        # determine complexity of class instance transforms and Bypass transform
        self.getComplexity()
        cdef tuple complexity = Matrix._getComplexity(self)
        self._profileBypassFwd.complexity = complexity[0]
        self._profileBypassBwd.complexity = complexity[1]

        # Explore the performance profiles of nested fastmat classes
        # Fills in the profile fields overheadNested and effortNested
        self._exploreNestedProfiles()

        # compile profile of this class to determine transform performance
        cdef float nan = np.nan
        cdef MatrixCalibration calClass = getMatrixCalibration(self.__class__)
        if calClass is None:
            finishProfile(&(self._profileForward), nan, nan)
            finishProfile(&(self._profileBackward), nan, nan)
        else:
            finishProfile(&(self._profileForward),
                          calClass.offsetForward, calClass.gainForward)
            finishProfile(&(self._profileBackward),
                          calClass.offsetBackward, calClass.gainBackward)

        # compile profile of base class to determine bypass performance
        cdef MatrixCalibration calBase  = getMatrixCalibration(Matrix)
        if calBase is None:
            finishProfile(&(self._profileBypassFwd), nan, nan)
            finishProfile(&(self._profileBypassBwd), nan, nan)
        else:
            finishProfile(&(self._profileBypassFwd),
                          calBase.offsetForward, calBase.gainForward)
            finishProfile(&(self._profileBypassBwd),
                          calBase.offsetBackward, calBase.gainBackward)

        # disable transform bypass if profile is either missing or incomplete
        if not (profileIsValid(&self._profileForward) and
                profileIsValid(&self._profileBackward) and
                profileIsValid(&self._profileBypassFwd) and
                profileIsValid(&self._profileBypassBwd)):
            self._bypassAllow = False

    cpdef _exploreNestedProfiles(self):
        r"""Explore Nested Profiles

        Explore the runtime properties of all nested fastmat matrices. Use ane
        iterator on self._content by default to sum the profile properties of
        all nested classes of meta-classes by default. basic-classes either
        have an empty tuple for _content or need to overwrite this method.
        """
        cdef Matrix item
        cdef bint bypass

        for item in self:
            bypass = (item._bypassAllow and
                      (item._array is not None or item._bypassAutoArray))
            profileUpdate(&(self._profileForward), 1, bypass,
                          &(item._profileForward), &(item._profileBypassFwd))
            profileUpdate(&(self._profileBackward), 1, bypass,
                          &(item._profileBackward), &(item._profileBypassBwd))

    ############################################## class methods
    def __init__(self, arrMatrix):
        r"""Initialize Matrix instance

        """
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
            self._array.shape[0],        # numN
            self._array.shape[1],        # numM
            self._array.dtype            # data type of matrix
        )

    def _initProperties(
        self,
        intsize numN,
        intsize numM,
        object dataType,
        **properties
    ):
        r"""Initial Properties

        """

        # assign basic class properties (at c-level)
        self._info.nDim = 2
        self._info.numN = numN
        self._info.numM = numM
        self._info.dtype = _getTypeInfo(dataType)

        # get and assign c-level properties
        self._cythonCall          = properties.pop('cythonCall', False)
        self._forceInputAlignment = properties.pop('forceInputAlignment', False)
        self._widenInputDatatype  = properties.pop('widenInputDatatype', False)
        self._useFortranStyle     = properties.pop('fortranStyle', True)
        self._bypassAllow         = properties.pop('bypassAllow',
                                                   flags.bypassAllow)

        # determine new value of _bypassAutoArray: take the value in `flags`
        # as default but check that no nested child instance has set
        # bypassAutoArray to False. If the parent class would disregard this,
        # A nested instances' AutoArray function would be called implicitly
        # although it shouldn't be.
        cdef bint autoArray = (flags.bypassAutoArray and
                               all(not item.bypassAutoArray for item in self))
        self._bypassAutoArray     = properties.pop('bypassAutoArray', autoArray)

        # initialize performance profile
        # NOTE: If a valid profile is not available (either no calibration data
        # is found or the complexity model could not be evaluated properly),
        # self._bypassAllow will be disabled explicitly in self._initProfiles()
        # This way bypassing decisions are no longer dependent on (potentially)
        # volatile floating-point comparisions against NaN values, as was the
        # case previously)
        self._initProfiles()

    def __repr__(self):
        r"""Representation

        Return a string representing this very class instance. For
        identification module name, class name, instance id and shape will be
        returned formatted as string.
        """

        return "<%s[%dx%d]:0x%12x>" %(
            self.__class__.__name__,
            self.numN,
            self.numM,
            id(self)
        )

    def __str__(self):
        r"""String

        Return a string representing the classes' contents in a more
        human-readable and human-interpretable fashion. Currently the
        call is redirected to self.__repr__().
        """
        return self.getArray().__str__()

    def __len__(self):
        """Return number of nested elements in matrix instance."""
        return 0 if self._content is None else len(self._content)

    def __iter__(self):
        """Iterate through all nested objects of this matrix instance."""
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
        """Return the sum of this matrix instance and another."""
        if isinstance(element, Matrix):
            return Sum(self, element)
        else:
            raise TypeError("Not an addition of fastmat matrices.")

    def __radd__(self, element):
        """Return the sum of another matrix instance and this."""
        if isinstance(element, Matrix):
            return Sum(self, element)
        else:
            raise TypeError("Not an addition of fastmat matrices.")

    def __sub__(self, element):
        """Return the difference of this matrix instance and another."""
        if isinstance(element, Matrix):
            return Sum(
                self, Product(element, np.int8(-1), typeExpansion=np.int8))
        else:
            raise TypeError("Not a subtraction of fastmat matrices.")

    def __rsub__(self, element):
        """Return the difference of another matrix instance and this."""
        if isinstance(element, Matrix):
            return Sum(
                element, Product(self, np.int8(-1), typeExpansion=np.int8))
        else:
            raise TypeError("Not a subtraction of fastmat matrices.")

    def __mul__(self, factor):
        """Return the product of this matrix and another or a scalar."""
        if isinstance(factor, np.ndarray):
            return self.forward(factor)
        else:
            return Product(self, factor, typeExpansion=np.int8)

    def __rmul__(self, factor):
        """Return the product of a scalar and this matrix."""
        if np.isscalar(factor) or isinstance(factor, Matrix):
            return Product(factor, self, typeExpansion=np.int8)
        else:
            raise TypeError("Invalid product term for fastmat Matrix.")

    def __div__(self, divisor):
        """Return the product of a matrix by the reciproce of a given scalar."""
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
    cpdef _forwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        raise NotImplementedError("No _forwardC method implemented in class.")

    cpdef np.ndarray _forward(self, np.ndarray arrX):
        """Forward

        Perform a forward transform for general matrices. This method will get
        overwritten in child classes of Matrix to implement specific transforms.
        This base function will also be called when a cython-call object does
        not define a _forward() method. One circumstance leading to this is
        when the runtime estimation within the forward()-entry point decides to
        bootstrap a dense array representation from within forward(). Then
        _getArray() cannot simply call forward() as this would leed to an
        infinite loop. Then, _forward() will be called directly, leading to this
        issue with cythonCall classes that only define a _forwardC()
        """
        cdef nptype typeInput, typeOutput
        if self._cythonCall:
            # Create output array
            typeInput = typeSelection[np.PyArray_TYPE(arrX)]
            typeOutput = typeInfo[typeInput].promote[
                self._info.dtype[0].fusedType]

            arrOutput = _arrEmpty(2, self.numN, arrX.shape[1],
                                  typeInfo[typeOutput].typeNum)
            self._forwardC(arrX, arrOutput, typeInput, typeOutput)
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

        """
        # local variable holding return array
        cdef np.ndarray arrInput = arrX, arrOutput
        cdef int ndimInput = arrInput.ndim
        cdef ftype typeInput, typeOutput

        # check dimension count and sizes
        if ndimInput > 2:
            raise ValueError("Input data array must be at most 2D")

        # arrInput.N must match self.M in forward() and .N in backward()
        if arrInput.shape[0] != self.numM:
            raise ValueError("Dimension mismatch %s <-!-> %s" %(
                str(self.shape), str((<object> arrInput).shape)))

        # force array of data to be two-dimensional and determine vector count
        if ndimInput == 1:
            arrInput = _arrReshape(arrInput, 2, self.numM, 1, np.NPY_ANYORDER)
        cdef intsize M = arrInput.shape[1]

        # Determine output data type
        typeInput = typeSelection[np.PyArray_TYPE(arrInput)]
        typeOutput = typeInfo[typeInput].promote[self._info.dtype[0].fusedType]

        # force input data type to fulfill some requirements if needed
        #  * check for data type match
        #  * check for data alignment (contiguousy and segmentation)
        typeForce = (typeInfo[typeOutput].typeNum if self._widenInputDatatype
                     else typeInfo[typeInput].typeNum)
        if self._forceInputAlignment:
            arrInput = _arrForceTypeAlignment(arrInput, typeForce, 0,
                                              self._useFortranStyle)
        else:
            if self._widenInputDatatype:
                arrInput = _arrForceType(arrInput, typeForce)

        # estimate runtimes according profileForward and profileBypass
        # if the dot-product bypass strategy leads to smaller runtimes, do it!
        cdef float estimateForward  = (self._profileForward.overhead +
                                       self._profileForward.effort * M)
        cdef float estimateBypass   = (self._profileBypassFwd.overhead +
                                       self._profileBypassFwd.effort * M)
        if (self._bypassAllow and estimateBypass < estimateForward and
                (self._array is not None or self._bypassAutoArray)):
            arrOutput = self.array.dot(arrInput)
        else:
            # call fast transform with either cython or python style
            if self._cythonCall:
                # Create output array
                arrOutput = _arrEmpty(
                    2, self.numN, M if ndimInput > 1 else 1,
                    typeInfo[typeOutput].typeNum)

                # Call calculation routine (fused type dispatch must be
                # be done locally in each class, typeInfo gets passed)
                self._forwardC(arrInput, arrOutput, typeInput, typeOutput)
            else:
                arrOutput = self._forward(arrInput)

        if ndimInput == 1:
            # reshape back to single-dimensional vector
            arrOutput = _arrReshape(arrOutput, 1, self.numN, 1, np.NPY_ANYORDER)

        return arrOutput

    cpdef _backwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        raise NotImplementedError("No _backwardC method implemented in class.")

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        """Backward

        Perform a backward transform for general matrices. This method
        may get overwritten in child classes of Matrix
        """
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

        """
        # local variable holding return array
        cdef np.ndarray arrInput = arrX, arrOutput
        cdef int ndimInput = arrInput.ndim
        cdef ftype typeInput, typeOutput
        cdef nptype typeForce

        # check dimension count and sizes
        if ndimInput > 2:
            raise ValueError("Input data array must be at most 2D")

        # arrInput.N must match self.M in forward() and .N in backward()
        if arrInput.shape[0] != self.numN:
            raise ValueError("Dimension mismatch %s.H <-!-> %s" %(
                str(self.shape), str((<object> arrInput).shape)))

        # force array of data to be two-dimensional and determine vector count
        if ndimInput == 1:
            arrInput = _arrReshape(arrInput, 2, self.numN, 1, np.NPY_ANYORDER)
        cdef intsize M = arrInput.shape[1]

        # Determine output data type
        typeInput = typeSelection[np.PyArray_TYPE(arrInput)]
        typeOutput = typeInfo[typeInput].promote[self._info.dtype[0].fusedType]

        # force input data type to fulfill some requirements if needed
        #  * check for data type match
        #  * check for data alignment (contiguousy and segmentation)
        typeForce = (typeInfo[typeOutput].typeNum if self._widenInputDatatype
                     else typeInfo[typeInput].typeNum)
        if self._forceInputAlignment:
            arrInput = _arrForceTypeAlignment(arrInput, typeForce, 0,
                                              self._useFortranStyle)
        else:
            if self._widenInputDatatype:
                arrInput = _arrForceType(arrInput, typeForce)

        # estimate runtimes according profileForward and profileBypass
        # if the dot-product bypass strategy leads to smaller runtimes, do it!
        cdef float estimateBackward = (self._profileBackward.overhead +
                                       self._profileBackward.effort * M)
        cdef float estimateBypass   = (self._profileBypassBwd.overhead +
                                       self._profileBypassBwd.effort * M)
        if (self._bypassAllow and estimateBypass < estimateBackward and
                (self._arrayH is not None or self._bypassAutoArray)):
            if self._arrayH is None:
                self._arrayH = self.array.T.conj()

            arrOutput = self._arrayH.dot(arrInput)
        else:
            # call fast transform with either cython or python style
            if self._cythonCall:
                # Create output array
                arrOutput = _arrEmpty(
                    2, self.numM, arrInput.shape[1] if ndimInput > 1 else 1,
                    typeInfo[typeOutput].typeNum)

                # Call calculation routine (fused type dispatch must be
                # be done locally in each class, typeInfo gets passed)
                self._backwardC(arrInput, arrOutput, typeInput, typeOutput)
            else:
                arrOutput = self._backward(arrInput)

        if ndimInput == 1:
            # reshape back to single-dimensional vector
            arrOutput = _arrReshape(arrOutput, 1, self.numM, 1, np.NPY_ANYORDER)

        return arrOutput

    ############################################## class reference
    cpdef np.ndarray reference(self):
        # r"""Return Slow Explicit Version of Instance
        #
        # Return an explicit representation of the matrix without using any
        # fastmat code. Provides type checks and raises errors if the matrix
        # type (self.dtype) cannot hold the reference data.
        # """
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
        # r"""Reference
        #
        # Return matrix representation without using a single bit of fastmat
        # code. Overwrite this method in child classes to define its
        # reference.
        # """
        return self._array

    def _forwardReferenceInit(self):
        self._forwardReferenceMatrix = self.reference()

    def _forwardReference(self,
                          arrX
                          ):
        # r"""Calculate the forward transform by non-fastmat means.
        # """
        if self._forwardReferenceMatrix is None:
            self._forwardReferenceInit()

        return self._forwardReferenceMatrix.dot(arrX)

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        if self.__class__ == Matrix:
            # Test code for Matrix base class
            return {
                TEST.COMMON: {
                    TEST.NUM_N      : 35,
                    TEST.NUM_M      : TEST.Permutation([30, TEST.NUM_N]),
                    'mType'         : TEST.Permutation(TEST.ALLTYPES),
                    TEST.PARAMALIGN : TEST.Permutation(TEST.ALLALIGNMENTS),
                    'arrM'          : TEST.ArrayGenerator({
                        TEST.DTYPE  : 'mType',
                        TEST.SHAPE  : (TEST.NUM_N, TEST.NUM_M),
                        TEST.ALIGN  : TEST.PARAMALIGN
                    }),
                    TEST.OBJECT     : Matrix,
                    TEST.INITARGS   : (lambda param: [param['arrM']()]),
                    TEST.NAMINGARGS : dynFormat("%s", 'arrM')
                },
                TEST.CLASS: {},
                TEST.TRANSFORMS: {}
            }
        elif isinstance(self, (Hermitian, Conjugate, Transpose)):
            # Test code for the three Transposition classes that are also
            # defined in this submodule. As the Transpositions are directly
            # derived from the Matrix base class we can put the relevant code
            # directly in here and adapt it to work for all three classes.

            # Note that the resulting matrix shape differs for Hermitian and
            # Transpose so we must reflect this when specifying the shape of the
            # to-be-generated data array. As Permutations of a dimension is not
            # uncommon the corresponding fields must be linked in DATASHAPE
            numN = 35
            numM = 30
            swap = isinstance(self, Conjugate)
            return {
                TEST.COMMON: {
                    TEST.NUM_N      : numN,
                    TEST.NUM_M      : TEST.Permutation([numM, TEST.NUM_N]),
                    'mType'         : TEST.Permutation(TEST.ALLTYPES),
                    TEST.PARAMALIGN : TEST.Permutation(TEST.ALLALIGNMENTS),
                    'arrM'          : TEST.ArrayGenerator({
                        TEST.DTYPE  : 'mType',
                        TEST.SHAPE  : (TEST.NUM_N, TEST.NUM_M),
                        TEST.ALIGN  : TEST.PARAMALIGN
                    }),
                    TEST.DATASHAPE  : (TEST.NUM_M if swap else TEST.NUM_N,
                                       TEST.DATACOLS),
                    TEST.DATASHAPE_T: (TEST.NUM_N if swap else TEST.NUM_M,
                                       TEST.DATACOLS),
                    TEST.OBJECT     : self.__class__,
                    TEST.INITARGS   : (lambda param: [Matrix(param['arrM']())]),
                    TEST.NAMINGARGS : dynFormat("%s", 'arrM')
                },
                TEST.CLASS: {},
                TEST.TRANSFORMS: {}
            }
        else:
            # Any other class should go and define its own tests!
            return {}

    def _getBenchmark(self):
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
        elif isinstance(self, (Hermitian, Conjugate, Transpose)):
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

    def _getDocumentation(self):
        from .inspect import DOC
        return ""

################################################################################
cdef class Hermitian(Matrix):
    r""" Hermitian Transpose of a Matrix

    """
    ############################################## class methods

    def __init__(self, Matrix matrix):
        if not isinstance(matrix, Matrix):
            raise TypeError("Hermitian: Not a fastmat Matrix")

        self._nested = matrix
        self._content = (matrix, )
        self._initProperties(matrix.shape[1], matrix.shape[0], matrix.dtype,
                             cythonCall=matrix._cythonCall,
                             widenInputDatatype=matrix._widenInputDatatype,
                             forceInputAlignment=matrix._forceInputAlignment)

    def __repr__(self):
        r"""

        """
        return "<%s.H>" %(self._nested.__repr__())

    ############################################## class property override
    cpdef np.ndarray _getArray(self):
        return self._nested._getArray().T.conj()

    cpdef np.ndarray _getCol(self, intsize idx):
        return _conjugate(self._nested._getRow(idx))

    cpdef np.ndarray _getRow(self, intsize idx):
        return _conjugate(self._nested._getCol(idx))

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        return np.conjugate(self._nested._getItem(idxM, idxN))

    cpdef object _getLargestEV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return self._nested.largestEV

    cpdef object _getLargestSV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return self._nested.largestSV

    cpdef Matrix _getT(self):
        return getConjugate(self._nested)

    cpdef Matrix _getH(self):
        return self._nested

    cpdef Matrix _getConj(self):
        return Transpose(self._nested)

    ############################################## class performance estimation
    cpdef tuple _getComplexity(self):
        return (self.numM + self.numN, self.numN + self.numM)

    cpdef _forwardC(self, np.ndarray arrX, np.ndarray arrRes,
                    ftype typeX, ftype typeRes):
        r"""Calculate the forward transform of this matrix, cython-style."""
        self._nested._backwardC(arrX, arrRes, typeX, typeRes)

    cpdef _backwardC(self, np.ndarray arrX, np.ndarray arrRes,
                     ftype typeX, ftype typeRes):
        r"""Calculate the backward transform of this matrix, cython-style."""
        self._nested._forwardC(arrX, arrRes, typeX, typeRes)

    cpdef np.ndarray _forward(self, np.ndarray arrX):
        r"""Calculate the forward transform of this matrix"""
        return self._nested._backward(arrX)

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        r"""Calculate the backward transform of this matrix"""
        return self._nested._forward(arrX)

    cpdef np.ndarray _reference(self):
        r"""Reference

        Return an explicit representation of the matrix without using any
        fastmat code.
        """
        return self._nested.reference().T.conj()


################################################################################

cdef inline Matrix getConjugate(Matrix matrix):
    """ Conjugate factory

    Return the conjugate of matrix if matrix has a complex data type. Otherwise
    return matrix. Acts as factory for the Conjugate metaclass.
    """
    return (Conjugate(matrix) if matrix._info.dtype[0].isComplex else matrix)


################################################################################
cdef class Conjugate(Matrix):
    r""" Conjugate of a Matrix

    """
    ############################################## class methods

    def __init__(self, Matrix matrix):
        if not isinstance(matrix, Matrix):
            raise TypeError("Conjugate: Not a fastmat Matrix")

        self._nested = matrix
        self._content = (matrix, )
        self._initProperties(matrix.shape[0], matrix.shape[1], matrix.dtype,
                             cythonCall=matrix._cythonCall,
                             widenInputDatatype=matrix._widenInputDatatype,
                             forceInputAlignment=matrix._forceInputAlignment)

    def __repr__(self):
        r"""Representation

        """
        return "<conj(%s)>" %(self._nested.__repr__())

    ############################################## class property override
    cpdef np.ndarray _getArray(self):
        return self._nested._getArray().conj()

    cpdef np.ndarray _getCol(self, intsize idx):
        return _conjugate(self._nested._getCol(idx))

    cpdef np.ndarray _getRow(self, intsize idx):
        return _conjugate(self._nested._getRow(idx))

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        return np.conjugate(self._nested._getItem(idxN, idxM))

    cpdef object _getLargestEV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return self._nested.largestEV

    cpdef object _getLargestSV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return self._nested.largestSV

    cpdef Matrix _getT(self):
        return Hermitian(self._nested)

    cpdef Matrix _getH(self):
        return Transpose(self._nested)

    cpdef Matrix _getConj(self):
        return self._nested

    ############################################## class performance estimation
    cpdef tuple _getComplexity(self):
        cdef float complexity = self.numN + self.numM
        return (complexity, complexity)

    ############################################## class forward / backward
    cpdef _forwardC(self, np.ndarray arrX, np.ndarray arrRes,
                    ftype typeX, ftype typeRes):
        r"""Calculate the forward transform of this matrix, cython-style.
        """
        cdef np.ndarray arrInput = _conjugate(arrX)
        self._nested.forwardC(arrInput, arrRes, typeX, typeRes)
        _conjugateInplace(arrRes)

    cpdef _backwardC(self, np.ndarray arrX, np.ndarray arrRes,
                     ftype typeX, ftype typeRes):
        r"""Calculate the backward transform of this matrix, cython-style.
        """
        cdef np.ndarray arrInput = _conjugate(arrX)
        self._nested.backwardC(arrInput, arrRes, typeX, typeRes)
        _conjugateInplace(arrRes)

    cpdef np.ndarray _forward(self, np.ndarray arrX):
        r"""Calculate the forward transform of this matrix
        """
        cdef np.ndarray arrRes = self._nested._forward(_conjugate(arrX))
        _conjugateInplace(arrRes)
        return arrRes

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        r"""Calculate the backward transform of this matrix

        """
        cdef np.ndarray arrRes = self._nested._backward(_conjugate(arrX))
        _conjugateInplace(arrRes)
        return arrRes

    ########################################## references: test / benchmark
    cpdef np.ndarray _reference(self):
        r"""Reference

        Return an explicit representation of the matrix without using any
        fastmat code.
        """
        return self._nested._reference().conj()


################################################################################
################################################## Transpose class
cdef class Transpose(Hermitian):
    r"""Transpose of a Matrix

    """
    ############################################## class methods

    def __init__(self, Matrix matrix):
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
        self._initProperties(matrix.shape[1], matrix.shape[0], matrix.dtype,
                             cythonCall=matrix._cythonCall,
                             widenInputDatatype=matrix._widenInputDatatype,
                             forceInputAlignment=matrix._forceInputAlignment)

    def __repr__(self):
        r"""

        """
        return "<%s.T>" %(self._nestedConj.__repr__())

    ############################################## class property override
    cpdef np.ndarray _getArray(self):
        return self._nestedConj._getArray().T

    cpdef np.ndarray _getCol(self, intsize idx):
        return self._nestedConj._getRow(idx)

    cpdef np.ndarray _getRow(self, intsize idx):
        return self._nestedConj._getCol(idx)

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        return self._nestedConj._getItem(idxM, idxN)

    cpdef object _getLargestEV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return self._nestedConj.largestEV

    cpdef object _getLargestSV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return self._nestedConj.largestSV

    cpdef Matrix _getT(self):
        return self._nestedConj

    cpdef Matrix _getH(self):
        return getConjugate(self._nestedConj)

    cpdef Matrix _getConj(self):
        return Hermitian(self._nestedConj)

    ########################################## references: test / benchmark
    cpdef np.ndarray _reference(self):
        r""" Return reference array

        Return an explicit representation of the matrix without using any
        fastmat code.
        """
        return self._nestedConj._reference().T
