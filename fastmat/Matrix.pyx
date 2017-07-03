# -*- coding: utf-8 -*-
'''
  fastmat/Matrix.pyx
 -------------------------------------------------- part of the fastmat package

  Base class for all fast matrix transformations in fastmat.


  Author      : wcw, sempersn
  Introduced  : 2016-04-08
 ------------------------------------------------------------------------------

   Copyright 2016 Sebastian Semper, Christoph Wagner
       https://www.tu-ilmenau.de/ems/

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

 ------------------------------------------------------------------------------
TODO:
    - add caches for cols / rows
    - implement generation of normalized matrix of general matrices
'''
import types
import sys

import numpy as np
cimport numpy as np

# initialize numpy C-API interface for cython extension-type classes.
# This call is required once for every module that uses PyArray_ calls
# WARNING: DO NOT REMOVE THIS LINE OR SEGFAULTS ARE ABOUT TO HAPPEN!
np.import_array()

from .helpers.types cimport *
from .helpers.cmath cimport *
#_conjugate, _arrEmpty, _arrReshape, _arrForceContType
from .helpers.resource import getMemoryFootprint
from .Transpose cimport *
from .Product cimport Product
from .Sum cimport Sum
from .Diag cimport Diag


################################################################################
################################################## class Matrix
cdef class Matrix(object):
    '''
    class Matrix of fastmat

    Matrix is the base class of which all fastmat matrices are derived from.
    Many routines, which are available for all classes, are defined here.
    Please note: Many properties defined here link to private variables which
    are overwritten in child classes.
        e.g. self.numN --> self._numN, which may be overwritten in
        child.__init__() to reflect childs' dimension.
    '''

    ############################################## class properties
    # dtype - Property (read-only)
    # Report the matrix elements' data type
    property dtype:
        def __get__(self):
            return np.PyArray_TypeObjectFromType(self._info.dtype.typeNum)

    # dtypeNum - Property (read-only)
    # Report the matrix elements' data type (fastmat fused type number)
    property dtypeNum:
        def __get__(self):
            return self._info.dtype[0].fusedType

    # numN - Property (read-only)
    # Report the matrix row count
    property numN:
        def __get__(self):
            return self._info.numN

    # numM - Property (read-only)
    # Report the matrix column count
    property numM:
        def __get__(self):
            return self._info.numM

    # shape - Property (read-only)
    # Report the matrix shape as tuple of (rows, columns)
    property shape:
        def __get__(self):
            return (self.numN, self.numM)

    # tag - Property (read-write)
    # Description tag of Matrix
    property tag:
        def __get__(self):
            return (self._tag)

        def __set__(self, tag):
            self._tag = tag

    # nbytes - Property(read)
    # Size of the Matrix object
    property nbytes:
        def __get__(self):
            return getMemoryFootprint(self)

    # _nbytesReference - Property(read)
    # Size of the Matrix object
    property nbytesReference:
        def __get__(self):
            if self._forwardReferenceMatrix is None:
                self._forwardReferenceInit()

            return getMemoryFootprint(self._forwardReferenceMatrix)

    ############################################## Matrix content property
    # content - Property (read-only)
    # Return the Matrix contents
    property content:
        def __get__(self):
            if self.__class__ == Matrix:
                return self._data

    ############################################## generic algebric properties
    # gram - Property (read-only)
    # Return the gram matrix for this fastmat class
    property gram:
        def __get__(self):
            return (self.getGram() if self._gram is None
                    else self._gram)

    # normalized - Property (read-only)
    # Return a normalized matrix for this instance
    property normalized:
        def __get__(self):
            return (self.getNormalized() if self._normalized is None
                    else self._normalized)

    # largestEV - Property (read-only)
    # Return the largest eigenvalue for this matrix instance
    property largestEV:
        def __get__(self):
            return (self.getLargestEV() if self._largestEV is None
                    else self._largestEV)

    # largestSV - Property (read-only)
    # Return the largestSV for this matrix instance
    property largestSV:
        def __get__(self):
            return (self.getLargestSV() if self._largestSV is None
                    else self._largestSV)

    # T - Property (read-only)
    # Return the transpose of the matrix as fastmat class
    property T:
        def __get__(self):
            return (self.getT() if self._T is None else self._T)

    # H - Property (read-only)
    # Return the hermitian transpose of the matrix as fastmat class
    property H:
        def __get__(self):
            return (self.getH() if self._H is None else self._H)

    # conj - Property (read-only)
    # Return the conjugate of the matrix as fastmat class
    property conj:
        def __get__(self):
            return (self.getConj() if self._conj is None else self._conj)

    ############################################## generic math implementations
    # if math functionality is to be overloaded n child classes, overload these
    # generic functions with specialized ones.
    # self._getXXX are intended to only incorporate the actual generation where
    # self.getXXX provide additional sanity checks, caching abilities as well
    # as specification of input argument default values

    cpdef np.ndarray _getCol(self, intsize idx):
        if self.__class__ == Matrix:
            return self._data[:, idx]

        cdef np.ndarray arrData = _arrZero(1, self._info.numM, 1,
                                           self._info.dtype[0].typeNum)
        arrData[idx] = 1
        return self.forward(arrData)

    cpdef np.ndarray _getRow(self, intsize idx):
        if self.__class__ == Matrix:
            return self._data[idx, :]

        cdef np.ndarray arrData = _arrZero(1, self._info.numN, 1,
                                           self._info.dtype[0].typeNum)
        arrData[idx] = 1
        return self.backward(arrData).conj()

    cpdef object _getLargestEV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        '''
        Determine largest eigen-value of a fastmat matrix.

        The following variables are used:
        eps            - relative stopping threshold
        numNormOld     - norm of vector from last iteration
        numNormNew     - norm of vector in current iteration
        vecBNew        - normalized current iterate
        vecBOld        - normalized last iterate
        '''
        cdef np.ndarray vecBOld, vecBNew, vecBNewHat

        # determine an convergance threshold if eps is deliberately set to zero
        if eps == 0:
            eps = relEps * _getTypeEps(safeTypeExpansion(self.dtype)) * \
                (self.numN if self.numN >= self.numM else self.numM)

        if self.numN != self.numM:
            raise ValueError("Determination of largest Eigenvalue yet only " +
                             "implemented for square matrices.")

        # sample one point uniformly in space, have a zero-vector reference
        vecBNew = np.random.randn(self.numM).astype(
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

    cpdef object _getLargestSV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        '''
        Determine largest singular value of a fastmat matrix.

        The following variables are used:
        eps            - relative stopping threshold
        vecBNew        - normalized current iterate
        vecBOld        - normalized last iterate
        '''
        cdef np.ndarray vecBOld, vecBNew
        cdef Matrix matGram = self.gram
        cdef intsize ii

        # determine an convergance threshold if eps is deliberately set to zero
        if eps == 0:
            eps = relEps * _getTypeEps(safeTypeExpansion(self.dtype)) * \
                (self.numN if self.numN >= self.numM else self.numM)

        # sample one initial sample, have a zero-vector reference
        vecB = np.random.randn(self.numM).astype(
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

    cpdef Matrix _getT(self):
        return TransposeFactory(self, TRANSPOSE_T)

    cpdef Matrix _getH(self):
        return TransposeFactory(self, TRANSPOSE_H)

    cpdef Matrix _getConj(self):
        return TransposeFactory(self, TRANSPOSE_C)

    cpdef Matrix _getGram(self):
        return Product(self.H, self)

    cpdef Matrix _getNormalized(self):
        # determine type of normalization diagonal matrix
        diagType = safeTypeExpansion(self.dtype)

        # array that contains the norms of each column
        cdef np.ndarray arrDiag = np.empty(self.numM, dtype=diagType)

        # number of elements we consider at once during normalization
        cdef int numStrideSize = 2 ** 8

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
            raise ValueError(
                "At least one column has zero norm. Normalization aborted.")

        # finally invert the diagonal and generate normalized matrix
        return self * Diag(1. / arrDiag)

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        return (self._data[idxN, idxM] if self.__class__ == Matrix
                else self._getCol(idxM)[idxN])

    ############################################## generic math entry points
    # if math functionality is to be overloaded n child classes, overloag these
    # self._getXXX functions as self.getXXX will be used to incorporate sanity
    # checks and default input abstraction
    def getCol(self, idx):
        # TODO it might be nice to offer some cache for the columns here
        if idx < 0 or idx >= self.numM:
            raise ValueError("Column index must not exceed matrix dimensions.")

        return self._getCol(idx)

    def getCols(self, indices):
        '''Return selected columns of self.toarray()'''
        cdef np.ndarray arrResult, arrIdx
        cdef intsize ii, numSize

        if np.isscalar(indices):
            arrResult = self.getCol(indices)
        else:
            arrIdx = np.array(indices)
            if arrIdx.ndim > 1:
                raise ValueError("Index array must have at most one dimension.")

            numSize = arrIdx.size
            arrResult = _arrEmpty(2, self._info.numN, numSize,
                                  self._info.dtype[0].typeNum)

            for ii in range(numSize):
                arrResult[:, ii] = self.getCol(arrIdx[ii])

        return arrResult

    def getRow(self, idx):
        # TODO it might be nice to offer some cache for the columns here
        if idx < 0 or idx >= self.numN:
            raise ValueError("Row index must not exceed matrix dimensions.")

        return self._getRow(idx)

    def getRows(self, indices):
        '''Return selected rows of self.toarray()'''
        cdef np.ndarray arrResult, arrIdx
        cdef intsize ii, numSize

        if np.isscalar(indices):
            arrResult = self.getRow(indices)
        else:
            arrIdx = np.array(indices)
            if arrIdx.ndim > 1:
                raise ValueError("Index array must have at most one dimension.")

            numSize = arrIdx.size
            arrResult = _arrEmpty(2, numSize, self._info.numM,
                                  self._info.dtype[0].typeNum)

            for ii in range(numSize):
                arrResult[ii, :] = self.getRow(arrIdx[ii])

        return arrResult

    def getLargestSV(self, maxSteps=10000,
                     relEps=1., eps=0., alwaysReturn=False):
        result = self._getLargestSV(maxSteps, relEps, eps, alwaysReturn)
        self._largestSV = self._largestSV if np.isnan(result) else result
        return result

    def getLargestEV(self, maxSteps=10000,
                     relEps=1., eps=0., alwaysReturn=False):
        if self._numN != self._numM:
            raise ValueError("Matrix must be quadratic for largestEV")

        result = self._getLargestEV(maxSteps, relEps, eps, alwaysReturn)
        self._largestEV = self._largestEV if np.isnan(result) else result
        return result

    def getT(self):
        self._T = self._getT()
        return self._T

    def getH(self):
        self._H = self._getH()
        return self._H

    def getConj(self):
        self._conj = self._getConj()
        return self._conj

    def getGram(self):
        self._gram = self._getGram()
        return self._gram

    def getNormalized(self):
        self._normalized = self._getNormalized()
        return self._normalized

    def __getitem__(self, tplIdx):
        '''
        Return the element at index tplIdx. The element is evaluated using
        self.forward() on a column-selecting vector, which is then indexed at
        the row-index to retrieve the element. This is quite slow and memory
        consuming for large matrices due to the use of numpy.eye().
        '''
        if len(tplIdx) != 2:
            raise ValueError("Matrix must have two indices.")

        cdef intsize idxN = tplIdx[0], idxM = tplIdx[1]
        cdef intsize N = self._info.numN, M = self._info.numM

        if idxN < 0 or idxN >= N or idxM < 0 or idxM >= M:
            raise IndexError("Index %s exceeds matrix dimensions %s." %(
                str(tplIdx), str(self.shape)))

        return self._getItem(idxN, idxM)

    ############################################## class methods
    def __init__(self, arrMatrix):
        '''Initialize Matrix instance with numpy-array from arrMatrix.'''
        dims = len(arrMatrix.shape)
        if dims == 2:
            self._data = np.copy(arrMatrix)
        elif dims < 2:
            self._data = np.reshape(np.copy(arrMatrix), (len(arrMatrix), 1))
        else:
            raise NotImplementedError("Matrix data array must be 2D")

        # set properties of matrix
        self._initProperties(
            self._data.shape[0],        # numN
            self._data.shape[1],        # numM
            self._data.dtype            # data type of matrix
        )

    def _initProperties(
        self,
        intsize numN,
        intsize numM,
        object dataType,
        **properties
    ):
        '''
        Initialize properties of the matrix representation.
        Use explicit values for type properties.
        '''

        # assign basic class properties (at c-level)
        self._info.nDim = 2
        self._info.numN = numN
        self._info.numM = numM
        self._info.dtype = _getTypeInfo(dataType)

        # get and assign c-level properties
        self._cythonCall          = properties.pop('cythonCall', False)
        self._forceInputAlignment = properties.pop('forceInputAlignment', False)
        self._widenInputDatatype  = properties.pop('widenInputDatatype', False)

    cpdef np.ndarray toarray(self):
        '''
        Return an explicit representation of the matrix as numpy-array.
        '''
        # check whether this instance is a Matrix or some child class
        if self.__class__ == Matrix:
            # it is a Matrix, so return our internal data
            return self._data
        else:
            # it is any child class. Use self.forward() to generate a ndarray
            # representation of this matrix. This calculation is very slow but
            # also very general. Therefore it is used also in child classes
            # when no explicit code for self.toarray() is provided.
            return self.forward(np.eye(self.numM, dtype=self.dtype))

    def __array__(self):
        '''
        Return an expanded array representation of matrix. Invoked internally
        during broadcasting, redirect to self.toarray().
        '''
        return self.toarray()

    def __repr__(self):
        '''
        Return a string representing this very class instance. For
        identification module name, class name, instance id and shape will be
        returned formatted as string.
        '''
        return "<%s[%dx%d]:0x%12x>" %(
            self.__class__.__name__,
            self.numN,
            self.numM,
            id(self)
        )

    def __str__(self):
        '''
        Return a string representing the classes' contents in a more
        human-readable and human-interpretable fashion. Currently the
        call is redirected to self.__repr__().
        '''
        return self.toarray().__str__()

    ############################################## class operator overloading
    def __add__(self, element):
        '''Return the sum of this matrix instance and another.'''
        return Sum(self, element)

    def __sub__(self, element):
        '''Return the difference of this matrix instance and another.'''
        return Sum(self, (-1) * element)

    def __mul__(self, factor):
        '''Return the product of this matrix and another or a scalar.'''
        if isinstance(factor, np.ndarray):
            return self.forward(factor)
        else:
            return Product(self, factor)

    def __rmul__(self, factor):
        '''Return the product of a scalar and this matrix.'''
        if np.isscalar(factor) or isinstance(Matrix, factor):
            return Product(self, factor)
        else:
            raise TypeError(
                "Product term must be either scalar or a fastmat matrix.")

    ############################################## class copy/deepcopy handling
    def __copy__(self):
        '''
        Performs a copy() on this class instance. As fastmat matrix classes are
        defined immutable nothing must be copied, so retuning self suffices.
        '''
        return self

    def __deepcopy__(self, dict memo):
        '''
        Performs a deepcopy() on this class instance. As fastmat matrix classes
        are defined immutable nothing must be copied, so retuning self suffices.
        '''
        return self

    ############################################## class forward / backward
    cpdef _forwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        raise NotImplementedError("No _forwardC method implemented in class.")

    cpdef _backwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        raise NotImplementedError("No _backwardC method implemented in class.")

    cpdef np.ndarray _forward(self, np.ndarray arrX):
        '''
        Perform a forward transform for general matrices. This method
        may get overwritten in child classes of Matrix
        '''
        return self._data.dot(arrX)

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        '''
        Perform a backward transform for general matrices. This method
        may get overwritten in child classes of Matrix
        '''
        return _conjugate(self._data.T).dot(arrX)

    cpdef np.ndarray forward(self, np.ndarray arrX):
        '''
        Calculate the forward transform (self * x).
        Dimension-checking is performed to ensure valid fast transforms as these
        may succeed even when dimensions do not match. To support both single-
        and multidimensional input vectors x, single dimensional input will be
        reshaped to (n, 1) before processing and flattened to (n) after
        completion. This allows the use of both vectors and arrays. The actual
        transform code gets called by the callbacks specified in funcPython and
        funcCython, depending on the state of self._cythonCall.

        !!! Do not override this method !!!
        '''
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

        # force array of data to be two-dimensional
        if ndimInput == 1:
            arrInput = _arrReshape(arrInput, 2, self.numM, 1, np.NPY_ANYORDER)

        # Determine output data type
        typeInput = typeSelection[np.PyArray_TYPE(arrInput)]
        typeOutput = typeInfo[typeInput].promote[self._info.dtype[0].fusedType]

        # force input data type to fulfill some requirements if needed
        #  * check for data type match
        #  * check for data alignment (contiguousy and segmentation)
        typeForce = typeInfo[typeOutput].typeNum \
            if self._widenInputDatatype else typeInfo[typeInput].typeNum
        if self._forceInputAlignment:
            arrInput = _arrForceTypeAlignment(arrInput, typeForce, 0)
        else:
            if self._widenInputDatatype:
                arrInput = _arrForceType(arrInput, typeForce)

        # call with either cython & python styles
        if self._cythonCall:
            # Create output array
            arrOutput = _arrEmpty(
                2, self.numN, arrInput.shape[1] if ndimInput > 1 else 1,
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

    cpdef np.ndarray backward(self, np.ndarray arrX):
        '''
        Calculate the backward transform (self ^ H * x) where ^H is the
        hermitian transpose operator.
        Dimension-checking is performed to ensure valid fast transforms as these
        may succeed even when dimensions do not match. To support both single-
        and multidimensional input vectors x, single dimensional input will be
        reshaped to (n, 1) before processing and flattened to (n) after
        completion. This allows the use of both vectors and arrays. The actual
        transform code gets called by the callbacks specified in funcPython and
        funcCython, depending on the state of self._cythonCall.

        !!! Do not override this method !!!
        '''
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

        # force array of data to be two-dimensional
        if ndimInput == 1:
            arrInput = _arrReshape(arrInput, 2, self.numN, 1, np.NPY_ANYORDER)

        # Determine output data type
        typeInput = typeSelection[np.PyArray_TYPE(arrInput)]
        typeOutput = typeInfo[typeInput].promote[self._info.dtype[0].fusedType]

        # force input data type to fulfill some requirements if needed
        #  * check for data type match
        #  * check for data alignment (contiguousy and segmentation)
        typeForce = typeInfo[typeOutput].typeNum \
            if self._widenInputDatatype else typeInfo[typeInput].typeNum
        if self._forceInputAlignment:
            arrInput = _arrForceTypeAlignment(arrInput, typeForce, 0)
        else:
            if self._widenInputDatatype:
                arrInput = _arrForceType(arrInput, typeForce)

        # call with either cython & python styles
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
        '''
        Return an explicit representation of the matrix without using any
        fastmat code. Provides type checks and raises errors if the matrix type
        (self.dtype) cannot hold the reference data.
        '''
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
        '''
        Return matrix representation without using a single bit of fastmat code.
        Overwrite this method in child classes to define its reference.
        '''
        return self._data

    def _forwardReferenceInit(self):
        self._forwardReferenceMatrix = self.reference()

    def _forwardReference(self,
                          arrX
                          ):
        '''Calculate the forward transform by non-fastmat means.'''
        if self._forwardReferenceMatrix is None:
            self._forwardReferenceInit()

        return self._forwardReferenceMatrix.dot(arrX)


################################################################################
################################################################################
from .helpers.unitInterface import *
################################################### Testing
test = {
    NAME_COMMON: {
        TEST_NUM_N: 35,
        TEST_NUM_M: Permutation([30, TEST_NUM_N]),
        'mType': Permutation(typesAll),
        TEST_PARAMALIGN: Permutation(alignmentsAll),
        'arrM': ArrayGenerator({
            NAME_DTYPE  : 'mType',
            NAME_SHAPE  : (TEST_NUM_N, TEST_NUM_M),
            NAME_ALIGN  : TEST_PARAMALIGN
            #            NAME_CENTER : 2,
        }),
        TEST_OBJECT: Matrix,
        TEST_INITARGS: (lambda param : [param['arrM']()]),
        TEST_NAMINGARGS: dynFormatString("%s", 'arrM')
    },
    TEST_CLASS: {
        # test basic class methods
    }, TEST_TRANSFORMS: {
        # test forward and backward transforms
    }
}


################################################## Benchmarks
benchmark = {
    NAME_COMMON: {
        NAME_DOCU       :
            r'$\bm M = \bm 0_{2^k \times 2^k}$; so $n = 2^k$ for $k \in \N$',
    },
    BENCH_OVERHEAD: {
        BENCH_FUNC_GEN  : (lambda c : Matrix(np.zeros((2 ** c, 2 ** c)))),
        BENCH_FUNC_SIZE : (lambda c : 2 ** c)
    }
}


################################################## Documentation
docLaTeX = r"""
\subsection{General Matrix (\texttt{fastmat.Matrix})}
\subsubsection{Definition and Interface}
This class serves several purposes. The first one is, that it provides the
functionality to also work with unstructured matrices in conjunction with the
other structured matrices. This is enabled by simply being a wrapper around a
standard NumPy Array \cite{mat_walt2011numpy}.

Second, it serves as the base where all other clases are derived from. This
means it contains properties and methods that are shared across all other \fm{}
classes.

\begin{snippet}
\begin{lstlisting}[language=Python]
# import the package
import fastmat as fm
import numpy as np

# define the parameter
n = 10

# construct the matrix
I = fm.Matrix(np.zeros((4,4)))
\end{lstlisting}

This yields a $4 \times 4$ zero matrix which is represented by a \np{} array
and as such does not make use of the structural information anymore.
\end{snippet}

\subsubsection{Properties}

\begin{itemize}
    \item{Mathematical properties of Matrices}
    \item{Result generated on first request only, then cached}
    \item{Defined in Baseclass, available in all subclasses}
    \item{Behaviour override in child classes possible}
\end{itemize}

These are the properties currently available for all matrices across \fm{}.
\begin{itemize}
\item Transpose (\texttt{Matrix.T})
\item Hermitian Transpose (\texttt{Matrix.H})
\item Conjugate (\texttt{Matrix.conj})
\item Normalized Matrix (\texttt{Matrix.normalized})
\item Gram Matrix (\texttt{Matrix.gram})
\item Largest Eigenvalue (\texttt{Matrix.largestEV})
\end{itemize}
%
%
For a given matrix $\bm A \in \C^{n \times n}$, so $\bm A$ square, we calculate
the absolute value of the largest eigenvalue $\lambda \in \C$. The eigenvalues
obey the equation
\begin{align}
\bm A \cdot \bm v= \lambda \cdot \bm v,
\end{align}
where $\bm v$ is a non-zero vector.

\begin{itemize}
\item \textbf{Input:} matrix $\bm A$, parameter $0 < \varepsilon \ll 1$ as a
stopping criterion
\item \textbf{Output:} largest eigenvalue $\sigma_{\rm max}(\bm A)$
\end{itemize}

\textit{Note:} This algorithm performs well if the two largest eigenvalues are
not very close to each other on a relative scale with respect to their absolute
value. Otherwise it might get trouble converging properly.

\begin{snippet}
\begin{lstlisting}[language=Python]
# import the packages
import numpy.linalg as npl
import numpy as np
import fastmat as fm

# define the matrices
n = 5
H = fm.Hadamard(n)
D = fm.Diag(np.linspace(
    1,2**n,2**n
    ))

K1 = fm.Product(H, D)
K2 = K1.toarray()

# calculate the eigenvalue
x1 = K1.largestEV
x2 = npl.eigvals(K2)
x2 = np.sort(np.abs(x2))[-1]

# check if the solutions match
print(x1-x2)
\end{lstlisting}

We define a matrix-matrix product of a Hadamard matrix and a diagonal matrix.
Then we also cast it into a \texttt{numpy}-array and use the integrated EVD. For
demonstration, try to increase $n$ to $>10$ and see what happens.
\end{snippet}

\subsubsection{Largest Singular Value (\texttt{Matrix.largestSV})}
%
For a given matrix $\bm A \in \C^{n \times m}$, we calculate the largest
singular value $\sigma_{\rm max}(\bm A) > 0$, which is the largest entry of the
diagonal matrix $\bm \Sigma \in \C^{n \times m}$ in the decomposition
\begin{align}
\bm A = \bm U \bm \Sigma \bm V^\herm,
\end{align}
where $\bm U$ and $\bm V$ are matrices of the appropriate dimensions. This is
done via the so called power iteration of $\bm A^\herm \cdot \bm A$.

\begin{itemize}
\item \textbf{Input:} matrix $\bm A$, parameter $0 < \varepsilon \ll 1$ as a
stopping criterion
\item \textbf{Output:} largest singular value $\sigma_{\rm max}(\bm A)$
\end{itemize}

\textit{Note:} This algorithm performs well if the two largest singular values
are not very close to each other on a relative scale. Otherwise it might get
trouble converging properly.

\begin{snippet}
\begin{lstlisting}[language=Python]
# import packages
import numpy.linalg as npl
import numpy as np
import fastmat as fm

# define involved matrices
n = 5
H = fm.Hadamard(n)
F = fm.Fourier(2**n)
K1 = fm.Kron(H, F)
K2 = K1.toarray()

# calculate the largest SV
# and a reference solution
x1 = largestSV(K1.largestSV
x2 = npl.svd(K2,compute_uv=0)[0]

# check if they match
print(x1-x2)
\end{lstlisting}

We define a Kronecker product of a Hadamard matrix and a Fourier matrix. Then we
also cast it into a \texttt{numpy}-array and use the integrated SVD. For
demonstration, try to increase $n$ to $>10$ and see what happens.
\end{snippet}

\begin{thebibliography}{9}
\bibitem{mat_walt2011numpy}
St\'efan van der Walt, S. Chris Colbert and Ga\"el Varoquaux
\emph{The NumPy Array: A Structure for Efficient Numerical Computation},
Computing in Science and Engineering,
Volume 13,
2011.
\end{thebibliography}

"""
