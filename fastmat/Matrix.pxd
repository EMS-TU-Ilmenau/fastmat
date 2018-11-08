# -*- coding: utf-8 -*-

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

import cython
import numpy as np
cimport numpy as np

from .core.types cimport *

################################################## class FastmatFlags
cdef class FastmatFlags:
    cpdef public bint bypassAllow
    cpdef public bint bypassAutoArray


################################################## class MatrixCalibration
cdef class MatrixCalibration(dict):
    cpdef tuple getCall(self, targetCall)


################################################## class MatrixCallProfile
cdef class MatrixCallProfile(object):
    cdef readonly np.float32_t  complexityAlg
    cdef readonly np.float32_t  timeAlgCallOverhead
    cdef readonly np.float32_t  timeAlgPerUnit

    cdef readonly np.float32_t  timeNestedCallOverhead
    cdef readonly np.float32_t  timeNestedPerUnit

    cdef readonly np.float32_t  complexityBypass
    cdef readonly np.float32_t  timeBypassCallOverhead
    cdef readonly np.float32_t  timeBypassPerUnit

    cpdef void addNestedProfile(self, intsize, bint, MatrixCallProfile)
    cpdef bint isValid(self)
    cpdef bint isBypassFaster(self, intsize numVectors)
    cpdef tuple estimateRuntime(self, intsize M)


################################################## class Matrix
cdef struct TRANSFORM:
    intsize numVectors
    ftype fInput
    ftype fInternal
    ftype fOutput
    ntype nInternal
    ntype nOutput

cdef class Matrix:

    cdef bint       _cythonCall                  # use _C() transforms in class
    cdef bint       _forceContiguousInput        # force contiguous input data
    #                                            # to be F-contiguous
    cdef bint       _fortranStyle                # if true, select Fortran style
    cdef bint       _widenInputDatatype          # widen input data type upfront

    ############################################## class variables
    cdef public np.ndarray  _array               # ndarray dense representation
    cdef public np.ndarray  _arrayH              # backward dense representation
    cdef public tuple       _content             # nested fastmat matrices

    cdef public Matrix      _gram                # cache for gram matrix
    cdef public Matrix      _normalized          # cache for normalized matrix
    cdef public object      _largestEV           # cache for largestEV
    cdef public object      _largestSV           # cache for largestSV
    cdef public object      _scipyLinearOperator # interface to scipy
    cdef public Matrix      _T                   # cache for transpose matrix
    cdef public Matrix      _H                   # cache for adjunct matrix
    cdef public Matrix      _conj                # cache for conjugate matrix

    cdef readonly intsize   numN                 # row-count of matrix
    cdef readonly intsize   numM                 # column-count of matrix
    cdef readonly ntype     numpyType            # numpy typenum
    cdef readonly ftype     fusedType            # fastmat fused typenum
    cdef readonly ftype     _minFusedType        # minimal fused type used
    #                                            # internally for transform

    cdef public bint        bypassAllow          # if true, transform may be
    #                                            # bypassed based on runtime
    #                                            # estimation decision
    cdef public bint        bypassAutoArray      # if true, a dense array repre-
    #                                            # sentation for bypassing a
    #                                            # transform will automatically
    #                                            # be generated when required
    cdef public str         tag                  # Description of matrix

    cdef public object  _forwardReferenceMatrix  # ndarray representing Matrix
    #                                            # reference

    ############################################## class profiling
    cdef public MatrixCallProfile profileForward
    cdef public MatrixCallProfile profileBackward

    ############################################## class implementation methods
    cpdef np.ndarray _getArray(self)
    cpdef np.ndarray _getCol(self, intsize)
    cpdef np.ndarray _getRow(self, intsize)
    cpdef object _getItem(self, intsize, intsize)
    cpdef object _getLargestEV(self, intsize, float, float, bint)
    cpdef object _getLargestSV(self, intsize, float, float, bint)
    cpdef object _getScipyLinearOperator(self)
    cpdef Matrix _getGram(self)
    cpdef Matrix _getNormalized(self)
    cpdef Matrix _getT(self)
    cpdef Matrix _getH(self)
    cpdef Matrix _getConj(self)

    ############################################## computation profiling
    cpdef tuple _getComplexity(self)
    cdef void _initProfiles(self)
    cpdef _exploreNestedProfiles(self)
    cpdef tuple estimateRuntime(self, intsize numVectors=?)

    cdef np.ndarray _prepareInputArray(self, np.ndarray, intsize, TRANSFORM *)
    cpdef _forwardC(self, np.ndarray, np.ndarray, ftype, ftype)
    cpdef _backwardC(self, np.ndarray, np.ndarray, ftype, ftype)
    cpdef np.ndarray _forward(self, np.ndarray)
    cpdef np.ndarray _backward(self, np.ndarray)

    ############################################## class interface methods
    cpdef np.ndarray forward(self, np.ndarray)
    cpdef np.ndarray backward(self, np.ndarray)
    cpdef np.ndarray getArray(self)

    ############################################## class reference
    cpdef np.ndarray _reference(self)
    cpdef np.ndarray reference(self)

################################################## Hermitian,Conjugate,Transpose
cdef class Hermitian(Matrix):
    cdef public Matrix _nested                   # nested fastmat baseclass

cdef Matrix getConjugate(Matrix)
cdef class Conjugate(Matrix):
    cdef public Matrix _nested                   # nested fastmat baseclass

cdef class Transpose(Hermitian):
    cdef public Matrix _nestedConj               # nested fastmat baseclass
