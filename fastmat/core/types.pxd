# -*- coding: utf-8 -*-

# Copyright 2016 Sebastian Semper, Christoph Wagner
#     https://www.tu-ilmenau.de/it-ems/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cython
import numpy as np
cimport numpy as np

################################################## define unique type-id types

ctypedef np.uint8_t ftype
ctypedef np.uint8_t nptype
ctypedef np.npy_intp intsize

################################################## define global fused types
#
# fastmat-wide definitions of fused types which specify the datatypes allowed
# in memoryviews of ndarrays.
#
# Fused types work in a way that each argument of each fused type used as
# parameters must be of the same explicit type. i.e. if a function accepts
# two parameters of the same-named fused type (allowing float and double),
# then both of these parameters must either be float or double and no combina-
# tions between these types are allowed. To allow combinations, each parameter
# must have its own fused type, even if they contain the same definitions.
# Further, these types also must not be derived from other fused types.

# WARNING: be sure to keep the NUM_TYPES definition in types.h correct!
cdef extern from "types.h":
    enum: NUM_TYPES

ctypedef enum TYPES:
    TYPE_INT8
    TYPE_INT32
    TYPE_INT64
    TYPE_FLOAT32
    TYPE_FLOAT64
    TYPE_COMPLEX64
    TYPE_COMPLEX128
    TYPE_INVALID
    TYPE_NUM

################################ fused type classes

ctypedef fused TYPE_INT:
    np.int8_t
    np.int32_t
    np.int64_t

ctypedef fused TYPE_REAL:
    np.float32_t
    np.float64_t

ctypedef fused TYPE_COMPLEX:
    np.complex64_t
    np.complex128_t

ctypedef fused TYPE_FLOAT:
    np.float32_t
    np.float64_t
    np.complex64_t
    np.complex128_t

ctypedef fused TYPE_ALL:
    np.int8_t
    np.int32_t
    np.int64_t
    np.float32_t
    np.float64_t
    np.complex64_t
    np.complex128_t


################################ one round of fused types for input arguments
ctypedef fused TYPE_IN:
    np.int8_t
    np.int32_t
    np.int64_t
    np.float32_t
    np.float64_t
    np.complex64_t
    np.complex128_t

ctypedef fused TYPE_IN_R:
    np.int8_t
    np.int32_t
    np.int64_t
    np.float32_t
    np.float64_t

ctypedef fused TYPE_IN_I:
    np.int8_t
    np.int32_t
    np.int64_t


################################ one round of fused types for operand arguments
ctypedef fused TYPE_OP:
    np.int8_t
    np.int32_t
    np.int64_t
    np.float32_t
    np.float64_t
    np.complex64_t
    np.complex128_t

ctypedef fused TYPE_OP_R:
    np.int8_t
    np.int32_t
    np.int64_t
    np.float32_t
    np.float64_t

ctypedef fused TYPE_OP_I:
    np.int8_t
    np.int32_t
    np.int64_t


################################ type information structures

ctypedef struct INFO_TYPE_s:
    nptype          typeNum
    ftype           fusedType
    ftype           *promote
    np.float64_t    eps
    np.float64_t    min
    np.float64_t    max
    int             dsize
    bint            isNumber
    bint            isInt
    bint            isFloat
    bint            isComplex


ctypedef struct INFO_ARR_s:
    INFO_TYPE_s     *dtype
    int             nDim
    intsize         numN
    intsize         numM


################################################## type handling

cdef INFO_TYPE_s typeInfo[NUM_TYPES]
cdef ftype typeSelection[<int> np.NPY_NTYPES]

cdef INFO_TYPE_s *_getTypeInfo(object)
cdef nptype _getNpType(np.ndarray arr)
cdef ftype _getFType(np.ndarray arr)

cpdef np.float64_t _getTypeEps(object dtype)
cpdef np.float64_t _getTypeMin(object dtype)
cpdef np.float64_t _getTypeMax(object dtype)

################################################## type class checks

cpdef isInteger(object obj)
cpdef isFloat(object obj)
cpdef isComplex(object obj)

################################################## type Promotion stuff

cdef ftype _promoteTypeNums(ftype, ftype)
cpdef object safeTypeExpansion(object)
