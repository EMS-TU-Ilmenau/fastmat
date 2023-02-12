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

from libc.stdlib cimport malloc

import numpy as np
cimport numpy as np

# initialize numpy C-API interface for cython extension-type classes.
# Theis call is required once for every module that uses PyArray_ calls
# WARNING: DO NOT REMOVE THIS LINE OR SEGFAULTS ARE ABOUT TO HAPPEN!
np.import_array()

################################################## type handling

cdef inline void getDtypeInfo(np.dtype dtype, INFO_TYPE_s *info):
    """Short summary.

    Parameters
    ----------
    dtype : np.dtype
        Description of parameter `dtype`.
    *info : INFO_TYPE_s
        Description of parameter `*info`.

    Returns
    -------
    inline void
        Description of returned object.

    """
    '''Fill in type descriptor (INTO_TYPE_s) for given a numpy.dtype.'''
    info[0].numpyType   = dtype.type_num
    info[0].fusedType   = typeSelection[info[0].numpyType]
    info[0].dsize       = dtype.itemsize
    info[0].isNumber    = np.PyDataType_ISNUMBER(dtype)
    info[0].isInt       = np.PyDataType_ISINTEGER(dtype)
    info[0].isFloat     = np.PyDataType_ISFLOAT(dtype)
    info[0].isComplex   = np.PyDataType_ISCOMPLEX(dtype)

    info[0].eps         = 0
    info[0].min         = 0
    info[0].max         = 0
    if dtype != bool:
        if info[0].isFloat or info[0].isComplex:
            # floating-point types
            dtypeInfo       = np.finfo(dtype)
            info[0].eps     = dtypeInfo.eps
            info[0].min     = dtypeInfo.min
            info[0].max     = dtypeInfo.max

        if info[0].isInt:
            # integer-based types
            dtypeInfo       = np.iinfo(dtype)
            info[0].min     = dtypeInfo.min
            info[0].max     = dtypeInfo.max

    # by default, no promotion array will be present
    # also, don't allocate memory for it
    info[0].promote     = NULL


cdef ftype approximateType(INFO_TYPE_s *dtype):
    """Short summary.

    Parameters
    ----------
    *dtype : INFO_TYPE_s
        Description of parameter `*dtype`.

    Returns
    -------
    ftype
        Description of returned object.

    """
    '''Return most suited fastmat datatype id for given type descriptor.'''
    if dtype[0].isNumber:
        if dtype[0].isInt:
            if dtype[0].dsize <= 1:
                return TYPE_INT8
            elif dtype[0].dsize <= 2:
                return TYPE_INT16
            elif dtype[0].dsize <= 4:
                return TYPE_INT32
            elif dtype[0].dsize <= 8:
                return TYPE_INT64
            else:
                return TYPE_FLOAT64
        elif dtype[0].isComplex:
            return TYPE_COMPLEX64 if (dtype[0].dsize <= 8) else TYPE_COMPLEX128
        else:
            if dtype[0].dsize == 1:
                # bool ends here as it is a number, but neither int nor complex
                return TYPE_INT8
            else:
                return TYPE_FLOAT32 if (dtype[0].dsize <= 4) else TYPE_FLOAT64

    return TYPE_INVALID


cdef INFO_TYPE_s *getTypeInfo(object dtype):
    """Short summary.

    Parameters
    ----------
    dtype : object
        Description of parameter `dtype`.

    Returns
    -------
    INFO_TYPE_s
        Description of returned object.

    """
    '''
    Return a pointer to a type descriptor (INFO_TYPE_s) for a given type
    object. Takes any valid type descriptor as input (np.dtype, int, type).
    This function fetches actual data from the pre-generated `typeInfo` array.

    If dtype is an ndarray, the type of the ndarray will be determined.
    '''
    cdef np.dtype ntype

    if type(dtype) == type:
        # if dtype is a python type object, convert it to a numpy type container
        ntype = np.dtype(dtype)
    elif type(dtype) == int:
        # then check for common types: np.dtype and int (numpy typenum)
        ntype = np.PyArray_DescrFromType(dtype)
    elif isinstance(dtype, np.ndarray):
        # if dtype is an ndarray, take the array type
        ntype = dtype.dtype
    elif not isinstance(dtype, np.dtype):
        # throw an error if itself not a dtype
        raise TypeError("Invalid type information %s" % (str(dtype)))
    else:
        ntype = dtype

    cdef ftype fusedType = typeSelection[ntype.type_num]
    if fusedType == TYPE_INVALID:
        raise TypeError("Not a fastmat fused type: %s" %(str(dtype)))

    return &(typeInfo[fusedType])


cdef ntype getNumpyType(object obj):
    """Short summary.

    Parameters
    ----------
    obj : object
        Description of parameter `obj`.

    Returns
    -------
    ntype
        Description of returned object.

    """
    '''Return numpy type number for a given or a given array's data type.'''
    cdef INFO_TYPE_s *info = getTypeInfo(obj)
    return info[0].numpyType


cdef ftype getFusedType(object obj):
    """Short summary.

    Parameters
    ----------
    obj : object
        Description of parameter `obj`.

    Returns
    -------
    ftype
        Description of returned object.

    """
    '''Return fastmat type number for a given or a given array's data type.'''
    cdef INFO_TYPE_s *info = getTypeInfo(obj)
    return info[0].fusedType


cpdef np.float64_t getTypeEps(object obj):
    """Short summary.

    Parameters
    ----------
    obj : object
        Description of parameter `obj`.

    Returns
    -------
    np.float64_t
        Description of returned object.

    """
    '''Return eps for a given or a given array's data type.'''
    cdef INFO_TYPE_s *info = getTypeInfo(obj)
    return info[0].eps


cpdef np.float64_t getTypeMin(object obj):
    """Short summary.

    Parameters
    ----------
    obj : object
        Description of parameter `obj`.

    Returns
    -------
    np.float64_t
        Description of returned object.

    """
    '''
    Return the minimum representable value for a given or a given array's data
    type.
    '''
    cdef INFO_TYPE_s *info = getTypeInfo(obj)
    return info[0].min


cpdef np.float64_t getTypeMax(object obj):
    """Short summary.

    Parameters
    ----------
    obj : object
        Description of parameter `obj`.

    Returns
    -------
    np.float64_t
        Description of returned object.

    """
    '''
    Return the maximum representable value for a given or a given array's data
    type.
    '''
    cdef INFO_TYPE_s *info = getTypeInfo(obj)
    return info[0].max

cpdef isInteger(object obj):
    """Short summary.

    Parameters
    ----------
    obj : object
        Description of parameter `obj`.

    Returns
    -------
    type
        Description of returned object.

    """
    '''Return whether a given data type or an array's data type is integer.'''
    cdef INFO_TYPE_s *info = getTypeInfo(obj)
    return info[0].isInt

cpdef isFloat(object obj):
    """Short summary.

    Parameters
    ----------
    obj : object
        Description of parameter `obj`.

    Returns
    -------
    type
        Description of returned object.

    """
    '''
    Return whether a given data type or an array's data type is floating point.
    '''
    cdef INFO_TYPE_s *info = getTypeInfo(obj)
    return info[0].isFloat

cpdef isComplex(object obj):
    """Short summary.

    Parameters
    ----------
    obj : object
        Description of parameter `obj`.

    Returns
    -------
    type
        Description of returned object.

    """
    '''Return whether a given data type or an array's data type is complex.'''
    cdef INFO_TYPE_s *info = getTypeInfo(obj)
    return info[0].isComplex

################################################## type Promotion stuff

cdef ftype promoteFusedTypes(ftype type1, ftype type2):
    """Short summary.

    Parameters
    ----------
    type1 : ftype
        Description of parameter `type1`.
    type2 : ftype
        Description of parameter `type2`.

    Returns
    -------
    ftype
        Description of returned object.

    """
    if (type1 < 0) or (type2 >= NUM_TYPES) or \
       (type2 < 0) or (type2 >= NUM_TYPES):
        raise ValueError("Invalid type numbers for promotion")

    return typeInfo[type1].promote[type2]

cpdef object safeTypeExpansion(object dtype):
    """Short summary.

    Parameters
    ----------
    dtype : object
        Description of parameter `dtype`.

    Returns
    -------
    object
        Description of returned object.

    """
    return (np.float32 if (dtype == np.int8) or (dtype == np.int16)
            else (np.float64 if (dtype == np.int32) or (dtype == np.int64)
                  else dtype))


### put some type promotion code here! Initialize type table list, specifying
### type conversion rules for this very machine. Define a number of basic types
### which are fused to declare different implementation specializations. Then,
### handle promotion by simple table-lookups in simple steps:
###   (1)  break down the multitude of different types, classify unsupported
###        unsupported ones
###   (2)  generate a type promotion matrix for the remaining types
###   (3)  translate type back for array generation
###
### After a type is correctly promoted and its corresponding result array cre-
### ated, also extract memoryviews with these types for the call specialization
### following.
###
### np.NPY_NTYPES is equal to the total number of numpy types available
### NUM_TYPES is equal to the number of fused types supported (incl. ~INVALID)
### typeInfo contains information about each fused type used in fastmat
### typeSelection can be used to fit numpy data types to fused types

# definine association between numpy data types to fused data types
# CAUTION: must be initialized before typeInfo as the fusedType ID will be taken
#          from typeSelection[]
cdef INFO_TYPE_s ttDescr
cdef np.dtype numpyDtype
cdef int tt, ii

# determine suitable type number for numpy types
for tt in range(<int> np.NPY_NTYPES):
    # fusedType field will not be correct as typeSelection is not yet filled.
    getDtypeInfo(np.PyArray_DescrFromType(tt), &ttDescr)
    typeSelection[tt] = approximateType(&ttDescr)

# set fused type definitions for specific numpy data types
for tt, dtype in {
    TYPE_INT8: np.int8,
    TYPE_INT16: np.int16,
    TYPE_INT32: np.int32,
    TYPE_INT64: np.int64,
    TYPE_FLOAT32: np.float32,
    TYPE_FLOAT64: np.float64,
    TYPE_COMPLEX64: np.complex64,
    TYPE_COMPLEX128: np.complex128,
    TYPE_INVALID: np.void
}.items():
    getDtypeInfo(np.PyArray_DescrFromTypeObject(dtype), &(typeInfo[tt]))

# initialize type promotion stuff:
for tt in range(NUM_TYPES):
    typeInfo[tt].promote = <ftype *> malloc(sizeof(int) * NUM_TYPES)
    for ii in range(NUM_TYPES):
        t1 = np.PyArray_TypeObjectFromType(typeInfo[tt].numpyType)
        t2 = np.PyArray_TypeObjectFromType(typeInfo[ii].numpyType)
        if tt < <int> TYPE_INVALID and ii < <int> TYPE_INVALID:
            numpyDtype = np.promote_types(t1, t2)
            typeInfo[tt].promote[ii] = typeSelection[numpyDtype.type_num]
        else:
            numpyDtype = np.dtype(np.void)
            typeInfo[tt].promote[ii] = TYPE_INVALID

# debug output: show internal data structures


def _typeSelection():
    """Short summary.

    Returns
    -------
    type
        Description of returned object.

    """
    '''Print internals to assist debugging.'''
    print("typeSelection =")
    lst = [[] for ii in range(NUM_TYPES)]
    for ii in range(<int> np.NPY_NTYPES):
        lst[typeSelection[ii]].append(
            "%d:%s" % (ii, str(np.PyArray_DescrFromType(ii))))

    for ii in range(NUM_TYPES):
        print("\t%d:%s" % (ii, lst[ii]))


def _typeInfo():
    """Short summary.

    Returns
    -------
    type
        Description of returned object.

    """
    print("typeInfo =")
    for tt in range(NUM_TYPES):
        lst = [typeInfo[tt].promote[ii] for ii in range(NUM_TYPES)]
        ii = typeInfo[tt].numpyType
        print("\t%d:%s" % (tt, {
            'numpyType' : "%d:%s" % (ii, str(np.PyArray_DescrFromType(ii))),
            'fusedType' : typeInfo[tt].fusedType,
            'dsize'     : typeInfo[tt].dsize,
            'isNumber'  : typeInfo[tt].isNumber,
            'isInt'     : typeInfo[tt].isInt,
            'isFloat'   : typeInfo[tt].isFloat,
            'isComplex' : typeInfo[tt].isComplex,
            'promote'   : lst,
            'eps'       : typeInfo[tt].eps,
            'min'       : typeInfo[tt].min,
            'max'       : typeInfo[tt].max
        }))
