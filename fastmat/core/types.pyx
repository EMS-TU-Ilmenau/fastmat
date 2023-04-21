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
    """
    Fill in type descriptor structure (INTO_TYPE_s) for a given numpy.dtype.

    Parameters
    ----------
    dtype : np.dtype
        The numpy.dtype object to retrieve type information for.

    info : INFO_TYPE_s *
        A pointer to the `INFO_TYPE_s` structure to be filled.

    Returns
    -------
    None
    """
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
    """
    Return most suited fastmat datatype id for given type descriptor.

    Parameters
    ----------
    *dtype : INFO_TYPE_s
        An `INFO_TYPE_s` structure holding type info to find a ftype for.

    Returns
    -------
    ftype
        The determined fastmat-type.
    """
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


cdef INFO_TYPE_s * getTypeInfo(object obj) except *:
    """
    Retrieve a type information structure from a given numpy/python type.

    Takes any valid type descriptor as input (np.dtype, int, type).
    If dtype is an ndarray, the type of the ndarray will be determined.

    NOTE: This function fetches actual data from the pre-generated `typeInfo`
          array. Therefore, you must never modify the returned structure.

    Parameters
    ----------
    obj : object
        The type object to retrieve the most fitting type information for.

    Returns
    -------
    INFO_TYPE_s *
        A pointer to the type descriptor structure `INFO_TYPE_s`, which
        resembles the requested type most closely.

        NOTE: Never modify this structure!

    Raises
    ------
    TypeError
        When the type is not supported by fastmat.
    """
    cdef np.dtype ntype

    if type(obj) == type:
        # if dtype is a python type object, convert it to a numpy type container
        ntype = np.dtype(obj)
    elif type(obj) == int:
        # then check for common types: np.dtype and int (numpy typenum)
        ntype = np.PyArray_DescrFromType(obj)
    elif isinstance(obj, np.ndarray):
        # if dtype is an ndarray, take the array type
        ntype = obj.dtype
    elif not isinstance(obj, np.dtype):
        # throw an error if itself not a dtype
        raise TypeError("Invalid type information %s" % (str(obj), ))
    else:
        ntype = obj

    cdef ftype fusedType = typeSelection[ntype.type_num]
    if fusedType >= TYPE_INVALID:
        raise TypeError("Not a fastmat fused type: %s" %(str(obj), ))

    return &(typeInfo[fusedType])


cpdef ntype getNumpyType(object obj) except *:
    """
    Return numpy type number for a given data type (or array).

    Parameters
    ----------
    obj : object
        The object type or `numpy.ndarray` to query for.

    Returns
    -------
    ntype
        The numpy type number for that type.

    Raises
    ------
    TypeError
        When the type is not supported by fastmat.
    """
    cdef INFO_TYPE_s *info = getTypeInfo(obj)
    return info[0].numpyType


cpdef ftype getFusedType(object obj) except *:
    """
    Return fastmat type number for a given data type (or array).

    Parameters
    ----------
    obj : object
        The object type or `numpy.ndarray` to query for.

    Returns
    -------
    ftype
        The fastmat type number for that type.

    Raises
    ------
    TypeError
        When the type is not supported by fastmat.
    """
    '''Return fastmat type number for a given or a given array's data type.'''
    cdef INFO_TYPE_s *info = getTypeInfo(obj)
    return info[0].fusedType


cpdef np.float64_t getTypeEps(object obj) except *:
    """
    Return eps for a given data type (or array).

    Parameters
    ----------
    obj : object
        The object type or `numpy.ndarray` to query for.

    Returns
    -------
    np.float64_t
        The epsilon value for that type.

    Raises
    ------
    TypeError
        When the type is not supported by fastmat.
    """
    cdef INFO_TYPE_s *info = getTypeInfo(obj)
    return info[0].eps


cpdef np.float64_t getTypeMin(object obj) except *:
    """
    Return the minimum representable value for a given data type (or array).

    Parameters
    ----------
    obj : object
        The object type or `numpy.ndarray` to query for.

    Returns
    -------
    np.float64_t
        The minimum representable value for that type.

    Raises
    ------
    TypeError
        When the type is not supported by fastmat.
    """
    cdef INFO_TYPE_s *info = getTypeInfo(obj)
    return info[0].min


cpdef np.float64_t getTypeMax(object obj) except *:
    """
    Return the maximum representable value for a given data type (or array).

    Parameters
    ----------
    obj : object
        The object type or `numpy.ndarray` to query for.

    Returns
    -------
    np.float64_t
        The maximum representable value for that type.

    Raises
    ------
    TypeError
        When the type is not supported by fastmat.
    """
    cdef INFO_TYPE_s *info = getTypeInfo(obj)
    return info[0].max

cpdef bint isInteger(object obj) except *:
    """
    Return whether a given data type or an array's data type is integer.

    Parameters
    ----------
    obj : object
        The object type to query for.

    Returns
    -------
    type
        True if the data type is of integer kind.

    Raises
    ------
    TypeError
        When the type is not supported by fastmat.
    """
    cdef INFO_TYPE_s *info = getTypeInfo(obj)
    return info[0].isInt

cpdef bint isFloat(object obj) except *:
    """
    Return whether a given data type or an array's data type is floating point.

    Parameters
    ----------
    obj : object
        The object type to query for.

    Returns
    -------
    type
        True if the data type is of floating point kind.

    Raises
    ------
    TypeError
        When the type is not supported by fastmat.
    """
    cdef INFO_TYPE_s *info = getTypeInfo(obj)
    return info[0].isFloat

cpdef bint isComplex(object obj) except *:
    """
    Return whether a given data type or an array's data type is complex.

    Parameters
    ----------
    obj : object
        The object type to query for.

    Returns
    -------
    bool
        True if the data type is of complex kind.

    Raises
    ------
    TypeError
        When the type is not supported by fastmat.
    """
    cdef INFO_TYPE_s *info = getTypeInfo(obj)
    return info[0].isComplex

################################################## type Promotion stuff

cdef ftype promoteFusedTypes(ftype type1, ftype type2) except *:
    """
    Return the fastmat type most suited for results based on given types.

    Parameters
    ----------
    type1 : ftype
        The fastmat type identifier for one operation argument.

    type2 : ftype
        The fastmat type identifier for another operation argument.

    Returns
    -------
    ftype
        The fastmat type number that safely expands both argument types, such
        that an operation on both can be held without numerical accuracy loss.

    Raises
    ------
    ValueError
        Invalid fastmat type identifier are passed.
    """
    if (type1 < 0) or (type2 >= NUM_TYPES) or \
       (type2 < 0) or (type2 >= NUM_TYPES):
        raise ValueError("Invalid type numbers for promotion")

    return typeInfo[type1].promote[type2]

cpdef object safeTypeExpansion(object dtype):
    """
    Return a floating type expanding the given type with full accuracy.

    Parameters
    ----------
    dtype : object
        A type object to be expanded to float without numerical accuracy loss.

    Returns
    -------
    object
        The safely expanded datatype
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
    """
    Print type map and selection table.

    NOTE: This function is used for debugging of extension-type internals.

    Returns
    -------
    None
    """
    print("typeSelection =")
    lst = [[] for ii in range(NUM_TYPES)]
    for ii in range(<int> np.NPY_NTYPES):
        lst[typeSelection[ii]].append(
            "%d:%s" % (ii, str(np.PyArray_DescrFromType(ii))))

    for ii in range(NUM_TYPES):
        print("\t%d:%s" % (ii, lst[ii]))


def _typeInfo():
    """
    Print the type information table, also accessed by getTypeInfo.

    NOTE: This function is used for debugging of extension-type internals.

    Returns
    -------
    None
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
