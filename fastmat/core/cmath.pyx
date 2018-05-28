# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False

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

from libc.string cimport memcpy, memset
from libc.math cimport ceil

import numpy as np
cimport numpy as np

from .types cimport *
from timeit import timeit

# initialize numpy C-API interface for cython extension-type classes.
# Theis call is required once for every module that uses PyArray_ calls
# WARNING: DO NOT REMOVE THIS LINE OR SEGFAULTS ARE ABOUT TO HAPPEN!
np.import_array()

################################################################################
###  Complexity estimation routines
################################################################################
cdef int _findFFTFactors(int targetLength, int maxFactor,
                         int state, int bestState):
    cdef int ff, length, complexity, newState
    for ff in range(maxFactor, 0, -1):
        length = (state & 0xFFFF) * ff
        complexity = (state >> 16) + ff + 1
        newState = (complexity << 16) + length
        if newState <= bestState and length < targetLength:
            # still now filled targeLength? -> recurse deeper but limit max
            # factor of next stage for speed (still complete)
            bestState = _findFFTFactors(targetLength, ff, newState, bestState)
        else:
            # this iteration ist better than wahat we already found?
            if newState < bestState:
                bestState = newState

            continue

    return bestState


cpdef intsize _findOptimalFFTSize(intsize order, int maxStage):
    cdef intsize paddedSize = 1, minimalSize = order
    cdef float remaining = minimalSize
    cdef int x, complexity, length
    # fill the factor reasonably far with 4-DFTs but leave some room for
    # choosing a close-to-optimal combination of DFT sizes to come as
    # close as reasonable to the targeted size
    while remaining > 64:
        paddedSize *= 4
        remaining /= 4

    x = int(ceil(remaining))

    # for what's left of minimalSize choose the "last mile" optimally, i.e.
    # accept some overshoot while heading for the optimal stage combination
    if x != 1:
        # start a recursion. Pass a default as "current best solution":
        # three stages of 4-DFT, (with complexity of 15 [3 * 4 + 1])
        length = 64
        complexity = 3 * (4 + 1)
        factor = _findFFTFactors(x, maxStage, 1,
                                 (complexity << 16) + length) & 0xFFFF
        paddedSize *= factor

    return paddedSize


cpdef float _getFFTComplexity(intsize N):
    '''
    Return an estimate on the complexity of a typical FFT algorithm.

    Consider a DFT of size N to be pieced together by smaller DFTs corresponding
    to the prime factors of N. Each factor F introduces one level of (N / F)
    DFTs of size F each and a scalar multiplication of the stages' output. The
    composite complexity of all stages is extended by one stage for
    initialization of input and permutation of output.
    '''

    # determine complexity on-the-fly during prime factor decomposition
    # single should be just enough as the largest N is 2^63 which leads to
    # an floating-point exponent of +126 that single still can hold
    cdef intsize nn, factor, x
    cdef float complexity = 0, floatN = N
    cdef ii

    # let nn denote the remaining "non-explored" fraction of N
    nn = N

    # first, consider Radix-4 stages as these are the most efficient and
    # desirable.
    while nn % 4 == 0:
        complexity += 4 + 1
        nn /= 4

    # as 2 * 2 == 4 there can only be one more factor of 2
    if nn > 1 and nn & 1 == 0:
        complexity += 2 + 1
        nn /= 2

    # now consider all odd numbers larger or equal to three as dividers
    # only search up to sqrt(nn), which represents the largest single
    # factor with product nn.
    ii = 3
    while (nn > 1) and (ii * ii < nn):
        if nn % ii == 0:
            complexity += ii + 1
            nn /= ii
        else:
            ii += 2

    # if nn is still not 1 the remainder must be a prime by itself
    # also include the output permutation stage
    if nn > 1:
        complexity += nn + 1

    return floatN * (complexity + 1)


################################################################################
###  Profiling calls
################################################################################


def profileCall(reps, call, *args):
    '''
    wrapper for measuring the runtime of 'call' by averaging the runtime
    of many repeated calls.

        reps        - number of repetitions to be averaged over
        call        - pointer to the evaluatee call
        *args       - a list of parameters for the callee
    '''
    cdef object arg1
    cdef object arg2
    cdef intsize N

    N = 1 if reps < 1 else reps

    def _inner():
        cdef intsize ii
        for _ii in range(N):
            call(*args)

    def _inner1():
        cdef intsize ii
        for _ii in range(N):
            call(arg1)

    def _inner2():
        cdef intsize ii
        for _ii in range(N):
            call(arg1, arg2)

    if len(args) == 1:
        arg1 = args[0]
        runtime = timeit(_inner1, number=1)
    elif len(args) == 2:
        arg1 = args[0]
        arg2 = args[1]
        runtime = timeit(_inner2, number=1)
    else:
        _innerArgs = args
        runtime = timeit(_inner, number=1)

    # return results
    return {
        'avg': runtime / reps,
        'time': runtime,
        'cnt': reps
    }


################################################################################
###  Array creation routines
################################################################################

################################################## _arrZero()
cpdef np.ndarray _arrZero(
    int dims,
    intsize numN,
    intsize numM,
    ntype dtype,
    bint fortranStyle=True
):
    '''
    Create and zero-init new ndarray of specified shape and data type
    (up to two dimensions).
    '''
    cdef np.npy_intp shape[2]
    shape[0] = numN
    shape[1] = numM

    return np.PyArray_ZEROS(
        dims if dims < 2 else 2,    # Nr. Dimensions
        & shape[0],                 # Sizes of Dimensions
        dtype,                      # Data Type of elements
        fortranStyle                # FORTRAN-style result
    )


################################################## _arrEmpty()
cpdef np.ndarray _arrEmpty(
    int dims,
    intsize numN,
    intsize numM,
    ntype dtype,
    bint fortranStyle=True
):
    '''
    Create an empty ndarray of specified shape and data type
    (up to two dimensions)
    '''
    cdef np.npy_intp shape[2]
    shape[0] = numN
    shape[1] = numM

    return np.PyArray_EMPTY(
        dims if dims < 2 else 2,    # Nr. Dimensions
        & shape[0],                 # Sizes of Dimensions
        dtype,                      # Data Type of elements
        fortranStyle                # FORTRAN-style result
    )

################################################## _arrSqueeze1DF()
cdef np.ndarray _arrSqueeze1D(
    object data,
    int flags
):
    '''
    Return a squeezed, at least 1D version of the given data structure and
    consider special flags for the initial object conversion.
    '''
    cdef np.ndarray arrResult = np.PyArray_FROM_O(
        np.PyArray_Squeeze(np.PyArray_FROM_OF(data, flags))
    )

    return (arrResult if arrResult.ndim >= 1 else
            _arrReshape(arrResult, 1, arrResult.size, 0, np.NPY_ANYORDER))


################################################## _arrSqueeze1D()
cpdef np.ndarray _arrSqueeze(
    object data
):
    '''
    Return a squeezed, at least 1D version of the given data structure.
    '''
    return _arrSqueeze1D(data, 0)


################################################## _arrCopy1D()
cpdef np.ndarray _arrSqueezedCopy(
    object data
):
    '''
    Return a squeezed, at least 1D copy of the given data structure.
    '''
    return _arrSqueeze1D(data, np.NPY_ENSURECOPY)


################################################## _arrReshape()
cpdef np.ndarray _arrReshape(
    np.ndarray arr,
    int dims,
    intsize numN,
    intsize numM,
    np.NPY_ORDER order
):
    cdef np.PyArray_Dims shape2D
    cdef np.npy_intp shape[2]
    shape2D.ptr = &shape[0]
    shape2D.len = dims if dims < 2 else 2
    shape[0] = numN
    shape[1] = numM

    return np.PyArray_Newshape(arr, &shape2D, order)


################################################## _arrResize()
cpdef bint _arrResize(
    np.ndarray arr,
    int dims,
    intsize numN,
    intsize numM,
    np.NPY_ORDER order
):
    cdef np.PyArray_Dims shape2D
    cdef np.npy_intp shape[2]
    shape2D.ptr = &shape[0]
    shape2D.len = dims if dims < 2 else 2
    shape[0] = numN
    shape[1] = numM

    return np.PyArray_Resize(arr, &shape2D, False, order) is None


################################################## _arrCopyExt()
cpdef np.ndarray _arrCopyExt(
    np.ndarray arr,
    ntype dtype,
    int flags
):
    return np.PyArray_FROM_OTF(arr, dtype, flags)


################################################## _arrForceType()
cpdef np.ndarray _arrForceType(
    np.ndarray arr,
    ntype typeArr
):
    return arr if (np.PyArray_TYPE(arr) == typeArr) \
        else np.PyArray_FROM_OT(arr, typeArr)


################################################## _arrForceAlignment()
cpdef np.ndarray _arrForceAlignment(
    np.ndarray arr,
    int flags,
    bint fortranStyle=True
):
    if np.PyArray_ISONESEGMENT(arr) and \
       (np.PyArray_ISFORTRAN(arr) == fortranStyle) and \
       np.PyArray_ISCONTIGUOUS(arr):
        return arr

    if fortranStyle:
        flags += np.NPY_F_CONTIGUOUS
    else:
        flags += np.NPY_C_CONTIGUOUS

    return np.PyArray_FROM_OF(arr, np.NPY_OWNDATA + np.NPY_ENSUREARRAY + flags)


################################################## _arrForceTypeAlignment()
cpdef np.ndarray _arrForceTypeAlignment(
    np.ndarray arr,
    ntype typeArr,
    int flags,
    bint fortranStyle=True
):
    if (np.PyArray_TYPE(arr) == typeArr) and \
            np.PyArray_ISONESEGMENT(arr) and \
            (np.PyArray_ISFORTRAN(arr) == fortranStyle) and \
            np.PyArray_ISCONTIGUOUS(arr):
        return arr

    if fortranStyle:
        flags += np.NPY_F_CONTIGUOUS
    else:
        flags += np.NPY_C_CONTIGUOUS

    return np.PyArray_FROM_OTF(arr, typeArr,
                               np.NPY_OWNDATA + np.NPY_ENSUREARRAY + flags)


################################################## fused-type conjugate
cdef void _conjugateMV1(
    TYPE_COMPLEX[:] input,
    TYPE_COMPLEX[:] output
):
    '''
    Conjugate contents of memoryview in input and write to output.
    If the memoryview data type is not complex return False, indicating
    no processing of input and changes to output.
    '''
    cdef intsize nn, N = input.shape[0]
    # conjugate elements from input, write to output
    for nn in range(N):
        output[nn].real = input[nn].real
        output[nn].imag = -input[nn].imag


cdef void _conjugateMV2(
    TYPE_COMPLEX[:, :] input,
    TYPE_COMPLEX[:, :] output
):
    '''
    Conjugate contents of memoryview in input and write to output.
    If the memoryview data type is not complex return False, indicating
    no processing of input and changes to output.
    '''
    cdef intsize nn, mm, N = input.shape[0], M = input.shape[1]
    # conjugate elements from input, write to output
    for mm in range(M):
        for nn in range(N):
            output[nn, mm].real = input[nn, mm].real
            output[nn, mm].imag = -input[nn, mm].imag


cpdef np.ndarray _conjugate(np.ndarray arr):
    '''
    Conjugate numpy ndarray non-destructively.
    Return a conjugated copy if complex input or original data view if real.
    '''
    # determine type of array
    typeArr = typeSelection[np.PyArray_TYPE(arr)]
    if typeArr == TYPE_INVALID:
        raise NotImplementedError("Type %d not supported." % (typeArr))

    # check if complex. If not, abort
    if not ((typeArr == TYPE_COMPLEX64) or (typeArr == TYPE_COMPLEX128)):
        return arr

    # check if unsupported data type
    if typeArr == TYPE_INVALID:
        raise NotImplementedError("Type %d not supported." % (typeArr))

    # create empty output ndarray with equal shape as input (FORTRAN-style)
    cdef int numDims = arr.ndim
    cdef np.ndarray arrOutput = _arrEmpty(
        numDims,
        arr.shape[0],
        arr.shape[1] if numDims > 1 else 1,
        typeInfo[typeArr].numpyType
    )

    # perform conjugation
    if numDims == 1:
        if typeArr == TYPE_COMPLEX64:
            _conjugateMV1[np.complex64_t](arr, arrOutput)
        elif typeArr == TYPE_COMPLEX128:
            _conjugateMV1[np.complex128_t](arr, arrOutput)
    elif numDims == 2:
        if typeArr == TYPE_COMPLEX64:
            _conjugateMV2[np.complex64_t](arr, arrOutput)
        elif typeArr == TYPE_COMPLEX128:
            _conjugateMV2[np.complex128_t](arr, arrOutput)

    return arrOutput


################################################## in-place fused-type conjugate

cdef void _conjInplaceCore(
    np.ndarray arr,
    TYPE_COMPLEX typeArr
):
    '''
    Conjugate a raw data buffer of type TYPE_COMPLEX and cnt elements,
    assuming to be consecutive in memory
    '''
    cdef intsize ii, cnt = np.PyArray_SIZE(arr)
    cdef TYPE_COMPLEX *pData = <TYPE_COMPLEX *> arr.data

    # conjugate elements from input, write to output
    for ii in range(cnt):
        pData[ii].imag = -pData[ii].imag


cpdef bint _conjugateInplace(np.ndarray arr):
    '''
    Conjugate numpy ndarray in input and write to output.
    Performs as a wrapper for ndarrays
    '''
    # determine type of array
    cdef ftype typeArr = getFusedType(arr)

    # check if complex. If so, conjugate.
    if typeArr == TYPE_COMPLEX64:
        _conjInplaceCore[np.complex64_t](arr, typeArr)
        return True
    elif typeArr == TYPE_COMPLEX128:
        _conjInplaceCore[np.complex128_t](arr, typeArr)
        return True
    elif typeArr == TYPE_INVALID:
        raise NotImplementedError("Type %d not supported." % (typeArr))

    return False


################################################## in-place fused-type conjugate
cdef np.float64_t _norm(
    TYPE_ALL *vec,
    intsize N
):
    '''Compute the norm of the vector `vec`.'''
    cdef intsize nn
    cdef TYPE_ALL val
    cdef np.float64_t norm = 0.

    for nn in range(N):
        val = vec[nn]
        if (TYPE_ALL == np.complex64_t) or (TYPE_ALL == np.complex128_t):
            norm += val.real * val.real
            norm += val.imag * val.imag
        else:
            norm += val * val

    return norm


cdef np.float64_t _normMV(TYPE_ALL[:] vec):
    '''Compute the norm of the vector `vec`.'''
    cdef intsize nn, N = vec.shape[0]
    cdef TYPE_ALL val
    cdef np.float64_t norm = 0.

    for nn in range(N):
        val = vec[nn]
        if (TYPE_ALL == np.complex64_t) or (TYPE_ALL == np.complex128_t):
            norm += val.real * val.real + val.imag * val.imag
        else:
            norm += val * val

    return norm


cdef TYPE_ALL _corrMV(
    TYPE_ALL[:] vec1,
    TYPE_ALL[:] vec2
):
    '''Compute the correlation of the vectors `vec1` and `vec2`.'''
    cdef intsize nn, N = vec1.shape[0]
    cdef TYPE_ALL corr = 0

    for nn in range(N):
        corr = vec1[nn] * vec2[nn]

    return corr


################################################## fused-type multiply
cdef void _opCoreI(
    np.ndarray arrIn,
    np.ndarray arrOp,
    np.ndarray arrOut,
    TYPE_IN_I tIn,
    TYPE_OP_I tOp,
    TYPE_INT tOut,
    OP_MODE mode,
    intsize param
):
    '''
    '''
    # never use arrIn.shape[1] without checking dimension count of arrIn !
    cdef intsize kk, nn, mm
    cdef intsize N = arrIn.shape[0]
    cdef intsize NO = arrOut.shape[0], MO = arrOut.shape[1]

    # pointers to data field, assume ordering as contiguous columns
    cdef TYPE_IN_I * pIn = <TYPE_IN_I * > arrIn.data
    cdef TYPE_OP_I * pOp = <TYPE_OP_I * > arrOp.data
    cdef TYPE_INT * pOut = <TYPE_INT * > arrOut.data

    # pointers to a single column (arrOptor)
    cdef TYPE_IN_I * pInVec
    cdef TYPE_OP_I * pOpVec
    cdef TYPE_INT * pOutVec

    # memoryviews
    cdef TYPE_OP_I[:] mvOp

    if mode == MODE_MUL:
        # element-wise multiplication of arrIn and arrOp in m-Dimension
        # arrOp is assumed to be a one-dimensional vector
        for mm in range(MO):
            # possible issue with strides for non-aligned data
            pInVec = &(pIn[N * mm])
            pOutVec = &(pOut[N * mm])
            for nn in range(N):
                pOutVec[nn] = pInVec[nn] * pOp[nn]
    elif mode == MODE_DOTROW:
        # matrix dot-product or arrOp and arrIn.

        if (arrOp.ndim > 1) or (arrOp.shape[0] != N):
            raise ValueError("Dimension mismatch in _dotRowOffset")

        # iterate over columns of input / output data
        mvOp = arrOp
        for mm in range(MO):
            pOutVec = &(pOut[NO * mm + param])
            pInVec = &(pIn[N * mm])

            pOutVec[0] = pInVec[0] * mvOp[0]
            for nn in range(1, N):
                pOutVec[0] += pInVec[nn] * mvOp[nn]


cdef void _opCoreF(
    np.ndarray arrIn,
    np.ndarray arrOp,
    np.ndarray arrOut,
    TYPE_IN_R tIn,
    TYPE_OP_R tOp,
    TYPE_REAL tOut,
    OP_MODE mode,
    intsize param
):
    '''
    '''
    # never use arrIn.shape[1] without checking dimension count of arrIn !
    cdef intsize kk, nn, mm
    cdef intsize N = arrIn.shape[0]
    cdef intsize NO = arrOut.shape[0], MO = arrOut.shape[1]

    # pointers to data field, assume ordering as contiguous columns
    cdef TYPE_IN_R * pIn = <TYPE_IN_R * > arrIn.data
    cdef TYPE_OP_R * pOp = <TYPE_OP_R * > arrOp.data
    cdef TYPE_REAL * pOut = <TYPE_REAL * > arrOut.data

    # pointers to a single column (arrOptor)
    cdef TYPE_IN_R * pInVec
    cdef TYPE_OP_R * pOpVec
    cdef TYPE_REAL * pOutVec

    # memoryviews
    cdef TYPE_OP_R[:] mvOp

    if mode == MODE_MUL:
        # element-wise multiplication of arrIn and arrOp in m-Dimension
        # arrOp is assumed to be a one-dimensional vector
        for mm in range(MO):
            # possible issue with strides for non-aligned data
            pInVec = &(pIn[N * mm])
            pOutVec = &(pOut[N * mm])
            for nn in range(N):
                pOutVec[nn] = pInVec[nn] * pOp[nn]
    elif mode == MODE_DOTROW:
        # matrix dot-product or arrOp and arrIn.

        if (arrOp.ndim > 1) or (arrOp.shape[0] != N):
            raise ValueError("Dimension mismatch in _dotRowOffset")

        # iterate over columns of input / output data
        mvOp = arrOp
        for mm in range(MO):
            pOutVec = &(pOut[NO * mm + param])
            pInVec = &(pIn[N * mm])

            pOutVec[0] = pInVec[0] * mvOp[0]
            for nn in range(1, N):
                pOutVec[0] += pInVec[nn] * mvOp[nn]


cdef void _opCoreC(
    np.ndarray arrIn,
    np.ndarray arrOp,
    np.ndarray arrOut,
    TYPE_IN tIn,
    TYPE_OP tOp,
    TYPE_COMPLEX tOut,
    OP_MODE mode,
    intsize param
):
    '''
    '''
    # never use arrIn.shape[1] without checking dimension count of arrIn !
    cdef intsize kk, nn, mm
    cdef intsize N = arrIn.shape[0]
    cdef intsize NO = arrOut.shape[0], MO = arrOut.shape[1]

    # pointers to data field, assume ordering as contiguous columns
    cdef TYPE_IN * pIn = <TYPE_IN * > arrIn.data
    cdef TYPE_OP * pOp = <TYPE_OP * > arrOp.data
    cdef TYPE_COMPLEX * pOut = <TYPE_COMPLEX * > arrOut.data

    # pointers to a single column (arrOptor)
    cdef TYPE_IN * pInVec
    cdef TYPE_OP * pOpVec
    cdef TYPE_COMPLEX * pOutVec

    # memoryviews
    cdef TYPE_OP[:] mvOp

    if mode == MODE_MUL:
        # element-wise multiplication of arrIn and arrOp in m-Dimension
        # arrOp is assumed to be a one-dimensional vector
        for mm in range(MO):
            # possible issue with strides for non-aligned data
            pInVec = &(pIn[N * mm])
            pOutVec = &(pOut[N * mm])
            for nn in range(N):
                pOutVec[nn] = pInVec[nn] * pOp[nn]
    elif mode == MODE_DOTROW:
        # matrix dot-product or arrOp and arrIn.

        if (arrOp.ndim > 1) or (arrOp.shape[0] != N):
            raise ValueError("Dimension mismatch in _dotRowOffset")

        # iterate over columns of input / output data
        mvOp = arrOp
        for mm in range(MO):
            pOutVec = &(pOut[NO * mm + param])
            pInVec = &(pIn[N * mm])

            pOutVec[0] = pInVec[0] * mvOp[0]
            for nn in range(1, N):
                pOutVec[0] = pOutVec[0] + pInVec[nn] * mvOp[nn]


################################ dispatch integer results
cdef void _opI(
    np.ndarray arrIn,
    np.ndarray arrOp,
    np.ndarray arrOut,
    ftype tIn,
    ftype tOp,
    TYPE_INT tOut,
    OP_MODE mode,
    intsize param
):
    '''    '''
    # dispatch specialization of core routine according tOptor
    if tIn == TYPE_INT8:
        if tOp == TYPE_INT8:
            _opCoreI[np.int8_t, np.int8_t, TYPE_INT](
                arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
        elif tOp == TYPE_INT32:
            _opCoreI[np.int8_t, np.int32_t, TYPE_INT](
                arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
        elif tOp == TYPE_INT64:
            _opCoreI[np.int8_t, np.int64_t, TYPE_INT](
                arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tIn == TYPE_INT32:
        if tOp == TYPE_INT8:
            _opCoreI[np.int32_t, np.int8_t, TYPE_INT](
                arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
        elif tOp == TYPE_INT32:
            _opCoreI[np.int32_t, np.int32_t, TYPE_INT](
                arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
        elif tOp == TYPE_INT64:
            _opCoreI[np.int32_t, np.int64_t, TYPE_INT](
                arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tIn == TYPE_INT64:
        if tOp == TYPE_INT8:
            _opCoreI[np.int64_t, np.int8_t, TYPE_INT](
                arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
        elif tOp == TYPE_INT32:
            _opCoreI[np.int64_t, np.int32_t, TYPE_INT](
                arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
        elif tOp == TYPE_INT64:
            _opCoreI[np.int64_t, np.int64_t, TYPE_INT](
                arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)


################################ dispatch float results
cdef void _opRIn(
    np.ndarray arrIn,
    np.ndarray arrOp,
    np.ndarray arrOut,
    ftype tIn,
    TYPE_OP_R tOp,
    TYPE_REAL tOut,
    OP_MODE mode,
    intsize param
):
    '''     '''
    # dispatch specialization of core routine according tOptor
    if tIn == TYPE_INT8:
        _opCoreF[np.int8_t, TYPE_OP_R, TYPE_REAL](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tIn == TYPE_INT32:
        _opCoreF[np.int32_t, TYPE_OP_R, TYPE_REAL](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tIn == TYPE_INT64:
        _opCoreF[np.int64_t, TYPE_OP_R, TYPE_REAL](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tIn == TYPE_FLOAT32:
        _opCoreF[np.float32_t, TYPE_OP_R, TYPE_REAL](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tIn == TYPE_FLOAT64:
        _opCoreF[np.float64_t, TYPE_OP_R, TYPE_REAL](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)


cdef void _opR(
    np.ndarray arrIn,
    np.ndarray arrOp,
    np.ndarray arrOut,
    ftype tIn,
    ftype tOp,
    TYPE_REAL tOut,
    OP_MODE mode,
    intsize param
):
    '''    '''
    # dispatch specialization of core routine according tOptor
    if tOp == TYPE_INT8:
        _opRIn[np.int8_t, TYPE_REAL](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tOp == TYPE_INT32:
        _opRIn[np.int32_t, TYPE_REAL](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tOp == TYPE_INT64:
        _opRIn[np.int64_t, TYPE_REAL](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tOp == TYPE_FLOAT32:
        _opRIn[np.float32_t, TYPE_REAL](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tOp == TYPE_FLOAT64:
        _opRIn[np.float64_t, TYPE_REAL](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)


################################ dispatch complex results
cdef void _opCIn(
    np.ndarray arrIn,
    np.ndarray arrOp,
    np.ndarray arrOut,
    ftype tIn,
    TYPE_OP tOp,
    TYPE_COMPLEX tOut,
    OP_MODE mode,
    intsize param
):
    '''    '''
    # dispatch specialization of core routine according tOptor
    if tIn == TYPE_INT8:
        _opCoreC[np.int8_t, TYPE_OP, TYPE_COMPLEX](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tIn == TYPE_INT32:
        _opCoreC[np.int32_t, TYPE_OP, TYPE_COMPLEX](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tIn == TYPE_INT64:
        _opCoreC[np.int64_t, TYPE_OP, TYPE_COMPLEX](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tIn == TYPE_FLOAT32:
        _opCoreC[np.float32_t, TYPE_OP, TYPE_COMPLEX](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tIn == TYPE_FLOAT64:
        _opCoreC[np.float64_t, TYPE_OP, TYPE_COMPLEX](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tIn == TYPE_COMPLEX64:
        _opCoreC[np.complex64_t, TYPE_OP, TYPE_COMPLEX](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tIn == TYPE_COMPLEX128:
        _opCoreC[np.complex128_t, TYPE_OP, TYPE_COMPLEX](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)


cdef void _opC(
    np.ndarray arrIn,
    np.ndarray arrOp,
    np.ndarray arrOut,
    ftype tIn,
    ftype tOp,
    TYPE_COMPLEX tOut,
    OP_MODE mode,
    intsize param
):
    '''    '''
    # dispatch specialization of core routine according tOptor
    if tOp == TYPE_INT8:
        _opCIn[np.int8_t, TYPE_COMPLEX](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tOp == TYPE_INT32:
        _opCIn[np.int32_t, TYPE_COMPLEX](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tOp == TYPE_INT64:
        _opCIn[np.int64_t, TYPE_COMPLEX](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tOp == TYPE_FLOAT32:
        _opCIn[np.float32_t, TYPE_COMPLEX](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tOp == TYPE_FLOAT64:
        _opCIn[np.float64_t, TYPE_COMPLEX](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tOp == TYPE_COMPLEX64:
        _opCIn[np.complex64_t, TYPE_COMPLEX](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tOp == TYPE_COMPLEX128:
        _opCIn[np.complex128_t, TYPE_COMPLEX](
            arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)


cdef void _op(
    np.ndarray arrIn,
    np.ndarray arrOp,
    np.ndarray arrOut,
    ftype tIn,
    ftype tOp,
    ftype tOut,
    OP_MODE mode,
    intsize param
):
    # dispatch specialization of core routines according typeData
    if tOut == TYPE_INT8:
        _opI[np.int8_t](arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tOut == TYPE_INT32:
        _opI[np.int32_t](arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tOut == TYPE_INT64:
        _opI[np.int64_t](arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tOut == TYPE_FLOAT32:
        _opR[np.float32_t](arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tOut == TYPE_FLOAT64:
        _opR[np.float64_t](arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tOut == TYPE_COMPLEX64:
        _opC[np.complex64_t](arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)
    elif tOut == TYPE_COMPLEX128:
        _opC[np.complex128_t](arrIn, arrOp, arrOut, tIn, tOp, tOut, mode, param)

################################ entry point: _multiply()

cpdef _multiply(
    np.ndarray arrIn,
    np.ndarray arrOp,
    np.ndarray arrOut,
    ftype tIn,
    ftype tOp,
    ftype tOut
):
    '''
    '''
    _op(arrIn, arrOp, arrOut, tIn, tOp, tOut, MODE_MUL, 0)


################################ entry point: _dotRow()

cpdef _dotSingleRow(
    np.ndarray arrIn,
    np.ndarray arrOp,
    np.ndarray arrOut,
    ftype tIn,
    ftype tOp,
    ftype tOut,
    intsize iRow
):
    _op(arrIn, arrOp, arrOut, tIn, tOp, tOut, MODE_DOTROW, iRow)
