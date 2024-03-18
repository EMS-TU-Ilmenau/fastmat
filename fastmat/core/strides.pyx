# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False

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

from libc.string cimport memcpy, memset
from libc.math cimport ceil

import numpy as np
cimport numpy as np

from .types cimport *

################################################################################
###  Basic stride operations
################################################################################
cdef void strideInit(
    STRIDE_s * stride, np.ndarray arr, np.uint8_t axis
) except *:
    """Initialize a stride operator from an existing `ndarray`.

    Parameters
    ----------
    stride : STRIDE_s *
        Pointer to the `STRIDE_s` data structure generated.

    arr : np.ndarray
        The `ndarray` to base the stride structure on.

    axis : np.uint8_t
        The axis that should be interpreted

    Returns
    -------
    None

    """
    if axis > 1:
        raise ValueError("Striding operations support 2D-arrays only.")

    stride[0].dtype         = getFusedType(arr.dtype)
    stride[0].base          = arr.data                  # array base pointer
    stride[0].strideElement = arr.strides[axis]         # the selected axis
    stride[0].strideVector  = arr.strides[axis ^ 1]     # the other axis
    stride[0].numElements   = arr.shape[axis]           # array element count
    stride[0].numVectors    = arr.shape[axis ^ 1]       # vector count
    stride[0].sizeItem      = np.PyArray_ITEMSIZE(arr)  # size of one item


cdef void strideCopy(STRIDE_s * strideDst, STRIDE_s * strideSrc):
    """Copy a stride structure

    Parameters
    ----------
    strideDst : STRIDE_s *
        A pointer to the destination `STRIDE_s` object

    strideSrc : STRIDE_s *
        A pointer to the source `STRIDE_s` object

    Returns
    -------
    None

    """
    memcpy(strideDst, strideSrc, sizeof(STRIDE_s))


cdef void strideSliceVectors(
    STRIDE_s * stride, intsize first, intsize last, intsize step
):
    """Modify a stride object by subselecting along the vector axis.

    Parameters
    ----------
    stride : STRIDE_s *
        A pointer to the `STRIDE_s` object to be modified.

    first : intsize
        The first index to start indexing from. Must be inside the open
        interval [0, length[. If negative, it is assumed to be the last index.

    last : intsize
        The last index to stop indexing at. Must be inside the open
        interval [0, length[. If negative, it is assumed to be the last index.

    step : intsize
        The stepping size as number of elements. Can be negative.

    Returns
    -------
    None
    """
    if first < 0:
        first = stride[0].numVectors - 1

    if last < 0:
        last = stride[0].numVectors - 1

    stride[0].base += first * stride[0].strideVector
    stride[0].numVectors = \
        ((last - first) if step == 0 else (last - first) // step) + 1
    stride[0].strideVector = stride[0].strideVector * step


cdef void strideSliceElements(
    STRIDE_s * stride, intsize first, intsize last, intsize step
):
    """Modify a stride object by subselecting along the element axis.

    Parameters
    ----------
    stride : STRIDE_s *
        A pointer to the `STRIDE_s` object to be modified.

    first : intsize
        The first index to start indexing from. Must be inside the open
        interval [0, length[. If negative, it is assumed to be the last index.

    last : intsize
        The last index to stop indexing at. Must be inside the open
        interval [0, length[. If negative, it is assumed to be the last index.

    step : intsize
        The stepping size as number of elements. Can be negative.

    Returns
    -------
    None
    """
    if first < 0:
        first = stride[0].numElements - 1

    if last < 0:
        last = stride[0].numElements - 1

    stride[0].base += first * stride[0].strideElement
    stride[0].numElements = \
        ((last - first) if step == 0 else (last - first) // step) + 1
    stride[0].strideElement = stride[0].strideElement * step


cdef void strideSubgrid(
    STRIDE_s * stride,
    intsize idxVector, intsize idxElement,
    intsize steppingVectors, intsize steppingElements,
    intsize numVectors, intsize numElements
):
    """Modify a stride object to span a linear vector into a new 2D grid.

    Parameters
    ----------
    stride : STRIDE_s *
        A pointer to the `STRIDE_s` object to be modified.

    idxVector : intsize
        The starting vector index in the stride, using the old indexing.
        Must be strictly inside the exis bounds.

    idxElement : intsize
        The starting element index in the stride, using the old indexing.
        Must be strictly inside the exis bounds.

    steppingVectors : intsize
        The subgrid step size along the new Vector axis.

    steppingElements : intsize
        The subgrid step size along the new Element axis.

    numVectors : intsize
        The new size along the new Vector axis.

    numElements : intsize
        The new size along the new Element axis.

    Returns
    -------
    None
    """
    # compute new base pointer (offset subgrid stride from current one)
    stride[0].base += (stride[0].strideElement * idxElement +
                       stride[0].strideVector * idxVector)

    # now compute new spans and strides for element and vector axes. Both of
    # them originate from the vector at position (idxElement, idxVector).
    # This allows mapping butterfly operation slicing directly to a 2D stride!
    stride[0].strideVector = stride[0].strideElement * steppingVectors
    stride[0].strideElement = stride[0].strideElement * steppingElements
    stride[0].numElements = numElements
    stride[0].numVectors = numVectors


cdef void strideFlipVectors(STRIDE_s * stride):
    """Flip the strided array along its the Vector axis.

    Parameters
    ----------
    stride : STRIDE_s *
        A pointer to the `STRIDE_s` object to be modified.

    Returns
    -------
    None
    """
    stride[0].base += (stride[0].numVectors - 1) * stride[0].strideVector
    stride[0].strideVector = -stride[0].strideVector


cdef void strideFlipElements(STRIDE_s * stride):
    """Flip the strided array along its the Element axis.

    Parameters
    ----------
    stride : STRIDE_s *
        A pointer to the `STRIDE_s` object to be modified.

    Returns
    -------
    None
    """
    stride[0].base += (stride[0].numElements - 1) * stride[0].strideElement
    stride[0].strideElement = -stride[0].strideElement


cdef void stridePrint(STRIDE_s * stride, text='') except *:
    """Print information about a strided array to stdout.

    Parameters
    ----------
    stride : STRIDE_s *
        A pointer to the `STRIDE_s` object to be printed.

    text : str
        Additional information to be appended in the output

    Returns
    -------
    type
        Description of returned object.

    """
    print("[%dx%d,%d] @ 0x%012X + %d * nn + %d * mm (%d Bytes) %s" %(
        stride[0].numElements, stride[0].numVectors, stride[0].dtype,
        <intsize> (stride[0].base),
        stride[0].strideElement, stride[0].strideVector, stride[0].sizeItem,
        ": %s" %(text, ) if len(text) > 0 else ""))


################################################################################
###  Operations with strides
################################################################################
cdef void opCopyVector(STRIDE_s * strideDst, intsize idxVectorDst,
                       STRIDE_s * strideSrc, intsize idxVectorSrc) except *:
    """Copy data from one strided array to another along their Element axes.

    Parameters
    ----------
    strideDst : STRIDE_s *
        A pointer to the destination strided array.

    idxVectorDst : intsize
        The index along the Vector axis to put the copied Element-axis data to.

    strideSrc : STRIDE_s *
        A pointer to the source strided array.

    idxVectorSrc : intsize
        The index along the Element axis to put the copied Vector-axis data to.

    Returns
    -------
    None
    """
    cdef intsize nn
    cdef intsize dstStride = strideDst[0].strideElement
    cdef intsize srcStride = strideSrc[0].strideElement
    cdef intsize numElements = strideDst[0].numElements
    cdef np.uint8_t sizeItem = strideDst[0].sizeItem

    # sanity check
    if ((strideDst[0].sizeItem != strideSrc[0].sizeItem or
         strideDst[0].numElements != strideSrc[0].numElements)):
        raise TypeError("Strides differ in vector length or element size.")

    # determine addresses
    cdef char *ptrSrc = (strideSrc[0].base +
                         strideSrc[0].strideVector * idxVectorSrc)
    cdef char *ptrDst = (strideDst[0].base +
                         strideDst[0].strideVector * idxVectorDst)

    if dstStride == srcStride == sizeItem:
        # element-contiguous in both strides
        memcpy(ptrDst, ptrSrc, dstStride * numElements)
    elif sizeItem == 8:
        for nn in range(numElements):
            (<np.int64_t *> ptrDst)[0] = (<np.int64_t *> ptrSrc)[0]
            ptrDst += dstStride
            ptrSrc += srcStride
    elif sizeItem == 4:
        for nn in range(numElements):
            (<np.int32_t *> ptrDst)[0] = (<np.int32_t *> ptrSrc)[0]
            ptrDst += dstStride
            ptrSrc += srcStride
    elif sizeItem == 2:
        for nn in range(numElements):
            (<np.int16_t *> ptrDst)[0] = (<np.int16_t *> ptrSrc)[0]
            ptrDst += dstStride
            ptrSrc += srcStride
    elif sizeItem == 1:
        for nn in range(numElements):
            ptrDst[0] = ptrSrc[0]
            ptrDst += dstStride
            ptrSrc += srcStride
    else:
        for nn in range(numElements):
            memcpy(ptrDst, ptrSrc, sizeItem)
            ptrDst += dstStride
            ptrSrc += srcStride


cdef void opZeroVector(STRIDE_s * stride, intsize idxVector):
    """Zero all elements of a strided array belonging to a Vector axis index.

    Parameters
    ----------
    strideDst : STRIDE_s *
        A pointer to the strided array to be modified.

    idxVectorDst : intsize
        The index along the Vector axis, for which to zero all data elements.

    Returns
    -------
    None
    """
    cdef intsize nn
    cdef char *ptr = stride[0].base + stride[0].strideVector * idxVector
    cdef intsize strideElement = stride[0].strideElement
    cdef intsize numElements = stride[0].numElements
    cdef np.uint8_t sizeItem = stride[0].sizeItem

    if strideElement == sizeItem:
        # element-contiguous: easy memset call
        memset(ptr, 0, sizeItem * numElements)
    elif strideElement == -sizeItem:
        # flipped, but still element-contiguous: easy memset call
        memset(ptr - sizeItem * (numElements - 1), 0, sizeItem * numElements)

    # do it the long and manual way
    elif sizeItem == 8:
        for nn in range(numElements):
            (<np.int64_t *> ptr)[0] = 0
            ptr += strideElement
    elif sizeItem == 4:
        for nn in range(numElements):
            (<np.int32_t *> ptr)[0] = 0
            ptr += strideElement
    elif sizeItem == 2:
        for nn in range(numElements):
            (<np.int16_t *> ptr)[0] = 0
            ptr += strideElement
    elif sizeItem == 1:
        for nn in range(numElements):
            ptr[0] = 0
            ptr += strideElement
    else:
        for nn in range(numElements):
            memset(ptr, 0, sizeItem)
            ptr += strideElement


cdef void opZeroVectors(STRIDE_s * stride):
    """Zero all elements of a strided array along both axes.

    Parameters
    ----------
    stride : STRIDE_s *
        A pointer to the strided array to be modified.

    Returns
    -------
    None
    """
    cdef intsize nn, mm, sizeChunk
    cdef np.uint8_t sizeItem = stride[0].sizeItem
    cdef intsize numElements = stride[0].numElements
    cdef intsize numVectors = stride[0].numVectors
    cdef intsize strideElement = stride[0].strideElement
    cdef intsize strideVector = stride[0].strideVector
    cdef char *ptr = stride[0].base
    cdef char *ptrVector

    if ((abs(strideElement) == sizeItem and
         abs(strideVector) == numElements * sizeItem)):
        # both axes are contiguous: jackpot!
        sizeChunk = strideVector * numVectors
        if sizeChunk < 0:
            sizeChunk = abs(sizeChunk)
            ptr -= sizeChunk - sizeItem
        memset(ptr, 0, sizeChunk)
    elif abs(strideElement) == sizeItem:
        # element-contiguous: loop memset calls
        sizeChunk = strideElement * numElements
        if sizeChunk < 0:
            sizeChunk = abs(sizeChunk)
            ptr -= sizeChunk - sizeItem
        for mm in range(numVectors):
            memset(ptr, 0, sizeChunk)
            ptr += strideVector
    elif abs(strideVector) == sizeItem:
        # vector-contiguous: loop memset calls
        sizeChunk = strideVector * numVectors
        if sizeChunk < 0:
            sizeChunk = abs(sizeChunk)
            ptr -= sizeChunk - sizeItem
        for nn in range(numElements):
            memset(ptr, 0, sizeChunk)
            ptr += strideElement

    # do it the long and manual way
    elif sizeItem == 8:
        for mm in range(numVectors):
            ptrVector = ptr
            for nn in range(numElements):
                (<np.int64_t *> ptrVector)[0] = 0
                ptrVector += strideElement
            ptr += strideVector
    elif sizeItem == 4:
        for mm in range(numVectors):
            ptrVector = ptr
            for nn in range(numElements):
                (<np.int32_t *> ptrVector)[0] = 0
                ptrVector += strideElement
            ptr += strideVector
    elif sizeItem == 2:
        for mm in range(numVectors):
            ptrVector = ptr
            for nn in range(numElements):
                (<np.int16_t *> ptrVector)[0] = 0
                ptrVector += strideElement
            ptr += strideVector
    elif sizeItem == 1:
        for mm in range(numVectors):
            ptrVector = ptr
            for nn in range(numElements):
                ptrVector[0] = 0
                ptrVector += strideElement
            ptr += strideVector
    else:
        for mm in range(numVectors):
            ptrVector = ptr
            for nn in range(numElements):
                memset(ptrVector, 0, sizeItem)
                ptrVector += strideElement
            ptr += strideVector
