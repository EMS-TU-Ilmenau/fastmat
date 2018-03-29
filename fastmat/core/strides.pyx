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

################################################################################
###  Basic stride operations
################################################################################
cdef void strideInit(STRIDE_s *stride, np.ndarray arr, np.uint8_t axis):
    if axis > 1:
        raise ValueError("Striding operations support 2D-arrays only.")

    stride[0].dtype         = getFusedType(arr)
    if stride[0].dtype >= TYPE_INVALID:
        raise ValueError("Striding does not support given array data type.")

    stride[0].base          = arr.data                  # array base pointer
    stride[0].strideElement = arr.strides[axis]         # the selected axis
    stride[0].strideVector  = arr.strides[axis ^ 1]     # the other axis
    stride[0].numElements   = arr.shape[axis]           # array element count
    stride[0].numVectors    = arr.shape[axis ^ 1]       # vector count
    stride[0].sizeItem      = np.PyArray_ITEMSIZE(arr)  # size of one item


cdef void strideCopy(STRIDE_s *strideDst, STRIDE_s *strideSrc):
    memcpy(strideDst, strideSrc, sizeof(STRIDE_s))


cdef void strideSliceVectors(STRIDE_s *stride,
                             intsize start, intsize stop, intsize step):
    # CAUTION: NOT VERIFIED YET
    if start < 0:
        start = stride[0].numVectors

    if stop < 0:
        stop = stride[0].numVectors

    stride[0].base += start * stride[0].strideVector
    stride[0].numVectors = stop if step == 0 else (stop - start) / step
    stride[0].strideVector = stride[0].strideVector * step

cdef void strideSliceElements(STRIDE_s *stride,
                              intsize start, intsize stop, intsize step):
    if start < 0:
        start = stride[0].numElements

    if stop < 0:
        stop = stride[0].numElements

    stride[0].base += start * stride[0].strideElement
    stride[0].numElements = stop if step == 0 else (stop - start) / step
    stride[0].strideElement = stride[0].strideElement * step

cdef void strideSubgridVector(STRIDE_s *stride,
                              intsize idxVector, intsize idxElement,
                              intsize steppingElements, intsize numElements,
                              intsize steppingVectors, intsize numVectors):
    # CAUTION: NOT VERIFIED YET

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

cdef void strideFlipVectors(STRIDE_s *stride):
    stride[0].base += (stride[0].numVectors - 1) * stride[0].strideVector
    stride[0].strideVector = -stride[0].strideVector

cdef void strideFlipElements(STRIDE_s *stride):
    stride[0].base += (stride[0].numElements - 1) * stride[0].strideElement
    stride[0].strideElement = -stride[0].strideElement

cdef stridePrint(STRIDE_s *stride, text=''):
    print("[%dx%d,%d] @ 0x%012X + %d * nn + %d * mm (%d Bytes) %s" %(
        stride[0].numElements, stride[0].numVectors, stride[0].dtype,
        <intsize> (stride[0].base),
        stride[0].strideElement, stride[0].strideVector, stride[0].sizeItem,
        ": %s" %(text, ) if len(text) > 0 else ""))

################################################################################
###  Operations with strides
################################################################################
cdef opCopyVector(STRIDE_s *strideDst, intsize idxVectorDst,
                  STRIDE_s *strideSrc, intsize idxVectorSrc):
    # CAUTION: NOT VERIFIED YET

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


cdef opZeroVector(STRIDE_s *stride, intsize idxVector):
    # CAUTION: NOT VERIFIED YET
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
    elif sizeItem == 1:
        for nn in range(numElements):
            ptr[0] = 0
            ptr += strideElement
    else:
        for nn in range(numElements):
            memset(ptr, 0, sizeItem)
            ptr += strideElement

cdef opZeroVectors(STRIDE_s *stride):
    # CAUTION: NOT VERIFIED YET
    cdef intsize nn, mm, sizeChunk
    cdef np.uint8_t sizeItem = stride[0].sizeItem
    cdef intsize numElements = stride[0].numElements
    cdef intsize numVectors = stride[0].numVectors
    cdef intsize strideElement = stride[0].strideElement
    cdef intsize strideVector = stride[0].strideVector
    cdef char *ptr = stride[0].base
    cdef char *ptrVector

    if ((strideElement == sizeItem and
         strideVector == numElements * sizeItem)):
        # both axes are contiguous: jackpot!
        memset(ptr, 0, strideVector * numVectors)
    elif strideElement == sizeItem:
        # element-contiguous: loop memset calls
        sizeChunk = numElements * sizeItem
        for mm in range(numVectors):
            memset(ptr, 0, sizeChunk)
            ptr += strideVector
    elif strideVector == sizeItem:
        # vector-contiguous: loop memset calls
        sizeChunk = numVectors * sizeItem
        for nn in range(numElements):
            memset(ptr, 0, sizeChunk)
            ptr += strideElement

    # do it the long and manual way
    elif sizeItem == 8:
        for mm in range(numVectors):
            ptrVector = ptr + strideVector
            for nn in range(numElements):
                (<np.int64_t *> ptrVector)[0] = 0
                ptrVector += strideElement
    elif sizeItem == 4:
        for mm in range(numVectors):
            ptrVector = ptr + strideVector
            for nn in range(numElements):
                (<np.int32_t *> ptrVector)[0] = 0
                ptrVector += strideElement
    elif sizeItem == 1:
        for mm in range(numVectors):
            ptrVector = ptr + strideVector
            for nn in range(numElements):
                ptrVector[0] = 0
                ptrVector += strideElement
    else:
        for mm in range(numVectors):
            ptrVector = ptr + strideVector
            for nn in range(numElements):
                memset(ptrVector, 0, sizeItem)
                ptrVector += strideElement
