# -*- coding: utf-8 -*-
# cython: boundscheck=False, wraparound=False
'''
  demo/matrixSAFT.py
 -------------------------------------------------- part of the fastmat demos

  Demonstration on how to use fastMat for SAFT.


  Author      : wcw
  Introduced  :
 ------------------------------------------------------------------------------

   Copyright 2016 Sebastian Semper, Christoph Wagner
       https://www.tu-ilmenau.de/it-ems/

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
'''
# import numpy functionality
import numpy as np
cimport numpy as np

ctypedef fused DATATYPE:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.float32_t
    np.float64_t
    np.complex64_t
    np.complex128_t

###############################################################################
# cython-optimized fastmat matrix class core for efficient mask matrices
###############################################################################
cdef void _SaftMaskCore(
    DATATYPE[:, :] mvIn,
    DATATYPE[:, :] mvOut,
    int sizeItem,
    int numBlocks,
    tuple masks,
    bint backward
):
    cdef int mm, ee, ii, iiA, iiB, oo, vv, iiAe, iiBe, ooe, dC
    cdef int numElements, numMasks, numSize, numVecs
    cdef np.uint16_t[:] maskRow, maskCol
    cdef bint applyAbove

    cdef DATATYPE value

    numSize = mvIn.shape[0]
    numVecs = mvIn.shape[1]
    numMasks = len(masks)
    for mm in range(numMasks):
        # get mask and determine number of elements
        maskRow = masks[mm][:, 1 if backward else 0]
        maskCol = masks[mm][:, 0 if backward else 1]
        numElements = maskRow.shape[0]

        for bb in range(numBlocks):

            # compute elements on and below the main block diagonal
            # determine indices into input and output memoryviews
            oo = bb * sizeItem
            iiB = (bb - mm) * sizeItem
            iiA = (bb + mm) * sizeItem
            applyAbove = iiA < numSize and (iiA != iiB)

            for ee in range(numElements):
                ooe = oo + maskRow[ee]

                dC = maskCol[ee]
                iiAe = iiA + dC
                iiBe = iiB + dC

                for vv in range(numVecs):
                    value = mvIn[iiBe, vv] if iiBe >= 0 else 0
                    if applyAbove:
                        value += mvIn[iiAe, vv]
                    if value != 0:
                        mvOut[ooe, vv] = mvOut[ooe, vv] + value




################################################## type-dispatch
cpdef np.ndarray SaftMaskCore(
    np.ndarray arrIn,
    int sizeItem,
    int numBlocks,
    tuple masks,
    bint backward
):
    cdef np.ndarray arrOut = np.zeros((<object> arrIn).shape, dtype=arrIn.dtype)

    if arrIn.dtype == np.int8:
        _SaftMaskCore[np.int8_t](
            arrIn, arrOut, sizeItem, numBlocks, masks, backward)
    elif arrIn.dtype == np.int16:
        _SaftMaskCore[np.int16_t](
            arrIn, arrOut, sizeItem, numBlocks, masks, backward)
    elif arrIn.dtype == np.int32:
        _SaftMaskCore[np.int32_t](
            arrIn, arrOut, sizeItem, numBlocks, masks, backward)
    elif arrIn.dtype == np.int64:
        _SaftMaskCore[np.int64_t](
            arrIn, arrOut, sizeItem, numBlocks, masks, backward)
    elif arrIn.dtype == np.float32:
        _SaftMaskCore[np.float32_t](
            arrIn, arrOut, sizeItem, numBlocks, masks, backward)
    elif arrIn.dtype == np.float64:
        _SaftMaskCore[np.float64_t](
            arrIn, arrOut, sizeItem, numBlocks, masks, backward)
    elif arrIn.dtype == np.complex64:
        _SaftMaskCore[np.complex64_t](
            arrIn, arrOut, sizeItem, numBlocks, masks, backward)
    elif arrIn.dtype == np.complex128:
        _SaftMaskCore[np.complex128_t](
            arrIn, arrOut, sizeItem, numBlocks, masks, backward)
    else:
        raise TypeError("Data type '%s' not supported in SaftMask.forward()" %(
            str(arrIn.dtype)))

    return arrOut
