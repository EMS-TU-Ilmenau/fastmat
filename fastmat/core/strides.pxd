# -*- coding: utf-8 -*-
'''
  fastmat/core/strides.pxd
 -------------------------------------------------- part of the fastmat package

  Header file for cython fastmat striding math library. (types and funcs)

  Author      : wcw
  Introduced  : 2017-11-13
 ------------------------------------------------------------------------------

   Copyright 2016 Sebastian Semper, Christoph Wagner

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
import cython
import numpy as np
cimport numpy as np

from .types cimport *

cimport numpy as np

################################################## Stride object
ctypedef struct STRIDE_s:
    char *          base
    intsize         strideElement
    intsize         strideVector
    intsize         numElements
    intsize         numVectors
    np.uint8_t      sizeItem
    ftype           dtype

################################################## Basic stride operations
cdef void strideInit(STRIDE_s *, np.ndarray, np.uint8_t)
cdef void strideCopy(STRIDE_s *, STRIDE_s *)
cdef void strideSliceVectors(STRIDE_s *, intsize, intsize, intsize)
cdef void strideSliceElements(STRIDE_s *, intsize, intsize, intsize)
cdef void strideSubgridVector(STRIDE_s *,
                              intsize, intsize,
                              intsize, intsize, intsize, intsize)
cdef void strideFlipVectors(STRIDE_s *)
cdef void strideFlipElements(STRIDE_s *)

cdef stridePrint(STRIDE_s *, text=?)


################################################## Operations with strides
cdef opCopyVector(STRIDE_s *, intsize, STRIDE_s *, intsize)
cdef opZeroVector(STRIDE_s *, intsize)
cdef opZeroVectors(STRIDE_s *)
