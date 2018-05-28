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

from .types cimport *

cimport numpy as np

################################################## math routines

ctypedef enum OP_MODE:
    MODE_MUL
    MODE_DOTROW

cdef np.float64_t _norm(TYPE_ALL *, intsize)
cdef np.float64_t _normMV(TYPE_ALL[:])
cdef TYPE_ALL _corrMV(TYPE_ALL[:], TYPE_ALL[:])


################################################## complexity estimation
cdef int _findFFTFactors(int, int, int, int)
cpdef intsize _findOptimalFFTSize(intsize, int)
cpdef float _getFFTComplexity(intsize)

################################################## Array generation routines
cpdef np.ndarray _arrZero(int, intsize, intsize, ntype, bint fortranStyle=?)
cpdef np.ndarray _arrEmpty(int, intsize, intsize, ntype, bint fortranStyle=?)

cpdef np.ndarray _arrSqueeze(object)
cpdef np.ndarray _arrSqueezedCopy(object)
cpdef np.ndarray _arrReshape(np.ndarray, int, intsize, intsize, np.NPY_ORDER)
cpdef bint _arrResize(np.ndarray, int, intsize, intsize, np.NPY_ORDER)

################################################## Array formatting
cpdef np.ndarray _arrCopyExt(np.ndarray, ntype, int)
cpdef np.ndarray _arrForceType(np.ndarray, ntype)
cpdef np.ndarray _arrForceAlignment(np.ndarray, int, bint fortranStyle=?)
cpdef np.ndarray _arrForceTypeAlignment(np.ndarray, ntype, int,
                                        bint fortranStyle=?)

cpdef bint _conjugateInplace(np.ndarray)
cpdef np.ndarray _conjugate(np.ndarray)
cpdef _multiply(np.ndarray, np.ndarray, np.ndarray, ftype, ftype, ftype)
cpdef _dotSingleRow(np.ndarray, np.ndarray, np.ndarray,
                    ftype, ftype, ftype, intsize)
