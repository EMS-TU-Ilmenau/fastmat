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
cimport numpy as np

from .Matrix cimport Matrix
from .core.types cimport *

ctypedef np.uint32_t lfsrReg_t

################################################################################
################################################## class LfsrConv
cdef class LFSRCirculant(Matrix):

    cdef public int         _regSize
    cdef public lfsrReg_t   _regTaps
    cdef public lfsrReg_t   _resetState

    cdef public np.ndarray  _vecC
    cdef public np.ndarray  _states

    cdef np.ndarray _getStates(self)
    cdef np.ndarray _getVecC(self)

    cdef void _core(self, np.ndarray, np.ndarray, bint, bint)
    cdef void _roll(self, np.ndarray, intsize)

    cpdef _forwardC(self, np.ndarray, np.ndarray, ftype, ftype)
    cpdef _backwardC(self, np.ndarray, np.ndarray, ftype, ftype)
