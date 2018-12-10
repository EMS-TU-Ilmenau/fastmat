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

from ..core.types cimport *
from ..Matrix cimport Matrix
from .Algorithm cimport Algorithm

################################################## class Algorithm
cdef class OMP(Algorithm):
    cdef public intsize numN
    cdef public intsize numM
    cdef public intsize numL
    cdef public Matrix fmatA
    cdef public Matrix fmatC
    cdef public np.ndarray arrB
    cdef public np.ndarray arrX
    cdef public np.ndarray arrXtmp
    cdef public np.ndarray arrResidual
    cdef public np.ndarray arrSupport
    cdef public np.ndarray matPinv
    cdef public np.ndarray arrA
    cdef public np.ndarray v2
    cdef public np.ndarray v2n
    cdef public np.ndarray v2y
    cdef public np.ndarray newCols
    cdef public np.ndarray arrC
    cdef public np.ndarray newIndex
    cdef public intsize numMaxSteps
    cdef public intsize numStep
