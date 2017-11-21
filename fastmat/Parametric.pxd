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

ctypedef object(*function)(object, object)

################################################################################
################################################## class Hadamard
cdef class Parametric(Matrix):

    ############################################## class variables
    cdef public np.ndarray  _vecX                # support vector of X-dimension
    cdef public np.ndarray  _vecY                # support vector of Y-dimension
    cdef public object      _fun                 # Parameterizing function

    cdef object      _funDtype                   # element function data type
    cdef bint        _rangeAccess                # if True, _fun will be called
    #                                            # for every element separately
    #                                            # (no range access calls)

    ############################################## class methods
    cdef void _core(self, np.ndarray, np.ndarray, ftype, ftype, ftype, bint)

    cpdef _forwardC(self, np.ndarray, np.ndarray, ftype, ftype)
    cpdef _backwardC(self, np.ndarray, np.ndarray, ftype, ftype)

    cpdef np.ndarray _reference(self)
