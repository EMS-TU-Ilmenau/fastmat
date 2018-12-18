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

import cython
cimport numpy as np

from .Partial cimport Partial
from .core.types cimport *

################################################################################
################################################## class Circulant
cdef class Circulant(Partial):

    ############################################## class variables
    cdef public np.ndarray _tenC                 # matrix diagonal-entry tensor

    ############################################## class methods
    cpdef np.ndarray _preProcSlice(
        self, np.ndarray, int, np.ndarray, np.ndarray)
    cpdef np.ndarray _genArrS(self, np.ndarray, np.ndarray, bint verbose=?)
    cpdef np.ndarray _reference(self)
