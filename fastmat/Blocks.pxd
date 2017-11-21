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

################################################################################
################################################## class Blocks
cdef class Blocks(Matrix):

    ############################################## class variables
    cdef public tuple _rows                      # list of list of Matrix terms
    #                                            # (array of matrix terms)
    cdef public tuple _cols                      # same as _rows, transposed
    cdef public tuple _rowN                      # heights of rows
    cdef public tuple _colM                      # widths of columns
    cdef public intsize _numRows                 # Row-count of term array
    cdef public intsize _numCols                 # Column-count of term array

    ############################################## class methods
    cpdef _forwardC(self, np.ndarray, np.ndarray, ftype, ftype)
    cpdef _backwardC(self, np.ndarray, np.ndarray, ftype, ftype)

    cpdef np.ndarray _reference(self)
