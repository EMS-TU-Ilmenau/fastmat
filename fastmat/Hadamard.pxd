# -*- coding: utf-8 -*-
'''
  fastmat/Hadamard.pxd
 -------------------------------------------------- part of the fastmat package

  Header file for Hadamard class (structural description).


  Author      : wcw
  Introduced  : 2016-09-27
 ------------------------------------------------------------------------------

   Copyright 2016 Sebastian Semper, Christoph Wagner
       https://www.tu-ilmenau.de/ems/

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
cimport numpy as np

from .Matrix cimport Matrix
from .core.types cimport *

################################################################################
################################################## class Hadamard
cdef class Hadamard(Matrix):

    ############################################## class variables
    cdef int         _order                      # order of the hadamard matrix

    ############################################## class methods
    cdef void _coreLoop(self, TYPE_IN *, TYPE_IN *, intsize, intsize)
    cdef void _core(self, np.ndarray, np.ndarray, TYPE_IN)

    cpdef _forwardC(self, np.ndarray, np.ndarray, ftype, ftype)
    cpdef _backwardC(self, np.ndarray, np.ndarray, ftype, ftype)

    cpdef np.ndarray _reference(self)
