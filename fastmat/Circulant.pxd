# -*- coding: utf-8 -*-
'''
  fastmat/Circulant.pxd
 -------------------------------------------------- part of the fastmat package

  Header file for Circulant class (structural description).


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

from .Partial cimport Partial
from .helpers.types cimport *

################################################################################
################################################## class Circulant
cdef class Circulant(Partial):

    ############################################## class variables
    cdef public np.ndarray _vecC                 # matrix diagonal entry vector

    ############################################## class internals
    cdef void _roll(self, np.ndarray, intsize)

    ############################################## class methods
    cpdef np.ndarray toarray(self)
    cpdef np.ndarray _reference(self)
