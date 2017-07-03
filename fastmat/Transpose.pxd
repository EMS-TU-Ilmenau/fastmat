# -*- coding: utf-8 -*-
'''
  fastmat/Transpose.pxd
 -------------------------------------------------- part of the fastmat package

  Header file for Transpose class (structural description).

  Contains:
     - TpType        - Transposition type names
     - TpFlags        - Transposition flags for defining a transposed matrix
     - Transpose    - fastmat class for transposed matrices
     - TransposeFactory - Factory for generation of transposed matrices

  Author      : wcw
  Introduced  : 2016-09-30
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

from .helpers.types cimport *
from .Matrix cimport Matrix

################################################################################
################################################## enum TpType
ctypedef enum TpType:
    TRANSPOSE_NONE,
    TRANSPOSE_C,
    TRANSPOSE_H,
    TRANSPOSE_T


################################################################################
################################################## class TpFlags
cdef class TpFlags(object):

    ############################################## class variables
    cdef public bint applyH                      # apply hermitian transpose
    cdef public bint applyC                      # apply complex conjugate

    ############################################## class methods
    cpdef TpType decode(self)
    cpdef encode(self, TpType)
    cpdef apply(self, TpFlags)


################################################################################
################################################## class Transpose
cdef class Transpose(Matrix):

    ############################################## class variables
    cdef public TpFlags _flags                   # TpFlags to be applied
    cdef public Matrix _content                  # matrix to be transformed

    ############################################## class methods
    cpdef _forwardC(self, np.ndarray, np.ndarray, ftype, ftype)
    cpdef _backwardC(self, np.ndarray, np.ndarray, ftype, ftype)

    cpdef np.ndarray _forward(self, np.ndarray)
    cpdef np.ndarray _backward(self, np.ndarray)
    cpdef np.ndarray _reference(self)


################################################################################
################################################## factory TransposeFactory()
cpdef Matrix TransposeFactory(Matrix, TpType)
