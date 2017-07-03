# -*- coding: utf-8 -*-
'''
  fastmat/Matrix.pxd
 -------------------------------------------------- part of the fastmat package

  Header file for Matrix base class (structural description).


  Author      : wcw
  Introduced  : 2016-09-24
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
import numpy as np
cimport numpy as np

from .helpers.types cimport *

################################################## class Matrix
cdef class Matrix:

    ############################################## class variables
    cdef public np.ndarray  _data                # ndarray representing Matrix
    cdef public object      _forwardReferenceMatrix
    #                                            # ndarray representing Matrix
    #                                            # reference
    cdef public Matrix  _gram                    # cache for gram matrix
    cdef public Matrix  _normalized              # cache for normalized matrix
    cdef public object  _largestEV               # cache for largestEV
    cdef public object  _largestSV               # cache for largestSV
    cdef public Matrix  _T                       # cache for transpose matrix
    cdef public Matrix  _H                       # cache for adjunct matrix
    cdef public Matrix  _conj                    # cache for conjugate matrix

    cdef intsize     _numN                       # row-count of matrix
    cdef intsize     _numM                       # column-count of matrix
    cdef INFO_ARR_s  _info                       # fields representing matrix
    #                                            #  dtype - data type descriptor
    #                                            #  numN  - row count
    #                                            #  numM  - column count
    #                                            #  nDim  - dimension count
    cdef bint   _cythonCall                      # use _C() transforms in class
    cdef bint   _forceInputAlignment             # force alignment of input data
    #                                            # to be F-contiguous
    cdef bint   _widenInputDatatype              # widen input data type upfront
    cdef str    _tag                             # Description of matrix

    ############################################## class implementation methods
    cpdef np.ndarray _getCol(self, intsize)
    cpdef np.ndarray _getRow(self, intsize)
    cpdef object _getLargestEV(self, intsize, float, float, bint)
    cpdef object _getLargestSV(self, intsize, float, float, bint)
    cpdef Matrix _getT(self)
    cpdef Matrix _getH(self)
    cpdef Matrix _getConj(self)
    cpdef Matrix _getGram(self)
    cpdef Matrix _getNormalized(self)
    cpdef object _getItem(self, intsize, intsize)

    cpdef _forwardC(self, np.ndarray, np.ndarray, ftype, ftype)
    cpdef _backwardC(self, np.ndarray, np.ndarray, ftype, ftype)
    cpdef np.ndarray _forward(self, np.ndarray)
    cpdef np.ndarray _backward(self, np.ndarray)

    ############################################## class interface methods
    cpdef np.ndarray forward(self, np.ndarray)
    cpdef np.ndarray backward(self, np.ndarray)
    cpdef np.ndarray toarray(self)

    ############################################## class testing
    cpdef np.ndarray _reference(self)
    cpdef np.ndarray reference(self)
