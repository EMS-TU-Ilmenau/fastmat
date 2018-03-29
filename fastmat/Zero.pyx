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

cimport numpy as np
import numpy as np

from .Matrix cimport Matrix
from .core.cmath cimport _arrZero
from .core.types cimport *


cdef class Zero(Matrix):

    r"""


    .. math::
        x \mapsto  0

    The zero matrix only needs the dimension :math:`n` of the vectors it acts
    on. It is very fast and very good!

    >>> import fastmat as fm
    >>>
    >>> # define the parameter
    >>> n = 10
    >>>
    >>> # construct the matrix
    >>> O = fm.Zero(n)
    """

    def __init__(self, numN, numM):
        '''Initialize Matrix instance with its dimensions'''
        # set properties of matrix
        self._initProperties(numN, numM, np.int8)

    ############################################## class property override
    cpdef np.ndarray _getArray(self):
        return _arrZero(2, self.numN, self.numM, self.numpyType)

    cpdef np.ndarray _getCol(self, intsize idx):
        return _arrZero(1, self.numN, 1, self.numpyType)

    cpdef np.ndarray _getRow(self, intsize idx):
        return _arrZero(1, self.numM, 1, self.numpyType)

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        return 0

    cpdef object _getLargestEV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return 0.

    cpdef object _getLargestSV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return 0.

    cpdef Matrix _getT(self):
        return Zero(self.numM, self.numN)

    cpdef Matrix _getH(self):
        return Zero(self.numM, self.numN)

    cpdef Matrix _getConj(self):
        return self

    cpdef Matrix _getGram(self):
        return Zero(self.numM, self.numM)

    cpdef Matrix _getNormalized(self):
        raise ValueError("A Zero Matrix cannot be normalized.")

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        return (self.numN, self.numM)

    ############################################## class forward / backward
    cpdef np.ndarray _forward(self, np.ndarray arrX):
        '''Calculate the forward transform of this matrix'''
        return _arrZero(
            arrX.ndim, self.numN, arrX.shape[1] if arrX.ndim > 1 else 1,
            getNumpyType(arrX))

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        '''Calculate the backward transform of this matrix'''
        return _arrZero(
            arrX.ndim, self.numM, arrX.shape[1] if arrX.ndim > 1 else 1,
            getNumpyType(arrX))

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        return np.zeros((self.numN, self.numM), dtype=self.dtype)

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                TEST.NUM_N      : 35,
                TEST.NUM_M      : TEST.Permutation([30, TEST.NUM_N]),
                TEST.OBJECT     : Zero,
                TEST.INITARGS   : [TEST.NUM_N, TEST.NUM_M],
                TEST.NAMINGARGS : dynFormat("%dx%d", TEST.NUM_N, TEST.NUM_M)
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN      : (lambda c: Zero(c, c)),
            },
            BENCH.FORWARD: {},
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN      : (lambda c: Zero(2 ** c, 2 ** c)),
            }
        }

    def _getDocumentation(self):
        return ""
