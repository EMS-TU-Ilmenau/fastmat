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

import numpy as np
cimport numpy as np

from .Matrix cimport Matrix
from .core.types cimport *
from .core.cmath cimport _arrZero

cdef class Eye(Matrix):
    r"""

    For :math:`x \in \mathbb{C}^n` we

    map .. math::
        x \mapsto  x.

    note::
        Eye.forward(x) returns the exact same object as the given input array x.
        Make sure to issue an explicit .copy() in case you need it!

    The identity matrix only needs the dimension :math:`n` of the vectors it
    acts on.

    >>> # import the package
    >>> import fastmat
    >>> # set the parameter
    >>> n = 10
    >>> # construct the identity
    >>> I = fastmat.Eye(n)

    This yields the identity matrix :math:`I_{10}` with dimension :math:`10`.
    """

    def __init__(self, numN):
        '''
        Initialize Matrix instance with its dimensions.

        Generated a [numN x numM] matrix of small integers
        '''
        self._initProperties(numN, numN, np.int8)

    cpdef np.ndarray _getArray(Eye self):
        '''
        Return an explicit representation of the matrix as numpy-array.
        '''
        return np.eye(self.numN, dtype=self.dtype)

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        cdef np.ndarray arrRes

        arrRes = _arrZero(1, self.numN, 1, self._info.dtype[0].typeNum)
        arrRes[idx] = 1

        return arrRes

    cpdef np.ndarray _getRow(self, intsize idx):
        return self._getCol(idx)

    cpdef object _getLargestSV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return 1.

    cpdef object _getLargestEV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return 1.

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        return 1 if (idxN == idxM) else 0

    cpdef Matrix _getNormalized(self):
        return self

    cpdef Matrix _getGram(self):
        return self

    cpdef Matrix _getT(self):
        return self

    cpdef Matrix _getH(self):
        return self

    cpdef Matrix _getConj(self):
        return self

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        return (0., 0.)

    ############################################## class forward / backward
    cpdef np.ndarray _forward(Eye, np.ndarray arrX):
        '''Calculate the forward transform of this matrix'''
        return arrX

    cpdef np.ndarray _backward(Eye self, np.ndarray arrX):
        '''Calculate the backward transform of this matrix'''
        return arrX

    ############################################## class reference
    cpdef np.ndarray _reference(Eye self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        return np.eye(self.numN, dtype=self.dtype)

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST
        return {
            TEST.COMMON: {
                TEST.NUM_N      : 35,
                TEST.NUM_M      : TEST.NUM_N,
                TEST.OBJECT     : Eye,
                TEST.INITARGS   : [TEST.NUM_N]
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c: Eye(c)),
            },
            BENCH.FORWARD: {},
            BENCH.SOLVE: {
                BENCH.FUNC_GEN  : (lambda c: Eye(c))
            },
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c: Eye(2 ** c)),
            }
        }

    def _getDocumentation(self):
        return ""
