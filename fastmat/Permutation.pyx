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

cdef class Permutation(Matrix):

    r"""

    For a given permutation :math:`\sigma \in S_n` and a vector
    :math:`x \in \mathbb{C}^n` we map

    .. math::
        x \mapsto \left(x_{\sigma(i)}\right)_{i = 1}^n.

    >>> # import the package
    >>> import fastmat
    >>>
    >>> # set the permutation
    >>> sigma = np.array([3,1,2,0])
    >>>
    >>> # construct the identity
    >>> P = fastmat.Permutation(sigma)

    .. math::
        J = \begin{bmatrix}
        0 & 0 & 0 & 1 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        1 & 0 & 0 & 0
        \end{bmatrix}
    """

    property sigma:
        r"""Return the defining permutation

        *(read only)*
        """

        def __get__(self):
            return self._sigma

    def __init__(self, sigma):
        '''
        Initialize Matrix instance with its dimensions.
        '''
        numN = sigma.shape[0]

        if not np.allclose(np.sort(sigma), np.arange(numN)):
            raise ValueError("Not a permutation.")

        self._sigma = sigma
        self._tau = np.argsort(sigma)

        self._initProperties(numN, numN, np.int8)

    ############################################## class property override
    cpdef object _getItem(self, intsize idxN, intsize idxM):
        return 1 if (self._sigma[idxN] == idxM) else 0

    cpdef np.ndarray _getArray(self):
        return np.eye(self.numN, dtype=self.dtype)[self.sigma, :]

    cpdef object _getLargestSV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return 1.

    cpdef object _getLargestEV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return 1.

    cpdef Matrix _getNormalized(self):
        return self

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        return (self.numN, self.numM)

    ############################################## class forward / backward
    cpdef np.ndarray _forward(self, np.ndarray arrX):
        '''Calculate the forward transform of this matrix'''
        return arrX[self._sigma, :]

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        '''Calculate the backward transform of this matrix'''
        return arrX[self._tau, :]

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        return np.eye(self.numN, dtype=self.dtype)[self._sigma, :]

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST
        return {
            TEST.COMMON: {
                TEST.NUM_N      : 35,
                TEST.NUM_M      : TEST.NUM_N,
                TEST.OBJECT     : Permutation,
                TEST.INITARGS   : (lambda param :
                                   [np.random.permutation(param[TEST.NUM_N])])
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c:
                                   Permutation(np.random.permutation(c)))
            },
            BENCH.FORWARD: {},
            BENCH.SOLVE: {},
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c:
                                   Permutation(np.random.permutation(2 ** c)))
            }
        }

    def _getDocumentation(self):
        return ""
