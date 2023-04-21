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

import numpy as np
cimport numpy as np

from .Matrix cimport Matrix
from .core.types cimport *
from .core.cmath cimport _arrSqueezedCopy

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

    def __init__(self, sigma, **options):
        '''
        Initialize a Permutation matrix instance.

        Parameters
        ----------
        sigma : :py:class:`numpy.ndarray`
            A 1d vector of type int mapping the row indices to column indices
            uniquely.

        **options : optional
            Additional keyworded arguments. Supports all optional arguments
            supported by :py:class:`fastmat.Matrix`.
        '''
        if not isinstance(sigma, np.ndarray):
            sigma = np.array(sigma)

        self._sigma = _arrSqueezedCopy(sigma)
        if (sigma.ndim != 1) or (self._sigma.ndim != 1):
            raise ValueError(
                "Diag: Definition vector must have exactly one dimension.")

        numRows = sigma.shape[0]
        if not np.allclose(np.sort(sigma), np.arange(numRows)):
            raise ValueError("Not a permutation.")

        self._tau = np.argsort(sigma)
        self._initProperties(numRows, numRows, np.int8, **options)

    ############################################## class property override
    cpdef object _getItem(self, intsize idxRow, intsize idxCol):
        return 1 if (self._sigma[idxRow] == idxCol) else 0

    cpdef np.ndarray _getArray(self):
        return np.eye(self.numRows, dtype=self.dtype)[self.sigma, :]

    cpdef object _getLargestSingularValue(self):
        return 1.

    cpdef object _getLargestEigenValue(self):
        return 1.

    cpdef np.ndarray _getColNorms(self):
        return np.ones((self.numCols, ), dtype=self.dtype)

    cpdef np.ndarray _getRowNorms(self):
        return np.ones((self.numRows, ), dtype=self.dtype)

    cpdef Matrix _getColNormalized(self):
        return self

    cpdef Matrix _getRowNormalized(self):
        return self

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        return (self.numRows, self.numCols)

    ############################################## class forward / backward
    cpdef np.ndarray _forward(self, np.ndarray arrX):
        return arrX[self._sigma, :]

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        return arrX[self._tau, :]

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        return np.eye(self.numRows, dtype=self.dtype)[self._sigma, :]

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST
        return {
            TEST.COMMON: {
                TEST.NUM_ROWS   : 35,
                TEST.NUM_COLS   : TEST.NUM_ROWS,
                TEST.OBJECT     : Permutation,
                TEST.INITARGS   : (
                    lambda param: [np.random.permutation(param[TEST.NUM_ROWS])]
                )
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
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c:
                                   Permutation(np.random.permutation(2 ** c)))
            }
        }
