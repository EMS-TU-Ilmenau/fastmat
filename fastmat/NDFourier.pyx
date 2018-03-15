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
from .Eye cimport Eye
from .core.cmath cimport *
from .core.strides cimport *

cdef class NDFourier(Matrix):
    r"""

    The :math:`d`-dimensional Fourier Transform on vectorized
    :math:`d`-dimensional data realizes the mapping

    .. math::
        x \mapsto \left(\mathcal{F}_{n_1} \otimes \dots \otimes
        \mathcal{F}_{n_d}\right) \cdot  x,

    where the :math:`{\mathcal{F}}_{n_i}` are Fourier matrices of order
    :math:`n_i` and :math:`\otimes` denotes the Kronecker product.

    >>> # import the package
    >>> import fastmat as fm
    >>>
    >>> # define parameter
    >>> n = 4
    >>>
    >>> # construct the matrix
    >>> F = fm.NDFourier(n, n)

    This yields a 2D Fourier matrix of size 16. As a library to provide the
    Fast Fourier Transform we used the one provided by NumPy [1]_.

    .. todo::
        - real valued transforms
        - chirp-z transform for all dimensions individually
    """

    property numD:
        r"""Return the number of dimensions of the Fourier Transform.

        *(read-only)*
        """

        def __get__(self):
            return self._numD

    property order:
        r"""Return the Order of the Fourier matrix.

        *(read-only)*
        """

        def __get__(self):
            return self._order

    def __init__(self, *order, **options):
        '''Initialize Matrix instance with a list of child matrices'''

        # store order of Fourier Transform
        self._order = np.copy(np.squeeze(np.array(order)))

        # store number of dimensions to transform over
        self._numD = len(order)

        # set properties of matrix
        self._initProperties(
            np.prod(self._order),
            np.prod(self._order),
            np.complex128
        )

    # cpdef np.ndarray _getArray(self):
    #     '''
    #     Return an explicit representation of the matrix as numpy-array.
    #     '''
    #     return np.fft.fftn(
    #             np.eye(self._order, dtype=self.dtype), axis=0)

    cpdef np.ndarray _getRow(self, intsize idx):
        return self._getCol(idx)

    cpdef object _getLargestSV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return np.sqrt(self.numN)

    cpdef object _getLargestEV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return np.sqrt(self._order)

    cpdef Matrix _getNormalized(self):
        return NDFourier(self._order) * (1. / np.sqrt(self.numN))

    cpdef Matrix _getGram(self):
        return Eye(self.numN) * self.numN

    # cpdef object _getItem(self, intsize idxN, intsize idxM):
    #     return np.exp(idxN * idxM * -2j * np.pi / self.order).astype(
    #         self.dtype)

    # ############################################## class property override
    # cpdef tuple _getComplexity(self):
    #     cdef float complexity = _getFFTComplexity(self._order)
    #     return (complexity, complexity + self._order)

    ############################################## class forward / backward
    cpdef np.ndarray _forward(self, np.ndarray arrX):
        '''Calculate the forward transform of this matrix'''

        # first we reshape the input to the size of the ND transform
        # and then we apply the transform along the last axis over the first
        # d axes and finally reshape everything back
        return np.fft.fftn(
            arrX.reshape(tuple(self._order) + (-1, )),
            axes=tuple(range(self._numD))
        ).reshape((-1, arrX.shape[1]))

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        '''Calculate the backward transform of this matrix'''

        # first we reshape the input to the size of the ND transform
        # and then we apply the transform along the last axis over the first
        # d axes and finally reshape everything back
        # since we do a backward and not the inverse, we have to apply a
        # scaling factor
        return np.fft.ifftn(
            arrX.reshape(tuple(self._order) + (-1, )),
            axes=tuple(range(self._numD))
        ).reshape((-1, arrX.shape[1])) * self.numN

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''

        # We iteratively build up the kronecker product of the 1D-Fourier
        # transforms

        arrRes = np.ones((1))

        for ii in range(self.numD):
            arrRes = np.kron(arrRes, np.exp(
                np.multiply(
                    *np.meshgrid(
                        np.arange(self._order[ii]),
                        np.arange(self._order[ii])
                    )) * -2j * np.pi / self._order[ii]
            ))

        return arrRes

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                # define matrix sizes and parameters
                'numS'          : TEST.Permutation([7, 12]),
                TEST.NUM_N      : lambda param: param['numS'] ** 2,
                TEST.NUM_M      : TEST.NUM_N,
                'optimize'      : TEST.Permutation([False, True]),

                # define constructor for test instances and naming of test
                TEST.OBJECT     : NDFourier,
                TEST.INITARGS   : lambda param:
                [param['numS'], param['numS']],
                TEST.INITKWARGS : {'optimize': 'optimize'},
                TEST.TOL_POWER  : 3.,
                TEST.NAMINGARGS : dynFormat("%d, optimize=%d", TEST.NUM_N,
                                            'optimize'),
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c: NDFourier(c, c)),
                BENCH.FUNC_SIZE : (lambda c: c ** 2)
            },
            BENCH.FORWARD: {},
            BENCH.SOLVE: {},
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c: NDFourier(2 ** c, 2 ** c)),
                BENCH.FUNC_SIZE : (lambda c: 2 ** (2 * c))
            },
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, datatype: NDFourier(2 ** c,
                												 2 ** c)),
                BENCH.FUNC_SIZE : (lambda c: 2 ** (2 * c))
            }
        }

    def _getDocumentation(self):
        return ""
