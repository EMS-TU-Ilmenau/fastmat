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

from .core.types cimport *
from .Matrix cimport Matrix


cdef class Polynomial(Matrix):
    r"""

    For given coefficients :math:`a_k,\dots,a_0 \in \mathbb{C}` and a linear
    mapping :math:`A \in \mathbb{C}^{n \times n}`, we define

    .. math::
        M = a_n  A^n + a_{n-1}  A^{n-1} + a_1  A + a_0  I.

    The transform :math:`M \cdot  x` can be calculated efficiently with
    Horner's method.

    >>> # import the package
    >>> import fastmat as fm
    >>>
    >>> # define the transforms
    >>> H = fm.Hadamard(n)
    >>>
    >>> # define the coefficient array
    >>> arr_a = [1, 2 + 1j, -3.0, 0.0]
    >>>
    >>> # define the polynomial
    >>> M = fm.Polynomial(H, arr_a)

    Let :math:`H_n` be the Hadamard matrix of order :math:`n`. And let
    :math:`a = (1, 2 + i, -3, 0) \in \mathbb{C}^{4}` be a coefficient vector,
    then the polynomial is defined as

    .. math::
        M =  H_n^3 + (2+i)  H_n^2 - 3  H_n.
    """

    property coeff:
        r"""Return the polynomial coefficient vector.

        *(read only)*
        """

        def __get__(self):
            return self._coeff

    def __init__(self, mat, coeff, **options):
        '''Initialize Matrix instance'''

        if mat.numN != mat.numM:
            raise ValueError("Polynomial: Matrix must be square.")

        dtype = np.promote_types(mat.dtype, coeff.dtype)

        # handle type expansion with default depending on matrix type
        # default: expand small types due to accumulation during transforms
        # skip by specifying `typeExpansion=None` or override with `~=...`
        typeExpansion = options.get('typeExpansion', safeTypeExpansion(dtype))
        dtype = (dtype if typeExpansion is None
                 else np.promote_types(dtype, typeExpansion))

        self._content = (mat,)
        self._coeff = np.flipud(coeff.astype(dtype))
        self._coeffConj = self._coeff.conj()

        # set properties of matrix
        self._initProperties(self._content[0].numN, self._content[0].numM,
                             dtype)

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        cdef float complexity = len(self._coeff) * (self.numN + self.numM)
        return (complexity, complexity)

    ############################################## class forward / backward
    cpdef np.ndarray _forward(self, np.ndarray arrX):
        '''Apply the Horner scheme to a polynomial of matrices.'''
        cdef cc, cnt = self._coeff.shape[0]
        cdef np.ndarray arrRes, arrIn = arrX

        arrRes = np.inner(arrX, self._coeff[0])

        # use inner for element-wise scalar mul as inner does type promotion
        for cc in range(1, cnt):
            arrRes  = self._content[0].forward(arrRes) + np.inner(
                arrX, self._coeff[cc])

        return arrRes

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        '''Apply the Horner scheme to a polynomial of matrices.'''
        cdef cc, cnt = self._coeffConj.shape[0]
        cdef np.ndarray arrRes

        arrRes = np.inner(arrX, self._coeffConj[0])

        # use inner for element-wise scalar mul as inner does type promotion
        for cc in range(1, cnt):
            arrRes  = self._content[0].backward(arrRes) + np.inner(
                arrX, self._coeffConj[cc])

        return arrRes

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        cdef intsize ii, ind = 0
        cdef np.ndarray arrRes, tmp

        dtype = np.promote_types(self.dtype, np.float64)
        arrRes = np.zeros((self.numN, self.numN), dtype=dtype)

        for cc in np.flipud(self._coeff):
            arrTrafo = self._content[0].reference()
            tmp = np.eye(self._content[0].numN, dtype=dtype)
            for ii in range(ind):
                tmp = arrTrafo.dot(tmp)

            arrRes = arrRes + np.inner(tmp, cc)
            ind += 1

        return arrRes

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                'order'         : 5,
                TEST.TOL_POWER  : 'order',
                TEST.NUM_N      : 7,
                TEST.NUM_M      : 7,

                'mTypeC'        : TEST.Permutation(TEST.ALLTYPES),
                'mTypeM'        : TEST.Permutation(TEST.ALLTYPES),
                TEST.PARAMALIGN : TEST.Permutation(TEST.ALLALIGNMENTS),
                'vecC'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeC',
                    TEST.SHAPE  : ('order', ),
                    TEST.ALIGN  : TEST.PARAMALIGN
                }),
                'arrM'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeM',
                    TEST.SHAPE  : (TEST.NUM_N, TEST.NUM_M)
                }),

                TEST.OBJECT     : Polynomial,
                TEST.INITARGS   : (lambda param: [Matrix(param['arrM']()),
                                                  param['vecC']()]),
                TEST.NAMINGARGS : dynFormat("%s,%s", 'vecC', 'arrM'),
                TEST.TOL_POWER  : 'order'
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH
        from .Eye import Eye
        from .Hadamard import Hadamard
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c: Polynomial(
                    Hadamard(c), np.random.uniform(1, 2, 6))),
                BENCH.FUNC_SIZE : (lambda c: 2 ** c),
                BENCH.FUNC_STEP : (lambda c: c + 1)
            },
            BENCH.FORWARD: {},
            BENCH.SOLVE: {},
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c: Polynomial(
                    Eye(2 ** c), np.random.uniform(1, 2, 10)))
            }
        }

    def _getDocumentation(self):
        return ""
