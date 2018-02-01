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

from .Outer cimport Outer
from .Product cimport Product
from .Matrix cimport Matrix

from .Diag cimport Diag

from .core.types cimport *

cdef class LowRank(Product):
    r"""

    Generally one can consider the "complexity" of a matrix as the number of
    its rows :math:`n` and columns :math:`m`. The rank of a matrix
    :math:`A \in \mathbb{C}^{n \times m}` always obeys the bound

    .. math::
        \mathrm{rk}(A) \leqslant \min \{n,m\}.

    If one carries out the normal matrix vector multiplication, one assumes
    the rank to be essentially close to this upper bound. However if the rank
    of :math:`A` is far lower than the minimum of its dimensions, then one
    carries out a lot of redundant tasks, when applying this matrix to a
    vector. But if one computes the singular value decomposition (SVD) of
    :math:`A =  U  \Sigma  V^\mathrm{H}`, then one can express :math:`A` as a
    sum of rank-:math:`1` matrices as

    .. math::
        A = \sum\limits_{i = 1}^{r} \sigma_i  u_{i}  v^\mathrm{H}_{i}.

    If :math:`r = \mathrm{rk}(A)` is much smaller than the minimum of the
    dimensions, then one can save a lot of computational effort in applying
    :math:`A` to a vector.

    >>> # import the package
    >>> import fastmat as fm
    >>> import numpy as np
    >>>
    >>> # define all parameters
    >>> S = np.random.randn(2)
    >>> U = np.random.randn(20,2)
    >>> V = np.random.randn(20,2)
    >>>
    >>> # define the matrix
    >>> L = fm.LowRank(S, U, V)

    We define a matrix
    :math:`L =  U  S  V^\mathrm{H} \in \mathbb{R}^{20 \times 20}` with rank
    :math:`2`.
    """

    property vecS:
        r"""Return the vector of non-zero singular values entries.

        *(read-only)*
        """

        def __get__(self):
            return self._vecS

    property arrU:
        r"""Return the array of left orthogonal vectors, i.e. the image

        *(read-only)*
        """

        def __get__(self):
            return self._arrU

    property arrV:
        r"""Return the array of right orthogonal vectors

        the orthogonal complement of the kernel

        *(read-only)*
        """

        def __get__(self):
            return self._arrV

    def __init__(self, vecS, arrU, arrV):

        # complain if dimension does not match
        if vecS.ndim != 1:
            raise ValueError(
                "Singular value vector must have exactly one active dimension.")

        if arrU.ndim != 2 or arrV.ndim != 2:
            raise ValueError("Defining arrays must be 2D.")

        if vecS.shape[0] != arrU.shape[1]:
            raise ValueError("Size of U must fit to rank.")

        # determine matrix data data type from parameters, then adjust them to
        # match (parameters)
        dtype = np.promote_types(
            np.promote_types(arrU.dtype, arrV.dtype), vecS.dtype)

        # store singular values vector as copy of vecS
        # store copies of left and right orthogonal matrices
        self._arrU = arrU.astype(dtype, copy=True, subok=False)
        self._arrV = arrV.astype(dtype, copy=True, subok=False)
        self._vecS = np.atleast_1d(vecS).astype(dtype, copy=True, subok=False)

        super(LowRank, self).__init__(
            Matrix(self._arrU), Diag(vecS), Matrix(self._arrV.conj().T))

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        cdef float complexity = self.numN + self.numM + self._vecS.shape[0]
        return (complexity, complexity)

    ########################################################################
    ## forward and backward are taken from sum, product and outer
    ########################################################################

    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        cdef np.ndarray arrU = self._arrU.astype(
            np.promote_types(self._arrU.dtype, np.float64))
        cdef np.ndarray arrV = self._arrV.astype(
            np.promote_types(self._arrV.dtype, np.float64))
        cdef np.ndarray vecS = self._vecS.astype(
            np.promote_types(self._vecS.dtype, np.float64))

        return arrU.dot(np.diag(vecS).dot(arrV.conj().T))

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                'order'         : 4,
                TEST.TOL_POWER  : (lambda param: np.sqrt(param['order'])),
                TEST.NUM_N      : 7,
                TEST.NUM_M      : TEST.Permutation([11, TEST.NUM_N]),
                'mTypeS'        : TEST.Permutation(TEST.ALLTYPES),
                'mTypeU'        : TEST.Permutation(TEST.FEWTYPES),
                'mTypeV'        : TEST.Permutation(TEST.ALLTYPES),
                'vecS'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeS',
                    TEST.SHAPE  : ('order',),
                }),
                'arrU'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeU',
                    TEST.SHAPE  : (TEST.NUM_N, 'order'),
                }),
                'arrV'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeV',
                    TEST.SHAPE  : (TEST.NUM_M, 'order'),
                }),
                TEST.INITARGS   : (lambda param : [param['vecS'](),
                                                   param['arrU'](),
                                                   param['arrV']()]),
                TEST.OBJECT     : LowRank,
                TEST.NAMINGARGS : dynFormat("%s,%s,%s", 'arrU', 'vecS', 'arrV'),
                TEST.TOL_POWER  : 3.
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH, arrTestDist
        return {
            BENCH.COMMON: {
                BENCH.FUNC_SIZE : (lambda c: 10 * c + 1),
                BENCH.FUNC_GEN  : (lambda c: LowRank(
                    arrTestDist((c + 1, ), dtype=np.float, center=2),
                    arrTestDist((10 * c + 1, c + 1), dtype=np.float, center=2),
                    arrTestDist((10 * c + 1, c + 1), dtype=np.float, center=2)))
            },
            BENCH.FORWARD: {},
            BENCH.OVERHEAD: {
                BENCH.FUNC_SIZE : (lambda c: 2 ** c),
                BENCH.FUNC_GEN  : (lambda c: LowRank(
                    arrTestDist((c, ), dtype=np.float),
                    arrTestDist((2 ** c, c), dtype=np.float),
                    arrTestDist((2 ** c, c), dtype=np.float)))
            },
            BENCH.DTYPES: {
                BENCH.FUNC_SIZE : (lambda c: 2 ** c),
                BENCH.FUNC_GEN  : (lambda c, datatype: LowRank(
                    arrTestDist((c, ), dtype=np.float),
                    arrTestDist((2 ** c, c), dtype=np.float),
                    arrTestDist((2 ** c, c), dtype=np.float)))
            }
        }

    def _getDocumentation(self):
        return ""
