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
from scipy.sparse import spmatrix

from .Matrix cimport Matrix
from .core.types cimport *
from .core.cmath cimport _conjugate, _arrSqueeze

cdef class Sparse(Matrix):
    r"""


    .. math::
        x \mapsto  S x,

    where :math:`S` is a ``scipy.sparse`` matrix. To provide a high level of
    generality, the user has to make use of the standard ``scipy.sparse``
    matrix constructors and pass them to ``fastmat`` during construction.
    After that a ``Sparse`` matrix can be used like every other type in
    ``fastmat``

    >>> # import the package
    >>> import fastmat as fm
    >>>
    >>> # import scipy to get
    >>> # the constructor
    >>> import scipy.sparse.rand as r
    >>>
    >>> # set the matrix size
    >>> n = 100
    >>>
    >>> # construct the sparse matrix
    >>> S = fm.Sparse(
    >>>         r(
    >>>             n,
    >>>             n,
    >>>             0.01,
    >>>             format='csr'
    >>>         ))

    This yields a random sparse matrix with 1\% of its entries occupied drawn
    from a random distribution.

    It is also possible to directly cast SciPy sparse matrices into the
    `fastmat`` sparse matrix format as follows.

    >>> # import the package
    >>> import fastmat as fm
    >>>
    >>> # import scipy to get
    >>> # the constructor
    >>> import scipy.sparse as ss
    >>>
    >>> # construct the SciPy sparse matrix
    >>> S_scipy = ss.csr_matrix(
    >>>         [
    >>>             [1, 0, 0],
    >>>             [1, 0, 0],
    >>>             [0, 0, 1]
    >>>         ]
    >>>     )
    >>>
    >>> # construct the fastmat sparse matrix
    >>> S = fm.Sparse(S_scipy)

    .. note::
        The ``format`` specifier drastically influences performance during
        multiplication of these matrices. From our experience ``'csr'`` works
        best in these cases.

    For this matrix class we used the already tried and tested routines of
    SciPy [1]_, so we merely provide a convenient wrapper to integrate nicely
    into ``fastmat``.
    """

    property spArray:
        r"""Return the scipy sparse matrix .

        *(read only)*
        """

        def __get__(self):
            return self._spArray

    property spArrayH:
        r"""Return the scipy sparse matrix' hermitian transpose.

        *(read only)*
        """

        def __get__(self):
            if self._spArrayH is None:
                self._spArrayH = self._spArray.T.conj().tocsr()
            return self._spArrayH

    def __init__(self, value):
        '''Initialize Matrix instance'''
        if not isinstance(value, spmatrix):
            raise TypeError("Sparse: Use Matrix() for numpy ndarrays."
                            if isinstance(value, np.ndarray)
                            else "Sparse: Matrix is not a scipy spmatrix")

        self._spArray = value.tocsr()

        # set properties of matrix
        self._initProperties(self._spArray.shape[0], self._spArray.shape[1],
                             self._spArray.dtype)

    cpdef np.ndarray _getArray(self):
        '''
        Return an explicit representation of the matrix as numpy-array.
        '''
        return self._spArray.toarray()

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        return _arrSqueeze(self._spArray.getcol(idx).toarray())

    cpdef np.ndarray _getRow(self, intsize idx):
        return _arrSqueeze(_conjugate(self.spArrayH.getcol(idx).toarray()))

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        return self._spArray[idxN, idxM]

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        cdef float complexity = self._spArray.getnnz()
        return (complexity, complexity)

    ############################################## class forward / backward
    cpdef np.ndarray _forward(self, np.ndarray arrX):
        '''Calculate the forward transform of this matrix'''
        return self._spArray.dot(arrX)

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        '''Calculate the backward transform of this matrix'''
        return self.spArrayH.dot(arrX)

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        return self._spArray.toarray()

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat, arrSparseTestDist
        return {
            TEST.COMMON: {
                TEST.NUM_N      : 25,
                TEST.NUM_M      : TEST.Permutation([30, TEST.NUM_N]),
                'mType'         : TEST.Permutation(TEST.ALLTYPES),
                'density'       : .1,
                TEST.INITARGS   : (lambda param: [
                    arrSparseTestDist(
                        (param[TEST.NUM_N], param[TEST.NUM_M]), param['mType'],
                        density=param['density'], compactFullyOccupied=True
                    )]),
                TEST.OBJECT     : Sparse,
                'strType'       : (lambda param: TEST.TYPENAME[param['mType']]),
                TEST.NAMINGARGS : dynFormat("%s,%s", 'strType', TEST.NUM_M)
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH
        import scipy.sparse as sps
        return {
            BENCH.FORWARD: {
                BENCH.FUNC_GEN  : (lambda c: Sparse(
                    sps.rand(c, c, 0.1, format='csr')))
            },
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c: Sparse(
                    sps.rand(2 ** c, 2 ** c, 0.1, format='csr')))
            },
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, dt: (
                    Sparse(sps.rand(2 ** c, 2 ** c, 0.1,
                                    format='csr', dtype=dt))
                    if isFloat(dt) else None))
            }
        }

    def _getDocumentation(self):
        return ""
