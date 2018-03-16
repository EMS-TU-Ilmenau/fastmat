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
from .core.cmath cimport _conjugate
from .core.types cimport *


cdef class Product(Matrix):
    r"""


    .. math::
        M = \prod\limits_i  A_i

    where the :math:`A_{i}` can be fast transforms of \*any\* type.

    >>> # import the package
    >>> import fastmat as fm
    >>>
    >>> # define the product terms
    >>> A = fm.Circulant(x_A)
    >>> B = fm.Circulant(x_B)
    >>>
    >>> # construct the product
    >>> M = fm.Product(A.H, B)

    Assume we have two circulant matrices :math:`A` and :math:`B`. Then we
    define

    .. math::
        M =  A_c^\mathrm{H}  B_c.

    """

    def __init__(self, *matrices, **options):
        '''Initialize Matrix instance'''

        # evaluate options passed to Product
        debug = options.get('debug', False)

        # initialize product content to [1]
        lstFactors = []
        self._scalar = np.int8(1)

        # determine data type of matrix: store datatype in immutable object to
        # allow access from within subfunctions (python2/3 compatible)
        datatype = [self._scalar.dtype]

        def __promoteType(dtype):
            datatype[0] = np.promote_types(datatype[0], dtype)

        # add all product terms, direct and nested
        def __addFactors(factors):
            for factor in factors:
                if np.isscalar(factor):
                    # explicit type promotion: avoid type shortening of scalars
                    __promoteType(np.array(factor).dtype)
                    self._scalar = np.inner(self._scalar, factor)
                    continue

                if not isinstance(factor, Matrix):
                    raise TypeError("Product: Term not scalar nor Matrix.")

                if isinstance(factor, Product):
                    if factor._scalar != 1:
                        self._scalar = self._scalar * factor._scalar

                    __addFactors(factor.content)
                else:
                    # store fastmat-matrix-content: determine data type
                    # -> promotion of factor types
                    lstFactors.append(factor)

                # promote data type for all fastmat matrices
                __promoteType(factor.dtype)

        __addFactors(matrices)
        dtype = datatype[0]

        # handle type expansion with default depending on matrix type
        # default: expand small types due to accumulation during transforms
        # skip by specifying `typeExpansion=None` or override with `~=...`
        typeExpansion = options.get('typeExpansion', safeTypeExpansion(dtype))
        dtype = (dtype if typeExpansion is None
                 else np.promote_types(dtype, typeExpansion))

        # sanity check of the supplied amount of product terms
        if len(lstFactors) < 1:
            raise ValueError("Product has no terms.")

        # iterate elements and check if their numN fit the previous numM
        cdef intsize numN = lstFactors[0].numN
        cdef intsize numM = lstFactors[0].numM
        cdef int ii
        for ii in range(1, len(lstFactors)):
            factor = lstFactors[ii]
            if factor.numN != numM:
                raise ValueError(
                    "Product: Dimension mismatch for term %d [%dx%d]" %(
                        ii, numN, numM))
            numM = factor.numM

        # force scalar datatype to match matrix datatype (calculation accuracy)
        self._scalar = self._scalar.astype(dtype)

        # also, make factor list immutable
        self._content = tuple(lstFactors)

        # set properties of matrix
        self._initProperties(numN, numM, dtype)

        if debug:
            print("fastmat.Product instance %12x containing:" %(id(self)))
            if self._scalar != 1:
                print("  [0]: scalar %s" %(self._scalar))
            for ii, factor in enumerate(lstFactors):
                print("  [%d]: %s" %(ii, factor.__repr__()))

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        # regular product except the last term is just term._getCol
        cdef int cnt = len(self._content)
        cdef int ii                     # index (0 .. cnt - 1)
        cdef int iii = cnt - 1          # index (cnt - 1 .. 0)
        cdef np.ndarray arrRes = self._content[iii].getCol(idx)

        # use inner for element-wise scalar mul as inner does type promotion
        if self._scalar != 1:
            arrRes = np.inner(arrRes, self._scalar)
        else:
            arrRes = arrRes.astype(np.promote_types(arrRes.dtype, self.dtype))

        for ii in range(1, cnt):
            iii = cnt - 1 - ii
            arrRes = self._content[iii].forward(arrRes)

        return arrRes

    cpdef np.ndarray _getRow(self, intsize idx):
        # regular product w/ backward except the last term is just term._getRow
        cdef int cnt = len(self._content)
        cdef int ii = 0
        cdef np.ndarray arrRes = _conjugate(self._content[ii].getRow(idx))

        # use inner for element-wise scalar mul as inner does type promotion
        if self._scalar != 1:
            if np.iscomplex(self._scalar):
                arrRes = np.inner(arrRes, self._scalar.conjugate())
            else:
                arrRes = np.inner(arrRes, self._scalar)
        else:
            arrRes = arrRes.astype(np.promote_types(arrRes.dtype, self.dtype))

        for ii in range(1, cnt):
            arrRes = self._content[ii].backward(arrRes)

        # don't forget to return the conjugate as we use the backward
        return _conjugate(arrRes)

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        cdef float complexity = len(self._content)
        return (complexity, complexity)

    ############################################## class forward / backward
    cpdef np.ndarray _forward(self, np.ndarray arrX):
        '''Calculate the forward transform of this matrix'''

        cdef int cnt = len(self._content)
        cdef int ii                     # index (0 .. cnt - 1)
        cdef int iii = cnt - 1          # index (cnt - 1 .. 0)
        cdef np.ndarray arrRes = arrX

        # use inner for element-wise scalar mul as inner does type promotion
        if self._scalar != 1:
            arrRes = np.inner(arrRes, self._scalar)
        else:
            arrRes = arrRes.astype(np.promote_types(arrRes.dtype, self.dtype))

        for ii in range(0, cnt):
            iii = cnt - 1 - ii
            arrRes = self._content[iii].forward(arrRes)

        return arrRes

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        '''Calculate the backward transform of this matrix'''

        cdef int cnt = len(self._content)
        cdef int ii
        cdef np.ndarray arrRes = arrX

        # use inner for element-wise scalar mul as inner does type promotion
        if self._scalar != 1:
            if np.iscomplex(self._scalar):
                arrRes = np.inner(arrRes, self._scalar.conjugate())
            else:
                arrRes = np.inner(arrRes, self._scalar)
        else:
            arrRes = arrRes.astype(np.promote_types(arrRes.dtype, self.dtype))

        for ii in range(0, cnt):
            arrRes = self._content[ii].backward(arrRes)

        return arrRes

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        cdef ii, cnt = len(self._content)
        cdef Matrix term
        cdef np.ndarray arrRes

        dtype = np.promote_types(np.float64, self.dtype)

        arrRes = np.inner(
            self._content[cnt - 1].reference(), self._scalar.astype(dtype))

        for ii in range(1, cnt):
            term = self._content[cnt - ii - 1]
            arrRes = term.reference().dot(arrRes)

        return arrRes

    def _forwardReferenceInit(self):
        self._forwardReferenceMatrix = []
        for ii, term in enumerate(self._content):
            self._forwardReferenceMatrix.append(term.reference())

    def _forwardReference(self, arrX):
        '''Calculate the forward transform by non-fastmat means.'''

        # check if operation list initialized. If not, then do it!
        if not isinstance(self._forwardReferenceMatrix, list):
            self._forwardReferenceInit()

        # perform operations list
        arrRes = arrX
        for ii in range(len(self._content), 0, -1):
            arrRes = self._forwardReferenceMatrix[ii - 1].dot(arrRes)

        return arrRes

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                TEST.NUM_N      : 7,
                TEST.NUM_M      : TEST.Permutation([10, TEST.NUM_N]),
                'mType1'        : TEST.Permutation(TEST.ALLTYPES),
                'mType2'        : TEST.Permutation(TEST.FEWTYPES),
                'sType'         : TEST.Permutation(TEST.ALLTYPES),
                'arr1'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mType1',
                    TEST.SHAPE  : (TEST.NUM_N, TEST.NUM_M)
                }),
                'arr2'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mType2',
                    TEST.SHAPE  : (TEST.NUM_M , TEST.NUM_N)
                }),
                'arr3'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mType1',
                    TEST.SHAPE  : (TEST.NUM_N , TEST.NUM_M)
                }),
                'num4'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'sType',
                    TEST.SHAPE  : (1,)
                }),
                TEST.OBJECT     : Product,
                TEST.INITARGS   : (lambda param: [param['num4']()[0],
                                                  Matrix(param['arr1']()),
                                                  Matrix(param['arr2']()),
                                                  Matrix(param['arr3']())]),
                'strType'       : (lambda param: TEST.TYPENAME[param['sType']]),
                TEST.NAMINGARGS: dynFormat("%s*%s*%s*%s",
                                           'strType', 'arr1', 'arr2', 'arr3'),
                TEST.TOL_POWER  : 3.
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH
        from .Fourier import Fourier
        from .Hadamard import Hadamard
        from .Eye import Eye
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c:
                                   Product(Hadamard(c), Fourier(2 ** c))),
                BENCH.FUNC_SIZE : (lambda c: 2 ** c),
                BENCH.FUNC_STEP : (lambda c: c + 1)
            },
            BENCH.FORWARD: {},
            BENCH.SOLVE: {},
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c: Product(*([Eye(2 ** c)] * 2 ** c)))
            }
        }

    def _getDocumentation(self):
        return ""
