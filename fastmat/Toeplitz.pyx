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
from .core.cmath cimport *
from .Matrix cimport Matrix
from .Partial cimport Partial
from .Product cimport Product
from .Fourier cimport Fourier
from .Diag cimport Diag


################################################################################
################################################## class Toeplitz
cdef class Toeplitz(Partial):
    r"""

    A Toeplitz matrix :math:`T \in \mathbb{C}^{n \times m}` realizes the mapping

    .. math::
        x \mapsto  T \cdot  x,

    where :math:`x \in C^n` and

    .. math::
        T = \begin{bmatrix}
        t_1 & t_{-1} & \dots & t_{-(m-1)} \\
        t_2 & t_1 & \ddots & t_{-(n-2)} \\
        \vdots & \vdots & \ddots & \vdots \\
        t_n & t_{n-1} & \dots & t_1
        \end{bmatrix}.

    This means that a Toeplitz matrix is uniquely defined by the
    :math:`n + m - 1` values that are on the diagonals.

    >>> # import the package
    >>> import fastmat as fm
    >>> import numpy as np
    >>>
    >>> # define the parameters
    >>> d1 = np.array([1,0,3,6])
    >>> d2 = np.array([5,7,9])
    >>>
    >>> # construct the transform
    >>> T = fm.Toeplitz(d1,d2)

    This yields

    .. math::
        d_1 = (1,0,3,6)^\mathrm{T}

    .. math::
        d_2 = (5,7,9)^\mathrm{T}

    .. math::
        T = \begin{bmatrix}
        1 & 5 & 7 & 9 \\
        0 & 1 & 5 & 7 \\
        3 & 0 & 1 & 5 \\
        6 & 3 & 0 & 1
        \end{bmatrix}

    Since the multiplication with a Toeplitz matrix makes use of the FFT, it
    can be very slow, if the sum of the dimensions of :math:`d_1` and
    :math:`d_2` are far away from a power of :math:`2`, :math:`3` or
    :math:`4`. This can be alleviated if one applies smart zeropadding during
    the transformation.
    This can be activated as follows.

    >>> # import the package
    >>> import fastmat as fm
    >>> import numpy as np
    >>>
    >>> # define the parameters
    >>> d1 = np.array([1,0,3,6])
    >>> d2 = np.array([5,7,9])
    >>>
    >>> # construct the transform
    >>> T = fm.Toeplitz(d1,d2,pad='true')

    This yields the same matrix and transformation as above, but it might be
    faster depending on the dimensions involved in the problem.

    This class depends on ``Fourier``, ``Diag``, ``Product`` and
    ``Partial``.
    """

    property vecC:
        r"""Return the column-defining vector of Toeplitz matrix."""

        def __get__(self):
            return self._vecC

    property vecR:
        r"""Return the row-defining vector of Toeplitz matrix."""

        def __get__(self):
            return self._vecR

    def __init__(self, vecC, vecR, **options):
        '''
        Initialize Toeplitz Matrix instance.

        The toeplitz matrix is embedded into a circulant matrix. See class
        Circulant for further details.

        Toeplitz([ c C C C C ], [ r R R R ]) represents this matrix:
          [ c r R R R ]
          [ C c r R R ]
          [ C C c r R ]
          [ C C C c r ]
          [ C C C C c ]

        The Circulant it is derived from has the following generator:
          [ c C C C C R R R r ] [9 x 9]

        ...and may be zero-padded to a [16 x 16] Matrix for efficiency:
          [ c C C C C 0 0 0 0 0 0 0 R R R r]

        Valid options
        -------------
         'pad'=[false TRUE]
            perform zero-padding of Circulant for efficiency

        All options specified will also be passed on to the geneation of the
        underlying Product instance initialization.
        '''

        # save generating vectors. Matrix sizes will be set by Product
        dataType = np.promote_types(vecC.dtype, vecR.dtype)
        self._vecC = vecC = np.atleast_1d(np.squeeze(vecC.astype(dataType,
                                                                 copy=True)))
        self._vecR = vecR = np.atleast_1d(np.squeeze(vecR.astype(dataType,
                                                                 copy=True)))

        # evaluate options passed to class
        cdef bint optimize = options.get('optimize', True)
        cdef int maxStage = options.get('maxStage', 4)

        # perform padding (if enabled) and generate vector
        cdef intsize size = len(vecC) + len(vecR)
        cdef intsize vecSize = size

        # determine if zero-padding of the convolution to achieve a better FFT
        # size is beneficial or not
        if optimize:
            vecSize = _findOptimalFFTSize(size, maxStage)

            assert vecSize >= size

            if _getFFTComplexity(size) <= _getFFTComplexity(vecSize):
                vecSize = size

        if vecSize > size:
            # zero-padding pays off, so do it!
            vec = np.concatenate([self._vecC,
                                  np.zeros((vecSize - size,),
                                           dtype=dataType),
                                  np.flipud(self._vecR)])
        else:
            vec = np.concatenate([self._vecC, np.flipud(self._vecR)])

        # Describe as circulant matrix with product of data and vector
        # in fourier domain. Both fourier matrices cause scaling of the
        # data vector by N, which will be compensated in Diag().

        # Create inner product
        cdef Fourier FN = Fourier(vecSize)
        cdef Product P = Product(FN.H, Diag(np.fft.fft(vec, axis=0) / vecSize),
                                 FN, **options)

        # initialize Partial of Product
        kwargs = {}
        if size != len(self._vecC):
            kwargs['N'] = np.arange(len(self._vecC))

        if size != len(self._vecR) + 1:
            kwargs['M'] = np.arange(len(self._vecR) + 1)

        super(Toeplitz, self).__init__(P, **kwargs)

        # Currently Fourier matrices bloat everything up to complex double
        # precision, therefore make sure vecC and vecR matches the precision of
        # the matrix itself
        if self.dtype != self._vecC.dtype:
            self._vecC = self._vecC.astype(self.dtype)

        if self.dtype != self._vecR.dtype:
            self._vecR = self._vecR.astype(self.dtype)

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        cdef intsize N = self.numN
        cdef np.ndarray arrRes

        if idx == 0:
            return self._vecC
        elif idx >= N:
            # double slicing needed, otherwise fail when M = N + 1
            return self._vecR[idx - N:idx][::-1]
        else:
            arrRes = _arrEmpty(1, N, 0, self._info.dtype[0].typeNum)
            arrRes[:idx] = self._vecR[idx - 1::-1]
            arrRes[idx:] = self._vecC[:N - idx]
            return arrRes

    cpdef np.ndarray _getRow(self, intsize idx):
        cdef intsize M = self.numM
        cdef np.ndarray arrRes

        if idx >= M - 1:
            # double slicing needed, otherwise fail when N = M + 1
            return self._vecC[idx - M + 1:idx + 1][::-1]
        else:
            arrRes = _arrEmpty(1, M, 0, self._info.dtype[0].typeNum)
            arrRes[:idx + 1] = self._vecC[idx::-1]
            arrRes[idx + 1:] = self._vecR[:M - 1 - idx]
            return arrRes

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        cdef intsize distance = idxN - idxM
        return (self._vecR[-distance - 1] if distance < 0
                else self._vecC[distance])

    cpdef np.ndarray _getArray(self):
        '''
        Return an explicit representation of the matrix as numpy-array.
        '''
        return self._reference()

    cpdef Matrix _getNormalized(self):
        # NOTE: This method suffers accuracy losses when elements with lower
        # indices are large in magnitude compared to ones with higher index!
        cdef intsize ii, N = self.numN, M = self.numM
        cdef intsize iiMax = (N + 1) if M > N else M

        # fill in a placeholder array
        cdef np.ndarray arrNorms = _arrZero(1, M, 1, np.NPY_FLOAT64)

        # compute the absolute value of the squared elements in the defining
        # vectors
        cdef np.ndarray vecCSqr = np.square(np.abs(self._vecC))
        cdef np.ndarray vecRSqr = np.square(np.abs(self._vecR))

        # the first column is easy
        arrNorms[0] = vecCSqr.sum()

        # then follow the iterative approach. Every subsequent column features
        # one more element of vecR and one less of vecC. Continue until vecC is
        # eaten up completely
        for ii in range(1, iiMax):
            arrNorms[ii] = arrNorms[ii - 1] + vecRSqr[ii - 1] - vecCSqr[N - ii]

        # then (as vecC is eaten up), proceed with rolling the remaining
        # elements of vecR until they are represented fully in arrNorms
        for ii in range(iiMax, M):
            arrNorms[ii] = (arrNorms[ii - 1] +
                            vecRSqr[ii - 1] - vecRSqr[ii - iiMax])

        return self * Diag(1. / np.sqrt(arrNorms))

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        return (0., 0.)

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        _reference borrowing from Partial is too slow. Therefore,
        construct a reference directly from the vectors.
        '''
        cdef intsize ii, N = self.numN, M = self.numM
        cdef np.ndarray arrRes = np.empty((N, M), dtype=self.dtype)

        # put columns in lower-triangular part of matrix
        for ii in range(0, min(N, M)):
            arrRes[ii:N, ii] = self._vecC[0:(N - ii)]

        # put rows in upper-triangular part of matrix
        for ii in range(0, min(N, M - 1)):
            arrRes[ii, (ii + 1):M] = self._vecR[0:(M - ii - 1)]

        return arrRes

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                # 35 is just any number that causes no padding
                # 41 is the first size for which bluestein is faster
                TEST.NUM_N      : TEST.Permutation([5, 41]),
                'num_M'         : TEST.Permutation([4, 6]),
                TEST.NUM_M      : (lambda param: param['num_M'] + 1),
                'mTypeH'        : TEST.Permutation(TEST.ALLTYPES),
                'mTypeV'        : TEST.Permutation(TEST.FEWTYPES),
                'optimize'      : True,
                'vecH'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeH',
                    TEST.SHAPE  : (TEST.NUM_N, )
                }),
                'vecV'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeV',
                    TEST.SHAPE  : ('num_M', )
                }),
                TEST.INITARGS   : (lambda param : [param['vecH'](),
                                                   param['vecV']()]),
                TEST.INITKWARGS : {'optimize' : 'optimize'},
                TEST.OBJECT     : Toeplitz,
                TEST.NAMINGARGS : dynFormat("%s,%s,optimize=%s",
                                            'vecH', 'vecV', str('optimize')),
                TEST.TOL_POWER  : 2
            },
            TEST.CLASS: {
                # perform thorough testing of slicing during array construction
                # therefore, aside the symmetric shape case also test shapes
                # that differ by +1/-1 and +x/-x in row and col size
                TEST.NUM_N      : 4,
                'num_M'         : TEST.Permutation([2, 3, 4, 5, 6]),
            },
            TEST.TRANSFORMS: {
                # test differences in padding only for the transforms
                'padding'       : TEST.Permutation([True, False]),
            }
        }

    def _getBenchmark(self):
        from .inspect import BENCH, arrTestDist
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c: Toeplitz(
                    arrTestDist((c, ), dtype=np.float),
                    arrTestDist((c - 1, ), dtype=np.float)))
            },
            BENCH.FORWARD: {},
            BENCH.SOLVE: {},
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c: Toeplitz(
                    arrTestDist((2 ** c, ), dtype=np.float),
                    arrTestDist((2 ** c - 1, ), dtype=np.float)))
            },
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, datatype: Toeplitz(
                    arrTestDist((2 ** c, ), dtype=datatype),
                    arrTestDist((2 ** c - 1, ), dtype=datatype)))
            }
        }

    def _getDocumentation(self):
        return ""
