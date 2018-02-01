# -*- coding: utf-8 -*-

#cython: boundscheck=False, wraparound=False
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
################################################## class Circulant
cdef class Circulant(Partial):

    r"""Circulant Matrix ``fastmat.Circulant``



    Circulant matrices realize the following mapping

    .. math::
        x \mapsto  C \cdot  x =  c *  x,

    with :math:`x \in \mathbb{C}^n` and

    .. math::
        C = \begin{bmatrix}
            c_1     & c_n       & \dots     & c_2       \\
            c_2     & c_1       & \ddots    & c_3       \\
            \vdots  & \vdots    & \ddots    & \vdots    \\
            c_n     & c_{n-1}   & \dots     & c_1
        \end{bmatrix}.

    This means that :math:`C` is completely defined by its first column and
    realizes the convolution with the vector :math:`c`.

    >>> # import the package
    >>> import fastmat as fm
    >>> import numpy as np
    >>>
    >>> # construct the
    >>> # parameter
    >>> n = 4
    >>> c = np.array([1, 0, 3, 6])
    >>>
    >>> # construct the matrix
    >>> C = fm.Circulant(c)

    This yields

    .. math::
        c = (1,0,3,6)^\mathrm{T}

    .. math::
        C = \begin{bmatrix}
            1 & 6 & 3 & 0 \\
            0 & 1 & 6 & 3 \\
            3 & 0 & 1 & 6 \\
            6 & 3 & 0 & 1
        \end{bmatrix}

    This class depends on ``Fourier``, ``Diag``, ``Product`` and
    ``Partial``.

    .. todo::
        - sort out when optimizations are possible (real/complex, such stuff)
    """

    property vecC:
        r"""Return the matrix-defining column vector of the circulant matrix"""

        def __get__(self):
            return self._vecC

    ############################################## class methods
    def __init__(self, vecC, **options):
        '''
        Initialize Circulant Matrix instance.

        Circulant([ d c C C C]) represents this matrix:
          [ d C C C c ]
          [ c d C C C ]
          [ C c d C C ]
          [ C C c d C ]
          [ C C C c d ]

        the generating vector passed as vecC may be zero-padded to increase
        computation efficiency of involved fft operations. However, the padded
        zeros must be appended to the center of the generating column. The
        generating vector must head the column vector to define the resulting
        lower triangular matrix. The upper triangular is defined by a reversed
        copy of the vector at the end of column vector. This is short by one
        entry, which is already placed at the very first position. The minimal
        column vector length is one element short of twice the generating vector
        length. Thus, padding is only efficient for really bad behaved fourier
        transforms.
          [ d c C C C 0 0 0 0 0 0 0 c C C C]

        Valid options
        -------------
         'pad'=[FALSE true]
            perform zero-padding of Circulant for efficiency. By default set to
            False as padding introduces significant overhead to non-padded for
            most but some sizes.

        All options specified will also be passed on to the generation of the
        underlying Product instance initialization.
        '''

        # save generating vector. Matrix sizes will be set by Product
        # during its intitalization (invoked by 'super' below)
        self._vecC = vecC = np.atleast_1d(np.squeeze(np.copy(vecC)))

        assert self._vecC.ndim == 1, "Column-definition vector must be 1D."
        assert len(self._vecC) >= 1, "Vector must have at least one entry"

        cdef bint optimize = options.get('optimize', True)
        cdef int maxStage = options.get('maxStage', 4)

        cdef intsize size = len(vecC)
        cdef intsize paddedSize, minimalSize = size * 2 - 1
        cdef np.ndarray arrIndices

        # determine if zero-padding of the convolution to achieve a better FFT
        # size is beneficial or not
        if optimize:
            paddedSize = _findOptimalFFTSize(minimalSize, maxStage)

            assert paddedSize >= size * 2 - 1

            if _getFFTComplexity(size) > _getFFTComplexity(paddedSize):
                # zero-padding pays off, so do it!
                vecC = np.concatenate([vecC,
                                       np.zeros((paddedSize - minimalSize,),
                                                dtype=vecC.dtype),
                                       vecC[1:]])
                size = paddedSize

        # Describe circulant matrix as product of data and vector in fourier
        # domain. Both fourier matrices cause scaling of the data vector by
        # size, which will be compensated in Diag().

        # Create inner product
        cdef Fourier FN = Fourier(size)
        cdef Product P = Product(FN.H, Diag(np.fft.fft(vecC, axis=0) / size),
                                 FN, **options)

        # initialize Partial of Product. Only use Partial when padding size
        if size == len(self._vecC):
            super(Circulant, self).__init__(P)
        else:
            # generate index array once to save memory by one shared reference
            arrIndices = np.arange(len(self._vecC))
            super(Circulant, self).__init__(P, N=arrIndices, M=arrIndices)

        # Currently Fourier matrices bloat everything up to complex double
        # precision, therefore make sure vecC matches the precision of the
        # matrix itself
        if self.dtype != self._vecC.dtype:
            self._vecC = self._vecC.astype(self.dtype)

    cpdef np.ndarray _getArray(self):
        '''Return an explicit representation of the matrix as numpy-array.'''
        return self._reference()

    ############################################## class property override
    cpdef object _getItem(self, intsize idxN, intsize idxM):
        return self._vecC[(idxN - idxM) % self.numN]

    cpdef np.ndarray _getCol(self, intsize idx):
        '''Return selected columns of self.array'''
        cdef np.ndarray arrRes = _arrEmpty(
            1, self.numN, 0, self._info.dtype[0].typeNum)
        self._roll(arrRes, idx)
        return arrRes

    cpdef np.ndarray _getRow(self, intsize idx):
        '''Return selected rows of self.array'''
        cdef np.ndarray arrRes = _arrEmpty(
            1, self.numN, 0, self._info.dtype[0].typeNum)
        self._roll(arrRes[::-1], self.numN - idx - 1)
        return arrRes

    cpdef Matrix _getGram(self):
        cdef Fourier F = self.content[0].content[2]
        return Circulant(np.fft.ifft(abs(np.fft.fft(self.vecC)) ** 2))

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        return (0., 0.)

    ############################################## internal roll core
    cdef void _roll(self, np.ndarray vecOut, intsize shift):
        '''Return self.vecC rolled by 'shift' elements.'''
        if shift == 0:
            vecOut[:] = self._vecC
        else:
            vecOut[:shift] = self._vecC[self.numN - shift:]
            vecOut[shift:] = self._vecC[:self.numN - shift]

    cpdef Matrix _getNormalized(self):
        return self * (1. / np.linalg.norm(self._vecC))

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        cdef np.ndarray arrRes
        cdef intsize ii, N = self.numN, M = self.numM

        arrRes = np.empty((N, M), dtype=self.dtype)
        arrRes[:, 0] = self._vecC
        for ii in range(N):
            self._roll(arrRes[:, ii], ii)

        return arrRes

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                # 35 is just any number that causes no padding
                # 41 is the first size for which bluestein is faster
                TEST.NUM_N      : TEST.Permutation([31, 41]),
                TEST.NUM_M      : TEST.NUM_N,
                'mTypeC'        : TEST.Permutation(TEST.ALLTYPES),
                'optimize'      : True,
                TEST.PARAMALIGN : TEST.Permutation(TEST.ALLALIGNMENTS),
                'vecC'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeC',
                    TEST.SHAPE  : (TEST.NUM_N, ),
                    TEST.ALIGN  : TEST.PARAMALIGN
                }),
                TEST.INITARGS   : (lambda param : [param['vecC']()]),
                TEST.INITKWARGS : {'optimize' : 'optimize'},
                TEST.OBJECT     : Circulant,
                TEST.NAMINGARGS : dynFormat("%s,optimize=%s",
                                            'vecC', str('optimize')),
                TEST.TOL_POWER  : 2.,
                TEST.TOL_MINEPS : _getTypeEps(np.float64)
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c:
                                   Circulant(np.random.randn(2 ** c))),
                BENCH.FUNC_SIZE : (lambda c: 2 ** c),
                BENCH.FUNC_STEP : (lambda c: c + 1),
            },
            BENCH.FORWARD: {
                BENCH.FUNC_GEN  : (lambda c: Circulant(np.random.randn(c),
                                                       pad=True)),
                BENCH.FUNC_SIZE : (lambda c: c),
                BENCH.FUNC_STEP : (lambda c: c * 10 ** (1. / 12))
            },
            BENCH.SOLVE: {},
            BENCH.OVERHEAD: {},
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, dt: Circulant(
                    np.random.randn(2 ** c).astype(dt)))
            }
        }

    def _getDocumentation(self):
        return ""
