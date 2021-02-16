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

from .core.cmath cimport *
from .core.strides cimport *
from .Eye cimport Eye
from .Matrix cimport Matrix

cdef class Fourier(Matrix):
    r"""

    The Fourier Transform realizes the mapping

    .. math::
        x \mapsto {\mathcal{F}}_n \cdot  x,

    where the Fourier matrix :math:`{\mathcal{F}}_n` is uniquely defined by
    the size of the vectors it acts on.

    >>> # import the package
    >>> import fastmat as fm
    >>>
    >>> # define parameter
    >>> n = 4
    >>>
    >>> # construct the matrix
    >>> F = fm.Fourier(n)

    This yields a Fourier :math:`{\mathcal{F}}_4` matrix of size :math:`4`.
    As a library to provide the Fast Fourier Transform we used the one provided
    by NumPy.

    .. todo::
        - real valued transforms
    """

    property order:
        r"""Return the Order of the Fourier matrix.

        *(read-only)*
        """

        def __get__(self):
            return self._order

    def __init__(self, order, **options):
        '''
        Initialize Fourier matrix instance.

        Parameters
        ----------
        order : int
            The order of the DFT matrix represented by this matrix instance.

        optimize : bool, optional
            Allow application of the Bluestein algorithm for badly conditioned
            fourier transform orders.

            Defaults to True.

        maxStage : int, optional
            Specify the maximum butterfly element size for the FFT. Larger
            values can reduce the required order for the FFTs computed in the
            Bluestein case. However, increasing only makes sense as long as an
            efficient implementation of the butterfly structures exist in your
            BLAS.

            Defaults to 4, which is safe to assume on all architectures.
            However, most implementations support sizes of 5 and on some cpu
            architectures, also 7.

        **options : optional
            Additional optional keyworded arguments. Supports all optional
            arguments supported by :py:class:`fastmat.Matrix`.
        '''

        cdef intsize paddedSize
        cdef np.ndarray arrSamples, vecConv, arrArgument, arrBuffer

        cdef bint optimize = options.get('optimize', True)
        cdef int maxStage = options.get('maxStage', 4)

        if order < 1:
            raise ValueError("Fourier order cannot be smaller than 1.")

        # store order of Fourier
        self._order = order

        # determine if computation of the FFT via convolution is beneficial
        # first, detect the smallest size of a reasonably fast chirp-Z transform
        # start with the minimal size of 2 * N - 1
        if optimize:
            paddedSize = _findOptimalFFTSize(order * 2 - 1, maxStage)
            # check once more nothing went wrong up to this point
            assert paddedSize >= order * 2 - 1

            # now as we know where to pad to if we choose to, decide what is a
            # lesser pain to do. self._numL acts as a marker on whether to act
            # as a plain fft wrapper (== 0) or whether to use the chirp-z
            # transform in the latter case self._numL specifies the internal
            # dimension
            self._numL = (0 if (_getFFTComplexity(self.order) <
                                (2 * _getFFTComplexity(paddedSize) +
                                 2 * paddedSize + 2 * self.order))
                          else paddedSize)

        # if we convolve, then we should prepare some stuff
        # the presence (non-None) of self._vecConvHat controls the behaviour
        # of the forward() and backward() transforms.
        if self._numL > 0:

            # create the sampling grid
            arrSamples = np.linspace(0, self.order - 1, self.order)

            # evaluate at these sample points for the first numL elements
            # of the vector, also compose a premultiplication array for
            # preprocessing input data prior to the convolution.
            # Make use of trigonometric funtion symmetry for much faster
            # compututation of the complex exponentials (compared to using
            # np.exp(...) twice).
            vecConv = _arrEmpty(1, self._numL, 1, np.NPY_COMPLEX128)
            self._preMult = _arrEmpty(1, self.order, 0, np.NPY_COMPLEX128)
            arrArgument = (arrSamples ** 2) * np.pi / self.order
            arrBuffer = np.cos(arrArgument)
            vecConv.real[:self.order] = arrBuffer
            self._preMult[:] = arrBuffer
            np.sin(arrArgument, out=arrBuffer)
            vecConv.imag[:self.order] = arrBuffer
            self._preMult[:] = -arrBuffer

            # now put a flipped version of the above at the very end to get
            # a circular convolution
            vecConv[self._numL - self.order + 1:] = vecConv[1:self.order][::-1]

            # ... and fill the remainder (in the middle of vecConv) with zeros
            vecConv[self.order:self._numL - self.order + 1] = 0

            # transfer function of convolution
            self._vecConvHat = np.fft.fft(vecConv)

        # set properties of matrix
        self._initProperties(
            self._order, self._order, np.complex128, **options
        )

    ############################################## class property override
    cpdef np.ndarray _getArray(self):
        return np.fft.fft(np.eye(self._order, dtype=self.dtype), axis=0)

    cpdef np.ndarray _getRow(self, intsize idx):
        return self._getCol(idx)

    cpdef object _getLargestSingularValue(self):
        return np.sqrt(self._order)

    cpdef object _getLargestEigenValue(self):
        return np.sqrt(self._order)

    cpdef np.ndarray _getColNorms(self):
        return np.full((self.numCols, ), np.sqrt(self._order))

    cpdef np.ndarray _getRowNorms(self):
        return np.full((self.numRows, ), np.sqrt(self._order))

    cpdef Matrix _getColNormalized(self):
        return self * (1. / np.sqrt(self._order))

    cpdef Matrix _getRowNormalized(self):
        return self * (1. / np.sqrt(self._order))

    cpdef Matrix _getGram(self):
        return Eye(self._order) * self.dtype(self._order)

    cpdef object _getItem(self, intsize idxRow, intsize idxCol):
        return np.exp(idxRow * idxCol * -2j * np.pi / self.order).astype(
            self.dtype
        )

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        # handle the Bluestein case differently to reflect the implementation
        cdef float complexity = _getFFTComplexity(
            self._order if self._numL == 0 else self._numL
        )
        if self._numL > 0:
            complexity = 2 * complexity + 2 * self._order + 2 * self._numL

        return (complexity, complexity + 2 * self._order)

    ############################################## class forward / backward
    cpdef np.ndarray _forward(self, np.ndarray arrX):
        cdef np.ndarray arrRes
        cdef STRIDE_s strResPadding
        cdef intsize mm, M = arrX.shape[1]

        if self._numL == 0:
            arrRes = np.fft.fft(arrX, axis=0)
        else:
            arrRes = _arrEmpty(
                2, self._numL, M,
                typeInfo[promoteFusedTypes(
                    self.fusedType, getFusedType(arrX))].numpyType)

            strideInit(&strResPadding, arrRes, 0)
            strideSliceElements(&strResPadding, self.order, -1, 1)
            opZeroVectors(&strResPadding)

            arrRes[:self.order, :] = (self._preMult.T * arrX.T).T

            arrRes = (self._vecConvHat.T * np.fft.fft(arrRes, axis=0).T).T

            arrRes = (self._preMult.T *
                      np.fft.ifft(arrRes, axis=0)[:self.order, :].T).T

        return arrRes

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        cdef np.ndarray arrRes = self._forward(_conjugate(arrX))
        _conjugateInplace(arrRes)
        return arrRes

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        return np.exp(
            np.multiply(
                *np.meshgrid(np.arange(self._order), np.arange(self._order)
                             )) * -2j * np.pi / self._order
        ).astype(self.dtype)

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                # define matrix sizes and parameters
                # 35 is just any number with non-2 and non-4 primes
                # 89 is the first size for which bluestein is faster
                TEST.NUM_ROWS   : TEST.Permutation([35, 89]),
                TEST.NUM_COLS   : TEST.NUM_ROWS,
                'optimize'      : TEST.Permutation([False, True]),

                # define constructor for test instances and naming of test
                TEST.OBJECT     : Fourier,
                TEST.INITARGS   : [TEST.NUM_ROWS],
                TEST.INITKWARGS : {'optimize': 'optimize'},
                TEST.TOL_POWER  : 3.,
                TEST.NAMINGARGS : dynFormat("%d, optimize=%d", TEST.NUM_ROWS,
                                            'optimize'),
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c: Fourier(c))
            },
            BENCH.FORWARD: {},
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c: Fourier(2 ** c))
            },
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, datatype: Fourier(2 ** c))
            }
        }
