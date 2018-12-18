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

    property tenT:
        r"""Return the defining Tensor of Toeplitz matrix."""

        def __get__(self):
            return self._tenT

    property vecC:
        r"""Return the column-defining vector of Toeplitz matrix."""

        def __get__(self):
            import warnings
            warnings.warn('vecC is deprecated.', FutureWarning)
            return self._tenT[:self._arrDimCols[0]]

    property vecR:
        r"""Return the row-defining vector of Toeplitz matrix."""

        def __get__(self):
            import warnings
            warnings.warn('vecR is deprecated.', FutureWarning)
            return self._tenT[self._arrDimCols[0]:]

    def __init__(self, *args, **options):
        '''
        Initialize Toeplitz matrix instance.

        Parameter for one-dimensional case
        ----------------------------------
        vecC : :py:class:`numpy.ndarray`
            The generating column vector of the toeplitz matrix describing the
            first column of the matrix.

        vecR : :py:class:`numpy.ndarray`
            The generating row vector of the toeplitz matrix excluding the
            element corresponding to the first column, which is already defined
            in `vecC`.

        **options:
            See below.

        Parameter for one-or-multi-dimensional case
        -------------------------------------------
        tenT : :py:class:`numpy.ndarray`
            The generating nd-array defining the toeplitz tensor. The matrix
            data type is determined by the data type of this array. In this
            parameter variant the column- and row-defining vectors are given
            in one single vector. The intersection point between these two
            vectors is given in the `splitpoint` option.

        **options:
            See below.

        Options
        -------
        splitpoint : :py:class:`numpy.ndarray`
            A 1d vector specifying the split-point for row/column definition
            of each vector. If this option is not specified each level
            :math:`i` of `tenT` is assumed to have a square shape of size
            :math:`T \in \mathbb{C}^{d_i \times d_i}` with the corresponding
            dimension of `tenT` having :math:`d_i * 2 - 1` entries.

            Defaults to a splitpoint vetor corresponding to all-square levels.


        Also see the special options of :py:class:`fastmat.Fourier`, which are
        also supported by this matrix and the general options offered by
        :py:meth:`fastmat.Matrix.__init__`.


        '''

        # multiplex different parameter variants during initialization
        cdef np.ndarray vecC, vecR
        cdef np.ndarray arrSplit = np.array(options.pop('splitpoint', []))
        if len(args) == 1:
            # define the Matrix by a tensor defining its levels over axes
            self._arrT = args[0]
        elif len(args) == 2:
            if not all(isinstance(aa, np.ndarray) for aa in args):
                raise ValueError(
                    "You must specify two 1D-ndarrays containing the " +
                    "column- and row-definition vectors or one ndarray tensor"
                )

            if arrSplit.size != 0:
                raise ValueError(
                    "You must not define split points when supplying " +
                    "column- and row-definition vectors."
                )

            dataType = np.promote_types(args[0].dtype, args[1].dtype)
            vecC = _arrSqueeze(args[0].astype(dataType))
            vecR = _arrSqueeze(args[1].astype(dataType))
            if (vecC.ndim != 1) or (vecR.ndim != 1):
                raise ValueError(
                    "Column- and row-definition vectors must be 1D."
                )

            arrSplit = np.array(vecC.size)
            self._arrT = np.hstack(vecC, vecR)
        else:
            raise ValueError(
                "Invalid number of arguments to Toeplitz: Expecting exactly " +
                "one or two fixed arguments"
            )

        cdef np.ndarray tplShape = (<object> self._arrT).shape

        # If no splitpoint vector was either given in options or generated from
        # column- and row-definition vectors, assume square levels such that
        # each dimension must obey the axis size relation (2 * n - 1)
        if arrSplit.size == 0:
            if not all(((ll + 1) % 2 == 0)
                       for ll in tplShape):
                raise ValueError(
                    "Defining a tensor with non-square levels requires " +
                    "explicit split points."
                )

            arrSplit = (np.ndarray(tplShape) + 1) // 2
        if arrSplit.size != self._arrT.ndim:
            raise ValueError(
                "The split point vector must have one entry for each " +
                "dimension of the defining tensor"
            )
        elif arrSplit.ndim != 1:
            raise ValueError(
                "The split point vector must be 1D"
            )
        elif any(ll < 1 or ll >= tplShape[ii]
                 for ii, ll in enumerate(arrSplit)):
            raise ValueError(
                "Entry in splitpoint vector out of defining tensor bounds"
            )


        # save generating vectors. Matrix sizes will be set by Product
        dataType = self._arrT.dtype
        vecC = _arrSqueeze(vecC.astype(dataType, copy=True))
        vecR = _arrSqueeze(vecR.astype(dataType, copy=True))
        self._arrDimCols = np.array(vecC.size)
        self._arrDimRows = np.array(vecR.size + 1)
        self._tenT = np.hstack((vecC, vecR))

        # evaluate options passed to class
        cdef bint optimize = options.get('optimize', True)
        cdef int maxStage = options.get('maxStage', 4)

        # perform padding (if enabled) and generate vector
        cdef intsize size = self._tenT.size
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
            vec = np.concatenate([vecC,
                                  np.zeros((vecSize - size,),
                                           dtype=dataType),
                                  np.flipud(vecR)])
        else:
            vec = np.concatenate([vecC, np.flipud(vecR)])

        # Describe as circulant matrix with product of data and vector
        # in fourier domain. Both fourier matrices cause scaling of the
        # data vector by N, which will be compensated in Diag().

        # Create inner product
        cdef Fourier FN = Fourier(vecSize, **options)
        cdef Product P = Product(
            FN.H,
            Diag(np.fft.fft(vec, axis=0) / vecSize, **options),
            FN,
            **options
        )

        # initialize Partial of Product
        cdef dict kwargs = options.copy()
        kwargs['rows'] = (np.arange(self._arrDimCols[0])
                          if size != self._arrDimCols[0] else None)
        kwargs['cols'] = (np.arange(self._arrDimRows[0])
                          if size != self._arrDimRows[0] else None)

        super(Toeplitz, self).__init__(P, **kwargs)

        # Currently Fourier matrices bloat everything up to complex double
        # precision, therefore make sure vecC and vecR matches the precision of
        # the matrix itself
        if self.dtype != self._tenT.dtype:
            self._tenT = self._tenT.astype(self.dtype)

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        cdef np.ndarray arrRes
        cdef intsize numCols = self._arrDimCols[0]

        if idx == 0:
            return self._tenT[:numCols]
        elif idx >= self.numRows:
            # double slicing needed, otherwise fail when numCols = numRows + 1
            return self._tenT[numCols + idx - self.numRows:numCols + idx][::-1]
        else:
            arrRes = _arrEmpty(1, self.numRows, 0, self.numpyType)
            arrRes[:idx] = self._tenT[numCols + idx - 1::-1]
            arrRes[idx:] = self._tenT[:self.numRows - idx]
            return arrRes

    cpdef np.ndarray _getRow(self, intsize idx):
        cdef np.ndarray arrRes
        cdef intsize numCols = self._arrDimCols[0]

        if idx >= self.numCols - 1:
            # double slicing needed, otherwise fail when numRows = numCols + 1
            return self._tenT[idx - self.numCols + 1:idx + 1][::-1]
        else:
            arrRes = _arrEmpty(1, self.numCols, 0, self.numpyType)
            arrRes[:idx + 1] = self._tenT[idx::-1]
            arrRes[idx + 1:] = self._tenT[
                numCols:numCols + self.numCols - 1 - idx
            ]
            return arrRes

    cpdef object _getItem(self, intsize idxRow, intsize idxCol):
        cdef intsize distance = idxRow - idxCol
        return (self._tenT[-distance - 1] if distance < 0
                else self._tenT[distance])

    cpdef np.ndarray _getArray(self):
        return self._reference()

    cpdef np.ndarray _getColNorms(self):
        # NOTE: This method suffers accuracy losses when elements with lower
        # indices are large in magnitude compared to ones with higher index!
        cdef intsize iiMax = ((self.numRows + 1) if self.numCols > self.numRows
                              else self.numCols)

        # fill in a placeholder array
        cdef np.ndarray arrNorms = _arrZero(1, self.numCols, 1, np.NPY_FLOAT64)

        # compute the absolute value of the squared elements in the defining
        # vectors
        cdef intsize numCols = self._arrDimCols[0]
        cdef np.ndarray vecCSqr = np.square(np.abs(self._tenT[:numCols]))
        cdef np.ndarray vecRSqr = np.square(np.abs(self._tenT[numCols:]))

        # the first column is easy
        arrNorms[0] = vecCSqr.sum()

        # then follow the iterative approach. Every subsequent column features
        # one more element of vecR and one less of vecC. Continue until vecC is
        # eaten up completely
        for ii in range(1, iiMax):
            arrNorms[ii] = (arrNorms[ii - 1] + vecRSqr[ii - 1] -
                            vecCSqr[self.numRows - ii])

        # then (as vecC is eaten up), proceed with rolling the remaining
        # elements of vecR until they are represented fully in arrNorms
        for ii in range(iiMax, self.numCols):
            arrNorms[ii] = (arrNorms[ii - 1] +
                            vecRSqr[ii - 1] - vecRSqr[ii - iiMax])

        return np.sqrt(arrNorms)

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        return (0., 0.)

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        # _reference overloading from Partial is too slow. Therefore, construct
        # a reference directly from the vectors.
        cdef intsize ii, numCols = self._arrDimCols[0]
        cdef np.ndarray arrRes = np.empty(
            (self.numRows, self.numCols), dtype=self.dtype
        )

        # put columns in lower-triangular part of matrix
        for ii in range(0, min(self.numRows, self.numCols)):
            arrRes[ii:self.numRows, ii] = self._tenT[
                :(self.numRows - ii)
            ]

        # put rows in upper-triangular part of matrix
        for ii in range(0, min(self.numRows, self.numCols - 1)):
            arrRes[ii, (ii + 1):self.numCols] = self._tenT[
                numCols:numCols + (self.numCols - ii - 1)
            ]

        return arrRes

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                TEST.DATAALIGN  : TEST.ALIGNMENT.DONTCARE,
                # 35 is just any number that causes no padding
                # 41 is the first size for which bluestein is faster
                TEST.NUM_ROWS   : TEST.Permutation([5, 41]),
                'num_cols'      : TEST.Permutation([4, 6]),
                TEST.NUM_COLS   : (lambda param: param['num_cols'] + 1),
                'mTypeH'        : TEST.Permutation(TEST.FEWTYPES),
                'mTypeV'        : TEST.Permutation(TEST.FEWTYPES),
                'optimize'      : True,
                'vecH'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeH',
                    TEST.SHAPE  : (TEST.NUM_ROWS, )
                }),
                'vecV'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeV',
                    TEST.SHAPE  : ('num_cols', )
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
                TEST.NUM_ROWS      : 4,
                'num_M'         : TEST.Permutation([2, 3, 4, 5, 6]),
            },
            TEST.TRANSFORMS: {
                # during class tests we do not need to verify bluestein again
                TEST.NUM_ROWS   : TEST.Permutation([7]),
            },
            'interface': {
                TEST.TEMPLATE   : TEST.TRANSFORMS,
                TEST.INITKWARGS : {
                    'optimize'      : 'optimize',
                    'splitpoint'    : 'splitpoint'
                },
                # during class tests we do not need to verify bluestein again
                TEST.NUM_ROWS   : TEST.Permutation([7]),
                'splitpoint'    :
            }
        }

    def _getBenchmark(self):
        from .inspect import BENCH, arrTestDist
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c: Toeplitz(
                    arrTestDist((c, ), dtype=np.float32),
                    arrTestDist((c - 1, ), dtype=np.float32)))
            },
            BENCH.FORWARD: {},
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c: Toeplitz(
                    arrTestDist((2 ** c, ), dtype=np.float32),
                    arrTestDist((2 ** c - 1, ), dtype=np.float32)))
            },
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, datatype: Toeplitz(
                    arrTestDist((2 ** c, ), dtype=datatype),
                    arrTestDist((2 ** c - 1, ), dtype=datatype)))
            }
        }
