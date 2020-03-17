# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False

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
from .Hadamard cimport Hadamard
from .core.types cimport *
from .core.cmath cimport *
from .core.strides cimport *

cdef inline np.int8_t lfsrOutBit(lfsrReg_t state) nogil:
    return (-1 if state & 1 else 1)

cdef inline lfsrReg_t lfsrGenStep(lfsrReg_t state, lfsrReg_t polynomial,
                                  lfsrReg_t mask) nogil:
    cdef lfsrReg_t tmp = state & polynomial
    tmp ^= tmp >> 1
    tmp ^= tmp >> 2
    tmp = (tmp & 0x11111111) * 0x11111111
    if tmp & 0x10000000:
        state |= mask

    return state >> 1

cdef inline lfsrReg_t lfsrTapStep(lfsrReg_t state, lfsrReg_t polynomial,
                                  lfsrReg_t mask) nogil:
    state = state << 1
    if (state & mask) != 0:
        state = state ^ (polynomial | mask)

    return state

cdef class LFSRCirculant(Matrix):
    r"""

    Linear Feedback Shift Registers (LFSR) as implemented in this class are
    finite state machines generating sequences of symbols from the finite
    field :math:`F=[-1, +1]`. A shift register of size :math:`N` is a cascade
    of :math:`N` storage elements :math:`a_n` for :math:`n = 0,\dots,N-1`,
    each holding one symbol of :math:`F`. The state of the shift register is
    defined by the states of :math:`a_0,\dots,a_{N-1}`. :ref:`[5]<ref5>`

    The next state of the register is generated from the current state by
    moving the contents of each storage element to the next lower index by
    setting :math:`a_{n-1} = a_n` for :math:`n \geq 1`, hence the name shift
    register. The element :math:`a_0` of the current state is discarded
    completely in the next state. A subset :math:`T` of all storage elements
    with cardinality of 1 or greater is used for generating the next symbol
    :math:`a_{N-1}` by multiplication within :math:`F`. :math:`T` is called
    the tap configuration of the shift register.

    The output sequence of the register is the sequence of symbols
    :math:`a_0` for each state of the register. When the shift register
    repeats one of its previous states after :math:`L` state transistions,
    the output sequence also repeats and thus is periodic with a length
    :math:`L`. Evaluation of the sequence starts with all storage elements set
    to an initial state :math:`I`. Only periodic sequences of length
    :math:`L > 1` are considered if they also repeat all states including the
    initial state and thus form a hamilton circle an the graph corresponding
    to the chosen shift register size :math:`N` and tap configuration
    :math:`T`.

    Instanciation of this matrix class requires supplying the requested
    register size :math:`N`, the tap configuration and the initial state.
    The latter two are required to be supplied as binary words of up to
    :math:`N` bits. A one bit on position :math:`i` in the tap configuration
    adds :math:`a_i` as \*feedback tap\* to :math:`T`. At least one feedback
    tap must be supplied. The bits in the given initial state word :math:`r`
    will be mapped to the initial register state, where :math:`r_n = 0` sets
    :math:`a_n = +1` and :math:`r_n = 1` sets :math:`a_n = -1`. If no :math:`r`
    is given, it is assumed to be all-ones.

    >>> # import the package
    >>> import fastmat as fm
    >>>
    >>> # construct the parameter
    >>> polynomial = 0b11001
    >>> start = 0b1010
    >>>
    >>> # construct the matrix
    >>> L = fm.LFSRCirculant(polynomial, start)
    >>> s = L.vecC

    This yields a Circulant matrix where the column-definition vector is the
    output of a LFSR of size 4, which is configured to generate a maximum
    length sequence of length 15 and a cyclic shift corresponding to the given
    initial state.

    .. math::
        s = [+1, -1, +1, -1, -1, +1, +1, -1, +1, +1, +1, -1, -1, -1, -1]

    .. math::
        L = \begin{bmatrix}
        +1 &   -1   & -1 &        & -1 \\
        -1 &   +1   & -1 & \dots  & +1 \\
        +1 &   -1   & +1 &        & -1 \\
        & \vdots &    & \ddots &    \\
        -1 &   -1   & -1 &        & +1 \\
        \end{bmatrix}

    This class depends on ``Hadamard``.
    """

    property size:
        '''Deprecated. Will be removed in future releases'''
        def __get__(self):
            import warnings
            warnings.warn(
                'size is deprecated. WIll be removed in furure releases.',
                FutureWarning
            )
            return self.order

    property taps:
        '''Deprecated. See .polynomial'''
        def __get__(self):
            import warnings
            warnings.warn('taps is deprecated. Use polynomial.',
                          FutureWarning)
            return self.polynomial

    property vecC:
        r"""Return the sequence defining the circular convolution.

        *(read only)*
        """

        def __get__(self):
            return (self._getVecC() if self._vecC is None
                    else self._vecC)

    property states:
        r"""Return the internal register states during the sequence.

        *(read-only)*
        """

        def __get__(self):
            return (self._getStates() if self._states is None
                    else self._states)

    def __init__(self, polynomial, start, **options):
        '''
        Initialize a LFSR Circulant matrix instance.

        The definition vector of the circulant matrix is defined by the output
        [+1/-1] of a binary Linear Feedback Shift Register (LFSR) with the
        given defining parameters over one period.

        Parameters
        ----------
        polynomial : int
            The characteristic polynomial corresponding to the shift register
            sequence. Every set bit k in this value corresponds to one feedback
            tap at storage element k of the register or the monome x^k of the
            characterisctic polynomial that forms a cycle in the galois field
            GF2 of the order corresponding to the highest non-zero monome x^K
            in the polynomial.

        start : int
            The initial value of the storage elements of the register.

        **options : optional
            Additional keyworded arguments. Supports all optional arguments
            supported by :py:class:`fastmat.Matrix`.

            All optional arguments will be passed on to all
            :py:class:`fastmat.Matrix` instances that are generated during
            initialization.
        '''

        self.polynomial = polynomial
        self.order = 0
        cdef lfsrReg_t mask = 1
        while (~mask & polynomial > mask):
            mask <<=1
            self.order += 1

        mask = 1 << self.order

        # determine register size (determines order of embedded Hadamard)
        self.start = start & (mask - 1)

        if self.order > 31 or self.order < 1:
            raise ValueError("Only polynomials of order 1 to 31 are supported.")

        if self.start == 0:
            raise ValueError("Initial state must be non-zero.")

        # generate embedded Hadamard matrix
        self._content = (Hadamard(self.order), )

        # determine matrix size (square) by determining the period of the
        # sequence generated by the given register configuration
        cdef lfsrReg_t state = lfsrGenStep(self.start, self.polynomial, mask)
        cdef intsize period = 1

        while state != self.start:
            state = lfsrGenStep(state, polynomial, mask)
            period += 1
            if period >= mask or state == 0:
                raise ValueError(
                    "Register configuration produces invalid sequence.")

        self.period = period

        # extension note: it should be possible -- by properly tweaking the
        # involved permutation matrices -- to also construct a convolution
        # matrix of a section of the sequence. For that it might be neccessary
        # to have different starting values for the two polynomials generating
        # the actual permutation operations. If that turns out true, a fourth
        # parameter might be passed specifying the desired length of the
        # convolution operator matrix (from the specified start point). As the
        # period is generated anyway it should also be possible to determine
        # the start value for the alternate polynomial. However, confirmation
        # is required before implementation

        # set properties of matrix
        self._cythonCall = True
        self._initProperties(self.period, self.period, np.int8, **options)

    ############################################## class property override
    cpdef object _getItem(self, intsize idxRow, intsize idxCol):
        return self.vecC[(idxRow - idxCol) % self.numRows]

    cpdef np.ndarray _getCol(self, intsize idx):
        cdef np.ndarray arrRes = _arrEmpty(1, self.numRows, 0, self.numpyType)
        self._roll(arrRes, idx)
        return arrRes

    cpdef np.ndarray _getRow(self, intsize idx):
        cdef np.ndarray arrRes = _arrEmpty(1, self.numRows, 0, self.numpyType)
        self._roll(arrRes[::-1], self.numRows - idx - 1)
        return arrRes

    cpdef np.ndarray _getColNorms(self):
        return np.full((self.numCols, ), np.sqrt(self.numCols))

    cpdef np.ndarray _getRowNorms(self):
        return np.full((self.numRows, ), np.sqrt(self.numRows))

    cpdef Matrix _getColNormalized(self):
        return self * (1. / np.sqrt(self.numCols))

    cpdef Matrix _getRowNormalized(self):
        return self * (1. / np.sqrt(self.numRows))

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        cdef float complexity = 2 * self._content[0].numRows + 2 * self.numRows
        return (complexity, complexity)

    ############################################## class core methods
    cdef void _roll(self, np.ndarray vecOut, intsize shift):
        if shift == 0:
            vecOut[:] = self.vecC
        else:
            vecOut[:shift] = self.vecC[self.numRows - shift:]
            vecOut[shift:] = self.vecC[:self.numRows - shift]

    cdef np.ndarray _getStates(self):
        cdef intsize nn
        cdef np.ndarray arrStates
        cdef lfsrReg_t[:] mvStates

        cdef ntype typeStates = np.dtype(np.uint32).type_num
        cdef lfsrReg_t mask = 1 << self.order
        cdef lfsrReg_t polynomial = self.polynomial
        cdef lfsrReg_t state = self.start

        arrStates = _arrEmpty(1, self.numRows, 1, typeStates)
        mvStates = arrStates
        for nn in range(self.numRows):
            mvStates[nn] = state & (mask - 1)
            state = lfsrGenStep(state, polynomial, mask)

        self._states = arrStates
        return arrStates

    cdef np.ndarray _getVecC(self):
        cdef lfsrReg_t mask = 1 << self.order
        cdef lfsrReg_t polynomial = self.polynomial
        cdef lfsrReg_t state = self.start
        cdef intsize nn

        cdef np.ndarray arrRes
        cdef np.int8_t[:] mvRes
        cdef ntype typeElement = np.dtype(np.int8).type_num

        arrRes = _arrEmpty(1, self.numRows, 1, typeElement)
        mvRes = arrRes
        for nn in range(self.numRows):
            mvRes[nn] = lfsrOutBit(state)
            state = lfsrGenStep(state, polynomial, mask)

        self._vecC = arrRes
        return self._vecC

    cdef void _core(self, np.ndarray arrIn, np.ndarray arrOut,
                    bint flipIn, bint flipOut):

        cdef intsize ii, nn, N = arrIn.shape[0], M = arrIn.shape[1]
        cdef lfsrReg_t state, mask  = 1 << self.order
        cdef lfsrReg_t polynomial   = self.polynomial
        cdef np.ndarray arrData
        cdef STRIDE_s strIn, strData, strOut

        # initialize data Array, no zero init required for maximum length seqs
        if N == mask - 1:
            arrData = _arrEmpty(2, mask, M, getNumpyType(arrIn), False)
        else:
            arrData = _arrZero(2, mask, M, getNumpyType(arrIn), False)

        strideInit(&strIn, arrIn, 1)
        strideInit(&strOut, arrOut, 1)
        strideInit(&strData, arrData, 1)

        # ensure the first element of each vector in arrData is zero
        opZeroVector(&strData, 0)

        # flipping the arrays achieves circulant shape without actually shifting
        # anything (the original algorithm computes deconvolution which builds
        # an Hankel-like matrix with the sequence elements along anti-diagonals)
        # For the conversion to take place flip the input array during the
        # forward, and the output array during the backward transform.
        # Note, that the flip can be efficiently realized by twiddling around
        # with row indexing

        # apply input permutation P(c->g) from arrIn to arrData (through slices)
        # reset state of address register G resembles sequence start point
        state = self.start

        # copy first element from each imput vector to Hadamard transform input
        # data position denoted by the register's start value (copy row to row)
        # then, copy remaining elements (flip them if requested by flipIn)
        # don't flip the first item (which we will we copy explicitly now)
        opCopyVector(&strData, state, &strIn, 0)
        strideSliceVectors(&strIn, 1, -1, 1)        # equivalent to strIn[1:]

        # and now for something completely different: addressing hack!
        # to flip the input we can fiddle around with the array base pointer
        # and its strides. Inverting the slice-stride (keeping row distance but
        # changing direction of traversal) and setting the base pointer to the
        # last element in the array achieves what we want.
        # However, index 0 will be mapped directly to the can as it literally
        # starts after the array data. After that however indexing begins with
        # the last row for index 1.
        if flipIn:
            strideFlipVectors(&strIn)

        for nn in range(0, N - 1):
            state = lfsrGenStep(state, polynomial, mask)
            opCopyVector(&strData, state, &strIn, nn)

        # apply hadamard-walsh transform from arrData to arrData
        arrData = self._content[0].forward(arrData)

        # reinit slice as arrData has changed
        strideInit(&strData, arrData, 1)

        # apply output permutation P(r->t) fromm arrData to arrOut (slices!)
        # reset state for address register T must always be 1 (unclear in paper)
        state = 1

        # copy elements from Hadamard transform output denoted by the Tap
        # register's start value to first element of result (copy row to row)
        # then, copy remaining elements (flip them if requested by flipOut)
        # don't flip the first item (which we will we copy explicitly now)
        opCopyVector(&strOut, 0, &strData, state)
        strideSliceVectors(&strOut, 1, -1, 1)        # equivalent to strOut[1:]

        # same addressing hack as above
        if flipOut:
            strideFlipVectors(&strOut)

        for nn in range(0, N - 1):
            state = lfsrTapStep(state, polynomial, mask)
            opCopyVector(&strOut, nn, &strData, state)

    ############################################## class forward / backward
    cpdef _forwardC(self, np.ndarray arrX, np.ndarray arrRes,
                    ftype typeX, ftype typeRes):
        # dispatch input ndarray to type specialization
        self._core(arrX, arrRes, True, False)

    cpdef _backwardC(self, np.ndarray arrX, np.ndarray arrRes,
                     ftype typeX, ftype typeRes):
        # dispatch input ndarray to type specialization
        self._core(arrX, arrRes, False, True)

    ############################################### class reference
    cpdef np.ndarray _reference(self):
        cdef np.ndarray arrRes, vecSequence
        cdef np.int8_t[:] mvSequence
        cdef int ii, state, polynomial, mask, tmp, cnt

        state = self.start
        polynomial = self.polynomial
        mask = 1 << self.order
        vecSequence = np.empty((self.numRows, ), dtype=np.int8)
        mvSequence = vecSequence
        for ii in range(self.numRows):
            mvSequence[ii] = (state & 1) * -2 + 1

            tmp = state & polynomial
            cnt = 0
            while tmp:
                cnt ^= 1
                tmp &= tmp - 1

            if cnt:
                state |= mask

            state >>= 1

        arrRes = np.empty((self.numRows, self.numRows), dtype=self.dtype)
        for ii in range(self.numRows):
            arrRes[:, ii] = np.roll(vecSequence, ii)

        return arrRes

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                # define matrix sizes and parameters
                'polynomial'    : TEST.Permutation([0x19, 0x17]),
                'start'         : TEST.Permutation([0xD, 0x1]),
                'period'        : (
                    lambda param: (15 if param['polynomial'] == 0x19 else 7)
                ),
                TEST.NUM_ROWS   : 'period',
                TEST.NUM_COLS   : TEST.NUM_ROWS,

                # define constructor for test instances and naming of test
                TEST.OBJECT     : LFSRCirculant,
                TEST.INITARGS   : ['polynomial', 'start'],
                TEST.NAMINGARGS : dynFormat(
                    "%x,%x", 'polynomial', 'start'
                )
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH

        # specify tap configurations for various maximum-length-sequences
        # the list contains the tap configurations of registers corresponding
        # to a register size of the list index of the element + 1.
        db = {ll + 1: tt for ll, tt in enumerate([
            0x3, 0x7, 0xB, 0x19, 0x25, 0x61, 0x89, 0x1C3,
            0x211, 0x481, 0xB03, 0x1C11, 0x2C41, 0x5803, 0xC001,
            0x1A011, 0x24001, 0x40801, 0xE4001, 0x120001, 0x280001,
            0x600001, 0x840001, 0x1C20001, 0x2400001, 0x7100001, 0xE400001,
            0x12000001, 0x28000001, 0x70000081, 0x90000001
        ])}

        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (
                    lambda c: LFSRCirculant(db[c], 0xFFFFFFFF)
                ),
                BENCH.FUNC_SIZE : (lambda c: (2 ** c) - 1),
                BENCH.FUNC_STEP : (lambda c: c + 1),
            },
            BENCH.FORWARD: {},
            BENCH.OVERHEAD: {},
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (
                    lambda c, dt: LFSRCirculant(db[c], 0xFFFFFFFF, minType=dt)
                )
            }
        }
