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
import numpy.linalg as npl

from .Algorithm import Algorithm
from ..Matrix import Matrix


class OMP(Algorithm):
    r"""Orthogonal Matching Pursuit

    **Definition and Interface**:
    For a given matrix :math:`A \in \mathbb{C}^{m \times N}` with
    :math:`m \ll N` and a vector :math:`b \in \mathbb{C}^m` we approximately
    solve

    .. math::
        \min\limits_{ x \in \mathbb{C}^N} \Vert x \Vert_0
        \quad\mathrm{s.t.}\quad A \cdot x =  x.

    If it holds that :math:`b =  A \cdot  x_0` for some :math:`k`-sparse
    :math:`x_0` and :math:`k` is low enough, we can recover :math:`x_0` via
    OMP [2]_.

    This type of problem as the one described above occurs in Compressed
    Sensing and Sparse Signal Recovery, where signals are approximated by
    sparse representations.

    >>> # import the packages
    >>> import numpy.linalg as npl
    >>> import numpy as np
    >>> import fastmat as fm
    >>> import fastmat.algorithms as fma
    >>> # define the dimensions
    >>> # and the sparsity
    >>> n, k = 512, 3
    >>> # define the sampling positions
    >>> t = np.linspace(0, 20 * np.pi, n)
    >>> # construct the convolution matrix
    >>> c = np.cos(2 * t)
    >>> C = fm.Circulant(c)
    >>> # create the ground truth
    >>> x = np.zeros(n)
    >>> x[npr.choice(range(n), k, replace=0)] = 1
    >>> b = C * x
    >>> # reconstruct it
    >>> omp = fma.OMP(C, numMaxSteps=100)
    >>> y = omp.process(b)
    >>> # test if they are close in the
    >>> # domain of C
    >>> print(npl.norm(C * y - b))

    We describe a sparse deconvolution problem, where the signal is in
    :math:`\mathbb{R}^{512}` and consists of :math:`3` windowed cosine pulses
    of the form :math:`c` with circulant displacement. Then we take the
    convolution and try to recover the location of the pulses using the OMP
    algorithm.

    .. note::
     The algorithm exploits two mathematical shortcuts. First it obviously
     uses the fast transform of the involved system matrix during the
     correlation step and second it uses a method to calculate the pseudo
     inverse after a rank-:math:`1` update of the matrix.

    .. todo::
      - optimize einsum-stuff

    Parameters
    ----------
    fmatA : fm.Matrix
        the system matrix
    arrB : np.ndarray
        the measurement vector
    numMaxSteps : int
        the desired sparsity order

    Returns
    -------
    np.ndarray
        solution array
    """

    def __init__(self, fmatA, **kwargs):

        # check the must-have parameters
        if not isinstance(fmatA, Matrix):
            raise TypeError("fmatA must be a fastmat matrix")
        self.fmatA = fmatA

        # set default parameters (and create attributes)
        self.numMaxSteps = None

        # Update with extra arguments
        self.updateParameters(**kwargs)

    def _process(self, arrB):
        #     fmatA           - input system matrix
        #     arrB            - input data vector (measurements)
        #     numMaxSteps            - specified sparsity order, i.e. number of
        #                       iterations to run
        #     numN,numM       - number of rows / columns of the system matrix
        #     numL            - number of problems to solve
        #     arrDiag         - array that contains column norms of fmatA
        #     numStrideSize   - size of strides during norm calculation
        #     numStrides      - number of whole strides to go through
        #     numPreSteps     - number of entries that do not fit in whole
        #                       strides
        #

        if arrB.ndim > 2:
            raise ValueError("Only n x m arrays are supported for OMP")

        if arrB.ndim == 1:
            self.arrB = arrB.reshape((-1, 1))
        else:
            self.arrB = arrB

        # get the number of vectors to operate on
        self.numN, self.numM, self.numL = \
            self.fmatA.numN, self.fmatA.numM, self.arrB.shape[1]

        self.fmatC = self.fmatA.normalized

        # determine return value data type
        self.returnType = np.promote_types(
            np.promote_types(self.fmatC.dtype, self.arrB.dtype),
            np.float32
        )

        # temporary array to store only support entries in
        self.arrXtmp = np.zeros(
            (self.numMaxSteps, self.numL),
            dtype=self.returnType
        )

        # initital residual is the measurement
        self.arrResidual = self.arrB.astype(self.returnType, copy=True)

        # list containing the support
        self.arrSupport = np.empty(
            (self.numMaxSteps, self.numL),
            dtype=np.intp
        )

        # matrix B that contains the pseudo inverse of A restricted to the
        # support
        self.matPinv = np.zeros(
            (self.numMaxSteps, self.numN, self.numL), dtype=self.returnType
        )

        # A restricted to the support
        self.arrA = np.zeros(
            (self.numN, self.numMaxSteps, self.numL), dtype=self.returnType
        )

        # different helper variables
        self.v2 = np.empty((self.numN, self.numL), dtype=self.returnType)
        self.v2n = np.empty((self.numN, self.numL), dtype=self.returnType)
        self.v2y = np.empty((self.numL, ), dtype=self.returnType)
        self.newCols = np.empty((self.numN, self.numL), dtype=self.returnType)
        self.arrC = np.empty((self.numM, self.numL), dtype=self.returnType)
        self.newIndex = np.empty((self.numL, ), dtype=np.intp)

        # iterativly build up the solution
        for self.numStep in range(self.numMaxSteps):
            # shorten access to index variable
            ii = self.numStep

            # do the normalized correlation step
            self.arrC = np.abs(self.fmatC.backward(self.arrResidual))

            # pick the maximum index in each correlation array
            self.newIndex = np.apply_along_axis(np.argmax, 0, self.arrC)

            # add these to the support
            self.arrSupport[ii, :] = self.newIndex

            # get the newly picked columns of A
            self.newCols = self.fmatA.getCols(self.newIndex)

            # store them into the submatrix
            self.arrA[:, ii, :] = self.newCols

            # in the first step everything is simple
            if ii == 0:
                self.v2 = self.newCols
                self.v2n = (self.v2 / npl.norm(self.v2, axis=0) ** 2).conj()

                self.v2y = np.einsum('ji,ji->i', self.v2n, self.arrB)

                self.arrXtmp[0, :] = self.v2y
                self.matPinv[0, :, :] = self.v2n
            else:
                self.v1 = np.einsum(
                    'ijk,jk->ik', self.matPinv[:ii, :, :], self.newCols
                )

                self.v2 = self.newCols - np.einsum(
                    'ijk,jk->ik', self.arrA[:, :ii, :], self.v1
                )
                self.v2n = (self.v2 / npl.norm(self.v2, axis=0) ** 2).conj()

                self.v2y = np.einsum('ji,ji->i', self.v2n, self.arrB)

                self.arrXtmp[:ii, :] -= self.v2y * self.v1
                self.arrXtmp[ii, :] += self.v2y

                self.matPinv[:ii, :, :] -= np.einsum(
                    'ik,jk->jik', self.v2n, self.v1
                )
                self.matPinv[ii, :, :] = self.v2n

            # update the residual
            self.arrResidual -= self.v2y * self.v2

        # return the computed vector
        self.arrX = np.zeros((self.numM, self.numL), dtype=self.returnType)
        self.arrX[self.arrSupport, np.arange(self.numL)] = self.arrXtmp

        return self.arrX

    @staticmethod
    def _getTest():
        from ..inspect import TEST, dynFormat, arrSparseTestDist
        from ..core.types import getTypeEps
        from ..Product import Product
        from ..Hadamard import Hadamard
        from ..Matrix import Matrix

        def testOMP(test):
            # prepare vectors
            numM = test[TEST.NUM_M]
            test[TEST.REFERENCE] = test[TEST.ALG_MATRIX].reference()
            test[TEST.RESULT_REF] = np.hstack([
                arrSparseTestDist(
                    (numM, 1),
                    dtype=test[TEST.DATATYPE],
                    density=1. * test['numK'] / numM
                ).toarray()
                for nn in range(test[TEST.DATACOLS])
            ])
            test[TEST.RESULT_INPUT] = test[TEST.ALG_MATRIX].array.dot(
                test[TEST.RESULT_REF]
            )
            test[TEST.RESULT_OUTPUT] = test[TEST.INSTANCE].process(
                test[TEST.RESULT_INPUT]
            )

        return {
            TEST.ALGORITHM: {
                'order': 6,
                TEST.NUM_N: (lambda param: 3 * param['order']),
                TEST.NUM_M: (lambda param: 2 ** param['order']),
                'numK': 'order',
                'maxSteps': 3,
                TEST.ALG_MATRIX: lambda param:
                    Product(Matrix(np.random.uniform(
                        -100, 100, (getattr(param, TEST.NUM_M),
                                    getattr(param, TEST.NUM_M))).astype(
                                        param['typeA'])),
                            Hadamard(param.order),
                            typeExpansion=param['typeA']),
                'typeA': TEST.Permutation(TEST.ALLTYPES),

                TEST.OBJECT: OMP,
                TEST.INITARGS: [TEST.ALG_MATRIX],
                TEST.INITKWARGS: {
                    'numMaxSteps': 'maxSteps'
                },

                TEST.DATAALIGN: TEST.ALIGNMENT.DONTCARE,
                TEST.INIT_VARIANT: TEST.IgnoreFunc(testOMP),

                'strTypeA': (lambda param: TEST.TYPENAME[param['typeA']]),
                TEST.NAMINGARGS: dynFormat(
                    "(%dx%d)*Hadamard(%s)[%s]",
                    TEST.NUM_N,
                    TEST.NUM_M,
                    'order',
                    'strTypeA'
                ),

                # matrix inversion always expands data type to floating-point
                TEST.TYPE_PROMOTION: np.float32,
                TEST.TOL_MINEPS: getTypeEps(np.float32),
                TEST.TOL_POWER: 5.,
                TEST.CHECK_PROXIMITY: False
            },
        }

    @staticmethod
    def _getBenchmark():
        return {}

    @staticmethod
    def _getDocumentation():
        from ..inspect import DOC
        return ""
