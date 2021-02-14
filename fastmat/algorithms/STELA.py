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

from .Algorithm import Algorithm
from ..Matrix import Matrix


class STELA(Algorithm):
    r"""Soft-Thresholding with simplified Exact Line search Algorithm (STELA)

    The algorithm is presented in [1]_ with derivation and convergence results.

    **Definition and Interface**:
    For a given matrix :math:`A \in \mathbb{C}^{m \times N}` with
    :math:`m \ll N` and a vector :math:`b \in \mathbb{C}^m` we approximately
    solve

    .. math::
        \min\limits_{ x \in \mathbb{C}^N}\Vert{ A \cdot  x -  b}\Vert^2_2 +
        \lambda \cdot \Vert x \Vert_1,

    where :math:`\lambda > 0` is a regularization parameter to steer the
    trade-off between data fidelity and sparsity of the solution.

    >>> # import the packages
    >>> import numpy.linalg as npl
    >>> import numpy as np
    >>> import fastmat as fm
    >>> import fastmat.algorithms as fma
    >>> # define the dimensions and the sparsity
    >>> n, k = 512, 3
    >>> # define the sampling positions
    >>> t = np.linspace(0, 20 * np.pi, n)
    >>> # construct the convolution matrix
    >>> c = np.cos(2 * t) * np.exp(-t ** 2)
    >>> C = fm.Circulant(c)
    >>> # create the ground truth
    >>> x = np.zeros(n)
    >>> x[np.random.choice(range(n), k, replace=0)] = 1
    >>> b = C * x
    >>> # reconstruct it
    >>> stela = fma.STELA(C, numLambda=0.005, numMaxSteps=100)
    >>> y = stela.process(b)
    >>> # test if they are close in the
    >>> # domain of C
    >>> print(npl.norm(C * y - b))

    We solve a sparse deconvolution problem, where the atoms are harmonics
    windowed by a gaussian envelope. The ground truth :math:`x` is build out
    of three pulses at arbitrary locations.

    .. note::
        The proper choice of :math:`\lambda` is crucial for good perfomance
        of this algorithm, but this is not an easy task. Unfortunately we are
        not in the place here to give you a rule of thumb what to do, since it
        highly depends on the application at hand. Again, consult [1]_ for any
        further considerations of this matter.

    .. [1]  Y. Yang, M. Pesavento, "A Unified Successive Pseudoconvex
            Approximation Framework", IEEE Transactions on Signal Processing,
            vol. 65, no. 13, pp. 3313-3327, Dec 2017


    Parameters
    ----------
    fmatA : fm.Matrix
        the system matrix
    arrB : np.ndarray
        the measurement vector
    numLambda : float, optional
        the thresholding parameter; default is 0.1
    numMaxSteps : int, optional
        maximum number of steps; default is 100
    numMaxError : float, optional
        maximum error tolerance; default is 1e-6

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
        self.numLambda = 0.1
        self.numMaxSteps = 100
        self.numMaxError = 1e-6

        # initialize callbacks
        self.cbStep = None

        # Update with extra arguments
        self.updateParameters(**kwargs)

    def softThreshold(self, arrX, numAlpha):
        r"""
        Do a soft thresholding step.
        """

        # arrM         - positive part of arrX - numAlpha
        # arrX         - vector to be thresholded
        # numAlpha     - thresholding threshold
        arrM = np.maximum(np.abs(arrX) - numAlpha, 0)
        return np.multiply((arrM / (arrM + numAlpha)), arrX)

    def _process(self, arrB):
        if arrB.ndim > 2 or arrB.ndim < 1:
            raise ValueError("Only n x m arrays are supported for STELA")

        if arrB.ndim == 1:
            self.arrB = arrB.reshape((-1, 1))
        else:
            self.arrB = arrB

        if self.numMaxSteps <= 0:
            raise ValueError(
                "STELA would like to do at least one step for you"
            )

        dtypeType = np.promote_types(np.float64, self.arrB.dtype)

        # step size
        self.arrGamma = np.zeros(self.arrB.shape[1], dtype=dtypeType)

        # current state vector
        self.arrX = np.zeros(
            (self.fmatA.numCols, self.arrB.shape[1]), dtype=dtypeType
        )

        # current gradient
        self.arrGrad = np.zeros_like(self.arrX, dtype=dtypeType)

        # residual vector
        self.arrRes = (-self.arrB).astype(dtypeType)

        # some intermediate vectors
        self.arrBx = np.zeros_like(self.arrX, dtype=dtypeType)
        self.arrABxx = np.zeros_like(self.arrB, dtype=dtypeType)

        # backprojection of the residual vector
        self.arrZ = self.fmatA.backward(self.arrRes)

        # squared norms of the system matrix
        self.arrD = (1.0 / self.fmatA._getColNorms() ** 2).reshape((-1, 1))

        # vector for the stopping criterion
        self.arrStop = np.ones(self.arrX.shape[1])

        # this vector keeps track of the still active measurements, where
        # we did not converge yet
        self.arrActive = np.ones(self.arrX.shape[1]) == 1

        # start iterating
        for self.numStep in range(self.numMaxSteps):

            # some utility vector (the scaled gradient) (17)
            self.arrGrad = self.arrD * self.arrX - self.arrZ

            # calculate the stopping criterion
            arrDiff = np.maximum(
                np.minimum(self.arrZ.real - self.arrX.real, +self.numLambda),
                -self.numLambda,
            )
            if dtypeType == complex:
                arrDiff = arrDiff + 1j * np.maximum(
                    np.minimum(
                        self.arrZ.imag - self.arrX.imag, +self.numLambda
                    ),
                    -self.numLambda,
                )
            self.arrStop = np.linalg.norm(
                self.arrZ - arrDiff,
                axis=0,
            )
            # now check if we converged for any snapshot
            self.arrActive = self.arrStop > self.numMaxError

            # if no snapshot is active anymore, we can stop entirely
            if np.sum(self.arrActive) == 0:
                return self.arrX

            # update of the intermediate vector (16)
            self.arrBx[:, self.arrActive] = (
                self.softThreshold(
                    self.arrGrad[:, self.arrActive], self.numLambda
                )
                / self.arrD
            )

            # cache some operations
            self.arrABxx[:, self.arrActive] = self.fmatA.forward(
                self.arrBx[:, self.arrActive] - self.arrX[:, self.arrActive]
            )

            # we can do exact line search in this case (19)
            # axis=0 accounts for the fact, that we might have multiple
            # measurements at hand.
            self.arrGamma[self.arrActive] = np.maximum(
                np.minimum(
                    -(
                        np.real(
                            np.sum(
                                np.multiply(
                                    np.conj(self.arrRes[:, self.arrActive]),
                                    self.arrABxx[:, self.arrActive],
                                ),
                                axis=0,
                            )
                        )
                        + self.numLambda
                        * (
                            np.sum(
                                np.abs(self.arrBx[:, self.arrActive])
                                - np.abs(self.arrX[:, self.arrActive]),
                                axis=0,
                            )
                        )
                    )
                    / np.sum(
                        np.abs(self.arrABxx[:, self.arrActive]) ** 2, axis=0
                    ),
                    1,
                ),
                0,
            )

            # update step (5)
            self.arrX[:, self.arrActive] += (
                self.arrBx[:, self.arrActive] - self.arrX[:, self.arrActive]
            ).dot(np.diag(self.arrGamma[self.arrActive]))

            # residual update (20)
            self.arrRes[:, self.arrActive] += (
                self.arrGamma[self.arrActive] * self.arrABxx[:, self.arrActive]
            )
            self.arrZ[:, self.arrActive] = self.fmatA.backward(
                self.arrRes[:, self.arrActive]
            )

            self.handleCallback(self.cbStep)
            self.handleCallback(self.cbTrace)

        # return the unthresholded values for all non-zero support elements
        # if we did not converge in the given number of steps
        return self.arrX

    @staticmethod
    def _getTest():
        from ..inspect import TEST, dynFormat, arrSparseTestDist
        from ..Product import Product
        from ..Hadamard import Hadamard
        from ..Matrix import Matrix

        def testSTELA(test):
            # prepare vectors
            numCols = test[TEST.NUM_COLS]
            test[TEST.REFERENCE] = test[TEST.ALG_MATRIX].reference()
            test[TEST.RESULT_REF] = np.hstack(
                [
                    arrSparseTestDist(
                        (numCols, 1),
                        dtype=test[TEST.DATATYPE],
                        density=1.0 * test["numK"] / numCols,
                    ).toarray()
                    for nn in range(test[TEST.DATACOLS])
                ]
            )
            test[TEST.RESULT_INPUT] = test[TEST.ALG_MATRIX].array.dot(
                test[TEST.RESULT_REF]
            )
            test[TEST.RESULT_OUTPUT] = test[TEST.INSTANCE].process(
                test[TEST.RESULT_INPUT]
            )

        return {
            TEST.ALGORITHM: {
                "order": 6,
                TEST.NUM_ROWS: (lambda param: 3 * param["order"]),
                TEST.NUM_COLS: (lambda param: 2 ** param["order"]),
                "numK": "order",
                "lambda": 0.1,
                "maxSteps": 3,
                TEST.ALG_MATRIX: lambda param: Product(
                    Matrix(
                        np.random.uniform(
                            -100,
                            100,
                            (
                                getattr(param, TEST.NUM_COLS),
                                getattr(param, TEST.NUM_COLS),
                            ),
                        ).astype(param["typeA"])
                    ),
                    Hadamard(param.order),
                    typeExpansion=param["typeA"],
                ),
                "typeA": TEST.Permutation(TEST.FLOATTYPES),
                TEST.OBJECT: STELA,
                TEST.INITARGS: [TEST.ALG_MATRIX],
                TEST.INITKWARGS: {
                    "numLambda": "lambda",
                    "numMaxSteps": "maxSteps",
                },
                TEST.DATAALIGN: TEST.ALIGNMENT.DONTCARE,
                TEST.INIT_VARIANT: TEST.IgnoreFunc(testSTELA),
                "strTypeA": (lambda param: TEST.TYPENAME[param["typeA"]]),
                TEST.NAMINGARGS: dynFormat(
                    "(%dx%d)*Hadamard(%s)[%s]",
                    TEST.NUM_ROWS,
                    TEST.NUM_COLS,
                    "order",
                    "strTypeA",
                ),
                # matrix inversion always expands data type to floating-point
                TEST.TYPE_PROMOTION: np.float64,
                TEST.CHECK_PROXIMITY: False,
            },
        }

    @staticmethod
    def _getBenchmark():
        return {}

    @staticmethod
    def _getDocumentation():
        from ..inspect import DOC

        return ""
