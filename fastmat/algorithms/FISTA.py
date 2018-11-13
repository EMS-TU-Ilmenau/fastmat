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

from .Algorithm import Algorithm
from ..Matrix import Matrix


class FISTA(Algorithm):
    r"""Fast Iterative Shrinking-Thresholding Algorithm (FISTA)

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
    >>> c = np.cos(2 * t)
    >>> C = fm.Circulant(c)
    >>> # create the ground truth
    >>> x = np.zeros(n)
    >>> x[npr.choice(range(n), k, replace=0)] = 1
    >>> b = C * x
    >>> # reconstruct it
    >>> fista = fma.FISTA(C, numLambda=0.005, numMaxSteps=100)
    >>> y = fista.process(b)
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

    .. todo::
        - Todos for ISTA
        - Check if its working

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

        # Update with extra arguments
        self.updateParameters(**kwargs)

    def softThreshold(self, arrX, numAlpha):
        r"""
        Do a soft thresholding step.
        """

        # arrM         - positive part of arrX - numAlpha
        # arrX         - vector to be thresholded
        # numAlpha     - thresholding threshold

        self.arrM = np.maximum(np.abs(arrX) - numAlpha, 0)
        return np.multiply((self.arrM / (self.arrM + numAlpha)), arrX)

    def _process(self, arrB):
        # Wrapper around the FISTA algrithm to allow processing of arrays of
        # signals
        #     fmatA         - input system matrix
        #     arrB          - input data vector (measurements)
        #     numLambda     - balancing parameter in optimization problem
        #                     between data fidelity and sparsity
        #     numMaxSteps   - maximum number of steps to run
        #     numL          - step size during the conjugate gradient step
        if arrB.ndim > 2:
            raise ValueError("Only n x m arrays are supported for FISTA")

        if arrB.ndim == 1:
            self.arrB = arrB.reshape((-1, 1))
        else:
            self.arrB = arrB

        # calculate the largest singular value to get the right step size
        self.numL = 1.0 / (self.fmatA.largestSingularVal ** 2)
        self.t = 1

        self.arrX = np.zeros(
            (self.fmatA.numCols, self.arrB.shape[1]),
            dtype=np.promote_types(np.float32, self.arrB.dtype)
        )
        # initial arrY
        self.arrY = np.copy(self.arrX)
        # start iterating
        for self.numStep in range(self.numMaxSteps):
            self.arrXold = np.copy(self.arrX)

            # do the gradient step and threshold
            self.arrStep = self.arrY - self.numL * self.fmatA.backward(
                self.fmatA.forward(self.arrY) - self.arrB
            )
            self.arrX = self.softThreshold(
                self.arrStep, self.numL * self.numLambda * 0.5
            )

            # update t
            tOld = self.t
            self.t = (1 + np.sqrt(1 + 4 * self.t ** 2)) / 2

            # update arrY
            self.arrY = self.arrX + ((tOld - 1) / self.t) * (
                self.arrX - self.arrXold
            )

        # return the unthresholded values for all non-zero support elements
        return np.where(self.arrX != 0, self.arrStep, self.arrX)

    @staticmethod
    def _getTest():
        from ..inspect import TEST, dynFormat, arrSparseTestDist
        from ..Product import Product
        from ..Hadamard import Hadamard
        from ..Matrix import Matrix

        def testFISTA(test):
            # prepare vectors
            numCols = test[TEST.NUM_COLS]
            test[TEST.REFERENCE] = test[TEST.ALG_MATRIX].reference()
            test[TEST.RESULT_REF] = np.hstack([
                arrSparseTestDist(
                    (numCols, 1),
                    dtype=test[TEST.DATATYPE],
                    density=1. * test['numK'] / numCols
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
                'order'         : 6,
                TEST.NUM_ROWS   : (lambda param: 3 * param['order']),
                TEST.NUM_COLS   : (lambda param: 2 ** param['order']),
                'numK'          : 'order',
                'lambda'        : 1.,
                'maxSteps'      : 10,
                'typeA'         : TEST.Permutation(TEST.ALLTYPES),

                TEST.ALG_MATRIX : lambda param:
                    Product(Matrix(np.random.uniform(
                        -100, 100, (getattr(param, TEST.NUM_COLS),
                                    getattr(param, TEST.NUM_COLS))).astype(
                                        param['typeA'])),
                            Hadamard(param.order),
                            typeExpansion=param['typeA']),
                TEST.OBJECT     : FISTA,
                TEST.INITARGS   : [TEST.ALG_MATRIX],
                TEST.INITKWARGS : {
                    'numLambda'     : 'lambda',
                    'numMaxSteps'   : 'maxSteps'
                },


                TEST.DATAALIGN  : TEST.ALIGNMENT.DONTCARE,
                TEST.INIT_VARIANT: TEST.IgnoreFunc(testFISTA),

                'strTypeA'      : (lambda param: TEST.TYPENAME[param['typeA']]),
                TEST.NAMINGARGS: dynFormat(
                    "(%dx%d)*Hadamard(%s)[%s]",
                    TEST.NUM_ROWS,
                    TEST.NUM_COLS,
                    'order',
                    'strTypeA'
                ),

                # matrix inversion always expands data type to floating-point
                TEST.TYPE_PROMOTION: np.float32,
                TEST.CHECK_PROXIMITY: False
            },
        }

    @staticmethod
    def _getBenchmark():
        return {}
