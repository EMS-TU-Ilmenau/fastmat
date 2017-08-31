# -*- coding: utf-8 -*-
'''
  fastmat/algs/OMP.py
 -------------------------------------------------- part of the fastmat package

  Implementation of orthogonal matching pursuit (OMP) in
  fastmat.


  Author      : sempersn
  Introduced  : 2016-04-08
 ------------------------------------------------------------------------------

   Copyright 2016 Sebastian Semper, Christoph Wagner
       https://www.tu-ilmenau.de/ems/

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

 ------------------------------------------------------------------------------

  TODO:
    - optimize einsum-stuff
'''
import numpy as np
import numpy.linalg as npl

from ..base import Algorithm

################################################################################
###  OMP: Sparse recovery algorithm
################################################################################


def OMP(fmatA, arrY, numK):
    '''
    Wrapper around the ISTA algrithm to allow processing of arrays of signals
        fmatA           - input system matrix
        arrY            - input data vector (measurements)
        numK            - specified sparsity order, i.e. number of iterations
                          to run
        numN,numM       - number of rows / columns of the system matrix
        numL            - number of problems to solve
        arrDiag         - array that contains column norms of fmatA
        numStrideSize   - size of strides during norm calculation
        numStrides      - number of whole strides to go through
        numPreSteps     - number of entries that do not fit in whoel strides
    '''
    if len(arrY.shape) > 2:
        raise ValueError("Only n x m arrays are supported for OMP")

    # get the number of vectors to operate on
    numN, numM, numL = fmatA.numN, fmatA.numM, arrY.shape[1]

    fmatC = fmatA.normalized

    # determine return value data type
    returnType = np.promote_types(
        np.promote_types(fmatC.dtype, arrY.dtype), np.float32)

    # temporary array to store only support entries in
    arrXtmp     = np.zeros((numK, numL), dtype=returnType)

    # initital residual is the measurement
    arrResidual = arrY.astype(returnType, copy=True)

    # list containing the support
    arrSupport  = np.empty((numK, numL),       dtype='int')

    # matrix B that contains the pseudo inverse of A restricted to the support
    arrB        = np.zeros((numK, numN, numL), dtype=returnType)

    # A restricted to the support
    arrA        = np.zeros((numN, numK, numL), dtype=returnType)

    # different helper variables
    v2       = np.empty((numN, numL), dtype=returnType)
    v2n      = np.empty((numN, numL), dtype=returnType)
    v2y      = np.empty((numL),       dtype=returnType)
    newCols  = np.empty((numN, numL), dtype=returnType)
    arrC     = np.empty((numM, numL), dtype=returnType)
    newIndex = np.empty(numL,         dtype='int')

    # iterativly build up the solution
    for ii in range(numK):

        # do the normalized correlation step
        arrC = np.abs(fmatC.backward(arrResidual))

        # pick the maximum index in each correlation array
        newIndex = np.apply_along_axis(np.argmax, 0, arrC)

        # add these to the support
        arrSupport[ii, :] = newIndex

        # get the newly picked columns of A
        newCols = fmatA.getCols(newIndex)

        # store them into the submatrix
        arrA[:, ii, :] = newCols

        # in the first step everything is simple
        if ii == 0:
            v2  = newCols
            v2n = (v2 / npl.norm(v2, axis=0) ** 2).conj()

            v2y = np.einsum('ji,ji->i', v2n, arrY)

            arrXtmp[0, :] = v2y
            arrB[0, :, :] = v2n
        else:
            v1 = np.einsum('ijk,jk->ik', arrB[:ii, :, :], newCols)

            v2 = newCols - np.einsum('ijk,jk->ik', arrA[: , :ii, :], v1)
            v2n = (v2 / npl.norm(v2, axis=0) ** 2).conj()

            v2y = np.einsum('ji,ji->i', v2n, arrY)

            arrXtmp[:ii, :] -= v2y * v1
            arrXtmp[ii , :] += v2y

            arrB[:ii, :, :] -= np.einsum('ik,jk->jik', v2n, v1)
            arrB[ii, :, :] = v2n

        # update the residual
        arrResidual -= v2y * v2

    # return the computed vector
    arrX = np.zeros((numM, numL), dtype=returnType)
    arrX[arrSupport, np.arange(numL)] = arrXtmp

    return arrX


################################################################################
###  Maintainance and Documentation
################################################################################

################################################## inspection inerface
class OMPinspect(Algorithm):

    def _getTest(self):
        from ..inspect import TEST, dynFormat, arrSparseTestDist
        from ..Hadamard import Hadamard
        from ..Matrix import Matrix

        def testOmp(test):
            # prepare vectors
            numM = test[TEST.NUM_M]
            test[TEST.RESULT_REF]       = np.hstack(
                [arrSparseTestDist((numM, 1), dtype=test[TEST.DATATYPE],
                                   density=1. * test['numK'] / numM).todense()
                 for nn in range(test[TEST.DATACOLS])])
            test[TEST.RESULT_INPUT]     = (test[TEST.INSTANCE] *
                                           test[TEST.RESULT_REF])
            test[TEST.RESULT_OUTPUT]    = OMP(test[TEST.INSTANCE],
                                              test[TEST.RESULT_INPUT],
                                              test['numK'])

        return {
            TEST.ALGORITHM: {
                'order'         : 5,
                TEST.NUM_N      : (lambda param: 2 ** param['order']),
                TEST.NUM_M      : TEST.NUM_N,
                'numK'          : 5,
                'typeA'         : TEST.Permutation(TEST.ALLTYPES),

                TEST.OBJECT     : Matrix,
                TEST.DATAALIGN  : TEST.ALIGNMENT.DONTCARE,
                TEST.INITARGS   : (lambda param: [
                    Hadamard(param.order).array.astype(param['typeA'])]),

                TEST.INIT_VARIANT : TEST.IgnoreFunc(testOmp),

                'strTypeA'      : (lambda param: TEST.TYPENAME[param['typeA']]),
                TEST.NAMINGARGS : dynFormat("Hadamard(%s,%s)",
                                            'order', 'strTypeA'),

                # matrix inversion always expands data type to floating-point
                TEST.TYPE_PROMOTION : np.float32
            },
        }

    def _getBenchmark(self):
        from ..inspect import BENCH, arrTestDist
        from ..Matrix import Matrix
        from ..Fourier import Fourier
        from ..Product import Product
        from scipy import sparse as sps

        def createTarget(M, datatype):
            '''Create test target for algorithm performance evaluation.'''

            if M < 10:
                raise ValueError("Problem size too small for OMP benchmark")

            # assume a 1:5 ratio of measurements and problem size
            # assume a sparsity of half the number of measurements
            N = int(np.round(M / 5.0))
            K = int(N / 2)

            # generate matA = [random measurement matrix] * [Fourier dictionary]
            matA = Product(Matrix(arrTestDist((N, M), datatype)), Fourier(M))

            # generate attb from random baseline support (RHS)
            arrB = matA.forward(
                sps.rand(M, 1, 1.0 * K / M).todense().astype(datatype))

            return (OMP, [matA, arrB, K])

        return {
            BENCH.COMMON: {
                BENCH.NAME      : 'OMP Algorithm',
                BENCH.DOCU      : r"""We use $\bm A = \bm M \cdot \bm \Fs$,
                    where $\bm M$ was drawn from a standard Gaussian
                    distribution and $\bm \Fs$ is a Fourier matrix. The vector
                    $\bm b \in \C^m$ of equation \eqref{omp_problem} is
                    generated from multiplying $\bm A$ with a sparse vector
                    $\bm x$.""",
                BENCH.FUNC_GEN  : (lambda c: createTarget(10 * c, np.float64)),
                BENCH.FUNC_SIZE : (lambda c: 10 * c)
            },
            BENCH.PERFORMANCE: {
                BENCH.CAPTION   : 'OMP performance'
            },
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, dt: createTarget(10 * c, dt)),
                BENCH.FUNC_SIZE : (lambda c: 10 * c),
                BENCH.FUNC_STEP : (lambda c: c * 10 ** (1. / 12)),
            }
        }

    def _getDocumentation(self):
        from ..inspect import DOC
        return DOC.SUBSECTION(
            r'Orthogonal matching Pursuit (OMP) (\texttt{fastmat.algs.OMP})',
            DOC.SUBSUBSECTION(
                'Definition and Interface', r"""
For a given matrix $\bm A \in \C^{m \times N}$ with $m \ll N$ and a
vector $\bm b \in \C^m$ we approximately solve

\begin{align}\label{omp_problem}
    \Min\limits_{\bm x \in \C^N}\Norm{\bm x}_0 \Text{s.t.} \bm A \cdot
    \bm x = \bm x.
\end{align}

If it holds that $\bm b = \bm A \cdot \bm x_0$ for some $k$-sparse $\bm x_0$ and
$k$ is low enough, we can recover $\bm x_0$ via OMP \cite{omp_mallat1952omp}.

\begin{itemize}
\item \textbf{Input:} Full rank matrix $\bm A$, measurements $\bm b$, desired
    sparsity order $k$ of the reconstruction
\item \textbf{Output:} Reconstruction vector $\bm x$.
\end{itemize}

This type of problem as the one described above occurs in Compressed Sensing
and Sparse Signal Recovery, where signals are approximated by sparse
representations.""",
                DOC.SNIPPET('# import the packages',
                            'import numpy.linalg as npl',
                            'import numpy as np',
                            'import fastmat as fm',
                            'import fastmat.algs as fma',
                            '',
                            '# define the dimensions',
                            '# and the sparsity',
                            'n, k = 512, 3',
                            '',
                            '# define the sampling positions',
                            't = np.linspace(0, 20 * m.pi, n)',
                            '',
                            '# construct the convolution matrix',
                            'c = np.cos(2 * t) * np.exp(',
                            '    -(t - 10 * m.pi) ** 2 / .1)',
                            'C = fm.Circulant(c)',
                            '',
                            '# create the ground truth',
                            'x = np.zeros(n)',
                            'x[npr.choice(range(n),',
                            '             k, replace=0)] = 1',
                            'b = C * x',
                            '',
                            '# reconstruct it',
                            'y = fma.OMP(C, b, k)',
                            '',
                            '# test if they are close in the',
                            '# domain of C',
                            'print(npl.norm(C * y - b))',
                            caption=r"""
We describe a sparse deconvolution problem, where the signal is in $\R^{512}$
and consists of $3$ windowed osine pulses of the form $\bm c$ with circulant
displacement. Then we take the convolution and try to recover the location of
the pulses using the OMP algorithm."""),
                r"""
\textit{Hint:} The algorithm exploits two mathematical shortcuts. First it
obviously uses the fast transform of the involved system matrix during the
correlation step and second it uses a method to calculate the pseudo
inverse after a rank-$1$ update of the matrix."""
            ),
            DOC.SUBSUBSECTION(
                'Performance Benchmarks', r"""
We use $\bm A = \bm M \cdot \bm \Fs$, where $\bm M$ was drawn from a standard
Gaussian distribution and $\bm \Fs$ is a Fourier matrix. The vector $\bm b \in
\C^m$ of equation \eqref{ista_lasso} is generated from multiplying $\bm A$ with
a sparse vector $\bm x$.""",
                DOC.PLOTPERFORMANCE(),
                DOC.PLOTTYPESPEED(),
                DOC.PLOTTYPEMEMORY()
            ),
            DOC.BIBLIO(
                omp_mallat1952omp=DOC.BIBITEM(
                    r'S. G. Mallat and Zhifeng Zhang',
                    r'Matching pursuits with time-frequency dictionaries',
                    r"""
IEEE Transactions on Signal Processing, vol. 41, no. 12, pp. 3397-3415,
Dec 1993""")
            )
        )
