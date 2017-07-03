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
import numpy.random as npr
import numpy.linalg as npl
import scipy as sp

from ..helpers.unitInterface import *

from ..Matrix import Matrix
from ..Fourier import Fourier
from ..Hadamard import Hadamard
from ..Product import Product
from ..Diag import Diag
from ..Eye import Eye

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

################################################### Testing
def testOmp(test):

    # prepare vectors
    numM = test[TEST_NUM_M]
    test[TEST_RESULT_REF] = np.hstack(
        [arrSparseTestDist((numM, 1), dtype=test[TEST_DATATYPE],
                           density=1. * test['numK'] / numM).todense()
         for nn in range(test[TEST_DATACOLS])])
    test[TEST_RESULT_INPUT] = test[TEST_INSTANCE] * test[TEST_RESULT_REF]
    test[TEST_RESULT_OUTPUT] = OMP(
        test[TEST_INSTANCE], test[TEST_RESULT_INPUT], test['numK'])


test = {
    TEST_ALGORITHM: {
        'order'         : 5,
        TEST_NUM_N      : (lambda param: 2 ** param['order']),
        TEST_NUM_M      : TEST_NUM_N,
        'numK'          : 5,
        'typeA'         : Permutation(typesAll),

        TEST_OBJECT     : Matrix,
        TEST_DATAALIGN  : ALIGN_DONTCARE,
        TEST_INITARGS   : (lambda param: [
            Hadamard(param.order).toarray().astype(param['typeA'])]),

        TEST_INIT_VARIANT   : IgnoreFunc(testOmp),

        'strTypeA'      : (lambda param: NAME_TYPES[param['typeA']]),
        TEST_NAMINGARGS : dynFormatString("Hadamard(%s,%s)",
                                          'order', 'strTypeA'),

        # matrix inversion always expands data type to floating-point
        TEST_TYPE_PROMOTION     : np.float32
    },
}

################################################## Benchmarks


def createTarget(M, datatype):
    if M < 10:
        raise ValueError("Problem size too small for OMP benchmark")

    # assume a 1:5 ratio of measurements and problem size
    # assume a sparsity of half the number of measurements
    N = int(np.round(M / 5.0))
    K = int(N / 2)

    # generate matA from random measurement matrix and Fourier dictionary
    matA = Product(Matrix(arrTestDist((N, M), datatype)), Fourier(M))

    # generate attb from random baseline support (RHS)
    arrB = matA.forward(
        sp.sparse.rand(M, 1, 1.0 * K / M).todense().astype(datatype))

    return (OMP, [matA, arrB, K])


benchmark = {
    NAME_COMMON: {
        NAME_NAME       : 'OMP Algorithm',
        NAME_DOCU       : r'''
            We use $\bm A = \bm M \cdot \bm \Fs$, where $\bm M$ was drawn from a
            standard Gaussian distribution and $\bm \Fs$ is a Fourier matrix.
            The vector $\bm b \in \C^m$ of equation \eqref{omp_problem} is
            generated from multiplying $\bm A$ with a sparse vector $\bm x$.
            ''',
        BENCH_FUNC_GEN  : (lambda c: createTarget(10 * c, np.float64)),
        BENCH_FUNC_SIZE : (lambda c: 10 * c)
    },
    BENCH_PERFORMANCE: {
        NAME_CAPTION    : 'OMP performance'
    },
    BENCH_DTYPES: {
        BENCH_FUNC_GEN  : (lambda c, datatype: createTarget(10 * c, datatype)),
        BENCH_FUNC_SIZE : (lambda c: 10 * c),
        BENCH_FUNC_STEP : (lambda c: c * 10 ** (1. / 12)),
    }
}


################################################## Documentation
docLaTeX = r"""
\subsection{Orthogonal matching Pursuit (OMP) (\texttt{fastmat.algs.OMP})}
\subsubsection{Definition and Interface}
%
For a given matrix $\bm A \in \C^{m \times N}$ with $m \ll N$ and a
vector $\bm b \in \C^m$ we approximately solve
%
\begin{align}\label{omp_problem}
    \Min\limits_{\bm x \in \C^N}\Norm{\bm x}_0 \Text{s.t.} \bm A \cdot
    \bm x = \bm x.
\end{align}
%
If it holds that $\bm b = \bm A \cdot \bm x_0$ for some $k$-sparse $\bm x_0$ and
$k$ is low enough, we can recover $\bm x_0$ via OMP \cite{omp_mallat1952omp}.
%
\begin{itemize}
\item \textbf{Input:} Full rank matrix $\bm A$, measurements $\bm b$, desired
    sparsity order $k$ of the reconstruction
\item \textbf{Output:} Reconstruction vector $\bm x$.
\end{itemize}
%
This type of problem as the one described above occurs in Compressed Sensing
and Sparse Signal Recovery, where signals are approximated by sparse
representations.

\begin{snippet}
\begin{lstlisting}[language=Python]
# import the packages
import numpy.linalg as npl
import numpy as np
import fastmat as fm
import fastmat.algs as fma

# define the dimensions
# and the sparsity
n,k = 512,3

# define the sampling positions
t = np.linspace(0,20*m.pi,n)

# construct the convolution matrix
c = np.cos(2*t)
    *np.exp(-(t-10*m.pi)**2/0.1)
C = fm.Circulant(c)

# create the ground truth
x = np.zeros(n)
x[npr.choice(
    range(n),k,replace=0
    )] = 1
b = C * x

# reconstruct it
y = fma.OMP(C,b,k)

# test if they are close in the
# domain of C
print(npl.norm(C * y - b))
\end{lstlisting}

We describe a sparse deconvolution problem, where the signal is in $\R^{512}$
and consists of $3$ windowed osine pulses of the form $\bm c$ with circulant
displacement. Then we take the convolution and try to recover the location of
the pulses using the OMP algorithm.
\end{snippet}

\textit{Hint:} The algorithm exploits two mathematical shortcuts. First it
obviously uses the fast transform of the involved system matrix during the
correlation step and second it uses a method to calculate the pseudo
inverse after a rank-$1$ update of the matrix.

\begin{thebibliography}{9}
\bibitem{omp_mallat1952omp}
S. G. Mallat and Zhifeng Zhang
\emph{Matching pursuits with time-frequency dictionaries},
IEEE Transactions on Signal Processing, vol. 41, no. 12, pp. 3397-3415, Dec 1993
\end{thebibliography}
"""
