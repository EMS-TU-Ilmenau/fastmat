# -*- coding: utf-8 -*-
'''
  fastmat/algs/ISTA.py
 -------------------------------------------------- part of the fastmat package

  Implementation of iterative Shrinking-Thresholding Algorithm (ISTA) in
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
    - test ISTA for correctness
    - implement unit test code
    - specify when this transform was introduced
    - reformulate benchmark baseline definition to ensure compareability
'''
import numpy as np
import scipy as sp

from ..Product import Product
from ..helpers.unitInterface import *
from ..helpers.types import _getTypeEps

from ..Matrix import Matrix
from ..Fourier import Fourier
from ..Hadamard import Hadamard

################################################################################
###  ISTA: Sparse recovery algorithm
################################################################################


def _softThreshold(arrX, numAlpha):
    '''
    Do a soft Thresholding step.
      arrM         - positive part of arrX - numAlpha
      arrX         - vector to be thresholded
      numAlpha     - thresholding threshold
    '''
    arrM = np.maximum(np.abs(arrX) - numAlpha, 0)
    return np.multiply((arrM / (arrM + numAlpha)), arrX)


##################################################

def ISTA(
    fmatA,
    arrB,
    numLambda=0.1,
    numMaxSteps=100
):
    '''
    Wrapper around the ISTA algrithm to allow processing of arrays of signals
        fmatA         - input system matrix
        arrB          - input data vector (measurements)
        numLambda     - balancing parameter in optimization problem
                        between data fidelity and sparsity
        numMaxSteps   - maximum number of steps to run
        numL          - step size during the conjugate gradient step
    '''

    if len(arrB.shape) > 2:
        raise ValueError("Only n x m arrays are supported for ISTA")

    # calculate the largest singular value to get the right step size
    numL = 1.0 / (fmatA.largestSV ** 2)

    arrX = np.zeros(
        (fmatA.numM, arrB.shape[1]),
        dtype=np.promote_types(np.float32, arrB.dtype)
    )

    # start iterating
    for numStep in range(numMaxSteps):
        # do the gradient step and threshold

        arrStep = arrX - 2.0 * numL * fmatA.backward(fmatA.forward(arrX) - arrB)
        arrX = _softThreshold(arrStep, numL * numLambda)

    # return the unthresholded values for all non-zero support elements
    return np.where(arrX != 0, arrStep, arrX)


################################################################################
###  Maintainance and Documentation
################################################################################

################################################### Testing
def testISTA(test):

    # prepare vectors
    numM = test[TEST_NUM_M]
    test[TEST_RESULT_REF] = np.hstack(
        [arrSparseTestDist((numM, 1), dtype=test[TEST_DATATYPE],
                           density=1. * test['numK'] / numM).todense()
         for nn in range(test[TEST_DATACOLS])])
    test[TEST_RESULT_INPUT] = test[TEST_INSTANCE] * test[TEST_RESULT_REF]
    test[TEST_RESULT_OUTPUT] = ISTA(
        test[TEST_INSTANCE], test[TEST_RESULT_INPUT],
        numLambda=test['lambda'], numMaxSteps=test['maxSteps'])


test = {
    TEST_ALGORITHM: {
        'order'         : 5,
        TEST_NUM_N      : (lambda param: 3 * param['order']),
        TEST_NUM_M      : (lambda param: 2 ** param['order']),
        'numK'          : 5,
        'lambda'        : 7.,
        'maxSteps'      : 1000,
        'typeA'         : Permutation(typesAll),

        TEST_OBJECT     : Matrix,
        TEST_INITARGS   : (lambda param: [
            Product(Matrix(arrTestDist((getattr(param, TEST_NUM_M),
                                        getattr(param, TEST_NUM_M)),
                                       param['typeA'])),
                    Hadamard(param.order),
                    typeExpansion=param['typeA']).toarray()]),

        TEST_DATAALIGN      : ALIGN_DONTCARE,
        TEST_INIT_VARIANT   : IgnoreFunc(testISTA),

        'strTypeA'      : (lambda param: NAME_TYPES[param['typeA']]),
        TEST_NAMINGARGS : dynFormatString("(%dx%d)*Hadamard(%s)[%s]",
                                          TEST_NUM_N, TEST_NUM_M,
                                          'order', 'strTypeA'),

        # matrix inversion always expands data type to floating-point
        TEST_TYPE_PROMOTION     : np.float32,
        TEST_TOL_MINEPS         : _getTypeEps(np.float32),
        TEST_TOL_POWER          : 5.
        #TEST_CHECK_PROXIMITY    : False
    },
}

################################################## Benchmarks


def createBenchmarkTarget(M, datatype):
    if M < 10:
        raise ValueError("Problem size too small for ISTA benchmark")

    # assume a 1:5 ratio of measurements and problem size
    # assume a sparsity of half the number of measurements
    N = int(np.round(M / 5.0))
    K = int(N / 2)

    # generate matA (random measurement matrix, Fourier dictionary)
    matA = Product(Matrix(arrTestDist((N, M), datatype)), Fourier(M))

    # generate arrB from random baseline support (RHS)
    arrB = matA * sp.sparse.rand(M, 1, 1.0 * K / M).todense().astype(datatype)

    return (ISTA, [matA, arrB])


benchmark = {
    NAME_COMMON: {
        NAME_NAME       : 'ISTA Algorithm',
        NAME_DOCU       : r'''
            We use $\bm A = \bm M \cdot \bm \Fs$, where $\bm M$ was drawn from a
            standard Gaussian distribution and $\bm \Fs$ is a Fourier matrix.
            The vector $\bm b \in \C^m$ of equation \eqref{ista_lasso} is
            generated from multiplying $\bm A$ with a sparse vector $\bm x$.
            ''',
        BENCH_FUNC_GEN  : (lambda c: createBenchmarkTarget(10 * c, np.float64)),
        BENCH_FUNC_SIZE : (lambda c: 10 * c)
    },
    BENCH_PERFORMANCE: {
        NAME_CAPTION    : 'ISTA performance'
    },
    BENCH_DTYPES: {
        BENCH_FUNC_GEN  :
            (lambda c, datatype: createBenchmarkTarget(10 * c, datatype)),
        BENCH_FUNC_SIZE : (lambda c: 10 * c),
        BENCH_FUNC_STEP : (lambda c: c * 10 ** (1. / 12)),
    }
}


################################################## Documentation
docLaTeX = r"""
\subsection{Iterative Soft Thresholding Algorithm (ISTA)
 (\texttt{fastmat.algs.ISTA})}
\subsubsection{Definition and Interface}
%
For a given matrix $\bm A \in \C^{m \times N}$ with $m \ll N$ and a
vector $\bm b \in \C^m$ we approximately solve
%
\begin{align}\label{ista_lasso}
    \Min\limits_{\bm x \in \C^N}\Norm{\bm A \cdot \bm x - \bm b}^2_2 +
    \lambda\cdot\Norm{\bm x}_1,
\end{align}
%
where $\lambda > 0$ is a regularization parameter to steer the trade-off between
data fidelity and sparsity of the solution.
%
\begin{itemize}
\item \textbf{Input:} Full rank matrix $\bm A$, measurements $\bm b$, trade-off
    parameter $\lambda$
\item \textbf{Output:} Reconstruction vector $\bm x$.
\end{itemize}

The algorithm is presented in \cite{ista_beck1952ista} together with an
performance upgrade, which we hope to implement soon.

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
y = fma.ISTA(C,b,0.005,1000)

# test if they are close in the
# domain of C
print(npl.norm(C * y - b))
\end{lstlisting}

We solve a sparse deconvolution problem, where the atoms are harmonics windowed
by a gaussian envelope. The ground truth $\bm x$ is build out of three pulses at
arbitrary locations.
\end{snippet}

\textit{Hint:} The proper choice of $\lambda$ is crucial for good perfomance of
    this algorithm, but this is not an easy task. Unfortunately we are not
    in the place here to give you a rule of thumb what to do, since it
    highly depends on the application at hand. Again, consult
    \cite{ista_beck1952ista} for any further considerations of this matter.

\begin{thebibliography}{9}
\bibitem{ista_beck1952ista}
Amir Beck and Marc Teboulle
\emph{A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse
    Problems}, SIAM Journal on Imaging Sciences 2009 2:1, 183-202
\end{thebibliography}
"""
