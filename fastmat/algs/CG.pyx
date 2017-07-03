# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False, nonecheck=False
'''
  fastmat/algs/CG.pyx
 -------------------------------------------------- part of the fastmat package

  Implementation of conjugated gradient method solver for linear equation
  systems in fastmat.


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
'''
import numpy as np
cimport numpy as np

from ..helpers.cmath cimport _arrEmpty, _arrZero, _arrForceTypeAlignment, _norm
from ..helpers.types cimport *

from ..Matrix cimport Matrix


################################################################################
###  CG: wrapper around the single vector solver
################################################################################
cpdef np.ndarray CG(
    Matrix fmatA,
    np.ndarray arrB,
    float eps=0
):
    '''Solve linear equation system 'fmatA * x = arrB' for x.'''
    if arrB.ndim > 2:
        raise ValueError("Only N x M arrays are supported for CG")

    cdef np.dtype typeOut = np.promote_types(
        np.float32, np.promote_types(fmatA.dtype, arrB.dtype))
    cdef nptype npTypeOut = typeOut.type_num
    arrIn = _arrForceTypeAlignment(arrB, npTypeOut, np.NPY_FORCECAST)
    if eps == 0:
        eps = _getTypeEps(typeOut)

    # dispatch specialization of core routine according tOptor
    if typeOut == np.float32:
        return _CGcore[np.float32_t](fmatA, arrIn, npTypeOut, 0., eps)
    elif typeOut == np.float64:
        return _CGcore[np.float64_t](fmatA, arrIn, npTypeOut, 0., eps)
    elif typeOut == np.complex64:
        return _CGcore[np.complex64_t](fmatA, arrIn, npTypeOut, 0., eps)
    elif typeOut == np.complex128:
        return \
            _CGcore[np.complex128_t](fmatA, arrIn, npTypeOut, 0., eps)
    else:
        raise NotImplementedError("Output type %d not supported." % (typeOut))


cdef np.ndarray _CGcore(
    Matrix fmatA,
    np.ndarray arrB,
    nptype npTypeOut,
    TYPE_FLOAT typeTag,
    float eps
):
    '''
    Solve linear equation system 'fmatA * x = arrB' for x.
    The following variables are used:
    arrB         - input data array
    fmatA        - input system matrix
    arrR         - residual vector (TYPE_FLOAT, CONT)
    vecR         - residual vector (TYPE_FLOAT[:] in arrR)
    arrP         - next krylov subspace vector (TYPE_FLOAT[:N], CONT)
    vecP         - next krylov subspace vector (TYPE_FLOAT[:] in arrP)
    vecQ         - projection of krylov subspace vector onto cols of matA
    arrIn        - symmetrized right-hand side [TYPE_FLOAT, F, CONT]
    arrOut       - current solution [TYPE_FLOAT, F, CONT]
    vecOut       - vector of current solution (TYPE_FLOAT[:] in arrOut)
    numAlpha     - optimal step with
    numRNormNew  - new residual norm
    numRNormOld  - old residual norm
    eps          - stopping condition to the projected residual
    '''
    # NOTE: typeTag is used for telling the compiler the used specialization
    # fetch dimensions of arrB
    cdef intsize mm, nn, numStep
    cdef intsize N = arrB.shape[0]
    cdef intsize M = arrB.shape[1] if arrB.ndim > 1 else 1

    # allocate temporary variables used in vector loop
    cdef np.float64_t numRNormOld, numRNormNew, ratio
    cdef TYPE_FLOAT numValue, numCorr, numAlpha

    # allocate temporary buffers in memory and create memoryviews into them
    cdef np.ndarray arrOut, arrP = _arrEmpty(1, N, 1, npTypeOut)
    cdef TYPE_FLOAT * vecP = <TYPE_FLOAT * > arrP.data
    cdef np.ndarray arrQ
    cdef TYPE_FLOAT * vecQ

    # change right hand side of equation system according to symmetrization
    # force to be F-contiguous and of consistent data type (no ints here)
    cdef np.ndarray arrR = _arrForceTypeAlignment(
        fmatA.backward(arrB), npTypeOut, np.NPY_FORCECAST)
    cdef TYPE_FLOAT * pArrR = <TYPE_FLOAT * > arrR.data
    cdef TYPE_FLOAT * vecR
    arrOut = _arrZero(2, fmatA.numM, M, npTypeOut)
    cdef TYPE_FLOAT * pArrOut = <TYPE_FLOAT * > arrOut.data
    cdef TYPE_FLOAT * vecOut

    # iterate over vectors
    for mm in range(M):

        vecOut = &(pArrOut[mm * N])
        vecR   = &(pArrR[mm * N])

        # norm of residual
        numRNormOld = _norm(vecR, N)

        # projected residual (initialized with RHS-reformulation),
        # use to bootstrap krylov subspace vector by copy
        for nn in range(N):
            vecP[nn] = vecR[nn]

        # step counter
        numStep = 0

        # iterate until stopping criterion is met
        while numRNormOld > eps:
            arrQ = fmatA.gram.forward(arrP)
            vecQ = <TYPE_FLOAT * > arrQ.data

            # calculate next optimal step width according to
            # current residual and correlation between old and
            # new projection: correlate np.conj(arrP) (*) arrQ
            numCorr = 0
            for nn in range(N):
                numValue = vecP[nn]
                # conjugate elements of P on-the-fly
                if (TYPE_FLOAT == np.complex64_t) or \
                        (TYPE_FLOAT == np.complex128_t):
                    numValue.imag = -numValue.imag

                numCorr += numValue * vecQ[nn]

            numAlpha = numRNormOld / numCorr

            # update current solution and residual
            # here we use the orthogonality of arrP and arrQ
            # with respect to the inner product induced by A
            # while iterating, calculate norm of vecR on-the-fly
            for nn in range(N):
                vecOut[nn] = vecOut[nn] + numAlpha * vecP[nn]
                vecR[nn] = vecR[nn] - numAlpha * vecQ[nn]

            # new residual norm
            numRNormNew = _norm(vecR, N)

            # next krylov subspace vector
            value = numRNormNew / numRNormOld
            for nn in range(N):
                vecP[nn] = vecP[nn] * value + vecR[nn]

            numRNormOld = numRNormNew

            # step increment
            numStep += 1

    return arrOut


################################################################################
################################################################################
from ..helpers.unitInterface import *
from ..Diag import Diag

################################################### Testing


def testCG(test):

    # prepare vectors
    test[TEST_RESULT_REF] = arrTestDist(
        (test[TEST_NUM_M], test[TEST_DATACOLS]), dtype=test[TEST_DATATYPE])
    test[TEST_RESULT_INPUT] = test[TEST_INSTANCE] * test[TEST_RESULT_REF]
    test[TEST_RESULT_OUTPUT] = CG(test[TEST_INSTANCE], test[TEST_RESULT_INPUT])


test = {
    TEST_ALGORITHM: {
        TEST_NUM_N      : 27,
        TEST_NUM_M      : TEST_NUM_N,

        'typeA'         : Permutation(typesAll),
        'arrA'          : ArrayGenerator({
            NAME_DTYPE      : 'typeA',
            NAME_SHAPE      : (TEST_NUM_N, TEST_NUM_N)
        }),

        TEST_OBJECT     : Matrix,
        TEST_INITARGS   : (lambda param: [param.arrA()]),

        TEST_DATAALIGN      : ALIGN_DONTCARE,
        TEST_INIT_VARIANT   : IgnoreFunc(testCG),

        'strTypeA'      : (lambda param: NAME_TYPES[param['typeA']]),
        TEST_NAMINGARGS : dynFormatString("%s", 'arrA'),

        # matrix inversion always expands data type to floating-point
        TEST_TYPE_PROMOTION     : np.float32,
        #        TEST_CHECK_PROXIMITY    : False,
        TEST_TOL_POWER          : 6.
    },
}


################################################## Benchmarks
def createBenchmarkTarget(M, datatype):
    '''Create test target for algorithm performance evaluation.'''

    # generate matA (random diagonal matrix)
    matA = Diag(arrTestDist((M, 1), datatype))

    # generate arrb from random baseline support and matrix (RHS)
    arrB = matA.forward(arrTestDist((M, 1), datatype))

    return (CG, [matA, arrB])


benchmark = {
    NAME_COMMON: {
        NAME_NAME       : 'Method of Conjugate Gradients',
        NAME_DOCU       : r'$\bm A = \diag(\{1,\dots,n\})$',
        BENCH_FUNC_GEN  : (lambda c: createBenchmarkTarget(c, np.double))
    },
    BENCH_PERFORMANCE: {
        NAME_CAPTION    : 'CG performance'
    },
    BENCH_DTYPES: {
        BENCH_FUNC_GEN  : createBenchmarkTarget,
        BENCH_FUNC_SIZE : (lambda c: c),
        BENCH_FUNC_STEP : (lambda c: c * 10 ** (1. / 12)),
    }
}


################################################## Documentation
docLaTeX = r"""
\subsection{Conjugate Gradient Method (CG) (\texttt{fastmat.algs.CG})}
\subsubsection{Definition and Interface}
%
For a given full rank hermitian matrix $\bm A \in \C^n$ and a vector $\bm b \in
\C^n$, we solve
%
\begin{align}
\bm A \cdot \bm x = \bm b
\end{align}
%
for $\bm x \in \C^n$, i.e. $\bm x = \bm A^{-1} \cdot \bm b$. If $\bm A$ is not
hermitian, we solve
%
\begin{align}
\bm A^\herm \cdot \bm A \cdot \bm x = \bm A^\herm \cdot \bm b
\end{align}
instead. In this case it should be noted, that the condition number of $\bm
A^\herm \cdot \bm A$ might be a lot larger than the one of $\bm A$ an thus we
might run into stability problems for large and already ill-conditioned systems.
%
\begin{itemize}
\item \textbf{Input:} System matrix $\bm A$, right hand side $\bm b$ and
stopping tolerance for the residual $0 < \varepsilon \ll 1$.
\item \textbf{Output:} Solution $\bm x = \bm A^{-1} \cdot \bm b$
\end{itemize}

This algorithm was originally described in \cite{cg_stiefel1952cg} and is
applicable here, because it only uses the backward and forward projection of a
matrix.

\begin{snippet}
\begin{lstlisting}[language=Python]
# import the packages
import numpy.random as npr
import numpy as np
import fastmat as fm
import fastmat.algs as fma

# construct the matrix
n = 26
H = fm.Hadamard(n)

# define the right hand side
b = npr.randn(2**n)

# solve the system
y = fma.CG(H,b)

# check if solution is correct
print(
    np.allclose(b,H.forward(y))
    )
\end{lstlisting}

We construct a Hadamard matrix of order $26$, which would consume
\SI{4.5}{\peta\byte} of memory if we used \SI{1}{byte} integers to represent it
and solve above system of linear equations.
\end{snippet}

\begin{thebibliography}{9}
\bibitem{cg_stiefel1952cg}
Hestenes, Magnus R., Stiefel, Eduard
\emph{Methods of Conjugate Gradients for Solving Linear Systems},
Journal of Research of the National Bureau of Standards,
Volume 49,
1952.
\end{thebibliography}
"""
