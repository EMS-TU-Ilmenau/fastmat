# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False
'''
  fastmat/MLCirculant.py
 -------------------------------------------------- part of the fastmat package

  MLCirculant matrix.


  Author      : sempersn
  Introduced  : 2017-09-19
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

from .core.types cimport *
from .core.cmath cimport *
from .Matrix cimport Matrix
from .Partial cimport Partial
from .Product cimport Product
from .Fourier cimport Fourier
from .Diag cimport Diag
from .Kron cimport Kron
from .Circulant cimport Circulant


################################################################################
################################################## class Circulant
cdef class MLCirculant(Partial):

    ############################################## class properties
    # vecC - Property (read-only)
    # Return the matrix-defining column vector of the circulant matrix
    property tenC:
        def __get__(self):
            return self._tenC

    ############################################## class methods
    def __init__(self, tenC, **options):
        '''
        Initialize MLCirculant Matrix instance.
        '''

        self._tenC = np.atleast_1d(np.squeeze(np.copy(tenC)))

        assert self._tenC.ndim >= 1, "Column-definition vector must be >= 1D."

        # extract the level dimensions from the defining tensor
        self._arrN = np.array(np.array(self._tenC).shape)

        # get the size of the matrix
        numN = np.prod(self._arrN)

        # stages during optimization
        cdef int maxStage = options.get('maxStage', 4)

        # minimum number to pad to during optimization and helper arrays
        arrNpad = 2 * self._arrN - 1
        arrOptSize = np.zeros_like(self._arrN)
        arrDoOpt = np.zeros_like(self._arrN)

        cdef bint optimize = options.get('optimize', True)

        if optimize:
            # go through all level dimensions and get optimal FFT size
            for inn, nn in enumerate(arrNpad):
                arrOptSize[inn] = _findOptimalFFTSize(nn, maxStage)

                # use this size, if we get better in that level
                if (_getFFTComplexity(arrOptSize[inn]) <
                        _getFFTComplexity(self._arrN[inn])):
                    arrDoOpt[inn] = 1

        arrDoOpt = arrDoOpt == 1
        arrNopt = np.copy(self._arrN)

        # set the optimization size to the calculated one
        arrNopt[arrDoOpt] = arrOptSize[arrDoOpt]

        # get the size of the zero padded matrix
        numNopt = np.prod(arrNopt)

        # allocate memory for the tensor in MD fourier domain
        tenChat = np.empty_like(self._tenC, dtype='complex')
        tenChat[:] = self._tenC[:]

        # go through the array and apply the preprocessing in direction
        # of each axis. this cannot be done without the for loop, since
        # manipulations always influence the data for the next dimension
        for ii in range(len(self._arrN)):
            tenChat = np.apply_along_axis(
                self._preProcSlice,
                ii,
                tenChat,
                ii,
                arrNopt,
                self._arrN
            )

        # after correct zeropadding, go into fourier domain
        tenChat = np.fft.fftn(tenChat).reshape(numNopt) / numNopt

        # subselection array
        arrIndices = np.arange(numNopt)[self._genArrS(self._arrN, arrNopt)]

        # create the decomposing kronecker product
        cdef Kron KN = Kron(*list(map(
            lambda ii : Fourier(ii, optimize=False), arrNopt
        )))

        # now decompose the ML matrix as a product
        cdef Product P = Product(KN.H, Diag(tenChat),
                                 KN, **options)

        # initialize Partial of Product. Only use Partial when padding size
        if np.allclose(self._arrN, arrNopt):
            super(MLCirculant, self).__init__(P)
        else:
            super(MLCirculant, self).__init__(P, N=arrIndices, M=arrIndices)

        # Currently Fourier matrices bloat everything up to complex double
        # precision, therefore make sure tenC matches the precision of the
        # matrix itself
        if self.dtype != self._tenC.dtype:
            self._tenC = self._tenC.astype(self.dtype)

    cpdef Matrix _getNormalized(self):
        norm = np.linalg.norm(self._tenC.reshape((-1)))
        return self * Diag(np.ones(self.numN) / norm)

    cpdef np.ndarray _getArray(self):
        '''Return an explicit representation of the matrix as numpy-array.'''
        return self._reference()

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        return (0., 0.)

    def _preProcSlice(
        self,
        theSlice,
        numSliceInd,
        arrNopt,
        arrN
    ):
        '''
        preprocess one axis of the defining tensor. here we check for one
        dimension, whether it makes sense to  zero-pad or not by estimating
        the fft-complexity in each dimension.
        '''
        arrRes = np.empty(1)

        if arrNopt[numSliceInd] > arrN[numSliceInd]:
            z = np.zeros(arrNopt[numSliceInd] - 2 * arrN[numSliceInd] + 1)
            return np.concatenate((theSlice, z, theSlice[1:]))
        else:
            return np.copy(theSlice)

    def _genArrS(
        self,
        arrN,
        arrNout,
        verbose=False
    ):
        '''
        Iteratively filter out the non-zero elements in the padded version
        of X. I know, that one can achieve this from a zero-padded
        version of the tensor Xpadten, but the procedure itself is very
        helpful for understanding how the nested levels have an impact on
        the padding structure
        '''
        n = arrN.shape[0]
        numNout = np.prod(arrNout)
        arrS = np.arange(numNout) >= 0
        for ii in range(n):
            if verbose:
                print("state", arrS)
                print("modulus", np.mod(
                    np.arange(numNout),
                    np.prod(arrNout[:(n -ii)])
                ))
                print("inequ", arrN[n -1 -ii] * np.prod(arrNout[:(n -1 -ii)]))
                print("res", np.mod(
                    np.arange(numNout),
                    np.prod(arrNout[:(n -1 -ii)])
                ) < arrN[n -1 -ii] * np.prod(arrNout[:(n -1 -ii)]))
            # iteratively subselect more and more indices in arrS
            np.logical_and(
                arrS,
                np.mod(
                    np.arange(numNout),
                    np.prod(arrNout[ii:])
                ) < arrN[ii] * np.prod(arrNout[ii +1:]),
                arrS
            )
        return arrS

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''

        return self._refRecursion(self._arrN, self._tenC, False)

    def _refRecursion(
        self,
        arrN,               # dimensions in each level
        tenC,               # defining elements
        verbose=False       # verbosity flag
    ):
        '''
            Construct a multilevel circulant matrix
        '''
        # number of dimensions
        numD = arrN.shape[0]

        if verbose:
            print(numD, arrN)

        # get size of resulting block circulant matrix
        numN = np.prod(arrN)
        arrNprod = np.array(
            list(map(lambda ii : np.prod(arrN[ii:]), range(len(arrN) + 1)))
        )
        if verbose:
            print()
            print(tenC)
            print(arrNprod)

        C = np.zeros((numN, numN), dtype=self.dtype)

        if numD > 1:
            # iterate over dimensions
            for nn in range(arrN[0]):

                # calculate the submatrices
                if verbose:
                    print("Going one level deeper: %d" % (nn))

                subC = self._refRecursion(arrN[1 :], tenC[nn])

                # place them at the correct positions
                for ii in range(arrN[0]):
                    NN = (arrNprod[1] * ((nn + ii) % (arrN[0]))) % arrNprod[0]
                    MM = (arrNprod[1] * ii)
                    if verbose:
                        print("nn=%d, ii=%d, NN=%d, MM=%d, \
                                NNto=%d, MMto=%d, CN=%d, CM=%d"
                              % (nn, ii, NN, MM, NN + arrNprod[1],
                                 MM + arrNprod[1], subC.shape[0], subC.shape[1])
                              )
                        print(C[NN:NN + arrNprod[1], MM:MM + arrNprod[1]].shape)
                        print(C.shape)
                        print(arrN[0])

                    # do the actual placement
                    C[NN:NN + arrNprod[1], MM:MM + arrNprod[1]] = subC
            return C
        else:
            # if we are in the lowest level, we just return the circulant
            # block
            if verbose:
                print("Deepest level reached")

            return Circulant(tenC[:numN]).array

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                # 35 is just any number that causes no padding
                # 41 is the first size for which bluestein is faster
                TEST.NUM_N      : 27,
                TEST.NUM_M      : TEST.NUM_N,
                'mTypeC'        : TEST.Permutation(TEST.ALLTYPES),
                'optimize'      : True,
                TEST.PARAMALIGN : TEST.Permutation(TEST.ALLALIGNMENTS),
                'tenC'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mTypeC',
                    TEST.SHAPE  : (3, 3, 3),
                    TEST.ALIGN  : TEST.PARAMALIGN
                }),
                TEST.INITARGS   : (lambda param : [param['tenC']()]),
                TEST.INITKWARGS : {'optimize' : 'optimize'},
                TEST.OBJECT     : MLCirculant,
                TEST.NAMINGARGS : dynFormat("%s,optimize=%s",
                                            'tenC', str('optimize')),
                TEST.TOL_POWER  : 2.,
                TEST.TOL_MINEPS : _getTypeEps(np.float64)
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c:
                                   MLCirculant(np.random.randn(
                                       *(2 * [c + 1])
                                   ))),
                BENCH.FUNC_SIZE : (lambda c: (c + 1) ** 2),
                BENCH.FUNC_STEP : (lambda c: c * 10 ** (1. / 12))
            },
            BENCH.FORWARD: {},
            BENCH.SOLVE: {},
            BENCH.OVERHEAD: {},
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, dt: MLCirculant(
                    np.random.randn(*(2 * [c + 1])).astype(dt)))
            }
        }

    def _getDocumentation(self):
        from .inspect import DOC
        return DOC.SUBSECTION(
            r'Multilevel Circulant Matrix (\texttt{fastmat.MLCirculant})',
            DOC.SUBSUBSECTION(
                'Definition and Interface',
                r"""
Multilevel Circulant Matrices are not circulant by themselves, but consist
of multiply nested levels of circulant structures. To this end, let
$d \geqslant 2$, $\bm n = [n_1, \dots, n_d]$,
$\bm n_{1-} = [n_1,\dots, n_{d-1}]$ and $\bm m = [n_2,\dots, n_d]$.
Then, given a $d$-dimensional complex sequence
$\bm c = [c_{\bm{k}}]$ for $\bm{k} \in \N^d$
a $d$-level circulant matrix $\bm C_{\bm n,d}$ is recursively defined as
%
\[\bm C_{\bm n,d} =
\begingroup
\setlength\arraycolsep{0pt}
\begin{bmatrix}
    \bm{C}_{[1,\bm{m}],\ell}        & \bm{C}_{[n_1,\bm{m}],\ell}
    & \dots     & \bm{C}_{[2,\bm{m}],\ell}  \\
    \bm{C}_{[2,\bm{m}],\ell}        & \bm{C}_{[1,\bm{m}],\ell}
    & \dots     & \bm{C}_{[3,\bm{m}],\ell}  \\
    \vdots                              & \vdots
    & \ddots    & \vdots                        \\
    \bm{C}_{[n_1,\bm{m}],\ell}      & \bm{C}_{[n_1 - 1,\bm{m}],\ell}
    & \dots     & \bm{C}_{[1,\bm{m}],\ell}  \\
\end{bmatrix}.
\endgroup
\]
%
So for $\bm n = (2,2)$ and $\bm c \in \C^{2 \times 2}$ we get
%
\[
\begingroup
\setlength\arraycolsep{3pt}
\bm C_{[2,2],2} =
\begin{bmatrix}
\bm C_{[1,2],1} & \bm C_{[2,2],1} \\
\bm C_{[2,2],1} & \bm C_{[1,2],1}
\end{bmatrix}
=
\begin{bmatrix}
c_{1,1} & c_{1,2} & c_{2,1} & c_{2,2} \\
c_{1,2} & c_{1,1} & c_{2,2} & c_{2,1} \\
c_{2,1} & c_{2,2} & c_{1,1} & c_{1,2} \\
c_{2,2} & c_{2,1} & c_{1,2} & c_{1,1}
\end{bmatrix}.
\endgroup
\]
                """,
                DOC.SNIPPET('# import the package',
                            'import fastmat as fm',
                            'import numpy as np',
                            '',
                            '# construct the',
                            '# parameters',
                            'n = 2',
                            'l = 2',
                            'c = np.arange(n ** l).reshape((n,n))',
                            '',
                            '# construct the matrix',
                            'C = fm.MLCirculant(c)',
                            caption=r"""
This yields
\[\bm c = (1,0,3,6)^T\]
\[\bm C = \left(\begin{array}{cccc}
    0 & 1 & 2 & 2 \\
    1 & 0 & 3 & 2 \\
    2 & 3 & 0 & 1 \\
    3 & 2 & 1 & 0
\end{array}\right)\]"""),
                r"""
This class depends on \texttt{Fourier}, \texttt{Diag}, \texttt{Kron},
\texttt{Product} and \texttt{Partial}."""
            ),
            DOC.SUBSUBSECTION(
                'Performance Benchmarks', r"""
All benchmarks were performed on a matrix
$\bm{\mathcal{C}} \in \R^{n \times n}$ with $n \in \N$ and all
entries drawn from a  standard Gaussian distribution.""",
                DOC.PLOTFORWARD(),
                DOC.PLOTFORWARDMEMORY(),
                DOC.PLOTSOLVE(),
                DOC.PLOTOVERHEAD(),
                DOC.PLOTTYPESPEED(),
                DOC.PLOTTYPEMEMORY()
            )
        )
