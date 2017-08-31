# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False
'''
  fastmat/Kron.pyx
 -------------------------------------------------- part of the fastmat package

  Kronecker Product of matrices.


  Author      : wcw, sempersn
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

from .Matrix cimport Matrix, profileUpdate
from .Product cimport Product
from .core.types cimport *
from .core.cmath cimport _arrReshape, _arrEmpty

################################################################################
################################################## class Kron
cdef class Kron(Matrix):

    ############################################## class methods
    def __init__(self, *matrices, **options):
        '''Initialize Matrix instance with a list of child matrices'''

        cdef int ff, factorCount = len(matrices)
        cdef intsize numN = 1
        cdef Matrix factor

        #check the number of matrices
        self._content = tuple(matrices)
        if factorCount < 2:
            raise ValueError("Kronecker: Product must have at least two terms")

        # determine total size and data type of matrix
        dtype = np.int8

        for ff in range(factorCount):
            factor = self._content[ff]

            #check for symmetry of matrices
            if (factor.numN != factor.numM):
                raise ValueError("Kronecker: Product terms must be symmetric")

            # acknowledge size and data type of factor
            numN *= self._content[ff].numN
            dtype = np.promote_types(dtype, factor.dtype)

        # determine dimensions of all factor terms in one tuple
        self._dims = tuple([factor.numN for factor in self._content])

        # handle type expansion with default depending on matrix type
        # default: expand small types due to accumulation during transforms
        # skip by specifying `typeExpansion=None` or override with `~=...`
        typeExpansion = options.get('typeExpansion', safeTypeExpansion(dtype))
        dtype = (dtype if typeExpansion is None
                 else np.promote_types(dtype, typeExpansion))

        # set properties of matrix
        self._initProperties(numN, numN, dtype, widenInputDatatype=True)

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        # perform index decomposition
        cdef tuple idxTerms = np.unravel_index(idx, self._dims)
        cdef intsize ii, cnt = len(self._content)
        cdef np.ndarray arrRes

        arrRes = self._content[0].getCols(idxTerms[0]).astype(self.dtype)
        for ii in range(1, cnt):
            arrRes = np.kron(arrRes, self._content[ii].getCols(idxTerms[ii]))

        return arrRes

    cpdef np.ndarray _getRow(self, intsize idx):
        cdef tuple idxTerms = np.unravel_index(idx, self._dims)
        cdef intsize ii, cnt = len(self._content)
        cdef np.ndarray arrRes

        arrRes = self._content[0].getRows(idxTerms[0]).astype(self.dtype)
        for ii in range(1, cnt):
            arrRes = np.kron(arrRes, self._content[ii].getRows(idxTerms[ii]))

        return arrRes

    cpdef object _getLargestEV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return np.prod(np.array(
            [term._getLargestEV(
                maxSteps, relEps, eps, alwaysReturn).astype(np.float64)
             for term in self._content]))

    cpdef object _getLargestSV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return np.prod(np.array(
            [term._getLargestSV(
                maxSteps, relEps, eps, alwaysReturn).astype(np.float64)
             for term in self._content]))

    cpdef Matrix _getNormalized(self):
        # redo normalization for product terms, which were normalized for a
        # different data type. Otherwide the best possible accuracy cannot be
        # achieved if a wide-type kronecker product containing matrices with
        # narrow data types.
        cdef list terms = list(self._content)
        cdef Matrix term
        cdef intsize tt

        if (self.dtypeNum == TYPE_COMPLEX128 or self.dtypeNum == TYPE_FLOAT64):
            for tt in range(len(terms)):
                term = terms[tt]
                if not (term.dtypeNum == TYPE_COMPLEX128 or
                        term.dtypeNum == TYPE_FLOAT64):
                    terms[tt] = Product(term, typeExpansion=np.float64)

        for tt in range(len(terms)):
            terms[tt] = terms[tt].normalized
        return Kron(*terms)

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        cdef tuple idxTermsN = np.unravel_index(idxN, self._dims)
        cdef tuple idxTermsM = np.unravel_index(idxM, self._dims)
        cdef intsize ii, cnt = len(self._content)

        numResult = self._content[0][idxTermsN[0], idxTermsM[0]].astype(
            self.dtype)

        for ii in range(1, cnt):
            numResult *= self._content[ii][idxTermsN[ii], idxTermsM[ii]]

        return numResult

    ############################################## class property override
    cpdef tuple _getComplexity(self):
        cdef intsize N = len(self._content)
        return (2 * N * self.numN, 2 * N * self.numN)

    cpdef _exploreNestedProfiles(self):
        '''
        Explore the runtime properties of all nested fastmat matrices. Use ane
        iterator on self._content by default to sum the profile properties of
        all nested classes of meta-classes by default. basic-classes either
        have an empty tuple for _content or need to overwrite this method.
        '''
        cdef Matrix item
        cdef intsize scale
        cdef bint bypass
        for item in self:
            scale = self.numN / item.numN
            bypass = (item._bypassAllow and
                      (item._array is not None or item._bypassAutoArray))
            profileUpdate(&(self._profileForward), scale, bypass,
                          &(item._profileForward), &(item._profileBypassFwd))
            profileUpdate(&(self._profileBackward), scale, bypass,
                          &(item._profileBackward), &(item._profileBypassBwd))

    ############################################## class forward / backward
    cpdef np.ndarray _forward(self, np.ndarray arrX):
        '''Calculate the forward transform of this matrix'''

        # detect if we have an array of signals (assume the vector to
        # be two-dimensional)
        cdef intsize numSize = arrX.shape[0]
        cdef intsize numVecs = arrX.shape[1]
        cdef intsize ii, termCnt = len(self._content)
        cdef Matrix term
        cdef np.ndarray arrData = arrX

        # keep track of the identity compositions' head identity size
        cdef intsize headIN = 1

        for ii in range(termCnt):
            term = self._content[ii]

            # increase size of identity composition for this term
            headIN *= term.numN

            # reshape to match composition size
            arrData = _arrReshape(
                arrData,
                2, headIN, numVecs * self.numN / headIN,
                np.NPY_CORDER)
            # reshape to match term size
            arrData = _arrReshape(
                arrData,
                2, term.numN, numVecs * self.numN / term.numN,
                np.NPY_FORTRANORDER)

            # apply transform and reshape back to current comp. size
            arrData = _arrReshape(
                term.forward(arrData),
                2, headIN, numVecs * self.numN / headIN,
                np.NPY_FORTRANORDER)

        # reshape to matrix output dimensions
        return _arrReshape(arrData, 2, self.numN, numVecs, np.NPY_CORDER)

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        '''Calculate the backward transform of this matrix'''

        # detect if we have an array of signals (assume the vector to
        # be two-dimensional)
        cdef intsize numSize = arrX.shape[0]
        cdef intsize numVecs = arrX.shape[1]
        cdef intsize ii, termCnt = len(self._content)
        cdef Matrix term
        cdef np.ndarray arrData = arrX

        # keep track of the identity compositions' head identity size
        cdef intsize headIN = 1

        for ii in range(termCnt):
            term = self._content[ii]

            # increase size of identity composition for this term
            headIN *= term.numN

            # reshape to match composition size
            arrData = _arrReshape(
                arrData,
                2, headIN, numVecs * self.numN / headIN,
                np.NPY_CORDER)
            # reshape to match term size
            arrData = _arrReshape(
                arrData,
                2, term.numN, numVecs * self.numN / term.numN,
                np.NPY_FORTRANORDER)

            # apply transform and reshape back to current comp. size
            arrData = _arrReshape(
                term.backward(arrData),
                2, headIN, numVecs * self.numN / headIN,
                np.NPY_FORTRANORDER)

        # reshape to matrix output dimensions
        return _arrReshape(arrData, 2, self.numN, numVecs, np.NPY_CORDER)

    ############################################## class reference
    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using
        any fastmat code.
        '''
        cdef np.ndarray arrRes
        cdef int ff, factorCount = len(self._content)

        arrRes = self._content[0].reference().astype(
            np.promote_types(self.dtype, np.float64))
        for ff in range(1, factorCount):
            arrRes = np.kron(arrRes, self._content[ff].reference())

        return arrRes

    ############################################## class inspection, QM
    def _getTest(self):
        from .inspect import TEST, dynFormat
        return {
            TEST.COMMON: {
                TEST.NUM_N      : 5 * 4 * 3,
                TEST.NUM_M      : TEST.NUM_N,
                'mType1'        : TEST.Permutation(TEST.ALLTYPES),
                'mType2'        : TEST.Permutation(TEST.ALLTYPES),
                'arr1'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mType1',
                    TEST.SHAPE  : (5, 5)
                }),
                'arr2'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mType2',
                    TEST.SHAPE  : (4, 4)
                }),
                'arr3'          : TEST.ArrayGenerator({
                    TEST.DTYPE  : 'mType1',
                    TEST.SHAPE  : (3, 3)
                }),
                TEST.INITARGS: (lambda param : [Matrix(param['arr1']()),
                                                Matrix(param['arr2']()),
                                                Matrix(param['arr3']())]),
                TEST.OBJECT     : Kron,
                TEST.NAMINGARGS : dynFormat("%so%so%s", 'arr1', 'arr2', 'arr3'),
                TEST.TOL_POWER  : 4.
            },
            TEST.CLASS: {},
            TEST.TRANSFORMS: {}
        }

    def _getBenchmark(self):
        from .inspect import BENCH, arrTestDist
        from .Fourier import Fourier
        from .Diag import Diag
        from .Matrix import Matrix
        from .Eye import Eye
        return {
            BENCH.COMMON: {
                BENCH.FUNC_GEN  : (lambda c: Kron(
                    Fourier(2 * c),
                    Diag(np.random.uniform(2, 3, (2 * c))),
                    Matrix(arrTestDist((2 * c, 2 * c), dtype=np.complex)))),
                BENCH.FUNC_SIZE : (lambda c: 8 * c ** 3)
            },
            BENCH.FORWARD: {},
            BENCH.SOLVE: {},
            BENCH.OVERHEAD: {
                BENCH.FUNC_GEN  : (lambda c: Kron(*([Eye(2)] * c))),
                BENCH.FUNC_SIZE : (lambda c: (2) ** c)
            },
            BENCH.DTYPES: {
                BENCH.FUNC_GEN  : (lambda c, dt: Kron(
                    Fourier(2 * c),
                    Diag(np.random.uniform(2, 3, (2 * c)).astype(dt)),
                    Matrix(arrTestDist((2 * c, 2 * c), dt)))),
                BENCH.FUNC_SIZE : (lambda c: 8 * c ** 3)
            }
        }

    def _getDocumentation(self):
        from .inspect import DOC
        return DOC.SUBSECTION(
            r'Kronecker Product (\texttt{fastmat.Kron})',
            DOC.SUBSUBSECTION(
                'Definition and Interface', r"""
For matrices $\bm A_i \in \C^{n_i \times n_i}$ for $i = 1,\dots,k$ the
Kronecker product
\[\bm A_1 \otimes \bm A_2 \otimes \dots \otimes \bm A_k\]
can be defined recursively because of associativity from the Kronecker product
of $\bm A \in \C^{n \times m}$ and $\bm B \in \C^{r \times s}$ defined as
\[\bm A \otimes \bm B =
\left(\begin{array}{ccc}
    a_{11} \bm B    & \dots     & a_{1m} \bm B  \\
    \vdots          & \ddots    & \vdots        \\
    a_{n1} \bm B    & \dots     & a_{nm} \bm B
\end{array}\right).\]
We make use of a decomposition into a standard matrix product to speed up the
matrix-vector multiplication which is introduced in
\cite{kron_fernandes1998automata_networks}. This then yields multiple benefits:
\begin{itemize}
\item It already brings down the complexity of the forward and backward
    projection if the factors $\bm A_i$ have no fast transformations.
\item It is not necessary to compute the matrix representation of the product,
    which saves \textbf{a lot} of memory.
\item When fast transforms of the factors are available the calculations can be
    sped up further.
\end{itemize}""",
                DOC.SNIPPET('# import the package',
                            'import fastmat as fm',
                            '',
                            '# define the factors',
                            'C = fm.Circulant(x_C)',
                            'H = fm.Hadamard(n)',
                            '',
                            '# define the Kronecker',
                            '# product',
                            'P = fm.Kron(C.H, H)',
                            caption=r"""
Assume we have a circulant matrix $\bm C$ with first column $\bm x_c$ and a
Hadamard matrix $\bm{\mathcal{H}}_n$ of order $n$. Then we define
    \[\bm P = \bm C^H \otimes \bm H_n.\]""")
            ),
            DOC.SUBSUBSECTION(
                'Performance Benchmarks', r"""
All but the overhead benchmarks were performed on a matrix
$\bm K = \bm \Fs_{2k} \otimes \bm D_{2k} \otimes \bm M_{2k}$, where $\bm \Fs$
is a Fourier Matrix, $\bm D$ is diagonal and $\bm M$ is unstructured and
complex valued; so $n = 8 k^3$ for $k \in \N$""",
                DOC.PLOTFORWARD(),
                DOC.PLOTFORWARDMEMORY(),
                DOC.PLOTSOLVE(),
                DOC.PLOTOVERHEAD(doc=r"""
$\bm K = \prod_k \bm I_k$;
so $n=k^4$ for $k \in \N$"""),
                DOC.PLOTTYPESPEED(),
                DOC.PLOTTYPEMEMORY()
            ),
            DOC.BIBLIO(
                kron_fernandes1998automata_networks=DOC.BIBITEM(
                    r"""
Fernandes, Paulo and Plateau, Brigitte and Stewart, William J.,""",
                    r"""
Efficient Descriptor-Vector Multiplications in Stochastic Automata Networks""",
                    r'Journal of the ACM, New York, Volume 45, 1998')
            )
        )
