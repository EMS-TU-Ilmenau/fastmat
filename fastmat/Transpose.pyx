# -*- coding: utf-8 -*-
'''
  fastmat/Transpose.pyx
 -------------------------------------------------- part of the fastmat package

  Transpositions of fastmat matrices. Includes Type definitions and Factories.

  Contains:
     - TpFlags        - Transposition flags for defining a transposed matrix
     - Transpose    - fastmat class for transposed matrices
     - TransposeFactory - Factory for generation of transposed matrices

  Author      : wcw, sempersn
  Introduced  : 2016-07-10
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
cimport numpy as np

from .Matrix cimport Matrix
from .helpers.types cimport *
from .helpers.cmath cimport _conjugate, _conjugateInplace

################################################################################
################################################## class TpFlags
cdef class TpFlags(object):
    '''
    TpFlags
    A class to represent a transpose as superposition of processing steps.
    '''

    ############################################## class properties
    # type - Property (read-write)
    # Return / Set transposition type. (see self.decode / self.encode)
    property type:
        def __get__(self):
            return self.decode()

        def __set__(self, value):
            self.encode(value)

    # isSuperfluous - Property (read-only)
    # Decide whether a transposition is needed or can be ommitted.
    property isSuperfluous:
        def __get__(self):
            return not (self.applyH or self.applyC)

    ############################################## class copy/deepcopy handling
    def __copy__(self):
        '''Return copy of this instance containing no shared objects.'''
        return TpFlags(self.decode())

    def __deepcopy__(self, dict memo):
        '''Return copy of this instance containing no shared objects.'''
        return TpFlags(self.decode())

    ############################################## class methods
    def __init__(self, transposition):
        '''Initialize transposition flag container.'''
        self.encode(transposition)

    def __repr__(self):
        '''Return a string representing the transposition.'''
        return "%s:%s" %(self.__class__.__name__, str(self))

    def __str__(self):
        '''Return the transposition type indicator as string.'''
        return ['1', 'C', 'H', 'T'][self.decode()]

    cpdef TpType decode(self):
        '''Return transposition type.'''
        if self.applyH:
            return TRANSPOSE_T if self.applyC else TRANSPOSE_H
        else:
            return TRANSPOSE_C if self.applyC else TRANSPOSE_NONE

    cpdef encode(self, TpType value):
        '''Set transposition flags according the transposition type.'''

        self.applyH = (value == TRANSPOSE_H) or (value == TRANSPOSE_T)
        self.applyC = (value == TRANSPOSE_C) or (value == TRANSPOSE_T)

    cpdef apply(self, TpFlags value):
        '''Combine two transpositions by applying one on the other.'''

        if not isinstance(value, self.__class__):
            # not of same type: cast it (converting unknown argument to flags)
            value = self.__class__(value)

        self.applyH = self.applyH != value.applyH
        self.applyC = self.applyC != value.applyC

        return self


################################################################################
################################################## class Transpose
cdef class Transpose(Matrix):
    '''
    class fastmat.Diag
    '''

    ########################################## Class local variables
    #
    #  _flags        Flags: Transpositions to be applied
    #  _content        fastmat matrix to be transformed

    ########################################## Class properties

    # flags - Property (read-only)
    # Return flags describing which transpositions are applied to matrix
    property flags:
        def __get__(self):
            return self._flags

    # type - Property (read-only)
    # Name the type of transposition resulting from the applied flags
    property type:
        def __get__(self):
            return str(self._flags)

    # content - Property (read-only)
    # Return the fastmat matrix which is transformed by Transpose
    property content:
        def __get__(self):
            return self._content

    ############################################## class property override
    cpdef np.ndarray _getCol(self, intsize idx):
        cdef np.ndarray arrRes
        arrRes = (self._content._getRow(idx) if self._flags.applyH
                  else self._content._getCol(idx))
        return (_conjugate(arrRes) if self._flags.applyC ^ self._flags.applyH
                else arrRes)

    cpdef np.ndarray _getRow(self, intsize idx):
        cdef np.ndarray arrRes
        arrRes = (self._content._getCol(idx) if self._flags.applyH
                  else self._content._getRow(idx))
        return (_conjugate(arrRes) if self._flags.applyC ^ self._flags.applyH
                else arrRes)

    cpdef object _getLargestEV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return self._content.largestEV

    cpdef object _getLargestSV(self, intsize maxSteps,
                               float relEps, float eps, bint alwaysReturn):
        return self._content.largestSV

    cpdef object _getItem(self, intsize idxN, intsize idxM):
        item = self._content._getItem(idxM if self._flags.applyH else idxN,
                                      idxN if self._flags.applyH else idxM)
        return (np.conjugate(item) if self._flags.applyC ^ self._flags.applyH
                else item)

    cpdef np.ndarray toarray(self):
        '''
        Return an explicit representation of the matrix as numpy-array.
        '''
        cdef np.ndarray arrRes = self._content.toarray()
        if self._flags.applyH:
            arrRes = arrRes.T
            # conjugate of [.H === .T.conj()] in line above is skipped to avoid
            # double conjugation. therefor negation hier
            return (_conjugate(arrRes) if not self._flags.applyC else arrRes)
        return (_conjugate(arrRes) if self._flags.applyC else arrRes)

    ############################################## class methods
    def __init__(self, mat, transposition_flags):
        '''Initialize Matrix instance with a list of child matrices'''

        if not issubclass(type(mat), Matrix):
            raise TypeError("Only fastmat matrices allowed in Transpose.")

        if not isinstance(transposition_flags, TpFlags):
            raise TypeError("A TpFlags object must be given to Transpose.")

        self._content = mat
        self._flags = transposition_flags

        # set properties of matrix
        # applying H effectively exchanges matrix dimensions
        self._initProperties(
            mat.numM if self._flags.applyH else mat.numN,
            mat.numN if self._flags.applyH else mat.numM,
            self._content.dtype,
            cythonCall=self._content._cythonCall,
            widenInputDatatype=self._content._widenInputDatatype,
            forceInputAlignment=self._content._forceInputAlignment
        )

    def __repr__(self):
        '''
        Return a string representation of this class instance.
        The __repr__() method of the nested transformation gets extended by an
        info about the applied transposition.
        '''
        return "<%s.%s:0x%12x>" %(
            self._content.__repr__(), self.type, id(self))

    def __str__(self):
        '''
        Return a human-readable string representation of the classes' contents.
        The __str__() method of the nested transformation gets extended by an
        info about the applied transposition.
        '''
        return "<%s.'%s':0x%12x>" %(
            self._content.__str__(), self.type, id(self))

    ############################################## class forward / backward
    cpdef _forwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        '''
        Calculate the forward transform of this matrix, cython-style.
        '''
        cdef np.ndarray arrInput
        arrInput = _conjugate(arrX) if self._flags.applyC else arrX

        # apply hermitian transpose by swapping _forward and _backward
        if self._flags.applyH:
            self._content._backwardC(arrInput, arrRes, typeX, typeRes)
        else:
            self._content._forwardC(arrInput, arrRes, typeX, typeRes)

        if self._flags.applyC:
            _conjugateInplace(arrRes)

    cpdef _backwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        '''
        Calculate the backward transform of this matrix, cython-style.
        '''
        cdef np.ndarray arrInput
        arrInput = _conjugate(arrX) if self._flags.applyC else arrX

        # apply hermitian transpose by swapping _forward and _backward
        if self._flags.applyH:
            self._content._forwardC(arrInput, arrRes, typeX, typeRes)
        else:
            self._content._backwardC(arrInput, arrRes, typeX, typeRes)

        if self._flags.applyC:
            _conjugateInplace(arrRes)

    cpdef np.ndarray _forward(self, np.ndarray arrX):
        '''Calculate the forward transform of this matrix'''
        cdef np.ndarray arrRes
        arrRes = _conjugate(arrX) if self._flags.applyC else arrX

        # apply hermitian transpose by swapping _forward and _backward
        if self._flags.applyH:
            arrRes = self._content._backward(arrRes)
        else:
            arrRes = self._content._forward(arrRes)

        if self._flags.applyC:
            _conjugateInplace(arrRes)

        return arrRes

    cpdef np.ndarray _backward(self, np.ndarray arrX):
        '''Calculate the backward transform of this matrix'''
        cdef np.ndarray arrRes
        arrRes = _conjugate(arrX) if self._flags.applyC else arrX

        # apply hermitian transpose by swapping _forward and _backward
        if self._flags.applyH:
            arrRes = self._content._forward(arrRes)
        else:
            arrRes = self._content._backward(arrRes)

        if self._flags.applyC:
            _conjugateInplace(arrRes)

        return arrRes

    ########################################## references: test / benchmark

    cpdef np.ndarray _reference(self):
        '''
        Return an explicit representation of the matrix without using any
        fastmat code.
        '''
        cdef np.ndarray ref = self._content.reference()
        if self._flags.applyH:
            ref = ref.T if self._flags.applyC else ref.T.conj()
        else:
            if self._flags.applyC:
                ref = ref.conj()

        return ref


################################################################################
################################################## TransposeFactory()
cpdef Matrix TransposeFactory(
    Matrix mat,
    TpType transposition
):
    '''
    Return the minimal needed Transpose for a given transformation and matrix.
    The factory resolves stacked transpositions and is able to return the
    original class in case no transposition is needed after resolving.

    Returns either a fastmat.Transpose instance or the given matrix itself.
    '''
    # determine flagset for operation representation
    cdef TpFlags flags = TpFlags(transposition)

    # chain-scan transformation for nested Transpose-classes and merge them
    # with set of current transposition flags
    instance = mat
    while isinstance(instance, Transpose):
        flags.apply(instance.flags)
        instance = instance.content

    return instance if flags.isSuperfluous else Transpose(instance, flags)


################################################################################
################################################################################
from .helpers.unitInterface import *
from .Sum cimport Sum

################################################## Testing
test = {
    NAME_COMMON: {
        # define matrix sizes and parameters
        TEST_NUM_N: 25,
        TEST_NUM_M: Permutation([TEST_NUM_N, 20]),
        'mType': Permutation(typesAll),
        'arrM': ArrayGenerator({
            NAME_DTYPE  : 'mType',
            NAME_SHAPE  : (TEST_NUM_N, TEST_NUM_M)
            #            NAME_CENTER : 2,
        }),

        # define data array for test
        # test single- and multi-column and various types of data array
        # NOTE: the shape of the data array must match matrix dimensions
        #       depending on the transformation type data dims may be changed!
        # TEST_DATAARRAY, TEST_DATATYPE are defaults for class tests
        TEST_DATASHAPE: (
            (lambda param: \
                param[TEST_NUM_N] if param['transp'].applyH \
             else param[TEST_NUM_M]),
            TEST_DATACOLS
        ),
        TEST_DATASHAPE_T: (
            (lambda param: \
                param[TEST_NUM_M] if param['transp'].applyH \
             else param[TEST_NUM_N]),
            TEST_DATACOLS
        ),

        # test all transformation types
        'transp': Permutation([
            TpFlags(TRANSPOSE_NONE),
            TpFlags(TRANSPOSE_C),
            TpFlags(TRANSPOSE_T),
            TpFlags(TRANSPOSE_H)
        ]),

        # define constructor for test instances and naming of test
        TEST_OBJECT: Transpose,
        'transformee': Permutation([Matrix, Sum]),
        TEST_INITARGS: [
            (lambda param: Matrix(param['arrM']()) \
                if param['transformee'] == Matrix \
                else Sum(*((Matrix(param['arrM']()), ) * 3))),
            'transp'
        ],

        # name the test instances individually to reflect test scenario
        'strT': (lambda param: str(param['transp'])),
        'strType': (lambda param: param['transformee'].__name__),
        TEST_NAMINGARGS: dynFormatString(
            "%s%s.%s", 'strType', 'arrM', 'strT')
    }, TEST_CLASS: {
        TEST_NUM_N: 24,
        TEST_NUM_M: Permutation([TEST_NUM_N, 32]),
    }, TEST_TRANSFORMS: {
    }
}


################################################## Benchmarks
from .Eye import Eye

benchmark = {
    NAME_COMMON: {
        NAME_DOCU       : r'''$\bm M = \bm I^H_{2^k}$; so $n = 2^k$'''
    },
    BENCH_OVERHEAD: {
        BENCH_FUNC_GEN  : (lambda c : Eye(2 ** c).T)
    }
}


################################################## Documentation
docLaTeX = r"""
\subsection{Transposition (\texttt{fastmat.Transpose})}
\subsubsection{Definition and Interface}
This class allows to form $\bm A^\trans$, $\bm A^\herm$ and $\bm A^*$ for a
given transformation $\bm A$.
\begin{snippet}
\begin{lstlisting}[language=Python]
# import the package
import fastmat as fm
import numpy as np
# define some fastmat matrix
c = np.arange(10)
C = fm.Circulant(c)

# define the transposition
CT = fm.Transpose(C, 'T')

# define the hermitian
# transpose
CH = fm.Transpose(C, 'H')

# define the complex conjugate
CC = fm.Transpose(C, 'C')
\end{lstlisting}

Let $\bm C$ be a circulant matrix. Then the transform equivalent to $\bm
C^\herm$ can be defined as $\bm T$.
\end{snippet}

\begin{snippet}
The same results as above can be achieved much simpler via the following code.

\begin{lstlisting}[language=Python]
# import the package
import fastmat as fm
import numpy as np

# define some fastmat matrix
c = np.arange(10)
C = fm.Circulant(c)

# define the transposition
CT = C.T

# define the hermitian transpose
CH = C.H

# define the complex conjugate
CC = C.C
\end{lstlisting}
\end{snippet}
"""
