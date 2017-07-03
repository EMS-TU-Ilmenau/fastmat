# -*- coding: utf-8 -*-
'''
  fastmat/helpers/unitHandler.py
 -------------------------------------------------- part of the fastmat package

  Interface to unit testing, benchmarks and documentation generation.


  Author      : wcw
  Introduced  : 2017-01-12
 ------------------------------------------------------------------------------

   Copyright 2016 Sebastian Semper, Christoph Wagner

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
import copy
import inspect
import pprint
import six
import numpy as np
from scipy import sparse
from .types import _getTypeMax

################################################## Define string names


class TypeDict(dict):

    def __getitem__(self, key):
        if key not in self:
            key = key.type if isinstance(key, np.dtype) else None
        return dict(self).get(key, '???')


NAME_TYPES = TypeDict({
    np.int8:       'i08', np.int32:      'i32', np.int64:      'i64',
    np.float32:    'f32', np.float64:    'f64',
    np.complex64:  'c32', np.complex128: 'c64', None:          '???'
})
NAME_DATA           = 'data'
NAME_FWDATA         = 'dataForward'
NAME_BWDATA         = 'dataBackward'
NAME_DTYPE          = 'dtype'
NAME_SHAPE          = 'shape'
NAME_SHAPE_T        = 'shapeBackward'
NAME_ALIGN          = 'align'
NAME_CENTER         = 'center'
NAME_FORMAT         = 'format'
NAME_NAME           = 'name'

NAME_UNIT           = 'unit'
NAME_TEST           = 'test'
NAME_BENCHMARK      = 'benchmark'
NAME_DOCU           = 'docLaTeX'
NAME_FILENAME       = 'filename'
NAME_CAPTION        = 'caption'
NAME_COMMON         = 'common'
NAME_TEMPLATE       = 'template'
NAME_VARIANT        = 'variant'


def dynFormatString(s, *keys):
    return s.replace('%', '%%(%s)') % keys


################################################## define CONSTS for test params
TEST_CLASS          = 'class'
TEST_TRANSFORMS     = 'transform'
TEST_ALGORITHM      = 'algorithm'

TEST_OBJECT         = 'object'
TEST_ARGS           = 'testArgs'
TEST_KWARGS         = 'testKwargs'
TEST_NUM_N          = 'numN'
TEST_NUM_M          = 'numM'
TEST_DATACOLS       = 'numCols'
TEST_DATATYPE       = 'dataType'
TEST_DATASHAPE      = 'dataShape'
TEST_DATAALIGN      = 'dataAlign'
TEST_DATASHAPE_T    = 'dataShapeBackward'
TEST_DATAGEN        = 'dataGenerator'
TEST_DATACENTER     = 'dataDistCenter'
TEST_DATAARRAY      = 'arrData'
TEST_INIT           = 'init'
TEST_INITARGS       = 'args'
TEST_INITKWARGS     = 'kwargs'
TEST_INIT_VARIANT   = 'initVariant'
TEST_INSTANCE       = 'instance'
TEST_REFERENCE      = 'reference'
TEST_QUERY          = 'query'
TEST_NAMINGARGS     = 'namingArgs'
TEST_NAMING         = 'naming'
TEST_TOL_MINEPS     = 'tolMinEps'
TEST_TOL_POWER      = 'tolPower'
TEST_IGNORE         = 'ignore'
TEST_PARAMALIGN     = 'alignment'
TEST_RESULT_TOLERR  = 'testTolError'
TEST_RESULT_INPUT   = 'testInput'
TEST_RESULT_OUTPUT  = 'testOutput'
TEST_RESULT_REF     = 'testReference'
TEST_RESULT         = 'testResult'
TEST_RESULT_IGNORED = 'testResultIgnored'
TEST_RESULT_INFO    = 'testInfo'
TEST_ALG            = 'algorithm'
TEST_ALG_ARGS       = 'algorithmArgs'
TEST_ALG_KWARGS     = 'algorithmKwargs'
TEST_REFALG         = 'refAlgorithm'
TEST_REFALG_ARGS    = 'refAlgorithmArgs'
TEST_REFALG_KWARGS  = 'refAlgorithmKwargs'

TEST_CHECK_PROXIMITY    = 'checkProximity'
TEST_CHECK_DATATYPE     = 'checkDataType'
TEST_TYPE_EXPECTED      = 'typeExpected'
TEST_TYPE_PROMOTION     = 'typePromotion'

ALIGN_DONTCARE      = '-'
ALIGN_FCONT         = 'F'
ALIGN_CCONT         = 'C'
ALIGN_STRIDE        = 'S'

################################################## define CONSTS for benchmarks
BENCH_FORWARD       = 'forward'
BENCH_SOLVE         = 'solve'
BENCH_OVERHEAD      = 'overhead'
BENCH_DTYPES        = 'dtypes'
BENCH_PERFORMANCE   = 'performance'

BENCH_FUNC_INIT     = 'funcInit'
BENCH_FUNC_GEN      = 'funcConstr'
BENCH_FUNC_SIZE     = 'funcSize'
BENCH_FUNC_STEP     = 'funcStep'

BENCH_STEP_START_K  = 'startK'
BENCH_STEP_LOGSTEPS = 'logSteps'

BENCH_LIMIT_MEMORY  = 'maxMem'
BENCH_LIMIT_INIT    = 'maxInit'
BENCH_LIMIT_ITER    = 'maxIter'
BENCH_LIMIT_SIZE    = 'maxSize'

BENCH_RESULT_MEMORY = 'totalMem'
BENCH_RESULT_INIT   = 'initTime'
BENCH_RESULT_ITER   = 'iterTime'
BENCH_RESULT_K      = 'K'
BENCH_RESULT_SIZE   = 'numN'

BENCH_MEAS_MINTIME  = 'measMinTime'
BENCH_MEAS_MINREPS  = 'measMinReps'
BENCH_MEAS_MINRUNS  = 'measMinRuns'
BENCH_MEAS_COMPACT  = 'compactStats'

################################################## default sets of data types
typesInt            = [np.int8, np.int32, np.int64]
typesFloat          = [np.float32, np.float64]
typesComplex        = [np.complex64, np.complex128]

typesAll            = typesInt + typesFloat + typesComplex
typesSingleIFC      = [np.int32, np.float32, np.complex64]
typesDoubleIFC      = [np.int64, np.float64, np.complex128]
typesSmallIFC       = [np.int8, np.int32, np.float32, np.complex64]

alignmentsAll       = [ALIGN_FCONT, ALIGN_CCONT, ALIGN_STRIDE]

################################################## Permutation


class Permutation(list):

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__,
                           super(Permutation, self).__repr__())

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__,
                           super(Permutation, self).__str__())


class VariantPermutation(Permutation):
    pass

################################################## IgnoreDict


class IgnoreDict(dict):
    pass


################################################## IgnoreFunc
class IgnoreFunc(object):

    def __init__(self, fun):
        self._fun = fun

    def __call__(self, *args, **kwargs):
        return self._fun(*args, **kwargs)


################################################## arrTestDist()
def arrTestDist(shape, dtype, center=0):
    if np.prod(shape) < 1:
        return np.array([])

    def draw():
        '''
        Draw a random floating-point number from a test distribution.
        Remove the part around zero from the distribution and keep the distance
        between minimal and maximal absolute values (dynamics) relatively small
        '''
        return ((np.random.uniform(2., 1., size=shape) + center) *
                np.random.choice([-1, 1], shape))

    if np.issubdtype(dtype, np.int):
        result = np.random.choice(
            [center - 2, center - 1, center + 1, center + 2], shape).astype(
                dtype)
    else:
        if np.issubdtype(dtype, np.float):
            result = draw().astype(dtype)
        elif np.issubdtype(dtype, np.complex):
            result = (draw() + np.real(center) +
                      1j * (draw() + np.imag(center))).astype(dtype)
        else:
            raise TypeError("arrTestDist: unsupported type %s" % (dtype))

    # increase the largest element in magnitude a little bit more to avoid
    # too close neighbours to the largest element in the distribution
    # this helps at least largestSV in Diag matrices to converge ;)
    idxMax = np.unravel_index(np.abs(result).argmax(), result.shape)
    if np.issubdtype(dtype, np.int):
        result[idxMax] += np.sign(result[idxMax])
    else:
        result[idxMax] *= 1.5

    return result


################################################## arrSparseTestDist()
def arrSparseTestDist(shape, dtype,
                      density=0.1, center=0, compactFullyOccupied=False):
    numSize = np.prod(shape)
    if compactFullyOccupied:
        # draw just enough samples randomly such that every row and column is
        # occupied with at least one element. Ignore the density parameter
        numElements = max(shape)

        # draw mm and nn coordinates from coordinate space. Modulo operation
        # wraps indices larger than the actual row or column dimension
        suppX = np.mod(
            np.random.choice(numElements, numElements, replace=False), shape[0])
        suppY = np.mod(
            np.random.choice(numElements, numElements, replace=False), shape[1])
    else:
        # draw a relative amount of samples randomly over vectorized index space
        numElements = int(numSize * density)
        supp = np.random.choice(np.arange(numSize), numElements, replace=False)
        suppX = np.mod(supp, shape[0])
        suppY = np.divide(supp, shape[0])

    # determine the actual element values distributed over the sparse array
    # from arrTestDist with a 1D-array spanning the required element count
    arrElements = arrTestDist((numElements, ), dtype, center=center)

    return sparse.coo_matrix(
        (arrElements, (suppX, suppY)), shape=shape, dtype=dtype)


################################################## arrAlign()
def arrAlign(arr, alignment=ALIGN_DONTCARE):
    if alignment == ALIGN_DONTCARE:
        return np.asanyarray(arr)
    elif alignment == ALIGN_FCONT:
        return np.asfortranarray(arr)
    elif alignment == ALIGN_CCONT:
        return np.ascontiguousarray(arr)
    elif alignment == ALIGN_STRIDE:
        # define spacing between elements
        spacing = 3
        # determine maximum value
        try:
            maxValue = np.iinfo(arr.dtype).max
        except ValueError:
            try:
                maxValue = np.finfo(arr.dtype).max
            except ValueError:
                maxValue = 1.

        # generate large random array with maximized data type utilization
        arrFill = (maxValue * (np.random.rand(
            *(dim * spacing for dim in arr.shape)) - 0.5)).astype(arr.dtype)

        # fill-in the array data and return a view of the to-be-aligned array
        if arrFill.ndim == 1:
            arrFill[1::spacing] = arr
            return arrFill[1::spacing]
        elif arrFill.ndim == 2:
            arrFill[1::spacing, 1::spacing] = arr
            return arrFill[1::spacing, 1::spacing]
        elif arrFill.ndim == 3:
            arrFill[1::spacing, 1::spacing, 1::spacing] = arr
            return arrFill[1::spacing, 1::spacing, 1::spacing]
        else:
            raise ValueError("Only arrays of dimensions <3 are supported.")
    else:
        raise ValueError("Unknown alignment identificator '%s'" %(alignment))


################################################## arrayGenerator
class ArrayGenerator(dict):

    @property
    def forwardData(self):
        if NAME_FWDATA not in self:
            # generate random array and set specific alignment style
            self[NAME_FWDATA] = arrAlign(
                arrTestDist(self[NAME_SHAPE], self[NAME_DTYPE],
                            center=self.get(NAME_CENTER, 0)),
                alignment=self.get(NAME_ALIGN, ALIGN_DONTCARE))

        return self[NAME_FWDATA]

    @property
    def backwardData(self):
        if NAME_BWDATA not in self:
            # generate random array and set specific alignment style
            self[NAME_BWDATA] = arrAlign(
                arrTestDist(self[NAME_SHAPE_T], self[NAME_DTYPE],
                            center=self.get(NAME_CENTER, 0)),
                alignment=self.get(NAME_ALIGN, ALIGN_DONTCARE))

        return self[NAME_BWDATA]

    def __call__(self):
        return self.forwardData

    def __str__(self):
        '''Compose a compact description of the represented array.'''
        tags = []

        # generate shape-token: check for both shape variants (fw and bw).
        # if they differ, print both as "fw/bw", otherwise print the dim only
        fwShape = self[NAME_SHAPE] if NAME_SHAPE in self else ()
        bwShape = self[NAME_SHAPE_T] if NAME_SHAPE_T in self else ()

        def printDim(fw, bw):
            if fw is None:
                return '-' if bw is None else str(bw)
            else:
                return (str(fw) if bw is None
                        else "%s/%s" %(fw, bw) if (fw != bw) else str(fw))

        strShape = 'x'.join([
            printDim(fw, bw)
            for fw, bw in six.moves.zip_longest(fwShape, bwShape)])
        if len(strShape) > 0:
            tags.append(strShape)

        # print the data type of the array
        value = self.get(NAME_DTYPE, '')
        if isinstance(value, type):
            tags.append(NAME_TYPES.get(value, str(value)))

        value = self.get(NAME_ALIGN, None)
        if value in alignmentsAll:
            tags.append(value)

        return str("[%s]" % (','.join(tags)))

    def __repr__(self):
        return self.__str__()
