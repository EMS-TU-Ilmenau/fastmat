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

import itertools
import numpy as np
from pprint import pprint

from .common import *
from ..core.types import getTypeEps, safeTypeExpansion
from ..Matrix import Matrix
from ..algorithms import Algorithm
from ..Diag import Diag
from ..Eye import Eye
from ..Parametric import Parametric
from ..Zero import Zero


################################################## TEST CONSTANT-def class
class TEST(NAME):
    CLASS           = 'class'
    TRANSFORMS      = 'transform'
    ALGORITHM       = 'algorithm'

    OBJECT          = 'object'
    ARGS            = 'testArgs'
    KWARGS          = 'testKwargs'
    NUM_ROWS        = 'numRows'
    NUM_COLS        = 'numCols'
    DATACOLS        = 'numVectors'
    DATATYPE        = 'dataType'
    DATASHAPE       = 'dataShape'
    DATAALIGN       = 'dataAlign'
    DATASHAPE_T     = 'dataShapeBackward'
    DATAGEN         = 'dataGenerator'
    DATACENTER      = 'dataDistCenter'
    DATAARRAY       = 'arrData'
    INIT            = 'init'
    INITARGS        = 'args'
    INITKWARGS      = 'kwargs'
    INIT_VARIANT    = 'initVariant'
    INSTANCE        = 'instance'
    REFERENCE       = 'reference'
    QUERY           = 'query'
    NAMINGARGS      = 'namingArgs'
    NAMING          = 'naming'
    TOL_MINEPS      = 'tolMinEps'
    TOL_POWER       = 'tolPower'
    IGNORE          = 'ignore'
    PARAMALIGN      = 'alignment'
    RESULT_TOLERR   = 'testTolError'
    RESULT_INPUT    = 'testInput'
    RESULT_OUTPUT   = 'testOutput'
    RESULT_REF      = 'testReference'
    RESULT_IGNORED  = 'testResultIgnored'
    RESULT_INFO     = 'testInfo'
    RESULT_TYPE     = 'testResultType'
    RESULT_PROX     = 'testResultProximity'
    ALG             = 'algorithm'
    ALG_MATRIX      = 'algorithmMatrix'
    ALG_ARGS        = 'algorithmArgs'
    ALG_KWARGS      = 'algorithmKwargs'
    REFALG          = 'refAlgorithm'
    REFALG_ARGS     = 'refAlgorithmArgs'
    REFALG_KWARGS   = 'refAlgorithmKwargs'

    CHECK_PROXIMITY = 'checkProximity'
    CHECK_DATATYPE  = 'checkDataType'
    TYPE_EXPECTED   = 'typeExpected'
    TYPE_PROMOTION  = 'typePromotion'

################################################################################
##################################################  test implementations


################################################## tryQuery
def tryQuery(nameTest, query, argument):
    try:
        return query(argument)
    except Exception as e:
        print("Exception %s in %s: %s" %(
            repr(e), nameTest, query,
        ))
        pprint(argument)
        raise e


################################################## compareResults
def compareResults(test, query):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.
    query : type
        Description of parameter `query`.

    Returns
    -------
    type
        Description of returned object.

    """

    def getOption(option, default):
        return query.get(option, test.get(option, default))

    arrOutput=query[TEST.RESULT_OUTPUT]
    arrReference=query[TEST.RESULT_REF]
    arrMatrix=test[TEST.REFERENCE]

    # to extract the operation data type and shape when checking an algorithm
    # we cannot get this information from TEST.INSTANCE as this then refers to
    # the algorithm instance object itself and not to the matrix it operates
    # on. We need to use the TEST.ALG_MATRIX as this holds the required matrix
    # instance object used within the actual algorithm
    instance = query[TEST.INSTANCE] = getOption(
        TEST.ALG_MATRIX,
        test[TEST.INSTANCE]
    )

    # get comparision options from test and query dicts. Priority has query
    minimalType=np.dtype(getOption(TEST.TYPE_PROMOTION, np.int8))
    expectedType=np.dtype(getOption(TEST.TYPE_EXPECTED,
                                    np.promote_types(instance.dtype,
                                                     minimalType)))
    maxEps=getOption(TEST.TOL_MINEPS, 0.)
    tolPower=getOption(TEST.TOL_POWER, 1.)
    ignoreType=not getOption(TEST.CHECK_DATATYPE, True)
    ignoreProximity=not getOption(TEST.CHECK_PROXIMITY, True)

    # check if shapes match. If they do, check all elements
    if arrOutput.shape != arrReference.shape:
        query[TEST.RESULT] = False
        query[TEST.RESULT_IGNORED] = False
        query[TEST.RESULT_INFO] = fmtRed(
            "%s!=%s" %(str(arrOutput.shape), str(arrReference.shape)))
        return query

    # if the query was not generated from input data, assume an int8-Eye-Matrix
    if TEST.RESULT_INPUT in query:
        arrInput=query[TEST.RESULT_INPUT]
        # check that input vector actually contains energy
        if np.linalg.norm(arrInput) == 0:
            print("Test query data causing exception:")
            pprint(query)
            raise ValueError("All-zero input vector detected in test")

        expectedType = np.promote_types(expectedType, arrInput.dtype)
        maxEps = max(maxEps, getTypeEps(arrInput.dtype))

    # compare returned output data type to expected (qType) to verify
    # functionality of fastmat built-in type promotion mechanism
    resultType = np.can_cast(arrOutput.dtype, expectedType, casting='no')

    maxEps = max(maxEps,
                 getTypeEps(np.promote_types(minimalType, instance.dtype)),
                 getTypeEps(np.promote_types(minimalType, arrReference.dtype)))

    # determine allowed tolerance maxima (allow accuracy degradation of chained
    # operations by representing multiple stages by a power on operation count
    # the test distribution function generates random arrays with their absolute
    # element values in the range [0.4 .. 0.8]. This can be described by a
    # `dynamics`-factor of 2 per computation stage (parameter TEST.TOL_POWER)
    dynamics = 2.
    maxDim = max(arrMatrix.shape + instance.shape)
    tolError = 5 * dynamics * maxEps * (dynamics * np.sqrt(maxDim)) ** tolPower
    query[TEST.RESULT_TOLERR] = tolError

    maxRef = float(np.amax(np.abs(arrReference)))
    maxDiff = float(np.amax(np.abs(arrOutput - arrReference)))
    error = (maxDiff / maxRef if maxRef != 0 else maxDiff)
    resultProximity=(error <= tolError) or (maxRef <= tolError)

    # determine final result
    query[TEST.RESULT] = (resultType and resultProximity)
    # result ignored: whenever the main result is not true but an ignore in one
    # of the tests would cause it to become true
    query[TEST.RESULT_IGNORED] = ((resultType or ignoreType) and
                                  (resultProximity or ignoreProximity) and
                                  not query[TEST.RESULT])
    query[TEST.RESULT_TYPE] = (resultType, ignoreType,
                               arrOutput.dtype, expectedType)
    query[TEST.RESULT_PROX] = (resultProximity, ignoreProximity, error, maxRef)

    return query


################################################## formatResult()
def formatResult(result):
    """Short summary.

    Parameters
    ----------
    result : type
        Description of parameter `result`.

    Returns
    -------
    type
        Description of returned object.

    """
    # if a result info was already generated, return this. Otherwise generate it
    if TEST.RESULT_INFO in result:
        return result[TEST.RESULT_INFO]

    # fetch detailed results from result
    resultType, ignoreType, outputType, expectedType=result[TEST.RESULT_TYPE]
    resultProx, ignoreProx, error, maxRef=result[TEST.RESULT_PROX]

    # format string output
    if resultType:
        strInfo=fmtGreen(NAME.TYPENAME[outputType])
    elif ignoreType:
        strInfo=fmtYellow(NAME.TYPENAME[outputType])
    else:
        strInfo=fmtRed("%s!=%s"% (NAME.TYPENAME[outputType],
                                  NAME.TYPENAME[expectedType]))

    if not resultProx:
        if error > 0 and maxRef > 0:
            try:
                strDiff = "%+2.2d" %(
                    int(min(99, np.round(np.log10(1. * error / maxRef)))))
            except OverflowError:
                strDiff = '---'
        else:
            strDiff = '000'

        strInfo=strInfo + (fmtYellow if ignoreProx else fmtRed)(strDiff)

    return strInfo


################################################## initTest()
def initTest(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    # generate test object instance for given parameter set
    test[TEST.INSTANCE] = test[TEST.OBJECT](
        *test.get(TEST.INITARGS, ()),
        **test.get(TEST.INITKWARGS, {}))

    if isinstance(test[TEST.INSTANCE], Matrix):
        # generate plain reference array
        test[TEST.REFERENCE] = test[TEST.INSTANCE].reference()


################################################## testFailDump()
def testFailDump(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    print("Test query causing the exception:")
    pprint(test)


################################################## testArrays()
def testArrays(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    query={TEST.RESULT_INPUT      : test[TEST.RESULT_INPUT],
           TEST.RESULT_OUTPUT     : test[TEST.RESULT_OUTPUT],
           TEST.RESULT_REF        : test[TEST.RESULT_REF]}
    return compareResults(test, query)


################################################## testForward()
def testForward(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    query={}
    arrInput=query[TEST.RESULT_INPUT]=test[TEST.DATAARRAY].forwardData
    arrInputCheck = arrInput.copy()
    query[TEST.RESULT_OUTPUT]=test[TEST.INSTANCE].forward(arrInput)
    if not np.array_equal(arrInput, arrInputCheck):
        testFailDump(test)
        raise ValueError(".forward() modified input array.")

    query[TEST.RESULT_REF]=test[TEST.REFERENCE].dot(arrInput)
    return compareResults(test, query)


################################################## testBackward()
def testBackward(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    query={}
    arrInput=query[TEST.RESULT_INPUT]=test[TEST.DATAARRAY].backwardData
    arrInputCheck = arrInput.copy()
    query[TEST.RESULT_OUTPUT]=test[TEST.INSTANCE].backward(arrInput)
    if not np.array_equal(arrInput, arrInputCheck):
        testFailDump(test)
        raise ValueError(".backward() modified input array.")

    query[TEST.RESULT_REF]=test[TEST.REFERENCE].T.conj().dot(arrInput)
    return compareResults(test, query)


################################################## testArray()
def testArray(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    query={}
    instance=test[TEST.INSTANCE]
    query[TEST.RESULT_OUTPUT]=instance.array
    query[TEST.RESULT_REF]=test[TEST.REFERENCE].astype(instance.dtype)
    return compareResults(test, query)


################################################## testInterface()
def testInterface(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    query={}
    instance=test[TEST.INSTANCE]
    query[TEST.RESULT_OUTPUT] = (
        instance - instance + (np.int8(-1) * instance * np.int8(-1)) +
        instance - instance
    ).array
    query[TEST.RESULT_REF]=test[TEST.REFERENCE].astype(instance.dtype)
    return compareResults(test, query)


################################################## testGetItem()
def testGetItem(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    query={}
    instance=test[TEST.INSTANCE]
    arrOutput=np.zeros(instance.shape, instance.dtype)
    for nn, mm in itertools.product(range(instance.numRows),
                                    range(instance.numCols)):
        arrOutput[nn, mm]=instance[nn, mm]
    query[TEST.RESULT_OUTPUT]=arrOutput
    query[TEST.RESULT_REF]=test[TEST.REFERENCE].astype(instance.dtype)
    return compareResults(test, query)


################################################## testGetColsSingle()
def testGetColsSingle(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    query = {}
    instance = test[TEST.INSTANCE]
    arrOutput = np.empty(instance.shape, instance.dtype)
    for mm in range(instance.numCols):
        vecCol = instance.getCols(mm)
        if vecCol.ndim != 1:
            print('testGetColsSingle', vecCol.shape, vecCol)
            query[TEST.RESULT], query[TEST.RESULT_IGNORED] = False, False
            query[TEST.RESULT_INFO] = fmtRed('!=(N,)')
            return query

        arrOutput[:, mm] = vecCol

    query[TEST.RESULT_OUTPUT] = arrOutput
    query[TEST.RESULT_REF] = test[TEST.REFERENCE].astype(instance.dtype)
    return compareResults(test, query)


################################################## testGetColsMultiple()
def testGetColsMultiple(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """

    query={}
    instance=test[TEST.INSTANCE]
    arrOutput=instance.getCols([c for c in range(instance.numCols)])
    query[TEST.RESULT_OUTPUT]=arrOutput
    query[TEST.RESULT_REF]=test[TEST.REFERENCE].astype(instance.dtype)
    return compareResults(test, query)


################################################## test: getRowsSingle
def testGetRowsSingle(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """

    query = {}
    instance = test[TEST.INSTANCE]
    arrOutput = np.empty((instance.numRows, instance.numCols), instance.dtype)
    for nn in range(instance.numRows):
        vecRow = instance.getRows(nn)
        import sys
        if vecRow.ndim != 1:
            print('testGetRowsSingle', vecRow.shape, vecRow)
            result, ignored = False, False
            query[TEST.RESULT], query[TEST.RESULT_IGNORED] = False, False
            query[TEST.RESULT_INFO] = fmtRed('!=(N,)')
            return query

        arrOutput[nn, :] = vecRow

    query[TEST.RESULT_OUTPUT] = arrOutput
    query[TEST.RESULT_REF] = test[TEST.REFERENCE].astype(instance.dtype)
    return compareResults(test, query)


################################################## test: getRowsMultiple
def testGetRowsMultiple(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    query={}
    instance=test[TEST.INSTANCE]
    arrOutput=instance.getRows([r for r in range(instance.numRows)])
    query[TEST.RESULT_OUTPUT]=arrOutput
    query[TEST.RESULT_REF]=test[TEST.REFERENCE].astype(instance.dtype)
    return compareResults(test, query)


################################################## test: column,rowNorms
###                                            ### test: column,rowNormalized
def _testNorms(test, funcTestcase, expectedType):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.
    funcTestcase : type
        Description of parameter `funcTestcase`.
    expectedType : type
        Description of parameter `expectedType`.

    Returns
    -------
    type
        Description of returned object.

    """
    instance, reference = test[TEST.INSTANCE], test[TEST.REFERENCE]

    # usually expect the normalized matrix to be promoted in type complexity
    # due to division by column-norm during the process. However there exist
    # matrices that treat the problem differently. Exclude the expected pro-
    # motion for them.
    query=({} if isinstance(instance, (Diag, Eye, Zero))
           else {TEST.TYPE_EXPECTED: expectedType})

    # ignore actual type of generated gram:
    query[TEST.CHECK_DATATYPE] = False
    query[TEST.TOL_MINEPS] = getTypeEps(safeTypeExpansion(instance.dtype))

    try:
        funcTestcase(query, instance, reference)
        return compareResults(test, query)
    except ValueError:
        if isinstance(instance, Zero):
            # failing normalization is expected for Zero matrix.
            result, ignored=True, False
        elif isinstance(instance, Parametric):
            # Parametric normalization is excused for now (int8 trouble)
            result, ignored=False, True
        else:
            result, ignored=False, False

        query[TEST.RESULT], query[TEST.RESULT_IGNORED] = result, ignored
        query[TEST.RESULT_INFO] = '!RNK'
        return query


def testColNorms(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    def columnTestcase(query, instance, reference):
        query[TEST.RESULT_OUTPUT] = instance.colNorms
        query[TEST.RESULT_REF] = np.apply_along_axis(
            np.linalg.norm, 0, reference
        )

    return _testNorms(test, columnTestcase, np.float64)


def testRowNorms(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    def rowTestcase(query, instance, reference):
        query[TEST.RESULT_OUTPUT] = instance.rowNorms
        query[TEST.RESULT_REF] = np.apply_along_axis(
            np.linalg.norm, 1, reference
        )

    return _testNorms(test, rowTestcase, np.float64)


def testColNormsColNormalized(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    def columnTestcase(query, instance, reference):
        query[TEST.RESULT_OUTPUT] = instance.colNormalized.array
        query[TEST.RESULT_REF] = np.einsum(
            'ij,j->ij',
            reference,
            1. / np.apply_along_axis(np.linalg.norm, 0, reference))

    return _testNorms(test, columnTestcase,
                      np.promote_types(test[TEST.INSTANCE].dtype, np.float64))


def testRowNormalized(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    def rowTestcase(query, instance, reference):
        query[TEST.RESULT_OUTPUT] = instance.rowNormalized.array
        query[TEST.RESULT_REF] = np.einsum(
            'i,ij->ij',
            1. / np.apply_along_axis(np.linalg.norm, 1, reference),
            reference)

    return _testNorms(test, rowTestcase,
                      np.promote_types(test[TEST.INSTANCE].dtype, np.float64))


################################################## test: largestSV (property)
def testLargestSV(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    query={TEST.TYPE_EXPECTED: np.float64}
    instance=test[TEST.INSTANCE]

    # account for "extra computation stage" (gram) in largestSingularValue
    query[TEST.TOL_POWER] = test.get(TEST.TOL_POWER, 1.) * 2
    query[TEST.TOL_MINEPS] = getTypeEps(safeTypeExpansion(instance.dtype))

    # determine reference result
    largestSingularValue = np.linalg.svd(
        test[TEST.REFERENCE],
        compute_uv=False
    )[0]

    query[TEST.RESULT_REF] = np.array(
        largestSingularValue,
        dtype=np.promote_types(largestSingularValue.dtype, np.float32)
    )

    # largestSingularValue may not converge fast enough
    # for a bad random starting point
    # so retry some times before throwing up
    for tries in range(9):
        maxSteps=100. * 10. ** (tries / 2.)
        query[TEST.RESULT_OUTPUT]=np.array(
            instance.getLargestSingularValue())
        result=compareResults(test, query)
        if result[TEST.RESULT]:
            break
    return result


################################################## test: gram (property)
def testGram(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    instance, reference=test[TEST.INSTANCE], test[TEST.REFERENCE]

    # usually expect the normalized matrix to be promoted in type complexity
    # due to division by column-norm during the process. However there exist
    # matrices that treat the problem differently. Exclude the expected pro-
    # motion for them.
    query=({} if isinstance(instance, (Diag, Eye, Zero))
           else {TEST.TYPE_PROMOTION: np.float32})

    # account for "extra computation stage" in gram
    query[TEST.TOL_POWER]=test.get(TEST.TOL_POWER, 1.) * 2

    query[TEST.RESULT_OUTPUT]=instance.gram.array
    query[TEST.RESULT_REF]=reference.astype(
        np.promote_types(np.float32, reference.dtype)).T.conj().dot(reference)

    # ignore actual type of generated gram:
    query[TEST.CHECK_DATATYPE]=False

    return compareResults(test, query)


################################################## test: T (property)
def testTranspose(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    query={}
    instance=test[TEST.INSTANCE]
    query[TEST.RESULT_OUTPUT]=instance.T.array
    query[TEST.RESULT_REF]=test[TEST.REFERENCE].T
    return compareResults(test, query)


################################################## test: H (property)
def testHermitian(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    query={}
    instance=test[TEST.INSTANCE]
    query[TEST.RESULT_OUTPUT]=instance.H.array
    query[TEST.RESULT_REF]=test[TEST.REFERENCE].T.conj()
    return compareResults(test, query)


################################################## test: conj (property)
def testConjugate(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    query={}
    instance=test[TEST.INSTANCE]
    query[TEST.RESULT_OUTPUT]=instance.conj.array
    query[TEST.RESULT_REF]=test[TEST.REFERENCE].conj()
    return compareResults(test, query)


################################################## test: Algorithm
def testAlgorithm(test):
    """Short summary.

    Parameters
    ----------
    test : type
        Description of parameter `test`.

    Returns
    -------
    type
        Description of returned object.

    """
    # generate input data vector. As this one is needed for some algs, do one
    # dereferentiation step on a complete dictionary, including test and result
    query=test.copy()
    arrInput, query[TEST.RESULT_INPUT]=query[TEST.DATAARRAY].forwardData
    paramDereferentiate(query)

    # now call the algorithm executor functions for FUT and reference
    arrInputCheck = arrInput.copy()
    query[TEST.RESULT_OUTPUT]=query[TEST.ALG](
        *query.get(TEST.ALG_ARGS, ()),
        **query.get(TEST.ALG_KWARGS, {}))
    query[TEST.RESULT_REF]=query[TEST.REFALG](
        *query.get(TEST.REFALG_ARGS,      query.get(TEST.ALG_ARGS, [])),
        **query.get(TEST.REFALG_KWARGS,   query.get(TEST.ALG_KWARGS, {})))

    if not np.array_equal(arrInput, arrInputCheck):
        testFailDump(test)
        raise ValueError("Algorithm modified input array.")

    return compareResults(test, query)


################################################################################
################################################## class TestRunner
class Test(Worker):
    """Short summary.

    Parameters
    ----------
    targetClass : type
        Description of parameter `targetClass`.
    **options : type
        Description of parameter `**options`.

    Attributes
    ----------
    _verboseFull : type
        Description of attribute `_verboseFull`.

    """

    _verboseFull=False

    def __init__(self, targetClass, **options):

        # extract options
        extraOptions = options.get('extraOptions',  {})

        # by default, enable verbosity for issues and isolate problems
        self.cbStatus=self.printStatus
        self.cbResult=self.findProblems

        # initialize output collectors
        self.problems = AccessDict({})
        self.irregularities = AccessDict({})

        # define test-specific defaults
        defaults={
            TEST.COMMON: {
                NAME.NAME       : dynFormat("%s.%s", NAME.CLASS, NAME.TARGET),
                TEST.NAMINGARGS : dynFormat(
                    "[%dx%d]", TEST.NUM_ROWS, TEST.NUM_COLS
                ),
                TEST.NAMING     : dynFormat(
                    "%s(%s)", NAME.CLASS, TEST.NAMINGARGS)

            }, TEST.CLASS: {
                # define tests
                TEST.QUERY: IgnoreDict({
                    'arr'   : testArray,
                    'iface' : testInterface,
                    'item'  : testGetItem,
                    'gCs'   : testGetColsSingle,
                    'gCm'   : testGetColsMultiple,
                    'gRs'   : testGetRowsSingle,
                    'gRm'   : testGetRowsMultiple,
                    'CnVec' : testColNorms,
                    'RnVec' : testRowNorms,
                    'CnMat' : testColNormsColNormalized,
                    'RnMat' : testRowNormalized,
                    'lSV'   : testLargestSV,
                    'gram'  : testGram,
                    'T'     : testTranspose,
                    'H'     : testHermitian,
                    'conj'  : testConjugate
                })

            }, TEST.TRANSFORMS: {
                # define default naming and captions
                TEST.NAMING     : dynFormat(
                    "%s(%s)*%s", NAME.CLASS, TEST.NAMINGARGS, TEST.DATAARRAY),

                # define default arrB ArrayGenerator
                TEST.DATACOLS   : Permutation([1, 5]),
                TEST.DATATYPE   : VariantPermutation(TEST.ALLTYPES),
                TEST.DATASHAPE  : (TEST.NUM_COLS, TEST.DATACOLS),
                TEST.DATASHAPE_T: (TEST.NUM_ROWS, TEST.DATACOLS),
                TEST.DATAALIGN  : VariantPermutation(TEST.ALLALIGNMENTS),
                TEST.DATACENTER : 0,
                TEST.DATAARRAY  : ArrayGenerator({
                    NAME.DTYPE      : TEST.DATATYPE,
                    NAME.SHAPE      : TEST.DATASHAPE,
                    NAME.SHAPE_T    : TEST.DATASHAPE_T,
                    NAME.ALIGN      : TEST.DATAALIGN,
                    NAME.CENTER     : TEST.DATACENTER
                }),

                # define tests
                TEST.QUERY      : IgnoreDict({
                    'F'     : testForward,
                    'B'     : testBackward,
                })

            }, TEST.ALGORITHM: {
                # define default naming and captions
                TEST.TOL_POWER  : 1.,
                TEST.NAMING     : dynFormat(
                    "%s(%s)*%s", NAME.CLASS, TEST.NAMINGARGS, TEST.DATAARRAY),

                # define default arrB ArrayGenerator
                TEST.DATACOLS   : Permutation([1, 5]),
                TEST.DATATYPE   : VariantPermutation(TEST.LARGETYPES),
                TEST.DATASHAPE  : (TEST.NUM_COLS, TEST.DATACOLS),
                TEST.DATASHAPE_T: (TEST.NUM_ROWS, TEST.DATACOLS),
                TEST.DATAALIGN  : VariantPermutation(TEST.ALLALIGNMENTS),
                TEST.DATACENTER : 0,
                TEST.DATAARRAY  : ArrayGenerator({
                    NAME.DTYPE      : TEST.DATATYPE,
                    NAME.SHAPE      : TEST.DATASHAPE,
                    NAME.SHAPE_T    : TEST.DATASHAPE_T,
                    NAME.ALIGN      : TEST.DATAALIGN,
                    NAME.CENTER     : TEST.DATACENTER
                }),

                # define tests
                TEST.QUERY: IgnoreDict({
                    'arr'     : testArrays
                })

            }
        }

        # call parent initialization with Test-specific options
        super(Test, self).__init__(
            targetClass, targetOptionMethod='_getTest',
            runnerDefaults=defaults, extraOptions=extraOptions)

    def _runTest(self, name, options):
        """Short summary.

        Parameters
        ----------
        name : type
            Description of parameter `name`.
        options : type
            Description of parameter `options`.

        Returns
        -------
        type
            Description of returned object.

        """

        # build list of tests as complete permutation of parameter variations
        tests=uniqueNameDict({})
        lstTests=paramPermute(options)
        for test in lstTests:
            # prepare test dictionary (dereferentiate links, evaluate functions)
            paramDereferentiate(test)
            paramEvaluate(test)

            # name each test instance from the key naming rule option ('naming')
            # 'naming' is a format string, which selects its fields from the
            # option dictionary made up by the target dictionary itself.
            tests[getattr(test, TEST.NAMING, test[NAME.TARGET])]=test

            # sanity-check proper definition of test target
            obj=test.get(TEST.OBJECT, None)
            if (obj is None or
                    not issubclass(obj, (Matrix, Algorithm))):
                raise ValueError("%s['%s'] not a fastmat class or test case." %(
                    test.name, TEST.OBJECT))

        # When done, convert tests back to regular dictionary, enabling editing
        tests=dict(tests)

        # allow a second level of Permutation for VariantPermutation instances
        # determine variants of target and generate variant description tag name
        lstVariantPermutations=[name for name, value in options.items()
                                if isinstance(value, VariantPermutation)]
        descrVariants=','.join(lstVariantPermutations)

        # determine field lengths for nice printing of columns
        lenName=max(map(len, tests.keys()))

        # self.results is constructed as multi-level dictionary:
        #     ~[target-name][test-name][variant-name][query-name]: results dict
        # initialize result structure for [test-name] level
        # create the [variant-level] dictionary as uniqueNameDict to prevent
        # overwriting existing variant results with the same name
        resultTarget={name: uniqueNameDict({}) for name in tests.keys()}

        # iterate through tests
        for nameTest, test in sorted(tests.items()):
            # initialize instances and required stuff for each test
            tryQuery(nameTest, initTest, test)

            # get a pointer to the test result dictionary
            resultTest=resultTarget[nameTest]

            lstVariants=paramPermute(
                test, PermutationClass=VariantPermutation)
            variants=[paramDereferentiate(variant) for variant in lstVariants]
            for variant in variants:
                variant[NAME.VARIANT]=''.join(
                    [NAME.TYPENAME[variant[key]]
                     if isinstance(variant[key], type)
                     else str(variant[key])
                     for key in lstVariantPermutations])

                # call variant initialization routine, if defined
                if TEST.INIT_VARIANT in variant:
                    variant[TEST.INIT_VARIANT](variant)

                # execute test queries, collect results as [query-name] level
                # and store in [variant-name] level into test result structure.
                resultTest[variant[NAME.VARIANT]]={
                    name: tryQuery(nameTest, query, variant)
                    for name, query in test[TEST.QUERY].items()
                }

            self.emitStatus(nameTest, resultTest, lenName, descrVariants)

        return resultTarget

    def _run(self, name, options):
        """Short summary.

        Parameters
        ----------
        name : type
            Description of parameter `name`.
        options : type
            Description of parameter `options`.

        Returns
        -------
        type
            Description of returned object.

        """

        maxTries = 3
        for numTry in range(maxTries):
            resultTarget = self._runTest(name, options.copy())

            result = all(all(all(resultQuery[TEST.RESULT] or
                                 resultQuery[TEST.RESULT_IGNORED]
                                 for resultQuery in resultVariant.values())
                             for resultVariant in resultTest.values())
                         for resultTest in resultTarget.values())

            if result:
                break
            else:
                print("Test %s.%s failed in during try #%d/%d.%s" %(
                    options[NAME.CLASS], name, numTry + 1, maxTries,
                    " Retrying ..." if numTry < maxTries - 1 else ""))

        return resultTarget

    def printStatus(self, nameTest, resultTest, lenName=-1, descrVariants=""):
        """Short summary.

        Parameters
        ----------
        nameTest : type
            Description of parameter `nameTest`.
        resultTest : type
            Description of parameter `resultTest`.
        lenName : type
            Description of parameter `lenName`.
        descrVariants : type
            Description of parameter `descrVariants`.

        Returns
        -------
        type
            Description of returned object.

        """
        # if full output isn't actually requested, crawl all query results in
        # the hierarchy of this test for negative results. If none are found,
        # skip printing.
        if not self._verboseFull:
            if all(all((query.get(TEST.RESULT, True) or
                        query.get(TEST.IGNORE, False))
                       for query in variant.values())
                   for variant in resultTest.values()):
                return

        lenNameV=max(len(nameV) for nameV in resultTest.keys())
        if lenName < 1:
            lenName=len(nameTest)

        # construct named prefix describing the test and its following variants
        strPrefix="%-*s " %(lenName + len(descrVariants),
                            nameTest + ":" + descrVariants)

        # convert results to info strings
        strQueries={nameV: ["%s:%s" %(nameQ, formatResult(query))
                            for nameQ, query in sorted(variant.items())]
                    for nameV, variant in resultTest.items()}

        # determine print-lengths of query strings
        lenQueries={nameV: [len(fmtEscape(strQuery)) for strQuery in variant]
                    for nameV, variant in strQueries.items()}

        # determine highest print-length for any query
        maxLenQuery=max(max(variant) for variant in lenQueries.values())

        # normalize string lengths in strQueries
        for nameV, lengths in lenQueries.items():
            strQueries[nameV]=["%s%s" %(strQueries[nameV][qq],
                                        " " * (maxLenQuery - lenQuery))
                               for qq, lenQuery in enumerate(lengths)]

        # join query strings to form variant strings
        strVariants=["%*s %s" %(lenNameV, nameV, " ".join(lstStr))
                     for nameV, lstStr in sorted(strQueries.items())]

        lenVariants=[len(fmtEscape(strV)) for strV in strVariants]
        maxLenVariants=max(lenVariants)

        # normalize string lengths in strVariants
        strVariants=["%s%s" %(strV, " " * (maxLenVariants - lenVariants[vv]))
                     for vv, strV in enumerate(strVariants)]
        lenVariants=(0 if len(strVariants) == 0
                     else len(fmtEscape(strVariants[0])))

        # calculate amount of entries printable per line
        width=self.consoleWidth
        itemsPerLine=(
            1 if width < 0
            else max(1, ((width - len(strPrefix)) // (1 + lenVariants))))

        for ii in range(0, len(strVariants), itemsPerLine):
            print("%s|%s" %(strPrefix if ii == 0 else " " * len(strPrefix),
                            " |".join(strVariants[ii: ii + itemsPerLine])))

    # control verbosity of _run() and printStatus()
    @property
    def verbosity(self):
        return self.cbStatus == self.printStatus, self._verboseFull

    @verbosity.setter
    def verbosity(self, value):
        if isinstance(value, bool):
            value=(value, )

        if isinstance(value, tuple):
            if len(value) >= 1:
                self.cbStatus=(self.printStatus if value[0] is True
                               else None)
            if len(value) >= 2:
                self._verboseFull=value[1]

    def findProblems(self, nameTarget, targetResult):
        """Short summary.

        Parameters
        ----------
        nameTarget : type
            Description of parameter `nameTarget`.
        targetResult : type
            Description of parameter `targetResult`.

        Returns
        -------
        type
            Description of returned object.

        """
        # crawl all query results for problems and irregularities. Collect them
        # if found. Also, construct a considerable name.
        problems={}
        irregularities={}
        for nameT, test in sorted(targetResult.items()):
            for nameV, variant in sorted(test.items()):
                for nameQ, query in sorted(variant.items()):
                    if not query.get(TEST.RESULT, False):
                        nameTestQuery="%s%s.%s" %(nameT, nameV, nameQ)
                        if query.get(TEST.RESULT_IGNORED, False):
                            irregularities[nameTestQuery]=query
                        else:
                            problems[nameTestQuery]=query

        self.problems[nameTarget] = convertToAccessDicts(problems)
        self.irregularities[nameTarget] = convertToAccessDicts(irregularities)

        return self.problems[nameTarget]
