# -*- coding: utf-8 -*-
'''
  fastmat/util/routines/test.py
 -------------------------------------------------- part of the fastmat package

  Routines for testing classes and algorithms.


  Author      : wcw
  Introduced  : 2017-01-05
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
import sys
import os
import inspect
import time
import timeit
import itertools
from collections import deque
import pprint

import numpy as np

# import fastmat, try as global package or locally from two floors up
try:
    import fastmat
except ImportError:
    sys.path.insert(0, os.path.join('..', '..'))
    import fastmat
from fastmat.helpers.unitInterface import *
from fastmat.helpers.types import _getTypeEps, safeTypeExpansion
from fastmat import Matrix

from .printing import *
from .parameter import *

################################################################################
##################################################  helper classes and functions

################################################## class uniqueNameDict


class uniqueNameDict(dict):
    '''
    Modified dictionary: suffixes key names with integer to maintain uniqueness.
    '''

    def __setitem__(self, key, value):
        index = 1
        strKey = key
        while True:
            if strKey not in self:
                break
            index += 1
            strKey = "%s_%03d" %(key, index)

        dict.__setitem__(self, strKey, value)


################################################## compareResults()
def _compareResults(test, query, instanceCache):

    arrOutput = query[TEST_RESULT_OUTPUT]
    arrReference = query[TEST_RESULT_REF]
    arrMatrix = test[TEST_REFERENCE]
    instance = test[TEST_INSTANCE]
    strInfo = ":"

    def getOption(option, default):
        return query.get(option, test.get(option, default))

    def appendResult(string, result, ignore, strFail, strSuccess=None):
        strSuccess = fmtGreen(strSuccess) if strSuccess is not None else ""
        return string + (strSuccess if result
                         else (fmtYellow if ignore else fmtRed)(strFail))

    # check if shapes match. If they do, check all elements
    if arrOutput.shape != arrReference.shape:
        return False, fmtRed("%s!=%s" %(
            str(arrOutput.shape), str(arrReference.shape))), False

    # if the query was not generated from input data, assume an int8-Eye-Matrix
    minimalType = getOption(TEST_TYPE_PROMOTION, np.int8)
    expectedType = getOption(TEST_TYPE_EXPECTED,
                             np.promote_types(instance.dtype, minimalType))

    maxEps = getOption(TEST_TOL_MINEPS, 0)

    if TEST_RESULT_INPUT in query:
        arrInput = query[TEST_RESULT_INPUT]
        expectedType = np.promote_types(expectedType, arrInput.dtype)
        maxEps = max(maxEps, _getTypeEps(arrInput.dtype))

    # compare returned output data type to expected (qType) to verify
    # functionality of fastmat built-in type promotion mechanism
    resultType = np.can_cast(arrOutput.dtype, expectedType, casting='no')
    ignoreType = not getOption(TEST_CHECK_DATATYPE, True)
    strInfo = appendResult(strInfo, resultType, ignoreType,
                           (NAME_TYPES[arrOutput.dtype] if ignoreType
                            else "%s!=%s"% (NAME_TYPES[arrOutput.dtype],
                                            NAME_TYPES[expectedType])),
                           NAME_TYPES[arrOutput.dtype])

    maxEps = max(maxEps,
                 _getTypeEps(np.promote_types(minimalType, instance.dtype)),
                 _getTypeEps(np.promote_types(minimalType, arrReference.dtype)))

    # determine allowed tolerance maxima (allow accuracy degradation of chained
    # operations by representing multiple stages by a power on operation count
    # the test distribution function generates random arrays with their absolute
    # element values in the range [0.4 .. 0.8]. This can be described by a
    # `dynamics`-factor of 2 per computation stage (parameter TEST_TOL_POWER)
    dynamics = 2
    tolError = 5 * dynamics * maxEps * (dynamics * np.sqrt(
        max(arrMatrix.shape + instance.shape))) ** getOption(TEST_TOL_POWER, 1.)
    query[TEST_RESULT_TOLERR]  = tolError

    maxRef = np.amax(np.abs(arrReference))
    maxDiff = np.amax(np.abs(arrOutput - arrReference))
    error = (maxDiff / maxRef if maxRef != 0 else maxDiff)
    resultProximity = (error <= tolError) or (maxRef <= tolError)
    ignoreProximity = not getOption(TEST_CHECK_PROXIMITY, True)
    if error > 0 and maxRef > 0:
        s = "%+2.2d" %(int(max(-99, np.round(np.log10(error / maxRef)))))
    else:
        s = 3 * ' '

    strInfo = appendResult(strInfo, resultProximity, ignoreProximity, s)

    # determine final result
    result  = (resultType and resultProximity)
    ignored = ((resultType or ignoreType) and
               (resultProximity or ignoreProximity))
    return result, strInfo, ignored


_instanceCompareCache = {}


def compareResults(test, query):
    # introduce a cache for matrix test instance cache parameters
    # reset instanceCompareCache if test instance differs
    global _instanceCompareCache
    testInstance = test.get(TEST_INSTANCE, None)
    if _instanceCompareCache.get(TEST_INSTANCE, None) != testInstance or \
            testInstance is None:
        _instanceCompareCache = {TEST_INSTANCE: testInstance}

    # run test
    query[TEST_RESULT], query[TEST_RESULT_INFO], query[TEST_RESULT_IGNORED] \
        = _compareResults(test, query, _instanceCompareCache)
    return query


################################################## initTest()
def initTest(test):
    # generate test object instance for given parameter set
    test[TEST_INSTANCE] = test[TEST_OBJECT](
        *test.get(TEST_INITARGS, ()),
        **test.get(TEST_INITKWARGS, {}))

    # generate plain reference array
    test[TEST_REFERENCE] = test[TEST_INSTANCE].reference()


################################################################################
##################################################  test implementations

################################################## testArrays()
def testArrays(test):
    query = {TEST_RESULT_INPUT      : test[TEST_RESULT_INPUT],
             TEST_RESULT_OUTPUT     : test[TEST_RESULT_OUTPUT],
             TEST_RESULT_REF        : test[TEST_RESULT_REF]}
    return compareResults(test, query)


################################################## testForward()
def testForward(test):
    query = {}
    arrInput = query[TEST_RESULT_INPUT] = test[TEST_DATAARRAY].forwardData
    query[TEST_RESULT_OUTPUT]  = test[TEST_INSTANCE].forward(arrInput)
    query[TEST_RESULT_REF]     = test[TEST_REFERENCE].dot(arrInput)
    return compareResults(test, query)


################################################## testBackward()
def testBackward(test):
    query = {}
    arrInput = query[TEST_RESULT_INPUT] = test[TEST_DATAARRAY].backwardData
    query[TEST_RESULT_OUTPUT]  = test[TEST_INSTANCE].backward(arrInput)
    query[TEST_RESULT_REF]     = test[TEST_REFERENCE].T.conj().dot(arrInput)
    return compareResults(test, query)


################################################## testToarray()
def testToarray(test):
    query = {}
    instance = test[TEST_INSTANCE]
    query[TEST_RESULT_OUTPUT]  = instance.toarray()
    query[TEST_RESULT_REF]     = test[TEST_REFERENCE].astype(instance.dtype)
    return compareResults(test, query)


################################################## testGetItem()
def testGetItem(test):
    query = {}
    instance = test[TEST_INSTANCE]
    arrOutput = np.zeros(instance.shape, instance.dtype)
    for nn, mm in itertools.product(range(instance.numN), range(instance.numM)):
        arrOutput[nn, mm] = instance[nn, mm]
    query[TEST_RESULT_OUTPUT]  = arrOutput
    query[TEST_RESULT_REF]     = test[TEST_REFERENCE].astype(instance.dtype)
    return compareResults(test, query)


################################################## testGetColsSingle()
def testGetColsSingle(test):
    query = {}
    instance = test[TEST_INSTANCE]
    arrOutput = np.zeros(instance.shape, instance.dtype)
    for mm in range(instance.numM):
        arrOutput[:, mm] = instance.getCols(mm)
    query[TEST_RESULT_OUTPUT]  = arrOutput
    query[TEST_RESULT_REF]     = test[TEST_REFERENCE].astype(instance.dtype)
    return compareResults(test, query)


################################################## testGetColsMultiple()
def testGetColsMultiple(test):
    query = {}
    instance = test[TEST_INSTANCE]
    arrOutput = instance.getCols([c for c in range(instance.numM)])
    query[TEST_RESULT_OUTPUT]  = arrOutput
    query[TEST_RESULT_REF]     = test[TEST_REFERENCE].astype(instance.dtype)
    return compareResults(test, query)


################################################## test: getRowsSingle
def testGetRowsSingle(test):
    query = {}
    instance = test[TEST_INSTANCE]
    arrOutput = np.zeros((instance.numN, instance.numM), instance.dtype)
    for nn in range(instance.numN):
        arrOutput[nn, :] = instance.getRows(nn)
    query[TEST_RESULT_OUTPUT]  = arrOutput
    query[TEST_RESULT_REF]     = test[TEST_REFERENCE].astype(instance.dtype)
    return compareResults(test, query)


################################################## test: getRowsMultiple
def testGetRowsMultiple(test):
    query = {}
    instance = test[TEST_INSTANCE]
    arrOutput = instance.getRows([r for r in range(instance.numN)])
    query[TEST_RESULT_OUTPUT]  = arrOutput
    query[TEST_RESULT_REF]     = test[TEST_REFERENCE].astype(instance.dtype)
    return compareResults(test, query)


################################################## test: normalized (property)
def testNormalized(test):
    instance, reference = test[TEST_INSTANCE], test[TEST_REFERENCE]

    # usually expect the normalized matrix to be promoted in type complexity
    # due to division by column-norm during the process. However there exist
    # matrices that treat the problem differently. Exclude the expected pro-
    # motion for them.
    query = ({} if isinstance(instance,
                              (fastmat.Diag, fastmat.Eye, fastmat.Zero))
             else {TEST_TYPE_PROMOTION: np.float32})

    # ignore actual type of generated gram:
    query[TEST_CHECK_DATATYPE] = False
    query[TEST_TOL_MINEPS] = _getTypeEps(safeTypeExpansion(instance.dtype))

    try:
        query[TEST_RESULT_OUTPUT]  = instance.normalized.toarray()
        query[TEST_RESULT_REF]     = np.einsum(
            'ij,j->ij', reference,
            1. / np.apply_along_axis(np.linalg.norm, 0, reference))
        return compareResults(test, query)
    except ValueError:
        if isinstance(instance, fastmat.Zero):
            # failing normalization is expected for Zero matrix.
            result, ignored = True, False
        elif isinstance(instance, fastmat.Parametrc):
            # Parametric normalization is excused for now (int8 trouble)
            result, ignored = False, True
        else:
            result, ignored = False, False

        query[TEST_RESULT], query[TEST_RESULT_IGNORED] = result, ignored
        query[TEST_RESULT_INFO] = '!RNK'
        return query


################################################## test: largestSV (property)
def testLargestSV(test):
    query = {TEST_TYPE_EXPECTED: np.float64}
    instance = test[TEST_INSTANCE]

    # account for "extra computation stage" (gram) in largestSV
    query[TEST_TOL_POWER] = test.get(TEST_TOL_POWER, 1.) * 2
    query[TEST_TOL_MINEPS] = _getTypeEps(safeTypeExpansion(instance.dtype))

    # determine reference result
    largestSV = np.linalg.svd(test[TEST_REFERENCE], compute_uv=False)[0]
    query[TEST_RESULT_REF]     = np.array(
        largestSV, dtype=np.promote_types(largestSV.dtype, np.float64))

    # largestSV may not converge fast enough for a bad random starting point
    # so retry some times before throwing up
    for tries in range(9):
        maxSteps = 100. * 10. ** (tries / 2.)
        query[TEST_RESULT_OUTPUT] = np.array(
            instance.getLargestSV(maxSteps=maxSteps, alwaysReturn=True))
        result = compareResults(test, query)
        if result[TEST_RESULT]:
            break
    return result


################################################## test: gram (property)
def testGram(test):
    instance, reference = test[TEST_INSTANCE], test[TEST_REFERENCE]

    # usually expect the normalized matrix to be promoted in type complexity
    # due to division by column-norm during the process. However there exist
    # matrices that treat the problem differently. Exclude the expected pro-
    # motion for them.
    query = ({} if isinstance(instance,
                              (fastmat.Diag, fastmat.Eye, fastmat.Zero))
             else {TEST_TYPE_PROMOTION: np.float32})

    # account for "extra computation stage" in gram
    query[TEST_TOL_POWER] = test.get(TEST_TOL_POWER, 1.) * 2

    query[TEST_RESULT_OUTPUT]  = instance.gram.toarray()
    query[TEST_RESULT_REF]     = reference.astype(
        np.promote_types(np.float32, reference.dtype)).T.conj().dot(reference)

    # ignore actual type of generated gram:
    query[TEST_CHECK_DATATYPE] = False

    return compareResults(test, query)


################################################## test: T (property)
def testTranspose(test):
    query = {}
    instance = test[TEST_INSTANCE]
    query[TEST_RESULT_OUTPUT]  = instance.T.toarray()
    query[TEST_RESULT_REF]     = test[TEST_REFERENCE].T
    return compareResults(test, query)


################################################## test: H (property)
def testHermitian(test):
    query = {}
    instance = test[TEST_INSTANCE]
    query[TEST_RESULT_OUTPUT]  = instance.H.toarray()
    query[TEST_RESULT_REF]     = test[TEST_REFERENCE].T.conj()
    return compareResults(test, query)


################################################## test: conj (property)
def testConjugate(test):
    query = {}
    instance = test[TEST_INSTANCE]
    query[TEST_RESULT_OUTPUT]  = instance.conj.toarray()
    query[TEST_RESULT_REF]     = test[TEST_REFERENCE].conj()
    return compareResults(test, query)


################################################## test: Algorithm
def testAlgorithm(test):
    # generate input data vector. As this one is needed for some algs, do one
    # dereferentiation step on a complete dictionary, including test and result
    query = test.copy()
    query[TEST_RESULT_INPUT] = query[TEST_DATAARRAY].forwardData
    paramDereferentiate(query)

    # now call the algorithm executor functions for FUT and reference
    query[TEST_RESULT_OUTPUT]  = query[TEST_ALG](
        *query.get(TEST_ALG_ARGS, ()),
        **query.get(TEST_ALG_KWARGS, {}))
    query[TEST_RESULT_REF]     = query[TEST_REFALG](
        *query.get(TEST_REFALG_ARGS,      query.get(TEST_ALG_ARGS, [])),
        **query.get(TEST_REFALG_KWARGS,   query.get(TEST_ALG_KWARGS, {})))

    return compareResults(test, query)


################################################## test template definition
testTemplates = {
    NAME_COMMON: {
        TEST_NAMINGARGS : dynFormatString("[%dx%d]", TEST_NUM_N, TEST_NUM_M),
        TEST_NAMING     : dynFormatString("%s(%s)", NAME_UNIT, TEST_NAMINGARGS),
        NAME_NAME       : dynFormatString("%s.%s", NAME_UNIT, NAME_TEST)

    }, TEST_CLASS: {
        # define tests
        TEST_QUERY: IgnoreDict({
            'ta'    : testToarray,
            'gi'    : testGetItem,
            'gCs'   : testGetColsSingle,
            'gCm'   : testGetColsMultiple,
            'gRs'   : testGetRowsSingle,
            'gRm'   : testGetRowsMultiple,
            'nor'   : testNormalized,
            'lSV'   : testLargestSV,
            'gr'    : testGram,
            'T'     : testTranspose,
            'H'     : testHermitian,
            'conj'  : testConjugate
        })

    }, TEST_TRANSFORMS: {
        # define default naming and captions
        TEST_NAMING     : dynFormatString(
            "%s(%s)*%s", NAME_UNIT, TEST_NAMINGARGS, TEST_DATAARRAY),

        # define default arrB ArrayGenerator
        TEST_DATACOLS   : Permutation([1, 5]),
        TEST_DATATYPE   : VariantPermutation(typesAll),
        TEST_DATASHAPE  : (TEST_NUM_M, TEST_DATACOLS),
        TEST_DATASHAPE_T: (TEST_NUM_N, TEST_DATACOLS),
        TEST_DATAALIGN  : VariantPermutation(alignmentsAll),
        TEST_DATACENTER : 0,
        TEST_DATAARRAY  : ArrayGenerator({
            NAME_DTYPE      : TEST_DATATYPE,
            NAME_SHAPE      : TEST_DATASHAPE,
            NAME_SHAPE_T    : TEST_DATASHAPE_T,
            NAME_ALIGN      : TEST_DATAALIGN,
            NAME_CENTER     : TEST_DATACENTER
        }),

        # define tests
        TEST_QUERY      : IgnoreDict({
            'F'     : testForward,
            'B'     : testBackward,
        })

    }, TEST_ALGORITHM: {
        # define default naming and captions
        TEST_TOL_POWER  : 1.,
        TEST_NAMING     : dynFormatString(
            "%s(%s)*%s", NAME_UNIT, TEST_NAMINGARGS, TEST_DATAARRAY),

        # define default arrB ArrayGenerator
        TEST_DATACOLS   : Permutation([1, 5]),
        TEST_DATATYPE   : VariantPermutation(typesAll),
        TEST_DATASHAPE  : (TEST_NUM_M, TEST_DATACOLS),
        TEST_DATASHAPE_T: (TEST_NUM_N, TEST_DATACOLS),
        TEST_DATAALIGN  : VariantPermutation(alignmentsAll),
        TEST_DATACENTER : 0,
        TEST_DATAARRAY  : ArrayGenerator({
            NAME_DTYPE      : TEST_DATATYPE,
            NAME_SHAPE      : TEST_DATASHAPE,
            NAME_SHAPE_T    : TEST_DATASHAPE_T,
            NAME_ALIGN      : TEST_DATAALIGN,
            NAME_CENTER     : TEST_DATACENTER
        }),

        # define tests
        TEST_QUERY: IgnoreDict({
            'arr'     : testArrays
        })

    }
}


################################################## runTest()
def runTest(target, *extraArgs, **options):
    '''
    Run test for the specified class and testparameter dictionary 'target'.
    '''
    # allow one test to appear on one line if nothing but header and stats will
    # be printed along the run
    nothingPrintedYet = [True]

    def localPrint(msg):
        if nothingPrintedYet[0]:
            print("")
            nothingPrintedYet[0] = False
        print(msg)

    timeTest = time.time()
    verbose = options.get('verbose', False)
    fullOutput = options.get('fullOutput', False)
    lenLine = getConsoleSize()[1]

    target = paramApplyDefaults(
        target, testTemplates, target[NAME_TEST], extraArgs)

    # print title of test
    printSection(target.name, newline=False)

    # build list of tests as complete permutation of parameter variations
    tests = uniqueNameDict({})
    lstTests = paramPermute(target)
    for test in lstTests:
        # prepare test dictionary (dereferentiate links, evaluate functions)
        paramDereferentiate(test)
        paramEvaluate(test)

        # name each test instance from the key naming rule option in 'naming'.
        # 'naming' is a format string, which selects its fields from the format
        # dict made up by the target dictionary itself.
        tests[getattr(test, TEST_NAMING, test[NAME_TEST])] = test

        # sanity-check proper definition of test target
        obj = test.get(TEST_OBJECT, None)
        if (obj is None or
                not (isinstance(obj, IgnoreFunc) or issubclass(obj, Matrix))):
            raise ValueError("%s['%s'] not a fastmat class or test case." %(
                test.name, TEST_OBJECT))

    # When done, convert tests back to regular dictionary, enabling editing
    tests = dict(tests)

    # allow a second level of Permutation for VariantPermutation instances
    # determine variants of target and generate variant description tag name
    lstVariantPermutations = [name for name, value in target.items()
                              if type(value) == VariantPermutation]
    nameVariants = ','.join(lstVariantPermutations)
    lenNameVariants = len(nameVariants)

    # state amount of different test cases created.
    if fullOutput:
        localPrint(" * generated %d test cases." %(len(tests)))

    # prepare formatting: set name column with to longest test name occurred
    lenName = max(map(len, tests.keys()))

    # iterate through tests
    testProblems = {}
    testCount = 0
    testIgnoredCount = 0
    for nameTest, test in sorted(tests.items()):
        testResult = True
        # initialize instances, data vectors and required stuff for each test
        initTest(test)

        lstVariants = paramPermute(test, PermutationClass=VariantPermutation)
        variants = [paramDereferentiate(variant) for variant in lstVariants]
        for variant in variants:
            variant[NAME_VARIANT] = ''.join(
                [NAME_TYPES[variant[key]] if isinstance(variant[key], type)
                 else str(variant[key])
                 for key in lstVariantPermutations])

        # iterate through variants of current test
        dictVariantResults = uniqueNameDict({})
        variantResult = True
        variantIgnoredCount = 0
        for variant in variants:
            # if variant defines individual initialization routine, call it now
            if TEST_INIT_VARIANT in variant:
                variant[TEST_INIT_VARIANT](variant)

            # check if this variant is flagged as "ignore"
            ignoreVar = variant.get(TEST_IGNORE, lambda param: False)(variant)

            # setup structure for collecting string results
            dictQueryResults = uniqueNameDict({})

            # execute test queries
            for nameQuery, query in sorted(test[TEST_QUERY].items()):

                if not query:
                    dictQueryResults[nameQuery] = fmtYellow(nameQuery)
                    variantIgnoredCount += 1
                    continue

                # valid test query, evaluate its result!
                testCount += 1
                result = query(variant)
                queryResult = result[TEST_RESULT]
                queryIgnored = ignoreVar or result[TEST_RESULT_IGNORED]

                # print yellow if failed test shall be ignored,
                if queryIgnored and not queryResult:
                    funColor = fmtYellow
                else:
                    variantResult &= queryResult
                    funColor = fmtGreen if queryResult else fmtRed

                # add query result to print stack
                dictQueryResults[nameQuery] = funColor("%s%s" %(
                    nameQuery, result[TEST_RESULT_INFO] if verbose else ""))

                # if the test failed, add to problem list, incl. test params
                if not queryResult:
                    nameTestQuery = "%s%s.%s" %(
                        nameTest, variant[NAME_VARIANT], nameQuery)
                    result.update(variant)
                    result[TEST_QUERY] = nameTestQuery
                    if queryIgnored:
                        variantIgnoredCount += 1
                    else:
                        testProblems[nameTestQuery] = result

            # collect query results to variant result if there are some results
            dictVariantResults[variant[NAME_VARIANT]] = dictQueryResults

        # finish line
        testIgnoredCount += variantIgnoredCount
        if not variantResult or variantIgnoredCount > 0 or fullOutput:
            # build list of query data to print
            items = {variant: [query for query in sorted(queries.values())]
                     for variant, queries in sorted(dictVariantResults.items())}
            # determine the common length of all query entries
            lenQuery = max([len(fmtEscape(query))
                            for queries in items.values() for query in queries])
            # match the length (printables only!) of every query in list
            for queries in items.values():
                for ii, query in enumerate(queries):
                    queries[ii] = "%s%s" %(
                        query, " " * max(0, lenQuery - len(fmtEscape(query))))
            # combine query entries to variant string tokens
            items = ["%s %s" %(variant, " ".join(queries))
                     for variant, queries in sorted(items.items())]
            lenVariant = max([len(fmtEscape(variant)) for variant in items]) + 2
            # determine number of variants to be printed side-by-side
            itemsPerLine = max(
                1,
                (lenLine - lenName - lenNameVariants - 1) // max(1, lenVariant))
            # print it! ... finally
            for ii in range(0, len(items), itemsPerLine):
                localPrint("%-*s|%s" %(
                    lenName + lenNameVariants + 1,
                    "%s:%s" %(nameTest, nameVariants) if ii == 0 else '',
                    "| ".join(items[ii:ii + itemsPerLine])
                ))

    timeTest = time.time() - timeTest
    print("   > performed %d tests in %.2f seconds (%s%s)" %(
        testCount, timeTest,
        (fmtGreen if len(testProblems) == 0 else fmtRed)
        ("%d problems" % (len(testProblems))),
        ("" if testIgnoredCount == 0
         else fmtYellow(", %d irregularities (ignored)" %(testIgnoredCount)))
    ))
    if not nothingPrintedYet[0]:
        print("")

    return testCount, testProblems, testIgnoredCount
