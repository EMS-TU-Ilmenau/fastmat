# -*- coding: utf-8 -*-
'''
  fastmat/util/routines/benchmark.py
 -------------------------------------------------- part of the fastmat package

  Routines for benchmarking function calls and package modules.


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
import time
import timeit
try:
    from reprlib import repr
except ImportError:
    pass

import numpy as np

# import fastmat, try as global package or locally from two floors up
try:
    import fastmat
except ImportError:
    sys.path.insert(0, os.path.join('..', '..'))
    import fastmat
from fastmat.helpers.unitInterface import *
from fastmat.helpers.resource import getMemoryFootprint
from fastmat.helpers.profile import *

# import routines
from .printing import *
from .statistics import weightedPercentile
from .parameter import *


################################################################################
##################################################  Call runtime evaluation

################################################## timingStats()
def timingStats(
    stats,
    compactStats=False
):
    '''
    Return the runtime statistics of the given list of measurements.
    Each measurement is stored as such a tuple:
    (averageRuntime, totalRuntime, numberRepetitions)

    A dictionary with clearly named statistical properties is returned.
    '''

    # accumulate all individual stats
    times   = np.array([stat['avg'] for stat in stats], dtype=np.double)
    weights = np.array([stat['cnt'] for stat in stats], dtype=np.double)

    # calculate som stats
    reps    = np.sum(weights)
    avg     = np.average(times, weights=weights)
    minimum = np.min(times)

    # prepare output
    result  = {'avg': avg}

    # if compactStats is True, skip further stats
    if compactStats:
        return result

    # calculate weighted variance
    var = sum(weights * ((times - avg) ** 2)) / (reps * (1 - 1 / len(stats)))

    # calculate quartiles
    percs   = [25, 75]
    wp      = weightedPercentile(times, percentile=percs, weights=weights)

    result.update({'var': var, 'cnt': reps, 'min': minimum})
    result.update({"p%d" %(percs[i]): wp[i] for i in range(len(percs))})

    return result


################################################## timeCall()
def timeCall(call, *args, **options):
    '''
    Return runtime statistics for execution of a parameterizable method call
    specified by call(*args).

    Behaviour of the measurements can be adjusted with these options:
      minMeasTime       - minimum total time the evaluation shall take
      minRepetitions    - minimum number of calls needed for evaluation
      minRuns           - minimum number of single measurements (of one or more
                          repetitions each) required
    '''
    stats = []

    # begin with one repetition and approach accumulated runtime (step[1]) to
    # MEAS_TIME by adjusting number of repetitions (num)
    MEAS_TIME = 0.1
    num = 1
    step = {'avg': 0., 'time': 0., 'cnt': num}
    while step['time'] < (options[BENCH_MEAS_MINTIME] * num) / (num + 1) * 0.9:
        step = timeReps(num, call, *args)
        num = num * 2 if step['time'] < 1e-6 \
            else max(1, int(options[BENCH_MEAS_MINTIME] / step['avg']))

    # store as first run with [num] reps
    stats.append(step)

    # perform repeated measurement to reach at least MIN_ITERATIONS
    runs = max(int(np.ceil((options[BENCH_MEAS_MINREPS] - num) / num)),
               options[BENCH_MEAS_MINRUNS] - 1)
    if runs > 0:
        stats.extend([timeReps(num, call, *args) for ii in range(0, runs)])

    return timingStats(stats, options[BENCH_MEAS_COMPACT])


################################################## timeCalls()
def timeCalls(calls, **options):
    '''
    wrapper for running test runtime measurements for a number of calls

        target      - targeted resolution of measurement
        calls       - list of dics describing each call with these entries:
            > 'func'   (method pointer)
            > 'args'   (*list of arguments)
            > 'kwargs' (**dictionary of keyword-args)
            > 'tag'    (descriptive tag which serves as prefix for
                        resulting statistics of each call)
        options     - dictionary with parameters for measurements

    A dictionary containing the measurement results of each call, preceded with
    its call tag is returned. All returned keys will be in `camelCase`.
    '''
    return {'%s%s' % (tag, key.title()): value
            for tag, call in calls.items()
            for key, value in (
                timeCall(call['func'], *(call['args']), **options)).items()}


################################################################################
##################################################  TESTS: specification

################################################## testInitSolve()
def testInitSolve(funcConstr, numSize, numN):
    instance = funcConstr(numSize)
    mem1 = instance.nbytes

    func1 = fastmat.algs.CG
    func2 = np.linalg.solve

    args1 = [instance, arrTestDist((numN, 1), instance.dtype)]
    args2 = [instance._forwardReference(np.eye(numN)), args1[1]]

    if instance._forwardReferenceMatrix is None:
        instance._forwardReferenceInit()

    mem2 = instance.nbytesReference

    return {
        'fastmat': {
            'func': func1,
            'args': args1,
            'Mem': mem1
        }, 'numpy': {
            'func': func2,
            'args': args2,
            'Mem': mem2
        }
    }


################################################## testInitForward()
def testInitForward(funcConstr, numSize, numN):
    instance = funcConstr(numSize)
    mem1 = instance.nbytes

    func1 = instance.forward
    func2 = instance._forwardReference

    args = [arrTestDist((numN, 1), instance.dtype)]
    mem2 = instance.nbytesReference

    return {
        'fastmat': {
            'func': func1,
            'args': args,
            'Mem': mem1
        }, 'numpy': {
            'func': func2,
            'args': args,
            'Mem': mem2
        }
    }


################################################## testInitOverhead()
def testInitOverhead(funcConstr, numSize, numN):
    instance = funcConstr(numSize)
    mem = instance.nbytes

    funcF = instance.forward
    funcB = instance.backward

    args = [arrTestDist((numN, 1), instance.dtype)]

    return {
        'forward': {
            'func': funcF,
            'args': args,
            'Mem': mem
        }, 'backward': {
            'func': funcB,
            'args': args,
            'Mem': mem
        }
    }


################################################## testInitDtype()
def testInitDtype(funcConstr, numSize, numN):
    def generate(dtype):
        instance = funcConstr(numSize, dtype)

        # if no instance was created, tag this target as non-existant
        if instance is None:
            return None

        # either a Matrix instance or a call tuple structure is returned
        if isinstance(instance, fastmat.Matrix):
            return {
                'func': instance.forward,
                'args': [arrTestDist((numN, 1), dtype)],
                'Mem': instance.nbytes
            }
        else:
            return {
                'func': instance[0],
                'args': instance[1],
                'Mem': instance[1][0].nbytes
            }

    # create instances for all datatypes listed
    result = {name: generate(dtype)
              for dtype, name in NAME_TYPES.items() if dtype is not None}

    # prune None entries
    result = {key: val for key, val in result.items() if (val is not None)}
    return result


################################################## testInitPerformance()
def testInitPerformance(funcConstr, numSize, numN):
    func, args = funcConstr(numSize)
    mem = args[0].nbytes

    return {
        'perf': {
            'func': func,
            'args': args,
            'Mem': mem
        }
    }


################################################################################
##################################################  I / O
def extractHeader(lstResults):
    entries = {}
    for rr in lstResults:
        entries.update(rr)

    # have some priority key names to begin header with. Use the list ordering!
    priority = [BENCH_RESULT_SIZE, BENCH_RESULT_MEMORY,
                BENCH_RESULT_INIT, BENCH_RESULT_ITER]

    # return the priority keys, followed by the other keys (also sorted)
    return [key for key in priority if key in entries.keys()] + \
        [key for key in sorted(entries.keys()) if key not in priority]


def saveTable(
    lstResults,
    strFilename,
    str_format='%.6e',
    chr_delimiter=',',
    chr_newline=os.linesep
):
    '''
    Print data table to csv file. The result is organized as a list of dicts
    containing information in key=value pairs. The header will be extracted
    from the keys of the result data dictionaries. If the file does not exist
    it will be created.
    '''

    # force existance of output file
    dirname = os.path.dirname(strFilename)
    print(dirname, strFilename)
    if dirname != '' and not os.path.exists(dirname):
        os.makedirs(dirname)

    # generate header information
    lstHeader = extractHeader(lstResults)

    # populate output data array
    arrData = np.zeros((len(lstResults), len(lstHeader)))
    for ii, result in enumerate(lstResults):
        for kk in result.keys():
            arrData[ii, lstHeader.index(kk)] = result[kk]

    # save csv-file to memory
    np.savetxt(
        strFilename, arrData,
        fmt=str_format,
        delimiter=chr_delimiter,
        header=chr_delimiter.join(lstHeader),
        newline=chr_newline,
        comments=''
    )


################################################################################
################################################## benchmark template definition
benchmarkTemplates = {
    NAME_COMMON: {
        # default naming of benchmark target
        NAME_NAME           : dynFormatString("%s.%s",
                                              NAME_UNIT, NAME_BENCHMARK),
        NAME_CAPTION        : dynFormatString("%s", NAME_NAME),

        # default benchmark abort criteria
        BENCH_LIMIT_MEMORY  : 1e6,          # kilobytes
        BENCH_LIMIT_INIT    : 5.,           # seconds
        BENCH_LIMIT_ITER    : 0.5,          # seconds for all targets combined
        BENCH_LIMIT_SIZE    : 1e9,          # problem size

        # default benchmark parameters
        BENCH_STEP_START_K  : 1,

        # default benchmark control functionals
        BENCH_FUNC_INIT : None,
        BENCH_FUNC_GEN  : None,
        BENCH_FUNC_SIZE : (lambda k: k),
        BENCH_FUNC_STEP : (lambda k: k * 10 ** (1. / 12)),

        # default benchmark timing parameters
        BENCH_MEAS_MINTIME  : 0.01,
        BENCH_MEAS_MINREPS  : 10,
        BENCH_MEAS_MINRUNS  : 5,
        BENCH_MEAS_COMPACT  : False
    },
    BENCH_FORWARD: {
        NAME_CAPTION    : dynFormatString("%s (Multiplication)", NAME_NAME),
        BENCH_FUNC_INIT : testInitForward
    },
    BENCH_SOLVE: {
        NAME_CAPTION    : dynFormatString("%s (Solving a LES)", NAME_NAME),
        BENCH_FUNC_INIT : testInitSolve
    },
    BENCH_OVERHEAD: {
        NAME_CAPTION    : dynFormatString("%s (call overhead)", NAME_NAME),
        BENCH_FUNC_INIT : testInitOverhead,
        BENCH_FUNC_SIZE : (lambda k: 2 ** k),
        BENCH_FUNC_STEP : (lambda k: k + 1)
    },
    BENCH_PERFORMANCE: {
        NAME_CAPTION    : dynFormatString(
            "%s (Algorithm performance)", NAME_NAME),
        BENCH_FUNC_INIT : testInitPerformance
    },
    BENCH_DTYPES: {
        NAME_CAPTION    : dynFormatString(
            "%s (impact of data type)", NAME_NAME),
        BENCH_FUNC_INIT : testInitDtype,
        BENCH_FUNC_SIZE : (lambda k: 2 ** k),
        BENCH_FUNC_STEP : (lambda k: k + 1),
        'compactStats'  : True
    }
}


################################################## runEvaluation()
def runEvaluation(*params, **options):
    '''
    Perform a runtime performance evaluation.
    '''
    param = paramApplyDefaults(
        options, benchmarkTemplates, options[NAME_BENCHMARK], params)

    printTitle(getattr(param, NAME_CAPTION))

    # perform the benchmarks
    result, lstResults, header = {}, [], None
    valK, numN = param[BENCH_STEP_START_K], 0
    while (result.get(BENCH_RESULT_MEMORY, 1) < param[BENCH_LIMIT_MEMORY] and
            result.get(BENCH_RESULT_INIT, 0.) < param[BENCH_LIMIT_INIT] and
            result.get(BENCH_RESULT_ITER, 0.) < param[BENCH_LIMIT_ITER] and
            result.get(BENCH_RESULT_SIZE, 1)  < param[BENCH_LIMIT_SIZE]):
        # run test, init results dictionary
        result.clear()

        # estimate next size
        valK, lastValK = param[BENCH_FUNC_STEP](valK), valK
        if valK == lastValK:
            raise ValueError(
                "Stepping function causes endless loop: abort. [%s]" %(param))
        numK = int(round(valK))
        numN, lastNumN = param[BENCH_FUNC_SIZE](numK), numN

        # if problem size did not change since last iter, skip
        if lastNumN == numN:
            #            print("skipping %f (%d, %d)"%(valK, lastNumN, numN))
            continue
        result[BENCH_RESULT_SIZE] = numN

        ####################### construct test instance and initialize call list
        calls = {}
        result[BENCH_RESULT_INIT] = timeit.timeit(
            lambda : calls.update(param[BENCH_FUNC_INIT](
                param[BENCH_FUNC_GEN], numK, numN)),
            number=1)

        # determine total memory consumption of all call-related stuff and
        # matrix memory consumption of each call target separately
        totalMemUsed = getMemoryFootprint(calls) / 1e3
        for tag in calls.keys():
            # memory consumption of each calls' matrix instance
            memCall = calls[tag]['Mem'] / 1e3
            totalMemUsed += memCall
            result["%sMem" %(tag)] = memCall

        result[BENCH_RESULT_MEMORY] = totalMemUsed

        # do one extra warm-up run before starting actual measurements
        # to fetch and all required code and fetch data memory
        timeCalls(calls, **param)

        # determine runtime statistics and compose results
        # alongside, check timing limits
        result[BENCH_RESULT_ITER] = 0
        result.update(timeCalls(calls, **param))
        for tag in calls.keys():
            result[BENCH_RESULT_ITER] += result["%sAvg" %(tag)]

        lstResults.append(result.copy())

        # format instant output
        def formatToken(token):
            try:
                # int or scientific notation
                return ("%10d" if isinstance(token, int) else "%10.3e") %(token)
            except TypeError:
                # if that fails: make a string out of whatever it is
                return "%11s" %(token)

        # print header in first iteration
        if header is None:
            header = extractHeader(lstResults)
            strHeader = " ".join([formatToken(f) for f in header])
            print(strHeader)
            print('-' * len(strHeader))

        # output results in same order as header
        lstCells = ['.'] * len(header)
        for key in result.keys():
            try:
                lstCells[header.index(key)] = result[key]
            except KeyError:
                pass

        # print the whole stuff
        print("[%s]" %(", ".join([formatToken(f) for f in lstCells])))

    # if option was submitted, directly save the output to file
    if NAME_FILENAME in param:
        saveTable(lstResults, param[NAME_FILENAME])

    return lstResults
