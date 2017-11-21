# -*- coding: utf-8 -*-

# Copyright 2016 Sebastian Semper, Christoph Wagner
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

import os
import numpy as np
from timeit import timeit
import time

from .common import *
from ..core.cmath import profileCall
from ..core.resource import getMemoryFootprint
from ..Matrix import Matrix
from ..algs.CG import CG
from ..version import __version__


################################################## BENCH CONSTANT-def class
class BENCH(NAME):
    FORWARD         = 'forward'
    SOLVE           = 'solve'
    OVERHEAD        = 'overhead'
    DTYPES          = 'dtypes'
    PERFORMANCE     = 'performance'

    FUNC_INIT       = 'funcInit'
    FUNC_GEN        = 'funcConstr'
    FUNC_SIZE       = 'funcSize'
    FUNC_STEP       = 'funcStep'

    STEP_START_K    = 'startK'
    STEP_LOGSTEPS   = 'logSteps'
    STEP_MINITEMS   = 'minItems'

    LIMIT_MEMORY    = 'maxMem'
    LIMIT_INIT      = 'maxInit'
    LIMIT_ITER      = 'maxIter'
    LIMIT_SIZE      = 'maxSize'

    RESULT_MEMORY   = 'totalMem'
    RESULT_INIT     = 'initTime'
    RESULT_ITER     = 'iterTime'
    RESULT_K        = 'K'
    RESULT_SIZE     = 'numN'
    RESULT_OVH_NESTED_F = 'ovhFwd'
    RESULT_OVH_NESTED_B = 'ovhBwd'
    RESULT_EFF_NESTED_F = 'effFwd'
    RESULT_EFF_NESTED_B = 'effBwd'
    RESULT_COMPLEXITY_F = 'cpxFwd'
    RESULT_COMPLEXITY_B = 'cpxBwd'
    RESULT_ESTIMATE_FWD = 'estFwd'
    RESULT_ESTIMATE_BWD = 'estBwd'
    RESULT_ITEMS    = '#'

    MEAS_MINTIME    = 'measMinTime'
    MEAS_MINREPS    = 'measMinReps'
    MEAS_MINRUNS    = 'measMinRuns'
    MEAS_COMPACT    = 'compactStats'


################################################################################
##################################################  benchmark implementations

################################################## weightedPercentile()
def weightedPercentile(data, **options):
    """
    O(nlgn) implementation for weighted_percentile, with linear interpolation
    #between weights.

    date:       Aug 25 '16 @ 13:37
    user:       nayyrv [Aug 16 '15 @ 10:52]
    source:     www.stackoverflow.com/questions/
                    21844024/weighted-percentile-using-numpy
    ``  This is my function, it give identical behaviour to `np.percentile(
        np.repeat(data, weights), percentile)` With less memory overhead,
        np.percentile is an O(n) implementation so it's potentially faster for
        small weights. It has all the edge cases sorted out - it's an exace
        solution. The interpolation answers above assume linear, when it's a
        step for most of he case, except when the weight is 1.

        Say we have data [1,2,3] with weights [3, 11, 7] and I want the
        25% percentile. My ecdf is going to be [3, 10, 21] and I'm looking for
        the 5th value. The interpolation will see [3, 1] and [10, 2] as the
        matches and interpolate gving 1.28 despite being entirely in the
        2nd bin with a value of 2.
    ``

    """
    # extract options
    percentile = np.array(options.get('percentile', [75, 25])) / 100.
    weights = options.get('weights', None)

    if weights is None:
        weights=np.ones(data.shape)

    dataIndSort=np.argsort(data)
    dataSort=data[dataIndSort]
    weights_sort=weights[dataIndSort]
    ecdf=np.cumsum(weights_sort)
    percentilePos=percentile * (weights.sum() - 1) + 1

    # need the 1 offset at the end due to ecdf not starting at 0
    locations=np.searchsorted(ecdf, percentilePos)
    outPercentiles=np.zeros(percentilePos.shape)

    for i, empiricalLocation in enumerate(locations):
        # iterate across the requested outPercentiles
        if ecdf[empiricalLocation - 1] == np.floor(percentilePos[i]):
            # i.e. is the percentile in between two separate values
            uppWeight=percentilePos[i] - ecdf[empiricalLocation - 1]
            lowWeight=1 - uppWeight

            outPercentiles[i]=lowWeight * dataSort[empiricalLocation - 1] + \
                uppWeight * dataSort[empiricalLocation]
        else:
            # i.e. the percentile is entirely in one bin
            outPercentiles[i]=dataSort[empiricalLocation]

    return outPercentiles


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
    times=np.array([stat['avg'] for stat in stats], dtype=np.double)
    weights=np.array([stat['cnt'] for stat in stats], dtype=np.double)

    # calculate som stats
    reps=np.sum(weights)
    avg=np.average(times, weights=weights)
    minimum=np.min(times)

    # prepare output
    result={'min': minimum}

    # if compactStats is True, skip further stats
    if compactStats:
        return result

    # calculate weighted variance
    var=sum(weights * ((times - avg) ** 2)) / (reps * (1 - 1 / len(stats)))

    # calculate quartiles
    percs=[25, 75]
    wp=weightedPercentile(times, percentile=percs, weights=weights)

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
    stats=[]

    # begin with one repetition and approach accumulated runtime (step[1])
    num=1
    step={'avg': 0., 'time': 0., 'cnt': num}
    while step['time'] < (options[BENCH.MEAS_MINTIME] * num) / (num + 1) * 0.9:
        step=profileCall(num, call, *args)
        num=num * 2 if step['time'] < 1e-6 \
            else max(1, int(options[BENCH.MEAS_MINTIME] / step['avg']))

    # store as first run with [num] reps
    stats.append(step)

    # perform repeated measurement to reach at least MIN_ITERATIONS
    runs=max(int(np.ceil((options[BENCH.MEAS_MINREPS] - num) / num)),
             options[BENCH.MEAS_MINRUNS] - 1)
    if runs > 0:
        stats.extend([profileCall(num, call, *args) for ii in range(0, runs)])

    return timingStats(stats, options[BENCH.MEAS_COMPACT])


################################################## timeCalls()
def timeCalls(calls, **options):
    '''
    wrapper for running test runtime measurements for a number of calls

        target      - targeted resolution of measurement
        calls       - list of dics describing each call with these entries:
            > 'func'   (method pointer)
            > 'args'   (*list of arguments)
            > 'kwargs' (**dictionary of keyword-args)
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
    instance=funcConstr(numSize)
    mem1=instance.nbytes

    func1=CG
    func2=np.linalg.solve

    args1=[instance, arrTestDist((numN, 1), instance.dtype)]
    args2=[instance._forwardReference(np.eye(numN)), args1[1]]

    if instance._forwardReferenceMatrix is None:
        instance._forwardReferenceInit()

    mem2=instance.nbytesReference

    return {
        'fastmat': {
            'func': func1,
            'args': args1,
            'Mem': mem1
        }, 'fm10': {
            'func': func1,
            'args': [args1[0], np.resize(args1[1], (numN, 10))]
        },  'numpy': {
            'func': func2,
            'args': args2,
            'Mem': mem2
        }, 'np10': {
            'func': func2,
            'args': [args1[0], np.resize(args1[1], (numN, 10))]
        }
    }


################################################## testInitForward()
def testInitForward(funcConstr, numSize, numN):
    instance=funcConstr(numSize)
    mem1=instance.nbytes

    func1=instance.forward
    func2=instance._forwardReference

    args=[arrTestDist((numN, 1), instance.dtype)]
    mem2=instance.nbytesReference

    return {
        'fastmat': {
            'func': func1,
            'args': args,
            'Mem': mem1
        }, 'fm10': {
            'func': func1,
            'args': [np.resize(args[0], (numN, 10))]
        }, 'fm100': {
            'func': func1,
            'args': [np.resize(args[0], (numN, 100))]
        }, 'numpy': {
            'func': func2,
            'args': args,
            'Mem': mem2
        }, 'np10': {
            'func': func2,
            'args': [np.resize(args[0], (numN, 10))]
        }
    }


################################################## testInitOverhead()
def testInitOverhead(funcConstr, numSize, numN):
    instance=funcConstr(numSize)
    mem=instance.nbytes

    funcF=instance.forward
    funcB=instance.backward

    M = 1

    args=[arrTestDist((numN, M), instance.dtype)]

    profFwd, profBwd = instance.profileForward, instance.profileBackward
    estFwd, estBwd = instance.estimateRuntime(M)

    return {
        'forward': {
            'func': funcF,
            'args': args,
            'Mem': mem,
            BENCH.RESULT_OVH_NESTED_F   : profFwd['overheadNested'],
            BENCH.RESULT_EFF_NESTED_F   : profFwd['effortNested'],
            BENCH.RESULT_COMPLEXITY_F   : profFwd['complexity'],
            BENCH.RESULT_ESTIMATE_FWD   : estFwd
        }, 'backward': {
            'func': funcB,
            'args': args,
            'Mem': mem,
            BENCH.RESULT_OVH_NESTED_B   : profBwd['overheadNested'],
            BENCH.RESULT_EFF_NESTED_B   : profBwd['effortNested'],
            BENCH.RESULT_COMPLEXITY_B   : profBwd['complexity'],
            BENCH.RESULT_ESTIMATE_BWD   : estBwd
        }
    }


################################################## testInitDtype()
def testInitDtype(funcConstr, numSize, numN):
    def generate(dtype):
        instance=funcConstr(numSize, dtype)

        # if no instance was created, tag this target as non-existant
        if instance is None:
            return None

        # either a Matrix instance or a call tuple structure is returned
        if isinstance(instance, Matrix):
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
    result={name: generate(dtype)
            for dtype, name in NAME.TYPENAME.items() if dtype is not None}

    # prune None entries
    result={key: val for key, val in result.items() if (val is not None)}
    return result


################################################## testInitPerformance()
def testInitPerformance(funcConstr, numSize, numN):
    func, args=funcConstr(numSize)
    mem=args[0].nbytes

    return {
        'perf': {
            'func': func,
            'args': args,
            'Mem': mem
        }, 'perf10': {
            'func': func,
            'args': [args[0], np.resize(args[1],
                                        (args[1].shape[0], 10))] + args[2:]
        }
    }


################################################################################
################################################## class BenchmarkRunner
class Benchmark(Worker):

    def __init__(self, targetClass, **options):

        # extract options
        extraOptions = options.get('extraOptions', {})

        # by default, enable verbosity for issues and isolate problems
        self.cbStatus=self.printStatus

        # define defaults
        defaults={
            BENCH.COMMON: {
                # default naming of benchmark target
                BENCH.NAME          : dynFormat(
                    "%s.%s", NAME.CLASS, NAME.TARGET),
                BENCH.CAPTION       : dynFormat("%s", BENCH.NAME),

                # default benchmark abort criteria
                BENCH.LIMIT_MEMORY  : 1e6,  # kilobytes
                BENCH.LIMIT_INIT    : 1.,   # seconds
                BENCH.LIMIT_ITER    : 1.,  # seconds for all targets combined
                BENCH.LIMIT_SIZE    : 1e9,  # problem size

                # default benchmark parameters
                BENCH.STEP_START_K  : 1,
                BENCH.STEP_MINITEMS : 5,

                # default benchmark control functionals
                BENCH.FUNC_INIT     : None,
                BENCH.FUNC_GEN      : None,
                BENCH.FUNC_SIZE     : (lambda k: k),
                BENCH.FUNC_STEP     : (lambda k: k * 10 ** (1. / 12)),

                # default benchmark timing parameters
                BENCH.MEAS_MINTIME  : 0.003,
                BENCH.MEAS_MINREPS  : 7,
                BENCH.MEAS_MINRUNS  : 7,
                BENCH.MEAS_COMPACT  : True
            },
            BENCH.FORWARD: {
                BENCH.CAPTION       : dynFormat("%s (Multiplication)",
                                                BENCH.NAME),
                BENCH.FUNC_INIT     : testInitForward
            },
            BENCH.SOLVE: {
                BENCH.CAPTION       : dynFormat("%s (Solving a LSE)",
                                                BENCH.NAME),
                BENCH.FUNC_INIT     : testInitSolve
            },
            BENCH.OVERHEAD: {
                BENCH.CAPTION       : dynFormat("%s (call overhead)",
                                                BENCH.NAME),
                BENCH.FUNC_INIT     : testInitOverhead,
                BENCH.FUNC_SIZE     : (lambda k: 2 ** k),
                BENCH.FUNC_STEP     : (lambda k: k + 1)
            },
            BENCH.PERFORMANCE: {
                BENCH.CAPTION       : dynFormat("%s (Algorithm performance)",
                                                BENCH.NAME),
                BENCH.FUNC_INIT     : testInitPerformance
            },
            BENCH.DTYPES: {
                BENCH.CAPTION       : dynFormat("%s (impact of data type)",
                                                BENCH.NAME),
                BENCH.FUNC_INIT     : testInitDtype,
                BENCH.FUNC_SIZE     : (lambda k: 2 ** k),
                BENCH.FUNC_STEP     : (lambda k: k + 1)
            }
        }

        # call parent initialization with Test-specific options
        super(Benchmark, self).__init__(
            targetClass, targetOptionMethod='_getBenchmark',
            runnerDefaults=defaults, extraOptions=extraOptions)

    def getResult(self, nameResult, *nameParams):
        result=self.results.get(nameResult)
        header=result[BENCH.HEADER]
        indices=[header.index(key)
                 for key in (nameParams if len(nameParams) > 0 else header)]
        return result[BENCH.RESULT][:, indices]

    def _run(self, name, options):
        '''
        Perform a runtime performance evaluation.
        '''
        # output benchmark title
        self.emitStatus(fmtBold(">> %s" %(getattr(paramDict(options),
                                                  BENCH.CAPTION))))

        # perform the benchmarks
        run, result, header={}, {}, None
        result[BENCH.RESULT]=[]
        valK, numN=options[BENCH.STEP_START_K], 0
        numItems = 0
        while (run.get(BENCH.RESULT_MEMORY, 1) < options[BENCH.LIMIT_MEMORY] and
               run.get(BENCH.RESULT_INIT, 0.) < options[BENCH.LIMIT_INIT] and
               run.get(BENCH.RESULT_ITER, 0.) < options[BENCH.LIMIT_ITER] and
               run.get(BENCH.RESULT_SIZE, 1)  < options[BENCH.LIMIT_SIZE] or
               numItems < options[BENCH.STEP_MINITEMS]):
            # run test, init result dictionary for this run
            run.clear()

            # estimate next size
            valK, lastValK=options[BENCH.FUNC_STEP](valK), valK
            if valK == lastValK:
                raise ValueError(
                    "Stepping function caused endless loop: abort. [%s]" %(
                        options))
            numK=int(round(valK))
            numN, lastNumN=options[BENCH.FUNC_SIZE](numK), numN

            # if problem size did not change since last iter, skip
            if lastNumN == numN:
                #print("skipping %f (%d, %d)"%(valK, lastNumN, numN))
                continue
            run[BENCH.RESULT_SIZE] = numN

            # count number of elements in result to force minimal amount of
            numItems += 1

            ####################### construct test instance and init call list
            calls={}
            run[BENCH.RESULT_INIT]=timeit(
                lambda : calls.update(options[BENCH.FUNC_INIT](
                    options[BENCH.FUNC_GEN], numK, numN)),
                number=1)

            # determine total memory consumption of all call-related stuff and
            # matrix memory consumption of each call target separately
            totalMemUsed=getMemoryFootprint(calls) / 1e3
            for tag, call in calls.items():
                # memory consumption of each calls' matrix instance
                if 'Mem' in call:
                    memCall=call['Mem'] / 1e3
                    totalMemUsed += memCall
                    run["%sMem" %(tag)]=memCall

                # copy effort columns if defined in testInitCalls
                for key in [BENCH.RESULT_OVH_NESTED_F,
                            BENCH.RESULT_OVH_NESTED_B,
                            BENCH.RESULT_EFF_NESTED_F,
                            BENCH.RESULT_EFF_NESTED_B,
                            BENCH.RESULT_COMPLEXITY_F,
                            BENCH.RESULT_COMPLEXITY_B,
                            BENCH.RESULT_ESTIMATE_FWD,
                            BENCH.RESULT_ESTIMATE_BWD]:
                    if key in call:
                        run[key] = call[key]

            run[BENCH.RESULT_MEMORY]=totalMemUsed

            # do one extra warm-up run before starting actual measurements
            # to fetch and all required code and fetch data memory
            timeCalls(calls, **options)

            # determine runtime statistics and compose runs
            # alongside, check timing limits
#            run[BENCH.RESULT_ITER] = 0
            iterTime = time.time()
            run.update(timeCalls(calls, **options))
#            for tag in calls.keys():
#                run[BENCH.RESULT_ITER] += run["%sAvg" %(tag)]
            run[BENCH.RESULT_ITER] = time.time() - iterTime

            result[BENCH.RESULT].append(run.copy())

            # format instant output
            def formatToken(token):
                try:
                    # int or scientific notation
                    return ("%10d" if isinstance(token, int)
                            else "%10.3e") %(token)
                except TypeError:
                    # if that fails: make a string out of whatever it is
                    return "%11s" %(token)

            # print header in first iteration
            if BENCH.HEADER not in result:
                # determine
                keys=set([key
                          for step in result[BENCH.RESULT] for key in step])

                # let some key names have priority in ordering of header.
                priority=[BENCH.RESULT_SIZE, BENCH.RESULT_MEMORY,
                          BENCH.RESULT_INIT, BENCH.RESULT_ITER]

                # return the priority keys (in order of specification),
                # followed by the other keys (alphabetically sorted)
                result[BENCH.HEADER]=(
                    [key for key in priority if key in keys] +
                    [key for key in sorted(keys) if key not in priority])

                # emit header status line (--> printing)
                strHeader=" ".join([formatToken(f)
                                    for f in result[BENCH.HEADER]])
                self.emitStatus(strHeader + '\n' + '-' * len(strHeader))

            # output runs in same order as header
            row=['.'] * len(result[BENCH.HEADER])
            for key in run.keys():
                try:
                    row[result[BENCH.HEADER].index(key)]=run[key]
                except KeyError:
                    pass

            # print the whole stuff
            self.emitStatus("[%s]" %(", ".join([formatToken(f) for f in row])))

        # collate result dictionary structure to numpy array
        res, header=result[BENCH.RESULT], result[BENCH.HEADER]
        arrRes=np.zeros((len(res), len(header)), dtype=np.float64)
        for ii, item in enumerate(res):
            arrRes[ii, :]=np.array([item.get(key, np.NaN)
                                    for key in header])
        result[BENCH.RESULT]=arrRes

        return result

    # control verbosity of _run() and printStatus()
    @property
    def verbosity(self):
        return self.cbStatus == self.printStatus

    @verbosity.setter
    def verbosity(self, value):
        self.cbStatus=self.printStatus if value is True else None

    def printStatus(self, message):
        print(message)

    def getFilename(self, nameResult, path='', addVersionTag=True):
        return os.path.join(path, "%s.%s%s.csv" %(
            self.target.__class__.__name__, nameResult,
            ".%s" %(__version__) if addVersionTag else ""))

    def saveResult(self, nameResult, outPath='', addVersionTag=True,
                   strFormat='%.6e', chrDelimiter=',', chrNewline=os.linesep):
        '''
        Print data table to csv file. The result is organized as a list of
        dicts containing information in key=value pairs. The header will be
        extracted from the keys of the result data dictionaries. If the file
        does not exist it will be created.
        '''
        if nameResult not in self.results:
            raise ValueError("No benchmark result of such name found")

        result=self.results[nameResult]
        arrResult=result[BENCH.RESULT]
        lstHeader=result[BENCH.HEADER]

        filename=self.getFilename(nameResult, path=outPath,
                                  addVersionTag=addVersionTag)
        # force existance of output file
        dirname=os.path.dirname(filename)
        if dirname != '' and not os.path.exists(dirname):
            os.makedirs(dirname)

        # save csv-file to memory
        np.savetxt(filename, arrResult,
                   header=chrDelimiter.join(lstHeader), comments='',
                   fmt=strFormat, newline=chrNewline, delimiter=chrDelimiter)

        return filename

    def plotResult(self, nameResult, outPath='', addVersionTag=True,
                   strFormat='%.6e', chrDelimiter=',', chrNewline=os.linesep):
        '''
        Print data table to csv file. The result is organized as a list of
        dicts containing information in key=value pairs. The header will be
        extracted from the keys of the result data dictionaries. If the file
        does not exist it will be created.
        '''
        if nameResult not in self.results:
            raise ValueError("No benchmark result of such name found")

        result=self.results[nameResult]
        arrResult=result[BENCH.RESULT]
        lstHeader=result[BENCH.HEADER]

        filename=self.getFilename(nameResult, path=outPath,
                                  addVersionTag=addVersionTag)

        filename = filename.replace('csv', 'png')

        # force existance of output file
        dirname=os.path.dirname(filename)
        if dirname != '' and not os.path.exists(dirname):
            os.makedirs(dirname)

        import matplotlib.pyplot as plt

        plt.loglog(arrResult[:, 0], arrResult[:, 1:])

        plt.savefig(
            filename,
            dpi=300,
            transparent=True,
            bbox_inches='tight'
        )

        return filename
