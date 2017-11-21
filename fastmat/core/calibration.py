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

import numpy as np
from ..Matrix import MatrixCalibration

# stores calibration data as tuple (offsetFwd/Bwd, gainFwd/Bwd) over class
calData = {}


def saveCalibration(filename):
    import json
    import os
    outData = {target.__name__: cal.export() for target, cal in calData.items()}

    # force existance of output file
    dirname=os.path.dirname(filename)
    if dirname != '' and not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(filename, 'w') as f:
        string = json.dumps(outData)
        f.write(string.replace('],', '],' + os.linesep) + os.linesep)


def loadCalibration(filename):
    import json
    from .. import classes
    classNames = {item.__name__: item for item in classes}

    with open(filename, 'r') as f:
        inData = json.load(f)

    calData.clear()
    for name, data in inData.items():
        item = classNames.get(name, None)
        if item is not None:
            calData[item] = MatrixCalibration(*data)


def getMatrixCalibration(target):
    return calData.get(target, None)


def calibrateClass(target, **options):
    from fastmat.inspect import Benchmark, BENCH
    from .. import flags

    calOptions = {
        BENCH.LIMIT_ITER    : 0.1,
        BENCH.LIMIT_INIT    : 0.1,
        BENCH.LIMIT_SIZE    : 1e6,
        BENCH.LIMIT_MEMORY  : 1e5,
        BENCH.STEP_MINITEMS : 3,
        BENCH.MEAS_MINTIME  : 3e-3,
        BENCH.MEAS_MINRUNS  : 3,
        BENCH.MEAS_MINREPS  : 3,
        BENCH.FUNC_STEP     : lambda x: x + 1,
        'verbose'           : False
    }
    calOptions.update(options)

    # allow the function to skip estimation of calibration values and return
    # benchmark results that were generated with parameters matching the ones
    # that were used for the calibration
    benchmarkOnly = options.get('benchmarkOnly', False)
    if not benchmarkOnly:
        _bypassAllow = flags.bypassAllow
        flags.bypassAllow = False

    B = Benchmark(target)
    B.verbosity = calOptions.get('verbose', False)
    B.run('overhead', **calOptions)

    # transform bypass deactivated when calibrating
    if benchmarkOnly:
        return B
    else:
        flags.bypassAllow = _bypassAllow

    def estimate(nameTime, nameComplexity, nameOvhNested, nameEffNested):
        arr = np.nan_to_num(B.getResult('overhead', 'numN',
                                        nameTime, nameComplexity,
                                        nameOvhNested, nameEffNested))

        # remove runtimes of nested classes from the result
        arrTime             = arr[:, 1]
        arrComplexity       = arr[:, 2]
        arrOverheadNested   = arr[:, 3]
        arrEffortNested     = arr[:, 4]
        arrNested           = np.nan_to_num(arrOverheadNested + arrEffortNested)

        arrTime -= arrNested

        # keep entries with strictly positive runtimes and positive complexity
        arrIter         = arr[(arrTime > 0.) & (arrComplexity >= 0.), 1:3]
        arrTimeIter     = arrIter[:, 0]
        arrCplxIter     = arrIter[:, 1]

        offset, gain = 0., 0.
        if arrIter.shape[0] > 0:
            # estimate gain
            cpxRange = np.ptp(arrCplxIter)
            gain = (np.ptp(arrTimeIter) / cpxRange if cpxRange > 0. else 0.)

            # estimate offset iteratively
            for nn in range(20):
                delta = arrTimeIter - gain * arrCplxIter
                delta = delta[delta > -offset]
                if delta.shape[0] < 3:
                    break

                dOffset = (0 if delta.shape[0] < 1 else np.median(delta))
                offset += dOffset
                arrTimeIter -= dOffset

                if dOffset < offset * 1e-2:
                    break

        return offset, gain

    ovhForward, effForward = estimate('forwardMin',
                                      BENCH.RESULT_COMPLEXITY_F,
                                      BENCH.RESULT_OVH_NESTED_F,
                                      BENCH.RESULT_EFF_NESTED_F)

    ovhBackward, effBackward = estimate('backwardMin',
                                        BENCH.RESULT_COMPLEXITY_B,
                                        BENCH.RESULT_OVH_NESTED_B,
                                        BENCH.RESULT_EFF_NESTED_B)

    cal = MatrixCalibration(ovhForward, ovhBackward, effForward, effBackward)
    calData[target] = cal

    return cal, B
