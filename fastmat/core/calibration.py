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

import numpy as np
from ..Matrix import MatrixCalibration

################################################################################
################################################## MatrixCalibration call names
CALL_FORWARD = 'forward'
CALL_BACKWARD = 'backward'


################################################################################
################################################## calibration storage
def saveCalibration(filename):
    """Save package calibration data in JSON format to file.

    The top level is a dictionary containing calibration data for each class,
    as a :py:class:`MatrixCalibration` object, and identified by the class
    object's basename as string. The :py:class:`MatrixCalibration` object --
    being a :py:class:`dict` itself -- will be represented transparently by
    JSON.

    Parameters
    ----------
    filename : str
        Filename to write the configuration data to.

    Returns
    -------
    None
    """
    import json
    import os
    outData = {target.__name__: cal for target, cal in calData.items()}

    # force existance of output file
    dirname=os.path.dirname(filename)
    if dirname != '' and not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(filename, 'w') as f:
        string = json.dumps(outData)
        f.write(string.replace('],', '],' + os.linesep) + os.linesep)


def loadCalibration(filename):
    """Short summary.

    Parameters
    ----------
    filename : type
        Description of parameter `filename`.

    Returns
    -------
    type
        Description of returned object.

    """
    import json
    from .. import classes
    classNames = {item.__name__: item for item in classes}

    with open(filename, 'r') as f:
        inData = json.load(f)

    calData.clear()
    for name, data in inData.items():

        # check the format of the data
        data = {
            key: tuple(value)
            for key, value in data.items()
        }

        # if all is sound, put the calibration data in the database
        item = classNames.get(name, None)
        if item is not None:
            calData[item] = MatrixCalibration(data)


################################################################################
################################################## calibration data access
# stores calibration data as MatrixCalibration over class
calData = {}


def getMatrixCalibration(target):
    """Return a :py:class:`MatrixCalibration` object with the calibration data
    for the fastmat baseclass target was instantiated from.

    Parameters
    ----------
    target : type
        Description of parameter `target`.

    Returns
    -------
    :py:class:`MatrixCalibration`
        If no calibration data exists, `None` will be returned.

    """
    return calData.get(target, None)


################################################################################
################################################## calibration routines
def calibrateClass(target, **options):
    """Calibrate a fastmat matrix baseclass using the specified benchmark.

    The generated calibration data will be cached in `calData` and is then
    available during instantiation of upcoming fastmat classes and can be
    imported/exported to disk using the routines `loadCalibration` and
    `saveCalibration`.

    Parameters
    ----------
    target : :py:class:`Matrix`
        The Matrix class to be calibrated. Any existing calibration data will
        be overwritten when the calibration succeeded.
    **options : dict
        Additional keyworded arguments specifying one of the following options:

    Options
    -------
    benchmarkOnly : bool
        If true, only perform the benchmark evaluation and do not generate
        calibration data (or update the corresponding entries in `calData`).
    verbose : bool
        Controls the `BENCH.verbosity` flag of the :py:class:`BENCH` instance,
        resulting in increased verbosity during the test.
    others : object
        Any other entries will be added to the option set used during the
        actual benchmarking run.

    Returns
    -------
    tuple (:py:class:`MatrixCalibration`, :py:class:`BENCH`)
        If the option `benchmarkOnly` is True, return the generated calibration
        data and the benchmark instance (containing all benchmark data
        collected) as a tuple

    :py:class:`BENCH`
        If the option `benchmarkOnly` is False, return the benchmark instance.

    By default the following default benchmark options for calibration runs
    will be chosen unless explicitly overwritten or extended by further entries
    in `options`:

    Default options
    ---------------
    maxIter : float = 0.1
        Abort iteration if evaluation of one problem takes more than 0.1s.
    maxInit : float = 0.1
        Abort iteration if preparation of one problem takes more than 0.1s.
    maxSize : float = 10
    00000
        Abort iteration if problem size of 1 million is exceeded
    maxMem : float = 100000
        Abort iteration if 100 MB memory usage is exceeded.
    minItems : int = 3
        Require the evaluation of at least three different problem sizes.
    measMinTime : float = 0.003
        Require the measurement interval to be at least 3ms. Increase
        repetition count of the evaluation of one problem size is faster than
        that.
    meas_minReps : int = 3
        Require at least three repetitions to be performed in one measurement
        interval
    meas_minReps : int = 3
        Require at least three independent measurements for one evaluation.
    funcStep : int callable(int)
        Provision to increase problem size after each evaluation as lamba
        function returning the next problem size, based on the current.
        Defaults to `lambda x: x + 1`

    """
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

        return (offset, gain)

    cal = MatrixCalibration({
        CALL_FORWARD: estimate(
            'forwardMin',
            BENCH.RESULT_COMPLEXITY_F,
            BENCH.RESULT_OVH_NESTED_F,
            BENCH.RESULT_EFF_NESTED_F
        ),
        CALL_BACKWARD: estimate(
            'backwardMin',
            BENCH.RESULT_COMPLEXITY_B,
            BENCH.RESULT_OVH_NESTED_B,
            BENCH.RESULT_EFF_NESTED_B
        )
    })

    calData[target] = cal

    return cal, B


def calibrateAll(**options):
    """Short summary.

    Parameters
    ----------
    **options : type
        Description of parameter `**options`.

    Returns
    -------
    type
        Description of returned object.

    """
    from .. import classes
    verbose = options.pop('verbose', False)
    for cc in classes:
        output = calibrateClass(cc, **options)
        if verbose:
            print("%s: %s" %(cc.__name__, str(output[0])))
