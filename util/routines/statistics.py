# -*- coding: utf-8 -*-
'''
  fastmat/util/routines/statistics.py
 -------------------------------------------------- part of the fastmat package

  Routines for advanced statistical metrics.


  Author      : wcw
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
import sys
import os

import numpy as np
import scipy as sp

################################################################################
################################################## statistic evaluations


def weightedPercentile(
    data,
    percentile=([75, 25]),
    weights=None
):
    """
    O(nlgn) implementation for weighted_percentile, with linear interpolation
    between weights.

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
    percentile      = np.array(percentile) / 100.0
    if weights is None:
        weights     = np.ones(data.shape)

    dataIndSort     = np.argsort(data)
    dataSort        = data[dataIndSort]
    weights_sort    = weights[dataIndSort]
    ecdf            = np.cumsum(weights_sort)
    percentilePos   = percentile * (weights.sum() - 1) + 1

    # need the 1 offset at the end due to ecdf not starting at 0
    locations = np.searchsorted(ecdf, percentilePos)
    outPercentiles = np.zeros(percentilePos.shape)

    for i, empiricalLocation in enumerate(locations):
        # iterate across the requested outPercentiles
        if ecdf[empiricalLocation - 1] == np.floor(percentilePos[i]):
            # i.e. is the percentile in between two separate values
            uppWeight = percentilePos[i] - ecdf[empiricalLocation - 1]
            lowWeight = 1 - uppWeight

            outPercentiles[i] = \
                lowWeight * dataSort[empiricalLocation - 1] + \
                uppWeight * dataSort[empiricalLocation]
        else:
            # i.e. the percentile is entirely in one bin
            outPercentiles[i] = dataSort[empiricalLocation]

    return outPercentiles
