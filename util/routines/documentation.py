# -*- coding: utf-8 -*-
'''
  fastmat/util/routines/documentation.py
 -------------------------------------------------- part of the fastmat package

  Routines for generating documentation for units of fastmat.


  Author      : wcw
  Introduced  : 2017-01-10
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


################################################################################
##################################################  LaTeX output generators

def plotTimeWithQuartiles(caption, makro, *plots):

    output = r"""\tikzpicturedependsonfile{%(file)s}""" %(plots[0])
    output = output + "\n" + r"""%s{""" %(makro)

    legend = []
    for plot in plots:
        legend.append(plot['legend'])
#    \addplot table[col sep=comma,y=%(target)sP25]{%(file)s};
#    \addplot table[col sep=comma,y=%(target)sP75]{%(file)s};
        output = output + r"""
    \addplot table[col sep=comma,y=%(target)sMin]{%(file)s};""" %(plot)

    output = output + r"""
    \legend{%s}""" %(",".join(["%s" %(ll) for ll in legend]))

    return "%s\n}{%s}\n\n" %(output, caption)


def plotMemConsumption(caption, makro, *plots):

    output = r"""\tikzpicturedependsonfile{%(file)s}""" %(plots[0])
    output = output + "\n" + r"""%s{""" %(makro)

    legend = []
    for plot in plots:
        legend.append(plot['legend'])
        output = output + r"""
    \addplot table[col sep=comma,y=%(target)sMem]{%(file)s};""" %(plot)

    output = output + r"""
    \legend{%s}""" %(",".join(legend))

    return "%s\n}{%s}\n\n" %(output, caption)


def plotBar(caption, makro, *plots):

    output = r"""%s{""" % (makro)

    legend = []
    for plot in plots:
        legend.append(plot['legend'])
        output = output + r"""
    \addplot table[col sep=comma,y=%(target)sMem]{%(file)s};""" %(plot)

    output = output + r"""
    \legend{%s}""" %(",".join(legend))

    return "%s\n}{%s}\n" %(output, caption)


def docResultOutput(result):
    '''
    Generate LaTeX benchmark result output (plots) for the benchmark target
    result specified. Result is a dictionary describing one benchmark.
    '''
    # get names of benchmark and result file. If file does not exist, exit
    benchmark = result.get('benchmark', '')
    f = result.get('filename', '')
    if not os.path.isfile(f):
        return ""

    # define default plot setup for fastmat / numpy comparison
    fastmatNumpy = (
        {'file': f, 'target': 'fastmat', 'legend': 'fastmat'},
        {'file': f, 'target': 'numpy',   'legend': 'numpy'},
    )

    # begin compiling result string
    doc = ""

    if benchmark == "forward":
        # forward benchmark
        # add speeds plots first, then add memory usage plots
        doc += plotTimeWithQuartiles(
            'Forward projection', '\speed', *fastmatNumpy)
        doc += plotMemConsumption('Memory Usage', '\mem', *fastmatNumpy)
    elif benchmark == "solve":
        # solve benchmark
        doc += plotTimeWithQuartiles('Solving a LES', '\speed', *fastmatNumpy)
    elif "overhead" in f:
        # overhead benchmark
        doc += plotTimeWithQuartiles(
            'Call overhead', '\overhead',
            {'file': f, 'target': 'forward', 'legend': 'forward'},
            {'file': f, 'target': 'backward', 'legend': 'backward'})
    elif "performance" in f:
        # performance benchmark
        doc += plotTimeWithQuartiles(
            'Performance', '\speed',
            {'file': f, 'target': 'perf', 'legend': 'runtime'})
    else:
        # unknown benchmark: ignore
        return ""

    # add description (as defined in benchmark)
    descr = result.get('description', None)
    if descr:
        doc += "Tested Structure: %s\n" % (descr)

    return doc
