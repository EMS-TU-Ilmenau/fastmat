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

import inspect
import os
import sys
from datetime import datetime

from .common import *
from .benchmark import BENCH, Benchmark

modInspect = [sys.modules['fastmat.inspect.%s' %(name, )]
              for name in ['common', 'benchmark']]


################################################## DOC class
class DOC(NAME):
    CAPTION     = 'caption'
    TITLE       = 'title'
    XLABEL      = 'xLabel'
    YLABEL      = 'yLabel'
    DATASETS    = 'datasets'
    LEGEND      = 'legend'

    MAKRO       = 'makro'
    MAKRO_SPEED     = r'\speed'
    MAKRO_MEMORY    = r'\mem'
    MAKRO_OVERHEAD  = r'\overhead'
    MAKRO_TYPES     = r'\dtypes'

    OUTPATH     = 'outPath'
    FILENAME    = 'filename'
    TARGET      = 'target'
    COLUMN      = 'col'

    class __DOC__(object):
        def __init__(self, *args):
            self.items = args

        def __str__(self):
            return self.get()

        def __call__(self, **options):
            return self.get(**options)

        def get(self, **options):
            return os.linesep.join([''] + [
                ii.get(**options) if isinstance(ii, DOC.__DOC__)
                else str(ii).strip()
                for ii in self.items] + [''])

    class SECTION(__DOC__):
        def __init__(self, title, *contents):
            self.items = [r"\section{%s}" %(title)]
            self.items.extend(contents)

    class SUBSECTION(__DOC__):
        def __init__(self, title, *contents):
            self.items = [r"\subsection{%s}" %(title)]
            self.items.extend(contents)

    class SUBSUBSECTION(__DOC__):
        def __init__(self, title, *contents):
            self.items = [r"\subsubsection{%s}" %(title)]
            self.items.extend(contents)

    class BIBITEM(__DOC__):
        def __init__(self, author, title, source):
            self.items = [author, r"\emph{%s}" %(title), source]

    class BIBLIO(__DOC__):
        def __init__(self, **entries):
            self.items = [r"\begin{thebibliography}{9}"]
            for tag, entry in entries.items():
                self.items.extend([r"\bibitem{%s}" %(tag), entry])

            self.items.append(r"\end{thebibliography}")

    class SNIPPET(__DOC__):
        def __init__(self, *code, **opts):
            # output listing environment (with optional title)
            self.items = [
                r"\begin{snippet}%s" %("[%s]" %(opts[DOC.TITLE])
                                       if DOC.TITLE in opts else ""),
                r"\begin{lstlisting}[language=Python]"]

            # now add the contents
            self.items.extend(code if isinstance(code, (list, tuple))
                              else [code])

            # output caption if needed
            self.items.append(r"\end{lstlisting}")
            if DOC.CAPTION in opts:
                self.items.append(opts[DOC.CAPTION])

            # ... and finish the environment
            self.items.append(r"\end{snippet}")

    class PLOT(__DOC__):
        def __init__(self, benchName, **options):
            self.benchName = benchName
            self.params = options

        def get(self, **options):
            # merge options and parameters
            params = mergeDicts(self.params, options)
            params.setdefault(DOC.CAPTION, '')
            params.setdefault(DOC.TITLE, '')

            # check if data file actually exists
            benchmark = params.get(NAME.BENCHMARK, None)
            if benchmark is None or self.benchName not in benchmark.options:
                return ""

            outPath = params.get(DOC.OUTPATH, '')
            addVersionTag = params.get('addVersionTag', True)
            filename = benchmark.getFilename(self.benchName, path=outPath,
                                             addVersionTag=addVersionTag)
            params[DOC.FILENAME] = filename
            params[DOC.TARGET] = self.benchName

            # if result file exists, determine its age and compare it to the
            # age of the modules involved in the benchmark target.
            # This also involves all classes the benchmark target depends on
            if os.path.isfile(filename):
                # determine age of result file
                fileAge = os.path.getmtime(filename)

                # now determine a list of all modules containing code used by
                # the particular class and the quality management sections
                modules = [module
                           for module in inspect.getmro(
                               benchmark.target.__class__)
                           if module.__module__ != '__builtin__']
                modules.extend(modInspect)

                # now determine the newest all modules found as moduleage
                def getModuleAge(module):
                    try:
                        return os.path.getmtime(inspect.getfile(module))
                    except TypeError:
                        return 0

                moduleAge = max([getModuleAge(module) for module in modules])
            else:
                fileAge, moduleAge = -1, 0

            # the file is considered up-to-date if fileAge does exceed moduleAge
            # if necessary, rerun benchmark and update file
            if fileAge < moduleAge:
                if moduleAge == 0:
                    print("Output '%s' not found for target '%s'. Rebuild." %(
                        filename, self.benchName))
                else:
                    def strTime(timestamp):
                        return datetime.fromtimestamp(timestamp).strftime(
                            "%Y-%m-%d %H:%M:%S")

                    print(("Output '%s' outdated for target '%s'. " +
                           "(%s < %s). Rebuild.") %(filename, self.benchName,
                                                    strTime(fileAge),
                                                    strTime(moduleAge)))

                benchmark.run(self.benchName)
                benchmark.saveResult(self.benchName, outPath=outPath,
                                     addVersionTag=addVersionTag)

            # output header
            self.items = [r'\tikzpicturedependsonfile{%(filename)s}' %(params),
                          r'%(makro)s{' %(params)]

            # now do the actual plotting
            for dataset in params[DOC.DATASETS]:
                self.items.append(r"""
\addplot table[x=numN, col sep=comma,y=%s]{%s};""" %(dataset,
                                                     params[DOC.FILENAME]))

            # now output the legend
            self.items.append(r'\legend{%s};' %(",".join(params[DOC.LEGEND])))

            # output footer
            self.items.extend([r'}{%s}{%s}' %(params.get(DOC.TITLE),
                                              params.get(DOC.YLABEL))])

            return super(DOC.PLOT, self).get()

    ############################################## meta targets for plotting
    class PLOTSPEED(PLOT):
        def __init__(self, name, **options):
            super(DOC.PLOTSPEED, self).__init__(name, **mergeDicts(
                {DOC.DATASETS   : ['fastmatMin', 'numpyMin'],
                 DOC.LEGEND     : ['fastmat', 'numpy'],
                 DOC.MAKRO      : DOC.MAKRO_SPEED,
                 DOC.YLABEL     : 'Runtime [s]'}, options))

    class PLOTMEMORY(PLOT):
        def __init__(self, name, **options):
            super(DOC.PLOTMEMORY, self).__init__(name, **mergeDicts(
                {DOC.TITLE      : 'Memory consumption',
                 DOC.DATASETS   : ['fastmatMem', 'numpyMem'],
                 DOC.LEGEND     : ['fastmat', 'numpy'],
                 DOC.MAKRO      : DOC.MAKRO_MEMORY,
                 DOC.YLABEL     : 'Memory [kB]'}, options))

    class PLOTTYPES(PLOT):
        def __init__(self, name, **options):
            typeList = options.get(
                'typelist', ['c32', 'c64', 'f32', 'f64', 'i08', 'i32', 'i64'])
            col = options.get(DOC.COLUMN)
            super(DOC.PLOTTYPES, self).__init__(name, **mergeDicts(
                {DOC.DATASETS   : ["%s%s" %(str, col) for str in typeList],
                 DOC.LEGEND     : typeList,
                 DOC.MAKRO      : DOC.MAKRO_TYPES}, options))

    ############################################## plot targets using metas
    class PLOTFORWARD(PLOTSPEED):
        def __init__(self, **options):
            super(DOC.PLOTFORWARD, self).__init__(BENCH.FORWARD, **mergeDicts(
                {DOC.TITLE      : 'Forward Projection'}, options))

    class PLOTSOLVE(PLOTSPEED):
        def __init__(self, **options):
            super(DOC.PLOTSOLVE, self).__init__(BENCH.SOLVE, **mergeDicts(
                {DOC.TITLE      : 'Solving a LSE'}, options))

    class PLOTFORWARDMEMORY(PLOTMEMORY):
        def __init__(self, **options):
            super(DOC.PLOTFORWARDMEMORY, self).__init__(
                BENCH.FORWARD, **mergeDicts({DOC.TITLE: 'Class Memory usage'},
                                            options))

    class PLOTTYPESPEED(PLOTTYPES):
        def __init__(self, **options):
            super(DOC.PLOTTYPESPEED, self).__init__(BENCH.DTYPES, **mergeDicts(
                {DOC.TITLE      : 'Datatype Impact (Speed)',
                 DOC.COLUMN     : 'Min',
                 DOC.YLABEL     : 'Runtime [s]'}, options))

    class PLOTTYPEMEMORY(PLOTTYPES):
        def __init__(self, **options):
            super(DOC.PLOTTYPEMEMORY, self).__init__(
                BENCH.DTYPES,
                **mergeDicts({DOC.TITLE      : 'Datatype Impact (Memory)',
                              DOC.COLUMN     : 'Mem',
                              DOC.YLABEL     : 'Memory [kB]'}, options))

    ############################################## stand-alone plot targets
    class PLOTOVERHEAD(PLOTSPEED):
        def __init__(self, **options):
            super(DOC.PLOTOVERHEAD, self).__init__(BENCH.OVERHEAD, **mergeDicts(
                {DOC.TITLE      : 'Transform Runtime',
                 DOC.DATASETS   : ['forwardMin', 'backwardMin'],
                 DOC.LEGEND     : ['forward', 'backward']}, options))

    class PLOTPERFORMANCE(PLOTSPEED):
        def __init__(self, **options):
            super(DOC.PLOTPERFORMANCE, self).__init__(
                BENCH.PERFORMANCE, **mergeDicts({
                    DOC.TITLE      : 'Runtime Performance',
                    DOC.DATASETS   : ['perfMin'],
                    DOC.LEGEND     : ['runtime']}, options))


################################################################################
################################################## Documentation function
def Documentation(targetClass, **options):

    # the test target is a fastmat class to be instantiated in the runners
    if not inspect.isclass(targetClass):
        raise ValueError("target in init of Runner must be a class type")

    # fetch benchmark options from options
    benchmarkOptions = options.pop('benchmarkOptions', {})

    # generate a benchmark instance for access to classes benchmarks
    # also use this instance to access documentation options
    # if no options can be retrieved, exit.
    if DOC.BENCHMARK not in options:
        options[DOC.BENCHMARK] = Benchmark(targetClass,
                                           extraOptions=benchmarkOptions)

    benchmark = options[DOC.BENCHMARK]
    doc = getattr(benchmark.target, '_getDocumentation', lambda: "")()
    return (doc.get(**options) if isinstance(doc, DOC.__DOC__)
            else str(doc))
