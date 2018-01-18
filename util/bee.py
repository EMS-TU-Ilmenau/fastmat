# -*- coding: utf-8 -*-

# Copyright 2016 Sebastian Semper, Christoph Wagner
# https://www.tu-ilmenau.de/it-ems/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Utility for shell interaction with class code from python packages.

Usecases:
 -  find all modules in a (sub-)package, which contain certain functions
    --check-only --function
 -  provide a string interface to module-level functions enabling shell
    piping to python package internals --argument --function
 - flexible options for character encoding --encode utf-8

Example:  retrieve LaTeX documentation of class Circulant, execute the
following one-liner from the doc/ directory of fastmat:

python doc_extract.py --path .. --package fastmat --name Circulant
--function docLaTeX --encode utf-8

"""

import sys
import inspect
import os
import argparse
from pprint import pprint
import time

import numpy as np


def importMatplotlib():
    '''Conditional import of matplotlib'''
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("matplotlib not found. Please consider installing it to proceed.")
        sys.exit(0)

    return plt


colors = ['#003366', '#FF6600', '#CC0000', '#FCE1E1']

################################################## import fastmat
try:
    import fastmat
except ImportError:
    sys.path.append('..')
    sys.path.append('.')
    import fastmat

from fastmat.inspect import Test, Benchmark, Documentation, TEST, BENCH, DOC
from fastmat.inspect.common import AccessDict, \
    fmtBold, fmtGreen, fmtYellow, fmtRed

classBaseContainers = (fastmat.Matrix, fastmat.Algorithm)

################################################## package constant definition
packageName = 'fastmat'
toolTitle = 'FastMat command line package interface tool (working Bee)'

EXT_BENCH_RESULT = 'csv'
EXT_TEST_RESULT = 'mat'

################################################################################
##################################################  subcommand argparser


def getObjects(
    obj,
    aSelector=(lambda obj, name: (name[0] != '_'))
):
    '''
    Return list of objects in obj whose properties satisfy aSelector.

    aClass may be a class type or a module. aSelector is a lambda function,
    filtering private objects whose names start with an underscore ('_').
    '''
    return sorted([item for item in list(obj.__dict__.keys())
                   if aSelector(obj, item)])


def unpackExtraArgs(extraArgs):
    result = {}
    if extraArgs is not None:
        for p in extraArgs:
            for pp in list(p.split(',')):
                tokens = pp.split('=')
                if len(tokens) >= 2:
                    string = "=".join(tokens[1:])
                    try:
                        val = int(string)
                    except ValueError:
                        try:
                            val = float(string)
                        except ValueError:
                            val = string
                    result[tokens[0]] = val
    return result


class CommandArgParser(object):
    _commandType = '<command>'

    def __init__(self, *arguments, **options):
        '''Detect commands and dispatch to specific parsers for each.'''

        # store passed (and already processed portion of command line)
        prevArgs = options.get('prevArgs', '')
        self._prevArgs = prevArgs

        # configure parser
        validCmds = '|'.join(getObjects(self.__class__))
        self.cmdParser = argparse.ArgumentParser(
            usage="%s %s [arguments]" % (prevArgs, self._commandType))

        # first-pass argument.parse_args defaults to [1:], selecting command
        self.cmdParser.add_argument(self._commandType, help=validCmds)
        argument = arguments[0] if len(arguments) > 0 else ''
        self._args = self.cmdParser.parse_args((argument,))

        # create argument parser for actual subcommands
        self.argParser = argparse.ArgumentParser(
            usage="%s %s [arguments]" % (prevArgs, argument))

        # try to call method with this name
        try:
            # dispatch all other arguments to subcommand-stage
            dispatch = getattr(self, argument)
        except AttributeError:
            print(os.linesep + toolTitle)
            self.cmdParser.print_help()
            exit(1)

        # for the next stage subcommands handling, pass current arguments too.
        if inspect.isclass(dispatch):
            dispatch(*(arguments[1:]), prevArgs=' '.join([prevArgs, argument]))
        else:
            dispatch(*(arguments[1:]))


################################################################################
################################################## command line argument parser
class Bee(CommandArgParser):

    ############################################## command: list
    class list(CommandArgParser):
        _commandType = '<option>'

        def _parseArgs(self, *arguments):
            '''Process further cmd-line arguments, store them in self.args.'''
            self.argParser.add_argument(
                '-e', '--extended',
                action='store_true',
                help='Show extended information'
            )
            self.args = self.argParser.parse_args(arguments)

        def _printDataset(self, dataset, newline='\n', separator=' '):
            '''List contents of dataset according to modifiers in self.args.'''
            if self.args.extended:
                pprint(dataset)
            else:
                # extract all key names of dataset and print the list
                print(separator.join(
                    sorted([loc for loc in dataset.keys()])) + newline)

        def _parseAndPrint(self, dataset, *arguments):
            '''Combine self._parseArgs and self._printDataset.'''
            self._parseArgs(*arguments)
            self._printDataset(dataset)

        def classes(self, *arguments):
            '''Return list of available classes in package.'''
            self._parseAndPrint(packageClasses, *arguments)

        def algs(self, *arguments):
            '''Return list of available algorithms in package.'''
            self._parseAndPrint(packageAlgs, *arguments)

        def index(self, *arguments):
            '''Return list of available units in package.'''
            self._parseAndPrint(packageIndexTree, *arguments)

        def makedump(self, *arguments):
            '''Dump all information for makefile processing.'''
            self._parseArgs()

            printSetup = {'newline': ';', 'separator': ':'}
            for dataset in [packageClasses, packageAlgs]:
                self._printDataset(dataset, **printSetup)

    ############################################## selection helper function
    def _select(self, nameType, arguments, lstElements):
        self.argParser.add_argument(
            '-s', '--select',
            nargs='+',
            default=[],
            help=("Allows customizing the set of %(name)s to be run by " +
                  "combining filtering with picking. Append an arbitrary " +
                  "number of selectors in the format 'unit.%(name)s'. " +
                  "Selectors specifying only one category ('unit.' or " +
                  "'.%(name)s') introduce mask filter capabilities, entries. " +
                  "If any mask filters are specified, all targets matching " +
                  "both the combined unit and the combined %(name)s masks " +
                  "will be added to the job. Any explicitly picked targets " +
                  "('unit.%(name)s') will be added to the job regardless of " +
                  "also specified mask filters. If no mask filters are " +
                  "specified, but some targets are picked explicitly, only " +
                  "the picked targets will be %(name)sed. If neither mask " +
                  "filters are specified nor targets are pciked, all " +
                  "available targets will be %(name)sed. Arguments with no " +
                  "separators ('.') will be ignored completely.") % {
                'name': nameType}
        )
        self.argParser.add_argument(
            '-o', '--options',
            nargs='+',
            default=[],
            help=("You may specify additional parameters for the %(name)ss " +
                  "by putting extra <NAME>=<VALUE> either with this option " +
                  "or where they do not interfere with other options. You " +
                  "may separate multiple parameters either with whitespaces " +
                  "or commas.") % {'name': nameType}
        )

        # parse arguments, all extra stuff goes to 'extraParam'
        self.args, self.extraArgs = self.argParser.parse_known_args(arguments)

        # add extraParams added with the -o option
        self.extraArgs = tuple(list(self.extraArgs) + self.args.options)

        # compile job list (--select argument), default: all
        # prepare a dictionary with {element:[]} structure to keep track of all
        # selected targets for each element.
        if len(self.args.select) == 0:
            # This defaults to 'select all targets of all elements'
            return {element: [] for element in lstElements}
        else:
            # default to no selected elements(every element selects one element
            # representing an illegal target -> nothing will be tested unless
            # another element is added
            dummyElement = ''
            selected = {element: set([dummyElement]) for element in lstElements}

            # prepare lists to collect parameters
            selectElement = set([])

            # classify options in lists
            for name in self.args.select:
                tokens = name.split('.')
                if len(tokens) < 2:
                    continue

                element, target = '.'.join(tokens[:-1]), tokens[-1]
                if tokens[0] == '':
                    # begins with '.' -> ".target" selection mask
                    # if specific targets were selected, add them to all
                    # elements resulting in full selection of target
                    for lstTargets in selected.values():
                        lstTargets.add(target)
                elif tokens[-1] == '':
                    # ends on '.' -> "element." selection mask
                    # collect these elements for now
                    selectElement.add(element)
                else:
                    # element.target specific selection mask
                    # only select this specific element:target pair
                    if element in lstElements:
                        selected[element].add(tokens[-1])

            # to select some elements completely the selection list of that
            # particular element only needs to become empty
            for element in selectElement:
                if element in lstElements:
                    selected[element].clear()

            # now remove all elements containing only the dummy element (these
            # were actually not selected at all)
            selected = {
                element: lstTargets
                for element, lstTargets in selected.items()
                if len(lstTargets) != 1 or dummyElement not in lstTargets}

            # as cleanup stage remove the empty-blocking '' elements from all
            # sets which contain one other parameter as well (> two elements)
            # also do convert bayk to list
            for element, lstElements in selected.items():
                if len(lstElements) > 1:
                    lstElements.discard(dummyElement)

                selected[element] = sorted(list(selected[element]))

            # return the selector as list
            return selected

    ############################################## command: test
    def test(self, *arguments):
        '''Run unit tests in package.'''
        # parse arguments and get selection from list
        self.argParser.add_argument(
            '-w', '--write-failed',
            action='store_true',
            help="If specified, stores failed test results to a .mat file " +
            "for each failed test instance, including parameters, input and " +
            "evaluation output"
        )
        self.argParser.add_argument(
            '-i', '--interact',
            action='store_true',
            help="Start interactive session for investigating the identified " +
            "problems"
        )
        self.argParser.add_argument(
            '-v', '--verbose',
            action='store_true',
            help="Increase verbosity of output."
        )
        self.argParser.add_argument(
            '-f', '--full',
            action='store_true',
            help="Produce full output."
        )
        self.argParser.add_argument(
            '-c', '--calibration',
            type=str,
            default='',
            help="Filename to read calibration data from. Loads calibration " +
            "data for fastmat classes which is then used by the package."
        )

        # stop time
        timeTotal = time.time()

        # determine the selection of classes-under-test and their tergets
        selection = self._select(TEST.TEST, arguments, packageIndex.keys())

        # sanity check
        if len(selection) < 1:
            self.argParser.exit(0, "test: job list is empty.\n")

        # generate test worker classes
        options = unpackExtraArgs(self.extraArgs)
        tests = {name: Test(packageIndex[name], extraOptions=options)
                 for name in selection.keys()}

        # if workers have no targets, discard them (evaluated during __init__)
        tests = {name: worker
                 for name, worker in tests.items()
                 if len(worker.options.keys()) > 0}

        # run tests
        cntTests, cntProblems, cntIrregularities = [0], [0], [0]
        for nameTest, test in sorted(tests.items()):
            # measure timing as lap timer in callback
            timeLap = [time.time()]

            def cbResult(name, targetResult):
                # runtime of last test
                timeSingle = time.time() - timeLap[0]

                # statistics and output
                test.findProblems(name, targetResult)
                numProb = len(test.problems[name])
                numIrr = len(test.irregularities[name])
                numTests = sum(sum(len(variant.values())
                                   for variant in test.values())
                               for test in targetResult.values())
                cntProblems[0] += numProb
                cntIrregularities[0] += numIrr
                cntTests[0] += numTests

                print((fmtBold(">> %s:") + " ran %d tests in %.2f seconds") %(
                    '.'.join([nameTest, name]), numTests, timeSingle
                ) + " (%s%s)" %(
                    (fmtGreen if numProb == 0 else fmtRed)
                    ("%d problems" % (numProb)),
                    ("" if numIrr == 0
                     else fmtYellow(
                         ", %d irregularities (ignored)" %(numIrr)))
                ))
                # refresh timer for next measurement
                timeLap[0] = time.time()

            # set callbacks, verbosity and start testing all selected targets
            test.verbosity = self.args.verbose, self.args.full
            test.cbResult = cbResult
            test.run(*(selection[nameTest]))

        # forget about all tests with no targets executed (the selection list
        # did not match the targets offered by the test) to exclude it from
        # the unit result reporting following
        tests = {name: test
                 for name, test in tests.items()
                 if len(test.results.keys()) > 0}

        # analyze dependencies of units involved in problems
        testClasses = {name: packageIndex[name]
                       for name in tests.keys()}
        # STAGE 1: detect structural dependencies (crawl inheritance model)
        # a dependency is a class of which the respective class is a subclass of
        # limit search to classes derived from one of the class containers
        testDependencies = {
            name: set(inspect.getmro(classType))
            for name, classType in testClasses.items()}

        # STAGE 2: add dependencies introduced in test case instances
        def crawlContent(item, targetSet):
            for nestedItem in item:
                targetSet.add(nestedItem.__class__)
                crawlContent(nestedItem, targetSet)

        for name, testWorker in tests.items():
            targetSet = testDependencies[name]
            for target in testWorker.results.values():
                for nameTest, test in target.items():
                    if len(test) > 0:
                        aVariant = list(test.values())[0]
                        if len(aVariant) > 0:
                            aQuery = list(aVariant.values())[0]
                            crawlContent(aQuery.get(TEST.INSTANCE, ()),
                                         targetSet)

        # filter out the containers themselves as they are obviousely the
        # "mothers of it all" and as each class is also a subclass to itself,
        # skip it as well to improve readability of output
        testDependencies = {
            name: [item
                   for item in dependencies
                   if (issubclass(item, classBaseContainers) and
                       item not in classBaseContainers and
                       item != testClasses[name])]
            for name, dependencies in testDependencies.items()}

        # split units into two main lists (unit has problems / has no problems)
        problemUnits = [name
                        for name, test in sorted(tests.items())
                        if sum(len(problems)
                               for problems in test.problems.values()) > 0]
        fineUnits = [name
                     for name in sorted(tests.keys())
                     if name not in problemUnits]

        # now divide the `has problems` list further based on dependencies
        dependentProblems = [name
                             for name in problemUnits
                             if len(testDependencies[name]) > 0]
        independentProblems = [name
                               for name in problemUnits
                               if name not in dependentProblems]

        # stop time, print results and show summary of dependencies
        timeTotal = time.time() - timeTotal
        cntTests = cntTests[0]
        cntProblems, cntIrregularities = (cntProblems[0], cntIrregularities[0])

        print("\nResults:")
        print(("   > found %d problem(s) in %d classes%s " +
               "(ran %d tests in %.2f seconds)") %(
            cntProblems, len(problemUnits),
            ("" if cntIrregularities == 0
             else " (%d issues ignored)" %(cntIrregularities)),
            cntTests, timeTotal))

        if len(fineUnits) > 0:
            print("   > Classes with no problems:")
            print("%s[%s]\n" % (
                " " * 8, ", ".join([fmtGreen(unit)
                                    for unit in fineUnits])))

        if len(independentProblems) > 0:
            print("   > Classes with independent problems (start fixing here):")
            print("%s[%s]\n" % (
                " " * 8, ", ".join([fmtRed(unit)
                                    for unit in sorted(independentProblems)])))

        if len(dependentProblems) > 0:
            nameLength = max(len(name) for name in dependentProblems)
            print("   > Classes with problems that depend on other classes")

            def fmtDummy(string):
                return string

            for name in dependentProblems:
                depList = [((fmtYellow if dep in problemUnits else fmtGreen)
                            if dep in tests.keys() else fmtDummy)(dep.__name__)
                           for dep in testDependencies[name]]
                print("%s%*s: [%s]" % (
                    " " * 8, nameLength, fmtRed(name),
                    ", ".join(sorted(depList))))
            print("\n")

        # make test dictionary easy accessible
        tests = AccessDict(tests)

        # interactive session, anyone?
        if self.args.interact and cntProblems > 0:
            # register signaling for graceful exit
            import atexit

            def quitGracefully():
                print("Leaving interactive testing session.")

            atexit.register(quitGracefully)

            # scope dictionary
            scope = {}

            # set numpy print options
            np.set_printoptions(linewidth=200, threshold=10, edgeitems=3,
                                formatter={'float_kind'     : '{: .3}'.format,
                                           'complex_kind'   : '{: .3e}'.format})

            # scope dictionary
            scope.update(globals())
            scope.update(locals())

            # start interactive session
            import code
            code.interact(r"  > Entering interactive testing session.",
                          None, scope)

        # if problems occurred, exit with non-zero errorcode
        if cntProblems > 0:
            sys.exit(1)

    ############################################## command: benchmark
    def benchmark(self, *arguments):
        '''Run performance evaluation of units in package.'''

        # parse arguments and get selection from list
        self.argParser.add_argument(
            '-p', '--path-results',
            type=str,
            default='.',
            help="Save the benchmark results to the path specified."
        )
        self.argParser.add_argument(
            '-c', '--calibration',
            type=str,
            default='',
            help="Filename to read calibration data from. Loads calibration " +
            "data for fastmat classes which is then used by the package."
        )
        self.argParser.add_argument(
            '-t', '--no-version-tag',
            action='store_true',
            help="Do not add version tag to csv benchmark output."
        )
        selection = self._select(
            BENCH.BENCHMARK, arguments, packageIndex.keys())

        # load calibration data
        if len(self.args.calibration) > 0:
            fastmat.core.loadCalibration(self.args.calibration)
            print("  > Loaded calibration data from %s." %(
                self.args.calibration))

        # sanity check
        if len(selection) < 1:
            self.argParser.error("benchmark: job list is empty.")

        # generate benchmark worker classes
        benchmarks = {
            name: Benchmark(packageIndex[name],
                            extraOptions=unpackExtraArgs(self.extraArgs))
            for name in selection.keys()}

        # run benchmarks
        for nameBench, bench in sorted(benchmarks.items()):
            def cbResult(name, targetResult):
                # output, anyone?
                if len(self.args.path_results) > 0:
                    print("   > saved data '%s' to '%s'" % (
                        name,
                        bench.saveResult(
                            name, outPath=self.args.path_results,
                            addVersionTag=not self.args.no_version_tag)))
                print("")

            # start tests of all targets of this element
            bench.cbResult = cbResult
            bench.run(*(selection[nameBench]))

    ############################################## command: documentation
    def documentation(self, *arguments):
        '''Extract built-in documentation from units in package.'''
        self.argParser.add_argument(
            'units',
            nargs='*',
            default=[],
            help="Specify the documentation sections to include in the TeX " +
            "output. If this list is empty, all sections in the package will " +
            "be considered. Any key=value pairs will be forwarded to " +
            "benchmark operations"
        )
        self.argParser.add_argument(
            '-p', '--path-results',
            type=str,
            default='.',
            help='Path to the benchmark results.'
        )
        self.argParser.add_argument(
            '-o', '--output',
            type=str,
            default='',
            help="Filename to write the output to. If not file is specified " +
            "the output will be directed to STDOUT."
        )
        self.argParser.add_argument(
            '-c', '--calibration',
            type=str,
            default='',
            help="Filename to read calibration data from. Loads calibration " +
            "data for fastmat classes which is then used by the package."
        )
        self.argParser.add_argument(
            '-t', '--no-version-tag',
            action='store_true',
            help="Do not add version tag to csv benchmark output."
        )

        # parse arguments, all extra stuff goes to 'extraParam'
        self.args, extraArgs = self.argParser.parse_known_args(arguments)
        self.args.units.extend(extraArgs)

        # load calibration data
        if len(self.args.calibration) > 0:
            fastmat.core.loadCalibration(self.args.calibration)
            print("  > Loaded calibration data from %s." %(
                self.args.calibration))

        # compile set of extra options for optional benchmarking
        benchmarkOptions = unpackExtraArgs([name
                                            for name in self.args.units
                                            if '=' in name])
        options = {'benchmarkOptions'   : benchmarkOptions,
                   'addVersionTag'      : not self.args.no_version_tag,
                   DOC.OUTPATH          : self.args.path_results}

        selection = [name for name in self.args.units if '=' not in name]

        # if selection is empty, take all sections of package
        if len(selection) == 0:
            selection = (sorted(packageClasses.keys()) +
                         sorted(packageAlgs.keys()))
        else:
            selection = [name for name in selection if name in packageIndex]

        if len(selection) < 1:
            self.argParser.error("documentation: job list is empty.")

        output = ["% THIS OUTPUT WAS GENERATED AUTOMATICALLY AND " +
                  "MAY BE OVERWRITTEN ANYTIME"]

        # add machine info ahead of documentation
        import platform
        machineInfo = (
            " ".join([" \\verb|%s|" %(token)
                      for token in platform.platform().split("-")]).strip("`"),
            " ".join([" \\verb|%s|" %(token)
                      for token in " ".join(
                          platform.uname()).split(" ")]).strip("'"),
            platform.processor(),
            '---'
        )
        output.append(DOC.SUBSECTION('General information', r"""
This section was automatically generated on a system with the following
specifications

\vspace{5mm}\begin{centering}
  \begin{tabular}[t]{ m{.45\columnwidth} | m{.45\columnwidth} }
    \textbf{\large System} & \textbf{\large Kernel} \\
      \begin{flushleft}%s\end{flushleft} & \begin{flushleft}%s\end{flushleft} \\
      \textbf{\large Processor} & \textbf{\large Memory} \\
      \verb|%s| & \verb|%s|
  \end{tabular}
\end{centering}""" % machineInfo))

        # get documentation of selected units
        for name in selection:
            # add class name to tex output and make output open on a new column
            output.extend(['', '%' * 80,
                           "%% Documentation of unit '%s'" %(name),
                           '%' * 80, '',
                           r'\vfill\null\columnbreak'])

            # output class documentation
            output.append(Documentation(packageIndex[name], **options))

        # finally, print output
        strOutput = os.linesep.join([str(item)
                                     if isinstance(item, DOC.__DOC__) else item
                                     for item in output])
        if len(self.args.output) == 0:
            print(strOutput)
        else:
            with open(self.args.output, "w") as f:
                f.write(strOutput)

            print(" >> output written to '%s'" %(self.args.output))

    ############################################## command: calibrate
    def calibrate(self, *arguments):
        '''Extract built-in documentation from units in package.'''
        self.argParser.add_argument(
            'classes',
            nargs='*',
            default=[],
            help="If an output file is specified, select the classes to " +
            "perform a calibration on. If as sole item the string `all` is " +
            "given, all classes in fastmat will be selected."
        )
        self.argParser.add_argument(
            '-i', '--input',
            type=str,
            default='',
            help="Filename to read calibration data from. May be used to " +
            "load existing calibration data for plotting."
        )
        self.argParser.add_argument(
            '-o', '--output',
            type=str,
            default='',
            help="Perform a calibration of all selected classes. Write the " +
            "resulting calibration data to OUTPUT"
        )
        self.argParser.add_argument(
            '-p', '--plot',
            action='store_true',
            help="Plot the calibration data for all selected classes. If no " +
            "new calibration was performed a fresh set of overhead " +
            "benchmarks will be generated and used for plotting against."
        )
        self.argParser.add_argument(
            '-v', '--verbose',
            action='store_true',
            help="Be verbose about the calibration."
        )

        # parse arguments
        self.args = self.argParser.parse_args(arguments)

        # determine a list of selected classes: interpret codewords
        if len(self.args.classes) == 1 and self.args.classes[0] == 'all':
            classes = fastmat.classes
        else:
            classNames = {item.__name__: item for item in fastmat.classes}
            classes = [classNames[name]
                       for name in self.args.classes
                       if name in classNames]

        # explore dependencies of class benchmarks in two stages:
        # STAGE I: determine class inheritance relations (code dependencies)
        dependencies = {
            item: set([child for child in inspect.getmro(item)
                       if (issubclass(child, fastmat.Matrix) and
                           not (child == item))])
            for item in fastmat.classes}

        # STAGE II: add nested matrix classes in benchmark instances
        #           (data dependencies)
        def nodesOfTree(tree):
            nodes = set()
            if tree is not None:
                for child in tree:
                    nodes.add(child)
                    nodes.update(nodesOfTree(child))

            return nodes

        instances = [
            Benchmark(item).options[BENCH.OVERHEAD][BENCH.FUNC_GEN](2)
            for item in fastmat.classes]

        for inst in instances:
            deps = dependencies.setdefault(inst.__class__, set())
            deps.update(node.__class__ for node in nodesOfTree(inst))

        # now compile a new selection list, which also includes dependent but
        # currently not yet calibrated classes
        # traverse the list in order of dependency count to put baseclasses
        # before deeply-nested metalasses. sort alphabetically if count matches
        selected = []
        for item in sorted(classes,
                           key=lambda x: (len(dependencies[x]), x.__name__)):
            # the class in `item` was selected for calibration. Make
            # sure all dependent classes are calibrated as well
            def updateSelected(node):
                # recurse over all subnodes of node
                for dep in dependencies[node]:
                    if dep not in selected:
                        updateSelected(dep)

                # finally, add the node that just got crawled it not
                # already done before
                if (node not in selected and
                        node not in fastmat.core.calData):
                    selected.append(node)

            # add all the requested classes to the job list. Make sure
            # the order is such that no class is added before all of its
            # dependencies are fulfilled by the list contents so far
            updateSelected(item)

        self.args.classes = selected

        # load calibration data from file if specified
        if len(self.args.input) > 0:
            fastmat.core.loadCalibration(self.args.input)
            print("  > Loaded calibration data from %s." %(self.args.input))

        # generate calibration data for the classes specified
        benchmarks = {}
        # first, figure out in which order to perform the calibrations
        # also

        # finally run the benchmarks
        for item in self.args.classes:
            print("  > Running calibration of class '%s'" %(item.__name__))
            cal, bench = fastmat.core.calibrateClass(item,
                                                     verbose=self.args.verbose)
            benchmarks[item] = bench

        # write calibration data to file if an output file is specified
        if len(self.args.output) > 0:
            fastmat.core.saveCalibration(self.args.output)
            print("  > Written calibration data to %s." %(self.args.output))

        # plot the current set of calibration data and perform benchmarks if
        # we have no data for some of the classes
        if self.args.plot:
            missing = [item for item in fastmat.core.calData
                       if item not in benchmarks]
            if len(missing) > 0:
                print("  > Generating performance measurements for class plots")
                for item in missing:
                    print("     - %s" %(item.__name__))
                    benchmarks[item] = fastmat.core.calibrateClass(
                        item, verbose=self.args.verbose, benchmarkOnly=True)

            plt = importMatplotlib()
            print("  > Plotting %d figures." %(len(fastmat.core.calData)))

            def plot(arrN, arrTime, arrNested, arrEstimate,
                     title, legend=False):
                '''
                Plot a time measurement and -model for each a Forward() and
                Backward() transform
                '''
                plt.loglog(arrN, arrTime[:, 0],
                           color=colors[0], linestyle='dotted')
                plt.loglog(arrN, arrTime[:, 1],
                           color=colors[1], linestyle='dotted')
                plt.loglog(arrN, arrNested[:, 0],
                           color=colors[0], linestyle='dashed')
                plt.loglog(arrN, arrNested[:, 1],
                           color=colors[1], linestyle='dashed')
                plt.loglog(arrN, arrEstimate[:, 0],
                           color=colors[0], linestyle='solid')
                plt.loglog(arrN, arrEstimate[:, 1],
                           color=colors[1], linestyle='solid')
                if legend:
                    plt.legend(['%s-%s' %(a, b)
                                for a in ['time measure', 'nested matrices',
                                          'estimated model']
                                for b in ['Forward()', 'Backward()']])
                plt.title(title)

            bb = 1
            numBlocks = int(np.ceil(np.sqrt(len(benchmarks))))
            for item, bench in benchmarks.items():
                # arrN holds the problem size of each benchmark
                # arrTime holds the measured durations of forward and backward
                # arrNested states the amount of time spent in nested classes
                # for the forward (col 0) and backward (col 1) transform
                arrN = bench.getResult('overhead', 'numN')

                arrTime = bench.getResult('overhead',
                                          'forwardMin',
                                          'backwardMin')
                arrNested = bench.getResult('overhead',
                                            BENCH.RESULT_OVH_NESTED_F,
                                            BENCH.RESULT_OVH_NESTED_B,
                                            BENCH.RESULT_EFF_NESTED_F,
                                            BENCH.RESULT_EFF_NESTED_B)
                arrNested = arrNested[:, 0:2] + arrNested[:, 2:4]

                # now load the calibration data and generate the estimate based
                # on the calibration and the transforms' complexity estimates
                cal = fastmat.core.getMatrixCalibration(item)
                matCal = np.diag(np.array([cal.gainForward, cal.gainBackward]))
                arrComplexity = bench.getResult('overhead',
                                                BENCH.RESULT_COMPLEXITY_F,
                                                BENCH.RESULT_COMPLEXITY_B)
                arrEstimate = (
                    np.array([[cal.offsetForward, cal.offsetBackward]]) +
                    arrNested + arrComplexity.dot(matCal))

                # final step: plotting into a new subplot each
                plt.subplot(numBlocks, numBlocks, bb)
                bb = bb + 1
                plot(arrN, arrTime, arrNested, arrEstimate, item.__name__)

            # plot an empty diagram with the same line parameters as the other
            # plots, but this time with a legend -> generates a legend frame
            plt.subplot(numBlocks, numBlocks, bb)
            plot(np.empty((0, 2)), np.empty((0, 2)),
                 np.empty((0, 2)), np.empty((0, 2)), 'Legend', legend=True)

            plt.show()

    ############################################## command: performance
    def performance(self, *arguments):
        '''Extract built-in documentation from units in package.'''
        self.argParser.add_argument(
            'classes',
            nargs='*',
            default=[],
            help="Select the classes to show in performance plots. " +
            "`all` is default."
        )
        self.argParser.add_argument(
            '-c', '--calibration',
            type=str,
            default='',
            help="Filename to read calibration data from. Loads calibration " +
            "data for fastmat classes which is then used by the package."
        )

        # parse arguments
        self.args = self.argParser.parse_args(arguments)

        # determine a list of selected classes: interpret codewords
        if len(self.args.classes) < 1:
            classes = fastmat.classes
        else:
            classNames = {item.__name__: item for item in fastmat.classes}
            classes = [classNames[name]
                       for name in self.args.classes
                       if name in classNames]

        # load calibration data from file if specified
        if len(self.args.calibration) > 0:
            fastmat.core.loadCalibration(self.args.calibration)
            print("  > Loaded calibration data from %s." %(
                self.args.calibration))

        # run the benchmarks and plot the result
        plt = importMatplotlib()

        def plot(arrN, arrTime, arrEstimate, title, legend=False):
            '''
            Plot a time measurement and -model for each a Forward() and
            Backward() transform
            '''
            plotLegend = ['time measure']
            plt.loglog(arrN, arrTime[:, 0],
                       color=colors[0], linestyle='dotted')
            plt.loglog(arrN, arrTime[:, 1],
                       color=colors[1], linestyle='dotted')
            if arrEstimate.shape[0] == arrN.shape[0]:
                plt.loglog(arrN, arrEstimate[:, 0],
                           color=colors[0], linestyle='solid')
                plt.loglog(arrN, arrEstimate[:, 1],
                           color=colors[1], linestyle='solid')
                plotLegend.append('estimated model')

            if legend:
                plt.legend(['%s-%s' %(a, b)
                            for a in plotLegend
                            for b in ['Forward()', 'Backward()']])
            plt.title(title)

        print("  > Generating performance measurements for class plots")
        bb = 1
        numBlocks = int(np.ceil(np.sqrt(len(classes) + 1)))
        for item in classes:
            print("     - %s" %(item.__name__))
            bench = fastmat.core.calibrateClass(item, benchmarkOnly=True)

            # arrN holds the problem size of each benchmark
            # arrTime holds the measured durations of forward and backward
            # arrNested states the amount of time spent in nested classes
            # for the forward (col 0) and backward (col 1) transform
            arrN = bench.getResult('overhead', 'numN').squeeze()

            arrTime = bench.getResult('overhead',
                                      'forwardMin',
                                      'backwardMin')

            # fetch the runtime estimation from the benchmarks
            arrEstimate = bench.getResult('overhead',
                                          BENCH.RESULT_ESTIMATE_FWD,
                                          BENCH.RESULT_ESTIMATE_BWD)

            # final step: plotting into a new subplot each
            plt.subplot(numBlocks, numBlocks, bb)
            bb = bb + 1
            plot(arrN, arrTime, arrEstimate, item.__name__)

        # plot an empty diagram with the same line parameters as the other
        # plots, but this time with a legend -> generates a legend frame
        plt.subplot(numBlocks, numBlocks, bb)
        plot(np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)),
             'Legend', legend=True)

        print("  > Plotting %d figures." %(len(classes)))
        plt.show()


################################################################################
################################################## Script entry point
if __name__ == '__main__':
    ################################################## Collect information
    # keep track of already seen items during the search
    seenItems=set([])

    def crawlModule(module, level, targets):
        elements={}

        # determine elements of package
        items=getObjects(module)

        # determine type of package elements
        for itemName in items:
            # fetch element
            try:
                item=getattr(module, itemName)
            except AttributeError:
                continue

            # determine module name and package descendency
            # Also check, that the following conditions are met:
            #  -> the module is a part of our package
            #  -> we haven't seen it yet
            #  -> the except AttributeError catches errors caused by numeric
            #     function names sometimes generated by cython
            try:
                moduleName=item.__module__ \
                    if not inspect.ismodule(item) else item.__name__
                modulePath=moduleName.split('.')
            except AttributeError:
                continue

            try:
                if modulePath[0] != packageName or (item in seenItems):
                    continue
            except TypeError:
                continue

            seenItems.add(item)

            # if the element is a submodule of fastmat, go deeper
            if inspect.ismodule(item):
                result = crawlModule(item, level + 1, targets)
                if len(result) > 0:
                    elements[itemName] = result
            elif (inspect.isclass(item) and issubclass(item, targets)):
                elements[itemName] = item

        return elements

    # Index elements in fastmat by location
    packageIndexTree = crawlModule(fastmat, 1, classBaseContainers)

    ############################################## Collect package infos
    # Flatten Index for categorization
    def appendDictFlattened(
        output,
        prefix,
        dictionary,
        separator='.'
    ):
        '''Flatten dict, prefixing nested levels with its key value'''
        if len(prefix) > 0:
            prefix=prefix + separator

        # iterate elements, nest if dict was found, add otherwise
        for key, item in dictionary.items():
            if isinstance(item, dict):
                appendDictFlattened(output, prefix + key, item)
            else:
                output[prefix + key]=item

    packageIndex={}
    appendDictFlattened(packageIndex, '', packageIndexTree)

    # now filter out the fastmat.Algorithm class as it is just a classifier
    packageIndex={name: element
                  for name, element in packageIndex.items()
                  if element is not fastmat.Algorithm}

    ############################################## analyze structure
    # categorize into classes (class) and algorithms (routine)
    packageClasses={loc: element
                    for loc, element in packageIndex.items()
                    if issubclass(element, fastmat.Matrix)}

    packageAlgs={loc: element
                 for loc, element in packageIndex.items()
                 if issubclass(element, fastmat.Algorithm)}

    ################################################## Parse and Run
    Bee(*(sys.argv[1:]), prevArgs=' '.join(sys.argv[:1]))
