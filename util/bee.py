# -*- coding: utf-8 -*-
'''
  util/bee.py
 -------------------------------------------------- part of the fastmat package

  Utility for shell interaction with class code from python packages.

  Usecases:
    - find all modules in a (sub-)package, which contain certain functions
        --check-only --function

    - provide a string interface to module-level functions enabling shell
      piping to python package internals
        --argument --function

    - flexible options for character encoding
        --encode utf-8

  Example:  retrieve LaTeX documentation of class Circulant, execute the
            following one-liner from the doc/ directory of fastmat:

            python doc_extract.py --path .. --package fastmat --name Circulant
            --function docLaTeX --encode utf-8


  Author      : wcw
  Introduced  : 2017-01-03
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
import inspect
import os
import argparse
import importlib
import pprint
import time

import numpy as np
from scipy import io as spio
from numbers import Number

# import fastmat, try as global package or locally from one floor up
try:
    import fastmat
except ImportError:
    sys.path.append('..')
    sys.path.append('.')
    import fastmat

from fastmat.helpers.unitInterface import *
from routines.printing import *

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
            printTitle(toolTitle)
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
            self.argParser.add_argument(
                '-f', '--filename',
                action='store_true',
                help="Print 'filename' key of dataset values if dictionary " +
                "unless extended output is enabled."
            )
            self.args = self.argParser.parse_args(arguments)

        def _printDataset(
            self,
            dataset,
            newline='\n',
            separator=' '
        ):
            '''List contents of dataset according to modifiers in self.args.'''
            if self.args.extended:
                pprint.pprint(dataset)
            else:
                if self.args.filename:
                    # if filename switch is given, extract values of the
                    # 'filename' key of all dictionaries in dataset
                    lst = [
                        d[NAME_FILENAME] for d in dataset.values()
                        if isinstance(d, dict) and d.get(NAME_FILENAME, None)
                    ]
                else:
                    # otherwise extract all key names of dataset
                    lst = [loc for loc in dataset.keys()]

                # print the list
                print(separator.join(sorted(lst)) + newline)

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

        def units(self, *arguments):
            '''Return list of available units in package.'''
            self._parseAndPrint(packageUnits, *arguments)

        def benchmarks(self, *arguments):
            '''Return list of benchmark targets used in package.'''
            self._parseAndPrint(packageBenchmarks, *arguments)

        def benchmarktargets(self, *arguments):
            '''Return list of available benchmark targets.'''
            self._parseAndPrint(packageBenchmarkTargets, *arguments)

        def documentation(self, *arguments):
            '''Return list of units with documentation included.'''
            self._parseAndPrint(packageDocumentation, *arguments)

        def tests(self, *arguments):
            '''Return list of test targets used in package.'''
            self._parseAndPrint(packageTests, *arguments)

        def testtargets(self, *arguments):
            '''Return list of test targets used in package.'''
            self._parseAndPrint(packageTestTargets, *arguments)

        def makedump(self, *arguments):
            '''Dump all information for makefile processing.'''
            self._parseArgs()

            printSetup = {'newline': ';', 'separator': ':'}
            for dataset in [packageClasses, packageAlgs,
                            packageBenchmarkTargets]:
                self.args.filename = False
                self._printDataset(dataset, **printSetup)
                self.args.filename = True
                self._printDataset(dataset, **printSetup)

    ############################################## selection helper function
    def _select(self, nameType, arguments, lstTypes, lstTargets):
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
        # perform deep copy to allow modifying data locally later on
        selected = {key: value.copy() for key, value in lstTargets.items()}
        if len(self.args.select) > 0:
            # prepare lists to collect parameters
            filterUnit = set([])
            filterType = set([])
            picked = set([])

            # classify options in lists
            for name in self.args.select:
                tokens = name.split('.')
                if len(tokens) < 2:
                    continue

                if tokens[0] == '':
                    # begins with '.' -> ".benchmark" mask
                    filterType.add(tokens[-1])
                elif tokens[-1] == '':
                    # ends on '.' -> "unit." mask
                    filterUnit.add('.'.join(tokens[:-1]))
                else:
                    picked.add(name)

            # if a filter contains no elements, disable it by filling its
            # "enable mask" with all possible elements
            if len(filterUnit) == 0 and len(filterType) > 0:
                filterUnit = set(packageUnits.keys())

            if len(filterType) == 0 and len(filterUnit) > 0:
                filterType = set(lstTypes.keys())

            # remove items not picked and not meeting filtermasks
            popList = [
                key for key, target in selected.items()
                if not ((target[NAME_UNIT] in filterUnit and
                         target[nameType] in filterType) or key in picked)]

            for key in popList:
                selected.pop(key)

        return selected

    ############################################## command: test
    def test(self, *arguments):
        '''Run unit tests in package.'''
        from routines import test

        # parse arguments and get selection from list
        self.argParser.add_argument(
            '-w', '--write-failed',
            action='store_true',
            help="If specified, stores failed test results to a .mat file " +
            "for each failed test instance, including parameters, input and " +
            "evaluation output"
        )
        self.argParser.add_argument(
            '-p', '--path-results',
            type=str,
            default='.',
            help="Path to put the result files for failed tests."
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
        tests = self._select(
            'test', arguments,
            packageTests, packageTestTargets
        )

        # sanity check
        if len(tests) < 1:
            self.argParser.exit(0, "test: job list is empty.\n")

        # update test result filenames to point to path specified
        for target in tests.values():
            target[NAME_FILENAME] = os.path.normpath(os.path.join(
                self.args.path_results, target[NAME_FILENAME]))

        # run tests
        problems = {}
        testUnits = set([])
        testCount = 0
        testIgnoredCount = 0
        timeTotal = time.time()
        for target in sorted(tests.keys()):
            testUnits.add(tests[target]['unit'])

            # run tests (prints results instantly)
            unitTestCount, problems[target], unitTestIgnoredCnt = \
                test.runTest(
                    tests[target], *self.extraArgs,
                    fullOutput=self.args.full,
                    verbose=self.args.verbose
            )
            testCount += unitTestCount
            testIgnoredCount += unitTestIgnoredCnt

            # now comes saving the output data. But first check if this is
            # actually what we want
            if not self.args.write_failed or len(problems[target]) == 0:
                continue

            # prepare data to be stored
            #  * convert non-representables to strings (keep numbers and arrays)
            def _convert(data):
                # behave recursive for all types of containers
                if isinstance(data, dict):
                    for key, value in data.items():
                        data[key] = _convert(value)

                    return data
                elif isinstance(data, list):
                    for ii, item in enumerate(data):
                        data[ii] = _convert(item)

                    return data
                elif isinstance(data, tuple):
                    lst = list(data)
                    _convert(lst)
                    return tuple(lst)

                # now check for elements
                elif isinstance(data, (np.ndarray, Number, str)):
                    return data
                # everything else needs to be converted
                elif isinstance(data, fastmat.Matrix):
                    return data.__repr__()
                else:
                    return str(data)

            # save test data, ensure output path exists
            # (convert result data to storeable format)
            filename = tests[target][NAME_FILENAME]
            path = os.path.dirname(filename)
            if not os.path.exists(path):
                os.makedirs(path)

            spio.savemat(filename, _convert(results))
            print("   > results saved to '%s'" % (filename))

        timeTotal = time.time() - timeTotal

        # analyze dependencies of units involved in problems
        problemUnits = set(['.'.join(name.split('.')[:-1])
                            for name, lst in problems.items() if len(lst) > 0])

        independentUnits = []
        dependentUnits = {}
        fineUnits = []
        unitDependencies = {}
        for unit in testUnits:
            if unit not in problemUnits:
                fineUnits.append(unit)
                continue

            dependencies = sorted([dep
                                   for dep in vars(packageUnits[unit]['module'])
                                   if (dep in packageUnits) and (dep != unit)])
            unitDependencies[unit] = dependencies

            if len(dependencies) == 0:
                independentUnits.append(unit)
            else:
                dependentUnits[unit] = dependencies

        # flatten unit-name based list of problems
        problems = {name: problem
                    for unitProblems in problems.values()
                    for name, problem in unitProblems.items()}

        # print results
        printTitle("Results")
        print(("   > found %d problem(s) in %d units%s " +
               "(ran %d tests in %.2f seconds)") %(
            len(problems), len(problemUnits),
            ("" if testIgnoredCount == 0
             else " (%d issues ignored)" %(testIgnoredCount)),
            testCount, timeTotal))

        if len(fineUnits) > 0:
            print("   > Units with no problems:")
            print("%s[%s]\n" % (
                " " * 8, ", ".join([fmtGreen(unit)
                                    for unit in fineUnits])))

        if len(independentUnits) > 0:
            print("   > Units with independent problems (start fixing here):")
            print("%s[%s]\n" % (
                " " * 8, ", ".join([fmtRed(unit)
                                    for unit in sorted(independentUnits)])))

        if len(dependentUnits) > 0:
            nameLength = max(len(name) for name in dependentUnits)
            print("   > Units with problems, depending on other " +
                  "units with problems")
            for unit, depList in sorted(dependentUnits.items()):
                print("%s%*s: [%s]" % (
                    " " * 8,
                    nameLength, fmtRed(unit),
                    ", ".join([
                        dep if dep not in testUnits
                        else [fmtGreen, fmtYellow][dep in problemUnits](dep)
                        for dep in sorted(depList)])))
            print("\n")

        # interactive session, anyone?
        if self.args.interact and len(problems) > 0:
            # register signaling for graceful exit
            import atexit
            try:
                import matplotlib.pyplot as mpl
            except ImportError:
                pass

            def quitGracefully():
                print("Leaving interactive testing session.")

            atexit.register(quitGracefully)

            # scope dictionary
            scope={}

            # define helper function for filtering problems
            def select(*filters):
                return {name: item for name, item in problems.items()
                        if all(f in name for f in filters)}

            def get(index, data=problems, verbose=True):
                key = list(data.keys())[index]
                if verbose:
                    print("selected problem '%s'." %(fmtBold(key)))
                return data[key]

            def fetch(problem, dataSet=problems, scope=scope):
                def _load(key, name, doc=""):
                    if key in problem:
                        print("retrieved %s <= [%s] %s" %(
                            name, key, "" if len(doc) == 0 else " (%s)" %(doc)))
                        scope[name]=problem[key]

                # fetch from scope (interpret as index) if no problem passed
                if problem not in scope:
                    problem = get(problem, data=dataSet)

                # load fields from problem
                _load(TEST_INSTANCE, 'inst',
                      doc="matrix instance")
                _load(TEST_REFERENCE, 'arrMat',
                      doc="plain matrix representation")
                _load(TEST_RESULT_INPUT, 'arrIn',
                      doc="input to tested code segment")
                _load(TEST_RESULT_OUTPUT, 'arrOut',
                      doc="output of tested code segment")
                _load(TEST_RESULT_REF, 'arrRef',
                      doc="reference calculation result")
                _load(TEST_RESULT_TOLERR, 'tolErr')

                # calculate some other measures
                try:

                    if all(key in scope for key in ['arrRef', 'arrOut']):
                        arrOut, arrRef = scope['arrOut'], scope['arrRef']

                        eps0 = scope['eps0'] = \
                            fastmat.helpers.types._getTypeEps(arrOut.dtype)

                        arrDiff = arrOut - arrRef
                        absErr = scope['absErr'] = (np.abs(arrDiff).max() /
                                                    np.abs(arrRef).max())
                        print("absErr = %e [tolErr = %e]" %(
                            absErr, scope['tolErr']))

                        if eps0 != 0:
                            relErr = scope['relErr'] = absErr / eps0
                            print("relErr = %e [eps = %e]" %(relErr, eps0))
                except NameError:
                    pass

            def getDiff(data=problems):
                return {name: (item['testOutput'] - item['testReference'])
                        for name, item in data.items()}

            def getAbs(data=problems):
                return {name: abs(item) for name, item in data.items()}

            def getMax(data=problems):
                return {name: item.max() for name, item in data.items()}

            def show(data=problems):
                pprint.pprint(data)

            # set numpy print options
            np.set_printoptions(
                linewidth=200,
                threshold=10,
                edgeitems=3,
                formatter={
                    'float_kind'    : '{: .3}'.format,
                    'complex_kind'  : '{: .3e}'.format
                }
            )

            # scope dictionary
            scope.update(globals())
            scope.update(locals())

            # start interactive session
            import code
            code.interact(
                r"""
   > Entering interactive testing session.
   > Available commands:
      * select('selector', ...) - Select subset from 'problems'
                                  (includes auto-completion)
      * get(index, dataSet)     - Return single problem #(index) from dataSet
      * show(dataSet)           - Print all problems in dataSet
      * fetch(problem)          - extract data fields from single problem
      * getDiff(dataSet)        - return diff arrays for problems in dataSet
      * getAbs(dictArrays)      - return absolute of all arrays in dictArrays
      * getMax(dictArrays)      - return maximum of all values in dictArrays
                """, None, scope)

        # if problems occurred, exit with non-zero errorcode
        if len(problems) > 0:
            sys.exit(1)

    ############################################## command: benchmark
    def benchmark(self, *arguments):
        '''Run performance evaluation of units in package.'''
        from routines import benchmark

        # parse arguments and get selection from list
        self.argParser.add_argument(
            '-p', '--path-results',
            type=str,
            default='.',
            help="Save the benchmark results to the path specified."
        )
        benchmarks=self._select(
            NAME_BENCHMARK, arguments,
            packageBenchmarks, packageBenchmarkTargets)

        # sanity check
        if len(benchmarks) < 1:
            self.argParser.error("benchmark: job list is empty.")

        # update benchmark result filenames to point to path specified
        for target in benchmarks.values():
            target[NAME_FILENAME]=os.path.normpath(os.path.join(
                self.args.path_results, target[NAME_FILENAME]))

        # run benchmarks
        for target in sorted(benchmarks.keys()):
            bench=benchmarks[target]

            # run benchmark with set of bench options and extra cmd arguments
            benchmark.runEvaluation(*self.extraArgs, **bench)
            print("   > results saved to '%s'" % (bench[NAME_FILENAME]))

    ############################################## command: documentation
    def documentation(self, *arguments):
        '''Extract built-in documentation from units in package.'''
        from routines import documentation

        self.argParser.add_argument(
            'units',
            nargs='*',
            default=packageDocumentation.keys(),
            help="Specify the documentation sections to include in the TeX " +
            "output. Defaults to all units having available documentation."
        )
        self.argParser.add_argument(
            '-p', '--path-results',
            type=str,
            default='.',
            help='Path to the benchmark results.'
        )
        self.argParser.add_argument(
            '-s', '--section',
            type=str,
            default=None,
            help="If specified, open a new \section with the caption given"
        )

        # parse arguments, all extra stuff goes to 'extraParam'
        self.args, self.extraParam=self.argParser.parse_known_args(arguments)

        # get list of selected units, ensure they contain some doc
        docs={name: packageDocumentation[name].copy()
              for name in self.args.units if name in packageDocumentation}

        if len(docs) < 1:
            self.argParser.error("documentation: job list is empty.")

        # get the file names of benchmark result files for each module, update
        # the filenames with the specified path where they shall be stored and
        # check if they exist. Store the compiled list in the unit's doc info
        for nameDoc, doc in docs.items():
            # copy benchmark target definitions for this unit
            doc['results']={name: bench.copy()
                            for name, bench in packageBenchmarkTargets.items()
                            if bench.get(NAME_UNIT, None) == nameDoc}

            # iterate over benchmark targets, updating filenames
            for nameBench, bench in doc['results'].items():
                bench[NAME_FILENAME]=os.path.normpath(
                    os.path.join(self.args.path_results, bench[NAME_FILENAME]))

        # print header
        print("% BEWARE: THIS FILE WAS GENERATED AUTOMATICALLY. ALL EDITS " +
              "MAY BE OVERWRITTEN WITHOUT PRIOR NOTICE %")

        # output section header if applicable
        if self.args.section:
            print("\section{%s}" % (self.args.section))

        # output all documentation information to tex file.
        for nameDoc, doc in sorted(docs.items()):
            # state which unit gets printed
            printTitle(
                "Documentation of unit '%s'" % (nameDoc),
                repChar='%', strBorder='%%', width=80, style=lambda str: str)

            # output class documentation
            print(doc[NAME_DOCU])

            # output benchmark results if applicable
            results={name: result
                     for name, result in sorted(doc.get('results', {}).items())
                     if os.path.isfile(result.get(NAME_FILENAME, ''))}

            if len(results) > 0:
                print("\subsubsection{Time and Memory complexity}")

            for result in results.values():
                print(documentation.docResultOutput(result))

            # print one empty line
            print("")


################################################################################
################################################## Script entry point
if __name__ == '__main__':
    ################################################## Collect information
    # keep track of already seen items during the search
    seenItems=set([])

    def crawlModule(module, level):
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
            #  -> the item is named after its container
            #  -> we haven't seen it yet
            #  -> the except AttributeError catches errors caused by numeric
            #     function names sometimes generated by cython
            try:
                moduleName=item.__module__ \
                    if not inspect.ismodule(item) else item.__name__
                modulePath=moduleName.split('.')
            except AttributeError:
                continue

            if modulePath[0] != packageName or modulePath[-1] != itemName or \
               (item in seenItems):
                continue

            seenItems.add(item)

            # if the element is a submodule of fastmat, go deeper
            if inspect.ismodule(item):
                result=crawlModule(item, level + 1)
                if len(result) > 0:
                    elements[itemName]=result
            elif inspect.isroutine(item) or inspect.isclass(item):
                elements[itemName]=item

        return elements

    # Index elements in fastmat (classes, routines, modules)
    packageIndex=crawlModule(fastmat, 1)

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

    packageIndexFlat={}
    appendDictFlattened(packageIndexFlat, '', packageIndex)

    ############################################## analyze structure
    packageUnits={}
    for loc, item in packageIndexFlat.items():
        # assemble generic set of infos
        absPath='.'.join([packageName, loc])
        info={
            'object': item,
            'absModulePath': absPath,
            'relModulePath': loc,
            'name': item.__name__
        }

        # try to load container module and add some more
        try:
            module=importlib.import_module(absPath)
            info['module']=module
            info[NAME_FILENAME]=module.__file__

            # try to load entry points for related helpers
            def setInfoIfPresent(key):
                obj=getattr(module, key, None)
                if obj is not None:
                    info[key]=obj

            for key in [NAME_BENCHMARK, NAME_DOCU, NAME_TEST]:
                setInfoIfPresent(key)

        except ImportError:
            continue

        packageUnits[loc]=info

    # categorize into classes (class) and algorithms (routine)
    packageClasses={key: item
                    for key, item in packageUnits.items()
                    if inspect.isclass(item['object'])}

    packageAlgs={key: item
                 for key, item in packageUnits.items()
                 if inspect.isroutine(item['object'])}

    ################################################## collect benchmarks
    # compile list of all used benchmark types and list which units use them
    # also, compile list of available benchmark targets
    packageBenchmarks={}
    packageBenchmarkTargets={}
    for unitName, unit in packageUnits.items():
        if isinstance(unit.get(NAME_BENCHMARK, None), dict):
            benchmarks=unit[NAME_BENCHMARK]
            for benchmarkName in benchmarks.keys():
                if benchmarkName == NAME_COMMON:
                    continue

                packageBenchmarks.setdefault(benchmarkName, []).append(unitName)
                target={}
                target.update(benchmarks.get(NAME_COMMON, {}))
                target.update(benchmarks[benchmarkName])
                target.setdefault('template', benchmarkName)
                target[NAME_UNIT]=unitName
                target[NAME_BENCHMARK]=benchmarkName
                name="%s.%s" % (unitName, benchmarkName)
                target[NAME_FILENAME]=os.path.join(
                    '.', '.'.join([name, EXT_BENCH_RESULT]))
                packageBenchmarkTargets[name]=target

    ################################################## collect tests
    # compile list of all used test types and list which units use them also,
    # compile list of available test targets
    packageTests={}
    packageTestTargets={}
    for unitName, unit in packageUnits.items():
        if isinstance(unit.get(NAME_TEST, None), dict):
            tests=unit[NAME_TEST]
            for testName in tests.keys():
                if testName == NAME_COMMON:
                    continue

                packageTests.setdefault(testName, []).append(unitName)
                test={}
                test.update(tests.get(NAME_COMMON, {}))
                test.update(tests[testName])
                test[NAME_UNIT]=unitName
                test[NAME_TEST]=testName
                name="%s.%s" % (unitName, testName)
                test[NAME_FILENAME]=os.path.join(
                    '.', '.'.join([name, EXT_TEST_RESULT]))
                packageTestTargets[name]=test

    ################################################## collect documentation
    packageDocumentation={key: item for key, item in packageUnits.items()
                          if isinstance(item.get(NAME_DOCU, None), str)}

    ################################################## Parse and Run
    Bee(*(sys.argv[1:]), prevArgs=' '.join(sys.argv[:1]))
