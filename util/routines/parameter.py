# -*- coding: utf-8 -*-
'''
  fastmat/util/routines/parameter.py
 -------------------------------------------------- part of the fastmat package

  Routines for managing parameter dictionaries supporting templating, value
  permutation, dereferentiation and in-place evaluation of functionals.


  Author      : wcw
  Introduced  : 2017-03-17
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

import copy
import inspect
import itertools
import pprint
import six
import re

# import fastmat, try as global package or locally from two floors up
try:
    import fastmat
except ImportError:
    sys.path.insert(0, os.path.join('..', '..'))
    import fastmat
from fastmat.helpers.unitInterface import *
from fastmat import Matrix

try:
    from itertools import izip
except ImportError:         # python 3.x
    izip = zip


################################################## paramDict
reFormatString = re.compile(r'%\(.+\)')


class paramDict(dict):

    def __getattr__(self, key):
        # evaluate nested format-string parameters, update format results
        value, lastValue = super(self.__class__, self).__getitem__(key), None

        while id(lastValue) != id(value):
            lastValue = value
            if isinstance(value, str):
                if value in self and value != key:
                    value = getattr(self, value)
#                    self[key] = value
                elif reFormatString.search(value):
                    value = value %self
#                    self[key] = value

            elif (inspect.isroutine(value) and
                  not isinstance(value, IgnoreFunc)):
                value = value(self)
                self[key] = value

        return value


################################################## paramApplyDefaults()
def paramApplyDefaults(
    params,
    templates=None,
    templateKey=None,
    extraArgs=None
):

    # have some defaults from templates
    # first, fetch the template as defaults, then update with target,
    # then assign the whole dict to target
    result = {}
    if templates is not None:
        # 1. COMMON - section of templates (lowest-priority)
        result.update(templates.get(NAME_COMMON, {}))
        # 2. the templates-section corresponding to the templateKey
        result.update(templates.get(templateKey, {}))
        # 3. specific reference to a template by the 'template' key in params
        if NAME_TEMPLATE in params:
            result.update(templates.get(params[NAME_TEMPLATE], {}))

    # 4. actual parameters (params)
    result.update(params)

    # 5. extraArgs (which usually come from command-line) (top-priority)
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

    return paramDict(result)


################################################## paramPermute()
def paramPermute(dictionary, copy=True, PermutationClass=Permutation):
    '''
    Return a list of cartesian-product combination of all dictionary values
    holding a Permutation list object. If copy is True, the resulting list will
    be applied back to input dictionary copies before returning, resulting in
    a list of copies with the permutation replacements being made on the input
    dictionary. Parameter permutations must be indicated by wrapping a list in
    a Permutation class instance.
    '''
    # isolate the permutation parameters from the dictionary
    parameters = {key: list(value)
                  for key, value in dictionary.items()
                  if type(value) == PermutationClass}

    # perform a cartesian product -> list of permuted instances
    permutations = [dict(izip(parameters, x))
                    for x in itertools.product(*six.itervalues(parameters))]
    if len(permutations) == 1:
        permutations = [{}]

    # apply the permutations to the input dictionary (generating copies)
    def dictCopyAndMerge(source, merge):
        result = source.copy()
        result.update(merge)
        return paramDict(result)
    return [dictCopyAndMerge(dictionary, permutation)
            for permutation in permutations] if copy else permutations


################################################## paramDereferentiate()
def paramDereferentiate(currentLevel, paramDict=None):
    '''
    Replace all text value identifyers matching a target key name with the
    key values it's pointing to allowing parameter links. Then, recurse through
    an arbitrary depth of container types (dicts, lists, tuples) found in the
    first stage, continuing dereferentiation.
    Returns the modified currentLevel container.
    '''
    # set paramDict in first level, determine iterator for currentLevel
    iterator = currentLevel.items() \
        if isinstance(currentLevel, dict) else enumerate(currentLevel)
    if paramDict is None:
        paramDict = currentLevel

    # STAGE 1: Replace all dictionary values matching a key name in paramDict
    #          with the corresponding value of that paramDict-entry. Also build
    #          a container list for stage two.
    dictIterables = {}
    for key, value in iterator:
        if isinstance(value, (list, tuple, dict)):
            dictIterables[key] = value
        else:
            if not isinstance(value, str):
                continue
            try:
                paramValue = paramDict.get(value, None)
                if (paramValue is not None and
                        not isinstance(paramValue, Permutation)):
                    currentLevel[key] = paramValue
                    if isinstance(paramValue, (list, tuple, dict)):
                        dictIterables[key] = paramValue
            except TypeError:
                continue

    # STAGE 2: Crawl the containers found in stage 1, repeating the process.
    #          Note that nested containers are copied in the process.
    #          The parameter dictionary paramDict stays the same for all levels.
    for key, iterable in dictIterables.items():
        # copy the container to allow modification of values
        newIterable = copy.copy(iterable)
        if isinstance(iterable, tuple):
            # cast the immutable tuple type to list allowing modifications, cast
            # back to tuple afterwards
            newIterable = list(newIterable)
            paramDereferentiate(newIterable, paramDict)
            newIterable = tuple(newIterable)
        else:
            paramDereferentiate(newIterable, paramDict)
        # overwrite former iterable with new copy
        currentLevel[key] = newIterable

    return currentLevel


################################################## paramEvaluate()
def paramEvaluate(currentLevel, paramDict=None):
    '''
    Evaluate all functions found in currentLevel with paramDict as argument.
    Repeat the process for nested containers.
    Returns the modified currentLevel container.
    '''
    # set paramDict in first level, determine iterator for currentLevel
    iterator = currentLevel.items() \
        if isinstance(currentLevel, dict) else enumerate(currentLevel)
    if paramDict is None:
        paramDict = currentLevel

    # STAGE 1: Evaluate the functions found in currentLevel. Also build a
    #          container list for stage two.
    dictIterables = {}
    for key, value in iterator:
        if isinstance(value, (list, tuple, dict)):
            if not isinstance(value, IgnoreDict):
                dictIterables[key] = value
        elif inspect.isroutine(value):
            currentLevel[key] = value(paramDict)

    # STAGE 2: Crawl the containers found in stage 1, repeating the process
    #          The parameter dictionary paramDict stays the same for all levels.
    for key, iterable in dictIterables.items():
        if isinstance(iterable, tuple):
            # cast the immutable tuple type to list allowing modifications, cast
            # back to tuple afterwards
            newIterable = list(currentLevel[key])
            paramEvaluate(newIterable, paramDict)
            currentLevel[key] = tuple(newIterable)
        else:
            paramEvaluate(currentLevel[key], paramDict)

    return currentLevel
