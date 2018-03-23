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

import copy
import inspect
import itertools
import os
import re
import six
import struct
import numpy as np
from scipy import sparse
from platform import system as pfSystem

from ..core.types import isInteger

try:
    from itertools import izip
except ImportError:         # python 3.x
    izip = zip

currentOS = pfSystem()


################################################################################
################################################## CONSTANT definition classes
class ALIGNMENT(object):
    DONTCARE        = '-'
    FCONT           = 'F'
    CCONT           = 'C'
    STRIDE          = 'S'


################################################################################
################################################## Permutation funcs and classes

class AccessDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            found = [kk for kk in sorted(self.keys())
                     if kk.startswith(key)]
            return ([self[kk] for kk in found] if len(found) > 2
                    else (None if len(found) == 0
                          else self[found[0]]))

    def __repr__(self):
        return str(self.keys())


def convertToAccessDicts(level):
    for key, value in level.items():
        if isinstance(value, dict) and (value is not value):
            level[key] = convertToAccessDicts(value)

    return AccessDict(level)


################################################## class uniqueNameDict
class uniqueNameDict(dict):
    '''
    Modified dictionary: suffixes key names with integer to maintain uniqueness.
    '''

    def __setitem__(self, key, value):
        index = 1
        strKey = key
        while True:
            if strKey not in self:
                break
            index += 1
            strKey = "%s_%03d" %(key, index)

        dict.__setitem__(self, strKey, value)


################################################## class paramDict
reFormatString = re.compile(r'%\(.+\)')


class paramDict(dict):

    def __getattr__(self, key):
        # evaluate nested format-string parameters, update format results
        value, lastValue = super(paramDict, self).__getitem__(key), None

        while id(lastValue) != id(value):
            lastValue = value
            if isinstance(value, str):
                if value in self and value != key:
                    value = getattr(self, value)
                elif reFormatString.search(value):
                    value = value %self

            elif (inspect.isroutine(value) and
                  not isinstance(value, IgnoreFunc)):
                value = value(self)
                self[key] = value

        return value


################################################## class Permutation
class Permutation(list):

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__,
                           super(Permutation, self).__repr__())

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__,
                           super(Permutation, self).__str__())


################################################## class VariantPermutation
class VariantPermutation(Permutation):
    pass


################################################## class IgnoreDict
class IgnoreDict(dict):
    pass


################################################## class IgnoreFunc
class IgnoreFunc(object):

    def __init__(self, fun):
        self._fun = fun

    def __call__(self, *args, **kwargs):
        return self._fun(*args, **kwargs)


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
        result.update(templates.get(NAME.COMMON, {}))
        # 2. the templates-section corresponding to the templateKey
        result.update(templates.get(templateKey, {}))
        # 3. specific reference to a template by the 'template' key in params
        if NAME.TEMPLATE in params:
            result.update(templates.get(params[NAME.TEMPLATE], {}))

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
                  if isinstance(value, PermutationClass)}

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


################################################## mergeDicts()
# a little helper to safely merge two dictionaries
def mergeDicts(a, b):
    '''
    Merge the dictionaries a and b such that entries in b have priority and the
    input Dictionary a remains unchanged.
    '''
    c=a.copy()
    c.update(b)
    return c

################################################################################
################################################## array generators, test distr.


################################################## arrTestDist()
def arrTestDist(shape, dtype, center=0):
    def _draw(shape):
        '''
        Draw a random floating-point number from a test distribution.
        Remove the part around zero from the distribution and keep the distance
        between minimal and maximal absolute values (dynamics) relatively small
        '''
        return (np.random.uniform(2., 1., size=shape) *
                np.random.choice([-1, 1], shape))

    if np.prod(shape) < 1:
        return np.array([])

    if dtype in (np.int8, np.uint8, np.int32, np.uint32, np.int64, np.uint64):
        result = np.random.choice(
            [center - 2, center - 1, center + 1, center + 2], shape).astype(
                dtype)
    else:
        if dtype in (np.float32, np.float64):
            result = _draw(shape).astype(dtype) + center
        elif dtype in (np.complex64, np.complex128):
            result = (_draw(shape) + np.real(center) +
                      1j * (_draw(shape) + np.imag(center))).astype(dtype)
        else:
            raise TypeError("arrTestDist: unsupported type %s" % (dtype))

    # increase the largest element in magnitude a little bit more to avoid
    # too close neighbours to the largest element in the distribution
    # this helps at least largestSV in Diag matrices to converge ;)
    idxMax = np.unravel_index(np.abs(result).argmax(), result.shape)
    if isInteger(dtype):
        result[idxMax] += np.sign(result[idxMax])
    else:
        result[idxMax] *= 1.5

    return result

################################################## arrSparseTestDist()


def arrSparseTestDist(shape, dtype,
                      density=0.1, center=0, compactFullyOccupied=False):
    numSize = np.prod(shape)
    if compactFullyOccupied:
        # draw just enough samples randomly such that every row and column is
        # occupied with at least one element. Ignore the density parameter
        numElements = max(shape)

        # draw mm and nn coordinates from coordinate space. Modulo operation
        # wraps indices larger than the actual row or column dimension
        suppX = np.mod(
            np.random.choice(numElements, numElements, replace=False), shape[0])
        suppY = np.mod(
            np.random.choice(numElements, numElements, replace=False), shape[1])
    else:
        # draw a relative amount of samples randomly over vectorized index space
        numElements = int(numSize * density)
        supp = np.random.choice(np.arange(numSize), numElements, replace=False)
        suppX = np.mod(supp, shape[0])
        suppY = np.divide(supp, shape[0])

    # determine the actual element values distributed over the sparse array
    # from arrTestDist with a 1D-array spanning the required element count
    arrElements = arrTestDist((numElements, ), dtype, center=center)

    return sparse.coo_matrix(
        (arrElements, (suppX, suppY)), shape=shape, dtype=dtype)


################################################## arrAlign()
def arrAlign(arr, alignment=ALIGNMENT.DONTCARE):
    if alignment == ALIGNMENT.DONTCARE:
        return np.asanyarray(arr)
    elif alignment == ALIGNMENT.FCONT:
        return np.asfortranarray(arr)
    elif alignment == ALIGNMENT.CCONT:
        return np.ascontiguousarray(arr)
    elif alignment == ALIGNMENT.STRIDE:
        # define spacing between elements
        spacing = 3
        # determine maximum value
        try:
            maxValue = np.iinfo(arr.dtype).max
        except ValueError:
            try:
                maxValue = np.finfo(arr.dtype).max
            except ValueError:
                maxValue = 1.

        # generate large random array with maximized data type utilization
        arrFill = (maxValue * (np.random.rand(
            *(dim * spacing for dim in arr.shape)) - 0.5)).astype(arr.dtype)

        # fill-in the array data and return a view of the to-be-aligned array
        arrPart = arrFill[(np.s_[1::spacing], ) * arrFill.ndim]
        arrPart[:] = arr
        return arrPart
    else:
        raise ValueError("Unknown alignment identificator '%s'" %(alignment))


################################################## arrayGenerator
class ArrayGenerator(dict):

    @property
    def forwardData(self):
        if NAME.FWDATA not in self:
            # generate random array and set specific alignment style
            self[NAME.FWDATA] = arrAlign(
                arrTestDist(self[NAME.SHAPE], self[NAME.DTYPE],
                            center=self.get(NAME.CENTER, 0)),
                alignment=self.get(NAME.ALIGN, ALIGNMENT.DONTCARE))

        return self[NAME.FWDATA]

    @property
    def backwardData(self):
        if NAME.BWDATA not in self:
            # generate random array and set specific alignment style
            self[NAME.BWDATA] = arrAlign(
                arrTestDist(self[NAME.SHAPE_T], self[NAME.DTYPE],
                            center=self.get(NAME.CENTER, 0)),
                alignment=self.get(NAME.ALIGN, ALIGNMENT.DONTCARE))

        return self[NAME.BWDATA]

    def __call__(self):
        return self.forwardData

    def __str__(self):
        '''Compose a compact description of the represented array.'''
        tags = []

        # generate shape-token: check for both shape variants (fw and bw).
        # if they differ, print both as "fw/bw", otherwise print the dim only
        fwShape = self[NAME.SHAPE] if NAME.SHAPE in self else ()
        bwShape = self[NAME.SHAPE_T] if NAME.SHAPE_T in self else ()

        def printDim(fw, bw):
            if fw is None:
                return '-' if bw is None else str(bw)
            else:
                return (str(fw) if bw is None
                        else "%s/%s" %(fw, bw) if (fw != bw) else str(fw))

        strShape = 'x'.join([
            printDim(fw, bw)
            for fw, bw in six.moves.zip_longest(fwShape, bwShape)])
        if len(strShape) > 0:
            tags.append(strShape)

        # print the data type of the array
        value = self.get(NAME.DTYPE, '')
        if isinstance(value, type):
            tags.append(NAME.TYPENAME.get(value, str(value)))

        value = self.get(NAME.ALIGN, None)
        if value in NAME.ALLALIGNMENTS:
            tags.append(value)

        return str("[%s]" % (','.join(tags)))

    def __repr__(self):
        return self.__str__()

################################################################################
################################################## inspection routines


def showContent(instance, seen=None, prefix=""):
    '''
    Print a readable dependency tree of fastmat class instances

    Parameters
    ----------
    instance : :py:class:`fastmat.Matrix`
        Matrix instance to get inspected

    Notes
    -----
    The function outputs instance as top-level node and then walks through all
    elements of instance.content, recursively calling itself with extended
    prefix. To avoid endless loops, the function ensures the recursion is only
    applied once to every element by keeping track already visited elements.

    >>> showContent(Eye(4) * -1 * -1 + Eye(4) - Eye(4))
    <Sum[4x4]:0x7fe447f40770>
    +-<Product[4x4]:0x7fe447f3d188>
    | +-<Eye[4x4]:0x7fe447f402b0>
    +-<Eye[4x4]:0x7fe447f403e0>
    +-<Product[4x4]:0x7fe447f3da10>
      +-<Eye[4x4]:0x7fe447f40640>
    '''
    if seen is None:
        seen = set([])

    print((prefix[:-2] + "+-" if len(prefix) > 0 else "") + repr(instance))
    if instance not in seen:
        seen.add(instance)
        last = len(instance)
        for ii, item in enumerate(instance):
            showContent(item, seen=seen,
                        prefix=prefix + ("| " if ii < last - 1 else "  "))

################################################################################
################################################## CONSTANT definitions


################################################## class TypeDict
class TypeDict(dict):
    def __getitem__(self, key):
        if key not in self:
            key = key.type if isinstance(key, np.dtype) else None
        return dict(self).get(key, '???')


class COLOR():
    '''Give escape format strings (color, face) a name.'''
    END    = "\033[0m"
    BOLD   = "\033[1m"
    LINE   = "\033[4m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    PURPLE = "\033[95m"
    AQUA   = "\033[96m"


def fmtStr(string, color):
    '''Print a string quoted by some format specifiers.'''
    # colored output only supported with linux
    return ("%s%s%s" %(color, string, COLOR.END)
            if currentOS == 'Linux'
            else string)


def fmtGreen(string):
    '''Print string in green.'''
    return fmtStr(string, COLOR.GREEN)


def fmtRed(string):
    '''Print string in red.'''
    return fmtStr(string, COLOR.RED)


def fmtYellow(string):
    '''Print string in yellow.'''
    return fmtStr(string, COLOR.YELLOW)


def fmtBold(string):
    '''Print string in bold face.'''
    return fmtStr(string, COLOR.BOLD)


reAnsiEscape = None


def fmtEscape(string):
    '''Return a string with all ASCII escape sequences removed.'''
    global reAnsiEscape
    if reAnsiEscape is None:
        reAnsiEscape = re.compile(r'\x1b[^m]*m')

    return reAnsiEscape.sub('', string)


def dynFormat(s, *keys):
    return s.replace('%', '%%(%s)') % keys


################################################## getConsoleSize()

fallbackConsoleSize = (80, 25)
if (currentOS in ['Linux', 'Darwin']) or currentOS.startswith('CYGWIN'):
    import fcntl
    import termios
    # source: https://gist.github.com/jtriley/1108174

    def getConsoleSize():
        def ioctl_GWINSZ(fd):
            try:
                return struct.unpack(
                    'hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
            except EnvironmentError:
                return None

        cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
        if not cr:
            try:
                fd = os.open(os.ctermid(), os.O_RDONLY)
                cr = ioctl_GWINSZ(fd)
                os.close(fd)
            except EnvironmentError:
                pass

        if not cr:
            try:
                cr = (os.environ['LINES'], os.environ['COLUMNS'])
            except (EnvironmentError, KeyError):
                cr = fallbackConsoleSize

        return int(cr[0]), int(cr[1])
elif currentOS == 'Windows':
    def getConsoleSize():
        cr = fallbackConsoleSize
        try:
            from ctypes import windll, create_string_buffer
            # stdin handle is -10
            # stdout handle is -11
            # stderr handle is -12
            h = windll.kernel32.GetStdHandle(-12)
            csbi = create_string_buffer(22)
            res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
            if res:
                (left, top, right, bottom) = struct.unpack("10x4h4x", csbi.raw)
                cr = (right - left, bottom - top)
        except (ImportError, EnvironmentError):
            pass

        return cr
else:
    def getConsoleSize():
        return fallbackConsoleSize


################################################## worker's CONSTANT classes
class NAME(object):
    DATA            = 'data'
    FWDATA          = 'dataForward'
    BWDATA          = 'dataBackward'
    DTYPE           = 'dtype'
    SHAPE           = 'shape'
    SHAPE_T         = 'shapeBackward'
    ALIGN           = 'align'
    CENTER          = 'center'
    FORMAT          = 'format'
    NAME            = 'name'

    CLASS           = 'class'
    TARGET          = 'target'
    FILENAME        = 'filename'
    CAPTION         = 'caption'
    BENCHMARK       = 'bench'
    DOCU            = 'docu'
    TEST            = 'test'
    COMMON          = 'common'
    TEMPLATE        = 'template'
    VARIANT         = 'variant'
    RESULT          = 'result'
    HEADER          = 'header'

    TYPENAME        = TypeDict({
        np.int8:       'i08', np.int32:      'i32', np.int64:      'i64',
        np.float32:    'f32', np.float64:    'f64',
        np.complex64:  'c32', np.complex128: 'c64', None:          '???'
    })

    ALLTYPES        = [np.int8, np.int32, np.int64,
                       np.float32, np.float64,
                       np.complex64, np.complex128]
    FEWTYPES        = [np.int8, np.int32, np.float32, np.complex64]

    SINGLETYPES     = [np.int32, np.float32, np.complex64]
    DOUBLETYPES     = [np.int64, np.float64, np.complex128]

    INTTYPES        = [np.int8, np.int32, np.int64]
    FLOATTYPES      = [np.float32, np.float64]
    COMPLEXTYPES    = [np.complex64, np.complex128]

    LARGETYPES      = [np.int32, np.int64, np.float32, np.float64,
                       np.complex64, np.complex128]

    ALLALIGNMENTS   = [ALIGNMENT.FCONT, ALIGNMENT.CCONT, ALIGNMENT.STRIDE]

    # repeat some of the definitions in this unit to allow compact imports
    ALIGNMENT       = ALIGNMENT
    ArrayGenerator  = ArrayGenerator
    Permutation     = Permutation
    IgnoreFunc      = IgnoreFunc


################################################################################
################################################## class Runner
class Worker(object):
    '''
    options - dictionary structure containing options for multiple targets.
        {   'nameOfTarget': {'parameter': 123,
                                'anotherParameter': 456
         }, NAME.DEFAULTS:  {'parameter': default parameter unless overwritten
                                          within the selected target
         }}

    results - output of each target's as specified in options
    '''

    cbStatus=None
    cbResult=None
    target=None

    def __init__(self, targetClass, **options):
        '''
        Setup an inspection environment on a fastmat class specified in target.
        Along the way an empty instance of target will be created and aside
        the default options, which may be specified in runnerDefaults, an
        arbitrarily named method of target may be specified to return more
        specific options when called.
        extraOptions may be specified to overwrite any parameter with highest
        priority.
        Both extraOptions and runnerDefaults must be specified as a two-level
        dictionary with the outer level specifying target names as keys and the
        inner level the actual parameter for the target.
        '''
        # the test target is a fastmat class to be instantiated in the runners
        if not inspect.isclass(targetClass):
            raise ValueError("target in init of Runner must be a class type")

        # set defaults for options
        targetOptionMethod = options.get('targetOptionMethod', None)
        runnerDefaults = options.get('runnerDefaults', {})
        extraOptions = options.get('extraOptions', {})

        self.target=targetClass.__new__(targetClass)

        # targetOptionMethod specifies a method name of target which will
        # return a dictionary with class-specific options if called. If this
        # functionality is not needed, targetOptionsMethod or runnerDefaults
        # may be left to their default values.
        targetOptionMethod=getattr(self.target, targetOptionMethod, None)
        options={name: mergeDicts(target, extraOptions)
                 for name, target in ({} if targetOptionMethod is None
                                      else targetOptionMethod()).items()}

        # determine keys for final output
        keys=[name for name in options.keys() if name != NAME.COMMON]

        # start with the defaults for the selected keys as a baseline
        common=runnerDefaults.get(NAME.COMMON, {})
        self.options={name: mergeDicts(common, runnerDefaults.get(name, {}))
                      for name in keys}

        # a NAME.TEMPLATE parameter in a target will cause the specified
        # target to be extend by the default options of another target.
        # Priority order: extraOptions > options > defaults > COMMON
        for name, target in self.options.items():
            template=(options[name][NAME.TEMPLATE]
                      if name in options and NAME.TEMPLATE in options[name]
                      else (runnerDefaults[name][NAME.TEMPLATE]
                            if (name in runnerDefaults and
                                NAME.TEMPLATE in runnerDefaults[name])
                            else None))

            if template is not None:
                target.update(
                    runnerDefaults.get(template, {}))

        # now add our specific options from options and extraoptions
        common=options.get(NAME.COMMON, {})
        self.options={name: mergeDicts(self.options.get(name, {}),
                                       mergeDicts(common, options[name]))
                      for name in keys}

        # finally, tag each target with its target name so that parameter links
        # may address it over the key [NAME.TARGET]. Also, write down the name
        # of the class in [NAME.CLASS]
        for name, target in self.options.items():
            target[NAME.TARGET]=name
            target[NAME.CLASS]=targetClass.__name__

        # initialize output
        self.results=AccessDict({})

    def emitStatus(self, *args):
        if self.cbStatus is not None:
            self.cbStatus(*args)

    def emitResult(self, *args):
        if self.cbResult is not None:
            self.cbResult(*args)

    def run(self, *targetNames, **extraArgs):
        '''
        Execute all selected targets by a list of targetNames.
        If *targetNames is empty all targets will be run.
        The output of each :math:`TARGET` will be written to
        self.results[:math:`TARGET`]
        '''
        # determine console width
        self.consoleWidth=getConsoleSize()[1]

        if len(targetNames) == 0:
            targetNames=self.options.keys()

        targets={name: self.options[name]
                 for name in targetNames if name in self.options}

        for name, target in sorted(targets.items()):
            options=target.copy()

            if len(extraArgs) > 0:
                options.update(extraArgs)

            result = self._run(name, options)

            # make all result dicts of type accessDict for easy access
            self.results[name] = convertToAccessDicts(result)
            if self.cbResult is not None:
                self.cbResult(name, result)
