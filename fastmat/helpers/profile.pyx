# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False, nonecheck=False
'''
  fastmat/helpers/profile.pyx
 -------------------------------------------------- part of the fastmat package

  Lean call wrappers for fast profiling of python or cython code


  Author      : wcw
  Introduced  : 2016-09-25
 ------------------------------------------------------------------------------

   Copyright 2016 Sebastian Semper, Christoph Wagner

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
import timeit

from .types cimport *


def timeReps(reps, method, *args):
    '''
    wrapper for measuring the runtime of 'method' by averaging the runtime
    of many repeated calls.

        reps        - number of repetitions to be averaged over
        method      - pointer to the evaluatee method
        *args       - a list of parameters for the callee
    '''
    cdef object arg1
    cdef object arg2
    cdef intsize N

    N = 1 if reps < 1 else reps

    def _inner():
        cdef intsize ii
        for _ii in range(N):
            method(*args)

    def _inner1():
        cdef intsize ii
        for _ii in range(N):
            method(arg1)

    def _inner2():
        cdef intsize ii
        for _ii in range(N):
            method(arg1, arg2)

    if len(args) == 1:
        arg1 = args[0]
        runtime = timeit.timeit(_inner1, number=1)
    elif len(args) == 2:
        arg1 = args[0]
        arg2 = args[1]
        runtime = timeit.timeit(_inner2, number=1)
    else:
        _innerArgs = args
        runtime = timeit.timeit(_inner, number=1)

    # return results
    return {
        'avg': runtime / reps,
        'time': runtime,
        'cnt': reps
    }
