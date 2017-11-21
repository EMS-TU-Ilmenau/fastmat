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

import sys
from itertools import chain
from collections import deque
import numpy as np
from scipy.sparse import spmatrix
from ..Matrix import Matrix

################################################## getMemoryFootprint()


def getMemoryFootprint(obj, **options):
    '''
    Return the approximate memory footprint of an object with all of its
    contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
            OtherContainerClass: OtherContainerClass.get_elements}
    '''
    # extract options
    handlers = options.get('handlers', {})
    verbose = options.get('verbose', False)

    # keep track of object ids already seen
    seen = set()

    # estimate size of objects without __sizeof__()
    default_size = sys.getsizeof(0)

    # setup handlers for various data types
    def dict_handler(d):
        return chain.from_iterable(d.items())

    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    all_handlers.update(handlers)

    # walker for getting size of object considering corner and special cases
    def sizeof(obj):
        # skip objects already seen once
        if id(obj) in seen:
            return 0

        seen.add(id(obj))

        s = sys.getsizeof(obj, default_size)

        # check fastmat matrices:
        # only cdefs with explicit `public` tag will be seen here !
        if isinstance(obj, Matrix):
            for key in dir(obj):
                if key[0] == '_' and key[1] != '_':
                    item = getattr(obj, key)
                    if not callable(item) and (item is not None):
                        s += sizeof(item)

        # check for ndarrays (have special properties holding data)
        if isinstance(obj, np.ndarray):
            if obj.base is not None:
                # add memory size of base (if not yet added)
                # some numpy versions don't report properly to getsizeof()
                added = sizeof(obj.base)
                return s + added if added > s else s

            # some numpy versions don't report properly to getsizeof()
            added = obj.nbytes
            s += added if added > s else 0

        # check for sparse arrays (have special properties holding data)
        if isinstance(obj, spmatrix):
            s += sizeof(obj.__dict__)

        if verbose:
            print("==%d, [%s@%x]" %(s, type(obj), id(obj)))

        # check for known container types
        for typ, handler in all_handlers.items():
            if isinstance(obj, typ):
                size = sum(map(sizeof, handler(obj)))
                s += size
                break

        return s

    return sizeof(obj)
