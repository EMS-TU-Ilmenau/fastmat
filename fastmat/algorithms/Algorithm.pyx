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
cimport numpy as np

from copy import copy
#from cpython.object cimport PyObject_GenericSetAttr
from ..core.resource import getMemoryFootprint

################################################################################
################################################## class Algorithm
cdef class Algorithm(object):
    r"""Algorithm Base Class

    The baseclass of all algorithms that operate on Matrices. This abstract
    baseclass introduces general framework concepts such as interfaces for
    parameter specification, algorithm execution, logging and callbacks.

    >>> algI = fma.ISTA(Fourier(10))
    >>> algI.cbResult = lambda i: print(i.arrResult)
    >>> algI.cbStep = lambda i: print(i.numStep)
    >>> algI.cbTrace = fma.Algorithm.snapshot
    >>> algI.process(np.ones(10) + np.random.randn(10))
    >>> plt.imshow(np.hstack((np.abs(tt.arrX) for tt in algI.trace)))
    >>> plt.show()
    """

    def __init__(self):
        if type(self) is Algorithm:
            raise NotImplementedError(
                "Algorithm baseclass cannot be instantiated."
            )

    property cbTrace:
        def __get__(self):
            return self._cbTrace

        def __set__(self, value):
            self._cbTrace = value

    property cbResult:
        def __get__(self):
            return self._cbResult

        def __set__(self, value):
            self._cbResult = value

    property trace:
        def __get__(self):
            if self._trace is None:
                self._trace = []

            return self._trace

        def __set__(self, value):
            if not isinstance(value, list):
                raise TypeError("Algorithm trace must be a list")

            self._trace = value

    ############################################## algorithm resource handling
    # nbytes - Property(read)
    # Size of the Matrix object
    property nbytes:
        # r"""Number of bytes in memory used by this instance
        # """
        def __get__(self):
            return getMemoryFootprint(self)

#    def __setattr__(self, attr, val):
#        try:
#            PyObject_GenericSetAttr(self, attr, val)
#        except AttributeError:
#            if self.__dict__ is None:
#                self.__dict__ = {}
#
#            self.__dict__[attr] = val
#            print("bla")

    def updateParameters(self, **kwargs):
        r"""
        Update the parameters of the algorithm instance with the supllied
        keyworded arguments.

        Apply the set of parameters specified in kwargs by iteratively passing
        them to setattr(self, ...). Specifying an parameter which does not have
        a mathing attribute in the algorithm class will cause an AttributeError
        to be raised.
        """
        for key, value in kwargs.items():
            if not hasattr(self, key) and (self._attributes is not None and
                                           key not in self._attributes):
                raise AttributeError(
                    "Attribute '%s' not defined in %s" %(
                        key, self.__class__.__name__)
                )

            setattr(self, key, value)

    def process(self, np.ndarray arrB, **kwargs):
        r"""
        Process an array of data by the algorithm.

        This method also accepts passing additional parameters as keyworded
        arguments. These arguments will be applied to the algorithm instance
        using self.updateParameters().

        If no additional parameters are required the self._process() method
        may also be called directly for slightly higher call performance.
        """
        self.updateParameters(**kwargs)

        # perform the actual computation and service the result callback
        arrResult = self._process(arrB)
        self.handleCallback(self.cbResult)

        return arrResult

    cpdef np.ndarray _process(self, np.ndarray arrB):
        r"""
        Process an array of data by the algorithm.

        Please check out self.process() for further information.
        """
        # Raise an not implemented Error as this is an (abstract) baseclass.
        raise NotImplementedError("Algorithm is not implemented yet.")

    cpdef snapshot(self):
        r"""
        Add the current instances' state (without the trace) to the trace.
        """
        # temporarily remove the trace from the object to allow copying
        # without circular references. Then put the trace back in
        cdef list trace
        trace, self._trace = self._trace, []
        if trace is None:
            trace = []

        trace.append(copy(self))
        self._trace = trace

    cpdef handleCallback(self, callback):
        r"""
        Call the callback if it is not None.
        """
        if callback is not None:
            return callback(self)

        return None
