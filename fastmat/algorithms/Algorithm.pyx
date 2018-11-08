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

################################################################################
################################################## class Algorithm
cdef class Algorithm(object):
    r"""Algorithm Base Class


    **Description:**
    The baseclass of all algorithms that operate on Matrices. This abstract
    baseclass introduces general framework concepts such as interfaces for
    parameter specification, algorithm execution, logging and callbacks.

    DEMO:
    >>> algI = fma.ISTA(Fourier(10))
    >>> algI.cbResult = lambda i: print(i.arrResult)
    >>> algI.cbStep = lambda i: print(i.numStep)
    >>> algI.cbTrace = fma.Algorithm.snapshot
    >>> algI.process(np.ones(10) + np.random.randn(10))
    >>> plt.imshow(np.hstack((np.abs(tt.arrX) for tt in algI.trace)))
    >>> plt.show()
    """

    trace = []

    cbTrace = None
    cbResult = None

    def __init__(self):
        raise NotImplementedError("Algorithm baseclass cannot be instantiated.")

    def updateParameters(self, **kwargs):
        r"""
        Update the parameters of the algorithm instance with **kwargs.

        Apply the set of parameters specified in kwargs by iteratively passing
        them to setattr(self, ...). Specifying an parameter which does not have
        a mathing attribute in the algorithm class will cause an AttributeError
        to be raised.
        """
        for key, value in kwargs.items():
            if not hasattr(self, key):
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

    cpdef _process(self, np.ndarray arrB):
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
        trace, self.trace = self.trace, []
        trace.append(copy(self))
        self.trace = trace

    cpdef handleCallback(self, callback):
        r"""
        Call the callback if it is not None.
        """
        if callback is not None:
            return callback(self)

        return None
