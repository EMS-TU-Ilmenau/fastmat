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

################################################################################
################################################## class Algorithm
cdef class Algorithm(object):
    r"""Algorithm Base Class


    **Description:**
    The baseclass of all algorithms that operate on Matrices. This abstract
    baseclass introduces general framework concepts such as interfaces for
    parameter specification, algorithm execution, logging and callbacks.
    """

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
        self.updateParameters(**kwargs);
        return self._process(arrB)


    cpdef _process(self, np.ndarray arrB):
        r"""
        Process an array of data by the algorithm.

        Please check out self.process() for further information.
        """
        # Raise an not implemented Error as this is an (abstract) baseclass.
        raise NotImplementedError("Algorithm is not implemented yet.")

