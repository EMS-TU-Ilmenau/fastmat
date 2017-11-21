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

r"""
fastmat.algs
============

We provide some algorithms, which make use of the speedups provided by fastmat
to allow easy production use out of the box.

Algorithms
----------

.. automodule:: fastmat.algs.ISTA
    :members:
    :undoc-members:

.. automodule:: fastmat.algs.OMP
    :members:
    :undoc-members:

.. automodule:: fastmat.algs.FISTA
    :members:
    :undoc-members:

.. automodule:: fastmat.algs.CG
    :members:
    :undoc-members:


"""
# algorithms for sparse recovery
from .ISTA import ISTA, ISTAinspect
from .OMP import OMP, OMPinspect
from .FISTA import FISTA, FISTAinspect
# numeric algorithms
from .CG import CG, CGinspect
