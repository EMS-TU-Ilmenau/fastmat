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
fastmat.algorithms
============

We provide some algorithms, which make use of the speedups provided by fastmat
to allow easy production use out of the box.

Algorithms
----------

.. automodule:: fastmat.algorithms.ISTA
    :members:
    :undoc-members:

.. automodule:: fastmat.algorithms.FISTA
    :members:
    :undoc-members:

.. automodule:: fastmat.algorithms.OMP
    :members:
    :undoc-members:

.. automodule:: fastmat.algorithms.STELA
    :members:
    :undoc-members:

"""
# algorithms for sparse recovery
from .Algorithm import Algorithm
from .ISTA import ISTA
from .OMP import OMP
from .FISTA import FISTA
from .STELA import STELA