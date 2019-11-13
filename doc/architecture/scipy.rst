..  Copyright 2018 Sebastian Semper, Christoph Wagner
        https://www.tu-ilmenau.de/it-ems/

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

.. _scipy:

SciPy Interface
===============

Motivation
----------

SciPy offers a large zoo of algorithms which exploit the possibility to pass a
so called LinearOperator, which only provides methods for forward and backward
transforms together with some simple properties like a datatype and shape
parameters. This is exactly what we can provide for a specific instance of a
fastmat Matrix. To this end, each fastmat Matrix has the (read only) property
scipyLinearOperator, which provides a SciPy Linear operator realizing the
transform specified by the fastmat object.

This allows to combine fastmat and SciPy in the most efficient manner possible.
Here, fastmat provides the simple and efficient description of a huge variety
of linear operators, which can then be used neatly and trouble free in SciPy.

Usage
-----

The interface is provided by the factory
:py:meth:`fastmat.Matrix.scipyLinearOperator`, which returns an instance of
:py:class:`scipy.sparse.linalg.LinearOperator`. All fastmat matrices can be
used from `scipy` methods supporting this interface, offering a wide range of
functionality -- by combining both worlds -- to the user.

For a code example, see :ref:`example-scipy-linear-operator`.
