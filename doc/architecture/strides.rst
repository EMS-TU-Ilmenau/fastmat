..  Copyright 2016 Sebastian Semper, Christoph Wagner
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

.. _strides:

Low-Overhead Array Striding Interface
=====================================

Fastmat offers a special C-level interface allowing the creation, processing
and manipulation of views into the underlying data of :py:class:`numpy.ndarray`
objects without the overhead of creating view or memoryview objects of that
array object. As the implementation is based on C structures, no interaction
with the python object interface is necessary, thus increasing the efficiency
of advanced linear operators from within cython code. By mimimizing memory
operations occuring during view creation, structure- or object allocation or
copying, this helps minimizing the already low overhead on using cython
memoryviews further.


The main container for defining and using strides is the ``STRIDE_s`` structure:

.. code-block:: cython

    ctypedef struct STRIDE_s:
        char *          base
        intsize         strideElement
        intsize         strideVector
        intsize         numElements
        intsize         numVectors
        np.uint8_t      sizeItem
        ftype           dtype

:ref:`fastmat Type Identifier<ftype>`

The striding interface supports:
  * Two-dimensional :py:class:`numpy.ndarray` objects
  * Non-contiguous (striding) access into the data
  * Modifying views (substriding

``fastmat.core.strides``
------------------------

.. automodule:: fastmat.core.strides
    :members:
    :undoc-members:
    :show-inheritance:
