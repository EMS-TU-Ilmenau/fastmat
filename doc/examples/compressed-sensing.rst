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

.. highlight:: py

.. role:: python(code)
   :language: py

.. _example-compressed-sensing:


Compressed Sensing example
==========================

We set up a linear forward model using a Fourier matrix as dictionary and
reconstruct the underlying sparse vector from linear projections using a matrix
with elements drawn randomly from a Gaussian distribution.

.. plot:: examples/compressed-sensing.py
  :include-source:
