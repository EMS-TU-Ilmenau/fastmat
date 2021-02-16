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

############################
  Testing and Benchmarking
############################

.. toctree::
   :glob:
   :hidden:

   inspection/*

In this section we will show two built-in systems that allow you to
:ref:`test<testing>` and :ref:`benchmarking<benchmarking>` any fastmat Matrix
implementation. Testing is important to verify that a certain implementation
actually does what you'd expect of it and is virtually *the* essential
cornerstone to writing your own user defined matrix class implementation.

The scope of the implemented testing system expands to these areas
(not complete):

  * Testing of mathematical correctness
  * Testing of computational accuracy
  * Testing of various data types (and combinations thereof)
  * Testing of parameter combinations
  * Testing for different platforms ans versions (mostly done by our CI setup)

Benchmarking, however, is a valuable goodie to have in your toolbox to evaluate
the performance of your implementation and to find culprits that impact the
runtime or memory consumption of your implementation. Make sure to also tune in
to out :ref:`optimization<optimization>` section, where we detail on how to use
the information gained from benchmarking your classes productively to improve
your implementation.
