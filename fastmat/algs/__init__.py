# -*- coding: utf-8 -*-
'''
  fastmat/algs/__init__.py
 -------------------------------------------------- part of the fastmat demos


  Author      : wcw
  Introduced  :
 ------------------------------------------------------------------------------

   Copyright 2016 Sebastian Semper, Christoph Wagner
       https://www.tu-ilmenau.de/ems/

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

 ------------------------------------------------------------------------------
'''
# algorithms for sparse recovery
from .ISTA import ISTA, ISTAinspect
from .OMP import OMP, OMPinspect

# numeric algorithms
from .CG import CG, CGinspect
