# -*- coding: utf-8 -*-
'''
  fastmat/__init__.py
 -------------------------------------------------- part of the fastmat demos


  Author      : wcw
  Introduced  :
 ------------------------------------------------------------------------------

   Copyright 2016 Sebastian Semper, Christoph Wagner

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

# import fundamental types and classes first, also behavioural flags
from .Matrix import Matrix, Hermitian, Conjugate, Transpose, flags
from .Product import Product

# import all fastmat classes
from .BlockDiag import BlockDiag
from .Blocks import Blocks
from .Circulant import Circulant
from .Diag import Diag
from .Eye import Eye
from .Fourier import Fourier
from .Hadamard import Hadamard
from .Kron import Kron
from .LowRank import LowRank
from .Outer import Outer
from .Parametric import Parametric
from .Partial import Partial
from .Permutation import Permutation
from .Polynomial import Polynomial
from .Sparse import Sparse
from .Sum import Sum
from .Toeplitz import Toeplitz
from .Zero import Zero

# new stuff
from .LFSRCirculant import LFSRCirculant

# import algorithms subpackage
from .base import Algorithm
from . import algs


# define package version (gets overwritten by setup script)
from .version import __version__

# compile a list of all matrix classes offered by the package


def isClass(item):
    try:
        return issubclass(item, Matrix)
    except TypeError:
        return False


classes = list(item for item in locals().values() if isClass(item))
