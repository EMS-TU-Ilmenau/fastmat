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

r"""
In many fields of engineering linear transforms play a key role during modeling
and solving real world problems. Often these linear transforms have an inherent
structure which reduces the degrees of freedom in their parametrization.
Moreover, this structure allows to describe the action of a linear mapping on a
given vector more efficiently than the general one.

This structure can be exploited twofold. First, the storage of these transforms
in form of matrices, on computers normally an array of numbers in
:math:`\mathbb{C}` or :math:`\mathbb{R}`, might be unnecessary.
So for each structure there is a more concise way of representation, which
leads to a benefit in memory consumption when using these linear transforms.
Second, the structure allows more efficient calculations when applying the
linear transform to a vector. This may result in a drop in algorithmic
complexity which implies that computing time can be saved.

Still, these structural benefits have to be exploited and it is not often easy
to accomplish this in a save and reuseable way. Moreover, in applications you
often think of the linear transforms as a matrix and your way of working with
it is streamlined to this way of thinking, which is only natural, but does not
directly allow to exploit the structure.

So, there are different ways of thinking in what is natural and in what is
efficient. This is the gap fastmat tries to bridge by allowing you to work with
the provided objects as if they were common matrices represented as arrays of
numbers, while the algorithms that make up the internals are highly adapted to
the specific structure at hand. It provides you with a set of tools to work
with linear transforms while hiding the algorithmic complexity and exposing the
benefits in memory and calculation efficiency without too much overhead.

This way you can worry about really urgent matters to you, like research and
development of algorithms and leave the internals to fastmat.
"""

# import fundamental types and classes first, also behavioural flags
from .Matrix import Matrix, Hermitian, Conjugate, Transpose, flags
from .Matrix import Inverse, PseudoInverse
from .Product import Product

# import all fastmat classes
from .BlockDiag import BlockDiag
from .Blocks import Blocks
from .Circulant import Circulant
from .Diag import Diag
from .DiagBlocks import DiagBlocks
from .Eye import Eye
from .Fourier import Fourier
from .Hadamard import Hadamard
from .Kron import Kron
from .LFSRCirculant import LFSRCirculant
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

# define package version (gets overwritten by setup script)
from .version import __version__

# compile a list of all matrix classes offered by the package


def isClass(item):
    try:
        return issubclass(item, Matrix)
    except TypeError:
        return False


classes = list(item for item in locals().values() if isClass(item))
