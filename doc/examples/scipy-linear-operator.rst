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


.. _example-scipy-linear-operator:

Solve a System of Linear Equations with Preconditioning
=======================================================

The preconditioner used for solving can also be provided as a
``LinearOperator``.

.. code-block:: python

    import fastmat as fm
    import numpy as np
    from scipy.sparse.linalg import cgs
    # diagonal matrix with no zeros
    d = np.random.uniform(1, 20, 2 ** 10)

    # fastmat object
    H = fm.Diag(d)

    # use the new property to generate a scipy linear operator
    Hs = H.scipyLinearOperator

    # also generate a Preconditioning linear operator,
    # which in this case is the exact inverse
    Ms = fm.Diag(1.0 / d).scipyLinearOperator

    # get a baseline
    x = np.random.uniform(1, 20, 2 ** 10)
    y = np.linalg.solve(H.array, x)
    cgs(Hs, x, tol=1e-10)
    cgs(Hs, x, tol=1e-10, M=Ms)
