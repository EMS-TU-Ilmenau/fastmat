# -*- coding: utf8 -*-
'''
  demo/compOmpIsta.py
 -------------------------------------------------- part of the fastmat demos


  Author      : wcw
  Introduced  :
 ------------------------------------------------------------------------------

   Copyright 2016 Sebastian Semper, Christoph Wagner
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

 ------------------------------------------------------------------------------
'''

import sys                                        # system calls
import time                                       # measuring time

import numpy                                      # math library
import scipy.sparse                               # math library


################################################## import modules
try:
    import matplotlib.pyplot as plt               # plotting
    import matplotlib.gridspec as gridspec        # grid specification
except ImportError:
    print("matplotlib not found. Please consider installing it to proceed.")
    sys.exit(0)

try:
    import fastmat
except ImportError:
    sys.path.append('..')
    import fastmat


################################################## helper functions
def printTime(note, call, *args, **kwargs):
    '''Measure runtime of call and print it to stdout.'''
    sys.stdout.write(" * %s ... " % (note))
    sys.stdout.flush()

    # measure runtime of call, print it and return result
    t = time.time()
    result = call(*args, **kwargs)
    print("%.2e s" % (time.time() - t))
    return result


def plotComplex(plot, vec):
    '''Plot real and imaginary part of vector concatenated along x'''
    plot.axes.get_xaxis().set_visible(False)
    plot.imshow(
        numpy.concatenate([vec.real, vec.imag], axis=1),
        interpolation='none',
        aspect='auto'
    )

################################################## script


print("fastmat demo: Sparse-Signal-Recovery Demo using OMP and ISTA")
print("------------------------------------------------------------")

# define some constants
N = 100        # number of measurements (height of dictionary)
M = 500        # width of dictionary
K = 30        # sparsity (non-zero components in support)

# define baseline random support (1.0 forces dtype to be float)
s = scipy.sparse.rand(M, 1, 1.0 * K / M).todense()
x0 = s - 1j * s

# define some dictionary
# (dictionary matrix holds first N entries of M-Fourier matrix)
# option to choose a complex one or a non-complex one
matG = fastmat.Matrix(
        numpy.random.normal(0.0, numpy.sqrt(N), (N, M)) + \
        1j * fastmat.Matrix(numpy.random.normal(0.0, numpy.sqrt(N), (N, M)))
    )

# option 1
mat = fastmat.Product(
    matG,
    fastmat.Fourier(M)
)

# option 2
#mat = matG

# generate measurements from baseline support
b = mat * x0

# run OMP and ISTA
# result = fastmat.OMP(mat, b, K)
xOMP = printTime("running OMP",
                 fastmat.algs.OMP, mat, b, K)
# result = fastmat.ISTA(mat, b, numLambda=1e4)
xISTA = printTime("running ISTA",
                  fastmat.algs.ISTA, mat, b,
                  numLambda = 1e6, numMaxSteps = 1000)

# plot:
fig = plt.figure()
grid = gridspec.GridSpec(
    2, 3,
    height_ratios=[1, 7],
    width_ratios=[M / (2 * N), M / (2 * N), 1])

# plot dictionary. as it is a partial Fourier, only plot element phases
plot = fig.add_subplot(grid[0, :-1], title='Dictionary matrix').imshow(
    numpy.angle(mat.array),
    interpolation='none', aspect='auto')

# plot measurements next to matrix
plotComplex(fig.add_subplot(grid[0, -1], title='Measurements'), b)

# plot support vectors (baseline left, OMP center, ISTA right)
plotComplex(fig.add_subplot(grid[1, 0], title='OMP (real|imag)'), xOMP)
plotComplex(fig.add_subplot(grid[1, 1], title='ISTA (real|imag)'), xISTA)
plotComplex(fig.add_subplot(grid[1, 2], title='baseline'), x0)

# set title of whole figure
fig.suptitle(
    "Sparse Signal Recovery Demo with OMP and ISTA using fastmat",
    fontweight='bold')

# arrange tightly and plot
grid.tight_layout(fig)
plt.show()
