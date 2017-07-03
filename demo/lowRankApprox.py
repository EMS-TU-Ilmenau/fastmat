# -*- coding: utf-8 -*-
'''
  demo/lowRankApprox.py
 -------------------------------------------------- part of the fastmat demos

  Demonstration how to do low rank approximations of dense matrices
  that have a rapidly decaying spectrum.

  Author      : sempersn
  Introduced  : 2016-09-28
 ------------------------------------------------------------------------------
  PARAMETERS:
    numMatSize      - size of the system matrix
    numApproxQual   - approximation quality in [0,1]

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

import sys
import time

# import numpy functionality
import numpy.random as npr
import numpy.linalg as npl
import numpy as np

# import fastmat, try as global package or locally from one floor up
try:
    import fastmat
except ImportError:
    sys.path.insert(0, '..')
    import fastmat

# import smooth printing routines
sys.path.insert(0, '../util')
from routines.printing import frameLine, frameText, printTitle

################################################################################
#                          PARAMETER SECTION
################################################################################
# please edit these parameters as you please, but be cautious with the problem
# size, because the traditional method might fail or take a long time
################################################################################
numMatSize = 2 ** 9
numApproxQual = 0.3

################################################################################
#                          CALCULATION SECTION
################################################################################

printTitle(
    "Approximation of an almost low rank matrix using the LowRank class",
    width=80
)

# calc the number of dimensions to approximate
numSize = int(numApproxQual * numMatSize)

# draw a dense random and normalized array
frameText(" Generate the Matrix")
arrFull = npr.randn(numMatSize, numMatSize) / np.sqrt(numMatSize)

# get the SVD
frameText(" Calc the SVD")
arrU, vecSigma, arrV = npl.svd(arrFull)

frameText(" Truncate the singular values")
vecT = np.linspace(0, 3, numMatSize - numSize - 1)
vecSigma[numSize + 1:] = vecSigma[numSize + 1:] * 0.1 * np.exp(-vecT)

frameText(" Rebuild the matrix with truncated singular values")
arrFull = arrU.dot( np.diag(vecSigma).dot(arrV.T) )

matFull = fastmat.Matrix(arrFull)
matApprox = fastmat.LowRank(
    vecSigma[:numSize],arrU[:,:numSize],arrV[:,:numSize]
)

frameText(" Generate the linear system")
vecX = npr.randn(numMatSize)
vecB = matFull * vecX

s = time.time()
y1 = matFull * vecX
numDenseForwardTime = time.time() - s

s = time.time()
y2 = matApprox * vecX
numApproxForwardTime = time.time() - s

s = time.time()
x1 = fastmat.algs.CG(matFull,vecB)
numDenseSolveTime = time.time() - s

s = time.time()
x2 = fastmat.algs.CG(matApprox,vecB)
numApproxSolveTime = time.time() - s

numApproxErr1 = npl.norm(x1 - x2) / npl.norm(x1)
numApproxErr2 = npl.norm(matFull * x1 - matFull * x2) / npl.norm(vecB)

################################################################################
#                               OUTPUT SECTION
################################################################################
printTitle("RESULTS", width=80)
frameText(" Dense Multiplication             % 10.3f ms" %
          (1000 * numDenseForwardTime))
frameText(" Approximated Multiplication      % 10.3f ms" %
          (1000 * numApproxForwardTime))
frameText(" Solve Dense Linear System        % 10.3f ms" %
          (1000 * numDenseSolveTime))
frameText(" Solve Approximated Linear System % 10.3f ms" %
          (1000 * numApproxSolveTime))
frameText(" Relative Error in Coefficient Domain      % 10.3f" %
          (numApproxErr1))
frameText(" Relative Error in Signal Domain           % 10.3f" %
          (numApproxErr2))
printTitle("K THX, BYE", width=80)

#import matplotlib.pyplot as plt
#plt.plot(vecSigma)
#plt.show()
#plt.plot(x1)
#plt.plot(x2)
#plt.show()
#plt.plot(x1)
#plt.plot(x2)
#plt.show()
