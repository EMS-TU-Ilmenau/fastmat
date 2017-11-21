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

import sys
import time

# import numpy functionality
import numpy.random as npr
import numpy.linalg as npl
import numpy as np

################################################## import modules
try:
    import fastmat
except ImportError:
    sys.path.append('..')
    import fastmat

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

print("fastmat demo: Approximation of an almost low rank matrix with LowRank()")
print("-----------------------------------------------------------------------")

# calc the number of dimensions to approximate
numSize = int(numApproxQual * numMatSize)

# draw a dense random and normalized array
print(" * Generate the Matrix")
arrFull = npr.randn(numMatSize, numMatSize) / np.sqrt(numMatSize)

# get the SVD
print(" * Calc the SVD")
arrU, vecSigma, arrV = npl.svd(arrFull)

print(" * Truncate the singular values")
vecT = np.linspace(0, 3, numMatSize - numSize - 1)
vecSigma[numSize + 1:] = vecSigma[numSize + 1:] * 0.1 * np.exp(-vecT)

print(" * Rebuild the matrix with truncated singular values")
arrFull = arrU.dot( np.diag(vecSigma).dot(arrV.T) )

matFull = fastmat.Matrix(arrFull)
matApprox = fastmat.LowRank(
    vecSigma[:numSize],arrU[:,:numSize],arrV[:,:numSize]
)

print(" * Generate the linear system")
vecX = npr.randn(numMatSize)
vecB = matFull * vecX

s = time.time()
y1 = matFull * vecX
timeDenseFwd = time.time() - s

s = time.time()
y2 = matApprox * vecX
timeApproxFwd = time.time() - s

s = time.time()
x1 = fastmat.algs.CG(matFull,vecB)
timeDenseSolve = time.time() - s

s = time.time()
x2 = fastmat.algs.CG(matApprox,vecB)
timeApproxSolve = time.time() - s

numApproxErr1 = npl.norm(x1 - x2) / npl.norm(x1)
numApproxErr2 = npl.norm(matFull * x1 - matFull * x2) / npl.norm(vecB)

################################################################################
#                               OUTPUT SECTION
################################################################################
print("\nResults:")
print("   Dense Multiplication             %14.3f ms" %(1e3 * timeDenseFwd))
print("   Approximated Multiplication      %14.3f ms" %(1e3 * timeApproxFwd))
print("   Solve Dense Linear System        %14.3f ms" %(1e3 * timeDenseSolve))
print("   Solve Approximated Linear System %14.3f ms" %(1e3 * timeApproxSolve))
print("   Relative Error in Coefficient Domain %10.3f" %(numApproxErr1))
print("   Relative Error in Signal Domain      %10.3f" %(numApproxErr2))
