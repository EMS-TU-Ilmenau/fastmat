# -*- coding: utf-8 -*-
'''
  demo/matrixSAFT.py
 -------------------------------------------------- part of the fastmat demos

  Demonstration on how to use fastMat for SAFT.


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
import sys
from timeit import timeit

# import numpy functionality
import numpy as np
import scipy.io as sio
import scipy.sparse as sps

################################################## import modules
try:
    from matplotlib.pyplot import *               # plotting
except ImportError:
    print("matplotlib not found. Please consider installing it to proceed.")
    sys.exit(0)

try:
    import fastmat
except ImportError:
    sys.path.append('..')
    import fastmat


################################################## import cython-optimized core
import pyximport; pyximport.install()
from matrixSAFTmask import SaftMaskCore


###############################################################################
# SAFT demo
###############################################################################
print("fastmat demo: Synthetic Aperture Focussing Technique (SAFT)")
print("-----------------------------------------------------------")

# x - Parameter für die Matrix (Unterscheidung)
# a - Rekovektor
# b -  Messvektor
# s_x - Spezifisches X
# j - Zeilenindex Matrix - Zeitrichtung (numN)
# dx - Schrittweite in Breite (M) (numD)
# dz - Schrittweite in Tiefe (N) (numS)


refData = sio.loadmat('datenSAFT.mat')
dX = refData['dx'] * 1000
dZ = refData['dz'] * 1000
numN        = 2040
numK        = 50
numD        = 1000 if numN == 2040 else numN / 2

for key, value in {
    'Samples in time dimension (numN)': numN,
    'Samples in spatial dimension (numD)': numD,
    'Summation window width (numK)': numK}.items():
    print("%40s = %g" %(key, value))



##################################################  Generate SAFT aperture masks
print("\nGenerating aperture masks and SAFT matrix:")

# create an array to store index mapping
# (row-index [:, 0] to col-index[:, 1], data is [:, 2])
arrI = np.zeros((numN, 3), dtype=np.int32)
arrI[:, 0] = np.arange(numN)
arrI[:, 2] = np.ones(numN, dtype=np.int32)

sizeFull = numN * numD
sizeItem = numN
cntBlocks = sizeFull // sizeItem

# Generate sparse matrices: iterate over numK
matSparse = []
for kk in range(0, numK):
    # determine row-to-col index mapping for kk
    arrI[:, 1] = np.sqrt((kk * dX) ** 2 + (arrI[:, 0] * dZ) ** 2) / dZ + 0.5

    # select all entries within matrix range
    selection = np.where((arrI[:, 1] < sizeItem) * (arrI[:, 1] >= 0))[0]
    mapping = arrI[selection].astype(np.int16)

    # stop when submatrices are empty
    if selection.size < 1:
        break

    matSparse.append(
        sps.coo_matrix((mapping[:, 2], (mapping[:, 0], mapping[:, 1])),
                       shape=(sizeItem, sizeItem), dtype=np.int8))
fastmatSparse = [fastmat.Sparse(mat.tocsc()) for mat in matSparse]
cntK = len(matSparse)



################################################## user-defined SAFT class
class SaftMaskClass(fastmat.Matrix):
    def __init__(self, numBlocks, sizeItem, *masks):
        if any((mask.ndim != 2 or mask.shape[1] != 2) for mask in masks):
            raise ValueError("Masks of SaftMask must be of shape Nx2!")
        self._masks = tuple(mask.astype(np.uint16, copy=True, order='C')
                            for mask in masks)
        self._sizeItem = sizeItem
        self._numBlocks = numBlocks
        self._numMasks = len(masks)
        numN = numBlocks * self._sizeItem
        self._initProperties(numN, numN, np.int8)

    def _forward(self, arrX):
        return SaftMaskCore(arrX, self._sizeItem, self._numBlocks, self._masks,
                            False)

    def _backward(self, arrX):
        return SaftMaskCore(arrX, self._sizeItem, self._numBlocks, self._masks,
                            True)

################################################## Use special fastmat class
print(" %-20s : %s" %("'matSaftMaskClass'",
                      "user-defined fastmat class, cython-optimized"))
matSaftMaskClass = SaftMaskClass(numD, sizeItem,
                                 *(tuple(np.vstack((mat.row, mat.col)).T
                                         for mat in matSparse)))
print(" " * 24 + repr(matSaftMaskClass))


figure(1)
rc('image', aspect='auto', interpolation='none')

arrScanData = refData['data'].astype(np.float32)
#print(arrScanData.min(), arrScanData.max(), arrScanData.dtype)
#vecInput = arrScanData.reshape((-1, 1)).astype(np.float32)
vecInput = arrScanData.T.reshape((-1, 1))

vecOutput = None
def doIt():
    global vecOutput
    vecOutput = matSaftMaskClass * vecInput

cnt = 5
doIt()
print(" %20s = %12.4e s (%d repetitions)" %(
    "runtime", timeit(doIt, number=cnt) / cnt, cnt))

arrReconstruction = vecOutput.T.reshape(arrScanData.T.shape).T

subplot(1, 3, 1)
imshow(arrScanData)

subplot(1, 12, 5)
imshow(vecInput)
subplot(1, 12, 6)
imshow(vecOutput)

subplot(1, 2, 2)
imshow(arrReconstruction)

show()
