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
import timeit

# import numpy functionality
import numpy as np
import scipy as sp

################################################## import modules
try:
    import fastmat
except ImportError:
    sys.path.append('..')
    import fastmat
from fastmat.core.resource import getMemoryFootprint

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

numN        = 2040
numK        = 50
numD        = 1000 if numN == 2040 else numN / 2
dX          = 0.5
dZ          = 0.034

for key, value in {
    'Samples in time dimension (numN)': numN,
    'Samples in spatial dimension (numD)': numD,
    'Summation window width (numK)': numK}.items():
    print("%40s = %g" %(key, value))



print("\nGenerating masks and SAFT matrices:")



##################################################  Generate SAFT aperture masks
print(" %-20s : %s" %("'matSparse'",
                      "SAFT aperture masks as scipy-sparse matrices"))

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
        sp.sparse.coo_matrix(
                    (mapping[:, 2], (mapping[:, 0], mapping[:, 1])),
                    shape=(sizeItem, sizeItem), dtype=np.int8))
fastmatSparse = [fastmat.Sparse(mat.tocsc()) for mat in matSparse]
cntK = len(matSparse)




###############################################################################
###  Generate SAFT matrix using fastmat Kronecker approach
###############################################################################

################################################## generate actual matrix
print(" %-20s : %s" %("'matSaftKron'",
                      "using fastmat built-in classes"))
matSaftKron = fastmat.Sum(
    *[fastmat.Kron(
            fastmat.Eye(numD) if kk == 0
            else fastmat.Sparse(sp.sparse.eye(numD, k=kk, dtype=np.int8) +
                                sp.sparse.eye(numD, k=-kk, dtype=np.int8)),
            fastmat.Sparse(mat.tocsr()))
      for kk, mat in enumerate(matSparse)])
print(" " * 4 + repr(matSaftKron))




###############################################################################
###  Generate SAFT matrix with a user-defined fastmat clas
###############################################################################

################################################## user-defined SAFT class
# define a fastmat class representing SAFT
class SaftClass(fastmat.Matrix):
    def __init__(self, numN, sizeItem, *items):
        self._items = items
        self._sizeItem = sizeItem
        self._numBlocks = numN // sizeItem
        self._numItems = len(items)
        self._initProperties(numN, numN, np.int8)

    def _core(self, arrX, backward):
        numI = self._sizeItem
        numB = self._numBlocks

        arrRes = np.empty((self.numN, arrX.shape[1]), dtype=arrX.dtype)
        viewIn  = [arrX[mm * numI:(mm + 1) * numI, :] for mm in range(numB)]
        viewOut = [arrRes[nn * numI:(nn + 1) * numI, :] for nn in range(numB)]

        for kk in range(self._numItems):
            item = self._items[kk]
            itemDot = item.vdot if backward else item.dot
            for nn in range(numB):
                if kk == 0:
                    arrArg = viewIn[nn]
                    viewOut[nn][:] = itemDot(arrArg)
                else:
                    kkl, kkr = nn - kk, nn + kk
                    arrArg = (viewIn[kkr] if kkl < 0
                              else (viewIn[kkl] if kkr >= numB
                                    else viewIn[kkl] + viewIn[kkr]))
                    viewOut[nn][:] += itemDot(arrArg)

        return arrRes

    def _forward(self, arrX):
        return self._core(arrX, False)

    def _backward(self, arrX):
        return self._core(arrX, True)

################################################## generate actual matrix
print(" %-20s : %s" %("'matSaftClass'",
                      "user-defined fastmat class"))
matSaftClass = SaftClass(sizeFull, sizeItem, *matSparse)
print(" " * 4 + repr(matSaftClass))

###############################################################################
# fastmat matrix class for efficient mask matrices
###############################################################################

################################################## import cython-optimized core
import pyximport; pyximport.install()
from matrixSAFTmask import SaftMaskCore

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
print(" " * 4 + repr(matSaftMaskClass))




###############################################################################
# Generate a sparse reference matrix with scipy (as reference)
###############################################################################
print(" %-20s : %s" %("'matSaftReference'",
                      "using scipy only"))

# determine element count upfront to initialize storage memory efficiently:
numElements = 0
for kk, mat in enumerate(matSparse):
    numElements += (1 if kk == 0 else 2) * (cntBlocks - kk) * mat.nnz
arrCol = np.empty((numElements), dtype=np.int32)
arrRow = np.empty((numElements), dtype=np.int32)
arrData = np.empty((numElements), dtype=np.int8)

# now fill that array
ii = 0
for nn in range(0, cntBlocks):
    offsetRow = nn * sizeItem
    for mm in range(0, cntBlocks):
        offsetCol = mm * sizeItem
        kk = abs(nn - mm)
        if kk >= cntK:
            continue
        mat = matSparse[kk]
        arrCol[ii:ii + mat.nnz] = mat.col + offsetCol
        arrRow[ii:ii + mat.nnz] = mat.row + offsetRow
        arrData[ii:ii + mat.nnz] = mat.data
        ii += mat.nnz
    sys.stdout.write("  [converting row %d/%d]\r" %(nn, cntBlocks))
    sys.stdout.flush()
matSaftReference = sp.sparse.coo_matrix((arrData, (arrRow, arrCol)),
    shape=(sizeFull, sizeFull), dtype=np.int8)
print(" " * 4 + repr(matSaftReference))




###############################################################################
# Generate data vectors to apply the matrices to
###############################################################################




###############################################################################
# Benchmark performance and accuracy of matrices
###############################################################################

# define a sample data vector
v = np.arange(sizeFull).reshape((sizeFull, 1)).astype(np.int64)

print("\nChecking accuracy of the various methods:")
print(" - metric used: distance norm of transform output for a given vector")
vReference = matSaftReference.dot(v)
for name, matrix in sorted({
    "matSaftKron"       : matSaftKron,
    "matSaftClass"      : matSaftClass,
    "matSaftMaskClass"  : matSaftMaskClass}.items()):
    print("%40s : %12.4e" %("%s(v) vs. reference(v)" %(name),
                            np.linalg.norm(matrix * v - vReference)))


print("\nBenchmarking forward transform performance of matrices:")
def timing(title, fun, reps):
    print("%40s : %10.4g s"% (title, timeit.timeit(fun, number=reps) / reps))

for title, fun in sorted({
    "matSaftKron"       : lambda: matSaftKron * v,
    "matSaftClass"      : lambda: matSaftClass * v,
    "matSaftMaskClass"  : lambda: matSaftMaskClass * v,
    "matSaftReference"  : lambda: matSaftReference.dot(v)}.items()):
    timing(title, fun, 1)


print("\nRAM usage for the various variants:")
for size, title in sorted({
    getMemoryFootprint(matSaftKron)         : "matSaftKron",
    getMemoryFootprint(matSaftClass)        : "matSaftClass",
    getMemoryFootprint(matSaftMaskClass)    : "matSaftMaskClass",
    getMemoryFootprint(matSaftReference)    : "matSaftReference",
    getMemoryFootprint(matSparse)           : "matSparse"}.items()):
    print("%40s : %10d kBytes" %(title, size / 1000))
