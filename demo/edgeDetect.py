# -*- coding: utf-8 -*-
'''
  demo/edgeDetect.py
 -------------------------------------------------- part of the fastmat demos


  Author      : sempersn
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

# add local fastmat directory to system path
import sys
import time

# import numpy functionality
import numpy as np
from scipy import ndimage

################################################## import modules
try:
    import matplotlib.pyplot as plt               # plotting
except ImportError:
    print("matplotlib not found. Please consider installing it to proceed.")
    sys.exit(0)

try:
    import fastmat
except ImportError:
    sys.path.append('..')
    import fastmat


print("fastmat demo: Edge detection using fast correlation")
print("---------------------------------------------------")

# read the image
arrHead = ndimage.imread("head.png", flatten=True)

# extract the dimensions
numN, numM = arrHead.shape

# create the correlation vector
c = np.zeros(numN)
c[0] = 0.5
c[1] = -0.5

# create the correlation matrix
print(" * Create the correlation matrix")
fmatLx1 = fastmat.Circulant(c)

# calculate the gradient in the directions
# using the efficient way
print(" * Perform edge detection while exploiting structure")
s = time.time()
arrEdgesX = np.abs(fmatLx1 * arrHead)
arrEdgesY = np.abs(fmatLx1 * arrHead.T).T
numFastTime = time.time() - s

# cast the matrix to a standard numpy array
print(" * Generate unstructured reference matrix for speed comparison")
fmatLx2 = fastmat.Matrix(fmatLx1.array)

# calculate the gradient in the directions
# using the inefficient way
print(" * Perform edge detection without exploting structure")
s = time.time()
arrEdgesX = np.abs(fmatLx2 * arrHead)
arrEdgesY = np.abs(fmatLx2 * arrHead.T).T
numSlowTime = time.time() - s

# calc the total gradient
arrEdges = np.sqrt(arrEdgesX**2 + arrEdgesY**2)

print("\nResults:")
print("   Fast Detection: %.3fs" %(numFastTime))
print("   Reference     : %.3fs" %(numSlowTime))
print("   Speedup Factor: %3.3f" %(numSlowTime / numFastTime))

plt.figure(1)
plt.subplot(211)
plt.imshow(arrHead, cmap=plt.cm.gray)
plt.subplot(212)
plt.imshow(arrEdges, cmap=plt.cm.gray)
plt.show()
