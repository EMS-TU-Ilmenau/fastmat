# -*- coding: utf-8 -*-
'''
  demo/sparseReco.py
 -------------------------------------------------- part of the fastmat demos

  Demonstration on how to use fastMat for sparse deconvolution, where one can
  specify most of the parameters involved.


  Author      : sempersn
  Introduced  : 2016-09-28
 ------------------------------------------------------------------------------
  PARAMETERS:
    numSignalSize   - size of the signal and thus also the dictionary
    numSlices       - number of signal to process
    numPulses       - number of pulses in the signal
    numPulseWidth   - width of a single pulse, where 0.5 would be half of
                      the signal length
    numPulseFreq    - frequency relative to the whole length, i.e. number
                      of oszillations over the whole signal

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
# size, because the traditional method might fail (> 2^13 with 8GB RAM)
################################################################################
numSignalSize = 2**11
numSlices = 1
numPulses = 10
numPulseWidth = 0.01
numPulseFreq = 200


################################################################################
def genPulse(numN, numWidth, numFreq):
    '''
    Construct a pulsed in a gaussian envelope that gets cut off
    to get a signal with compact support, given frequency of the
    cosine as well as the window width.
        numN        - number of samples of the whole pulse
        numWidth    - width of the gaussian window
        numFreq     - frequency of the underlying cosine
        arrT        - array that contains the time values
        arrCos      - Cosine of the time array
        arrGauss    - Envelope of in form of a cut-off Gaussian
        arrP        - the final pulse
    '''

    # create the time array as an
    # open intervall from 0 to 2*pi
    arrT = np.linspace(0, 2 * np.pi * (1. - 1. / numN), numN)

    # construct the cosine of the intervall
    # with the required frequency
    arrCos = np.cos(arrT * numFreq)

    # create a windowed gaussian from
    # -2*sigma to +2*sigma
    arrGauss = np.exp(
        -((arrT - (2 * 2 * np.pi * numWidth)) / (2 * np.pi * numWidth))**2)
    arrGauss[arrT > 4 * 2 * np.pi * numWidth] = 0

    # multiply the two to get the actual pulse
    arrP = arrCos * arrGauss

    # return a normalized version of this pulse
    return arrP / np.sqrt(np.inner(arrP, arrP))


def genGroundTruth(numN, numM, numK):
    '''
    '''

    arrX = np.zeros((numN, numM))

    for ii in range(numM):
        arrInd = int(0.1 * numN) + npr.choice(
            range(int(numN - 0.2 * numN)), numK, replace=False)
        arrX[arrInd, ii] = npr.uniform(1, 2, numK)

    return arrX


################################################################################
#                          CALCULATION SECTION
################################################################################

printTitle("Sparse reconstruction example using efficient convolution",
           width=80)


print(" * Generate the windowed pulse")
s = time.time()
arrP = genPulse(numSignalSize, numPulseWidth, numPulseFreq)
numPulseTime = time.time() - s


# generate a circulant dictionary which will serve as linear operator for the
# convolution
print(" * Create the dictionary")
s = time.time()
matC = fastmat.Circulant(arrP)
numDictionaryTime = time.time() - s


# create an explicit version of the dictionary
# that disregards the circulant structure
print(" * Create the unstructured matrix for speed comparison")
matCHat = fastmat.Matrix(matC.array)


# generate a random sequence of spikes where position and amplitudes are random
print(" * Generating the ground truth as a sequence of spikes")
s = time.time()
arrX = genGroundTruth(numSignalSize, numSlices, numPulses)
numGroundTime = time.time() - s


# do the measurement (apply the forward model), i.e. do the convolution
print(" * Apply the forward model to generate the signal")
s = time.time()
arrY = matC * arrX
numForwardTime = time.time() - s


# call the sparse recovery algorithm to extract the original pulses
print(" * Do the reconstruction while exploiting structure")
s = time.time()
arrR1 = fastmat.algs.OMP(matC, arrY, numPulses)
numFastTime = time.time() - s

print(" * Do the reconstruction without exploiting structure")
s = time.time()
arrR2 = fastmat.algs.OMP(matCHat, arrY, numPulses)
numSlowTime = time.time() - s

################################################################################
#                               OUTPUT SECTION
################################################################################

printTitle("RESULTS", width=80)
frameText("   Pulse Generation         : % 10.3f ms" % (1000 * numPulseTime))
frameText("   Dictionary Generation    : % 10.3f ms" %
          (1000 * numDictionaryTime))
frameText("   Ground Truth Generation  : % 10.3f ms" % (1000 * numGroundTime))
frameText("   Forward Model            : % 10.3f ms" % (1000 * numForwardTime))
frameLine()
frameText("   Fast Reconstruction Time : % 10.3f s" % (numFastTime))
frameText("   Slow Reconstruction Time : % 10.3f s" % (numSlowTime))
frameLine()
frameText("   Efficient Matrix Storage : % 10.3f MB" % (matC.nbytes / 2.**20))
frameText("   Reference Matrix Storage : % 10.3f MB" %
          (matCHat.nbytes / 2.**20))
frameLine()
frameText("   Speedup factor           : % 10.2f" % (numSlowTime / numFastTime))
frameLine()
