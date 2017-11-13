# fastmat

[![Build Status](https://www.travis-ci.org/EMS-TU-Ilmenau/fastmat.svg?branch=master)](https://www.travis-ci.org/EMS-TU-Ilmenau/fastmat)[![Coverage Status](https://coveralls.io/repos/github/EMS-TU-Ilmenau/fastmat/badge.svg?branch=master)](https://coveralls.io/github/EMS-TU-Ilmenau/fastmat?branch=master)

## Description
Scientific computing requires handling large composed or structured matrices.
Fastmat is a framework for handling large composed or structured matrices.
It allows expressing and using them in a mathematically intuitive way while
storing and handling them internally in an efficient way. This approach allows
huge savings in computational time and memory requirements compared to using
dense matrix representations.

### Languages
- Python
- Cython

### Dependencies
- Python >= 2.7 or >=3.4
- Numpy >= 1.07
- Scipy >= 1.08
- Cython >= 1.18

### soft-dependencies
- matplotlib: for demos and tools that make use of plotting functions

### Authors
- Sebastian Semper - Technische Universität Ilmenau, Institute for Mathematics
- Christoph Wagner - Technische Universität Ilmenau,
                     Institute for Information Technology, EMS Group

### Contact
- sebastian.semper@tu-ilmenau.de
- christoph.wagner@tu-ilmenau.de
- https://www.tu-ilmenau.de/ems/

## Documentation / HELP !
Please have a look at the documentation, which is included in the source
distribution at github or may be built locally on your machine by running
    `make doc`

If you experience any trouble please do not hesitate to contact us or to open
an issue on our github projectpage: https://github.com/EMS-TU-Ilmenau/fastmat

### FAQ

#### Installation fails with *ImportError: No module named Cython.Build*
Something went wrong with resolving the dependencies of fastmat during setup.
This issue will be addressed in release 0.1.1. Please check if the problem
persists with this version. You may try to bypass the problem by running
    `pip install cython numpy scipy`
and retrying the installation of fastmat.

#### Windows: Installation fails with various "file not found" errors
Often, this is caused by missing header files. Unfortunately windows ships
without a c-compiler and the header files necessary to compile native binary
code. If you use the Intel Distribution for Python this can be resolved by
installing the Visual Studio Build tools with the version as recommended by
the version of the Intel Distribution for Python that you are using.

#### Issue not resolved yet?
Please contact us or leave your bug report in the *issue* section. Thank You!


## Citation / Acknowledgements
If you want to use fastmat or parts of it in your private, scientific or
commercial project or work you are required to acknowledge visibly to all users
that fastmat was used for your work and put a reference to the project and the
EMS Group at TU Ilmenau.

If you use fastmat for your scientific work you are further required to cite
the following publication affiliated with the project:
- `to be announced soon. Please tune back in regularly to check on updates.`

## Installation
fastmat currently supports both Linux and Windows. Building the
documentation and running benchmarks is currently only supported under linux
with windows support underway. You may choose one of these installation methods:

### installation with pip:

fastmat is included in the Python Package Index (PyPI) and can be installed
from the commandline by running one easy and straightforward command:
    `pip install fastmat`

When installing with pip all dependencies of the package will be installed
along. With release 0.1.1 python wheels will be offered for many versions
greatly improving installation time and effort, especially under windows.

### installation from source: doing it manually
- download the source distribution from our github repository:
    https://github.com/EMS-TU-Ilmenau/fastmat/archive/master.zip
- unpack its contents and navigate to the project root directory
#### LINUX
##### Python 2.x:
- Make sure you have the following packages installed:
  * numpy
  * scipy
  * cython
- Call "make install MODE=--user PYTHON=python2" to install fastmat for your local user $PYTHONPATH
- Call "sudo make install PYTHON=python2" to install fastmat as super user onto your local machine
- Call "sudo make uninstall" to remove a previous local machine install
- Call "make uninstall MODE=--user" to remove a previous local user install
##### Python 3.x:
- Make sure you have the following packages installed:
  * numpy
  * scipy
  * cython3
- Call "make install MODE=--user PYTHON=python3" to install fastmat for your local user $PYTHONPATH
- Call "sudo make install PYTHON=python3" to install fastmat as super user onto your local machine
- Call "sudo make uninstall" to remove a previous local machine install
- Call "make uninstall MODE=--user" to remove a previous local user install
#### WINDOWS
- Make sure you have the following software packages installed properly:
  * Intel Distribution for Python 2.7
  * Microsoft Visual C++ Compiler 9.0 for Python 2.7
  * GNU make for Windows 3.81 or newer
- Add the paths to python.exe (Intel Distribution, defaults to C:\IntelPython27\) and make.exe (GNU make) to the @PATH@ environment variable
- Call "make install MODE=--user" to install fastmat for your local user $PYTHONPATH
- Call "make install" as super user to install fastmat onto your local machine
- Call "make uninstall" to remove a previous local machine install.
- Call "make uninstall MODE=--user" to remove a provious local user install



## Demos
Feel free to have a look at the demos in the `demo/` directory of the source
distribution. Please make sure to have fastmat already installed when running
these.

Please note that the edgeDetect demo requires the Python Imaging Library (PIL)
installed and the SAFT demos do compile a cython-core of a user defined matrix
class beforehand thus having a delaying the first time they're executed.
