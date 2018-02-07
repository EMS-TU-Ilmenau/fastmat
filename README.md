# fastmat
[![Version](https://img.shields.io/pypi/v/fastmat.svg)](https://pypi.python.org/pypi/fastmat)
[![Status](https://img.shields.io/pypi/status/fastmat.svg)](https://pypi.python.org/pypi/fastmat)
[![Python wheels](https://img.shields.io/pypi/wheel/fastmat.svg)](https://pypi.python.org/pypi/fastmat)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/8af2b42d33cc4f5ba5b91566032c44ed)](https://www.codacy.com/app/ChristophWWagner/fastmat-EMS-TU-Ilmenau?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=EMS-TU-Ilmenau/fastmat&amp;utm_campaign=Badge_Grade)
[![Documentation Status](https://readthedocs.org/projects/fastmat/badge/?version=latest)](http://fastmat.readthedocs.io/en/latest/?badge=latest)

[![License](https://img.shields.io/pypi/l/fastmat.svg)](https://pypi.python.org/pypi/fastmat)
[![Python versions](https://img.shields.io/pypi/pyversions/fastmat.svg)](https://pypi.python.org/pypi/fastmat)
[![Implementation](https://img.shields.io/pypi/implementation/fastmat.svg)](https://pypi.python.org/pypi/fastmat)

[![Coverage Status](https://coveralls.io/repos/github/EMS-TU-Ilmenau/fastmat/badge.svg?branch=master)](https://coveralls.io/github/EMS-TU-Ilmenau/fastmat?branch=master)
[![GitHub issues](https://img.shields.io/github/issues/EMS-TU-Ilmenau/fastmat.svg)](https://github.com/EMS-TU-Ilmenau/fastmat/issues)
**stable:** [![Build Status](https://www.travis-ci.org/EMS-TU-Ilmenau/fastmat.svg?branch=stable)](https://www.travis-ci.org/EMS-TU-Ilmenau/fastmat) **master:** [![Build Status](https://www.travis-ci.org/EMS-TU-Ilmenau/fastmat.svg?branch=master)](https://www.travis-ci.org/EMS-TU-Ilmenau/fastmat)

## Description
Scientific computing requires handling large composed or structured matrices.
Fastmat is a framework for handling large composed or structured matrices.
It allows expressing and using them in a mathematically intuitive way while
storing and handling them internally in an efficient way. This approach allows
huge savings in computational time and memory requirements compared to using
dense matrix representations.

### Dependencies
- Python >= 2.7 or >=3.4
- Numpy >= 1.7
- Scipy >= 1.0
- Cython >= 1.19
- soft dependencies:
    - matplotlib: for demos and tools that make use of plotting functions

### Authors & Contact Information
- Sebastian Semper | sebastian.semper@tu-ilmenau.de  
  Technische Universität Ilmenau, Institute for Mathematics, EMS Group
- Christoph Wagner | christoph.wagner@tu-ilmenau.de  
  Technische Universität Ilmenau, Institute for Information Technology, EMS Group
- **<https://www.tu-ilmenau.de/it-ems/>**

## Citation / Acknowledgements
If you use fastmat, or parts of it, for commercial purposes you are required
to acknowledge the use of fastmat visibly to all users of your work and put a
reference to the project and the EMS Group at TU Ilmenau.

If you use fastmat for your scientific work you are required to mention the
EMS Group at TU Ilmenau and cite the following publication affiliated with the
project:
 > C. Wagner and S. Semper, _Fast Linear Transforms in Python_,  
 > arXiV:1710.09578, 2017
 >
 > -- <cite>https://arxiv.org/abs/1710.09578</cite>

- **<https://www.tu-ilmenau.de/it-ems/>**

## Installation
fastmat currently supports Linux, Windows and Mac OS. You may choose one of
these installation methods:

### Installing with pip:

fastmat is included in the Python Package Index (PyPI) and can be installed
from the commandline by running one easy and straightforward command:
    `pip install fastmat`

When installing with pip all dependencies of the package will be installed
along. With release 0.1.1 python wheels will be offered for many versions
greatly improving installation time and effort.

### Manually installing from source
- download the source distribution from our github repository:
    https://github.com/EMS-TU-Ilmenau/fastmat/archive/stable.zip
- unpack its contents and navigate to the project root directory
- run `pip install .` to install fastmat on your computer
- you may also install fastmat without pip, using the offered makefile:
    * type `make install` to install fastmat on your computer
    * If you intend to install the package locally for your user type
      `make install MODE=--user` instead
    * You may add a version specifier for all `make` targets that directly or indirectly invoke Python:
      `make install PYTHON=python2`
      `make compile PYTHON=python3`
    * If you only would like to compile the package to use it from this local
      directory without installing it, type `make compile`
    * An uninstallation of a previously run `make install`is possible, provided the installation log file `setup.files` has been preserved
      Invoking `make uninstall` without a local `setup.files` causes another installation for generating the setup file log prior to uninstalling
- **NOTE: Windows users**
  If you intent on building fastmat from source on a windows platform, make sure you have installed a c compiler environment and make interpreter. One way to accomplish this is to install these tools for Python 2.7 (you may also chose different ones, of course):
    * Intel Distribution for Python 2.7
    * Microsoft Visual C++ Compiler 9.0 for Python 2.7
    * GNU make for Windows 3.81 or newer
    * depending on your system: The relevant header files

## Demos
Feel free to have a look at the demos in the `demo/` directory of the source
distribution. Please make sure to have fastmat already installed when running
these.

Please note that the edgeDetect demo requires the Python Imaging Library (PIL)
installed and the SAFT demos do compile a cython-core of a user defined matrix
class beforehand thus having a delaying the first time they're executed.

## Documentation / HELP !
Please have a look at the documentation, which is included in the source
distribution at github or may be built locally on your machine by running
    `make doc`

If you experience any trouble please do not hesitate to contact us or to open
an issue on our github projectpage: https://github.com/EMS-TU-Ilmenau/fastmat

### FAQ

Please check out our project documentation at [readthedocs](https://fastmat.readthedocs.io/).

#### Windows: Installation fails with various "file not found" errors
Often, this is caused by missing header files. Unfortunately windows ships
without a c-compiler and the header files necessary to compile native binary
code. If you use the Intel Distribution for Python this can be resolved by
installing the Visual Studio Build tools with the version as recommended by
the version of the Intel Distribution for Python that you are using.

#### Issue not resolved yet?
Please contact us or leave your bug report in the *issue* section. Thank You!
