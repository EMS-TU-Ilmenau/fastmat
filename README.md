# fastmat

## Version 0.1 beta

## Description
Scientific comuting requires handling large composed or structured matrices
Fastmat is a framework for handling large composed or structured matrices.
It allows expressing them mathematically intuitive while storing and handling
them efficient, achieving fast operations requiring less memory.

## Languages
- Python
- Cython


## Authors
- Sebastian Semper - Technische Universität Ilmenau, Institute for Mathematics
- Christoph Wagner - Technische Universität Ilmenau,
					 Institute for Information Technology, EMS Group


## Contact
- sebastian.semper@tu-ilmenau.de
- christoph.wagner@tu-ilmenau.de
- https://www.tu-ilmenau.de/ems/


## Citation
If you use fastmat in your own scientific work, you are required to
cite the following publication affiliated with the project
- tba


## Dependencies
- Numpy
- Scipy
- Python
- Cython


## Installation
### LINUX
#### Python 2.x:
- Make sure you have the following packages installed:
  * numpy
  * scipy
  * cython
- Call "make install MODE=--user PYTHON=python2" to install fastmat for your local user $PYTHONPATH
- Call "sudo make install PYTHON=python2" to install fastmat as super user onto your local machine
- Call "sudo make uninstall" to remove a previous local machine install
- Call "make uninstall MODE=--user" to remove a previous local user install

#### Python 3.x:
- Make sure you have the following packages installed:
  * numpy
  * scipy
  * cython3
- Call "make install MODE=--user PYTHON=python3" to install fastmat for your local user $PYTHONPATH
- Call "sudo make install PYTHON=python3" to install fastmat as super user onto your local machine
- Call "sudo make uninstall" to remove a previous local machine install
- Call "make uninstall MODE=--user" to remove a previous local user install

### WINDOWS
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
Feel free to have a look at the demos in the `demo/` directory. Please make sure to have fastmat already installed.
