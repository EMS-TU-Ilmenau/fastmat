# -*- coding: utf-8 -*-
'''
  setup.py
 -------------------------------------------------- part of the fastmat package

  Setup script for installation of fastmat package

  Usecases:
    - install fastmat package system-wide on your machine (needs su privileges)
        EXAMPLE:        'python setup.py install'

    - install fastmat package for your local user only (no privileges needed)
        EXAMPLE:        'python setup.py install --user'

    - compile all cython source files locally
        EXAMPLE:        'python setup.py build_ext --inplace'


  Author      : sempersn
  Introduced  : 2016-04-08
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

# import modules
from Cython.Build import cythonize
import platform
import sys
import os
import re
import subprocess
import numpy


packageName     = 'fastmat'
packageVersion  = '0.1b'
strName         = "%s %s" % (packageName, packageVersion)
strVersionFile  = "%s/version.py" %(packageName)

VERSION_PY      = """
# -*- coding: utf-8 -*-
# This file carries the module's version information which will be updated
# during execution of the installation script, setup.py. Distribution tarballs
# contain a pre-generated copy of this file.

__version__ = '%s'
"""


def WARNING(str):
    print("\033[91m WARNING:\033[0m %s" % (str))


def updateVersion():
    # check if source directory is a git repository
    if not os.path.exists(".git"):
        print("Installing from something other than a Git repository; " +
              "Leaving %s alone." %(strVersionFile))
        return

    # fetch current tag and commit description from git
    try:
        p = subprocess.Popen(
            ["git", "describe", "--tags", "--dirty", "--always"],
            stdout=subprocess.PIPE
        )
    except EnvironmentError:
        print("Not a git repository; leaving version string '%s' untouched." %(
            strVersionFile))
        return

    stdout = p.communicate()[0].strip()
    if stdout is not str:
        stdout = stdout.decode()

    if p.returncode != 0:
        print(("Unable to fetch version from git repository; " +
               "leaving version string '%s' untouched.") %(strVersionFile))
        return

    # output results to version string, extract package version number from tag
    versionMatch = re.match("[.+\d+]+\d*", stdout)
    if versionMatch:
        packageVersion = versionMatch.group(0)
        print("Fetched package version number from git tag (%s)." %(
            packageVersion))

    with open(strVersionFile, "w") as f:
        f.write(VERSION_PY % (stdout))
    print("Set %s to '%s'" %(strVersionFile, stdout))


# load setup and extension from distutils. Under windows systems, setuptools is
# used, which monkey-patches distutils to get around 'missing vcvarsall.bat'
try:
    from setuptools import setup, Extension
except ImportError:
    WARNING("Could not import setuptools, falling back to using distutils.")
    WARNING("You might want to consider installing python-setuptools.")
    from distutils.core import setup, Extension


# define different compiler arguments for each platform
strPlatform = platform.system()
if strPlatform == 'Windows':
    # Microsoft Visual C++ Compiler 9.0
    compilerArguments = ['/O2', '/fp:precise']
    linkerArguments = []
elif strPlatform == 'Linux':
    # assuming Linux and gcc
    compilerArguments = ['-O3', '-march=native', '-fopenmp', '-ffast-math']
    linkerArguments = ['-fopenmp']
else:
    raise NotImplementedError("Your platform is not supported by %s: %s" % (
        strName, strPlatform))
print("Building %s for %s." % (strName, strPlatform))

extensionArguments = {
    'include_dirs':
    [numpy.get_include(), 'fastmat/helpers', 'util/routines'],
    'extra_compile_args': compilerArguments,
    'extra_link_args': linkerArguments
    #'define_macros'        : [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
}

# get version from git and update fastmat/__init__.py accordingly
updateVersion()

# get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md')) as f:
    longDescription = f.read()

# setup package
setup(
    name=packageName,
    version=packageVersion,
    description='fast linear transforms in Python',
    long_description=longDescription,
    author='Christoph Wagner, Sebastian Semper, EMS group TU Ilmenau',
    author_email='christoph.wagner@tu-ilmenau.de',
    url='https://ems-tu-ilmenau.github.io/fastmat/',
    license='Apache Software License',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics'
        'Topic :: Software Development :: Libraries'
    ],
    keywords='linear transforms efficient algorithms mathematics',
    install_requires=[
        'cython',
        'numpy',
        'scipy',
        'matplotlib'
    ],
    packages=[
        'fastmat',
        'fastmat/algs',
        'fastmat/helpers'
    ],
    ext_modules=cythonize([
        Extension("*", ["fastmat/*.pyx"], **extensionArguments),
        Extension("*", ["fastmat/algs/*.pyx"], **extensionArguments),
        Extension("*", ["fastmat/helpers/*.pyx"], **extensionArguments)
    ])
)
