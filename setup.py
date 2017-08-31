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
import platform
import sys
import os
import re
import subprocess

packageName     = 'fastmat'
packageVersion  = '0.1'           # provide a version tag as fallback
fullVersion     = packageVersion
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


def getCurrentVersion():
    global packageVersion
    global fullVersion

    # check if there is a manual version override
    if os.path.isfile(".version"):
        with open(".version", "r") as f:
            stdout = f.read().split('\n')[0]
        print("Override of version string to '%s' (from .version file )" %(
            stdout))

        packageVersion = stdout

    else:
        # check if source directory is a git repository
        if not os.path.exists(".git"):
            print(("Installing from something other than a Git repository; " +
                   "Version file '%s' untouched.") %(strVersionFile))
            return

        # fetch current tag and commit description from git
        try:
            p = subprocess.Popen(
                ["git", "describe", "--tags", "--dirty", "--always"],
                stdout=subprocess.PIPE
            )
        except EnvironmentError:
            print("Not a git repository; Version file '%s' not touched." %(
                strVersionFile))
            return

        stdout = p.communicate()[0].strip()
        if stdout is not str:
            stdout = stdout.decode()

        if p.returncode != 0:
            print(("Unable to fetch version from git repository; " +
                   "leaving version file '%s' untouched.") %(strVersionFile))
            return

        # output results to version string, extract package version number
        # from git tag
        fullVersion = stdout
        versionMatch = re.match("[.+\d+]+\d*[abr]\d*", fullVersion)
        if versionMatch:
            packageVersion = versionMatch.group(0)
            print("Fetched package version number from git tag (%s)." %(
                packageVersion))


# load setup and extension from distutils. Under windows systems, setuptools is
# used, which monkey-patches distutils to get around 'missing vcvarsall.bat'
try:
    from setuptools import setup, Extension
except ImportError:
    WARNING("Could not import setuptools, falling back to using distutils.")
    WARNING("You might want to consider installing python-setuptools.")
    from distutils.core import setup, Extension


# get version from git and update fastmat/__init__.py accordingly
getCurrentVersion()

# make sure there exists a version.py file in the project
with open(strVersionFile, "w") as f:
    f.write(VERSION_PY % (fullVersion))
print("Set %s to '%s'" %(strVersionFile, fullVersion))

# get the long description from the README file.
# CAUTION: Python2/3 utf encoding shit calls needs some adjustments
fileName = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')
f = (open(fileName, 'r') if sys.version_info < (3, 0)
     else open(fileName, 'r', encoding='utf-8'))
longDescription = f.read()
f.close()

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
        packageName, strPlatform))
print("Building %s v%s for %s." % (packageName, packageVersion, strPlatform))


# override list type to allow lazy cythonization: cythonize and compile only
# after install_requires installed cython
class lazyCythonize(list):
    def __init__(self, callback):
        self._list, self.callback = None, callback

    def c_list(self):
        if self._list is None:
            self._list = self.callback()

        return self._list

    def __iter__(self):
        for e in self.c_list():
            yield e

    def __getitem__(self, ii):
        return self.c_list()[ii]

    def __len__(self):
        return len(self.c_list())


def extensions():
    try:
        from Cython.Build import cythonize
        import numpy

        extensionArguments = {
            'include_dirs':
            [numpy.get_include(), 'fastmat/core', 'util/routines'],
            'extra_compile_args': compilerArguments,
            'extra_link_args': linkerArguments
        }

        return cythonize([
            Extension("*", ["fastmat/*.pyx"], **extensionArguments),
            Extension("*", ["fastmat/algs/*.pyx"], **extensionArguments),
            Extension("*", ["fastmat/core/*.pyx"], **extensionArguments)
        ])

    except ImportError:
        print("\n\n" +
              "  Could not import one or more of the required modules:\n" +
              "    cython, numpy, scipy\n\n" +
              "  Possibly an error in resolving package dependencies.\n" +
              "  Please try installing them manually and try again:\n" +
              "    pip install cython numpy scipy matplotlib\n\n" +
              "  Sorry for the inconvenience. We are going to address this " +
              "soon.\n")
        sys.exit(1)


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
        'Development Status :: 5 - Production/Stable',
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
        'Topic :: Scientific/Engineering :: Mathematics',
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
        'fastmat/core',
        'fastmat/inspect'
    ],
    ext_modules=lazyCythonize(extensions)
)
