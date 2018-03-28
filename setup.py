# -*- coding: utf-8 -*-
# Copyright 2016 Sebastian Semper, Christoph Wagner
#     https://www.tu-ilmenau.de/it-ems/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''Setup script for installation of fastmat package

  Usecases:
    - install fastmat package system-wide on your machine (needs su privileges)
        EXAMPLE:        'python setup.py install'

    - install fastmat package for your local user only (no privileges needed)
        EXAMPLE:        'python setup.py install --user'

    - compile all cython source files locally
        EXAMPLE:        'python setup.py build_ext --inplace'
'''

# import modules
import platform
import sys
import os
import re
import subprocess
from Cython.Distutils import build_ext
from future.utils import iteritems

packageName     = 'fastmat'
packageVersion  = '0.1.1'           # provide a version tag as fallback
fullVersion     = packageVersion
strVersionFile  = "%s/version.py" %(packageName)

VERSION_PY      = """
# -*- coding: utf-8 -*-
# This file carries the module's version information which will be updated
# during execution of the installation script, setup.py. Distribution tarballs
# contain a pre-generated copy of this file.

__version__ = '%s'
"""


def WARNING(string):
    print("\033[91m WARNING:\033[0m %s" % (string))


###############################################################################
###  Import setuptools
###############################################################################

# load setup and extensions from setuptools. If that fails, try distutils
try:
    from setuptools import setup, Extension
except ImportError:
    WARNING("Could not import setuptools.")
    raise


###############################################################################
###  Determine package version
###############################################################################
def getCurrentVersion():
    global packageVersion
    global fullVersion

    # check if there is a manual version override
    if os.path.isfile(".version"):
        with open(".version", "r") as f:
            stdout = f.read().split('\n')[0]
        print("Override of version string to '%s' (from .version file )" % (
            stdout))

        packageVersion = stdout

    else:
        # check if source directory is a git repository
        if not os.path.exists(".git"):
            print(("Installing from something other than a Git repository; " +
                   "Version file '%s' untouched.") % (strVersionFile))
            return

        # fetch current tag and commit description from git
        try:
            p = subprocess.Popen(
                ["git", "describe", "--tags", "--dirty", "--always"],
                stdout=subprocess.PIPE
            )
        except EnvironmentError:
            print("Not a git repository; Version file '%s' not touched." % (
                strVersionFile))
            return

        stdout = p.communicate()[0].strip()
        if stdout is not str:
            stdout = stdout.decode()

        if p.returncode != 0:
            print(("Unable to fetch version from git repository; " +
                   "leaving version file '%s' untouched.") % (strVersionFile))
            return

        # output results to version string, extract package version number
        # from git tag
        fullVersion = stdout
        versionMatch = re.match("[.+\d+]+\d*[abr]\d*", fullVersion)
        if versionMatch:
            packageVersion = versionMatch.group(0)
            print("Fetched package version number from git tag (%s)." %(
                packageVersion))


# get version from git and update fastmat/__init__.py accordingly
getCurrentVersion()

# make sure there exists a version.py file in the project
with open(strVersionFile, "w") as f:
    f.write(VERSION_PY % (fullVersion))
print("Set %s to '%s'" %(strVersionFile, fullVersion))


###############################################################################
###  CUDA specific routines
###############################################################################

def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib'
    and values giving the absolute path to each directory/file.

    Everything has to be provided in the env variables CUDAHOME, CUDALIB and
    CUDAINCLUDE. The path to nvcc is derived from CUDAHOME.
    """

    # home directory for the cuda binaries
    if 'CUDAHOME' in os.environ:
        strHomePath = os.environ['CUDAHOME']
    else:
        raise EnvironmentError("CUDAHOME environment variable not set!")

    # absolute path to nvcc
    strNvccPath = os.path.join(strHomePath, 'bin', 'nvcc')

    # absolute path to all the header files
    if 'CUDAINCLUDE' in os.environ:
        strIncludePath = os.environ['CUDAINCLUDE']
    else:
        raise EnvironmentError("CUDAINCLUDE environment variable not set!")

    # absolute path to all the shared libraries
    if 'CUDALIB' in os.environ:
        strLibraryPath = os.environ['CUDALIB']
    else:
        raise EnvironmentError("CUDALIB environment variable not set!")

    # collect the paths in a nice dictionary
    cudaConfig = {
        'home': strHomePath,
        'nvcc': strNvccPath,
        'include': strIncludePath,
        'lib': strLibraryPath
    }

    # check if all provided paths are valid
    for k, v in iteritems(cudaConfig):
        if not os.path.exists(v):
            raise EnvironmentError(
                'The path to %s could not be located in %s!' % (k, v)
            )

    return cudaConfig

def customize_compiler_for_nvcc(self, cudaConfig):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _compile methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        print(cc_args)
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', cudaConfig['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            # self.set_executable('compiler_so', '/usr/bin/clan)
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile

def getExtensionList(lstExtDef):
    lstExtensions = []
    for ee in lstExtDef:
        lstExtensions.append(
            Extension(
                ee['name'],
                sources=ee['sources'],
                **dctExtensionKwargs
            )
        )
    return lstExtensions


# supplemental compilation arguments for the two compilers we
# are using
gccExtraCompileArgs = []
nvccExtraCompileArgs = [
    '-arch=sm_30',
    '--ptxas-options=-v',
    '-c',
    '--compiler-options',
    "'-fPIC'"
]

# initialize the paths related to cuda
if "--cuda" in sys.argv:
    cudaConfig = locate_cuda()

    cudaConfig.update({
        'linkedLibs': ['cudart', 'cufft'],
    })

    # define the customized compiler
    class custom_build_ext(build_ext):
        def build_extensions(self):
            customize_compiler_for_nvcc(self.compiler, cudaConfig)
            build_ext.build_extensions(self)

    extraCompileArgs = {
        'gcc': gccExtraCompileArgs,
        'nvcc': nvccExtraCompileArgs
    }

    sys.argv.remove("--cuda")
else:
    cudaConfig = {
        'home': "",
        'nvcc': "",
        'include': ".",
        'lib': "."
    }

    extraCompileArgs = gccExtraCompileArgs

    cudaConfig.update({
        'linkedLibs': []
    })

    class custom_build_ext(build_ext):
        def build_extensions(self):
            build_ext.build_extensions(self)


###############################################################################
###  Prepare other files and construct compile arguments
###############################################################################

# get the long description from the README file.
# CAUTION: Python2/3 utf encoding shit calls needs some adjustments
fileName = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')
f = (open(fileName, 'r') if sys.version_info < (3, 0)
     else open(fileName, 'r', encoding='utf-8'))
longDescription = f.read()
f.close()

# define different compiler arguments for each platform
strPlatform = platform.system()

linkerArguments = []
if strPlatform == 'Windows':
    # Microsoft Visual C++ Compiler 9.0
    gccExtraCompileArgs += ['/O2', '/fp:precise']
elif strPlatform == 'Linux':
    # assuming Linux and gcc
    gccExtraCompileArgs += ['-O3', '-march=native', '-ffast-math']
elif strPlatform == 'Darwin':
    # assuming Linux and gcc
    gccExtraCompileArgs += ['-O3', '-march=native', '-ffast-math']
else:
    WARNING("Your platform is currently not supported by %s: %s" % (
        packageName, strPlatform))

# define default cython directives, these may get extended along the script
cythonDirectives = {}
defineMacros = []
CMD_COVERAGE = '--enable-cython-tracing'
if CMD_COVERAGE in sys.argv:
    sys.argv.remove(CMD_COVERAGE)
    cythonDirectives['linetrace'] = True
    cythonDirectives['binding'] = True
    defineMacros += [('CYTHON_TRACE_NOGIL', '1'),
                     ('CYTHON_TRACE', '1')]
    print("Enabling cython line tracing allowing code coverage analysis")

print("Building %s v%s for %s." % (packageName, packageVersion, strPlatform))


###############################################################################
###  Enable flexible dependency handling by installing missing base components
###############################################################################
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
    except ImportError:
        def cythonize(*args, **kwargs):
            print("Hint: Wrapping import of cythonize in extensions()")
            from Cython.Build import cythonize
            return cythonize(*args, **kwargs)

    try:
        import numpy
        lstIncludes = [numpy.get_include()]
    except ImportError:
        lstIncludes = []

    extensionArguments = {
        'include_dirs':
            lstIncludes + ['fastmat', 'fastmat/core', 'fastmat/inspect', 'util', cudaConfig['include']],
        'library_dirs': [cudaConfig['lib']],
        'language': 'c++',
        'libraries': ['cudart', 'cufft'],
        'runtime_library_dirs': [cudaConfig['lib']],
        'extra_compile_args': extraCompileArgs,
        'extra_link_args': linkerArguments,
        'define_macros': defineMacros
    }

    # extensionCudaArguments = extensionArguments.copy()
    # extensionCudaArguments.update({'language': 'c++'})

    return cythonize(
        [
            Extension(
                "*",
                ["fastmat/*.pyx"],
                **extensionArguments
            ),
            Extension("*", ["fastmat/algs/*.pyx"], **extensionArguments),
            Extension("*", ["fastmat/core/*.pyx"], **extensionArguments),
            Extension(
                "fastmat.BlkTwoLvlToepWrp",
                ["fastmat/BlkTwoLvlToepWrp.pyx", "fastmat/BlkTwoLvlToepCu.cu"],
                **extensionArguments
            )
        ],
        compiler_directives=cythonDirectives,
        nthreads=4
    )


# determine requirements for install and setup
def checkRequirement(lstRequirements, importName, requirementName):
    '''
    Don't add packages unconditionally as this involves the risk of updating an
    already installed package. Sometimes this may break during install or mix
    up dependencies after install. Consider an update only if the requested
    package is not installed at all or if we are building an installation wheel.
    '''
    try:
        __import__(importName)
    except ImportError:
        lstRequirements.append(requirementName)
    else:
        if 'bdist_wheel' in sys.argv[1:]:
            lstRequirements.append(requirementName)


setupRequires = []
installRequires = []
checkRequirement(setupRequires, 'setuptools', 'setuptools>=18.0')
checkRequirement(setupRequires, 'Cython', 'cython>=0.19')
checkRequirement(setupRequires, 'numpy', 'numpy')
checkRequirement(installRequires, 'six', 'six')
checkRequirement(installRequires, 'scipy', 'scipy')

print("Requirements for setup: %s" %(setupRequires))
print("Requirements for install: %s" %(installRequires))

###############################################################################
### The documentation
###############################################################################


def doc_opts():
    try:
        from sphinx.setup_command import BuildDoc
    except ImportError:
        return {}

    class OwnDoc(BuildDoc):

        def __init__(self, *args, **kwargs):
            # os.system(
            #     sys.executable + " util/bee.py benchmark -p doc/_static/bench"
            # )
            super(OwnDoc, self).__init__(*args, **kwargs)

    return OwnDoc


###############################################################################
### The actual setup
###############################################################################
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
        'Framework :: IPython',
        'Framework :: Jupyter',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries'
    ],
    keywords='linear transforms efficient algorithms mathematics',
    setup_requires=setupRequires,
    install_requires=installRequires,
    packages=[
        'fastmat',
        'fastmat/algs',
        'fastmat/core',
        'fastmat/inspect'
    ],
    cmdclass={
        'build_doc': doc_opts(),

        # inject our custom compile patching magic
        'build_ext': custom_build_ext
        },
    command_options={
        'build_doc': {
            'project': ('setup.py', packageName),
            'version': ('setup.py', packageVersion),
            'release': ('setup.py', fullVersion),
            'copyright': ('setup.py', '2017, ' + packageName)
        }},
    ext_modules=lazyCythonize(extensions)
)
