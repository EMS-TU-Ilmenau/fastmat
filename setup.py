# -*- coding: utf-8 -*-
# Copyright 2018 Sebastian Semper, Christoph Wagner
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
try:
    import sysconfig
except ImportError:
    from setuptools import sysconfig


def WARNING(string):
    print("\033[91mWARNING:\033[0m %s" % (string))


def ERROR(string, e):
    print("\033[91mERROR:\033[0m %s" % (string))
    if isinstance(e, int):
        sys.exit(e)
    else:
        raise e


def INFO(string):
    print("\033[96mINFO:\033[0m %s" % (string))


# load setup and extensions from setuptools. If that fails, try distutils
try:
    from setuptools import setup, Extension
except ImportError:
    WARNING("Could not import setuptools.")
    raise

# global package constants
packageName     = 'fastmat'
packageVersion  = '<INVALID>'
strVersionFile  = "%s/version.py" %(packageName)

VERSION_PY = """
# -*- coding: utf-8 -*-
# This file carries the module's version information which will be updated
# during execution of the installation script, setup.py. Distribution tarballs
# contain a pre-generated copy of this file.

__version__ = '%s'
"""

##############################################################################
### function and class declaration section. DO NOT PUT SCRIPT CODE IN BETWEEN
##############################################################################


# Enable flexible dependency handling by installing missing base components
class lazyCythonize(list):
    '''
    Override list type to allow lazy cythonization.
    Cythonize and compile only after install_requires are actually installed.
    '''

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
    '''
    Handle generation of extensions (a.k.a "managing cython compilery").
    '''
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
        lstIncludes + ['fastmat/core', 'fastmat/inspect'],
        'extra_compile_args': compilerArguments,
        'extra_link_args': linkerArguments,
        'define_macros': defineMacros
    }

    # me make damn sure, that disutils does not mess with our
    # build process
    global useGccOverride
    if useGccOverride:
        INFO('Overriding compiler setup for `gcc -shared`')
        sysconfig.get_config_vars()['CFLAGS'] = ''
        sysconfig.get_config_vars()['OPT'] = ''
        sysconfig.get_config_vars()['PY_CFLAGS'] = ''
        sysconfig.get_config_vars()['PY_CORE_CFLAGS'] = ''
        sysconfig.get_config_vars()['CC'] = 'gcc'
        sysconfig.get_config_vars()['CXX'] = 'g++'
        sysconfig.get_config_vars()['BASECFLAGS'] = ''
        sysconfig.get_config_vars()['CCSHARED'] = '-fPIC'
        sysconfig.get_config_vars()['LDSHARED'] = 'gcc -shared'
        sysconfig.get_config_vars()['CPP'] = ''
        sysconfig.get_config_vars()['CPPFLAGS'] = ''
        sysconfig.get_config_vars()['BLDSHARED'] = ''
        sysconfig.get_config_vars()['CONFIGURE_LDFLAGS'] = ''
        sysconfig.get_config_vars()['LDFLAGS'] = ''
        sysconfig.get_config_vars()['PY_LDFLAGS'] = ''

    return cythonize(
        [Extension("*", ["fastmat/*.pyx"], **extensionArguments),
         Extension("*", ["fastmat/algorithms/*.pyx"], **extensionArguments),
         Extension("*", ["fastmat/core/*.pyx"], **extensionArguments)],
        compiler_directives=cythonDirectives,
        nthreads=4
    )


# determine requirements for install and setup
def checkRequirement(lstRequirements, importName, requirementName):
    '''
    Don't add packages unconditionally as this involves the risk of updating an
    already installed package. Sometimes this may break during install or mix
    up dependencies after install. Consider an update only if the requested
    package is not installed at all or if we are building an installation
    wheel.
    '''
    try:
        __import__(importName)
    except ImportError:
        lstRequirements.append(requirementName)
    else:
        if 'bdist_wheel' in sys.argv[1:]:
            lstRequirements.append(requirementName)


def doc_opts():
    '''
    Introduce a command-line setup target to generate the sphinx doc.
    '''
    try:
        from sphinx.setup_command import BuildDoc

        class OwnDoc(BuildDoc, object):

            def __init__(self, *args, **kwargs):
                # check if we have the necessary sphinx add-ons installed
                import pip
                global sphinxRequires
                failed = []
                for requirement in sphinxRequires:
                    try:
                        __import__(requirement)
                    except ImportError:
                        failed.append(requirement)

                if len(failed) > 0:
                    ERROR(
                        "Following pypi packages are missing: %s" %(failed, ),
                        1
                    )

                super(OwnDoc, self).__init__(*args, **kwargs)

        return OwnDoc

    except ImportError:
        WARNING(
            "Unable to import Sphinx. Building docs is currently unavailable."
        )
        return None


##############################################################################
### The actual script. KEEP THE `import filter` ALIVE AT ALL TIMES
##############################################################################

if __name__ == '__main__':
    # get version from git and update fastmat/__init__.py accordingly
    try:
        with open(".version", "r") as f:
            lines = [str(s) for s in [ln.strip() for ln in f] if len(s)]
        packageVersion = lines[0]
    except IOError as e:
        Error("Setting package version", e)
    except IndexError as e:
        Error("Version file is empty", e)

    # make sure there exists a version.py file in the project
    with open(strVersionFile, "w") as f:
        f.write(VERSION_PY % (packageVersion))
    print("Set %s to '%s'" % (strVersionFile, packageVersion))

    # get the long description from the README file.
    # CAUTION: Python2/3 utf encoding shit calls needs some adjustments
    fileName = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'README.md'
    )

    f = (open(fileName, 'r') if sys.version_info < (3, 0)
         else open(fileName, 'r', encoding='utf-8'))
    longDescription = f.read()
    f.close()

    pypiName = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'pypi.md'
    )

    f = (open(fileName, 'r') if sys.version_info < (3, 0)
         else open(fileName, 'r', encoding='utf-8'))
    pypiDescription = f.read()
    f.close()

    # Build for generic (legacy) architectures when enviroment variable
    # (FASTMAT_GENERIC) is defined
    if 'FASTMAT_COMPILER_OPTIONS' in os.environ:
        marchFlag = os.environ['FASTMAT_COMPILER_OPTIONS']
        mtuneFlag = ''
        WARNING("Passing special build options: " + marchFlag)
    elif (
        ('FASTMAT_GENERIC' in os.environ) and
        (bool(int(os.environ['FASTMAT_GENERIC'])))
    ):
        marchFlag = '-march=x86-64'
        mtuneFlag = '-mtune=core2'
        WARNING("Building package for generic architectures")
    else:
        marchFlag = '-march=native'
        mtuneFlag = '-mtune=native'

    # define different compiler arguments for each platform
    strPlatform = platform.system()
    compilerArguments = []
    linkerArguments = []
    useGccOverride = False
    if strPlatform == 'Windows':
        # Microsoft Visual C++ Compiler 9.0
        compilerArguments += ['/O2', '/fp:precise', marchFlag]
    elif strPlatform == 'Linux':
        # assuming Linux and gcc
        compilerArguments.extend(['-Ofast', marchFlag])
        if len(mtuneFlag):
            compilerArguments.append(mtuneFlag)

        useGccOverride = True
    elif strPlatform == 'Darwin':
        # assuming Darwin
        compilerArguments.extend(['-Ofast', marchFlag])
        if len(mtuneFlag):
            compilerArguments.append(mtuneFlag)
    else:
        WARNING("Your platform is currently not supported by %s: %s" % (
            packageName, strPlatform))

    # define default cython directives, these may get extended along the script
    cythonDirectives = {'language_level': '3str'}
    defineMacros = []
    CMD_COVERAGE = '--enable-cython-tracing'
    if CMD_COVERAGE in sys.argv:
        sys.argv.remove(CMD_COVERAGE)
        cythonDirectives['linetrace'] = True
        cythonDirectives['binding'] = True
        defineMacros += [('CYTHON_TRACE_NOGIL', '1'),
                         ('CYTHON_TRACE', '1')]
        print("Enabling cython line tracing allowing code coverage analysis")

    print("Building %s v%s for %s." % (
        packageName,
        packageVersion,
        strPlatform)
    )

    # check if all requirements are met prior to actually calling setup()
    setupRequires = []
    installRequires = []
    sphinxRequires = ['sphinx', 'sphinx_rtd_theme', 'numpydoc', 'matplotlib']
    checkRequirement(setupRequires, 'setuptools', 'setuptools>=18.0')
    checkRequirement(setupRequires, 'Cython', 'cython>=0.29')
    if sys.version_info < (3, 5):
        checkRequirement(setupRequires, 'numpy', 'numpy<1.17')
    else:
        checkRequirement(setupRequires, 'numpy', 'numpy>=1.16.3')

    checkRequirement(installRequires, 'six', 'six')
    checkRequirement(installRequires, 'scipy', 'scipy>=1.0')

    print("Requirements for setup: %s" % (setupRequires))
    print("Requirements for install: %s" % (installRequires))

    # everything's set. Fire in the hole.
    setup(
        name=packageName,
        version=packageVersion,
        description='fast linear transforms in Python',
        long_description=pypiDescription,
        long_description_content_type='text/markdown',
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
            'Operating System :: POSIX :: Other',
            'Operating System :: MacOS :: MacOS X',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Software Development :: Libraries'
        ],
        keywords='linear transforms efficient algorithms mathematics',
        setup_requires=setupRequires,
        install_requires=installRequires,
        packages=[
            'fastmat',
            'fastmat/algorithms',
            'fastmat/core',
            'fastmat/inspect'
        ],
        cmdclass={'build_doc': doc_opts()},
        command_options={
            'build_doc': {
                'project': ('setup.py', packageName),
                'version': ('setup.py', packageVersion),
                'release': ('setup.py', packageVersion),
                'copyright': ('setup.py', '2017, ' + packageName)
            }},
        ext_modules=lazyCythonize(extensions)
    )
