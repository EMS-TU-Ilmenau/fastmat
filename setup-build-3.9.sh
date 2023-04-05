# make sure to use the oldest numpy version that is acceptable
# for building binaries, since the ABI of numpy is designed to
# be forward-compatible with future releases but not necessarily
# also backward-pcompatible with packages that were compiled
# against later releases.
# numpy 1.16.3 is the oldest release without a major security flaw
# rel: https://snyk.io/vuln/SNYK-PYTHON-NUMPY-73513

# The latest numpy version of the corresponding C-API version 0x13 is 1.19.3, which supports wheels beginning from Python 3.6.
# With the combination numpy==1.16.3 and scipy==1.1.0, both Python 2.7 and 3.6 can still be supported.
# The latest Python version to support both is Python 3.7, which will be used to build these points.
# Note that there's no ARM support yet for these old versions
# For {num,sci}py of this version combination, wheels exists for the python versions and architectures indicated below
# The latest numpy version of the corresponding C-API version 0x13 is 1.19.3, to which scipy 1.5.4 fits
# For {num,sci}py of this version combination, wheels exists for the python versions and architectures indicated below
python -m pip install --only-binary :all: \
    cython>=0.29 numpy==1.19.3 scipy==1.5.4 cibuildwheel==1.9 twine

export CIBW_BUILD="cp36* cp37* cp38* cp39*"
export CIBW_ARCHS_WINDOWS="AMD64 x86"
export CIBW_ARCHS_LINUX="x86_64 i686 aarch64"
export CIBW_ARCHS_MACOS="x86_64 arm64"
