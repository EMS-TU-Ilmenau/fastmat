# For Python 3.11, wheel support starts with numpy 1.21, for which the C-API is version 0x16.
# There is no wheeled version of C-API 0x15, so there could potentially be issues for source-builds of numpy on Python 3.11
# The youngest version of this C-API version is 1.24.2, which will be used to build against.
# Corresponding scipy version is 1.10.1
python -m pip install --only-binary :all: \
    cython>=0.29 numpy==1.24.2 scipy==1.10.1 cibuildwheel==2.12.1 twine

export CIBW_BUILD="cp311*"
export CIBW_ARCHS_WINDOWS="AMD64"
export CIBW_ARCHS_LINUX="x86_64 aarch64"
export CIBW_ARCHS_MACOS="x86_64 arm64"
