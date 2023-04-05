# For Python 3.10, wheel support starts with numpy 1.20, for which the C-API is version 0x14.
# The youngest version of this C-API version is 1.21.6, which will be used to build against.
# Corresponding scipy version is 1.7.3
python -m pip install --only-binary :all: \
    cython>=0.29 numpy==1.21.6 scipy==1.7.3 cibuildwheel==2.12.1 twine

export CIBW_BUILD="cp310*"
export CIBW_ARCHS_WINDOWS="AMD64 x86"
export CIBW_ARCHS_LINUX="x86_64 i686 aarch64"
export CIBW_ARCHS_MACOS="x86_64 arm64"
