#include <cuda.h>
#include <complex>
#include <cufft.h>
#include <cuComplex.h>

class BlkTwoLvlToepGPU{

    // pointer to the GPU memory where the arrays are stored
    cuDoubleComplex* arrX_device;
    cuDoubleComplex* arrT_device;
    cuDoubleComplex* arrY_device;

    // number of blocks
    int nZ;

    // size of the first level (NOT number of defining elements!)
    int nX;

    // size of the second level (NOT number if  defining elements!)
    int nY;

    // size of whole matrix
    int nN;

    // size of defining elements for one single block in memory
    size_t memSingleBlock;

    // size of the whole input vector
    size_t memWholeVec;

    // FFT plans
    cufftHandle planWhole; // plan for the whole input and output
    cufftHandle planBlock; // plan for the single blocks

    public:
        // the constructor
        BlkTwoLvlToepGPU(
            int,
            int,
            int
        );

        ~BlkTwoLvlToepGPU();

        void forward(
            std::complex<double>*,
            std::complex<double>*
        );
};
