#include "BlkTwoLvlToepCu.h"
#include <cuda.h>
#include <complex>
#include <assert.h>
#include <cuComplex.h>
#include <iostream>

using namespace std;


void __global__ kernelDefElements(
    int ii,
    int jj,
    int Nx,
    int Ny,
    cuDoubleComplex* t
){
    int gidx = threadIdx.x + blockDim.x*blockIdx.x;
    int gidy = threadIdx.y + blockDim.y*blockIdx.y;

    if (gidy < (2 * Ny - 1)){
        if (gidx < (2 * Nx - 1)){
            t[(2 * Ny - 1) * gidx + gidy] = make_cuDoubleComplex(1.0,0.0);
        }
    }
}

void __global__ kernelMult(
    int ii,
    int jj,
    int Nx,
    int Ny,
    cuDoubleComplex* t,
    cuDoubleComplex* x,
    cuDoubleComplex* y
){
    int gidx = threadIdx.x + blockDim.x*blockIdx.x;
    int gidy = threadIdx.y + blockDim.y*blockIdx.y;

    if (gidy < (2 * Ny - 1)){
        if (gidx < (2 * Nx - 1)){
            y[
                (2 * Ny - 1) * gidx + gidy
            ] = cuCmul(
                t[(2 * Ny - 1) * gidx + gidy],
                x[(2 * Ny - 1) * gidx + gidy]
            );
        }
    }
}

BlkTwoLvlToepGPU::BlkTwoLvlToepGPU (
    int nZ_,
    int nX_,
    int nY_
){
    nZ = nZ_;
    nX = nX_;
    nY = nY_;

    nN = nZ * nX * nY;

    memWholeVec = sizeof(complex<double>) * nN;
    memSingleBlock = sizeof(complex<double>) * (2 * nX - 1) * (2 * nY - 1);


    cudaError_t err;

    // allocate memory for defining elements of the blocks
    err = cudaMalloc((void**) &arrT_device, memSingleBlock);
    assert(err == 0);

    // allocate memory for the whole input vector
    err = cudaMalloc((void**) &arrX_device, memWholeVec);
    assert(err == 0);

    // memory for the whole output vector
    err = cudaMalloc((void**) &arrY_device, memWholeVec);
    assert(err == 0);

    int fftDims[2] = {2 * nX - 1, 2 * nY - 1};
    cufftPlanMany(
        &planWhole,
        2,
        fftDims,
        NULL,
        0,
        0,
        NULL,
        0,
        0,
        CUFFT_Z2Z,
        1
    );

    cufftPlan2d(
        &planBlock,
        2 * nX - 1,
        2 * nY - 1,
        CUFFT_Z2Z
    );


}

void BlkTwoLvlToepGPU::forward(
    std::complex<double>* arrX,
    std::complex<double>* arrRes
) {
    /*
    Calculation Routine for the Forward Projection

    0. Step:
        Initalization

    1. Step:
        Copy the input data arrX, which is a 1D array
        from the host to the device

    2. Step:
        Apply correct zeropadding

    3. Step:
        Calculate a 2D FFT over the input on the device
        directly.

    4. Step:
        a) Iterate over the blocks vertically
            b) Iterate over the blocks horizontally
                - calc the elements of the current block in
                  parallel -> kernelDefElements
                - wait for all threads to finish
                - calc the 2dfft of the current generating
                  elements inplace
                - do the multiplication simultaneously
                  and in parallel and write in
                  arrY_device -> kernelMult
                - wait for all threads to finish

    5. Step:
        do an inplace 2DiFFT

    6.
        copy back the memory from arrY_device to
        arrRes


    */

    // 0. Step
    dim3 blockSize(32,32);
    dim3 gridSize(nX / 32 + 1, nY / 32 + 1);

    // 1. Step
    cudaMemcpy(
        arrX_device,
        reinterpret_cast<cuDoubleComplex*>(arrX),
        memWholeVec,
        cudaMemcpyHostToDevice
    );

    // 3. Step
    cufftExecZ2Z(
        planWhole,
        reinterpret_cast<cufftDoubleComplex*>(arrX_device),
        reinterpret_cast<cufftDoubleComplex*>(arrX_device),
        CUFFT_FORWARD
    );

    // 4. Step
    // a)
    for (int ii = 0; ii < nZ; ii++){
        // b)
        for (int jj = 0; jj < nZ; jj++){
            kernelDefElements<<<gridSize, blockSize>>>(
                ii,
                jj,
                nX,
                nY,
                reinterpret_cast<cuDoubleComplex*>(arrT_device)
            );

            cout << ii << endl;

            cudaThreadSynchronize();

            cufftExecZ2Z(
                planBlock,
                reinterpret_cast<cufftDoubleComplex*>(arrT_device),
                reinterpret_cast<cufftDoubleComplex*>(arrT_device),
                CUFFT_FORWARD
            );

            cudaThreadSynchronize();

            kernelMult<<<gridSize, blockSize>>>(
                ii,
                jj,
                nX,
                nY,
                arrT_device,
                arrX_device,
                arrY_device
            );

            cudaThreadSynchronize();
        }
    }

    // 5. Step
    cufftExecZ2Z(
        planWhole,
        reinterpret_cast<cufftDoubleComplex*>(arrY_device),
        reinterpret_cast<cufftDoubleComplex*>(arrY_device),
        CUFFT_INVERSE
    );

    // 6. Step
    cudaMemcpy(
        arrRes,
        reinterpret_cast<complex<double>*>(arrY_device),
        memWholeVec,
        cudaMemcpyDeviceToHost
    );
}


BlkTwoLvlToepGPU::~BlkTwoLvlToepGPU() {
    cudaFree(arrX_device);
    cudaFree(arrT_device);
    cudaFree(arrY_device);
}
