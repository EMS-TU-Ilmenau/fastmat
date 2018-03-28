import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "BlkTwoLvlToepCu.h":
    cdef cppclass C_BlkTwoLvlToepGPU "BlkTwoLvlToepGPU":
        C_BlkTwoLvlToepGPU(int, int, int)
        void forward(np.complex128_t*, np.complex128_t*)

cdef class BlkTwoLvlToepGPU:
    cdef C_BlkTwoLvlToepGPU* f

    def __cinit__(
        self,
        int nZ,
        int nX,
        int nY
    ):
        self.f = new C_BlkTwoLvlToepGPU(nZ, nX, nY)

    def forward(
        self,
        np.ndarray[ndim=1, dtype=np.complex128_t] arrIn,
        np.ndarray[ndim=1, dtype=np.complex128_t] arrOut
    ):
        self.f.forward(&arrIn[0], &arrOut[0])
