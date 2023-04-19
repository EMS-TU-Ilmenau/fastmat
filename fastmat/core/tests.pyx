# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False

# Copyright 2023 Sebastian Semper, Christoph Wagner
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

import numpy as np
cimport numpy as np

from .strides cimport *
from .types cimport getTypeInfo
from .types import _typeInfo, _typeSelection

import unittest


class TestCore(unittest.TestCase):
    # opCopyVector(STRIDE_s *, intsize, STRIDE_s *, intsize)
    # opZeroVector(STRIDE_s *, intsize)
    # opZeroVectors(STRIDE_s *)
    # strideCopy(STRIDE_s *, STRIDE_s *)
    # strideSliceVectors(STRIDE_s *, intsize, intsize, intsize)
    # strideSliceElements(STRIDE_s *, intsize, intsize, intsize)
    # strideSubgridVector(
    #     STRIDE_s *, intsize, intsize, intsize, intsize, intsize, intsize
    # )
    # strideFlipVectors(STRIDE_s *)
    # strideFlipElements(STRIDE_s *)
    # stridePrint(STRIDE_s *, text=?)

    def test_strides(self):
        cdef STRIDE_s stride, stride_arr
        cdef STRIDE_s stride_arr_B, stride_arr_B_flipped, stride_subgrid
        cdef STRIDE_s stride_arr_C, stride_arr_C_flipped
        cdef object dtype, order
        cdef np.ndarray arr, arr_ref, arr_base, arr_B, arr_C
        cdef intsize ii, nn, axis, n = 9, m = 12, d = 3

        def queryStrideInit(axis, dtype):
            cdef STRIDE_s stride
            cdef np.ndarray arr = np.zeros((10, 10), dtype=dtype)
            strideInit(&stride, arr, axis)

        print()
        self.assertRaises(ValueError, lambda: queryStrideInit(2, int))
        self.assertRaises(OverflowError, lambda: queryStrideInit(256, int))
        self.assertRaises(TypeError, lambda: queryStrideInit(0, object))
        for dtype in [
            np.int8, np.int16, np.int32, np.int64,
            np.float32, np.float64, np.complex64, np.complex128
        ]:
            for order in ['F', 'C', 'S']:
                for axis in range(0, 2):
                    ###
                    ###  strideInit(), strideCopy(), stridePrint()
                    ###
                    # construct a reference array, to which all striding
                    # operations will be applied in numpy syntax. For the
                    # 'S' (aka 'strided') order, that should reflect the
                    # actual array's contents with similar numpy
                    slice_base = lambda arr_base: arr_base[::d, ::d]
                    arr = np.arange(n * m, dtype=dtype)
                    if order == 'S':
                        arr_base = np.ones((n * d, m * d), dtype=dtype)
                        slice_base(arr_base)[...] = arr.reshape((n, m))
                        arr = slice_base(arr_base)
                        arr_ref = arr.copy()
                        self.assertTrue(arr_base.flags.owndata)
                        self.assertFalse(arr.flags.owndata)
                    else:
                        arr = arr.reshape((n, m)).copy(order=order)
                        arr_base = arr[...]
                        arr_ref = arr.copy()
                        self.assertTrue(arr.flags.owndata)

                    self.assertTrue(arr_ref.flags.owndata)

                    strideInit(&stride, arr, axis)
                    stridePrint(&stride, text="axis=%d, order=%s, dtype=%s" %(
                        axis, order, repr(dtype)
                    ))

                    # test that the stride-modified array matches the numpy-
                    # modified reference array. Also check for presence of the
                    # "guard-ones" present in the base array for 'S'-order
                    def check(arr, arr_ref, arr_base):
                        np.testing.assert_array_equal(arr, arr_ref)
                        if arr_base.size != arr.size:
                            arr_vanilla_base = np.ones_like(arr_base)
                            slice_base(arr_vanilla_base)[...] = arr
                            np.testing.assert_array_equal(
                                arr_base, arr_vanilla_base
                            )

                    # first check our ground truth, after initialization
                    check(arr, arr_ref, arr_base)

                    ###
                    ###  strideFlipElements(), strideFlipVectors(),
                    ###  strideSliceVectors(),
                    ###  opCopyVector(), opZeroVector()
                    ###
                    # Construct two test grounds, one each for the
                    # ~Vectors and ~Elements function ...
                    strideCopy(&stride_arr, &stride)
                    arr_base_B = np.ones_like(arr_base)
                    arr_base_C = np.ones_like(arr_base)
                    if order == 'S':
                        arr_B = slice_base(arr_base_B)
                        arr_C = slice_base(arr_base_C)
                    else:
                        arr_B = arr_base_B[...]
                        arr_C = arr_base_C[...]
                    # ... , construct the slices ...
                    strideInit(&stride_arr_B, arr_B, axis)
                    strideCopy(&stride_arr_B_flipped, &stride_arr_B)
                    strideInit(&stride_arr_C, arr_C, axis)
                    strideCopy(&stride_arr_C_flipped, &stride_arr_C)
                    # ..., subselect them in an interleaving fashion ...
                    # The indexing offsets correspond to the if-clause of the
                    # for-loop below and ensures, that both opCopyVector and
                    # opZeroVector will see all indices in all slice ranges
                    # We also need to make sure that the arr_B_flipped stride
                    # touches the correct end, due to the extra flip
                    strideSliceVectors(
                        &stride_arr_B_flipped,
                        (stride_arr.numVectors + 0) % 2, -1, 2
                    )
                    strideSliceVectors(
                        &stride_arr_C,
                        (stride_arr.numVectors + 0 + axis) % 2, -1, 2
                    )
                    strideSliceVectors(
                        &stride_arr_B,
                        (stride_arr.numVectors + 1) % 2, -1, 2
                    )
                    strideSliceVectors(
                        &stride_arr_C_flipped,
                        (stride_arr.numVectors + 1 + axis) % 2, -1, 2
                    )
                    strideFlipVectors(&stride_arr_B_flipped)
                    strideFlipElements(&stride_arr_C_flipped)

                    # stridePrint(&stride_arr, text='arr')
                    # stridePrint(&stride_arr_B, text='arr_B')
                    # stridePrint(&stride_arr_B_flipped, text='arr_B_flipped')
                    # stridePrint(&stride_arr_C, text='arr_C')
                    # stridePrint(&stride_arr_C_flipped, text='arr_C_flipped')
                    # ... , copy or zero like mad ...
                    for nn in range(stride_arr.numVectors):
                        ii = nn >> 1
                        if nn % 2 == 0:
                            opZeroVector(&stride_arr_B, ii)
                            opZeroVector(&stride_arr_C, ii)
                        else:
                            opCopyVector(
                                &stride_arr_B_flipped, ii, &stride_arr, nn
                            )
                            opCopyVector(
                                &stride_arr_C_flipped, ii, &stride_arr, nn
                            )

                    # ... and check against a similarly numpy-modified array.
                    # But keep the stride axis selection in mind, which means
                    # that the references must be swapped accordingly
                    arr_ref_B = arr_ref[:, ::-1].copy()
                    arr_ref_C = arr_ref[::-1, :].copy()
                    if axis == 1:
                        arr_ref_B[0::2, :] = 0
                        arr_ref_C[0::2, :] = 0
                    else:
                        arr_ref_B[:, 1::2] = 0
                        arr_ref_C[:, 0::2] = 0

                    if axis == 1:
                        arr_ref_B, arr_ref_C = arr_ref_C, arr_ref_B

                    # print(repr(arr_B))
                    # print(repr(arr_ref_B))
                    # print(repr(arr_base_B))
                    # print(repr(arr_B - arr_ref_B))
                    # print()
                    # print(repr(arr_C))
                    # print(repr(arr_ref_C))
                    # print(repr(arr_base_C))
                    # print(repr(arr_C - arr_ref_C))
                    # print()

                    check(arr_B, arr_ref_B, arr_base_B)
                    check(arr_C, arr_ref_C, arr_base_C)

                    ###
                    ###  strideSliceElements(), strideSubgridVector(),
                    ###  opZeroVectors()
                    ###
                    # Construct two test grounds, one each for the
                    # ~Vectors and ~Elements function ...
                    strideCopy(&stride_arr, &stride)
                    arr_base_B = np.ones_like(arr_base)
                    arr_B = slice_base(arr_base_B) if order == 'S' \
                        else arr_base_B[...]

                    # ... , construct the slices ...
                    strideInit(&stride_arr_B, arr_B, axis)

                    arr_B[...] = arr
                    strideCopy(&stride_subgrid, &stride_arr_B)
                    opZeroVectors(&stride_subgrid)
                    check(arr_B, np.zeros_like(arr_B), arr_base_B)

                    np.set_printoptions(linewidth=150, edgeitems=100)

                    arr_B[...] = arr
                    strideFlipVectors(&stride_subgrid)
                    strideFlipElements(&stride_subgrid)
                    opZeroVectors(&stride_subgrid)
                    check(arr_B, np.zeros_like(arr_B), arr_base_B)

                    arr_B[...] = arr
                    eleFirst = stride_subgrid.numElements // 3
                    eleLast = (stride_subgrid.numElements * 2 - 1) // 3
                    strideCopy(&stride_subgrid, &stride_arr_B)
                    strideSliceElements(&stride_subgrid, eleFirst, eleLast, 1)
                    opZeroVectors(&stride_subgrid)
                    arr_ref_B = arr_ref.copy()
                    if axis == 0:
                        arr_ref_B[eleFirst:eleLast + 1, :] = 0
                    else:
                        arr_ref_B[:, eleFirst:eleLast + 1] = 0

                    check(arr_B, arr_ref_B, arr_base_B)

                    arr_B[...] = arr
                    strideCopy(&stride_subgrid, &stride_arr_B)
                    # Construct a subgrid of size [2 x 4],
                    # starting from index [1, 1] in F-contiguous style
                    strideSubgrid(&stride_subgrid, 1, 1, 2, 1, 4, 2)
                    strideSliceElements(&stride_subgrid, -1, -1, 1)
                    opZeroVectors(&stride_subgrid)
                    arr_ref_B = arr_ref.copy()
                    view = arr_ref_B[1:1 + 8, 1] if axis == 0 \
                        else arr_ref_B[1, 1:1 + 8]

                    view.reshape((2, 4), order='F')[-1, :] = 0

                    check(arr_B, arr_ref_B, arr_base_B)

    def test_types(self):

        print("")
        _typeInfo()
        _typeSelection()

        def queryTypeInfo(objType):
            cdef INFO_TYPE_s * info
            info = getTypeInfo(objType)
            return info[0].fusedType

        print(queryTypeInfo(np.int16))
        self.assertRaises(TypeError, lambda: queryTypeInfo(object))
