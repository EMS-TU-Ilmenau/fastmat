..  Copyright 2016 Sebastian Semper, Christoph Wagner
        https://www.tu-ilmenau.de/it-ems/

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

.. _types:

Data Types in fastmat
=====================

To achieve high performance, fastmat is designed to support common data types
only, namely

  * Floating point with single and double precision (`float32` and `float64`)
  * Complex floating point with single and double precision (`complex64` and
    `complex128`)
  * Signed integer of selected fixed sizes (`int8`, `int16`, `int32`, `int64`)

Some implementation of fastmat matrices use numpy under the hood. Although
those could technically be able to deal with other data types offered by
``numpy`` as well, using other types than those listed above is disencouraged.
This is important to ensure consistency throughout the intestines of
``fastmat``, which is important for being able to reliably test the package.

The following sections detail the organization and handling of types in
``fastmat`` and explain the mechanisms how fastmat handles type promotion. The
final section references the internal type API of ``fastmat``.

Type handling
-------------

**ftype**

To distinguish between the supported data types ``fastmat`` uses the ``ftype``
internally as type identifier. All data type checks within the package as well
as the type promotion logic is internally handled via these type numbers, that
correspond directly to the associated ``numpy.dtype`` given in this table:

+--------------------------+------------------+-------------------+-------+
| Data Type                | ``numpy.dtype``  | fastmat ``ftype`` | Short |
+--------------------------+------------------+-------------------+-------+
| Signed Integer 8 bit     | ``int8_t``       |        0          |  i8   |
| Signed Integer 16 bit    | ``int16_t``      |        1          |  i16  |
| Signed Integer 32 bit    | ``int32_t``      |        2          |  i32  |
| Signed Integer 64 bit    | ``int64_t``      |        3          |  i64  |
| Single-precision Float   | ``float32_t``    |        4          |  f32  |
| Double-Precision Float   | ``float64_t``    |        5          |  f64  |
| Single-Precision Complex | ``complex64_t``  |        6          |  c64  |
| Double-Precision Complex | ``complex128_t`` |        7          |  c128 |
+--------------------------+------------------+-------------------+-------+

Type promotion
--------------

Type promotion matrix of binary operators of kind ``f(A, B)`` as used
throughout ``fastmat``:

+--------------------+-------------------------------------------------------+
|                    |                          B                            |
|                    +---------------------------+-------------+-------------+
|   Type promotion   |            int            |    float    |   complex   |
|                    +------+------+------+------+------+------+------+------+
|                    | i8   | i16  | i32  | i64  | f32  | f64  | c64  | c128 |
+---+----------------+------+------+------+------+------+------+------+------+
|   |     int 8      | i8   | i16  | i32  | i64  | f32  | f64  | c64  | c128 |
|   +----------------+------+------+------+------+------+------+------+------+
|   |     int 16     | i16  | i16  | i32  | i64  | f32  | f64  | c64  | c128 |
|   +----------------+------+------+------+------+------+------+------+------+
|   |     int 32     | i32  | i32  | i32  | i64  | f64  | f64  | c128 | c128 |
|   +----------------+------+------+------+------+------+------+------+------+
|   |     int 64     | i64  | i64  | i64  | i64  | f64  | f64  | c128 | c128 |
| A +----------------+------+------+------+------+------+------+------+------+
|   |    float 32    | f32  | f32  | f64  | f64  | f64  | f64  | c128 | c128 |
|   +----------------+------+------+------+------+------+------+------+------+
|   |    float 64    | f64  | f64  | f64  | f64  | f64  | f64  | c128 | c128 |
|   +----------------+------+------+------+------+------+------+------+------+
|   |   complex 64   | c64  | c64  | c128 | c128 | c64  | c128 | c64  | c128 |
|   +----------------+------+------+------+------+------+------+------+------+
|   |   complex 128  | c128 | c128 | c128 | c128 | c128 | c128 | c128 | c128 |
+---+----------------+------+------+------+------+------+------+------+------+

Example:
    The forward operator of a :py:class:`fastmat.Matrix` of type `float 32`
    will, if provided with an `int 32` input vector, yield an output vector of
    type `float 64`.

.. Note::
    The output data type will be expanded to fit the mantissa of any of the
    operands best. As `int 32` has a wider mantissa than `float 32` offers,
    the example type promotion will yield `float 64` to maintain accuracy.

.. Note::
    Data types will not be expanded automatically to the next larger data
    type for the sake of preventing overflows. You'll need to specifically
    expand the data type -- where necessary -- by specifying ``minType=?``
    during the generation of your :py:class:`fastmat.Matrix` instance.


``fastmat.core.types``
----------------------

.. automodule:: fastmat.core.types
    :members:
    :undoc-members:
    :show-inheritance:
