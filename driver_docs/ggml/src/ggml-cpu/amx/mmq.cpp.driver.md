# Purpose
This C++ source code file is designed to optimize matrix multiplication operations using Intel's Advanced Matrix Extensions (AMX) and AVX-512 instructions. The file provides a set of functions and templates to handle various data types and quantization formats, specifically targeting high-performance computing environments. The code is structured to support different quantized data types, such as `block_q4_0`, `block_q4_1`, `block_q8_0`, and others, which are used to represent quantized matrices in a compact form. These quantized types are then processed using AMX and AVX-512 instructions to perform efficient matrix multiplications.

The file includes several key components: macros for dispatching operations based on data types, templates for unrolling loops to optimize performance, and functions for packing and unpacking data into AMX-friendly formats. It also defines specialized kernels for performing matrix multiplications using AMX and AVX-512 instructions, which are tailored to handle different block sizes and quantization formats. The code is intended to be part of a larger library or application that requires high-performance matrix operations, such as machine learning frameworks or scientific computing applications. It provides a public API for initializing tile configurations, converting weights to packed formats, and executing matrix multiplications with quantized data, making it a crucial component for applications that leverage Intel's hardware acceleration features.
# Imports and Dependencies

---
- `amx.h`
- `mmq.h`
- `ggml-impl.h`
- `ggml-cpu-impl.h`
- `ggml-cpu-quants.h`
- `ggml-quants.h`
- `algorithm`
- `type_traits`
- `sys/syscall.h`
- `unistd.h`


# Data Structures

---
### Unroll<!-- {{#data_structure:(anonymous)::Unroll}} -->
- **Type**: ``struct``
- **Description**: The `Unroll` struct is a template-based utility designed to facilitate loop unrolling at compile time. It recursively calls itself with a decremented template parameter `n` until it reaches the base case, which is specialized for `n = 1`. This struct is used to apply a function `f` to a sequence of integral constants, effectively unrolling a loop of size `n` by invoking the function with each index from `0` to `n-1`. The `ALWAYS_INLINE` attribute ensures that the function calls are inlined, optimizing performance by reducing function call overhead.
- **Member Functions**:
    - [`(anonymous)::Unroll::operator()`](#(anonymous)::Unroll::operator())

**Methods**

---
#### Unroll::operator\(\)<!-- {{#callable:(anonymous)::Unroll::operator()}} -->
The `operator()` function in the `Unroll` struct template recursively applies a given function `f` to a sequence of arguments, decrementing the template parameter `n` until it reaches zero.
- **Inputs**:
    - `f`: A callable object (function or function object) that takes an integral constant and additional arguments.
    - `args`: A variadic list of arguments to be passed to the function `f`.
- **Control Flow**:
    - The function is called recursively by creating an instance of `Unroll<n - 1>` and invoking its `operator()` with the same function `f` and arguments `args...`.
    - After the recursive call, the function `f` is invoked with an integral constant representing the current value of `n - 1` and the arguments `args...`.
- **Output**: This function does not return any value; it performs operations through side effects of the function `f`.
- **See also**: [`(anonymous)::Unroll`](#(anonymous)::Unroll)  (Data Structure)



---
### PackedTypes<!-- {{#data_structure:(anonymous)::PackedTypes}} -->
- **Type**: ``struct``
- **Members**:
    - `type`: Defines a type alias for a specific template specialization.
- **Description**: `PackedTypes` is a template structure that provides a type alias for specific template specializations. It is used to map certain types, such as `block_q4_0`, to a corresponding type, like `int8_t`. This structure is part of a larger system that deals with type traits and type conversions, particularly in the context of quantized data types and their processing in a high-performance computing environment.


---
### do\_compensate<!-- {{#data_structure:(anonymous)::do_compensate}} -->
- **Type**: ``struct``
- **Description**: The `do_compensate` struct is a template specialization of `std::integral_constant` that evaluates to `true` if the template parameter `T` is the same type as `block_q8_0`. This struct is used as a type trait to determine if compensation is needed for a given type `T` in certain operations, particularly in the context of quantized data processing.
- **Inherits From**:
    - `std::integral_constant<bool,
    std::is_same<T, block_q8_0>::value>`


---
### do\_unpack<!-- {{#data_structure:(anonymous)::do_unpack}} -->
- **Type**: `struct`
- **Description**: The `do_unpack` struct is a template specialization of `std::integral_constant` that evaluates to `true` if the template type `T` is either `block_q4_0` or `block_q4_1`. This struct is used as a type trait to determine if a given type should be unpacked, specifically for types related to quantized data blocks in the context of the code.
- **Inherits From**:
    - `std::integral_constant<bool,
    std::is_same<T, block_q4_0>::value ||
    std::is_same<T, block_q4_1>::value>`


---
### is\_type\_qkk<!-- {{#data_structure:(anonymous)::is_type_qkk}} -->
- **Type**: `struct`
- **Description**: The `is_type_qkk` struct is a type trait that inherits from `std::integral_constant<bool, ...>` and is used to determine if a given type `T` is one of the specified types: `block_q4_K`, `block_q5_K`, `block_q6_K`, or `block_iq4_xs`. It evaluates to `true` if `T` matches any of these types, otherwise it evaluates to `false`. This struct is useful for compile-time type checking and conditional compilation based on type properties.
- **Inherits From**:
    - `std::integral_constant<bool,
    std::is_same<T, block_q4_K>::value ||
    std::is_same<T, block_q5_K>::value ||
    std::is_same<T, block_q6_K>::value ||
    std::is_same<T, block_iq4_xs>::value>`


---
### tile\_config\_t<!-- {{#data_structure:(anonymous)::tile_config_t}} -->
- **Type**: `struct`
- **Members**:
    - `palette_id`: An 8-bit unsigned integer representing the palette identifier, initialized to 0.
    - `start_row`: An 8-bit unsigned integer representing the starting row, initialized to 0.
    - `reserved_0`: An array of 14 8-bit unsigned integers reserved for future use, initialized to 0.
    - `colsb`: An array of 16 16-bit unsigned integers representing column sizes, initialized to 0.
    - `rows`: An array of 16 8-bit unsigned integers representing row sizes, initialized to 0.
- **Description**: The `tile_config_t` struct is designed to configure AMX (Advanced Matrix Extensions) tile operations, specifically for handling matrix multiplication with quantized data types. It includes fields for identifying the palette, specifying the starting row, and arrays for storing column and row sizes. The `reserved_0` array is included for potential future extensions or alignment purposes. This struct is integral to setting up the tile configuration for efficient matrix operations using AMX instructions.


---
### acc\_C<!-- {{#data_structure:(anonymous)::acc_C}} -->
- **Type**: ``struct``
- **Description**: The `acc_C` structure is a templated struct that is defined with three template parameters: `TA`, `TB`, and a boolean `is_acc`. It is used as a placeholder for specialized implementations of matrix accumulation operations, particularly in the context of quantized matrix multiplication using Intel's Advanced Matrix Extensions (AMX) and AVX512VNNI instructions. The struct itself does not contain any members or fields, but it is specialized for different combinations of template parameters elsewhere in the code to perform specific operations.


---
### tinygemm\_kernel\_avx<!-- {{#data_structure:(anonymous)::tinygemm_kernel_avx}} -->
- **Type**: `struct`
- **Description**: The `tinygemm_kernel_avx` is a templated C++ struct designed to perform matrix multiplication using AVX (Advanced Vector Extensions) instructions. It is parameterized by types `TA`, `TB`, `TC` for the matrix elements and integer constants `BLOCK_M`, `BLOCK_N`, `BLOCK_K` for block sizes. The struct contains a static method `apply` which is intended to execute the matrix multiplication operation, although in the provided code, the method body is empty and uses `GGML_UNUSED` to suppress unused parameter warnings. This struct is part of a larger codebase that likely implements optimized matrix operations for specific hardware capabilities.
- **Member Functions**:
    - [`(anonymous)::tinygemm_kernel_avx::apply`](#(anonymous)::tinygemm_kernel_avx::apply)

**Methods**

---
#### tinygemm\_kernel\_avx::apply<!-- {{#callable:(anonymous)::tinygemm_kernel_avx::apply}} -->
The `apply` function is a static method in the `tinygemm_kernel_avx` struct that takes five parameters but does not perform any operations with them.
- **Inputs**:
    - `K`: An integer representing a dimension size, typically used in matrix operations.
    - `A`: A pointer to a constant array of type `TA`, marked with `RESTRICT`, indicating it points to the first element of a matrix or data block.
    - `B`: A pointer to a constant array of type `TB`, marked with `RESTRICT`, indicating it points to the first element of a matrix or data block.
    - `C`: A pointer to an array of type `TC`, marked with `RESTRICT`, indicating it points to the first element of a matrix or data block where results might be stored.
    - `ldc`: An integer representing the leading dimension of the matrix `C`, typically used in matrix operations to indicate the number of elements between successive rows.
- **Control Flow**:
    - The function begins by declaring the parameters `K`, `A`, `B`, `C`, and `ldc` as unused using the `GGML_UNUSED` macro.
    - No operations or computations are performed within the function body.
- **Output**: The function does not return any value or produce any output.
- **See also**: [`(anonymous)::tinygemm_kernel_avx`](#(anonymous)::tinygemm_kernel_avx)  (Data Structure)



---
### tinygemm\_kernel\_vnni<!-- {{#data_structure:(anonymous)::tinygemm_kernel_vnni}} -->
- **Type**: `struct`
- **Description**: The `tinygemm_kernel_vnni` is a templated struct in C++ designed for matrix multiplication operations using the VNNI (Vector Neural Network Instructions) on Intel architectures. It is parameterized by types `TA`, `TB`, `TC`, and integer constants `BLOCK_M`, `BLOCK_N`, and `BLOCK_K`, which define the data types and block sizes for the matrix multiplication. This struct is part of a larger implementation that leverages advanced matrix extensions and quantized data types to optimize performance on specific hardware.


# Functions

---
### operator\(\)<!-- {{#callable:(anonymous)::operator()}} -->
The `operator()` function is a template function that invokes a given function `f` with an integral constant and additional arguments.
- **Inputs**:
    - `Func`: A callable object (function or functor) that will be invoked.
    - `args`: A variadic template parameter representing additional arguments to be passed to the function `f`.
- **Control Flow**:
    - The function takes a callable `f` and a variadic number of arguments `args`.
    - It calls the function `f` with an `std::integral_constant<int, 0>` and the provided `args`.
- **Output**: The function does not return any value; it simply invokes the callable `f`.


---
### ggml\_tile\_config\_init<!-- {{#callable:(anonymous)::ggml_tile_config_init}} -->
The `ggml_tile_config_init` function initializes a thread-local tile configuration for AMX (Advanced Matrix Extensions) operations, ensuring it is set up only once per thread.
- **Inputs**: None
- **Control Flow**:
    - A static thread-local boolean `is_first_time` is checked to determine if the function has been called before in the current thread.
    - If `is_first_time` is false, the function returns immediately, skipping initialization.
    - A static thread-local `tile_config_t` structure `tc` is declared to hold the tile configuration.
    - The current tile configuration is stored in `current_tc` using `_tile_storeconfig`.
    - The function checks if the configuration needs to be updated by comparing `current_tc` with `tc` using `memcmp`.
    - If the configuration has changed or is uninitialized (`tc.palette_id == 0`), it sets up the tile configuration with specific parameters using `TC_CONFIG_TILE` macros.
    - The new configuration is loaded using `_tile_loadconfig`.
    - Finally, `is_first_time` is set to false to prevent reinitialization in future calls.
- **Output**: The function does not return any value; it modifies thread-local state to initialize tile configuration.


---
### get\_tile\_size<!-- {{#callable:(anonymous)::get_tile_size}} -->
The `get_tile_size` function calculates the size of a tile based on the template type `TB` and various conditions related to the type.
- **Inputs**:
    - `TB`: A template parameter representing the type for which the tile size is being calculated.
- **Control Flow**:
    - Initialize `tile_size` to `TILE_N * sizeof(TB)`.
    - Check if `do_compensate<TB>::value` is true; if so, add `TILE_N * sizeof(int32_t)` to `tile_size`.
    - Check if `TB` is `block_q4_K` or `block_q5_K`; if so, add `TILE_N * 4` to `tile_size`.
    - Check if `TB` is `block_iq4_xs`; if so, add `TILE_N * 2` to `tile_size`.
    - Return the calculated `tile_size`.
- **Output**: An integer representing the calculated size of the tile.


---
### get\_row\_size<!-- {{#callable:(anonymous)::get_row_size}} -->
The `get_row_size` function calculates the size of a row in bytes for a given type and block size, considering specific type-based compensations and adjustments.
- **Inputs**:
    - `TB`: A template parameter representing the data type of the elements in the row.
    - `BLOCK_K`: A template parameter representing the block size used for dividing the total size.
    - `K`: An integer representing the total size or length for which the row size is being calculated.
- **Control Flow**:
    - Calculate the number of blocks (KB) by dividing K by BLOCK_K.
    - Initialize row_size as the product of KB and the size of type TB.
    - Check if the type TB requires compensation using the do_compensate trait; if true, add KB times the size of int32_t to row_size.
    - Check if TB is either block_q4_K or block_q5_K; if true, add KB times 4 to row_size.
    - Check if TB is block_iq4_xs; if true, add KB times 2 to row_size.
    - Return the calculated row_size.
- **Output**: The function returns an integer representing the calculated row size in bytes.


---
### FP16\_TO\_FP32<!-- {{#callable:(anonymous)::FP16_TO_FP32}} -->
The `FP16_TO_FP32` function converts a 16-bit floating-point value to a 32-bit floating-point value using AVX-512 instructions.
- **Inputs**:
    - `val`: A 16-bit floating-point value (`ggml_half`) to be converted to a 32-bit floating-point value.
- **Control Flow**:
    - Initialize a 256-bit integer vector `v` with the input value `val` and fill the rest with zeros using `_mm256_setr_epi16`.
    - Convert the 16-bit floating-point values in `v` to 32-bit floating-point values using `_mm512_cvtph_ps`, storing the result in `o`.
    - Extract the lower 32-bit floating-point value from `o` using `_mm512_cvtss_f32` and return it.
- **Output**: A 32-bit floating-point value (`float`) converted from the input 16-bit floating-point value.


---
### FP16\_TO\_FP32\_VEC<!-- {{#callable:(anonymous)::FP16_TO_FP32_VEC}} -->
The function `FP16_TO_FP32_VEC` converts a 16-bit floating-point value to a 512-bit vector of 32-bit floating-point values using AVX-512 instructions.
- **Inputs**:
    - `val`: A 16-bit floating-point value (`ggml_half`) to be converted.
- **Control Flow**:
    - The function initializes a 256-bit integer vector `v` by setting all 16-bit elements to the input value `val` using `_mm256_set1_epi16`.
    - It then converts the 256-bit integer vector `v` from 16-bit floating-point to 512-bit vector of 32-bit floating-point values using `_mm512_cvtph_ps`.
- **Output**: A 512-bit vector (`__m512`) containing 32-bit floating-point values converted from the input 16-bit floating-point value.


---
### \_mm512\_reduce\_max\_ps<!-- {{#callable:(anonymous)::_mm512_reduce_max_ps}} -->
The function `_mm512_reduce_max_ps` computes the maximum value from a 512-bit vector of single-precision floating-point numbers.
- **Inputs**:
    - `x`: A 512-bit vector (`__m512`) containing 16 single-precision floating-point numbers.
- **Control Flow**:
    - Initialize `v` with the input vector `x`.
    - Shuffle the vector `v` using `_mm512_shuffle_f32x4` with control mask `0x4E` and store the result in `v1`.
    - Compute the element-wise maximum of `v` and `v1` using `_mm512_max_ps` and store the result back in `v`.
    - Shuffle the vector `v` again using `_mm512_shuffle_f32x4` with control mask `0xB1` and store the result in `v1`.
    - Compute the element-wise maximum of `v` and `v1` using `_mm512_max_ps` and store the result back in `v`.
    - Shuffle the vector `v` using `_mm512_shuffle_ps` with control mask `0x4E` and store the result in `v1`.
    - Compute the element-wise maximum of `v` and `v1` using `_mm512_max_ps` and store the result back in `v`.
    - Shuffle the vector `v` using `_mm512_shuffle_ps` with control mask `0xB1` and store the result in `v1`.
    - Compute the element-wise maximum of `v` and `v1` using `_mm512_max_ps` and store the result back in `v`.
    - Convert the first element of the vector `v` to a scalar float using `_mm512_cvtss_f32` and return it.
- **Output**: A single `float` representing the maximum value found in the input vector `x`.


---
### transpose\_8x8\_32bit<!-- {{#callable:(anonymous)::transpose_8x8_32bit}} -->
The `transpose_8x8_32bit` function transposes an 8x8 matrix of 32-bit integers using AVX2 intrinsics.
- **Inputs**:
    - `v`: A pointer to an array of 8 __m256i vectors representing the input 8x8 matrix of 32-bit integers.
    - `v1`: A pointer to an array of 8 __m256i vectors used as temporary storage during the transposition process.
- **Control Flow**:
    - Unpack the lower and higher 32-bit elements of each pair of input vectors and store them in the temporary storage `v1`.
    - Shuffle the 32-bit elements from `v1` to create the first stage of the transposed matrix and store them back in `v`.
    - Shuffle the 128-bit elements from `v` to finalize the transposition and store the result in `v1`.
- **Output**: The function modifies the input arrays `v` and `v1` in place to contain the transposed matrix.


---
### transpose\_16x4\_32bit<!-- {{#callable:(anonymous)::transpose_16x4_32bit}} -->
The `transpose_16x4_32bit` function transposes a 16x4 matrix of 32-bit integers using AVX-512 intrinsics.
- **Inputs**:
    - `r`: A pointer to an array of four __m512i vectors representing the input 16x4 matrix.
    - `d`: A pointer to an array of four __m512i vectors used as temporary storage during the transposition process.
- **Control Flow**:
    - Initialize a constant __m512i index1 with a specific permutation pattern for 32-bit integers.
    - Permute the elements of each __m512i vector in the input array r using the index1 pattern and store the results in the temporary array d.
    - Shuffle the 32-bit integers between pairs of vectors in d to form new vectors in r.
    - Further shuffle the 32-bit integers between pairs of vectors in r to form the final transposed vectors in d.
- **Output**: The function modifies the input arrays r and d in place to contain the transposed matrix.


---
### transpose\_16x16\_32bit<!-- {{#callable:(anonymous)::transpose_16x16_32bit}} -->
The `transpose_16x16_32bit` function transposes a 16x16 matrix of 32-bit integers using AVX-512 intrinsics.
- **Inputs**:
    - `v`: A pointer to an array of 16 __m512i vectors, representing a 16x16 matrix of 32-bit integers.
- **Control Flow**:
    - Initialize an array `v1` of 16 __m512i vectors to store intermediate results.
    - Perform unpacking of 32-bit integers from the input vectors `v` into `v1` using `_mm512_unpacklo_epi32` and `_mm512_unpackhi_epi32`.
    - Unpack 64-bit integers from `v1` back into `v` using `_mm512_unpacklo_epi64` and `_mm512_unpackhi_epi64`.
    - Shuffle 128-bit lanes within `v` using `_mm512_shuffle_i32x4` to further transpose the matrix.
    - Perform a final shuffle of 128-bit lanes within `v1` to complete the transposition.
- **Output**: The function modifies the input array `v` in place to contain the transposed matrix.


---
### quantize\_row\_q8\_K\_vnni<!-- {{#callable:(anonymous)::quantize_row_q8_K_vnni}} -->
The `quantize_row_q8_K_vnni` function quantizes a row of floating-point numbers into a specific quantized format using AVX-512 VNNI instructions.
- **Inputs**:
    - `x`: A pointer to an array of floating-point numbers to be quantized.
    - `vy`: A pointer to the destination where the quantized data will be stored, cast to a `block_q8_K` type.
    - `k`: An integer representing the number of elements in the input array `x`, which must be a multiple of `QK_K`.
- **Control Flow**:
    - Assert that `k` is a multiple of `QK_K` to ensure proper block processing.
    - Calculate the number of blocks `KB` by dividing `k` by `QK_K`.
    - Initialize arrays to hold 16 float vectors, quantized vectors, and packed quantized vectors.
    - Iterate over each block `i` from 0 to `KB-1`.
    - For each block, compute the maximum absolute value of the elements in the block.
    - Calculate the inverse scale factor `iscale` based on the maximum absolute value and store the quantization factor in the destination block.
    - Scale and round the floating-point numbers to the nearest integer using the calculated scale factor.
    - Pack the quantized integers into 8-bit vectors and store them in the destination block.
    - Transpose the packed vectors and compute the byte sums using VNNI instructions.
    - Store the computed byte sums in the destination block.
- **Output**: The function does not return a value; it modifies the data at the location pointed to by `vy` to store the quantized results.
- **Functions called**:
    - [`(anonymous)::_mm512_reduce_max_ps`](#(anonymous)::_mm512_reduce_max_ps)
    - [`(anonymous)::transpose_16x4_32bit`](#(anonymous)::transpose_16x4_32bit)


---
### from\_float<block\_q8\_0><!-- {{#callable:(anonymous)::from_float<block_q8_0>}} -->
The function `from_float<block_q8_0>` converts an array of floating-point numbers to a quantized format using the `block_q8_0` type.
- **Inputs**:
    - `x`: A pointer to an array of floating-point numbers to be converted.
    - `vy`: A pointer to a memory location where the quantized data will be stored, cast to a `block_q8_0` type.
    - `k`: An integer representing the number of elements in the array `x` to be converted.
- **Control Flow**:
    - The function calls [`quantize_row_q8_0`](../ggml-cpu-quants.c.driver.md#quantize_row_q8_0) with the provided inputs `x`, `vy` cast to `block_q8_0`, and `k`.
- **Output**: The function does not return a value; it modifies the memory pointed to by `vy` to store the quantized data.
- **Functions called**:
    - [`quantize_row_q8_0`](../ggml-cpu-quants.c.driver.md#quantize_row_q8_0)


---
### from\_float<block\_q8\_1><!-- {{#callable:(anonymous)::from_float<block_q8_1>}} -->
The function `from_float<block_q8_1>` converts an array of floats to a quantized format using the `block_q8_1` type.
- **Inputs**:
    - `x`: A pointer to an array of floats that need to be quantized.
    - `vy`: A pointer to a memory location where the quantized data will be stored, cast to a `block_q8_1` type.
    - `k`: An integer representing the number of elements in the array `x` to be quantized.
- **Control Flow**:
    - The function calls [`quantize_row_q8_1`](../ggml-cpu-quants.c.driver.md#quantize_row_q8_1) with the provided inputs `x`, `vy`, and `k`.
- **Output**: The function does not return a value; it modifies the memory pointed to by `vy` to store the quantized data.
- **Functions called**:
    - [`quantize_row_q8_1`](../ggml-cpu-quants.c.driver.md#quantize_row_q8_1)


---
### from\_float<block\_q8\_K><!-- {{#callable:(anonymous)::from_float<block_q8_K>}} -->
The `from_float<block_q8_K>` function converts a block of floating-point numbers into a quantized format using a reference implementation.
- **Inputs**:
    - `x`: A pointer to an array of floating-point numbers to be quantized.
    - `vy`: A pointer to a memory location where the quantized data will be stored.
    - `k`: The number of elements in the array `x` to be quantized.
- **Control Flow**:
    - The function checks a preprocessor condition to determine which quantization implementation to use.
    - If the condition is true, it calls [`quantize_row_q8_K_ref`](../../ggml-quants.c.driver.md#quantize_row_q8_K_ref) to perform the quantization using a reference implementation.
    - If the condition is false, it would call [`quantize_row_q8_K_vnni`](#(anonymous)::quantize_row_q8_K_vnni) (though this path is currently not used).
- **Output**: The function does not return a value; it modifies the memory pointed to by `vy` to store the quantized data.
- **Functions called**:
    - [`quantize_row_q8_K_ref`](../../ggml-quants.c.driver.md#quantize_row_q8_K_ref)
    - [`(anonymous)::quantize_row_q8_K_vnni`](#(anonymous)::quantize_row_q8_K_vnni)


---
### unpack\_A<!-- {{#callable:(anonymous)::unpack_A}} -->
The `unpack_A` function loads data from a source array `A` into a destination array `tile` using AVX2 intrinsics for efficient memory operations.
- **Inputs**:
    - `tile`: A pointer to an int8_t array where the unpacked data will be stored.
    - `A`: A pointer to a block_q8_K array from which data will be loaded.
    - `lda`: An integer representing the leading dimension of the array `A`.
    - `k`: An integer representing the offset in the `qs` array within each block of `A`.
    - `nr`: An integer representing the number of rows to process, which must be less than or equal to TILE_M.
- **Control Flow**:
    - Assert that `nr` is less than or equal to TILE_M to ensure valid processing range.
    - Iterate over each row `m` from 0 to `nr-1`.
    - For each row, load a 256-bit vector from the `qs` array of the `m`-th block of `A`, offset by `k * 32`.
    - Store the loaded vector into the `tile` array at the position corresponding to the current row `m`.
- **Output**: The function does not return a value; it modifies the `tile` array in place.


---
### unpack\_A<block\_q6\_K><!-- {{#callable:(anonymous)::unpack_A<block_q6_K>}} -->
The function `unpack_A<block_q6_K>` unpacks and zero-pads a block of quantized data from a source array into a destination tile array for further processing.
- **Inputs**:
    - `tile`: A pointer to the destination array where the unpacked and zero-padded data will be stored.
    - `A`: A pointer to the source array of type `block_q8_K` containing the quantized data to be unpacked.
    - `lda`: The leading dimension of the source array `A`, representing the number of elements between successive rows.
    - `k`: An integer representing the starting index in the source array `A` from which data will be unpacked.
    - `nr`: An integer representing the number of rows to be processed, which must be less than or equal to `TILE_M`.
- **Control Flow**:
    - Assert that `nr` is less than or equal to `TILE_M` to ensure valid processing range.
    - Initialize a zero vector `zero` of type `__m128i` for padding purposes.
    - Iterate over each row `m` from 0 to `nr-1`.
    - For each row, load a 128-bit vector `v` from the source array `A` starting at the specified index `k` and row `m`.
    - Insert the zero vector into the upper half of a 256-bit vector `r` to achieve zero-padding from 16 to 32 bits.
    - Store the resulting 256-bit vector `r` into the destination tile array at the corresponding position.
- **Output**: The function does not return a value; it modifies the `tile` array in place by storing the unpacked and zero-padded data.


---
### bytes\_from\_nibbles\_32<!-- {{#callable:(anonymous)::bytes_from_nibbles_32}} -->
The `bytes_from_nibbles_32` function converts a 16-byte array of nibbles into a 32-byte array of bytes, extracting the lower 4 bits from each nibble.
- **Inputs**:
    - `rsi`: A pointer to a 16-byte array of `uint8_t` values, where each byte contains two nibbles.
- **Control Flow**:
    - Load 16 bytes from the input pointer `rsi` into a 128-bit SIMD register `tmp`.
    - Shift the `tmp` register right by 4 bits to separate the high nibbles and combine it with the original `tmp` to form a 256-bit SIMD register `bytes`.
    - Create a 256-bit SIMD register `lowMask` with all bytes set to 0xF to mask out the high bits.
    - Perform a bitwise AND operation between `bytes` and `lowMask` to extract the lower 4 bits of each byte, effectively converting nibbles to bytes.
- **Output**: A 256-bit SIMD register (`__m256i`) containing the converted bytes, where each byte represents a nibble from the input.


---
### bytes\_from\_nibbles\_64<!-- {{#callable:(anonymous)::bytes_from_nibbles_64}} -->
The function `bytes_from_nibbles_64` combines two sets of 4-bit nibbles from two input arrays into a single 64-byte vector using AVX-512 instructions.
- **Inputs**:
    - `qs`: A pointer to an array of 8-bit unsigned integers representing the lower 4 bits of the nibbles.
    - `qh`: A pointer to an array of 8-bit unsigned integers representing the higher 4 bits of the nibbles.
    - `k`: An integer used to shift the higher bits to their correct position.
- **Control Flow**:
    - Initialize a 256-bit mask `lowMask` to isolate the lower 4 bits of each byte.
    - Initialize a 256-bit mask `hmask` to isolate the higher bits, shifted by `k`.
    - Load 256 bits from `qs` and `qh` into `q5bits` and `hbits` respectively.
    - Extract the lower 4 bits from `q5bits` and store in `q5l_0`.
    - Extract and shift the higher bits from `hbits`, combine with `q5l_0` to form `q5_0`.
    - Shift `hmask` by 1 bit to prepare for the next higher bit extraction.
    - Extract the next set of lower 4 bits from `q5bits`, shift and combine with the next set of higher bits from `hbits` to form `q5_1`.
    - Combine `q5_0` and `q5_1` into a 512-bit vector and return.
- **Output**: A 512-bit integer vector (`__m512i`) containing the combined bytes from the nibbles.


---
### bytes\_from\_nibbles\_128<!-- {{#callable:(anonymous)::bytes_from_nibbles_128}} -->
The `bytes_from_nibbles_128` function converts two sets of 64 nibbles from two input byte arrays into two 512-bit vectors of bytes, using AVX-512 and AVX-256 intrinsics.
- **Inputs**:
    - `r0`: A reference to an __m512i variable where the first 512-bit vector of bytes will be stored.
    - `r1`: A reference to an __m512i variable where the second 512-bit vector of bytes will be stored.
    - `qs`: A pointer to a uint8_t array containing the first set of 64 nibbles (32 bytes).
    - `qh`: A pointer to a uint8_t array containing the second set of 64 nibbles (32 bytes).
- **Control Flow**:
    - Initialize two 256-bit masks, m4 and m2, to isolate lower 4 bits and 2 bits respectively.
    - Load 256 bits from the qs array into q6bits1 and q6bits2, and from qh into q6bitsH.
    - Extract and shift the high nibbles from q6bitsH into q6h_0, q6h_1, q6h_2, and q6h_3 using bitwise AND and shift operations.
    - Combine the low nibbles from q6bits1 and q6bits2 with the shifted high nibbles to form q6_0, q6_1, q6_2, and q6_3.
    - Insert q6_0 and q6_1 into r0, and q6_2 and q6_3 into r1, forming two 512-bit vectors.
- **Output**: The function outputs two 512-bit vectors, r0 and r1, each containing 64 bytes derived from the input nibbles.


---
### packNibbles<!-- {{#callable:(anonymous)::packNibbles}} -->
The `packNibbles` function combines two 512-bit integer vectors by bitwise OR-ing the first vector with the second vector left-shifted by 4 bits.
- **Inputs**:
    - `r0`: A 512-bit integer vector (__m512i) representing the first input.
    - `r1`: A 512-bit integer vector (__m512i) representing the second input.
- **Control Flow**:
    - The function takes two 512-bit integer vectors, r0 and r1, as input.
    - It left-shifts each 16-bit element of r1 by 4 bits using the _mm512_slli_epi16 intrinsic.
    - It performs a bitwise OR operation between r0 and the shifted r1 using the _mm512_or_si512 intrinsic.
    - The result of the OR operation is returned as the output.
- **Output**: A 512-bit integer vector (__m512i) that is the result of the bitwise OR operation between r0 and the left-shifted r1.


---
### pack\_qs<!-- {{#callable:(anonymous)::pack_qs}} -->
The `pack_qs` function packs quantized data from an input array into a packed format suitable for vectorized processing using AVX512 instructions.
- **Inputs**:
    - `packed_B`: A pointer to the destination buffer where the packed data will be stored.
    - `B`: A pointer to the source array containing the quantized data to be packed.
    - `KB`: An integer representing the stride or step size in the source array `B`.
- **Control Flow**:
    - Initialize a temporary buffer `tmp` to store intermediate results.
    - Load 8 blocks of quantized data from `B` into `v` using [`bytes_from_nibbles_32`](#(anonymous)::bytes_from_nibbles_32).
    - Transpose the 8x8 block of 32-bit integers in `v` into `v2` using [`transpose_8x8_32bit`](#(anonymous)::transpose_8x8_32bit).
    - Store the transposed data from `v2` into the `tmp` buffer.
    - Repeat the loading, transposing, and storing process for the next 8 blocks of data from `B`.
    - Pack the data from `tmp` into 512-bit vectors using [`packNibbles`](#(anonymous)::packNibbles) and store them in `packed_B`.
- **Output**: The function does not return a value; it modifies the `packed_B` buffer in place to contain the packed data.
- **Functions called**:
    - [`(anonymous)::bytes_from_nibbles_32`](#(anonymous)::bytes_from_nibbles_32)
    - [`(anonymous)::transpose_8x8_32bit`](#(anonymous)::transpose_8x8_32bit)
    - [`(anonymous)::packNibbles`](#(anonymous)::packNibbles)


---
### pack\_qs<block\_q8\_0><!-- {{#callable:(anonymous)::pack_qs<block_q8_0>}} -->
The function `pack_qs<block_q8_0>` transposes and packs 8x8 blocks of 32-bit integers from a source array into a destination array using AVX2 intrinsics.
- **Inputs**:
    - `packed_B`: A pointer to the destination buffer where the packed data will be stored.
    - `B`: A pointer to the source array of `block_q8_0` structures containing the data to be packed.
    - `KB`: An integer representing the stride or step size in the source array `B`.
- **Control Flow**:
    - Initialize two arrays of __m256i vectors, `v` and `v2`, each with 8 elements.
    - Load 8 blocks of 32-bit integers from the source array `B` into the `v` array using `_mm256_loadu_si256`.
    - Call [`transpose_8x8_32bit`](#(anonymous)::transpose_8x8_32bit) to transpose the 8x8 block of 32-bit integers from `v` to `v2`.
    - Store the transposed data from `v2` into the destination buffer `packed_B` using `_mm256_storeu_si256`.
    - Repeat the process for the next set of 8 blocks from the source array `B`, offset by 8 * `KB`, and store the transposed data into `packed_B` with an additional offset of 32 bytes.
- **Output**: The function does not return a value; it modifies the `packed_B` buffer in place.
- **Functions called**:
    - [`(anonymous)::transpose_8x8_32bit`](#(anonymous)::transpose_8x8_32bit)


---
### pack\_qs<block\_q4\_K><!-- {{#callable:(anonymous)::pack_qs<block_q4_K>}} -->
The function `pack_qs<block_q4_K>` packs quantized data from a `block_q4_K` structure into a packed format suitable for vectorized processing.
- **Inputs**:
    - `packed_B`: A pointer to the destination buffer where the packed data will be stored.
    - `B`: A pointer to the source array of `block_q4_K` structures containing the quantized data to be packed.
    - `KB`: An integer representing the number of blocks in the source array `B`.
- **Control Flow**:
    - Initialize an array of 16 __m512i vectors to hold intermediate data.
    - Cast the destination buffer pointer `packed_B` to a character pointer `pb`.
    - Iterate over the blocks in `B` in chunks of 64 (since QK_K is 256 and we handle 2 groups at a time).
    - For each block, convert 64 nibbles from the source `B` into bytes using [`bytes_from_nibbles_64`](#(anonymous)::bytes_from_nibbles_64) and store them in the `v` array.
    - Transpose the 16x16 matrix of 32-bit integers in `v` to prepare for packing.
    - Pack the transposed data into 128-bit chunks using [`packNibbles`](#(anonymous)::packNibbles) and store them in the destination buffer `pb`, advancing the pointer by 64 bytes after each store.
- **Output**: The function does not return a value; it modifies the buffer pointed to by `packed_B` to contain the packed data.
- **Functions called**:
    - [`(anonymous)::bytes_from_nibbles_64`](#(anonymous)::bytes_from_nibbles_64)
    - [`(anonymous)::transpose_16x16_32bit`](#(anonymous)::transpose_16x16_32bit)
    - [`(anonymous)::packNibbles`](#(anonymous)::packNibbles)


---
### pack\_qs<block\_q5\_K><!-- {{#callable:(anonymous)::pack_qs<block_q5_K>}} -->
The function `pack_qs<block_q5_K>` packs quantized data from a `block_q5_K` structure into a packed format suitable for efficient processing using AVX-512 instructions.
- **Inputs**:
    - `packed_B`: A pointer to the destination buffer where the packed data will be stored.
    - `B`: A pointer to the source array of `block_q5_K` structures containing the quantized data to be packed.
    - `KB`: An integer representing the stride or number of `block_q5_K` elements in the source array `B`.
- **Control Flow**:
    - Initialize an array of 16 __m512i vectors and a lowMask for extracting lower 4 bits.
    - Calculate pointers for storing packed lower 4 bits and higher 1 bit data.
    - Iterate over groups of 64 elements in the source data, processing two groups at a time.
    - For each group, convert nibbles to bytes using [`bytes_from_nibbles_64`](#(anonymous)::bytes_from_nibbles_64) and store in the vector array `v`.
    - Transpose the 16x16 matrix of 32-bit integers in `v` to prepare for packing.
    - Pack the lower 4 bits of two groups into the destination buffer using [`packNibbles`](#(anonymous)::packNibbles) and store them.
    - Pack the higher 1 bit of two groups by shifting and adding bits from the vector array `v`, then store them in the destination buffer.
- **Output**: The function does not return a value; it modifies the `packed_B` buffer in place to contain the packed quantized data.
- **Functions called**:
    - [`(anonymous)::bytes_from_nibbles_64`](#(anonymous)::bytes_from_nibbles_64)
    - [`(anonymous)::transpose_16x16_32bit`](#(anonymous)::transpose_16x16_32bit)
    - [`(anonymous)::packNibbles`](#(anonymous)::packNibbles)


---
### pack\_qs<block\_q6\_K><!-- {{#callable:(anonymous)::pack_qs<block_q6_K>}} -->
The function `pack_qs<block_q6_K>` packs quantized data from a `block_q6_K` structure into a packed format suitable for efficient processing using AVX-512 instructions.
- **Inputs**:
    - `packed_B`: A pointer to the destination buffer where the packed data will be stored.
    - `B`: A pointer to the source array of `block_q6_K` structures containing the quantized data to be packed.
    - `KB`: An integer representing the stride or number of blocks in the source array `B`.
- **Control Flow**:
    - Initialize an array of 32 __m512i vectors and a lowMask for bit manipulation.
    - Calculate pointers for the packed data buffer `pb` and `ph` for storing lower and higher bits respectively.
    - Iterate over the blocks in `B` in chunks of 128 (QK_K / 128).
    - For each block, iterate over the number of tiles `TILE_N`.
    - Call [`bytes_from_nibbles_128`](#(anonymous)::bytes_from_nibbles_128) to convert nibbles to bytes and store them in the `v` array.
    - Transpose the `v` array to rearrange the data for efficient packing.
    - Pack the lower 4 bits of the data using [`packNibbles`](#(anonymous)::packNibbles) and store them in `pb`.
    - Pack the higher 2 bits of the data using bit manipulation and store them in `ph`.
- **Output**: The function does not return a value; it modifies the `packed_B` buffer in place to contain the packed quantized data.
- **Functions called**:
    - [`(anonymous)::bytes_from_nibbles_128`](#(anonymous)::bytes_from_nibbles_128)
    - [`(anonymous)::transpose_16x16_32bit`](#(anonymous)::transpose_16x16_32bit)
    - [`(anonymous)::packNibbles`](#(anonymous)::packNibbles)


---
### pack\_qs<block\_iq4\_xs><!-- {{#callable:(anonymous)::pack_qs<block_iq4_xs>}} -->
The function `pack_qs<block_iq4_xs>` packs quantized data from a source array into a packed format using AVX-512 instructions for efficient processing.
- **Inputs**:
    - `packed_B`: A pointer to the destination buffer where the packed data will be stored.
    - `B`: A pointer to the source array of type `block_iq4_xs` containing the quantized data to be packed.
    - `KB`: An integer representing the stride or number of blocks in the source array `B`.
- **Control Flow**:
    - Initialize an array `v` of 16 `__m512i` vectors to hold intermediate data.
    - Cast the `packed_B` pointer to a `char` pointer `pb` for byte-wise operations.
    - Iterate over `k` from 0 to `QK_K / 64`, processing each group of 64 elements.
    - For each `k`, iterate over `n` from 0 to `TILE_N`, processing each tile.
    - Load 32 nibbles from `B` into two `__m256i` vectors `r0` and `r1` using [`bytes_from_nibbles_32`](#(anonymous)::bytes_from_nibbles_32).
    - Combine `r0` and `r1` into a single `__m512i` vector and store it in `v[n]`.
    - Transpose the 16x16 matrix of 32-bit integers in `v` using [`transpose_16x16_32bit`](#(anonymous)::transpose_16x16_32bit).
    - Iterate over `n` in steps of 2, packing pairs of vectors from `v` into nibbles using [`packNibbles`](#(anonymous)::packNibbles).
    - Store the packed nibbles into the `packed_B` buffer and advance the pointer `pb` by 64 bytes.
- **Output**: The function does not return a value; it modifies the `packed_B` buffer in place to store the packed data.
- **Functions called**:
    - [`(anonymous)::bytes_from_nibbles_32`](#(anonymous)::bytes_from_nibbles_32)
    - [`(anonymous)::transpose_16x16_32bit`](#(anonymous)::transpose_16x16_32bit)
    - [`(anonymous)::packNibbles`](#(anonymous)::packNibbles)


---
### pack\_B<!-- {{#callable:(anonymous)::pack_B}} -->
The `pack_B` function packs a block of data `B` into a packed format `packed_B` using quantization and scale adjustments for efficient matrix multiplication.
- **Inputs**:
    - `packed_B`: A pointer to the memory location where the packed data will be stored.
    - `B`: A pointer to the block of data of type `block_iq4_xs` that needs to be packed.
    - `KB`: An integer representing the number of blocks in the data `B`.
- **Control Flow**:
    - The function begins by calling [`pack_qs`](#(anonymous)::pack_qs) to pack the quantized data from `B` into `packed_B`.
    - It calculates the starting address for scales and `d` within `packed_B` using pointer arithmetic.
    - A loop iterates over each block in `B` (up to `TILE_N`), extracting and packing scale values.
    - For each block, it extracts the high and low scale values, adjusts them, and stores them in the `scales` array.
    - The `d` value for each block is directly copied from `B` to the corresponding position in `packed_B`.
- **Output**: The function does not return a value; it modifies the `packed_B` in place to contain the packed and quantized data.
- **Functions called**:
    - [`(anonymous)::pack_qs`](#(anonymous)::pack_qs)


---
### s8s8\_compensation<!-- {{#callable:(anonymous)::s8s8_compensation}} -->
The `s8s8_compensation` function calculates and stores the compensation values for packed 8-bit signed integer data in a specific memory layout.
- **Inputs**:
    - `packed_B`: A pointer to the packed data buffer, which contains quantized data and will be modified to include compensation values.
- **Control Flow**:
    - Initialize an offset to locate the compensation section in the packed data layout.
    - Set a zero-initialized 512-bit integer vector `vcomp` for accumulating compensation values.
    - Set a 512-bit integer vector `off` with all elements set to 0x80 to adjust the signed 8-bit values to unsigned.
    - Iterate over 8 blocks of 64 bytes each from the packed data, loading them into a 512-bit integer vector `vb`.
    - For each block, update `vcomp` by performing a dot product of `off` and `vb`, accumulating the results.
    - Store the final compensation values from `vcomp` into the compensation section of the packed data.
- **Output**: The function does not return a value but modifies the `packed_B` buffer to include the calculated compensation values.


---
### unpack\_mins\_and\_scales<!-- {{#callable:(anonymous)::unpack_mins_and_scales}} -->
The `unpack_mins_and_scales` function converts 8 pairs of 6-bit minimum and scale values from a byte array into 8-bit values stored in a 32-bit integer array.
- **Inputs**:
    - `scales`: A pointer to a byte array containing 6-bit packed minimum and scale values.
    - `utmp`: A pointer to a 32-bit integer array where the unpacked 8-bit values will be stored.
- **Control Flow**:
    - Define three constant masks: kmask1, kmask2, and kmask3 for bit manipulation.
    - Copy the first 12 bytes from the scales array to the utmp array using memcpy.
    - Calculate utmp[3] by shifting and masking utmp[2] and utmp[1], then combining the results.
    - Store the masked value of utmp[1] in a temporary variable uaux.
    - Calculate utmp[1] by shifting and masking utmp[0] and utmp[2], then combining the results.
    - Assign the value of uaux to utmp[2].
    - Mask utmp[0] with kmask1 to finalize its value.
- **Output**: The function does not return a value; it modifies the utmp array in place to store the unpacked 8-bit values.


---
### unpack\_B<block\_q4\_0><!-- {{#callable:(anonymous)::unpack_B<block_q4_0>}} -->
The function `unpack_B<block_q4_0>` unpacks packed 4-bit quantized data into an 8-bit format, adjusting the values by a constant offset.
- **Inputs**:
    - `tile`: A pointer to an int8_t array where the unpacked data will be stored.
    - `packed_B`: A pointer to the packed data that needs to be unpacked.
- **Control Flow**:
    - Initialize two constant __m512i vectors, `off` and `lowMask`, for offset and masking operations respectively.
    - Iterate over the packed data in steps of 2, processing 8 blocks in total.
    - For each block, load 512 bits of data from `packed_B` into a __m512i vector `bytes`.
    - Extract the lower 4 bits of each byte using `lowMask` and subtract `off` to get `r0`.
    - Shift the `bytes` right by 4 bits, mask again with `lowMask`, and subtract `off` to get `r1`.
    - Store the results `r0` and `r1` into the `tile` array at calculated positions.
- **Output**: The function does not return a value; it modifies the `tile` array in place with the unpacked data.


---
### unpack\_B<block\_q4\_1><!-- {{#callable:(anonymous)::unpack_B<block_q4_1>}} -->
The function `unpack_B<block_q4_1>` unpacks packed 4-bit quantized data into a tile of 8-bit data using AVX-512 instructions.
- **Inputs**:
    - `tile`: A pointer to a uint8_t array where the unpacked data will be stored.
    - `packed_B`: A pointer to the packed data that needs to be unpacked.
- **Control Flow**:
    - Initialize a constant `lowMask` with a value of 0xF to mask the lower 4 bits of each byte.
    - Iterate over the packed data in steps of 2, processing 8 bytes at a time.
    - For each iteration, load 64 bytes of packed data into a 512-bit register using `_mm512_loadu_si512`.
    - Mask the lower 4 bits of the loaded data to get the first set of unpacked bytes using `_mm512_and_si512`.
    - Shift the loaded data right by 4 bits and mask again to get the second set of unpacked bytes.
    - Store the first set of unpacked bytes into the `tile` array at the current position.
    - Store the second set of unpacked bytes into the `tile` array at an offset of 64 bytes from the current position.
- **Output**: The function does not return a value; it modifies the `tile` array in place to contain the unpacked data.


---
### unpack\_B<!-- {{#callable:(anonymous)::unpack_B}} -->
The `unpack_B` function unpacks packed data from a source buffer into a destination tile buffer, processing it in groups and applying bitwise operations to extract and store the data.
- **Inputs**:
    - `tile`: A pointer to an int8_t array where the unpacked data will be stored.
    - `packed_B`: A pointer to the source buffer containing the packed data.
    - `k`: An integer representing the index of the group to be unpacked.
- **Control Flow**:
    - Calculate the size of each packed group using the formula `QK_K / 2 * TILE_N / 8`.
    - Determine the starting position of the packed group in the source buffer using the index `k`.
    - Initialize a mask to extract the lower 4 bits of each byte.
    - Iterate over the packed data in steps of 2, loading 512-bit chunks from the source buffer.
    - For each chunk, extract the lower and upper 4 bits of each byte using bitwise operations.
    - Store the extracted data into the destination tile buffer in two separate 512-bit chunks.
- **Output**: The function does not return a value; it modifies the `tile` buffer in place.


---
### unpack\_B<block\_q5\_K><!-- {{#callable:(anonymous)::unpack_B<block_q5_K>}} -->
The function `unpack_B<block_q5_K>` unpacks packed quantized data from a specific format into a tile of int8_t values, using AVX-512 intrinsics for vectorized operations.
- **Inputs**:
    - `tile`: A pointer to an int8_t array where the unpacked data will be stored.
    - `packed_B`: A pointer to the packed data that needs to be unpacked.
    - `k`: An integer index used to calculate the offset for accessing the packed data.
- **Control Flow**:
    - Calculate the size of the lower 4 bits group and set the pointer `pb` to the start of this group in `packed_B` using the index `k`.
    - Calculate the size of the higher 1 bit group and set the pointer `ph` to the start of this group in `packed_B` using the index `k`.
    - Load the higher bits from `ph` into a 512-bit register `hbits`.
    - Initialize masks for extracting lower and higher bits.
    - Iterate over 8 elements in steps of 2, performing the following operations:
    - Load 512 bits of data from `pb` into `bytes`.
    - Extract the lower 4 bits and higher 1 bit from `bytes` and `hbits`, respectively, using bitwise operations and shifts.
    - Combine the lower and higher bits to form the unpacked values `r0` and `r1`.
    - Store the unpacked values `r0` and `r1` into the `tile` array at calculated positions.
- **Output**: The function does not return a value; it modifies the `tile` array in place with the unpacked data.


---
### unpack\_B<block\_q6\_K><!-- {{#callable:(anonymous)::unpack_B<block_q6_K>}} -->
The function `unpack_B<block_q6_K>` unpacks a packed block of data into a tile of int8_t values, using specific bit manipulation and vectorized operations.
- **Inputs**:
    - `tile`: A pointer to an int8_t array where the unpacked data will be stored.
    - `packed_B`: A pointer to the packed data that needs to be unpacked.
    - `k`: An integer index used to calculate the offset within the packed data.
- **Control Flow**:
    - Calculate the size of the lower 4 bits group and set the pointer `pb` to the start of the relevant section in `packed_B` using `k` as an offset.
    - Calculate the size of the higher 2 bits group and set the pointer `ph` to the start of the relevant section in `packed_B` using `k` as an offset.
    - Load the higher bits from `ph` into a 512-bit vector `hbits`.
    - Set up constant vectors for offset and masks used in bit manipulation.
    - Load the lower bits from `pb` into a 512-bit vector `bytes`.
    - Extract and manipulate the lower and higher bits from `bytes` and `hbits` to form two result vectors `r0` and `r1`.
    - Store the results into the `tile` array after adjusting with the offset vector.
    - Repeat the process for the next set of 64 bytes in `pb` and store the results in the next section of the `tile` array.
- **Output**: The function does not return a value; it modifies the `tile` array in place with the unpacked data.


---
### unpack\_B<block\_iq4\_xs><!-- {{#callable:(anonymous)::unpack_B<block_iq4_xs>}} -->
The function `unpack_B<block_iq4_xs>` unpacks packed data from a specific format into a tile of int8_t values using AVX-512 intrinsics.
- **Inputs**:
    - `tile`: A pointer to an int8_t array where the unpacked data will be stored.
    - `packed_B`: A pointer to the packed data that needs to be unpacked.
    - `k`: An integer representing the index of the group of packed data to be processed.
- **Control Flow**:
    - Initialize a constant __m512i vector `values128` with specific values for shuffling.
    - Calculate the size of each packed group using `QK_K / 2 * TILE_N / 8`.
    - Set a pointer `pb` to the start of the packed data group using the index `k`.
    - Define a mask `lowMask` to extract the lower 4 bits of each byte.
    - Iterate over `n` from 0 to 8 in steps of 2 to process each pair of 32-byte segments.
    - Load 32 bytes of packed data into a __m512i vector `bytes`.
    - Shuffle and unpack the lower 4 bits of `bytes` using `values128` and store the result in `r0`.
    - Shuffle and unpack the higher 4 bits of `bytes` using `values128` and store the result in `r1`.
    - Store the unpacked results `r0` and `r1` into the `tile` array at calculated offsets.
- **Output**: The function does not return a value; it modifies the `tile` array in place with the unpacked data.


---
### apply<!-- {{#callable:(anonymous)::apply}} -->
The `apply` function performs a matrix multiplication operation using AVX-512 and VNNI instructions, processing quantized data from matrices A and B, and storing the result in matrix C.
- **Inputs**:
    - `KB`: An integer representing the number of blocks in the K dimension.
    - `_A`: A pointer to the first input matrix A, which is quantized and packed.
    - `_B`: A pointer to the second input matrix B, which is quantized and packed.
    - `C`: A pointer to the output matrix C, where the result of the matrix multiplication will be stored.
    - `ldc`: An integer representing the leading dimension of matrix C, used for accessing elements in a row-major order.
- **Control Flow**:
    - Initialize constants and variables for processing, including AVX-512 vectors for loading and storing data.
    - Load and unpack data from matrix A into AVX-512 vectors.
    - Initialize the result vectors for matrix C to zero.
    - Iterate over the columns of matrix B, loading and processing data in blocks using AVX-512 instructions.
    - Perform dot product operations between the unpacked data from A and B, accumulating results in the C vectors.
    - Apply compensation and scaling factors to the accumulated results to adjust for quantization effects.
    - Store the final computed results back into the output matrix C.
- **Output**: The function does not return a value; it modifies the output matrix C in place with the results of the matrix multiplication.


---
### get\_quants\_size<block\_q4\_K><!-- {{#callable:(anonymous)::get_quants_size<block_q4_K>}} -->
The function `get_quants_size<block_q4_K>` calculates the size of quantized data for a specific block type `block_q4_K` based on predefined constants.
- **Inputs**: None
- **Control Flow**:
    - The function is a template specialization for `block_q4_K`.
    - It calculates the size by dividing `QK_K` by 2 and multiplying the result by `TILE_N`.
    - The function returns the calculated size as a constant integer.
- **Output**: The function returns an integer representing the size of quantized data for `block_q4_K`.


---
### get\_quants\_size<block\_q5\_K><!-- {{#callable:(anonymous)::get_quants_size<block_q5_K>}} -->
The function `get_quants_size<block_q5_K>` calculates the size of quantized data for a specific block type `block_q5_K` based on predefined constants.
- **Inputs**: None
- **Control Flow**:
    - The function is a template specialization for `block_q5_K`.
    - It calculates the size using the formula `(QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N`.
    - The function returns the calculated size as a constant integer.
- **Output**: The function returns an integer representing the size of quantized data for `block_q5_K`.


---
### get\_quants\_size<block\_q6\_K><!-- {{#callable:(anonymous)::get_quants_size<block_q6_K>}} -->
The function `get_quants_size<block_q6_K>` calculates the size of quantized data for the `block_q6_K` type based on predefined constants.
- **Inputs**: None
- **Control Flow**:
    - The function is a template specialization for `block_q6_K`.
    - It calculates the size using the formula `(QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N`.
    - The function returns the calculated size as an integer.
- **Output**: An integer representing the size of quantized data for `block_q6_K`.


---
### get\_quants\_size<block\_iq4\_xs><!-- {{#callable:(anonymous)::get_quants_size<block_iq4_xs>}} -->
The function `get_quants_size<block_iq4_xs>` returns the size of quantized data for the `block_iq4_xs` type, calculated as `(QK_K / 2) * TILE_N`.
- **Inputs**: None
- **Control Flow**:
    - The function is a template specialization for `block_iq4_xs`.
    - It calculates the size of quantized data using the formula `(QK_K / 2) * TILE_N`.
    - The function returns the calculated size as a constant integer.
- **Output**: An integer representing the size of quantized data for `block_iq4_xs`.


---
### scale\_C<!-- {{#callable:(anonymous)::scale_C}} -->
The `scale_C` function scales and accumulates integer matrix tiles using quantization scales from packed data.
- **Inputs**:
    - `tile`: A pointer to an array of int32_t representing the input matrix tile to be scaled.
    - `sumi`: A pointer to an array of int32_t where the scaled and accumulated results will be stored.
    - `packed_B`: A pointer to the packed data containing quantization scales.
    - `k`: An integer representing the index for accessing the correct quantization scale.
    - `nr`: An integer representing the number of rows to process in the tile.
- **Control Flow**:
    - Retrieve the quantization scales from the packed data using the index `k` and convert them to 32-bit integers.
    - Iterate over each row `m` in the tile up to `nr`.
    - For each row, load the current accumulated sum from `sumi` if `is_acc` is true, otherwise initialize it to zero.
    - Load the corresponding row from the input tile.
    - Multiply the tile row by the quantization scale and add the result to the accumulated sum.
    - Store the updated accumulated sum back into `sumi`.
- **Output**: The function does not return a value; it modifies the `sumi` array in place with the scaled and accumulated results.


---
### convert\_B\_packed\_format<!-- {{#callable:(anonymous)::convert_B_packed_format}} -->
The `convert_B_packed_format` function converts a matrix B into a packed format suitable for efficient computation using AMX (Advanced Matrix Extensions) by organizing it into tiles.
- **Inputs**:
    - `packed_B`: A pointer to the memory location where the packed format of matrix B will be stored.
    - `B`: A pointer to the original matrix B that needs to be converted into a packed format.
    - `N`: The number of columns in matrix B.
    - `K`: The number of rows in matrix B.
- **Control Flow**:
    - Calculate the number of tiles in the N and K dimensions, NB and KB, respectively, using TILE_N and BLOCK_K.
    - Determine the tile size using the `get_tile_size` function, which depends on the type TB.
    - Use a parallel loop over the number of tiles in the N dimension (NB) to distribute the work across multiple threads.
    - For each tile in the N dimension, iterate over the tiles in the K dimension (KB).
    - For each tile, calculate the starting index in the N dimension and call [`pack_B`](#(anonymous)::pack_B) to pack the corresponding block of matrix B into the packed format.
- **Output**: The function does not return a value; it modifies the `packed_B` memory to store the packed format of matrix B.
- **Functions called**:
    - [`(anonymous)::pack_B`](#(anonymous)::pack_B)


---
### tinygemm\_kernel\_amx<!-- {{#callable:(anonymous)::tinygemm_kernel_amx}} -->
The `tinygemm_kernel_amx` function performs a matrix multiplication using advanced matrix extensions (AMX) for quantized data types, specifically optimized for small matrix sizes and specific quantized formats.
- **Inputs**:
    - `M`: The number of rows in matrix A.
    - `N`: The number of columns in matrix B.
    - `KB`: The number of blocks in the K dimension, where K is the shared dimension between matrices A and B.
    - `_A`: A pointer to the matrix A data, which is in a quantized format.
    - `_B`: A pointer to the matrix B data, which is in a quantized format.
    - `C`: A pointer to the output matrix C, where the result of the matrix multiplication will be stored.
    - `ldc`: The leading dimension of matrix C, which is the number of elements between successive rows in memory.
- **Control Flow**:
    - The function begins by asserting that the input matrices meet certain size constraints and casts the input pointers to the appropriate types.
    - It initializes several thread-local storage arrays for intermediate tile data and results.
    - The function iterates over the blocks in the K dimension (KB) to perform the matrix multiplication in a tiled manner.
    - For each block, it unpacks the quantized data from matrix B and loads it into AMX tiles.
    - It performs the dot product of the tiles from matrices A and B using AMX instructions, storing the results in intermediate tile arrays.
    - The function accumulates the results across multiple blocks and scales them appropriately using the quantization scales.
    - Finally, it writes the accumulated results to the output matrix C.
- **Output**: The function does not return a value; it writes the result of the matrix multiplication to the output matrix C.


---
### ggml\_backend\_amx\_get\_alloc\_size<!-- {{#callable:ggml_backend_amx_get_alloc_size}} -->
The function `ggml_backend_amx_get_alloc_size` calculates the required allocation size for a tensor based on its type and dimensions, specifically for use with AMX kernels.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure, which contains information about the tensor's type and dimensions.
- **Control Flow**:
    - Retrieve the tensor's type and dimensions (K and N).
    - Define a lambda function `get_tensor_size` to calculate the size of a tensor row based on its type and dimension K.
    - Use `GGML_DISPATCH_QTYPES` macro to determine the row size for quantized types by calling `get_row_size` with the appropriate type and block size.
    - Check if the tensor type has AMX kernels using `qtype_has_amx_kernels`.
    - If the tensor type supports AMX kernels, return the total size by multiplying N with the row size calculated by `get_tensor_size`.
    - If the tensor type does not support AMX kernels, return the size calculated by [`ggml_nbytes`](../../ggml.c.driver.md#ggml_nbytes).
- **Output**: Returns a `size_t` value representing the required allocation size for the tensor.
- **Functions called**:
    - [`ggml_nbytes`](../../ggml.c.driver.md#ggml_nbytes)


---
### ggml\_backend\_amx\_convert\_weight<!-- {{#callable:ggml_backend_amx_convert_weight}} -->
The `ggml_backend_amx_convert_weight` function converts a tensor's weight data into a packed format suitable for AMX operations, ensuring the entire tensor is processed.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor whose weights are to be converted.
    - `data`: A pointer to the data that contains the weights to be converted.
    - `offset`: A size_t value representing the offset in the tensor data where conversion should start, which must be 0.
    - `size`: A size_t value representing the size of the data to be converted, which must match the total size of the tensor data.
- **Control Flow**:
    - The function asserts that the offset is 0 and the size matches the total size of the tensor data, ensuring only full tensor conversion is supported.
    - It retrieves the type of the tensor from the `tensor` structure.
    - The function extracts the dimensions `K` and `N` from the tensor's `ne` array, representing in_features and out_features respectively.
    - It uses a macro `GGML_DISPATCH_QTYPES` to handle different quantized data types, executing a lambda function for the specific type.
    - Within the lambda, it calls `convert_B_packed_format` to convert the data into a packed format suitable for AMX operations, using the tensor's data pointer, the input data, and the dimensions `N` and `K`.
- **Output**: The function does not return a value; it modifies the tensor's data in place to convert it into a packed format.
- **Functions called**:
    - [`ggml_nbytes`](../../ggml.c.driver.md#ggml_nbytes)


---
### ggml\_backend\_amx\_desired\_wsize<!-- {{#callable:ggml_backend_amx_desired_wsize}} -->
The function `ggml_backend_amx_desired_wsize` calculates the desired workspace size for a given tensor operation based on its type and dimensions.
- **Inputs**:
    - `dst`: A pointer to a `ggml_tensor` structure representing the destination tensor for which the desired workspace size is being calculated.
- **Control Flow**:
    - Retrieve the source tensor `src0` from the destination tensor `dst`.
    - Determine the type of the source tensor `src0` and store it in `TYPE`.
    - Check if the type is a floating-point type (specifically `GGML_TYPE_F16`).
    - If the type is floating-point, return 0 as no workspace is needed.
    - Retrieve the dimensions `M` and `K` from the destination tensor `dst` and source tensor `src0`, respectively.
    - Initialize `desired_wsize` to 0.
    - Use the `GGML_DISPATCH_QTYPES` macro to handle different quantized types, calculating the row size and updating `desired_wsize` accordingly.
    - Return the calculated `desired_wsize`.
- **Output**: The function returns a `size_t` value representing the desired workspace size for the tensor operation.


---
### ggml\_backend\_amx\_mul\_mat<!-- {{#callable:ggml_backend_amx_mul_mat}} -->
The function `ggml_backend_amx_mul_mat` performs matrix multiplication using Advanced Matrix Extensions (AMX) and AVX512 instructions, handling both floating-point and quantized data types.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` which contains parameters for computation, including workspace data and thread information.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result of the matrix multiplication will be stored.
- **Control Flow**:
    - Retrieve source tensors `src0` and `src1` from `dst`.
    - Determine the data type of `src0` and check if it is a floating-point type.
    - Calculate matrix dimensions M, N, and K from `dst` and `src0`.
    - If the data type is floating-point, use AVX512 instructions for matrix multiplication with parallel processing.
    - If the data type is quantized, prepare workspace and convert data as needed.
    - For single-row matrices (M == 1), use VNNI instructions for optimized processing.
    - For larger matrices, use AMX instructions to handle multiple tiles in parallel.
    - Synchronize threads using a barrier to ensure all data is ready before proceeding.
- **Output**: The function does not return a value; it modifies the `dst` tensor in-place to store the result of the matrix multiplication.
- **Functions called**:
    - [`ggml_barrier`](../ggml-cpu.c.driver.md#ggml_barrier)
    - [`(anonymous)::ggml_tile_config_init`](#(anonymous)::ggml_tile_config_init)


