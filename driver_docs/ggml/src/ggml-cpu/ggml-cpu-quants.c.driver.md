# Purpose
The provided C code is a sophisticated and highly optimized implementation designed for efficient quantization and vector dot product operations, primarily in the context of machine learning and signal processing applications. It is part of a larger library that supports various quantization schemes and is tailored for different hardware architectures, including ARM NEON, AVX, AVX2, Power9 Vector, and LoongArch ASX, leveraging SIMD instructions to enhance performance. The code focuses on narrow functionality, providing specialized functions for quantizing data into different bit-width formats and computing dot products between quantized vectors, which are crucial for reducing memory footprint and computational load on resource-constrained devices. It is not a standalone executable but a set of internal functions and macros intended to be integrated into larger systems, with conditional compilation ensuring portability and optimization across diverse platforms. Overall, the code exemplifies a performance-oriented approach to handling quantized data, essential for high-throughput applications in machine learning and signal processing.
# Imports and Dependencies

---
- `ggml-common.h`
- `ggml-quants.h`
- `ggml-cpu-quants.h`
- `ggml-impl.h`
- `ggml-cpu-impl.h`
- `ggml-cpu.h`
- `math.h`
- `string.h`
- `assert.h`
- `float.h`
- `stdlib.h`
- `stdio.h`


# Global Variables

---
### table\_b2b\_0
- **Type**: ``static const uint64_t` array`
- **Description**: `table_b2b_0` is a static constant array of 256 elements, each of type `uint64_t`. It is initialized with a macro `B8(00, 10)`, which likely expands to a 64-bit value, and the comment suggests a bitwise operation involving shifting.
- **Use**: This array is used to store precomputed 64-bit values for efficient lookup operations, likely in a context where bitwise manipulations are required.


---
### table\_b2b\_1
- **Type**: ``static const uint64_t``
- **Description**: `table_b2b_1` is a static constant array of 64-bit unsigned integers with a size of 256 elements, initialized with a macro `B8(10, 00)`. The array is likely used for lookup purposes, given its size and static nature.
- **Use**: This variable is used to store precomputed values for efficient access, likely in a lookup table context.


# Data Structures

---
### vec\_index\_t
- **Type**: `union`
- **Members**:
    - `vec_index`: A 128-bit vector containing eight 16-bit unsigned integers.
    - `index`: An array of eight 16-bit unsigned integers.
- **Description**: The `vec_index_t` is a union data structure that allows for the storage of either a 128-bit vector of eight 16-bit unsigned integers or an array of eight 16-bit unsigned integers. This union provides flexibility in accessing the data either as a SIMD vector for parallel processing or as a standard array for individual element manipulation.


---
### index\_t
- **Type**: `union`
- **Members**:
    - `vec`: An array of two __m256i vector types, typically used for SIMD operations.
    - `index`: An array of 16 uint32_t integers, providing direct access to individual index values.
- **Description**: The `index_t` union is a data structure that allows for efficient manipulation of data using SIMD (Single Instruction, Multiple Data) operations. It provides two views of the same data: as an array of two __m256i vectors, which are 256-bit wide and suitable for parallel processing, and as an array of 16 32-bit unsigned integers, allowing for individual access to each index. This design is useful in performance-critical applications where both vectorized operations and individual element access are required.


# Functions

---
### mul\_sum\_i8\_pairs<!-- {{#callable:mul_sum_i8_pairs}} -->
The `mul_sum_i8_pairs` function computes the sum of products of absolute values of two 8-bit integer vectors.
- **Inputs**:
    - `x`: An `__m128i` vector containing the first set of 8-bit integers.
    - `y`: An `__m128i` vector containing the second set of 8-bit integers.
- **Control Flow**:
    - The function first computes the absolute values of the elements in vector `x` using `__lsx_vsigncov_b`.
    - Next, it computes the signed values of vector `y` based on the signs of vector `x`.
    - Then, it performs a multiplication of the absolute values of `x` and the signed values of `y` using [`lsx_maddubs_h`](#lsx_maddubs_h), resulting in 16-bit intermediate values.
    - Finally, it multiplies the result by a vector of ones using [`lsx_madd_h`](#lsx_madd_h) to produce the final output.
- **Output**: The function returns an `__m128i` vector containing the sum of the products of the absolute values of `x` and the signed values of `y`.
- **Functions called**:
    - [`lsx_maddubs_h`](#lsx_maddubs_h)
    - [`lsx_madd_h`](#lsx_madd_h)


---
### hsum\_float\_8<!-- {{#callable:hsum_float_8}} -->
The `hsum_float_8` function computes the horizontal sum of eight single-precision floating-point values stored in a 256-bit vector.
- **Inputs**:
    - `x`: A 256-bit vector of type `__m256` containing eight single-precision floating-point values.
- **Control Flow**:
    - The function begins by extracting the upper 128 bits of the input vector `x` using [`lasx_extractf128`](#lasx_extractf128).
    - It then adds the extracted upper 128 bits to the lower 128 bits of `x` using the `__lsx_vfadd_s` function.
    - Next, it performs a pairwise addition on the resulting vector to sum the elements, utilizing `__lsx_vpickod_d` to pick the odd elements.
    - The function further adds the result to a vector that combines the even elements using `__lsx_vinsgr2vr_w` and `__lsx_vpickve2gr_w`.
    - Finally, it returns the first element of the resulting vector as the output.
- **Output**: The function returns a single float value, which is the horizontal sum of the eight input floating-point values.
- **Functions called**:
    - [`lasx_extractf128`](#lasx_extractf128)


---
### hsum\_i32\_8<!-- {{#callable:hsum_i32_8}} -->
The `hsum_i32_8` function computes the horizontal sum of 8 32-bit integers packed in a `__m256i` vector.
- **Inputs**:
    - `a`: A `__m256i` vector containing 8 packed 32-bit integers.
- **Control Flow**:
    - The function begins by creating two temporary vectors, `tmp1` and `tmp2`, using the `__lasx_xvpermi_q` intrinsic to permute the input vector `a`.
    - Next, it extracts the lower 128 bits of `tmp1` and `tmp2` into `tmp1_128` and `tmp2_128` respectively using [`lasx_extracti128_lo`](#lasx_extracti128_lo).
    - The function then computes the sum of the two 128-bit vectors using `__lsx_vadd_w`, resulting in `sum128`.
    - It separates the even and odd indexed elements of `sum128` into `ev` and `od` using `__lsx_vpickev_w` and `__lsx_vpickod_w`.
    - The even and odd sums are then added together to produce `sum64`.
    - Finally, the function extracts the two 32-bit integers from `sum64` using `__lsx_vpickve2gr_w` and returns their sum.
- **Output**: The function returns an integer representing the total sum of the 8 packed 32-bit integers in the input vector.
- **Functions called**:
    - [`lasx_extracti128_lo`](#lasx_extracti128_lo)


---
### hsum\_i32\_4<!-- {{#callable:hsum_i32_4}} -->
The `hsum_i32_4` function computes the horizontal sum of four 32-bit integers stored in a `__m128i` vector.
- **Inputs**:
    - `a`: A `__m128i` vector containing four 32-bit integers.
- **Control Flow**:
    - The function begins by creating an even indexed vector `ev` by picking even indexed elements from the input vector `a`.
    - Next, it creates an odd indexed vector `od` by picking odd indexed elements from the input vector `a`.
    - The function then computes the sum of the even and odd indexed vectors, resulting in a new vector `sum64`.
    - The two sums from `sum64` are extracted into `sum64_1` and `sum64_2` using the `__lsx_vpickve2gr_w` function.
    - Finally, the function returns the total sum by adding `sum64_1` and `sum64_2`.
- **Output**: The function returns an integer representing the horizontal sum of the four 32-bit integers from the input vector.


---
### bytes\_from\_bits\_32<!-- {{#callable:bytes_from_bits_32}} -->
Converts a 32-bit array of bytes into a 256-bit vector using specific shuffling and masking operations.
- **Inputs**:
    - `x`: A pointer to an array of 8-bit unsigned integers (bytes) that will be converted.
- **Control Flow**:
    - The function begins by declaring a 32-bit unsigned integer `x32` and copies 4 bytes from the input pointer `x` into `x32` using `memcpy`.
    - A shuffle mask is defined using [`lasx_set_d`](#lasx_set_d) to specify how the bytes will be rearranged.
    - The function then creates a 256-bit vector by replicating the 32-bit integer `x32` into a vector using `__lasx_xvreplgr2vr_w`.
    - The bytes are shuffled according to the defined mask using [`lasx_shuffle_b`](#lasx_shuffle_b).
    - A bit mask is created using `__lasx_xvreplgr2vr_d` to modify the bits of the resulting vector.
    - The bitwise OR operation is performed between the shuffled bytes and the bit mask using `__lasx_xvor_v`.
    - Finally, the function returns a vector where the bytes are compared against a vector of -1 using `__lasx_xvseq_b`.
- **Output**: Returns a 256-bit vector of bytes that has been shuffled and masked based on the input 32-bit integer.
- **Functions called**:
    - [`lasx_set_d`](#lasx_set_d)
    - [`lasx_shuffle_b`](#lasx_shuffle_b)


---
### bytes\_from\_nibbles\_32<!-- {{#callable:bytes_from_nibbles_32}} -->
Converts an array of nibbles (4-bit values) into bytes using SIMD operations.
- **Inputs**:
    - `rsi`: A pointer to an array of 8-bit unsigned integers (nibbles) that will be processed.
- **Control Flow**:
    - The function reads 16 bytes (128 bits) from the input array pointed to by `rsi` using `__lsx_vld`.
    - It then shifts the lower 4 bits of each byte to the right by 4 bits using `__lsx_vsrli_h`, effectively separating the high nibbles from the low nibbles.
    - The high and low nibbles are combined into a single 128-bit vector using [`lasx_insertf128`](#lasx_insertf128).
    - Finally, the combined vector is masked to retain only the lower 4 bits of each byte using `__lasx_xvandi_b`.
- **Output**: Returns a 256-bit vector containing the bytes formed from the nibbles in the input array.
- **Functions called**:
    - [`lasx_insertf128`](#lasx_insertf128)


---
### sum\_i16\_pairs\_float<!-- {{#callable:sum_i16_pairs_float}} -->
The `sum_i16_pairs_float` function computes the sum of pairs of 16-bit integers from a 256-bit integer vector and returns the result as a 256-bit floating-point vector.
- **Inputs**:
    - `x`: A 256-bit integer vector containing 16-bit integers.
- **Control Flow**:
    - The function first packs the 16-bit integers in `x` into pairs using `__lasx_xvpackod_h`, resulting in a new vector `v`.
    - Next, it computes the sum of the original vector `x` and the packed vector `v` using `__lasx_xvaddwev_w_h`, producing `summed_pairs`.
    - Finally, the summed pairs are converted to a floating-point vector using `__lasx_xvffint_s_w` and returned.
- **Output**: A 256-bit floating-point vector representing the summed pairs of the original 16-bit integers.


---
### mul\_sum\_us8\_pairs\_float<!-- {{#callable:mul_sum_us8_pairs_float}} -->
Multiplies two vectors of unsigned 8-bit integers and sums the resulting pairs as floating-point values.
- **Inputs**:
    - `ax`: A vector of unsigned 8-bit integers represented as a `__m256i` type.
    - `sy`: Another vector of unsigned 8-bit integers represented as a `__m256i` type.
- **Control Flow**:
    - The function begins by performing a multiplication of the input vectors `ax` and `sy` using the [`lasx_maddubs_h`](#lasx_maddubs_h) intrinsic, which computes the dot product and produces 16-bit intermediate results.
    - The resulting 16-bit values are then passed to the [`sum_i16_pairs_float`](#sum_i16_pairs_float) function, which sums the pairs of these values and converts them into floating-point format.
- **Output**: Returns a `__m256` type containing the summed floating-point values derived from the pairs of 16-bit results.
- **Functions called**:
    - [`lasx_maddubs_h`](#lasx_maddubs_h)
    - [`sum_i16_pairs_float`](#sum_i16_pairs_float)


---
### mul\_sum\_i8\_pairs\_float<!-- {{#callable:mul_sum_i8_pairs_float}} -->
Multiplies pairs of 8-bit integers from two vectors and sums the results as floating-point values.
- **Inputs**:
    - `x`: A vector of 256 bits containing 32 signed 8-bit integers.
    - `y`: A vector of 256 bits containing 32 signed 8-bit integers.
- **Control Flow**:
    - The function first computes the dot product of the input vectors `x` and `y` using the [`lasx_madd_h_b`](#lasx_madd_h_b) intrinsic, which multiplies pairs of 8-bit integers and accumulates the results.
    - The resulting vector, `dot`, contains the accumulated sums of the products of the pairs.
    - Finally, the function calls [`sum_i16_pairs_float`](#sum_i16_pairs_float) to convert the accumulated sums in `dot` into floating-point values and returns the result.
- **Output**: A vector of 256 bits containing the summed floating-point results of the products of the input integer pairs.
- **Functions called**:
    - [`lasx_madd_h_b`](#lasx_madd_h_b)
    - [`sum_i16_pairs_float`](#sum_i16_pairs_float)


---
### packNibbles<!-- {{#callable:packNibbles}} -->
The `packNibbles` function compresses 256-bit integer data into 128-bit format by manipulating the bits and performing saturation operations.
- **Inputs**:
    - `bytes`: A 256-bit integer (`__m256i`) containing the data to be packed, where each 16-bit lane holds two nibbles.
- **Control Flow**:
    - The function begins by creating a mask for the low byte using `__lasx_xvreplgr2vr_h`.
    - It then separates the high and low nibbles of the input `bytes` using bitwise operations.
    - The high nibbles are right-shifted by 4 bits to align them with the low nibbles.
    - The low and shifted high nibbles are combined using a bitwise OR operation.
    - The combined 256-bit result is then rearranged to prepare for compression into 128-bit format.
    - Two temporary variables are created to hold the maximum values of the lanes against zero and are saturated to a maximum of 7.
    - Finally, the function returns a packed 128-bit integer by interleaving the saturated results.
- **Output**: The output is a 128-bit integer (`__m128i`) that contains the packed and saturated values derived from the input `bytes`.


---
### mul\_add\_epi8\_sse<!-- {{#callable:mul_add_epi8_sse}} -->
The `mul_add_epi8_sse` function performs a multiply-add operation on two 128-bit integer vectors, treating the elements as signed 8-bit integers.
- **Inputs**:
    - `x`: A `__m128i` type input representing the first vector of signed 8-bit integers.
    - `y`: A `__m128i` type input representing the second vector of signed 8-bit integers.
- **Control Flow**:
    - The function first computes the absolute values of the elements in vector `x` using `_mm_sign_epi8`, storing the result in `ax`.
    - Next, it computes the signed values of the elements in vector `y` based on the sign of the corresponding elements in vector `x`, storing the result in `sy`.
    - Finally, it performs a multiply-add operation on the vectors `ax` and `sy` using `_mm_maddubs_epi16`, which multiplies the corresponding elements and adds them together, returning the result.
- **Output**: The output is a `__m128i` type value that contains the results of the multiply-add operation, represented as a vector of signed 16-bit integers.


---
### mul\_sum\_i8\_quad\_float<!-- {{#callable:mul_sum_i8_quad_float}} -->
The `mul_sum_i8_quad_float` function computes a weighted sum of products of 8-bit integers and returns the result as a 256-bit vector of floats.
- **Inputs**:
    - `x_1_0`: First 128-bit integer vector containing the first set of 8-bit integers for multiplication.
    - `x_1_1`: Second 128-bit integer vector containing the second set of 8-bit integers for multiplication.
    - `x_2_0`: Third 128-bit integer vector containing the third set of 8-bit integers for multiplication.
    - `x_2_1`: Fourth 128-bit integer vector containing the fourth set of 8-bit integers for multiplication.
    - `y_1_0`: Fifth 128-bit integer vector containing the first set of 8-bit integers to be multiplied with `x_1_0`.
    - `y_1_1`: Sixth 128-bit integer vector containing the second set of 8-bit integers to be multiplied with `x_1_1`.
    - `y_2_0`: Seventh 128-bit integer vector containing the third set of 8-bit integers to be multiplied with `x_2_0`.
    - `y_2_1`: Eighth 128-bit integer vector containing the fourth set of 8-bit integers to be multiplied with `x_2_1`.
- **Control Flow**:
    - The function initializes a constant vector `mone` with the value 1.
    - It computes the products of the corresponding elements of the input vectors `x_1_0` with `y_1_0`, `x_1_1` with `y_1_1`, `x_2_0` with `y_2_0`, and `x_2_1` with `y_2_1` using the [`mul_add_epi8_sse`](#mul_add_epi8_sse) function.
    - The results of these multiplications are then negated by multiplying with `mone` using the `_mm_madd_epi16` function.
    - The intermediate results are summed for both sets of products using `_mm_add_epi32` to produce two final sums, `p_1` and `p_2`.
    - Finally, the function converts the 32-bit integer results `p_1` and `p_2` into a 256-bit floating-point vector using `_mm256_cvtepi32_ps`.
- **Output**: The output is a 256-bit vector of floats representing the combined weighted sums of the products of the input integer vectors.
- **Functions called**:
    - [`mul_add_epi8_sse`](#mul_add_epi8_sse)


---
### quad\_fp16\_delta\_float<!-- {{#callable:quad_fp16_delta_float}} -->
The `quad_fp16_delta_float` function computes a 256-bit vector containing two float values derived from the products of two pairs of half-precision floating-point values.
- **Inputs**:
    - `x0`: The first half-precision floating-point value for the first product.
    - `y0`: The second half-precision floating-point value for the first product.
    - `x1`: The first half-precision floating-point value for the second product.
    - `y1`: The second half-precision floating-point value for the second product.
- **Control Flow**:
    - The function converts the half-precision values `x0`, `y0`, `x1`, and `y1` to single-precision floating-point using the `GGML_FP16_TO_FP32` macro.
    - It computes the products of `x0` and `y0`, and `x1` and `y1`.
    - The results are packed into a 256-bit vector using `_mm256_set_m128`, where the first 128 bits contain the product of `x1` and `y1`, and the second 128 bits contain the product of `x0` and `y0`.
- **Output**: The function returns a 256-bit vector of type `__m256`, where the lower half contains the product of `x0` and `y0`, and the upper half contains the product of `x1` and `y1`, both represented as single-precision floating-point values.


---
### hsum\_float\_4x4<!-- {{#callable:hsum_float_4x4}} -->
Computes the horizontal sum of four 128-bit vectors containing floating-point values.
- **Inputs**:
    - `a`: A `__m128` vector containing four single-precision floating-point values.
    - `b`: A `__m128` vector containing four single-precision floating-point values.
    - `c`: A `__m128` vector containing four single-precision floating-point values.
    - `d`: A `__m128` vector containing four single-precision floating-point values.
- **Control Flow**:
    - Uses the [`lsx_hadd_s`](#lsx_hadd_s) intrinsic to perform horizontal addition on the input vectors in pairs.
    - First, it horizontally adds vectors `a` and `b` to produce `res_0`.
    - Then, it horizontally adds vectors `c` and `d` to produce `res_1`.
    - Next, it horizontally adds `res_0` and `res_1` to combine the results.
    - The result is further reduced by applying horizontal addition multiple times to condense the sum into a single value.
- **Output**: Returns the first element of the final reduced vector, which represents the total horizontal sum of the four input vectors.
- **Functions called**:
    - [`lsx_hadd_s`](#lsx_hadd_s)


---
### lsx\_packs\_w<!-- {{#callable:lsx_packs_w}} -->
The `lsx_packs_w` function performs saturation and packing of two 128-bit integer vectors.
- **Inputs**:
    - `a`: A `__m128i` type vector containing the first set of integers to be processed.
    - `b`: A `__m128i` type vector containing the second set of integers to be processed.
- **Control Flow**:
    - The function first applies saturation to the vector `a` using the `__lsx_vsat_w` intrinsic, limiting the values to a range of -15 to 15, and stores the result in `tmp`.
    - Next, it applies the same saturation process to the vector `b`, storing the result in `tmp1`.
    - Finally, the function combines the saturated results from `tmp` and `tmp1` by picking even elements from `tmp` and odd elements from `tmp1` using the `__lsx_vpickev_h` intrinsic, and returns the packed result.
- **Output**: The output is a `__m128i` type vector that contains the packed and saturated results of the input vectors `a` and `b`.


---
### lsx\_packs\_h<!-- {{#callable:lsx_packs_h}} -->
The `lsx_packs_h` function performs saturation packing of two 128-bit integer vectors into a single vector.
- **Inputs**:
    - `a`: A `__m128i` type vector containing the first set of integers to be processed.
    - `b`: A `__m128i` type vector containing the second set of integers to be processed.
- **Control Flow**:
    - The function begins by declaring two `__m128i` variables, `tmp` and `tmp1`, to hold intermediate results.
    - It calls the `__lsx_vsat_h` function with the input vector `a`, which saturates the values in `a` to a specified range ([-128, 127]) and stores the result in `tmp`.
    - Next, it calls the `__lsx_vsat_h` function with the input vector `b`, saturating its values and storing the result in `tmp1`.
    - Finally, the function combines the saturated results from `tmp` and `tmp1` using the `__lsx_vpickev_b` function, which interleaves the bytes from both vectors, and returns the resulting vector.
- **Output**: The output is a `__m128i` type vector that contains the packed and saturated values from the input vectors `a` and `b`.


---
### lsx\_packus\_h<!-- {{#callable:lsx_packus_h}} -->
The `lsx_packus_h` function packs two 128-bit vectors of unsigned integers into a single vector, applying saturation and interleaving operations.
- **Inputs**:
    - `a`: A `__m128i` vector containing the first set of unsigned integers to be processed.
    - `b`: A `__m128i` vector containing the second set of unsigned integers to be processed.
- **Control Flow**:
    - The function first applies saturation to the elements of vector `a` using `__lsx_vsat_hu`, limiting the maximum value to 127.
    - It then applies saturation to the elements of vector `b` in the same manner.
    - Finally, the function interleaves the saturated results from `a` and `b` using `__lsx_vpickev_b`, returning the packed result.
- **Output**: The function returns a `__m128i` vector that contains the interleaved and saturated results of the input vectors `a` and `b`.


---
### lsx\_maddubs\_h<!-- {{#callable:lsx_maddubs_h}} -->
The `lsx_maddubs_h` function performs a packed multiplication of unsigned bytes followed by an addition of the results.
- **Inputs**:
    - `a`: A `__m128i` type input representing the first packed vector of unsigned bytes.
    - `b`: A `__m128i` type input representing the second packed vector of unsigned bytes.
- **Control Flow**:
    - The function begins by declaring two `__m128i` variables, `tmp1` and `tmp2`, to hold intermediate results.
    - It computes `tmp1` by calling `__lsx_vmulwev_h_b(a, b)`, which performs a packed multiplication of the even-indexed bytes of `a` and `b`.
    - Next, it computes `tmp2` by calling `__lsx_vmulwod_h_b(a, b)`, which performs a packed multiplication of the odd-indexed bytes of `a` and `b`.
    - Finally, it returns the result of adding `tmp1` and `tmp2` using the `__lsx_vsadd_h` function.
- **Output**: The output is a `__m128i` type value that contains the sum of the products of the even and odd indexed bytes from the inputs `a` and `b`.


---
### lsx\_madd\_h<!-- {{#callable:lsx_madd_h}} -->
Performs a multiply-accumulate operation on two `__m128i` vectors using half-word elements.
- **Inputs**:
    - `a`: First input vector of type `__m128i` containing half-word elements.
    - `b`: Second input vector of type `__m128i` containing half-word elements.
- **Control Flow**:
    - The function begins by declaring two `__m128i` variables, `tmp1` and `tmp2`.
    - `tmp1` is assigned the result of multiplying the even half-words of `a` and `b` using the `__lsx_vmulwev_w_h` intrinsic.
    - `tmp2` is assigned the result of multiplying the odd half-words of `a` and `b` using the `__lsx_vmulwod_w_h` intrinsic.
    - Finally, the function returns the sum of `tmp1` and `tmp2` using the `__lsx_vadd_w` intrinsic.
- **Output**: Returns a `__m128i` vector that contains the accumulated results of the half-word multiplications.


---
### lsx\_set\_w<!-- {{#callable:lsx_set_w}} -->
The `lsx_set_w` function creates a 128-bit integer vector from four 32-bit integers.
- **Inputs**:
    - `a`: The first 32-bit integer to be placed in the vector.
    - `b`: The second 32-bit integer to be placed in the vector.
    - `c`: The third 32-bit integer to be placed in the vector.
    - `d`: The fourth 32-bit integer to be placed in the vector.
- **Control Flow**:
    - The function initializes a vector `__ret` of type `v4i32` with the values of `d`, `c`, `b`, and `a` in that order.
    - The vector `__ret` is then cast to the type `__m128i` and returned as the output.
- **Output**: The function returns a `__m128i` type representing a 128-bit integer vector containing the four input integers arranged as (d, c, b, a).


---
### lsx\_shuffle\_b<!-- {{#callable:lsx_shuffle_b}} -->
The `lsx_shuffle_b` function performs a conditional shuffle of bytes between two `__m128i` vectors based on a mask derived from the second vector.
- **Inputs**:
    - `a`: The first input vector of type `__m128i` containing bytes to be shuffled.
    - `b`: The second input vector of type `__m128i` used to generate the shuffle mask.
- **Control Flow**:
    - A mask is created by replicating the value 0x8f into a vector, which is used to extract specific bits from vector `b`.
    - The low 4 bits and sign bits of vector `b` are masked and combined with 0x10 to prepare for positive values.
    - A comparison is made against zero to create a mask that indicates which bytes in `b` are non-negative.
    - The mask is applied to filter out bytes from `b` that are less than zero.
    - Finally, the `__lsx_vshuf_b` function is called to shuffle bytes from vector `a` based on the computed mask.
- **Output**: The function returns a new `__m128i` vector containing the shuffled bytes from vector `a`, determined by the mask derived from vector `b`.


---
### lsx\_hadd\_h<!-- {{#callable:lsx_hadd_h}} -->
The `lsx_hadd_h` function performs horizontal addition of two 128-bit integer vectors containing halfword elements.
- **Inputs**:
    - `a`: A `__m128i` type vector containing halfword elements to be added.
    - `b`: A `__m128i` type vector containing halfword elements to be added.
- **Control Flow**:
    - The function begins by picking even halfword elements from vector `b` and odd halfword elements from vector `a` using the `__lsx_vpickev_h` intrinsic, storing the result in `tmp1`.
    - Next, it picks odd halfword elements from vector `b` and even halfword elements from vector `a` using the `__lsx_vpickod_h` intrinsic, storing the result in `tmp2`.
    - Finally, it adds the two temporary vectors `tmp1` and `tmp2` together using the `__lsx_vadd_h` intrinsic and returns the result.
- **Output**: The output is a `__m128i` type vector that contains the horizontal sum of the halfword elements from the input vectors `a` and `b`.


---
### lsx\_hadd\_w<!-- {{#callable:lsx_hadd_w}} -->
The `lsx_hadd_w` function performs horizontal addition of two 128-bit integer vectors.
- **Inputs**:
    - `a`: A `__m128i` type vector containing four 32-bit integers.
    - `b`: A `__m128i` type vector containing four 32-bit integers.
- **Control Flow**:
    - The function begins by creating a new vector `tmp1` that contains the even-indexed elements of vector `b` interleaved with the elements of vector `a` using the `__lsx_vpickev_w` intrinsic.
    - The next vector `tmp2` is created to hold the odd-indexed elements of vector `b` interleaved with the elements of vector `a` using the `__lsx_vpickod_w` intrinsic.
    - Finally, the function returns the result of adding `tmp1` and `tmp2` together using the `__lsx_vadd_w` intrinsic.
- **Output**: The output is a `__m128i` type vector that contains the horizontal sum of the input vectors `a` and `b`, effectively combining their elements.


---
### lsx\_hadd\_s<!-- {{#callable:lsx_hadd_s}} -->
The `lsx_hadd_s` function performs horizontal addition of two 128-bit vectors containing single-precision floating-point values.
- **Inputs**:
    - `a`: A `__m128` vector containing four single-precision floating-point values.
    - `b`: A `__m128` vector containing four single-precision floating-point values.
- **Control Flow**:
    - The function begins by using the `__lsx_vpickev_w` intrinsic to select even-indexed elements from vector `b` and odd-indexed elements from vector `a`, storing the result in `tmp1`.
    - Next, it uses the `__lsx_vpickod_w` intrinsic to select odd-indexed elements from vector `b` and even-indexed elements from vector `a`, storing the result in `tmp2`.
    - Finally, the function adds the two temporary vectors `tmp1` and `tmp2` using the `__lsx_vfadd_s` intrinsic and returns the result.
- **Output**: A `__m128` vector containing the result of the horizontal addition of the input vectors `a` and `b`, which combines the elements in a specific manner.


---
### \_\_\_\_m256i<!-- {{#callable:____m256i}} -->
Converts a single `__m128i` input into a `__m256i` output using vector permutation instructions.
- **Inputs**:
    - `in`: A `__m128i` type input representing a 128-bit vector that will be converted to a 256-bit vector.
- **Control Flow**:
    - Initializes a `__m256i` variable `out` to zero using `__lasx_xvldi(0)`.
    - Uses inline assembly to perform a series of vector permutation operations based on the input `in`.
    - Iterates over all registers defined in `__ALL_REGS` to apply the permutation if the output register matches.
    - For each matching output register, iterates over all input registers defined in `VREGS_PREFIX` to apply the `xvpermi.q` instruction.
    - The permutation instruction rearranges the elements of the input vector to form the output vector.
- **Output**: Returns a `__m256i` type output that contains the permuted result of the input `__m128i` vector.


---
### lasx\_set\_q<!-- {{#callable:lasx_set_q}} -->
The `lasx_set_q` function combines two `__m128i` inputs into a single `__m256i` output using inline assembly for vector permutation.
- **Inputs**:
    - `inhi`: A `__m128i` type input representing the high part of the output vector.
    - `inlo`: A `__m128i` type input representing the low part of the output vector.
- **Control Flow**:
    - The function begins by declaring a variable `out` of type `__m256i` to hold the result.
    - It uses inline assembly to perform vector operations based on the input registers.
    - The assembly code iterates over all registers, checking if the input high and low parts match specific vector registers.
    - If a match is found, it performs a vector permutation operation using `xvpermi.q` to combine the inputs into the output.
    - The function also includes conditional checks to perform a bitwise OR operation if the output register matches the high input register.
    - Finally, the result stored in `out` is returned.
- **Output**: The function outputs a `__m256i` type value that combines the two `__m128i` inputs into a single 256-bit vector.


---
### lasx\_extracti128\_lo<!-- {{#callable:lasx_extracti128_lo}} -->
Extracts the lower 128 bits from a 256-bit integer input.
- **Inputs**:
    - `in`: A 256-bit integer input of type `__m256i` from which the lower 128 bits will be extracted.
- **Control Flow**:
    - The function uses inline assembly to perform the extraction operation.
    - It checks if the output variable `out` is not the same as the input variable `in` to avoid unnecessary operations.
    - It iterates over all registers defined by `__ALL_REGS` and checks if the output register matches the current register.
    - For each matching register, it checks if the input register matches the current register and performs a bitwise OR operation to extract the lower bits.
- **Output**: Returns a 128-bit integer of type `__m128i` containing the lower part of the input 256-bit integer.


---
### lasx\_extracti128\_hi<!-- {{#callable:lasx_extracti128_hi}} -->
Extracts the high 128 bits from a 256-bit integer using inline assembly.
- **Inputs**:
    - `in`: A `__m256i` type input representing a 256-bit integer from which the high 128 bits will be extracted.
- **Control Flow**:
    - The function uses inline assembly to perform operations on the input data.
    - It iterates over all registers defined by `__ALL_REGS` to find the appropriate register for the output.
    - For each register, it checks if the output register matches and then iterates over the input registers.
    - If a match is found for the input register, it executes the `xvpermi.q` instruction to extract the high 128 bits.
    - The assembly code is structured with conditional directives to ensure the correct registers are used.
- **Output**: Returns a `__m128i` type value containing the high 128 bits extracted from the input 256-bit integer.


---
### lasx\_set\_w<!-- {{#callable:lasx_set_w}} -->
The `lasx_set_w` function initializes a 256-bit integer vector with eight 32-bit integer values.
- **Inputs**:
    - `e7`: The highest 32-bit integer value to be stored in the vector.
    - `e6`: The second highest 32-bit integer value to be stored in the vector.
    - `e5`: The third highest 32-bit integer value to be stored in the vector.
    - `e4`: The fourth highest 32-bit integer value to be stored in the vector.
    - `e3`: The fifth highest 32-bit integer value to be stored in the vector.
    - `e2`: The sixth highest 32-bit integer value to be stored in the vector.
    - `e1`: The seventh highest 32-bit integer value to be stored in the vector.
    - `e0`: The lowest 32-bit integer value to be stored in the vector.
- **Control Flow**:
    - The function begins by declaring a variable `__ret` of type `v8i32`, which is an array of eight 32-bit integers.
    - The values passed as arguments (e0 to e7) are assigned to the corresponding positions in the `__ret` array.
    - Finally, the function returns the `__ret` array cast to the type `__m256i`, which represents a 256-bit integer vector.
- **Output**: The output is a `__m256i` type representing a 256-bit integer vector containing the eight input integer values.


---
### lasx\_set\_d<!-- {{#callable:lasx_set_d}} -->
Sets four 64-bit integers into a 256-bit vector.
- **Inputs**:
    - `a`: The first 64-bit integer to be set in the vector.
    - `b`: The second 64-bit integer to be set in the vector.
    - `c`: The third 64-bit integer to be set in the vector.
    - `d`: The fourth 64-bit integer to be set in the vector.
- **Control Flow**:
    - The function initializes a vector `__ret` of type `v4i64` with the values of `d`, `c`, `b`, and `a` in that order.
    - The vector `__ret` is then cast to the type `__m256i` and returned as the output.
- **Output**: Returns a 256-bit vector containing the four 64-bit integers in reverse order of their input.


---
### lasx\_insertf128<!-- {{#callable:lasx_insertf128}} -->
Inserts a 128-bit integer vector into a 256-bit integer vector.
- **Inputs**:
    - `x`: A `__m128i` type representing the first 128-bit integer vector to be inserted.
    - `y`: A `__m128i` type representing the second 128-bit integer vector to be inserted.
- **Control Flow**:
    - The function directly calls [`lasx_set_q`](#lasx_set_q) with the inputs `x` and `y`.
    - No conditional statements or loops are present in the function.
- **Output**: Returns a `__m256i` type which is the result of the [`lasx_set_q`](#lasx_set_q) function, combining the two input vectors.
- **Functions called**:
    - [`lasx_set_q`](#lasx_set_q)


---
### lasx\_shuffle\_b<!-- {{#callable:lasx_shuffle_b}} -->
The `lasx_shuffle_b` function performs a byte-wise shuffle of two 256-bit vectors based on a masking operation.
- **Inputs**:
    - `a`: A 256-bit vector of type `__m256i` that serves as the primary input for the shuffle operation.
    - `b`: A 256-bit vector of type `__m256i` that is used to generate a mask for the shuffle operation.
- **Control Flow**:
    - A constant integer `f` is initialized with the value 0x8f, which is used to create a mask.
    - The function creates a mask vector `mask_f` by replicating the value of `f` across all bytes using `__lasx_xvreplgr2vr_b`.
    - A zero vector `zero` is initialized using `__lasx_xvldi(0)`.
    - The function computes `tmp0` by performing a bitwise AND operation between vector `b` and `mask_f` to extract specific bits.
    - The `tmp0` vector is then modified by performing a bitwise OR operation with 0x10 to prepare the mask for positive values.
    - A comparison is made to create a mask vector that indicates which elements of `tmp0` are greater than or equal to zero using `__lasx_xvsle_b`.
    - The final mask is applied to `tmp0` to filter out elements that do not meet the criteria using a bitwise AND operation.
    - The function returns the result of the shuffle operation performed on vector `a` using the modified mask and the zero vector.
- **Output**: The output is a 256-bit vector of type `__m256i` that contains the shuffled bytes from vector `a` based on the computed mask.


---
### lasx\_extu8\_16<!-- {{#callable:lasx_extu8_16}} -->
Extracts 16 unsigned 8-bit integers from a 128-bit integer vector and extends them to a 256-bit integer vector.
- **Inputs**:
    - `a`: A `__m128i` type input representing a 128-bit integer vector containing 16 unsigned 8-bit integers.
- **Control Flow**:
    - The function calls `__lasx_vext2xv_hu_bu` with the input `a` converted to `__m256i` type.
    - The `__lasx_vext2xv_hu_bu` function is responsible for extending the 8-bit integers to a 256-bit format.
- **Output**: Returns a `__m256i` type value that contains the extended 256-bit integer vector derived from the input 128-bit vector.
- **Functions called**:
    - [`____m256i`](#____m256i)


---
### lasx\_ext8\_16<!-- {{#callable:lasx_ext8_16}} -->
Extends a 128-bit integer vector to a 256-bit integer vector by duplicating its elements.
- **Inputs**:
    - `a`: A `__m128i` type representing a 128-bit integer vector that will be extended.
- **Control Flow**:
    - The function takes a single input of type `__m128i`.
    - It calls the intrinsic function `__lasx_vext2xv_h_b` to perform the extension operation.
    - The input vector `a` is cast to `__m256i` type before being passed to the intrinsic function.
- **Output**: Returns a `__m256i` type representing the extended 256-bit integer vector.
- **Functions called**:
    - [`____m256i`](#____m256i)


---
### lasx\_ext16\_32<!-- {{#callable:lasx_ext16_32}} -->
Extends a 128-bit integer vector to a 256-bit integer vector by extracting elements.
- **Inputs**:
    - `a`: A `__m128i` type input representing a 128-bit integer vector.
- **Control Flow**:
    - The function directly calls `__lasx_vext2xv_w_h` with the input `a` converted to `__m256i` type.
    - The result of the call to `__lasx_vext2xv_w_h` is returned as the output.
- **Output**: Returns a `__m256i` type output which is a 256-bit integer vector created from the input 128-bit vector.
- **Functions called**:
    - [`____m256i`](#____m256i)


---
### lasx\_extracti128<!-- {{#callable:lasx_extracti128}} -->
Extracts a 128-bit integer from a 256-bit integer based on the specified position.
- **Inputs**:
    - `a`: A 256-bit integer represented as `__m256i` from which a 128-bit integer will be extracted.
    - `pos`: An integer indicating the position of the 128-bit integer to extract; 0 for the lower half and 1 for the upper half.
- **Control Flow**:
    - The function checks the value of `pos` to determine which half of the 256-bit integer to extract.
    - If `pos` is 0, it calls the [`lasx_extracti128_lo`](#lasx_extracti128_lo) function to extract the lower 128 bits.
    - If `pos` is 1, it calls the [`lasx_extracti128_hi`](#lasx_extracti128_hi) function to extract the upper 128 bits.
- **Output**: Returns a 128-bit integer represented as `__m128i`, which is either the lower or upper half of the input 256-bit integer based on the value of `pos`.
- **Functions called**:
    - [`lasx_extracti128_lo`](#lasx_extracti128_lo)
    - [`lasx_extracti128_hi`](#lasx_extracti128_hi)


---
### lasx\_extractf128<!-- {{#callable:lasx_extractf128}} -->
Extracts a 128-bit vector from a 256-bit vector based on the specified position.
- **Inputs**:
    - `a`: A 256-bit vector of type `__m256` from which a 128-bit vector will be extracted.
    - `pos`: An integer indicating the position of the 128-bit vector to extract; 0 for the lower half and 1 for the upper half.
- **Control Flow**:
    - The function checks the value of `pos` to determine which half of the 256-bit vector to extract.
    - If `pos` is 0, it calls [`lasx_extracti128_lo`](#lasx_extracti128_lo) to extract the lower 128 bits.
    - If `pos` is 1, it calls [`lasx_extracti128_hi`](#lasx_extracti128_hi) to extract the upper 128 bits.
- **Output**: Returns a 128-bit vector of type `__m128` containing the extracted bits from the specified position of the input 256-bit vector.
- **Functions called**:
    - [`lasx_extracti128_lo`](#lasx_extracti128_lo)
    - [`lasx_extracti128_hi`](#lasx_extracti128_hi)


---
### lasx\_maddubs\_h<!-- {{#callable:lasx_maddubs_h}} -->
Performs a packed multiplication and addition operation on two vectors of 16-bit integers.
- **Inputs**:
    - `a`: A vector of 16-bit integers representing the first operand for the multiplication.
    - `b`: A vector of 16-bit integers representing the second operand for the multiplication.
- **Control Flow**:
    - The function begins by declaring two `__m256i` variables, `tmp1` and `tmp2`, to hold intermediate results.
    - `tmp1` is computed by calling the intrinsic `__lasx_xvmulwev_h_b`, which performs a packed multiplication of the even elements of `a` and `b`.
    - `tmp2` is computed by calling the intrinsic `__lasx_xvmulwod_h_b`, which performs a packed multiplication of the odd elements of `a` and `b`.
    - Finally, the function returns the result of adding `tmp1` and `tmp2` using the intrinsic `__lasx_xvsadd_h`.
- **Output**: Returns a vector of 16-bit integers that is the result of adding the products of the even and odd indexed elements of the input vectors `a` and `b`.


---
### lasx\_madd\_h<!-- {{#callable:lasx_madd_h}} -->
Performs a multiply-accumulate operation on two vectors of 16-bit integers.
- **Inputs**:
    - `a`: A vector of 16-bit integers representing the first operand for the multiply-accumulate operation.
    - `b`: A vector of 16-bit integers representing the second operand for the multiply-accumulate operation.
- **Control Flow**:
    - The function begins by declaring two `__m256i` variables, `tmp1` and `tmp2`, to hold intermediate results.
    - It computes the element-wise multiplication of the lower halves of `a` and `b` using `__lasx_xvmulwev_w_h`, storing the result in `tmp1`.
    - Next, it computes the element-wise multiplication of the upper halves of `a` and `b` using `__lasx_xvmulwod_w_h`, storing the result in `tmp2`.
    - Finally, it adds the results stored in `tmp1` and `tmp2` using `__lasx_xvadd_w` and returns the final result.
- **Output**: Returns a vector of 32-bit integers that contains the sum of the products of the corresponding elements from the two input vectors.


---
### lasx\_packs\_w<!-- {{#callable:lasx_packs_w}} -->
The `lasx_packs_w` function packs two 256-bit integer vectors by saturating their values and then interleaving the results.
- **Inputs**:
    - `a`: A 256-bit integer vector that will be saturated and packed.
    - `b`: A second 256-bit integer vector that will also be saturated and packed.
- **Control Flow**:
    - The function begins by declaring two `__m256i` variables, `tmp` and `tmp1`.
    - It applies the saturation operation to the input vector `a` using the `__lasx_xvsat_w` intrinsic, which limits the values to a range defined by 15, storing the result in `tmp`.
    - Similarly, it saturates the input vector `b` and stores the result in `tmp1`.
    - Finally, it interleaves the saturated results from `tmp` and `tmp1` using the `__lasx_xvpickev_h` intrinsic, returning the packed result.
- **Output**: The function returns a 256-bit integer vector that contains the interleaved and saturated values from the input vectors `a` and `b`.


---
### lasx\_packs\_h<!-- {{#callable:lasx_packs_h}} -->
The `lasx_packs_h` function packs two 256-bit integer vectors into a single vector with saturation and even-odd selection.
- **Inputs**:
    - `a`: A `__m256i` type vector containing the first set of integers to be packed.
    - `b`: A `__m256i` type vector containing the second set of integers to be packed.
- **Control Flow**:
    - The function first applies saturation to the vector `a` using the `__lasx_xvsat_h` intrinsic with a saturation limit of 7, storing the result in `tmp`.
    - Next, it applies saturation to the vector `b` in a similar manner, storing the result in `tmp1`.
    - Finally, the function combines the two saturated vectors by picking even elements from `tmp` and odd elements from `tmp1` using the `__lasx_xvpickev_b` intrinsic, and returns the resulting vector.
- **Output**: The function returns a `__m256i` type vector that contains the packed and saturated results of the input vectors `a` and `b`.


---
### lasx\_madd\_h\_b<!-- {{#callable:lasx_madd_h_b}} -->
The `lasx_madd_h_b` function performs a multiply-accumulate operation on two 256-bit integer vectors.
- **Inputs**:
    - `a`: A `__m256i` type vector containing the first set of integers for the operation.
    - `b`: A `__m256i` type vector containing the second set of integers for the operation.
- **Control Flow**:
    - The function begins by declaring two `__m256i` variables, `tmp1` and `tmp2`.
    - `tmp1` is assigned the result of the element-wise multiplication of the lower half of `a` and `b` using the `__lasx_xvmulwev_h_b` intrinsic.
    - `tmp2` is assigned the result of the element-wise multiplication of the upper half of `a` and `b` using the `__lasx_xvmulwod_h_b` intrinsic.
    - Finally, the function returns the sum of `tmp1` and `tmp2` using the `__lasx_xvadd_h` intrinsic.
- **Output**: The output is a `__m256i` type vector that contains the result of the multiply-accumulate operation, which combines the results of the two multiplications followed by their addition.


---
### lasx\_xvrepl128vei\_h<!-- {{#callable:lasx_xvrepl128vei_h}} -->
The `lasx_xvrepl128vei_h` function replicates a specified 128-bit segment of a 256-bit integer based on the provided index.
- **Inputs**:
    - `a`: A 256-bit integer of type `__m256i` from which a 128-bit segment will be replicated.
    - `b`: An unsigned integer that specifies which 128-bit segment to replicate, ranging from 0 to 7.
- **Control Flow**:
    - The function uses a `switch` statement to determine the value of `b`.
    - For each case from 0 to 7, it calls the `__lasx_xvrepl128vei_h` function with `a` and the corresponding index.
    - If `b` is outside the range of 0 to 7, the function invokes `__builtin_unreachable()` to indicate that this case should not occur.
- **Output**: The function returns a 256-bit integer of type `__m256i`, which contains the replicated 128-bit segment from the input `a` based on the index specified by `b`.


---
### lasx\_xvandi\_b\_bit<!-- {{#callable:lasx_xvandi_b_bit}} -->
The `lasx_xvandi_b_bit` function applies a bitwise AND operation on a 256-bit integer using a specified bit position.
- **Inputs**:
    - `a`: A 256-bit integer represented as a `__m256i` type.
    - `b`: An unsigned integer specifying the bit position to be used in the AND operation.
- **Control Flow**:
    - The function uses a `switch` statement to determine the value of `b`.
    - For each case from 0 to 7, it calls the `__lasx_xvandi_b` function with `a` and a bitmask created by left-shifting 1 by `b` positions.
    - If `b` is outside the range of 0 to 7, the function invokes `__builtin_unreachable()` to indicate that this case should not occur.
- **Output**: The function returns a `__m256i` type result, which is the result of the bitwise AND operation between `a` and the generated bitmask.


---
### quantize\_row\_q4\_0<!-- {{#callable:quantize_row_q4_0}} -->
This function quantizes a row of floating-point values using a reference quantization function.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that represent the data to be quantized.
    - `y`: A pointer to a memory location where the quantized output will be stored.
    - `k`: An integer value that may represent the number of elements to quantize or a specific quantization parameter.
- **Control Flow**:
    - The function calls [`quantize_row_q4_0_ref`](../ggml-quants.c.driver.md#quantize_row_q4_0_ref), passing the input parameters `x`, `y`, and `k` directly to it.
    - There are no conditional statements or loops in this function; it simply delegates the work to the referenced function.
- **Output**: The function does not return a value; instead, it modifies the memory pointed to by `y` to store the quantized results.
- **Functions called**:
    - [`quantize_row_q4_0_ref`](../ggml-quants.c.driver.md#quantize_row_q4_0_ref)


---
### quantize\_row\_q4\_1<!-- {{#callable:quantize_row_q4_1}} -->
The `quantize_row_q4_1` function quantizes a row of floating-point values using a reference quantization function.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that represent the data to be quantized.
    - `y`: A pointer to the output buffer where the quantized data will be stored.
    - `k`: An integer value that likely represents the number of elements to quantize or a specific quantization parameter.
- **Control Flow**:
    - The function calls [`quantize_row_q4_1_ref`](../ggml-quants.c.driver.md#quantize_row_q4_1_ref), passing the input parameters `x`, `y`, and `k` directly to it.
    - There are no conditional statements or loops in this function; it simply delegates the work to the referenced function.
- **Output**: The function does not return a value; instead, it modifies the output buffer pointed to by `y` with the quantized results.
- **Functions called**:
    - [`quantize_row_q4_1_ref`](../ggml-quants.c.driver.md#quantize_row_q4_1_ref)


---
### quantize\_row\_q5\_0<!-- {{#callable:quantize_row_q5_0}} -->
This function serves as a wrapper that calls the [`quantize_row_q5_0_ref`](../ggml-quants.c.driver.md#quantize_row_q5_0_ref) function to quantize a row of data.
- **Inputs**:
    - `x`: A pointer to an array of floats representing the input data to be quantized.
    - `y`: A pointer to a memory location where the quantized output will be stored.
    - `k`: An integer representing a parameter that may influence the quantization process.
- **Control Flow**:
    - The function directly calls [`quantize_row_q5_0_ref`](../ggml-quants.c.driver.md#quantize_row_q5_0_ref) with the provided inputs without any additional processing or checks.
- **Output**: The function does not return a value; instead, it modifies the output in the memory location pointed to by `y`.
- **Functions called**:
    - [`quantize_row_q5_0_ref`](../ggml-quants.c.driver.md#quantize_row_q5_0_ref)


---
### quantize\_row\_q5\_1<!-- {{#callable:quantize_row_q5_1}} -->
The `quantize_row_q5_1` function quantizes a row of data by calling a reference quantization function.
- **Inputs**:
    - `x`: A pointer to an array of floats representing the data to be quantized.
    - `y`: A pointer to a memory location where the quantized output will be stored.
    - `k`: An integer representing the quantization parameter or index.
- **Control Flow**:
    - The function directly calls [`quantize_row_q5_1_ref`](../ggml-quants.c.driver.md#quantize_row_q5_1_ref), passing the same arguments it received.
    - There are no conditional statements or loops in this function, making it a straightforward delegation to another function.
- **Output**: The function does not return a value; instead, it modifies the output in the memory location pointed to by `y`.
- **Functions called**:
    - [`quantize_row_q5_1_ref`](../ggml-quants.c.driver.md#quantize_row_q5_1_ref)


---
### quantize\_row\_q8\_0<!-- {{#callable:quantize_row_q8_0}} -->
The `quantize_row_q8_0` function quantizes a row of floating-point values into a specific format using various SIMD optimizations based on the architecture.
- **Inputs**:
    - `x`: A pointer to an array of `float` values that represent the input data to be quantized.
    - `vy`: A pointer to a destination where the quantized output will be stored, structured as an array of `block_q8_0`.
    - `k`: An integer representing the number of elements in the input array `x`, which must be a multiple of `QK8_0`.
- **Control Flow**:
    - The function begins by asserting that `QK8_0` is equal to 32 and that `k` is a multiple of `QK8_0`.
    - It calculates the number of blocks `nb` by dividing `k` by `QK8_0`.
    - Depending on the defined architecture (e.g., ARM NEON, WASM SIMD, AVX, etc.), it executes different SIMD-based loops to process the input data.
    - For each block, it loads the input data, computes the maximum absolute value, and normalizes the data based on this maximum.
    - The normalized values are then quantized and stored in the output structure.
- **Output**: The function outputs quantized data in the `vy` pointer, where each block contains a normalized value and an array of quantized integers.
- **Functions called**:
    - [`vmaxvq_f32`](ggml-cpu-impl.h.driver.md#vmaxvq_f32)
    - [`vcvtnq_s32_f32`](ggml-cpu-impl.h.driver.md#vcvtnq_s32_f32)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)
    - [`lasx_extractf128`](#lasx_extractf128)
    - [`lasx_extracti128`](#lasx_extracti128)
    - [`lsx_packs_w`](#lsx_packs_w)
    - [`lsx_packs_h`](#lsx_packs_h)
    - [`quantize_row_q8_0_ref`](../ggml-quants.c.driver.md#quantize_row_q8_0_ref)


---
### quantize\_row\_q8\_1<!-- {{#callable:quantize_row_q8_1}} -->
Quantizes a row of floating-point values into a specific format and computes a summary statistic.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that are to be quantized.
    - `vy`: A pointer to a destination where the quantized data will be stored.
    - `k`: An integer representing the number of elements in the input array, which must be a multiple of `QK8_1`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK8_1`.
    - It calculates the number of blocks `nb` by dividing `k` by `QK8_1`.
    - Depending on the compilation flags, it uses different SIMD instructions (ARM NEON, WebAssembly SIMD, AVX, etc.) to process the data.
    - For each block, it loads the floating-point values, computes the maximum absolute value, and derives a scaling factor `d`.
    - It normalizes the values by multiplying with the inverse of `d`, rounds them, and converts them to integers.
    - The quantized values are stored in the output structure, along with a summary statistic `s` which is the scaled sum of the quantized values.
- **Output**: The function outputs a structure containing the quantized values and a summary statistic, both stored in the memory location pointed to by `vy`.
- **Functions called**:
    - [`vmaxvq_f32`](ggml-cpu-impl.h.driver.md#vmaxvq_f32)
    - [`vcvtnq_s32_f32`](ggml-cpu-impl.h.driver.md#vcvtnq_s32_f32)
    - [`vaddvq_s32`](ggml-cpu-impl.h.driver.md#vaddvq_s32)
    - [`hsum_i32_8`](#hsum_i32_8)
    - [`hsum_i32_4`](#hsum_i32_4)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)
    - [`lasx_extractf128`](#lasx_extractf128)
    - [`lasx_extracti128`](#lasx_extracti128)
    - [`lsx_packs_w`](#lsx_packs_w)
    - [`lsx_packs_h`](#lsx_packs_h)
    - [`quantize_row_q8_1_ref`](../ggml-quants.c.driver.md#quantize_row_q8_1_ref)


---
### nearest\_int<!-- {{#callable:nearest_int}} -->
Converts a floating-point number to its nearest integer representation using bit manipulation.
- **Inputs**:
    - `fval`: A floating-point number that is to be converted to the nearest integer.
- **Control Flow**:
    - Assert that the absolute value of `fval` does not exceed 4194303.f to ensure it is within a valid range.
    - Add 12582912.f to `fval` to adjust its value for the subsequent bit manipulation.
    - Use `memcpy` to copy the adjusted float value into an integer variable `i` to interpret its bits directly.
    - Apply a bitwise AND operation with 0x007fffff to isolate the fractional part of the float, and then subtract 0x00400000 to obtain the final integer result.
- **Output**: Returns the nearest integer representation of the input floating-point number.


---
### make\_q3\_quants<!-- {{#callable:make_q3_quants}} -->
The `make_q3_quants` function computes quantization levels for an array of floating-point values, optionally optimizing for root mean square error.
- **Inputs**:
    - `n`: The number of elements in the input array `x`.
    - `nmax`: The maximum quantization level.
    - `x`: A pointer to an array of floating-point values to be quantized.
    - `L`: A pointer to an array where the quantization levels will be stored.
    - `do_rmse`: A boolean flag indicating whether to optimize for root mean square error.
- **Control Flow**:
    - Initialize `max` and `amax` to zero.
    - Iterate through the input array `x` to find the maximum absolute value and its corresponding value.
    - If the maximum absolute value is less than a defined threshold (`GROUP_MAX_EPS`), set all elements in `L` to zero and return 0.
    - Calculate the inverse scale factor `iscale` based on the maximum value and `nmax`.
    - If `do_rmse` is true, perform an optimization loop up to 5 times to adjust quantization levels in `L` based on weighted sums.
    - In each iteration of the optimization, check if changing a quantization level improves the overall error, and update `L` accordingly.
    - After optimization, adjust the quantization levels in `L` by adding `nmax`.
    - If `do_rmse` is false, directly compute and store quantization levels in `L` without optimization.
    - Return the computed value based on whether `do_rmse` was true or false.
- **Output**: The function returns a float value representing either the average error for the quantization levels (if `do_rmse` is true) or the inverse scale factor (if `do_rmse` is false).
- **Functions called**:
    - [`nearest_int`](#nearest_int)


---
### make\_qkx1\_quants<!-- {{#callable:make_qkx1_quants}} -->
The `make_qkx1_quants` function quantizes an array of floating-point values into discrete levels based on specified parameters.
- **Inputs**:
    - `n`: The number of elements in the input array `x`.
    - `nmax`: The maximum quantization level.
    - `x`: A pointer to an array of floating-point values to be quantized.
    - `L`: A pointer to an array where the quantized levels will be stored.
    - `the_min`: A pointer to a float where the minimum value will be stored.
    - `ntry`: The number of iterations to attempt for refining the quantization.
    - `alpha`: A smoothing factor used to update the minimum value.
- **Control Flow**:
    - Initialize `min` and `max` with the first element of `x` and iterate through the array to find the overall minimum and maximum values.
    - If `max` equals `min`, set all elements in `L` to 0, set `the_min` to 0, and return 0.
    - Adjust `min` to 0 if it is greater than 0, then calculate the initial `iscale` and `scale` based on the range of values.
    - Enter a loop that runs for `ntry` iterations to refine the quantization levels.
    - Within the loop, calculate the quantized level for each element in `x`, updating `L` and checking if any changes occurred.
    - Compute the new `scale` based on the quantized levels and update `min` using the weighted average of the previous `min` and the computed sum.
    - If no changes were made to `L`, exit the loop early.
    - Store the negative of the final `min` in `the_min` and return the final `scale`.
- **Output**: The function returns a float representing the final scale used for quantization, and updates `the_min` with the negative of the computed minimum value.
- **Functions called**:
    - [`nearest_int`](#nearest_int)


---
### get\_scale\_min\_k4<!-- {{#callable:get_scale_min_k4}} -->
Extracts and computes scaled values from an input array based on the index provided.
- **Inputs**:
    - `j`: An integer index used to determine which elements to access in the input array.
    - `q`: A pointer to a constant array of 8-bit unsigned integers from which values are read.
    - `d`: A pointer to an 8-bit unsigned integer where the computed value will be stored.
    - `m`: A pointer to an 8-bit unsigned integer where the computed value will be stored.
- **Control Flow**:
    - If the index `j` is less than 4, the function extracts the lower 6 bits of `q[j]` and `q[j + 4]` and stores them in `d` and `m` respectively.
    - If `j` is 4 or greater, it computes `d` and `m` using bitwise operations on `q[j + 4]` and `q[j - 4]`, manipulating specific bits to form the final values.
- **Output**: The function does not return a value; instead, it modifies the values pointed to by `d` and `m` based on the computations performed.


---
### quantize\_row\_q2\_K<!-- {{#callable:quantize_row_q2_K}} -->
This function quantizes a row of data using a reference quantization function.
- **Inputs**:
    - `x`: A pointer to an array of floats representing the data to be quantized.
    - `vy`: A pointer to a memory location where the quantized output will be stored.
    - `k`: An integer representing the quantization level or parameter.
- **Control Flow**:
    - The function calls [`quantize_row_q2_K_ref`](../ggml-quants.c.driver.md#quantize_row_q2_K_ref) with the provided inputs.
    - No additional logic or control structures are present in this function.
- **Output**: The function does not return a value; instead, it modifies the output stored at the location pointed to by `vy`.
- **Functions called**:
    - [`quantize_row_q2_K_ref`](../ggml-quants.c.driver.md#quantize_row_q2_K_ref)


---
### quantize\_row\_q3\_K<!-- {{#callable:quantize_row_q3_K}} -->
Calls the [`quantize_row_q3_K_ref`](../ggml-quants.c.driver.md#quantize_row_q3_K_ref) function to quantize a row of data.
- **Inputs**:
    - `x`: A pointer to an array of floats representing the data to be quantized.
    - `vy`: A pointer to a memory location where the quantized output will be stored.
    - `k`: An integer representing a parameter that may influence the quantization process.
- **Control Flow**:
    - The function directly calls [`quantize_row_q3_K_ref`](../ggml-quants.c.driver.md#quantize_row_q3_K_ref) with the provided arguments.
    - No additional logic or control structures are present in this function.
- **Output**: The function does not return a value; instead, it modifies the output in the memory location pointed to by `vy`.
- **Functions called**:
    - [`quantize_row_q3_K_ref`](../ggml-quants.c.driver.md#quantize_row_q3_K_ref)


---
### quantize\_row\_q4\_K<!-- {{#callable:quantize_row_q4_K}} -->
The `quantize_row_q4_K` function quantizes a row of floating-point values into a specified format.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that need to be quantized.
    - `vy`: A pointer to a memory location where the quantized output will be stored.
    - `k`: An integer representing the number of elements to quantize, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K` to ensure valid input.
    - It then casts the output pointer `vy` to a pointer of type `block_q4_K` for proper storage of the quantized data.
    - Finally, it calls the [`quantize_row_q4_K_ref`](../ggml-quants.c.driver.md#quantize_row_q4_K_ref) function, passing the input array `x`, the output pointer `y`, and the count `k` to perform the actual quantization.
- **Output**: The function does not return a value; instead, it modifies the memory pointed to by `vy` to store the quantized representation of the input data.
- **Functions called**:
    - [`quantize_row_q4_K_ref`](../ggml-quants.c.driver.md#quantize_row_q4_K_ref)


---
### quantize\_row\_q5\_K<!-- {{#callable:quantize_row_q5_K}} -->
Quantizes a row of floating-point values into a specified format using a reference quantization function.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that need to be quantized.
    - `vy`: A pointer to a memory location where the quantized output will be stored.
    - `k`: An integer representing the number of elements to quantize, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K`, ensuring valid input for quantization.
    - It then casts the output pointer `vy` to a pointer of type `block_q5_K` for proper storage of the quantized data.
    - Finally, it calls the [`quantize_row_q5_K_ref`](../ggml-quants.c.driver.md#quantize_row_q5_K_ref) function, passing the input array `x`, the output pointer `y`, and the count `k` to perform the actual quantization.
- **Output**: The function does not return a value; instead, it modifies the memory pointed to by `vy` to store the quantized representation of the input data.
- **Functions called**:
    - [`quantize_row_q5_K_ref`](../ggml-quants.c.driver.md#quantize_row_q5_K_ref)


---
### quantize\_row\_q6\_K<!-- {{#callable:quantize_row_q6_K}} -->
Quantizes a row of floating-point values into a specified format using a reference quantization function.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that are to be quantized.
    - `vy`: A pointer to a memory location where the quantized output will be stored.
    - `k`: An integer representing the number of elements to quantize, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K`, ensuring valid input for quantization.
    - It then casts the output pointer `vy` to a pointer of type `block_q6_K` for proper storage of the quantized data.
    - Finally, it calls the [`quantize_row_q6_K_ref`](../ggml-quants.c.driver.md#quantize_row_q6_K_ref) function, passing the input array `x`, the output pointer `y`, and the count `k` to perform the actual quantization.
- **Output**: The function does not return a value; instead, it modifies the memory pointed to by `vy` to store the quantized representation of the input data.
- **Functions called**:
    - [`quantize_row_q6_K_ref`](../ggml-quants.c.driver.md#quantize_row_q6_K_ref)


---
### quantize\_row\_tq1\_0<!-- {{#callable:quantize_row_tq1_0}} -->
Quantizes a row of floating-point values into a specified format.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that need to be quantized.
    - `vy`: A pointer to a memory location where the quantized output will be stored.
    - `k`: An integer representing the number of elements to quantize, which must be a multiple of QK_K.
- **Control Flow**:
    - The function begins by asserting that the value of k is a multiple of QK_K to ensure valid input.
    - It then casts the output pointer vy to a specific type (block_tq1_0) for storing the quantized data.
    - Finally, it calls the [`quantize_row_tq1_0_ref`](../ggml-quants.c.driver.md#quantize_row_tq1_0_ref) function to perform the actual quantization process using the input array x and the output pointer y.
- **Output**: The function does not return a value; instead, it modifies the memory pointed to by vy to store the quantized representation of the input data.
- **Functions called**:
    - [`quantize_row_tq1_0_ref`](../ggml-quants.c.driver.md#quantize_row_tq1_0_ref)


---
### quantize\_row\_tq2\_0<!-- {{#callable:quantize_row_tq2_0}} -->
Quantizes a row of floating-point values into a specified format.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that need to be quantized.
    - `vy`: A pointer to a memory location where the quantized output will be stored.
    - `k`: An integer representing the number of elements to quantize, which must be a multiple of `QK_K`.
- **Control Flow**:
    - Checks that `k` is a multiple of `QK_K` using an assertion to ensure valid input.
    - Assigns the output pointer `vy` to a typed pointer `y` of type `block_tq2_0` for further processing.
    - Calls the function [`quantize_row_tq2_0_ref`](../ggml-quants.c.driver.md#quantize_row_tq2_0_ref) to perform the actual quantization of the input array `x` into the output `y`.
- **Output**: The function does not return a value; instead, it modifies the memory pointed to by `vy` to store the quantized representation of the input data.
- **Functions called**:
    - [`quantize_row_tq2_0_ref`](../ggml-quants.c.driver.md#quantize_row_tq2_0_ref)


---
### quantize\_row\_q8\_K<!-- {{#callable:quantize_row_q8_K}} -->
The `quantize_row_q8_K` function quantizes a row of floating-point values into a specific format using SIMD operations.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized.
    - `y`: A pointer to a destination where the quantized data will be stored.
    - `k`: An integer representing the number of elements in the input array, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function checks if `k` is a multiple of `QK_K` and calculates the number of blocks `nb` to process.
    - For each block, it initializes `min_vec` and `max_vec` to find the minimum and maximum values in the block using SIMD operations.
    - If the maximum absolute value (`amax`) is zero, it sets the corresponding output to zero and continues to the next block.
    - If `amax` is non-zero, it calculates the scale factor and processes 16 elements at a time, quantizing them and storing the results.
    - It computes the sums of the quantized values for further analysis and stores them in the output structure.
- **Output**: The function outputs quantized data in the `y` pointer, along with computed sums for each block.
- **Functions called**:
    - [`quantize_row_q8_K_ref`](../ggml-quants.c.driver.md#quantize_row_q8_K_ref)


---
### get\_scale\_shuffle\_q3k<!-- {{#callable:get_scale_shuffle_q3k}} -->
The `get_scale_shuffle_q3k` function retrieves a 256-bit integer vector from a predefined shuffle array based on the provided index.
- **Inputs**:
    - `i`: An integer index used to access the `k_shuffle` array.
- **Control Flow**:
    - The function defines a static array `k_shuffle` containing 128 elements that specify the shuffle pattern.
    - It calculates the address of the desired element in the `k_shuffle` array by adding the input index `i` to the base address of the array.
    - The function then uses the `__lasx_xvld` intrinsic to load a 256-bit integer vector from the calculated address.
- **Output**: The function returns a `__m256i` type value, which is a 256-bit integer vector containing the shuffled values from the `k_shuffle` array.


---
### get\_scale\_shuffle\_k4<!-- {{#callable:get_scale_shuffle_k4}} -->
The `get_scale_shuffle_k4` function retrieves a 256-bit integer vector from a predefined shuffle table based on the input index.
- **Inputs**:
    - `i`: An integer index used to access the corresponding entry in the `k_shuffle` array.
- **Control Flow**:
    - The function defines a static array `k_shuffle` containing 256 predefined values for shuffling.
    - It calculates the address of the desired entry in the `k_shuffle` array by adding the input index `i` to the base address of the array.
    - The function uses the `__lasx_xvld` intrinsic to load a 256-bit integer vector from the calculated address.
- **Output**: The function returns a `__m256i` type value, which is a 256-bit integer vector containing the shuffled values from the `k_shuffle` array.


---
### get\_scale\_shuffle<!-- {{#callable:get_scale_shuffle}} -->
The `get_scale_shuffle` function retrieves a 128-bit integer vector from a predefined shuffle table based on the input index.
- **Inputs**:
    - `i`: An integer index used to access the shuffle table.
- **Control Flow**:
    - The function defines a static array `k_shuffle` containing 128 predefined values for shuffling.
    - It uses the input index `i` to calculate the address of the desired element in the `k_shuffle` array.
    - The function then loads a 128-bit integer vector from the calculated address using the `__lsx_vld` intrinsic.
- **Output**: The function returns a `__m128i` type value, which is a 128-bit integer vector containing the shuffled values from the `k_shuffle` array.


---
### ggml\_vec\_dot\_q4\_0\_q8\_0<!-- {{#callable:ggml_vec_dot_q4_0_q8_0}} -->
Calculates the dot product of two quantized vectors with optional scaling.
- **Inputs**:
    - `n`: The total number of elements in the vectors, must be a multiple of `QK8_0`.
    - `s`: Pointer to the output array where the result of the dot product will be stored.
    - `bs`: The stride for the output array, indicating how many elements to skip for the next result.
    - `vx`: Pointer to the first input vector, which is quantized using 4 bits per element.
    - `bx`: The byte size of each block in the first input vector.
    - `vy`: Pointer to the second input vector, which is quantized using 8 bits per element.
    - `by`: The byte size of each block in the second input vector.
    - `nrc`: Indicates the number of result components to compute; can be 1 or 2 depending on the architecture.
- **Control Flow**:
    - The function begins by asserting that `n` is a multiple of `QK8_0` and checks the value of `nrc` based on the architecture.
    - If the architecture supports ARM's matrix multiplication for 8-bit integers and `nrc` is 2, it processes two blocks of the input vectors in parallel using SIMD instructions.
    - For each block, it extracts the quantized values, converts them from 4-bit to 8-bit, and performs the dot product while applying scaling factors derived from the input vectors.
    - If `nrc` is 1 or if the architecture does not support the special case, it falls back to a standard loop that processes the input vectors in pairs, accumulating the results into a floating-point sum.
    - The final result is stored in the output array `s`.
- **Output**: The function outputs the computed dot product of the two input vectors, scaled by their respective scaling factors, into the array pointed to by `s`.
- **Functions called**:
    - [`ggml_cpu_get_sve_cnt`](ggml-cpu.c.driver.md#ggml_cpu_get_sve_cnt)
    - [`ggml_vdotq_s32`](ggml-cpu-impl.h.driver.md#ggml_vdotq_s32)
    - [`vaddvq_f32`](ggml-cpu-impl.h.driver.md#vaddvq_f32)
    - [`bytes_from_nibbles_32`](#bytes_from_nibbles_32)
    - [`mul_sum_i8_pairs_float`](#mul_sum_i8_pairs_float)
    - [`hsum_float_8`](#hsum_float_8)
    - [`mul_add_epi8_sse`](#mul_add_epi8_sse)
    - [`sum_i16_pairs_float`](#sum_i16_pairs_float)
    - [`quad_fp16_delta_float`](#quad_fp16_delta_float)
    - [`mul_sum_i8_pairs`](#mul_sum_i8_pairs)
    - [`hsum_float_4x4`](#hsum_float_4x4)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)


---
### ggml\_vec\_dot\_q4\_1\_q8\_1<!-- {{#callable:ggml_vec_dot_q4_1_q8_1}} -->
Calculates the dot product of two quantized vectors with specific optimizations for different architectures.
- **Inputs**:
    - `n`: The total number of elements in the vectors, which must be a multiple of `QK8_1`.
    - `s`: Pointer to the output array where the result of the dot product will be stored.
    - `bs`: The stride for the output array, indicating how many elements to skip for the next output.
    - `vx`: Pointer to the first input vector, which is quantized using the `block_q4_1` format.
    - `bx`: The byte offset for the first input vector.
    - `vy`: Pointer to the second input vector, which is quantized using the `block_q8_1` format.
    - `by`: The byte offset for the second input vector.
    - `nrc`: An integer indicating the number of result components to compute, which can be 1 or 2 depending on the architecture.
- **Control Flow**:
    - The function begins by asserting that `n` is a multiple of `QK8_1` and checks the value of `nrc` based on the architecture.
    - If the architecture supports ARM's matrix multiplication for 8-bit integers and `nrc` is 2, it processes two sets of quantized vectors in parallel using SIMD instructions.
    - For each block of the input vectors, it computes intermediate sums and accumulates results using vectorized operations.
    - If `nrc` is not 2 or the architecture does not support it, the function falls back to a scalar implementation that processes the vectors sequentially.
    - In the scalar implementation, it computes the dot product by iterating through the elements of the input vectors and accumulating the results.
- **Output**: The result of the dot product is stored in the output array pointed to by `s`, with the final sum being the accumulated value from the computations.
- **Functions called**:
    - [`ggml_vdotq_s32`](ggml-cpu-impl.h.driver.md#ggml_vdotq_s32)
    - [`vaddvq_f32`](ggml-cpu-impl.h.driver.md#vaddvq_f32)
    - [`bytes_from_nibbles_32`](#bytes_from_nibbles_32)
    - [`mul_sum_us8_pairs_float`](#mul_sum_us8_pairs_float)
    - [`hsum_float_8`](#hsum_float_8)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)
    - [`ggml_vec_dot`](ggml-cpu-impl.h.driver.md#ggml_vec_dot)


---
### ggml\_vec\_dot\_q5\_0\_q8\_0<!-- {{#callable:ggml_vec_dot_q5_0_q8_0}} -->
Calculates the dot product of two quantized vectors with specific formats and stores the result in a provided output variable.
- **Inputs**:
    - `n`: The total number of elements in the vectors, which must be a multiple of `QK8_0`.
    - `s`: A pointer to a float where the result of the dot product will be stored.
    - `bs`: The block size, which is unused in the function.
    - `vx`: A pointer to the first input vector in the `block_q5_0` format.
    - `bx`: The block size for the first vector, which is unused in the function.
    - `vy`: A pointer to the second input vector in the `block_q8_0` format.
    - `by`: The block size for the second vector, which is unused in the function.
    - `nrc`: An integer that is expected to be 1, which is unused in the function.
- **Control Flow**:
    - The function begins by asserting that `n` is a multiple of `QK8_0`, `qk` is equal to `QK5_0`, and `nrc` is 1.
    - It initializes an accumulator for the sum and sets up a loop to process the input vectors in blocks.
    - Depending on the compilation flags, it uses SIMD instructions to optimize the dot product calculation.
    - For each block, it extracts the necessary bits from the input vectors, performs the dot product, and accumulates the results.
    - After processing all blocks, it handles any remaining elements in a standard loop.
    - Finally, it stores the computed sum in the output variable `s`.
- **Output**: The function outputs the computed dot product as a float value stored in the variable pointed to by `s`.
- **Functions called**:
    - [`ggml_vdotq_s32`](ggml-cpu-impl.h.driver.md#ggml_vdotq_s32)
    - [`vaddvq_f32`](ggml-cpu-impl.h.driver.md#vaddvq_f32)
    - [`bytes_from_nibbles_32`](#bytes_from_nibbles_32)
    - [`bytes_from_bits_32`](#bytes_from_bits_32)
    - [`mul_sum_i8_pairs_float`](#mul_sum_i8_pairs_float)
    - [`hsum_float_8`](#hsum_float_8)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)


---
### ggml\_vec\_dot\_q5\_1\_q8\_1<!-- {{#callable:ggml_vec_dot_q5_1_q8_1}} -->
Calculates the dot product of two quantized vectors with specific formats.
- **Inputs**:
    - `n`: The number of elements in the vectors, must be a multiple of `QK8_1`.
    - `s`: Pointer to a float where the result of the dot product will be stored.
    - `bs`: Size of the block for the first vector, not used in the function.
    - `vx`: Pointer to the first quantized vector of type `block_q5_1`.
    - `bx`: Size of the block for the second vector, not used in the function.
    - `vy`: Pointer to the second quantized vector of type `block_q8_1`.
    - `by`: Size of the block for the second vector, not used in the function.
    - `nrc`: An integer that is expected to be 1, not used in the function.
- **Control Flow**:
    - The function begins by asserting that `n` is divisible by `QK8_1` and that `nrc` equals 1.
    - It initializes variables for accumulation and iterates over the number of blocks in the input vectors.
    - Depending on the compilation flags, it uses SIMD instructions (like ARM NEON, AVX, etc.) to perform vectorized operations for efficiency.
    - For each block, it extracts and processes the quantized values, computes partial sums, and accumulates results.
    - After processing all blocks, it computes the final sum and stores it in the provided pointer.
- **Output**: The function outputs the computed dot product as a float in the location pointed to by `s`.
- **Functions called**:
    - [`ggml_vdotq_s32`](ggml-cpu-impl.h.driver.md#ggml_vdotq_s32)
    - [`vaddvq_f32`](ggml-cpu-impl.h.driver.md#vaddvq_f32)
    - [`bytes_from_nibbles_32`](#bytes_from_nibbles_32)
    - [`bytes_from_bits_32`](#bytes_from_bits_32)
    - [`mul_sum_us8_pairs_float`](#mul_sum_us8_pairs_float)
    - [`hsum_float_8`](#hsum_float_8)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)


---
### ggml\_vec\_dot\_q8\_0\_q8\_0<!-- {{#callable:ggml_vec_dot_q8_0_q8_0}} -->
Computes the dot product of two quantized vectors with optional scaling.
- **Inputs**:
    - `n`: The total number of elements in the vectors.
    - `s`: Pointer to the output float array where the result will be stored.
    - `bs`: The stride for the output array.
    - `vx`: Pointer to the first input vector, which is quantized.
    - `bx`: The stride for the first input vector.
    - `vy`: Pointer to the second input vector, which is quantized.
    - `by`: The stride for the second input vector.
    - `nrc`: Number of result components to compute, which can be 1 or 2 depending on the architecture.
- **Control Flow**:
    - The function begins by asserting that the input size `n` is a multiple of `QK8_0`.
    - It checks the value of `nrc` to determine if it should compute one or two result components.
    - If the architecture supports ARM's matrix multiplication for 8-bit integers and `nrc` is 2, it computes the dot product using SIMD instructions for efficiency.
    - For other architectures or when `nrc` is 1, it uses a different approach based on the available vector extensions (SVE, NEON, AVX, etc.) to compute the dot product.
    - The function accumulates the results of the dot products, applying scaling factors derived from the input vectors.
    - Finally, it stores the computed sum in the output array `s`.
- **Output**: The function outputs the computed dot product of the two input vectors, scaled appropriately, into the provided output array.
- **Functions called**:
    - [`ggml_cpu_get_sve_cnt`](ggml-cpu.c.driver.md#ggml_cpu_get_sve_cnt)
    - [`ggml_vdotq_s32`](ggml-cpu-impl.h.driver.md#ggml_vdotq_s32)
    - [`vaddvq_f32`](ggml-cpu-impl.h.driver.md#vaddvq_f32)
    - [`mul_sum_i8_pairs_float`](#mul_sum_i8_pairs_float)
    - [`hsum_float_8`](#hsum_float_8)
    - [`mul_sum_i8_quad_float`](#mul_sum_i8_quad_float)
    - [`quad_fp16_delta_float`](#quad_fp16_delta_float)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)
    - [`ggml_vec_dot`](ggml-cpu-impl.h.driver.md#ggml_vec_dot)


---
### ggml\_vec\_dot\_tq1\_0\_q8\_K<!-- {{#callable:ggml_vec_dot_tq1_0_q8_K}} -->
Calculates the dot product of two quantized vectors using optimized SIMD instructions.
- **Inputs**:
    - `n`: The number of elements in the vectors.
    - `s`: Pointer to a float where the result of the dot product will be stored.
    - `bs`: Block size, which is unused in the function.
    - `vx`: Pointer to the first vector, which is a quantized structure.
    - `bx`: Block size for the first vector, which is unused in the function.
    - `vy`: Pointer to the second vector, which is also a quantized structure.
    - `by`: Block size for the second vector, which is unused in the function.
    - `nrc`: An integer that must be equal to 1, otherwise an assertion fails.
- **Control Flow**:
    - The function begins by asserting that `nrc` is equal to 1 and marks several parameters as unused.
    - It calculates the number of blocks `nb` based on the input size `n` divided by a constant `QK_K`.
    - Depending on the defined architecture (ARM NEON or AVX2), it initializes the sum and processes the input vectors using SIMD instructions.
    - For each block, it loads the quantized values from both vectors, applies necessary transformations and multiplications, and accumulates the results.
    - Finally, it computes the final dot product and stores it in the output pointer `s`.
- **Output**: The function outputs the computed dot product as a float value stored at the address pointed to by `s`.
- **Functions called**:
    - [`vaddvq_s32`](ggml-cpu-impl.h.driver.md#vaddvq_s32)
    - [`vaddlvq_s16`](ggml-cpu-impl.h.driver.md#vaddlvq_s16)
    - [`hsum_float_8`](#hsum_float_8)


---
### ggml\_vec\_dot\_tq2\_0\_q8\_K<!-- {{#callable:ggml_vec_dot_tq2_0_q8_K}} -->
Calculates the dot product of two quantized vectors using optimized SIMD instructions.
- **Inputs**:
    - `n`: The number of elements in the vectors.
    - `s`: Pointer to a float where the result of the dot product will be stored.
    - `bs`: Block size, which is unused in the function.
    - `vx`: Pointer to the first vector, which is a block of quantized data.
    - `bx`: Block size for the first vector, which is unused in the function.
    - `vy`: Pointer to the second vector, which is also a block of quantized data.
    - `by`: Block size for the second vector, which is unused in the function.
    - `nrc`: An integer that must be equal to 1, otherwise the function asserts.
- **Control Flow**:
    - The function begins by asserting that nrc is equal to 1 and marks several parameters as unused.
    - It calculates the number of blocks (nb) based on the input size n.
    - Depending on the architecture (ARM NEON or AVX2), it initializes variables for accumulating the dot product.
    - For each block, it processes the quantized data in chunks, loading data from the input vectors and performing dot product calculations using SIMD instructions.
    - The results from the dot products are accumulated and adjusted using additional sums from the input vectors.
    - Finally, the accumulated result is stored in the location pointed to by s.
- **Output**: The function outputs the computed dot product of the two quantized vectors into the float pointed to by s.
- **Functions called**:
    - [`vaddvq_s32`](ggml-cpu-impl.h.driver.md#vaddvq_s32)
    - [`vaddlvq_s16`](ggml-cpu-impl.h.driver.md#vaddlvq_s16)
    - [`hsum_float_8`](#hsum_float_8)


---
### ggml\_vec\_dot\_q2\_K\_q8\_K<!-- {{#callable:ggml_vec_dot_q2_K_q8_K}} -->
Calculates the dot product of two quantized vectors with specific scaling and summation logic.
- **Inputs**:
    - `n`: The number of elements in the vectors.
    - `s`: Pointer to a float where the result of the dot product will be stored.
    - `bs`: Block size, which is not used in the function.
    - `vx`: Pointer to the first vector of type `block_q2_K`.
    - `bx`: Block size for the first vector, which is not used in the function.
    - `vy`: Pointer to the second vector of type `block_q8_K`.
    - `by`: Block size for the second vector, which is not used in the function.
    - `nrc`: An integer that must be equal to 1, otherwise an assertion fails.
- **Control Flow**:
    - The function begins by asserting that `nrc` is equal to 1.
    - It initializes pointers to the input vectors and calculates the number of blocks `nb` based on `n`.
    - The function checks the vector length and executes different code paths based on whether it is 128, 256, or 512.
    - For each block, it computes a dot product using vectorized operations, handling the quantized values and their scales.
    - The accumulated results are stored in `acc_sum`, which is then summed and stored in the output variable `s`.
- **Output**: The function outputs the computed dot product as a float value stored at the address pointed to by `s`.
- **Functions called**:
    - [`ggml_vld1q_s16_x2`](ggml-cpu-impl.h.driver.md#ggml_vld1q_s16_x2)
    - [`vaddvq_s32`](ggml-cpu-impl.h.driver.md#vaddvq_s32)
    - [`ggml_vld1q_u8_x2`](ggml-cpu-impl.h.driver.md#ggml_vld1q_u8_x2)
    - [`ggml_vld1q_s8_x2`](ggml-cpu-impl.h.driver.md#ggml_vld1q_s8_x2)
    - [`get_scale_shuffle_q3k`](#get_scale_shuffle_q3k)
    - [`hsum_float_8`](#hsum_float_8)
    - [`lasx_ext8_16`](#lasx_ext8_16)
    - [`lasx_madd_h`](#lasx_madd_h)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)
    - [`lasx_madd_h_b`](#lasx_madd_h_b)
    - [`lasx_xvrepl128vei_h`](#lasx_xvrepl128vei_h)


---
### ggml\_vec\_dot\_q3\_K\_q8\_K<!-- {{#callable:ggml_vec_dot_q3_K_q8_K}} -->
Calculates the dot product of two quantized vectors with scaling.
- **Inputs**:
    - `n`: The number of elements in the vectors, must be a multiple of QK_K.
    - `s`: Pointer to a float where the result of the dot product will be stored.
    - `bs`: Block size, currently unused.
    - `vx`: Pointer to the first quantized vector of type `block_q3_K`.
    - `bx`: Block size for the first vector, currently unused.
    - `vy`: Pointer to the second quantized vector of type `block_q8_K`.
    - `by`: Block size for the second vector, currently unused.
    - `nrc`: Number of return codes, must be 1.
- **Control Flow**:
    - The function begins by asserting that `n` is a multiple of `QK_K` and that `nrc` equals 1.
    - It initializes masks and prepares to process the quantized vectors using SIMD instructions based on the architecture.
    - For each block of the input vectors, it calculates the dot product using the quantized values and their corresponding scales.
    - The results are accumulated into a sum, which is then scaled by the corresponding floating-point values from the input vectors.
    - Finally, the computed sum is stored in the output pointer `s`.
- **Output**: The function outputs the computed dot product as a float in the location pointed to by `s`.
- **Functions called**:
    - [`ggml_vld1q_u8_x2`](ggml-cpu-impl.h.driver.md#ggml_vld1q_u8_x2)
    - [`ggml_vld1q_s8_x4`](ggml-cpu-impl.h.driver.md#ggml_vld1q_s8_x4)
    - [`vaddvq_s32`](ggml-cpu-impl.h.driver.md#vaddvq_s32)
    - [`ggml_vdotq_s32`](ggml-cpu-impl.h.driver.md#ggml_vdotq_s32)
    - [`get_scale_shuffle_q3k`](#get_scale_shuffle_q3k)
    - [`hsum_float_8`](#hsum_float_8)
    - [`lsx_set_w`](#lsx_set_w)
    - [`lasx_ext8_16`](#lasx_ext8_16)
    - [`lasx_xvandi_b_bit`](#lasx_xvandi_b_bit)
    - [`lasx_madd_h_b`](#lasx_madd_h_b)
    - [`lasx_madd_h`](#lasx_madd_h)
    - [`lasx_xvrepl128vei_h`](#lasx_xvrepl128vei_h)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)
    - [`ggml_vec_dot`](ggml-cpu-impl.h.driver.md#ggml_vec_dot)


---
### ggml\_vec\_dot\_q4\_K\_q8\_K<!-- {{#callable:ggml_vec_dot_q4_K_q8_K}} -->
Calculates the dot product of two quantized vectors with specific optimizations for different architectures.
- **Inputs**:
    - `n`: The number of elements in the vectors, which must be a multiple of `QK_K`.
    - `s`: Pointer to the output array where the result of the dot product will be stored.
    - `bs`: The stride for the output array, indicating how far apart the results should be stored.
    - `vx`: Pointer to the first input vector, which is quantized using `block_q4_K` format.
    - `bx`: The byte offset for the first input vector.
    - `vy`: Pointer to the second input vector, which is quantized using `block_q8_K` format.
    - `by`: The byte offset for the second input vector.
    - `nrc`: An integer indicating the number of return channels, which can be 1 or 2 depending on the architecture.
- **Control Flow**:
    - The function begins by asserting that `n` is a multiple of `QK_K` and checks the value of `nrc` based on the architecture.
    - It initializes pointers to the input vectors and calculates the number of blocks `nb` based on `n`.
    - Depending on the architecture (ARM with INT8 support, SVE, NEON, etc.), it processes the input vectors using different optimized loops.
    - For each block, it decodes the scales and mins from the input vectors and performs the dot product calculations.
    - The results are accumulated in a temporary variable, which is then adjusted by applying scales and biases.
    - Finally, the computed result is stored in the output array `s`.
- **Output**: The function outputs the computed dot product result in the array pointed to by `s`, which is a floating-point representation of the dot product of the two input vectors.
- **Functions called**:
    - [`vpaddq_s16`](ggml-cpu-impl.h.driver.md#vpaddq_s16)
    - [`vaddvq_s32`](ggml-cpu-impl.h.driver.md#vaddvq_s32)
    - [`ggml_cpu_get_sve_cnt`](ggml-cpu.c.driver.md#ggml_cpu_get_sve_cnt)
    - [`ggml_vld1q_u8_x2`](ggml-cpu-impl.h.driver.md#ggml_vld1q_u8_x2)
    - [`ggml_vld1q_s8_x2`](ggml-cpu-impl.h.driver.md#ggml_vld1q_s8_x2)
    - [`ggml_vdotq_s32`](ggml-cpu-impl.h.driver.md#ggml_vdotq_s32)
    - [`get_scale_shuffle_k4`](#get_scale_shuffle_k4)
    - [`hsum_float_8`](#hsum_float_8)
    - [`lsx_set_w`](#lsx_set_w)
    - [`lsx_hadd_h`](#lsx_hadd_h)
    - [`lasx_extracti128`](#lasx_extracti128)
    - [`lsx_madd_h`](#lsx_madd_h)
    - [`__lsx_vreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lsx_vreplfr2vr_s)
    - [`lasx_insertf128`](#lasx_insertf128)
    - [`lasx_xvrepl128vei_h`](#lasx_xvrepl128vei_h)
    - [`lasx_madd_h_b`](#lasx_madd_h_b)
    - [`lasx_madd_h`](#lasx_madd_h)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)
    - [`vec_padd_s16`](ggml-cpu-impl.h.driver.md#vec_padd_s16)
    - [`ggml_vec_dot`](ggml-cpu-impl.h.driver.md#ggml_vec_dot)


---
### ggml\_vec\_dot\_q5\_K\_q8\_K<!-- {{#callable:ggml_vec_dot_q5_K_q8_K}} -->
Calculates the dot product of two quantized vectors with scaling and bias adjustments.
- **Inputs**:
    - `n`: The total number of elements in the vectors, which must be a multiple of `QK_K`.
    - `s`: A pointer to a float where the result of the dot product will be stored.
    - `bs`: The block size, which is unused in the function.
    - `vx`: A pointer to the first quantized vector of type `block_q5_K`.
    - `bx`: The block size for the first vector, which is unused in the function.
    - `vy`: A pointer to the second quantized vector of type `block_q8_K`.
    - `by`: The block size for the second vector, which is unused in the function.
    - `nrc`: An integer that must be equal to 1, which is unused in the function.
- **Control Flow**:
    - The function begins by asserting that `n` is a multiple of `QK_K` and that `nrc` is equal to 1.
    - It initializes pointers to the quantized vectors and calculates the number of blocks `nb` based on `n`.
    - Depending on the compilation flags, it uses different SIMD instructions (e.g., ARM NEON, AVX2, etc.) to perform the dot product efficiently.
    - For each block, it retrieves the necessary data, processes the scales and minimums, and computes the dot product using the quantized values.
    - The results are accumulated into a floating-point sum, which is adjusted by the scaling factors and minimums.
    - Finally, the computed sum is stored in the location pointed to by `s`.
- **Output**: The function outputs the computed dot product as a float, stored in the variable pointed to by `s`.
- **Functions called**:
    - [`vpaddq_s16`](ggml-cpu-impl.h.driver.md#vpaddq_s16)
    - [`vaddvq_s32`](ggml-cpu-impl.h.driver.md#vaddvq_s32)
    - [`ggml_vld1q_u8_x2`](ggml-cpu-impl.h.driver.md#ggml_vld1q_u8_x2)
    - [`ggml_vld1q_s8_x4`](ggml-cpu-impl.h.driver.md#ggml_vld1q_s8_x4)
    - [`ggml_vdotq_s32`](ggml-cpu-impl.h.driver.md#ggml_vdotq_s32)
    - [`get_scale_shuffle_k4`](#get_scale_shuffle_k4)
    - [`hsum_float_8`](#hsum_float_8)
    - [`lsx_set_w`](#lsx_set_w)
    - [`lsx_hadd_h`](#lsx_hadd_h)
    - [`lasx_extracti128`](#lasx_extracti128)
    - [`lsx_madd_h`](#lsx_madd_h)
    - [`__lsx_vreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lsx_vreplfr2vr_s)
    - [`lasx_insertf128`](#lasx_insertf128)
    - [`lasx_xvrepl128vei_h`](#lasx_xvrepl128vei_h)
    - [`lasx_xvandi_b_bit`](#lasx_xvandi_b_bit)
    - [`lasx_madd_h_b`](#lasx_madd_h_b)
    - [`lasx_madd_h`](#lasx_madd_h)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)
    - [`vec_padd_s16`](ggml-cpu-impl.h.driver.md#vec_padd_s16)
    - [`ggml_vec_dot`](ggml-cpu-impl.h.driver.md#ggml_vec_dot)


---
### ggml\_vec\_dot\_q6\_K\_q8\_K<!-- {{#callable:ggml_vec_dot_q6_K_q8_K}} -->
Calculates the dot product of two quantized vectors with optional scaling and bias adjustments.
- **Inputs**:
    - `n`: The number of elements in the vectors, which must be a multiple of `QK_K`.
    - `s`: Pointer to the output array where the result of the dot product will be stored.
    - `bs`: The stride for the output array, indicating how many elements to skip for the next result.
    - `vx`: Pointer to the first input vector, which is quantized using `block_q6_K` format.
    - `bx`: The byte offset for the first input vector.
    - `vy`: Pointer to the second input vector, which is quantized using `block_q8_K` format.
    - `by`: The byte offset for the second input vector.
    - `nrc`: Indicates the number of result components to compute, which can be 1 or 2 depending on the architecture.
- **Control Flow**:
    - The function begins by asserting that `n` is a multiple of `QK_K` and checks the value of `nrc` based on the architecture.
    - It initializes pointers to the quantized input vectors and calculates the number of blocks `nb` based on `n`.
    - If `nrc` is 2, it processes two sets of quantized vectors in parallel, de-quantizing them into 8-bit integers.
    - For each block, it computes the dot product using SIMD operations, accumulating results into a temporary sum.
    - After processing all blocks, it adjusts the accumulated sum by applying bias and scaling factors.
    - Finally, the result is stored in the output array `s`.
- **Output**: The function outputs the computed dot product result in the array pointed to by `s`, with the results potentially adjusted by scaling and bias.
- **Functions called**:
    - [`vaddvq_s32`](ggml-cpu-impl.h.driver.md#vaddvq_s32)
    - [`ggml_cpu_get_sve_cnt`](ggml-cpu.c.driver.md#ggml_cpu_get_sve_cnt)
    - [`ggml_vld1q_s16_x2`](ggml-cpu-impl.h.driver.md#ggml_vld1q_s16_x2)
    - [`ggml_vld1q_u8_x2`](ggml-cpu-impl.h.driver.md#ggml_vld1q_u8_x2)
    - [`ggml_vld1q_u8_x4`](ggml-cpu-impl.h.driver.md#ggml_vld1q_u8_x4)
    - [`ggml_vld1q_s8_x4`](ggml-cpu-impl.h.driver.md#ggml_vld1q_s8_x4)
    - [`ggml_vdotq_s32`](ggml-cpu-impl.h.driver.md#ggml_vdotq_s32)
    - [`get_scale_shuffle`](#get_scale_shuffle)
    - [`hsum_float_8`](#hsum_float_8)
    - [`lasx_ext8_16`](#lasx_ext8_16)
    - [`lasx_madd_h_b`](#lasx_madd_h_b)
    - [`lasx_madd_h`](#lasx_madd_h)
    - [`lasx_xvrepl128vei_h`](#lasx_xvrepl128vei_h)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)
    - [`ggml_vec_dot`](ggml-cpu-impl.h.driver.md#ggml_vec_dot)


---
### ggml\_vec\_dot\_iq2\_xxs\_q8\_K<!-- {{#callable:ggml_vec_dot_iq2_xxs_q8_K}} -->
Calculates the dot product of two quantized vectors with various optimizations based on the architecture.
- **Inputs**:
    - `n`: The number of elements in the vectors, which must be a multiple of `QK_K`.
    - `s`: A pointer to a float where the result of the dot product will be stored.
    - `bs`: A size_t representing the block size, which is unused in the function.
    - `vx`: A pointer to the first vector, which is expected to be of type `block_iq2_xxs`.
    - `bx`: A size_t representing the block size for the first vector, which is unused in the function.
    - `vy`: A pointer to the second vector, which is expected to be of type `block_q8_K`.
    - `by`: A size_t representing the block size for the second vector, which is unused in the function.
    - `nrc`: An integer that is expected to be 1, which is unused in the function.
- **Control Flow**:
    - The function begins by asserting that `n` is a multiple of `QK_K` and that `nrc` equals 1.
    - It initializes pointers to the input vectors and calculates the number of blocks `nb` as `n / QK_K`.
    - Depending on the defined architecture (e.g., ARM NEON, AVX2, AVX, etc.), it executes different optimized code paths for calculating the dot product.
    - In each architecture-specific block, it iterates over the number of blocks, performing calculations to accumulate the dot product using SIMD instructions.
    - The final result is scaled and stored in the output pointer `s`.
- **Output**: The function outputs the computed dot product of the two quantized vectors, scaled by a factor of 0.125 or 0.25 depending on the architecture.
- **Functions called**:
    - [`ggml_vld1q_s8_x4`](ggml-cpu-impl.h.driver.md#ggml_vld1q_s8_x4)
    - [`ggml_vdotq_s32`](ggml-cpu-impl.h.driver.md#ggml_vdotq_s32)
    - [`vaddvq_s32`](ggml-cpu-impl.h.driver.md#vaddvq_s32)
    - [`hsum_float_8`](#hsum_float_8)
    - [`lasx_set_d`](#lasx_set_d)
    - [`lasx_maddubs_h`](#lasx_maddubs_h)
    - [`lasx_madd_h`](#lasx_madd_h)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)


---
### ggml\_vec\_dot\_iq2\_xs\_q8\_K<!-- {{#callable:ggml_vec_dot_iq2_xs_q8_K}} -->
Calculates the dot product of two quantized vectors with various optimizations based on the architecture.
- **Inputs**:
    - `n`: The number of elements in the vectors, which must be a multiple of `QK_K`.
    - `s`: A pointer to a float where the result of the dot product will be stored.
    - `bs`: The block size, which is unused in the function.
    - `vx`: A pointer to the first vector's data, which is expected to be of type `block_iq2_xs`.
    - `bx`: The block size for the first vector, which is unused in the function.
    - `vy`: A pointer to the second vector's data, which is expected to be of type `block_q8_K`.
    - `by`: The block size for the second vector, which is unused in the function.
    - `nrc`: An integer that is expected to be 1, which is unused in the function.
- **Control Flow**:
    - The function begins by asserting that `n` is a multiple of `QK_K` and that `nrc` equals 1.
    - It initializes pointers to the input vectors and calculates the number of blocks `nb` based on `n`.
    - Depending on the defined architecture (e.g., ARM NEON, AVX2, AVX, etc.), it executes different optimized code paths for calculating the dot product.
    - In each architecture-specific block, it iterates over the number of blocks, performing element-wise multiplications and accumulating results using SIMD instructions.
    - The final result is scaled and stored in the output pointer `s`.
- **Output**: The function outputs the computed dot product of the two quantized vectors, scaled by a factor of 0.125, stored in the float pointed to by `s`.
- **Functions called**:
    - [`vzip1_u8`](ggml-cpu-impl.h.driver.md#vzip1_u8)
    - [`vzip2_u8`](ggml-cpu-impl.h.driver.md#vzip2_u8)
    - [`ggml_vld1q_s8_x4`](ggml-cpu-impl.h.driver.md#ggml_vld1q_s8_x4)
    - [`ggml_vdotq_s32`](ggml-cpu-impl.h.driver.md#ggml_vdotq_s32)
    - [`vpaddq_s32`](ggml-cpu-impl.h.driver.md#vpaddq_s32)
    - [`vaddvq_s32`](ggml-cpu-impl.h.driver.md#vaddvq_s32)
    - [`get_scale_shuffle`](#get_scale_shuffle)
    - [`hsum_float_8`](#hsum_float_8)
    - [`lasx_shuffle_b`](#lasx_shuffle_b)
    - [`lasx_set_d`](#lasx_set_d)
    - [`lasx_extracti128`](#lasx_extracti128)
    - [`lasx_insertf128`](#lasx_insertf128)
    - [`lasx_maddubs_h`](#lasx_maddubs_h)
    - [`lasx_ext8_16`](#lasx_ext8_16)
    - [`lsx_shuffle_b`](#lsx_shuffle_b)
    - [`lasx_madd_h`](#lasx_madd_h)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)


---
### ggml\_vec\_dot\_iq2\_s\_q8\_K<!-- {{#callable:ggml_vec_dot_iq2_s_q8_K}} -->
Calculates the dot product of two quantized vectors with various optimizations based on the architecture.
- **Inputs**:
    - `n`: The number of elements in the vectors, which must be a multiple of `QK_K`.
    - `s`: A pointer to a float where the result of the dot product will be stored.
    - `bs`: The batch size, which is unused in the function.
    - `vx`: A pointer to the first vector of type `block_iq2_s`.
    - `bx`: The size of the first vector, which is unused in the function.
    - `vy`: A pointer to the second vector of type `block_q8_K`.
    - `by`: The size of the second vector, which is unused in the function.
    - `nrc`: A parameter that is expected to be 1 and is unused in the function.
- **Control Flow**:
    - The function begins by asserting that `n` is a multiple of `QK_K` and that `nrc` equals 1.
    - It initializes pointers to the input vectors and calculates the number of blocks `nb` based on `n`.
    - Depending on the defined architecture (e.g., ARM NEON, AVX2, etc.), it executes optimized code paths for calculating the dot product.
    - For each block, it retrieves the necessary data from the input vectors, performs quantization and scaling, and computes partial sums.
    - The final result is accumulated and scaled before being stored in the output pointer `s`.
- **Output**: The function outputs the computed dot product of the two quantized vectors, scaled by a factor of 0.125, stored in the location pointed to by `s`.
- **Functions called**:
    - [`ggml_vld1q_u8_x2`](ggml-cpu-impl.h.driver.md#ggml_vld1q_u8_x2)
    - [`ggml_vld1q_s8_x4`](ggml-cpu-impl.h.driver.md#ggml_vld1q_s8_x4)
    - [`ggml_vqtbl1q_u8`](ggml-cpu-impl.h.driver.md#ggml_vqtbl1q_u8)
    - [`ggml_vdotq_s32`](ggml-cpu-impl.h.driver.md#ggml_vdotq_s32)
    - [`vaddvq_s32`](ggml-cpu-impl.h.driver.md#vaddvq_s32)
    - [`get_scale_shuffle_k4`](#get_scale_shuffle_k4)
    - [`hsum_float_8`](#hsum_float_8)
    - [`lasx_ext8_16`](#lasx_ext8_16)
    - [`lasx_set_d`](#lasx_set_d)
    - [`lasx_shuffle_b`](#lasx_shuffle_b)
    - [`lasx_maddubs_h`](#lasx_maddubs_h)
    - [`lasx_madd_h`](#lasx_madd_h)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)


---
### ggml\_vec\_dot\_iq3\_xxs\_q8\_K<!-- {{#callable:ggml_vec_dot_iq3_xxs_q8_K}} -->
Calculates the dot product of two quantized vectors using various SIMD optimizations.
- **Inputs**:
    - `n`: The number of elements in the vectors, which must be a multiple of `QK_K`.
    - `s`: A pointer to a float where the result of the dot product will be stored.
    - `bs`: The block size, which is unused in the function.
    - `vx`: A pointer to the first quantized vector data structure.
    - `bx`: The block size for the first vector, which is unused in the function.
    - `vy`: A pointer to the second quantized vector data structure.
    - `by`: The block size for the second vector, which is unused in the function.
    - `nrc`: An integer that is expected to be 1, which is unused in the function.
- **Control Flow**:
    - The function begins by asserting that `n` is a multiple of `QK_K` and that `nrc` equals 1.
    - It initializes pointers to the quantized vector data structures and calculates the number of blocks `nb` based on `n`.
    - Depending on the defined architecture (e.g., ARM NEON, AVX2, AVX, etc.), it executes different SIMD optimized code paths to compute the dot product.
    - For each block, it retrieves the necessary data, performs element-wise multiplications, and accumulates the results.
    - Finally, the computed dot product is scaled and stored in the output pointer `s`.
- **Output**: The function outputs the computed dot product of the two quantized vectors, scaled by a factor of 0.25 or 0.5 depending on the architecture.
- **Functions called**:
    - [`ggml_vld1q_s8_x4`](ggml-cpu-impl.h.driver.md#ggml_vld1q_s8_x4)
    - [`ggml_vdotq_s32`](ggml-cpu-impl.h.driver.md#ggml_vdotq_s32)
    - [`vaddvq_s32`](ggml-cpu-impl.h.driver.md#vaddvq_s32)
    - [`hsum_float_8`](#hsum_float_8)
    - [`lasx_set_w`](#lasx_set_w)
    - [`lasx_set_d`](#lasx_set_d)
    - [`lasx_maddubs_h`](#lasx_maddubs_h)
    - [`lasx_madd_h`](#lasx_madd_h)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)


---
### ggml\_vec\_dot\_iq3\_s\_q8\_K<!-- {{#callable:ggml_vec_dot_iq3_s_q8_K}} -->
Calculates the dot product of two quantized vectors using various SIMD optimizations.
- **Inputs**:
    - `n`: The number of elements in the vectors, which must be a multiple of `QK_K`.
    - `s`: A pointer to a float where the result of the dot product will be stored.
    - `bs`: The block size, which is unused in the function.
    - `vx`: A pointer to the first quantized vector data structure.
    - `bx`: The block size for the first vector, which is unused in the function.
    - `vy`: A pointer to the second quantized vector data structure.
    - `by`: The block size for the second vector, which is unused in the function.
    - `nrc`: An integer that is expected to be 1, which is unused in the function.
- **Control Flow**:
    - The function begins by asserting that `n` is a multiple of `QK_K` and that `nrc` equals 1.
    - It initializes pointers to the quantized vector data structures and calculates the number of blocks `nb` based on `n`.
    - Depending on the defined architecture (e.g., ARM NEON, AVX2, AVX, etc.), it executes different SIMD optimized code paths for the dot product calculation.
    - For each block, it retrieves the necessary data from the quantized vectors, performs calculations using SIMD instructions, and accumulates the results.
    - Finally, it stores the computed dot product result in the location pointed to by `s`.
- **Output**: The function outputs the computed dot product of the two quantized vectors as a float, stored in the variable pointed to by `s`.
- **Functions called**:
    - [`ggml_vld1q_u8_x2`](ggml-cpu-impl.h.driver.md#ggml_vld1q_u8_x2)
    - [`ggml_vld1q_s8_x4`](ggml-cpu-impl.h.driver.md#ggml_vld1q_s8_x4)
    - [`ggml_vqtbl1q_u8`](ggml-cpu-impl.h.driver.md#ggml_vqtbl1q_u8)
    - [`ggml_vdotq_s32`](ggml-cpu-impl.h.driver.md#ggml_vdotq_s32)
    - [`vaddvq_s32`](ggml-cpu-impl.h.driver.md#vaddvq_s32)
    - [`hsum_float_8`](#hsum_float_8)
    - [`lasx_set_w`](#lasx_set_w)
    - [`lasx_extu8_16`](#lasx_extu8_16)
    - [`lasx_ext16_32`](#lasx_ext16_32)
    - [`lasx_extracti128`](#lasx_extracti128)
    - [`lasx_shuffle_b`](#lasx_shuffle_b)
    - [`lasx_maddubs_h`](#lasx_maddubs_h)
    - [`lasx_madd_h`](#lasx_madd_h)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)


---
### mul\_add\_epi8<!-- {{#callable:mul_add_epi8}} -->
Performs a multiplication and addition operation on two 256-bit vectors of 8-bit integers.
- **Inputs**:
    - `x`: A `__m256i` type vector containing the first set of 8-bit integers.
    - `y`: A `__m256i` type vector containing the second set of 8-bit integers.
- **Control Flow**:
    - The function begins by multiplying the lower half of the 8-bit integers in `x` and `y` using the `__lasx_xvmulwev_h_b` intrinsic, storing the result in `a`.
    - Next, it multiplies the upper half of the 8-bit integers in `x` and `y` using the `__lasx_xvmulwod_h_b` intrinsic, storing the result in `b`.
    - Finally, it adds the results stored in `a` and `b` using the `__lasx_xvadd_h` intrinsic and returns the final result.
- **Output**: Returns a `__m256i` type vector that contains the sum of the products of the corresponding elements from the two input vectors.


---
### ggml\_vec\_dot\_iq1\_s\_q8\_K<!-- {{#callable:ggml_vec_dot_iq1_s_q8_K}} -->
Calculates the dot product of two quantized vectors with various optimizations based on the architecture.
- **Inputs**:
    - `n`: The number of elements in the vectors, which must be a multiple of `QK_K`.
    - `s`: A pointer to a float where the result of the dot product will be stored.
    - `bs`: The block size, which is unused in the function.
    - `vx`: A pointer to the first quantized vector (type `block_iq1_s`).
    - `bx`: The block size for the first vector, which is unused in the function.
    - `vy`: A pointer to the second quantized vector (type `block_q8_K`).
    - `by`: The block size for the second vector, which is unused in the function.
    - `nrc`: A parameter that is expected to be 1 and is unused in the function.
- **Control Flow**:
    - The function begins by asserting that `n` is a multiple of `QK_K` and that `nrc` equals 1.
    - It initializes pointers to the input vectors and calculates the number of blocks `nb` based on `n`.
    - Depending on the defined architecture (e.g., ARM NEON, AVX2, AVX, etc.), it executes different optimized code paths for calculating the dot product.
    - For each block, it retrieves quantized values and computes partial sums using vectorized operations.
    - The results from each block are accumulated into a final sum, which is then stored in the output pointer `s`.
- **Output**: The function outputs the computed dot product as a float value stored at the address pointed to by `s`.
- **Functions called**:
    - [`ggml_vld1q_s8_x4`](ggml-cpu-impl.h.driver.md#ggml_vld1q_s8_x4)
    - [`ggml_vdotq_s32`](ggml-cpu-impl.h.driver.md#ggml_vdotq_s32)
    - [`vaddvq_s32`](ggml-cpu-impl.h.driver.md#vaddvq_s32)
    - [`mul_add_epi8`](#mul_add_epi8)
    - [`hsum_float_8`](#hsum_float_8)
    - [`mul_add_epi8_sse`](#mul_add_epi8_sse)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)


---
### ggml\_vec\_dot\_iq1\_m\_q8\_K<!-- {{#callable:ggml_vec_dot_iq1_m_q8_K}} -->
Calculates the dot product of two quantized vectors with specific scaling and delta adjustments.
- **Inputs**:
    - `n`: The number of elements in the vectors, which must be a multiple of `QK_K`.
    - `s`: A pointer to a float where the result of the dot product will be stored.
    - `bs`: The block size, which is unused in the function.
    - `vx`: A pointer to the first quantized vector data structure of type `block_iq1_m`.
    - `bx`: The block size for the first vector, which is unused in the function.
    - `vy`: A pointer to the second quantized vector data structure of type `block_q8_K`.
    - `by`: The block size for the second vector, which is unused in the function.
    - `nrc`: A parameter that is expected to be 1 and is unused in the function.
- **Control Flow**:
    - The function begins by asserting that `n` is a multiple of `QK_K` and that `nrc` equals 1.
    - It initializes pointers to the input vectors and calculates the number of blocks `nb` based on `n`.
    - Depending on the compilation flags, it uses either ARM NEON, AVX2, AVX, or a fallback method to perform the dot product.
    - For each block, it retrieves the quantized values and scales, computes intermediate sums using vectorized operations, and accumulates the results.
    - Finally, it stores the computed dot product result in the location pointed to by `s`.
- **Output**: The function outputs the computed dot product as a float value stored at the address pointed to by `s`.
- **Functions called**:
    - [`ggml_vld1q_s8_x4`](ggml-cpu-impl.h.driver.md#ggml_vld1q_s8_x4)
    - [`vpaddq_s32`](ggml-cpu-impl.h.driver.md#vpaddq_s32)
    - [`ggml_vdotq_s32`](ggml-cpu-impl.h.driver.md#ggml_vdotq_s32)
    - [`vaddvq_s32`](ggml-cpu-impl.h.driver.md#vaddvq_s32)
    - [`mul_add_epi8`](#mul_add_epi8)
    - [`hsum_float_8`](#hsum_float_8)
    - [`mul_add_epi8_sse`](#mul_add_epi8_sse)


---
### ggml\_vec\_dot\_iq4\_nl\_q8\_0<!-- {{#callable:ggml_vec_dot_iq4_nl_q8_0}} -->
Calculates the dot product of two quantized vectors with specific optimizations for various architectures.
- **Inputs**:
    - `n`: The number of elements in the vectors, which must be a multiple of `QK4_NL`.
    - `s`: A pointer to a float where the result of the dot product will be stored.
    - `bs`: The block size, which is unused in the function.
    - `vx`: A pointer to the first input vector, which is quantized using `block_iq4_nl`.
    - `bx`: The block size for the first input vector, which is unused in the function.
    - `vy`: A pointer to the second input vector, which is quantized using `block_q8_0`.
    - `by`: The block size for the second input vector, which is unused in the function.
    - `nrc`: An integer that must be equal to 1, indicating the number of return codes.
- **Control Flow**:
    - The function begins by asserting that `nrc` is equal to 1 and that `n` is a multiple of `QK4_NL`.
    - It initializes pointers to the input vectors and calculates the number of blocks `nb` based on `n`.
    - The function then enters a loop that processes pairs of blocks from the input vectors, using SIMD instructions for optimization based on the architecture.
    - For each pair of blocks, it loads the quantized values, performs dot product calculations, and accumulates the results into `sumf`.
    - After processing all pairs, it handles any remaining blocks in a separate loop, performing similar calculations.
    - Finally, the result is stored in the location pointed to by `s`.
- **Output**: The function outputs the computed dot product of the two input vectors, stored in the float pointed to by `s`.
- **Functions called**:
    - [`ggml_vqtbl1q_s8`](ggml-cpu-impl.h.driver.md#ggml_vqtbl1q_s8)
    - [`ggml_vdotq_s32`](ggml-cpu-impl.h.driver.md#ggml_vdotq_s32)
    - [`vaddvq_s32`](ggml-cpu-impl.h.driver.md#vaddvq_s32)
    - [`mul_add_epi8`](#mul_add_epi8)
    - [`hsum_float_8`](#hsum_float_8)
    - [`mul_sum_i8_quad_float`](#mul_sum_i8_quad_float)
    - [`quad_fp16_delta_float`](#quad_fp16_delta_float)
    - [`lasx_insertf128`](#lasx_insertf128)
    - [`lsx_shuffle_b`](#lsx_shuffle_b)
    - [`lasx_madd_h`](#lasx_madd_h)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)
    - [`ggml_vec_dot`](ggml-cpu-impl.h.driver.md#ggml_vec_dot)


---
### ggml\_vec\_dot\_iq4\_xs\_q8\_K<!-- {{#callable:ggml_vec_dot_iq4_xs_q8_K}} -->
Calculates the dot product of two quantized vectors with various optimizations based on the architecture.
- **Inputs**:
    - `n`: The number of elements in the vectors, which must be a multiple of `QK_K`.
    - `s`: A pointer to a float where the result of the dot product will be stored.
    - `bs`: The block size, which is unused in the function.
    - `vx`: A pointer to the first input vector, which is quantized in IQ4 format.
    - `bx`: The block size for the first input vector, which is unused in the function.
    - `vy`: A pointer to the second input vector, which is quantized in Q8 format.
    - `by`: The block size for the second input vector, which is unused in the function.
    - `nrc`: A parameter that must be equal to 1, which is asserted at the beginning of the function.
- **Control Flow**:
    - The function begins by asserting that `nrc` is 1 and that `n` is a multiple of `QK_K`.
    - It initializes pointers to the quantized input vectors and calculates the number of blocks `nb` based on `n`.
    - Depending on the defined architecture (e.g., ARM NEON, AVX2, AVX, etc.), it executes optimized code paths for calculating the dot product.
    - For each block, it processes the quantized data, performs vectorized operations to compute partial sums, and applies scaling factors.
    - Finally, it accumulates the results and stores the final dot product in the output pointer `s`.
- **Output**: The function outputs the computed dot product of the two quantized vectors into the float pointer `s`.
- **Functions called**:
    - [`ggml_vld1q_u8_x2`](ggml-cpu-impl.h.driver.md#ggml_vld1q_u8_x2)
    - [`ggml_vld1q_s8_x4`](ggml-cpu-impl.h.driver.md#ggml_vld1q_s8_x4)
    - [`ggml_vqtbl1q_s8`](ggml-cpu-impl.h.driver.md#ggml_vqtbl1q_s8)
    - [`ggml_vdotq_s32`](ggml-cpu-impl.h.driver.md#ggml_vdotq_s32)
    - [`vaddvq_s32`](ggml-cpu-impl.h.driver.md#vaddvq_s32)
    - [`mul_add_epi8`](#mul_add_epi8)
    - [`hsum_float_8`](#hsum_float_8)
    - [`mul_add_epi8_sse`](#mul_add_epi8_sse)
    - [`lasx_insertf128`](#lasx_insertf128)
    - [`lasx_madd_h`](#lasx_madd_h)
    - [`__lasx_xvreplfr2vr_s`](ggml-cpu-impl.h.driver.md#__lasx_xvreplfr2vr_s)
    - [`ggml_vec_dot`](ggml-cpu-impl.h.driver.md#ggml_vec_dot)


---
### quantize\_row\_iq4\_nl<!-- {{#callable:quantize_row_iq4_nl}} -->
The `quantize_row_iq4_nl` function quantizes a row of floating-point values using a specified quantization method.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that represent the data to be quantized.
    - `y`: A pointer to a memory location where the quantized output will be stored.
    - `k`: An integer representing the number of elements in the input array `x`, which must be a multiple of `QK4_NL`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK4_NL` to ensure valid input.
    - If the assertion passes, it calls the [`quantize_row_iq4_nl_ref`](../ggml-quants.c.driver.md#quantize_row_iq4_nl_ref) function to perform the actual quantization process.
- **Output**: The function does not return a value; instead, it modifies the memory pointed to by `y` to store the quantized results.
- **Functions called**:
    - [`quantize_row_iq4_nl_ref`](../ggml-quants.c.driver.md#quantize_row_iq4_nl_ref)


---
### quantize\_row\_iq4\_xs<!-- {{#callable:quantize_row_iq4_xs}} -->
The `quantize_row_iq4_xs` function quantizes a row of floating-point values into a specified format.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that represent the data to be quantized.
    - `y`: A pointer to the output buffer where the quantized data will be stored.
    - `k`: An integer that specifies the number of elements to quantize, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that the value of `k` is a multiple of `QK_K` to ensure valid quantization parameters.
    - Next, it calls the [`quantize_iq4_xs`](../ggml-quants.c.driver.md#quantize_iq4_xs) function, passing the input array `x`, the output buffer `y`, a constant value of 1, the size `k`, and a NULL pointer for additional parameters.
- **Output**: The function does not return a value; instead, it modifies the output buffer `y` to contain the quantized representation of the input data.
- **Functions called**:
    - [`quantize_iq4_xs`](../ggml-quants.c.driver.md#quantize_iq4_xs)


