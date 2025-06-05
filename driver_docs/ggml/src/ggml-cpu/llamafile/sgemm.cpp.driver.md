# Purpose
This C++ source code file is part of a library that implements optimized matrix multiplication routines for various data types and architectures. The primary focus of the file is to provide high-performance implementations of matrix multiplication, specifically for the operation \( C = A^T \times B \), where \( A^T \) is the transpose of matrix \( A \). The code is designed to exploit hardware capabilities such as SIMD (Single Instruction, Multiple Data) instructions available on different CPU architectures, including x86 with AVX/AVX2/AVX512, ARM with NEON, and PowerPC with MMA. The file includes multiple template-based classes and functions that handle different data types like float, BF16, FP16, and quantized types (Q4, Q5, IQ4_NL), and it supports multithreading to further enhance performance.

The file defines a public API function [`llamafile_sgemm`](#llamafile_sgemm), which serves as an entry point for performing matrix multiplication. This function checks the compatibility of the input matrices' data types and dimensions, and then delegates the computation to the appropriate specialized class based on the detected CPU architecture and data type. The implementation includes various optimizations such as vectorized arithmetic operations, fused multiply-add (FMA) operations, and vectorized memory loading to minimize memory bandwidth usage and maximize computational throughput. The code is structured to be highly modular, allowing for easy extension and adaptation to new architectures or data types.
# Imports and Dependencies

---
- `sgemm.h`
- `ggml-impl.h`
- `ggml-cpu-impl.h`
- `ggml-quants.h`
- `atomic`
- `array`
- `type_traits`


# Data Structures

---
### tinyBLAS<!-- {{#data_structure:(anonymous)::tinyBLAS}} -->
- **Type**: `class`
- **Members**:
    - `params`: A pointer to ggml_compute_params, storing computation parameters.
    - `A`: A constant pointer to the first input matrix of type TA.
    - `B`: A constant pointer to the second input matrix of type TB.
    - `C`: A pointer to the output matrix of type TC.
    - `k`: An integer representing the number of columns in A and rows in B.
    - `lda`: An integer representing the leading dimension of A.
    - `ldb`: An integer representing the leading dimension of B.
    - `ldc`: An integer representing the leading dimension of C.
- **Description**: The `tinyBLAS` class template is designed for performing optimized matrix multiplication on CPUs, specifically for the operation C = Aᵀ * B, where A is transposed. It is a highly parameterized class that supports different data types and vectorization strategies, depending on the CPU architecture. The class is initialized with pointers to matrices A, B, and C, along with their respective leading dimensions and the number of columns/rows involved in the multiplication. The class provides a `matmul` method to execute the matrix multiplication, leveraging vectorized operations and multithreading to enhance performance, especially for matrices that fit within the CPU cache.
- **Member Functions**:
    - [`(anonymous)::tinyBLAS::tinyBLAS`](#(anonymous)::tinyBLAS::tinyBLAS)
    - [`(anonymous)::tinyBLAS::matmul`](#(anonymous)::tinyBLAS::matmul)
    - [`(anonymous)::tinyBLAS::mnpack`](#(anonymous)::tinyBLAS::mnpack)
    - [`(anonymous)::tinyBLAS::gemm_bloc`](#(anonymous)::tinyBLAS::gemm_bloc)
    - [`(anonymous)::tinyBLAS::gemm`](#(anonymous)::tinyBLAS::gemm)

**Methods**

---
#### tinyBLAS::tinyBLAS<!-- {{#callable:(anonymous)::tinyBLAS::tinyBLAS}} -->
The `tinyBLAS` constructor initializes a `tinyBLAS` object with given matrix parameters and pointers.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing computation parameters.
    - `k`: An integer representing the number of columns in matrix A and rows in matrix B.
    - `A`: A pointer to the first input matrix, which is transposed.
    - `lda`: An integer representing the leading dimension (row stride) of matrix A.
    - `B`: A pointer to the second input matrix, which is not transposed.
    - `ldb`: An integer representing the leading dimension (row stride) of matrix B.
    - `C`: A pointer to the output matrix where the result of the matrix multiplication will be stored.
    - `ldc`: An integer representing the leading dimension (row stride) of matrix C.
- **Control Flow**:
    - The constructor initializes the `tinyBLAS` object by assigning the provided parameters and pointers to the class's member variables.
    - No additional logic or computation is performed within the constructor.
- **Output**: The constructor does not return any value; it initializes the object state.
- **See also**: [`(anonymous)::tinyBLAS`](#(anonymous)::tinyBLAS)  (Data Structure)


---
#### tinyBLAS::matmul<!-- {{#callable:(anonymous)::tinyBLAS::matmul}} -->
The `matmul` function performs matrix multiplication with specific optimizations based on the input dimensions and vector register size, returning a boolean indicating success or failure.
- **Inputs**:
    - `m`: The number of rows in the matrix A and the resulting matrix C.
    - `n`: The number of columns in the matrix B and the resulting matrix C.
- **Control Flow**:
    - Check if the member variable `k` is divisible by the template parameter `KN`; if not, return false.
    - Depending on the value of `VECTOR_REGISTERS`, choose different block sizes and call the `mnpack` function with specific parameters if `m` is divisible by 16, 8, or 4, and return true if successful.
    - If none of the conditions are met, return false.
- **Output**: A boolean value indicating whether the matrix multiplication was successfully performed with the given optimizations.
- **See also**: [`(anonymous)::tinyBLAS`](#(anonymous)::tinyBLAS)  (Data Structure)


---
#### tinyBLAS::mnpack<!-- {{#callable:(anonymous)::tinyBLAS::mnpack}} -->
The `mnpack` function is a recursive template function that determines the appropriate block size for matrix multiplication and calls the `gemm` function if the block size matches, or recursively reduces the block size until a valid configuration is found.
- **Inputs**:
    - `m`: The number of rows in the matrix.
    - `n`: The number of columns in the matrix.
    - `SIZE_N`: The current block size for the columns.
    - `BN`: The block size for the rows.
- **Control Flow**:
    - Check if `SIZE_N` equals `RN`; if true, call `gemm<RM, RN, BM>(m, n, BN)` and return.
    - If `RN` is greater than 1, recursively call `mnpack<RM, RN-1, BM>(m, n, SIZE_N, BN)` to try a smaller block size.
    - If `RN` is not greater than 1, log an error message indicating an unsupported block size and assert false.
- **Output**: The function does not return a value; it either calls another function or logs an error and asserts false.
- **See also**: [`(anonymous)::tinyBLAS`](#(anonymous)::tinyBLAS)  (Data Structure)


---
#### tinyBLAS::gemm\_bloc<!-- {{#callable:(anonymous)::tinyBLAS::gemm_bloc}} -->
The `gemm_bloc` function performs a block matrix multiplication for a specific sub-block of matrices A and B, accumulating the results into matrix C.
- **Inputs**:
    - `ii`: The starting row index for the sub-block of matrix A.
    - `jj`: The starting column index for the sub-block of matrix B.
- **Control Flow**:
    - Initialize a local accumulator matrix Cv with dimensions RN x RM to store intermediate results.
    - Iterate over the shared dimension 'k' in steps of KN, processing sub-blocks of A and B.
    - If RM <= RN, load RM vectors from A and iterate over RN vectors from B, performing multiply-add operations to accumulate results in Cv.
    - If RM > RN, load RN vectors from B and iterate over RM vectors from A, performing multiply-add operations to accumulate results in Cv.
    - After processing all sub-blocks, iterate over Cv to compute the horizontal sum of each element and store the result in the corresponding position in matrix C.
- **Output**: The function does not return a value; it updates the matrix C with the results of the block matrix multiplication.
- **Functions called**:
    - [`(anonymous)::madd`](#(anonymous)::madd)
    - [`(anonymous)::hsum`](#(anonymous)::hsum)
- **See also**: [`(anonymous)::tinyBLAS`](#(anonymous)::tinyBLAS)  (Data Structure)


---
#### tinyBLAS::gemm<!-- {{#callable:(anonymous)::tinyBLAS::gemm}} -->
The `gemm` function performs a multithreaded matrix multiplication operation using a tiled approach for efficient computation on CPU.
- **Inputs**:
    - `m`: The number of rows in the matrix A and the resulting matrix C.
    - `n`: The number of columns in the matrix B and the resulting matrix C.
    - `BN`: The block size used for tiling the computation.
- **Control Flow**:
    - The function begins by asserting that the number of rows m is divisible by the product of RM and BM, ensuring proper tiling.
    - It calculates the number of tiles in the y-direction (ytiles) and x-direction (xtiles) based on the input dimensions and tile sizes.
    - The function determines the number of jobs (nb_job) by multiplying ytiles with the number of block tiles in the x-direction (NB_BN).
    - If the current thread is the first one (ith == 0), it initializes the current_chunk atomic variable to the number of threads (nth).
    - A barrier is used to synchronize all threads before starting the computation.
    - Each thread processes a chunk of the matrix multiplication by iterating over jobs, calculating the starting indices for the tiles, and calling the `gemm_bloc` function for each tile.
    - The function uses atomic operations to fetch and increment the current job index, ensuring that each thread processes a unique chunk.
    - Another barrier is used to synchronize all threads after the computation is complete.
- **Output**: The function does not return any value; it performs the matrix multiplication in-place on the provided matrices.
- **Functions called**:
    - [`ggml_barrier`](../ggml-cpu.c.driver.md#ggml_barrier)
    - [`(anonymous)::BLOC_POS`](#(anonymous)::BLOC_POS)
- **See also**: [`(anonymous)::tinyBLAS`](#(anonymous)::tinyBLAS)  (Data Structure)



---
### tinyBLAS\_Q0\_ARM<!-- {{#data_structure:(anonymous)::tinyBLAS_Q0_ARM}} -->
- **Type**: `class`
- **Members**:
    - `A`: A constant pointer to the first input matrix of type TA.
    - `B`: A constant pointer to the second input matrix of type block_q8_0.
    - `C`: A constant pointer to the output matrix of type float.
    - `k`: A constant integer representing the number of columns in A and rows in B.
    - `lda`: A constant integer representing the leading dimension of A.
    - `ldb`: A constant integer representing the leading dimension of B.
    - `ldc`: A constant integer representing the leading dimension of C.
    - `ith`: An integer representing the thread index.
    - `nth`: An integer representing the total number of threads.
- **Description**: The `tinyBLAS_Q0_ARM` class is a template class designed for performing optimized matrix multiplication on ARM architectures with dot product support. It is specifically tailored for handling matrices where the first matrix is of a generic type `TA` and the second matrix is of type `block_q8_0`. The class is initialized with matrix dimensions and pointers to the matrices, and it provides a method `matmul` to perform the matrix multiplication operation. The class is optimized for multithreaded execution, allowing for efficient computation by dividing the workload among multiple threads.
- **Member Functions**:
    - [`(anonymous)::tinyBLAS_Q0_ARM::tinyBLAS_Q0_ARM`](#(anonymous)::tinyBLAS_Q0_ARM::tinyBLAS_Q0_ARM)
    - [`(anonymous)::tinyBLAS_Q0_ARM::matmul`](#(anonymous)::tinyBLAS_Q0_ARM::matmul)
    - [`(anonymous)::tinyBLAS_Q0_ARM::mnpack`](#(anonymous)::tinyBLAS_Q0_ARM::mnpack)
    - [`(anonymous)::tinyBLAS_Q0_ARM::gemm`](#(anonymous)::tinyBLAS_Q0_ARM::gemm)
    - [`(anonymous)::tinyBLAS_Q0_ARM::load_lo`](#(anonymous)::tinyBLAS_Q0_ARM::load_lo)
    - [`(anonymous)::tinyBLAS_Q0_ARM::load_hi`](#(anonymous)::tinyBLAS_Q0_ARM::load_hi)
    - [`(anonymous)::tinyBLAS_Q0_ARM::load_lo`](#(anonymous)::tinyBLAS_Q0_ARM::load_lo)
    - [`(anonymous)::tinyBLAS_Q0_ARM::load_hi`](#(anonymous)::tinyBLAS_Q0_ARM::load_hi)

**Methods**

---
#### tinyBLAS\_Q0\_ARM::tinyBLAS\_Q0\_ARM<!-- {{#callable:(anonymous)::tinyBLAS_Q0_ARM::tinyBLAS_Q0_ARM}} -->
The `tinyBLAS_Q0_ARM` constructor initializes an instance of the `tinyBLAS_Q0_ARM` class with matrix multiplication parameters and pointers to matrices A, B, and C.
- **Inputs**:
    - `k`: The number of columns in matrix A and the number of rows in matrix B.
    - `A`: A pointer to the first input matrix, which is transposed.
    - `lda`: The leading dimension or row stride of matrix A.
    - `B`: A pointer to the second input matrix, which is not transposed.
    - `ldb`: The leading dimension or row stride of matrix B.
    - `C`: A pointer to the output matrix where the result of the multiplication will be stored.
    - `ldc`: The leading dimension or row stride of matrix C.
    - `ith`: The thread index, indicating which thread is executing this instance.
    - `nth`: The total number of threads available for execution.
- **Control Flow**:
    - The constructor initializes the member variables A, B, C, k, lda, ldb, ldc, ith, and nth with the provided arguments.
    - No additional logic or operations are performed within the constructor body.
- **Output**: An instance of the `tinyBLAS_Q0_ARM` class is created with the specified parameters, ready for matrix multiplication operations.
- **See also**: [`(anonymous)::tinyBLAS_Q0_ARM`](#(anonymous)::tinyBLAS_Q0_ARM)  (Data Structure)


---
#### tinyBLAS\_Q0\_ARM::matmul<!-- {{#callable:(anonymous)::tinyBLAS_Q0_ARM::matmul}} -->
The `matmul` function initiates a matrix multiplication operation by calling the [`mnpack`](#(anonymous)::tinyBLAS::mnpack) method with specified row and column indices.
- **Inputs**:
    - `m`: The number of rows in the matrix A and the resulting matrix C.
    - `n`: The number of columns in the matrix B and the resulting matrix C.
- **Control Flow**:
    - The function `matmul` is called with two parameters, `m` and `n`, representing the dimensions of the matrices involved in the multiplication.
    - Inside `matmul`, the [`mnpack`](#(anonymous)::tinyBLAS::mnpack) method is invoked with the parameters `0, m, 0, n`, which sets the starting and ending indices for the rows and columns to be processed.
- **Output**: The function does not return any value; it initiates the matrix multiplication process by calling [`mnpack`](#(anonymous)::tinyBLAS::mnpack).
- **Functions called**:
    - [`(anonymous)::tinyBLAS::mnpack`](#(anonymous)::tinyBLAS::mnpack)
- **See also**: [`(anonymous)::tinyBLAS_Q0_ARM`](#(anonymous)::tinyBLAS_Q0_ARM)  (Data Structure)


---
#### tinyBLAS\_Q0\_ARM::mnpack<!-- {{#callable:(anonymous)::tinyBLAS_Q0_ARM::mnpack}} -->
The [`mnpack`](#(anonymous)::tinyBLAS_Q0_AVX::mnpack) function recursively partitions a matrix multiplication problem into smaller subproblems and calls a specialized `gemm` function based on the dimensions of the subproblem.
- **Inputs**:
    - `m0`: The starting row index for the submatrix of matrix A.
    - `m`: The ending row index for the submatrix of matrix A.
    - `n0`: The starting column index for the submatrix of matrix B.
    - `n`: The ending column index for the submatrix of matrix B.
- **Control Flow**:
    - Calculate the minimum of the row and column dimensions minus their starting indices, and use these to determine the case in a switch statement.
    - Based on the case, set `mc` and `nc` to specific values and call the `gemm` function with template parameters corresponding to these values.
    - Calculate `mp` and `np` as the next partition points for the matrix multiplication.
    - Recursively call [`mnpack`](#(anonymous)::tinyBLAS_Q0_AVX::mnpack) for the submatrices defined by the new partition points `mp` and `np`.
- **Output**: The function does not return a value; it performs matrix multiplication on submatrices and updates the result matrix in place.
- **Functions called**:
    - [`(anonymous)::tinyBLAS_Q0_AVX::mnpack`](#(anonymous)::tinyBLAS_Q0_AVX::mnpack)
- **See also**: [`(anonymous)::tinyBLAS_Q0_ARM`](#(anonymous)::tinyBLAS_Q0_ARM)  (Data Structure)


---
#### tinyBLAS\_Q0\_ARM::gemm<!-- {{#callable:(anonymous)::tinyBLAS_Q0_ARM::gemm}} -->
The `gemm` function performs a tiled matrix multiplication on a subset of matrices A and B, storing the result in matrix C, using a multi-threaded approach.
- **Inputs**:
    - `m0`: The starting row index for the matrix A.
    - `m`: The ending row index for the matrix A.
    - `n0`: The starting column index for the matrix B.
    - `n`: The ending column index for the matrix B.
- **Control Flow**:
    - Calculate the number of tiles in the y-direction (`ytiles`) and x-direction (`xtiles`) based on the dimensions of the matrices and the tile sizes `RM` and `RN`.
    - Compute the total number of tiles (`tiles`) and determine the number of tiles each thread should process (`duty`).
    - Calculate the starting (`start`) and ending (`end`) tile indices for the current thread based on its ID (`ith`) and the total number of threads (`nth`).
    - Iterate over the assigned tiles for the current thread, calculating the starting indices `ii` and `jj` for each tile in matrices A and B respectively.
    - Initialize a local accumulator matrix `Cv` for the current tile.
    - Perform the matrix multiplication for the current tile by iterating over the shared dimension `k`, updating `Cv` using vectorized operations.
    - Store the accumulated results from `Cv` into the appropriate positions in matrix C.
- **Output**: The function does not return a value; it modifies the matrix C in place with the results of the matrix multiplication.
- **Functions called**:
    - [`(anonymous)::tinyBLAS_Q0_ARM::load_lo`](#(anonymous)::tinyBLAS_Q0_ARM::load_lo)
    - [`(anonymous)::tinyBLAS_Q0_ARM::load_hi`](#(anonymous)::tinyBLAS_Q0_ARM::load_hi)
    - [`(anonymous)::unhalf`](#(anonymous)::unhalf)
    - [`(anonymous)::hsum`](#(anonymous)::hsum)
- **See also**: [`(anonymous)::tinyBLAS_Q0_ARM`](#(anonymous)::tinyBLAS_Q0_ARM)  (Data Structure)


---
#### tinyBLAS\_Q0\_ARM::load\_lo<!-- {{#callable:(anonymous)::tinyBLAS_Q0_ARM::load_lo}} -->
The `load_lo` function loads the lower 16 bytes of a `block_q8_0` structure into a NEON vector register.
- **Inputs**:
    - `b`: A pointer to a `block_q8_0` structure, which contains a member `qs` that is an array of bytes.
- **Control Flow**:
    - The function takes a pointer to a `block_q8_0` structure as input.
    - It uses the NEON intrinsic `vld1q_s8` to load the first 16 bytes from the `qs` array of the `block_q8_0` structure into a 128-bit NEON vector register.
- **Output**: The function returns an `int8x16_t` NEON vector containing the loaded bytes.
- **See also**: [`(anonymous)::tinyBLAS_Q0_ARM`](#(anonymous)::tinyBLAS_Q0_ARM)  (Data Structure)


---
#### tinyBLAS\_Q0\_ARM::load\_hi<!-- {{#callable:(anonymous)::tinyBLAS_Q0_ARM::load_hi}} -->
The `load_hi` function loads a high 128-bit vector from a `block_q8_0` structure's `qs` array starting at the 16th element.
- **Inputs**:
    - `b`: A pointer to a `block_q8_0` structure, which contains an array `qs` of 8-bit integers.
- **Control Flow**:
    - The function accesses the `qs` array of the `block_q8_0` structure pointed to by `b`.
    - It calculates the address of the 16th element in the `qs` array.
    - It uses the `vld1q_s8` function to load a 128-bit vector from this address.
- **Output**: Returns a 128-bit vector (`int8x16_t`) containing 16 8-bit integers loaded from the specified position in the `qs` array.
- **See also**: [`(anonymous)::tinyBLAS_Q0_ARM`](#(anonymous)::tinyBLAS_Q0_ARM)  (Data Structure)


---
#### tinyBLAS\_Q0\_ARM::load\_lo<!-- {{#callable:(anonymous)::tinyBLAS_Q0_ARM::load_lo}} -->
The `load_lo` function loads the lower 4 bits of each byte from a `block_q4_0` structure, converts them to signed 8-bit integers, and subtracts 8 from each.
- **Inputs**:
    - `b`: A pointer to a `block_q4_0` structure, which contains a member `qs` that is an array of bytes.
- **Control Flow**:
    - Load 16 bytes from the `qs` array of the `block_q4_0` structure using `vld1q_u8`.
    - Perform a bitwise AND operation with `0x0f` to extract the lower 4 bits of each byte using `vandq_u8`.
    - Convert the result from unsigned 8-bit integers to signed 8-bit integers using `vreinterpretq_s8_u8`.
    - Subtract 8 from each element of the vector using `vsubq_s8`.
- **Output**: Returns a vector of signed 8-bit integers (`int8x16_t`) where each element is the lower 4 bits of the corresponding byte in `b->qs`, interpreted as a signed integer and adjusted by subtracting 8.
- **See also**: [`(anonymous)::tinyBLAS_Q0_ARM`](#(anonymous)::tinyBLAS_Q0_ARM)  (Data Structure)


---
#### tinyBLAS\_Q0\_ARM::load\_hi<!-- {{#callable:(anonymous)::tinyBLAS_Q0_ARM::load_hi}} -->
The `load_hi` function loads the higher 4 bits of each byte from a `block_q4_0` structure, interprets them as signed 8-bit integers, and subtracts 8 from each.
- **Inputs**:
    - `b`: A pointer to a `block_q4_0` structure, which contains a member `qs` that is an array of bytes.
- **Control Flow**:
    - Load 16 bytes from the `qs` array of the `block_q4_0` structure pointed to by `b`.
    - Shift each byte right by 4 bits to isolate the higher 4 bits of each byte.
    - Reinterpret the resulting 4-bit values as signed 8-bit integers.
    - Subtract 8 from each of these signed 8-bit integers.
- **Output**: Returns a 128-bit NEON vector (`int8x16_t`) containing the processed signed 8-bit integers.
- **See also**: [`(anonymous)::tinyBLAS_Q0_ARM`](#(anonymous)::tinyBLAS_Q0_ARM)  (Data Structure)



---
### tinyBLAS\_Q0\_AVX<!-- {{#data_structure:(anonymous)::tinyBLAS_Q0_AVX}} -->
- **Type**: `class`
- **Members**:
    - `A`: Pointer to the first input matrix of type TA.
    - `B`: Pointer to the second input matrix of type TB.
    - `C`: Pointer to the output matrix of type TC.
    - `k`: The number of columns in A and rows in B.
    - `lda`: Leading dimension of matrix A.
    - `ldb`: Leading dimension of matrix B.
    - `ldc`: Leading dimension of matrix C.
    - `ith`: Thread index for parallel execution.
    - `nth`: Total number of threads for parallel execution.
    - `iq4nlt`: A 128-bit integer vector used for loading specific values.
- **Description**: The `tinyBLAS_Q0_AVX` class is a template-based implementation designed for optimized matrix multiplication using AVX instructions. It supports multithreaded execution and is specialized for handling matrices that fit within the CPU cache, aiming to maximize performance. The class is parameterized by three types, TA, TB, and TC, representing the data types of the input matrices A, B, and the output matrix C, respectively. It includes several private methods for performing matrix multiplication with different block sizes, leveraging AVX2 and F16C instructions when available. The class is initialized with matrix dimensions and pointers, and it provides a `matmul` method to execute the matrix multiplication operation.
- **Member Functions**:
    - [`(anonymous)::tinyBLAS_Q0_AVX::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX::tinyBLAS_Q0_AVX)
    - [`(anonymous)::tinyBLAS_Q0_AVX::matmul`](#(anonymous)::tinyBLAS_Q0_AVX::matmul)
    - [`(anonymous)::tinyBLAS_Q0_AVX::mnpack`](#(anonymous)::tinyBLAS_Q0_AVX::mnpack)
    - [`(anonymous)::tinyBLAS_Q0_AVX::gemm4xN`](#(anonymous)::tinyBLAS_Q0_AVX::gemm4xN)
    - [`(anonymous)::tinyBLAS_Q0_AVX::gemmMx4`](#(anonymous)::tinyBLAS_Q0_AVX::gemmMx4)
    - [`(anonymous)::tinyBLAS_Q0_AVX::gemm`](#(anonymous)::tinyBLAS_Q0_AVX::gemm)
    - [`(anonymous)::tinyBLAS_Q0_AVX::load`](#(anonymous)::tinyBLAS_Q0_AVX::load)
    - [`(anonymous)::tinyBLAS_Q0_AVX::load0`](#(anonymous)::tinyBLAS_Q0_AVX::load0)
    - [`(anonymous)::tinyBLAS_Q0_AVX::load1`](#(anonymous)::tinyBLAS_Q0_AVX::load1)
    - [`(anonymous)::tinyBLAS_Q0_AVX::load`](#(anonymous)::tinyBLAS_Q0_AVX::load)
    - [`(anonymous)::tinyBLAS_Q0_AVX::load0`](#(anonymous)::tinyBLAS_Q0_AVX::load0)
    - [`(anonymous)::tinyBLAS_Q0_AVX::load1`](#(anonymous)::tinyBLAS_Q0_AVX::load1)
    - [`(anonymous)::tinyBLAS_Q0_AVX::load`](#(anonymous)::tinyBLAS_Q0_AVX::load)
    - [`(anonymous)::tinyBLAS_Q0_AVX::load0`](#(anonymous)::tinyBLAS_Q0_AVX::load0)
    - [`(anonymous)::tinyBLAS_Q0_AVX::load1`](#(anonymous)::tinyBLAS_Q0_AVX::load1)
    - [`(anonymous)::tinyBLAS_Q0_AVX::load`](#(anonymous)::tinyBLAS_Q0_AVX::load)
    - [`(anonymous)::tinyBLAS_Q0_AVX::load0`](#(anonymous)::tinyBLAS_Q0_AVX::load0)
    - [`(anonymous)::tinyBLAS_Q0_AVX::load1`](#(anonymous)::tinyBLAS_Q0_AVX::load1)
    - [`(anonymous)::tinyBLAS_Q0_AVX::updot`](#(anonymous)::tinyBLAS_Q0_AVX::updot)
    - [`(anonymous)::tinyBLAS_Q0_AVX::denibble`](#(anonymous)::tinyBLAS_Q0_AVX::denibble)
    - [`(anonymous)::tinyBLAS_Q0_AVX::bittobyte`](#(anonymous)::tinyBLAS_Q0_AVX::bittobyte)

**Methods**

---
#### tinyBLAS\_Q0\_AVX::tinyBLAS\_Q0\_AVX<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::tinyBLAS_Q0_AVX}} -->
The `tinyBLAS_Q0_AVX` constructor initializes a matrix multiplication object with given matrices and dimensions, and preloads a specific set of integer values into a SIMD register for later use in computations.
- **Inputs**:
    - `k`: The number of columns in matrix A and rows in matrix B.
    - `A`: Pointer to the first input matrix, which is transposed.
    - `lda`: Leading dimension of matrix A, representing the row stride.
    - `B`: Pointer to the second input matrix, which is not transposed.
    - `ldb`: Leading dimension of matrix B, representing the row stride.
    - `C`: Pointer to the output matrix where the result will be stored.
    - `ldc`: Leading dimension of matrix C, representing the row stride.
    - `ith`: The thread index for parallel execution.
    - `nth`: The total number of threads for parallel execution.
- **Control Flow**:
    - Initialize member variables with the provided arguments.
    - Define a constant array `kvalues_iq4nl` with 16 specific int8_t values.
    - Load these values into a SIMD register `iq4nlt` using `_mm_loadu_si128`.
- **Output**: The constructor does not return any value; it initializes the object state for subsequent matrix multiplication operations.
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::matmul<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::matmul}} -->
The `matmul` function initiates a matrix multiplication operation by calling the [`mnpack`](#(anonymous)::tinyBLAS::mnpack) function with specified row and column indices.
- **Inputs**:
    - `m`: The number of rows in the matrix A.
    - `n`: The number of columns in the matrix B.
- **Control Flow**:
    - The function `matmul` is called with two parameters, `m` and `n`, representing the dimensions of the matrices involved in the multiplication.
    - Inside `matmul`, the function [`mnpack`](#(anonymous)::tinyBLAS::mnpack) is called with the parameters `0, m, 0, n`, which sets the starting and ending indices for the rows and columns to be processed.
- **Output**: The function does not return any value; it initiates the matrix multiplication process.
- **Functions called**:
    - [`(anonymous)::tinyBLAS::mnpack`](#(anonymous)::tinyBLAS::mnpack)
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::mnpack<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::mnpack}} -->
The [`mnpack`](#(anonymous)::tinyBLAS::mnpack) function recursively partitions a matrix multiplication problem into smaller subproblems and executes optimized matrix multiplication routines based on the dimensions of the submatrices.
- **Inputs**:
    - `m0`: The starting row index for the submatrix of matrix A.
    - `m`: The ending row index for the submatrix of matrix A.
    - `n0`: The starting column index for the submatrix of matrix B.
    - `n`: The ending column index for the submatrix of matrix B.
- **Control Flow**:
    - Calculate the minimum of the row and column dimensions of the submatrices, limited to 4, and combine them into a single value to determine the case in the switch statement.
    - Use a switch statement to select the appropriate matrix multiplication routine based on the combined value of the submatrix dimensions.
    - For each case, set the values of `mc` and `nc` to define the block size for the matrix multiplication routine.
    - Call the appropriate templated `gemm` function to perform the matrix multiplication for the current block size.
    - Calculate the next partition points `mp` and `np` for further recursive calls.
    - Recursively call [`mnpack`](#(anonymous)::tinyBLAS::mnpack) for the submatrices defined by the new partition points `mp` and `np`.
- **Output**: The function does not return a value; it performs matrix multiplication and updates the result in place.
- **Functions called**:
    - [`(anonymous)::tinyBLAS::mnpack`](#(anonymous)::tinyBLAS::mnpack)
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::gemm4xN<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::gemm4xN}} -->
The `gemm4xN` function performs a matrix multiplication operation on a 4xN block of matrices using AVX2 and F16C instructions for optimized performance.
- **Inputs**:
    - `m0`: The starting row index for the matrix A.
    - `m`: The ending row index for the matrix A.
    - `n0`: The starting column index for the matrix B.
    - `n`: The ending column index for the matrix B.
- **Control Flow**:
    - Calculate the number of 4-row tiles (ytiles) and RN-column tiles (xtiles) based on the input dimensions.
    - Determine the total number of tiles and distribute the workload among threads using the duty cycle, start, and end indices.
    - Iterate over each assigned tile, calculating the starting indices for the current tile in matrices A and B.
    - Initialize a 4xRN block of accumulators (Cv) to store intermediate results.
    - For each element in the shared dimension (k), compute the delta values for a 4-element block of matrix A and convert them to float.
    - Load 256-bit vectors from matrix A and replicate the delta values across the 256-bit lane.
    - For each column in the RN block, compute the product of the delta values and the corresponding element from matrix B, and update the accumulators using fused multiply-add operations.
    - After processing all elements in the shared dimension, perform a horizontal sum on the accumulators and store the results in the output matrix C.
- **Output**: The function updates the matrix C with the results of the matrix multiplication for the specified 4xN block.
- **Functions called**:
    - [`(anonymous)::load`](#(anonymous)::load)
    - [`(anonymous)::unhalf`](#(anonymous)::unhalf)
    - [`(anonymous)::madd`](#(anonymous)::madd)
    - [`(anonymous)::tinyBLAS_Q0_AVX::updot`](#(anonymous)::tinyBLAS_Q0_AVX::updot)
    - [`(anonymous)::hsum`](#(anonymous)::hsum)
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::gemmMx4<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::gemmMx4}} -->
The `gemmMx4` function performs a matrix multiplication operation on a submatrix of dimensions Mx4 using AVX2 and F16C instructions, distributing the workload across multiple threads.
- **Inputs**:
    - `m0`: The starting row index for the submatrix of matrix A.
    - `m`: The ending row index for the submatrix of matrix A.
    - `n0`: The starting column index for the submatrix of matrix B.
    - `n`: The ending column index for the submatrix of matrix B.
- **Control Flow**:
    - Calculate the number of tiles in the y-direction (`ytiles`) and x-direction (`xtiles`) based on the input dimensions and the template parameter `RM`.
    - Determine the total number of tiles (`tiles`) and the number of tiles each thread should process (`duty`).
    - Calculate the starting (`start`) and ending (`end`) tile indices for the current thread based on its ID (`ith`) and the total number of threads (`nth`).
    - Iterate over each tile assigned to the current thread, calculating the starting indices `ii` and `jj` for the submatrices of A and B, respectively.
    - Initialize a 2D array `Cv` to store intermediate results for the current tile.
    - For each element in the k-dimension, compute the delta values for the B matrix and convert them to float using AVX2 instructions.
    - Load the corresponding elements from matrices A and B, compute the product, and update the `Cv` array using fused multiply-add operations.
    - After processing all elements in the k-dimension, sum the results in `Cv` and store them in the output matrix C.
- **Output**: The function does not return a value; it updates the matrix C with the results of the matrix multiplication.
- **Functions called**:
    - [`(anonymous)::load`](#(anonymous)::load)
    - [`(anonymous)::unhalf`](#(anonymous)::unhalf)
    - [`(anonymous)::madd`](#(anonymous)::madd)
    - [`(anonymous)::tinyBLAS_Q0_AVX::updot`](#(anonymous)::tinyBLAS_Q0_AVX::updot)
    - [`(anonymous)::hsum`](#(anonymous)::hsum)
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::gemm<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::gemm}} -->
The `gemm` function performs a matrix multiplication operation on submatrices of matrices A and B, storing the result in matrix C, using AVX2 or SSE instructions for optimization.
- **Inputs**:
    - `m0`: The starting row index for the submatrix of A.
    - `m`: The ending row index for the submatrix of A.
    - `n0`: The starting column index for the submatrix of B.
    - `n`: The ending column index for the submatrix of B.
- **Control Flow**:
    - Calculate the number of tiles in the y and x dimensions based on RM and RN, respectively.
    - Determine the total number of tiles and the duty (number of tiles per thread).
    - Calculate the start and end indices for the current thread's work based on its ID (ith) and the total number of threads (nth).
    - Iterate over each tile assigned to the current thread.
    - For each tile, calculate the starting indices (ii, jj) for the submatrices of A and B.
    - Initialize an accumulator matrix Cv for the current tile.
    - Iterate over the shared dimension k, performing vectorized dot products and accumulations using AVX2 or SSE instructions.
    - Store the accumulated results back into the appropriate positions in matrix C.
- **Output**: The function does not return a value; it modifies the matrix C in place with the results of the matrix multiplication.
- **Functions called**:
    - [`(anonymous)::tinyBLAS_Q0_AVX::updot`](#(anonymous)::tinyBLAS_Q0_AVX::updot)
    - [`(anonymous)::load`](#(anonymous)::load)
    - [`(anonymous)::madd`](#(anonymous)::madd)
    - [`(anonymous)::unhalf`](#(anonymous)::unhalf)
    - [`(anonymous)::hsum`](#(anonymous)::hsum)
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::load<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::load}} -->
The `load` function loads a 256-bit integer from the `qs` member of a `block_q8_0` structure into an AVX2 register.
- **Inputs**:
    - `b`: A pointer to a `block_q8_0` structure, which contains a member `qs` that is an array of bytes.
- **Control Flow**:
    - The function casts the `qs` member of the `block_q8_0` structure to a pointer to `__m256i`.
    - It then uses the `_mm256_loadu_si256` intrinsic to load the data into a 256-bit AVX2 register.
- **Output**: A 256-bit integer (`__m256i`) loaded from the `qs` member of the `block_q8_0` structure.
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::load0<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::load0}} -->
The `load0` function loads 128 bits of data from a `block_q8_0` structure into an `__m128i` SIMD register using unaligned memory access.
- **Inputs**:
    - `b`: A pointer to a `block_q8_0` structure, which contains a member `qs` that is an array of data to be loaded.
- **Control Flow**:
    - The function casts the `qs` member of the `block_q8_0` structure to a `const __m128i *` pointer.
    - It then uses the `_mm_loadu_si128` intrinsic to load 128 bits of data from the address pointed to by the casted pointer into an `__m128i` register.
- **Output**: The function returns an `__m128i` register containing the loaded data.
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::load1<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::load1}} -->
The `load1` function loads a 128-bit integer from a specific offset within a `block_q8_0` structure using unaligned memory access.
- **Inputs**:
    - `b`: A pointer to a `block_q8_0` structure, which contains a member `qs` that is an array of bytes.
- **Control Flow**:
    - The function casts the `qs` member of the `block_q8_0` structure to a pointer of type `__m128i`.
    - It then adds an offset of 1 to this pointer to access the second 128-bit block.
    - The function uses `_mm_loadu_si128` to load the 128-bit integer from this offset, allowing for unaligned memory access.
- **Output**: Returns a `__m128i` type, which is a 128-bit integer loaded from the specified offset in the `qs` array of the `block_q8_0` structure.
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::load<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::load}} -->
The `load` function loads and processes a 256-bit vector from a `block_q4_0` structure by denibbling its `qs` field and subtracting 8 from each byte.
- **Inputs**:
    - `b`: A pointer to a `block_q4_0` structure, which contains a `qs` field representing a block of quantized data.
- **Control Flow**:
    - Call the [`denibble`](#(anonymous)::tinyBLAS_Q0_AVX::denibble) function on `b->qs` to extract the lower 4 bits of each byte in the `qs` field.
    - Use `_mm256_set1_epi8(8)` to create a 256-bit vector with all bytes set to 8.
    - Subtract the 256-bit vector of 8s from the result of `denibble(b->qs)` using `_mm256_sub_epi8`.
- **Output**: Returns a 256-bit integer vector (`__m256i`) where each byte is the result of subtracting 8 from the corresponding denibbled byte of `b->qs`.
- **Functions called**:
    - [`(anonymous)::tinyBLAS_Q0_AVX::denibble`](#(anonymous)::tinyBLAS_Q0_AVX::denibble)
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::load0<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::load0}} -->
The `load0` function loads a 128-bit block of data from a `block_q4_0` structure, applies a bitwise AND operation with a mask of 15, and then subtracts 8 from each byte.
- **Inputs**:
    - `b`: A pointer to a `block_q4_0` structure, which contains a member `qs` that is an array of bytes to be processed.
- **Control Flow**:
    - Load a 128-bit block of data from the `qs` member of the `block_q4_0` structure using `_mm_loadu_si128`.
    - Apply a bitwise AND operation between the loaded data and a constant mask of 15 using `_mm_and_si128`.
    - Subtract 8 from each byte of the result using `_mm_sub_epi8`.
- **Output**: Returns a 128-bit integer vector (`__m128i`) where each byte is the result of the bitwise operations applied to the corresponding byte in the input data.
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::load1<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::load1}} -->
The `load1` function loads a 128-bit integer from a `block_q4_0` structure, processes it to extract and adjust the higher nibble values, and returns the result as a 128-bit integer.
- **Inputs**:
    - `b`: A pointer to a `block_q4_0` structure, which contains a member `qs` that is an array of bytes.
- **Control Flow**:
    - Load a 128-bit integer from the memory location pointed to by `b->qs` using `_mm_loadu_si128`.
    - Shift the loaded integer right by 4 bits to access the higher nibbles using `_mm_srli_epi16`.
    - Mask the result with `0x0F` to isolate the higher nibbles using `_mm_and_si128`.
    - Subtract 8 from each byte in the result using `_mm_sub_epi8` to adjust the values.
    - Return the processed 128-bit integer.
- **Output**: A 128-bit integer (`__m128i`) containing the processed higher nibble values from the input block.
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::load<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::load}} -->
The `load` function combines two 256-bit integer vectors derived from a `block_q5_0` structure using bitwise operations.
- **Inputs**:
    - `b`: A pointer to a `block_q5_0` structure, which contains the data to be loaded and processed.
- **Control Flow**:
    - The function calls [`denibble`](#(anonymous)::tinyBLAS_Q0_AVX::denibble) on `b->qs` to extract and process the lower 4 bits of each byte in the `qs` array.
    - It calls [`bittobyte`](#(anonymous)::tinyBLAS_Q0_AVX::bittobyte) on `b->qh` to convert the high bits into a byte representation.
    - The results of [`denibble`](#(anonymous)::tinyBLAS_Q0_AVX::denibble) and [`bittobyte`](#(anonymous)::tinyBLAS_Q0_AVX::bittobyte) are combined using a bitwise OR operation to form a single 256-bit integer vector.
- **Output**: A 256-bit integer vector (`__m256i`) that is the result of combining the processed data from the `block_q5_0` structure.
- **Functions called**:
    - [`(anonymous)::tinyBLAS_Q0_AVX::denibble`](#(anonymous)::tinyBLAS_Q0_AVX::denibble)
    - [`(anonymous)::tinyBLAS_Q0_AVX::bittobyte`](#(anonymous)::tinyBLAS_Q0_AVX::bittobyte)
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::load0<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::load0}} -->
The `load0` function loads and processes a 128-bit block of data from a `block_q5_0` structure, applying bitwise operations to extract and manipulate specific bits.
- **Inputs**:
    - `b`: A pointer to a `block_q5_0` structure containing the data to be loaded and processed.
- **Control Flow**:
    - Load a 128-bit block from the `qs` field of the `block_q5_0` structure into an `__m128i` variable `x`.
    - Copy a 32-bit integer from the `qh` field of the `block_q5_0` structure into a `uint32_t` variable `x32`.
    - Perform a bitwise AND operation on `x` with a mask of 15 to isolate the lower 4 bits of each byte, storing the result in `qxl`.
    - Create a mask `bytesl` by comparing a constant with a bitwise OR of another constant and a shuffled version of `x32`.
    - Invert `bytesl` and perform a bitwise AND with a mask of 0xF0 to clear specific bits.
    - Combine `qxl` and `bytesl` using a bitwise OR operation to produce the final result.
- **Output**: Returns an `__m128i` value representing the processed 128-bit block with specific bits extracted and manipulated.
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::load1<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::load1}} -->
The `load1` function loads and processes a 128-bit block of data from a `block_q5_0` structure, applying bitwise operations to extract and combine specific bits into a single 128-bit integer.
- **Inputs**:
    - `b`: A pointer to a `block_q5_0` structure containing the data to be loaded and processed.
- **Control Flow**:
    - Load a 128-bit integer from the `qs` field of the `block_q5_0` structure using `_mm_loadu_si128`.
    - Copy a 32-bit integer from the `qh` field of the `block_q5_0` structure using `memcpy`.
    - Extract the higher 4 bits of each byte from the loaded 128-bit integer using `_mm_srli_epi16` and `_mm_and_si128`.
    - Create a mask to identify specific bytes using `_mm_cmpeq_epi8` and `_mm_or_si128` with a shuffled version of the copied 32-bit integer.
    - Apply the mask to clear certain bits using `_mm_andnot_si128`.
    - Combine the processed higher bits and masked bytes into a single 128-bit integer using `_mm_or_si128`.
- **Output**: A 128-bit integer (`__m128i`) that combines the processed higher bits and masked bytes from the input `block_q5_0` structure.
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::load<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::load}} -->
The `load` function loads data from a `block_iq4_nl` structure into a 256-bit AVX2 register by combining two 128-bit loads.
- **Inputs**:
    - `b`: A pointer to a `block_iq4_nl` structure from which data is to be loaded.
- **Control Flow**:
    - The function calls `load1(b)` to load the second 128 bits from the `block_iq4_nl` structure.
    - The function calls `load0(b)` to load the first 128 bits from the `block_iq4_nl` structure.
    - The function combines the two 128-bit results into a single 256-bit AVX2 register using `MM256_SET_M128I`.
- **Output**: A 256-bit AVX2 integer register (`__m256i`) containing the combined data from the `block_iq4_nl` structure.
- **Functions called**:
    - [`(anonymous)::tinyBLAS_Q0_AVX::load1`](#(anonymous)::tinyBLAS_Q0_AVX::load1)
    - [`(anonymous)::tinyBLAS_Q0_AVX::load0`](#(anonymous)::tinyBLAS_Q0_AVX::load0)
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::load0<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::load0}} -->
The `load0` function loads a 128-bit integer from a block of type `block_iq4_nl` and shuffles its bytes using a predefined lookup table.
- **Inputs**:
    - `b`: A pointer to a `block_iq4_nl` structure, which contains the data to be loaded and processed.
- **Control Flow**:
    - Load a 128-bit integer from the `qs` field of the `block_iq4_nl` structure pointed to by `b` using `_mm_loadu_si128`.
    - Perform a bitwise AND operation between the loaded integer and a constant mask of 15 using `_mm_and_si128`.
    - Shuffle the bytes of the result using `_mm_shuffle_epi8` with the predefined lookup table `iq4nlt`.
- **Output**: Returns a 128-bit integer (`__m128i`) with its bytes shuffled according to the lookup table `iq4nlt`.
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::load1<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::load1}} -->
The `load1` function loads a 128-bit integer from a specified block and shuffles its bytes using a predefined lookup table.
- **Inputs**:
    - `b`: A pointer to a `block_iq4_nl` structure, which contains the data to be loaded and processed.
- **Control Flow**:
    - Load a 128-bit integer from the memory location pointed to by `b->qs` using `_mm_loadu_si128`.
    - Perform a right logical shift by 4 bits on each 16-bit element of the loaded integer using `_mm_srli_epi16`.
    - Mask the result with 15 (0x0F) using `_mm_and_si128` to isolate the lower 4 bits of each byte.
    - Shuffle the bytes of the lookup table `iq4nlt` using `_mm_shuffle_epi8` with the masked result as the shuffle control mask.
    - Return the shuffled 128-bit integer.
- **Output**: A 128-bit integer (`__m128i`) that is the result of shuffling the bytes of the lookup table `iq4nlt` based on the processed input data.
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::updot<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::updot}} -->
The `updot` function computes a dot product of two 256-bit integer vectors and returns the result as a 256-bit floating-point vector.
- **Inputs**:
    - `u`: A 256-bit integer vector (__m256i) representing the first operand for the dot product.
    - `s`: A 256-bit integer vector (__m256i) representing the second operand for the dot product.
- **Control Flow**:
    - Initialize a 256-bit integer vector `res` to store the intermediate result.
    - Check if the AVX512VNNI and AVX512VL instruction sets are available; if so, use `_mm256_dpbusd_epi32` to compute the dot product of `u` and `s` with zero initialization.
    - If AVX512VNNI is not available but AVXVNNI is, use `_mm256_dpbusd_avx_epi32` for the dot product.
    - If neither AVX512VNNI nor AVXVNNI is available, use `_mm256_madd_epi16` and `_mm256_maddubs_epi16` to compute the dot product.
    - Convert the resulting 256-bit integer vector `res` to a 256-bit floating-point vector using `_mm256_cvtepi32_ps`.
- **Output**: A 256-bit floating-point vector (__m256) representing the dot product of the input vectors `u` and `s`.
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::denibble<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::denibble}} -->
The `denibble` function extracts the lower 4 bits (nibble) from each byte in a 128-bit block of data and returns it as a 256-bit AVX2 vector.
- **Inputs**:
    - `p`: A pointer to a block of data of type `uint8_t`, which is expected to be at least 16 bytes long.
- **Control Flow**:
    - Load 128 bits of data from the memory location pointed to by `p` into an `__m128i` variable `x`.
    - Shift the bits in `x` right by 4 positions to isolate the upper nibbles.
    - Insert the shifted result into a 256-bit AVX2 vector, placing it in the upper 128 bits.
    - Perform a bitwise AND operation with a 256-bit vector where all bytes are set to 15 (0x0F) to extract the lower nibbles from both the original and shifted data.
    - Return the resulting 256-bit vector.
- **Output**: A 256-bit AVX2 vector (`__m256i`) containing the lower 4 bits of each byte from the input data, duplicated across the lower and upper 128-bit lanes.
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)


---
#### tinyBLAS\_Q0\_AVX::bittobyte<!-- {{#callable:(anonymous)::tinyBLAS_Q0_AVX::bittobyte}} -->
The `bittobyte` function converts a 4-byte sequence from a given pointer into a 256-bit vector with specific bit manipulations using AVX2 instructions.
- **Inputs**:
    - `p`: A pointer to a constant uint8_t array, representing the input byte sequence to be processed.
- **Control Flow**:
    - Copy 4 bytes from the input pointer `p` into a 32-bit unsigned integer `x32`.
    - Create a 256-bit vector `bytes` by comparing each byte of a shuffled version of `x32` with a specific pattern using AVX2 instructions.
    - Perform a bitwise AND NOT operation on `bytes` with a constant vector to mask certain bits.
    - Return the resulting 256-bit vector.
- **Output**: A 256-bit vector (__m256i) with specific bits set based on the input byte sequence and the applied bit manipulations.
- **See also**: [`(anonymous)::tinyBLAS_Q0_AVX`](#(anonymous)::tinyBLAS_Q0_AVX)  (Data Structure)



---
### tinyBLAS\_BF16\_PPC<!-- {{#data_structure:(anonymous)::tinyBLAS_BF16_PPC}} -->
- **Type**: `class`
- **Members**:
    - `A`: Pointer to the first input matrix.
    - `B`: Pointer to the second input matrix.
    - `C`: Pointer to the output matrix.
    - `k`: Number of columns in A and rows in B.
    - `lda`: Leading dimension of matrix A.
    - `ldb`: Leading dimension of matrix B.
    - `ldc`: Leading dimension of matrix C.
    - `ith`: Thread index for parallel execution.
    - `nth`: Total number of threads for parallel execution.
- **Description**: The `tinyBLAS_BF16_PPC` class is a template class designed for performing matrix multiplication using the BF16 data type on PowerPC architectures. It is optimized for multithreaded execution, allowing efficient computation of matrix products by dividing the workload across multiple threads. The class stores pointers to the input matrices A and B, the output matrix C, and various parameters such as the dimensions of the matrices and the thread configuration. The class provides methods for packing matrices, performing vectorized operations, and executing the matrix multiplication using different kernel sizes based on the dimensions of the matrices.
- **Member Functions**:
    - [`(anonymous)::tinyBLAS_BF16_PPC::tinyBLAS_BF16_PPC`](#(anonymous)::tinyBLAS_BF16_PPC::tinyBLAS_BF16_PPC)
    - [`(anonymous)::tinyBLAS_BF16_PPC::matmul`](#(anonymous)::tinyBLAS_BF16_PPC::matmul)
    - [`(anonymous)::tinyBLAS_BF16_PPC::vector_permute_store`](#(anonymous)::tinyBLAS_BF16_PPC::vector_permute_store)
    - [`(anonymous)::tinyBLAS_BF16_PPC::packNormal`](#(anonymous)::tinyBLAS_BF16_PPC::packNormal)
    - [`(anonymous)::tinyBLAS_BF16_PPC::mnpack`](#(anonymous)::tinyBLAS_BF16_PPC::mnpack)
    - [`(anonymous)::tinyBLAS_BF16_PPC::KERNEL_4x8`](#(anonymous)::tinyBLAS_BF16_PPC::KERNEL_4x8)
    - [`(anonymous)::tinyBLAS_BF16_PPC::KERNEL_8x4`](#(anonymous)::tinyBLAS_BF16_PPC::KERNEL_8x4)
    - [`(anonymous)::tinyBLAS_BF16_PPC::KERNEL_8x8`](#(anonymous)::tinyBLAS_BF16_PPC::KERNEL_8x8)
    - [`(anonymous)::tinyBLAS_BF16_PPC::gemm_small`](#(anonymous)::tinyBLAS_BF16_PPC::gemm_small)
    - [`(anonymous)::tinyBLAS_BF16_PPC::gemm_Mx8`](#(anonymous)::tinyBLAS_BF16_PPC::gemm_Mx8)
    - [`(anonymous)::tinyBLAS_BF16_PPC::kernel`](#(anonymous)::tinyBLAS_BF16_PPC::kernel)
    - [`(anonymous)::tinyBLAS_BF16_PPC::gemm`](#(anonymous)::tinyBLAS_BF16_PPC::gemm)

**Methods**

---
#### tinyBLAS\_BF16\_PPC::tinyBLAS\_BF16\_PPC<!-- {{#callable:(anonymous)::tinyBLAS_BF16_PPC::tinyBLAS_BF16_PPC}} -->
The `tinyBLAS_BF16_PPC` constructor initializes a matrix multiplication object for BF16 data type on PowerPC architecture with given matrix dimensions and thread information.
- **Inputs**:
    - `k`: The number of columns in matrix A and rows in matrix B.
    - `A`: Pointer to the first input matrix A, which is transposed.
    - `lda`: Leading dimension of matrix A, representing the row stride.
    - `B`: Pointer to the second input matrix B.
    - `ldb`: Leading dimension of matrix B, representing the row stride.
    - `C`: Pointer to the output matrix C.
    - `ldc`: Leading dimension of matrix C, representing the row stride.
    - `ith`: The thread index for parallel execution.
    - `nth`: The total number of threads for parallel execution.
- **Control Flow**:
    - The constructor initializes member variables with the provided arguments.
    - It sets up the object to perform matrix multiplication using the BF16 data type on PowerPC architecture.
- **Output**: The constructor does not return any value; it initializes the object state for subsequent matrix multiplication operations.
- **See also**: [`(anonymous)::tinyBLAS_BF16_PPC`](#(anonymous)::tinyBLAS_BF16_PPC)  (Data Structure)


---
#### tinyBLAS\_BF16\_PPC::matmul<!-- {{#callable:(anonymous)::tinyBLAS_BF16_PPC::matmul}} -->
The `matmul` function initiates a matrix multiplication operation by calling the [`mnpack`](#(anonymous)::tinyBLAS::mnpack) method with specified row and column indices.
- **Inputs**:
    - `m`: The number of rows in the matrix.
    - `n`: The number of columns in the matrix.
- **Control Flow**:
    - The function calls the [`mnpack`](#(anonymous)::tinyBLAS::mnpack) method with parameters (0, m, 0, n).
- **Output**: The function does not return any value; it performs an operation as a side effect.
- **Functions called**:
    - [`(anonymous)::tinyBLAS::mnpack`](#(anonymous)::tinyBLAS::mnpack)
- **See also**: [`(anonymous)::tinyBLAS_BF16_PPC`](#(anonymous)::tinyBLAS_BF16_PPC)  (Data Structure)


---
#### tinyBLAS\_BF16\_PPC::vector\_permute\_store<!-- {{#callable:(anonymous)::tinyBLAS_BF16_PPC::vector_permute_store}} -->
The `vector_permute_store` function performs vector permutation and storage operations on input vectors based on the specified number of vectors (`numVec`).
- **Inputs**:
    - `c`: A pointer to an array of `vec_t` vectors, which are the input vectors to be permuted and stored.
    - `numVec`: An integer specifying the number of vectors to process, which can be 2, 4, or 8.
    - `vecOffset`: A pointer to an unsigned char array where the permuted vectors will be stored.
- **Control Flow**:
    - Initialize temporary vectors `t` and `s` and define permutation patterns `swiz1`, `swiz2`, `swiz3`, and `swiz4`.
    - Check if `numVec` is 2, 4, or 8 and execute the corresponding permutation and storage logic.
    - For `numVec == 2`, perform permutations using `swiz1` and `swiz3/swiz4`, then store the results in `vecOffset`.
    - For `numVec == 4`, perform permutations using `swiz1`, `swiz2`, `swiz3`, and `swiz4`, then store the results in `vecOffset`.
    - For `numVec == 8`, perform permutations in two loops for the first and second halves of the vectors using `swiz1`, `swiz2`, `swiz3`, and `swiz4`, then store the results in `vecOffset`.
- **Output**: The function does not return a value; it stores the permuted vectors in the memory location pointed to by `vecOffset`.
- **See also**: [`(anonymous)::tinyBLAS_BF16_PPC`](#(anonymous)::tinyBLAS_BF16_PPC)  (Data Structure)


---
#### tinyBLAS\_BF16\_PPC::packNormal<!-- {{#callable:(anonymous)::tinyBLAS_BF16_PPC::packNormal}} -->
The `packNormal` function efficiently packs a matrix into a vector format using vectorized operations, optimizing for different row and column configurations.
- **Inputs**:
    - `a`: A pointer to the input matrix of type `TA`.
    - `lda`: The leading dimension of the matrix `a`, indicating the number of elements between successive rows.
    - `rows`: The number of rows in the matrix to be packed.
    - `cols`: The number of columns in the matrix to be packed.
    - `vec`: A pointer to the output vector where the packed matrix will be stored.
- **Control Flow**:
    - Initialize pointers and arrays for offsets and vectorized data storage.
    - Calculate the number of 8-row blocks in the matrix and iterate over them.
    - For each 8-row block, handle the case where the number of columns is 4 by loading data into vector registers and storing it using a permutation store function.
    - For columns greater than 4, iterate over 8-column blocks, load data into vector registers, and store it using a permutation store function.
    - Handle remaining rows in groups of 4 or fewer, using similar vectorized operations and conditional logic to manage different column configurations.
    - Adjust pointers and offsets accordingly after processing each block of data.
- **Output**: The function does not return a value; it modifies the `vec` array in place to store the packed matrix data.
- **Functions called**:
    - [`(anonymous)::tinyBLAS_BF16_PPC::vector_permute_store`](#(anonymous)::tinyBLAS_BF16_PPC::vector_permute_store)
- **See also**: [`(anonymous)::tinyBLAS_BF16_PPC`](#(anonymous)::tinyBLAS_BF16_PPC)  (Data Structure)


---
#### tinyBLAS\_BF16\_PPC::mnpack<!-- {{#callable:(anonymous)::tinyBLAS_BF16_PPC::mnpack}} -->
The [`mnpack`](#(anonymous)::tinyBLAS::mnpack) function recursively partitions a matrix multiplication problem into smaller subproblems and calls optimized GEMM (General Matrix Multiply) functions based on the dimensions of the submatrices.
- **Inputs**:
    - `m0`: The starting row index for the submatrix of matrix A.
    - `m`: The ending row index for the submatrix of matrix A.
    - `n0`: The starting column index for the submatrix of matrix B.
    - `n`: The ending column index for the submatrix of matrix B.
- **Control Flow**:
    - Calculate the remaining rows (`m_rem`) and columns (`n_rem`) to be processed, limited to a maximum of 8.
    - Based on the values of `m_rem` and `n_rem`, select an appropriate GEMM function to handle the current submatrix dimensions.
    - If both `m_rem` and `n_rem` are at least 8, call `gemm<8,8>`; otherwise, choose smaller block sizes and call the corresponding GEMM function.
    - If `m_rem` is less than 4 and `n_rem` is at least 8, use a switch statement to handle cases for `m_rem` values of 1, 2, or 3, calling `gemm_Mx8` with the appropriate template parameter.
    - For other combinations of `m_rem` and `n_rem`, use a switch statement to select the appropriate `gemm_small` function based on the combined value of `(m_rem << 4) | n_rem`.
    - Calculate the next partition points `mp` and `np` for further recursive calls.
    - Recursively call [`mnpack`](#(anonymous)::tinyBLAS::mnpack) for the submatrices defined by the new partition points `mp` and `np`.
- **Output**: The function does not return a value; it performs matrix multiplication on submatrices and updates the result matrix in place.
- **Functions called**:
    - [`(anonymous)::tinyBLAS::mnpack`](#(anonymous)::tinyBLAS::mnpack)
- **See also**: [`(anonymous)::tinyBLAS_BF16_PPC`](#(anonymous)::tinyBLAS_BF16_PPC)  (Data Structure)


---
#### tinyBLAS\_BF16\_PPC::KERNEL\_4x8<!-- {{#callable:(anonymous)::tinyBLAS_BF16_PPC::KERNEL_4x8}} -->
The `KERNEL_4x8` function performs a matrix multiplication operation on 4x8 blocks of matrices A and B, accumulating the results into two accumulators, and then saves the results into matrix C.
- **Inputs**:
    - `ii`: The row index offset for matrix A.
    - `jj`: The column index offset for matrix B.
- **Control Flow**:
    - Initialize vectors `vec_A`, `vec_B`, and `vec_C` for storing matrix blocks and accumulators `acc_0` and `acc_1` for results.
    - Set accumulators `acc_0` and `acc_1` to zero using `__builtin_mma_xxsetaccz`.
    - Iterate over the matrix dimension `k` in steps of 8.
    - In each iteration, pack 4x8 block of matrix A and 8x8 block of matrix B into `vec_A` and `vec_B` respectively using [`packNormal`](#(anonymous)::tinyBLAS_BF16_PPC::packNormal).
    - For each of the 4 rows in `vec_A`, perform matrix multiplication with corresponding columns in `vec_B` and accumulate results in `acc_0` and `acc_1` using `__builtin_mma_xvbf16ger2pp`.
    - Save the accumulated results from `acc_0` and `acc_1` into matrix C at positions determined by `ii` and `jj` using `SAVE_ACC`.
- **Output**: The function does not return a value; it modifies the matrix C by saving the accumulated results from the matrix multiplication.
- **Functions called**:
    - [`(anonymous)::tinyBLAS_BF16_PPC::packNormal`](#(anonymous)::tinyBLAS_BF16_PPC::packNormal)
- **See also**: [`(anonymous)::tinyBLAS_BF16_PPC`](#(anonymous)::tinyBLAS_BF16_PPC)  (Data Structure)


---
#### tinyBLAS\_BF16\_PPC::KERNEL\_8x4<!-- {{#callable:(anonymous)::tinyBLAS_BF16_PPC::KERNEL_8x4}} -->
The `KERNEL_8x4` function performs a matrix multiplication operation on 8x4 blocks of matrices A and B, accumulating the results into two accumulators and storing the results in matrix C.
- **Inputs**:
    - `ii`: The row index offset for matrix A.
    - `jj`: The column index offset for matrix B.
- **Control Flow**:
    - Initialize vectors vec_A, vec_B, and vec_C for storing matrix blocks and accumulators acc_0 and acc_1 for results.
    - Set accumulators acc_0 and acc_1 to zero using `__builtin_mma_xxsetaccz`.
    - Iterate over the matrix dimension `k` in steps of 8.
    - For each iteration, pack 8x8 block of matrix A and 8x4 block of matrix B into vec_A and vec_B respectively using [`packNormal`](#(anonymous)::tinyBLAS_BF16_PPC::packNormal).
    - Perform matrix multiplication using `__builtin_mma_xvbf16ger2pp` to update accumulators acc_0 and acc_1 with the results of the outer product of vec_A and vec_B.
    - Store the results from accumulators acc_0 and acc_1 into matrix C using `SAVE_ACC`.
- **Output**: The function does not return a value; it modifies the matrix C in place by storing the results of the matrix multiplication.
- **Functions called**:
    - [`(anonymous)::tinyBLAS_BF16_PPC::packNormal`](#(anonymous)::tinyBLAS_BF16_PPC::packNormal)
- **See also**: [`(anonymous)::tinyBLAS_BF16_PPC`](#(anonymous)::tinyBLAS_BF16_PPC)  (Data Structure)


---
#### tinyBLAS\_BF16\_PPC::KERNEL\_8x8<!-- {{#callable:(anonymous)::tinyBLAS_BF16_PPC::KERNEL_8x8}} -->
The `KERNEL_8x8` function performs an 8x8 matrix multiplication using vectorized operations and accumulates the results into four accumulators, which are then saved to the output matrix.
- **Inputs**:
    - `ii`: The row index offset for matrix A.
    - `jj`: The column index offset for matrix B.
- **Control Flow**:
    - Initialize four vector accumulators to zero using `__builtin_mma_xxsetaccz`.
    - Iterate over the range of `k` in steps of 8, packing 8x8 blocks of matrices A and B into vectors `vec_A` and `vec_B` respectively using [`packNormal`](#(anonymous)::tinyBLAS_BF16_PPC::packNormal).
    - For each of the first four elements in the packed vectors, perform vectorized multiply-accumulate operations using `__builtin_mma_xvbf16ger2pp` to update the four accumulators.
    - Save the results from the accumulators into the output matrix C using the `SAVE_ACC` macro.
- **Output**: The function does not return a value; it modifies the output matrix C by saving the accumulated results from the vector operations.
- **Functions called**:
    - [`(anonymous)::tinyBLAS_BF16_PPC::packNormal`](#(anonymous)::tinyBLAS_BF16_PPC::packNormal)
- **See also**: [`(anonymous)::tinyBLAS_BF16_PPC`](#(anonymous)::tinyBLAS_BF16_PPC)  (Data Structure)


---
#### tinyBLAS\_BF16\_PPC::gemm\_small<!-- {{#callable:(anonymous)::tinyBLAS_BF16_PPC::gemm_small}} -->
The `gemm_small` function performs a small-scale matrix multiplication using a specified number of rows and columns per tile, optimized for parallel execution across multiple threads.
- **Inputs**:
    - `m0`: The starting row index for the matrix multiplication.
    - `m`: The ending row index for the matrix multiplication.
    - `n0`: The starting column index for the matrix multiplication.
    - `n`: The ending column index for the matrix multiplication.
- **Control Flow**:
    - Calculate the number of tiles in the y and x dimensions based on the input dimensions and the template parameters RM and RN.
    - Determine the total number of tiles and the number of tiles each thread is responsible for (duty).
    - Calculate the start and end indices for the current thread's tile processing range.
    - Iterate over each tile assigned to the current thread, calculating the starting indices for the sub-matrix multiplication.
    - Initialize the accumulator to zero for the matrix multiplication.
    - Pack the sub-matrices A and B into vector registers for efficient processing.
    - Perform the matrix multiplication using vectorized operations and accumulate the results.
    - Disassemble the accumulator into a vector and store the results back into the matrix C.
- **Output**: The function does not return a value; it modifies the matrix C in place with the results of the matrix multiplication.
- **Functions called**:
    - [`(anonymous)::tinyBLAS_BF16_PPC::packNormal`](#(anonymous)::tinyBLAS_BF16_PPC::packNormal)
- **See also**: [`(anonymous)::tinyBLAS_BF16_PPC`](#(anonymous)::tinyBLAS_BF16_PPC)  (Data Structure)


---
#### tinyBLAS\_BF16\_PPC::gemm\_Mx8<!-- {{#callable:(anonymous)::tinyBLAS_BF16_PPC::gemm_Mx8}} -->
The `gemm_Mx8` function performs a matrix multiplication operation on a submatrix of dimensions RMx8 using PowerPC's MMA (Matrix-Multiply Assist) instructions.
- **Inputs**:
    - `m0`: The starting row index for the submatrix in matrix A.
    - `m`: The ending row index for the submatrix in matrix A.
    - `n0`: The starting column index for the submatrix in matrix B.
    - `n`: The ending column index for the submatrix in matrix B.
- **Control Flow**:
    - Calculate the number of tiles in the y-direction (`ytiles`) and x-direction (`xtiles`) based on the dimensions of the submatrix and the tile size (RMx8).
    - Determine the total number of tiles (`tiles`) and the number of tiles each thread should process (`duty`).
    - Calculate the starting (`start`) and ending (`end`) tile indices for the current thread based on its ID (`ith`) and the total number of threads (`nth`).
    - Iterate over each tile assigned to the current thread, calculating the starting indices (`ii`, `jj`) for the submatrix in matrices A and B.
    - Initialize two accumulators (`acc_0`, `acc_1`) to zero using MMA instructions.
    - For each block of 8 columns in the submatrix, pack the corresponding rows from matrices A and B into vector registers (`vec_A`, `vec_B`).
    - Perform matrix multiplication using MMA instructions to update the accumulators with the results of the outer product of `vec_A` and `vec_B`.
    - Disassemble the accumulators into result vectors (`vec_C`) and store the results back into the appropriate positions in matrix C.
- **Output**: The function does not return a value; it modifies the matrix C in place with the results of the matrix multiplication.
- **Functions called**:
    - [`(anonymous)::tinyBLAS_BF16_PPC::packNormal`](#(anonymous)::tinyBLAS_BF16_PPC::packNormal)
- **See also**: [`(anonymous)::tinyBLAS_BF16_PPC`](#(anonymous)::tinyBLAS_BF16_PPC)  (Data Structure)


---
#### tinyBLAS\_BF16\_PPC::kernel<!-- {{#callable:(anonymous)::tinyBLAS_BF16_PPC::kernel}} -->
The `kernel` function selects and executes a specific matrix multiplication kernel based on the template parameters `RM` and `RN`.
- **Inputs**:
    - `ii`: An integer representing the starting row index for the matrix operation.
    - `jj`: An integer representing the starting column index for the matrix operation.
- **Control Flow**:
    - The function uses `if constexpr` to check if `RM` and `RN` match specific values (4x8, 8x8, or 8x4).
    - If `RM` and `RN` are 4 and 8, respectively, it calls `KERNEL_4x8(ii, jj)`.
    - If `RM` and `RN` are 8 and 8, respectively, it calls `KERNEL_8x8(ii, jj)`.
    - If `RM` and `RN` are 8 and 4, respectively, it calls `KERNEL_8x4(ii, jj)`.
    - If none of the conditions are met, a static assertion fails, indicating unsupported `RM`/`RN` values.
- **Output**: The function does not return a value; it performs operations based on the selected kernel.
- **Functions called**:
    - [`(anonymous)::tinyBLAS_BF16_PPC::KERNEL_4x8`](#(anonymous)::tinyBLAS_BF16_PPC::KERNEL_4x8)
    - [`(anonymous)::tinyBLAS_BF16_PPC::KERNEL_8x8`](#(anonymous)::tinyBLAS_BF16_PPC::KERNEL_8x8)
    - [`(anonymous)::tinyBLAS_BF16_PPC::KERNEL_8x4`](#(anonymous)::tinyBLAS_BF16_PPC::KERNEL_8x4)
- **See also**: [`(anonymous)::tinyBLAS_BF16_PPC`](#(anonymous)::tinyBLAS_BF16_PPC)  (Data Structure)


---
#### tinyBLAS\_BF16\_PPC::gemm<!-- {{#callable:(anonymous)::tinyBLAS_BF16_PPC::gemm}} -->
The `gemm` function performs a tiled matrix multiplication using a specified number of rows and columns per tile, distributing the work across multiple threads.
- **Inputs**:
    - `m0`: The starting row index for the matrix multiplication.
    - `m`: The ending row index for the matrix multiplication.
    - `n0`: The starting column index for the matrix multiplication.
    - `n`: The ending column index for the matrix multiplication.
- **Control Flow**:
    - Calculate the number of tiles in the y-direction (`ytiles`) and x-direction (`xtiles`) based on the input dimensions and template parameters `RM` and `RN`.
    - Compute the total number of tiles (`tiles`) as the product of `ytiles` and `xtiles`.
    - Determine the number of tiles each thread should process (`duty`) by dividing `tiles` by the number of threads (`nth`).
    - Calculate the starting (`start`) and ending (`end`) tile indices for the current thread based on its index (`ith`).
    - Iterate over the range of tiles assigned to the current thread, calculating the starting row (`ii`) and column (`jj`) for each tile.
    - Invoke the `kernel` function with the calculated row and column indices to perform the matrix multiplication for each tile.
- **Output**: The function does not return a value; it performs matrix multiplication and stores the result in a pre-defined location.
- **See also**: [`(anonymous)::tinyBLAS_BF16_PPC`](#(anonymous)::tinyBLAS_BF16_PPC)  (Data Structure)



---
### tinyBLAS\_Q0\_PPC<!-- {{#data_structure:(anonymous)::tinyBLAS_Q0_PPC}} -->
- **Type**: `class`
- **Members**:
    - `A`: Pointer to the first input matrix of type TA.
    - `B`: Pointer to the second input matrix of type TB.
    - `C`: Pointer to the output matrix of type TC.
    - `k`: The number of columns in A and rows in B.
    - `lda`: Leading dimension of matrix A.
    - `ldb`: Leading dimension of matrix B.
    - `ldc`: Leading dimension of matrix C.
    - `ith`: Thread index for parallel execution.
    - `nth`: Total number of threads for parallel execution.
- **Description**: The `tinyBLAS_Q0_PPC` class is a template-based C++ class designed for performing optimized matrix multiplication on PowerPC architectures using vectorized operations. It supports multithreaded execution and is specialized for handling matrices with specific data types (TA, TB, TC). The class encapsulates pointers to input matrices A and B, and the output matrix C, along with their respective leading dimensions. It also manages the number of threads and the current thread index for parallel processing. The class provides methods for packing matrices, computing matrix products, and saving results, leveraging PowerPC's vector processing capabilities to enhance performance.
- **Member Functions**:
    - [`(anonymous)::tinyBLAS_Q0_PPC::tinyBLAS_Q0_PPC`](#(anonymous)::tinyBLAS_Q0_PPC::tinyBLAS_Q0_PPC)
    - [`(anonymous)::tinyBLAS_Q0_PPC::matmul`](#(anonymous)::tinyBLAS_Q0_PPC::matmul)
    - [`(anonymous)::tinyBLAS_Q0_PPC::save_res`](#(anonymous)::tinyBLAS_Q0_PPC::save_res)
    - [`(anonymous)::tinyBLAS_Q0_PPC::compute`](#(anonymous)::tinyBLAS_Q0_PPC::compute)
    - [`(anonymous)::tinyBLAS_Q0_PPC::packNormalInt4`](#(anonymous)::tinyBLAS_Q0_PPC::packNormalInt4)
    - [`(anonymous)::tinyBLAS_Q0_PPC::packNormal`](#(anonymous)::tinyBLAS_Q0_PPC::packNormal)
    - [`(anonymous)::tinyBLAS_Q0_PPC::mnpack`](#(anonymous)::tinyBLAS_Q0_PPC::mnpack)
    - [`(anonymous)::tinyBLAS_Q0_PPC::KERNEL_4x8`](#(anonymous)::tinyBLAS_Q0_PPC::KERNEL_4x8)
    - [`(anonymous)::tinyBLAS_Q0_PPC::KERNEL_8x4`](#(anonymous)::tinyBLAS_Q0_PPC::KERNEL_8x4)
    - [`(anonymous)::tinyBLAS_Q0_PPC::KERNEL_8x8`](#(anonymous)::tinyBLAS_Q0_PPC::KERNEL_8x8)
    - [`(anonymous)::tinyBLAS_Q0_PPC::gemm_small`](#(anonymous)::tinyBLAS_Q0_PPC::gemm_small)
    - [`(anonymous)::tinyBLAS_Q0_PPC::kernel`](#(anonymous)::tinyBLAS_Q0_PPC::kernel)
    - [`(anonymous)::tinyBLAS_Q0_PPC::gemm`](#(anonymous)::tinyBLAS_Q0_PPC::gemm)

**Methods**

---
#### tinyBLAS\_Q0\_PPC::tinyBLAS\_Q0\_PPC<!-- {{#callable:(anonymous)::tinyBLAS_Q0_PPC::tinyBLAS_Q0_PPC}} -->
The `tinyBLAS_Q0_PPC` constructor initializes a matrix multiplication object with given matrices and parameters for parallel processing.
- **Inputs**:
    - `k`: The number of columns in matrix A and rows in matrix B.
    - `A`: Pointer to the first input matrix, which is transposed.
    - `lda`: Leading dimension of matrix A, representing the row stride.
    - `B`: Pointer to the second input matrix, which is not transposed.
    - `ldb`: Leading dimension of matrix B, representing the row stride.
    - `C`: Pointer to the output matrix where the result will be stored.
    - `ldc`: Leading dimension of matrix C, representing the row stride.
    - `ith`: The thread index for parallel processing.
    - `nth`: The total number of threads for parallel processing.
- **Control Flow**:
    - The constructor initializes member variables with the provided arguments.
    - It sets up the object to perform matrix multiplication using the specified matrices and dimensions.
    - The constructor does not perform any computation itself; it only prepares the object for later operations.
- **Output**: The constructor does not return any value; it initializes the object state for matrix multiplication.
- **See also**: [`(anonymous)::tinyBLAS_Q0_PPC`](#(anonymous)::tinyBLAS_Q0_PPC)  (Data Structure)


---
#### tinyBLAS\_Q0\_PPC::matmul<!-- {{#callable:(anonymous)::tinyBLAS_Q0_PPC::matmul}} -->
The `matmul` function initiates a matrix multiplication operation by calling the [`mnpack`](#(anonymous)::tinyBLAS::mnpack) function with specified row and column indices.
- **Inputs**:
    - `m`: The number of rows in the matrix.
    - `n`: The number of columns in the matrix.
- **Control Flow**:
    - The function `matmul` is called with two parameters: `m` and `n`, representing the dimensions of the matrix.
    - Inside `matmul`, the function [`mnpack`](#(anonymous)::tinyBLAS::mnpack) is called with the parameters `0, m, 0, n`, which likely initiates a matrix packing or multiplication process starting from the top-left corner of the matrix.
- **Output**: The function does not return any value; it performs its operations internally, likely affecting the state of the class instance or the matrices involved.
- **Functions called**:
    - [`(anonymous)::tinyBLAS::mnpack`](#(anonymous)::tinyBLAS::mnpack)
- **See also**: [`(anonymous)::tinyBLAS_Q0_PPC`](#(anonymous)::tinyBLAS_Q0_PPC)  (Data Structure)


---
#### tinyBLAS\_Q0\_PPC::save\_res<!-- {{#callable:(anonymous)::tinyBLAS_Q0_PPC::save_res}} -->
The `save_res` function stores computed results from a vector of floats into a specific location in a matrix `C` using given indices and dimensions.
- **Inputs**:
    - `ii`: The starting row index in matrix `C` where the results will be stored.
    - `jj`: The starting column index in matrix `C` where the results will be stored.
    - `idx`: The starting index in the `fin_res` vector from which results will be read.
    - `fin_res`: A pointer to a vector of floats containing the computed results to be stored in matrix `C`.
- **Control Flow**:
    - The function iterates over two nested loops: the outer loop runs `RM` times and the inner loop runs `RN` times, where `RM` and `RN` are template parameters.
    - For each combination of `I` and `J` from the loops, it calculates the position in matrix `C` using the formula `C + ii + ((jj + J) * ldc) + I`.
    - It then assigns the value from `fin_res[idx + I] + J` to the calculated position in `C`.
- **Output**: The function does not return a value; it modifies the matrix `C` in place.
- **See also**: [`(anonymous)::tinyBLAS_Q0_PPC`](#(anonymous)::tinyBLAS_Q0_PPC)  (Data Structure)


---
#### tinyBLAS\_Q0\_PPC::compute<!-- {{#callable:(anonymous)::tinyBLAS_Q0_PPC::compute}} -->
The `compute` function performs vectorized arithmetic operations on input data to update the final result vector using matrix multiplication and addition.
- **Inputs**:
    - `ACC`: A pointer to an accumulator of type `acc_t` which holds intermediate results of matrix multiplication.
    - `c_idx`: An integer index indicating the starting position in the `comparray` for the current computation.
    - `s_idx`: An integer index indicating the starting position in the `fin_res` and `vs` arrays for the current computation.
    - `comparray`: A reference to a `std::array` of integers with a template size `size`, used to store precomputed values for the computation.
    - `vs`: A pointer to a vector of floats, representing a vector of scaling factors for the computation.
    - `fin_res`: A pointer to a vector of floats, representing the final result vector to be updated with the computation results.
- **Control Flow**:
    - Initialize vectors `vec_C`, `CA`, and `res` to hold intermediate computation values.
    - Disassemble the accumulator `ACC` into `vec_C` using `__builtin_mma_disassemble_acc`.
    - Iterate over a loop of size 4 to perform computations for each vector element.
    - For each iteration, compute `CA[i]` by scaling and converting `comparray` values to floats, then multiply by -128.0.
    - Add the converted `vec_C[i]` to `CA[i]` and store the result in `res[i]`.
    - Update `fin_res[s_idx+i]` by performing a multiply-add operation with `res[i]`, `vs[s_idx+i]`, and the current value of `fin_res[s_idx+i]`.
- **Output**: The function updates the `fin_res` vector with the computed results, modifying it in place.
- **See also**: [`(anonymous)::tinyBLAS_Q0_PPC`](#(anonymous)::tinyBLAS_Q0_PPC)  (Data Structure)


---
#### tinyBLAS\_Q0\_PPC::packNormalInt4<!-- {{#callable:(anonymous)::tinyBLAS_Q0_PPC::packNormalInt4}} -->
The `packNormalInt4` function processes a matrix of type `TA` and packs it into a vector format suitable for SIMD operations, while also computing and storing sums of the processed data in a comparison array.
- **Inputs**:
    - `a`: A pointer to the input matrix of type `TA`.
    - `lda`: The leading dimension of the matrix `a`, representing the number of elements between successive rows.
    - `rows`: The number of rows in the matrix `a` to be processed.
    - `cols`: The number of columns in the matrix `a` to be processed.
    - `vec`: A pointer to the output vector of type `VA` where the packed data will be stored.
    - `comparray`: A reference to a `std::array` of integers where the computed sums of the processed data will be stored.
- **Control Flow**:
    - Initialize pointers and vectors for processing the matrix data.
    - Loop over the matrix rows in blocks of 8, adjusting pointers for each block.
    - Within each block, loop over the columns in blocks of 4, processing each set of 4 columns.
    - For each set of columns, load data into vector registers, apply bitwise operations, and compute sums.
    - Store the computed sums in the `comparray` and reset sum vectors for the next iteration.
    - Pack the processed data into the `vec` output using vector permutations and store operations.
    - Handle remaining rows and columns that do not fit into the 8x4 block structure.
- **Output**: The function does not return a value but modifies the `vec` and `comparray` to store the packed data and computed sums, respectively.
- **See also**: [`(anonymous)::tinyBLAS_Q0_PPC`](#(anonymous)::tinyBLAS_Q0_PPC)  (Data Structure)


---
#### tinyBLAS\_Q0\_PPC::packNormal<!-- {{#callable:(anonymous)::tinyBLAS_Q0_PPC::packNormal}} -->
The `packNormal` function optimizes the packing of matrix data into a vectorized format for efficient processing, with optional bit-flipping for data transformation.
- **Inputs**:
    - `a`: A pointer to the input matrix of type `TB`.
    - `lda`: The leading dimension of the matrix `a`, representing the number of elements between successive rows.
    - `rows`: The number of rows in the matrix `a` to be processed.
    - `cols`: The number of columns in the matrix `a` to be processed.
    - `vec`: A pointer to the output vector of type `VA` where the packed data will be stored.
    - `flip`: A boolean flag indicating whether to apply a bitwise XOR operation to flip the bits of the packed data.
- **Control Flow**:
    - Initialize pointers and vector variables for processing.
    - Calculate the number of 8-row blocks (`j`) and iterate over them.
    - For each block, set up row pointers (`aoffset1` to `aoffset8`) and increment the main offset (`aoffset`).
    - Calculate the number of 8-column blocks (`i`) and iterate over them.
    - Load data from the matrix into vector pairs (`C1` to `C8`) using `__builtin_vsx_lxvp`.
    - Disassemble vector pairs into individual vectors (`c1` to `c8`).
    - Perform vector permutations to rearrange data into the desired format (`t1` to `t8`).
    - If `flip` is true, apply XOR with `xor_vector` to flip bits of the vectors (`t5` to `t8`).
    - Store the packed vectors into the output vector `vec` at the appropriate offsets.
    - Repeat the process for remaining rows and columns that do not fit into 8-row or 8-column blocks.
- **Output**: The function does not return a value; it modifies the `vec` array in place to store the packed matrix data.
- **See also**: [`(anonymous)::tinyBLAS_Q0_PPC`](#(anonymous)::tinyBLAS_Q0_PPC)  (Data Structure)


---
#### tinyBLAS\_Q0\_PPC::mnpack<!-- {{#callable:(anonymous)::tinyBLAS_Q0_PPC::mnpack}} -->
The [`mnpack`](#(anonymous)::tinyBLAS::mnpack) function recursively partitions a matrix multiplication problem into smaller subproblems and applies optimized matrix multiplication kernels based on the dimensions of the subproblems.
- **Inputs**:
    - `m0`: The starting row index for the matrix multiplication.
    - `m`: The ending row index for the matrix multiplication.
    - `n0`: The starting column index for the matrix multiplication.
    - `n`: The ending column index for the matrix multiplication.
- **Control Flow**:
    - Calculate the remaining rows (`m_rem`) and columns (`n_rem`) to be processed, each limited to a maximum of 8.
    - Check various conditions based on `m_rem` and `n_rem` to determine the appropriate kernel size for matrix multiplication.
    - If `m_rem` and `n_rem` are both at least 8, use an 8x8 kernel.
    - If `m_rem` is at least 4 and `n_rem` is at least 8, use a 4x8 kernel.
    - If `m_rem` is at least 8 and `n_rem` is at least 4, use an 8x4 kernel.
    - If both `m_rem` and `n_rem` are at least 4, use a 4x4 small kernel.
    - For smaller values of `m_rem` or `n_rem`, use a switch-case to select the appropriate small kernel based on the combination of `m_rem` and `n_rem`.
    - Calculate the next partition points `mp` and `np` for further recursive calls.
    - Recursively call [`mnpack`](#(anonymous)::tinyBLAS::mnpack) for the subproblems defined by the new partition points.
- **Output**: The function does not return a value; it performs matrix multiplication operations recursively.
- **Functions called**:
    - [`(anonymous)::tinyBLAS::mnpack`](#(anonymous)::tinyBLAS::mnpack)
- **See also**: [`(anonymous)::tinyBLAS_Q0_PPC`](#(anonymous)::tinyBLAS_Q0_PPC)  (Data Structure)


---
#### tinyBLAS\_Q0\_PPC::KERNEL\_4x8<!-- {{#callable:(anonymous)::tinyBLAS_Q0_PPC::KERNEL_4x8}} -->
The `KERNEL_4x8` function performs a matrix multiplication operation on a 4x8 block of matrices A and B, using vectorized operations and accumulators to compute the result, which is then stored in matrix C.
- **Inputs**:
    - `ii`: The row index offset for matrix A.
    - `jj`: The column index offset for matrix B.
- **Control Flow**:
    - Initialize vector arrays `vec_A` and `vec_B` for storing packed matrix data, and accumulators `acc_0` and `acc_1` for intermediate results.
    - Determine if the matrix A is of type `block_q4_0` to decide the packing method for `vec_A`.
    - Iterate over the range of `k` to perform the matrix multiplication in blocks.
    - For each iteration, reset the accumulators `acc_0` and `acc_1` to zero using `__builtin_mma_xxsetaccz`.
    - Pack the matrix data from A and B into `vec_A` and `vec_B` using the appropriate packing function based on the type of A.
    - Perform vectorized multiply-accumulate operations using `__builtin_mma_xvi8ger4pp` to update the accumulators with the products of `vec_A` and `vec_B`.
    - Compute additional results using the `compute` function, which processes the accumulators and stores the results in `fin_res`.
    - Store the final results from `fin_res` into the matrix C using the `save_res` function.
- **Output**: The function does not return a value; it modifies the matrix C in place by storing the computed results.
- **Functions called**:
    - [`(anonymous)::unhalf`](#(anonymous)::unhalf)
- **See also**: [`(anonymous)::tinyBLAS_Q0_PPC`](#(anonymous)::tinyBLAS_Q0_PPC)  (Data Structure)


---
#### tinyBLAS\_Q0\_PPC::KERNEL\_8x4<!-- {{#callable:(anonymous)::tinyBLAS_Q0_PPC::KERNEL_8x4}} -->
The `KERNEL_8x4` function performs a matrix multiplication operation on 8x4 blocks of matrices A and B, using vectorized operations and accumulators to compute and store the results in matrix C.
- **Inputs**:
    - `ii`: The starting row index for the block of matrix A.
    - `jj`: The starting column index for the block of matrix B.
- **Control Flow**:
    - Initialize vector arrays `vec_A` and `vec_B` for storing packed data from matrices A and B, and accumulators `acc_0` and `acc_1` for storing intermediate results.
    - Determine if the type of matrix A is `block_q4_0` to decide the packing method for matrix A.
    - Iterate over the range of `k` to perform operations for each depth slice of the matrices.
    - For each iteration, reset the accumulators `acc_0` and `acc_1` to zero using `__builtin_mma_xxsetaccz`.
    - Pack the current slice of matrix A into `vec_A` using either `packNormalInt4` or `packNormal` based on the type of A.
    - Pack the current slice of matrix B into `vec_B` using `packNormal`.
    - Perform vectorized multiply-accumulate operations using `__builtin_mma_xvi8ger4pp` to update the accumulators with the products of `vec_A` and `vec_B`.
    - Compute the final results using the `compute` function, which processes the accumulators and stores the results in `fin_res`.
    - Store the computed results back into matrix C using the `save_res` function.
- **Output**: The function does not return a value; it modifies the matrix C in place by storing the computed results of the matrix multiplication.
- **Functions called**:
    - [`(anonymous)::unhalf`](#(anonymous)::unhalf)
- **See also**: [`(anonymous)::tinyBLAS_Q0_PPC`](#(anonymous)::tinyBLAS_Q0_PPC)  (Data Structure)


---
#### tinyBLAS\_Q0\_PPC::KERNEL\_8x8<!-- {{#callable:(anonymous)::tinyBLAS_Q0_PPC::KERNEL_8x8}} -->
The `KERNEL_8x8` function performs an 8x8 matrix multiplication using vectorized operations and accumulates the results into a final result matrix.
- **Inputs**:
    - `ii`: The starting row index for the sub-matrix of A.
    - `jj`: The starting column index for the sub-matrix of B.
- **Control Flow**:
    - Initialize vector arrays `vec_A` and `vec_B` for storing packed matrix data and accumulators `acc_0`, `acc_1`, `acc_2`, and `acc_3` for intermediate results.
    - Determine if the matrix A is of type `block_q4_0` to decide the packing method for A.
    - Iterate over the range of `k` to perform the matrix multiplication in blocks.
    - For each iteration, reset the accumulators to zero using `__builtin_mma_xxsetaccz`.
    - Pack the matrix A and B data into vector arrays `vec_A` and `vec_B` using `packNormalInt4` or `packNormal` based on the type of A.
    - Perform vectorized multiply-accumulate operations using `__builtin_mma_xvi8ger4pp` to update the accumulators with the products of `vec_A` and `vec_B`.
    - Compute the final results using the `compute` function, which adjusts the accumulated values based on a comparison array and stores them in `fin_res`.
    - Save the computed results into the output matrix C using the `save_res` function.
- **Output**: The function does not return a value; it modifies the output matrix C in place with the results of the matrix multiplication.
- **Functions called**:
    - [`(anonymous)::unhalf`](#(anonymous)::unhalf)
- **See also**: [`(anonymous)::tinyBLAS_Q0_PPC`](#(anonymous)::tinyBLAS_Q0_PPC)  (Data Structure)


---
#### tinyBLAS\_Q0\_PPC::gemm\_small<!-- {{#callable:(anonymous)::tinyBLAS_Q0_PPC::gemm_small}} -->
The `gemm_small` function performs a small-scale matrix multiplication using vectorized operations and multi-threading for efficiency.
- **Inputs**:
    - `m0`: The starting row index for the matrix A.
    - `m`: The ending row index for the matrix A.
    - `n0`: The starting column index for the matrix B.
    - `n`: The ending column index for the matrix B.
- **Control Flow**:
    - Calculate the number of tiles in the y and x dimensions based on the input dimensions and template parameters RM and RN.
    - Determine the total number of tiles and the duty (number of tiles per thread).
    - Calculate the start and end indices for the current thread's work based on its ID (ith) and the total number of threads (nth).
    - Initialize vector arrays for storing intermediate results and prefetch the first elements of matrices A and B.
    - Iterate over the assigned tiles, calculating the starting indices for each tile in matrices A and B.
    - For each tile, iterate over the shared dimension k, performing prefetching and vectorized operations to compute partial results.
    - Use vectorized operations to accumulate results into an accumulator and disassemble the accumulator into a vector for further processing.
    - Adjust results based on whether the matrix A is of type block_q4_0 and compute the final results using vectorized multiply-add operations.
    - Store the computed results back into the output matrix C using the save_res function.
- **Output**: The function does not return a value; it writes the computed matrix multiplication results into the output matrix C.
- **Functions called**:
    - [`(anonymous)::unhalf`](#(anonymous)::unhalf)
- **See also**: [`(anonymous)::tinyBLAS_Q0_PPC`](#(anonymous)::tinyBLAS_Q0_PPC)  (Data Structure)


---
#### tinyBLAS\_Q0\_PPC::kernel<!-- {{#callable:(anonymous)::tinyBLAS_Q0_PPC::kernel}} -->
The `kernel` function selects and executes a specific matrix multiplication kernel based on the template parameters `RM` and `RN`.
- **Inputs**:
    - `ii`: An integer representing the starting row index for the matrix operation.
    - `jj`: An integer representing the starting column index for the matrix operation.
- **Control Flow**:
    - The function uses `if constexpr` to check if `RM` and `RN` match specific values (4x8, 8x4, or 8x8).
    - If `RM` and `RN` are 4 and 8, respectively, it calls `KERNEL_4x8(ii, jj)`.
    - If `RM` and `RN` are 8 and 4, respectively, it calls `KERNEL_8x4(ii, jj)`.
    - If `RM` and `RN` are both 8, it calls `KERNEL_8x8(ii, jj)`.
    - If none of the conditions are met, a static assertion fails, indicating unsupported `RM`/`RN` values.
- **Output**: The function does not return a value; it performs operations based on the selected kernel.
- **Functions called**:
    - [`(anonymous)::tinyBLAS_BF16_PPC::KERNEL_4x8`](#(anonymous)::tinyBLAS_BF16_PPC::KERNEL_4x8)
    - [`(anonymous)::tinyBLAS_BF16_PPC::KERNEL_8x4`](#(anonymous)::tinyBLAS_BF16_PPC::KERNEL_8x4)
    - [`(anonymous)::tinyBLAS_BF16_PPC::KERNEL_8x8`](#(anonymous)::tinyBLAS_BF16_PPC::KERNEL_8x8)
- **See also**: [`(anonymous)::tinyBLAS_Q0_PPC`](#(anonymous)::tinyBLAS_Q0_PPC)  (Data Structure)


---
#### tinyBLAS\_Q0\_PPC::gemm<!-- {{#callable:(anonymous)::tinyBLAS_Q0_PPC::gemm}} -->
The `gemm` function performs a parallelized matrix multiplication operation on a submatrix defined by the input parameters, using a specified tile size for optimization.
- **Inputs**:
    - `m0`: The starting row index for the submatrix of matrix A.
    - `m`: The ending row index for the submatrix of matrix A.
    - `n0`: The starting column index for the submatrix of matrix B.
    - `n`: The ending column index for the submatrix of matrix B.
- **Control Flow**:
    - Calculate the number of tiles in the y-direction (`ytiles`) and x-direction (`xtiles`) based on the input dimensions and tile sizes `RM` and `RN`.
    - Compute the total number of tiles (`tiles`) as the product of `ytiles` and `xtiles`.
    - Determine the number of tiles each thread should process (`duty`) based on the total number of tiles and the number of threads (`nth`).
    - Calculate the starting (`start`) and ending (`end`) tile indices for the current thread based on its index (`ith`).
    - Iterate over the range of tiles assigned to the current thread, calculating the starting indices `ii` and `jj` for each tile.
    - Invoke the `kernel` function with the calculated indices `ii` and `jj` to perform the matrix multiplication for the current tile.
- **Output**: The function does not return a value; it performs operations directly on the matrices involved.
- **See also**: [`(anonymous)::tinyBLAS_Q0_PPC`](#(anonymous)::tinyBLAS_Q0_PPC)  (Data Structure)



---
### tinyBLAS\_PPC<!-- {{#data_structure:(anonymous)::tinyBLAS_PPC}} -->
- **Type**: `class`
- **Members**:
    - `A`: Pointer to the first input matrix, which is transposed.
    - `B`: Pointer to the second input matrix, which is not transposed.
    - `C`: Pointer to the output matrix.
    - `k`: Number of columns in A and rows in B.
    - `lda`: Leading dimension of matrix A.
    - `ldb`: Leading dimension of matrix B.
    - `ldc`: Leading dimension of matrix C.
    - `ith`: Thread index for parallel processing.
    - `nth`: Total number of threads for parallel processing.
- **Description**: The `tinyBLAS_PPC` class is a template class designed for performing optimized matrix multiplication on PowerPC architectures using the Matrix-Multiply Assist (MMA) instructions. It supports multithreaded operations and is optimized for matrices that fit within the CPU cache. The class is parameterized by three types, `TA`, `TB`, and `TC`, which represent the data types of the input matrices A and B, and the output matrix C, respectively. The class constructor initializes the matrix pointers and dimensions, as well as the thread information for parallel execution. The class provides a `matmul` method to perform the matrix multiplication, leveraging various kernel functions for different matrix block sizes to optimize performance.
- **Member Functions**:
    - [`(anonymous)::tinyBLAS_PPC::tinyBLAS_PPC`](#(anonymous)::tinyBLAS_PPC::tinyBLAS_PPC)
    - [`(anonymous)::tinyBLAS_PPC::matmul`](#(anonymous)::tinyBLAS_PPC::matmul)
    - [`(anonymous)::tinyBLAS_PPC::packTranspose`](#(anonymous)::tinyBLAS_PPC::packTranspose)
    - [`(anonymous)::tinyBLAS_PPC::KERNEL_4x4`](#(anonymous)::tinyBLAS_PPC::KERNEL_4x4)
    - [`(anonymous)::tinyBLAS_PPC::KERNEL_4x8`](#(anonymous)::tinyBLAS_PPC::KERNEL_4x8)
    - [`(anonymous)::tinyBLAS_PPC::KERNEL_8x4`](#(anonymous)::tinyBLAS_PPC::KERNEL_8x4)
    - [`(anonymous)::tinyBLAS_PPC::KERNEL_8x8`](#(anonymous)::tinyBLAS_PPC::KERNEL_8x8)
    - [`(anonymous)::tinyBLAS_PPC::mnpack`](#(anonymous)::tinyBLAS_PPC::mnpack)
    - [`(anonymous)::tinyBLAS_PPC::gemm_small`](#(anonymous)::tinyBLAS_PPC::gemm_small)
    - [`(anonymous)::tinyBLAS_PPC::gemm`](#(anonymous)::tinyBLAS_PPC::gemm)

**Methods**

---
#### tinyBLAS\_PPC::tinyBLAS\_PPC<!-- {{#callable:(anonymous)::tinyBLAS_PPC::tinyBLAS_PPC}} -->
The `tinyBLAS_PPC` constructor initializes a matrix multiplication object with given matrix dimensions, data pointers, and threading information.
- **Inputs**:
    - `k`: The number of columns in matrix A and rows in matrix B.
    - `A`: Pointer to the first input matrix A, which is transposed.
    - `lda`: Leading dimension of matrix A, representing the row stride.
    - `B`: Pointer to the second input matrix B.
    - `ldb`: Leading dimension of matrix B, representing the row stride.
    - `C`: Pointer to the output matrix C.
    - `ldc`: Leading dimension of matrix C, representing the row stride.
    - `ith`: The thread index for parallel execution.
    - `nth`: The total number of threads for parallel execution.
- **Control Flow**:
    - The constructor initializes member variables with the provided arguments.
    - It sets up the matrix dimensions and pointers for matrices A, B, and C.
    - It also configures the threading information with the thread index and total number of threads.
- **Output**: The constructor does not return any value; it initializes the object state.
- **See also**: [`(anonymous)::tinyBLAS_PPC`](#(anonymous)::tinyBLAS_PPC)  (Data Structure)


---
#### tinyBLAS\_PPC::matmul<!-- {{#callable:(anonymous)::tinyBLAS_PPC::matmul}} -->
The `matmul` function initiates a matrix multiplication operation by calling the [`mnpack`](#(anonymous)::tinyBLAS::mnpack) function with specified row and column indices.
- **Inputs**:
    - `m`: The number of rows in the matrix.
    - `n`: The number of columns in the matrix.
- **Control Flow**:
    - The function `matmul` is called with two parameters: `m` and `n`, representing the dimensions of the matrix.
    - Inside `matmul`, the function [`mnpack`](#(anonymous)::tinyBLAS::mnpack) is called with the parameters `0, m, 0, n`, which likely initiates a matrix packing or multiplication process starting from the first row and column up to the specified dimensions.
- **Output**: This function does not return any value; it likely modifies the state of the object or performs operations on matrices stored within the class.
- **Functions called**:
    - [`(anonymous)::tinyBLAS::mnpack`](#(anonymous)::tinyBLAS::mnpack)
- **See also**: [`(anonymous)::tinyBLAS_PPC`](#(anonymous)::tinyBLAS_PPC)  (Data Structure)


---
#### tinyBLAS\_PPC::packTranspose<!-- {{#callable:(anonymous)::tinyBLAS_PPC::packTranspose}} -->
The `packTranspose` function transposes and packs a matrix into a vectorized format for optimized matrix multiplication on PowerPC architectures.
- **Inputs**:
    - `a`: Pointer to the input matrix of type `TA`.
    - `lda`: Leading dimension of the matrix `a`, representing the number of elements between successive rows.
    - `rows`: Number of rows in the matrix `a` to be processed.
    - `cols`: Number of columns in the matrix `a` to be processed.
    - `vec`: Pointer to the output buffer where the transposed and packed matrix will be stored.
- **Control Flow**:
    - Initialize pointers for matrix offsets and vector pairs for vectorized operations.
    - Calculate the number of 8-row blocks in the matrix and iterate over them.
    - For each 8-row block, calculate the number of 8-column blocks and iterate over them.
    - Load 8 pairs of vectors from the matrix using `__builtin_vsx_lxvp` for each row offset.
    - Disassemble the vector pairs into individual vectors using `__builtin_vsx_disassemble_pair`.
    - Perform vectorized merge and permutation operations to transpose and pack the data.
    - Store the packed vectors into the output buffer using `vec_xst`.
    - Handle remaining columns if the number of columns is not a multiple of 8.
    - Handle remaining rows if the number of rows is not a multiple of 8.
- **Output**: The function does not return a value; it modifies the `vec` buffer in place to store the transposed and packed matrix data.
- **See also**: [`(anonymous)::tinyBLAS_PPC`](#(anonymous)::tinyBLAS_PPC)  (Data Structure)


---
#### tinyBLAS\_PPC::KERNEL\_4x4<!-- {{#callable:(anonymous)::tinyBLAS_PPC::KERNEL_4x4}} -->
The `KERNEL_4x4` function performs a 4x4 matrix multiplication using vectorized operations and accumulates the results into an accumulator, which is then saved to the output matrix.
- **Inputs**:
    - `ii`: The row index offset for matrix A.
    - `jj`: The column index offset for matrix B.
- **Control Flow**:
    - Initialize vector arrays `vec_A`, `vec_B`, and `vec_C` for storing packed matrix data and an accumulator `acc_0` for storing intermediate results.
    - Set the accumulator `acc_0` to zero using `__builtin_mma_xxsetaccz`.
    - Iterate over the range of `k` in steps of 4, where `k` is the shared dimension of matrices A and B.
    - In each iteration, pack and transpose a 4x4 block of matrix A starting at `(ii*lda)+l` into `vec_A` and a 4x4 block of matrix B starting at `(jj*ldb)+l` into `vec_B`.
    - Perform a series of vectorized multiply-accumulate operations using `__builtin_mma_xvf32gerpp` to update `acc_0` with the products of corresponding elements from `vec_A` and `vec_B`.
    - After the loop, save the accumulated results from `acc_0` into the output matrix C using the `SAVE_ACC` macro.
- **Output**: The function does not return a value; it modifies the output matrix C by saving the accumulated results from the matrix multiplication.
- **See also**: [`(anonymous)::tinyBLAS_PPC`](#(anonymous)::tinyBLAS_PPC)  (Data Structure)


---
#### tinyBLAS\_PPC::KERNEL\_4x8<!-- {{#callable:(anonymous)::tinyBLAS_PPC::KERNEL_4x8}} -->
The `KERNEL_4x8` function performs a 4x8 matrix multiplication using vectorized operations and accumulates the results into two accumulators, which are then stored in the output matrix.
- **Inputs**:
    - `ii`: The row index offset for matrix A.
    - `jj`: The column index offset for matrix B.
- **Control Flow**:
    - Initialize vector arrays `vec_A`, `vec_B`, and `vec_C` for storing matrix data and accumulators `acc_0` and `acc_1` for results.
    - Set both accumulators `acc_0` and `acc_1` to zero using `__builtin_mma_xxsetaccz`.
    - Iterate over the matrix dimension `k` in steps of 4.
    - For each iteration, transpose and pack a 4x4 block of matrix A and an 8x4 block of matrix B into `vec_A` and `vec_B` respectively using `packTranspose`.
    - Perform vectorized general matrix multiplication operations (`__builtin_mma_xvf32gerpp`) between `vec_A` and `vec_B` to update the accumulators `acc_0` and `acc_1`.
    - Store the results from `acc_0` and `acc_1` into the output matrix C using `SAVE_ACC`.
- **Output**: The function does not return a value but updates the output matrix C with the results of the matrix multiplication.
- **See also**: [`(anonymous)::tinyBLAS_PPC`](#(anonymous)::tinyBLAS_PPC)  (Data Structure)


---
#### tinyBLAS\_PPC::KERNEL\_8x4<!-- {{#callable:(anonymous)::tinyBLAS_PPC::KERNEL_8x4}} -->
The `KERNEL_8x4` function performs a matrix multiplication operation on 8x4 blocks using vectorized operations and accumulators.
- **Inputs**:
    - `ii`: The starting row index for the block of matrix A.
    - `jj`: The starting column index for the block of matrix B.
- **Control Flow**:
    - Initialize vector arrays `vec_A`, `vec_B`, and `vec_C` for storing blocks of matrices A and B, and the result, respectively.
    - Initialize accumulators `acc_0` and `acc_1` to zero using `__builtin_mma_xxsetaccz`.
    - Iterate over the range of `k` in steps of 4 to process blocks of the matrices.
    - For each iteration, transpose and pack 8x4 blocks of matrix A and 4x4 blocks of matrix B into `vec_A` and `vec_B` using `packTranspose`.
    - Perform a series of vectorized multiply-accumulate operations using `__builtin_mma_xvf32gerpp` to update the accumulators `acc_0` and `acc_1` with the products of elements from `vec_A` and `vec_B`.
    - Store the results from the accumulators into the result matrix C using `SAVE_ACC`.
- **Output**: The function does not return a value; it updates the matrix C with the results of the matrix multiplication.
- **See also**: [`(anonymous)::tinyBLAS_PPC`](#(anonymous)::tinyBLAS_PPC)  (Data Structure)


---
#### tinyBLAS\_PPC::KERNEL\_8x8<!-- {{#callable:(anonymous)::tinyBLAS_PPC::KERNEL_8x8}} -->
The `KERNEL_8x8` function performs an 8x8 matrix multiplication using vectorized operations and accumulates the results into four accumulators, which are then stored in the output matrix.
- **Inputs**:
    - `ii`: The row index offset for matrix A.
    - `jj`: The column index offset for matrix B.
- **Control Flow**:
    - Initialize four vector accumulators to zero using `__builtin_mma_xxsetaccz`.
    - Iterate over the range of `k` in steps of 8, where `k` is the shared dimension of matrices A and B.
    - For each iteration, transpose and pack an 8x8 block of matrix A and B into `vec_A` and `vec_B` respectively using `packTranspose`.
    - Perform vectorized outer product operations on pairs of vectors from `vec_A` and `vec_B`, updating the four accumulators with `__builtin_mma_xvf32gerpp`.
    - Store the results from the accumulators into the output matrix C using `SAVE_ACC`.
- **Output**: The function does not return a value but updates the output matrix C with the results of the matrix multiplication.
- **See also**: [`(anonymous)::tinyBLAS_PPC`](#(anonymous)::tinyBLAS_PPC)  (Data Structure)


---
#### tinyBLAS\_PPC::mnpack<!-- {{#callable:(anonymous)::tinyBLAS_PPC::mnpack}} -->
The [`mnpack`](#(anonymous)::tinyBLAS::mnpack) function recursively partitions a matrix multiplication problem into smaller subproblems based on the dimensions of the matrices and calls optimized GEMM (General Matrix Multiply) functions for efficient computation.
- **Inputs**:
    - `m0`: The starting row index for the matrix A.
    - `m`: The ending row index for the matrix A.
    - `n0`: The starting column index for the matrix B.
    - `n`: The ending column index for the matrix B.
- **Control Flow**:
    - Calculate the remaining rows `m_rem` and columns `n_rem` to be processed, limited to a maximum of 16.
    - Based on the values of `m_rem` and `n_rem`, determine the block sizes `mc` and `nc` for the subproblem and call the appropriate `gemm` or [`gemm_small`](#(anonymous)::tinyBLAS_BF16_PPC::gemm_small) function.
    - If `m_rem` and `n_rem` are both large enough, use a larger block size and call `gemm<8,8>`.
    - For smaller values of `m_rem` and `n_rem`, adjust `mc` and `nc` accordingly and call `gemm` or [`gemm_small`](#(anonymous)::tinyBLAS_BF16_PPC::gemm_small) with the appropriate template parameters.
    - If neither `m_rem` nor `n_rem` is large enough for the predefined cases, use a switch statement to handle specific small cases with [`gemm_small`](#(anonymous)::tinyBLAS_BF16_PPC::gemm_small).
    - Calculate the next partition points `mp` and `np` based on the current block sizes `mc` and `nc`.
    - Recursively call [`mnpack`](#(anonymous)::tinyBLAS::mnpack) for the next subproblems defined by the new partition points `mp` and `np`.
- **Output**: The function does not return a value; it performs matrix multiplication by recursively partitioning the problem and calling optimized subroutines.
- **Functions called**:
    - [`(anonymous)::tinyBLAS_BF16_PPC::gemm_small`](#(anonymous)::tinyBLAS_BF16_PPC::gemm_small)
    - [`(anonymous)::tinyBLAS::mnpack`](#(anonymous)::tinyBLAS::mnpack)
- **See also**: [`(anonymous)::tinyBLAS_PPC`](#(anonymous)::tinyBLAS_PPC)  (Data Structure)


---
#### tinyBLAS\_PPC::gemm\_small<!-- {{#callable:(anonymous)::tinyBLAS_PPC::gemm_small}} -->
The `gemm_small` function performs a small-scale matrix multiplication using a specified number of rows and columns for tiling, optimized for parallel execution across multiple threads.
- **Inputs**:
    - `m0`: The starting row index for the matrix multiplication.
    - `m`: The ending row index for the matrix multiplication.
    - `n0`: The starting column index for the matrix multiplication.
    - `n`: The ending column index for the matrix multiplication.
    - `RM`: The number of rows in each tile for the matrix multiplication.
    - `RN`: The number of columns in each tile for the matrix multiplication.
- **Control Flow**:
    - Calculate the number of tiles in the y and x directions based on RM and RN.
    - Determine the total number of tiles and distribute the workload across threads using the 'duty' variable.
    - Calculate the start and end indices for the current thread's workload.
    - Iterate over each job assigned to the current thread, calculating the starting indices for each tile.
    - Initialize accumulators and vectors for matrix multiplication.
    - Use conditional logic to handle cases where RM or RN is 1, broadcasting elements as needed.
    - Perform matrix multiplication using vectorized operations and accumulate results.
    - Disassemble the accumulator to store the results back into the output matrix C.
- **Output**: The function does not return a value; it modifies the matrix C in place with the results of the matrix multiplication.
- **See also**: [`(anonymous)::tinyBLAS_PPC`](#(anonymous)::tinyBLAS_PPC)  (Data Structure)


---
#### tinyBLAS\_PPC::gemm<!-- {{#callable:(anonymous)::tinyBLAS_PPC::gemm}} -->
The `gemm` function performs a tiled matrix multiplication using specified tile sizes and assigns the appropriate kernel function for the operation.
- **Inputs**:
    - `m0`: The starting row index for the matrix multiplication.
    - `m`: The ending row index for the matrix multiplication.
    - `n0`: The starting column index for the matrix multiplication.
    - `n`: The ending column index for the matrix multiplication.
- **Control Flow**:
    - Calculate the number of tiles in the y and x directions based on the input dimensions and tile sizes RM and RN.
    - Determine the total number of tiles and the duty (number of tiles per thread) based on the number of threads (nth).
    - Calculate the start and end indices for the current thread's work based on its index (ith).
    - Select the appropriate kernel function based on the tile sizes RM and RN.
    - Adjust the end index if it exceeds the total number of tiles.
    - Iterate over the assigned tiles, calculating the starting indices for each tile and invoking the selected kernel function for each tile.
- **Output**: The function does not return a value; it performs matrix multiplication and updates the result matrix in place.
- **See also**: [`(anonymous)::tinyBLAS_PPC`](#(anonymous)::tinyBLAS_PPC)  (Data Structure)



# Functions

---
### unhalf<!-- {{#callable:(anonymous)::unhalf}} -->
The `unhalf` function converts a 16-bit floating-point number to a 32-bit floating-point number.
- **Inputs**:
    - `d`: A 16-bit floating-point number of type `ggml_fp16_t` to be converted to a 32-bit floating-point number.
- **Control Flow**:
    - The function takes a single argument `d` of type `ggml_fp16_t`.
    - It calls the macro `GGML_FP16_TO_FP32` with `d` as the argument to perform the conversion.
- **Output**: A 32-bit floating-point number (`float`) that is the result of converting the input 16-bit floating-point number.


---
### add<!-- {{#callable:(anonymous)::add}} -->
The `add` function performs element-wise addition of two 8-element vectors of 16-bit floating-point numbers using ARM NEON intrinsics.
- **Inputs**:
    - `x`: An 8-element vector of 16-bit floating-point numbers (float16x8_t) to be added.
    - `y`: Another 8-element vector of 16-bit floating-point numbers (float16x8_t) to be added.
- **Control Flow**:
    - The function takes two float16x8_t vectors as input parameters.
    - It calls the ARM NEON intrinsic function `vaddq_f16` to perform element-wise addition of the two input vectors.
    - The result of the addition is returned.
- **Output**: An 8-element vector of 16-bit floating-point numbers (float16x8_t) representing the element-wise sum of the input vectors.


---
### sub<!-- {{#callable:(anonymous)::sub}} -->
The `sub` function performs element-wise subtraction of two 8-element vectors of 16-bit floating-point numbers using ARM NEON intrinsics.
- **Inputs**:
    - `x`: An 8-element vector of 16-bit floating-point numbers (float16x8_t) representing the minuend.
    - `y`: An 8-element vector of 16-bit floating-point numbers (float16x8_t) representing the subtrahend.
- **Control Flow**:
    - The function takes two float16x8_t vectors as input parameters.
    - It calls the ARM NEON intrinsic function `vsubq_f16` to perform element-wise subtraction of the two vectors.
    - The result of the subtraction is returned as a float16x8_t vector.
- **Output**: An 8-element vector of 16-bit floating-point numbers (float16x8_t) representing the result of the element-wise subtraction of `y` from `x`.


---
### mul<!-- {{#callable:(anonymous)::mul}} -->
The `mul` function performs element-wise multiplication of two 8-element vectors of 16-bit floating-point numbers using ARM NEON intrinsics.
- **Inputs**:
    - `x`: An 8-element vector of 16-bit floating-point numbers (float16x8_t).
    - `y`: An 8-element vector of 16-bit floating-point numbers (float16x8_t).
- **Control Flow**:
    - The function takes two float16x8_t vectors as input.
    - It calls the NEON intrinsic function `vmulq_f16` to perform element-wise multiplication of the two vectors.
    - The result of the multiplication is returned.
- **Output**: An 8-element vector of 16-bit floating-point numbers (float16x8_t) resulting from the element-wise multiplication of the input vectors.


---
### madd<!-- {{#callable:(anonymous)::madd}} -->
The `madd` function performs a fused multiply-add operation on three `float16x8_t` vectors using ARM NEON intrinsics.
- **Inputs**:
    - `a`: A `float16x8_t` vector representing the first operand for multiplication.
    - `b`: A `float16x8_t` vector representing the second operand for multiplication.
    - `c`: A `float16x8_t` vector representing the operand to be added to the product of `a` and `b`.
- **Control Flow**:
    - The function takes three `float16x8_t` vectors as input parameters: `a`, `b`, and `c`.
    - It uses the ARM NEON intrinsic `vfmaq_f16` to compute the fused multiply-add operation, which multiplies `a` and `b`, and then adds `c` to the result.
    - The result of the operation is returned as a `float16x8_t` vector.
- **Output**: A `float16x8_t` vector that is the result of the fused multiply-add operation on the input vectors `a`, `b`, and `c`.


---
### hsum<!-- {{#callable:(anonymous)::hsum}} -->
The `hsum` function computes the horizontal sum of all elements in a 512-bit wide SIMD register of single-precision floating-point numbers.
- **Inputs**:
    - `x`: A 512-bit wide SIMD register (__m512) containing single-precision floating-point numbers.
- **Control Flow**:
    - The function takes a single input, a 512-bit SIMD register `x`.
    - It calls the intrinsic function `_mm512_reduce_add_ps(x)` to compute the sum of all elements in the register.
    - The result of the sum is returned as a single floating-point number.
- **Output**: A single floating-point number representing the sum of all elements in the input SIMD register.


---
### load<!-- {{#callable:(anonymous)::load}} -->
The `load` function loads a 256-bit half-precision floating-point vector from a given memory address and converts it from single-precision to bfloat16 format using AVX-512 instructions.
- **Inputs**:
    - `p`: A pointer to a memory location containing single-precision floating-point values.
- **Control Flow**:
    - The function takes a pointer `p` to a float array as input.
    - It uses the `_mm512_loadu_ps` intrinsic to load 512 bits (16 floats) from the memory location pointed to by `p`.
    - The loaded single-precision floats are then converted to bfloat16 format using the `_mm512_cvtneps_pbh` intrinsic.
    - The resulting 256-bit bfloat16 vector is returned.
- **Output**: A 256-bit vector of bfloat16 values.


---
### BLOCK\_SIZE<!-- {{#callable:(anonymous)::BLOCK_SIZE}} -->
The `BLOCK_SIZE` function calculates the size of each block when dividing a given size `m` into blocks of a specified maximum size `M`.
- **Inputs**:
    - `M`: A template parameter representing the maximum size of each block.
    - `m`: A size_t parameter representing the total size to be divided into blocks.
- **Control Flow**:
    - Calculate the number of blocks `NB_BLOC_M` by dividing `m + M - 1` by `M` to ensure rounding up.
    - Check if `m` is evenly divisible by `NB_BLOC_M`.
    - If `m` is evenly divisible, return `m / NB_BLOC_M`.
    - If not, return `m / NB_BLOC_M + 1` to account for the remainder.
- **Output**: Returns an int64_t representing the size of each block.


---
### BLOC\_POS<!-- {{#callable:(anonymous)::BLOC_POS}} -->
The `BLOC_POS` function calculates the position of a block in a matrix based on its index, the number of blocks, and the block size.
- **Inputs**:
    - `ib`: The index of the block for which the position is being calculated.
    - `ibN`: The total number of blocks in the matrix.
    - `bloc_size`: The size of each block in the matrix.
- **Control Flow**:
    - Check if the block index `ib` is less than the total number of blocks `ibN`.
    - If `ib` is less than `ibN`, return the product of `ib` and `bloc_size`.
    - If `ib` is not less than `ibN`, return the sum of `ibN * bloc_size` and the product of `(ib - ibN)` and `(bloc_size - 1)`.
- **Output**: The function returns an `int64_t` value representing the calculated position of the block in the matrix.


---
### llamafile\_sgemm<!-- {{#callable:llamafile_sgemm}} -->
The `llamafile_sgemm` function performs optimized matrix multiplication on CPU for specific data types and conditions.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing computation parameters such as thread information.
    - `m`: The number of rows in matrices A and C.
    - `n`: The number of columns in matrices B and C.
    - `k`: The number of columns in matrix A and rows in matrix B.
    - `A`: A pointer to the first input matrix, which is always transposed.
    - `lda`: The leading dimension or row stride of matrix A.
    - `B`: A pointer to the second input matrix, which is never transposed.
    - `ldb`: The leading dimension or row stride of matrix B.
    - `C`: A pointer to the output matrix where the result is stored.
    - `ldc`: The leading dimension or row stride of matrix C.
    - `Atype`: The data type of matrix A, specified as a GGML type constant.
    - `Btype`: The data type of matrix B, specified as a GGML type constant.
    - `Ctype`: The data type of matrix C, specified as a GGML type constant.
- **Control Flow**:
    - The function begins by asserting that the dimensions and parameters are valid, ensuring non-negative dimensions and valid thread parameters.
    - It checks if the matrix multiplication is enabled for prompt processing, returning false if not applicable.
    - The function verifies that the output matrix C is of type `GGML_TYPE_F32`, returning false otherwise.
    - A switch statement is used to handle different data types of matrix A, with nested conditions for matrix B's data type.
    - For each supported combination of data types and hardware capabilities, a specialized `tinyBLAS` object is instantiated and its `matmul` method is called to perform the multiplication.
    - If the conditions for any specific data type or hardware are not met, the function returns false.
    - The function uses preprocessor directives to include or exclude code based on the available hardware features, such as AVX, AVX2, AVX512, ARM NEON, and MMA.
    - If none of the conditions are met, the function returns false, indicating that the request could not be serviced.
- **Output**: The function returns a boolean value indicating whether the matrix multiplication was successfully performed.


