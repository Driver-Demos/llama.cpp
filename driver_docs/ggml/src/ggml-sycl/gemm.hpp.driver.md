# Purpose
This C++ header file, `ggml_sycl_gemm.hpp`, is part of the LLVM Project and is designed to provide specialized functionality for performing General Matrix Multiplication (GEMM) operations using the SYCL and DNNL (Deep Neural Network Library) frameworks. The file defines a class `DnnlGemmWrapper` that encapsulates methods for executing matrix multiplication operations on SYCL-enabled devices, leveraging the DNNL library for optimized performance. The class provides two static methods: [`gemm`](#DnnlGemmWrappergemm) and [`row_gemm`](#DnnlGemmWrapperrow_gemm). The [`gemm`](#DnnlGemmWrappergemm) method is a versatile function that allows for the multiplication of two matrices with specified dimensions and strides, supporting batch processing. The [`row_gemm`](#DnnlGemmWrapperrow_gemm) method is a specialized version of [`gemm`](#DnnlGemmWrappergemm) for matrices stored in column-major order, specifically designed to compute the product of a transposed matrix A and matrix B.

The file includes necessary headers such as `dnnl.hpp` and `dnnl_sycl.hpp` to integrate DNNL functionalities with SYCL. It defines data types and memory format tags using DNNL's memory descriptors, which are crucial for specifying how data is laid out in memory. The `DnnlGemmWrapper` class provides a public API for performing matrix multiplications, making it a reusable component in applications that require efficient linear algebra computations on heterogeneous computing platforms. The use of templates and static assertions ensures type safety and flexibility, allowing the code to handle different data types like `float` and `sycl::half`. This header file is intended to be included in other C++ source files that require matrix multiplication capabilities, particularly in environments where SYCL and DNNL are used for high-performance computing tasks.
# Imports and Dependencies

---
- `ggml-sycl.h`
- `dnnl.hpp`
- `dnnl_sycl.hpp`


# Data Structures

---
### DnnlGemmWrapper<!-- {{#data_structure:DnnlGemmWrapper}} -->
- **Type**: `class`
- **Members**:
    - `dt`: Alias for dnnl::memory::data_type, representing data types used in memory operations.
    - `tag`: Alias for dnnl::memory::format_tag, representing memory format tags.
- **Description**: The `DnnlGemmWrapper` class is a utility wrapper designed to facilitate General Matrix Multiplication (GEMM) operations using the DNNL (Deep Neural Network Library) with SYCL support. It provides static methods to perform matrix multiplication, specifically tailored for handling matrices in a column-major format. The class includes type aliases for data types and format tags used in DNNL memory operations, and it offers a method to convert C++ types to DNNL data types. The primary functionality is encapsulated in the `gemm` method, which sets up and executes a matrix multiplication operation, and the `row_gemm` method, which is a specialized version for row-major matrices.
- **Member Functions**:
    - [`DnnlGemmWrapper::to_dt`](#DnnlGemmWrapperto_dt)
    - [`DnnlGemmWrapper::gemm`](#DnnlGemmWrappergemm)
    - [`DnnlGemmWrapper::row_gemm`](#DnnlGemmWrapperrow_gemm)

**Methods**

---
#### DnnlGemmWrapper::to\_dt<!-- {{#callable:DnnlGemmWrapper::to_dt}} -->
The `to_dt` function is a template function that maps a given type to a corresponding DNNL data type.
- **Inputs**:
    - `T`: A template parameter representing the type to be mapped to a DNNL data type.
- **Control Flow**:
    - The function checks if the type `T` is `float` using `std::is_same_v`; if true, it returns `dt::f32`.
    - If the type `T` is `sycl::half`, it returns `dt::f16`.
    - If `T` is neither `float` nor `sycl::half`, a static assertion fails, causing a compile-time error.
- **Output**: The function returns a `dt` value corresponding to the DNNL data type for the given type `T`.
- **See also**: [`DnnlGemmWrapper`](#DnnlGemmWrapper)  (Data Structure)


---
#### DnnlGemmWrapper::gemm<!-- {{#callable:DnnlGemmWrapper::gemm}} -->
The `gemm` function performs a General Matrix Multiply (GEMM) operation using the DNNL library, handling multiple batches of matrices with specified strides and dimensions.
- **Inputs**:
    - `ctx`: A reference to a `ggml_backend_sycl_context` object, which provides the SYCL context and DNNL engine/stream management.
    - `m`: An integer representing the number of rows in matrix A.
    - `n`: An integer representing the number of columns in matrix B.
    - `k`: An integer representing the number of columns in matrix A and rows in matrix B.
    - `a`: A pointer to the data of matrix A.
    - `at`: The data type of matrix A, specified as `dnnl::memory::data_type`.
    - `nra`: The number of elements to skip when moving to the next row in matrix A.
    - `nca`: The number of elements to skip when moving to the next column in matrix A.
    - `stride_a`: The number of elements to skip when moving to the next matrix A in a batch.
    - `b`: A pointer to the data of matrix B.
    - `bt`: The data type of matrix B, specified as `dnnl::memory::data_type`.
    - `nrb`: The number of elements to skip when moving to the next row in matrix B.
    - `ncb`: The number of elements to skip when moving to the next column in matrix B.
    - `stride_b`: The number of elements to skip when moving to the next matrix B in a batch.
    - `c`: A pointer to the data of matrix C, where the result will be stored.
    - `ct`: The data type of matrix C, specified as `dnnl::memory::data_type`.
    - `q`: A `queue_ptr` object representing the SYCL queue to be used for execution.
    - `batches_a`: The number of matrices in batch A.
    - `batches_b`: The number of matrices in batch B.
- **Control Flow**:
    - Retrieve the DNNL stream and engine from the context using the provided queue.
    - Define the dimensions for matrices A, B, and C based on the input parameters and batch sizes.
    - Define the strides for matrices A and B based on the input parameters.
    - Create memory descriptors for matrices A, B, and C using their dimensions, data types, and strides.
    - Set the scratchpad mode for the DNNL primitive attributes to user mode.
    - Create DNNL memory objects for matrices A and B using their descriptors and the engine.
    - Create a matmul primitive descriptor using the engine and the memory descriptors for A, B, and C.
    - Create a DNNL memory object for matrix C using the matmul primitive descriptor's destination descriptor and the engine.
    - Retrieve the scratchpad memory descriptor from the matmul primitive descriptor and obtain the corresponding memory object from the context.
    - Create a matmul primitive using the matmul primitive descriptor.
    - Prepare a map of arguments for the matmul primitive, including the source, weights, destination, and scratchpad memory objects.
    - Execute the matmul primitive using the DNNL stream and the prepared arguments map.
- **Output**: The function does not return a value; it performs the matrix multiplication and stores the result in the provided memory location for matrix C.
- **See also**: [`DnnlGemmWrapper`](#DnnlGemmWrapper)  (Data Structure)


---
#### DnnlGemmWrapper::row\_gemm<!-- {{#callable:DnnlGemmWrapper::row_gemm}} -->
The `row_gemm` function performs a matrix multiplication of two column-major matrices A and B, storing the result in matrix C, using the SYCL backend context.
- **Inputs**:
    - `ctx`: A reference to a `ggml_backend_sycl_context` object, which provides the SYCL backend context for the operation.
    - `m`: An integer representing the number of columns in matrix A.
    - `n`: An integer representing the number of columns in matrix B.
    - `k`: An integer representing the number of rows in both matrices A and B.
    - `a`: A pointer to the data of matrix A.
    - `at`: The data type of matrix A, specified as `dt`.
    - `b`: A pointer to the data of matrix B.
    - `bt`: The data type of matrix B, specified as `dt`.
    - `c`: A pointer to the data of matrix C, where the result will be stored.
    - `ct`: The data type of matrix C, specified as `dt`.
    - `q`: A `queue_ptr` object representing the SYCL queue to be used for the operation.
- **Control Flow**:
    - The function `row_gemm` is called with the provided parameters.
    - It calls the [`gemm`](#DnnlGemmWrappergemm) function with specific stride and batch parameters to perform the matrix multiplication.
    - The [`gemm`](#DnnlGemmWrappergemm) function handles the detailed setup and execution of the matrix multiplication using the DNNL library and SYCL context.
- **Output**: The function does not return a value; it stores the result of the matrix multiplication in the memory location pointed to by `c`.
- **Functions called**:
    - [`DnnlGemmWrapper::gemm`](#DnnlGemmWrappergemm)
- **See also**: [`DnnlGemmWrapper`](#DnnlGemmWrapper)  (Data Structure)



