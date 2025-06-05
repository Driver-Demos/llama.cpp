# Purpose
The provided C code is a comprehensive header file for the GGML Tensor Library, which is designed to facilitate various machine learning tasks through tensor operations, automatic differentiation, and basic optimization algorithms. The library is intended to be minimalistic, supporting tasks such as linear regression, support vector machines, and neural networks. It allows users to define functions using tensor operations, which are internally represented as computation graphs. These graphs enable the computation of function values and gradients, and the library provides optimization capabilities to refine these functions.

The header file defines a wide array of functionalities, including tensor creation, manipulation, and operations like addition, multiplication, and more complex operations such as convolution and pooling. It supports multi-dimensional tensors up to four dimensions and various data types, including FP16 and FP32. The library also includes mechanisms for memory management, multi-threading, and SIMD optimizations. Additionally, it provides a set of APIs for custom operations, loss functions, and quantization, making it a versatile tool for machine learning practitioners. The file is structured to be included in other C programs, providing a public API for the GGML library, and it includes conditional compilation directives to support different platforms and compilers.
# Imports and Dependencies

---
- `stdbool.h`
- `stddef.h`
- `stdint.h`
- `stdio.h`


# Data Structures

---
### ggml\_status
- **Type**: `enum`
- **Members**:
    - `GGML_STATUS_ALLOC_FAILED`: Indicates that memory allocation failed, represented by the value -2.
    - `GGML_STATUS_FAILED`: Indicates a general failure, represented by the value -1.
    - `GGML_STATUS_SUCCESS`: Indicates successful operation, represented by the value 0.
    - `GGML_STATUS_ABORTED`: Indicates that an operation was aborted, represented by the value 1.
- **Description**: The `ggml_status` enumeration defines a set of constants representing the status of operations within the GGML Tensor Library. It includes values for successful operations, general failures, allocation failures, and aborted operations, allowing functions to communicate their execution status effectively.


---
### ggml\_bf16\_t
- **Type**: `struct`
- **Members**:
    - `bits`: A 16-bit unsigned integer representing the bit pattern of the bfloat16 value.
- **Description**: The `ggml_bf16_t` structure is a simple data structure used to represent a bfloat16 (Brain Floating Point) number, which is a 16-bit floating-point format primarily used in machine learning applications. The structure contains a single member, `bits`, which stores the bit pattern of the bfloat16 value. This format is designed to provide a balance between range and precision, making it suitable for deep learning tasks where memory efficiency is crucial.


---
### ggml\_type
- **Type**: `enum`
- **Members**:
    - `GGML_TYPE_F32`: Represents a 32-bit floating point type.
    - `GGML_TYPE_F16`: Represents a 16-bit floating point type.
    - `GGML_TYPE_Q4_0`: Represents a quantized type with a specific format.
    - `GGML_TYPE_Q4_1`: Represents another quantized type with a different format.
    - `GGML_TYPE_Q5_0`: Represents a quantized type with a specific format.
    - `GGML_TYPE_Q5_1`: Represents another quantized type with a different format.
    - `GGML_TYPE_Q8_0`: Represents a quantized type with a specific format.
    - `GGML_TYPE_Q8_1`: Represents another quantized type with a different format.
    - `GGML_TYPE_Q2_K`: Represents a quantized type with a specific format.
    - `GGML_TYPE_Q3_K`: Represents another quantized type with a different format.
    - `GGML_TYPE_Q4_K`: Represents a quantized type with a specific format.
    - `GGML_TYPE_Q5_K`: Represents another quantized type with a different format.
    - `GGML_TYPE_Q6_K`: Represents a quantized type with a specific format.
    - `GGML_TYPE_Q8_K`: Represents another quantized type with a different format.
    - `GGML_TYPE_IQ2_XXS`: Represents an integer quantized type with a specific format.
    - `GGML_TYPE_IQ2_XS`: Represents another integer quantized type with a different format.
    - `GGML_TYPE_IQ3_XXS`: Represents an integer quantized type with a specific format.
    - `GGML_TYPE_IQ1_S`: Represents another integer quantized type with a different format.
    - `GGML_TYPE_IQ4_NL`: Represents an integer quantized type with a specific format.
    - `GGML_TYPE_IQ3_S`: Represents another integer quantized type with a different format.
    - `GGML_TYPE_IQ2_S`: Represents an integer quantized type with a specific format.
    - `GGML_TYPE_IQ4_XS`: Represents another integer quantized type with a different format.
    - `GGML_TYPE_I8`: Represents an 8-bit integer type.
    - `GGML_TYPE_I16`: Represents a 16-bit integer type.
    - `GGML_TYPE_I32`: Represents a 32-bit integer type.
    - `GGML_TYPE_I64`: Represents a 64-bit integer type.
    - `GGML_TYPE_F64`: Represents a 64-bit floating point type.
    - `GGML_TYPE_IQ1_M`: Represents an integer quantized type with a specific format.
    - `GGML_TYPE_BF16`: Represents a bfloat16 type.
    - `GGML_TYPE_TQ1_0`: Represents a tensor quantized type with a specific format.
    - `GGML_TYPE_TQ2_0`: Represents another tensor quantized type with a different format.
    - `GGML_TYPE_COUNT`: Represents the count of all defined types.
- **Description**: The `ggml_type` enum defines a set of data types used within the GGML Tensor Library, which includes various floating point, integer, and quantized types. These types are used to specify the data format of tensors in the library, allowing for operations on different numerical representations. The enum includes standard types like 32-bit and 64-bit floats and integers, as well as specialized quantized types for optimized storage and computation. The `GGML_TYPE_COUNT` member indicates the total number of types defined in the enum.


---
### ggml\_prec
- **Type**: `enum`
- **Members**:
    - `GGML_PREC_DEFAULT`: Represents the default precision, stored as ggml_tensor.op_params, with a value of 0.
    - `GGML_PREC_F32`: Represents 32-bit floating point precision, with a value of 10.
- **Description**: The `ggml_prec` enumeration defines precision levels for tensor operations within the GGML Tensor Library. It currently includes two precision levels: `GGML_PREC_DEFAULT`, which is the default precision setting, and `GGML_PREC_F32`, which specifies 32-bit floating point precision. This enumeration is used to control the precision of computations in tensor operations, allowing for flexibility in performance and accuracy trade-offs.


---
### ggml\_ftype
- **Type**: `enum`
- **Members**:
    - `GGML_FTYPE_UNKNOWN`: Represents an unknown file type with a value of -1.
    - `GGML_FTYPE_ALL_F32`: Represents a file type where all data is in 32-bit floating point format, with a value of 0.
    - `GGML_FTYPE_MOSTLY_F16`: Represents a file type where most data is in 16-bit floating point format, except for 1D tensors, with a value of 1.
    - `GGML_FTYPE_MOSTLY_Q4_0`: Represents a file type where most data is in Q4_0 format, except for 1D tensors, with a value of 2.
    - `GGML_FTYPE_MOSTLY_Q4_1`: Represents a file type where most data is in Q4_1 format, except for 1D tensors, with a value of 3.
    - `GGML_FTYPE_MOSTLY_Q4_1_SOME_F16`: Represents a file type where most data is in Q4_1 format, but some specific weights are in F16, with a value of 4.
    - `GGML_FTYPE_MOSTLY_Q8_0`: Represents a file type where most data is in Q8_0 format, except for 1D tensors, with a value of 7.
    - `GGML_FTYPE_MOSTLY_Q5_0`: Represents a file type where most data is in Q5_0 format, except for 1D tensors, with a value of 8.
    - `GGML_FTYPE_MOSTLY_Q5_1`: Represents a file type where most data is in Q5_1 format, except for 1D tensors, with a value of 9.
    - `GGML_FTYPE_MOSTLY_Q2_K`: Represents a file type where most data is in Q2_K format, except for 1D tensors, with a value of 10.
    - `GGML_FTYPE_MOSTLY_Q3_K`: Represents a file type where most data is in Q3_K format, except for 1D tensors, with a value of 11.
    - `GGML_FTYPE_MOSTLY_Q4_K`: Represents a file type where most data is in Q4_K format, except for 1D tensors, with a value of 12.
    - `GGML_FTYPE_MOSTLY_Q5_K`: Represents a file type where most data is in Q5_K format, except for 1D tensors, with a value of 13.
    - `GGML_FTYPE_MOSTLY_Q6_K`: Represents a file type where most data is in Q6_K format, except for 1D tensors, with a value of 14.
    - `GGML_FTYPE_MOSTLY_IQ2_XXS`: Represents a file type where most data is in IQ2_XXS format, except for 1D tensors, with a value of 15.
    - `GGML_FTYPE_MOSTLY_IQ2_XS`: Represents a file type where most data is in IQ2_XS format, except for 1D tensors, with a value of 16.
    - `GGML_FTYPE_MOSTLY_IQ3_XXS`: Represents a file type where most data is in IQ3_XXS format, except for 1D tensors, with a value of 17.
    - `GGML_FTYPE_MOSTLY_IQ1_S`: Represents a file type where most data is in IQ1_S format, except for 1D tensors, with a value of 18.
    - `GGML_FTYPE_MOSTLY_IQ4_NL`: Represents a file type where most data is in IQ4_NL format, except for 1D tensors, with a value of 19.
    - `GGML_FTYPE_MOSTLY_IQ3_S`: Represents a file type where most data is in IQ3_S format, except for 1D tensors, with a value of 20.
    - `GGML_FTYPE_MOSTLY_IQ2_S`: Represents a file type where most data is in IQ2_S format, except for 1D tensors, with a value of 21.
    - `GGML_FTYPE_MOSTLY_IQ4_XS`: Represents a file type where most data is in IQ4_XS format, except for 1D tensors, with a value of 22.
    - `GGML_FTYPE_MOSTLY_IQ1_M`: Represents a file type where most data is in IQ1_M format, except for 1D tensors, with a value of 23.
    - `GGML_FTYPE_MOSTLY_BF16`: Represents a file type where most data is in BF16 format, except for 1D tensors, with a value of 24.
- **Description**: The `ggml_ftype` enumeration defines various file types used in the GGML Tensor Library, each representing a specific data format or precision level for storing tensor data. These file types are primarily used to specify the format of tensor data, with most types indicating a predominant format (e.g., F16, Q4_0) while allowing exceptions for 1D tensors. This enumeration is crucial for managing and interpreting the data formats used in machine learning tasks within the library.


---
### ggml\_op
- **Type**: `enum`
- **Members**:
    - `GGML_OP_NONE`: Represents no operation, used as a default or placeholder.
    - `GGML_OP_DUP`: Duplicates a tensor.
    - `GGML_OP_ADD`: Performs element-wise addition of two tensors.
    - `GGML_OP_ADD1`: Adds a scalar to each element of a tensor.
    - `GGML_OP_ACC`: Accumulates values into a tensor.
    - `GGML_OP_SUB`: Performs element-wise subtraction between two tensors.
    - `GGML_OP_MUL`: Performs element-wise multiplication of two tensors.
    - `GGML_OP_DIV`: Performs element-wise division of two tensors.
    - `GGML_OP_SQR`: Squares each element of a tensor.
    - `GGML_OP_SQRT`: Computes the square root of each element in a tensor.
    - `GGML_OP_LOG`: Computes the natural logarithm of each element in a tensor.
    - `GGML_OP_SIN`: Computes the sine of each element in a tensor.
    - `GGML_OP_COS`: Computes the cosine of each element in a tensor.
    - `GGML_OP_SUM`: Sums all elements in a tensor.
    - `GGML_OP_SUM_ROWS`: Sums elements along the rows of a tensor.
    - `GGML_OP_MEAN`: Computes the mean of elements in a tensor.
    - `GGML_OP_ARGMAX`: Finds the index of the maximum value along a specified axis.
    - `GGML_OP_COUNT_EQUAL`: Counts the number of equal elements between two tensors.
    - `GGML_OP_REPEAT`: Repeats a tensor to match the shape of another tensor.
    - `GGML_OP_REPEAT_BACK`: Sums repeated elements back into the original shape.
    - `GGML_OP_CONCAT`: Concatenates two tensors along a specified dimension.
    - `GGML_OP_SILU_BACK`: Performs the backward pass of the SiLU activation function.
    - `GGML_OP_NORM`: Normalizes a tensor.
    - `GGML_OP_RMS_NORM`: Performs RMS normalization on a tensor.
    - `GGML_OP_RMS_NORM_BACK`: Performs the backward pass of RMS normalization.
    - `GGML_OP_GROUP_NORM`: Performs group normalization on a tensor.
    - `GGML_OP_L2_NORM`: Performs L2 normalization on a tensor.
    - `GGML_OP_MUL_MAT`: Performs matrix multiplication.
    - `GGML_OP_MUL_MAT_ID`: Performs indirect matrix multiplication.
    - `GGML_OP_OUT_PROD`: Computes the outer product of two tensors.
    - `GGML_OP_SCALE`: Scales a tensor by a scalar value.
    - `GGML_OP_SET`: Sets values in a tensor.
    - `GGML_OP_CPY`: Copies data from one tensor to another.
    - `GGML_OP_CONT`: Makes a tensor contiguous in memory.
    - `GGML_OP_RESHAPE`: Reshapes a tensor to a new shape.
    - `GGML_OP_VIEW`: Creates a view of a tensor with a different shape.
    - `GGML_OP_PERMUTE`: Permutes the dimensions of a tensor.
    - `GGML_OP_TRANSPOSE`: Transposes a tensor.
    - `GGML_OP_GET_ROWS`: Extracts specific rows from a tensor.
    - `GGML_OP_GET_ROWS_BACK`: Performs the backward pass for row extraction.
    - `GGML_OP_DIAG`: Extracts the diagonal of a tensor.
    - `GGML_OP_DIAG_MASK_INF`: Masks elements above the diagonal with negative infinity.
    - `GGML_OP_DIAG_MASK_ZERO`: Masks elements above the diagonal with zero.
    - `GGML_OP_SOFT_MAX`: Applies the softmax function to a tensor.
    - `GGML_OP_SOFT_MAX_BACK`: Performs the backward pass of the softmax function.
    - `GGML_OP_ROPE`: Applies rotary position embedding to a tensor.
    - `GGML_OP_ROPE_BACK`: Performs the backward pass of rotary position embedding.
    - `GGML_OP_CLAMP`: Clamps the values of a tensor within a specified range.
    - `GGML_OP_CONV_TRANSPOSE_1D`: Performs a 1D transposed convolution.
    - `GGML_OP_IM2COL`: Transforms data for convolution using the im2col method.
    - `GGML_OP_IM2COL_BACK`: Performs the backward pass of the im2col transformation.
    - `GGML_OP_CONV_2D_DW`: Performs a 2D depthwise convolution.
    - `GGML_OP_CONV_TRANSPOSE_2D`: Performs a 2D transposed convolution.
    - `GGML_OP_POOL_1D`: Applies a 1D pooling operation.
    - `GGML_OP_POOL_2D`: Applies a 2D pooling operation.
    - `GGML_OP_POOL_2D_BACK`: Performs the backward pass of a 2D pooling operation.
    - `GGML_OP_UPSCALE`: Upscales a tensor using nearest interpolation.
    - `GGML_OP_PAD`: Pads a tensor with zeros.
    - `GGML_OP_PAD_REFLECT_1D`: Pads a tensor with reflected values in 1D.
    - `GGML_OP_ARANGE`: Creates a tensor with a range of values.
    - `GGML_OP_TIMESTEP_EMBEDDING`: Embeds timesteps into a tensor.
    - `GGML_OP_ARGSORT`: Sorts elements and returns their indices.
    - `GGML_OP_LEAKY_RELU`: Applies the Leaky ReLU activation function.
    - `GGML_OP_FLASH_ATTN_EXT`: Performs extended flash attention.
    - `GGML_OP_FLASH_ATTN_BACK`: Performs the backward pass of flash attention.
    - `GGML_OP_SSM_CONV`: Performs a convolution using the SSM method.
    - `GGML_OP_SSM_SCAN`: Scans a tensor using the SSM method.
    - `GGML_OP_WIN_PART`: Partitions a tensor into non-overlapping windows.
    - `GGML_OP_WIN_UNPART`: Reverses the partitioning of a tensor into windows.
    - `GGML_OP_GET_REL_POS`: Gets relative positions for a tensor.
    - `GGML_OP_ADD_REL_POS`: Adds relative positions to a tensor.
    - `GGML_OP_RWKV_WKV6`: Performs the RWKV WKV6 operation.
    - `GGML_OP_GATED_LINEAR_ATTN`: Applies gated linear attention.
    - `GGML_OP_RWKV_WKV7`: Performs the RWKV WKV7 operation.
    - `GGML_OP_UNARY`: Applies a unary operation to a tensor.
    - `GGML_OP_MAP_CUSTOM1`: Applies a custom unary operation to a tensor.
    - `GGML_OP_MAP_CUSTOM2`: Applies a custom binary operation to two tensors.
    - `GGML_OP_MAP_CUSTOM3`: Applies a custom ternary operation to three tensors.
    - `GGML_OP_CUSTOM`: Applies a custom operation to a tensor.
    - `GGML_OP_CROSS_ENTROPY_LOSS`: Computes the cross-entropy loss.
    - `GGML_OP_CROSS_ENTROPY_LOSS_BACK`: Performs the backward pass of cross-entropy loss.
    - `GGML_OP_OPT_STEP_ADAMW`: Performs an optimization step using the AdamW algorithm.
    - `GGML_OP_COUNT`: Represents the total number of operations available.
- **Description**: The `ggml_op` enum defines a comprehensive set of operations available in the GGML Tensor Library, which is designed for machine learning tasks involving tensor computations. Each enumerator represents a specific tensor operation, ranging from basic arithmetic operations like addition and multiplication to more complex operations such as matrix multiplication, convolution, and various normalization techniques. The enum also includes operations for manipulating tensor shapes, applying activation functions, and performing optimization steps. This extensive list of operations allows users to construct and manipulate computation graphs for machine learning models efficiently.


---
### ggml\_unary\_op
- **Type**: `enum`
- **Members**:
    - `GGML_UNARY_OP_ABS`: Represents the absolute value operation.
    - `GGML_UNARY_OP_SGN`: Represents the signum function operation.
    - `GGML_UNARY_OP_NEG`: Represents the negation operation.
    - `GGML_UNARY_OP_STEP`: Represents the step function operation.
    - `GGML_UNARY_OP_TANH`: Represents the hyperbolic tangent operation.
    - `GGML_UNARY_OP_ELU`: Represents the Exponential Linear Unit operation.
    - `GGML_UNARY_OP_RELU`: Represents the Rectified Linear Unit operation.
    - `GGML_UNARY_OP_SIGMOID`: Represents the sigmoid function operation.
    - `GGML_UNARY_OP_GELU`: Represents the Gaussian Error Linear Unit operation.
    - `GGML_UNARY_OP_GELU_QUICK`: Represents a quick approximation of the GELU operation.
    - `GGML_UNARY_OP_SILU`: Represents the Sigmoid Linear Unit operation.
    - `GGML_UNARY_OP_HARDSWISH`: Represents the hard swish operation.
    - `GGML_UNARY_OP_HARDSIGMOID`: Represents the hard sigmoid operation.
    - `GGML_UNARY_OP_EXP`: Represents the exponential function operation.
    - `GGML_UNARY_OP_GELU_ERF`: Represents the GELU operation using the error function.
    - `GGML_UNARY_OP_COUNT`: Represents the count of unary operations available.
- **Description**: The `ggml_unary_op` enum defines a set of unary operations that can be applied to tensors within the GGML Tensor Library. These operations include various mathematical functions such as absolute value, signum, negation, step function, hyperbolic tangent, and several activation functions commonly used in neural networks like ReLU, ELU, and GELU. The enum provides a way to specify these operations in a standardized manner, facilitating their use in tensor computations and automatic differentiation processes.


---
### ggml\_object\_type
- **Type**: `enum`
- **Members**:
    - `GGML_OBJECT_TYPE_TENSOR`: Represents a tensor object type in the GGML library.
    - `GGML_OBJECT_TYPE_GRAPH`: Represents a computation graph object type in the GGML library.
    - `GGML_OBJECT_TYPE_WORK_BUFFER`: Represents a work buffer object type in the GGML library.
- **Description**: The `ggml_object_type` enumeration defines the different types of objects that can be used within the GGML library, which is a minimalistic tensor library for machine learning tasks. This enum includes types for tensors, computation graphs, and work buffers, each serving a specific role in the library's operations. Tensors are used for data representation, graphs for defining computation flows, and work buffers for temporary data storage during computations.


---
### ggml\_log\_level
- **Type**: `enum`
- **Members**:
    - `GGML_LOG_LEVEL_NONE`: Represents no logging, with a value of 0.
    - `GGML_LOG_LEVEL_DEBUG`: Represents debug level logging, with a value of 1.
    - `GGML_LOG_LEVEL_INFO`: Represents informational level logging, with a value of 2.
    - `GGML_LOG_LEVEL_WARN`: Represents warning level logging, with a value of 3.
    - `GGML_LOG_LEVEL_ERROR`: Represents error level logging, with a value of 4.
    - `GGML_LOG_LEVEL_CONT`: Represents continuation of the previous log, with a value of 5.
- **Description**: The `ggml_log_level` enumeration defines various levels of logging severity for the GGML library, ranging from no logging to error logging, and includes a special level for continuing the previous log. This allows for fine-grained control over the verbosity of log output, which can be useful for debugging and monitoring the library's operations.


---
### ggml\_tensor\_flag
- **Type**: `enum`
- **Members**:
    - `GGML_TENSOR_FLAG_INPUT`: Indicates the tensor is an input for the GGML compute graph.
    - `GGML_TENSOR_FLAG_OUTPUT`: Indicates the tensor is an output for the GGML compute graph.
    - `GGML_TENSOR_FLAG_PARAM`: Indicates the tensor contains trainable parameters.
    - `GGML_TENSOR_FLAG_LOSS`: Indicates the tensor defines loss for numerical optimization, with multiple loss tensors adding up.
- **Description**: The `ggml_tensor_flag` enumeration defines a set of flags used to categorize tensors within the GGML library's computation graph. These flags help in identifying the role of a tensor, such as whether it is an input, output, contains trainable parameters, or is used for loss calculation in optimization processes. This categorization is crucial for managing the flow and processing of data within machine learning tasks handled by the GGML library.


---
### ggml\_init\_params
- **Type**: `struct`
- **Members**:
    - `mem_size`: Specifies the size of the memory pool in bytes.
    - `mem_buffer`: Pointer to a memory buffer; if NULL, memory will be allocated internally.
    - `no_alloc`: Boolean flag indicating whether to allocate memory for tensor data.
- **Description**: The `ggml_init_params` structure is used to initialize the GGML library's memory management system. It defines parameters for setting up a memory pool, including the size of the memory (`mem_size`), a pointer to an external memory buffer (`mem_buffer`), and a flag (`no_alloc`) to control whether memory allocation for tensor data should be performed internally or not. This structure allows users to customize memory allocation strategies when working with tensors in the GGML library.


---
### ggml\_tensor
- **Type**: `struct`
- **Members**:
    - `type`: Specifies the data type of the tensor elements.
    - `buffer`: Pointer to the backend buffer where tensor data is stored.
    - `ne`: Array indicating the number of elements in each dimension of the tensor.
    - `nb`: Array indicating the stride in bytes for each dimension of the tensor.
    - `op`: Specifies the operation associated with the tensor.
    - `op_params`: Array for storing operation parameters, aligned as int32_t.
    - `flags`: Integer flags for additional tensor properties.
    - `src`: Array of pointers to source tensors used to compute the current tensor.
    - `view_src`: Pointer to the source tensor for views.
    - `view_offs`: Offset in bytes for views.
    - `data`: Pointer to the actual data of the tensor.
    - `name`: Character array for storing the name of the tensor.
    - `extra`: Pointer for storing extra information, such as for CUDA operations.
    - `padding`: Padding for alignment purposes.
- **Description**: The `ggml_tensor` structure is a fundamental component of the GGML Tensor Library, designed to represent multi-dimensional tensors used in machine learning computations. It encapsulates information about the tensor's data type, dimensions, and memory layout, including the number of elements and stride for each dimension. The structure also supports operations by storing operation types and parameters, and it maintains references to source tensors for computation graph construction. Additionally, it includes fields for managing tensor views, data storage, and extra metadata, making it versatile for various tensor operations and optimizations.


---
### ggml\_op\_pool
- **Type**: `enum`
- **Members**:
    - `GGML_OP_POOL_MAX`: Represents the maximum pooling operation.
    - `GGML_OP_POOL_AVG`: Represents the average pooling operation.
    - `GGML_OP_POOL_COUNT`: Indicates the count of pooling operations available.
- **Description**: The `ggml_op_pool` enumeration defines a set of constants used to specify different types of pooling operations in the GGML Tensor Library. These operations include maximum pooling (`GGML_OP_POOL_MAX`), average pooling (`GGML_OP_POOL_AVG`), and a count of the available pooling operations (`GGML_OP_POOL_COUNT`). This enumeration is likely used in functions that perform pooling operations on tensors, allowing the user to specify which type of pooling to apply.


---
### ggml\_scale\_mode
- **Type**: `enum`
- **Members**:
    - `GGML_SCALE_MODE_NEAREST`: Represents the nearest neighbor scaling mode with a value of 0.
    - `GGML_SCALE_MODE_BILINEAR`: Represents the bilinear scaling mode with a value of 1.
- **Description**: The `ggml_scale_mode` is an enumeration that defines different modes of scaling for tensor operations within the GGML Tensor Library. It provides two options: nearest neighbor scaling and bilinear scaling, which are used to determine how tensor data is interpolated during scaling operations. This enum is part of the library's functionality to handle various machine learning tasks efficiently by allowing different scaling techniques.


---
### ggml\_sort\_order
- **Type**: `enum`
- **Members**:
    - `GGML_SORT_ORDER_ASC`: Represents ascending sort order.
    - `GGML_SORT_ORDER_DESC`: Represents descending sort order.
- **Description**: The `ggml_sort_order` is an enumeration that defines the sorting order options for operations that require sorting within the GGML Tensor Library. It provides two possible values: `GGML_SORT_ORDER_ASC` for ascending order and `GGML_SORT_ORDER_DESC` for descending order. This enum is used to specify the desired order when performing sorting operations on tensors.


---
### ggml\_type\_traits
- **Type**: `struct`
- **Members**:
    - `type_name`: A pointer to a constant character string representing the name of the type.
    - `blck_size`: An integer representing the block size for the type.
    - `blck_size_interleave`: An integer representing the interleaved block size for elements.
    - `type_size`: A size_t value indicating the size of the type in bytes.
    - `is_quantized`: A boolean indicating whether the type is quantized.
    - `to_float`: A function pointer for converting the type to a float.
    - `from_float_ref`: A function pointer for converting from a float to the type.
- **Description**: The `ggml_type_traits` structure is used to define characteristics and operations associated with different data types in the GGML library. It includes information about the type's name, block size, interleaved block size, and size in bytes. Additionally, it indicates whether the type is quantized and provides function pointers for converting to and from float representations. This structure is essential for handling various data types within the library's tensor operations, ensuring that each type is processed correctly according to its specific traits.


---
### ggml\_sched\_priority
- **Type**: `enum`
- **Members**:
    - `GGML_SCHED_PRIO_LOW`: Represents a low scheduling priority with a value of -1.
    - `GGML_SCHED_PRIO_NORMAL`: Represents a normal scheduling priority.
    - `GGML_SCHED_PRIO_MEDIUM`: Represents a medium scheduling priority.
    - `GGML_SCHED_PRIO_HIGH`: Represents a high scheduling priority.
    - `GGML_SCHED_PRIO_REALTIME`: Represents a real-time scheduling priority.
- **Description**: The `ggml_sched_priority` enumeration defines various levels of scheduling priorities for thread management within the GGML library. These priorities range from low to real-time, allowing for fine-grained control over the execution order and urgency of tasks in a multi-threaded environment. This is particularly useful in optimizing performance and resource allocation in computational tasks.


---
### ggml\_threadpool\_params
- **Type**: `struct`
- **Members**:
    - `cpumask`: An array of booleans representing the mask of CPU cores, where all-zeros means using default affinity settings.
    - `n_threads`: An integer specifying the number of threads.
    - `prio`: An enumeration value indicating the thread priority.
    - `poll`: A 32-bit unsigned integer representing the polling level, with 0 for no polling and 100 for aggressive polling.
    - `strict_cpu`: A boolean indicating whether strict CPU placement is enforced.
    - `paused`: A boolean indicating if the threadpool should start in a paused state.
- **Description**: The `ggml_threadpool_params` structure is used to configure the parameters for a threadpool in the GGML library. It includes settings for CPU core affinity, the number of threads, thread priority, polling level, strict CPU placement, and whether the threadpool should start in a paused state. This structure allows for fine-tuning of threadpool behavior to optimize performance for specific use cases.


# Function Declarations (Public API)

---
### ggml\_abort<!-- {{#callable_declaration:ggml_abort}} -->
Aborts the program and prints an error message.
- **Description**: This function is intended to be called when a critical error occurs, and it should be used to terminate the program immediately. It takes a file name and line number as parameters to provide context for the error, along with a formatted message that describes the issue. It is important to note that this function will not return; it will call `abort()` to terminate the program. Therefore, it should be used in situations where recovery from the error is not possible or desired.
- **Inputs**:
    - `file`: A pointer to a null-terminated string representing the name of the source file where the error occurred. Must not be null.
    - `line`: An integer representing the line number in the source file where the error occurred. Must be a positive integer.
    - `fmt`: A format string for the error message, followed by a variable number of arguments. The format string must be a valid format for `printf`-style functions.
- **Output**: This function does not return a value and does not modify any inputs.
- **See also**: [`ggml_abort`](../src/ggml.c.driver.md#ggml_abort)  (Implementation)


---
### ggml\_status\_to\_string<!-- {{#callable_declaration:ggml_status_to_string}} -->
Converts a `ggml_status` enum value to its corresponding string representation.
- **Description**: This function is used to obtain a human-readable string that describes the status of an operation represented by the `ggml_status` enum. It can be called at any point after an operation that returns a `ggml_status` value to provide feedback on the success or failure of that operation. The function handles known status values and returns a default message for unknown statuses. It is important to ensure that the provided status is a valid `ggml_status` value; otherwise, the function will return a string indicating an unknown status.
- **Inputs**:
    - `status`: An enumeration value of type `ggml_status` that indicates the status of an operation. Valid values include `GGML_STATUS_ALLOC_FAILED`, `GGML_STATUS_FAILED`, `GGML_STATUS_SUCCESS`, and `GGML_STATUS_ABORTED`. The caller must ensure that the value passed is a valid `ggml_status` value; otherwise, the function will return a string indicating an unknown status.
- **Output**: Returns a pointer to a string that describes the status corresponding to the provided `ggml_status` value. If the status is unknown, it returns a string indicating that the status is unknown.
- **See also**: [`ggml_status_to_string`](../src/ggml.c.driver.md#ggml_status_to_string)  (Implementation)


---
### ggml\_fp16\_to\_fp32<!-- {{#callable_declaration:ggml_fp16_to_fp32}} -->
Converts a half-precision floating-point value to a single-precision floating-point value.
- **Description**: This function is used to convert a value of type `ggml_fp16_t`, which represents a half-precision floating-point number, into a single-precision floating-point number (float). It is typically called when there is a need to perform calculations that require higher precision than what half-precision can provide. The input value must be a valid `ggml_fp16_t` type, and the function will return the corresponding float value. There are no side effects, and the function does not modify the input value.
- **Inputs**:
    - `x`: A value of type `ggml_fp16_t` representing a half-precision floating-point number. This value must be a valid half-precision representation. The function does not take ownership of this value, and it is expected to be valid for the duration of the function call.
- **Output**: Returns a float that represents the single-precision equivalent of the input half-precision value.
- **See also**: [`ggml_fp16_to_fp32`](../src/ggml.c.driver.md#ggml_fp16_to_fp32)  (Implementation)


---
### ggml\_fp16\_to\_fp32\_row<!-- {{#callable_declaration:ggml_fp16_to_fp32_row}} -->
Converts an array of half-precision floats to single-precision floats.
- **Description**: This function is used to convert an array of half-precision floating-point values (FP16) to single-precision floating-point values (FP32). It should be called when you need to process or manipulate data that is originally in half-precision format, such as when interfacing with certain hardware or libraries that utilize FP16. The function expects the input array to be valid and the output array to have sufficient allocated memory to hold the converted values. It is important to ensure that the input pointer is not null and that the output pointer is also valid to avoid undefined behavior.
- **Inputs**:
    - `x`: A pointer to an array of `ggml_fp16_t` values. Must not be null. The array should contain at least `n` elements.
    - `y`: A pointer to an array of `float` values where the converted FP32 values will be stored. Must not be null and should be allocated with at least `n` elements.
    - `n`: An integer representing the number of elements to convert. Must be a non-negative value.
- **Output**: The function does not return a value. It populates the output array `y` with the converted single-precision float values.
- **See also**: [`ggml_fp16_to_fp32_row`](../src/ggml.c.driver.md#ggml_fp16_to_fp32_row)  (Implementation)


---
### ggml\_fp32\_to\_fp16\_row<!-- {{#callable_declaration:ggml_fp32_to_fp16_row}} -->
Converts an array of 32-bit floating point numbers to 16-bit floating point numbers.
- **Description**: This function is used to convert an array of `float` values (32-bit) to an array of `ggml_fp16_t` values (16-bit) for efficient storage or processing. It should be called with a valid pointer to the source array and a destination array that has been allocated with sufficient space to hold the converted values. The conversion is performed for `n` elements, where `n` must be a non-negative integer. If `n` is zero, the function will perform no operations.
- **Inputs**:
    - `x`: A pointer to the source array of `float` values. Must not be null. The caller retains ownership of this memory.
    - `y`: A pointer to the destination array of `ggml_fp16_t` values. Must not be null and must be allocated with at least `n` elements. The caller retains ownership of this memory.
    - `n`: The number of elements to convert. Must be a non-negative integer.
- **Output**: The function does not return a value. The output is written directly to the destination array `y`, which will contain the converted 16-bit floating point values.
- **See also**: [`ggml_fp32_to_fp16_row`](../src/ggml.c.driver.md#ggml_fp32_to_fp16_row)  (Implementation)


---
### ggml\_bf16\_to\_fp32<!-- {{#callable_declaration:ggml_bf16_to_fp32}} -->
Converts a bfloat16 value to a float32 value.
- **Description**: This function is used to convert a value from bfloat16 format to float32 format. It is typically called when there is a need to process or manipulate data that is originally in bfloat16 format, such as in machine learning applications. The input must be a valid `ggml_bf16_t` type, which represents a bfloat16 value. The function does not perform any checks on the input value, so it is the caller's responsibility to ensure that the input is valid. The output is a float32 representation of the input bfloat16 value.
- **Inputs**:
    - `x`: A `ggml_bf16_t` type representing the bfloat16 value to be converted. This parameter must not be null.
- **Output**: Returns a float32 value that is the converted representation of the input bfloat16 value.
- **See also**: [`ggml_bf16_to_fp32`](../src/ggml.c.driver.md#ggml_bf16_to_fp32)  (Implementation)


---
### ggml\_bf16\_to\_fp32\_row<!-- {{#callable_declaration:ggml_bf16_to_fp32_row}} -->
Converts an array of bfloat16 values to an array of float values.
- **Description**: This function is used to convert an array of bfloat16 values into an array of float values. It should be called when you have a valid pointer to an array of `ggml_bf16_t` values and a corresponding pointer to a float array where the results will be stored. The parameter `n` specifies the number of elements to convert. It is important to ensure that the destination array has enough space to hold the converted float values. If `n` is less than or equal to zero, no conversion will be performed.
- **Inputs**:
    - `x`: A pointer to an array of `ggml_bf16_t` values. Must not be null and should point to a valid memory region containing at least `n` elements.
    - `y`: A pointer to an array of float values where the converted results will be stored. Must not be null and should have enough space to hold at least `n` float values.
    - `n`: An integer representing the number of elements to convert. Must be greater than zero.
- **Output**: The function does not return a value. It populates the array pointed to by `y` with the converted float values corresponding to the bfloat16 values in the array pointed to by `x`.
- **See also**: [`ggml_bf16_to_fp32_row`](../src/ggml.c.driver.md#ggml_bf16_to_fp32_row)  (Implementation)


---
### ggml\_fp32\_to\_bf16\_row\_ref<!-- {{#callable_declaration:ggml_fp32_to_bf16_row_ref}} -->
Converts an array of 32-bit floating-point numbers to bfloat16 format.
- **Description**: This function is used to convert an array of 32-bit floating-point numbers into bfloat16 format, which is often used in machine learning applications for reduced precision. It should be called when you need to prepare data for models that utilize bfloat16 representation. The input array must not be null, and the output array must be allocated with sufficient space to hold the converted values. The function processes 'n' elements from the input array, where 'n' must be a non-negative integer. If 'n' is zero, the function will perform no operations.
- **Inputs**:
    - `x`: A pointer to the input array of 32-bit floating-point numbers. Must not be null.
    - `y`: A pointer to the output array where the converted bfloat16 values will be stored. Must not be null and must have enough allocated space to hold 'n' elements.
    - `n`: An integer representing the number of elements to convert. Must be a non-negative integer.
- **Output**: The function does not return a value. It populates the output array 'y' with the converted bfloat16 values.
- **See also**: [`ggml_fp32_to_bf16_row_ref`](../src/ggml.c.driver.md#ggml_fp32_to_bf16_row_ref)  (Implementation)


---
### ggml\_fp32\_to\_bf16\_row<!-- {{#callable_declaration:ggml_fp32_to_bf16_row}} -->
Converts an array of 32-bit floating-point numbers to bfloat16 format.
- **Description**: This function is used to convert an array of 32-bit floating-point numbers into bfloat16 format, which is often used in machine learning applications for reduced precision. It should be called when you need to prepare data for models that utilize bfloat16 representation. The input array must not be null, and the length of the array specified by the parameter `n` must be greater than zero. The output array will be populated with the converted bfloat16 values. If the input is invalid (e.g., null pointer or negative size), the behavior is undefined.
- **Inputs**:
    - `x`: A pointer to the input array of 32-bit floating-point numbers. Must not be null. The array should contain at least `n` elements.
    - `y`: A pointer to the output array where the converted bfloat16 values will be stored. Must not be null and should have enough space to hold at least `n` elements.
    - `n`: An integer representing the number of elements to convert. Must be greater than zero.
- **Output**: The function does not return a value. It populates the output array `y` with the converted bfloat16 values corresponding to the input array `x`.
- **See also**: [`ggml_fp32_to_bf16_row`](../src/ggml.c.driver.md#ggml_fp32_to_bf16_row)  (Implementation)


---
### ggml\_guid\_matches<!-- {{#callable_declaration:ggml_guid_matches}} -->
Compares two GUIDs for equality.
- **Description**: This function is used to determine if two GUIDs are identical. It should be called when you need to check if two GUIDs represent the same entity, such as when validating unique identifiers in a system. Both GUIDs must be valid and properly initialized; otherwise, the behavior is undefined. The function performs a direct comparison of the two GUIDs and returns true if they match, and false otherwise.
- **Inputs**:
    - `guid_a`: The first GUID to compare. Must not be null and should be a valid GUID.
    - `guid_b`: The second GUID to compare. Must not be null and should be a valid GUID.
- **Output**: Returns true if the two GUIDs are equal, otherwise returns false.
- **See also**: [`ggml_guid_matches`](../src/ggml.c.driver.md#ggml_guid_matches)  (Implementation)


---
### ggml\_cycles<!-- {{#callable_declaration:ggml_cycles}} -->
Returns the number of clock cycles since the program started.
- **Description**: This function can be called at any point in the program to retrieve the current number of clock cycles. It is useful for performance measurement and profiling, allowing developers to track the execution time of specific code segments. There are no preconditions for calling this function, and it does not modify any state or data.
- **Inputs**: None
- **Output**: Returns an `int64_t` value representing the number of clock cycles since the program started.
- **See also**: [`ggml_cycles`](../src/ggml.c.driver.md#ggml_cycles)  (Implementation)


---
### ggml\_cycles\_per\_ms<!-- {{#callable_declaration:ggml_cycles_per_ms}} -->
Returns the number of clock cycles per millisecond.
- **Description**: This function provides the number of clock cycles that occur in one millisecond, which can be useful for performance measurements and timing operations in applications. It is typically called when precise timing is required, such as benchmarking or profiling code execution. There are no specific preconditions for calling this function, and it does not modify any state or data.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns an integer value representing the number of clock cycles per millisecond.
- **See also**: [`ggml_cycles_per_ms`](../src/ggml.c.driver.md#ggml_cycles_per_ms)  (Implementation)


---
### ggml\_fopen<!-- {{#callable_declaration:ggml_fopen}} -->
Opens a file with the specified mode.
- **Description**: This function is used to open a file for reading or writing, depending on the specified mode. It accepts a UTF-8 encoded file name, making it suitable for cross-platform use, including Windows. The function should be called with a valid file name and mode string, where the mode string follows the standard C library conventions (e.g., "r" for reading, "w" for writing). If the file cannot be opened, the function will return a null pointer, which should be checked by the caller to handle any errors appropriately.
- **Inputs**:
    - `fname`: The name of the file to open, encoded in UTF-8. Must not be null.
    - `mode`: The mode in which to open the file, following standard C conventions (e.g., "r", "w"). Must not be null.
- **Output**: Returns a pointer to a `FILE` object associated with the opened file, or null if the file could not be opened.
- **See also**: [`ggml_fopen`](../src/ggml.c.driver.md#ggml_fopen)  (Implementation)


---
### ggml\_print\_object<!-- {{#callable_declaration:ggml_print_object}} -->
Prints the details of a `ggml_object`.
- **Description**: This function is used to log the properties of a `ggml_object`, including its type, offset, size, and the pointer to the next object in the linked list. It is typically called for debugging or informational purposes to understand the structure and state of the object. The function should be called with a valid pointer to a `ggml_object`, and it is important to ensure that the object is properly initialized before calling this function.
- **Inputs**:
    - `obj`: A pointer to a `ggml_object` structure. Must not be null and should point to a valid, initialized object. If the pointer is null or points to an uninitialized object, the behavior is undefined.
- **Output**: This function does not return a value and does not modify any inputs.
- **See also**: [`ggml_print_object`](../src/ggml.c.driver.md#ggml_print_object)  (Implementation)


---
### ggml\_print\_objects<!-- {{#callable_declaration:ggml_print_objects}} -->
Prints all objects in the specified context.
- **Description**: This function is used to enumerate and print details of all objects associated with a given context. It should be called after the context has been initialized and populated with objects. The function iterates through the linked list of objects starting from the beginning of the context's object list and prints each object's details. If the context is null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `struct ggml_context` that represents the context containing the objects to be printed. This parameter must not be null; passing a null pointer may lead to undefined behavior.
- **Output**: None
- **See also**: [`ggml_print_objects`](../src/ggml.c.driver.md#ggml_print_objects)  (Implementation)


---
### ggml\_nelements<!-- {{#callable_declaration:ggml_nelements}} -->
Returns the total number of elements in a tensor.
- **Description**: This function calculates the total number of elements in a given tensor by multiplying the sizes of all its dimensions. It is essential to call this function only after the tensor has been properly initialized and allocated. If the provided tensor pointer is null, the behavior is undefined, and the function may lead to a crash or unexpected results.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor. This pointer must not be null, and the tensor must be properly initialized. The function will access the tensor's dimension sizes to compute the total number of elements.
- **Output**: Returns an int64_t value representing the total number of elements in the tensor. If the tensor is empty or improperly initialized, the result may not be meaningful.
- **See also**: [`ggml_nelements`](../src/ggml.c.driver.md#ggml_nelements)  (Implementation)


---
### ggml\_nrows<!-- {{#callable_declaration:ggml_nrows}} -->
Returns the number of rows in a tensor.
- **Description**: This function is used to retrieve the number of rows in a multi-dimensional tensor. It is particularly useful when working with tensors that have more than one dimension, as it allows users to understand the structure of the tensor. The function should be called with a valid `ggml_tensor` pointer that has been properly initialized. If the provided tensor pointer is null, the behavior is undefined.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor from which to retrieve the number of rows. This pointer must not be null, and the tensor must be properly initialized and allocated.
- **Output**: Returns an integer value representing the number of rows in the tensor. The value is calculated based on the dimensions of the tensor, specifically the second, third, and fourth dimensions.
- **See also**: [`ggml_nrows`](../src/ggml.c.driver.md#ggml_nrows)  (Implementation)


---
### ggml\_nbytes<!-- {{#callable_declaration:ggml_nbytes}} -->
Calculates the number of bytes required for a tensor.
- **Description**: This function is used to determine the memory size needed to store the data of a tensor. It should be called after the tensor has been properly initialized and its dimensions set. If any dimension of the tensor is less than or equal to zero, the function will return zero, indicating that the tensor is invalid or empty. The function takes into account the tensor's data type and its dimensions to compute the total byte size required.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor. This parameter must not be null and should point to a valid tensor that has been initialized with valid dimensions.
- **Output**: Returns the size in bytes required to store the tensor's data. If the tensor is invalid (i.e., any dimension is less than or equal to zero), it returns zero.
- **See also**: [`ggml_nbytes`](../src/ggml.c.driver.md#ggml_nbytes)  (Implementation)


---
### ggml\_nbytes\_pad<!-- {{#callable_declaration:ggml_nbytes_pad}} -->
Calculates the padded size of a tensor in bytes.
- **Description**: This function is used to determine the total memory size required for a tensor, including padding to ensure proper alignment. It should be called when you need to allocate memory for a tensor and want to ensure that the memory is aligned according to the specified memory alignment requirements. The function expects a valid `ggml_tensor` pointer as input, and it will return the padded size in bytes. If the input tensor is null, the behavior is undefined.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor for which the padded size is to be calculated. This pointer must not be null; otherwise, the behavior is undefined.
- **Output**: Returns the size in bytes required for the tensor, including any necessary padding for memory alignment.
- **See also**: [`ggml_nbytes_pad`](../src/ggml.c.driver.md#ggml_nbytes_pad)  (Implementation)


---
### ggml\_blck\_size<!-- {{#callable_declaration:ggml_blck_size}} -->
Returns the block size for a given tensor type.
- **Description**: This function retrieves the block size associated with a specified tensor type, which is essential for memory allocation and tensor operations. It should be called with a valid `ggml_type` enumeration value, which represents the data type of the tensor. If an invalid type is provided, the behavior is undefined, so it is crucial to ensure that the type is within the defined range of `ggml_type`. This function does not modify any input parameters and is safe to call at any point after the library has been initialized.
- **Inputs**:
    - `type`: An enumeration of type `ggml_type` that specifies the tensor data type. Valid values range from `GGML_TYPE_F32` to `GGML_TYPE_COUNT`. Must not be null.
- **Output**: Returns the block size as an `int64_t` value, which indicates the number of elements in a block for the specified tensor type.
- **See also**: [`ggml_blck_size`](../src/ggml.c.driver.md#ggml_blck_size)  (Implementation)


---
### ggml\_type\_size<!-- {{#callable_declaration:ggml_type_size}} -->
Returns the size in bytes of the specified data type.
- **Description**: This function is used to determine the size in bytes of a specific data type defined in the `ggml_type` enumeration. It should be called with a valid `ggml_type` value, which represents the type of tensor elements. The function will return the size associated with that type, allowing users to allocate memory or perform operations based on the size of the data type. If an invalid type is provided, the behavior is undefined.
- **Inputs**:
    - `type`: An enumeration value of type `ggml_type` that specifies the data type. Valid values are defined in the `ggml_type` enum. Must not be null.
- **Output**: Returns the size in bytes of the specified data type as a `size_t` value.
- **See also**: [`ggml_type_size`](../src/ggml.c.driver.md#ggml_type_size)  (Implementation)


---
### ggml\_row\_size<!-- {{#callable_declaration:ggml_row_size}} -->
Calculates the size of a row in bytes for a given tensor type and number of elements.
- **Description**: This function is used to determine the memory size required for a row of a tensor based on its type and the number of elements. It is essential to call this function with a valid tensor type and a number of elements that is a multiple of the block size for that type. If the number of elements is not a multiple of the block size, the behavior is undefined, and the function may not return a valid size.
- **Inputs**:
    - `type`: Specifies the data type of the tensor. Valid values are defined in the `ggml_type` enumeration. Must not be null.
    - `ne`: Indicates the number of elements in the row. This value must be a non-negative integer and should be a multiple of the block size for the specified type.
- **Output**: Returns the size in bytes of a row for the specified tensor type and number of elements.
- **See also**: [`ggml_row_size`](../src/ggml.c.driver.md#ggml_row_size)  (Implementation)


---
### ggml\_type\_name<!-- {{#callable_declaration:ggml_type_name}} -->
Returns the name of the specified tensor type.
- **Description**: This function retrieves the name associated with a given tensor type, which is represented by the `ggml_type` enumeration. It is useful for obtaining a human-readable string that describes the type of tensor being used, which can aid in debugging or logging. The function should be called with a valid `ggml_type` value that is less than `GGML_TYPE_COUNT`. If an invalid type is provided (i.e., one that is not defined in the enumeration), the function will return the string "NONE".
- **Inputs**:
    - `type`: An enumeration value of type `ggml_type` representing the tensor type. Valid values are from `GGML_TYPE_F32` to `GGML_TYPE_IQ4_XS`, and must be less than `GGML_TYPE_COUNT`. If an invalid value is passed, the function will return "NONE".
- **Output**: Returns a pointer to a string containing the name of the specified tensor type. If the type is invalid, it returns the string "NONE".
- **See also**: [`ggml_type_name`](../src/ggml.c.driver.md#ggml_type_name)  (Implementation)


---
### ggml\_op\_name<!-- {{#callable_declaration:ggml_op_name}} -->
Returns the name of the specified tensor operation.
- **Description**: This function retrieves the name associated with a specific tensor operation, identified by the `op` parameter. It is useful for debugging or logging purposes, allowing developers to obtain a human-readable string representation of the operation type. The function should be called with a valid `ggml_op` enumeration value. If an invalid value is provided, the behavior is undefined, so it is important to ensure that the `op` parameter is within the valid range of defined operations.
- **Inputs**:
    - `op`: An enumeration value of type `ggml_op` that specifies the tensor operation for which the name is requested. Valid values are defined in the `ggml_op` enum. The caller must ensure that the value is within the defined range to avoid undefined behavior.
- **Output**: Returns a pointer to a string containing the name of the specified tensor operation. If the operation is invalid, the returned string may be undefined.
- **See also**: [`ggml_op_name`](../src/ggml.c.driver.md#ggml_op_name)  (Implementation)


---
### ggml\_op\_symbol<!-- {{#callable_declaration:ggml_op_symbol}} -->
Returns the symbol associated with a specified operation.
- **Description**: This function retrieves the string representation of a specified tensor operation, identified by the `op` parameter. It is useful for debugging or logging purposes, allowing developers to understand which operation is being performed. The function should be called with a valid `ggml_op` enumeration value. If an invalid value is provided, the behavior is undefined, so it is important to ensure that the `op` parameter is within the valid range of defined operations.
- **Inputs**:
    - `op`: An enumeration value of type `ggml_op` that specifies the operation for which the symbol is requested. Valid values are defined in the `ggml_op` enum. The caller must ensure that the value is within the defined range; otherwise, the behavior is undefined.
- **Output**: Returns a pointer to a string that represents the symbol of the specified operation. The returned string is valid as long as the library is in use and should not be modified or freed by the caller.
- **See also**: [`ggml_op_symbol`](../src/ggml.c.driver.md#ggml_op_symbol)  (Implementation)


---
### ggml\_unary\_op\_name<!-- {{#callable_declaration:ggml_unary_op_name}} -->
Returns the name of a unary operation.
- **Description**: This function retrieves the name associated with a specified unary operation from the `ggml_unary_op` enumeration. It is typically used when you need to display or log the name of a unary operation for debugging or informational purposes. The input parameter must be a valid enumeration value from the `ggml_unary_op` enum. If an invalid value is provided, the behavior is undefined, so it is important to ensure that the value is within the valid range.
- **Inputs**:
    - `op`: An enumeration value of type `ggml_unary_op` representing the unary operation. Valid values are defined in the `ggml_unary_op` enum. The caller must ensure that the value is valid; otherwise, the function may return an undefined result.
- **Output**: Returns a pointer to a string containing the name of the unary operation corresponding to the provided enumeration value.
- **See also**: [`ggml_unary_op_name`](../src/ggml.c.driver.md#ggml_unary_op_name)  (Implementation)


---
### ggml\_op\_desc<!-- {{#callable_declaration:ggml_op_desc}} -->
Returns the operation description of a tensor.
- **Description**: This function retrieves a string description of the operation associated with a given tensor. It should be called with a valid `ggml_tensor` that has been properly initialized. If the tensor represents a unary operation, the function will return the name of that unary operation; otherwise, it will return the name of the operation represented by the tensor. It is important to ensure that the tensor is not null before calling this function, as passing a null pointer may lead to undefined behavior.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` structure representing the tensor whose operation description is to be retrieved. This pointer must not be null and should point to a valid tensor that has been initialized.
- **Output**: Returns a pointer to a string that describes the operation associated with the tensor. The string is valid as long as the tensor remains in scope.
- **See also**: [`ggml_op_desc`](../src/ggml.c.driver.md#ggml_op_desc)  (Implementation)


---
### ggml\_element\_size<!-- {{#callable_declaration:ggml_element_size}} -->
Returns the size of an element in a tensor.
- **Description**: This function is used to determine the size in bytes of a single element in a tensor, which is essential for memory management and data manipulation tasks. It should be called with a valid `ggml_tensor` pointer that has been properly initialized. If the provided tensor pointer is null, the behavior is undefined, and the function may not return a valid size.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure. Must not be null and should point to a valid tensor that has been initialized. If the tensor is not properly initialized, the function may return an incorrect size.
- **Output**: Returns the size in bytes of a single element of the tensor, based on its type.
- **See also**: [`ggml_element_size`](../src/ggml.c.driver.md#ggml_element_size)  (Implementation)


---
### ggml\_is\_quantized<!-- {{#callable_declaration:ggml_is_quantized}} -->
Determines if a tensor type is quantized.
- **Description**: This function checks whether the specified tensor type is quantized, which is important for understanding how the tensor data is represented and processed. It should be called with a valid `ggml_type` enumeration value. If an invalid type is provided, the behavior is undefined, so it is crucial to ensure that the type is within the defined range of `ggml_type`. This function does not modify any input parameters.
- **Inputs**:
    - `type`: An enumeration value of type `ggml_type` that specifies the tensor type to check. Valid values are defined in the `ggml_type` enum. Must not be null or out of range; otherwise, the behavior is undefined.
- **Output**: Returns a boolean value: `true` if the specified tensor type is quantized, and `false` otherwise.
- **See also**: [`ggml_is_quantized`](../src/ggml.c.driver.md#ggml_is_quantized)  (Implementation)


---
### ggml\_ftype\_to\_ggml\_type<!-- {{#callable_declaration:ggml_ftype_to_ggml_type}} -->
Converts a file type enumeration to a corresponding tensor type.
- **Description**: This function is used to map a given file type enumeration to its corresponding tensor type. It should be called with a valid `ggml_ftype` value, which represents the format of the data being processed. The function will return the appropriate `ggml_type` based on the provided file type. If the input file type is unknown or not supported, the function will return `GGML_TYPE_COUNT`. It is important to ensure that the input value is a valid `ggml_ftype` to avoid unexpected results.
- **Inputs**:
    - `ftype`: An enumeration of type `ggml_ftype` that specifies the file type. Valid values include various types such as `GGML_FTYPE_ALL_F32`, `GGML_FTYPE_MOSTLY_F16`, etc. The caller must ensure that this parameter is a valid enumeration value; otherwise, the function may return `GGML_TYPE_COUNT`.
- **Output**: Returns an enumeration of type `ggml_type` that corresponds to the provided file type. If the input file type is unknown or unsupported, it returns `GGML_TYPE_COUNT`.
- **See also**: [`ggml_ftype_to_ggml_type`](../src/ggml.c.driver.md#ggml_ftype_to_ggml_type)  (Implementation)


---
### ggml\_is\_transposed<!-- {{#callable_declaration:ggml_is_transposed}} -->
Checks if a tensor is transposed.
- **Description**: This function determines whether the given tensor is stored in a transposed format, which is indicated by comparing the strides of its dimensions. It should be called with a valid `ggml_tensor` pointer that has been properly initialized. If the tensor is not initialized or is null, the behavior is undefined.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor to be checked. This pointer must not be null and should point to a valid tensor that has been initialized. If the pointer is null or invalid, the function's behavior is undefined.
- **Output**: Returns a boolean value: true if the tensor is transposed (i.e., the first dimension's stride is greater than the second dimension's stride), and false otherwise.
- **See also**: [`ggml_is_transposed`](../src/ggml.c.driver.md#ggml_is_transposed)  (Implementation)


---
### ggml\_is\_permuted<!-- {{#callable_declaration:ggml_is_permuted}} -->
Checks if a tensor is permuted.
- **Description**: This function is used to determine if a given tensor has been permuted, which means that its dimensions are not in a standard order. It should be called after the tensor has been created and initialized. The function evaluates the tensor's stride values to ascertain if any of them indicate a non-standard arrangement. If the tensor is null, the behavior is undefined, and the function may not return a valid result.
- **Inputs**:
    - `tensor`: A pointer to a `struct ggml_tensor`. This parameter must not be null, and it should point to a valid tensor structure. If the pointer is null, the function's behavior is undefined.
- **Output**: Returns a boolean value: true if the tensor is permuted, and false otherwise.
- **See also**: [`ggml_is_permuted`](../src/ggml.c.driver.md#ggml_is_permuted)  (Implementation)


---
### ggml\_is\_empty<!-- {{#callable_declaration:ggml_is_empty}} -->
Checks if a tensor is empty.
- **Description**: This function is used to determine whether a given tensor is empty, which is defined as having at least one dimension with zero elements. It is important to call this function when you need to validate the state of a tensor before performing operations that require non-empty tensors. The input tensor must be a valid pointer to a `struct ggml_tensor`. If the tensor is null, the behavior is undefined.
- **Inputs**:
    - `tensor`: A pointer to a `struct ggml_tensor`. Must not be null. If the tensor has any dimension with zero elements, the function will return true, indicating that the tensor is empty.
- **Output**: Returns true if the tensor is empty (i.e., any dimension has zero elements), otherwise returns false.
- **See also**: [`ggml_is_empty`](../src/ggml.c.driver.md#ggml_is_empty)  (Implementation)


---
### ggml\_is\_scalar<!-- {{#callable_declaration:ggml_is_scalar}} -->
Determines if a tensor is a scalar.
- **Description**: This function checks whether the provided tensor is a scalar, which is defined as having a size of 1 in all dimensions. It should be called when you need to verify if a tensor represents a single value, particularly before performing operations that require scalar inputs. The input tensor must be valid and properly initialized; otherwise, the behavior is undefined.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor to be checked. This pointer must not be null and should point to a valid tensor object.
- **Output**: Returns true if the tensor is a scalar (i.e., has a size of 1 in all dimensions), otherwise returns false.
- **See also**: [`ggml_is_scalar`](../src/ggml.c.driver.md#ggml_is_scalar)  (Implementation)


---
### ggml\_is\_vector<!-- {{#callable_declaration:ggml_is_vector}} -->
Checks if a tensor is a vector.
- **Description**: This function is used to determine if a given tensor is a vector, which is defined as a tensor with only one non-singleton dimension. It should be called with a valid `ggml_tensor` structure that has been properly initialized. The function checks the dimensions of the tensor and returns true if it is a vector (i.e., it has a shape of [n, 1, 1, 1] for some n), and false otherwise. It is important to ensure that the tensor is not null before calling this function, as passing a null pointer may lead to undefined behavior.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor to be checked. Must not be null. The function expects the tensor to be properly initialized with valid dimensions.
- **Output**: Returns true if the tensor is a vector, otherwise returns false.
- **See also**: [`ggml_is_vector`](../src/ggml.c.driver.md#ggml_is_vector)  (Implementation)


---
### ggml\_is\_matrix<!-- {{#callable_declaration:ggml_is_matrix}} -->
Determines if a tensor is a matrix.
- **Description**: This function checks if the provided tensor is a matrix by verifying that its dimensions are compatible with a matrix structure. Specifically, it ensures that the third and fourth dimensions of the tensor are both equal to 1, which is a requirement for a tensor to be considered a matrix in this context. It should be called with a valid `ggml_tensor` pointer that has been properly initialized. If the tensor is null or not properly initialized, the behavior is undefined.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor to be checked. This pointer must not be null and should point to a valid tensor that has been initialized. If the tensor is null or not properly initialized, the function's behavior is undefined.
- **Output**: Returns a boolean value: true if the tensor is a matrix (i.e., it has dimensions compatible with a matrix), and false otherwise.
- **See also**: [`ggml_is_matrix`](../src/ggml.c.driver.md#ggml_is_matrix)  (Implementation)


---
### ggml\_is\_3d<!-- {{#callable_declaration:ggml_is_3d}} -->
Checks if a tensor is 3D.
- **Description**: This function is used to determine if a given tensor is a 3-dimensional tensor. It should be called with a valid `ggml_tensor` pointer that has been properly initialized. The function checks the size of the fourth dimension of the tensor, which must be equal to 1 for the tensor to be considered 3D. If the tensor is not properly initialized or if the pointer is null, the behavior is undefined.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor to be checked. This pointer must not be null and should point to a valid tensor that has been initialized. The function does not handle null pointers or uninitialized tensors, which may lead to undefined behavior.
- **Output**: Returns true if the tensor is 3D (i.e., its fourth dimension is 1), and false otherwise.
- **See also**: [`ggml_is_3d`](../src/ggml.c.driver.md#ggml_is_3d)  (Implementation)


---
### ggml\_n\_dims<!-- {{#callable_declaration:ggml_n_dims}} -->
Returns the number of dimensions of a tensor.
- **Description**: This function is used to determine the number of dimensions of a given tensor. It should be called with a valid `ggml_tensor` pointer that has been properly initialized. The function checks the dimensions of the tensor and returns the highest dimension index that has more than one element. If all dimensions have only one element, it will return 1, indicating that the tensor is effectively a scalar. It is important to ensure that the `tensor` parameter is not null; passing a null pointer will lead to undefined behavior.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor whose dimensions are to be queried. This pointer must not be null and must point to a valid tensor object.
- **Output**: Returns an integer representing the number of dimensions of the tensor. The value will be between 1 and `GGML_MAX_DIMS`, inclusive.
- **See also**: [`ggml_n_dims`](../src/ggml.c.driver.md#ggml_n_dims)  (Implementation)


---
### ggml\_is\_contiguous<!-- {{#callable_declaration:ggml_is_contiguous}} -->
Checks if a tensor is contiguous in memory.
- **Description**: This function is used to determine whether the elements of a given tensor are stored in a contiguous block of memory, which is important for performance optimization in tensor operations. It should be called with a valid `ggml_tensor` pointer that has been properly initialized. If the tensor is null, the behavior is undefined, and the function may return false. This function is particularly useful before performing operations that require contiguous memory access.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure. Must not be null. If the tensor is not properly initialized or is null, the function may return false.
- **Output**: Returns true if the tensor's elements are contiguous in memory; otherwise, returns false.
- **See also**: [`ggml_is_contiguous`](../src/ggml.c.driver.md#ggml_is_contiguous)  (Implementation)


---
### ggml\_is\_contiguous\_0<!-- {{#callable_declaration:ggml_is_contiguous_0}} -->
Checks if a tensor is contiguous in memory.
- **Description**: This function is used to determine if the elements of a given tensor are stored in a contiguous block of memory. It is particularly useful when performing operations that require contiguous memory access for efficiency. The function should be called with a valid `ggml_tensor` pointer that has been properly initialized. If the tensor pointer is null, the behavior is undefined.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor to be checked. Must not be null; passing a null pointer results in undefined behavior.
- **Output**: Returns a boolean value: true if the tensor is contiguous in memory, false otherwise.
- **See also**: [`ggml_is_contiguous_0`](../src/ggml.c.driver.md#ggml_is_contiguous_0)  (Implementation)


---
### ggml\_is\_contiguous\_1<!-- {{#callable_declaration:ggml_is_contiguous_1}} -->
Checks if a tensor is contiguous for one dimension.
- **Description**: This function is used to determine if the elements of a given tensor are stored contiguously in memory for the first dimension. It is particularly useful when performing operations that require contiguous memory access for efficiency. The function should be called with a valid `ggml_tensor` pointer that has been properly initialized. If the tensor is null, the behavior is undefined.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor to be checked. Must not be null.
- **Output**: Returns true if the tensor's elements are contiguous in memory for the first dimension, otherwise returns false.
- **See also**: [`ggml_is_contiguous_1`](../src/ggml.c.driver.md#ggml_is_contiguous_1)  (Implementation)


---
### ggml\_is\_contiguous\_2<!-- {{#callable_declaration:ggml_is_contiguous_2}} -->
Checks if a tensor is contiguous in two dimensions.
- **Description**: This function is used to determine if the elements of a tensor are stored contiguously in memory for the first two dimensions. It should be called when you need to ensure that the tensor's data layout is suitable for certain operations that require contiguous memory access. The input tensor must be a valid pointer to a `ggml_tensor` structure; passing a null pointer will result in undefined behavior.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor to be checked. Must not be null. If the tensor is not properly initialized or is null, the behavior of the function is undefined.
- **Output**: Returns a boolean value: true if the tensor is contiguous in two dimensions, false otherwise.
- **See also**: [`ggml_is_contiguous_2`](../src/ggml.c.driver.md#ggml_is_contiguous_2)  (Implementation)


---
### ggml\_is\_contiguously\_allocated<!-- {{#callable_declaration:ggml_is_contiguously_allocated}} -->
Checks if a tensor is allocated in a contiguous block of memory.
- **Description**: This function is used to determine whether the memory allocated for a tensor is contiguous, meaning that all elements are stored in a single, continuous block of memory without gaps. It is important to call this function when you need to ensure that tensor operations that require contiguous memory can be performed efficiently. The input tensor must be valid and properly initialized; passing a null pointer or an uninitialized tensor may lead to undefined behavior.
- **Inputs**:
    - `tensor`: A pointer to a `struct ggml_tensor`. This parameter must not be null and should point to a valid tensor that has been initialized. If the tensor is null or uninitialized, the behavior of the function is undefined.
- **Output**: Returns a boolean value: true if the tensor is allocated in a contiguous block of memory, and false otherwise.
- **See also**: [`ggml_is_contiguously_allocated`](../src/ggml.c.driver.md#ggml_is_contiguously_allocated)  (Implementation)


---
### ggml\_is\_contiguous\_channels<!-- {{#callable_declaration:ggml_is_contiguous_channels}} -->
Checks if the tensor has contiguous channels.
- **Description**: This function is used to determine if the channels of a tensor are stored contiguously in memory. It should be called when you need to verify the memory layout of a tensor, particularly before performing operations that assume contiguous memory. The function checks the tensor's stride values to ensure that the first dimension is larger than the second, and that the third dimension's stride matches the size of the tensor's data type. It is important to ensure that the `tensor` parameter is valid and properly initialized before calling this function.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor to be checked. This pointer must not be null and should point to a valid tensor that has been initialized.
- **Output**: Returns `true` if the tensor's channels are contiguous in memory, otherwise returns `false`.
- **See also**: [`ggml_is_contiguous_channels`](../src/ggml.c.driver.md#ggml_is_contiguous_channels)  (Implementation)


---
### ggml\_are\_same\_shape<!-- {{#callable_declaration:ggml_are_same_shape}} -->
Determines if two tensors have the same shape.
- **Description**: This function is used to check if two tensors have the same dimensions. It is essential to call this function when you need to ensure that two tensors can be used together in operations that require matching shapes, such as addition or multiplication. The function compares the number of elements in each dimension of the two tensors. Both input tensors must be valid and initialized; otherwise, the behavior is undefined.
- **Inputs**:
    - `t0`: A pointer to the first tensor to compare. Must not be null and should point to a valid `ggml_tensor` structure.
    - `t1`: A pointer to the second tensor to compare. Must not be null and should point to a valid `ggml_tensor` structure.
- **Output**: Returns true if both tensors have the same shape (i.e., the same number of elements in each dimension), otherwise returns false.
- **See also**: [`ggml_are_same_shape`](../src/ggml.c.driver.md#ggml_are_same_shape)  (Implementation)


---
### ggml\_are\_same\_stride<!-- {{#callable_declaration:ggml_are_same_stride}} -->
Checks if two tensors have the same stride.
- **Description**: This function is used to determine if two tensors have the same memory layout in terms of stride, which is crucial for certain tensor operations that require aligned data. It should be called when you need to ensure that two tensors can be processed together without issues related to their memory layout. Both input tensors must be valid and properly initialized; otherwise, the behavior is undefined. The function does not modify the input tensors.
- **Inputs**:
    - `t0`: A pointer to the first tensor (`struct ggml_tensor`). Must not be null and should point to a valid tensor structure.
    - `t1`: A pointer to the second tensor (`struct ggml_tensor`). Must not be null and should point to a valid tensor structure.
- **Output**: Returns `true` if both tensors have the same stride in all dimensions, otherwise returns `false`.
- **See also**: [`ggml_are_same_stride`](../src/ggml.c.driver.md#ggml_are_same_stride)  (Implementation)


---
### ggml\_can\_repeat<!-- {{#callable_declaration:ggml_can_repeat}} -->
Determines if two tensors can repeat.
- **Description**: This function checks if the dimensions of two tensors allow for one to repeat the other. It is particularly useful in scenarios where tensor operations require matching dimensions, such as broadcasting. The function should be called with valid tensor pointers, and it will return true if the second tensor can be formed by repeating the first tensor along its dimensions. If either tensor is empty, the function will return true only if both tensors are empty.
- **Inputs**:
    - `t0`: A pointer to the first tensor. Must not be null. The tensor's dimensions are checked against the second tensor to determine if it can be repeated.
    - `t1`: A pointer to the second tensor. Must not be null. The tensor's dimensions are checked against the first tensor to determine if it can be formed by repeating the first tensor.
- **Output**: Returns true if the second tensor can be formed by repeating the first tensor along its dimensions, or if both tensors are empty. Returns false otherwise.
- **See also**: [`ggml_can_repeat`](../src/ggml.c.driver.md#ggml_can_repeat)  (Implementation)


---
### ggml\_tensor\_overhead<!-- {{#callable_declaration:ggml_tensor_overhead}} -->
Calculates the memory overhead for a tensor.
- **Description**: This function is used to determine the additional memory required for managing a tensor in the library. It is particularly useful for memory management and optimization tasks, allowing developers to estimate the total memory footprint of tensor operations. The function can be called at any time after the library has been initialized, and it does not modify any state or data.
- **Inputs**: None
- **Output**: Returns the size in bytes of the overhead associated with a tensor, which includes the size of the tensor structure and any additional metadata.
- **See also**: [`ggml_tensor_overhead`](../src/ggml.c.driver.md#ggml_tensor_overhead)  (Implementation)


---
### ggml\_validate\_row\_data<!-- {{#callable_declaration:ggml_validate_row_data}} -->
Validates the row data for a specified tensor type.
- **Description**: This function is used to ensure that the provided row data adheres to the expected format and constraints for a given tensor type. It should be called with valid parameters, specifically after determining the tensor type and the corresponding data size. The function checks for valid data types, ensures that the size of the data is a multiple of the size of the type, and performs additional validation based on the specific tensor type. If any validation fails, an error message is printed to stderr, and the function returns false. Otherwise, it returns true, indicating that the data is valid.
- **Inputs**:
    - `type`: Specifies the tensor type to validate against. Must be a valid `ggml_type` value, which should be in the range of 0 to `GGML_TYPE_COUNT - 1`. Passing an invalid type will result in a validation failure.
    - `data`: A pointer to the data to be validated. This must not be null and should point to a valid memory region containing the data for the specified tensor type.
    - `nbytes`: The size in bytes of the data pointed to by `data`. This must be a positive value and should be a multiple of the size of the specified tensor type. If it is not, validation will fail.
- **Output**: Returns true if the data is valid for the specified tensor type, otherwise returns false. If validation fails, an error message is printed to stderr.
- **See also**: [`ggml_validate_row_data`](../src/ggml-quants.c.driver.md#ggml_validate_row_data)  (Implementation)


---
### ggml\_init<!-- {{#callable_declaration:ggml_init}} -->
Initializes a new context for tensor operations.
- **Description**: This function is used to create and initialize a `ggml_context`, which is essential for managing memory and tensor operations within the GGML library. It should be called before any tensor operations are performed. The `params` structure allows the user to specify the memory size and an optional pre-allocated memory buffer. If `mem_size` is set to zero, a default aligned memory size will be used. The function ensures that the context is properly initialized and ready for use, and it will allocate memory if no buffer is provided. It is important to ensure that the provided memory size is sufficient for the intended operations, as exceeding this size may lead to undefined behavior.
- **Inputs**:
    - `params`: A `ggml_init_params` structure that contains initialization parameters. The `mem_size` field specifies the size of the memory to allocate in bytes and must be non-negative. If `mem_buffer` is provided, it should point to a valid memory area of at least `mem_size` bytes; otherwise, the function will allocate memory internally. The `no_alloc` field indicates whether to skip memory allocation for tensor data.
- **Output**: Returns a pointer to a newly initialized `ggml_context`. This pointer should be used for subsequent tensor operations. If memory allocation fails, the behavior is undefined.
- **See also**: [`ggml_init`](../src/ggml.c.driver.md#ggml_init)  (Implementation)


---
### ggml\_reset<!-- {{#callable_declaration:ggml_reset}} -->
Resets the state of a context.
- **Description**: This function is used to reset the internal state of a `ggml_context`, which is essential for reusing the context for new operations without residual data from previous computations. It should be called when you want to clear all objects associated with the context, typically before starting a new computation or after freeing resources. The function does not perform any action if the provided context pointer is `NULL`, ensuring that it can be safely called without prior checks.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that represents the context to be reset. This pointer must not be null; if it is null, the function will simply return without making any changes.
- **Output**: The function does not return any value and does not modify any inputs other than the internal state of the provided context.
- **See also**: [`ggml_reset`](../src/ggml.c.driver.md#ggml_reset)  (Implementation)


---
### ggml\_free<!-- {{#callable_declaration:ggml_free}} -->
Frees the resources associated with a `ggml_context`.
- **Description**: This function should be called to release the memory and resources allocated for a `ggml_context` when it is no longer needed. It is important to ensure that the context has been properly initialized before calling this function. If the `ctx` parameter is `NULL`, the function will simply return without performing any action. If the context owns its memory buffer, that memory will be freed as well. This function does not return any value.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that represents the context to be freed. This pointer must not be null; if it is null, the function will return immediately without any action.
- **Output**: None
- **See also**: [`ggml_free`](../src/ggml.c.driver.md#ggml_free)  (Implementation)


---
### ggml\_used\_mem<!-- {{#callable_declaration:ggml_used_mem}} -->
Returns the amount of memory used by the context.
- **Description**: This function is used to determine the total amount of memory that has been allocated for the tensors within a given context. It should be called after the context has been initialized and tensors have been created. If the context is valid and has allocated memory, the function will return the total size of memory used; otherwise, it will return zero. This can be useful for monitoring memory usage and ensuring that the allocated memory does not exceed the limits set during initialization.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that represents the current context. Must not be null. If the context is invalid or has not allocated any memory, the function will return 0.
- **Output**: Returns the total number of bytes used by the tensors in the context. If no memory has been allocated, it returns 0.
- **See also**: [`ggml_used_mem`](../src/ggml.c.driver.md#ggml_used_mem)  (Implementation)


---
### ggml\_get\_no\_alloc<!-- {{#callable_declaration:ggml_get_no_alloc}} -->
Returns the no_alloc flag from the context.
- **Description**: This function retrieves the `no_alloc` flag from the provided `ggml_context`. The `no_alloc` flag indicates whether memory allocation for tensor data is disabled. It is useful for scenarios where the user wants to manage memory allocation manually or avoid unnecessary allocations. This function should be called after the context has been initialized and is valid only if the context pointer is not null.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure. Must not be null. The context must be properly initialized before calling this function.
- **Output**: Returns a boolean value: true if memory allocation is disabled (no_alloc is set), and false otherwise.
- **See also**: [`ggml_get_no_alloc`](../src/ggml.c.driver.md#ggml_get_no_alloc)  (Implementation)


---
### ggml\_set\_no\_alloc<!-- {{#callable_declaration:ggml_set_no_alloc}} -->
Sets the allocation behavior of the context.
- **Description**: This function configures whether the context should allocate memory for tensor data. It is typically used to control memory management behavior, especially in scenarios where memory allocation should be avoided, such as during certain computations or optimizations. It is important to call this function after initializing the context and before any tensor operations that may depend on the allocation behavior.
- **Inputs**:
    - `ctx`: A pointer to a `struct ggml_context`. Must not be null. This context holds the state and configuration for tensor operations.
    - `no_alloc`: A boolean value indicating whether to disable memory allocation for tensor data. If set to true, the context will not allocate memory for tensor data.
- **Output**: None
- **See also**: [`ggml_set_no_alloc`](../src/ggml.c.driver.md#ggml_set_no_alloc)  (Implementation)


---
### ggml\_get\_mem\_buffer<!-- {{#callable_declaration:ggml_get_mem_buffer}} -->
Retrieves the memory buffer associated with the context.
- **Description**: This function is used to access the memory buffer allocated for the `ggml_context`. It should be called after the context has been initialized with `ggml_init()`. The returned pointer can be used to manage or inspect the memory used for tensor operations. If the context is not properly initialized, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure. This must not be null and should point to a valid, initialized context. If the context is not initialized, the function's behavior is undefined.
- **Output**: Returns a pointer to the memory buffer associated with the provided context. If the context was initialized with a null memory buffer, this will return a null pointer.
- **See also**: [`ggml_get_mem_buffer`](../src/ggml.c.driver.md#ggml_get_mem_buffer)  (Implementation)


---
### ggml\_get\_mem\_size<!-- {{#callable_declaration:ggml_get_mem_size}} -->
Retrieves the memory size allocated for the context.
- **Description**: This function is used to obtain the total memory size allocated for a given `ggml_context`. It should be called after the context has been initialized with `ggml_init()`. The returned value indicates the amount of memory that has been allocated for the context, which can be useful for monitoring memory usage and ensuring that the allocated memory is sufficient for the intended operations.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure. This parameter must not be null and should point to a valid context that has been initialized.
- **Output**: Returns the size of the memory allocated for the context in bytes.
- **See also**: [`ggml_get_mem_size`](../src/ggml.c.driver.md#ggml_get_mem_size)  (Implementation)


---
### ggml\_get\_max\_tensor\_size<!-- {{#callable_declaration:ggml_get_max_tensor_size}} -->
Returns the maximum size of tensors in bytes.
- **Description**: This function is used to determine the maximum size of any tensor currently managed by the specified context. It should be called after initializing the context and before performing operations that depend on tensor sizes. The function iterates through all tensors in the context and calculates their sizes, returning the largest size found. If no tensors are present, the function will return zero.
- **Inputs**:
    - `ctx`: A pointer to a `struct ggml_context` that represents the current context. This parameter must not be null, as it is required to access the tensor data.
- **Output**: Returns the maximum size of the tensors in bytes as a `size_t` value. If there are no tensors, the function returns zero.
- **See also**: [`ggml_get_max_tensor_size`](../src/ggml.c.driver.md#ggml_get_max_tensor_size)  (Implementation)


---
### ggml\_new\_tensor<!-- {{#callable_declaration:ggml_new_tensor}} -->
Creates a new tensor.
- **Description**: This function is used to create a new tensor within a specified context. It must be called after initializing the context with `ggml_init()`. The tensor's dimensions and type must be specified, and the function will allocate memory for the tensor based on the provided dimensions. If the dimensions or type are invalid, the function will return `NULL`. It is important to ensure that the total memory required for the tensor does not exceed the memory buffer allocated for the context.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `type`: An enumeration value of type `ggml_type` that specifies the data type of the tensor. Valid values include various floating-point and integer types.
    - `n_dims`: An integer representing the number of dimensions for the tensor. Must be between 1 and 4, inclusive.
    - `ne`: A pointer to an array of `int64_t` values that specify the size of each dimension of the tensor. The array must have at least `n_dims` elements and must not be null.
- **Output**: Returns a pointer to the newly created `ggml_tensor` structure, or `NULL` if the tensor could not be created due to insufficient memory or invalid parameters.
- **See also**: [`ggml_new_tensor`](../src/ggml.c.driver.md#ggml_new_tensor)  (Implementation)


---
### ggml\_new\_tensor\_1d<!-- {{#callable_declaration:ggml_new_tensor_1d}} -->
Creates a new 1D tensor.
- **Description**: This function is used to create a new 1-dimensional tensor within a specified context. It should be called after initializing the context with `ggml_init()`. The tensor will be allocated in the memory buffer associated with the context. The function expects a valid context and a specified tensor type. If the number of elements specified is less than or equal to zero, the function will return `NULL`, indicating an error in tensor creation.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must not be null and should be initialized before calling this function.
    - `type`: An enumeration value of type `ggml_type` that specifies the data type of the tensor. Valid types include various floating-point and integer representations.
    - `ne0`: An integer representing the number of elements in the tensor. It must be greater than zero; otherwise, the function will return `NULL`.
- **Output**: Returns a pointer to the newly created `ggml_tensor` structure. If the creation fails (e.g., due to invalid parameters), it returns `NULL`.
- **See also**: [`ggml_new_tensor_1d`](../src/ggml.c.driver.md#ggml_new_tensor_1d)  (Implementation)


---
### ggml\_new\_tensor\_2d<!-- {{#callable_declaration:ggml_new_tensor_2d}} -->
Creates a new 2D tensor.
- **Description**: This function is used to create a new 2D tensor within a specified context. It should be called after initializing the context with `ggml_init()`. The dimensions of the tensor are defined by the parameters `ne0` and `ne1`, which represent the size of each dimension. It is important to ensure that the total size of the tensor does not exceed the memory buffer allocated for the context. If the provided dimensions are invalid or if the context is not properly initialized, the function may return a null pointer.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `type`: An enumeration value of type `ggml_type` that specifies the data type of the tensor. Valid values include various floating-point and integer types.
    - `ne0`: An integer representing the size of the first dimension of the tensor. Must be a positive integer.
    - `ne1`: An integer representing the size of the second dimension of the tensor. Must be a positive integer.
- **Output**: Returns a pointer to the newly created `ggml_tensor` structure. If the tensor creation fails, it returns null.
- **See also**: [`ggml_new_tensor_2d`](../src/ggml.c.driver.md#ggml_new_tensor_2d)  (Implementation)


---
### ggml\_new\_tensor\_3d<!-- {{#callable_declaration:ggml_new_tensor_3d}} -->
Creates a new 3D tensor.
- **Description**: This function is used to allocate a new 3-dimensional tensor in the specified context. It should be called after initializing the context with `ggml_init()`. The dimensions of the tensor are defined by the parameters `ne0`, `ne1`, and `ne2`, which represent the size of each dimension. The function will return a pointer to the newly created tensor, or `NULL` if the allocation fails. It is important to ensure that the context has sufficient memory allocated to accommodate the new tensor.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must not be null and should be initialized before calling this function.
    - `type`: An enumeration value of type `ggml_type` that specifies the data type of the tensor elements. Valid types include various floating-point and integer representations.
    - `ne0`: An integer representing the size of the first dimension of the tensor. Must be a non-negative value.
    - `ne1`: An integer representing the size of the second dimension of the tensor. Must be a non-negative value.
    - `ne2`: An integer representing the size of the third dimension of the tensor. Must be a non-negative value.
- **Output**: Returns a pointer to the newly created `ggml_tensor` structure. If the allocation fails, it returns `NULL`.
- **See also**: [`ggml_new_tensor_3d`](../src/ggml.c.driver.md#ggml_new_tensor_3d)  (Implementation)


---
### ggml\_new\_tensor\_4d<!-- {{#callable_declaration:ggml_new_tensor_4d}} -->
Creates a new 4D tensor.
- **Description**: This function is used to allocate a new 4-dimensional tensor in the specified context. It is essential to call this function after initializing the context with `ggml_init()`. The dimensions of the tensor are defined by the parameters `ne0`, `ne1`, `ne2`, and `ne3`, which represent the size of each dimension. The function will return a pointer to the newly created tensor, or NULL if the allocation fails.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `type`: An enumeration value of type `ggml_type` that specifies the data type of the tensor elements. Valid values include types such as `GGML_TYPE_F32`, `GGML_TYPE_F16`, etc.
    - `ne0`: An integer representing the size of the first dimension of the tensor. Must be non-negative.
    - `ne1`: An integer representing the size of the second dimension of the tensor. Must be non-negative.
    - `ne2`: An integer representing the size of the third dimension of the tensor. Must be non-negative.
    - `ne3`: An integer representing the size of the fourth dimension of the tensor. Must be non-negative.
- **Output**: Returns a pointer to the newly created `ggml_tensor` structure. If the allocation fails, it returns NULL.
- **See also**: [`ggml_new_tensor_4d`](../src/ggml.c.driver.md#ggml_new_tensor_4d)  (Implementation)


---
### ggml\_new\_buffer<!-- {{#callable_declaration:ggml_new_buffer}} -->
Allocates a new buffer of specified size.
- **Description**: This function is used to allocate a new buffer within the specified context. It should be called after initializing the context with `ggml_init()`. The allocated buffer can be used for various operations within the library. If the requested size exceeds the available memory, the behavior is undefined, so it is important to ensure that the size is within the limits of the allocated memory for the context.
- **Inputs**:
    - `ctx`: A pointer to a `struct ggml_context` that represents the context in which the buffer is allocated. Must not be null.
    - `nbytes`: The size in bytes of the buffer to allocate. Must be a positive value.
- **Output**: Returns a pointer to the newly allocated buffer. The pointer is valid as long as the context remains valid.
- **See also**: [`ggml_new_buffer`](../src/ggml.c.driver.md#ggml_new_buffer)  (Implementation)


---
### ggml\_dup\_tensor<!-- {{#callable_declaration:ggml_dup_tensor}} -->
Duplicates a tensor.
- **Description**: This function is used to create a duplicate of an existing tensor, which is useful when you need to work with a copy of a tensor without modifying the original. It must be called with a valid `ggml_context` that has been initialized and a non-null source tensor. The function will allocate memory for the new tensor based on the properties of the source tensor, including its type and dimensions.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` that has been initialized. Must not be null.
    - `src`: A pointer to the source `ggml_tensor` that is to be duplicated. Must not be null.
- **Output**: Returns a pointer to the newly created tensor that is a duplicate of the source tensor. If the operation fails, it may return null.
- **See also**: [`ggml_dup_tensor`](../src/ggml.c.driver.md#ggml_dup_tensor)  (Implementation)


---
### ggml\_view\_tensor<!-- {{#callable_declaration:ggml_view_tensor}} -->
Creates a view of a source tensor.
- **Description**: This function is used to create a new tensor that acts as a view of an existing tensor, allowing for operations on the new tensor to affect the original tensor. It is important to call this function with a valid context and a source tensor that has been properly initialized. The resulting view tensor will share the same data as the source tensor, meaning that modifications to the view will reflect in the source tensor. If the source tensor is null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations. Must not be null.
    - `src`: A pointer to the source `ggml_tensor` that will be viewed. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the view of the source tensor. This new tensor shares the same data as the source tensor.
- **See also**: [`ggml_view_tensor`](../src/ggml.c.driver.md#ggml_view_tensor)  (Implementation)


---
### ggml\_get\_first\_tensor<!-- {{#callable_declaration:ggml_get_first_tensor}} -->
Retrieves the first tensor from the context.
- **Description**: This function is used to obtain the first tensor object from the specified context. It should be called after the context has been initialized and populated with tensor objects. If there are no tensors present in the context, the function will return `NULL`. It is important to ensure that the context is valid and properly initialized before calling this function to avoid undefined behavior.
- **Inputs**:
    - `ctx`: A pointer to a `struct ggml_context` that represents the context from which to retrieve the tensor. This pointer must not be null and should point to a valid context that has been initialized and populated with tensor objects.
- **Output**: Returns a pointer to the first `struct ggml_tensor` found in the context. If no tensors are present, it returns `NULL`.
- **See also**: [`ggml_get_first_tensor`](../src/ggml.c.driver.md#ggml_get_first_tensor)  (Implementation)


---
### ggml\_get\_next\_tensor<!-- {{#callable_declaration:ggml_get_next_tensor}} -->
Retrieves the next tensor in the context.
- **Description**: This function is used to iterate through tensors that are stored in a given context. It should be called after obtaining a tensor from the context, and it will return the next tensor in the sequence. If there are no more tensors available, the function will return `NULL`. It is important to ensure that the `ctx` parameter is valid and that the `tensor` parameter is not `NULL`.
- **Inputs**:
    - `ctx`: A pointer to a `struct ggml_context` that represents the context from which tensors are retrieved. Must not be null.
    - `tensor`: A pointer to a `struct ggml_tensor` that represents the current tensor. Must not be null. The function will return the next tensor in the sequence relative to this tensor.
- **Output**: Returns a pointer to the next `struct ggml_tensor` in the context, or `NULL` if there are no more tensors available.
- **See also**: [`ggml_get_next_tensor`](../src/ggml.c.driver.md#ggml_get_next_tensor)  (Implementation)


---
### ggml\_get\_tensor<!-- {{#callable_declaration:ggml_get_tensor}} -->
Retrieves a tensor by its name from the context.
- **Description**: This function is used to obtain a pointer to a `ggml_tensor` identified by its name within a specified `ggml_context`. It is essential to call this function after initializing the context and populating it with tensors. If the specified name does not correspond to any tensor in the context, the function will return `NULL`. This function does not modify the context or the tensor.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that holds the tensors. Must not be null.
    - `name`: A string representing the name of the tensor to retrieve. Must not be null.
- **Output**: Returns a pointer to the `ggml_tensor` with the specified name if found; otherwise, returns `NULL`.
- **See also**: [`ggml_get_tensor`](../src/ggml.c.driver.md#ggml_get_tensor)  (Implementation)


---
### ggml\_unravel\_index<!-- {{#callable_declaration:ggml_unravel_index}} -->
Converts a flat index into multi-dimensional coordinates.
- **Description**: This function is used to convert a flat index into its corresponding multi-dimensional indices based on the shape of the specified tensor. It is particularly useful when working with multi-dimensional tensors, allowing users to easily access specific elements using a single index. The function should be called with a valid `ggml_tensor` that has been properly initialized. If the provided index exceeds the total number of elements in the tensor, the behavior is undefined, and the output parameters may not be modified.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the multi-dimensional tensor. Must not be null.
    - `i`: The flat index to be converted. It should be a non-negative integer less than the total number of elements in the tensor.
    - `i0`: A pointer to an int64_t where the first dimension index will be stored. Can be null if the first index is not needed.
    - `i1`: A pointer to an int64_t where the second dimension index will be stored. Can be null if the second index is not needed.
    - `i2`: A pointer to an int64_t where the third dimension index will be stored. Can be null if the third index is not needed.
    - `i3`: A pointer to an int64_t where the fourth dimension index will be stored. Can be null if the fourth index is not needed.
- **Output**: The function does not return a value. Instead, it populates the provided pointers with the corresponding multi-dimensional indices based on the input flat index.
- **See also**: [`ggml_unravel_index`](../src/ggml.c.driver.md#ggml_unravel_index)  (Implementation)


---
### ggml\_get\_unary\_op<!-- {{#callable_declaration:ggml_get_unary_op}} -->
Retrieves the unary operation type of a tensor.
- **Description**: This function is used to obtain the type of unary operation associated with a given tensor. It should be called only when the tensor's operation type is `GGML_OP_UNARY`. If the tensor does not represent a unary operation, the behavior is undefined.
- **Inputs**:
    - `tensor`: A pointer to a `struct ggml_tensor` that represents the tensor from which the unary operation type is to be retrieved. This pointer must not be null and must point to a valid tensor structure that has been initialized properly.
- **Output**: Returns an enumeration value of type `enum ggml_unary_op` that indicates the specific unary operation associated with the tensor. If the tensor does not represent a unary operation, the return value is not defined.
- **See also**: [`ggml_get_unary_op`](../src/ggml.c.driver.md#ggml_get_unary_op)  (Implementation)


---
### ggml\_get\_data<!-- {{#callable_declaration:ggml_get_data}} -->
Retrieves the data pointer from a tensor.
- **Description**: This function is used to access the raw data of a tensor represented by the `ggml_tensor` structure. It should be called after the tensor has been properly initialized and allocated. The returned pointer allows users to manipulate the tensor's data directly, but care must be taken to ensure that the tensor is not null and that the data is accessed in accordance with the tensor's dimensions and type.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure. Must not be null. The tensor must be properly initialized and allocated before calling this function.
- **Output**: Returns a pointer to the data of the specified tensor. If the tensor is null, the behavior is undefined.
- **See also**: [`ggml_get_data`](../src/ggml.c.driver.md#ggml_get_data)  (Implementation)


---
### ggml\_get\_data\_f32<!-- {{#callable_declaration:ggml_get_data_f32}} -->
Retrieves a pointer to the data of a tensor as a float array.
- **Description**: This function is used to access the underlying data of a tensor that is specifically of type `GGML_TYPE_F32`. It should be called only after ensuring that the tensor has been properly initialized and is of the correct type. If the tensor is not of type `GGML_TYPE_F32`, the behavior is undefined, and the function may lead to incorrect results or crashes. It is important to manage the lifetime of the tensor appropriately, as the returned pointer points to the internal data of the tensor.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure that represents the tensor from which data is to be retrieved. This pointer must not be null and must point to a tensor of type `GGML_TYPE_F32`.
- **Output**: Returns a pointer to the data of the tensor as a float array. The caller should not attempt to free this pointer, as it is managed by the tensor structure.
- **See also**: [`ggml_get_data_f32`](../src/ggml.c.driver.md#ggml_get_data_f32)  (Implementation)


---
### ggml\_get\_name<!-- {{#callable_declaration:ggml_get_name}} -->
Retrieves the name of a tensor.
- **Description**: This function is used to obtain the name of a tensor, which can be useful for debugging or logging purposes. It should be called with a valid `ggml_tensor` pointer that has been properly initialized. If the provided tensor pointer is null, the behavior is undefined, and the function may return a null pointer.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure. This parameter must not be null and should point to a valid tensor that has been initialized. If the tensor is not properly initialized, the function may return a null pointer.
- **Output**: Returns a pointer to a string containing the name of the tensor. If the tensor does not have a name set, this may return a null pointer.
- **See also**: [`ggml_get_name`](../src/ggml.c.driver.md#ggml_get_name)  (Implementation)


---
### ggml\_set\_name<!-- {{#callable_declaration:ggml_set_name}} -->
Sets the name of a tensor.
- **Description**: This function assigns a name to a specified tensor, which can be useful for identification and debugging purposes. It should be called after the tensor has been created and initialized. The name is stored in the tensor's internal structure, and it is limited to a maximum length defined by `GGML_MAX_NAME`. If the provided name exceeds this length, it will be truncated. The function returns a pointer to the modified tensor.
- **Inputs**:
    - `tensor`: A pointer to the `ggml_tensor` structure that will have its name set. Must not be null.
    - `name`: A pointer to a null-terminated string containing the name to be assigned to the tensor. Must not be null.
- **Output**: Returns a pointer to the modified `ggml_tensor` with the new name set.
- **See also**: [`ggml_set_name`](../src/ggml.c.driver.md#ggml_set_name)  (Implementation)


---
### ggml\_format\_name<!-- {{#callable_declaration:ggml_format_name}} -->
Formats the name of a tensor using a specified format.
- **Description**: This function is used to set the name of a `ggml_tensor` by formatting it according to the provided format string and additional arguments. It should be called after the tensor has been created and initialized. The formatted name will replace any existing name in the tensor. Ensure that the format string is valid and that the tensor is not null. If the formatted name exceeds the maximum allowed length, it will be truncated.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure that will have its name formatted. Must not be null.
    - `fmt`: A format string that specifies how to format the name. Must not be null.
    - `args`: Additional arguments for formatting the name according to the format string.
- **Output**: Returns a pointer to the modified `ggml_tensor` with the updated name.
- **See also**: [`ggml_format_name`](../src/ggml.c.driver.md#ggml_format_name)  (Implementation)


---
### ggml\_set\_input<!-- {{#callable_declaration:ggml_set_input}} -->
Marks a tensor as an input for the computation graph.
- **Description**: This function should be called to designate a `ggml_tensor` as an input variable within the computation graph. It is essential to invoke this function before performing any computations that depend on the input tensor. The function modifies the tensor's flags to indicate its role in the graph, which is crucial for operations such as automatic differentiation and optimization. Ensure that the `tensor` parameter is valid and properly initialized before calling this function.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure that represents the tensor to be marked as an input. This pointer must not be null and should point to a valid tensor that has been initialized.
- **Output**: None
- **See also**: [`ggml_set_input`](../src/ggml.c.driver.md#ggml_set_input)  (Implementation)


---
### ggml\_set\_output<!-- {{#callable_declaration:ggml_set_output}} -->
Marks a tensor as an output for the computation graph.
- **Description**: This function should be called to designate a `ggml_tensor` as an output tensor within the computation graph. It is typically used after the tensor has been created and before the graph computation is executed. The function modifies the tensor's flags to indicate its role in the graph, which is essential for proper management of the tensor during forward and backward passes in machine learning tasks. Ensure that the tensor is valid and properly initialized before calling this function.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure that represents the tensor to be marked as an output. This pointer must not be null and should point to a valid tensor that has been created using the appropriate tensor creation functions.
- **Output**: None
- **See also**: [`ggml_set_output`](../src/ggml.c.driver.md#ggml_set_output)  (Implementation)


---
### ggml\_set\_param<!-- {{#callable_declaration:ggml_set_param}} -->
Marks a tensor as a parameter.
- **Description**: This function is used to designate a `ggml_tensor` as a parameter tensor, which is essential for automatic differentiation and optimization processes. It should be called only when the tensor's operation type is `GGML_OP_NONE`, ensuring that the tensor is not already part of a computation graph. Attempting to set a parameter on a tensor that is already associated with an operation will result in an assertion failure.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure that represents the tensor to be marked as a parameter. This pointer must not be null and the tensor's operation type must be `GGML_OP_NONE` before calling this function.
- **Output**: None
- **See also**: [`ggml_set_param`](../src/ggml.c.driver.md#ggml_set_param)  (Implementation)


---
### ggml\_set\_loss<!-- {{#callable_declaration:ggml_set_loss}} -->
Marks a tensor as a loss tensor.
- **Description**: This function is used to designate a specific tensor as a loss tensor, which is essential for numerical optimization tasks. It should be called after the tensor has been created and must be a scalar tensor of type `GGML_TYPE_F32`. Attempting to set loss on a tensor that does not meet these criteria will result in an assertion failure.
- **Inputs**:
    - `tensor`: A pointer to a `struct ggml_tensor` that represents the tensor to be marked as a loss tensor. This tensor must be a scalar and of type `GGML_TYPE_F32`. The function does not take ownership of the tensor, and it must not be null.
- **Output**: None
- **See also**: [`ggml_set_loss`](../src/ggml.c.driver.md#ggml_set_loss)  (Implementation)


---
### ggml\_dup<!-- {{#callable_declaration:ggml_dup}} -->
Duplicates a tensor.
- **Description**: This function is used to create a duplicate of an existing tensor. It is essential to call this function when you need a separate copy of a tensor to avoid unintended modifications to the original tensor. The function requires a valid context and a tensor to duplicate. If the provided tensor is null, the function will handle it gracefully by returning null.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` to be duplicated. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that is a duplicate of the input tensor. If the input tensor is null, the function returns null.
- **See also**: [`ggml_dup`](../src/ggml.c.driver.md#ggml_dup)  (Implementation)


---
### ggml\_dup\_inplace<!-- {{#callable_declaration:ggml_dup_inplace}} -->
Duplicates a tensor in place.
- **Description**: This function is used to create a duplicate of an existing tensor within the same memory context. It is particularly useful when you want to maintain the original tensor while working with its duplicate. The function must be called with a valid `ggml_context` that has been initialized and a non-null `ggml_tensor` to duplicate. If the provided tensor is invalid or null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that must be initialized before calling this function. It must not be null.
    - `a`: A pointer to the `ggml_tensor` to be duplicated. This tensor must not be null and should be a valid tensor within the context.
- **Output**: Returns a pointer to the duplicated `ggml_tensor`. If the operation fails, the return value may be null.
- **See also**: [`ggml_dup_inplace`](../src/ggml.c.driver.md#ggml_dup_inplace)  (Implementation)


---
### ggml\_add<!-- {{#callable_declaration:ggml_add}} -->
Adds two tensors together.
- **Description**: This function is used to perform element-wise addition of two tensors, `a` and `b`, within a specified context. It is essential to ensure that both tensors are compatible in terms of their dimensions and data types before calling this function. The resulting tensor will be a new tensor that contains the sum of the corresponding elements from `a` and `b`. If the input tensors are not compatible, the behavior is undefined, and the function may return a null pointer.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the first `ggml_tensor` to be added. Must not be null and must have compatible dimensions with tensor `b`.
    - `b`: A pointer to the second `ggml_tensor` to be added. Must not be null and must have compatible dimensions with tensor `a`.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the addition. If the operation fails due to incompatible tensor dimensions, the return value may be null.
- **See also**: [`ggml_add`](../src/ggml.c.driver.md#ggml_add)  (Implementation)


---
### ggml\_add\_inplace<!-- {{#callable_declaration:ggml_add_inplace}} -->
Adds two tensors in place.
- **Description**: This function performs an in-place addition of two tensors, modifying the first tensor to hold the result of the addition. It is essential to ensure that both tensors are compatible in terms of their dimensions and data types before calling this function. The function should be invoked after initializing the context and creating the tensors. If the tensors are not compatible, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before use. It must not be null.
    - `a`: A pointer to the first `ggml_tensor` that will be modified to store the result. This tensor must be allocated and initialized, and it must not be null.
    - `b`: A pointer to the second `ggml_tensor` that will be added to the first tensor. This tensor must also be allocated and initialized, and it must not be null.
- **Output**: Returns a pointer to the modified `ggml_tensor` that contains the result of the addition. If the operation is successful, this will be the same as the pointer to `a`.
- **See also**: [`ggml_add_inplace`](../src/ggml.c.driver.md#ggml_add_inplace)  (Implementation)


---
### ggml\_add\_cast<!-- {{#callable_declaration:ggml_add_cast}} -->
Adds two tensors and casts the result to a specified type.
- **Description**: This function is used to perform element-wise addition of two tensors, `a` and `b`, while also allowing the result to be cast to a specified tensor type. It is essential to ensure that both input tensors are compatible in terms of their dimensions and types before calling this function. The resulting tensor will have the same dimensions as the input tensors, and its type will be determined by the `type` parameter. If the input tensors are not compatible, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the first `ggml_tensor` to be added. Must not be null.
    - `b`: A pointer to the second `ggml_tensor` to be added. Must not be null.
    - `type`: An enumeration value of type `ggml_type` that specifies the desired type of the resulting tensor. This should be a valid type defined in the `ggml_type` enum.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the addition, cast to the specified type. If the operation fails due to incompatible tensor types or dimensions, the behavior is undefined.
- **See also**: [`ggml_add_cast`](../src/ggml.c.driver.md#ggml_add_cast)  (Implementation)


---
### ggml\_add1<!-- {{#callable_declaration:ggml_add1}} -->
Adds two tensors element-wise.
- **Description**: This function performs an element-wise addition of two tensors, `a` and `b`, and returns a new tensor containing the result. It is important to ensure that both input tensors are compatible in terms of their dimensions; otherwise, the behavior is undefined. This function should be called after initializing the `ggml_context` and creating the tensors. The resulting tensor is a new allocation, and the caller is responsible for managing its memory.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the first tensor to be added. Must not be null and must be a valid tensor.
    - `b`: A pointer to the second tensor to be added. Must not be null and must be a valid tensor.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the addition. The caller is responsible for freeing this tensor when it is no longer needed.
- **See also**: [`ggml_add1`](../src/ggml.c.driver.md#ggml_add1)  (Implementation)


---
### ggml\_add1\_inplace<!-- {{#callable_declaration:ggml_add1_inplace}} -->
Performs an in-place addition of two tensors.
- **Description**: This function is used to add the values of two tensors together, modifying the first tensor in place. It is important to ensure that both tensors are compatible in terms of their dimensions and data types before calling this function. The operation will fail if the tensors do not have the same shape, and the function will return a null pointer in such cases. This function should be called when you want to update the first tensor with the result of the addition without allocating additional memory for a new tensor.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the first `ggml_tensor` which will be modified in place. Must not be null and must have a compatible shape with tensor `b`.
    - `b`: A pointer to the second `ggml_tensor` which will be added to tensor `a`. Must not be null and must have the same shape as tensor `a`.
- **Output**: Returns a pointer to the modified `ggml_tensor` `a`, which now contains the result of the addition. If the operation fails due to incompatible tensor shapes, it returns null.
- **See also**: [`ggml_add1_inplace`](../src/ggml.c.driver.md#ggml_add1_inplace)  (Implementation)


---
### ggml\_acc<!-- {{#callable_declaration:ggml_acc}} -->
Accumulates values from one tensor into another.
- **Description**: This function is used to accumulate values from tensor `b` into tensor `a` at a specified offset and with given strides for the first three dimensions. It is essential to ensure that the tensors are properly initialized and that the specified dimensions and offset are valid to avoid undefined behavior. The function modifies the destination tensor in place, and it is expected to be called when the context is active.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be valid and initialized before calling this function. It manages the memory and state for tensor operations.
    - `a`: A pointer to the destination `ggml_tensor` where the values will be accumulated. This tensor must be properly initialized and have sufficient space to accommodate the accumulated values.
    - `b`: A pointer to the source `ggml_tensor` from which values will be taken. This tensor must also be properly initialized and should not be null.
    - `nb1`: A size_t value representing the stride in bytes for the first dimension of tensor `a`. It must be greater than zero.
    - `nb2`: A size_t value representing the stride in bytes for the second dimension of tensor `a`. It must be greater than zero.
    - `nb3`: A size_t value representing the stride in bytes for the third dimension of tensor `a`. It must be greater than zero.
    - `offset`: A size_t value representing the byte offset in tensor `a` where the accumulation will start. It must be within the bounds of the tensor.
- **Output**: Returns a pointer to the modified `ggml_tensor` `a`, which now contains the accumulated values from tensor `b`.
- **See also**: [`ggml_acc`](../src/ggml.c.driver.md#ggml_acc)  (Implementation)


---
### ggml\_acc\_inplace<!-- {{#callable_declaration:ggml_acc_inplace}} -->
Accumulates values from one tensor into another in place.
- **Description**: This function is used to perform an in-place accumulation of values from tensor `b` into tensor `a`, based on specified dimensions and an offset. It is essential to ensure that both tensors are properly initialized and compatible in terms of their dimensions before calling this function. The function modifies the contents of tensor `a` directly, which means that the original data in `a` will be altered. If the dimensions specified by `nb1`, `nb2`, and `nb3` do not align correctly with the shapes of the tensors, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the destination tensor where values will be accumulated. This tensor must be properly initialized and allocated. Must not be null.
    - `b`: A pointer to the source tensor from which values will be taken for accumulation. This tensor must also be properly initialized and allocated. Must not be null.
    - `nb1`: The first dimension size for the accumulation operation. Must be a non-negative value.
    - `nb2`: The second dimension size for the accumulation operation. Must be a non-negative value.
    - `nb3`: The third dimension size for the accumulation operation. Must be a non-negative value.
    - `offset`: The byte offset into tensor `a` where the accumulation will start. Must be a non-negative value.
- **Output**: Returns a pointer to the modified tensor `a`, which now contains the accumulated values from tensor `b`.
- **See also**: [`ggml_acc_inplace`](../src/ggml.c.driver.md#ggml_acc_inplace)  (Implementation)


---
### ggml\_sub<!-- {{#callable_declaration:ggml_sub}} -->
Subtracts one tensor from another.
- **Description**: This function is used to perform element-wise subtraction between two tensors. It should be called after initializing the context and creating the tensors to be subtracted. The resulting tensor will have the same shape as the input tensors, and it is important to ensure that both input tensors are compatible in terms of dimensions. If the input tensors are not compatible, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the first `ggml_tensor` to be subtracted. Must not be null.
    - `b`: A pointer to the second `ggml_tensor` to be subtracted. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the subtraction. The caller is responsible for managing the memory of the resulting tensor.
- **See also**: [`ggml_sub`](../src/ggml.c.driver.md#ggml_sub)  (Implementation)


---
### ggml\_sub\_inplace<!-- {{#callable_declaration:ggml_sub_inplace}} -->
Performs in-place subtraction of two tensors.
- **Description**: This function is used to subtract one tensor from another directly in the memory of the first tensor, modifying it in place. It is important to ensure that both input tensors are compatible in terms of their dimensions and data types. The function should be called after initializing the tensors and allocating the necessary memory. If the tensors are not compatible, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the first `ggml_tensor`, which will be modified to hold the result of the subtraction. Must not be null and must be compatible with tensor `b`.
    - `b`: A pointer to the second `ggml_tensor`, which will be subtracted from tensor `a`. Must not be null and must be compatible with tensor `a`.
- **Output**: Returns a pointer to the modified tensor `a`, which now contains the result of the subtraction.
- **See also**: [`ggml_sub_inplace`](../src/ggml.c.driver.md#ggml_sub_inplace)  (Implementation)


---
### ggml\_mul<!-- {{#callable_declaration:ggml_mul}} -->
Multiplies two tensors.
- **Description**: This function is used to perform element-wise multiplication of two tensors, `a` and `b`, within a specified context, `ctx`. It is essential to ensure that both tensors have compatible shapes for multiplication; otherwise, the behavior is undefined. The function should be called after initializing the context and creating the tensors. The resulting tensor is a new tensor that holds the product of the two input tensors.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the first `ggml_tensor` to be multiplied. Must not be null and should have a compatible shape with tensor `b`.
    - `b`: A pointer to the second `ggml_tensor` to be multiplied. Must not be null and should have a compatible shape with tensor `a`.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the multiplication. The caller is responsible for managing the memory of the resulting tensor.
- **See also**: [`ggml_mul`](../src/ggml.c.driver.md#ggml_mul)  (Implementation)


---
### ggml\_mul\_inplace<!-- {{#callable_declaration:ggml_mul_inplace}} -->
Performs in-place multiplication of two tensors.
- **Description**: This function is used to multiply two tensors element-wise and store the result in the first tensor. It should be called when both tensors are already allocated and have compatible shapes for multiplication. The operation modifies the first tensor directly, and the second tensor remains unchanged. Ensure that the tensors are of the same type and dimensions, as the function does not handle type conversion or shape mismatches.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the first `ggml_tensor` that will be modified to hold the result of the multiplication. Must not be null and must have a compatible shape with tensor `b`.
    - `b`: A pointer to the second `ggml_tensor` that will be multiplied with tensor `a`. Must not be null and must have a compatible shape with tensor `a`.
- **Output**: Returns a pointer to the modified `ggml_tensor` `a`, which now contains the result of the multiplication.
- **See also**: [`ggml_mul_inplace`](../src/ggml.c.driver.md#ggml_mul_inplace)  (Implementation)


---
### ggml\_div<!-- {{#callable_declaration:ggml_div}} -->
Divides two tensors element-wise.
- **Description**: This function performs element-wise division of two tensors, `a` and `b`, and returns a new tensor containing the result. It is essential to ensure that both input tensors are compatible in terms of their dimensions; otherwise, the function may not behave as expected. The function should be called after initializing the `ggml_context` and creating the tensors. If either tensor is null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the first `ggml_tensor` to be divided. Must not be null and must have compatible dimensions with tensor `b`.
    - `b`: A pointer to the second `ggml_tensor` to be used as the divisor. Must not be null and must have compatible dimensions with tensor `a`.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the element-wise division of `a` by `b`. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_div`](../src/ggml.c.driver.md#ggml_div)  (Implementation)


---
### ggml\_div\_inplace<!-- {{#callable_declaration:ggml_div_inplace}} -->
Performs in-place division of two tensors.
- **Description**: This function is used to divide one tensor by another tensor in-place, modifying the first tensor directly. It is important to ensure that both tensors are compatible in terms of their dimensions and data types before calling this function. The operation will fail if the tensors do not have the same shape, and the function will return a null pointer in such cases. This function should be called after initializing the context and creating the tensors.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. It must not be null.
    - `a`: A pointer to the first `ggml_tensor`, which will be modified in-place. This tensor must not be null and should have compatible dimensions with tensor `b`.
    - `b`: A pointer to the second `ggml_tensor`, which will be used as the divisor. This tensor must not be null and should have the same dimensions as tensor `a`.
- **Output**: Returns a pointer to the modified tensor `a`, or null if the operation fails due to incompatible tensor shapes.
- **See also**: [`ggml_div_inplace`](../src/ggml.c.driver.md#ggml_div_inplace)  (Implementation)


---
### ggml\_sqr<!-- {{#callable_declaration:ggml_sqr}} -->
Computes the element-wise square of a tensor.
- **Description**: This function is used to compute the square of each element in the provided tensor. It should be called after initializing the context and creating the tensor. The input tensor must not be null, and it should be a valid tensor created within the same context. If the input tensor is invalid or null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. It must not be null.
    - `a`: A pointer to the `ggml_tensor` structure representing the input tensor. This tensor must be valid and created within the same context. It must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the squared values of the input tensor. The returned tensor is a new allocation and the caller is responsible for managing its memory.
- **See also**: [`ggml_sqr`](../src/ggml.c.driver.md#ggml_sqr)  (Implementation)


---
### ggml\_sqr\_inplace<!-- {{#callable_declaration:ggml_sqr_inplace}} -->
Performs an in-place square operation on a tensor.
- **Description**: This function modifies the input tensor by squaring its elements in place. It should be called when the tensor is already allocated and initialized. The operation is performed directly on the input tensor, meaning that the original values will be overwritten. Ensure that the tensor is not null before calling this function, as passing a null pointer will lead to undefined behavior.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` structure that will be modified in place. Must not be null and should be a valid tensor.
- **Output**: Returns a pointer to the modified `ggml_tensor`, which is the same as the input tensor.
- **See also**: [`ggml_sqr_inplace`](../src/ggml.c.driver.md#ggml_sqr_inplace)  (Implementation)


---
### ggml\_sqrt<!-- {{#callable_declaration:ggml_sqrt}} -->
Computes the square root of a tensor.
- **Description**: This function is used to compute the element-wise square root of the input tensor. It should be called after initializing the context and creating the input tensor. The input tensor must not be null, and it should contain non-negative values, as the square root of negative values is undefined. The function returns a new tensor that contains the square root of each element from the input tensor.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. It must not be null.
    - `a`: A pointer to the input `ggml_tensor` structure. This tensor must not be null and should contain non-negative values.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the square root of each element from the input tensor. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_sqrt`](../src/ggml.c.driver.md#ggml_sqrt)  (Implementation)


---
### ggml\_sqrt\_inplace<!-- {{#callable_declaration:ggml_sqrt_inplace}} -->
Computes the square root of a tensor in place.
- **Description**: This function modifies the input tensor by computing the square root of each element in the tensor. It is important to ensure that the input tensor is valid and properly initialized before calling this function. The operation is performed in place, meaning that the original tensor is updated with the new values, and no new tensor is created. This function should be used when you want to apply the square root operation directly to an existing tensor without allocating additional memory.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the tensor to be modified. Must not be null and must be a valid tensor.
- **Output**: Returns a pointer to the modified `ggml_tensor`, which is the same as the input tensor after the square root operation has been applied.
- **See also**: [`ggml_sqrt_inplace`](../src/ggml.c.driver.md#ggml_sqrt_inplace)  (Implementation)


---
### ggml\_log<!-- {{#callable_declaration:ggml_log}} -->
Computes the logarithm of each element in a tensor.
- **Description**: This function is used to apply the logarithm operation to each element of the input tensor. It should be called after initializing the context and creating the input tensor. The input tensor must not be null, and it should contain valid numerical values. If the input tensor contains non-positive values, the behavior is undefined, and the function may return a null pointer.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the state of the library. This must not be null and should be initialized before calling this function.
    - `a`: A pointer to the `ggml_tensor` structure representing the input tensor. This tensor must not be null and should contain valid numerical values. The function will compute the logarithm of each element in this tensor.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the logarithm of each element from the input tensor. If the input tensor is invalid or if an error occurs, the function may return null.
- **See also**: [`ggml_log`](../src/ggml.c.driver.md#ggml_log)  (Implementation)


---
### ggml\_log\_inplace<!-- {{#callable_declaration:ggml_log_inplace}} -->
Applies the logarithm function to a tensor in place.
- **Description**: This function modifies the input tensor by applying the logarithm operation to each element. It is intended for use when the tensor has already been initialized and contains valid data. The function should be called when the tensor is ready for transformation, and it will directly alter the contents of the input tensor without creating a new tensor. Ensure that the input tensor contains positive values, as the logarithm of non-positive values is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the state of the library. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure that contains the data to be transformed. Must not be null and should contain positive values.
- **Output**: Returns a pointer to the modified `ggml_tensor`, which now contains the logarithm of the original values.
- **See also**: [`ggml_log_inplace`](../src/ggml.c.driver.md#ggml_log_inplace)  (Implementation)


---
### ggml\_sin<!-- {{#callable_declaration:ggml_sin}} -->
Computes the sine of each element in a tensor.
- **Description**: This function is used to apply the sine operation element-wise on a given tensor. It should be called after initializing the context and creating the tensor. The input tensor must not be null, and it should be of a compatible type. If the input tensor is invalid or null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the state of the library. This must not be null and should be initialized before calling this function.
    - `a`: A pointer to the `ggml_tensor` structure representing the input tensor. This tensor must not be null and should be of a compatible type for the sine operation.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the sine of each element from the input tensor. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_sin`](../src/ggml.c.driver.md#ggml_sin)  (Implementation)


---
### ggml\_sin\_inplace<!-- {{#callable_declaration:ggml_sin_inplace}} -->
Applies the sine function to a tensor in place.
- **Description**: This function modifies the input tensor by applying the sine function to each element. It should be called when you want to compute the sine of the tensor's values without creating a new tensor. The input tensor must be valid and properly initialized; otherwise, the behavior is undefined. This function is particularly useful in scenarios where memory efficiency is a concern, as it avoids the overhead of creating a new tensor for the result.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. This must not be null and should be properly initialized before calling the function.
    - `a`: A pointer to a `ggml_tensor` structure representing the input tensor. This tensor must be valid and initialized. The function will modify this tensor in place, and it must not be null.
- **Output**: Returns a pointer to the modified `ggml_tensor`, which is the same as the input tensor `a`. This allows for method chaining if desired.
- **See also**: [`ggml_sin_inplace`](../src/ggml.c.driver.md#ggml_sin_inplace)  (Implementation)


---
### ggml\_cos<!-- {{#callable_declaration:ggml_cos}} -->
Computes the cosine of each element in a tensor.
- **Description**: This function is used to apply the cosine operation element-wise on a given tensor. It should be called after initializing the context and creating the tensor. The input tensor must not be null, and it should be of a valid type supported by the library. If the input tensor is invalid or null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. It manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` structure representing the input tensor. This tensor must be valid and properly initialized. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the cosine of each element from the input tensor. The output tensor will have the same shape as the input tensor.
- **See also**: [`ggml_cos`](../src/ggml.c.driver.md#ggml_cos)  (Implementation)


---
### ggml\_cos\_inplace<!-- {{#callable_declaration:ggml_cos_inplace}} -->
Applies the cosine function to a tensor in place.
- **Description**: This function modifies the input tensor by applying the cosine operation to each element. It should be called when the tensor is already allocated and initialized. The input tensor must not be null, and it is expected to be of a compatible type for the cosine operation. The function operates directly on the input tensor, meaning that the original data will be replaced with the cosine values.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the state of the library. This must not be null and should be initialized before calling this function.
    - `a`: A pointer to the `ggml_tensor` structure that contains the data to be modified. This tensor must be properly initialized and allocated. It must not be null.
- **Output**: Returns a pointer to the modified `ggml_tensor`, which now contains the cosine of the original values.
- **See also**: [`ggml_cos_inplace`](../src/ggml.c.driver.md#ggml_cos_inplace)  (Implementation)


---
### ggml\_sum<!-- {{#callable_declaration:ggml_sum}} -->
Computes the sum of all elements in a tensor.
- **Description**: This function is used to calculate the sum of all elements in the provided tensor. It should be called after initializing the `ggml_context` and creating the tensor. The input tensor must not be null, and it should be a valid tensor created using the library's tensor creation functions. The result is a new tensor that contains the sum as a scalar value.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. It must not be null.
    - `a`: A pointer to the `ggml_tensor` structure representing the input tensor. This tensor must be valid and created using the library's tensor creation functions. It must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the sum of all elements from the input tensor. The resulting tensor will have a single element.
- **See also**: [`ggml_sum`](../src/ggml.c.driver.md#ggml_sum)  (Implementation)


---
### ggml\_sum\_rows<!-- {{#callable_declaration:ggml_sum_rows}} -->
Sums the elements of a tensor along its rows.
- **Description**: This function is used to compute the sum of elements across the rows of a given tensor. It is particularly useful when you need to aggregate data along a specific dimension, such as in machine learning tasks where row-wise operations are common. The input tensor must be properly initialized and allocated in memory before calling this function. The resulting tensor will have its first dimension reduced to 1, while the other dimensions remain unchanged. If the input tensor is not valid or improperly initialized, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `struct ggml_context` that manages the memory and state for tensor operations. This must not be null and should be properly initialized before use.
    - `a`: A pointer to a `struct ggml_tensor` representing the input tensor. This tensor must be properly initialized and allocated in memory. It should have at least two dimensions, as the function sums along the rows. If `a` is null or does not meet the dimensional requirements, the behavior is undefined.
- **Output**: Returns a pointer to a new `struct ggml_tensor` that contains the sum of the input tensor's rows. The resulting tensor will have a shape where the first dimension is reduced to 1, and the other dimensions are the same as the input tensor's dimensions, except for the first one.
- **See also**: [`ggml_sum_rows`](../src/ggml.c.driver.md#ggml_sum_rows)  (Implementation)


---
### ggml\_mean<!-- {{#callable_declaration:ggml_mean}} -->
Calculates the mean of a tensor along its first dimension.
- **Description**: This function is used to compute the mean of the input tensor `a` along its first dimension, resulting in a new tensor that represents the average values. It is important to ensure that the input tensor is properly initialized and allocated before calling this function. The resulting tensor will have its first dimension reduced to 1, while the other dimensions remain unchanged. If the input tensor is null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the input tensor of type `ggml_tensor`. This tensor must be properly initialized and allocated. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the mean values computed from the input tensor. The first dimension of the output tensor will be 1, while the other dimensions will match those of the input tensor.
- **See also**: [`ggml_mean`](../src/ggml.c.driver.md#ggml_mean)  (Implementation)


---
### ggml\_argmax<!-- {{#callable_declaration:ggml_argmax}} -->
Returns the indices of the maximum values along the rows of a matrix.
- **Description**: This function is used to compute the indices of the maximum values in each row of a given matrix tensor. It must be called with a valid `ggml_context` and a tensor `a` that is confirmed to be a matrix. The resulting tensor will have a shape that corresponds to the number of rows in the input tensor, with each element representing the index of the maximum value in that row. If the input tensor is not a matrix, the function will assert and terminate.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` that represents a matrix. The tensor must have at least two dimensions, and the first dimension must be less than or equal to INT32_MAX. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` containing the indices of the maximum values for each row of the input tensor. The output tensor is a 1D tensor with a size equal to the number of rows in the input tensor.
- **See also**: [`ggml_argmax`](../src/ggml.c.driver.md#ggml_argmax)  (Implementation)


---
### ggml\_count\_equal<!-- {{#callable_declaration:ggml_count_equal}} -->
Counts the number of equal elements between two tensors.
- **Description**: This function is used to compare two tensors for equality element-wise. It must be called with two tensors that have the same shape, as determined by the `ggml_are_same_shape` function. If the tensors are not of the same shape, the behavior is undefined. The result is a new tensor that contains a single element representing the count of equal elements found in the two input tensors. This function is typically used in scenarios where one needs to evaluate the similarity between two datasets or tensors.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the first `ggml_tensor` to compare. Must not be null and must have the same shape as tensor `b`.
    - `b`: A pointer to the second `ggml_tensor` to compare. Must not be null and must have the same shape as tensor `a`.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains a single element representing the count of equal elements between tensors `a` and `b`. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_count_equal`](../src/ggml.c.driver.md#ggml_count_equal)  (Implementation)


---
### ggml\_repeat<!-- {{#callable_declaration:ggml_repeat}} -->
Repeats a tensor to match the shape of another tensor.
- **Description**: This function is used to create a new tensor that repeats the elements of the first tensor (`a`) to match the dimensions of the second tensor (`b`). It is essential to ensure that the shapes of the tensors are compatible for repetition, which can be verified using the `ggml_can_repeat` function. This function should be called after initializing the context and before performing any tensor operations that depend on the repeated tensor. If the shapes of `a` and `b` are incompatible, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the source tensor that will be repeated. Must not be null and must be compatible for repetition with tensor `b`.
    - `b`: A pointer to the target tensor that defines the shape to which tensor `a` will be repeated. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the repeated elements of tensor `a`, shaped to match tensor `b`. If the operation fails due to incompatible shapes, the behavior is undefined.
- **See also**: [`ggml_repeat`](../src/ggml.c.driver.md#ggml_repeat)  (Implementation)


---
### ggml\_repeat\_4d<!-- {{#callable_declaration:ggml_repeat_4d}} -->
Repeats a tensor to a specified 4D shape.
- **Description**: This function is used to create a new tensor by repeating an existing tensor `a` to fit a specified 4D shape defined by the parameters `ne0`, `ne1`, `ne2`, and `ne3`. It is important to ensure that the dimensions of the new tensor are multiples of the corresponding dimensions of the original tensor; otherwise, an assertion will fail. This function should be called after initializing the context with `ggml_init()`, and it will allocate memory for the new tensor in the context's memory pool.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` that will be repeated. Must not be null and should not be empty.
    - `ne0`: The size of the first dimension of the new tensor. Must be a non-negative integer that is a multiple of `a->ne[0]`.
    - `ne1`: The size of the second dimension of the new tensor. Must be a non-negative integer that is a multiple of `a->ne[1]`.
    - `ne2`: The size of the third dimension of the new tensor. Must be a non-negative integer that is a multiple of `a->ne[2]`.
    - `ne3`: The size of the fourth dimension of the new tensor. Must be a non-negative integer that is a multiple of `a->ne[3]`.
- **Output**: Returns a pointer to the newly created `ggml_tensor` that represents the repeated tensor. The caller is responsible for managing the memory of this tensor.
- **See also**: [`ggml_repeat_4d`](../src/ggml.c.driver.md#ggml_repeat_4d)  (Implementation)


---
### ggml\_repeat\_back<!-- {{#callable_declaration:ggml_repeat_back}} -->
Sums adjacent values in a tensor to fit the shape of another tensor.
- **Description**: This function is used to adjust the shape of tensor `a` to match tensor `b` by summing adjacent values along dimensions greater than zero. It is particularly useful in scenarios where the dimensions of the tensors do not match, and you need to aggregate values from `a` to fit into `b`. Before calling this function, ensure that the tensors are compatible for the operation, as the function will assert that `b` can repeat `a`. The result will be a new tensor that reflects the summed values.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the source tensor that will be summed. Must not be null.
    - `b`: A pointer to the target tensor that defines the desired shape. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the summed values of `a` reshaped to fit `b`. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_repeat_back`](../src/ggml.c.driver.md#ggml_repeat_back)  (Implementation)


---
### ggml\_concat<!-- {{#callable_declaration:ggml_concat}} -->
Concatenates two tensors along a specified dimension.
- **Description**: This function is used to concatenate two tensors, `a` and `b`, along a specified dimension, `dim`. It is important to ensure that both tensors have the same type and compatible shapes in all dimensions except for the one being concatenated. The function must be called after initializing the context with `ggml_init()`. If the specified dimension is invalid or if the tensor types do not match, the function will assert and terminate.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the first tensor to concatenate. Must not be null and must have a valid type.
    - `b`: A pointer to the second tensor to concatenate. Must not be null and must have the same type as tensor `a`.
    - `dim`: An integer specifying the dimension along which to concatenate the tensors. Must be in the range [0, GGML_MAX_DIMS).
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the concatenation of tensors `a` and `b` along the specified dimension. The resulting tensor will have a shape that combines the sizes of `a` and `b` along the concatenation dimension.
- **See also**: [`ggml_concat`](../src/ggml.c.driver.md#ggml_concat)  (Implementation)


---
### ggml\_abs<!-- {{#callable_declaration:ggml_abs}} -->
Computes the absolute value of each element in a tensor.
- **Description**: This function is used to apply the absolute value operation to each element of the input tensor. It should be called after initializing the context and creating the input tensor. The function returns a new tensor containing the absolute values of the elements from the input tensor. If the input tensor is null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the input `ggml_tensor` whose absolute values are to be computed. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` containing the absolute values of the elements from the input tensor. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_abs`](../src/ggml.c.driver.md#ggml_abs)  (Implementation)


---
### ggml\_abs\_inplace<!-- {{#callable_declaration:ggml_abs_inplace}} -->
Applies the absolute value operation to a tensor in place.
- **Description**: This function modifies the input tensor by applying the absolute value operation to each element. It should be called when you want to ensure that all values in the tensor are non-negative. The input tensor must be valid and properly initialized; otherwise, the behavior is undefined. This operation is performed in place, meaning that the original tensor is altered, and no new tensor is created.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the tensor to be modified. Must not be null and must be a valid tensor.
- **Output**: Returns a pointer to the modified `ggml_tensor`, which is the same as the input tensor.
- **See also**: [`ggml_abs_inplace`](../src/ggml.c.driver.md#ggml_abs_inplace)  (Implementation)


---
### ggml\_sgn<!-- {{#callable_declaration:ggml_sgn}} -->
Computes the sign of each element in a tensor.
- **Description**: This function is used to apply the sign operation to each element of the input tensor, returning a new tensor containing the results. It should be called after initializing the context and creating the input tensor. The input tensor must not be null, and it is expected to be a valid tensor object. If the input tensor is invalid or null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. It must not be null.
    - `a`: A pointer to the `ggml_tensor` structure representing the input tensor. It must not be null and should be a valid tensor object.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the sign of each element from the input tensor. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_sgn`](../src/ggml.c.driver.md#ggml_sgn)  (Implementation)


---
### ggml\_sgn\_inplace<!-- {{#callable_declaration:ggml_sgn_inplace}} -->
Applies the sign function to a tensor in place.
- **Description**: This function modifies the input tensor by applying the sign function, which sets each element to -1, 0, or 1 based on its value. It is important to ensure that the input tensor is valid and properly initialized before calling this function. The operation is performed in place, meaning that the original tensor is updated directly without allocating new memory for the result.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the tensor to be modified. Must not be null and should be properly initialized.
- **Output**: Returns a pointer to the modified `ggml_tensor`, which is the same as the input tensor.
- **See also**: [`ggml_sgn_inplace`](../src/ggml.c.driver.md#ggml_sgn_inplace)  (Implementation)


---
### ggml\_neg<!-- {{#callable_declaration:ggml_neg}} -->
Negates the values of a tensor.
- **Description**: This function is used to compute the element-wise negation of a given tensor. It should be called with a valid `ggml_context` that has been initialized and a non-null tensor `a`. The resulting tensor will have the same shape and type as the input tensor, but with all values negated. If the input tensor is null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` that must be initialized before calling this function. It manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` that contains the values to be negated. This tensor must be valid and not null. The function will return a new tensor with the negated values.
- **Output**: Returns a pointer to a new `ggml_tensor` containing the negated values of the input tensor. The output tensor will have the same dimensions and type as the input tensor.
- **See also**: [`ggml_neg`](../src/ggml.c.driver.md#ggml_neg)  (Implementation)


---
### ggml\_neg\_inplace<!-- {{#callable_declaration:ggml_neg_inplace}} -->
Negates the values of a tensor in place.
- **Description**: This function is used to negate the values of a given tensor directly in memory, modifying the original tensor. It should be called when you want to apply a negation operation without creating a new tensor, which can be more efficient in terms of memory usage. The function requires a valid `ggml_context` and a `ggml_tensor` that has been properly initialized. If the input tensor is null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` that will be negated. This tensor must be properly initialized and allocated. Must not be null.
- **Output**: Returns a pointer to the same `ggml_tensor` that was passed in, now containing the negated values.
- **See also**: [`ggml_neg_inplace`](../src/ggml.c.driver.md#ggml_neg_inplace)  (Implementation)


---
### ggml\_step<!-- {{#callable_declaration:ggml_step}} -->
Applies the step function to a tensor.
- **Description**: This function is used to apply the step activation function to the input tensor, which is commonly used in neural networks. It should be called after initializing the context and creating the input tensor. The step function outputs 0 for negative input values and 1 for non-negative input values. The resulting tensor will have the same shape as the input tensor. Ensure that the input tensor is valid and properly initialized; otherwise, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the input tensor. Must not be null and should be properly initialized.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of applying the step function to the input tensor.
- **See also**: [`ggml_step`](../src/ggml.c.driver.md#ggml_step)  (Implementation)


---
### ggml\_step\_inplace<!-- {{#callable_declaration:ggml_step_inplace}} -->
Applies the step function to a tensor in place.
- **Description**: This function modifies the input tensor by applying the step function, which sets all negative values to zero and keeps positive values unchanged. It is intended for use when you need to apply this transformation directly to an existing tensor without creating a new one. The function should be called after initializing the `ggml_context` and ensuring that the input tensor is valid. If the input tensor is null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the state of the library. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the tensor to be modified. Must not be null.
- **Output**: Returns a pointer to the modified `ggml_tensor`, which is the same as the input tensor.
- **See also**: [`ggml_step_inplace`](../src/ggml.c.driver.md#ggml_step_inplace)  (Implementation)


---
### ggml\_tanh<!-- {{#callable_declaration:ggml_tanh}} -->
Computes the hyperbolic tangent of a tensor.
- **Description**: This function is used to apply the hyperbolic tangent operation to a given tensor, producing a new tensor as the result. It should be called after initializing the context and creating the input tensor. The input tensor must not be null, and it should be properly allocated and initialized. The function handles the operation in a way that ensures the output tensor is created based on the input tensor's dimensions and data type.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the input `ggml_tensor` on which the hyperbolic tangent operation will be performed. Must not be null and should be properly initialized.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of applying the hyperbolic tangent operation to the input tensor. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_tanh`](../src/ggml.c.driver.md#ggml_tanh)  (Implementation)


---
### ggml\_tanh\_inplace<!-- {{#callable_declaration:ggml_tanh_inplace}} -->
Applies the hyperbolic tangent function to a tensor in place.
- **Description**: This function modifies the input tensor by applying the hyperbolic tangent operation directly to its elements. It is intended for use when the tensor has already been created and initialized. The function should be called with a valid `ggml_context` and a non-null `ggml_tensor`. If the input tensor is invalid or null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` that will be modified in place. Must not be null and should be a valid tensor.
- **Output**: Returns a pointer to the modified `ggml_tensor`, which is the same as the input tensor.
- **See also**: [`ggml_tanh_inplace`](../src/ggml.c.driver.md#ggml_tanh_inplace)  (Implementation)


---
### ggml\_elu<!-- {{#callable_declaration:ggml_elu}} -->
Applies the ELU activation function to a tensor.
- **Description**: This function is used to apply the Exponential Linear Unit (ELU) activation function to a given tensor. It is typically called during the forward pass of a neural network to introduce non-linearity. The input tensor must be valid and properly initialized, and the function should be called after the context has been set up with `ggml_init`. If the input tensor is null, the function will not perform any operation and may return a null pointer.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the input tensor. Must not be null and should be properly initialized.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of applying the ELU function to the input tensor. If the input tensor is invalid, the return value may be null.
- **See also**: [`ggml_elu`](../src/ggml.c.driver.md#ggml_elu)  (Implementation)


---
### ggml\_elu\_inplace<!-- {{#callable_declaration:ggml_elu_inplace}} -->
Applies the ELU activation function in place.
- **Description**: This function modifies the input tensor by applying the Exponential Linear Unit (ELU) activation function directly to its elements. It is intended for use in neural network computations where the ELU activation is required. The function should be called after the tensor has been properly initialized and allocated. It is important to ensure that the input tensor is valid and not null; otherwise, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the state of the library. Must not be null.
    - `a`: A pointer to the `ggml_tensor` structure that contains the data to which the ELU activation will be applied. Must not be null.
- **Output**: Returns a pointer to the modified `ggml_tensor` that now contains the results of the ELU activation applied in place.
- **See also**: [`ggml_elu_inplace`](../src/ggml.c.driver.md#ggml_elu_inplace)  (Implementation)


---
### ggml\_relu<!-- {{#callable_declaration:ggml_relu}} -->
Applies the ReLU activation function to a tensor.
- **Description**: This function is used to apply the Rectified Linear Unit (ReLU) activation function to the input tensor. It is typically called during the forward pass of a neural network to introduce non-linearity. The input tensor must be valid and properly initialized, and the function will return a new tensor containing the result of applying the ReLU operation. If the input tensor is null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. This must not be null and should be properly initialized before calling the function.
    - `a`: A pointer to a `ggml_tensor` structure representing the input tensor. This tensor must be valid and initialized. If it is null, the function's behavior is undefined.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of applying the ReLU function to the input tensor. The output tensor will have the same shape as the input tensor, with all negative values replaced by zero.
- **See also**: [`ggml_relu`](../src/ggml.c.driver.md#ggml_relu)  (Implementation)


---
### ggml\_leaky\_relu<!-- {{#callable_declaration:ggml_leaky_relu}} -->
Applies the leaky ReLU activation function to a tensor.
- **Description**: This function is used to apply the leaky ReLU activation function to a given tensor, which is commonly used in neural networks to introduce non-linearity. It can be called after initializing the context and creating the input tensor. The `negative_slope` parameter determines the slope of the function for negative input values, allowing for a small, non-zero gradient when the input is less than zero. The `inplace` parameter specifies whether the operation should modify the input tensor directly or create a new tensor for the result. If `inplace` is set to true, the original tensor will be modified, and the function will return a view of it. If false, a new tensor will be created, and the original tensor remains unchanged.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the input `ggml_tensor` on which the leaky ReLU operation will be applied. Must not be null.
    - `negative_slope`: A float value representing the slope for negative input values. It should be a non-negative value.
    - `inplace`: A boolean value indicating whether to perform the operation in-place (modifying the input tensor) or to create a new tensor. If true, the input tensor is modified; if false, a new tensor is created.
- **Output**: Returns a pointer to the resulting `ggml_tensor` after applying the leaky ReLU function. If `inplace` is true, this will be a view of the input tensor; otherwise, it will be a new tensor containing the results.
- **See also**: [`ggml_leaky_relu`](../src/ggml.c.driver.md#ggml_leaky_relu)  (Implementation)


---
### ggml\_relu\_inplace<!-- {{#callable_declaration:ggml_relu_inplace}} -->
Applies the ReLU activation function in place.
- **Description**: This function modifies the input tensor by applying the ReLU (Rectified Linear Unit) activation function directly to its elements. It should be called when you want to apply the ReLU operation to a tensor without creating a new tensor, thus saving memory. The input tensor must be valid and properly initialized; otherwise, the behavior is undefined. The function operates in place, meaning the original tensor is altered.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the input tensor. Must not be null and must be properly initialized.
- **Output**: Returns a pointer to the modified `ggml_tensor` that has undergone the ReLU operation.
- **See also**: [`ggml_relu_inplace`](../src/ggml.c.driver.md#ggml_relu_inplace)  (Implementation)


---
### ggml\_sigmoid<!-- {{#callable_declaration:ggml_sigmoid}} -->
Applies the sigmoid function to a tensor.
- **Description**: This function is used to compute the sigmoid activation for each element in the input tensor. It should be called after initializing the `ggml_context` and creating the input tensor. The function will return a new tensor containing the results of the sigmoid operation. If the input tensor is null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the input tensor. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of applying the sigmoid function to each element of the input tensor.
- **See also**: [`ggml_sigmoid`](../src/ggml.c.driver.md#ggml_sigmoid)  (Implementation)


---
### ggml\_sigmoid\_inplace<!-- {{#callable_declaration:ggml_sigmoid_inplace}} -->
Applies the sigmoid function to a tensor in place.
- **Description**: This function modifies the input tensor by applying the sigmoid activation function to each element. It is intended for use when you want to transform the values of a tensor directly without creating a new tensor. The function should be called after the tensor has been properly initialized and allocated. It is important to ensure that the input tensor is valid and not null; otherwise, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure that contains the data to be transformed. Must not be null and should be properly initialized.
- **Output**: Returns a pointer to the modified `ggml_tensor`, which now contains the results of the sigmoid function applied to its elements.
- **See also**: [`ggml_sigmoid_inplace`](../src/ggml.c.driver.md#ggml_sigmoid_inplace)  (Implementation)


---
### ggml\_gelu<!-- {{#callable_declaration:ggml_gelu}} -->
Applies the GELU activation function to a tensor.
- **Description**: This function is used to apply the Gaussian Error Linear Unit (GELU) activation function to the input tensor. It should be called after initializing the context and creating the input tensor. The input tensor must not be null, and it should be of a compatible type. The function will return a new tensor that contains the result of applying the GELU function to each element of the input tensor.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the input tensor. Must not be null and should be of a compatible type for the GELU operation.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the GELU activation applied to the input tensor. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_gelu`](../src/ggml.c.driver.md#ggml_gelu)  (Implementation)


---
### ggml\_gelu\_inplace<!-- {{#callable_declaration:ggml_gelu_inplace}} -->
Applies the GELU activation function in place.
- **Description**: This function modifies the input tensor by applying the Gaussian Error Linear Unit (GELU) activation function directly to its elements. It is intended for use when the tensor has already been created and initialized, and it should be called when you want to apply the GELU activation without creating a new tensor. The input tensor must not be null, and it is expected to be of a compatible type for the operation. If the input tensor is invalid or not suitable for the operation, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the state of the library. This must not be null.
    - `a`: A pointer to the `ggml_tensor` structure that will be modified in place. This tensor must not be null and should be of a type compatible with the GELU operation.
- **Output**: Returns a pointer to the modified `ggml_tensor`, which is the same as the input tensor after applying the GELU activation.
- **See also**: [`ggml_gelu_inplace`](../src/ggml.c.driver.md#ggml_gelu_inplace)  (Implementation)


---
### ggml\_gelu\_erf<!-- {{#callable_declaration:ggml_gelu_erf}} -->
Computes the Gaussian Error Linear Unit (GELU) using the error function.
- **Description**: This function is used to apply the GELU activation function to a tensor, which is commonly used in neural networks to introduce non-linearity. It should be called after initializing the `ggml_context` and creating the input tensor. The input tensor must not be null, and it should be of a compatible type. If the input tensor is invalid or null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the input tensor. Must not be null and should be of a compatible type for the operation.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of applying the GELU activation function to the input tensor. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_gelu_erf`](../src/ggml.c.driver.md#ggml_gelu_erf)  (Implementation)


---
### ggml\_gelu\_erf\_inplace<!-- {{#callable_declaration:ggml_gelu_erf_inplace}} -->
Applies the GELU activation function using the error function in place.
- **Description**: This function modifies the input tensor by applying the Gaussian Error Linear Unit (GELU) activation function using the error function (erf) in place. It is intended for use in neural network computations where the input tensor has already been created and initialized. The function should be called after the tensor has been allocated and populated with data. It is important to ensure that the input tensor is valid and not null; otherwise, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the input tensor. This tensor will be modified in place. Must not be null.
- **Output**: Returns a pointer to the modified `ggml_tensor` that has undergone the GELU activation function. This is the same tensor that was passed as input.
- **See also**: [`ggml_gelu_erf_inplace`](../src/ggml.c.driver.md#ggml_gelu_erf_inplace)  (Implementation)


---
### ggml\_gelu\_quick<!-- {{#callable_declaration:ggml_gelu_quick}} -->
Applies the GELU activation function quickly.
- **Description**: This function is used to apply the GELU (Gaussian Error Linear Unit) activation function to a given tensor. It is typically called during the forward pass of a neural network to introduce non-linearity. The function should be invoked after the tensor has been created and initialized. It is important to ensure that the input tensor is valid and properly allocated; otherwise, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the input tensor. Must not be null and should be properly initialized.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of applying the GELU activation function to the input tensor. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_gelu_quick`](../src/ggml.c.driver.md#ggml_gelu_quick)  (Implementation)


---
### ggml\_gelu\_quick\_inplace<!-- {{#callable_declaration:ggml_gelu_quick_inplace}} -->
Applies the GELU activation function in place.
- **Description**: This function modifies the input tensor by applying the GELU (Gaussian Error Linear Unit) activation function directly to its elements. It is intended for use when the input tensor has already been created and initialized. The function should be called when the tensor is ready for activation, and it will alter the values of the tensor in place, meaning that the original tensor will be updated with the new values. Ensure that the input tensor is valid and properly allocated before calling this function.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` structure that contains the data to which the GELU activation will be applied. Must not be null and should be a valid tensor.
- **Output**: Returns a pointer to the modified `ggml_tensor` that now contains the results of the GELU activation applied in place.
- **See also**: [`ggml_gelu_quick_inplace`](../src/ggml.c.driver.md#ggml_gelu_quick_inplace)  (Implementation)


---
### ggml\_silu<!-- {{#callable_declaration:ggml_silu}} -->
Applies the Sigmoid Linear Unit (SiLU) activation function to a tensor.
- **Description**: This function is used to apply the SiLU activation function to the input tensor, which is commonly used in neural networks to introduce non-linearity. It should be called after the tensor has been created and initialized. The input tensor must not be null, and the function will return a new tensor that contains the result of applying the SiLU function. If the input tensor is invalid or null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the input tensor. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of applying the SiLU function to the input tensor. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_silu`](../src/ggml.c.driver.md#ggml_silu)  (Implementation)


---
### ggml\_silu\_inplace<!-- {{#callable_declaration:ggml_silu_inplace}} -->
Applies the SiLU activation function in place.
- **Description**: This function modifies the input tensor by applying the SiLU (Sigmoid Linear Unit) activation function directly to its elements. It is intended for use when the tensor has been properly initialized and allocated within a valid `ggml_context`. The function operates in place, meaning that the original tensor is updated with the new values, and no new tensor is created. It is important to ensure that the input tensor is not null before calling this function, as passing a null pointer will lead to undefined behavior.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure that represents the input tensor. This tensor must be properly initialized and allocated. Must not be null.
- **Output**: Returns a pointer to the modified `ggml_tensor`, which is the same as the input tensor `a`.
- **See also**: [`ggml_silu_inplace`](../src/ggml.c.driver.md#ggml_silu_inplace)  (Implementation)


---
### ggml\_silu\_back<!-- {{#callable_declaration:ggml_silu_back}} -->
Computes the backward pass of the SiLU activation function.
- **Description**: This function is used to compute the gradient of the SiLU (Sigmoid Linear Unit) activation function during backpropagation in neural networks. It should be called after the forward pass of the SiLU activation has been computed. The function takes two tensors as input: the first tensor represents the input to the SiLU function, while the second tensor represents the gradient of the loss with respect to the output of the SiLU function. The result is a new tensor that contains the gradient of the loss with respect to the input of the SiLU function.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` representing the input to the SiLU function. Must not be null.
    - `b`: A pointer to a `ggml_tensor` representing the gradient of the loss with respect to the output of the SiLU function. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the gradient of the loss with respect to the input of the SiLU function.
- **See also**: [`ggml_silu_back`](../src/ggml.c.driver.md#ggml_silu_back)  (Implementation)


---
### ggml\_hardswish<!-- {{#callable_declaration:ggml_hardswish}} -->
Applies the hard swish activation function to a tensor.
- **Description**: This function is used to apply the hard swish activation function to the input tensor. It should be called after initializing the context and creating the input tensor. The function takes a context and a tensor as parameters, and it returns a new tensor that contains the result of applying the hard swish operation. If the input tensor is null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the input tensor. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the hard swish operation applied to the input tensor.
- **See also**: [`ggml_hardswish`](../src/ggml.c.driver.md#ggml_hardswish)  (Implementation)


---
### ggml\_hardsigmoid<!-- {{#callable_declaration:ggml_hardsigmoid}} -->
Applies the hard sigmoid activation function to a tensor.
- **Description**: This function is used to apply the hard sigmoid activation function to the input tensor, which is commonly used in neural networks. It should be called after initializing the context and creating the input tensor. The function will return a new tensor that contains the result of the hard sigmoid operation applied element-wise to the input tensor. If the input tensor is null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the input tensor. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the hard sigmoid operation. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_hardsigmoid`](../src/ggml.c.driver.md#ggml_hardsigmoid)  (Implementation)


---
### ggml\_exp<!-- {{#callable_declaration:ggml_exp}} -->
Computes the exponential of each element in a tensor.
- **Description**: This function is used to apply the exponential function to each element of the input tensor. It should be called after initializing the context and creating the input tensor. The input tensor must not be null, and it should be a valid tensor created with the appropriate functions from the library. If the input tensor is invalid or null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the input tensor. Must not be null and should be a valid tensor.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of applying the exponential function to each element of the input tensor. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_exp`](../src/ggml.c.driver.md#ggml_exp)  (Implementation)


---
### ggml\_exp\_inplace<!-- {{#callable_declaration:ggml_exp_inplace}} -->
Applies the exponential function to each element of a tensor in place.
- **Description**: This function is used to compute the exponential of each element in the specified tensor, modifying the tensor directly. It should be called when the tensor has been properly initialized and allocated. The input tensor must not be null, and it is expected to be of a compatible type for the exponential operation. If the input tensor is invalid or null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the tensor to which the exponential function will be applied. Must not be null and should be of a compatible type.
- **Output**: Returns a pointer to the modified `ggml_tensor` after applying the exponential function. The original tensor is mutated in place.
- **See also**: [`ggml_exp_inplace`](../src/ggml.c.driver.md#ggml_exp_inplace)  (Implementation)


---
### ggml\_norm<!-- {{#callable_declaration:ggml_norm}} -->
Normalizes a tensor along its rows.
- **Description**: This function is used to normalize the input tensor `a` by adjusting its values based on the specified epsilon value. It is typically called when you need to ensure that the tensor values are scaled appropriately, which is common in various machine learning tasks. The function should be invoked after the tensor has been created and initialized. If the input tensor is null, the function will handle it gracefully without causing a crash.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` that needs to be normalized. Must not be null.
    - `eps`: A float value that serves as a small constant to prevent division by zero during normalization. It should be a positive value.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the normalized values. If the input tensor is null, the function will return null.
- **See also**: [`ggml_norm`](../src/ggml.c.driver.md#ggml_norm)  (Implementation)


---
### ggml\_norm\_inplace<!-- {{#callable_declaration:ggml_norm_inplace}} -->
Normalizes a tensor in place.
- **Description**: This function is used to normalize the values of a tensor along its rows, which is particularly useful in machine learning tasks to ensure that the data is scaled appropriately. It must be called with a valid `ggml_context` and a non-null tensor. The `eps` parameter is used to prevent division by zero during normalization; it should be a small positive value. If the tensor is empty or the context is invalid, the function will handle these cases gracefully without causing a crash.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that must be valid and initialized before calling this function. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure that represents the tensor to be normalized. Must not be null.
    - `eps`: A float value that serves as a small epsilon to prevent division by zero. It should be a small positive number.
- **Output**: Returns a pointer to the normalized `ggml_tensor`, which is the same tensor passed in as input, modified in place.
- **See also**: [`ggml_norm_inplace`](../src/ggml.c.driver.md#ggml_norm_inplace)  (Implementation)


---
### ggml\_rms\_norm<!-- {{#callable_declaration:ggml_rms_norm}} -->
Normalizes the input tensor using root mean square normalization.
- **Description**: This function is used to apply root mean square normalization to a given tensor, which is useful in various machine learning tasks to stabilize the training process. It should be called after the tensor has been created and initialized. The `eps` parameter is used to prevent division by zero, and it should be a small positive value. If the input tensor is null, the function will handle it gracefully, returning a null pointer.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` structure that represents the input tensor to be normalized. Must not be null.
    - `eps`: A float value that serves as a small constant to prevent division by zero. It should be a positive value.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the normalized values. If the input tensor is null, the return value will also be null.
- **See also**: [`ggml_rms_norm`](../src/ggml.c.driver.md#ggml_rms_norm)  (Implementation)


---
### ggml\_rms\_norm\_inplace<!-- {{#callable_declaration:ggml_rms_norm_inplace}} -->
Normalizes a tensor in place using RMS normalization.
- **Description**: This function performs in-place RMS normalization on the specified tensor, which is useful for stabilizing the training of machine learning models. It should be called after the tensor has been properly initialized and allocated. The function modifies the input tensor directly, and the normalization is performed using the specified epsilon value to prevent division by zero. If the input tensor is null, the function will not perform any operation.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` structure that will be normalized in place. Must not be null.
    - `eps`: A float value representing a small constant added to the denominator for numerical stability. It should be a positive value to ensure proper normalization.
- **Output**: Returns a pointer to the normalized `ggml_tensor`, which is the same as the input tensor since the operation is performed in place.
- **See also**: [`ggml_rms_norm_inplace`](../src/ggml.c.driver.md#ggml_rms_norm_inplace)  (Implementation)


---
### ggml\_group\_norm<!-- {{#callable_declaration:ggml_group_norm}} -->
Performs group normalization on a tensor.
- **Description**: This function applies group normalization to the input tensor, which is useful in various machine learning tasks to stabilize the learning process. It should be called after the tensor has been created and initialized. The function takes the number of groups and a small epsilon value to prevent division by zero during normalization. If the input tensor is not properly shaped or if the number of groups is invalid, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the input tensor of type `ggml_tensor`. This tensor must be properly initialized and allocated. Must not be null.
    - `n_groups`: An integer representing the number of groups for normalization. It must be a positive integer that divides the total number of elements in the tensor evenly.
    - `eps`: A small floating-point value used to prevent division by zero during normalization. It should be a positive number.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the group normalization. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_group_norm`](../src/ggml.c.driver.md#ggml_group_norm)  (Implementation)


---
### ggml\_group\_norm\_inplace<!-- {{#callable_declaration:ggml_group_norm_inplace}} -->
Performs in-place group normalization on a tensor.
- **Description**: This function is used to apply group normalization to a tensor, which is particularly useful in deep learning applications to stabilize the training process. It modifies the input tensor directly, so it should be called when the tensor is ready for normalization. The function expects the tensor to be properly initialized and allocated in memory. It is important to ensure that the number of groups specified does not exceed the number of channels in the tensor, as this may lead to undefined behavior.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` that will be normalized in place. This tensor must be properly initialized and allocated. Must not be null.
    - `n_groups`: An integer specifying the number of groups for normalization. Must be a positive integer and should not exceed the number of channels in the tensor.
    - `eps`: A small float value added to the variance to prevent division by zero. Must be a positive float.
- **Output**: Returns a pointer to the normalized `ggml_tensor`, which is the same as the input tensor `a` since the operation is performed in place.
- **See also**: [`ggml_group_norm_inplace`](../src/ggml.c.driver.md#ggml_group_norm_inplace)  (Implementation)


---
### ggml\_l2\_norm<!-- {{#callable_declaration:ggml_l2_norm}} -->
Computes the L2 normalization of a tensor.
- **Description**: This function is used to perform L2 normalization on a given tensor, which is a common operation in machine learning and data processing. It should be called after initializing the `ggml_context` and creating the tensor to be normalized. The function takes an epsilon value to prevent division by zero, which can be particularly useful when the tensor may contain very small values. If the input tensor is null, the function will handle it gracefully, returning a null pointer.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that must be initialized before calling this function. It manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` that will be normalized. This tensor must be properly initialized and allocated. If this pointer is null, the function will return null.
    - `eps`: A float value that serves as a small constant to prevent division by zero during normalization. It should be a non-negative value.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the L2 normalized values of the input tensor. If the input tensor is null, the function returns null.
- **See also**: [`ggml_l2_norm`](../src/ggml.c.driver.md#ggml_l2_norm)  (Implementation)


---
### ggml\_l2\_norm\_inplace<!-- {{#callable_declaration:ggml_l2_norm_inplace}} -->
Performs in-place L2 normalization on a tensor.
- **Description**: This function is used to normalize the elements of a tensor along its rows using L2 normalization, which scales the tensor such that the sum of the squares of its elements equals one. It should be called when the tensor is already allocated and initialized. The function modifies the input tensor directly, and the normalization is performed in-place, meaning that the original tensor is updated with the normalized values. The `eps` parameter is used to prevent division by zero, and it should be a small positive value.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` that will be normalized. This tensor must be initialized and allocated prior to calling this function. Must not be null.
    - `eps`: A small positive float value used to prevent division by zero during normalization. It should be greater than zero.
- **Output**: Returns a pointer to the normalized `ggml_tensor`, which is the same as the input tensor `a` since the operation is performed in-place.
- **See also**: [`ggml_l2_norm_inplace`](../src/ggml.c.driver.md#ggml_l2_norm_inplace)  (Implementation)


---
### ggml\_rms\_norm\_back<!-- {{#callable_declaration:ggml_rms_norm_back}} -->
Computes the backward pass of RMS normalization.
- **Description**: This function is used to compute the gradients for the backward pass of the RMS normalization operation. It should be called after performing a forward pass of the RMS normalization operation. The function takes two input tensors: `a`, which represents the input tensor, and `b`, which represents the gradient of the output tensor with respect to the loss. The `eps` parameter is a small value added for numerical stability during the computation. It is important to ensure that the input tensors are properly initialized and that the context is valid before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `struct ggml_context` that manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `struct ggml_tensor` representing the input tensor for which gradients are being computed. Must not be null.
    - `b`: A pointer to a `struct ggml_tensor` representing the gradient of the output tensor with respect to the loss. Must not be null.
    - `eps`: A float value representing a small constant added for numerical stability. It should be a non-negative value.
- **Output**: Returns a pointer to a new `struct ggml_tensor` that contains the computed gradients for the input tensor `a`. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_rms_norm_back`](../src/ggml.c.driver.md#ggml_rms_norm_back)  (Implementation)


---
### ggml\_mul\_mat<!-- {{#callable_declaration:ggml_mul_mat}} -->
Performs matrix multiplication of two tensors.
- **Description**: This function is used to multiply two tensors, `a` and `b`, where `a` must have dimensions compatible for matrix multiplication with `b`. It is essential to ensure that the number of columns in `a` matches the number of rows in `b`. The function should be called after initializing the context with `ggml_init()`. If the tensors are not compatible for multiplication, the function will assert and terminate. The result is a new tensor that holds the product of the two input tensors.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the first tensor to be multiplied. It must not be null and must have dimensions compatible with the second tensor for matrix multiplication.
    - `b`: A pointer to the second tensor to be multiplied. It must not be null and must have dimensions compatible with the first tensor for matrix multiplication.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the matrix multiplication. The caller is responsible for managing the memory of the resulting tensor.
- **See also**: [`ggml_mul_mat`](../src/ggml.c.driver.md#ggml_mul_mat)  (Implementation)


---
### ggml\_mul\_mat\_set\_prec<!-- {{#callable_declaration:ggml_mul_mat_set_prec}} -->
Changes the precision of a matrix multiplication.
- **Description**: This function is used to set the precision for matrix multiplication operations on a specified tensor. It should be called when the tensor is already defined as a matrix multiplication operation (i.e., its operation type is `GGML_OP_MUL_MAT`). The precision can be adjusted to optimize performance or accuracy based on the requirements of the computation. If the tensor does not represent a matrix multiplication operation, the behavior is undefined.
- **Inputs**:
    - `a`: A pointer to a `ggml_tensor` structure that represents the tensor for which the precision is to be set. This tensor must have been created and must represent a matrix multiplication operation. Must not be null.
    - `prec`: An enumeration value of type `ggml_prec` that specifies the desired precision. Valid values include `GGML_PREC_DEFAULT` and `GGML_PREC_F32`. The function will set the precision accordingly.
- **Output**: None
- **See also**: [`ggml_mul_mat_set_prec`](../src/ggml.c.driver.md#ggml_mul_mat_set_prec)  (Implementation)


---
### ggml\_mul\_mat\_id<!-- {{#callable_declaration:ggml_mul_mat_id}} -->
Performs matrix multiplication with an identity mapping based on specified indices.
- **Description**: This function is used to perform matrix multiplication between two tensors, where the second tensor is indexed by a third tensor of identifiers. It is essential to ensure that the input tensors are correctly shaped: the first tensor must be a 3D tensor with a single matrix per expert, the second tensor must also be a 3D tensor, and the identifiers tensor must be a 2D tensor with a specific structure. The function will assert the validity of these conditions before proceeding. If any of the assertions fail, the function will not execute, ensuring that the user is informed of the incorrect input shapes.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `as`: A pointer to a 3D `ggml_tensor` representing the first matrix in the multiplication. It must have a shape where the last dimension is 1, indicating a single matrix per expert.
    - `b`: A pointer to a 3D `ggml_tensor` representing the second matrix in the multiplication. It must also conform to the expected 3D shape.
    - `ids`: A pointer to a 2D `ggml_tensor` of type `GGML_TYPE_I32` that contains the indices for selecting rows from the second tensor. It must have a shape that matches the requirements for indexing the second tensor.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the matrix multiplication. The resulting tensor will have a shape determined by the dimensions of the input tensors.
- **See also**: [`ggml_mul_mat_id`](../src/ggml.c.driver.md#ggml_mul_mat_id)  (Implementation)


---
### ggml\_out\_prod<!-- {{#callable_declaration:ggml_out_prod}} -->
Computes the outer product of two tensors.
- **Description**: This function is used to compute the outer product of two input tensors, `a` and `b`, producing a new tensor that represents the result. It is essential to ensure that the dimensions of the tensors are compatible for the outer product operation, specifically that `a` is not transposed and that it can be broadcasted to the dimensions of `b`. The function should be called after initializing the context with `ggml_init`. If the input tensors do not meet the required conditions, the function will assert and terminate.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the first tensor (`ggml_tensor`) involved in the outer product. Must not be null and must not be transposed.
    - `b`: A pointer to the second tensor (`ggml_tensor`) involved in the outer product. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the outer product. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_out_prod`](../src/ggml.c.driver.md#ggml_out_prod)  (Implementation)


---
### ggml\_scale<!-- {{#callable_declaration:ggml_scale}} -->
Scales a tensor by a given factor.
- **Description**: This function is used to scale the values of a tensor by a specified scalar factor. It is typically called when you need to adjust the magnitude of the tensor's values, such as in normalization or preprocessing steps. The function requires a valid `ggml_context` and a non-null tensor `a`. If the tensor is null or if the context is invalid, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` to be scaled. Must not be null.
    - `s`: A float representing the scaling factor. This value can be any finite float, including positive, negative, or zero.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the scaled values of the input tensor. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_scale`](../src/ggml.c.driver.md#ggml_scale)  (Implementation)


---
### ggml\_scale\_inplace<!-- {{#callable_declaration:ggml_scale_inplace}} -->
Scales a tensor in place by a given factor.
- **Description**: This function modifies the tensor `a` by scaling its elements with the specified factor `s`. It is important to ensure that the tensor has been properly initialized and allocated before calling this function. The operation is performed in place, meaning that the original tensor is updated directly, and no new tensor is created. If the input tensor is null, the function will handle it gracefully without performing any operation.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be valid and initialized. This context is used for managing memory and operations related to tensors.
    - `a`: A pointer to the `ggml_tensor` structure that will be scaled. This tensor must not be null and should be properly initialized. The function will modify this tensor directly.
    - `s`: A float value representing the scaling factor. This value can be any finite float. If `s` is zero, the tensor will be set to zero.
- **Output**: Returns a pointer to the modified `ggml_tensor` that has been scaled in place.
- **See also**: [`ggml_scale_inplace`](../src/ggml.c.driver.md#ggml_scale_inplace)  (Implementation)


---
### ggml\_set<!-- {{#callable_declaration:ggml_set}} -->
Sets the values of a tensor based on another tensor.
- **Description**: This function is used to set the values of a tensor `a` using the values from another tensor `b`, starting from a specified offset and considering the specified dimensions. It is important to ensure that the tensors are compatible in terms of their dimensions and that the context `ctx` has been properly initialized. The function modifies the tensor `a` in place, and it is expected to be called when the tensor `a` is already allocated and ready to be updated.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must not be null and should be initialized before calling this function.
    - `a`: A pointer to the `ggml_tensor` structure that will be modified. This tensor must be allocated and should have sufficient space to accommodate the values being set.
    - `b`: A pointer to the `ggml_tensor` structure containing the values to set in tensor `a`. This tensor must not be null and should have compatible dimensions with `a`.
    - `nb1`: The size of the first dimension to consider when setting values. This should be a positive integer.
    - `nb2`: The size of the second dimension to consider when setting values. This should be a positive integer.
    - `nb3`: The size of the third dimension to consider when setting values. This should be a positive integer.
    - `offset`: The byte offset in tensor `a` from which to start setting values. This should be a non-negative integer.
- **Output**: Returns a pointer to the modified tensor `a`.
- **See also**: [`ggml_set`](../src/ggml.c.driver.md#ggml_set)  (Implementation)


---
### ggml\_set\_inplace<!-- {{#callable_declaration:ggml_set_inplace}} -->
Sets a tensor's values in place.
- **Description**: This function modifies the values of a tensor by setting a specified view of it to the values of another tensor. It is intended for use when you want to update a portion of an existing tensor without creating a new one. The function should be called after initializing the context and ensuring that both tensors are valid. The dimensions and offsets must be carefully managed to avoid out-of-bounds access, as invalid parameters may lead to undefined behavior.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized and valid. Must not be null.
    - `a`: A pointer to the `ggml_tensor` that will be modified. Must not be null.
    - `b`: A pointer to the `ggml_tensor` whose values will be copied into `a`. Must not be null.
    - `nb1`: The stride in bytes for the first dimension of the view. Must be a positive value.
    - `nb2`: The stride in bytes for the second dimension of the view. Must be a positive value.
    - `nb3`: The stride in bytes for the third dimension of the view. Must be a positive value.
    - `offset`: The byte offset into the tensor `a` where the values will be set. Must be a non-negative value.
- **Output**: Returns a pointer to the modified `ggml_tensor` `a`, which now contains the updated values.
- **See also**: [`ggml_set_inplace`](../src/ggml.c.driver.md#ggml_set_inplace)  (Implementation)


---
### ggml\_set\_1d<!-- {{#callable_declaration:ggml_set_1d}} -->
Sets a 1D tensor's values from another tensor.
- **Description**: This function is used to set the values of a 1D tensor by copying data from another tensor, starting at a specified byte offset. It is important to ensure that the destination tensor has been properly initialized and that the source tensor contains valid data. The function should be called after the context has been initialized and the tensors have been created. If the provided tensors are invalid or if the offset exceeds the bounds of the destination tensor, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must not be null and should be initialized before calling this function.
    - `a`: A pointer to the destination `ggml_tensor` that will receive the values. This tensor must be a 1D tensor and must not be null.
    - `b`: A pointer to the source `ggml_tensor` from which values will be copied. This tensor must also be a 1D tensor and must not be null.
    - `offset`: A size_t value representing the byte offset in the destination tensor where the values from the source tensor will be copied. This value must be within the bounds of the destination tensor.
- **Output**: Returns a pointer to the destination tensor `a` after the values have been set. If the operation is successful, the tensor will contain the updated values from the source tensor.
- **See also**: [`ggml_set_1d`](../src/ggml.c.driver.md#ggml_set_1d)  (Implementation)


---
### ggml\_set\_1d\_inplace<!-- {{#callable_declaration:ggml_set_1d_inplace}} -->
Sets the values of a 1D tensor in place.
- **Description**: This function modifies the contents of a specified 1D tensor by setting its values based on another tensor, starting from a given byte offset. It is essential to ensure that the destination tensor is properly initialized and has sufficient size to accommodate the values being set. The function should be called when the context is valid and the tensors involved are compatible in terms of dimensions and data types. If the input tensors are invalid or incompatible, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the destination `ggml_tensor` that will be modified. This tensor must be a 1D tensor and must not be null.
    - `b`: A pointer to the source `ggml_tensor` whose values will be copied into the destination tensor. This tensor must also be a 1D tensor and must not be null.
    - `offset`: A size_t value representing the byte offset in the destination tensor where the values from the source tensor will be set. The offset must be within the bounds of the destination tensor.
- **Output**: Returns a pointer to the modified destination tensor `a`. If the operation is successful, the tensor will contain the new values set from tensor `b` starting at the specified offset.
- **See also**: [`ggml_set_1d_inplace`](../src/ggml.c.driver.md#ggml_set_1d_inplace)  (Implementation)


---
### ggml\_set\_2d<!-- {{#callable_declaration:ggml_set_2d}} -->
Sets a 2D view of a tensor.
- **Description**: This function is used to create a 2D view of a tensor `a` by modifying its contents based on the values from tensor `b`. It is essential to call this function after initializing the context and ensuring that both tensors are properly allocated. The `nb1` parameter specifies the number of bytes for the first dimension, while the `offset` parameter indicates the starting point in bytes for the view. If the provided tensors are not compatible, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must not be null and should be initialized before calling this function.
    - `a`: A pointer to the `ggml_tensor` structure that will be modified to create a 2D view. This tensor must be properly allocated and initialized.
    - `b`: A pointer to the `ggml_tensor` structure that contains the data to be used for setting the view. This tensor must also be properly allocated and initialized.
    - `nb1`: A size_t value representing the number of bytes for the first dimension of the view. It must be a positive value.
    - `offset`: A size_t value representing the byte offset from which to start the view. It must be a non-negative value.
- **Output**: Returns a pointer to the modified `ggml_tensor` structure representing the 2D view of tensor `a`. If the operation fails, the behavior is undefined.
- **See also**: [`ggml_set_2d`](../src/ggml.c.driver.md#ggml_set_2d)  (Implementation)


---
### ggml\_set\_2d\_inplace<!-- {{#callable_declaration:ggml_set_2d_inplace}} -->
Sets a 2D view of tensor `b` into tensor `a`.
- **Description**: This function is used to create a 2D view of tensor `b` within tensor `a`, modifying `a` in place. It is essential to ensure that `a` has been properly initialized and allocated before calling this function. The `nb1` parameter specifies the number of bytes to be used for the first dimension of the view, while the `offset` parameter indicates the starting point in `a` where the view will be set. If the dimensions of `b` do not match the specified view in `a`, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must not be null and should be initialized before use.
    - `a`: A pointer to the `ggml_tensor` structure representing the destination tensor. This tensor must be properly initialized and allocated.
    - `b`: A pointer to the `ggml_tensor` structure representing the source tensor. This tensor must also be properly initialized and allocated.
    - `nb1`: A size_t value representing the number of bytes for the first dimension of the view. It must be a valid size that fits within the dimensions of tensor `a`.
    - `offset`: A size_t value representing the byte offset in tensor `a` where the view of tensor `b` will be set. It must be within the bounds of tensor `a`.
- **Output**: Returns a pointer to the modified tensor `a`, which now includes the view of tensor `b`.
- **See also**: [`ggml_set_2d_inplace`](../src/ggml.c.driver.md#ggml_set_2d_inplace)  (Implementation)


---
### ggml\_cpy<!-- {{#callable_declaration:ggml_cpy}} -->
Copies the contents of one tensor to another.
- **Description**: This function is used to copy the data from one tensor to another tensor. It is important to ensure that the destination tensor has been properly allocated and is compatible with the source tensor in terms of dimensions and data type. The function should be called after initializing the context and creating the tensors. If the destination tensor is not properly allocated or if the source tensor is invalid, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. It must not be null.
    - `a`: A pointer to the source `ggml_tensor` from which data will be copied. This tensor must be valid and properly initialized.
    - `b`: A pointer to the destination `ggml_tensor` where the data will be copied. This tensor must be allocated and compatible with the source tensor in terms of dimensions and data type.
- **Output**: Returns a pointer to the destination tensor `b` after the copy operation. If the operation fails, the behavior is undefined.
- **See also**: [`ggml_cpy`](../src/ggml.c.driver.md#ggml_cpy)  (Implementation)


---
### ggml\_cast<!-- {{#callable_declaration:ggml_cast}} -->
Casts a tensor to a specified type.
- **Description**: This function is used to create a new tensor that is a casted version of an existing tensor to a specified data type. It should be called when you need to convert a tensor to a different type for operations that require specific data formats. The function expects that the context (`ctx`) has been properly initialized and that the tensor (`a`) is valid and not null. If the specified type is incompatible with the tensor's data, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. It must not be null.
    - `a`: A pointer to the `ggml_tensor` that is to be cast. This tensor must be valid and not null.
    - `type`: An enumeration value of type `ggml_type` that specifies the target data type for the cast. It must be a valid type defined in the `ggml_type` enum.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the casted version of the input tensor. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_cast`](../src/ggml.c.driver.md#ggml_cast)  (Implementation)


---
### ggml\_cont<!-- {{#callable_declaration:ggml_cont}} -->
Makes a tensor contiguous.
- **Description**: This function is used to ensure that the specified tensor is stored in a contiguous block of memory. It should be called when you need to perform operations that require the tensor data to be contiguous, such as certain mathematical computations or optimizations. The input tensor must be valid and properly initialized; otherwise, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` that needs to be made contiguous. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that is contiguous. The original tensor remains unchanged.
- **See also**: [`ggml_cont`](../src/ggml.c.driver.md#ggml_cont)  (Implementation)


---
### ggml\_cont\_1d<!-- {{#callable_declaration:ggml_cont_1d}} -->
Creates a 1-dimensional contiguous tensor.
- **Description**: This function is used to create a new 1-dimensional tensor that is contiguous in memory. It should be called after initializing a `ggml_context` and requires a valid tensor `a` as input. The parameter `ne0` specifies the number of elements in the new tensor. If `a` is null or if `ne0` is less than or equal to zero, the function will not create a tensor and may lead to undefined behavior.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that must be initialized before calling this function. Must not be null.
    - `a`: A pointer to an existing `ggml_tensor` structure that serves as a reference for creating the new tensor. Must not be null.
    - `ne0`: An integer representing the number of elements in the new tensor. Must be greater than zero.
- **Output**: Returns a pointer to the newly created `ggml_tensor` that is contiguous in memory. If the operation fails, it may return null.
- **See also**: [`ggml_cont_1d`](../src/ggml.c.driver.md#ggml_cont_1d)  (Implementation)


---
### ggml\_cont\_2d<!-- {{#callable_declaration:ggml_cont_2d}} -->
Creates a 2D tensor from an existing tensor.
- **Description**: This function is used to create a new 2D tensor that is a continuation of an existing tensor. It should be called after initializing the `ggml_context` and when you have a valid tensor to work with. The dimensions of the new tensor are specified by the parameters `ne0` and `ne1`, which represent the size of the first and second dimensions, respectively. If the provided tensor is not compatible with the specified dimensions, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that must be initialized before calling this function. Must not be null.
    - `a`: A pointer to an existing `ggml_tensor` structure that serves as the base for the new tensor. Must not be null.
    - `ne0`: An integer representing the size of the first dimension of the new tensor. Must be greater than zero.
    - `ne1`: An integer representing the size of the second dimension of the new tensor. Must be greater than zero.
- **Output**: Returns a pointer to the newly created `ggml_tensor` structure representing the 2D tensor. If the operation fails, the return value is undefined.
- **See also**: [`ggml_cont_2d`](../src/ggml.c.driver.md#ggml_cont_2d)  (Implementation)


---
### ggml\_cont\_3d<!-- {{#callable_declaration:ggml_cont_3d}} -->
Creates a 3D tensor from the specified dimensions.
- **Description**: This function is used to create a new 3D tensor with the specified dimensions. It should be called after initializing the `ggml_context`. The function takes an existing tensor as input and reshapes it into a 3D tensor with the specified sizes for each dimension. If the input tensor is not compatible with the specified dimensions, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the existing `ggml_tensor` that will be reshaped into a 3D tensor. Must not be null.
    - `ne0`: The size of the first dimension of the new tensor. Must be a positive integer.
    - `ne1`: The size of the second dimension of the new tensor. Must be a positive integer.
    - `ne2`: The size of the third dimension of the new tensor. Must be a positive integer.
- **Output**: Returns a pointer to the newly created `ggml_tensor` representing the 3D tensor. If the operation fails, the return value is undefined.
- **See also**: [`ggml_cont_3d`](../src/ggml.c.driver.md#ggml_cont_3d)  (Implementation)


---
### ggml\_cont\_4d<!-- {{#callable_declaration:ggml_cont_4d}} -->
Creates a contiguous 4D tensor from the specified dimensions.
- **Description**: This function is used to create a new 4D tensor that is contiguous in memory, based on the provided dimensions. It should be called after initializing a `ggml_context` and requires an existing tensor `a` whose total number of elements matches the product of the specified dimensions (ne0, ne1, ne2, ne3). If the number of elements in `a` does not match the expected total, the function will assert and terminate. The resulting tensor will have the same data type as `a` and will be associated with the operation type `GGML_OP_CONT`.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory for tensor operations. Must not be null.
    - `a`: A pointer to an existing `ggml_tensor` that serves as the source tensor. The number of elements in `a` must equal the product of ne0, ne1, ne2, and ne3.
    - `ne0`: The size of the first dimension of the new tensor. Must be a non-negative integer.
    - `ne1`: The size of the second dimension of the new tensor. Must be a non-negative integer.
    - `ne2`: The size of the third dimension of the new tensor. Must be a non-negative integer.
    - `ne3`: The size of the fourth dimension of the new tensor. Must be a non-negative integer.
- **Output**: Returns a pointer to the newly created `ggml_tensor` that is contiguous in memory. If the input tensor `a` does not have the correct number of elements, the function will terminate the program.
- **See also**: [`ggml_cont_4d`](../src/ggml.c.driver.md#ggml_cont_4d)  (Implementation)


---
### ggml\_reshape<!-- {{#callable_declaration:ggml_reshape}} -->
Reshapes a tensor based on the shape of another tensor.
- **Description**: This function is used to reshape a tensor `a` to match the shape of another tensor `b`. It is important to ensure that the number of elements in both tensors is the same before calling this function, as it will assert this condition. The function can be called after initializing the context and creating the tensors. The resulting tensor will have the same data type as `a` and will be allocated in the context's memory. Note that `b` can be non-contiguous in memory, but its shape must be valid.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory for tensor operations. Must not be null.
    - `a`: A pointer to the source tensor that is to be reshaped. Must not be null and must be contiguous.
    - `b`: A pointer to the tensor that defines the new shape. Must not be null and can be non-contiguous.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the reshaped version of tensor `a`. The new tensor will have the same data type as `a` and will be allocated in the context's memory.
- **See also**: [`ggml_reshape`](../src/ggml.c.driver.md#ggml_reshape)  (Implementation)


---
### ggml\_reshape\_1d<!-- {{#callable_declaration:ggml_reshape_1d}} -->
Reshapes a tensor to a one-dimensional format.
- **Description**: This function is used to reshape an existing tensor into a one-dimensional tensor with a specified number of elements. It must be called with a tensor that is contiguous in memory and has a number of elements equal to the specified size. If the input tensor does not meet these conditions, the behavior is undefined. The resulting tensor will have the same data type as the input tensor and will be allocated in the context provided.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory allocation for tensors. Must not be null.
    - `a`: A pointer to the `ggml_tensor` to be reshaped. This tensor must be contiguous and its number of elements must match `ne0`. Must not be null.
    - `ne0`: An integer representing the desired number of elements in the reshaped tensor. Must be greater than zero.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the reshaped one-dimensional tensor. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_reshape_1d`](../src/ggml.c.driver.md#ggml_reshape_1d)  (Implementation)


---
### ggml\_reshape\_2d<!-- {{#callable_declaration:ggml_reshape_2d}} -->
Reshapes a tensor to a 2D format.
- **Description**: This function is used to reshape an existing tensor into a 2-dimensional tensor with specified dimensions. It must be called with a tensor that is contiguous in memory and whose total number of elements matches the product of the new dimensions. If the input tensor does not meet these conditions, the function will assert and terminate. The resulting tensor will have the same data type as the input tensor and will be allocated in the context provided.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and other resources. Must not be null.
    - `a`: A pointer to the source `ggml_tensor` that is to be reshaped. Must not be null and must be contiguous.
    - `ne0`: The size of the first dimension of the new tensor. Must be a positive integer.
    - `ne1`: The size of the second dimension of the new tensor. Must be a positive integer.
- **Output**: Returns a pointer to the newly created `ggml_tensor` that represents the reshaped tensor. The original tensor remains unchanged.
- **See also**: [`ggml_reshape_2d`](../src/ggml.c.driver.md#ggml_reshape_2d)  (Implementation)


---
### ggml\_reshape\_3d<!-- {{#callable_declaration:ggml_reshape_3d}} -->
Reshapes a 3D tensor.
- **Description**: This function is used to reshape an existing tensor into a new 3D shape defined by the specified dimensions. It must be called with a tensor that is contiguous in memory and whose total number of elements matches the product of the new dimensions. If the input tensor does not meet these conditions, the behavior is undefined. The function returns a new tensor that represents the reshaped version of the input tensor.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and other resources. Must not be null.
    - `a`: A pointer to the input tensor to be reshaped. This tensor must be contiguous in memory and its total number of elements must equal the product of `ne0`, `ne1`, and `ne2`. Must not be null.
    - `ne0`: The size of the first dimension of the new shape. Must be a non-negative integer.
    - `ne1`: The size of the second dimension of the new shape. Must be a non-negative integer.
    - `ne2`: The size of the third dimension of the new shape. Must be a non-negative integer.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the reshaped tensor. The original tensor remains unchanged.
- **See also**: [`ggml_reshape_3d`](../src/ggml.c.driver.md#ggml_reshape_3d)  (Implementation)


---
### ggml\_reshape\_4d<!-- {{#callable_declaration:ggml_reshape_4d}} -->
Reshapes a 4D tensor.
- **Description**: This function is used to reshape an existing tensor into a new 4-dimensional shape defined by the specified dimensions. It must be called with a tensor that is contiguous in memory and whose total number of elements matches the product of the new dimensions. If the input tensor does not meet these conditions, the function will assert and terminate the program. The resulting tensor will have the same data type as the input tensor and will be associated with the same context.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory and other resources for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` to be reshaped. This tensor must be contiguous and its total number of elements must equal the product of `ne0`, `ne1`, `ne2`, and `ne3`. Must not be null.
    - `ne0`: The size of the first dimension of the new tensor. Must be a non-negative integer.
    - `ne1`: The size of the second dimension of the new tensor. Must be a non-negative integer.
    - `ne2`: The size of the third dimension of the new tensor. Must be a non-negative integer.
    - `ne3`: The size of the fourth dimension of the new tensor. Must be a non-negative integer.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the reshaped tensor. The new tensor will have the same data as the original tensor but organized according to the specified dimensions.
- **See also**: [`ggml_reshape_4d`](../src/ggml.c.driver.md#ggml_reshape_4d)  (Implementation)


---
### ggml\_view\_1d<!-- {{#callable_declaration:ggml_view_1d}} -->
Creates a 1D view of a tensor.
- **Description**: This function is used to create a 1D view of an existing tensor, allowing access to a specific portion of its data. It should be called after initializing the `ggml_context` and when the tensor `a` is already created. The `ne0` parameter specifies the number of elements in the view, while the `offset` parameter indicates the starting point in the original tensor's data. If the specified view size exceeds the bounds of the original tensor, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the source tensor from which the view is created. This tensor must be valid and initialized. Must not be null.
    - `ne0`: An integer representing the number of elements in the view. It must be greater than zero and should not exceed the number of elements available in the original tensor from the specified offset.
    - `offset`: A size_t value indicating the byte offset in the original tensor's data from which the view starts. This offset must be within the bounds of the original tensor's data.
- **Output**: Returns a pointer to the newly created 1D view tensor. If the view cannot be created due to invalid parameters, the behavior is undefined.
- **See also**: [`ggml_view_1d`](../src/ggml.c.driver.md#ggml_view_1d)  (Implementation)


---
### ggml\_view\_2d<!-- {{#callable_declaration:ggml_view_2d}} -->
Creates a 2D view of a tensor.
- **Description**: This function is used to create a 2D view of an existing tensor, allowing for operations on a specific sub-region of the tensor's data. It should be called after initializing the `ggml_context` and providing a valid tensor. The parameters `ne0` and `ne1` specify the dimensions of the view, while `nb1` defines the byte stride for the rows. The `offset` parameter indicates the starting point in the original tensor's data from which the view will be created. If the specified dimensions and offsets do not align with the original tensor's structure, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized and valid. Must not be null.
    - `a`: A pointer to the source `ggml_tensor` from which the view will be created. Must not be null.
    - `ne0`: The number of elements in the first dimension of the view. Must be a positive integer.
    - `ne1`: The number of elements in the second dimension of the view. Must be a positive integer.
    - `nb1`: The byte stride for the rows in the view. Must be a positive integer.
    - `offset`: The byte offset from the start of the original tensor's data to begin the view. Must be a non-negative integer.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the 2D view of the original tensor. The returned tensor shares the same data as the original tensor, and any modifications to the view will affect the original tensor.
- **See also**: [`ggml_view_2d`](../src/ggml.c.driver.md#ggml_view_2d)  (Implementation)


---
### ggml\_view\_3d<!-- {{#callable_declaration:ggml_view_3d}} -->
Creates a 3D view of an existing tensor.
- **Description**: This function is used to create a new tensor view that represents a 3D slice of an existing tensor. It is particularly useful when you want to work with a specific portion of a tensor without copying the data. The function must be called with a valid `ggml_context` and an existing tensor. The dimensions specified by `ne0`, `ne1`, and `ne2` must be positive integers, and the `nb1` and `nb2` parameters should represent the byte strides for the second and third dimensions, respectively. The `offset` parameter specifies the starting point in the source tensor's data from which the view will be created. If the parameters do not conform to the expected ranges, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure. Must not be null and should be initialized before calling this function.
    - `a`: A pointer to an existing `ggml_tensor` structure that serves as the source tensor. Must not be null.
    - `ne0`: The size of the first dimension of the view. Must be a positive integer.
    - `ne1`: The size of the second dimension of the view. Must be a positive integer.
    - `ne2`: The size of the third dimension of the view. Must be a positive integer.
    - `nb1`: The byte stride for the second dimension. Must be a positive integer.
    - `nb2`: The byte stride for the third dimension. Must be a positive integer.
    - `offset`: The byte offset in the source tensor from which the view starts. Must be a non-negative integer.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the 3D view of the source tensor. The returned tensor shares the same data as the source tensor.
- **See also**: [`ggml_view_3d`](../src/ggml.c.driver.md#ggml_view_3d)  (Implementation)


---
### ggml\_view\_4d<!-- {{#callable_declaration:ggml_view_4d}} -->
Creates a 4D view of an existing tensor.
- **Description**: This function is used to create a new tensor view that represents a 4-dimensional slice of an existing tensor. It is particularly useful when you want to work with a specific sub-region of a tensor without copying the data. The function must be called with a valid `ggml_context` and a non-null source tensor. The dimensions and strides must be specified correctly to ensure that the view is set up properly. If the specified dimensions or strides are invalid, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure. Must not be null and should be initialized before calling this function.
    - `a`: A pointer to the source `ggml_tensor` from which the view will be created. Must not be null.
    - `ne0`: The size of the first dimension of the view. Must be a positive integer.
    - `ne1`: The size of the second dimension of the view. Must be a positive integer.
    - `ne2`: The size of the third dimension of the view. Must be a positive integer.
    - `ne3`: The size of the fourth dimension of the view. Must be a positive integer.
    - `nb1`: The stride in bytes for the second dimension. Must be a positive integer.
    - `nb2`: The stride in bytes for the third dimension. Must be a positive integer.
    - `nb3`: The stride in bytes for the fourth dimension. Must be a positive integer.
    - `offset`: The byte offset from the start of the source tensor's data. Must be a non-negative integer.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the 4D view of the source tensor. The returned tensor shares the same data as the source tensor, and any modifications to the view will affect the original tensor.
- **See also**: [`ggml_view_4d`](../src/ggml.c.driver.md#ggml_view_4d)  (Implementation)


---
### ggml\_permute<!-- {{#callable_declaration:ggml_permute}} -->
Permutes the dimensions of a tensor.
- **Description**: This function is used to rearrange the dimensions of a tensor according to the specified axes. It is essential to call this function with valid axis indices that are distinct and within the range defined by `GGML_MAX_DIMS`. The function will create a new tensor that reflects the permutation of the original tensor's dimensions. If the provided axes are invalid or not distinct, the behavior is undefined, and the function may assert failure.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure, which manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` to be permuted. Must not be null.
    - `axis0`: An integer representing the first axis to permute. Must be in the range [0, GGML_MAX_DIMS).
    - `axis1`: An integer representing the second axis to permute. Must be in the range [0, GGML_MAX_DIMS) and distinct from axis0.
    - `axis2`: An integer representing the third axis to permute. Must be in the range [0, GGML_MAX_DIMS) and distinct from axis0 and axis1.
    - `axis3`: An integer representing the fourth axis to permute. Must be in the range [0, GGML_MAX_DIMS) and distinct from axis0, axis1, and axis2.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the permuted version of the input tensor. The original tensor remains unchanged.
- **See also**: [`ggml_permute`](../src/ggml.c.driver.md#ggml_permute)  (Implementation)


---
### ggml\_transpose<!-- {{#callable_declaration:ggml_transpose}} -->
Transposes a given tensor.
- **Description**: This function is used to transpose a tensor, effectively swapping its dimensions. It should be called with a valid tensor that has at least two dimensions. The resulting tensor will have its dimensions rearranged, and the original tensor remains unchanged. It is important to ensure that the tensor being transposed is not null.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure, which manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` to be transposed. Must not be null and must have at least two dimensions.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the transposed version of the input tensor. The original tensor remains unchanged.
- **See also**: [`ggml_transpose`](../src/ggml.c.driver.md#ggml_transpose)  (Implementation)


---
### ggml\_get\_rows<!-- {{#callable_declaration:ggml_get_rows}} -->
Retrieves specific rows from a tensor.
- **Description**: This function is used to extract rows from a tensor based on specified indices provided in another tensor. It is essential to ensure that the input tensors are correctly initialized and that the dimensions of the tensors meet the required conditions: the third dimension of the first tensor must match the second dimension of the second tensor, and the second tensor must be of type `GGML_TYPE_I32`. This function should be called after the initialization of the context and the tensors, and it will return a new tensor containing the specified rows.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must not be null and should be properly initialized before calling this function.
    - `a`: A pointer to the source tensor from which rows will be extracted. This tensor must be properly initialized and should have at least three dimensions.
    - `b`: A pointer to a tensor containing the indices of the rows to be extracted. This tensor must be of type `GGML_TYPE_I32`, have a size of at least one, and its second dimension must be equal to one.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the extracted rows. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_get_rows`](../src/ggml.c.driver.md#ggml_get_rows)  (Implementation)


---
### ggml\_get\_rows\_back<!-- {{#callable_declaration:ggml_get_rows_back}} -->
Retrieves rows from a tensor based on specified indices.
- **Description**: This function is used to obtain a new tensor that contains rows from a given tensor, specified by the indices provided in another tensor. It is essential to ensure that the first tensor is a matrix and the second tensor is a vector of indices of type `GGML_TYPE_I32`. The third tensor is used solely to determine the shape of the output tensor, and it must also be a matrix with a compatible number of rows. This function should be called after initializing the context and the tensors involved.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. It must not be null.
    - `a`: A pointer to the source tensor from which rows will be retrieved. This tensor must be a matrix.
    - `b`: A pointer to a tensor containing row indices. This tensor must be a vector of type `GGML_TYPE_I32`.
    - `c`: A pointer to a tensor that defines the shape of the output tensor. This tensor must be a matrix and its first dimension must match the number of rows in tensor `a`.
- **Output**: Returns a pointer to a new tensor containing the specified rows from tensor `a`. The output tensor is allocated in the context provided and has a data type of `GGML_TYPE_F32`.
- **See also**: [`ggml_get_rows_back`](../src/ggml.c.driver.md#ggml_get_rows_back)  (Implementation)


---
### ggml\_diag<!-- {{#callable_declaration:ggml_diag}} -->
Creates a diagonal tensor from a given vector.
- **Description**: This function is used to create a diagonal tensor from a one-dimensional input tensor. The input tensor must have a shape where the second dimension is equal to 1, indicating that it is a vector. The resulting tensor will have a shape where the diagonal elements are populated with the values from the input tensor, while all off-diagonal elements will be set to zero. It is important to ensure that the input tensor meets the dimensionality requirement; otherwise, the function will not behave as expected.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the input vector. This tensor must have a shape of [n, 1], where n is the number of elements in the vector. If the shape is not as expected, the behavior is undefined.
- **Output**: Returns a pointer to a new `ggml_tensor` structure representing the diagonal tensor created from the input vector. The resulting tensor will have a shape of [n, n, d2, d3], where d2 and d3 are the dimensions of the input tensor beyond the first two.
- **See also**: [`ggml_diag`](../src/ggml.c.driver.md#ggml_diag)  (Implementation)


---
### ggml\_diag\_mask\_inf<!-- {{#callable_declaration:ggml_diag_mask_inf}} -->
Sets elements above the diagonal of a tensor to negative infinity.
- **Description**: This function is used to create a diagonal mask for a tensor, setting all elements above the diagonal to negative infinity. It is particularly useful in scenarios where you want to prevent certain values from being considered in computations, such as in attention mechanisms in neural networks. The function should be called with a valid `ggml_context` and a tensor that has been properly initialized. The `n_past` parameter specifies how many past elements should be masked, and it must be a non-negative integer. If `n_past` exceeds the dimensions of the tensor, the function will handle it gracefully by clamping the value.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the state of the library. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure that represents the tensor to be masked. Must not be null.
    - `n_past`: An integer specifying the number of past elements to mask. Must be a non-negative integer.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the masked tensor. The original tensor is not modified.
- **See also**: [`ggml_diag_mask_inf`](../src/ggml.c.driver.md#ggml_diag_mask_inf)  (Implementation)


---
### ggml\_diag\_mask\_inf\_inplace<!-- {{#callable_declaration:ggml_diag_mask_inf_inplace}} -->
Sets elements above the diagonal of a tensor to negative infinity.
- **Description**: This function modifies the input tensor in place by setting all elements above the diagonal to negative infinity, which is useful for masking in certain machine learning applications. It should be called after the tensor has been properly initialized and allocated. The function does not create a new tensor but rather alters the existing one, so the caller should ensure that the tensor is not needed in its original state after this operation.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure that represents the tensor to be modified. Must not be null.
    - `n_past`: An integer specifying the number of past elements to consider for masking. This value should be non-negative.
- **Output**: Returns a pointer to the modified `ggml_tensor`, which is the same as the input tensor `a`.
- **See also**: [`ggml_diag_mask_inf_inplace`](../src/ggml.c.driver.md#ggml_diag_mask_inf_inplace)  (Implementation)


---
### ggml\_diag\_mask\_zero<!-- {{#callable_declaration:ggml_diag_mask_zero}} -->
Sets elements above the diagonal of a tensor to zero.
- **Description**: This function is used to create a diagonal mask for a tensor, setting all elements above the diagonal to zero. It is particularly useful in scenarios where you want to prevent information leakage from future time steps in sequence models. The function should be called with a valid `ggml_context` and a tensor that has been properly initialized. The `n_past` parameter specifies how many past elements should be preserved, and it must be non-negative. If `n_past` exceeds the dimensions of the tensor, the function will handle it gracefully by clamping the value.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the input tensor. Must not be null and should be properly initialized.
    - `n_past`: An integer specifying the number of past elements to preserve. Must be non-negative. If it exceeds the tensor's dimensions, it will be clamped to the maximum valid value.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the masked tensor, with elements above the diagonal set to zero.
- **See also**: [`ggml_diag_mask_zero`](../src/ggml.c.driver.md#ggml_diag_mask_zero)  (Implementation)


---
### ggml\_diag\_mask\_zero\_inplace<!-- {{#callable_declaration:ggml_diag_mask_zero_inplace}} -->
Sets elements above the diagonal of a tensor to zero in place.
- **Description**: This function modifies the input tensor by setting all elements above the diagonal to zero, effectively creating a masked diagonal tensor. It should be called when you need to apply a diagonal mask to a tensor, particularly in scenarios involving sequence modeling or attention mechanisms. The function operates in place, meaning the original tensor is modified directly, and it is important to ensure that the tensor has been properly initialized and allocated before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` that will be modified. This tensor must be properly initialized and allocated. Must not be null.
    - `n_past`: An integer specifying the number of past elements to consider for masking. This value should be non-negative, and the function will handle it appropriately, ensuring that it does not exceed the dimensions of the tensor.
- **Output**: Returns a pointer to the modified `ggml_tensor`, which is the same as the input tensor `a`, reflecting the changes made by the function.
- **See also**: [`ggml_diag_mask_zero_inplace`](../src/ggml.c.driver.md#ggml_diag_mask_zero_inplace)  (Implementation)


---
### ggml\_soft\_max<!-- {{#callable_declaration:ggml_soft_max}} -->
Computes the softmax of a tensor.
- **Description**: This function is used to apply the softmax operation to a given tensor, which is commonly used in machine learning to convert raw scores into probabilities. It should be called after initializing the `ggml_context` and creating the input tensor. The input tensor must be non-null and properly allocated. The function will return a new tensor containing the softmax values, and it is the caller's responsibility to manage the memory of the returned tensor.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that must be initialized before calling this function. It must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the input tensor. This tensor must be properly allocated and initialized. It must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the softmax values of the input tensor. The caller is responsible for managing the memory of this tensor.
- **See also**: [`ggml_soft_max`](../src/ggml.c.driver.md#ggml_soft_max)  (Implementation)


---
### ggml\_soft\_max\_inplace<!-- {{#callable_declaration:ggml_soft_max_inplace}} -->
Performs in-place softmax operation on a tensor.
- **Description**: This function is used to apply the softmax operation directly to the input tensor, modifying it in place. It is typically called when you need to convert raw scores (logits) into probabilities, especially in the context of machine learning models. The input tensor must be properly initialized and allocated before calling this function. If the input tensor is null, the function will handle it gracefully without performing any operation.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` on which the softmax operation will be performed. This tensor must be properly initialized and allocated. If it is null, the function will not perform any operation.
- **Output**: Returns a pointer to the modified `ggml_tensor` after applying the softmax operation. The returned pointer is the same as the input tensor pointer.
- **See also**: [`ggml_soft_max_inplace`](../src/ggml.c.driver.md#ggml_soft_max_inplace)  (Implementation)


---
### ggml\_soft\_max\_ext<!-- {{#callable_declaration:ggml_soft_max_ext}} -->
Computes the softmax of a tensor with optional scaling and masking.
- **Description**: This function is used to compute the softmax of a given tensor, optionally applying a mask and scaling the result. It is particularly useful in scenarios where you want to apply softmax to a tensor while considering additional factors such as a mask for selective activation and a scaling factor to adjust the output. The function should be called with a valid `ggml_context` and tensors for `a` and `mask`. The `scale` parameter allows for adjusting the softmax output, while `max_bias` can be used to introduce a bias in the computation. If `mask` is not provided, it can be set to NULL.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the input tensor for which the softmax operation will be computed. Must not be null.
    - `mask`: A pointer to a tensor that acts as a mask for the softmax operation. This can be NULL if no masking is required.
    - `scale`: A float value that scales the input tensor before applying softmax. Valid range is any finite float value.
    - `max_bias`: A float value that adds a bias to the softmax computation. This can be set to 0.0f if no bias is needed.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the softmax operation. The output tensor will have the same shape as the input tensor `a`.
- **See also**: [`ggml_soft_max_ext`](../src/ggml.c.driver.md#ggml_soft_max_ext)  (Implementation)


---
### ggml\_soft\_max\_ext\_back<!-- {{#callable_declaration:ggml_soft_max_ext_back}} -->
Computes the backward pass of the extended softmax operation.
- **Description**: This function is used to compute the gradients for the extended softmax operation, which is typically called during the backpropagation phase of training a neural network. It should be invoked after the forward pass of the softmax operation has been executed. The function takes two tensors as input: the first tensor represents the output of the softmax operation, while the second tensor represents the gradients of the loss with respect to the output. The function also requires a scaling factor and a maximum bias value, which can be adjusted based on the specific requirements of the model. It is important to ensure that the input tensors are properly initialized and have compatible dimensions.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` structure representing the output of the softmax operation. Must not be null.
    - `b`: A pointer to the `ggml_tensor` structure representing the gradients of the loss with respect to the output tensor. Must not be null.
    - `scale`: A float value used to scale the output gradients. It can be any finite float value.
    - `max_bias`: A float value that represents the maximum bias to be applied. It can be any finite float value.
- **Output**: Returns a pointer to a `ggml_tensor` structure containing the computed gradients for the input tensor `a`. If the operation fails, it may return null.
- **See also**: [`ggml_soft_max_ext_back`](../src/ggml.c.driver.md#ggml_soft_max_ext_back)  (Implementation)


---
### ggml\_soft\_max\_ext\_back\_inplace<!-- {{#callable_declaration:ggml_soft_max_ext_back_inplace}} -->
Performs in-place backward computation for the extended softmax operation.
- **Description**: This function is intended to be called during the backward pass of a neural network training process, specifically after the extended softmax operation has been applied. It computes the gradients with respect to the input tensor `a` based on the gradients stored in tensor `b`. The `scale` parameter allows for scaling the input before the softmax operation, while `max_bias` can be used to adjust the maximum value bias. It is important to ensure that the tensors `a` and `b` are properly initialized and have compatible dimensions before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the input tensor for which gradients are to be computed. Must not be null and should be properly initialized.
    - `b`: A pointer to the tensor containing gradients from the next layer. Must not be null and should have compatible dimensions with `a`.
    - `scale`: A float value used to scale the input tensor `a` before applying the softmax operation. Valid range is any finite float.
    - `max_bias`: A float value that adjusts the maximum value bias in the softmax computation. Valid range is any finite float.
- **Output**: Returns a pointer to the tensor `a`, which now contains the computed gradients. The output tensor is modified in place.
- **See also**: [`ggml_soft_max_ext_back_inplace`](../src/ggml.c.driver.md#ggml_soft_max_ext_back_inplace)  (Implementation)


---
### ggml\_rope<!-- {{#callable_declaration:ggml_rope}} -->
Computes rotary position embeddings.
- **Description**: This function is used to compute rotary position embeddings, which are useful in various machine learning tasks, particularly in transformer models. It should be called after initializing the `ggml_context` and creating the necessary tensors. The function takes two input tensors, `a` and `b`, where `a` typically represents the input data and `b` contains position indices. The number of dimensions for the operation is specified by `n_dims`, and the `mode` parameter allows for different styles of rotary embeddings. It is important to ensure that the input tensors are properly allocated and initialized before calling this function.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the first input tensor, which must be a valid tensor created in the context. Must not be null.
    - `b`: A pointer to the second input tensor, which must also be a valid tensor created in the context. Must not be null.
    - `n_dims`: An integer specifying the number of dimensions for the operation. Valid values are typically between 1 and 4, corresponding to the supported tensor dimensions.
    - `mode`: An integer that specifies the mode of operation for the rotary embeddings. The valid range of values depends on the specific implementation and should be documented in the library's guidelines.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the computed rotary position embeddings. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_rope`](../src/ggml.c.driver.md#ggml_rope)  (Implementation)


---
### ggml\_rope\_inplace<!-- {{#callable_declaration:ggml_rope_inplace}} -->
Applies rotary position embedding in place.
- **Description**: This function modifies the tensor `a` by applying rotary position embedding based on the values in tensor `b`. It is intended for use in scenarios where rotary embeddings are required, such as in transformer models. The function should be called after initializing the context and ensuring that both input tensors are properly allocated and have compatible dimensions. The `n_dims` parameter specifies the number of dimensions to consider, while the `mode` parameter determines the specific behavior of the embedding. It is important to ensure that the tensors are not null and that their dimensions are appropriate for the operation.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the `ggml_tensor` that will be modified in place. Must not be null and should be properly allocated.
    - `b`: A pointer to the `ggml_tensor` containing the positions for the rotary embedding. Must not be null and should have compatible dimensions with `a`.
    - `n_dims`: An integer specifying the number of dimensions to apply the embedding to. Must be a positive integer.
    - `mode`: An integer that specifies the mode of operation for the rotary embedding. Valid values depend on the specific implementation and should be documented elsewhere.
- **Output**: Returns a pointer to the modified `ggml_tensor` `a`, which now contains the applied rotary position embedding.
- **See also**: [`ggml_rope_inplace`](../src/ggml.c.driver.md#ggml_rope_inplace)  (Implementation)


---
### ggml\_rope\_ext<!-- {{#callable_declaration:ggml_rope_ext}} -->
Creates a custom rotary position embedding.
- **Description**: This function is used to generate a custom rotary position embedding based on the provided tensors and parameters. It should be called after initializing the `ggml_context` and before any computations that require the rotary embeddings. The function takes multiple tensors as input, including the context tensor and frequency factors, and it is important to ensure that the dimensions of the input tensors are compatible. If any of the input tensors are null or if the dimensions do not match the expected sizes, the function may return an error or undefined behavior.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the first input tensor, which must be a valid tensor. Must not be null.
    - `b`: A pointer to the second input tensor, which must be a valid tensor. Must not be null.
    - `c`: A pointer to the third input tensor, which can be null if not used. If provided, it must be a valid tensor.
    - `n_dims`: An integer representing the number of dimensions for the tensors. Must be between 1 and 4.
    - `mode`: An integer that specifies the mode of operation. Valid values depend on the specific implementation and should be documented separately.
    - `n_ctx_orig`: An integer representing the original context size. Must be a positive integer.
    - `freq_base`: A floating-point value representing the base frequency. Must be a non-negative float.
    - `freq_scale`: A floating-point value representing the frequency scale. Must be a positive float.
    - `ext_factor`: A floating-point value representing the extension factor. Must be a positive float.
    - `attn_factor`: A floating-point value representing the attention factor. Must be a positive float.
    - `beta_fast`: A floating-point value representing the fast beta parameter. Must be a non-negative float.
    - `beta_slow`: A floating-point value representing the slow beta parameter. Must be a non-negative float.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the computed rotary position embedding. If the operation fails, it may return null.
- **See also**: [`ggml_rope_ext`](../src/ggml.c.driver.md#ggml_rope_ext)  (Implementation)


---
### ggml\_rope\_multi<!-- {{#callable_declaration:ggml_rope_multi}} -->
Creates a multimodal rotary position embedding.
- **Description**: This function is used to generate a multimodal rotary position embedding based on the provided tensors and parameters. It must be called after initializing the context and requires that the input tensors meet specific dimensional and type requirements. The function will assert conditions on the input tensors, such as ensuring that the second tensor is a vector of integers and that the first tensor has the correct shape. If the third tensor is provided, it must be of type float and have a sufficient number of elements. The function modifies the context and returns a new tensor that represents the computed rotary position embedding.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the first input tensor, which must have at least three dimensions. Must not be null.
    - `b`: A pointer to the second input tensor, which must be a vector of integers (type `GGML_TYPE_I32`). Its first dimension must be four times the size of the third dimension of tensor `a`. Must not be null.
    - `c`: An optional pointer to the third input tensor, which must be of type float (type `GGML_TYPE_F32`) and have at least half the number of dimensions specified by `n_dims`. Can be null.
    - `n_dims`: An integer specifying the number of dimensions for the rotary position embedding. Must be a positive integer.
    - `sections`: An array of four integers that specify sections for the embedding. Must not be null.
    - `mode`: An integer that specifies the mode of operation. The least significant bit must be zero.
    - `n_ctx_orig`: An integer representing the original context size. Must be a positive integer.
    - `freq_base`: A float representing the base frequency for the embedding. Must be a non-negative float.
    - `freq_scale`: A float representing the frequency scale for the embedding. Must be a non-negative float.
    - `ext_factor`: A float representing the external factor for the embedding. Must be a non-negative float.
    - `attn_factor`: A float representing the attention factor for the embedding. Must be a non-negative float.
    - `beta_fast`: A float representing the fast beta parameter for the embedding. Must be a non-negative float.
    - `beta_slow`: A float representing the slow beta parameter for the embedding. Must be a non-negative float.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the computed multimodal rotary position embedding. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_rope_multi`](../src/ggml.c.driver.md#ggml_rope_multi)  (Implementation)


---
### ggml\_rope\_ext\_inplace<!-- {{#callable_declaration:ggml_rope_ext_inplace}} -->
Performs an in-place rotary position embedding extension.
- **Description**: This function is used to apply a rotary position embedding extension to a tensor in place, modifying the tensor directly. It should be called after initializing the context and creating the necessary tensors. The function takes multiple parameters that control the behavior of the embedding, including the number of dimensions, the mode of operation, and various frequency and scaling factors. It is important to ensure that the input tensors are properly allocated and compatible in size, as invalid inputs may lead to undefined behavior.
- **Inputs**:
    - `ctx`: A pointer to a `struct ggml_context`, which must be initialized before calling this function. It must not be null.
    - `a`: A pointer to a `struct ggml_tensor` that will be modified in place. This tensor must be properly allocated and compatible with the operation.
    - `b`: A pointer to a `struct ggml_tensor` that provides positional information. This tensor must be properly allocated and compatible with the operation.
    - `c`: A pointer to a `struct ggml_tensor` that contains frequency factors. This tensor is optional and can be null.
    - `n_dims`: An integer representing the number of dimensions of the tensors involved. It must be a positive integer.
    - `mode`: An integer that specifies the mode of operation. Valid values depend on the specific implementation and should be documented separately.
    - `n_ctx_orig`: An integer representing the original context size. It must be a positive integer.
    - `freq_base`: A float representing the base frequency. It should be a non-negative value.
    - `freq_scale`: A float representing the frequency scale. It should be a non-negative value.
    - `ext_factor`: A float representing the extension factor. It should be a non-negative value.
    - `attn_factor`: A float representing the attention factor. It should be a non-negative value.
    - `beta_fast`: A float representing the fast beta parameter. It should be a non-negative value.
    - `beta_slow`: A float representing the slow beta parameter. It should be a non-negative value.
- **Output**: Returns a pointer to the modified `struct ggml_tensor` that was passed as the first argument, reflecting the changes made by the function.
- **See also**: [`ggml_rope_ext_inplace`](../src/ggml.c.driver.md#ggml_rope_ext_inplace)  (Implementation)


---
### ggml\_im2col<!-- {{#callable_declaration:ggml_im2col}} -->
Converts data into a format suitable for convolution.
- **Description**: This function is used to transform input data into a format that allows for efficient convolution operations, particularly in neural networks. It should be called after initializing the `ggml_context` and before performing convolution operations. The function supports both 2D and 1D data formats, determined by the `is_2D` parameter. It is important to ensure that the dimensions of the input tensors are compatible; otherwise, assertions will trigger errors. The output tensor is created in the specified destination type, and the function will allocate memory for it.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the convolution kernel tensor. Must not be null.
    - `b`: A pointer to the input data tensor. Must not be null.
    - `s0`: An integer representing the stride in the first dimension. Must be a positive integer.
    - `s1`: An integer representing the stride in the second dimension. Must be a positive integer.
    - `p0`: An integer representing the padding in the first dimension. Must be a non-negative integer.
    - `p1`: An integer representing the padding in the second dimension. Must be a non-negative integer.
    - `d0`: An integer representing the dilation in the first dimension. Must be a positive integer.
    - `d1`: An integer representing the dilation in the second dimension. Must be a positive integer.
    - `is_2D`: A boolean indicating whether the operation is for 2D data. If true, the function will treat the input as 2D; otherwise, it will treat it as 1D.
    - `dst_type`: An enumeration value specifying the destination tensor type. Must be a valid `ggml_type`.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the transformed data suitable for convolution operations. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_im2col`](../src/ggml.c.driver.md#ggml_im2col)  (Implementation)


---
### ggml\_im2col\_back<!-- {{#callable_declaration:ggml_im2col_back}} -->
Converts data into a format suitable for convolution.
- **Description**: This function is used to transform input data into a format that allows for efficient convolution operations, particularly in neural network implementations. It should be called after initializing the necessary context and tensors. The function takes parameters that define the stride, padding, and dilation for the convolution operation. It is important to ensure that the input tensors are properly allocated and initialized before calling this function. If the input parameters are invalid, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` representing the convolution kernel. Must not be null.
    - `b`: A pointer to the `ggml_tensor` representing the input data. Must not be null.
    - `ne`: A pointer to an array of int64_t that specifies the shape of the output tensor. The caller retains ownership.
    - `s0`: An integer specifying the stride in the first dimension. Must be a positive integer.
    - `s1`: An integer specifying the stride in the second dimension. Must be a positive integer.
    - `p0`: An integer specifying the padding in the first dimension. Must be a non-negative integer.
    - `p1`: An integer specifying the padding in the second dimension. Must be a non-negative integer.
    - `d0`: An integer specifying the dilation in the first dimension. Must be a positive integer.
    - `d1`: An integer specifying the dilation in the second dimension. Must be a positive integer.
    - `is_2D`: A boolean indicating whether the operation is 2D. If true, the function will treat the input as a 2D tensor.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the transformed data suitable for convolution operations. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_im2col_back`](../src/ggml.c.driver.md#ggml_im2col_back)  (Implementation)


---
### ggml\_conv\_1d<!-- {{#callable_declaration:ggml_conv_1d}} -->
Performs a 1D convolution operation.
- **Description**: This function is used to apply a 1D convolution operation on the input tensor `b` using the kernel tensor `a`. It requires a valid `ggml_context` to manage memory and tensor operations. The function takes stride, padding, and dilation parameters to control the convolution behavior. It is important to ensure that the input tensors are properly initialized and compatible in terms of dimensions before calling this function. The output tensor will be a new tensor containing the result of the convolution.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and operations. Must not be null.
    - `a`: A pointer to the convolution kernel tensor. Must not be null and should have compatible dimensions with the input tensor.
    - `b`: A pointer to the input data tensor. Must not be null and should have compatible dimensions with the kernel tensor.
    - `s0`: An integer representing the stride for the convolution operation. Must be a positive integer.
    - `p0`: An integer representing the padding for the convolution operation. Must be a non-negative integer.
    - `d0`: An integer representing the dilation for the convolution operation. Must be a positive integer.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the 1D convolution. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_conv_1d`](../src/ggml.c.driver.md#ggml_conv_1d)  (Implementation)


---
### ggml\_conv\_1d\_ph<!-- {{#callable_declaration:ggml_conv_1d_ph}} -->
Performs a 1D convolution with half padding.
- **Description**: This function is used to apply a 1D convolution operation on a tensor using a specified kernel. It is particularly useful in scenarios where the input tensor needs to be convolved with a kernel tensor, and it automatically applies half padding to the input tensor. The function should be called after initializing the context and creating the input tensors. It is important to ensure that the input tensors are properly allocated and initialized before calling this function. If the input tensors are not compatible in terms of dimensions or types, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` structure representing the convolution kernel. Must not be null and should have a valid shape.
    - `b`: A pointer to the `ggml_tensor` structure representing the input data to be convolved. Must not be null and should have a compatible shape with the kernel.
    - `s`: An integer representing the stride of the convolution. Must be a positive integer.
    - `d`: An integer representing the dilation of the convolution. Must be a non-negative integer.
- **Output**: Returns a pointer to a new `ggml_tensor` structure that contains the result of the convolution operation. The output tensor will have a shape determined by the convolution parameters and the input tensor dimensions.
- **See also**: [`ggml_conv_1d_ph`](../src/ggml.c.driver.md#ggml_conv_1d_ph)  (Implementation)


---
### ggml\_conv\_1d\_dw<!-- {{#callable_declaration:ggml_conv_1d_dw}} -->
Performs a depthwise 1D convolution operation.
- **Description**: This function is used to apply a depthwise convolution operation on a given input tensor using a specified kernel tensor. It is essential to call this function after initializing the `ggml_context` and ensuring that the input tensors are properly allocated and shaped. The function takes stride, padding, and dilation parameters to control the convolution behavior. If the input tensors do not meet the expected dimensions or types, the function may return an error or undefined behavior.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the convolution kernel tensor. This tensor must be properly initialized and should have the correct dimensions for the convolution operation.
    - `b`: A pointer to the input data tensor on which the convolution is to be performed. This tensor must also be properly initialized and should have compatible dimensions with the kernel tensor.
    - `s0`: An integer representing the stride for the convolution operation. This value should be greater than zero.
    - `p0`: An integer representing the padding to be applied to the input tensor. This value should be non-negative.
    - `d0`: An integer representing the dilation for the convolution operation. This value should be greater than zero.
- **Output**: Returns a pointer to a new tensor that contains the result of the depthwise convolution operation. The output tensor will have dimensions determined by the input tensor dimensions, stride, padding, and dilation parameters.
- **See also**: [`ggml_conv_1d_dw`](../src/ggml.c.driver.md#ggml_conv_1d_dw)  (Implementation)


---
### ggml\_conv\_1d\_dw\_ph<!-- {{#callable_declaration:ggml_conv_1d_dw_ph}} -->
Performs a depthwise 1D convolution with half padding.
- **Description**: This function is used to apply a depthwise 1D convolution operation on the input tensor `b` using the kernel tensor `a`. It is important to call this function after initializing the `ggml_context` and creating the tensors. The stride and dilation parameters control the convolution operation's behavior. The function automatically calculates the padding as half of the kernel size, which is useful for maintaining the input size. If the input tensors are not properly initialized or if the dimensions do not match the expected sizes, the function may return an error.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the convolution kernel tensor. This tensor must be properly initialized and should have a compatible shape for the convolution operation. Must not be null.
    - `b`: A pointer to the input data tensor on which the convolution will be applied. This tensor must also be properly initialized and should have a compatible shape. Must not be null.
    - `s0`: An integer representing the stride for the convolution operation. It should be a positive integer.
    - `d0`: An integer representing the dilation for the convolution operation. It should be a positive integer.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the convolution operation. If the operation fails, it may return a null pointer.
- **See also**: [`ggml_conv_1d_dw_ph`](../src/ggml.c.driver.md#ggml_conv_1d_dw_ph)  (Implementation)


---
### ggml\_conv\_transpose\_1d<!-- {{#callable_declaration:ggml_conv_transpose_1d}} -->
Performs a 1D transposed convolution operation.
- **Description**: This function is used to compute the transposed convolution of two tensors, which is commonly used in neural network architectures. It must be called with a valid `ggml_context` and two tensors: the first tensor `a` represents the convolution kernel, while the second tensor `b` represents the input data. The function requires specific parameters for stride, padding, and dilation, which must be set according to the desired output shape. It is important to ensure that the dimensions of the input tensors are compatible; specifically, the second dimension of `a` must match the first dimension of `b`, and the third dimension of `a` must be 1. The function will assert these conditions and will not proceed if they are not met.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the first tensor, which serves as the convolution kernel. It must be a matrix with the third dimension equal to 1.
    - `b`: A pointer to the second tensor, which serves as the input data. It must be a matrix, and its first dimension must match the second dimension of tensor `a`.
    - `s0`: An integer representing the stride for the convolution operation. Valid values depend on the desired output size.
    - `p0`: An integer representing the padding for the convolution operation. This function requires this value to be 0.
    - `d0`: An integer representing the dilation for the convolution operation. This function requires this value to be 1.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the transposed convolution operation. The output tensor's dimensions are determined by the input tensors and the specified parameters.
- **See also**: [`ggml_conv_transpose_1d`](../src/ggml.c.driver.md#ggml_conv_transpose_1d)  (Implementation)


---
### ggml\_conv\_2d<!-- {{#callable_declaration:ggml_conv_2d}} -->
Performs a 2D convolution operation.
- **Description**: This function is used to apply a 2D convolution operation on input data using a specified kernel. It is essential to call this function with valid tensor inputs representing the convolution kernel and the data to be convolved. The function handles various parameters such as stride, padding, and dilation, which control the convolution's behavior. Ensure that the input tensors are properly initialized and compatible in terms of dimensions. The output tensor will contain the result of the convolution operation.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state. Must not be null.
    - `a`: A pointer to a `ggml_tensor` representing the convolution kernel. Must not be null and should have a valid shape.
    - `b`: A pointer to a `ggml_tensor` representing the input data to be convolved. Must not be null and should have a compatible shape with the kernel.
    - `s0`: An integer representing the stride in the first dimension. Must be a positive integer.
    - `s1`: An integer representing the stride in the second dimension. Must be a positive integer.
    - `p0`: An integer representing the padding in the first dimension. Must be a non-negative integer.
    - `p1`: An integer representing the padding in the second dimension. Must be a non-negative integer.
    - `d0`: An integer representing the dilation in the first dimension. Must be a positive integer.
    - `d1`: An integer representing the dilation in the second dimension. Must be a positive integer.
- **Output**: Returns a pointer to a `ggml_tensor` containing the result of the 2D convolution operation. The output tensor will have a shape determined by the input dimensions, kernel size, stride, padding, and dilation parameters.
- **See also**: [`ggml_conv_2d`](../src/ggml.c.driver.md#ggml_conv_2d)  (Implementation)


---
### ggml\_conv\_2d\_sk\_p0<!-- {{#callable_declaration:ggml_conv_2d_sk_p0}} -->
Performs a 2D convolution operation with specific kernel and stride settings.
- **Description**: This function is used to apply a 2D convolution operation on a tensor using a specified kernel tensor. It is particularly useful in machine learning and image processing tasks where convolutional operations are required. The function should be called after initializing the context and creating the input tensors. The kernel tensor must have dimensions that match the expected input shape, and the input tensor must be properly allocated. If the input tensors are not valid or do not meet the expected dimensions, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the convolution kernel. This tensor must have a valid shape and type suitable for convolution operations.
    - `b`: A pointer to a `ggml_tensor` structure representing the input data on which the convolution is to be performed. This tensor must also have a valid shape and type.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the convolution operation. The output tensor will have dimensions determined by the kernel size and the input tensor dimensions.
- **See also**: [`ggml_conv_2d_sk_p0`](../src/ggml.c.driver.md#ggml_conv_2d_sk_p0)  (Implementation)


---
### ggml\_conv\_2d\_s1\_ph<!-- {{#callable_declaration:ggml_conv_2d_s1_ph}} -->
Performs a 2D convolution with a stride of 1 and half padding.
- **Description**: This function is used to apply a 2D convolution operation on the input tensor `b` using the kernel tensor `a`. It is specifically designed to be called when a stride of 1 is required, and it automatically applies half padding to the input tensor. The function should be invoked after initializing the `ggml_context` and creating the tensors. It is important to ensure that the dimensions of the tensors are compatible for convolution; otherwise, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the convolution kernel tensor. This tensor must have at least two dimensions, representing the height and width of the kernel. Must not be null.
    - `b`: A pointer to the input tensor on which the convolution is to be performed. This tensor must have at least two dimensions, and its dimensions must be compatible with the kernel tensor. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the convolution operation. The dimensions of the output tensor will depend on the dimensions of the input tensor and the kernel tensor, taking into account the stride and padding applied.
- **See also**: [`ggml_conv_2d_s1_ph`](../src/ggml.c.driver.md#ggml_conv_2d_s1_ph)  (Implementation)


---
### ggml\_conv\_2d\_dw<!-- {{#callable_declaration:ggml_conv_2d_dw}} -->
Performs a depthwise 2D convolution operation.
- **Description**: This function is used to apply a depthwise 2D convolution to the input tensor using the specified kernel tensor. It is typically called when performing convolutional operations in neural networks, particularly for models that utilize depthwise separable convolutions. The function requires that the input tensors are properly initialized and that the dimensions of the kernel and input tensors are compatible. The stride, padding, and dilation parameters control the convolution's behavior, allowing for flexibility in how the convolution is applied. It is important to ensure that the input tensors are not null and that the dimensions specified by the parameters are valid.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` structure representing the convolution kernel. Must not be null and should have a valid shape for convolution.
    - `b`: A pointer to the `ggml_tensor` structure representing the input data to be convolved. Must not be null and should have compatible dimensions with the kernel.
    - `s0`: An integer representing the stride in the first dimension. Must be a positive integer.
    - `s1`: An integer representing the stride in the second dimension. Must be a positive integer.
    - `p0`: An integer representing the padding in the first dimension. Must be a non-negative integer.
    - `p1`: An integer representing the padding in the second dimension. Must be a non-negative integer.
    - `d0`: An integer representing the dilation in the first dimension. Must be a positive integer.
    - `d1`: An integer representing the dilation in the second dimension. Must be a positive integer.
- **Output**: Returns a pointer to a new `ggml_tensor` structure containing the result of the depthwise 2D convolution operation. The output tensor will have dimensions determined by the input tensor dimensions, kernel size, stride, padding, and dilation parameters.
- **See also**: [`ggml_conv_2d_dw`](../src/ggml.c.driver.md#ggml_conv_2d_dw)  (Implementation)


---
### ggml\_conv\_2d\_dw\_direct<!-- {{#callable_declaration:ggml_conv_2d_dw_direct}} -->
Performs a direct depthwise 2D convolution.
- **Description**: This function is used to apply a depthwise 2D convolution operation on the input tensor `b` using the kernel tensor `a`. It is essential to ensure that the input tensor `a` has a single channel (i.e., its third dimension must be 1) and that the number of channels in `b` matches the number of channels in `a`. The function requires the caller to specify the strides, padding, and dilation for both dimensions of the convolution. It is important to note that the output tensor is allocated within the context provided, and the caller must manage the context's lifecycle appropriately.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the kernel tensor used for the convolution. It must have a shape where the third dimension is 1, indicating a single channel.
    - `b`: A pointer to the input tensor on which the convolution is applied. The number of channels in `b` must match the number of channels in `a`.
    - `stride0`: An integer specifying the stride for the first dimension of the convolution. Must be a positive integer.
    - `stride1`: An integer specifying the stride for the second dimension of the convolution. Must be a positive integer.
    - `pad0`: An integer specifying the padding for the first dimension. Must be a non-negative integer.
    - `pad1`: An integer specifying the padding for the second dimension. Must be a non-negative integer.
    - `dilation0`: An integer specifying the dilation for the first dimension. Must be a positive integer.
    - `dilation1`: An integer specifying the dilation for the second dimension. Must be a positive integer.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the convolution operation. The shape of the output tensor is determined by the input tensor dimensions, the kernel size, strides, padding, and dilation parameters.
- **See also**: [`ggml_conv_2d_dw_direct`](../src/ggml.c.driver.md#ggml_conv_2d_dw_direct)  (Implementation)


---
### ggml\_conv\_transpose\_2d\_p0<!-- {{#callable_declaration:ggml_conv_transpose_2d_p0}} -->
Performs a 2D transposed convolution operation.
- **Description**: This function is used to perform a 2D transposed convolution operation, which is commonly used in neural networks for upsampling feature maps. It should be called after initializing the `ggml_context` and requires two input tensors: the convolution kernel and the input data. The `stride` parameter specifies the step size for the convolution operation. The function will assert that the dimensions of the input tensors are compatible, specifically that the third dimension of the first tensor matches the second dimension of the second tensor. If the input tensors do not meet the required conditions, the function will abort.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the first `ggml_tensor`, representing the convolution kernel. Must not be null.
    - `b`: A pointer to the second `ggml_tensor`, representing the input data. Must not be null.
    - `stride`: An integer specifying the stride of the convolution operation. Must be a positive integer.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the transposed convolution operation. The dimensions of the output tensor are determined based on the input tensor sizes and the specified stride.
- **See also**: [`ggml_conv_transpose_2d_p0`](../src/ggml.c.driver.md#ggml_conv_transpose_2d_p0)  (Implementation)


---
### ggml\_pool\_1d<!-- {{#callable_declaration:ggml_pool_1d}} -->
Performs 1D pooling operation on a tensor.
- **Description**: This function is used to apply a pooling operation (either max or average) on a 1D tensor. It is typically called when you want to downsample the input tensor along one dimension, which is common in various machine learning tasks such as convolutional neural networks. The function requires a valid `ggml_context` and a source tensor, and it is important to ensure that the source tensor is properly initialized and has the correct dimensions. The kernel size, stride, and padding parameters must be specified, and they should be positive integers. If the parameters are invalid, the function may not behave as expected.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the input tensor on which the pooling operation will be performed. Must not be null and should be properly initialized.
    - `op`: An enumeration value specifying the type of pooling operation to perform (max or average). Must be a valid value from the `ggml_op_pool` enum.
    - `k0`: An integer representing the size of the pooling kernel. Must be a positive integer.
    - `s0`: An integer representing the stride of the pooling operation. Must be a positive integer.
    - `p0`: An integer representing the amount of padding to apply. Must be a non-negative integer.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the pooling operation. The dimensions of the output tensor will depend on the input tensor dimensions and the specified kernel size, stride, and padding.
- **See also**: [`ggml_pool_1d`](../src/ggml.c.driver.md#ggml_pool_1d)  (Implementation)


---
### ggml\_pool\_2d<!-- {{#callable_declaration:ggml_pool_2d}} -->
Performs 2D pooling operation on a tensor.
- **Description**: This function is used to apply a 2D pooling operation, such as max or average pooling, on a given tensor. It is typically called during the forward pass of a neural network to reduce the spatial dimensions of the input tensor while retaining important features. The function requires a valid `ggml_context` and a source tensor, and it expects the kernel size, stride, and padding values to be specified. The pooling operation is defined by the `op` parameter, which can be either max or average pooling. It is important to ensure that the input tensor is properly initialized and that the kernel size and stride values are appropriate for the dimensions of the input tensor.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the input tensor on which the pooling operation will be performed. Must not be null.
    - `op`: An enumeration value specifying the type of pooling operation to perform (e.g., max or average). Valid values are defined in the `ggml_op_pool` enum.
    - `k0`: An integer representing the size of the pooling kernel in the first dimension. Must be a positive integer.
    - `k1`: An integer representing the size of the pooling kernel in the second dimension. Must be a positive integer.
    - `s0`: An integer representing the stride in the first dimension. Must be a positive integer.
    - `s1`: An integer representing the stride in the second dimension. Must be a positive integer.
    - `p0`: A float representing the padding in the first dimension. Can be zero or a positive value.
    - `p1`: A float representing the padding in the second dimension. Can be zero or a positive value.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the pooling operation. The dimensions of the output tensor are determined by the input tensor dimensions, kernel size, stride, and padding values.
- **See also**: [`ggml_pool_2d`](../src/ggml.c.driver.md#ggml_pool_2d)  (Implementation)


---
### ggml\_pool\_2d\_back<!-- {{#callable_declaration:ggml_pool_2d_back}} -->
Computes the backward pooling operation for a 2D tensor.
- **Description**: This function is used to compute the gradients of the input tensor during the backpropagation phase of a neural network, specifically for 2D pooling operations. It should be called after a forward pooling operation has been performed, using the same parameters that were used in the forward pass. The function takes the input tensor and the forward output tensor, along with pooling operation parameters such as kernel size, stride, and padding. It is important to ensure that the input tensors are valid and properly initialized before calling this function.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the input tensor for which gradients are being computed. Must not be null.
    - `af`: A pointer to the forward output tensor from the previous pooling operation. Must not be null.
    - `op`: An enumeration value specifying the pooling operation type (e.g., max or average pooling). Valid values are defined in the `ggml_op_pool` enum.
    - `k0`: An integer representing the kernel size in the first dimension. Must be a positive integer.
    - `k1`: An integer representing the kernel size in the second dimension. Must be a positive integer.
    - `s0`: An integer representing the stride in the first dimension. Must be a positive integer.
    - `s1`: An integer representing the stride in the second dimension. Must be a positive integer.
    - `p0`: A float representing the padding in the first dimension. Can be zero or a positive value.
    - `p1`: A float representing the padding in the second dimension. Can be zero or a positive value.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the computed gradients. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_pool_2d_back`](../src/ggml.c.driver.md#ggml_pool_2d_back)  (Implementation)


---
### ggml\_upscale<!-- {{#callable_declaration:ggml_upscale}} -->
Upscales a tensor by a specified scale factor.
- **Description**: This function is used to increase the dimensions of a tensor by a given scale factor, effectively enlarging the tensor's size. It is important to call this function with a valid `ggml_context` and a properly initialized tensor. The `scale_factor` must be a positive integer, and the `mode` parameter determines the interpolation method used for upscaling. If the input tensor is null or the scale factor is less than or equal to zero, the function will not perform the operation and may return a null pointer.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the `ggml_tensor` to be upscaled. This tensor must be properly initialized and allocated. Must not be null.
    - `scale_factor`: An integer representing the factor by which to upscale the tensor's dimensions. Must be a positive integer.
    - `mode`: An enumeration value of type `ggml_scale_mode` that specifies the interpolation method to use. Valid values are `GGML_SCALE_MODE_NEAREST` for nearest neighbor interpolation and `GGML_SCALE_MODE_BILINEAR` for bilinear interpolation.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the upscaled version of the input tensor. If the operation fails (e.g., due to invalid parameters), it may return null.
- **See also**: [`ggml_upscale`](../src/ggml.c.driver.md#ggml_upscale)  (Implementation)


---
### ggml\_upscale\_ext<!-- {{#callable_declaration:ggml_upscale_ext}} -->
Interpolates a tensor to specified dimensions.
- **Description**: This function is used to upscale a tensor to new dimensions defined by the parameters `ne0`, `ne1`, `ne2`, and `ne3`. It is particularly useful in scenarios where you need to increase the size of a tensor for further processing or analysis. The function should be called with a valid `ggml_context` and a tensor that has been properly initialized. The scaling mode can be specified using the `mode` parameter, which determines the interpolation method. If the specified dimensions are smaller than the original tensor's dimensions, the function will not perform any operation and will return the original tensor.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` to be upscaled. Must not be null and should be a valid tensor.
    - `ne0`: The new size for the first dimension of the tensor. Must be a positive integer.
    - `ne1`: The new size for the second dimension of the tensor. Must be a positive integer.
    - `ne2`: The new size for the third dimension of the tensor. Must be a positive integer.
    - `ne3`: The new size for the fourth dimension of the tensor. Must be a positive integer.
    - `mode`: The scaling mode to use for interpolation, defined by the `ggml_scale_mode` enum. Valid values are `GGML_SCALE_MODE_NEAREST` for nearest neighbor interpolation and `GGML_SCALE_MODE_BILINEAR` for bilinear interpolation.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the upscaled version of the input tensor. If the operation fails, it may return null.
- **See also**: [`ggml_upscale_ext`](../src/ggml.c.driver.md#ggml_upscale_ext)  (Implementation)


---
### ggml\_pad<!-- {{#callable_declaration:ggml_pad}} -->
Pads a tensor with zeros.
- **Description**: This function is used to add padding to a tensor along its dimensions. It is particularly useful when preparing data for operations that require specific input shapes. The function must be called with a valid `ggml_context` and a tensor that has been previously created. The padding values for each dimension are specified by the parameters `p0`, `p1`, `p2`, and `p3`, which represent the amount of padding to add to each respective dimension. If any of these padding values are negative, the function will handle them gracefully by treating them as zero padding.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` to be padded. Must not be null and should be a valid tensor.
    - `p0`: An integer specifying the amount of padding to add to the first dimension. Must be non-negative.
    - `p1`: An integer specifying the amount of padding to add to the second dimension. Must be non-negative.
    - `p2`: An integer specifying the amount of padding to add to the third dimension. Must be non-negative.
    - `p3`: An integer specifying the amount of padding to add to the fourth dimension. Must be non-negative.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the padded version of the input tensor. The new tensor will have the same type as the input tensor, with dimensions increased by the specified padding values.
- **See also**: [`ggml_pad`](../src/ggml.c.driver.md#ggml_pad)  (Implementation)


---
### ggml\_pad\_reflect\_1d<!-- {{#callable_declaration:ggml_pad_reflect_1d}} -->
Pads a tensor with reflected values.
- **Description**: This function is used to pad a 1D tensor by reflecting its values at both ends. It is particularly useful when you want to extend the tensor while maintaining its boundary characteristics. The function must be called with a valid `ggml_context` and a tensor of type `GGML_TYPE_F32`. The padding sizes `p0` and `p1` must be non-negative and less than the size of the tensor's first dimension. If the provided tensor is not contiguous or does not meet the type requirement, the function will assert and terminate.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure, which must not be null and should be initialized before calling this function.
    - `a`: A pointer to the `ggml_tensor` to be padded. This tensor must not be null and must be of type `GGML_TYPE_F32`.
    - `p0`: An integer specifying the number of elements to pad at the beginning of the tensor. Must be non-negative and less than the size of the first dimension of tensor `a`.
    - `p1`: An integer specifying the number of elements to pad at the end of the tensor. Must be non-negative and less than the size of the first dimension of tensor `a`.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the padded tensor. The new tensor will have a size increased by `p0` and `p1` in its first dimension.
- **See also**: [`ggml_pad_reflect_1d`](../src/ggml.c.driver.md#ggml_pad_reflect_1d)  (Implementation)


---
### ggml\_timestep\_embedding<!-- {{#callable_declaration:ggml_timestep_embedding}} -->
Generates a timestep embedding tensor.
- **Description**: This function is used to create a timestep embedding tensor based on the provided timesteps and specified dimensions. It should be called after initializing the `ggml_context`. The function adjusts the dimension to be even if an odd value is provided, ensuring compatibility with the embedding requirements. The resulting tensor will have a shape determined by the number of timesteps and the specified dimension, and it will be initialized with the appropriate operation parameters.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `timesteps`: A pointer to a `ggml_tensor` containing the timesteps. This tensor must be one-dimensional and must not be null.
    - `dim`: An integer representing the desired dimension of the embedding. If this value is odd, it will be incremented by one to ensure it is even.
    - `max_period`: An integer specifying the maximum period for the embedding. This value should be a positive integer.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the timestep embedding. The shape of this tensor will be [N, dim], where N is the number of timesteps provided. If the function fails to allocate memory or if any input parameters are invalid, it may return a null pointer.
- **See also**: [`ggml_timestep_embedding`](../src/ggml.c.driver.md#ggml_timestep_embedding)  (Implementation)


---
### ggml\_argsort<!-- {{#callable_declaration:ggml_argsort}} -->
Sorts the elements of a tensor.
- **Description**: This function is used to sort the elements of a tensor along its first dimension, either in ascending or descending order, as specified by the `order` parameter. It is important to ensure that the input tensor is properly initialized and that its first dimension does not exceed the maximum allowable size. The function will return a new tensor containing the indices that would sort the input tensor, which can be useful for various applications such as ranking or selection tasks.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` to be sorted. The tensor must be properly initialized and its first dimension must be less than or equal to INT32_MAX.
    - `order`: An enumeration value of type `ggml_sort_order` that specifies the sorting order. It can be either `GGML_SORT_ORDER_ASC` for ascending order or `GGML_SORT_ORDER_DESC` for descending order.
- **Output**: Returns a pointer to a new `ggml_tensor` containing the indices that would sort the input tensor. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_argsort`](../src/ggml.c.driver.md#ggml_argsort)  (Implementation)


---
### ggml\_arange<!-- {{#callable_declaration:ggml_arange}} -->
Creates a tensor with a range of values.
- **Description**: This function generates a one-dimensional tensor containing a sequence of floating-point values starting from `start`, incrementing by `step`, and stopping before `stop`. It is essential to ensure that `stop` is greater than `start` to avoid unexpected behavior. The function should be called after initializing the `ggml_context`, and it will allocate memory for the new tensor within the context's memory buffer.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory for tensor operations. Must not be null.
    - `start`: The starting value of the sequence. This can be any floating-point number.
    - `stop`: The end value of the sequence, which must be greater than `start`. If `stop` is less than or equal to `start`, the function will not behave as expected.
    - `step`: The increment between each value in the sequence. This must be a positive floating-point number; if `step` is zero or negative, the function will not produce a valid tensor.
- **Output**: Returns a pointer to a `ggml_tensor` containing the generated range of values. The tensor will have a size determined by the number of steps calculated from the provided parameters.
- **See also**: [`ggml_arange`](../src/ggml.c.driver.md#ggml_arange)  (Implementation)


---
### ggml\_top\_k<!-- {{#callable_declaration:ggml_top_k}} -->
Returns the top k elements from a tensor.
- **Description**: This function is used to retrieve the top k elements from a given tensor along its first dimension. It is essential to ensure that the input tensor has at least k elements; otherwise, the function will assert and terminate. The result is a new tensor that contains the top k elements sorted in descending order. This function should be called after initializing the context and creating the input tensor.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` structure representing the input tensor from which the top k elements will be extracted. The tensor must have at least k elements along its first dimension.
    - `k`: An integer specifying the number of top elements to retrieve. Must be greater than 0 and less than or equal to the number of elements in the first dimension of tensor `a`.
- **Output**: Returns a pointer to a new `ggml_tensor` containing the top k elements from the input tensor, sorted in descending order. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_top_k`](../src/ggml.c.driver.md#ggml_top_k)  (Implementation)


---
### ggml\_flash\_attn\_ext<!-- {{#callable_declaration:ggml_flash_attn_ext}} -->
Computes the extended flash attention mechanism.
- **Description**: This function is designed to compute the extended flash attention mechanism, which is commonly used in transformer models. It should be called after initializing the `ggml_context` and creating the necessary input tensors. The function expects the query (`q`), key (`k`), value (`v`), and an optional mask tensor. The mask tensor, if provided, must be contiguous and appropriately sized to match the requirements of the attention mechanism. The function also takes scaling factors and a logit soft cap, which influence the attention computation. If the `max_bias` is greater than zero, the mask must be provided. The output is a new tensor representing the result of the attention computation.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `q`: A pointer to the query tensor, which must have a compatible shape for the attention computation. Must not be null.
    - `k`: A pointer to the key tensor, which must have a compatible shape for the attention computation. Must not be null.
    - `v`: A pointer to the value tensor, which must have a compatible shape for the attention computation. Must not be null.
    - `mask`: An optional pointer to a mask tensor, which must be contiguous and have the appropriate dimensions. If provided, it must not be null and should meet the padding requirements.
    - `scale`: A float value used to scale the attention scores. It can be any finite float value.
    - `max_bias`: A float value that represents the maximum bias to apply. It must be non-negative.
    - `logit_softcap`: A float value that caps the logits. It can be any finite float value.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the attention computation. The shape of the output tensor is determined by the dimensions of the input tensors.
- **See also**: [`ggml_flash_attn_ext`](../src/ggml.c.driver.md#ggml_flash_attn_ext)  (Implementation)


---
### ggml\_flash\_attn\_ext\_set\_prec<!-- {{#callable_declaration:ggml_flash_attn_ext_set_prec}} -->
Sets the precision for a tensor used in flash attention.
- **Description**: This function is used to configure the precision of a tensor that is involved in flash attention operations. It must be called with a valid tensor that has been initialized for flash attention, as indicated by its operation type. The precision can be set to different levels, which may affect the performance and accuracy of subsequent computations. It is important to ensure that the tensor is not null and that it is of the correct operation type before calling this function.
- **Inputs**:
    - `a`: A pointer to a `ggml_tensor` structure that represents the tensor for which the precision is being set. This tensor must not be null and must have its operation type set to `GGML_OP_FLASH_ATTN_EXT`. If the tensor does not meet these criteria, the behavior of the function is undefined.
    - `prec`: An enumeration value of type `ggml_prec` that specifies the desired precision level. Valid values include `GGML_PREC_DEFAULT` and `GGML_PREC_F32`. The function will set the precision of the tensor according to this value.
- **Output**: None
- **See also**: [`ggml_flash_attn_ext_set_prec`](../src/ggml.c.driver.md#ggml_flash_attn_ext_set_prec)  (Implementation)


---
### ggml\_flash\_attn\_ext\_get\_prec<!-- {{#callable_declaration:ggml_flash_attn_ext_get_prec}} -->
Retrieves the precision of a tensor used in flash attention.
- **Description**: This function is used to obtain the precision setting of a tensor that is specifically configured for flash attention operations. It should be called with a valid tensor that has been initialized and is associated with the `GGML_OP_FLASH_ATTN_EXT` operation. If the provided tensor does not meet this requirement, the behavior is undefined.
- **Inputs**:
    - `a`: A pointer to a `struct ggml_tensor` that represents the tensor for which the precision is being queried. This tensor must not be null and must be associated with the `GGML_OP_FLASH_ATTN_EXT` operation. If the tensor does not meet these criteria, the function's behavior is undefined.
- **Output**: Returns an enumeration value of type `enum ggml_prec` that indicates the precision of the specified tensor. The possible values include `GGML_PREC_DEFAULT` and `GGML_PREC_F32`, among others.
- **See also**: [`ggml_flash_attn_ext_get_prec`](../src/ggml.c.driver.md#ggml_flash_attn_ext_get_prec)  (Implementation)


---
### ggml\_flash\_attn\_back<!-- {{#callable_declaration:ggml_flash_attn_back}} -->
Computes the backward pass of the flash attention mechanism.
- **Description**: This function is intended to be called during the backward pass of a neural network that utilizes flash attention. It computes gradients for the input tensors `q`, `k`, `v`, and `d`, which represent query, key, value, and output tensors, respectively. The function expects that the input tensors have compatible shapes, and it will assert these conditions before proceeding. It is crucial to ensure that the tensors are properly initialized and that their dimensions align as specified in the documentation. The `masked` parameter indicates whether the attention should be masked, which can affect the computation of gradients. If any of the input tensors do not meet the expected shape requirements, the function will abort with an assertion failure.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations. Must not be null.
    - `q`: A pointer to a `ggml_tensor` representing the query tensor. It must have a shape of [D, N, ne2, ne3]. Must not be null.
    - `k`: A pointer to a `ggml_tensor` representing the key tensor. It must have a shape of [D, M, kvne2, ne3]. Must not be null.
    - `v`: A pointer to a `ggml_tensor` representing the value tensor. It must have a shape of [M, D, kvne2, ne3]. Must not be null.
    - `d`: A pointer to a `ggml_tensor` representing the output tensor. It must have a shape of [D, N, ne2, ne3]. Must not be null.
    - `masked`: A boolean value indicating whether the attention should be masked. Valid values are true or false.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the computed gradients for the input tensors. The shape and type of the returned tensor will depend on the input tensors and the operations performed.
- **See also**: [`ggml_flash_attn_back`](../src/ggml.c.driver.md#ggml_flash_attn_back)  (Implementation)


---
### ggml\_ssm\_conv<!-- {{#callable_declaration:ggml_ssm_conv}} -->
Performs a state-space model convolution.
- **Description**: This function is used to perform a convolution operation on a 3D tensor using a specified convolution kernel. It is essential to call this function with a valid context and ensure that the input tensor is 3D and the kernel tensor is a matrix. The function computes the output tensor based on the dimensions of the input tensor and the kernel, and it is expected to be used in scenarios involving state-space models. If the input tensors do not meet the required dimensionality, the function will assert and terminate.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations. Must not be null.
    - `sx`: A pointer to a `ggml_tensor` representing the input tensor, which must be a 3D tensor. Must not be null.
    - `c`: A pointer to a `ggml_tensor` representing the convolution kernel, which must be a matrix. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the convolution operation. The dimensions of the output tensor are determined by the input tensor and the kernel.
- **See also**: [`ggml_ssm_conv`](../src/ggml.c.driver.md#ggml_ssm_conv)  (Implementation)


---
### ggml\_ssm\_scan<!-- {{#callable_declaration:ggml_ssm_scan}} -->
Performs a state-space model scan operation.
- **Description**: This function is used to execute a state-space model scan operation, which is typically part of a larger machine learning or signal processing workflow. It requires that the input tensors are properly initialized and contiguous in memory. The function should be called after the necessary context has been set up and the input tensors have been created. It is important to ensure that the dimensions of the input tensors match the expected shapes, as mismatched dimensions will lead to errors. The function will return a new tensor that contains the result of the scan operation.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `s`: A pointer to a `ggml_tensor` representing the state tensor. Must be contiguous and have the correct dimensions.
    - `x`: A pointer to a `ggml_tensor` representing the input tensor. Must be contiguous and have the same shape as `dt`.
    - `dt`: A pointer to a `ggml_tensor` representing the delta time tensor. Must be contiguous and have the same shape as `x`.
    - `A`: A pointer to a `ggml_tensor` representing the state transition matrix. Must be a matrix and have dimensions compatible with `s`.
    - `B`: A pointer to a `ggml_tensor` representing the input-to-state matrix. Must be a 3D tensor and have dimensions compatible with `x`.
    - `C`: A pointer to a `ggml_tensor` representing the output tensor. Must be a 3D tensor and have dimensions compatible with `B`.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the state-space model scan operation. The returned tensor will have a size equal to the sum of the elements in `x` and `s`.
- **See also**: [`ggml_ssm_scan`](../src/ggml.c.driver.md#ggml_ssm_scan)  (Implementation)


---
### ggml\_win\_part<!-- {{#callable_declaration:ggml_win_part}} -->
Partitions a tensor into non-overlapping windows.
- **Description**: This function is used to partition a tensor into smaller, non-overlapping windows, which can be useful for various machine learning tasks such as processing images or sequences. It must be called with a tensor that has a specific shape, where the fourth dimension is equal to 1 and the tensor type is `GGML_TYPE_F32`. The function calculates the necessary padding and creates a new tensor that represents the partitioned windows. It is important to ensure that the input tensor meets these requirements to avoid unexpected behavior.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the input tensor to be partitioned. This tensor must have a shape where the fourth dimension is 1 and its type must be `GGML_TYPE_F32`. Must not be null.
    - `w`: An integer representing the width of the windows to create. Must be a positive integer.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the partitioned windows. The shape of the output tensor will depend on the input tensor's dimensions and the specified window width.
- **See also**: [`ggml_win_part`](../src/ggml.c.driver.md#ggml_win_part)  (Implementation)


---
### ggml\_win\_unpart<!-- {{#callable_declaration:ggml_win_unpart}} -->
Reverses the partitioning of a tensor into non-overlapping windows.
- **Description**: This function is used to reverse the effect of partitioning a tensor into non-overlapping windows, which is useful in various machine learning tasks. It must be called with a tensor that has been previously partitioned using the `ggml_win_part` function. The parameters `w0` and `h0` specify the original dimensions of the tensor before partitioning, while `w` indicates the width of the windows. The function will return a new tensor that combines the windows back into the original shape. If the input tensor is not of type `GGML_TYPE_F32`, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` that represents the partitioned tensor. Must not be null and must be of type `GGML_TYPE_F32`.
    - `w0`: An integer representing the original width of the tensor before partitioning. Must be a positive integer.
    - `h0`: An integer representing the original height of the tensor before partitioning. Must be a positive integer.
    - `w`: An integer representing the width of the windows used during partitioning. Must be a positive integer.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the unpartitioned data. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_win_unpart`](../src/ggml.c.driver.md#ggml_win_unpart)  (Implementation)


---
### ggml\_unary<!-- {{#callable_declaration:ggml_unary}} -->
Applies a unary operation to a tensor.
- **Description**: This function is used to perform a specified unary operation on a given tensor. It must be called with a valid `ggml_context` and a non-null tensor. The operation is defined by the `op` parameter, which should be one of the enumerated unary operations. The function will return a new tensor that contains the result of applying the unary operation to the input tensor. If the input tensor is invalid or the operation is not supported, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the input `ggml_tensor` on which the unary operation will be applied. Must not be null.
    - `op`: An enumerated value of type `ggml_unary_op` that specifies the unary operation to perform. Valid values include operations like absolute, negation, and various activation functions.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the unary operation. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_unary`](../src/ggml.c.driver.md#ggml_unary)  (Implementation)


---
### ggml\_unary\_inplace<!-- {{#callable_declaration:ggml_unary_inplace}} -->
Applies a unary operation to a tensor in place.
- **Description**: This function is used to apply a specified unary operation to a given tensor directly, modifying the tensor's data without creating a new tensor. It should be called after the tensor has been properly initialized and allocated. The operation is defined by the `op` parameter, which specifies the type of unary operation to perform. It is important to ensure that the tensor is valid and not null before calling this function, as passing an invalid tensor may lead to undefined behavior.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the `ggml_tensor` structure that will be modified in place. Must not be null.
    - `op`: An enumeration value of type `ggml_unary_op` that specifies the unary operation to apply. Valid values include operations like absolute, negation, and various activation functions.
- **Output**: Returns a pointer to the modified `ggml_tensor` that has had the unary operation applied. This is the same tensor that was passed in as the `a` parameter.
- **See also**: [`ggml_unary_inplace`](../src/ggml.c.driver.md#ggml_unary_inplace)  (Implementation)


---
### ggml\_get\_rel\_pos<!-- {{#callable_declaration:ggml_get_rel_pos}} -->
Retrieves a tensor representing relative positional encodings.
- **Description**: This function is used to obtain a tensor that encodes relative positional information based on the input tensor. It should be called when you need to incorporate relative positional encodings into your model, particularly in attention mechanisms. The function requires that the input tensor `a` has a specific shape, and both `qh` and `kh` must be equal, representing the query and key heights respectively. If the preconditions are not met, the function will assert and terminate.
- **Inputs**:
    - `ctx`: A pointer to a `struct ggml_context`, which manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `struct ggml_tensor` that serves as the input tensor. It must have a second dimension size of `2 * MAX(qh, kh) - 1`. Must not be null.
    - `qh`: An integer representing the height of the query. Must be non-negative.
    - `kh`: An integer representing the height of the key. Must be non-negative and equal to `qh`.
- **Output**: Returns a pointer to a new `struct ggml_tensor` that contains the relative positional encodings. The shape of the resulting tensor is determined by the input tensor and the values of `qh` and `kh`.
- **See also**: [`ggml_get_rel_pos`](../src/ggml.c.driver.md#ggml_get_rel_pos)  (Implementation)


---
### ggml\_add\_rel\_pos<!-- {{#callable_declaration:ggml_add_rel_pos}} -->
Adds relative positional encodings to a tensor.
- **Description**: This function is used to incorporate relative positional encodings into a tensor, which is particularly useful in models that require positional information, such as transformers. It should be called after initializing the context and creating the necessary tensors. The function expects the input tensor and the positional tensors to be compatible in terms of dimensions. If any of the input tensors are null, the function will handle this gracefully by returning null.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the input tensor to which the positional encodings will be added. Must not be null.
    - `pw`: A pointer to the tensor representing positional encodings for width. Must not be null.
    - `ph`: A pointer to the tensor representing positional encodings for height. Must not be null.
- **Output**: Returns a pointer to a new tensor that contains the result of adding the relative positional encodings to the input tensor. If any input tensor is null, the function returns null.
- **See also**: [`ggml_add_rel_pos`](../src/ggml.c.driver.md#ggml_add_rel_pos)  (Implementation)


---
### ggml\_add\_rel\_pos\_inplace<!-- {{#callable_declaration:ggml_add_rel_pos_inplace}} -->
Adds relative positional encodings to a tensor in place.
- **Description**: This function modifies the tensor `a` by adding relative positional encodings from tensors `pw` and `ph`. It is intended to be used in scenarios where positional information needs to be incorporated into the tensor data, such as in transformer models. The function should be called after initializing the `ggml_context` and ensuring that the tensors involved are properly allocated and of compatible dimensions. If any of the input tensors are null or incompatible, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must not be null and should be initialized before calling this function.
    - `a`: A pointer to the `ggml_tensor` that will be modified in place. This tensor must not be null and should be properly allocated.
    - `pw`: A pointer to the `ggml_tensor` representing the positional weights. This tensor must not be null and should have compatible dimensions with `a`.
    - `ph`: A pointer to the `ggml_tensor` representing the positional heights. This tensor must not be null and should have compatible dimensions with `a`.
- **Output**: Returns a pointer to the modified `ggml_tensor` `a`, which now includes the added relative positional encodings.
- **See also**: [`ggml_add_rel_pos_inplace`](../src/ggml.c.driver.md#ggml_add_rel_pos_inplace)  (Implementation)


---
### ggml\_rwkv\_wkv6<!-- {{#callable_declaration:ggml_rwkv_wkv6}} -->
Generates a new tensor based on the RWKV WKV6 operation.
- **Description**: This function is used to create a new tensor that results from the RWKV WKV6 operation, which combines multiple input tensors. It must be called with valid, contiguous tensors for `k`, `v`, `r`, `tf`, `td`, and `state`. The function will assert that all input tensors are contiguous and have the correct dimensions. The resulting tensor will have a shape that combines the dimensions of the input tensors, specifically concatenating the token dimensions with the state dimensions. It is important to ensure that the input tensors are properly initialized and allocated before calling this function.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory and state for tensor operations. Must not be null.
    - `k`: A pointer to the `ggml_tensor` representing the key tensor. Must be contiguous and have dimensions [S, H, n_tokens]. Must not be null.
    - `v`: A pointer to the `ggml_tensor` representing the value tensor. Must be contiguous and have dimensions [S, H, n_tokens]. Must not be null.
    - `r`: A pointer to the `ggml_tensor` representing the recurrent tensor. Must be contiguous and have dimensions [S, H, n_tokens]. Must not be null.
    - `tf`: A pointer to the `ggml_tensor` representing the tensor for transformation. Must be contiguous and have dimensions [S, H, n_tokens]. Must not be null.
    - `td`: A pointer to the `ggml_tensor` representing the tensor for decay. Must be contiguous and have dimensions [S, H, n_tokens]. Must not be null.
    - `state`: A pointer to the `ggml_tensor` representing the state tensor. Must be contiguous and have dimensions [S, S, H, n_seqs]. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the RWKV WKV6 operation. The shape of the resulting tensor will be [S * H, n_tokens + S * n_seqs, 1, 1].
- **See also**: [`ggml_rwkv_wkv6`](../src/ggml.c.driver.md#ggml_rwkv_wkv6)  (Implementation)


---
### ggml\_gated\_linear\_attn<!-- {{#callable_declaration:ggml_gated_linear_attn}} -->
Computes gated linear attention.
- **Description**: This function is used to compute gated linear attention, which is a crucial operation in various neural network architectures. It should be called after initializing the `ggml_context` and requires that all input tensors are contiguous in memory. The function takes multiple tensor inputs representing keys, values, queries, and a gating tensor, along with a state tensor and a scaling factor. It is important to ensure that the dimensions of the input tensors are compatible, as the function performs assertions to validate this. The output is a new tensor that encapsulates the result of the gated linear attention operation.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `k`: A pointer to the key tensor. Must be contiguous and have dimensions [S, H, n_tokens]. Must not be null.
    - `v`: A pointer to the value tensor. Must be contiguous and have dimensions [S, H, n_tokens]. Must not be null.
    - `q`: A pointer to the query tensor. Must be contiguous and have dimensions [S, H, n_tokens]. Must not be null.
    - `g`: A pointer to the gating tensor. Must be contiguous and have dimensions [S, H, n_tokens]. Must not be null.
    - `state`: A pointer to the state tensor. Must be contiguous and have dimensions [S, S, H, n_seqs]. Must not be null.
    - `scale`: A float value used to scale the output. This value can be any finite float.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the gated linear attention operation. The dimensions of the output tensor are [S * H, n_tokens + S * n_seqs, 1, 1].
- **See also**: [`ggml_gated_linear_attn`](../src/ggml.c.driver.md#ggml_gated_linear_attn)  (Implementation)


---
### ggml\_rwkv\_wkv7<!-- {{#callable_declaration:ggml_rwkv_wkv7}} -->
Creates a new tensor by concatenating several input tensors.
- **Description**: This function is used to create a new tensor that combines the data from multiple input tensors. It is essential to ensure that all input tensors are contiguous in memory and have compatible dimensions. The function must be called with valid tensor pointers, and it will assert the validity of the input tensors before proceeding. If any of the input tensors are not contiguous or do not meet the expected dimensions, the function will abort. The resulting tensor will have a shape that combines the specified dimensions of the input tensors, allowing for efficient data manipulation in machine learning tasks.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory and state for tensor operations. Must not be null.
    - `r`: A pointer to the tensor `r`, which must be contiguous. Represents one of the input tensors.
    - `w`: A pointer to the tensor `w`, which must be contiguous. Represents another input tensor.
    - `k`: A pointer to the tensor `k`, which must be contiguous. Represents an additional input tensor.
    - `v`: A pointer to the tensor `v`, which must be contiguous. Represents yet another input tensor.
    - `a`: A pointer to the tensor `a`, which must be contiguous. Represents an additional input tensor.
    - `b`: A pointer to the tensor `b`, which must be contiguous. Represents another input tensor.
    - `state`: A pointer to the tensor `state`, which must be contiguous. Represents the state tensor used in the operation.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the concatenated results of the input tensors. The shape of the resulting tensor is determined by the dimensions of the input tensors.
- **See also**: [`ggml_rwkv_wkv7`](../src/ggml.c.driver.md#ggml_rwkv_wkv7)  (Implementation)


---
### ggml\_map\_custom1<!-- {{#callable_declaration:ggml_map_custom1}} -->
Maps a custom operation over a tensor.
- **Description**: This function is used to apply a user-defined operation to each element of a tensor in a parallelized manner. It is particularly useful when you want to perform operations that are not natively supported by the library. The function should be called after initializing the `ggml_context` and creating the tensor. The `n_tasks` parameter allows you to specify the number of parallel tasks to use; if set to `GGML_N_TASKS_MAX`, the maximum number of tasks will be utilized. The `userdata` parameter can be used to pass additional data to the custom operation function. It is important to ensure that the tensor `a` is valid and properly initialized before calling this function.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the `ggml_tensor` structure that will be operated on. This tensor must be valid and properly initialized. Must not be null.
    - `fun`: A function pointer to the custom operation to be applied. This function must match the signature defined by `ggml_custom1_op_t`. Must not be null.
    - `n_tasks`: An integer specifying the number of tasks to use for parallel execution. It can be set to `GGML_N_TASKS_MAX` to use the maximum number of tasks.
    - `userdata`: A pointer to user-defined data that will be passed to the custom operation function. Can be null if not needed.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the results of applying the custom operation to the input tensor. If the operation fails, the return value may be null.
- **See also**: [`ggml_map_custom1`](../src/ggml.c.driver.md#ggml_map_custom1)  (Implementation)


---
### ggml\_map\_custom1\_inplace<!-- {{#callable_declaration:ggml_map_custom1_inplace}} -->
Maps a custom operation to a tensor in place.
- **Description**: This function applies a user-defined operation to each element of the specified tensor in place, modifying the original tensor. It is intended for use when you want to perform a custom operation on a tensor without creating a new tensor. The function should be called after initializing the `ggml_context` and creating the tensor. The `fun` parameter must point to a valid function that adheres to the expected signature for custom operations. If `n_tasks` is set to `GGML_N_TASKS_MAX`, the maximum number of tasks will be utilized for parallel processing.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the `ggml_tensor` that will be modified in place. Must not be null.
    - `fun`: A function pointer of type `ggml_custom1_op_t` that defines the custom operation to apply. Must not be null.
    - `n_tasks`: An integer specifying the number of tasks to use for parallel processing. Can be set to `GGML_N_TASKS_MAX` to use the maximum number of tasks.
    - `userdata`: A pointer to user-defined data that will be passed to the custom operation. Can be null.
- **Output**: Returns a pointer to the modified `ggml_tensor`. If the operation fails, the behavior is undefined, and the caller should ensure that the inputs are valid.
- **See also**: [`ggml_map_custom1_inplace`](../src/ggml.c.driver.md#ggml_map_custom1_inplace)  (Implementation)


---
### ggml\_map\_custom2<!-- {{#callable_declaration:ggml_map_custom2}} -->
Maps a custom operation over two tensors.
- **Description**: This function is used to apply a custom operation defined by the user to two input tensors, producing a new output tensor. It is essential to call this function after initializing the `ggml_context` and creating the input tensors. The operation is performed in parallel across multiple tasks, as specified by the `n_tasks` parameter. If `n_tasks` is set to `GGML_N_TASKS_MAX`, the function will utilize the maximum number of tasks available. The `userdata` parameter can be used to pass additional data to the custom operation function. It is important to ensure that the input tensors are compatible in terms of their dimensions and types, as the function does not perform validation on these aspects.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the first input tensor. Must not be null.
    - `b`: A pointer to the second input tensor. Must not be null.
    - `fun`: A function pointer to the custom operation to be applied. This function must match the signature defined for `ggml_custom2_op_t`.
    - `n_tasks`: An integer specifying the number of tasks to use for parallel execution. Can be set to `GGML_N_TASKS_MAX` to use the maximum number of tasks.
    - `userdata`: A pointer to user-defined data that will be passed to the custom operation function. Can be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of applying the custom operation to the input tensors. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_map_custom2`](../src/ggml.c.driver.md#ggml_map_custom2)  (Implementation)


---
### ggml\_map\_custom2\_inplace<!-- {{#callable_declaration:ggml_map_custom2_inplace}} -->
Maps a custom operation to two tensors in place.
- **Description**: This function applies a user-defined operation to two input tensors, modifying the first tensor directly. It is intended for use when you want to perform a custom operation on two tensors without creating a new tensor for the result. The function should be called after initializing the `ggml_context` and ensuring that both input tensors are valid and compatible in terms of their dimensions. The operation is performed in parallel across multiple tasks, as specified by the `n_tasks` parameter.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the first `ggml_tensor` that will be modified in place. Must not be null and should be compatible with the operation.
    - `b`: A pointer to the second `ggml_tensor` that will be used in the operation. Must not be null and should be compatible with the operation.
    - `fun`: A function pointer to the custom operation to be applied. This function must be defined by the user and should match the expected signature.
    - `n_tasks`: An integer specifying the number of tasks to use for parallel execution. Should be a positive integer or `GGML_N_TASKS_MAX` to use the maximum number of tasks.
    - `userdata`: A pointer to user-defined data that can be passed to the custom operation. Can be null if not needed.
- **Output**: Returns a pointer to the modified `ggml_tensor` `a`, which now contains the result of the custom operation.
- **See also**: [`ggml_map_custom2_inplace`](../src/ggml.c.driver.md#ggml_map_custom2_inplace)  (Implementation)


---
### ggml\_map\_custom3<!-- {{#callable_declaration:ggml_map_custom3}} -->
Maps a custom operation across three tensors.
- **Description**: This function is used to apply a custom operation defined by the user across three input tensors. It is particularly useful in scenarios where a specific operation needs to be performed on multiple tensors in a parallelized manner. The function should be called after initializing the `ggml_context` and creating the input tensors. The `n_tasks` parameter allows for specifying the number of tasks to be executed concurrently, with a value of `GGML_N_TASKS_MAX` indicating the maximum number of tasks. The `userdata` parameter can be used to pass additional data to the custom operation function. If any of the input tensors are invalid (e.g., null), the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the first input tensor. Must not be null.
    - `b`: A pointer to the second input tensor. Must not be null.
    - `c`: A pointer to the third input tensor. Must not be null.
    - `fun`: A function pointer to the custom operation to be applied. Must not be null.
    - `n_tasks`: An integer specifying the number of tasks to execute concurrently. Use `GGML_N_TASKS_MAX` for maximum tasks.
    - `userdata`: A pointer to user-defined data that can be passed to the custom operation. Can be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the result of the custom operation applied to the input tensors. If the operation fails, the return value may be null.
- **See also**: [`ggml_map_custom3`](../src/ggml.c.driver.md#ggml_map_custom3)  (Implementation)


---
### ggml\_map\_custom3\_inplace<!-- {{#callable_declaration:ggml_map_custom3_inplace}} -->
Maps a custom operation on three tensors in place.
- **Description**: This function applies a user-defined operation to three input tensors, `a`, `b`, and `c`, and stores the result back in tensor `a`. It is intended for use when you want to perform a custom operation on multiple tensors without creating additional copies of the data. The function should be called after initializing the `ggml_context` and before any computations that depend on the modified tensor. It is important to ensure that the input tensors are compatible in terms of dimensions and types, as the function does not perform validation on these aspects.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the memory and state for tensor operations. Must not be null.
    - `a`: A pointer to the first `ggml_tensor` that will hold the result of the operation. Must not be null.
    - `b`: A pointer to the second `ggml_tensor` used as input for the operation. Must not be null.
    - `c`: A pointer to the third `ggml_tensor` used as input for the operation. Must not be null.
    - `fun`: A function pointer to the custom operation to be applied. This function must be defined by the user and should match the expected signature.
    - `n_tasks`: An integer specifying the number of tasks to be used for parallel execution. Can be set to `GGML_N_TASKS_MAX` to use the maximum number of tasks.
    - `userdata`: A pointer to user-defined data that can be passed to the custom operation. This can be null if not needed.
- **Output**: Returns a pointer to the modified `ggml_tensor` `a`, which now contains the result of the custom operation.
- **See also**: [`ggml_map_custom3_inplace`](../src/ggml.c.driver.md#ggml_map_custom3_inplace)  (Implementation)


---
### ggml\_custom\_4d<!-- {{#callable_declaration:ggml_custom_4d}} -->
Creates a custom 4D tensor.
- **Description**: This function is used to create a new 4D tensor that is defined by a custom operation. It should be called after initializing the `ggml_context`. The function takes the dimensions of the tensor and a custom operation function, which will be applied to the tensor's source tensors. The number of source tensors must not exceed the maximum allowed, and the function will return a pointer to the newly created tensor. If the provided parameters are invalid, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must not be null and should be initialized before calling this function.
    - `type`: An enumeration value of type `ggml_type` that specifies the data type of the tensor. Valid values include various floating-point and integer types.
    - `ne0`: The size of the first dimension of the tensor, must be a non-negative integer.
    - `ne1`: The size of the second dimension of the tensor, must be a non-negative integer.
    - `ne2`: The size of the third dimension of the tensor, must be a non-negative integer.
    - `ne3`: The size of the fourth dimension of the tensor, must be a non-negative integer.
    - `args`: An array of pointers to `ggml_tensor` structures that serve as the source tensors for the custom operation. This array must not be null.
    - `n_args`: The number of source tensors provided in the `args` array, must be less than `GGML_MAX_SRC`.
    - `fun`: A pointer to a custom operation function that defines how the source tensors will be processed. This function must be compatible with the expected signature.
    - `n_tasks`: An integer specifying the number of tasks to be used for the operation. This can be set to `GGML_N_TASKS_MAX` to use the maximum number of tasks.
    - `userdata`: A pointer to user-defined data that can be passed to the custom operation function. This can be null if not needed.
- **Output**: Returns a pointer to the newly created `ggml_tensor` structure representing the custom 4D tensor. If the creation fails, the behavior is undefined.
- **See also**: [`ggml_custom_4d`](../src/ggml.c.driver.md#ggml_custom_4d)  (Implementation)


---
### ggml\_custom\_inplace<!-- {{#callable_declaration:ggml_custom_inplace}} -->
Creates a custom operation on a tensor.
- **Description**: This function is used to apply a custom operation defined by the user on a tensor. It should be called after initializing the `ggml_context` and creating the input tensor. The function takes a tensor `a` and a list of additional tensors `args` that are used as inputs for the custom operation. The number of additional arguments is specified by `n_args`, and the custom operation is defined by the function pointer `fun`. The operation can be executed in parallel by specifying the number of tasks with `n_tasks`. The `userdata` parameter can be used to pass additional data to the custom operation. It is important to ensure that `n_args` does not exceed the maximum allowed value, as defined by the library.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must be initialized before calling this function. Must not be null.
    - `a`: A pointer to the input tensor on which the custom operation will be applied. Must not be null.
    - `args`: An array of pointers to additional tensors that will be used as inputs for the custom operation. The array must contain at least `n_args` elements.
    - `n_args`: An integer specifying the number of additional arguments. Must be less than `GGML_MAX_SRC - 1`.
    - `fun`: A function pointer to the custom operation to be applied. This function must match the expected signature for custom operations.
    - `n_tasks`: An integer specifying the number of tasks to use for parallel execution. Can be set to `GGML_N_TASKS_MAX` to use the maximum number of tasks.
    - `userdata`: A pointer to user-defined data that can be passed to the custom operation. Can be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the result of the custom operation. The returned tensor is a view of the input tensor `a`.
- **See also**: [`ggml_custom_inplace`](../src/ggml.c.driver.md#ggml_custom_inplace)  (Implementation)


---
### ggml\_cross\_entropy\_loss<!-- {{#callable_declaration:ggml_cross_entropy_loss}} -->
Calculates the cross-entropy loss between two tensors.
- **Description**: This function computes the cross-entropy loss between two tensors, which is commonly used in classification tasks to measure the performance of a model. It should be called after ensuring that both input tensors have the same shape, as the function asserts this condition. The first tensor, `a`, represents the predicted logits, while the second tensor, `b`, contains the true labels. The result is a new tensor that holds the computed loss value. It is important to manage the memory context properly, as the function allocates a new tensor in the provided context.
- **Inputs**:
    - `ctx`: A pointer to a `struct ggml_context`, which manages memory for tensor operations. Must not be null.
    - `a`: A pointer to a `struct ggml_tensor` representing the predicted logits. Must not be null and must have the same shape as tensor `b`.
    - `b`: A pointer to a `struct ggml_tensor` representing the true labels. Must not be null and must have the same shape as tensor `a`.
- **Output**: Returns a pointer to a new `struct ggml_tensor` containing the computed cross-entropy loss. The caller is responsible for managing the memory of this tensor.
- **See also**: [`ggml_cross_entropy_loss`](../src/ggml.c.driver.md#ggml_cross_entropy_loss)  (Implementation)


---
### ggml\_cross\_entropy\_loss\_back<!-- {{#callable_declaration:ggml_cross_entropy_loss_back}} -->
Computes the backward pass of the cross-entropy loss.
- **Description**: This function is used to compute the gradients of the cross-entropy loss with respect to the input logits and labels. It should be called after the forward pass of the cross-entropy loss has been computed. The function expects the first tensor to be a scalar representing the logits, while the second and third tensors should have the same shape, representing the labels and the gradients of the loss, respectively. If the input tensors do not meet these requirements, the function may not behave as expected.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` representing the logits. This tensor must be a scalar.
    - `b`: A pointer to a `ggml_tensor` representing the labels. This tensor must have the same shape as the third parameter.
    - `c`: A pointer to a `ggml_tensor` representing the gradients of the cross-entropy loss. This tensor must have the same shape as the second parameter.
- **Output**: Returns a pointer to a new `ggml_tensor` that contains the computed gradients. The caller is responsible for managing the memory of the returned tensor.
- **See also**: [`ggml_cross_entropy_loss_back`](../src/ggml.c.driver.md#ggml_cross_entropy_loss_back)  (Implementation)


---
### ggml\_opt\_step\_adamw<!-- {{#callable_declaration:ggml_opt_step_adamw}} -->
Performs an optimization step using the AdamW algorithm.
- **Description**: This function is used to update the parameters of a model during training using the AdamW optimization algorithm. It should be called after computing the gradients of the loss with respect to the parameters. The function requires that the input tensor `a` is marked as a parameter tensor and that the shapes of `a`, `grad`, `m`, and `v` are the same. The `adamw_params` tensor must be of type `GGML_TYPE_F32` and contain exactly 7 elements, which typically include hyperparameters such as learning rate and weight decay. If any of these conditions are not met, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations. Must not be null.
    - `a`: A pointer to a `ggml_tensor` that represents the parameters to be updated. This tensor must have the flag `GGML_TENSOR_FLAG_PARAM` set and must not be null.
    - `grad`: A pointer to a `ggml_tensor` that contains the gradients of the loss with respect to the parameters. This tensor must have the same shape as `a` and must not be null.
    - `m`: A pointer to a `ggml_tensor` that holds the first moment estimate (mean of gradients). This tensor must have the same shape as `a` and must not be null.
    - `v`: A pointer to a `ggml_tensor` that holds the second moment estimate (variance of gradients). This tensor must have the same shape as `a` and must not be null.
    - `adamw_params`: A pointer to a `ggml_tensor` that contains the AdamW hyperparameters. This tensor must be of type `GGML_TYPE_F32` and must contain exactly 7 elements. Must not be null.
- **Output**: Returns a pointer to a new `ggml_tensor` that represents the updated parameters after the optimization step.
- **See also**: [`ggml_opt_step_adamw`](../src/ggml.c.driver.md#ggml_opt_step_adamw)  (Implementation)


---
### ggml\_build\_forward\_expand<!-- {{#callable_declaration:ggml_build_forward_expand}} -->
Expands the forward computation graph.
- **Description**: This function is used to expand the forward computation graph by adding a tensor to the specified computation graph. It should be called after initializing the computation graph and before performing any computations. The function assumes that the provided `cgraph` and `tensor` are valid and properly initialized. If either parameter is null, the behavior is undefined.
- **Inputs**:
    - `cgraph`: A pointer to a `struct ggml_cgraph` representing the computation graph. Must not be null.
    - `tensor`: A pointer to a `struct ggml_tensor` that represents the tensor to be added to the graph. Must not be null.
- **Output**: None
- **See also**: [`ggml_build_forward_expand`](../src/ggml.c.driver.md#ggml_build_forward_expand)  (Implementation)


---
### ggml\_build\_backward\_expand<!-- {{#callable_declaration:ggml_build_backward_expand}} -->
Builds the backward graph for gradient computation.
- **Description**: This function is used to prepare the backward computation graph for automatic differentiation. It must be called after the forward graph has been built and before any gradient computations are performed. The function expects a valid `ggml_context` and a `ggml_cgraph` that contains nodes with trainable parameters and loss tensors. If the `grad_accs` parameter is provided, it will be used to accumulate gradients; otherwise, new tensors will be created for loss tensors that require gradient accumulation. The function asserts that there are trainable parameters and loss tensors present in the graph, and it handles the initialization of gradient accumulators.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure, which must not be null and should be properly initialized before calling this function.
    - `cgraph`: A pointer to a `ggml_cgraph` structure that represents the computation graph. It must contain at least one node and valid gradients.
    - `grad_accs`: An array of pointers to `ggml_tensor` structures for gradient accumulation. This can be null, in which case the function will create new tensors for loss gradients.
- **Output**: This function does not return a value and does not mutate any inputs directly. It prepares the `cgraph` for backward computation by setting up necessary gradient accumulators.
- **See also**: [`ggml_build_backward_expand`](../src/ggml.c.driver.md#ggml_build_backward_expand)  (Implementation)


---
### ggml\_new\_graph<!-- {{#callable_declaration:ggml_new_graph}} -->
Creates a new computation graph.
- **Description**: This function is used to create a new computation graph within a specified context. It should be called after initializing the context with `ggml_init()`. The created graph can be used to define and compute tensor operations. The function allocates a default size for the graph, which can be adjusted using the custom graph creation function if needed. If the context is invalid or uninitialized, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `struct ggml_context` that represents the current context. This must not be null and should be properly initialized before calling this function.
- **Output**: Returns a pointer to a `struct ggml_cgraph` representing the newly created computation graph. If the graph creation fails, the return value will be null.
- **See also**: [`ggml_new_graph`](../src/ggml.c.driver.md#ggml_new_graph)  (Implementation)


---
### ggml\_new\_graph\_custom<!-- {{#callable_declaration:ggml_new_graph_custom}} -->
Creates a new computation graph with custom size and gradient tracking.
- **Description**: This function is used to create a new computation graph within a specified context. It allows the user to define the size of the graph and whether to track gradients for automatic differentiation. It is essential to call this function after initializing the context with `ggml_init()`. The size parameter determines the number of nodes in the graph, and if gradient tracking is enabled, additional memory will be allocated for gradient accumulation. If the specified size is zero, the function will return a null pointer. The function may also assert if the memory allocation does not match the expected size.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must not be null and should be initialized before calling this function.
    - `size`: A size_t value representing the number of nodes in the graph. Must be greater than zero; otherwise, the function will return null.
    - `grads`: A boolean value indicating whether to track gradients. If true, additional memory will be allocated for gradient storage.
- **Output**: Returns a pointer to a `ggml_cgraph` structure representing the newly created computation graph. If the allocation fails or the size is zero, it returns null.
- **See also**: [`ggml_new_graph_custom`](../src/ggml.c.driver.md#ggml_new_graph_custom)  (Implementation)


---
### ggml\_graph\_dup<!-- {{#callable_declaration:ggml_graph_dup}} -->
Duplicates a computation graph.
- **Description**: This function is used to create a duplicate of an existing computation graph, which can be useful for scenarios where you want to preserve the original graph while making modifications to the copy. It is important to call this function after initializing the context and creating the original graph. The `force_grads` parameter allows the user to specify whether to include gradients in the duplicated graph, which can be beneficial for certain optimization tasks.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which must not be null and should be initialized before calling this function.
    - `cgraph`: A pointer to the `ggml_cgraph` structure representing the original computation graph to be duplicated. This must not be null.
    - `force_grads`: A boolean value indicating whether to force the inclusion of gradients in the duplicated graph. Valid values are true or false.
- **Output**: Returns a pointer to a new `ggml_cgraph` structure that represents the duplicated graph. If the duplication fails, the behavior is undefined, and the caller should ensure that the original graph remains intact.
- **See also**: [`ggml_graph_dup`](../src/ggml.c.driver.md#ggml_graph_dup)  (Implementation)


---
### ggml\_graph\_cpy<!-- {{#callable_declaration:ggml_graph_cpy}} -->
Copies the contents of one computation graph to another.
- **Description**: This function is used to duplicate the structure and data of a source computation graph into a destination graph. It is essential to ensure that the destination graph has sufficient allocated size to accommodate the source graph's nodes and leaves. The function should be called after initializing both graphs, and it will assert that the destination graph has enough space for the source's elements. If the source graph contains gradients, the destination graph's gradients will be reset to zero. This function does not return a value.
- **Inputs**:
    - `src`: A pointer to the source `ggml_cgraph` structure that contains the graph to be copied. Must not be null.
    - `dst`: A pointer to the destination `ggml_cgraph` structure where the source graph will be copied. Must not be null and must have sufficient allocated size to hold the source graph's nodes and leaves.
- **Output**: None
- **See also**: [`ggml_graph_cpy`](../src/ggml.c.driver.md#ggml_graph_cpy)  (Implementation)


---
### ggml\_graph\_reset<!-- {{#callable_declaration:ggml_graph_reset}} -->
Resets the computation graph and its gradients.
- **Description**: This function should be called to reset the state of a computation graph before reusing it for new computations. It sets the gradients of all nodes in the graph to zero and initializes the gradients of loss nodes to one. It is important to ensure that the `cgraph` parameter is valid and properly initialized before calling this function. If the `cgraph` is null, the function will simply return without making any changes.
- **Inputs**:
    - `cgraph`: A pointer to a `struct ggml_cgraph` representing the computation graph. Must not be null and should be properly initialized before calling this function.
- **Output**: None
- **See also**: [`ggml_graph_reset`](../src/ggml.c.driver.md#ggml_graph_reset)  (Implementation)


---
### ggml\_graph\_clear<!-- {{#callable_declaration:ggml_graph_clear}} -->
Clears the computation graph.
- **Description**: This function is used to reset the state of a computation graph, which is represented by the `struct ggml_cgraph`. It should be called when you want to clear all nodes and leaves from the graph, effectively preparing it for a new computation. This function does not require any specific preconditions, but it is typically called when the graph is no longer needed or before starting a new computation. The function will reset the internal state of the graph, including the count of nodes and leaves, and clear any visited nodes.
- **Inputs**:
    - `cgraph`: A pointer to a `struct ggml_cgraph` that represents the computation graph to be cleared. This pointer must not be null, and the function assumes that the graph has been properly initialized before calling this function.
- **Output**: None
- **See also**: [`ggml_graph_clear`](../src/ggml.c.driver.md#ggml_graph_clear)  (Implementation)


---
### ggml\_graph\_size<!-- {{#callable_declaration:ggml_graph_size}} -->
Returns the size of the computation graph.
- **Description**: This function retrieves the size of a computation graph represented by the `ggml_cgraph` structure. It should be called after the graph has been created and populated with nodes. The size indicates the number of nodes in the graph, which can be useful for understanding the complexity of the computation being performed. There are no side effects associated with this function.
- **Inputs**:
    - `cgraph`: A pointer to a `ggml_cgraph` structure representing the computation graph. This pointer must not be null and should point to a valid graph that has been initialized and populated with nodes.
- **Output**: Returns an integer representing the size of the computation graph, specifically the number of nodes it contains.
- **See also**: [`ggml_graph_size`](../src/ggml.c.driver.md#ggml_graph_size)  (Implementation)


---
### ggml\_graph\_node<!-- {{#callable_declaration:ggml_graph_node}} -->
Retrieves a node from a computation graph.
- **Description**: This function is used to access a specific node in a computation graph represented by the `ggml_cgraph` structure. It can be called with a positive index to retrieve the corresponding node directly, or with a negative index to access nodes from the end of the list. The function asserts that the index is within valid bounds, ensuring that the caller does not attempt to access an out-of-range node.
- **Inputs**:
    - `cgraph`: A pointer to a `ggml_cgraph` structure representing the computation graph. Must not be null.
    - `i`: An integer index for the node to retrieve. Valid values are from -n_nodes to n_nodes-1, where n_nodes is the total number of nodes in the graph. If i is negative, it accesses nodes from the end of the list.
- **Output**: Returns a pointer to the `ggml_tensor` corresponding to the specified node. If the index is out of bounds, the behavior is undefined.
- **See also**: [`ggml_graph_node`](../src/ggml.c.driver.md#ggml_graph_node)  (Implementation)


---
### ggml\_graph\_nodes<!-- {{#callable_declaration:ggml_graph_nodes}} -->
Returns the nodes of a computation graph.
- **Description**: This function retrieves the array of nodes from a given computation graph. It should be called after the graph has been created and populated with nodes. The returned array allows users to access the individual nodes for further operations or analysis. It is important to ensure that the `cgraph` parameter is valid and properly initialized before calling this function.
- **Inputs**:
    - `cgraph`: A pointer to a `struct ggml_cgraph` representing the computation graph. This parameter must not be null and should point to a valid, initialized graph structure.
- **Output**: Returns a pointer to an array of pointers to `struct ggml_tensor`, representing the nodes in the computation graph. The array may be empty if no nodes have been added to the graph.
- **See also**: [`ggml_graph_nodes`](../src/ggml.c.driver.md#ggml_graph_nodes)  (Implementation)


---
### ggml\_graph\_n\_nodes<!-- {{#callable_declaration:ggml_graph_n_nodes}} -->
Returns the number of nodes in a computation graph.
- **Description**: This function retrieves the number of nodes present in a given computation graph, which is represented by the `struct ggml_cgraph`. It is essential to ensure that the `cgraph` parameter is valid and properly initialized before calling this function. If the `cgraph` is null or uninitialized, the behavior is undefined.
- **Inputs**:
    - `cgraph`: A pointer to a `struct ggml_cgraph` representing the computation graph. This parameter must not be null and should point to a valid, initialized graph structure.
- **Output**: Returns an integer representing the number of nodes in the specified computation graph.
- **See also**: [`ggml_graph_n_nodes`](../src/ggml.c.driver.md#ggml_graph_n_nodes)  (Implementation)


---
### ggml\_graph\_add\_node<!-- {{#callable_declaration:ggml_graph_add_node}} -->
Adds a tensor as a node to a computation graph.
- **Description**: This function is used to add a `tensor` to the specified `cgraph`, which represents a computation graph. It must be called when the graph has been properly initialized and before any computations are performed. The function asserts that the current number of nodes in the graph does not exceed its allocated size, ensuring that there is enough space to add the new node. If the `tensor` is valid and the graph is not full, it will be added as the next node in the graph.
- **Inputs**:
    - `cgraph`: A pointer to a `ggml_cgraph` structure representing the computation graph. Must not be null and should be properly initialized before calling this function.
    - `tensor`: A pointer to a `ggml_tensor` structure that represents the tensor to be added as a node. Must not be null and should be a valid tensor that has been created prior to this call.
- **Output**: None
- **See also**: [`ggml_graph_add_node`](../src/ggml.c.driver.md#ggml_graph_add_node)  (Implementation)


---
### ggml\_graph\_overhead<!-- {{#callable_declaration:ggml_graph_overhead}} -->
Calculates the memory overhead for the computation graph.
- **Description**: This function is used to determine the memory overhead associated with the computation graph in the GGML library. It should be called after initializing the library and before creating or manipulating any graphs. The overhead value can be useful for optimizing memory usage and ensuring that sufficient resources are allocated for graph operations. There are no side effects, and the function does not modify any input parameters.
- **Inputs**: None
- **Output**: Returns the size of the memory overhead in bytes as a `size_t` value.
- **See also**: [`ggml_graph_overhead`](../src/ggml.c.driver.md#ggml_graph_overhead)  (Implementation)


---
### ggml\_graph\_overhead\_custom<!-- {{#callable_declaration:ggml_graph_overhead_custom}} -->
Calculates the overhead size for a custom computation graph.
- **Description**: This function is used to determine the memory overhead required for a custom computation graph, which is essential for managing memory efficiently in machine learning tasks. It should be called when setting up a new graph, particularly when the size of the graph and whether gradients are needed are known. The function takes into account the specified size and whether gradients are to be tracked, returning the total overhead size needed for the graph. It is important to ensure that the size parameter is non-negative, as negative values may lead to undefined behavior.
- **Inputs**:
    - `size`: Specifies the size of the computation graph. Must be a non-negative value; negative values may lead to undefined behavior.
    - `grads`: A boolean indicating whether gradients should be tracked. This parameter does not have specific constraints.
- **Output**: Returns the size of the overhead in bytes required for the computation graph, calculated based on the input parameters.
- **See also**: [`ggml_graph_overhead_custom`](../src/ggml.c.driver.md#ggml_graph_overhead_custom)  (Implementation)


---
### ggml\_graph\_get\_tensor<!-- {{#callable_declaration:ggml_graph_get_tensor}} -->
Retrieves a tensor from the computation graph by its name.
- **Description**: This function is used to obtain a pointer to a `ggml_tensor` from a given computation graph, identified by its name. It searches through both the leaf and node tensors within the graph. The function should be called with a valid `ggml_cgraph` that has been properly initialized. If the specified name does not match any tensor in the graph, the function will return `NULL`. It is important to ensure that the name provided corresponds to an existing tensor to avoid unexpected `NULL` returns.
- **Inputs**:
    - `cgraph`: A pointer to a `ggml_cgraph` structure representing the computation graph. Must not be null and should be properly initialized before calling this function.
    - `name`: A string representing the name of the tensor to retrieve. This string must be null-terminated. If the name does not correspond to any tensor in the graph, the function will return `NULL`.
- **Output**: Returns a pointer to the `ggml_tensor` associated with the specified name if found; otherwise, returns `NULL`.
- **See also**: [`ggml_graph_get_tensor`](../src/ggml.c.driver.md#ggml_graph_get_tensor)  (Implementation)


---
### ggml\_graph\_get\_grad<!-- {{#callable_declaration:ggml_graph_get_grad}} -->
Retrieves the gradient tensor for a specified node in the computation graph.
- **Description**: This function is used to obtain the gradient tensor associated with a specific node in a computation graph. It should be called after the graph has been built and gradients have been computed. If the specified `node` is not found in the graph or if gradients are not available, the function will return `NULL`. Ensure that the `cgraph` parameter is valid and that the `node` is part of the graph before calling this function.
- **Inputs**:
    - `cgraph`: A pointer to a `struct ggml_cgraph` representing the computation graph. Must not be null.
    - `node`: A pointer to a `struct ggml_tensor` representing the node for which the gradient is requested. Must not be null.
- **Output**: Returns a pointer to the gradient tensor associated with the specified `node`, or `NULL` if the gradient is not available.
- **See also**: [`ggml_graph_get_grad`](../src/ggml.c.driver.md#ggml_graph_get_grad)  (Implementation)


---
### ggml\_graph\_get\_grad\_acc<!-- {{#callable_declaration:ggml_graph_get_grad_acc}} -->
Retrieves the accumulated gradient tensor for a specified node in a computation graph.
- **Description**: This function is used to obtain the accumulated gradient tensor associated with a specific node in a computation graph. It should be called after the graph has been built and gradients have been computed. If the specified `node` is not part of the graph or if the graph does not have accumulated gradients available, the function will return `NULL`. It is important to ensure that the `cgraph` and `node` parameters are valid and properly initialized before calling this function.
- **Inputs**:
    - `cgraph`: A pointer to a `ggml_cgraph` structure representing the computation graph. Must not be null.
    - `node`: A pointer to a `ggml_tensor` structure representing the node for which the accumulated gradient is requested. Must not be null.
- **Output**: Returns a pointer to the accumulated gradient tensor associated with the specified node, or `NULL` if the node is not found or if there are no accumulated gradients.
- **See also**: [`ggml_graph_get_grad_acc`](../src/ggml.c.driver.md#ggml_graph_get_grad_acc)  (Implementation)


---
### ggml\_graph\_print<!-- {{#callable_declaration:ggml_graph_print}} -->
Prints the details of a computation graph.
- **Description**: This function is used to output the structure and details of a computation graph represented by the `ggml_cgraph` structure. It should be called after the graph has been constructed and populated with nodes and leafs. The output includes the number of nodes and leafs, along with their respective properties such as dimensions and operation types. There are no side effects, but the function assumes that the `cgraph` parameter is valid and properly initialized.
- **Inputs**:
    - `cgraph`: A pointer to a `ggml_cgraph` structure representing the computation graph. This parameter must not be null and should point to a valid graph that has been initialized and populated with nodes and leafs.
- **Output**: This function does not return a value and does not modify any inputs.
- **See also**: [`ggml_graph_print`](../src/ggml.c.driver.md#ggml_graph_print)  (Implementation)


---
### ggml\_graph\_dump\_dot<!-- {{#callable_declaration:ggml_graph_dump_dot}} -->
Dumps the computation graph in DOT format to a specified file.
- **Description**: This function is used to export a visual representation of a computation graph, which can be useful for debugging or analysis. It should be called after the graph has been constructed and before any computations are performed. The output file will contain a DOT representation of the graph, which can be visualized using graph visualization tools. If the specified filename is invalid or if the graph is empty, the function may fail to create the output file.
- **Inputs**:
    - `gb`: A pointer to the source computation graph structure (`struct ggml_cgraph`). This graph should contain the nodes and edges that represent the computation. Must not be null.
    - `gf`: A pointer to the gradient computation graph structure (`struct ggml_cgraph`). This graph is used to determine which nodes have gradients. Must not be null.
    - `filename`: A string representing the path to the output file where the DOT representation will be saved. The path must be valid and writable.
- **Output**: None
- **See also**: [`ggml_graph_dump_dot`](../src/ggml.c.driver.md#ggml_graph_dump_dot)  (Implementation)


---
### ggml\_log\_set<!-- {{#callable_declaration:ggml_log_set}} -->
Sets a custom logging callback.
- **Description**: This function allows the user to specify a custom logging callback for handling log messages generated by the library. It should be called before any logging occurs, and if a null callback is provided, the default logging behavior will be used, which outputs messages to standard error. The `user_data` parameter can be used to pass additional context to the callback function, and it can be set to null if not needed.
- **Inputs**:
    - `log_callback`: A pointer to a logging callback function that conforms to the `ggml_log_callback` type. If set to null, the default logging behavior will be used.
    - `user_data`: A pointer to user-defined data that will be passed to the logging callback. This can be null if no additional context is needed.
- **Output**: None
- **See also**: [`ggml_log_set`](../src/ggml.c.driver.md#ggml_log_set)  (Implementation)


---
### ggml\_set\_zero<!-- {{#callable_declaration:ggml_set_zero}} -->
Sets all elements of a tensor to zero.
- **Description**: This function is used to reset the values of a tensor to zero. It should be called with a valid `ggml_tensor` that has been properly initialized. If the tensor is empty, the function will return it without making any changes. If the tensor has allocated memory, it will set all its elements to zero. This operation is useful when you want to clear the contents of a tensor before reusing it.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure that represents the tensor to be zeroed. This pointer must not be null and should point to a valid tensor that has been initialized. If the tensor is empty, it will be returned unchanged.
- **Output**: Returns a pointer to the same `ggml_tensor` structure after setting its elements to zero. If the tensor was empty, it returns the original tensor without modification.
- **See also**: [`ggml_set_zero`](../src/ggml.c.driver.md#ggml_set_zero)  (Implementation)


---
### ggml\_quantize\_init<!-- {{#callable_declaration:ggml_quantize_init}} -->
Initializes quantization settings for a specified type.
- **Description**: This function should be called to set up the quantization parameters for the specified data type before performing any quantization operations. It is important to call this function only once for each type, as subsequent calls with the same type will have no effect unless the quantization settings have been freed using `ggml_quantize_free`. The function is thread-safe, allowing it to be called from multiple threads without causing data corruption.
- **Inputs**:
    - `type`: Specifies the quantization type to initialize. Valid values are defined in the `ggml_type` enumeration. The function will handle invalid types by not performing any initialization.
- **Output**: None
- **See also**: [`ggml_quantize_init`](../src/ggml.c.driver.md#ggml_quantize_init)  (Implementation)


---
### ggml\_quantize\_free<!-- {{#callable_declaration:ggml_quantize_free}} -->
Frees resources allocated for quantization.
- **Description**: This function should be called to release any memory and resources that were allocated during the quantization process. It is important to invoke this function at the end of the program to prevent memory leaks. The function is thread-safe, ensuring that it can be called safely from multiple threads.
- **Inputs**: None
- **Output**: None
- **See also**: [`ggml_quantize_free`](../src/ggml.c.driver.md#ggml_quantize_free)  (Implementation)


---
### ggml\_quantize\_requires\_imatrix<!-- {{#callable_declaration:ggml_quantize_requires_imatrix}} -->
Determines if a quantization type requires an importance matrix.
- **Description**: This function is used to check whether a specific quantization type necessitates the use of an importance matrix. It is particularly relevant when preparing for quantization operations, as certain types may not function correctly without this additional data. The function should be called with a valid `ggml_type` enumeration value, and it will return a boolean indicating the requirement. Invalid or unsupported types will return false.
- **Inputs**:
    - `type`: Specifies the quantization type to check. It must be a valid value from the `ggml_type` enumeration. If an invalid type is provided, the function will return false.
- **Output**: Returns true if the specified quantization type requires an importance matrix; otherwise, returns false.
- **See also**: [`ggml_quantize_requires_imatrix`](../src/ggml.c.driver.md#ggml_quantize_requires_imatrix)  (Implementation)


---
### ggml\_quantize\_chunk<!-- {{#callable_declaration:ggml_quantize_chunk}} -->
Quantizes a chunk of data.
- **Description**: This function is used to quantize a specified chunk of floating-point data into a different format, which can be useful for reducing memory usage and improving performance in machine learning tasks. It must be called after initializing the quantization process with `ggml_quantize_init`. The function requires that the `start` index is aligned with the block size of the specified `type` and that it is a multiple of `n_per_row`. If the quantization type requires an importance matrix, the `imatrix` parameter must not be null. The function handles various quantization types and will assert if the input parameters are invalid.
- **Inputs**:
    - `type`: Specifies the quantization type. Must be a valid `ggml_type` enum value.
    - `src`: Pointer to the source data to be quantized. Must not be null.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null.
    - `start`: The starting index in the source data from which to begin quantization. Must be a multiple of the block size for the specified `type`.
    - `nrows`: The number of rows to quantize. Must be greater than zero.
    - `n_per_row`: The number of elements per row. Must be greater than zero.
    - `imatrix`: Pointer to the importance matrix, if required by the quantization type. Can be null if not required.
- **Output**: Returns the size of the quantized data in bytes. The output size should match the expected size based on the number of rows and the row size for the specified quantization type.
- **See also**: [`ggml_quantize_chunk`](../src/ggml.c.driver.md#ggml_quantize_chunk)  (Implementation)


---
### ggml\_get\_type\_traits<!-- {{#callable_declaration:ggml_get_type_traits}} -->
Retrieves the type traits for a specified tensor type.
- **Description**: This function is used to obtain the characteristics of a specific tensor type, such as its name, block size, and whether it is quantized. It should be called with a valid tensor type from the `ggml_type` enumeration. If the provided type is invalid (i.e., not less than `GGML_TYPE_COUNT`), the behavior is undefined.
- **Inputs**:
    - `type`: An enumeration value of type `ggml_type` that specifies the tensor type. Valid values are from `GGML_TYPE_F32` to `GGML_TYPE_COUNT - 1`. Must not be less than 0 and must be less than `GGML_TYPE_COUNT`.
- **Output**: Returns a pointer to a `ggml_type_traits` structure that contains the traits of the specified tensor type. If the type is invalid, the behavior is undefined.
- **See also**: [`ggml_get_type_traits`](../src/ggml.c.driver.md#ggml_get_type_traits)  (Implementation)


---
### ggml\_threadpool\_params\_default<!-- {{#callable_declaration:ggml_threadpool_params_default}} -->
Creates default thread pool parameters.
- **Description**: This function initializes a `ggml_threadpool_params` structure with default values based on the specified number of threads. It is typically used when setting up a thread pool for parallel processing tasks in the GGML library. The function should be called before using the thread pool to ensure that the parameters are correctly configured. The `n_threads` parameter determines how many threads will be utilized, and the function will set other parameters to their default values. It is important to note that the `n_threads` value should be a positive integer, as negative or zero values may lead to undefined behavior.
- **Inputs**:
    - `n_threads`: Specifies the number of threads to be used in the thread pool. Must be a positive integer. If a non-positive value is provided, the behavior is undefined.
- **Output**: Returns a `ggml_threadpool_params` structure populated with default values based on the specified number of threads.
- **See also**: [`ggml_threadpool_params_default`](../src/ggml.c.driver.md#ggml_threadpool_params_default)  (Implementation)


---
### ggml\_threadpool\_params\_init<!-- {{#callable_declaration:ggml_threadpool_params_init}} -->
Initializes thread pool parameters.
- **Description**: This function is used to initialize the parameters for a thread pool, which is essential for managing concurrent execution in multi-threaded applications. It should be called before using the thread pool to ensure that the parameters are set correctly. The function sets the number of threads, default priority, polling level, CPU placement behavior, and the initial paused state. It is important to note that the `n_threads` parameter must be a positive integer, and the function does not perform validation on the `p` pointer, which must be a valid memory address.
- **Inputs**:
    - `p`: A pointer to a `ggml_threadpool_params` structure that will be initialized. Must not be null.
    - `n_threads`: The number of threads to be used in the thread pool. Must be a positive integer.
- **Output**: None
- **See also**: [`ggml_threadpool_params_init`](../src/ggml.c.driver.md#ggml_threadpool_params_init)  (Implementation)


---
### ggml\_threadpool\_params\_match<!-- {{#callable_declaration:ggml_threadpool_params_match}} -->
Compares two thread pool parameter structures for equality.
- **Description**: This function is used to determine if two instances of `ggml_threadpool_params` are equivalent in terms of their configuration. It checks various fields such as the number of threads, priority, polling level, strict CPU placement, and CPU mask. It is important to ensure that both parameters are initialized before calling this function. If either parameter is null, the behavior is undefined.
- **Inputs**:
    - `p0`: Pointer to the first `ggml_threadpool_params` structure. Must not be null.
    - `p1`: Pointer to the second `ggml_threadpool_params` structure. Must not be null.
- **Output**: Returns true if both parameter structures are equal, otherwise returns false.
- **See also**: [`ggml_threadpool_params_match`](../src/ggml.c.driver.md#ggml_threadpool_params_match)  (Implementation)


