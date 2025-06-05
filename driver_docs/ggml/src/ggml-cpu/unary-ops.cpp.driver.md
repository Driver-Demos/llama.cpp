# Purpose
This C++ source code file provides a collection of unary mathematical operations that can be applied to tensors, which are multi-dimensional arrays commonly used in numerical computing and machine learning. The file defines a series of inline functions, each implementing a specific unary operation such as absolute value, sign, negation, step function, hyperbolic tangent, exponential linear unit (ELU), rectified linear unit (ReLU), sigmoid, hard sigmoid, exponential, hard swish, square, square root, sine, cosine, and logarithm. These operations are implemented as static inline functions for efficiency, allowing them to be inlined by the compiler to reduce function call overhead.

The file also includes template functions that facilitate the application of these unary operations to tensors of different data types, such as float32, float16, and bfloat16. The [`vec_unary_op`](#vec_unary_op) template function applies a specified unary operation to each element of a source tensor, converting data types as necessary. The [`apply_unary_op`](#apply_unary_op) and [`unary_op`](#unary_op) template functions manage the application of these operations, ensuring that the source and destination tensors are compatible in terms of shape and data type. The file defines a series of public API functions, each corresponding to a specific unary operation, which are intended to be used externally to perform forward computations on tensors. These functions are prefixed with `ggml_compute_forward_`, indicating their role in a larger framework or library for tensor computations, likely related to machine learning or numerical analysis.
# Imports and Dependencies

---
- `unary-ops.h`


# Functions

---
### op\_abs<!-- {{#callable:op_abs}} -->
The `op_abs` function computes the absolute value of a floating-point number using the `fabsf` function.
- **Inputs**:
    - `x`: A floating-point number for which the absolute value is to be calculated.
- **Control Flow**:
    - The function takes a single floating-point argument `x`.
    - It returns the result of the `fabsf` function applied to `x`, which computes the absolute value of `x`.
- **Output**: A floating-point number representing the absolute value of the input `x`.


---
### op\_sgn<!-- {{#callable:op_sgn}} -->
The `op_sgn` function returns the sign of a given floating-point number.
- **Inputs**:
    - `x`: A floating-point number whose sign is to be determined.
- **Control Flow**:
    - The function checks if the input `x` is greater than 0; if true, it returns 1.0.
    - If `x` is not greater than 0, it checks if `x` is less than 0; if true, it returns -1.0.
    - If neither condition is met, it returns 0.0, indicating that `x` is zero.
- **Output**: A floating-point number: 1.0 if `x` is positive, -1.0 if `x` is negative, and 0.0 if `x` is zero.


---
### op\_neg<!-- {{#callable:op_neg}} -->
The `op_neg` function negates a given floating-point number.
- **Inputs**:
    - `x`: A floating-point number to be negated.
- **Control Flow**:
    - The function takes a single floating-point argument `x`.
    - It returns the negation of `x` by applying the unary minus operator `-` to `x`.
- **Output**: A floating-point number which is the negation of the input `x`.


---
### op\_step<!-- {{#callable:op_step}} -->
The `op_step` function returns 1.0 if the input is greater than 0.0, otherwise it returns 0.0.
- **Inputs**:
    - `x`: A floating-point number to be evaluated by the step function.
- **Control Flow**:
    - The function checks if the input `x` is greater than 0.0.
    - If `x` is greater than 0.0, the function returns 1.0.
    - If `x` is not greater than 0.0, the function returns 0.0.
- **Output**: A floating-point number, either 1.0 or 0.0, based on the evaluation of the input.


---
### op\_tanh<!-- {{#callable:op_tanh}} -->
The `op_tanh` function computes the hyperbolic tangent of a given floating-point number using the `tanhf` function.
- **Inputs**:
    - `x`: A floating-point number for which the hyperbolic tangent is to be calculated.
- **Control Flow**:
    - The function directly calls the `tanhf` function with the input `x` and returns the result.
- **Output**: A floating-point number representing the hyperbolic tangent of the input `x`.


---
### op\_elu<!-- {{#callable:op_elu}} -->
The `op_elu` function computes the Exponential Linear Unit (ELU) activation for a given input value.
- **Inputs**:
    - `x`: A floating-point number representing the input value for which the ELU activation is to be computed.
- **Control Flow**:
    - The function checks if the input `x` is greater than 0.
    - If `x` is greater than 0, it returns `x` as the output.
    - If `x` is not greater than 0, it computes and returns `expm1f(x)`, which is equivalent to `exp(x) - 1`.
- **Output**: A floating-point number representing the ELU activation of the input `x`.


---
### op\_relu<!-- {{#callable:op_relu}} -->
The `op_relu` function implements the Rectified Linear Unit (ReLU) activation function, returning the input value if it is positive, otherwise returning zero.
- **Inputs**:
    - `x`: A floating-point number representing the input to the ReLU function.
- **Control Flow**:
    - The function checks if the input `x` is greater than 0.
    - If `x` is greater than 0, it returns `x`.
    - If `x` is not greater than 0, it returns 0.
- **Output**: A floating-point number which is either the input `x` if it is positive, or 0 if `x` is non-positive.


---
### op\_sigmoid<!-- {{#callable:op_sigmoid}} -->
The `op_sigmoid` function computes the sigmoid of a given floating-point number using the mathematical formula for the sigmoid function.
- **Inputs**:
    - `x`: A floating-point number for which the sigmoid function is to be computed.
- **Control Flow**:
    - The function calculates the exponential of the negation of the input `x` using `expf(-x)`.
    - It adds 1.0 to the result of the exponential calculation.
    - The function then divides 1.0 by the sum obtained in the previous step to compute the sigmoid value.
- **Output**: A floating-point number representing the sigmoid of the input `x`.


---
### op\_hardsigmoid<!-- {{#callable:op_hardsigmoid}} -->
The `op_hardsigmoid` function computes the hard sigmoid of a given float value by scaling and clamping it between 0 and 1.
- **Inputs**:
    - `x`: A float value for which the hard sigmoid is to be computed.
- **Control Flow**:
    - The function first adds 3.0 to the input value `x`.
    - It then divides the result by 6.0 to scale it.
    - The scaled value is clamped to be at least 0.0 using `fmaxf`.
    - The clamped value is further clamped to be at most 1.0 using `fminf`.
    - The final clamped value is returned as the result.
- **Output**: A float value representing the hard sigmoid of the input, clamped between 0.0 and 1.0.


---
### op\_exp<!-- {{#callable:op_exp}} -->
The `op_exp` function computes the exponential of a given floating-point number using the `expf` function.
- **Inputs**:
    - `x`: A floating-point number for which the exponential value is to be calculated.
- **Control Flow**:
    - The function takes a single floating-point argument `x`.
    - It directly returns the result of the `expf` function applied to `x`.
- **Output**: The function returns the exponential value of the input `x` as a floating-point number.


---
### op\_hardswish<!-- {{#callable:op_hardswish}} -->
The `op_hardswish` function computes the hard swish activation for a given float input.
- **Inputs**:
    - `x`: A floating-point number for which the hard swish activation is to be computed.
- **Control Flow**:
    - Calculate the expression `(x + 3.0f) / 6.0f` to shift and scale the input.
    - Clamp the result of the expression between 0.0f and 1.0f using `fmaxf` and `fminf`.
    - Multiply the clamped result by the original input `x` to compute the final hard swish value.
- **Output**: A floating-point number representing the hard swish activation of the input.


---
### op\_sqr<!-- {{#callable:op_sqr}} -->
The `op_sqr` function computes the square of a given floating-point number.
- **Inputs**:
    - `x`: A floating-point number to be squared.
- **Control Flow**:
    - The function takes a single floating-point argument `x`.
    - It returns the result of multiplying `x` by itself, effectively computing the square of `x`.
- **Output**: A floating-point number representing the square of the input `x`.


---
### op\_sqrt<!-- {{#callable:op_sqrt}} -->
The `op_sqrt` function computes the square root of a given floating-point number using the `sqrtf` function.
- **Inputs**:
    - `x`: A floating-point number for which the square root is to be calculated.
- **Control Flow**:
    - The function takes a single floating-point argument `x`.
    - It directly returns the result of the `sqrtf` function applied to `x`.
- **Output**: A floating-point number representing the square root of the input `x`.


---
### op\_sin<!-- {{#callable:op_sin}} -->
The `op_sin` function computes the sine of a given floating-point number using the `sinf` function.
- **Inputs**:
    - `x`: A floating-point number for which the sine value is to be calculated.
- **Control Flow**:
    - The function takes a single floating-point argument `x`.
    - It directly returns the result of the `sinf` function applied to `x`.
- **Output**: The function returns the sine of the input `x` as a floating-point number.


---
### op\_cos<!-- {{#callable:op_cos}} -->
The `op_cos` function computes the cosine of a given floating-point number using the `cosf` function.
- **Inputs**:
    - `x`: A floating-point number for which the cosine value is to be calculated.
- **Control Flow**:
    - The function directly calls the `cosf` function with the input `x` to compute the cosine value.
- **Output**: The function returns the cosine of the input `x` as a floating-point number.


---
### op\_log<!-- {{#callable:op_log}} -->
The `op_log` function computes the natural logarithm of a given floating-point number using the `logf` function.
- **Inputs**:
    - `x`: A floating-point number for which the natural logarithm is to be calculated.
- **Control Flow**:
    - The function directly calls the `logf` function with the input `x` to compute its natural logarithm.
- **Output**: The function returns the natural logarithm of the input `x` as a floating-point number.


---
### vec\_unary\_op<!-- {{#callable:vec_unary_op}} -->
The `vec_unary_op` function applies a specified unary operation to each element of an input array, converting the input and output types as necessary.
- **Inputs**:
    - `n`: The number of elements in the input array `x` to process.
    - `y`: A pointer to the output array where the results of the unary operation will be stored.
    - `x`: A pointer to the input array containing elements to which the unary operation will be applied.
- **Control Flow**:
    - Define type conversion functions `src0_to_f32` and `f32_to_dst` using a type conversion table for the input and output types, respectively.
    - Iterate over each element in the input array `x` from index 0 to `n-1`.
    - For each element, convert the input element to a float using `src0_to_f32`, apply the unary operation `op`, convert the result back to the destination type using `f32_to_dst`, and store it in the output array `y`.
- **Output**: The function does not return a value; it modifies the output array `y` in place with the results of the unary operation.


---
### apply\_unary\_op<!-- {{#callable:apply_unary_op}} -->
The `apply_unary_op` function applies a specified unary operation to each element of a source tensor and stores the result in a destination tensor, ensuring both tensors are contiguous and of the same shape.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` which contains parameters for computation, including threading information.
    - `dst`: A pointer to a `ggml_tensor` which serves as both the destination tensor for the operation and the source tensor's reference.
- **Control Flow**:
    - Retrieve the source tensor `src0` from the destination tensor's source array.
    - Assert that both `src0` and `dst` are contiguous and have the same shape.
    - Initialize local variables for tensor operation using `GGML_TENSOR_UNARY_OP_LOCALS`.
    - Assert that the byte sizes of the tensor elements match the expected sizes for `dst_t` and `src0_t`.
    - Determine the range of indices to process based on the thread parameters using `get_thread_range`.
    - Iterate over the specified range of indices, calculating the multi-dimensional indices for the current element.
    - Compute pointers to the current elements in the destination and source tensors.
    - Apply the unary operation to the current elements using `vec_unary_op`, storing the result in the destination tensor.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place by applying the unary operation to each element.


---
### unary\_op<!-- {{#callable:unary_op}} -->
The `unary_op` function applies a specified unary operation to a tensor, handling different data types and ensuring compatibility between source and destination tensors.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params`, which contains parameters for the computation.
    - `dst`: A pointer to `ggml_tensor`, which is the destination tensor where the result of the unary operation will be stored.
- **Control Flow**:
    - Retrieve the source tensor `src0` from the destination tensor `dst`.
    - Check the data types of `src0` and `dst` to determine the appropriate template instantiation for `apply_unary_op`.
    - If both `src0` and `dst` are of type `GGML_TYPE_F32`, call `apply_unary_op` with `float` types.
    - If both `src0` and `dst` are of type `GGML_TYPE_F16`, call `apply_unary_op` with `ggml_fp16_t` types.
    - If both `src0` and `dst` are of type `GGML_TYPE_BF16`, call `apply_unary_op` with `ggml_bf16_t` types.
    - If `src0` is `GGML_TYPE_BF16` and `dst` is `GGML_TYPE_F32`, call `apply_unary_op` with `ggml_bf16_t` for `src0` and `float` for `dst`.
    - If `src0` is `GGML_TYPE_F16` and `dst` is `GGML_TYPE_F32`, call `apply_unary_op` with `ggml_fp16_t` for `src0` and `float` for `dst`.
    - If none of the above conditions are met, print an error message and abort the program.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place with the result of the unary operation.


---
### ggml\_compute\_forward\_abs<!-- {{#callable:ggml_compute_forward_abs}} -->
The function `ggml_compute_forward_abs` applies the absolute value operation to each element of a tensor using the `unary_op` template function.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params`, which contains parameters for the computation, such as threading information.
    - `dst`: A pointer to `ggml_tensor`, which is the destination tensor where the result of the absolute value operation will be stored.
- **Control Flow**:
    - The function calls the `unary_op` template function with `op_abs` as the operation to be applied.
    - The `unary_op` function determines the data types of the source and destination tensors and calls `apply_unary_op` with the appropriate type conversion.
    - The `apply_unary_op` function iterates over the elements of the source tensor, applies the `op_abs` operation to each element, and stores the result in the destination tensor.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place to contain the absolute values of the elements from the source tensor.


---
### ggml\_compute\_forward\_sgn<!-- {{#callable:ggml_compute_forward_sgn}} -->
The function `ggml_compute_forward_sgn` applies the signum operation to each element of a source tensor and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params`, which contains parameters for the computation, such as threading information.
    - `dst`: A pointer to `ggml_tensor`, which is the destination tensor where the results of the signum operation will be stored.
- **Control Flow**:
    - The function calls `unary_op` with the `op_sgn` operation, passing `params` and `dst` as arguments.
    - Inside `unary_op`, the source tensor is retrieved from `dst->src[0]`.
    - The function checks the data types of the source and destination tensors to determine the appropriate template instantiation for `apply_unary_op`.
    - If the data types are supported, `apply_unary_op` is called to perform the signum operation on each element of the source tensor and store the result in the destination tensor.
    - If the data types are unsupported, an error message is printed and the program aborts.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place by applying the signum operation to each element of the source tensor.


---
### ggml\_compute\_forward\_neg<!-- {{#callable:ggml_compute_forward_neg}} -->
The function `ggml_compute_forward_neg` applies a negation operation to each element of a source tensor and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation, such as threading information.
    - `dst`: A pointer to a `ggml_tensor` structure that serves as both the destination tensor for the operation and the source tensor from which the data is negated.
- **Control Flow**:
    - The function calls the `unary_op` template function with `op_neg` as the operation to be applied.
    - Inside `unary_op`, the function checks the data types of the source and destination tensors to determine the appropriate template specialization of `apply_unary_op`.
    - The `apply_unary_op` function is then called, which iterates over the elements of the source tensor, applies the negation operation, and stores the results in the destination tensor.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place by applying the negation operation to its elements.


---
### ggml\_compute\_forward\_step<!-- {{#callable:ggml_compute_forward_step}} -->
The `ggml_compute_forward_step` function applies a step function to each element of a tensor using the specified compute parameters.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function calls the `unary_op` template function with `op_step` as the operation to be applied.
    - The `unary_op` function determines the data types of the source and destination tensors and calls `apply_unary_op` with the appropriate template parameters.
    - The `apply_unary_op` function iterates over the elements of the source tensor, applies the `op_step` function to each element, and stores the result in the destination tensor.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place by applying the step function to each of its elements.


---
### ggml\_compute\_forward\_tanh<!-- {{#callable:ggml_compute_forward_tanh}} -->
The function `ggml_compute_forward_tanh` applies the hyperbolic tangent operation to each element of a source tensor and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation, such as threading information.
    - `dst`: A pointer to a `ggml_tensor` structure that serves as both the destination tensor for the operation and the source tensor from which data is read.
- **Control Flow**:
    - The function calls the `unary_op` template function with `op_tanh` as the operation to be applied.
    - Inside `unary_op`, the function checks the data types of the source and destination tensors to determine the appropriate template specialization for `apply_unary_op`.
    - The `apply_unary_op` function is then called, which iterates over the elements of the source tensor, applies the `op_tanh` operation, and writes the results to the destination tensor.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place to contain the results of the tanh operation applied to its original data.


---
### ggml\_compute\_forward\_elu<!-- {{#callable:ggml_compute_forward_elu}} -->
The function `ggml_compute_forward_elu` applies the Exponential Linear Unit (ELU) activation function to a tensor using specified computation parameters.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params`, which contains parameters for the computation, such as threading information.
    - `dst`: A pointer to `ggml_tensor`, which is the destination tensor where the result of the ELU operation will be stored.
- **Control Flow**:
    - The function calls `unary_op` with the `op_elu` operation and the provided `params` and `dst` arguments.
    - The `unary_op` function determines the data type of the source and destination tensors and calls `apply_unary_op` with the appropriate type conversion functions.
    - The `apply_unary_op` function iterates over the elements of the source tensor, applies the ELU operation using `vec_unary_op`, and stores the results in the destination tensor.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place to contain the results of the ELU operation.


---
### ggml\_compute\_forward\_relu<!-- {{#callable:ggml_compute_forward_relu}} -->
The `ggml_compute_forward_relu` function applies the ReLU (Rectified Linear Unit) operation to a tensor using the provided compute parameters.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation, such as threading information.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result of the ReLU operation will be stored.
- **Control Flow**:
    - The function calls the `unary_op` template function with `op_relu` as the operation to be applied.
    - The `unary_op` function checks the data types of the source and destination tensors and calls `apply_unary_op` with the appropriate type conversion if the types are supported.
    - If the types are not supported, an error message is printed and the program aborts.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place to contain the result of the ReLU operation.


---
### ggml\_compute\_forward\_sigmoid<!-- {{#callable:ggml_compute_forward_sigmoid}} -->
The `ggml_compute_forward_sigmoid` function applies the sigmoid operation to each element of the input tensor and stores the result in the destination tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` which contains parameters for the computation, such as threading information.
    - `dst`: A pointer to a `ggml_tensor` where the result of the sigmoid operation will be stored; it also contains the source tensor as its first element in the `src` array.
- **Control Flow**:
    - The function calls `unary_op` with the `op_sigmoid` operation, passing the `params` and `dst` arguments.
    - Inside `unary_op`, the source tensor is extracted from `dst->src[0]`.
    - The function checks the data types of the source and destination tensors to determine the appropriate template instantiation for `apply_unary_op`.
    - If the data types are supported, `apply_unary_op` is called to perform the sigmoid operation on each element of the source tensor and store the result in the destination tensor.
    - If the data types are unsupported, an error message is printed and the program aborts.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place to contain the results of the sigmoid operation.


---
### ggml\_compute\_forward\_hardsigmoid<!-- {{#callable:ggml_compute_forward_hardsigmoid}} -->
The function `ggml_compute_forward_hardsigmoid` applies the hard sigmoid operation to each element of the input tensor and stores the result in the destination tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` which contains parameters for the computation, such as threading information.
    - `dst`: A pointer to a `ggml_tensor` which serves as both the input and output tensor for the operation.
- **Control Flow**:
    - The function calls `unary_op` with the `op_hardsigmoid` operation, passing the `params` and `dst` arguments.
    - Inside `unary_op`, the function checks the data types of the source and destination tensors to determine the appropriate template instantiation for `apply_unary_op`.
    - The `apply_unary_op` function is called with the `op_hardsigmoid` operation and the appropriate data types, which applies the hard sigmoid operation to each element of the input tensor and stores the result in the destination tensor.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place to contain the result of the hard sigmoid operation.


---
### ggml\_compute\_forward\_exp<!-- {{#callable:ggml_compute_forward_exp}} -->
The `ggml_compute_forward_exp` function applies the exponential operation to each element of the input tensor and stores the result in the destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation, such as threading information.
    - `dst`: A pointer to a `ggml_tensor` structure that serves as both the input and output tensor for the operation, where the result of the exponential operation will be stored.
- **Control Flow**:
    - The function calls the `unary_op` template function with `op_exp` as the operation to be applied.
    - The `unary_op` function checks the data types of the source and destination tensors and calls `apply_unary_op` with the appropriate type conversion if the types are supported.
    - The `apply_unary_op` function iterates over the elements of the input tensor, applies the `op_exp` operation to each element, and stores the result in the destination tensor.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place to contain the results of the exponential operation.


---
### ggml\_compute\_forward\_hardswish<!-- {{#callable:ggml_compute_forward_hardswish}} -->
The `ggml_compute_forward_hardswish` function applies the hard swish activation function to a tensor using specified computation parameters.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params`, which contains parameters for the computation, such as threading information.
    - `dst`: A pointer to `ggml_tensor`, which is the destination tensor where the result of the hard swish operation will be stored.
- **Control Flow**:
    - The function calls the `unary_op` template function with `op_hardswish` as the operation to be applied.
    - The `unary_op` function checks the data types of the source and destination tensors and calls `apply_unary_op` with the appropriate type conversion if the types are supported.
    - If the types are not supported, an error message is printed and the program aborts.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place to contain the result of the hard swish operation.


---
### ggml\_compute\_forward\_sqr<!-- {{#callable:ggml_compute_forward_sqr}} -->
The function `ggml_compute_forward_sqr` applies the square operation to each element of a source tensor and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation, such as threading information.
    - `dst`: A pointer to a `ggml_tensor` structure that serves as both the destination tensor for the operation and the source tensor from which the data is read.
- **Control Flow**:
    - The function calls the `unary_op` template function with `op_sqr` as the operation to be applied.
    - Inside `unary_op`, the function checks the data types of the source and destination tensors to determine the appropriate template instantiation of `apply_unary_op`.
    - The `apply_unary_op` function is then called, which iterates over the elements of the source tensor, applies the square operation, and writes the results to the destination tensor.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place by applying the square operation to each of its elements.


---
### ggml\_compute\_forward\_sqrt<!-- {{#callable:ggml_compute_forward_sqrt}} -->
The function `ggml_compute_forward_sqrt` applies the square root operation to each element of a source tensor and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params`, which contains parameters for the computation, such as threading information.
    - `dst`: A pointer to `ggml_tensor`, which is the destination tensor where the results of the square root operation will be stored.
- **Control Flow**:
    - The function calls the `unary_op` template function with `op_sqrt` as the operation to be applied.
    - Inside `unary_op`, the function checks the data types of the source and destination tensors to determine the appropriate template instantiation for `apply_unary_op`.
    - The `apply_unary_op` function is then called, which performs the square root operation on each element of the source tensor and stores the result in the destination tensor.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place to contain the square root of each element from the source tensor.


---
### ggml\_compute\_forward\_sin<!-- {{#callable:ggml_compute_forward_sin}} -->
The `ggml_compute_forward_sin` function applies the sine operation to each element of a source tensor and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation, such as threading information.
    - `dst`: A pointer to a `ggml_tensor` structure that serves as both the destination for the computed sine values and the source tensor from which the input values are taken.
- **Control Flow**:
    - The function calls the `unary_op` template function with `op_sin` as the operation to be applied.
    - Inside `unary_op`, the function checks the data types of the source and destination tensors to determine the appropriate template instantiation for `apply_unary_op`.
    - The `apply_unary_op` function is called with the appropriate type parameters, which iterates over the elements of the source tensor, applies the sine operation using `vec_unary_op`, and stores the results in the destination tensor.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place to contain the sine of each element from the source tensor.


---
### ggml\_compute\_forward\_cos<!-- {{#callable:ggml_compute_forward_cos}} -->
The function `ggml_compute_forward_cos` applies the cosine operation to each element of the input tensor and stores the result in the destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation, such as threading information.
    - `dst`: A pointer to a `ggml_tensor` structure that serves as both the destination for the computed results and the source of the input data.
- **Control Flow**:
    - The function calls the `unary_op` template function with `op_cos` as the operation to be applied.
    - The `unary_op` function checks the data types of the source and destination tensors to determine the appropriate template specialization for `apply_unary_op`.
    - The `apply_unary_op` function ensures that the source and destination tensors are contiguous and have the same shape.
    - The `apply_unary_op` function calculates the range of indices to process based on the threading parameters.
    - The `vec_unary_op` function is called to apply the cosine operation to each element in the specified range, converting data types as necessary.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place to contain the cosine of each element from the input tensor.


---
### ggml\_compute\_forward\_log<!-- {{#callable:ggml_compute_forward_log}} -->
The function `ggml_compute_forward_log` applies the natural logarithm operation to each element of a source tensor and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params`, which contains parameters for the computation, such as threading information.
    - `dst`: A pointer to `ggml_tensor`, which is the destination tensor where the results of the logarithm operation will be stored.
- **Control Flow**:
    - The function calls `unary_op` with the `op_log` operation, passing the `params` and `dst` arguments.
    - Inside `unary_op`, the source tensor `src0` is retrieved from `dst->src[0]`.
    - The function checks the data types of `src0` and `dst` to determine the appropriate template instantiation for `apply_unary_op`.
    - If the data types are supported, `apply_unary_op` is called to perform the logarithm operation on each element of the source tensor and store the result in the destination tensor.
    - If the data types are unsupported, an error message is printed and the program aborts.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place to contain the results of the logarithm operation.


