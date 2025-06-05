# Purpose
This code is a C header file that declares a set of functions for performing forward computations on tensors, likely as part of a machine learning or numerical computation library. Each function takes a pointer to a `ggml_compute_params` structure and a `ggml_tensor` structure, suggesting that these functions apply various mathematical operations (such as absolute value, sign, negation, and various activation functions like tanh, relu, and sigmoid) to the data contained within the tensor. The use of `#pragma once` ensures the file is included only once per compilation, preventing duplicate definitions. The `extern "C"` block indicates that these functions can be linked with C++ code, ensuring compatibility across C and C++ projects.
# Imports and Dependencies

---
- `common.h`


# Function Declarations (Public API)

---
### ggml\_compute\_forward\_abs<!-- {{#callable_declaration:ggml_compute_forward_abs}} -->
Computes the element-wise absolute value of a tensor.
- **Description**: This function computes the absolute value of each element in the destination tensor, `dst`, using the specified computation parameters. It is typically used in neural network operations where the absolute value transformation is required. The function must be called with valid computation parameters and a properly initialized destination tensor. The operation is performed in-place, modifying the contents of `dst`. Ensure that `dst` is allocated and has the appropriate dimensions before calling this function.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that specifies the computation parameters. This must not be null and should be properly initialized before calling the function.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor. This tensor will be modified in-place to store the result of the absolute value operation. It must be allocated and initialized before the function is called.
- **Output**: None
- **See also**: [`ggml_compute_forward_abs`](unary-ops.cpp.driver.md#ggml_compute_forward_abs)  (Implementation)


---
### ggml\_compute\_forward\_sgn<!-- {{#callable_declaration:ggml_compute_forward_sgn}} -->
Applies the sign function to each element of the destination tensor.
- **Description**: This function processes the elements of the destination tensor by applying the sign function, which determines the sign of each element. It is typically used in neural network computations or other mathematical operations where the sign of tensor elements is required. The function must be called with valid compute parameters and a properly initialized destination tensor. The destination tensor is modified in place, and the function does not return a value. Ensure that the destination tensor is correctly allocated and initialized before calling this function.
- **Inputs**:
    - `params`: A pointer to a ggml_compute_params structure containing the parameters for the computation. This must not be null and should be properly initialized before calling the function.
    - `dst`: A pointer to a ggml_tensor structure representing the destination tensor. This tensor will be modified in place to store the result of the sign function applied to each of its elements. The tensor must be properly allocated and initialized before calling the function.
- **Output**: None
- **See also**: [`ggml_compute_forward_sgn`](unary-ops.cpp.driver.md#ggml_compute_forward_sgn)  (Implementation)


---
### ggml\_compute\_forward\_neg<!-- {{#callable_declaration:ggml_compute_forward_neg}} -->
Applies the negation operation to a tensor.
- **Description**: This function is used to apply a negation operation to the elements of a tensor, effectively multiplying each element by -1. It is typically called when a negated version of the tensor is required for further computation. The function must be provided with valid compute parameters and a destination tensor, which will be modified to store the result of the negation operation. It is important to ensure that the destination tensor is properly initialized and allocated before calling this function.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing the parameters for the computation. This must not be null, and it should be properly initialized before calling the function.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the result of the negation operation. This tensor must be allocated and initialized prior to the function call, and it must not be null.
- **Output**: None
- **See also**: [`ggml_compute_forward_neg`](unary-ops.cpp.driver.md#ggml_compute_forward_neg)  (Implementation)


---
### ggml\_compute\_forward\_step<!-- {{#callable_declaration:ggml_compute_forward_step}} -->
Applies the step function to each element of the destination tensor.
- **Description**: This function processes the destination tensor by applying a step function to each of its elements. It is typically used in neural network computations where a step activation function is required. The function requires valid compute parameters and a destination tensor, which it modifies in place. Ensure that the destination tensor is properly initialized and that the compute parameters are correctly set before calling this function.
- **Inputs**:
    - `params`: A pointer to a ggml_compute_params structure containing the parameters for the computation. This must not be null and should be properly initialized before calling the function.
    - `dst`: A pointer to a ggml_tensor structure representing the destination tensor. This tensor will be modified in place, so it must be initialized and have sufficient capacity to store the results of the computation.
- **Output**: None
- **See also**: [`ggml_compute_forward_step`](unary-ops.cpp.driver.md#ggml_compute_forward_step)  (Implementation)


---
### ggml\_compute\_forward\_tanh<!-- {{#callable_declaration:ggml_compute_forward_tanh}} -->
Applies the hyperbolic tangent function to each element of the destination tensor.
- **Description**: This function computes the hyperbolic tangent (tanh) of each element in the destination tensor `dst`. It is typically used in neural network computations where the tanh activation function is required. The function must be called with valid compute parameters and a properly initialized destination tensor. It is important to ensure that the destination tensor is allocated and has the appropriate dimensions before calling this function. The function does not return a value, and any errors in computation are typically handled internally.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing the parameters for the computation. This must be a valid, non-null pointer, and the structure should be properly initialized before calling the function.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the result of the tanh operation. This tensor must be allocated and initialized prior to calling the function, and it must not be null. The function will overwrite the contents of this tensor with the computed results.
- **Output**: None
- **See also**: [`ggml_compute_forward_tanh`](unary-ops.cpp.driver.md#ggml_compute_forward_tanh)  (Implementation)


---
### ggml\_compute\_forward\_elu<!-- {{#callable_declaration:ggml_compute_forward_elu}} -->
Applies the ELU activation function to the input tensor.
- **Description**: This function applies the Exponential Linear Unit (ELU) activation function to each element of the input tensor specified by `dst`. It is typically used in neural network computations to introduce non-linearity. The function requires valid compute parameters and a destination tensor to store the results. Ensure that the `dst` tensor is properly initialized and allocated before calling this function. The function does not return a value but modifies the `dst` tensor in place.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing the parameters for the computation. This must not be null and should be properly initialized before calling the function.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result of the ELU operation will be stored. This tensor must be initialized and allocated with sufficient space to hold the results. The function modifies this tensor in place.
- **Output**: None
- **See also**: [`ggml_compute_forward_elu`](unary-ops.cpp.driver.md#ggml_compute_forward_elu)  (Implementation)


---
### ggml\_compute\_forward\_relu<!-- {{#callable_declaration:ggml_compute_forward_relu}} -->
Applies the ReLU activation function to the input tensor.
- **Description**: This function applies the Rectified Linear Unit (ReLU) activation function to each element of the input tensor, storing the result in the same tensor. It is typically used in neural network computations to introduce non-linearity. The function must be called with valid compute parameters and a properly initialized tensor. The ReLU function outputs zero for negative inputs and the input itself for non-negative inputs, effectively clamping negative values to zero.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing the parameters for the computation. This must not be null and should be properly initialized before calling the function.
    - `dst`: A pointer to a `ggml_tensor` structure representing the tensor to which the ReLU function will be applied. The tensor must be initialized and will be modified in place. The caller retains ownership of the tensor.
- **Output**: None
- **See also**: [`ggml_compute_forward_relu`](unary-ops.cpp.driver.md#ggml_compute_forward_relu)  (Implementation)


---
### ggml\_compute\_forward\_sigmoid<!-- {{#callable_declaration:ggml_compute_forward_sigmoid}} -->
Applies the sigmoid function to each element of the destination tensor.
- **Description**: This function computes the sigmoid activation function for each element in the provided destination tensor. It is typically used in neural network computations where the sigmoid function is needed to introduce non-linearity. The function requires a set of compute parameters and a destination tensor, which will be modified in place to contain the results of the sigmoid operation. Ensure that the destination tensor is properly initialized and allocated before calling this function.
- **Inputs**:
    - `params`: A pointer to a ggml_compute_params structure containing the parameters for the computation. This must not be null and should be properly initialized before calling the function.
    - `dst`: A pointer to a ggml_tensor structure that will be modified to store the result of the sigmoid operation. The tensor must be allocated and initialized before the function is called, and the caller retains ownership.
- **Output**: None
- **See also**: [`ggml_compute_forward_sigmoid`](unary-ops.cpp.driver.md#ggml_compute_forward_sigmoid)  (Implementation)


---
### ggml\_compute\_forward\_hardsigmoid<!-- {{#callable_declaration:ggml_compute_forward_hardsigmoid}} -->
Applies the hard sigmoid activation function to a tensor.
- **Description**: This function is used to apply the hard sigmoid activation function to each element of the input tensor, storing the result in the destination tensor. It is typically used in neural network computations where the hard sigmoid function is required as an activation function. The function must be called with valid compute parameters and a destination tensor that is properly initialized. The operation is performed in-place on the destination tensor, which means the original data in the destination tensor will be overwritten.
- **Inputs**:
    - `params`: A pointer to a ggml_compute_params structure containing the parameters for the computation. This must not be null and should be properly initialized before calling the function.
    - `dst`: A pointer to a ggml_tensor structure that will hold the result of the computation. This tensor must be initialized and will be modified in-place by the function.
- **Output**: None
- **See also**: [`ggml_compute_forward_hardsigmoid`](unary-ops.cpp.driver.md#ggml_compute_forward_hardsigmoid)  (Implementation)


---
### ggml\_compute\_forward\_exp<!-- {{#callable_declaration:ggml_compute_forward_exp}} -->
Computes the element-wise exponential of a tensor.
- **Description**: This function applies the exponential function to each element of the input tensor, storing the result in the destination tensor. It is typically used in neural network operations where element-wise transformations are required. The function must be called with valid compute parameters and a destination tensor that is properly initialized. The destination tensor will be modified to contain the results of the exponential operation.
- **Inputs**:
    - `params`: A pointer to a ggml_compute_params structure containing the parameters for the computation. This must not be null and should be properly initialized before calling the function.
    - `dst`: A pointer to a ggml_tensor structure where the result of the exponential operation will be stored. This must not be null and should be allocated and initialized before calling the function. The function will overwrite the contents of this tensor with the computed results.
- **Output**: None
- **See also**: [`ggml_compute_forward_exp`](unary-ops.cpp.driver.md#ggml_compute_forward_exp)  (Implementation)


---
### ggml\_compute\_forward\_hardswish<!-- {{#callable_declaration:ggml_compute_forward_hardswish}} -->
Applies the HardSwish activation function to a tensor.
- **Description**: This function applies the HardSwish activation function to the input tensor specified by `dst`. It is typically used in neural network computations where the HardSwish activation is required. The function expects valid compute parameters and a destination tensor, which will be modified in place to contain the result of the HardSwish operation. Ensure that the `params` and `dst` pointers are not null before calling this function, as invalid pointers may lead to undefined behavior.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` structure representing the destination tensor. This tensor will be modified in place to store the result of the HardSwish operation. Must not be null.
- **Output**: None
- **See also**: [`ggml_compute_forward_hardswish`](unary-ops.cpp.driver.md#ggml_compute_forward_hardswish)  (Implementation)


---
### ggml\_compute\_forward\_sqr<!-- {{#callable_declaration:ggml_compute_forward_sqr}} -->
Computes the element-wise square of a tensor.
- **Description**: This function computes the square of each element in the input tensor and stores the result in the destination tensor. It is typically used in neural network operations or other mathematical computations where element-wise squaring is required. The function must be called with valid compute parameters and a destination tensor that is properly initialized. The destination tensor will be modified to contain the squared values of the input tensor.
- **Inputs**:
    - `params`: A pointer to a ggml_compute_params structure containing the parameters for the computation. This must not be null and should be properly initialized before calling the function.
    - `dst`: A pointer to a ggml_tensor structure where the result will be stored. This tensor must be initialized and have the appropriate dimensions to store the squared values. The caller retains ownership of this tensor.
- **Output**: None
- **See also**: [`ggml_compute_forward_sqr`](unary-ops.cpp.driver.md#ggml_compute_forward_sqr)  (Implementation)


---
### ggml\_compute\_forward\_sqrt<!-- {{#callable_declaration:ggml_compute_forward_sqrt}} -->
Computes the element-wise square root of a tensor.
- **Description**: This function computes the square root of each element in the input tensor and stores the result in the same tensor. It is typically used in neural network operations or other mathematical computations where element-wise square root is required. The function must be called with valid compute parameters and a properly initialized tensor. The input tensor is modified in place, so ensure that the tensor is prepared for mutation before calling this function.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains the parameters for the computation. This must not be null and should be properly initialized before calling the function.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the tensor on which the square root operation will be performed. This tensor is modified in place, so it must be initialized and contain valid data before the function is called.
- **Output**: None
- **See also**: [`ggml_compute_forward_sqrt`](unary-ops.cpp.driver.md#ggml_compute_forward_sqrt)  (Implementation)


---
### ggml\_compute\_forward\_sin<!-- {{#callable_declaration:ggml_compute_forward_sin}} -->
Computes the element-wise sine of a tensor.
- **Description**: This function calculates the sine of each element in the input tensor and stores the result in the destination tensor. It is typically used in neural network computations or other mathematical operations where the sine function is required. The function assumes that the destination tensor is properly allocated and has the same dimensions as the input tensor. It must be called with valid compute parameters to ensure correct execution.
- **Inputs**:
    - `params`: A pointer to a ggml_compute_params structure containing the parameters for the computation. This must not be null and should be properly initialized before calling the function.
    - `dst`: A pointer to a ggml_tensor structure where the result of the sine operation will be stored. This tensor must be allocated and have the same dimensions as the input tensor. The caller retains ownership of this tensor.
- **Output**: None
- **See also**: [`ggml_compute_forward_sin`](unary-ops.cpp.driver.md#ggml_compute_forward_sin)  (Implementation)


---
### ggml\_compute\_forward\_cos<!-- {{#callable_declaration:ggml_compute_forward_cos}} -->
Computes the cosine of each element in the destination tensor.
- **Description**: This function applies the cosine operation to each element of the destination tensor specified by `dst`. It is typically used in neural network computations or other mathematical operations where the cosine transformation of tensor data is required. The function requires valid compute parameters and a destination tensor to be provided. It is important to ensure that the destination tensor is properly initialized and allocated before calling this function. The function does not return a value, and any errors or invalid inputs should be handled by the caller.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing the parameters for the computation. This must not be null, and the structure should be properly initialized before calling the function.
    - `dst`: A pointer to a `ggml_tensor` structure that will store the result of the cosine operation. This tensor must be allocated and initialized prior to the function call. The caller retains ownership of this tensor.
- **Output**: None
- **See also**: [`ggml_compute_forward_cos`](unary-ops.cpp.driver.md#ggml_compute_forward_cos)  (Implementation)


---
### ggml\_compute\_forward\_log<!-- {{#callable_declaration:ggml_compute_forward_log}} -->
Computes the natural logarithm of each element in the destination tensor.
- **Description**: This function applies the natural logarithm operation to each element of the destination tensor specified by `dst`. It is typically used in neural network computations or other mathematical operations where logarithmic transformations are required. The function must be called with valid compute parameters and a properly initialized tensor. The destination tensor is modified in place, and the input tensor values should be positive to avoid undefined behavior.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains the parameters for the computation. This must not be null, and it should be properly initialized before calling the function.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored. This tensor must be initialized and must contain positive values to ensure valid logarithmic computation. The function modifies this tensor in place.
- **Output**: None
- **See also**: [`ggml_compute_forward_log`](unary-ops.cpp.driver.md#ggml_compute_forward_log)  (Implementation)


