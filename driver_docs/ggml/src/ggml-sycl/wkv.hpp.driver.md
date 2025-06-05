# Purpose
This code is a C++ header file, as indicated by the use of include guards (`#ifndef`, `#define`, `#endif`) to prevent multiple inclusions. It provides a narrow functionality by declaring two functions, `ggml_sycl_op_rwkv_wkv6` and `ggml_sycl_op_rwkv_wkv7`, which are likely intended for operations related to SYCL (a C++-based parallel programming model) on RWKV (Recurrent Weighted Key-Value) models, as suggested by the function names. The functions take a `ggml_backend_sycl_context` reference and a pointer to a `ggml_tensor`, indicating that they are designed to perform operations on tensor data within a SYCL context. This header file is intended to be included in other C++ source files that require these specific SYCL operations, suggesting it is part of a larger library or framework dealing with machine learning or parallel computing tasks.
# Imports and Dependencies

---
- `common.hpp`


