# Purpose
This C++ header file defines the `common_sampler` structure and its associated functions, which extend the functionality of a `llama_sampler` by incorporating additional features such as grammar support, custom sampling logic, token history tracking, and performance metrics. The primary purpose of this file is to provide a common implementation of sampling logic that can be shared across various examples, allowing for flexible and efficient token sampling based on different parameters like temperature and grammar constraints. The file includes functions for initializing, freeing, and manipulating the `common_sampler`, as well as functions for sampling tokens with or without grammar checks, and for managing and accessing candidate tokens and performance data.

The file serves as an interface for a more sophisticated sampling mechanism that can be integrated into applications using the `llama` library. It provides a public API for initializing and managing the `common_sampler`, as well as for performing token sampling operations that respect both sampling chain logic and grammar constraints. The file also includes utility functions for accessing internal data, such as candidate tokens and the last accepted token, and for converting sampler types to strings or characters. This header file is intended to be included in other C++ source files, allowing developers to leverage its extended sampling capabilities in their applications.
# Imports and Dependencies

---
- `llama.h`
- `common.h`
- `string`
- `vector`


# Global Variables

---
### common\_sampler\_init
- **Type**: `struct common_sampler *`
- **Description**: The `common_sampler_init` function is a global function that initializes a `common_sampler` structure. This structure extends the `llama_sampler` with additional functionalities such as grammar support, custom sampler logic, and performance metrics.
- **Use**: This function is used to create and initialize a `common_sampler` instance with a given model and sampling parameters.


---
### common\_sampler\_clone
- **Type**: `struct common_sampler *`
- **Description**: The `common_sampler_clone` is a function that takes a pointer to a `common_sampler` structure and returns a new pointer to a `common_sampler` structure. This function is likely used to create a duplicate or copy of the existing `common_sampler` instance, allowing for operations on a separate instance without affecting the original.
- **Use**: This function is used to clone an existing `common_sampler` instance, creating a new instance with the same state as the original.


---
### common\_sampler\_get\_candidates
- **Type**: `llama_token_data_array *`
- **Description**: The `common_sampler_get_candidates` is a function that returns a pointer to a `llama_token_data_array`, which represents the internal list of current candidate tokens in the `common_sampler` structure. This function is part of the extended functionality provided by the `common_sampler`, which builds upon the `llama_sampler` to include additional features such as grammar support and custom sampling logic.
- **Use**: This function is used to access the current candidate tokens within a `common_sampler` instance, allowing users to examine the probabilities of non-sampled tokens.


---
### llama\_sampler\_init\_llg
- **Type**: `function`
- **Description**: The `llama_sampler_init_llg` function initializes a `llama_sampler` object using a given vocabulary, grammar kind, and grammar data. It is part of the llama_sampler API and is used to set up a sampler with specific grammar constraints.
- **Use**: This function is used to create and configure a `llama_sampler` instance with grammar support, which can then be used for sampling operations in the llama library.


