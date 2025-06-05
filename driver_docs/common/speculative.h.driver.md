# Purpose
This code is a C++ header file designed to provide a narrow, specialized functionality related to speculative token generation, likely in a machine learning or natural language processing context. It defines a structure `common_speculative_params` to hold parameters for speculative token generation, such as the maximum number of drafted tokens (`n_draft`) and a minimum probability threshold (`p_min`). The header declares functions for initializing and freeing a `common_speculative` structure, checking compatibility between contexts, and generating draft tokens using a draft model. The inclusion of `#pragma once` suggests it is intended to be imported into other C++ files to ensure single inclusion, and the use of external structures like `llama_context` and `llama_tokens` indicates it relies on other components defined in the included headers "llama.h" and "common.h".
# Imports and Dependencies

---
- `llama.h`
- `common.h`


# Global Variables

---
### common\_speculative\_init
- **Type**: `function pointer`
- **Description**: The `common_speculative_init` is a function pointer that returns a pointer to a `common_speculative` structure. It takes a pointer to a `llama_context` structure as its parameter, which is used to initialize the `common_speculative` structure.
- **Use**: This function is used to initialize a `common_speculative` structure with a given `llama_context`.


# Data Structures

---
### common\_speculative\_params<!-- {{#data_structure:common_speculative_params}} -->
- **Type**: `struct`
- **Members**:
    - `n_draft`: Maximum number of tokens that can be drafted.
    - `n_reuse`: An integer value, possibly indicating the number of tokens to reuse.
    - `p_min`: Minimum probability required to accept a token in the draft.
- **Description**: The `common_speculative_params` struct is designed to hold parameters for speculative token generation in a drafting process. It includes `n_draft`, which specifies the maximum number of tokens that can be drafted, `n_reuse`, which might indicate the number of tokens to reuse, and `p_min`, which sets the minimum probability threshold for accepting a token in the draft. These parameters are likely used to control the behavior of a speculative token generation algorithm, ensuring that only tokens meeting certain criteria are considered during the drafting process.


