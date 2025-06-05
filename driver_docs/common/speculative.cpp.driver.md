# Purpose
This C++ source code file is designed to facilitate speculative execution in a machine learning context, specifically for language models. It provides a set of functions and structures to manage speculative sampling and token generation using a draft model. The primary structure, `common_speculative`, encapsulates the context and sampling components necessary for this process. The file includes functions to initialize and free this structure, ensuring proper memory management and resource allocation. The [`common_speculative_init`](#common_speculative_init) function sets up the speculative execution environment by initializing a batch and a sampler with specific parameters, while [`common_speculative_free`](#common_speculative_free) handles the cleanup of resources.

The code also includes a compatibility check function, [`common_speculative_are_compatible`](#common_speculative_are_compatible), which ensures that the vocabularies of the draft and target models are sufficiently similar to allow for speculative execution. This involves checking the types and sizes of vocabularies, as well as the consistency of special tokens. The main functionality is provided by [`common_speculative_gen_draft`](#common_speculative_gen_draft), which generates a draft sequence of tokens based on a given prompt and parameters. This function reuses parts of previous drafts when possible, evaluates new tokens, and samples a specified number of draft tokens, ensuring that only high-confidence tokens are added to the result. The code is structured to be part of a larger system, likely a library, that can be integrated into applications requiring efficient speculative execution for language models.
# Imports and Dependencies

---
- `speculative.h`
- `log.h`
- `common.h`
- `sampling.h`
- `cstring`
- `algorithm`


# Data Structures

---
### common\_speculative<!-- {{#data_structure:common_speculative}} -->
- **Type**: `struct`
- **Members**:
    - `ctx`: A pointer to a llama_context structure, representing the context for llama operations.
    - `smpl`: A pointer to a common_sampler structure, used for sampling operations.
    - `batch`: An instance of llama_batch, used to manage batches of tokens.
    - `prompt`: An instance of llama_tokens, representing the prompt tokens for speculative generation.
- **Description**: The `common_speculative` struct is designed to facilitate speculative execution in a llama-based model. It holds pointers to a llama context and a sampler, as well as a batch and prompt tokens, which are essential for managing and executing speculative token generation. This struct is integral to the process of generating draft tokens and ensuring compatibility between different llama contexts.


# Functions

---
### common\_speculative\_init<!-- {{#callable:common_speculative_init}} -->
The `common_speculative_init` function initializes a `common_speculative` structure with a given llama context and sets up a sampler with specific parameters.
- **Inputs**:
    - `ctx_dft`: A pointer to a `llama_context` structure that serves as the default context for initializing the `common_speculative` structure.
- **Control Flow**:
    - Allocate memory for a new `common_speculative` structure and initialize its `ctx` field with `ctx_dft`, `smpl` with `nullptr`, `batch` with a new llama batch, and `prompt` as an empty token list.
    - Define a `common_params_sampling` structure and set its `no_perf` field to `false`.
    - Set the `top_k` parameter to 10 and initialize the `samplers` list with `COMMON_SAMPLER_TYPE_TOP_K`.
    - Initialize the `smpl` field of the `common_speculative` structure using [`common_sampler_init`](sampling.cpp.driver.md#common_sampler_init) with the model obtained from `ctx_dft` and the defined sampling parameters.
    - Return the initialized `common_speculative` structure.
- **Output**: A pointer to the newly initialized `common_speculative` structure.
- **Functions called**:
    - [`common_sampler_init`](sampling.cpp.driver.md#common_sampler_init)


---
### common\_speculative\_free<!-- {{#callable:common_speculative_free}} -->
The `common_speculative_free` function deallocates resources associated with a `common_speculative` structure.
- **Inputs**:
    - `spec`: A pointer to a `common_speculative` structure that needs to be freed.
- **Control Flow**:
    - Check if the `spec` pointer is `nullptr` and return immediately if it is.
    - Call [`common_sampler_free`](sampling.cpp.driver.md#common_sampler_free) to free the sampler associated with `spec`.
    - Call `llama_batch_free` to free the batch associated with `spec`.
    - Delete the `spec` object to free its memory.
- **Output**: This function does not return any value.
- **Functions called**:
    - [`common_sampler_free`](sampling.cpp.driver.md#common_sampler_free)


---
### common\_speculative\_are\_compatible<!-- {{#callable:common_speculative_are_compatible}} -->
The function `common_speculative_are_compatible` checks if two llama contexts are compatible for speculative execution by comparing their vocabularies and special tokens.
- **Inputs**:
    - `ctx_tgt`: A pointer to the target llama context.
    - `ctx_dft`: A pointer to the draft llama context.
- **Control Flow**:
    - Retrieve the models from both target and draft contexts using `llama_get_model`.
    - Retrieve the vocabularies from both models using `llama_model_get_vocab`.
    - Check if the vocabulary types of both contexts match using [`llama_vocab_type`](../include/llama.h.driver.md#llama_vocab_type); log and return false if they don't match.
    - Compare special tokens (BOS and EOS) and their addition flags between the two vocabularies; log and return false if they don't match.
    - Calculate the difference in vocabulary sizes and check if it exceeds `SPEC_VOCAB_MAX_SIZE_DIFFERENCE`; log and return false if it does.
    - Iterate over tokens starting from `SPEC_VOCAB_CHECK_START_TOKEN_ID` to the minimum of both vocabulary sizes, comparing token texts; log and return false if any token text differs.
    - Return true if all checks pass, indicating compatibility.
- **Output**: A boolean value indicating whether the two contexts are compatible for speculative execution.
- **Functions called**:
    - [`llama_vocab_type`](../include/llama.h.driver.md#llama_vocab_type)


---
### common\_speculative\_gen\_draft<!-- {{#callable:common_speculative_gen_draft}} -->
The `common_speculative_gen_draft` function generates a draft sequence of tokens by reusing parts of a previous context and sampling new tokens based on a given prompt and parameters.
- **Inputs**:
    - `spec`: A pointer to a `common_speculative` structure containing the context, sampler, batch, and prompt information.
    - `params`: A `common_speculative_params` structure containing parameters such as the number of draft tokens to generate and the minimum probability for token acceptance.
    - `prompt_tgt`: A reference to a `llama_tokens` object representing the target prompt tokens to be used for generating the draft.
    - `id_last`: A `llama_token` representing the last token ID to be added to the prompt.
- **Control Flow**:
    - Initialize references to the batch, context, sampler, and prompt from the `spec` structure.
    - Calculate the starting index `i_start` for the prompt based on the context size and draft parameters.
    - Iterate over the existing prompt to find the longest reusable segment that matches the target prompt.
    - If no reusable segment is found, clear the context and prompt; otherwise, adjust the prompt and context to reuse the identified segment.
    - Prepare a batch to evaluate new tokens from the target prompt and add them to the prompt.
    - Decode the batch if it contains tokens, updating the context with the new prompt state.
    - Reset the sampler and sample new tokens from the draft model, adding them to the result if they meet the probability threshold.
    - Return the generated draft tokens as the result.
- **Output**: A `llama_tokens` object containing the generated draft tokens.
- **Functions called**:
    - [`common_sampler_reset`](sampling.cpp.driver.md#common_sampler_reset)
    - [`common_sampler_sample`](sampling.cpp.driver.md#common_sampler_sample)
    - [`common_sampler_get_candidates`](sampling.cpp.driver.md#common_sampler_get_candidates)
    - [`common_sampler_accept`](sampling.cpp.driver.md#common_sampler_accept)


