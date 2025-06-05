# Purpose
This C++ source code file is designed to implement a sampling mechanism for a language model, specifically utilizing a feature called "llguidance" if it is enabled. The code is structured to integrate with a larger system, likely involving natural language processing or machine learning, where it provides specialized functionality for token sampling based on grammar constraints. The file includes several key components: it defines a structure `llama_sampler_llg` to hold the context for the sampler, including vocabulary, grammar kind, and tokenizer. It also implements a series of static functions that manage the lifecycle and operations of the sampler, such as initialization, token acceptance, application of grammar constraints, resetting, cloning, and freeing resources.

The code is modular and encapsulates the logic for handling grammar-based token sampling, which is crucial for tasks that require adherence to specific syntactic rules. It interfaces with external components like `LlgTokenizer` and `LlgMatcher` to perform tokenization and grammar matching, respectively. The file defines a public API through the [`llama_sampler_init_llg`](#llama_sampler_init_llg) function, which initializes the sampler with the given vocabulary and grammar data. This function is conditionally compiled based on the `LLAMA_USE_LLGUIDANCE` preprocessor directive, indicating that the llguidance feature is optional and can be toggled during the build process. The code is intended to be part of a larger library or application, providing a focused and specialized functionality within the broader context of language model sampling.
# Imports and Dependencies

---
- `sampling.h`
- `log.h`
- `llguidance.h`
- `cmath`


# Global Variables

---
### llama\_sampler\_llg\_i
- **Type**: `llama_sampler_i`
- **Description**: The `llama_sampler_llg_i` is a static instance of the `llama_sampler_i` structure, which serves as an interface for the llama sampler with LLGUIDANCE functionality. It contains function pointers to various operations such as name retrieval, token acceptance, application of sampling logic, resetting, cloning, and freeing resources.
- **Use**: This variable is used to define the interface for a llama sampler that utilizes LLGUIDANCE, providing the necessary function implementations for its operation.


# Data Structures

---
### llama\_sampler\_llg<!-- {{#data_structure:llama_sampler_llg}} -->
- **Type**: `struct`
- **Members**:
    - `vocab`: A pointer to a llama_vocab structure, representing the vocabulary used.
    - `grammar_kind`: A string representing the type of grammar being used.
    - `grammar_data`: A string containing the data related to the grammar.
    - `tokenizer`: A pointer to an LlgTokenizer, used for tokenizing input.
    - `grammar`: A pointer to an LlgMatcher, used for matching grammar rules.
- **Description**: The `llama_sampler_llg` struct is designed to facilitate language sampling with grammar guidance in the Llama framework. It holds references to a vocabulary, grammar type, and grammar data, as well as pointers to a tokenizer and a grammar matcher. This struct is integral to the process of applying grammar constraints during token sampling, ensuring that the generated language adheres to specified grammatical rules.


# Functions

---
### llama\_sampler\_llg\_new<!-- {{#callable:llama_sampler_llg_new}} -->
The `llama_sampler_llg_new` function initializes a new LlgMatcher object using a tokenizer and grammar specifications, with optional logging based on environment settings.
- **Inputs**:
    - `tokenizer`: A pointer to an LlgTokenizer object used for tokenization.
    - `grammar_kind`: A C-style string representing the type of grammar to be used.
    - `grammar_data`: A C-style string containing the grammar data to be used.
- **Control Flow**:
    - Initialize an LlgConstraintInit structure `cinit` and set its defaults using the provided tokenizer.
    - Retrieve the log level from the environment variable `LLGUIDANCE_LOG_LEVEL` and set `cinit.log_stderr_level` if the log level is specified.
    - Create a new LlgMatcher object `c` using `llg_new_matcher` with the initialized constraints, grammar kind, and grammar data.
    - Check for errors in the matcher using `llg_matcher_get_error`; if an error exists, log the error, free the matcher, and return `nullptr`.
    - If no errors are found, return the created LlgMatcher object `c`.
- **Output**: Returns a pointer to a newly created LlgMatcher object if successful, or `nullptr` if an error occurs during creation.


---
### llama\_sampler\_llg\_name<!-- {{#callable:llama_sampler_llg_name}} -->
The function `llama_sampler_llg_name` returns the name of the llama sampler as a constant string "llguidance".
- **Inputs**: None
- **Control Flow**:
    - The function takes a single argument, `const llama_sampler *`, which is not used in the function body.
    - It directly returns the string literal "llguidance".
- **Output**: A constant C-style string "llguidance".


---
### llama\_sampler\_llg\_accept\_impl<!-- {{#callable:llama_sampler_llg_accept_impl}} -->
The `llama_sampler_llg_accept_impl` function processes a token by consuming it with a grammar matcher if the grammar is present in the sampler context.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure, which contains the context for the sampler.
    - `token`: A `llama_token` that represents the token to be processed by the function.
- **Control Flow**:
    - Retrieve the context from the `llama_sampler` by casting its `ctx` member to `llama_sampler_llg` type.
    - Check if the `grammar` member of the context is not null.
    - If the `grammar` is present, call `llg_matcher_consume_token` with the grammar and the token to consume the token.
- **Output**: This function does not return any value; it performs an operation on the grammar within the sampler context.


---
### llama\_sampler\_llg\_apply<!-- {{#callable:llama_sampler_llg_apply}} -->
The `llama_sampler_llg_apply` function applies a grammar-based mask to a set of token data, modifying their logit values based on the mask.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure, which contains context information for the sampler.
    - `cur_p`: A pointer to a `llama_token_data_array` structure, which holds the current set of token data to be processed.
- **Control Flow**:
    - Retrieve the context (`ctx`) from the `llama_sampler` structure by casting its `ctx` member to `llama_sampler_llg` type.
    - Check if the `grammar` member of `ctx` is not null, indicating that grammar-based processing is enabled.
    - Retrieve the mask from the grammar using `llg_matcher_get_mask`.
    - If the mask is null, attempt to compute the mask using `llg_matcher_compute_mask`.
    - If mask computation fails, log an error, free the grammar matcher, set `grammar` to null, and return.
    - Iterate over each token in `cur_p->data`.
    - For each token, check if the corresponding bit in the mask is not set.
    - If the bit is not set, set the token's `logit` value to negative infinity.
- **Output**: The function does not return a value; it modifies the `logit` values of tokens in the `cur_p` array in place.


---
### llama\_sampler\_llg\_reset<!-- {{#callable:llama_sampler_llg_reset}} -->
The `llama_sampler_llg_reset` function resets the grammar matcher within a `llama_sampler` context if it exists.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure, which contains the context to be reset.
- **Control Flow**:
    - Cast the `ctx` member of the `llama_sampler` to a `llama_sampler_llg` pointer.
    - Check if the `grammar` member of the `llama_sampler_llg` context is not null.
    - If the `grammar` is not null, call `llg_matcher_reset` on the `grammar`.
- **Output**: This function does not return any value.


---
### llama\_sampler\_llg\_clone<!-- {{#callable:llama_sampler_llg_clone}} -->
The `llama_sampler_llg_clone` function creates a new `llama_sampler` object by cloning the state of an existing `llama_sampler` object, including its grammar and tokenizer if they exist.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` object that is to be cloned.
- **Control Flow**:
    - Cast the context of the input sampler `smpl` to `llama_sampler_llg` type.
    - Initialize a new `llama_sampler` object using [`llama_sampler_init_llg`](#llama_sampler_init_llg) with the vocabulary from the input sampler's context.
    - Cast the context of the newly created sampler to `llama_sampler_llg` type.
    - If the input sampler's context has a grammar, clone the grammar and tokenizer, and copy the grammar kind and data to the new sampler's context.
- **Output**: Returns a pointer to a new `llama_sampler` object that is a clone of the input sampler.
- **Functions called**:
    - [`llama_sampler_init_llg`](#llama_sampler_init_llg)


---
### llama\_sampler\_llg\_free<!-- {{#callable:llama_sampler_llg_free}} -->
The `llama_sampler_llg_free` function deallocates resources associated with a `llama_sampler` object, specifically freeing the grammar matcher and tokenizer if they exist, and then deletes the context.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` object whose resources are to be freed.
- **Control Flow**:
    - Retrieve the context from the `llama_sampler` object by casting `smpl->ctx` to `llama_sampler_llg`.
    - Check if the `grammar` field in the context is not null.
    - If `grammar` is not null, call `llg_free_matcher` to free the grammar matcher and `llg_free_tokenizer` to free the tokenizer.
    - Delete the context object using the `delete` operator.
- **Output**: The function does not return any value; it performs cleanup operations on the provided `llama_sampler` object.


---
### llama\_sampler\_llg\_tokenize\_fn<!-- {{#callable:llama_sampler_llg_tokenize_fn}} -->
The `llama_sampler_llg_tokenize_fn` function tokenizes a given byte sequence using a specified vocabulary and stores the resulting tokens in an output buffer.
- **Inputs**:
    - `user_data`: A pointer to the llama_vocab structure, which contains the vocabulary used for tokenization.
    - `bytes`: A pointer to the byte sequence that needs to be tokenized.
    - `bytes_len`: The length of the byte sequence to be tokenized.
    - `output_tokens`: A pointer to the buffer where the resulting tokens will be stored.
    - `output_tokens_len`: The length of the output buffer for storing tokens.
- **Control Flow**:
    - The function casts the `user_data` to a `llama_vocab` pointer to access the vocabulary.
    - It initializes an integer `r` to zero to store the result of the tokenization process.
    - A try-catch block is used to call the `llama_tokenize` function, which performs the tokenization using the provided vocabulary and byte sequence.
    - If an exception occurs during tokenization, the function aborts with an error message.
    - After tokenization, if the result `r` is negative, the function returns the negation of `r` to indicate an error.
    - If tokenization is successful, the function returns the result `r`, which represents the number of tokens produced.
- **Output**: The function returns a size_t value representing the number of tokens produced, or a negative value indicating an error during tokenization.


---
### llama\_sampler\_llg\_new\_tokenizer<!-- {{#callable:llama_sampler_llg_new_tokenizer}} -->
The function `llama_sampler_llg_new_tokenizer` creates and returns a new `LlgTokenizer` object based on the provided vocabulary, caching the tokenizer for future use with the same vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure representing the vocabulary to be used for creating the tokenizer.
- **Control Flow**:
    - Check if the cached vocabulary is the same as the input; if so, return a clone of the cached tokenizer.
    - Retrieve the end-of-sequence token from the vocabulary, using a fallback if necessary.
    - Determine the number of tokens in the vocabulary and allocate memory for token lengths and token bytes.
    - Iterate over each token in the vocabulary, detokenizing it and storing its size and byte representation.
    - Initialize an `LlgTokenizerInit` structure with the token data and other parameters.
    - Create a new `LlgTokenizer` using the initialized structure, handling any errors that occur.
    - Free the memory allocated for token bytes and lengths.
    - If a tokenizer was successfully created, update the cache with the new tokenizer and vocabulary.
    - Return a clone of the cached tokenizer.
- **Output**: A pointer to a newly created `LlgTokenizer` object, or `nullptr` if an error occurred during tokenizer creation.


---
### llama\_sampler\_init\_llg<!-- {{#callable:llama_sampler_init_llg}} -->
The function `llama_sampler_init_llg` initializes a `llama_sampler` object with LLGUIDANCE support, but returns `nullptr` if LLGUIDANCE is not enabled.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` object, representing the vocabulary to be used.
    - `grammar_kind`: A C-style string representing the kind of grammar to be used, or `nullptr` if no grammar is specified.
    - `grammar_data`: A C-style string containing the grammar data, or `nullptr` if no grammar is specified.
- **Control Flow**:
    - The function checks if the `LLAMA_USE_LLGUIDANCE` preprocessor directive is defined.
    - If `LLAMA_USE_LLGUIDANCE` is not defined, it logs a warning message indicating that LLGUIDANCE is not enabled.
    - The function returns `nullptr` immediately if LLGUIDANCE is not enabled.
- **Output**: The function returns a `nullptr` if LLGUIDANCE is not enabled, indicating that the sampler could not be initialized.


