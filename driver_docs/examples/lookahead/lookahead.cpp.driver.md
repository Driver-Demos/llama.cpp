# Purpose
This C++ source code file is an executable program designed to perform n-gram based token generation using a language model, likely for natural language processing tasks. The code initializes and utilizes a language model, referred to as "llama," to tokenize a given input prompt and generate subsequent tokens based on n-gram analysis. The program employs a lookahead decoding strategy, which involves maintaining a window of tokens and verifying n-grams to ensure the generated sequence adheres to certain constraints. The main components of the code include structures for managing n-gram data and containers, functions for initializing and interacting with the language model, and a loop that iteratively generates tokens while checking for end-of-sequence conditions.

The code is structured around a main function that orchestrates the initialization of the language model, tokenization of the input prompt, and the iterative process of token generation. It leverages several external components, such as "llama" for model operations and "common" for parameter parsing and initialization. The program defines a public API for token generation, which includes functions for sampling tokens, managing n-gram data, and handling the model's key-value cache. The code is designed to be executed as a standalone application, as indicated by the presence of a [`main`](#main) function, and it provides detailed logging of the token generation process, including performance metrics and configuration parameters.
# Imports and Dependencies

---
- `arg.h`
- `common.h`
- `sampling.h`
- `log.h`
- `llama.h`
- `cstdio`
- `string`
- `vector`
- `algorithm`


# Data Structures

---
### ngram\_data<!-- {{#data_structure:ngram_data}} -->
- **Type**: `struct`
- **Members**:
    - `active`: Indicates whether the n-gram data is active or not, initialized to false.
    - `seq_id`: Stores the sequence identifier, initialized to -1.
    - `i_batch`: A vector of integers representing batch indices.
    - `tokens`: A vector of llama_token objects representing the tokens in the n-gram.
- **Description**: The `ngram_data` struct is designed to encapsulate information related to n-grams in a sequence processing context. It includes a boolean flag `active` to indicate if the n-gram is currently in use, a `seq_id` to identify the sequence, and two vectors: `i_batch` for batch indices and `tokens` for storing the actual tokens of the n-gram. This structure is likely used in conjunction with other components to manage and process n-grams efficiently in a larger system.


---
### ngram\_container<!-- {{#data_structure:ngram_container}} -->
- **Type**: `struct`
- **Members**:
    - `n_total`: An integer representing the total number of n-grams stored.
    - `cnt`: A vector of integers that tracks the count of n-grams for each vocabulary token.
    - `head`: A vector of integers that indicates the current head position in the ring-buffer for each vocabulary token.
    - `tokens`: A vector of llama_token that stores n-grams in a ring-buffer format for each vocabulary token.
- **Description**: The `ngram_container` struct is designed to manage and store n-grams for a given vocabulary. It initializes vectors to keep track of the count and head position of n-grams for each token in the vocabulary. The `tokens` vector acts as a ring-buffer, storing n-grams of size N-1 for each token, with a capacity defined by G. This structure is useful for applications that require efficient storage and retrieval of n-grams, such as language modeling or text generation tasks.
- **Member Functions**:
    - [`ngram_container::ngram_container`](#ngram_containerngram_container)

**Methods**

---
#### ngram\_container::ngram\_container<!-- {{#callable:ngram_container::ngram_container}} -->
The `ngram_container` constructor initializes vectors to store counts, heads, and tokens for n-grams based on the given vocabulary size, n-gram size, and capacity.
- **Inputs**:
    - `n_vocab`: The number of unique tokens in the vocabulary.
    - `N`: The size of the n-grams to be managed.
    - `G`: The capacity of the ring-buffer for storing n-grams.
- **Control Flow**:
    - The constructor initializes the `cnt` vector to have a size equal to `n_vocab`, which will store counts for each vocabulary token.
    - The `head` vector is also initialized to have a size equal to `n_vocab`, which will track the head position for each token's n-gram buffer.
    - The `tokens` vector is resized to accommodate `n_vocab * G * (N - 1)` elements, which will store the n-grams for each token in a ring-buffer format.
- **Output**: The constructor does not return any value; it initializes the internal state of the `ngram_container` object.
- **See also**: [`ngram_container`](#ngram_container)  (Data Structure)



# Functions

---
### main<!-- {{#callable:main}} -->
The `main` function initializes and runs a language model using lookahead decoding, processing input tokens and generating output tokens while managing n-grams and cache sequences.
- **Inputs**:
    - `argc`: The number of command-line arguments.
    - `argv`: An array of command-line argument strings.
- **Control Flow**:
    - Initialize common parameters and parse command-line arguments.
    - Initialize the llama backend and NUMA settings based on parameters.
    - Load the target model and obtain the vocabulary.
    - Tokenize the input prompt and check if it exceeds the maximum token list size.
    - Log the tokenized input and initialize timing for encoding.
    - Evaluate the prompt using llama_decode and copy key-value sequences.
    - Initialize variables for prediction and acceptance counts, and set up batch processing.
    - Sample the first token and log it, then enter a loop for token generation.
    - Within the loop, clear and add tokens to the batch, including verification n-grams and lookahead tokens.
    - Decode the batch and handle errors, then sample and accept the next token.
    - Log the token and check for end-of-sequence conditions.
    - Update lookahead tokens and observed n-grams, managing cache sequences based on verification results.
    - Break the loop if prediction limits are reached or end-of-sequence is detected.
    - Log performance metrics and free resources before exiting.
- **Output**: Returns 0 on successful execution, or 1 if an error occurs during parameter parsing or decoding.
- **Functions called**:
    - [`common_params_parse`](../../common/arg.cpp.driver.md#common_params_parse)
    - [`llama_backend_init`](../../src/llama.cpp.driver.md#llama_backend_init)
    - [`llama_numa_init`](../../src/llama.cpp.driver.md#llama_numa_init)
    - [`common_sampler_init`](../../common/sampling.cpp.driver.md#common_sampler_init)
    - [`common_sampler_sample`](../../common/sampling.cpp.driver.md#common_sampler_sample)
    - [`common_sampler_accept`](../../common/sampling.cpp.driver.md#common_sampler_accept)
    - [`common_perf_print`](../../common/sampling.cpp.driver.md#common_perf_print)
    - [`common_sampler_free`](../../common/sampling.cpp.driver.md#common_sampler_free)
    - [`llama_backend_free`](../../src/llama.cpp.driver.md#llama_backend_free)


