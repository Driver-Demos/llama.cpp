# Purpose
This C++ source code file is primarily focused on implementing a sampling mechanism for a language model, specifically using a structure called `common_sampler`. The file includes a custom implementation of a [`ring_buffer`](#ring_bufferring_buffer) template class, which is a fixed-capacity circular buffer used to manage a sequence of tokens efficiently. The `common_sampler` structure is central to the file, encapsulating various sampling parameters and methods to interact with a language model's vocabulary and logits. It provides functionality to initialize, clone, reset, and free samplers, as well as to sample and accept tokens based on specified constraints and parameters.

The code defines a comprehensive API for managing and applying different sampling strategies, such as top-k, top-p, and temperature-based sampling, among others. It integrates with a language model's context to set logits and apply grammar constraints, ensuring that sampled tokens adhere to specified rules. The file also includes utility functions for converting sampler types to strings or characters and for handling alternative naming conventions. This code is intended to be part of a larger system, likely a library, that provides advanced token sampling capabilities for natural language processing tasks, and it is designed to be imported and used by other components within the system.
# Imports and Dependencies

---
- `sampling.h`
- `common.h`
- `log.h`
- `cmath`
- `unordered_map`
- `algorithm`


# Data Structures

---
### ring\_buffer<!-- {{#data_structure:ring_buffer}} -->
- **Type**: `struct`
- **Members**:
    - `capacity`: Stores the maximum number of elements the ring buffer can hold.
    - `sz`: Tracks the current number of elements in the ring buffer.
    - `first`: Index of the first element in the ring buffer.
    - `pos`: Index where the next element will be inserted in the ring buffer.
    - `data`: Vector that holds the elements of the ring buffer.
- **Description**: The `ring_buffer` is a templated data structure that implements a circular buffer with a fixed capacity, allowing for efficient insertion and removal of elements in a FIFO manner. It maintains a vector to store elements and uses indices to track the start and end of the buffer, automatically overwriting the oldest data when the buffer is full. This structure is useful for scenarios where a fixed-size buffer is needed, such as in streaming data applications.
- **Member Functions**:
    - [`ring_buffer::ring_buffer`](../src/llama-sampling.cpp.driver.md#ring_bufferring_buffer)
    - [`ring_buffer::front`](../src/llama-sampling.cpp.driver.md#ring_bufferfront)
    - [`ring_buffer::front`](../src/llama-sampling.cpp.driver.md#ring_bufferfront)
    - [`ring_buffer::back`](../src/llama-sampling.cpp.driver.md#ring_bufferback)
    - [`ring_buffer::back`](../src/llama-sampling.cpp.driver.md#ring_bufferback)
    - [`ring_buffer::push_back`](../src/llama-sampling.cpp.driver.md#ring_bufferpush_back)
    - [`ring_buffer::pop_front`](../src/llama-sampling.cpp.driver.md#ring_bufferpop_front)
    - [`ring_buffer::rat`](../src/llama-sampling.cpp.driver.md#ring_bufferrat)
    - [`ring_buffer::to_vector`](../src/llama-sampling.cpp.driver.md#ring_bufferto_vector)
    - [`ring_buffer::clear`](../src/llama-sampling.cpp.driver.md#ring_bufferclear)
    - [`ring_buffer::empty`](../src/llama-sampling.cpp.driver.md#ring_bufferempty)
    - [`ring_buffer::size`](../src/llama-sampling.cpp.driver.md#ring_buffersize)
    - [`ring_buffer::ring_buffer`](#ring_bufferring_buffer)
    - [`ring_buffer::front`](#ring_bufferfront)
    - [`ring_buffer::front`](#ring_bufferfront)
    - [`ring_buffer::back`](#ring_bufferback)
    - [`ring_buffer::back`](#ring_bufferback)
    - [`ring_buffer::push_back`](#ring_bufferpush_back)
    - [`ring_buffer::pop_front`](#ring_bufferpop_front)
    - [`ring_buffer::rat`](#ring_bufferrat)
    - [`ring_buffer::to_vector`](#ring_bufferto_vector)
    - [`ring_buffer::clear`](#ring_bufferclear)
    - [`ring_buffer::empty`](#ring_bufferempty)
    - [`ring_buffer::size`](#ring_buffersize)

**Methods**

---
#### ring\_buffer::ring\_buffer<!-- {{#callable:ring_buffer::ring_buffer}} -->
The `ring_buffer` constructor initializes a ring buffer with a specified capacity.
- **Inputs**:
    - `cap`: The capacity of the ring buffer, which determines the maximum number of elements it can hold.
- **Control Flow**:
    - The constructor initializes the `capacity` member variable with the provided `cap` value.
    - It initializes the `data` member variable as a vector with a size equal to `cap`.
- **Output**: A `ring_buffer` object with initialized capacity and data storage.
- **See also**: [`ring_buffer`](#ring_buffer)  (Data Structure)


---
#### ring\_buffer::front<!-- {{#callable:ring_buffer::front}} -->
The `front` function returns a reference to the first element in the ring buffer, throwing an exception if the buffer is empty.
- **Inputs**: None
- **Control Flow**:
    - Check if the size of the ring buffer (`sz`) is zero.
    - If the buffer is empty, throw a `std::runtime_error` with the message "ring buffer is empty".
    - Return a reference to the element at the `first` index of the `data` vector.
- **Output**: A reference to the first element in the ring buffer.
- **See also**: [`ring_buffer`](#ring_buffer)  (Data Structure)


---
#### ring\_buffer::front<!-- {{#callable:ring_buffer::front}} -->
The `front` function returns a reference to the first element in the ring buffer, throwing an exception if the buffer is empty.
- **Inputs**: None
- **Control Flow**:
    - Check if the size of the ring buffer (`sz`) is zero.
    - If the buffer is empty, throw a `std::runtime_error` with the message "ring buffer is empty".
    - If the buffer is not empty, return a reference to the element at the `first` index in the `data` vector.
- **Output**: A constant reference to the first element in the ring buffer.
- **See also**: [`ring_buffer`](#ring_buffer)  (Data Structure)


---
#### ring\_buffer::back<!-- {{#callable:ring_buffer::back}} -->
The `back` function returns a reference to the last element in the ring buffer, throwing an exception if the buffer is empty.
- **Inputs**: None
- **Control Flow**:
    - Check if the size of the ring buffer (`sz`) is zero.
    - If the buffer is empty, throw a `std::runtime_error` with the message "ring buffer is empty".
    - Return a reference to the element at the current position (`pos`) in the data vector.
- **Output**: A reference to the last element in the ring buffer.
- **See also**: [`ring_buffer`](#ring_buffer)  (Data Structure)


---
#### ring\_buffer::back<!-- {{#callable:ring_buffer::back}} -->
The `back` function returns the last element in the ring buffer, throwing an exception if the buffer is empty.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - Check if the size of the ring buffer (`sz`) is zero.
    - If the buffer is empty, throw a `std::runtime_error` with the message "ring buffer is empty".
    - If the buffer is not empty, return the element at the current position (`pos`) in the data vector.
- **Output**: A constant reference to the last element in the ring buffer.
- **See also**: [`ring_buffer`](#ring_buffer)  (Data Structure)


---
#### ring\_buffer::push\_back<!-- {{#callable:ring_buffer::push_back}} -->
The `push_back` function adds a new element to the ring buffer, managing the buffer's capacity by overwriting the oldest element if necessary.
- **Inputs**:
    - `value`: A constant reference to the element of type `T` to be added to the ring buffer.
- **Control Flow**:
    - Check if the current size `sz` is equal to the buffer's capacity.
    - If the buffer is full, increment the `first` index to overwrite the oldest element.
    - If the buffer is not full, increment the size `sz`.
    - Assign the new value to the current position `pos` in the buffer.
    - Increment the `pos` index, wrapping around using modulo operation with the capacity.
- **Output**: The function does not return a value; it modifies the ring buffer in place.
- **See also**: [`ring_buffer`](#ring_buffer)  (Data Structure)


---
#### ring\_buffer::pop\_front<!-- {{#callable:ring_buffer::pop_front}} -->
The `pop_front` function removes and returns the first element from a ring buffer, updating the buffer's state accordingly.
- **Inputs**: None
- **Control Flow**:
    - Check if the buffer size `sz` is zero; if so, throw a `std::runtime_error` indicating the buffer is empty.
    - Retrieve the value at the `first` index of the `data` vector and store it in `value`.
    - Update the `first` index to the next position in the buffer using modulo arithmetic with `capacity`.
    - Decrement the buffer size `sz` by one.
    - Return the retrieved `value`.
- **Output**: The function returns the element of type `T` that was at the front of the ring buffer.
- **See also**: [`ring_buffer`](#ring_buffer)  (Data Structure)


---
#### ring\_buffer::rat<!-- {{#callable:ring_buffer::rat}} -->
The `rat` function retrieves an element from the ring buffer at a specified reverse index, throwing an error if the index is out of bounds.
- **Inputs**:
    - `i`: A size_t index representing the reverse position from the end of the buffer to retrieve the element.
- **Control Flow**:
    - Check if the input index `i` is greater than or equal to the current size `sz` of the buffer; if so, throw a runtime error indicating an out-of-bounds access.
    - Calculate the actual index in the buffer using the formula `(first + sz - i - 1) % capacity`, which accounts for the circular nature of the buffer.
    - Return the element at the calculated index from the `data` vector.
- **Output**: A constant reference to the element of type `T` at the specified reverse index in the ring buffer.
- **See also**: [`ring_buffer`](#ring_buffer)  (Data Structure)


---
#### ring\_buffer::to\_vector<!-- {{#callable:ring_buffer::to_vector}} -->
The `to_vector` function converts the contents of a `ring_buffer` into a standard `std::vector`.
- **Inputs**: None
- **Control Flow**:
    - Initialize an empty `std::vector` named `result` to store the elements of the ring buffer.
    - Reserve space in `result` for `sz` elements to optimize memory allocation.
    - Iterate over the range from 0 to `sz`, using a loop variable `i`.
    - For each iteration, calculate the index `(first + i) % capacity` to access the correct element in the circular buffer.
    - Push the accessed element from `data` into the `result` vector.
    - Return the `result` vector containing all elements from the ring buffer in order.
- **Output**: A `std::vector<T>` containing the elements of the `ring_buffer` in the order they appear from `first` to `pos`.
- **See also**: [`ring_buffer`](#ring_buffer)  (Data Structure)


---
#### ring\_buffer::clear<!-- {{#callable:ring_buffer::clear}} -->
The `clear` function resets the state of the ring buffer by setting its size, first position, and current position to zero.
- **Inputs**: None
- **Control Flow**:
    - The function sets the `sz` member variable to 0, indicating the buffer is empty.
    - The `first` member variable is set to 0, resetting the starting point of the buffer.
    - The `pos` member variable is set to 0, resetting the current position in the buffer.
- **Output**: The function does not return any value; it modifies the internal state of the ring buffer object.
- **See also**: [`ring_buffer`](#ring_buffer)  (Data Structure)


---
#### ring\_buffer::empty<!-- {{#callable:ring_buffer::empty}} -->
The `empty` function checks if the ring buffer is empty by verifying if its size is zero.
- **Inputs**: None
- **Control Flow**:
    - The function returns the result of the comparison `sz == 0`, which checks if the size of the ring buffer is zero.
- **Output**: A boolean value indicating whether the ring buffer is empty (true if empty, false otherwise).
- **See also**: [`ring_buffer`](#ring_buffer)  (Data Structure)


---
#### ring\_buffer::size<!-- {{#callable:ring_buffer::size}} -->
The `size` function returns the current number of elements in the ring buffer.
- **Inputs**: None
- **Control Flow**:
    - The function simply returns the value of the member variable `sz`, which represents the current size of the ring buffer.
- **Output**: The function returns a `size_t` value representing the number of elements currently stored in the ring buffer.
- **See also**: [`ring_buffer`](#ring_buffer)  (Data Structure)



---
### common\_sampler<!-- {{#data_structure:common_sampler}} -->
- **Type**: `struct`
- **Members**:
    - `params`: Holds the sampling parameters for the common_sampler.
    - `grmr`: Pointer to a llama_sampler used for grammar sampling.
    - `chain`: Pointer to a llama_sampler used for chaining multiple samplers.
    - `prev`: A ring buffer storing previously sampled llama_tokens.
    - `cur`: A vector holding current llama_token_data for sampling.
    - `cur_p`: An array structure holding current token data for sampling operations.
- **Description**: The `common_sampler` struct is designed to facilitate token sampling in a language model context, utilizing various sampling strategies and parameters. It integrates multiple samplers, including grammar and chain samplers, to manage and apply different sampling techniques. The struct maintains a history of previously sampled tokens using a ring buffer and manages current token data for sampling operations. It is equipped with methods to set logits, sample tokens, and manage the sampling chain, making it a versatile component for language model sampling tasks.
- **Member Functions**:
    - [`common_sampler::set_logits`](#common_samplerset_logits)

**Methods**

---
#### common\_sampler::set\_logits<!-- {{#callable:common_sampler::set_logits}} -->
The `set_logits` function initializes the current token data array with logits for a given index in a llama context.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure, which contains the context for the llama model.
    - `idx`: An integer representing the index for which the logits are to be set.
- **Control Flow**:
    - Retrieve the logits for the given index from the llama context using `llama_get_logits_ith`.
    - Get the model and vocabulary from the llama context using `llama_get_model` and `llama_model_get_vocab`, respectively.
    - Determine the number of tokens in the vocabulary using `llama_vocab_n_tokens`.
    - Resize the `cur` vector to match the number of tokens in the vocabulary.
    - Iterate over each token in the vocabulary, initializing each entry in the `cur` vector with a `llama_token_data` structure containing the token ID, its corresponding logit, and a default value of 0.0f for the third field.
    - Initialize `cur_p` with the data from `cur`, its size, and default values for the other fields.
- **Output**: The function does not return a value; it modifies the `cur` vector and `cur_p` member of the `common_sampler` structure.
- **See also**: [`common_sampler`](#common_sampler)  (Data Structure)



---
### common\_params\_sampling<!-- {{#data_structure:common_params_sampling}} -->
- **Description**: [See definition](common.h.driver.md#common_params_sampling)
- **Member Functions**:
    - [`common_params_sampling::print`](#common_params_samplingprint)

**Methods**

---
#### common\_params\_sampling::print<!-- {{#callable:common_params_sampling::print}} -->
The `print` function in the `common_params_sampling` struct formats and returns a string representation of its sampling parameters.
- **Inputs**: None
- **Control Flow**:
    - A character array `result` of size 1024 is declared to hold the formatted string.
    - The `snprintf` function is used to format the sampling parameters into the `result` array, using various member variables of the `common_params_sampling` struct.
    - The formatted string includes parameters such as `repeat_last_n`, `repeat_penalty`, `frequency_penalty`, `presence_penalty`, `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_penalty_last_n`, `top_k`, `top_p`, `min_p`, `xtc_probability`, `xtc_threshold`, `typical_p`, `top_n_sigma`, `temp`, `mirostat`, `mirostat_lr`, and `mirostat_ent`.
    - The function returns a `std::string` constructed from the `result` character array.
- **Output**: A `std::string` containing a formatted representation of the sampling parameters.
- **See also**: [`common_params_sampling`](common.h.driver.md#common_params_sampling)  (Data Structure)



# Functions

---
### common\_sampler\_init<!-- {{#callable:common_sampler_init}} -->
The `common_sampler_init` function initializes a `common_sampler` structure with specified sampling parameters and a model, setting up grammar and sampling chains based on the provided configuration.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure representing the model to be used for sampling.
    - `params`: A reference to a `common_params_sampling` structure containing various parameters for sampling, such as grammar settings, sampler types, and other configuration options.
- **Control Flow**:
    - Retrieve the vocabulary from the provided model using `llama_model_get_vocab`.
    - Initialize default parameters for the sampler chain using [`llama_sampler_chain_default_params`](../src/llama.cpp.driver.md#llama_sampler_chain_default_params).
    - Set the `no_perf` parameter in the chain parameters based on the input `params`.
    - Check if the grammar in `params` starts with '%llguidance' and initialize the grammar sampler accordingly, using `llama_sampler_init_llg` if `LLAMA_USE_LLGUIDANCE` is enabled.
    - If the grammar does not start with '%llguidance', process grammar triggers to create regex patterns and tokens, then initialize the grammar sampler using `llama_sampler_init_grammar_lazy_patterns` or `llama_sampler_init_grammar`.
    - Create a new `common_sampler` object, initializing its fields with the provided parameters, the grammar sampler, and a new sampler chain.
    - Add a logit bias sampler to the chain using `llama_sampler_init_logit_bias`.
    - Depending on the `mirostat` parameter, add different samplers to the chain based on the types specified in `params.samplers`, using functions like `llama_sampler_init_dry`, `llama_sampler_init_top_k`, etc.
    - Return the initialized `common_sampler` object.
- **Output**: A pointer to a newly initialized `common_sampler` structure, or `nullptr` if the grammar initialization fails.
- **Functions called**:
    - [`llama_sampler_chain_default_params`](../src/llama.cpp.driver.md#llama_sampler_chain_default_params)


---
### common\_sampler\_free<!-- {{#callable:common_sampler_free}} -->
The `common_sampler_free` function deallocates memory and resources associated with a `common_sampler` object.
- **Inputs**:
    - `gsmpl`: A pointer to a `common_sampler` structure that needs to be freed.
- **Control Flow**:
    - Check if the `gsmpl` pointer is not null.
    - If `gsmpl` is valid, call `llama_sampler_free` on `gsmpl->grmr` to free the grammar sampler.
    - Call `llama_sampler_free` on `gsmpl->chain` to free the chain sampler.
    - Delete the `gsmpl` object to free its memory.
- **Output**: The function does not return any value; it performs cleanup operations on the provided `common_sampler` object.


---
### common\_sampler\_accept<!-- {{#callable:common_sampler_accept}} -->
The `common_sampler_accept` function accepts a token into the grammar and chain samplers of a `common_sampler` structure and appends the token to the previous tokens buffer.
- **Inputs**:
    - `gsmpl`: A pointer to a `common_sampler` structure which contains the grammar and chain samplers, as well as a buffer for previous tokens.
    - `token`: A `llama_token` that is to be accepted by the samplers and added to the previous tokens buffer.
    - `accept_grammar`: A boolean flag indicating whether the token should be accepted by the grammar sampler.
- **Control Flow**:
    - If `accept_grammar` is true, the function calls `llama_sampler_accept` on the grammar sampler (`gsmpl->grmr`) with the given token.
    - The function calls `llama_sampler_accept` on the chain sampler (`gsmpl->chain`) with the given token.
    - The token is appended to the `prev` ring buffer of the `common_sampler` structure.
- **Output**: This function does not return any value; it modifies the state of the `common_sampler` structure by updating its samplers and previous tokens buffer.


---
### common\_sampler\_reset<!-- {{#callable:common_sampler_reset}} -->
The `common_sampler_reset` function resets the state of the grammar and chain samplers within a `common_sampler` structure.
- **Inputs**:
    - `gsmpl`: A pointer to a `common_sampler` structure whose grammar and chain samplers need to be reset.
- **Control Flow**:
    - Call `llama_sampler_reset` on the `grmr` member of the `common_sampler` structure pointed to by `gsmpl`.
    - Call `llama_sampler_reset` on the `chain` member of the `common_sampler` structure pointed to by `gsmpl`.
- **Output**: This function does not return any value.


---
### common\_sampler\_clone<!-- {{#callable:common_sampler_clone}} -->
The `common_sampler_clone` function creates a new `common_sampler` object by copying the properties of an existing `common_sampler` object.
- **Inputs**:
    - `gsmpl`: A pointer to an existing `common_sampler` object that is to be cloned.
- **Control Flow**:
    - The function initializes a new `common_sampler` object using the properties of the input `gsmpl`.
    - It directly copies the `params`, `prev`, `cur`, and `cur_p` fields from `gsmpl`.
    - The `grmr` and `chain` fields are cloned using the `llama_sampler_clone` function.
- **Output**: A pointer to a newly created `common_sampler` object that is a clone of the input `gsmpl`.


---
### common\_perf\_print<!-- {{#callable:common_perf_print}} -->
The `common_perf_print` function prints performance metrics for a given llama context and common sampler if they are provided.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure, which contains context-specific data for performance measurement.
    - `gsmpl`: A pointer to a `common_sampler` structure, which contains sampling-related data for performance measurement.
- **Control Flow**:
    - Check if `gsmpl` is not null; if true, call `llama_perf_sampler_print` with `gsmpl->chain` to print sampler performance metrics.
    - Check if `ctx` is not null; if true, call `llama_perf_context_print` with `ctx` to print context performance metrics.
- **Output**: This function does not return any value; it performs output operations by printing performance metrics.


---
### common\_sampler\_sample<!-- {{#callable:common_sampler_sample}} -->
The `common_sampler_sample` function samples a token from a given context using a specified grammar and sampling chain, with optional grammar-first processing and resampling if necessary.
- **Inputs**:
    - `gsmpl`: A pointer to a `common_sampler` structure that contains the grammar and sampling chain configurations.
    - `ctx`: A pointer to a `llama_context` structure that provides the context for sampling.
    - `idx`: An integer index used to set the logits for the current sampling context.
    - `grammar_first`: A boolean flag indicating whether to apply the grammar sampler before the sampling chain.
- **Control Flow**:
    - Set the logits for the current sampling context using `gsmpl->set_logits(ctx, idx)`.
    - Retrieve references to the grammar sampler (`grmr`), sampling chain (`chain`), and current token data array (`cur_p`).
    - If `grammar_first` is true, apply the grammar sampler to `cur_p`.
    - Apply the sampling chain to `cur_p`.
    - Assert that a token has been selected (`cur_p.selected != -1`).
    - Retrieve the ID of the selected token from `cur_p`.
    - If `grammar_first` is true, return the selected token ID.
    - Create a single token data array with the selected token ID and apply the grammar sampler to it.
    - Check if the token is valid by verifying its logit is not negative infinity.
    - If the token is valid, return the token ID.
    - If the token is not valid, reset the logits and reapply the grammar sampler and sampling chain.
    - Assert that a token has been selected during resampling.
    - Return the ID of the newly selected token.
- **Output**: Returns a `llama_token` which is the ID of the sampled token, either from the initial sampling or after resampling if necessary.


---
### common\_sampler\_sample\_and\_accept\_n<!-- {{#callable:common_sampler_sample_and_accept_n}} -->
The function [`common_sampler_sample_and_accept_n`](#common_sampler_sample_and_accept_n) samples and accepts a sequence of tokens using a common sampler, given a draft sequence and a grammar-first flag.
- **Inputs**:
    - `gsmpl`: A pointer to a `common_sampler` structure, which contains sampling parameters and state.
    - `ctx`: A pointer to a `llama_context` structure, which provides context for the sampling process.
    - `draft`: A `llama_tokens` vector representing the draft sequence of tokens to be sampled and accepted.
    - `grammar_first`: A boolean flag indicating whether grammar should be applied before the sampling chain.
- **Control Flow**:
    - Initialize a vector `idxs` with indices from 0 to `draft.size()` inclusive.
    - Call the overloaded [`common_sampler_sample_and_accept_n`](#common_sampler_sample_and_accept_n) function with `gsmpl`, `ctx`, `idxs`, `draft`, and `grammar_first` as arguments.
    - Return the result of the called function.
- **Output**: A vector of `llama_token` representing the sampled and accepted tokens.
- **Functions called**:
    - [`common_sampler_sample_and_accept_n`](#common_sampler_sample_and_accept_n)


---
### common\_sampler\_get\_seed<!-- {{#callable:common_sampler_get_seed}} -->
The function `common_sampler_get_seed` retrieves the seed value from a `common_sampler` structure by accessing its `chain` member.
- **Inputs**:
    - `gsmpl`: A pointer to a `common_sampler` structure from which the seed value is to be retrieved.
- **Control Flow**:
    - The function calls `llama_sampler_get_seed` with the `chain` member of the `common_sampler` structure pointed to by `gsmpl`.
- **Output**: Returns a `uint32_t` representing the seed value obtained from the `chain` member of the `common_sampler` structure.


---
### common\_sampler\_get\_candidates<!-- {{#callable:common_sampler_get_candidates}} -->
The function `common_sampler_get_candidates` returns a pointer to the current token data array of a given `common_sampler` structure.
- **Inputs**:
    - `gsmpl`: A pointer to a `common_sampler` structure from which the current token data array is to be retrieved.
- **Control Flow**:
    - The function accesses the `cur_p` member of the `common_sampler` structure pointed to by `gsmpl`.
    - It returns the address of the `cur_p` member.
- **Output**: A pointer to a `llama_token_data_array` which represents the current token data array of the `common_sampler`.


---
### common\_sampler\_last<!-- {{#callable:common_sampler_last}} -->
The function `common_sampler_last` retrieves the most recent token from the `prev` ring buffer of a `common_sampler` structure.
- **Inputs**:
    - `gsmpl`: A pointer to a `common_sampler` structure, which contains a ring buffer of previously sampled tokens.
- **Control Flow**:
    - Access the `prev` ring buffer within the `common_sampler` structure pointed to by `gsmpl`.
    - Call the `rat` method on the `prev` ring buffer with an index of 0 to retrieve the most recent token.
- **Output**: Returns the most recent `llama_token` from the `prev` ring buffer of the `common_sampler`.


---
### common\_sampler\_print<!-- {{#callable:common_sampler_print}} -->
The `common_sampler_print` function generates a string representation of the sampler chain in a `common_sampler` structure.
- **Inputs**:
    - `gsmpl`: A pointer to a `common_sampler` structure, which contains the sampler chain to be printed.
- **Control Flow**:
    - Initialize a string `result` with the value "logits ".
    - Iterate over each sampler in the sampler chain of `gsmpl` using `llama_sampler_chain_n` to get the number of samplers.
    - For each sampler, retrieve it using `llama_sampler_chain_get` and append its name, obtained via `llama_sampler_name`, to `result` with a "-> " prefix.
    - Return the constructed `result` string.
- **Output**: A `std::string` representing the sequence of sampler names in the `common_sampler` chain, prefixed by "logits ".


---
### common\_sampler\_prev\_str<!-- {{#callable:common_sampler_prev_str}} -->
The function `common_sampler_prev_str` generates a string representation of the last `n` tokens from a `common_sampler`'s history, converting each token to its corresponding piece using a given context.
- **Inputs**:
    - `gsmpl`: A pointer to a `common_sampler` object, which contains the history of previously sampled tokens.
    - `ctx_main`: A pointer to a `llama_context` object, used to convert tokens to their string representations.
    - `n`: An integer specifying the number of recent tokens to include in the output string.
- **Control Flow**:
    - The function first adjusts `n` to be the minimum of `n` and the size of the `prev` ring buffer in `gsmpl`.
    - If `n` is less than or equal to zero, the function returns an empty string.
    - A `result` string is initialized with a reserved size of `8*n` to optimize memory allocation, assuming an average token length of 8 characters.
    - A loop iterates from `n-1` to 0, retrieving each token from the `prev` ring buffer using the `rat` method.
    - For each token, the function asserts that it is not `LLAMA_TOKEN_NULL` to ensure valid tokens are processed.
    - Each token is converted to its string representation using `common_token_to_piece` and appended to the `result` string.
    - Finally, the function returns the constructed `result` string.
- **Output**: A `std::string` containing the concatenated string representations of the last `n` tokens from the `common_sampler`'s history.


---
### common\_sampler\_type\_to\_chr<!-- {{#callable:common_sampler_type_to_chr}} -->
The function `common_sampler_type_to_chr` converts a `common_sampler_type` enumeration value to its corresponding character representation.
- **Inputs**:
    - `cnstr`: An enumeration value of type `common_sampler_type` representing a specific sampler type.
- **Control Flow**:
    - The function uses a switch statement to match the input `cnstr` with predefined `common_sampler_type` cases.
    - For each case, it returns a specific character that represents the sampler type.
    - If the input does not match any predefined case, it returns the character '?' as a default.
- **Output**: A character that represents the input `common_sampler_type`, or '?' if the type is not recognized.


---
### common\_sampler\_type\_to\_str<!-- {{#callable:common_sampler_type_to_str}} -->
The function `common_sampler_type_to_str` converts an enumeration value of type `common_sampler_type` to its corresponding string representation.
- **Inputs**:
    - `cnstr`: An enumeration value of type `common_sampler_type` representing a specific sampler type.
- **Control Flow**:
    - The function uses a switch statement to match the input enumeration value `cnstr` with predefined cases.
    - Each case corresponds to a specific sampler type and returns a string that represents that type.
    - If the input does not match any predefined case, the function returns an empty string.
- **Output**: A string representing the name of the sampler type corresponding to the input enumeration value, or an empty string if the input is not recognized.


---
### common\_sampler\_types\_from\_names<!-- {{#callable:common_sampler_types_from_names}} -->
The function `common_sampler_types_from_names` maps a list of sampler names to their corresponding `common_sampler_type` enums, optionally allowing alternative names.
- **Inputs**:
    - `names`: A vector of strings representing the names of samplers to be converted to their corresponding `common_sampler_type` enums.
    - `allow_alt_names`: A boolean flag indicating whether alternative names for samplers should be considered in the mapping process.
- **Control Flow**:
    - Initialize a map `sampler_canonical_name_map` with canonical sampler names as keys and their corresponding `common_sampler_type` enums as values.
    - Initialize a map `sampler_alt_name_map` with alternative sampler names as keys and their corresponding `common_sampler_type` enums as values.
    - Create an empty vector `samplers` to store the resulting `common_sampler_type` enums.
    - Iterate over each name in the `names` vector.
    - For each name, check if it exists in `sampler_canonical_name_map`; if found, add the corresponding enum to `samplers` and continue to the next name.
    - If `allow_alt_names` is true and the name was not found in the canonical map, check `sampler_alt_name_map`; if found, add the corresponding enum to `samplers` and continue to the next name.
    - If a name is not found in either map, log a warning message indicating the name could not be matched.
- **Output**: A vector of `common_sampler_type` enums corresponding to the input names, with unmatched names resulting in a warning log.


---
### common\_sampler\_types\_from\_chars<!-- {{#callable:common_sampler_types_from_chars}} -->
The function `common_sampler_types_from_chars` converts a string of characters into a vector of `common_sampler_type` enums based on predefined character mappings.
- **Inputs**:
    - `chars`: A string of characters where each character represents a specific `common_sampler_type`.
- **Control Flow**:
    - Initialize an unordered map `sampler_name_map` that maps characters to their corresponding `common_sampler_type` enums.
    - Create an empty vector `samplers` to store the resulting `common_sampler_type` values, reserving space equal to the size of `chars`.
    - Iterate over each character `c` in the input string `chars`.
    - For each character, check if it exists in the `sampler_name_map`.
    - If the character is found, append the corresponding `common_sampler_type` to the `samplers` vector.
    - If the character is not found, log a warning message indicating the unmatched character.
    - Return the `samplers` vector containing the matched `common_sampler_type` values.
- **Output**: A vector of `common_sampler_type` enums corresponding to the characters in the input string.
- **Functions called**:
    - [`common_sampler_type_to_chr`](#common_sampler_type_to_chr)


