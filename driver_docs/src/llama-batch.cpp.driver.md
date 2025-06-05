# Purpose
This C++ source code file is designed to manage and manipulate batches of data, specifically focusing on sequences and tokens within a batch. The code provides functionality for reserving, splitting, and adding sequences to batches, which are represented by the `llama_ubatch` and [`llama_sbatch`](#llama_sbatchllama_sbatch) structures. The primary purpose of this code is to handle the allocation and organization of data sequences, ensuring that sequences can be efficiently managed and processed, whether they are of equal length or require more complex handling due to varying lengths or embedded data. The code includes methods for both simple and equal sequence splitting, as well as a constructor for initializing these batches with specific parameters.

The file also defines a public API for batch initialization and management, including functions like [`llama_batch_get_one`](#llama_batch_get_one), [`llama_batch_init`](#llama_batch_init), and [`llama_batch_free`](#llama_batch_free), which are used to create, initialize, and free batches, respectively. These functions provide a clear interface for external code to interact with the batch management system, allowing for the creation of batches with specified token counts and embedding dimensions. The code is structured to ensure that memory is properly allocated and freed, preventing memory leaks and ensuring efficient use of resources. Overall, this file provides a comprehensive set of tools for managing data sequences within a batch processing context, with a focus on flexibility and efficiency.
# Imports and Dependencies

---
- `llama-batch.h`
- `cassert`
- `cstring`
- `algorithm`


# Data Structures

---
### llama\_sbatch<!-- {{#data_structure:llama_sbatch}} -->
- **Description**: [See definition](llama-batch.h.driver.md#llama_sbatch)
- **Member Functions**:
    - [`llama_sbatch::reserve_ubatch`](#llama_sbatchreserve_ubatch)
    - [`llama_sbatch::add_seq_to_ubatch`](#llama_sbatchadd_seq_to_ubatch)
    - [`llama_sbatch::split_simple`](#llama_sbatchsplit_simple)
    - [`llama_sbatch::split_equal`](#llama_sbatchsplit_equal)
    - [`llama_sbatch::split_seq`](#llama_sbatchsplit_seq)
    - [`llama_sbatch::llama_sbatch`](#llama_sbatchllama_sbatch)
    - [`llama_sbatch::llama_sbatch`](llama-batch.h.driver.md#llama_sbatchllama_sbatch)

**Methods**

---
#### llama\_sbatch::reserve\_ubatch<!-- {{#callable:llama_sbatch::reserve_ubatch}} -->
The `reserve_ubatch` function prepares and returns a new `llama_ubatch` structure by clearing empty sequences and resizing necessary data structures based on the input parameters.
- **Inputs**:
    - `n_ubatch`: The number of ubatch elements to reserve.
    - `has_embd`: A boolean indicating whether the ubatch should include embeddings.
- **Control Flow**:
    - Iterate over the `seq` vector in reverse to remove any sequences with zero length.
    - Add a new `ubatch_data` structure to the `udatas` vector.
    - Resize the `token`, `embd`, `pos`, `n_seq_id`, `seq_id`, and `output` vectors in the newly added `ubatch_data` based on `n_ubatch` and `has_embd`.
    - Initialize a `llama_ubatch` structure with pointers to the resized vectors and set initial values for its fields.
    - Return the initialized `llama_ubatch`.
- **Output**: A `llama_ubatch` structure initialized with the reserved data.
- **See also**: [`llama_sbatch`](llama-batch.h.driver.md#llama_sbatch)  (Data Structure)


---
#### llama\_sbatch::add\_seq\_to\_ubatch<!-- {{#callable:llama_sbatch::add_seq_to_ubatch}} -->
The `add_seq_to_ubatch` function adds a sequence of tokens and associated data from a source batch to a target micro-batch, ensuring compatibility and updating relevant metadata.
- **Inputs**:
    - `ubatch`: A reference to a `llama_ubatch` object where the sequence data will be added.
    - `seq`: A reference to a `llama_sbatch_seq` object representing the sequence to be added to the micro-batch.
    - `length`: The number of tokens from the sequence to add to the micro-batch.
- **Control Flow**:
    - Assert that the source batch is not null and the length is valid for the sequence.
    - Check that sequences of equal lengths are added to the batch, ensuring token-sequence clarity.
    - If the source batch has tokens, copy them to the micro-batch based on whether sequences are equal or not.
    - If the source batch has embeddings, copy them similarly to the micro-batch.
    - Copy position data from the source batch to the micro-batch, handling equal and unequal sequences differently.
    - Update sequence IDs in the micro-batch, handling equal and unequal sequences differently.
    - Handle logits by setting output flags and updating the output IDs list based on the logits configuration.
    - Update the number of tokens and sequences in the micro-batch, adjusting for equal or simple split sequences.
    - Adjust the sequence's offset and length, and decrease the total number of tokens in the source batch.
    - Assert that the total number of tokens in the micro-batch matches the expected count.
- **Output**: The function updates the `ubatch` with the sequence data, modifies the sequence's offset and length, and adjusts the total token count in the source batch.
- **See also**: [`llama_sbatch`](llama-batch.h.driver.md#llama_sbatch)  (Data Structure)


---
#### llama\_sbatch::split\_simple<!-- {{#callable:llama_sbatch::split_simple}} -->
The `split_simple` function creates a new `llama_ubatch` by splitting the first sequence in the `llama_sbatch` into a specified number of tokens, ensuring that the sequence is not mixed with other splits.
- **Inputs**:
    - `n_ubatch`: The maximum number of tokens to include in the new ubatch, constrained by the number of tokens available in the batch.
- **Control Flow**:
    - Determine the actual number of tokens to use for the ubatch by taking the minimum of `n_tokens` and `n_ubatch`.
    - Call [`reserve_ubatch`](#llama_sbatchreserve_ubatch) to allocate a new `llama_ubatch` with the determined number of tokens and check if embeddings are present.
    - Set `equal_seqs` to false for the new ubatch, indicating that sequences of unequal lengths are allowed.
    - Check if the sequence vector `seq` is not empty, and if so, proceed to process the first sequence.
    - Retrieve the first sequence `s` from `seq` and determine the length to use, which is the minimum of `s.length` and `n_ubatch`.
    - Assert that there is only one sequence in `seq` and that it has not been split before (`n_seq_id` is 0).
    - Add the sequence `s` to the ubatch using the [`add_seq_to_ubatch`](#llama_sbatchadd_seq_to_ubatch) function with the determined length.
    - Return the newly created `llama_ubatch`.
- **Output**: A `llama_ubatch` object containing the split sequence data.
- **Functions called**:
    - [`llama_sbatch::reserve_ubatch`](#llama_sbatchreserve_ubatch)
    - [`llama_sbatch::add_seq_to_ubatch`](#llama_sbatchadd_seq_to_ubatch)
- **See also**: [`llama_sbatch`](llama-batch.h.driver.md#llama_sbatch)  (Data Structure)


---
#### llama\_sbatch::split\_equal<!-- {{#callable:llama_sbatch::split_equal}} -->
The `split_equal` function creates a new `llama_ubatch` by splitting sequences into equal-length sub-batches, ensuring that each sub-batch does not exceed a specified number of tokens.
- **Inputs**:
    - `n_ubatch`: The maximum number of tokens allowed in the resulting sub-batch.
- **Control Flow**:
    - The function first adjusts `n_ubatch` to be the smaller of `n_tokens` and `n_ubatch`.
    - It reserves a new `llama_ubatch` with the adjusted `n_ubatch` size, considering whether embeddings are present.
    - If the sequence list `seq` is not empty, it initializes `length` and `n_tokens_in_ubatch` to zero.
    - It asserts that the first sequence's `n_seq_id` is greater than zero, ensuring no mixing with simple splits.
    - The function iterates over the sequences in reverse order, checking each sequence's length and adjusting `length` to the smaller of the sequence's length or `n_ubatch`.
    - It adds the sequence to the `ubatch` using [`add_seq_to_ubatch`](#llama_sbatchadd_seq_to_ubatch), updating `n_tokens_in_ubatch`.
    - The loop breaks if a sequence with `n_seq_id` greater than one is encountered or if adding another sequence would exceed `n_ubatch`.
- **Output**: The function returns a `llama_ubatch` containing sequences split into equal-length sub-batches, constrained by the specified `n_ubatch` size.
- **Functions called**:
    - [`llama_sbatch::reserve_ubatch`](#llama_sbatchreserve_ubatch)
    - [`llama_sbatch::add_seq_to_ubatch`](#llama_sbatchadd_seq_to_ubatch)
- **See also**: [`llama_sbatch`](llama-batch.h.driver.md#llama_sbatch)  (Data Structure)


---
#### llama\_sbatch::split\_seq<!-- {{#callable:llama_sbatch::split_seq}} -->
The `split_seq` function creates a sub-batch of sequences from the current batch, ensuring that the sequences are split according to the specified number of tokens and sequence constraints.
- **Inputs**:
    - `n_ubatch`: The maximum number of tokens to include in the sub-batch.
- **Control Flow**:
    - The function first adjusts `n_ubatch` to be the minimum of `n_tokens` and `n_ubatch`.
    - It calls [`reserve_ubatch`](#llama_sbatchreserve_ubatch) to allocate a new sub-batch with the specified number of tokens and checks if embeddings are present.
    - If the sequence list `seq` is not empty, it retrieves the last sequence from the list.
    - It calculates the length of the sequence to be added to the sub-batch, ensuring it does not exceed `n_ubatch`.
    - An assertion checks that the sequence ID is greater than zero, ensuring it is not a simple split.
    - The function then calls [`add_seq_to_ubatch`](#llama_sbatchadd_seq_to_ubatch) to add the sequence to the sub-batch with the calculated length.
    - Finally, it returns the newly created sub-batch.
- **Output**: The function returns a `llama_ubatch` object representing the newly created sub-batch of sequences.
- **Functions called**:
    - [`llama_sbatch::reserve_ubatch`](#llama_sbatchreserve_ubatch)
    - [`llama_sbatch::add_seq_to_ubatch`](#llama_sbatchadd_seq_to_ubatch)
- **See also**: [`llama_sbatch`](llama-batch.h.driver.md#llama_sbatch)  (Data Structure)


---
#### llama\_sbatch::llama\_sbatch<!-- {{#callable:llama_sbatch::llama_sbatch}} -->
The `llama_sbatch` constructor initializes a `llama_sbatch` object by sorting and organizing tokens from a given `llama_batch` based on sequence IDs and positions, optionally using a simple split method.
- **Inputs**:
    - `batch`: A reference to a `llama_batch` object containing tokens and associated metadata.
    - `n_embd`: A size_t value representing the number of embeddings.
    - `simple_split`: A boolean flag indicating whether to use a simple split method for sequences.
    - `logits_all`: A boolean flag indicating whether to consider all logits.
- **Control Flow**:
    - Assert that the number of tokens in the batch is non-negative.
    - Initialize member variables with the provided arguments and batch data.
    - Resize the `ids` vector to match the number of tokens and populate it with indices.
    - If `simple_split` is true, create a single sequence covering all tokens and return.
    - Sort the `ids` based on sequence IDs and positions, prioritizing shared prompts.
    - Initialize sequences by iterating over sorted tokens, grouping them by sequence IDs.
    - Sort the sequences to prioritize shared prompts and then by descending length.
- **Output**: The function does not return a value; it initializes the `llama_sbatch` object with sorted and organized sequences.
- **See also**: [`llama_sbatch`](llama-batch.h.driver.md#llama_sbatch)  (Data Structure)



---
### llama\_batch\_allocr<!-- {{#data_structure:llama_batch_allocr}} -->
- **Description**: [See definition](llama-batch.h.driver.md#llama_batch_allocr)
- **Member Functions**:
    - [`llama_batch_allocr::llama_batch_allocr`](#llama_batch_allocrllama_batch_allocr)

**Methods**

---
#### llama\_batch\_allocr::llama\_batch\_allocr<!-- {{#callable:llama_batch_allocr::llama_batch_allocr}} -->
The `llama_batch_allocr` constructor initializes a `llama_batch` object with default values for position, sequence ID, and logits if they are not already set.
- **Inputs**:
    - `in_batch`: A `llama_batch` structure that contains the batch data to be initialized.
    - `p0`: A `llama_pos` value representing the starting position for the tokens in the batch.
- **Control Flow**:
    - The function begins by assigning the input batch to the member variable `batch` and asserts that the number of tokens in the batch is greater than zero.
    - If the `pos` field of the batch is not set, it initializes the `pos` vector with positions starting from `p0` and assigns it to `batch.pos`.
    - If the `n_seq_id` field of the batch is not set, it initializes the `n_seq_id` vector with the size of `seq_id_0` and assigns it to `batch.n_seq_id`.
    - If the `seq_id` field of the batch is not set, it initializes the `seq_id` vector with pointers to `seq_id_0` and assigns it to `batch.seq_id`.
    - If the `logits` field of the batch is not set, it initializes the `logits` vector with a size equal to the number of tokens and sets the last element to true, then assigns it to `batch.logits`.
- **Output**: The function does not return a value; it initializes the `llama_batch` object with default values for certain fields if they are not already set.
- **See also**: [`llama_batch_allocr`](llama-batch.h.driver.md#llama_batch_allocr)  (Data Structure)



# Functions

---
### llama\_batch\_get\_one<!-- {{#callable:llama_batch_get_one}} -->
The `llama_batch_get_one` function initializes and returns a `llama_batch` structure with specified tokens and number of tokens, while other fields are set to null.
- **Inputs**:
    - `tokens`: A pointer to an array of `llama_token` which represents the tokens to be included in the batch.
    - `n_tokens`: An integer representing the number of tokens in the batch.
- **Control Flow**:
    - The function takes two parameters: a pointer to tokens and the number of tokens.
    - It returns a `llama_batch` structure initialized with the provided tokens and number of tokens.
    - All other fields in the `llama_batch` structure are set to null.
- **Output**: A `llama_batch` structure with the specified tokens and number of tokens, and other fields set to null.


---
### llama\_batch\_init<!-- {{#callable:llama_batch_init}} -->
The `llama_batch_init` function initializes a `llama_batch` structure by allocating memory for its components based on the provided parameters.
- **Inputs**:
    - `n_tokens_alloc`: The number of tokens to allocate memory for.
    - `embd`: The size of the embedding; if non-zero, memory for embeddings is allocated.
    - `n_seq_max`: The maximum number of sequences to allocate memory for each token.
- **Control Flow**:
    - Initialize a `llama_batch` structure with default values, setting all pointers to `nullptr` and `n_tokens` to 0.
    - Check if `embd` is non-zero; if so, allocate memory for embeddings using `malloc` for `n_tokens_alloc * embd` floats.
    - If `embd` is zero, allocate memory for tokens using `malloc` for `n_tokens_alloc` `llama_token` elements.
    - Allocate memory for `pos`, `n_seq_id`, and `seq_id` arrays using `malloc`, each with a size based on `n_tokens_alloc`.
    - For each token, allocate memory for `seq_id` using `malloc` for `n_seq_max` `llama_seq_id` elements.
    - Set the last element of `seq_id` to `nullptr` to mark the end of the sequence IDs.
    - Allocate memory for `logits` using `malloc` for `n_tokens_alloc` `int8_t` elements.
    - Return the initialized `llama_batch` structure.
- **Output**: A `llama_batch` structure with allocated memory for its components based on the input parameters.


---
### llama\_batch\_free<!-- {{#callable:llama_batch_free}} -->
The `llama_batch_free` function deallocates memory for various dynamically allocated arrays within a `llama_batch` structure.
- **Inputs**:
    - `batch`: A `llama_batch` structure containing pointers to dynamically allocated arrays that need to be freed.
- **Control Flow**:
    - Check if `batch.token` is not null and free it if so.
    - Check if `batch.embd` is not null and free it if so.
    - Check if `batch.pos` is not null and free it if so.
    - Check if `batch.n_seq_id` is not null and free it if so.
    - If `batch.seq_id` is not null, iterate through each non-null pointer in the array, freeing each one, and then free the `seq_id` array itself.
    - Check if `batch.logits` is not null and free it if so.
- **Output**: The function does not return any value; it performs memory deallocation for the input `llama_batch` structure.


