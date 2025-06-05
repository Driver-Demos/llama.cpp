# Purpose
The provided C++ source code defines a class [`llama_kv_cache_recurrent`](#llama_kv_cache_recurrentllama_kv_cache_recurrent), which is part of a larger system likely related to machine learning or neural network models, as suggested by the inclusion of terms like "model", "layer", and "embedding". This class is responsible for managing a recurrent key-value (KV) cache, which is a data structure used to store and retrieve information efficiently during the execution of a model. The cache is designed to handle sequences of data, allowing for operations such as adding, removing, copying, and keeping sequences, as well as managing their positions within the cache. The class also provides functionality for initializing, preparing, and updating the cache, as well as reading from and writing to an external I/O interface, which suggests that it supports persistence or state restoration.

The code is structured to handle multiple layers of a model, with each layer having its own set of key and value tensors. These tensors are managed using a context map that associates buffer types with contexts, allowing for efficient memory management. The class also includes methods for managing the state of the cache, such as clearing it, finding slots for new sequences, and checking the minimum and maximum positions of sequences. Additionally, the class [`llama_kv_cache_recurrent_state`](#llama_kv_cache_recurrent_statellama_kv_cache_recurrent_state) is defined to encapsulate the state of the cache, providing methods to navigate and apply updates to the cache. This code is part of a broader system, likely a library or module, that can be integrated into larger applications requiring efficient sequence management and state persistence in machine learning models.
# Imports and Dependencies

---
- `llama-kv-cache-recurrent.h`
- `llama-impl.h`
- `llama-io.h`
- `llama-batch.h`
- `llama-model.h`
- `algorithm`
- `cassert`
- `limits`
- `map`
- `stdexcept`


# Data Structures

---
### llama\_kv\_cache\_recurrent<!-- {{#data_structure:llama_kv_cache_recurrent}} -->
- **Description**: [See definition](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)
- **Member Functions**:
    - [`llama_kv_cache_recurrent::~llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrentllama_kv_cache_recurrent)
    - [`llama_kv_cache_recurrent::llama_kv_cache_recurrent`](#llama_kv_cache_recurrentllama_kv_cache_recurrent)
    - [`llama_kv_cache_recurrent::clear`](#llama_kv_cache_recurrentclear)
    - [`llama_kv_cache_recurrent::seq_rm`](#llama_kv_cache_recurrentseq_rm)
    - [`llama_kv_cache_recurrent::seq_cp`](#llama_kv_cache_recurrentseq_cp)
    - [`llama_kv_cache_recurrent::seq_keep`](#llama_kv_cache_recurrentseq_keep)
    - [`llama_kv_cache_recurrent::seq_add`](#llama_kv_cache_recurrentseq_add)
    - [`llama_kv_cache_recurrent::seq_div`](#llama_kv_cache_recurrentseq_div)
    - [`llama_kv_cache_recurrent::seq_pos_min`](#llama_kv_cache_recurrentseq_pos_min)
    - [`llama_kv_cache_recurrent::seq_pos_max`](#llama_kv_cache_recurrentseq_pos_max)
    - [`llama_kv_cache_recurrent::init_batch`](#llama_kv_cache_recurrentinit_batch)
    - [`llama_kv_cache_recurrent::init_full`](#llama_kv_cache_recurrentinit_full)
    - [`llama_kv_cache_recurrent::init_update`](#llama_kv_cache_recurrentinit_update)
    - [`llama_kv_cache_recurrent::prepare`](#llama_kv_cache_recurrentprepare)
    - [`llama_kv_cache_recurrent::find_slot`](#llama_kv_cache_recurrentfind_slot)
    - [`llama_kv_cache_recurrent::get_can_shift`](#llama_kv_cache_recurrentget_can_shift)
    - [`llama_kv_cache_recurrent::s_copy`](#llama_kv_cache_recurrents_copy)
    - [`llama_kv_cache_recurrent::s_mask`](#llama_kv_cache_recurrents_mask)
    - [`llama_kv_cache_recurrent::total_size`](#llama_kv_cache_recurrenttotal_size)
    - [`llama_kv_cache_recurrent::size_k_bytes`](#llama_kv_cache_recurrentsize_k_bytes)
    - [`llama_kv_cache_recurrent::size_v_bytes`](#llama_kv_cache_recurrentsize_v_bytes)
    - [`llama_kv_cache_recurrent::state_write`](#llama_kv_cache_recurrentstate_write)
    - [`llama_kv_cache_recurrent::state_read`](#llama_kv_cache_recurrentstate_read)
    - [`llama_kv_cache_recurrent::state_write_meta`](#llama_kv_cache_recurrentstate_write_meta)
    - [`llama_kv_cache_recurrent::state_write_data`](#llama_kv_cache_recurrentstate_write_data)
    - [`llama_kv_cache_recurrent::state_read_meta`](#llama_kv_cache_recurrentstate_read_meta)
    - [`llama_kv_cache_recurrent::state_read_data`](#llama_kv_cache_recurrentstate_read_data)
- **Inherits From**:
    - `llama_memory_i`

**Methods**

---
#### llama\_kv\_cache\_recurrent::llama\_kv\_cache\_recurrent<!-- {{#callable:llama_kv_cache_recurrent::llama_kv_cache_recurrent}} -->
The `llama_kv_cache_recurrent` constructor initializes a recurrent key-value cache for a llama model, setting up contexts and tensors for each layer based on the model's hyperparameters and specified configurations.
- **Inputs**:
    - `model`: A reference to a `llama_model` object, which provides the hyperparameters needed for cache initialization.
    - `type_k`: A `ggml_type` specifying the data type for the key tensors.
    - `type_v`: A `ggml_type` specifying the data type for the value tensors.
    - `offload`: A boolean indicating whether to offload computations to a device other than the CPU.
    - `kv_size`: A `uint32_t` representing the size of the key-value cache.
    - `n_seq_max`: A `uint32_t` representing the maximum number of sequences that can be handled by the cache.
- **Control Flow**:
    - Initialize the number of layers from the model's hyperparameters.
    - Log the initialization parameters including kv_size, n_seq_max, and data types for keys and values.
    - Set initial values for head, size, and used variables, and clear the cells vector, resizing it to kv_size.
    - Create a context for each buffer type using a lambda function that checks if a context already exists in a map, otherwise initializes a new context and stores it.
    - Reserve space in vectors k_l and v_l for key and value tensors for each layer.
    - Iterate over each layer to calculate embedding sizes for keys and values, determine the device name and buffer type, and log the device information.
    - For each layer, create a context for the buffer type, create new 1D tensors for keys and values, format their names, and store them in k_l and v_l vectors.
    - Allocate tensors and initialize buffers to avoid NaNs in padding, logging the buffer sizes.
    - Calculate and log the total memory size for keys and values.
- **Output**: The constructor does not return a value; it initializes the internal state of the `llama_kv_cache_recurrent` object.
- **Functions called**:
    - [`ggml_type_name`](../ggml/src/ggml.c.driver.md#ggml_type_name)
    - [`ggml_tensor_overhead`](../ggml/src/ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_init`](../ggml/src/ggml.c.driver.md#ggml_init)
    - [`ggml_backend_dev_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_name)
    - [`ggml_new_tensor_1d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_format_name`](../ggml/src/ggml.c.driver.md#ggml_format_name)
    - [`ggml_backend_alloc_ctx_tensors_from_buft`](../ggml/src/ggml-alloc.c.driver.md#ggml_backend_alloc_ctx_tensors_from_buft)
    - [`ggml_backend_buffer_clear`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_clear)
    - [`ggml_backend_buffer_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_name)
    - [`ggml_backend_buffer_get_size`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_get_size)
    - [`llama_kv_cache_recurrent::size_k_bytes`](#llama_kv_cache_recurrentsize_k_bytes)
    - [`llama_kv_cache_recurrent::size_v_bytes`](#llama_kv_cache_recurrentsize_v_bytes)
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::clear<!-- {{#callable:llama_kv_cache_recurrent::clear}} -->
The `clear` function resets the state of the `llama_kv_cache_recurrent` object by clearing all key-value cells and buffers.
- **Inputs**: None
- **Control Flow**:
    - Iterate over each cell in the `cells` vector up to the `size` of the cache.
    - For each cell, set `pos`, `src`, and `tail` to -1 and clear the `seq_id` set.
    - Reset `head` and `used` to 0.
    - Iterate over each buffer in `bufs` and clear it using [`ggml_backend_buffer_clear`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_clear).
- **Output**: The function does not return any value; it modifies the state of the `llama_kv_cache_recurrent` object in place.
- **Functions called**:
    - [`ggml_backend_buffer_clear`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_clear)
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::seq\_rm<!-- {{#callable:llama_kv_cache_recurrent::seq_rm}} -->
The `seq_rm` function removes a sequence or a range of positions from the key-value cache, ensuring that the cache's integrity is maintained by handling partial intersections and invalidating tails as necessary.
- **Inputs**:
    - `seq_id`: The identifier of the sequence to be removed or modified.
    - `p0`: The starting position of the range to be removed; if negative, it defaults to 0.
    - `p1`: The ending position of the range to be removed; if negative, it defaults to the maximum possible value for `llama_pos`.
- **Control Flow**:
    - Initialize `new_head` to the current size of the cache.
    - Adjust `p0` and `p1` to default values if they are negative.
    - Check if `seq_id` is valid; if not, return false.
    - If `seq_id` is non-negative, check and handle partial intersections with the tail cell; invalidate the tail if necessary.
    - If `seq_id` is negative, ensure the range includes everything or nothing; return false if not.
    - Iterate over all cells in the cache, clearing sequence IDs and updating cell states if they fall within the specified range.
    - Update the `head` of the cache if a new slot was freed up and is before the current `head`.
- **Output**: Returns a boolean indicating whether the operation was successful, with `true` for success and `false` for failure.
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::seq\_cp<!-- {{#callable:llama_kv_cache_recurrent::seq_cp}} -->
The `seq_cp` function copies the sequence data from a source sequence ID to a destination sequence ID within a specified positional range in the `llama_kv_cache_recurrent` class.
- **Inputs**:
    - `seq_id_src`: The source sequence ID from which data is to be copied.
    - `seq_id_dst`: The destination sequence ID to which data is to be copied.
    - `p0`: The starting position of the range to be copied; if negative, it defaults to 0.
    - `p1`: The ending position of the range to be copied; if negative, it defaults to the maximum possible value for `llama_pos`.
- **Control Flow**:
    - Check if the source and destination sequence IDs are the same; if so, return immediately as no action is needed.
    - Adjust `p0` and `p1` to ensure they are non-negative, setting them to 0 and the maximum possible value, respectively, if they are negative.
    - Verify that both the source and destination sequence IDs are within the valid range of the cache size.
    - If the destination sequence's tail is valid, clear the destination sequence ID from the corresponding cell and update the tail and usage count accordingly.
    - If the source sequence's tail is valid, add the destination sequence ID to the source cell's sequence ID set and update the destination's tail to point to the source's tail.
- **Output**: The function does not return a value; it modifies the internal state of the `llama_kv_cache_recurrent` object by copying sequence data from the source to the destination.
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::seq\_keep<!-- {{#callable:llama_kv_cache_recurrent::seq_keep}} -->
The `seq_keep` function updates the key-value cache by retaining only the specified sequence ID, clearing others, and adjusting the cache head if necessary.
- **Inputs**:
    - `seq_id`: The sequence ID to be retained in the cache.
- **Control Flow**:
    - Initialize `new_head` to the current size of the cache.
    - Iterate over each cell in the cache.
    - For each cell, if its sequence ID does not match `seq_id`, set its `tail` to -1.
    - If the cell does not contain `seq_id`, clear its position, source, and sequence ID, and decrement the `used` counter if the position was non-negative.
    - If the cell contains `seq_id`, clear and reinsert `seq_id` into its sequence ID set.
    - Update `new_head` to the current index if it is the first cleared cell.
    - After the loop, if `new_head` is less than the current `head`, update `head` to `new_head`.
- **Output**: The function does not return a value; it modifies the internal state of the `llama_kv_cache_recurrent` object.
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::seq\_add<!-- {{#callable:llama_kv_cache_recurrent::seq_add}} -->
The `seq_add` function adjusts the position of a sequence within a specified range by a given shift value in the `llama_kv_cache_recurrent` class.
- **Inputs**:
    - `seq_id`: An identifier for the sequence to be modified.
    - `p0`: The starting position of the range within which the sequence position should be adjusted.
    - `p1`: The ending position of the range within which the sequence position should be adjusted.
    - `shift`: The amount by which the sequence position should be shifted.
- **Control Flow**:
    - Check if the shift is zero; if so, return immediately as no adjustment is needed.
    - Ensure that p0 is not negative by setting it to zero if it is.
    - Ensure that p1 is not negative by setting it to the maximum possible value if it is.
    - Return early if p0 equals p1, indicating no range to adjust.
    - Check if the seq_id is within valid bounds and retrieve the tail cell associated with it.
    - If the tail cell exists and the sequence ID is present in the cell, and the cell's position is within the specified range, adjust the cell's position by the shift value.
- **Output**: The function does not return any value; it modifies the position of a sequence in the kv cache if conditions are met.
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::seq\_div<!-- {{#callable:llama_kv_cache_recurrent::seq_div}} -->
The `seq_div` function divides the position of a sequence within a specified range by a given divisor if certain conditions are met.
- **Inputs**:
    - `seq_id`: An identifier for the sequence to be processed.
    - `p0`: The starting position of the range to be considered; if negative, it defaults to 0.
    - `p1`: The ending position of the range to be considered; if negative, it defaults to the maximum possible value for `llama_pos`.
    - `d`: The divisor by which the position should be divided; if it is 1, the function returns immediately.
- **Control Flow**:
    - Check if the divisor `d` is 1, and if so, return immediately as no division is needed.
    - Adjust `p0` to 0 if it is negative, and `p1` to the maximum possible value if it is negative.
    - Return immediately if `p0` equals `p1`, indicating no range to process.
    - Check if `seq_id` is within the valid range of sequence identifiers.
    - Retrieve the tail cell associated with `seq_id` and check if it is valid.
    - If the tail cell has the sequence ID and its position is within the range `[p0, p1)`, divide the position by `d`.
- **Output**: The function does not return any value; it modifies the position of a sequence in the cache if applicable.
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::seq\_pos\_min<!-- {{#callable:llama_kv_cache_recurrent::seq_pos_min}} -->
The `seq_pos_min` function finds the minimum position of a given sequence ID within the key-value cache cells of the `llama_kv_cache_recurrent` class.
- **Inputs**:
    - `seq_id`: A sequence identifier of type `llama_seq_id` for which the minimum position is to be found.
- **Control Flow**:
    - Initialize `result` to the maximum possible value for `llama_pos`.
    - Iterate over each cell in the `cells` vector up to the `size` of the cache.
    - For each cell, check if it contains the given `seq_id` using the `has_seq_id` method.
    - If the cell contains the `seq_id`, update `result` to the minimum of its current value and the cell's position.
    - After the loop, check if `result` is still the maximum possible value, indicating that the `seq_id` was not found, and set `result` to -1 in that case.
    - Return the `result` value.
- **Output**: The function returns a `llama_pos` value representing the minimum position of the specified sequence ID, or -1 if the sequence ID is not found in any cell.
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::seq\_pos\_max<!-- {{#callable:llama_kv_cache_recurrent::seq_pos_max}} -->
The `seq_pos_max` function finds the maximum position value for a given sequence ID within the `llama_kv_cache_recurrent` class.
- **Inputs**:
    - `seq_id`: A `llama_seq_id` representing the sequence ID for which the maximum position is to be found.
- **Control Flow**:
    - Initialize `result` to -1, which will store the maximum position found.
    - Iterate over each cell in the `cells` vector up to the `size` of the cache.
    - For each cell, check if it contains the given `seq_id` using the `has_seq_id` method.
    - If the cell contains the `seq_id`, update `result` to the maximum of its current value and the cell's position.
    - Return the `result`, which is the maximum position found for the given `seq_id`.
- **Output**: Returns a `llama_pos` which is the maximum position value for the specified sequence ID, or -1 if the sequence ID is not found in any cell.
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::init\_batch<!-- {{#callable:llama_kv_cache_recurrent::init_batch}} -->
The `init_batch` function initializes a batch of sequences for processing in a recurrent key-value cache, splitting the batch into smaller sub-batches and preparing them for storage.
- **Inputs**:
    - `batch`: A `llama_batch` object representing the batch of sequences to be processed.
    - `n_ubatch`: A `uint32_t` specifying the number of sub-batches to split the batch into.
    - `embd_pooled`: A `bool` indicating whether embeddings are pooled, affecting how sub-batches are split.
    - `logits_all`: A `bool` indicating whether all logits should be considered during processing.
- **Control Flow**:
    - The function begins by creating a [`llama_sbatch`](llama-batch.h.driver.md#llama_sbatchllama_sbatch) object from the input `batch`, using the model's embedding size and the `logits_all` flag.
    - An empty vector `ubatches` is initialized to store the sub-batches.
    - A loop runs while there are tokens left in `sbatch`, splitting it into sub-batches using either `split_seq` or `split_equal` based on the `embd_pooled` flag, and appending each sub-batch to `ubatches`.
    - The [`prepare`](#llama_kv_cache_recurrentprepare) function is called with the `ubatches` vector to prepare them for storage.
    - If [`prepare`](#llama_kv_cache_recurrentprepare) fails, the function returns a `llama_kv_cache_recurrent_state` with a failure status.
    - If successful, the function returns a `llama_kv_cache_recurrent_state` with a success status, containing the prepared `sbatch` and `ubatches`.
- **Output**: A `llama_memory_state_ptr` pointing to a `llama_kv_cache_recurrent_state` object, indicating the success or failure of the batch initialization.
- **Functions called**:
    - [`llama_sbatch::llama_sbatch`](llama-batch.h.driver.md#llama_sbatchllama_sbatch)
    - [`llama_kv_cache_recurrent::prepare`](#llama_kv_cache_recurrentprepare)
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::init\_full<!-- {{#callable:llama_kv_cache_recurrent::init_full}} -->
The `init_full` function initializes a full memory state for the `llama_kv_cache_recurrent` class and returns a pointer to this state.
- **Inputs**: None
- **Control Flow**:
    - The function calls `std::make_unique` to create a new `llama_kv_cache_recurrent_state` object.
    - It passes `LLAMA_MEMORY_STATUS_SUCCESS` and `this` (the current instance of `llama_kv_cache_recurrent`) to the constructor of `llama_kv_cache_recurrent_state`.
- **Output**: A `llama_memory_state_ptr` which is a unique pointer to a `llama_kv_cache_recurrent_state` object initialized with success status and the current instance.
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::init\_update<!-- {{#callable:llama_kv_cache_recurrent::init_update}} -->
The `init_update` function initializes an update operation for the `llama_kv_cache_recurrent` class, returning a state indicating no update is needed.
- **Inputs**:
    - `lctx`: A pointer to a `llama_context` object, which is not used in this function.
    - `optimize`: A boolean flag indicating whether optimization should be applied, which is not used in this function.
- **Control Flow**:
    - The function begins by marking the `lctx` and `optimize` parameters as unused using the `GGML_UNUSED` macro.
    - It then creates a new `llama_kv_cache_recurrent_state` object with the status `LLAMA_MEMORY_STATUS_NO_UPDATE` using `std::make_unique`.
    - The function returns the newly created `llama_kv_cache_recurrent_state` object.
- **Output**: A `llama_memory_state_ptr`, which is a smart pointer to a `llama_kv_cache_recurrent_state` object with a status indicating no update is needed.
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::prepare<!-- {{#callable:llama_kv_cache_recurrent::prepare}} -->
The `prepare` function attempts to prepare the key-value cache for processing a batch of micro-batches (`ubatches`) by saving the current state and checking if the micro-batches can fit into the cache, although the current implementation is incomplete and always returns success.
- **Inputs**:
    - `ubatches`: A constant reference to a vector of `llama_ubatch` objects, representing micro-batches to be processed.
- **Control Flow**:
    - The function begins by saving the current state of the cache, including `cells`, `used`, and `head`, to local variables `org_cells`, `org_used`, and `org_head`.
    - A boolean variable `success` is initialized to `true`.
    - The function contains a commented-out loop that would iterate over each `ubatch` in `ubatches` to check if it can find a slot for each `ubatch` using the `find_slot` method, setting `success` to `false` if any `ubatch` cannot be accommodated.
    - The original state of the cache is restored by assigning `org_cells`, `org_used`, and `org_head` back to `cells`, `used`, and `head`, respectively.
    - The function returns the `success` variable, which is always `true` in the current implementation.
- **Output**: A boolean value indicating whether the preparation was successful, which is always `true` in the current implementation.
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::find\_slot<!-- {{#callable:llama_kv_cache_recurrent::find_slot}} -->
The `find_slot` function locates a contiguous slot in the key-value cache to place a given batch of sequences, ensuring that the cache is efficiently utilized and updated.
- **Inputs**:
    - `ubatch`: A `llama_ubatch` object containing the batch of sequences to be placed in the cache, including the number of tokens, sequences, and sequence IDs.
- **Control Flow**:
    - Check if the head of the cache should be reset to zero based on the number of unused cells and tokens.
    - Assert that the batch has an equal number of new tokens in each sequence.
    - Iterate over each sequence and sequence ID to ensure they fit within the cache size, logging an error if not.
    - Find the next empty cell in the cache starting from the current head position.
    - Determine the range of usable cells for the sequences, updating the cache cells as needed.
    - Reorder the cache cells to ensure they are contiguous and update sequence IDs and positions accordingly.
    - Update the head, number of used cells, and the range of used cells in the cache.
    - Return true if the number of used cells is greater than or equal to the number of sequences, indicating success.
- **Output**: A boolean value indicating whether a suitable slot was found and the batch was successfully placed in the cache.
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::get\_can\_shift<!-- {{#callable:llama_kv_cache_recurrent::get_can_shift}} -->
The `get_can_shift` function in the `llama_kv_cache_recurrent` class always returns `false`, indicating that shifting is not allowed.
- **Inputs**: None
- **Control Flow**:
    - The function is a constant method, meaning it does not modify the state of the object.
    - It simply returns the boolean value `false`.
- **Output**: A boolean value `false`, indicating that shifting is not possible.
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::s\_copy<!-- {{#callable:llama_kv_cache_recurrent::s_copy}} -->
The `s_copy` function retrieves and potentially updates the source index of a key-value cell in a cache, ensuring it is within bounds and only copied once.
- **Inputs**:
    - `i`: An integer index representing the position in the cache from which to retrieve the source index.
- **Control Flow**:
    - Calculate the cell ID by adding the input index `i` to the `head` attribute.
    - Use `const_cast` to obtain a non-const reference to the `kv_cell` at the calculated cell ID.
    - Check if the `src` attribute of the cell is out of bounds (less than 0 or greater than or equal to `size`), and if so, set it to the current cell ID.
    - Store the current `src` value in a local variable `res`.
    - Check if the `src` attribute is not equal to the current cell ID, and if so, update it to the current cell ID to ensure the copy only happens once.
    - Return the value stored in `res`.
- **Output**: An integer representing the source index of the key-value cell, which may have been updated to ensure it is within bounds and only copied once.
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::s\_mask<!-- {{#callable:llama_kv_cache_recurrent::s_mask}} -->
The `s_mask` function checks if a specific key-value cache cell has been initialized and updates its source if not, returning a float indicating the initialization status.
- **Inputs**:
    - `i`: An integer index representing the position in the key-value cache to be checked and potentially updated.
- **Control Flow**:
    - Calculate the cell ID by adding the input index `i` to the `head` attribute of the class.
    - Access the `kv_cell` at the calculated cell ID, using `const_cast` to allow modification despite the function being marked as `const`.
    - Check if the `src` attribute of the cell is non-negative, indicating it has been initialized, and store the result as a float.
    - If the `src` attribute is negative, indicating the cell is uninitialized, set it to the current cell ID to mark it as initialized.
    - Return the float result indicating whether the cell was initialized before the function call.
- **Output**: A float value indicating whether the cell at the specified index was initialized (1.0 if initialized, 0.0 if not).
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::total\_size<!-- {{#callable:llama_kv_cache_recurrent::total_size}} -->
The `total_size` function calculates the total size of all backend buffers in the `llama_kv_cache_recurrent` class.
- **Inputs**: None
- **Control Flow**:
    - Initialize a variable `size` to 0.
    - Iterate over each buffer in the `bufs` vector.
    - For each buffer, retrieve its size using [`ggml_backend_buffer_get_size`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_get_size) and add it to `size`.
    - Return the accumulated `size`.
- **Output**: Returns the total size of all backend buffers as a `size_t` value.
- **Functions called**:
    - [`ggml_backend_buffer_get_size`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_get_size)
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::size\_k\_bytes<!-- {{#callable:llama_kv_cache_recurrent::size_k_bytes}} -->
The `size_k_bytes` function calculates the total memory size in bytes of all key tensors in the `k_l` vector.
- **Inputs**: None
- **Control Flow**:
    - Initialize `size_k_bytes` to 0.
    - Iterate over each tensor `k` in the `k_l` vector.
    - For each tensor `k`, add its size in bytes (obtained using `ggml_nbytes(k)`) to `size_k_bytes`.
    - Return the accumulated `size_k_bytes`.
- **Output**: The function returns a `size_t` representing the total size in bytes of all key tensors in the `k_l` vector.
- **Functions called**:
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::size\_v\_bytes<!-- {{#callable:llama_kv_cache_recurrent::size_v_bytes}} -->
The `size_v_bytes` function calculates the total memory size in bytes used by the 'v' tensors in the `llama_kv_cache_recurrent` class.
- **Inputs**: None
- **Control Flow**:
    - Initialize a variable `size_v_bytes` to 0 to accumulate the total size.
    - Iterate over each tensor `v` in the vector `v_l`.
    - For each tensor `v`, add its size in bytes, obtained by calling `ggml_nbytes(v)`, to `size_v_bytes`.
    - Return the accumulated `size_v_bytes`.
- **Output**: The function returns a `size_t` representing the total size in bytes of all 'v' tensors in the `v_l` vector.
- **Functions called**:
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::state\_write<!-- {{#callable:llama_kv_cache_recurrent::state_write}} -->
The `state_write` function writes the state of the key-value cache for a specified sequence ID to an output stream.
- **Inputs**:
    - `io`: A reference to an object implementing the `llama_io_write_i` interface, used for writing data.
    - `seq_id`: An identifier for the sequence whose state is to be written; if set to -1, all sequences are considered.
- **Control Flow**:
    - Initialize a vector `cell_ranges` to store ranges of cells and a counter `cell_count` to count cells with the specified `seq_id`.
    - Iterate over all cells to identify those with the specified `seq_id` or all non-empty cells if `seq_id` is -1, updating `cell_count` and `cell_ranges` accordingly.
    - Perform a debug check to ensure the sum of cell counts in `cell_ranges` matches `cell_count`.
    - Write `cell_count` to the output stream using the `io` object.
    - Call [`state_write_meta`](#llama_kv_cache_recurrentstate_write_meta) to write metadata of the identified cell ranges to the output stream.
    - Call [`state_write_data`](#llama_kv_cache_recurrentstate_write_data) to write the actual data of the identified cell ranges to the output stream.
- **Output**: The function does not return a value; it writes data to the provided `io` stream.
- **Functions called**:
    - [`llama_kv_cache_recurrent::state_write_meta`](#llama_kv_cache_recurrentstate_write_meta)
    - [`llama_kv_cache_recurrent::state_write_data`](#llama_kv_cache_recurrentstate_write_data)
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::state\_read<!-- {{#callable:llama_kv_cache_recurrent::state_read}} -->
The `state_read` function reads and restores the state of the key-value cache from an input stream, handling errors by clearing or removing sequences if necessary.
- **Inputs**:
    - `io`: A reference to an object implementing the `llama_io_read_i` interface, used for reading data from an input stream.
    - `seq_id`: An identifier for the sequence to be restored; if set to -1, the entire cache is restored.
- **Control Flow**:
    - Read the number of cells (`cell_count`) from the input stream using `io.read_to`.
    - Initialize a boolean `res` to true, which will track the success of the read operations.
    - Call [`state_read_meta`](#llama_kv_cache_recurrentstate_read_meta) with `io`, `cell_count`, and `seq_id`, updating `res` with the result.
    - Call [`state_read_data`](#llama_kv_cache_recurrentstate_read_data) with `io` and `cell_count`, updating `res` with the result.
    - If `res` is false, check if `seq_id` is -1; if so, call `clear()` to reset the cache, otherwise call [`seq_rm`](#llama_kv_cache_recurrentseq_rm) to remove the sequence with `seq_id`.
    - Throw a `std::runtime_error` if the restoration fails.
- **Output**: The function does not return a value but may throw a `std::runtime_error` if the state restoration fails.
- **Functions called**:
    - [`llama_kv_cache_recurrent::state_read_meta`](#llama_kv_cache_recurrentstate_read_meta)
    - [`llama_kv_cache_recurrent::state_read_data`](#llama_kv_cache_recurrentstate_read_data)
    - [`llama_kv_cache_recurrent::clear`](#llama_kv_cache_recurrentclear)
    - [`llama_kv_cache_recurrent::seq_rm`](#llama_kv_cache_recurrentseq_rm)
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::state\_write\_meta<!-- {{#callable:llama_kv_cache_recurrent::state_write_meta}} -->
The `state_write_meta` function writes metadata of key-value cache cells to an output stream, including position and sequence IDs, for specified cell ranges.
- **Inputs**:
    - `io`: An output stream interface (`llama_io_write_i`) used to write data.
    - `cell_ranges`: A vector of pairs of unsigned integers, each representing a range of cell indices to process.
    - `seq_id`: A sequence identifier (`llama_seq_id`) used to filter which sequence IDs to write; if -1, all sequence IDs are written.
- **Control Flow**:
    - Iterate over each range in `cell_ranges`.
    - For each range, iterate over the cell indices from the start to the end of the range.
    - Retrieve the position (`pos`) and sequence IDs (`seq_id`) of each cell.
    - Determine the number of sequence IDs (`n_seq_id`) to write based on whether `seq_id` is -1.
    - Write the position and number of sequence IDs to the output stream.
    - If `n_seq_id` is non-zero, write each sequence ID to the output stream.
- **Output**: The function does not return a value; it writes metadata to the provided output stream.
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::state\_write\_data<!-- {{#callable:llama_kv_cache_recurrent::state_write_data}} -->
The `state_write_data` function writes the key and value data of a llama_kv_cache_recurrent object to an output stream, handling both non-transposed and transposed value cases.
- **Inputs**:
    - `io`: A reference to an object implementing the llama_io_write_i interface, used for writing data to an output stream.
    - `cell_ranges`: A constant reference to a vector of pairs of uint32_t, representing the ranges of cells to be written, where each pair indicates the start and end indices of a range.
- **Control Flow**:
    - Initialize v_trans to 0 and retrieve the number of layers from hparams.
    - Write v_trans and n_layer to the output stream using the io object.
    - Iterate over each layer to write key data:
    -   - Calculate the number of key embeddings for the current layer.
    -   - Write the key type and row size to the output stream.
    -   - For each range in cell_ranges, calculate the buffer size and write the key tensor data to the output stream.
    - Check if v_trans is false (non-transposed case):
    -   - Iterate over each layer to write value data:
    -     - Calculate the number of value embeddings for the current layer.
    -     - Write the value type and row size to the output stream.
    -     - For each range in cell_ranges, calculate the buffer size and write the value tensor data to the output stream.
    - If v_trans is true (transposed case):
    -   - Iterate over each layer to write transposed value data:
    -     - Calculate the number of value embeddings for the current layer.
    -     - Write the value type, element size, and embedding size to the output stream.
    -     - For each embedding, iterate over each range in cell_ranges, calculate the source offset and buffer size, and write the value tensor data to the output stream.
- **Output**: The function does not return a value; it writes data to the provided io object.
- **Functions called**:
    - [`ggml_row_size`](../ggml/src/ggml.c.driver.md#ggml_row_size)
    - [`ggml_type_size`](../ggml/src/ggml.c.driver.md#ggml_type_size)
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::state\_read\_meta<!-- {{#callable:llama_kv_cache_recurrent::state_read_meta}} -->
The `state_read_meta` function reads metadata from an input stream to restore the state of a key-value cache, either for a single sequence or the entire cache, and updates the cache accordingly.
- **Inputs**:
    - `io`: A reference to an `llama_io_read_i` object used to read data from an input stream.
    - `cell_count`: A `uint32_t` representing the number of cells to read from the input stream.
    - `dest_seq_id`: A `llama_seq_id` indicating the destination sequence ID for which the state is being restored; if set to -1, the entire cache is restored.
- **Control Flow**:
    - Check if `dest_seq_id` is not -1 to determine if restoring a single sequence or the entire cache.
    - If restoring a single sequence, remove the sequence from the cache, reserve a batch, and read positions and sequence IDs from the input stream.
    - Validate the sequence IDs and find a slot in the cache for the batch; return false if any validation fails.
    - If restoring the entire cache, clear the cache, read positions and sequence IDs for each cell, and validate them.
    - Update the cache's head and used cell count, and ensure each cell's source is set to its own ID.
    - Return true if the operation is successful, otherwise return false.
- **Output**: A boolean value indicating whether the metadata was successfully read and the cache state restored.
- **Functions called**:
    - [`llama_kv_cache_recurrent::seq_rm`](#llama_kv_cache_recurrentseq_rm)
    - [`llama_kv_cache_recurrent::find_slot`](#llama_kv_cache_recurrentfind_slot)
    - [`llama_kv_cache_recurrent::clear`](#llama_kv_cache_recurrentclear)
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent::state\_read\_data<!-- {{#callable:llama_kv_cache_recurrent::state_read_data}} -->
The `state_read_data` function reads and validates key-value cache data from an input stream for a specified number of cells, ensuring compatibility with the current model's parameters.
- **Inputs**:
    - `io`: A reference to an `llama_io_read_i` object, which provides the interface for reading data from an input stream.
    - `cell_count`: A `uint32_t` representing the number of cells to read from the input stream.
- **Control Flow**:
    - Read the transposition flag `v_trans` and the number of layers `n_layer` from the input stream.
    - Check if `n_layer` matches the expected number of layers from `hparams`; log an error and return false if not.
    - Check if `cell_count` exceeds the available cache size; log an error and return false if so.
    - Check if `v_trans` is false; log an error and return false if not.
    - Iterate over each layer to read and validate key data: read key type and row size, compare with expected values, log errors and return false on mismatches.
    - If `cell_count` is non-zero, read and set the keys for the entire cell range.
    - If `v_trans` is false, iterate over each layer to read and validate value data: read value type and row size, compare with expected values, log errors and return false on mismatches.
    - If `cell_count` is non-zero, read and set the values for the entire cell range.
    - If `v_trans` is true, iterate over each layer to read transposed value data: read value type, element size, and GQA embedding size, compare with expected values, log errors and return false on mismatches.
    - If `cell_count` is non-zero, read and set the transposed values for the entire cell range.
- **Output**: Returns a boolean indicating success (true) or failure (false) of the data reading and validation process.
- **Functions called**:
    - [`ggml_row_size`](../ggml/src/ggml.c.driver.md#ggml_row_size)
    - [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`ggml_type_size`](../ggml/src/ggml.c.driver.md#ggml_type_size)
- **See also**: [`llama_kv_cache_recurrent`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent)  (Data Structure)



---
### llama\_kv\_cache\_recurrent\_state<!-- {{#data_structure:llama_kv_cache_recurrent_state}} -->
- **Description**: [See definition](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent_state)
- **Member Functions**:
    - [`llama_kv_cache_recurrent_state::llama_kv_cache_recurrent_state`](#llama_kv_cache_recurrent_statellama_kv_cache_recurrent_state)
    - [`llama_kv_cache_recurrent_state::llama_kv_cache_recurrent_state`](#llama_kv_cache_recurrent_statellama_kv_cache_recurrent_state)
    - [`llama_kv_cache_recurrent_state::llama_kv_cache_recurrent_state`](#llama_kv_cache_recurrent_statellama_kv_cache_recurrent_state)
    - [`llama_kv_cache_recurrent_state::~llama_kv_cache_recurrent_state`](#llama_kv_cache_recurrent_statellama_kv_cache_recurrent_state)
    - [`llama_kv_cache_recurrent_state::next`](#llama_kv_cache_recurrent_statenext)
    - [`llama_kv_cache_recurrent_state::apply`](#llama_kv_cache_recurrent_stateapply)
    - [`llama_kv_cache_recurrent_state::out_ids`](#llama_kv_cache_recurrent_stateout_ids)
    - [`llama_kv_cache_recurrent_state::get_status`](#llama_kv_cache_recurrent_stateget_status)
    - [`llama_kv_cache_recurrent_state::get_ubatch`](#llama_kv_cache_recurrent_stateget_ubatch)
    - [`llama_kv_cache_recurrent_state::get_n_kv`](#llama_kv_cache_recurrent_stateget_n_kv)
    - [`llama_kv_cache_recurrent_state::get_head`](#llama_kv_cache_recurrent_stateget_head)
    - [`llama_kv_cache_recurrent_state::get_size`](#llama_kv_cache_recurrent_stateget_size)
    - [`llama_kv_cache_recurrent_state::get_k_l`](#llama_kv_cache_recurrent_stateget_k_l)
    - [`llama_kv_cache_recurrent_state::get_v_l`](#llama_kv_cache_recurrent_stateget_v_l)
    - [`llama_kv_cache_recurrent_state::s_copy`](#llama_kv_cache_recurrent_states_copy)
    - [`llama_kv_cache_recurrent_state::s_mask`](#llama_kv_cache_recurrent_states_mask)
- **Inherits From**:
    - `llama_memory_state_i`

**Methods**

---
#### llama\_kv\_cache\_recurrent\_state::llama\_kv\_cache\_recurrent\_state<!-- {{#callable:llama_kv_cache_recurrent_state::llama_kv_cache_recurrent_state}} -->
The `llama_kv_cache_recurrent_state` constructor initializes an instance with a given memory status.
- **Inputs**:
    - `status`: A `llama_memory_status` value representing the memory status to initialize the state with.
- **Control Flow**:
    - The constructor takes a `llama_memory_status` parameter named `status`.
    - It initializes the `status` member variable of the `llama_kv_cache_recurrent_state` class with the provided `status` argument.
- **Output**: An instance of `llama_kv_cache_recurrent_state` is created with the specified memory status.
- **See also**: [`llama_kv_cache_recurrent_state`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent_state)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent\_state::\~llama\_kv\_cache\_recurrent\_state<!-- {{#callable:llama_kv_cache_recurrent_state::~llama_kv_cache_recurrent_state}} -->
The destructor `~llama_kv_cache_recurrent_state` is a default destructor for the `llama_kv_cache_recurrent_state` class, which performs no specific actions upon object destruction.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as `= default;`, indicating that the compiler will generate the default destructor implementation.
    - No custom cleanup or resource deallocation is performed in this destructor.
- **Output**: There is no output from this destructor as it is a default implementation.
- **See also**: [`llama_kv_cache_recurrent_state`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent_state)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent\_state::next<!-- {{#callable:llama_kv_cache_recurrent_state::next}} -->
The `next` function in the `llama_kv_cache_recurrent_state` class advances the index to the next micro-batch and returns whether there are more micro-batches to process.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - The function asserts that the current status is `LLAMA_MEMORY_STATUS_SUCCESS`.
    - It increments the `i_next` index by one.
    - It checks if the incremented `i_next` is greater than or equal to the size of the `ubatches` vector.
    - If `i_next` is greater than or equal to the size of `ubatches`, it returns `false`.
    - Otherwise, it returns `true`.
- **Output**: A boolean value indicating whether there are more micro-batches to process (`true`) or not (`false`).
- **See also**: [`llama_kv_cache_recurrent_state`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent_state)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent\_state::apply<!-- {{#callable:llama_kv_cache_recurrent_state::apply}} -->
The `apply` method in the `llama_kv_cache_recurrent_state` class ensures that the current ubatch is processed by finding an appropriate slot in the key-value cache.
- **Inputs**:
    - `None`: This method does not take any input arguments.
- **Control Flow**:
    - The method asserts that the `status` is `LLAMA_MEMORY_STATUS_SUCCESS` to ensure the state is valid for processing.
    - It calls the `find_slot` method on the `kv` object with the current ubatch (`ubatches[i_next]`) to find a suitable slot in the cache.
    - The method returns `true` to indicate successful application.
- **Output**: The method returns a boolean value `true` indicating that the operation was successful.
- **See also**: [`llama_kv_cache_recurrent_state`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent_state)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent\_state::out\_ids<!-- {{#callable:llama_kv_cache_recurrent_state::out_ids}} -->
The `out_ids` function returns a reference to the `out_ids` vector from the `sbatch` object within the `llama_kv_cache_recurrent_state` class, ensuring the state is successful before doing so.
- **Inputs**: None
- **Control Flow**:
    - The function asserts that the `status` of the `llama_kv_cache_recurrent_state` object is `LLAMA_MEMORY_STATUS_SUCCESS` to ensure the state is valid.
    - It then returns the `out_ids` vector from the `sbatch` object.
- **Output**: A reference to a `std::vector<int64_t>` containing the output IDs from the `sbatch` object.
- **See also**: [`llama_kv_cache_recurrent_state`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent_state)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent\_state::get\_status<!-- {{#callable:llama_kv_cache_recurrent_state::get_status}} -->
The `get_status` function returns the current memory status of the `llama_kv_cache_recurrent_state` object.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the `status` member variable of the `llama_kv_cache_recurrent_state` class.
- **Output**: The function returns a `llama_memory_status` value, which represents the current status of the memory state.
- **See also**: [`llama_kv_cache_recurrent_state`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent_state)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent\_state::get\_ubatch<!-- {{#callable:llama_kv_cache_recurrent_state::get_ubatch}} -->
The `get_ubatch` function retrieves the current unprocessed batch (`ubatch`) from the `llama_kv_cache_recurrent_state` object.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - The function asserts that the `status` of the `llama_kv_cache_recurrent_state` object is `LLAMA_MEMORY_STATUS_SUCCESS` to ensure the state is valid for operation.
    - It then returns the `ubatch` at the index `i_next` from the `ubatches` vector.
- **Output**: The function returns a constant reference to a `llama_ubatch` object, which represents the current unprocessed batch.
- **See also**: [`llama_kv_cache_recurrent_state`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent_state)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent\_state::get\_n\_kv<!-- {{#callable:llama_kv_cache_recurrent_state::get_n_kv}} -->
The `get_n_kv` function returns the number of key-value pairs in the cache, depending on whether the cache is full or not.
- **Inputs**: None
- **Control Flow**:
    - Check if the `is_full` flag is true.
    - If `is_full` is true, return `kv->size`.
    - If `is_full` is false, return `kv->n`.
- **Output**: Returns a `uint32_t` representing the number of key-value pairs in the cache.
- **See also**: [`llama_kv_cache_recurrent_state`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent_state)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent\_state::get\_head<!-- {{#callable:llama_kv_cache_recurrent_state::get_head}} -->
The `get_head` function returns the head index of the key-value cache, which is 0 if the cache is full, otherwise it returns the current head index from the `kv` object.
- **Inputs**: None
- **Control Flow**:
    - Check if the `is_full` flag is true.
    - If `is_full` is true, return 0.
    - If `is_full` is false, return the `head` value from the `kv` object.
- **Output**: Returns a `uint32_t` representing the head index of the key-value cache.
- **See also**: [`llama_kv_cache_recurrent_state`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent_state)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent\_state::get\_size<!-- {{#callable:llama_kv_cache_recurrent_state::get_size}} -->
The `get_size` function returns the size of the key-value cache in the `llama_kv_cache_recurrent_state` class.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `kv` member of the `llama_kv_cache_recurrent_state` class, which is a pointer to a `llama_kv_cache_recurrent` object.
    - It retrieves the `size` attribute from the `kv` object.
    - The function then returns this `size` value.
- **Output**: The function returns a `uint32_t` representing the size of the key-value cache.
- **See also**: [`llama_kv_cache_recurrent_state`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent_state)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent\_state::get\_k\_l<!-- {{#callable:llama_kv_cache_recurrent_state::get_k_l}} -->
The `get_k_l` function retrieves a specific layer's key tensor from the key-value cache in a recurrent state.
- **Inputs**:
    - `il`: An integer representing the index of the layer for which the key tensor is to be retrieved.
- **Control Flow**:
    - The function accesses the `kv` member of the `llama_kv_cache_recurrent_state` class, which is a pointer to a `llama_kv_cache_recurrent` object.
    - It retrieves the key tensor at the specified layer index `il` from the `k_l` vector of the `llama_kv_cache_recurrent` object.
- **Output**: Returns a pointer to a `ggml_tensor` object representing the key tensor for the specified layer index.
- **See also**: [`llama_kv_cache_recurrent_state`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent_state)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent\_state::get\_v\_l<!-- {{#callable:llama_kv_cache_recurrent_state::get_v_l}} -->
The `get_v_l` function retrieves a specific layer's value tensor from the key-value cache of a recurrent state.
- **Inputs**:
    - `il`: An integer representing the index of the layer for which the value tensor is to be retrieved.
- **Control Flow**:
    - The function accesses the `v_l` vector within the `kv` object, which is a pointer to a `llama_kv_cache_recurrent` instance.
    - It returns the value tensor at the specified index `il` from the `v_l` vector.
- **Output**: A pointer to a `ggml_tensor` representing the value tensor for the specified layer index.
- **See also**: [`llama_kv_cache_recurrent_state`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent_state)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent\_state::s\_copy<!-- {{#callable:llama_kv_cache_recurrent_state::s_copy}} -->
The `s_copy` function in `llama_kv_cache_recurrent_state` class calls the `s_copy` method of the `kv` object with the given index `i` and returns its result.
- **Inputs**:
    - `i`: An integer index used to specify which element to copy in the `kv` cache.
- **Control Flow**:
    - The function takes an integer `i` as input.
    - It calls the `s_copy` method on the `kv` object, passing `i` as an argument.
    - The result of the `kv->s_copy(i)` call is returned.
- **Output**: The function returns an `int32_t` which is the result of the `s_copy` method from the `kv` object.
- **See also**: [`llama_kv_cache_recurrent_state`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent_state)  (Data Structure)


---
#### llama\_kv\_cache\_recurrent\_state::s\_mask<!-- {{#callable:llama_kv_cache_recurrent_state::s_mask}} -->
The `s_mask` function returns a float indicating whether a specific cell in the key-value cache has a valid source index.
- **Inputs**:
    - `i`: An integer index representing the position in the key-value cache to check.
- **Control Flow**:
    - The function calls the `s_mask` method of the `kv` object, passing the index `i` as an argument.
    - The `s_mask` method of the `kv` object checks if the source index of the cell at position `i + head` is valid (i.e., non-negative).
    - If the source index is valid, it returns 1.0; otherwise, it returns 0.0.
- **Output**: A float value, either 1.0 if the cell has a valid source index or 0.0 if it does not.
- **See also**: [`llama_kv_cache_recurrent_state`](llama-kv-cache-recurrent.h.driver.md#llama_kv_cache_recurrent_state)  (Data Structure)



