# Purpose
The provided C++ source code defines a class [`llama_kv_cache_unified`](#llama_kv_cache_unifiedllama_kv_cache_unified), which is part of a larger system likely related to machine learning or neural network operations, given the context of key-value caching and sequence handling. This class is responsible for managing a unified key-value cache, which is a critical component in optimizing the performance of models that require fast access to previously computed data, such as transformer models used in natural language processing. The class provides methods for initializing, updating, and managing the cache, including operations like clearing, copying, and defragmenting the cache. It also handles sequence operations such as adding, removing, and keeping sequences, which are essential for managing the temporal aspects of data in sequence-based models.

The code is structured to support various backend operations, as indicated by the use of `ggml_context` and related functions, which suggest integration with a backend library for tensor operations. The class also includes methods for handling input and output operations, such as reading and writing cache states, which are crucial for maintaining the cache's integrity across different sessions or instances. The presence of logging and error handling mechanisms indicates a focus on robustness and debugging support. Overall, this file provides a specialized and narrow functionality focused on efficient cache management for sequence data, which is a common requirement in advanced machine learning applications.
# Imports and Dependencies

---
- `llama-kv-cache-unified.h`
- `llama-impl.h`
- `llama-io.h`
- `llama-model.h`
- `llama-context.h`
- `algorithm`
- `cassert`
- `cmath`
- `limits`
- `map`
- `stdexcept`


# Data Structures

---
### state<!-- {{#data_structure:state}} -->
- **Type**: `struct`
- **Members**:
    - `head_old`: Stores the old position of the head before placing the ubatch.
    - `head_new`: Stores the new position of the head after placing the ubatch.
    - `cells`: Holds a copy of the old cells before placing the ubatch.
- **Description**: The `state` struct is used to capture the state of a key-value cache system, specifically tracking the position of the head before and after an update batch (ubatch) is placed. It also stores a snapshot of the cells before the update, allowing for potential rollback or analysis of changes. This struct is likely used in the context of managing memory or data structures that require precise tracking of changes over time.


---
### llm\_graph\_input\_k\_shift<!-- {{#data_structure:llm_graph_input_k_shift}} -->
- **Type**: `class`
- **Members**:
    - `k_shift`: A pointer to a ggml_tensor representing the K-shift tensor, which is an integer tensor with a size corresponding to kv_size.
    - `kv_self`: A constant pointer to a llama_kv_cache_unified object, representing the key-value cache associated with this input.
- **Description**: The `llm_graph_input_k_shift` class is a specialized input class for handling K-shift operations in a graph-based model. It inherits from `llm_graph_input_i` and is designed to work with a key-value cache (`llama_kv_cache_unified`). The class contains a tensor `k_shift` that is used to store the K-shift values, and a pointer `kv_self` to the associated key-value cache. This class provides functionality to set input data for the K-shift operation, which is crucial for managing shifts in the key-value cache during model execution.
- **Member Functions**:
    - [`llm_graph_input_k_shift::llm_graph_input_k_shift`](#llm_graph_input_k_shiftllm_graph_input_k_shift)
    - [`llm_graph_input_k_shift::~llm_graph_input_k_shift`](#llm_graph_input_k_shiftllm_graph_input_k_shift)
    - [`llm_graph_input_k_shift::set_input`](#llm_graph_input_k_shiftset_input)
- **Inherits From**:
    - [`llm_graph_input_i`](llama-graph.h.driver.md#llm_graph_input_i)

**Methods**

---
#### llm\_graph\_input\_k\_shift::llm\_graph\_input\_k\_shift<!-- {{#callable:llm_graph_input_k_shift::llm_graph_input_k_shift}} -->
The `llm_graph_input_k_shift` constructor initializes an instance of the class with a reference to a `llama_kv_cache_unified` object.
- **Inputs**:
    - `kv_self`: A pointer to a `llama_kv_cache_unified` object, which is used to initialize the `kv_self` member of the class.
- **Control Flow**:
    - The constructor takes a single argument, `kv_self`, which is a pointer to a `llama_kv_cache_unified` object.
    - It initializes the member variable `kv_self` with the provided argument.
- **Output**: There is no output from this constructor as it is used to initialize an object of the `llm_graph_input_k_shift` class.
- **See also**: [`llm_graph_input_k_shift`](#llm_graph_input_k_shift)  (Data Structure)


---
#### llm\_graph\_input\_k\_shift::\~llm\_graph\_input\_k\_shift<!-- {{#callable:llm_graph_input_k_shift::~llm_graph_input_k_shift}} -->
The destructor `~llm_graph_input_k_shift()` is a virtual default destructor for the `llm_graph_input_k_shift` class.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as a virtual function, ensuring that derived class destructors are called correctly when an object is deleted through a base class pointer.
    - The destructor is defaulted, meaning it has no custom implementation and relies on the compiler-generated destructor.
- **Output**: There is no output from this destructor as it is a default destructor with no custom logic.
- **See also**: [`llm_graph_input_k_shift`](#llm_graph_input_k_shift)  (Data Structure)


---
#### llm\_graph\_input\_k\_shift::set\_input<!-- {{#callable:llm_graph_input_k_shift::set_input}} -->
The `set_input` function sets the input for the `k_shift` tensor if it is not null by calling the `set_input_k_shift` method on the `kv_self` object.
- **Inputs**:
    - `ubatch`: A pointer to a `llama_ubatch` object, which is not used in the function.
- **Control Flow**:
    - The function begins by marking the `ubatch` parameter as unused with `GGML_UNUSED(ubatch);`.
    - It checks if the `k_shift` member variable is not null.
    - If `k_shift` is not null, it calls the `set_input_k_shift` method on the `kv_self` object, passing `k_shift` as an argument.
- **Output**: The function does not return any value.
- **See also**: [`llm_graph_input_k_shift`](#llm_graph_input_k_shift)  (Data Structure)



---
### llama\_kv\_cache\_unified<!-- {{#data_structure:llama_kv_cache_unified}} -->
- **Description**: [See definition](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)
- **Member Functions**:
    - [`llama_kv_cache_unified::~llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unifiedllama_kv_cache_unified)
    - [`llama_kv_cache_unified::llama_kv_cache_unified`](#llama_kv_cache_unifiedllama_kv_cache_unified)
    - [`llama_kv_cache_unified::clear`](#llama_kv_cache_unifiedclear)
    - [`llama_kv_cache_unified::seq_rm`](#llama_kv_cache_unifiedseq_rm)
    - [`llama_kv_cache_unified::seq_cp`](#llama_kv_cache_unifiedseq_cp)
    - [`llama_kv_cache_unified::seq_keep`](#llama_kv_cache_unifiedseq_keep)
    - [`llama_kv_cache_unified::seq_add`](#llama_kv_cache_unifiedseq_add)
    - [`llama_kv_cache_unified::seq_div`](#llama_kv_cache_unifiedseq_div)
    - [`llama_kv_cache_unified::seq_pos_min`](#llama_kv_cache_unifiedseq_pos_min)
    - [`llama_kv_cache_unified::seq_pos_max`](#llama_kv_cache_unifiedseq_pos_max)
    - [`llama_kv_cache_unified::init_batch`](#llama_kv_cache_unifiedinit_batch)
    - [`llama_kv_cache_unified::init_full`](#llama_kv_cache_unifiedinit_full)
    - [`llama_kv_cache_unified::init_update`](#llama_kv_cache_unifiedinit_update)
    - [`llama_kv_cache_unified::prepare`](#llama_kv_cache_unifiedprepare)
    - [`llama_kv_cache_unified::update`](#llama_kv_cache_unifiedupdate)
    - [`llama_kv_cache_unified::find_slot`](#llama_kv_cache_unifiedfind_slot)
    - [`llama_kv_cache_unified::apply_ubatch`](#llama_kv_cache_unifiedapply_ubatch)
    - [`llama_kv_cache_unified::get_can_shift`](#llama_kv_cache_unifiedget_can_shift)
    - [`llama_kv_cache_unified::get_size`](#llama_kv_cache_unifiedget_size)
    - [`llama_kv_cache_unified::get_has_shift`](#llama_kv_cache_unifiedget_has_shift)
    - [`llama_kv_cache_unified::get_n_kv`](#llama_kv_cache_unifiedget_n_kv)
    - [`llama_kv_cache_unified::get_k`](#llama_kv_cache_unifiedget_k)
    - [`llama_kv_cache_unified::get_v`](#llama_kv_cache_unifiedget_v)
    - [`llama_kv_cache_unified::cpy_k`](#llama_kv_cache_unifiedcpy_k)
    - [`llama_kv_cache_unified::cpy_v`](#llama_kv_cache_unifiedcpy_v)
    - [`llama_kv_cache_unified::set_input_kq_mask`](#llama_kv_cache_unifiedset_input_kq_mask)
    - [`llama_kv_cache_unified::set_input_k_shift`](#llama_kv_cache_unifiedset_input_k_shift)
    - [`llama_kv_cache_unified::set_input_pos_bucket`](#llama_kv_cache_unifiedset_input_pos_bucket)
    - [`llama_kv_cache_unified::total_size`](#llama_kv_cache_unifiedtotal_size)
    - [`llama_kv_cache_unified::size_k_bytes`](#llama_kv_cache_unifiedsize_k_bytes)
    - [`llama_kv_cache_unified::size_v_bytes`](#llama_kv_cache_unifiedsize_v_bytes)
    - [`llama_kv_cache_unified::build_rope_shift`](#llama_kv_cache_unifiedbuild_rope_shift)
    - [`llama_kv_cache_unified::build_graph_shift`](#llama_kv_cache_unifiedbuild_graph_shift)
    - [`llama_kv_cache_unified::build_graph_defrag`](#llama_kv_cache_unifiedbuild_graph_defrag)
    - [`llama_kv_cache_unified::defrag_prepare`](#llama_kv_cache_unifieddefrag_prepare)
    - [`llama_kv_cache_unified::is_masked_swa`](#llama_kv_cache_unifiedis_masked_swa)
    - [`llama_kv_cache_unified::state_write`](#llama_kv_cache_unifiedstate_write)
    - [`llama_kv_cache_unified::state_read`](#llama_kv_cache_unifiedstate_read)
    - [`llama_kv_cache_unified::state_write_meta`](#llama_kv_cache_unifiedstate_write_meta)
    - [`llama_kv_cache_unified::state_write_data`](#llama_kv_cache_unifiedstate_write_data)
    - [`llama_kv_cache_unified::state_read_meta`](#llama_kv_cache_unifiedstate_read_meta)
    - [`llama_kv_cache_unified::state_read_data`](#llama_kv_cache_unifiedstate_read_data)
    - [`llama_kv_cache_unified::get_padding`](#llama_kv_cache_unifiedget_padding)
- **Inherits From**:
    - `llama_memory_i`

**Methods**

---
#### llama\_kv\_cache\_unified::llama\_kv\_cache\_unified<!-- {{#callable:llama_kv_cache_unified::llama_kv_cache_unified}} -->
The `llama_kv_cache_unified` constructor initializes a key-value cache for a llama model, setting up contexts, layers, and buffers for efficient data handling.
- **Inputs**:
    - `model`: A reference to a `llama_model` object, which contains the model's hyperparameters and device information.
    - `filter`: A callback function used to filter out layers that should not be included in the cache.
    - `type_k`: The data type for the key tensors.
    - `type_v`: The data type for the value tensors.
    - `v_trans`: A boolean indicating whether the value tensor is transposed.
    - `offload`: A boolean indicating whether to offload computations to a device.
    - `kv_size`: The size of the key-value cache.
    - `n_seq_max`: The maximum number of sequences.
    - `n_pad`: The required padding for the cache.
    - `n_swa`: The number of sliding window attention (SWA) steps.
    - `swa_type`: The type of sliding window attention to use.
- **Control Flow**:
    - Assert that `kv_size` is divisible by `n_pad` to ensure proper padding.
    - Initialize a context map to manage different buffer types and create contexts as needed.
    - Set the initial head position to 0 and resize the cells vector to `kv_size`.
    - Iterate over each layer in the model, applying the filter callback to determine if the layer should be included.
    - For each included layer, determine the embedding sizes for keys and values, and set the appropriate device and buffer type.
    - Create a new context for the buffer type if it doesn't exist, and initialize key and value tensors for the layer.
    - Format the tensor names for debugging and logging purposes.
    - Map the model layer IDs to cache layer IDs and store the layer information.
    - Allocate and initialize buffers for each context to avoid NaNs in padding.
    - Log the total memory size used by the key and value tensors.
- **Output**: The function does not return a value; it initializes the internal state of the `llama_kv_cache_unified` object.
- **Functions called**:
    - [`ggml_tensor_overhead`](../ggml/src/ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_init`](../ggml/src/ggml.c.driver.md#ggml_init)
    - [`ggml_backend_dev_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_name)
    - [`ggml_new_tensor_2d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d)
    - [`ggml_format_name`](../ggml/src/ggml.c.driver.md#ggml_format_name)
    - [`ggml_backend_alloc_ctx_tensors_from_buft`](../ggml/src/ggml-alloc.c.driver.md#ggml_backend_alloc_ctx_tensors_from_buft)
    - [`ggml_backend_buffer_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_name)
    - [`ggml_backend_buffer_get_size`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_get_size)
    - [`ggml_backend_buffer_clear`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_clear)
    - [`llama_kv_cache_unified::size_k_bytes`](#llama_kv_cache_unifiedsize_k_bytes)
    - [`llama_kv_cache_unified::size_v_bytes`](#llama_kv_cache_unifiedsize_v_bytes)
    - [`ggml_type_name`](../ggml/src/ggml.c.driver.md#ggml_type_name)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::clear<!-- {{#callable:llama_kv_cache_unified::clear}} -->
The `clear` function resets the key-value cache by clearing all cells, resetting the head index, and clearing all backend buffers.
- **Inputs**: None
- **Control Flow**:
    - The function begins by resetting the `cells` object, which likely represents the storage for key-value pairs.
    - It sets the `head` variable to 0, indicating the start of the cache or buffer.
    - It iterates over each buffer in the `bufs` vector and calls [`ggml_backend_buffer_clear`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_clear) on each buffer to clear its contents.
- **Output**: The function does not return any value; it performs in-place modifications to the `llama_kv_cache_unified` object.
- **Functions called**:
    - [`ggml_backend_buffer_clear`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_clear)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::seq\_rm<!-- {{#callable:llama_kv_cache_unified::seq_rm}} -->
The `seq_rm` function removes sequences from a range of positions in the key-value cache, updating the head position if necessary.
- **Inputs**:
    - `seq_id`: An identifier for the sequence to be removed; if negative, any sequence is matched.
    - `p0`: The starting position of the range; if negative, defaults to 0.
    - `p1`: The ending position of the range; if negative, defaults to the maximum possible position.
- **Control Flow**:
    - Initialize `new_head` to the size of the `cells` vector.
    - If `p0` is negative, set it to 0; if `p1` is negative, set it to the maximum possible value for `llama_pos`.
    - If `seq_id` is non-negative, iterate over the `cells` vector and remove the sequence if it exists within the specified range, updating `new_head` if a slot is freed.
    - If `seq_id` is negative, iterate over the `cells` vector and remove any sequence within the specified range, updating `new_head` if a slot is freed.
    - If `new_head` is less than the current `head`, update `head` to `new_head`.
- **Output**: Returns `true` after attempting to remove sequences from the specified range.
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::seq\_cp<!-- {{#callable:llama_kv_cache_unified::seq_cp}} -->
The `seq_cp` function copies sequence identifiers from a source sequence to a destination sequence within a specified positional range in the `llama_kv_cache_unified` class.
- **Inputs**:
    - `seq_id_src`: The source sequence identifier from which sequence data is to be copied.
    - `seq_id_dst`: The destination sequence identifier to which sequence data is to be copied.
    - `p0`: The starting position of the range within which the sequence data is to be copied; defaults to 0 if negative.
    - `p1`: The ending position of the range within which the sequence data is to be copied; defaults to the maximum possible value if negative.
- **Control Flow**:
    - Check if the source and destination sequence identifiers are the same; if so, return immediately as no copying is needed.
    - Adjust `p0` to 0 if it is negative, ensuring the starting position is valid.
    - Adjust `p1` to the maximum possible value if it is negative, ensuring the ending position is valid.
    - Iterate over each cell in the `cells` vector.
    - For each cell, check if its position is within the range [p0, p1].
    - If the cell's position is within the range and it contains the source sequence identifier, add the destination sequence identifier to the cell.
- **Output**: The function does not return any value; it modifies the state of the `cells` vector by adding the destination sequence identifier to the appropriate cells.
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::seq\_keep<!-- {{#callable:llama_kv_cache_unified::seq_keep}} -->
The `seq_keep` function in the `llama_kv_cache_unified` class updates the cache to retain only the cells associated with a specific sequence ID and adjusts the head pointer if necessary.
- **Inputs**:
    - `seq_id`: A sequence identifier (`llama_seq_id`) used to determine which cells in the cache should be retained.
- **Control Flow**:
    - Initialize `new_head` to the size of the `cells` vector.
    - Iterate over each cell in the `cells` vector.
    - For each cell, check if it should be kept for the given `seq_id` using `cells.seq_keep(i, seq_id)`.
    - If a cell is to be kept and `new_head` is still set to the size of `cells`, update `new_head` to the current index `i`.
    - After the loop, if `new_head` is different from the size of `cells` and less than `head`, update `head` to `new_head`.
- **Output**: The function does not return a value; it modifies the internal state of the `llama_kv_cache_unified` object by potentially updating the `head` member variable.
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::seq\_add<!-- {{#callable:llama_kv_cache_unified::seq_add}} -->
The `seq_add` function adjusts the positions of sequence identifiers within a specified range in the key-value cache by a given shift amount.
- **Inputs**:
    - `seq_id`: A sequence identifier of type `llama_seq_id` that specifies which sequences to adjust.
    - `p0`: The starting position of the range, of type `llama_pos`, within which the sequence positions will be adjusted.
    - `p1`: The ending position of the range, of type `llama_pos`, within which the sequence positions will be adjusted.
    - `shift`: The amount, of type `llama_pos`, by which to adjust the sequence positions within the specified range.
- **Control Flow**:
    - Check if `shift` is zero; if so, return immediately as no adjustment is needed.
    - Initialize `new_head` to the current size of the `cells` vector.
    - Adjust `p0` and `p1` to ensure they are non-negative, setting `p0` to zero if negative and `p1` to the maximum possible value if negative.
    - Return early if `p0` equals `p1`, as there is no range to adjust.
    - Iterate over each cell in the `cells` vector.
    - For each cell, check if its position is within the range `[p0, p1)`; if not, continue to the next cell.
    - Check if the cell contains the specified `seq_id`; if so, attempt to add the `shift` to the cell's position.
    - If the position was successfully adjusted and `new_head` is still the size of `cells`, update `new_head` to the current index.
    - After the loop, update `head` to `new_head` if it was changed, otherwise reset `head` to zero.
- **Output**: The function does not return a value; it modifies the internal state of the `llama_kv_cache_unified` object by adjusting sequence positions and potentially updating the `head` index.
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::seq\_div<!-- {{#callable:llama_kv_cache_unified::seq_div}} -->
The `seq_div` function divides the positions of a specified sequence within a given range by a divisor, updating the cache accordingly.
- **Inputs**:
    - `seq_id`: The identifier of the sequence whose positions are to be divided.
    - `p0`: The starting position of the range within which the sequence positions are to be divided; if negative, it defaults to 0.
    - `p1`: The ending position of the range within which the sequence positions are to be divided; if negative, it defaults to the maximum possible value for `llama_pos`.
    - `d`: The divisor by which the sequence positions are to be divided; if it is 1, the function returns immediately without making any changes.
- **Control Flow**:
    - Check if the divisor `d` is 1, and if so, return immediately as no division is needed.
    - Adjust `p0` to 0 if it is negative, and `p1` to the maximum possible value if it is negative.
    - Return immediately if `p0` equals `p1`, as there is no range to process.
    - Iterate over each cell in the cache.
    - For each cell, check if its position is within the range `[p0, p1)` and if it contains the specified sequence `seq_id`.
    - If both conditions are met, divide the position of the sequence in that cell by `d`.
- **Output**: The function does not return a value; it modifies the positions of the specified sequence in the cache in place.
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::seq\_pos\_min<!-- {{#callable:llama_kv_cache_unified::seq_pos_min}} -->
The `seq_pos_min` function retrieves the minimum position of a sequence identified by `seq_id` from the `llama_kv_cache_unified` cache.
- **Inputs**:
    - `seq_id`: An identifier for the sequence whose minimum position is to be retrieved.
- **Control Flow**:
    - The function calls the `seq_pos_min` method on the `cells` member of the `llama_kv_cache_unified` class, passing the `seq_id` as an argument.
    - The `cells.seq_pos_min(seq_id)` method returns the minimum position of the specified sequence.
- **Output**: The function returns a `llama_pos` value representing the minimum position of the specified sequence in the cache.
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::seq\_pos\_max<!-- {{#callable:llama_kv_cache_unified::seq_pos_max}} -->
The `seq_pos_max` function retrieves the maximum position of a sequence identified by `seq_id` from the `llama_kv_cells_unified` object within the `llama_kv_cache_unified` class.
- **Inputs**:
    - `seq_id`: An identifier of type `llama_seq_id` representing the sequence for which the maximum position is to be retrieved.
- **Control Flow**:
    - The function calls the `seq_pos_max` method on the `cells` member of the `llama_kv_cache_unified` class, passing the `seq_id` as an argument.
    - The `cells` object, which is of type `llama_kv_cells_unified`, processes the request and returns the maximum position for the specified sequence.
- **Output**: The function returns a `llama_pos` value, which is the maximum position of the specified sequence within the cache.
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::init\_batch<!-- {{#callable:llama_kv_cache_unified::init_batch}} -->
The `init_batch` function initializes a batch of key-value cache entries for a given input batch, splitting it into smaller sub-batches and preparing them for processing.
- **Inputs**:
    - `batch`: A `llama_batch` object representing the input data to be processed.
    - `n_ubatch`: A `uint32_t` representing the number of tokens per sub-batch.
    - `embd_pooled`: A `bool` indicating whether embedding pooling is used (unused in this function).
    - `logits_all`: A `bool` indicating whether to process all logits.
- **Control Flow**:
    - The function begins by creating a [`llama_sbatch`](llama-batch.h.driver.md#llama_sbatchllama_sbatch) object from the input `batch`, using the number of embeddings from `hparams` and the `logits_all` flag.
    - It initializes an empty vector `ubatches` to store sub-batches.
    - A loop iterates while `sbatch.n_tokens` is greater than zero, splitting `sbatch` into sub-batches of size `n_ubatch` and adding them to `ubatches`.
    - The [`prepare`](#llama_kv_cache_unifiedprepare) function is called with `ubatches` to determine the head positions for each sub-batch.
    - If [`prepare`](#llama_kv_cache_unifiedprepare) returns an empty vector, indicating failure, the function returns a `llama_kv_cache_unified_state` with a failed status.
    - Otherwise, it returns a `llama_kv_cache_unified_state` initialized with the current object, `sbatch`, `heads`, and `ubatches`.
- **Output**: A `llama_memory_state_ptr` pointing to a `llama_kv_cache_unified_state` object, which represents the initialized state of the key-value cache for the batch.
- **Functions called**:
    - [`llama_sbatch::llama_sbatch`](llama-batch.h.driver.md#llama_sbatchllama_sbatch)
    - [`llama_kv_cache_unified::prepare`](#llama_kv_cache_unifiedprepare)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::init\_full<!-- {{#callable:llama_kv_cache_unified::init_full}} -->
The `init_full` function initializes a full memory state for the `llama_kv_cache_unified` class by creating a new `llama_kv_cache_unified_state` object.
- **Inputs**: None
- **Control Flow**:
    - The function calls `std::make_unique` to create a new `llama_kv_cache_unified_state` object, passing `this` (the current instance of `llama_kv_cache_unified`) as an argument.
    - The newly created `llama_kv_cache_unified_state` object is returned.
- **Output**: A `llama_memory_state_ptr` which is a smart pointer to a newly created `llama_kv_cache_unified_state` object.
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::init\_update<!-- {{#callable:llama_kv_cache_unified::init_update}} -->
The `init_update` function initializes and potentially updates the state of a key-value cache in a llama context, with optional defragmentation and shift operations.
- **Inputs**:
    - `lctx`: A pointer to a `llama_context` object, which provides context-specific parameters and methods for the operation.
    - `optimize`: A boolean flag indicating whether optimization, specifically defragmentation, should be performed.
- **Control Flow**:
    - Retrieve the current shift status using `get_has_shift()` and store it in `do_shift`.
    - Initialize a `defrag_info` object `dinfo` to store defragmentation details if needed.
    - Determine if defragmentation is necessary based on the `optimize` flag and the fragmentation threshold from `lctx`'s parameters.
    - Calculate the fragmentation level if the context size is large enough and compare it to the threshold to decide on defragmentation.
    - If defragmentation is required, prepare the defragmentation information using `defrag_prepare()` with the maximum number of nodes from `lctx`.
    - Create and return a new `llama_kv_cache_unified_state` object with the current cache, context, shift status, and defragmentation info.
- **Output**: A `llama_memory_state_ptr`, which is a smart pointer to a `llama_kv_cache_unified_state` object representing the updated state of the key-value cache.
- **Functions called**:
    - [`llama_kv_cells_unified::get_has_shift`](llama-kv-cells.h.driver.md#llama_kv_cells_unifiedget_has_shift)
    - [`llama_kv_cache_unified::defrag_prepare`](#llama_kv_cache_unifieddefrag_prepare)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::prepare<!-- {{#callable:llama_kv_cache_unified::prepare}} -->
The `prepare` function attempts to find suitable slots for a series of `llama_ubatch` objects in the cache, applies them, and then restores the cache to its original state if unsuccessful.
- **Inputs**:
    - `ubatches`: A constant reference to a vector of `llama_ubatch` objects, representing the batches to be placed in the cache.
- **Control Flow**:
    - Initialize an empty `ubatch_heads` vector `res` to store the new head positions.
    - Define a `state` structure to keep track of the old and new head positions and a copy of the old cells.
    - Initialize a `states` vector to store the state for each ubatch processed.
    - Set a `success` flag to true to track if all ubatches are successfully placed.
    - Iterate over each `ubatch` in `ubatches`.
    - For each `ubatch`, call [`find_slot`](#llama_kv_cache_unifiedfind_slot) to find a suitable slot in the cache.
    - If [`find_slot`](#llama_kv_cache_unifiedfind_slot) returns a negative value, set `success` to false and break the loop.
    - If a slot is found, push the new head position to `res` and store the current state in `states`.
    - Call [`apply_ubatch`](#llama_kv_cache_unifiedapply_ubatch) to place the `ubatch` in the cache at the found position.
    - After processing all ubatches, iterate over `states` in reverse order to restore the cache to its original state.
    - If `success` is false, return an empty vector.
    - Return the `res` vector containing the new head positions.
- **Output**: A vector of `uint32_t` representing the new head positions for each successfully placed `ubatch`, or an empty vector if placement was unsuccessful.
- **Functions called**:
    - [`llama_kv_cache_unified::find_slot`](#llama_kv_cache_unifiedfind_slot)
    - [`llama_kv_cache_unified::apply_ubatch`](#llama_kv_cache_unifiedapply_ubatch)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::update<!-- {{#callable:llama_kv_cache_unified::update}} -->
The `update` function in the `llama_kv_cache_unified` class updates the key-value cache by applying a K-shift and/or defragmenting the cache based on the provided context and defragmentation information.
- **Inputs**:
    - `lctx`: A pointer to a `llama_context` object, which provides the necessary context for the update operation.
    - `do_shift`: A boolean flag indicating whether a K-shift should be applied to the cache.
    - `dinfo`: A `defrag_info` object containing information about which cells in the cache need to be moved for defragmentation.
- **Control Flow**:
    - Initialize a boolean `updated` to false to track if any updates were made.
    - Retrieve the scheduler from the context `lctx`.
    - If `do_shift` is true, check if K-shift is supported; if not, abort the operation.
    - If K-shift is supported and needed, reset the scheduler, initialize a graph, build the graph for K-shift, allocate the graph, set inputs, and compute the graph; if successful, set `updated` to true.
    - Reset the shift in the cache cells after applying K-shift.
    - If `dinfo` is not empty, indicating defragmentation is needed, log the defragmentation process.
    - For each cell in `dinfo.ids`, move the cell to its new position if necessary, and reset the head to zero.
    - Reset the scheduler, initialize a graph, build the graph for defragmentation, allocate the graph, set inputs, and compute the graph; if successful, set `updated` to true.
    - Return the `updated` status indicating if any updates were made.
- **Output**: A boolean value indicating whether the cache was updated (true) or not (false).
- **Functions called**:
    - [`llama_kv_cache_unified::get_can_shift`](#llama_kv_cache_unifiedget_can_shift)
    - [`ggml_backend_sched_reset`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_reset)
    - [`llama_kv_cache_unified::build_graph_shift`](#llama_kv_cache_unifiedbuild_graph_shift)
    - [`ggml_backend_sched_alloc_graph`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_alloc_graph)
    - [`llama_kv_cache_unified::build_graph_defrag`](#llama_kv_cache_unifiedbuild_graph_defrag)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::find\_slot<!-- {{#callable:llama_kv_cache_unified::find_slot}} -->
The `find_slot` function searches for a contiguous slot in the key-value cache to insert a given batch of tokens, returning the starting position of the slot or -1 if no suitable slot is found.
- **Inputs**:
    - `ubatch`: A `llama_ubatch` object containing the batch of tokens to be inserted, including the number of tokens (`n_tokens`), their positions (`pos`), and sequence IDs (`seq_id`).
- **Control Flow**:
    - Initialize `n_tokens` from `ubatch.n_tokens` and set `head_cur` to the current head position.
    - Check if there are enough unused cells before the current head; if so, reset `head_cur` to 0 to start searching from the beginning.
    - If `n_tokens` exceeds the total size of cells, log an error and return -1.
    - Enter a loop to find a suitable slot:
    -   - If `head_cur + n_tokens` exceeds the cell size, reset `head_cur` to 0 and continue.
    -   - Initialize `seq_pos_min` to track minimum sequence positions for each sequence.
    -   - Iterate over each token in `ubatch` to check if the corresponding cell can be used:
    -     - A cell can be used if it is empty or occupied by a single sequence with a valid causal or SWA mask.
    -     - If a cell cannot be used, update `head_cur` and `n_tested`, then break the loop.
    -   - If a suitable slot is found, break the loop.
    -   - If `n_tested` exceeds the cell size, return -1 indicating failure to find a slot.
    - Return `head_cur` as the starting position of the found slot.
- **Output**: Returns an `int32_t` representing the starting position of the found slot in the cache, or -1 if no suitable slot is found.
- **Functions called**:
    - [`llama_kv_cache_unified::is_masked_swa`](#llama_kv_cache_unifiedis_masked_swa)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::apply\_ubatch<!-- {{#callable:llama_kv_cache_unified::apply_ubatch}} -->
The `apply_ubatch` function updates the key-value cache by inserting a batch of tokens at a specified head position, adjusting the cache's head pointer accordingly.
- **Inputs**:
    - `head_cur`: A `uint32_t` representing the current head position in the cache where the update will begin.
    - `ubatch`: A constant reference to a `llama_ubatch` object containing the tokens and sequence IDs to be inserted into the cache.
- **Control Flow**:
    - Iterate over each token in the `ubatch` using a loop.
    - For each token, check if the corresponding cell in the cache is not empty; if not, remove the existing data from that cell.
    - Set the position of the current cell in the cache to the position of the current token from `ubatch`.
    - Iterate over the sequence IDs for the current token and add each sequence ID to the current cell in the cache.
    - After processing all tokens, update the cache's head to point to the end of the newly inserted batch.
- **Output**: The function does not return a value; it modifies the state of the `llama_kv_cache_unified` object by updating its cells and head position.
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::get\_can\_shift<!-- {{#callable:llama_kv_cache_unified::get_can_shift}} -->
The `get_can_shift` function in the `llama_kv_cache_unified` class always returns `true`, indicating that shifting is supported.
- **Inputs**: None
- **Control Flow**:
    - The function is a constant method of the `llama_kv_cache_unified` class.
    - It simply returns the boolean value `true`.
- **Output**: A boolean value `true`, indicating that shifting is supported.
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::get\_size<!-- {{#callable:llama_kv_cache_unified::get_size}} -->
The `get_size` function returns the number of cells in the `llama_kv_cache_unified` cache.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `cells` member of the `llama_kv_cache_unified` class.
    - It calls the `size()` method on the `cells` object to get the number of elements.
    - The function returns the result of the `size()` method call.
- **Output**: The function returns a `uint32_t` representing the number of cells in the cache.
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::get\_has\_shift<!-- {{#callable:llama_kv_cache_unified::get_has_shift}} -->
The `get_has_shift` function checks if the `cells` object within the `llama_kv_cache_unified` class has a shift operation applied.
- **Inputs**: None
- **Control Flow**:
    - The function calls the `get_has_shift` method on the `cells` object.
    - It returns the result of the `cells.get_has_shift()` call.
- **Output**: A boolean value indicating whether the `cells` object has a shift operation applied.
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::get\_n\_kv<!-- {{#callable:llama_kv_cache_unified::get_n_kv}} -->
The `get_n_kv` function calculates the number of key-value pairs in the cache, considering padding and used cells.
- **Inputs**: None
- **Control Flow**:
    - The function calculates the maximum of `n_pad` and the padded value of `cells.used_max_p1()` using `GGML_PAD`.
    - It then returns the minimum of `cells.size()` and the previously calculated maximum value.
- **Output**: The function returns a `uint32_t` representing the number of key-value pairs in the cache, adjusted for padding.
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::get\_k<!-- {{#callable:llama_kv_cache_unified::get_k}} -->
The `get_k` function retrieves a 3D view of the 'k' tensor for a specified layer and number of key-value pairs from the llama_kv_cache_unified class.
- **Inputs**:
    - `ctx`: A pointer to a ggml_context object, which is used to manage memory and operations for the tensor.
    - `il`: An integer representing the layer index in the model for which the 'k' tensor is being retrieved.
    - `n_kv`: An unsigned integer specifying the number of key-value pairs to include in the view.
- **Control Flow**:
    - Retrieve the mapped layer index `ikv` from `map_layer_ids` using the input layer index `il`.
    - Access the 'k' tensor from the `layers` vector using the mapped index `ikv`.
    - Create and return a 3D view of the 'k' tensor using [`ggml_view_3d`](../ggml/src/ggml.c.driver.md#ggml_view_3d), with dimensions and row sizes determined by the hyperparameters and the input `n_kv`.
- **Output**: A pointer to a ggml_tensor representing a 3D view of the 'k' tensor for the specified layer and number of key-value pairs.
- **Functions called**:
    - [`ggml_view_3d`](../ggml/src/ggml.c.driver.md#ggml_view_3d)
    - [`ggml_row_size`](../ggml/src/ggml.c.driver.md#ggml_row_size)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::get\_v<!-- {{#callable:llama_kv_cache_unified::get_v}} -->
The `get_v` function retrieves a 3D view of the value tensor `v` for a specified layer and number of key-value pairs, considering whether the tensor is transposed or not.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` object, which is used to manage memory and operations for the tensor.
    - `il`: An integer representing the layer index in the model for which the value tensor is being retrieved.
    - `n_kv`: An unsigned integer specifying the number of key-value pairs to be considered in the view.
- **Control Flow**:
    - Retrieve the mapped layer index `ikv` using `il` from `map_layer_ids`.
    - Access the value tensor `v` from the `layers` vector using `ikv`.
    - Check if `v_trans` is false, indicating the tensor is not transposed.
    - If not transposed, create a 3D view of `v` with dimensions based on `hparams.n_embd_head_v`, `hparams.n_head_kv(il)`, and `n_kv`, and return it.
    - If transposed, create a 3D view of `v` with dimensions based on `n_kv`, `hparams.n_head_kv(il)`, and `hparams.n_embd_head_v`, and return it.
- **Output**: A pointer to a `ggml_tensor` representing a 3D view of the value tensor `v` for the specified layer and number of key-value pairs.
- **Functions called**:
    - [`ggml_view_3d`](../ggml/src/ggml.c.driver.md#ggml_view_3d)
    - [`ggml_row_size`](../ggml/src/ggml.c.driver.md#ggml_row_size)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::cpy\_k<!-- {{#callable:llama_kv_cache_unified::cpy_k}} -->
The `cpy_k` function copies the current key tensor `k_cur` into a specific view of the key tensor `k` associated with a given layer and head in the llama_kv_cache_unified class.
- **Inputs**:
    - `ctx`: A pointer to a ggml_context object, which provides the context for tensor operations.
    - `k_cur`: A pointer to a ggml_tensor object representing the current key tensor to be copied.
    - `il`: An integer representing the layer index in the model.
    - `head_cur`: An unsigned integer representing the current head index for the operation.
- **Control Flow**:
    - Retrieve the mapped layer index `ikv` for the given layer index `il` using `map_layer_ids`.
    - Access the key tensor `k` for the specified layer using the `ikv` index.
    - Determine the number of tokens `n_tokens` from the third dimension of `k_cur`.
    - Create a 1D view `k_view` of the key tensor `k` with dimensions based on `n_tokens` and the embedding size for the given layer and head.
    - Copy the contents of `k_cur` into `k_view` using the [`ggml_cpy`](../ggml/src/ggml.c.driver.md#ggml_cpy) function.
    - Return the resulting tensor from the copy operation.
- **Output**: A pointer to a ggml_tensor object representing the result of copying `k_cur` into the specified view of `k`.
- **Functions called**:
    - [`ggml_view_1d`](../ggml/src/ggml.c.driver.md#ggml_view_1d)
    - [`ggml_row_size`](../ggml/src/ggml.c.driver.md#ggml_row_size)
    - [`ggml_cpy`](../ggml/src/ggml.c.driver.md#ggml_cpy)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::cpy\_v<!-- {{#callable:llama_kv_cache_unified::cpy_v}} -->
The `cpy_v` function copies the current value tensor `v_cur` into a specific view of the value cache tensor `v` for a given layer and head position, potentially transposing it if required.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which provides the context for tensor operations.
    - `v_cur`: A pointer to the current value tensor that needs to be copied into the cache.
    - `il`: An integer representing the layer index in the model.
    - `head_cur`: An unsigned integer representing the current head position in the cache.
- **Control Flow**:
    - Retrieve the mapped layer index `ikv` using `il` from `map_layer_ids`.
    - Access the value tensor `v` for the layer `ikv`.
    - Determine the number of tokens `n_tokens` from the third dimension of `v_cur`.
    - Reshape `v_cur` to a 2D tensor with dimensions based on `n_embd_v_gqa(il)` and `n_tokens`.
    - If `v_trans` is false, create a 1D view `v_view` of `v` with appropriate dimensions and offsets.
    - If `v_trans` is true, create a 2D view `v_view` of `v`, transpose `v_cur`, and adjust dimensions and offsets accordingly.
    - Copy the reshaped `v_cur` into `v_view` using [`ggml_cpy`](../ggml/src/ggml.c.driver.md#ggml_cpy).
- **Output**: Returns a pointer to the resulting `ggml_tensor` after copying `v_cur` into the view `v_view`.
- **Functions called**:
    - [`ggml_reshape_2d`](../ggml/src/ggml.c.driver.md#ggml_reshape_2d)
    - [`ggml_view_1d`](../ggml/src/ggml.c.driver.md#ggml_view_1d)
    - [`ggml_row_size`](../ggml/src/ggml.c.driver.md#ggml_row_size)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_element_size`](../ggml/src/ggml.c.driver.md#ggml_element_size)
    - [`ggml_transpose`](../ggml/src/ggml.c.driver.md#ggml_transpose)
    - [`ggml_cpy`](../ggml/src/ggml.c.driver.md#ggml_cpy)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::set\_input\_kq\_mask<!-- {{#callable:llama_kv_cache_unified::set_input_kq_mask}} -->
The `set_input_kq_mask` function sets the key-query mask for a given tensor based on the sequence information and whether causal attention is applied.
- **Inputs**:
    - `dst`: A pointer to a `ggml_tensor` where the mask will be stored.
    - `ubatch`: A pointer to a `llama_ubatch` structure containing sequence information for the batch.
    - `causal_attn`: A boolean indicating whether causal attention is applied.
- **Control Flow**:
    - Retrieve the number of tokens, sequence tokens, and sequences from the `ubatch` structure.
    - Assert that the destination tensor's buffer is hosted on the CPU and cast its data to a float pointer.
    - Iterate over each sequence and token in the `ubatch`.
    - For each token, iterate over the key-value cells in the destination tensor.
    - Determine if each cell should be masked based on whether it is empty, belongs to the same sequence, is a future token (if causal attention is enabled), or is masked by SWA.
    - Set the mask value to negative infinity if masked, otherwise calculate a value based on the position difference if ALiBi is used.
    - Store the calculated mask value in the destination tensor.
    - After processing all tokens, mask any padded tokens in the destination tensor.
- **Output**: The function modifies the `dst` tensor in-place, setting its data to represent the key-query mask for the input batch.
- **Functions called**:
    - [`ggml_backend_buffer_is_host`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_is_host)
    - [`llama_kv_cache_unified::is_masked_swa`](#llama_kv_cache_unifiedis_masked_swa)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::set\_input\_k\_shift<!-- {{#callable:llama_kv_cache_unified::set_input_k_shift}} -->
The `set_input_k_shift` function populates a given tensor with shift values from a key-value cache, setting zero for empty cells.
- **Inputs**:
    - `dst`: A pointer to a `ggml_tensor` object that will be populated with shift values from the key-value cache.
- **Control Flow**:
    - Assert that the buffer of the destination tensor is hosted on the CPU using `GGML_ASSERT`.
    - Cast the data pointer of the destination tensor to an `int32_t` pointer.
    - Iterate over each cell in the `cells` vector.
    - For each cell, check if it is empty using `cells.is_empty(i)`.
    - If the cell is empty, set the corresponding position in the destination data to 0.
    - If the cell is not empty, set the corresponding position in the destination data to the shift value obtained from `cells.get_shift(i)`.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place.
- **Functions called**:
    - [`ggml_backend_buffer_is_host`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_is_host)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::set\_input\_pos\_bucket<!-- {{#callable:llama_kv_cache_unified::set_input_pos_bucket}} -->
The `set_input_pos_bucket` function populates a destination tensor with relative position bucket values for a given batch of tokens, based on their positions and the state of the key-value cache.
- **Inputs**:
    - `dst`: A pointer to a `ggml_tensor` that will be populated with position bucket values.
    - `ubatch`: A pointer to a `llama_ubatch` structure containing the batch of tokens and their positions.
- **Control Flow**:
    - Retrieve the number of tokens from the `ubatch` structure.
    - Assert that the destination tensor's buffer is hosted on the CPU and that the sequences in `ubatch` are not equal.
    - Cast the data pointer of the destination tensor to an `int32_t` pointer.
    - Retrieve the number of key-value pairs from the destination tensor's dimensions.
    - Iterate over a single head (as the loop is fixed to one iteration).
    - For each token in the batch, iterate over each key-value pair.
    - For each key-value pair, determine the position `p0` from the cache, defaulting to -1 if the cell is empty.
    - Calculate the relative position bucket using [`llama_relative_position_bucket`](llama-graph.cpp.driver.md#llama_relative_position_bucket) and store it in the destination tensor.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place.
- **Functions called**:
    - [`ggml_backend_buffer_is_host`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_is_host)
    - [`llama_relative_position_bucket`](llama-graph.cpp.driver.md#llama_relative_position_bucket)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::total\_size<!-- {{#callable:llama_kv_cache_unified::total_size}} -->
The `total_size` function calculates the total size of all backend buffers in the `llama_kv_cache_unified` class.
- **Inputs**: None
- **Control Flow**:
    - Initialize a variable `size` to 0.
    - Iterate over each buffer in the `bufs` vector.
    - For each buffer, retrieve its size using [`ggml_backend_buffer_get_size`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_get_size) and add it to `size`.
    - Return the accumulated `size`.
- **Output**: The function returns the total size as a `size_t` value, representing the sum of sizes of all backend buffers.
- **Functions called**:
    - [`ggml_backend_buffer_get_size`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_get_size)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::size\_k\_bytes<!-- {{#callable:llama_kv_cache_unified::size_k_bytes}} -->
The `size_k_bytes` function calculates the total memory size in bytes used by the 'k' tensors across all layers in the `llama_kv_cache_unified` class.
- **Inputs**: None
- **Control Flow**:
    - Initialize a variable `size_k_bytes` to 0 to accumulate the total size.
    - Iterate over each `layer` in the `layers` vector.
    - For each `layer`, add the number of bytes used by its 'k' tensor to `size_k_bytes` using the [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes) function.
    - Return the accumulated `size_k_bytes`.
- **Output**: The function returns a `size_t` representing the total size in bytes of all 'k' tensors in the cache.
- **Functions called**:
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::size\_v\_bytes<!-- {{#callable:llama_kv_cache_unified::size_v_bytes}} -->
The `size_v_bytes` function calculates the total memory size in bytes used by the 'v' tensors across all layers in the `llama_kv_cache_unified` class.
- **Inputs**: None
- **Control Flow**:
    - Initialize a variable `size_v_bytes` to 0 to accumulate the total size.
    - Iterate over each `layer` in the `layers` vector of the `llama_kv_cache_unified` class.
    - For each `layer`, add the size in bytes of the `v` tensor (obtained using [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)) to `size_v_bytes`.
    - Return the accumulated `size_v_bytes` value.
- **Output**: The function returns a `size_t` value representing the total size in bytes of all 'v' tensors in the cache.
- **Functions called**:
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::build\_rope\_shift<!-- {{#callable:llama_kv_cache_unified::build_rope_shift}} -->
The `build_rope_shift` function applies a rotational position encoding (RoPE) transformation to a given tensor, potentially involving dequantization and re-quantization, based on various parameters and conditions.
- **Inputs**:
    - `cparams`: A constant reference to a `llama_cparams` object containing configuration parameters for the operation.
    - `ctx`: A pointer to a `ggml_context` object, which provides the context for tensor operations.
    - `cur`: A pointer to a `ggml_tensor` object representing the current tensor to be transformed.
    - `shift`: A pointer to a `ggml_tensor` object representing the shift tensor used in the RoPE transformation.
    - `factors`: A pointer to a `ggml_tensor` object representing the factors used in the RoPE transformation.
    - `freq_base`: A float representing the base frequency for the RoPE transformation.
    - `freq_scale`: A float representing the frequency scale for the RoPE transformation.
- **Control Flow**:
    - Retrieve original context size and various parameters from `cparams` and `hparams`.
    - Determine the type of RoPE to use based on `hparams.rope_type` and apply a workaround if necessary.
    - Calculate the attention factor based on the model architecture and frequency scale.
    - Check if the current tensor `cur` is quantized.
    - If quantized, dequantize `cur` to float32, apply the RoPE transformation using [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext), and then re-quantize back to the original type.
    - If not quantized, apply the RoPE transformation in place using [`ggml_rope_ext_inplace`](../ggml/src/ggml.c.driver.md#ggml_rope_ext_inplace).
    - Return the transformed tensor.
- **Output**: A pointer to a `ggml_tensor` object representing the transformed tensor after applying the RoPE shift.
- **Functions called**:
    - [`ggml_is_quantized`](../ggml/src/ggml.c.driver.md#ggml_is_quantized)
    - [`ggml_cast`](../ggml/src/ggml.c.driver.md#ggml_cast)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_cpy`](../ggml/src/ggml.c.driver.md#ggml_cpy)
    - [`ggml_rope_ext_inplace`](../ggml/src/ggml.c.driver.md#ggml_rope_ext_inplace)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::build\_graph\_shift<!-- {{#callable:llama_kv_cache_unified::build_graph_shift}} -->
The `build_graph_shift` function constructs a computation graph for shifting key-value pairs in a neural network model using RoPE (Rotary Position Embedding) transformations.
- **Inputs**:
    - `cparams`: A constant reference to a `llama_cparams` object containing configuration parameters for the model.
    - `ctx`: A pointer to a `ggml_context` object, which is used for managing memory and computation resources.
    - `gf`: A pointer to a `ggml_cgraph` object, which represents the computation graph to be built.
- **Control Flow**:
    - Create a unique pointer `res` to an `llm_graph_result` object to store the result of the graph building process.
    - Retrieve the number of embedding heads for keys from the model's hyperparameters.
    - Create a unique pointer `inp` to an `llm_graph_input_k_shift` object, initializing it with the current object (`this`).
    - Allocate a new 1D tensor `k_shift` in the context `ctx` with integer type and size equal to the context size from `cparams`, and set it as an input tensor.
    - Iterate over each layer in the `layers` vector of the current object.
    - For each layer, retrieve the number of key-value heads and the number of embedding dimensions for keys from the model's hyperparameters.
    - Retrieve the base and scale frequencies for RoPE from the model for the current layer.
    - Get the RoPE factors tensor for the current layer from the model.
    - Create a 3D view of the key tensor for the current layer using the context `ctx` and the retrieved dimensions.
    - Call [`build_rope_shift`](#llama_kv_cache_unifiedbuild_rope_shift) to apply the RoPE transformation to the key tensor, using the `k_shift` tensor and RoPE factors.
    - Expand the computation graph `gf` with the transformed tensor `cur`.
    - Add the input `inp` to the result `res`.
- **Output**: A unique pointer to an `llm_graph_result` object, which contains the constructed computation graph for the key-value shift operation.
- **Functions called**:
    - [`ggml_new_tensor_1d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input)
    - [`ggml_view_3d`](../ggml/src/ggml.c.driver.md#ggml_view_3d)
    - [`ggml_row_size`](../ggml/src/ggml.c.driver.md#ggml_row_size)
    - [`llama_kv_cache_unified::build_rope_shift`](#llama_kv_cache_unifiedbuild_rope_shift)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::build\_graph\_defrag<!-- {{#callable:llama_kv_cache_unified::build_graph_defrag}} -->
The `build_graph_defrag` function constructs a defragmentation graph for a key-value cache using the provided defragmentation information.
- **Inputs**:
    - `cparams`: A constant reference to a `llama_cparams` object containing configuration parameters.
    - `ctx`: A pointer to a `ggml_context` object used for managing tensor operations.
    - `gf`: A pointer to a `ggml_cgraph` object representing the computation graph to be built.
    - `dinfo`: A constant reference to a `defrag_info` object containing information about which cells need to be moved during defragmentation.
- **Control Flow**:
    - Initialize a unique pointer `res` to a new `llm_graph_result` object.
    - Retrieve the `ids` vector from the `dinfo` object, which indicates the target positions for each cell.
    - Iterate over each index `i` in the `ids` vector.
    - For each index, check if the current position `i` is different from the target position `id` and if `id` is not equal to the size of `ids`.
    - If the current position needs to be moved, calculate the number of consecutive cells `nm` that can be moved together.
    - For each layer in the `layers` vector, create 2D views of the source and destination positions for both keys and values using [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d).
    - Copy the data from the source view to the destination view for both keys and values using [`ggml_cpy`](../ggml/src/ggml.c.driver.md#ggml_cpy) and expand the computation graph with [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand).
    - Adjust the loop index `i` to skip over the cells that have been moved.
- **Output**: A unique pointer to an `llm_graph_result` object representing the result of the defragmentation graph construction.
- **Functions called**:
    - [`ggml_row_size`](../ggml/src/ggml.c.driver.md#ggml_row_size)
    - [`ggml_type_size`](../ggml/src/ggml.c.driver.md#ggml_type_size)
    - [`ggml_backend_tensor_get`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_get)
    - [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
    - [`ggml_cpy`](../ggml/src/ggml.c.driver.md#ggml_cpy)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::defrag\_prepare<!-- {{#callable:llama_kv_cache_unified::defrag_prepare}} -->
The `defrag_prepare` function identifies and prepares a plan to defragment the key-value cache by moving non-empty cells to fill empty slots, optimizing the cache's memory usage.
- **Inputs**:
    - `n_max_nodes`: An integer representing the maximum number of nodes available for the defragmentation process.
- **Control Flow**:
    - Initialize variables for the number of layers, maximum used key-value cells, and used cells.
    - Assert that the number of used cells does not exceed the maximum key-value cells.
    - Calculate the maximum number of moves allowed based on the input `n_max_nodes` and the number of layers.
    - Resize the `ids` vector in the `defrag_info` structure to the size of `n_kv`, initializing all elements to `n_kv`.
    - Iterate over each used cell index `i0` to find empty cells and determine the size of contiguous empty slots (holes).
    - For each hole, find a corresponding number of non-empty cells from the end of the cache to fill the hole.
    - Move the identified non-empty cells to the empty slots, updating the `ids` vector to reflect the new positions.
    - Break the loop if the maximum number of moves is reached or if no more moves are possible.
    - Log the number of moves and expected graph nodes if any moves were made.
    - Return the `defrag_info` structure containing the move plan.
- **Output**: Returns a `defrag_info` structure containing a vector `ids` that maps each cell to its new position after defragmentation, or an empty `defrag_info` if no moves are made.
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::is\_masked\_swa<!-- {{#callable:llama_kv_cache_unified::is_masked_swa}} -->
The `is_masked_swa` function determines if a given range of positions is masked based on the SWA (Sliding Window Attention) type and parameters.
- **Inputs**:
    - `p0`: The starting position of the range to check, of type `llama_pos`.
    - `p1`: The ending position of the range to check, of type `llama_pos`.
- **Control Flow**:
    - The function asserts that both `p0` and `p1` are non-negative.
    - It checks the `swa_type` to determine the masking logic.
    - If `swa_type` is `LLAMA_SWA_TYPE_NONE`, no masking is applied.
    - If `swa_type` is `LLAMA_SWA_TYPE_STANDARD`, it checks if the difference `p1 - p0` is greater than or equal to `n_swa`; if so, it returns `true` indicating the range is masked.
    - If `swa_type` is `LLAMA_SWA_TYPE_CHUNKED`, it calculates the start of the chunk for `p1` and checks if `p0` is less than this chunk start; if so, it returns `true` indicating the range is masked.
    - If none of the conditions for masking are met, it returns `false`.
- **Output**: A boolean value indicating whether the range from `p0` to `p1` is masked based on the SWA type and parameters.
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::state\_write<!-- {{#callable:llama_kv_cache_unified::state_write}} -->
The `state_write` function writes the state of the key-value cache to an output stream, focusing on cells associated with a specific sequence ID.
- **Inputs**:
    - `io`: A reference to an object implementing the `llama_io_write_i` interface, used for writing data.
    - `seq_id`: An identifier for the sequence whose associated cells are to be written; if -1, all cells are considered.
- **Control Flow**:
    - Initialize a vector to store ranges of cells and a counter for the number of cells.
    - Iterate over all cells to identify those associated with the given `seq_id` and determine contiguous ranges of such cells.
    - Store the start and end indices of each range in the `cell_ranges` vector.
    - Perform a debug check to ensure the total number of cells matches the sum of cells in the identified ranges.
    - Write the total cell count to the output stream using the `io` object.
    - Call [`state_write_meta`](#llama_kv_cache_unifiedstate_write_meta) to write metadata for the identified cell ranges.
    - Call [`state_write_data`](#llama_kv_cache_unifiedstate_write_data) to write the actual data for the identified cell ranges.
- **Output**: The function does not return a value; it writes data to the provided `io` object.
- **Functions called**:
    - [`llama_kv_cache_unified::state_write_meta`](#llama_kv_cache_unifiedstate_write_meta)
    - [`llama_kv_cache_unified::state_write_data`](#llama_kv_cache_unifiedstate_write_data)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::state\_read<!-- {{#callable:llama_kv_cache_unified::state_read}} -->
The `state_read` function reads the state of the key-value cache from an input stream and restores it, handling errors by clearing or removing sequences if necessary.
- **Inputs**:
    - `io`: A reference to an object implementing the `llama_io_read_i` interface, used for reading data from an input stream.
    - `seq_id`: An identifier for the sequence to be restored; if set to -1, the entire cache is restored.
- **Control Flow**:
    - Read the number of cells (`cell_count`) from the input stream using `io.read_to`.
    - Initialize a boolean `res` to true and update it by calling [`state_read_meta`](#llama_kv_cache_unifiedstate_read_meta) and [`state_read_data`](#llama_kv_cache_unifiedstate_read_data) with `io`, `cell_count`, and `seq_id`.
    - If `res` is false, check if `seq_id` is -1; if true, call `clear()` to reset the cache, otherwise call `seq_rm(seq_id, -1, -1)` to remove the sequence.
    - Throw a `std::runtime_error` if the restoration fails.
- **Output**: The function does not return a value but throws an exception if the restoration process fails.
- **Functions called**:
    - [`llama_kv_cache_unified::state_read_meta`](#llama_kv_cache_unifiedstate_read_meta)
    - [`llama_kv_cache_unified::state_read_data`](#llama_kv_cache_unifiedstate_read_data)
    - [`llama_kv_cache_unified::clear`](#llama_kv_cache_unifiedclear)
    - [`llama_kv_cells_unified::seq_rm`](llama-kv-cells.h.driver.md#llama_kv_cells_unifiedseq_rm)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::state\_write\_meta<!-- {{#callable:llama_kv_cache_unified::state_write_meta}} -->
The `state_write_meta` function writes metadata about key-value cache cells to an output stream, including their positions and associated sequence IDs.
- **Inputs**:
    - `io`: A reference to an object implementing the `llama_io_write_i` interface, used for writing data.
    - `cell_ranges`: A vector of pairs, where each pair represents a range of cell indices (from inclusive, to exclusive) to be processed.
    - `seq_id`: A sequence ID to filter which sequence IDs to include in the metadata, or -1 to include all sequence IDs.
- **Control Flow**:
    - Iterate over each range in `cell_ranges`.
    - For each cell index in the current range, initialize an empty vector `seq_ids` to store sequence IDs.
    - Iterate over possible sequence IDs up to `n_seq_max`.
    - If the current sequence ID matches `seq_id` or `seq_id` is -1, check if the cell has this sequence ID and add it to `seq_ids` if true.
    - Retrieve the position of the current cell using `cells.pos_get(i)`.
    - Determine the number of sequence IDs (`n_seq_id`) in `seq_ids`.
    - Write the position and `n_seq_id` to the `io` stream.
    - Iterate over `seq_ids` and write each sequence ID to the `io` stream.
- **Output**: The function does not return a value; it writes metadata to the provided `io` stream.
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::state\_write\_data<!-- {{#callable:llama_kv_cache_unified::state_write_data}} -->
The `state_write_data` function writes the state data of the key-value cache to an output stream, handling both transposed and non-transposed value tensors.
- **Inputs**:
    - `io`: A reference to an `llama_io_write_i` object used for writing data to an output stream.
    - `cell_ranges`: A constant reference to a vector of pairs of `uint32_t`, representing the ranges of cells to be written.
- **Control Flow**:
    - Determine if the value tensor is transposed and the number of layers in the cache.
    - Write the transposition state and number of layers to the output stream.
    - Iterate over each layer to write key data: write key type, row size, and key tensor data for each cell range.
    - If the value tensor is not transposed, iterate over each layer to write value data: write value type, row size, and value tensor data for each cell range.
    - If the value tensor is transposed, iterate over each layer to write value data: write value type, element size, GQA embedding size, and value tensor data for each element range in each cell range.
- **Output**: The function does not return a value; it writes data to the provided `io` output stream.
- **Functions called**:
    - [`ggml_row_size`](../ggml/src/ggml.c.driver.md#ggml_row_size)
    - [`ggml_type_size`](../ggml/src/ggml.c.driver.md#ggml_type_size)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::state\_read\_meta<!-- {{#callable:llama_kv_cache_unified::state_read_meta}} -->
The `state_read_meta` function reads metadata from an input stream to restore the state of a key-value cache, either for a single sequence or the entire cache, depending on the provided sequence ID.
- **Inputs**:
    - `io`: A reference to an `llama_io_read_i` object used for reading data from an input stream.
    - `cell_count`: A `uint32_t` representing the number of cells to read from the input stream.
    - `dest_seq_id`: A `llama_seq_id` indicating the destination sequence ID; if not -1, it specifies a single sequence to restore.
- **Control Flow**:
    - Check if `dest_seq_id` is not -1 to determine if restoring a single sequence or the entire cache.
    - If restoring a single sequence, remove existing sequence data for `dest_seq_id` and prepare a batch for the specified number of cells.
    - Read position and sequence ID data for each cell, ensuring the sequence ID is valid, and store it in the batch.
    - Find a slot in the cache for the batch and apply it, updating the head position.
    - If restoring the entire cache, clear existing cache data and read position and sequence ID data for each cell, adding valid sequence IDs to the cache.
    - Return true if the operation is successful, otherwise return false if any errors occur.
- **Output**: A boolean value indicating whether the metadata was successfully read and applied to the cache.
- **Functions called**:
    - [`llama_kv_cells_unified::seq_rm`](llama-kv-cells.h.driver.md#llama_kv_cells_unifiedseq_rm)
    - [`llama_kv_cache_unified::find_slot`](#llama_kv_cache_unifiedfind_slot)
    - [`llama_kv_cache_unified::apply_ubatch`](#llama_kv_cache_unifiedapply_ubatch)
    - [`llama_kv_cache_unified::clear`](#llama_kv_cache_unifiedclear)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::state\_read\_data<!-- {{#callable:llama_kv_cache_unified::state_read_data}} -->
The `state_read_data` function reads and restores the state of a key-value cache from an input stream, ensuring compatibility with the current cache configuration.
- **Inputs**:
    - `io`: An input stream object of type `llama_io_read_i` used to read data from the source.
    - `cell_count`: A `uint32_t` representing the number of cells to be read and restored in the cache.
- **Control Flow**:
    - Read the transposition flag `v_trans` and the number of layers `n_layer` from the input stream.
    - Check if the number of layers `n_layer` matches the size of the current layers; log an error and return false if they don't match.
    - Check if `cell_count` exceeds the size of the current cells; log an error and return false if it does.
    - Check if the transposition flag `v_trans` matches the current cache's transposition setting; log an error and return false if it doesn't match.
    - Iterate over each layer to read and validate the key type and row size, logging errors and returning false if mismatches occur.
    - If `cell_count` is non-zero, read and set the keys for the entire cell range using the input stream.
    - If the cache is not transposed, iterate over each layer to read and validate the value type and row size, logging errors and returning false if mismatches occur, then read and set the values for the entire cell range.
    - If the cache is transposed, iterate over each layer to read and validate the value type, element size, and GQA embedding size, logging errors and returning false if mismatches occur, then read and set the values for each row in the transposed matrix.
- **Output**: Returns a boolean value `true` if the state is successfully read and restored, or `false` if any validation checks fail.
- **Functions called**:
    - [`ggml_row_size`](../ggml/src/ggml.c.driver.md#ggml_row_size)
    - [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`ggml_type_size`](../ggml/src/ggml.c.driver.md#ggml_type_size)
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)


---
#### llama\_kv\_cache\_unified::get\_padding<!-- {{#callable:llama_kv_cache_unified::get_padding}} -->
The `get_padding` function determines the required padding size based on whether flash attention is enabled in the given parameters.
- **Inputs**:
    - `cparams`: A constant reference to a `llama_cparams` object, which contains configuration parameters including whether flash attention is enabled.
- **Control Flow**:
    - Check if the `flash_attn` member of `cparams` is true.
    - If `flash_attn` is true, return 256u.
    - If `flash_attn` is false, return 32u.
- **Output**: Returns a `uint32_t` representing the padding size, either 256 or 32, depending on the flash attention setting.
- **See also**: [`llama_kv_cache_unified`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified)  (Data Structure)



---
### llama\_kv\_cache\_unified\_state<!-- {{#data_structure:llama_kv_cache_unified_state}} -->
- **Description**: [See definition](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified_state)
- **Member Functions**:
    - [`llama_kv_cache_unified_state::llama_kv_cache_unified_state`](#llama_kv_cache_unified_statellama_kv_cache_unified_state)
    - [`llama_kv_cache_unified_state::llama_kv_cache_unified_state`](#llama_kv_cache_unified_statellama_kv_cache_unified_state)
    - [`llama_kv_cache_unified_state::llama_kv_cache_unified_state`](#llama_kv_cache_unified_statellama_kv_cache_unified_state)
    - [`llama_kv_cache_unified_state::llama_kv_cache_unified_state`](#llama_kv_cache_unified_statellama_kv_cache_unified_state)
    - [`llama_kv_cache_unified_state::~llama_kv_cache_unified_state`](#llama_kv_cache_unified_statellama_kv_cache_unified_state)
    - [`llama_kv_cache_unified_state::next`](#llama_kv_cache_unified_statenext)
    - [`llama_kv_cache_unified_state::apply`](#llama_kv_cache_unified_stateapply)
    - [`llama_kv_cache_unified_state::out_ids`](#llama_kv_cache_unified_stateout_ids)
    - [`llama_kv_cache_unified_state::get_status`](#llama_kv_cache_unified_stateget_status)
    - [`llama_kv_cache_unified_state::get_ubatch`](#llama_kv_cache_unified_stateget_ubatch)
    - [`llama_kv_cache_unified_state::get_n_kv`](#llama_kv_cache_unified_stateget_n_kv)
    - [`llama_kv_cache_unified_state::get_k`](#llama_kv_cache_unified_stateget_k)
    - [`llama_kv_cache_unified_state::get_v`](#llama_kv_cache_unified_stateget_v)
    - [`llama_kv_cache_unified_state::cpy_k`](#llama_kv_cache_unified_statecpy_k)
    - [`llama_kv_cache_unified_state::cpy_v`](#llama_kv_cache_unified_statecpy_v)
    - [`llama_kv_cache_unified_state::set_input_k_shift`](#llama_kv_cache_unified_stateset_input_k_shift)
    - [`llama_kv_cache_unified_state::set_input_kq_mask`](#llama_kv_cache_unified_stateset_input_kq_mask)
    - [`llama_kv_cache_unified_state::set_input_pos_bucket`](#llama_kv_cache_unified_stateset_input_pos_bucket)
- **Inherits From**:
    - `llama_memory_state_i`

**Methods**

---
#### llama\_kv\_cache\_unified\_state::llama\_kv\_cache\_unified\_state<!-- {{#callable:llama_kv_cache_unified_state::llama_kv_cache_unified_state}} -->
The `llama_kv_cache_unified_state` constructor initializes an instance of the class with a given memory status.
- **Inputs**:
    - `status`: A `llama_memory_status` value representing the memory status to initialize the state with.
- **Control Flow**:
    - The constructor initializes the `status` member variable with the provided `status` argument.
- **Output**: An instance of `llama_kv_cache_unified_state` is created with the specified memory status.
- **See also**: [`llama_kv_cache_unified_state`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_state::\~llama\_kv\_cache\_unified\_state<!-- {{#callable:llama_kv_cache_unified_state::~llama_kv_cache_unified_state}} -->
The destructor `~llama_kv_cache_unified_state` is a default destructor for the `llama_kv_cache_unified_state` class, which performs no specific actions upon object destruction.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as `= default;`, indicating that it uses the compiler-generated default implementation.
    - No custom cleanup or resource deallocation is performed in this destructor.
- **Output**: There is no output from this destructor as it performs no operations.
- **See also**: [`llama_kv_cache_unified_state`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_state::next<!-- {{#callable:llama_kv_cache_unified_state::next}} -->
The `next` function in the `llama_kv_cache_unified_state` class increments the index of the next ubatch to process and checks if there are more ubatches to process.
- **Inputs**: None
- **Control Flow**:
    - The function asserts that the current status is `LLAMA_MEMORY_STATUS_SUCCESS`.
    - It increments the `i_next` index by one.
    - It checks if `i_next` is greater than or equal to the size of the `ubatches` vector.
    - If `i_next` is greater than or equal to the size of `ubatches`, it returns `false`.
    - Otherwise, it returns `true`.
- **Output**: A boolean value indicating whether there are more ubatches to process.
- **See also**: [`llama_kv_cache_unified_state`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_state::apply<!-- {{#callable:llama_kv_cache_unified_state::apply}} -->
The `apply` function in the `llama_kv_cache_unified_state` class updates the key-value cache by either applying a KV cache update or processing a specific micro-batch.
- **Inputs**: None
- **Control Flow**:
    - The function asserts that the current status is `LLAMA_MEMORY_STATUS_SUCCESS`.
    - It checks if the `ubatches` vector is empty, indicating a KV cache update, and calls `kv->update` with the context, shift flag, and defragmentation info.
    - If `ubatches` is not empty, it applies the current micro-batch using `kv->apply_ubatch` with the current head and micro-batch.
    - It updates `n_kv` with the current number of key-value pairs and sets `head` to the current head index.
    - The function returns `true` to indicate successful application.
- **Output**: The function returns a boolean value `true` indicating the successful application of the update or micro-batch.
- **See also**: [`llama_kv_cache_unified_state`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_state::out\_ids<!-- {{#callable:llama_kv_cache_unified_state::out_ids}} -->
The `out_ids` function returns a reference to the `out_ids` vector from the `sbatch` member of the `llama_kv_cache_unified_state` class, ensuring the state is successful before doing so.
- **Inputs**: None
- **Control Flow**:
    - The function asserts that the `status` member variable is equal to `LLAMA_MEMORY_STATUS_SUCCESS` to ensure the state is valid.
    - It then returns the `out_ids` vector from the `sbatch` member of the class.
- **Output**: A reference to a `std::vector<int64_t>` containing output IDs.
- **See also**: [`llama_kv_cache_unified_state`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_state::get\_status<!-- {{#callable:llama_kv_cache_unified_state::get_status}} -->
The `get_status` function returns the current memory status of the `llama_kv_cache_unified_state` object.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the `status` member variable of the `llama_kv_cache_unified_state` class.
- **Output**: The function returns a `llama_memory_status` value, which indicates the current status of the memory state.
- **See also**: [`llama_kv_cache_unified_state`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_state::get\_ubatch<!-- {{#callable:llama_kv_cache_unified_state::get_ubatch}} -->
The `get_ubatch` function retrieves the current ubatch from the `ubatches` vector based on the `i_next` index, ensuring the status is successful before doing so.
- **Inputs**: None
- **Control Flow**:
    - The function asserts that the `status` is `LLAMA_MEMORY_STATUS_SUCCESS` to ensure the operation is valid.
    - It returns the ubatch at the index `i_next` from the `ubatches` vector.
- **Output**: The function returns a constant reference to a `llama_ubatch` object from the `ubatches` vector.
- **See also**: [`llama_kv_cache_unified_state`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_state::get\_n\_kv<!-- {{#callable:llama_kv_cache_unified_state::get_n_kv}} -->
The `get_n_kv` function returns the current number of key-value pairs in the llama_kv_cache_unified_state object.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the member variable `n_kv`.
- **Output**: The function returns a `uint32_t` representing the number of key-value pairs.
- **See also**: [`llama_kv_cache_unified_state`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_state::get\_k<!-- {{#callable:llama_kv_cache_unified_state::get_k}} -->
The `get_k` function retrieves a 3D view of the 'k' tensor from the `llama_kv_cache_unified` object for a specified layer and number of key-value pairs.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` object, which is used to manage memory and operations for the tensor.
    - `il`: An integer representing the layer index for which the 'k' tensor is to be retrieved.
- **Control Flow**:
    - The function calls `kv->get_k` with the provided `ctx`, `il`, and `n_kv` to retrieve the 'k' tensor view.
    - The `kv` object is an instance of `llama_kv_cache_unified`, which manages the key-value cache.
    - The `n_kv` is a private member of the `llama_kv_cache_unified_state` class, representing the number of key-value pairs.
- **Output**: A pointer to a `ggml_tensor` object representing a 3D view of the 'k' tensor for the specified layer.
- **See also**: [`llama_kv_cache_unified_state`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_state::get\_v<!-- {{#callable:llama_kv_cache_unified_state::get_v}} -->
The `get_v` function retrieves a view of the 'V' tensor from the key-value cache for a specified layer index.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` object, which provides the context for tensor operations.
    - `il`: An integer representing the layer index for which the 'V' tensor view is requested.
- **Control Flow**:
    - The function calls the `get_v` method of the `kv` object, passing the context, layer index, and the number of key-value pairs (`n_kv`).
- **Output**: A pointer to a `ggml_tensor` object representing the view of the 'V' tensor for the specified layer.
- **See also**: [`llama_kv_cache_unified_state`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_state::cpy\_k<!-- {{#callable:llama_kv_cache_unified_state::cpy_k}} -->
The `cpy_k` function copies the current key tensor `k_cur` into a specific location in the key-value cache using the provided context and index.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` object, which provides the context for the operation.
    - `k_cur`: A pointer to a `ggml_tensor` object representing the current key tensor to be copied.
    - `il`: An integer representing the index or layer location in the cache where the key tensor should be copied.
- **Control Flow**:
    - The function calls the `cpy_k` method of the `kv` object, passing the context, current key tensor, index, and the `head` member variable.
    - The `kv->cpy_k` method performs the actual copying of the key tensor into the specified location in the cache.
- **Output**: Returns a pointer to a `ggml_tensor` object, which is the result of the copy operation.
- **See also**: [`llama_kv_cache_unified_state`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_state::cpy\_v<!-- {{#callable:llama_kv_cache_unified_state::cpy_v}} -->
The `cpy_v` function copies the current tensor `v_cur` into a specific location in the key-value cache using the provided context and index.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` object, which provides the context for the operation.
    - `v_cur`: A pointer to a `ggml_tensor` object representing the current tensor to be copied.
    - `il`: An integer representing the index or layer identifier where the tensor should be copied.
- **Control Flow**:
    - The function calls the `cpy_v` method of the `kv` object, passing the context, current tensor, index, and the `head` member variable.
    - The `head` member variable is used to determine the specific location in the cache where the tensor should be copied.
- **Output**: Returns a pointer to a `ggml_tensor` object, which is the result of the copy operation.
- **See also**: [`llama_kv_cache_unified_state`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_state::set\_input\_k\_shift<!-- {{#callable:llama_kv_cache_unified_state::set_input_k_shift}} -->
The `set_input_k_shift` function sets the input K-shift values in a given tensor by delegating the task to the `kv` member of the `llama_kv_cache_unified_state` class.
- **Inputs**:
    - `dst`: A pointer to a `ggml_tensor` where the K-shift values will be set.
- **Control Flow**:
    - The function calls the `set_input_k_shift` method on the `kv` member, passing the `dst` tensor as an argument.
- **Output**: The function does not return any value; it modifies the `dst` tensor in place.
- **See also**: [`llama_kv_cache_unified_state`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_state::set\_input\_kq\_mask<!-- {{#callable:llama_kv_cache_unified_state::set_input_kq_mask}} -->
The `set_input_kq_mask` function sets a mask on the input tensor for key-query operations based on the current state of the cache and the provided micro-batch, considering whether causal attention is applied.
- **Inputs**:
    - `dst`: A pointer to a `ggml_tensor` where the mask will be set.
    - `ubatch`: A pointer to a `llama_ubatch` structure representing the micro-batch for which the mask is being set.
    - `causal_attn`: A boolean indicating whether causal attention should be applied.
- **Control Flow**:
    - The function retrieves the number of tokens, sequence tokens, and sequences from the `ubatch`.
    - It asserts that the destination tensor's buffer is hosted on the CPU and retrieves a pointer to its data.
    - For each sequence in the micro-batch, it iterates over the sequence tokens and the cache cells to determine if a mask should be applied.
    - The mask is set based on whether the cache cell is empty, belongs to a different sequence, or represents a future token if causal attention is enabled.
    - If the `use_alibi` parameter is set, it adjusts the mask value based on the position difference.
    - Finally, it applies a mask to any padded tokens in the destination tensor.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place to apply the mask.
- **See also**: [`llama_kv_cache_unified_state`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_state::set\_input\_pos\_bucket<!-- {{#callable:llama_kv_cache_unified_state::set_input_pos_bucket}} -->
The `set_input_pos_bucket` function sets the input position bucket for a given destination tensor using the provided update batch.
- **Inputs**:
    - `dst`: A pointer to a `ggml_tensor` that represents the destination tensor where the position bucket will be set.
    - `ubatch`: A pointer to a `llama_ubatch` structure that contains the update batch information used to set the position bucket.
- **Control Flow**:
    - The function calls the `set_input_pos_bucket` method of the `kv` member of the `llama_kv_cache_unified_state` class, passing the `dst` and `ubatch` as arguments.
- **Output**: The function does not return any value; it modifies the `dst` tensor in place.
- **See also**: [`llama_kv_cache_unified_state`](llama-kv-cache-unified.h.driver.md#llama_kv_cache_unified_state)  (Data Structure)



