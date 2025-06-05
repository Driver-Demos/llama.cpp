# Purpose
The provided C++ source code defines a class [`llama_kv_cache_unified_iswa`](#llama_kv_cache_unified_iswallama_kv_cache_unified_iswa) and its associated state class [`llama_kv_cache_unified_iswa_state`](#llama_kv_cache_unified_iswa_statellama_kv_cache_unified_iswa_state). This code is part of a larger system that manages key-value (KV) caches, specifically designed to handle both standard and sliding window attention (SWA) mechanisms in a unified manner. The primary purpose of this file is to implement a caching mechanism that supports efficient sequence operations, such as adding, removing, copying, and keeping sequences, while maintaining compatibility with both base and SWA caches. The class constructor initializes these caches based on parameters derived from a `llama_model` object, and it provides methods to manipulate and query the cache state.

The [`llama_kv_cache_unified_iswa`](#llama_kv_cache_unified_iswallama_kv_cache_unified_iswa) class is a specialized component that extends the functionality of a base KV cache by incorporating SWA capabilities. It includes methods for initializing batch processing, updating cache states, and managing sequence positions. The class also logs information about cache creation and size, ensuring transparency in its operations. The associated state class, [`llama_kv_cache_unified_iswa_state`](#llama_kv_cache_unified_iswa_statellama_kv_cache_unified_iswa_state), manages the state of the cache during operations, ensuring that both base and SWA caches are synchronized and correctly updated. This file is likely part of a larger library or application that deals with machine learning models, particularly those that require efficient memory management for sequence data processing.
# Imports and Dependencies

---
- `llama-kv-cache-unified-iswa.h`
- `llama-impl.h`
- `llama-batch.h`
- `llama-model.h`
- `algorithm`
- `cassert`


# Data Structures

---
### llama\_kv\_cache\_unified\_iswa<!-- {{#data_structure:llama_kv_cache_unified_iswa}} -->
- **Description**: [See definition](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa)
- **Member Functions**:
    - [`llama_kv_cache_unified_iswa::~llama_kv_cache_unified_iswa`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswallama_kv_cache_unified_iswa)
    - [`llama_kv_cache_unified_iswa::llama_kv_cache_unified_iswa`](#llama_kv_cache_unified_iswallama_kv_cache_unified_iswa)
    - [`llama_kv_cache_unified_iswa::clear`](#llama_kv_cache_unified_iswaclear)
    - [`llama_kv_cache_unified_iswa::seq_rm`](#llama_kv_cache_unified_iswaseq_rm)
    - [`llama_kv_cache_unified_iswa::seq_cp`](#llama_kv_cache_unified_iswaseq_cp)
    - [`llama_kv_cache_unified_iswa::seq_keep`](#llama_kv_cache_unified_iswaseq_keep)
    - [`llama_kv_cache_unified_iswa::seq_add`](#llama_kv_cache_unified_iswaseq_add)
    - [`llama_kv_cache_unified_iswa::seq_div`](#llama_kv_cache_unified_iswaseq_div)
    - [`llama_kv_cache_unified_iswa::seq_pos_min`](#llama_kv_cache_unified_iswaseq_pos_min)
    - [`llama_kv_cache_unified_iswa::seq_pos_max`](#llama_kv_cache_unified_iswaseq_pos_max)
    - [`llama_kv_cache_unified_iswa::init_batch`](#llama_kv_cache_unified_iswainit_batch)
    - [`llama_kv_cache_unified_iswa::init_full`](#llama_kv_cache_unified_iswainit_full)
    - [`llama_kv_cache_unified_iswa::init_update`](#llama_kv_cache_unified_iswainit_update)
    - [`llama_kv_cache_unified_iswa::get_can_shift`](#llama_kv_cache_unified_iswaget_can_shift)
    - [`llama_kv_cache_unified_iswa::state_write`](#llama_kv_cache_unified_iswastate_write)
    - [`llama_kv_cache_unified_iswa::state_read`](#llama_kv_cache_unified_iswastate_read)
    - [`llama_kv_cache_unified_iswa::get_base`](#llama_kv_cache_unified_iswaget_base)
    - [`llama_kv_cache_unified_iswa::get_swa`](#llama_kv_cache_unified_iswaget_swa)
- **Inherits From**:
    - `llama_memory_i`

**Methods**

---
#### llama\_kv\_cache\_unified\_iswa::llama\_kv\_cache\_unified\_iswa<!-- {{#callable:llama_kv_cache_unified_iswa::llama_kv_cache_unified_iswa}} -->
The `llama_kv_cache_unified_iswa` constructor initializes a key-value cache system with separate configurations for standard and SWA (Stochastic Weight Averaging) caches based on the provided model and parameters.
- **Inputs**:
    - `model`: A reference to a `llama_model` object, which contains hyperparameters and other model-specific configurations.
    - `type_k`: A `ggml_type` indicating the data type for the keys in the cache.
    - `type_v`: A `ggml_type` indicating the data type for the values in the cache.
    - `v_trans`: A boolean flag indicating whether the values should be transposed.
    - `offload`: A boolean flag indicating whether the cache should be offloaded to a different storage or processing unit.
    - `swa_full`: A boolean flag indicating whether the SWA cache should use the full size of the base cache.
    - `kv_size`: A `uint32_t` specifying the size of the key-value cache.
    - `n_seq_max`: A `uint32_t` specifying the maximum number of sequences.
    - `n_ubatch`: A `uint32_t` specifying the number of micro-batches.
    - `n_pad`: A `uint32_t` specifying the padding size for alignment.
- **Control Flow**:
    - Initialize `filter_base` and `filter_swa` lambda functions to determine if a layer is part of the SWA or not based on the model's hyperparameters.
    - Calculate `size_base` as the provided `kv_size`.
    - Calculate `size_swa` as the minimum of `size_base` and a padded size based on `n_swa`, `n_seq_max`, `n_ubatch`, and `n_pad`.
    - If `swa_full` is true, set `size_swa` to `size_base` and log a warning about using full-size SWA cache.
    - Log the creation of the non-SWA KV cache with `size_base`.
    - Create a `kv_base` object using `llama_kv_cache_unified` with the non-SWA filter and other parameters.
    - Log the creation of the SWA KV cache with `size_swa`.
    - Create a `kv_swa` object using `llama_kv_cache_unified` with the SWA filter and other parameters.
- **Output**: The function does not return a value; it initializes the `llama_kv_cache_unified_iswa` object with two key-value caches, `kv_base` and `kv_swa`, configured according to the input parameters.
- **See also**: [`llama_kv_cache_unified_iswa`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa::clear<!-- {{#callable:llama_kv_cache_unified_iswa::clear}} -->
The `clear` function resets both the base and SWA key-value caches in the `llama_kv_cache_unified_iswa` class.
- **Inputs**: None
- **Control Flow**:
    - The function calls the `clear` method on the `kv_base` object, which is a unique pointer to a `llama_kv_cache_unified` instance.
    - The function then calls the `clear` method on the `kv_swa` object, which is another unique pointer to a `llama_kv_cache_unified` instance.
- **Output**: The function does not return any value.
- **See also**: [`llama_kv_cache_unified_iswa`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa::seq\_rm<!-- {{#callable:llama_kv_cache_unified_iswa::seq_rm}} -->
The `seq_rm` function attempts to remove a sequence of positions from both the base and SWA key-value caches and returns a boolean indicating the success of these operations.
- **Inputs**:
    - `seq_id`: An identifier for the sequence to be removed.
    - `p0`: The starting position of the sequence to be removed.
    - `p1`: The ending position of the sequence to be removed.
- **Control Flow**:
    - Initialize a boolean variable `res` to `true`.
    - Call the `seq_rm` method on `kv_base` with the provided `seq_id`, `p0`, and `p1`, and perform a bitwise AND operation with `res`.
    - Call the `seq_rm` method on `kv_swa` with the same parameters, and perform a bitwise AND operation with `res`.
    - Return the value of `res`.
- **Output**: A boolean value indicating whether the sequence removal was successful in both the base and SWA caches.
- **See also**: [`llama_kv_cache_unified_iswa`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa::seq\_cp<!-- {{#callable:llama_kv_cache_unified_iswa::seq_cp}} -->
The `seq_cp` function copies a sequence of data from a source sequence ID to a destination sequence ID within both the base and SWA key-value caches.
- **Inputs**:
    - `seq_id_src`: The source sequence ID from which data is to be copied.
    - `seq_id_dst`: The destination sequence ID to which data is to be copied.
    - `p0`: The starting position in the sequence from which to begin copying.
    - `p1`: The ending position in the sequence up to which data is to be copied.
- **Control Flow**:
    - The function calls the `seq_cp` method on the `kv_base` object, passing all input parameters to copy the sequence data from the source to the destination within the base cache.
    - The function then calls the `seq_cp` method on the `kv_swa` object, again passing all input parameters to perform the same copy operation within the SWA cache.
- **Output**: This function does not return any value; it performs operations on the internal state of the `kv_base` and `kv_swa` objects.
- **See also**: [`llama_kv_cache_unified_iswa`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa::seq\_keep<!-- {{#callable:llama_kv_cache_unified_iswa::seq_keep}} -->
The `seq_keep` function ensures that a sequence identified by `seq_id` is retained in both the base and SWA key-value caches.
- **Inputs**:
    - `seq_id`: An identifier for the sequence to be retained in the caches.
- **Control Flow**:
    - The function calls the `seq_keep` method on the `kv_base` object, passing the `seq_id` as an argument.
    - It then calls the `seq_keep` method on the `kv_swa` object, also passing the `seq_id` as an argument.
- **Output**: The function does not return any value; it performs operations on the internal state of the `kv_base` and `kv_swa` objects.
- **See also**: [`llama_kv_cache_unified_iswa`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa::seq\_add<!-- {{#callable:llama_kv_cache_unified_iswa::seq_add}} -->
The `seq_add` function adds a sequence to both the base and SWA key-value caches with specified positions and a shift value.
- **Inputs**:
    - `seq_id`: An identifier for the sequence to be added.
    - `p0`: The starting position of the sequence.
    - `p1`: The ending position of the sequence.
    - `shift`: The shift value to be applied to the sequence positions.
- **Control Flow**:
    - The function calls the `seq_add` method on the `kv_base` object, passing `seq_id`, `p0`, `p1`, and `shift` as arguments.
    - The function then calls the `seq_add` method on the `kv_swa` object with the same arguments.
- **Output**: This function does not return any value; it performs operations on the internal state of the `kv_base` and `kv_swa` objects.
- **See also**: [`llama_kv_cache_unified_iswa`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa::seq\_div<!-- {{#callable:llama_kv_cache_unified_iswa::seq_div}} -->
The `seq_div` function divides a sequence range by a given integer across both base and SWA key-value caches.
- **Inputs**:
    - `seq_id`: The identifier for the sequence to be divided.
    - `p0`: The starting position of the sequence range to be divided.
    - `p1`: The ending position of the sequence range to be divided.
    - `d`: The integer divisor used to divide the sequence range.
- **Control Flow**:
    - The function calls the `seq_div` method on the `kv_base` object with the provided sequence ID, start position, end position, and divisor.
    - It then calls the `seq_div` method on the `kv_swa` object with the same parameters.
- **Output**: The function does not return any value.
- **See also**: [`llama_kv_cache_unified_iswa`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa::seq\_pos\_min<!-- {{#callable:llama_kv_cache_unified_iswa::seq_pos_min}} -->
The `seq_pos_min` function retrieves the minimum sequence position for a given sequence ID from the SWA cache.
- **Inputs**:
    - `seq_id`: A sequence identifier of type `llama_seq_id` for which the minimum position is to be retrieved.
- **Control Flow**:
    - The function calls the `seq_pos_min` method on the `kv_swa` object, which is a pointer to a `llama_kv_cache_unified` instance.
    - It passes the `seq_id` argument to this method call.
- **Output**: The function returns a `llama_pos` value representing the minimum position of the specified sequence ID in the SWA cache.
- **See also**: [`llama_kv_cache_unified_iswa`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa::seq\_pos\_max<!-- {{#callable:llama_kv_cache_unified_iswa::seq_pos_max}} -->
The `seq_pos_max` function retrieves the maximum sequence position for a given sequence ID from the SWA cache.
- **Inputs**:
    - `seq_id`: An identifier for the sequence whose maximum position is to be retrieved.
- **Control Flow**:
    - The function calls the `seq_pos_max` method on the `kv_swa` object, passing the `seq_id` as an argument.
    - The result from the `kv_swa->seq_pos_max(seq_id)` call is returned.
- **Output**: The function returns a `llama_pos` value representing the maximum position of the specified sequence in the SWA cache.
- **See also**: [`llama_kv_cache_unified_iswa`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa::init\_batch<!-- {{#callable:llama_kv_cache_unified_iswa::init_batch}} -->
The `init_batch` function initializes a batch of unified key-value caches for a given input batch, splitting it into smaller sub-batches and preparing them for processing.
- **Inputs**:
    - `batch`: A `llama_batch` object representing the input data to be processed.
    - `n_ubatch`: A `uint32_t` value specifying the number of tokens per sub-batch.
    - `embd_pooled`: A `bool` indicating whether embedding pooling is used, though it is unused in this function.
    - `logits_all`: A `bool` indicating whether to process all logits.
- **Control Flow**:
    - The function begins by creating a [`llama_sbatch`](llama-batch.h.driver.md#llama_sbatchllama_sbatch) object from the input `batch`, using the number of embeddings from `hparams` and the `logits_all` flag.
    - An empty vector `ubatches` is initialized to store the sub-batches.
    - A loop runs while there are tokens left in `sbatch`, splitting it into sub-batches of size `n_ubatch` using `split_simple` and adding each sub-batch to `ubatches`.
    - The function prepares the `ubatches` for both `kv_base` and `kv_swa` caches, storing the results in `heads_base` and `heads_swa` respectively.
    - If either `heads_base` or `heads_swa` is empty, indicating a preparation failure, the function returns a new `llama_kv_cache_unified_iswa_state` with a failed status.
    - An assertion checks that `heads_base` and `heads_swa` have the same size.
    - Finally, the function returns a new `llama_kv_cache_unified_iswa_state` initialized with the current object, `sbatch`, `heads_base`, `heads_swa`, and `ubatches`.
- **Output**: A `llama_memory_state_ptr` pointing to a new `llama_kv_cache_unified_iswa_state` object, which represents the initialized state of the batch processing.
- **Functions called**:
    - [`llama_sbatch::llama_sbatch`](llama-batch.h.driver.md#llama_sbatchllama_sbatch)
- **See also**: [`llama_kv_cache_unified_iswa`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa::init\_full<!-- {{#callable:llama_kv_cache_unified_iswa::init_full}} -->
The `init_full` function initializes a full memory state for the `llama_kv_cache_unified_iswa` class by creating a new `llama_kv_cache_unified_iswa_state` object.
- **Inputs**: None
- **Control Flow**:
    - The function calls `std::make_unique` to create a new `llama_kv_cache_unified_iswa_state` object, passing `this` as an argument to the constructor.
    - The constructor of `llama_kv_cache_unified_iswa_state` initializes the state for both the base and SWA caches by calling their respective `init_full` methods.
- **Output**: A `llama_memory_state_ptr` which is a smart pointer to a newly created `llama_kv_cache_unified_iswa_state` object.
- **See also**: [`llama_kv_cache_unified_iswa`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa::init\_update<!-- {{#callable:llama_kv_cache_unified_iswa::init_update}} -->
The `init_update` function initializes and returns a new `llama_kv_cache_unified_iswa_state` object using the provided context and optimization flag.
- **Inputs**:
    - `lctx`: A pointer to a `llama_context` object, which provides the context for the update operation.
    - `optimize`: A boolean flag indicating whether optimization should be applied during the update.
- **Control Flow**:
    - The function calls `std::make_unique` to create a new `llama_kv_cache_unified_iswa_state` object.
    - The constructor of `llama_kv_cache_unified_iswa_state` is invoked with `this`, `lctx`, and `optimize` as arguments.
    - The newly created `llama_kv_cache_unified_iswa_state` object is returned.
- **Output**: A `llama_memory_state_ptr`, which is a smart pointer to the newly created `llama_kv_cache_unified_iswa_state` object.
- **See also**: [`llama_kv_cache_unified_iswa`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa::get\_can\_shift<!-- {{#callable:llama_kv_cache_unified_iswa::get_can_shift}} -->
The `get_can_shift` function checks if the sizes of the base and SWA key-value caches are equal.
- **Inputs**: None
- **Control Flow**:
    - The function retrieves the size of the `kv_base` cache using `kv_base->get_size()`.
    - It retrieves the size of the `kv_swa` cache using `kv_swa->get_size()`.
    - It compares the two sizes for equality and returns the result.
- **Output**: A boolean value indicating whether the sizes of `kv_base` and `kv_swa` are equal.
- **See also**: [`llama_kv_cache_unified_iswa`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa::state\_write<!-- {{#callable:llama_kv_cache_unified_iswa::state_write}} -->
The `state_write` function writes the state of both the base and SWA key-value caches to an output stream for a given sequence ID.
- **Inputs**:
    - `io`: A reference to an object implementing the `llama_io_write_i` interface, which handles the output stream for writing the state.
    - `seq_id`: An identifier for the sequence whose state is to be written; defaults to -1 if not specified.
- **Control Flow**:
    - The function calls the `state_write` method on the `kv_base` object, passing the `io` and `seq_id` parameters.
    - It then calls the `state_write` method on the `kv_swa` object, also passing the `io` and `seq_id` parameters.
- **Output**: The function does not return a value; it performs its operations through side effects on the `io` object.
- **See also**: [`llama_kv_cache_unified_iswa`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa::state\_read<!-- {{#callable:llama_kv_cache_unified_iswa::state_read}} -->
The `state_read` function reads the state of both the base and SWA key-value caches using the provided I/O interface and sequence ID.
- **Inputs**:
    - `io`: A reference to an object implementing the `llama_io_read_i` interface, used for reading state data.
    - `seq_id`: A `llama_seq_id` representing the sequence identifier for which the state is to be read.
- **Control Flow**:
    - The function calls the `state_read` method on the `kv_base` object, passing the `io` and `seq_id` parameters.
    - The function then calls the `state_read` method on the `kv_swa` object, also passing the `io` and `seq_id` parameters.
- **Output**: The function does not return any value; it performs state reading operations on the internal key-value caches.
- **See also**: [`llama_kv_cache_unified_iswa`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa::get\_base<!-- {{#callable:llama_kv_cache_unified_iswa::get_base}} -->
The `get_base` function returns a pointer to the base key-value cache of the `llama_kv_cache_unified_iswa` class.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `kv_base` member variable, which is a `std::unique_ptr` to a `llama_kv_cache_unified` object.
    - It calls the `get()` method on `kv_base` to retrieve the raw pointer to the `llama_kv_cache_unified` object.
    - The function returns this raw pointer.
- **Output**: A pointer to a `llama_kv_cache_unified` object representing the base key-value cache.
- **See also**: [`llama_kv_cache_unified_iswa`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa::get\_swa<!-- {{#callable:llama_kv_cache_unified_iswa::get_swa}} -->
The `get_swa` function returns a pointer to the SWA (Stochastic Weight Averaging) key-value cache associated with the `llama_kv_cache_unified_iswa` object.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `kv_swa` member variable, which is a unique pointer to a `llama_kv_cache_unified` object.
    - It calls the `get()` method on the `kv_swa` unique pointer to retrieve the raw pointer to the `llama_kv_cache_unified` object.
    - The function returns this raw pointer.
- **Output**: A raw pointer to a `llama_kv_cache_unified` object representing the SWA cache.
- **See also**: [`llama_kv_cache_unified_iswa`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa)  (Data Structure)



---
### llama\_kv\_cache\_unified\_iswa\_state<!-- {{#data_structure:llama_kv_cache_unified_iswa_state}} -->
- **Description**: [See definition](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa_state)
- **Member Functions**:
    - [`llama_kv_cache_unified_iswa_state::llama_kv_cache_unified_iswa_state`](#llama_kv_cache_unified_iswa_statellama_kv_cache_unified_iswa_state)
    - [`llama_kv_cache_unified_iswa_state::llama_kv_cache_unified_iswa_state`](#llama_kv_cache_unified_iswa_statellama_kv_cache_unified_iswa_state)
    - [`llama_kv_cache_unified_iswa_state::llama_kv_cache_unified_iswa_state`](#llama_kv_cache_unified_iswa_statellama_kv_cache_unified_iswa_state)
    - [`llama_kv_cache_unified_iswa_state::llama_kv_cache_unified_iswa_state`](#llama_kv_cache_unified_iswa_statellama_kv_cache_unified_iswa_state)
    - [`llama_kv_cache_unified_iswa_state::~llama_kv_cache_unified_iswa_state`](#llama_kv_cache_unified_iswa_statellama_kv_cache_unified_iswa_state)
    - [`llama_kv_cache_unified_iswa_state::next`](#llama_kv_cache_unified_iswa_statenext)
    - [`llama_kv_cache_unified_iswa_state::apply`](#llama_kv_cache_unified_iswa_stateapply)
    - [`llama_kv_cache_unified_iswa_state::out_ids`](#llama_kv_cache_unified_iswa_stateout_ids)
    - [`llama_kv_cache_unified_iswa_state::get_status`](#llama_kv_cache_unified_iswa_stateget_status)
    - [`llama_kv_cache_unified_iswa_state::get_ubatch`](#llama_kv_cache_unified_iswa_stateget_ubatch)
    - [`llama_kv_cache_unified_iswa_state::get_base`](#llama_kv_cache_unified_iswa_stateget_base)
    - [`llama_kv_cache_unified_iswa_state::get_swa`](#llama_kv_cache_unified_iswa_stateget_swa)
- **Inherits From**:
    - `llama_memory_state_i`

**Methods**

---
#### llama\_kv\_cache\_unified\_iswa\_state::llama\_kv\_cache\_unified\_iswa\_state<!-- {{#callable:llama_kv_cache_unified_iswa_state::llama_kv_cache_unified_iswa_state}} -->
The constructor `llama_kv_cache_unified_iswa_state::llama_kv_cache_unified_iswa_state` initializes an instance of the `llama_kv_cache_unified_iswa_state` class with a given memory status.
- **Inputs**:
    - `status`: A `llama_memory_status` value representing the initial memory status of the state.
- **Control Flow**:
    - The constructor initializes the `status` member variable with the provided `status` argument.
- **Output**: An instance of the `llama_kv_cache_unified_iswa_state` class with its `status` member initialized.
- **See also**: [`llama_kv_cache_unified_iswa_state`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa\_state::\~llama\_kv\_cache\_unified\_iswa\_state<!-- {{#callable:llama_kv_cache_unified_iswa_state::~llama_kv_cache_unified_iswa_state}} -->
The destructor `~llama_kv_cache_unified_iswa_state` is a default destructor for the `llama_kv_cache_unified_iswa_state` class, which performs no specific actions upon object destruction.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as `default`, meaning it relies on the compiler-generated default behavior for destructors.
    - No custom cleanup or resource deallocation is performed in this destructor.
- **Output**: There is no output from this destructor as it performs no operations.
- **See also**: [`llama_kv_cache_unified_iswa_state`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa\_state::next<!-- {{#callable:llama_kv_cache_unified_iswa_state::next}} -->
The `next` function advances the state of the `llama_kv_cache_unified_iswa_state` object to the next unprocessed batch and returns whether more batches are available.
- **Inputs**: None
- **Control Flow**:
    - The function begins by asserting that the `status` is `LLAMA_MEMORY_STATUS_SUCCESS`.
    - It calls the `next` method on both `state_base` and `state_swa` objects to advance their states.
    - The index `i_next` is incremented to point to the next batch in `ubatches`.
    - If `i_next` is greater than or equal to the size of `ubatches`, the function returns `false`, indicating no more batches are available.
    - Otherwise, it returns `true`, indicating there are more batches to process.
- **Output**: A boolean value indicating whether there are more batches to process (`true`) or not (`false`).
- **See also**: [`llama_kv_cache_unified_iswa_state`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa\_state::apply<!-- {{#callable:llama_kv_cache_unified_iswa_state::apply}} -->
The `apply` method in `llama_kv_cache_unified_iswa_state` class applies the state changes to both base and SWA states and returns a boolean indicating success.
- **Inputs**: None
- **Control Flow**:
    - The function begins by asserting that the `status` is `LLAMA_MEMORY_STATUS_SUCCESS`.
    - A boolean variable `res` is initialized to `true`.
    - The `apply` method of `state_base` is called, and its result is bitwise ANDed with `res`.
    - The `apply` method of `state_swa` is called, and its result is also bitwise ANDed with `res`.
    - The final value of `res` is returned.
- **Output**: A boolean value indicating whether both `state_base` and `state_swa` successfully applied their changes.
- **See also**: [`llama_kv_cache_unified_iswa_state`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa\_state::out\_ids<!-- {{#callable:llama_kv_cache_unified_iswa_state::out_ids}} -->
The `out_ids` function returns a reference to the `out_ids` vector from the `sbatch` object within the `llama_kv_cache_unified_iswa_state` class.
- **Inputs**: None
- **Control Flow**:
    - The function asserts that the `status` member variable is equal to `LLAMA_MEMORY_STATUS_SUCCESS` to ensure the state is valid before proceeding.
    - It then returns a reference to the `out_ids` vector from the `sbatch` object.
- **Output**: A reference to a `std::vector<int64_t>` containing output IDs.
- **See also**: [`llama_kv_cache_unified_iswa_state`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa\_state::get\_status<!-- {{#callable:llama_kv_cache_unified_iswa_state::get_status}} -->
The `get_status` method returns the current memory status of the `llama_kv_cache_unified_iswa_state` object.
- **Inputs**: None
- **Control Flow**:
    - The method directly returns the value of the `status` member variable of the `llama_kv_cache_unified_iswa_state` class.
- **Output**: The method returns a `llama_memory_status` which indicates the current status of the memory state.
- **See also**: [`llama_kv_cache_unified_iswa_state`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa\_state::get\_ubatch<!-- {{#callable:llama_kv_cache_unified_iswa_state::get_ubatch}} -->
The `get_ubatch` function returns the current `llama_ubatch` from the `ubatches` vector based on the `i_next` index, ensuring the state is successful before doing so.
- **Inputs**: None
- **Control Flow**:
    - The function asserts that the `status` is `LLAMA_MEMORY_STATUS_SUCCESS` to ensure the state is valid before proceeding.
    - It returns the `llama_ubatch` at the index `i_next` from the `ubatches` vector.
- **Output**: The function returns a constant reference to a `llama_ubatch` object from the `ubatches` vector.
- **See also**: [`llama_kv_cache_unified_iswa_state`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa\_state::get\_base<!-- {{#callable:llama_kv_cache_unified_iswa_state::get_base}} -->
The `get_base` function returns a pointer to the base state of the llama key-value cache unified state, ensuring the current status is successful.
- **Inputs**: None
- **Control Flow**:
    - The function begins by asserting that the `status` member variable is equal to `LLAMA_MEMORY_STATUS_SUCCESS`, ensuring that the operation is only performed if the state is in a successful status.
    - It then returns a pointer to the base state by casting the `state_base` member, which is a smart pointer, to a `const llama_kv_cache_unified_state *`.
- **Output**: A pointer to a `const llama_kv_cache_unified_state` object representing the base state of the cache.
- **See also**: [`llama_kv_cache_unified_iswa_state`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa_state)  (Data Structure)


---
#### llama\_kv\_cache\_unified\_iswa\_state::get\_swa<!-- {{#callable:llama_kv_cache_unified_iswa_state::get_swa}} -->
The `get_swa` function returns a pointer to the SWA (Stochastic Weight Averaging) state of the llama key-value cache unified state, ensuring the current status is successful.
- **Inputs**: None
- **Control Flow**:
    - The function asserts that the current status is `LLAMA_MEMORY_STATUS_SUCCESS` to ensure the state is valid before proceeding.
    - It returns a pointer to the SWA state by casting the `state_swa` member to a `const llama_kv_cache_unified_state *`.
- **Output**: A pointer to a `const llama_kv_cache_unified_state` representing the SWA state.
- **See also**: [`llama_kv_cache_unified_iswa_state`](llama-kv-cache-unified-iswa.h.driver.md#llama_kv_cache_unified_iswa_state)  (Data Structure)



