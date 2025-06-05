# Purpose
The provided C++ code defines two classes, `llama_kv_cache_recurrent` and `llama_kv_cache_recurrent_state`, which are part of a system for managing key-value (KV) caches in a recurrent manner, likely for use in machine learning or graph computation contexts. The `llama_kv_cache_recurrent` class inherits from `llama_memory_i`, indicating that it implements an interface for memory management. This class is responsible for initializing, updating, and managing the state of a KV cache, which includes operations such as clearing the cache, managing sequences, and reading/writing state data. The class also includes methods for preparing batches and finding slots for them, suggesting its role in handling dynamic data inputs efficiently. The `kv_cell` structure within the class represents individual cells in the cache, each associated with sequence identifiers, and provides methods to check and manipulate these associations.

The `llama_kv_cache_recurrent_state` class, inheriting from `llama_memory_state_i`, appears to manage the state of the KV cache for specific operations or sequences. It provides functionality to iterate over and apply states, manage output identifiers, and access specific tensors associated with the cache. This class is designed to handle both full-cache states and batch-specific states, indicating its flexibility in managing different operational contexts. The presence of methods for reading and writing state data suggests that these classes are part of a larger system that requires persistent state management, possibly for tasks involving recurrent neural networks or other iterative computational models. Overall, the code provides a specialized and modular approach to managing memory and state in a computational graph or machine learning framework.
# Imports and Dependencies

---
- `llama-batch.h`
- `llama-graph.h`
- `llama-memory.h`
- `set`
- `vector`


# Data Structures

---
### llama\_kv\_cache\_recurrent<!-- {{#data_structure:llama_kv_cache_recurrent}} -->
- **Type**: `class`
- **Members**:
    - `head`: The location where the batch will be placed in the cache.
    - `size`: Total number of cells, shared across all sequences.
    - `used`: Number of used cells, indicating at least one sequence ID is present.
    - `n`: Computed before each graph build, possibly indicating the number of sequences or operations.
    - `cells`: A vector of kv_cell structures representing the key-value cache cells.
    - `k_l`: A vector of ggml_tensor pointers for keys, organized per layer.
    - `v_l`: A vector of ggml_tensor pointers for values, organized per layer.
    - `hparams`: A reference to llama_hparams, holding hyperparameters for the model.
    - `n_seq_max`: Maximum number of sequences allowed, defaulting to 1.
    - `ctxs`: A vector of ggml_context_ptr, representing contexts for different buffer types.
    - `bufs`: A vector of ggml_backend_buffer_ptr, representing backend buffers for the cache.
- **Description**: The `llama_kv_cache_recurrent` class is a specialized data structure designed to manage a recurrent key-value cache for a machine learning model, specifically for handling sequences in a memory-efficient manner. It extends the `llama_memory_i` interface, providing methods to initialize, update, and manage the cache state, including operations like adding, removing, and copying sequences. The class maintains a collection of `kv_cell` structures, each representing a cache cell with positional and sequence ID information, and supports operations to find and allocate slots for new data batches. It also handles the serialization and deserialization of cache states, ensuring that the cache can be saved and restored efficiently. The class is optimized for recurrent state needs, with a focus on managing sequences across multiple layers of a model.
- **Member Functions**:
    - [`llama_kv_cache_recurrent::~llama_kv_cache_recurrent`](#llama_kv_cache_recurrentllama_kv_cache_recurrent)
    - [`llama_kv_cache_recurrent::llama_kv_cache_recurrent`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentllama_kv_cache_recurrent)
    - [`llama_kv_cache_recurrent::clear`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentclear)
    - [`llama_kv_cache_recurrent::seq_rm`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentseq_rm)
    - [`llama_kv_cache_recurrent::seq_cp`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentseq_cp)
    - [`llama_kv_cache_recurrent::seq_keep`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentseq_keep)
    - [`llama_kv_cache_recurrent::seq_add`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentseq_add)
    - [`llama_kv_cache_recurrent::seq_div`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentseq_div)
    - [`llama_kv_cache_recurrent::seq_pos_min`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentseq_pos_min)
    - [`llama_kv_cache_recurrent::seq_pos_max`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentseq_pos_max)
    - [`llama_kv_cache_recurrent::init_batch`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentinit_batch)
    - [`llama_kv_cache_recurrent::init_full`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentinit_full)
    - [`llama_kv_cache_recurrent::init_update`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentinit_update)
    - [`llama_kv_cache_recurrent::prepare`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentprepare)
    - [`llama_kv_cache_recurrent::find_slot`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentfind_slot)
    - [`llama_kv_cache_recurrent::get_can_shift`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentget_can_shift)
    - [`llama_kv_cache_recurrent::s_copy`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrents_copy)
    - [`llama_kv_cache_recurrent::s_mask`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrents_mask)
    - [`llama_kv_cache_recurrent::total_size`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrenttotal_size)
    - [`llama_kv_cache_recurrent::size_k_bytes`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentsize_k_bytes)
    - [`llama_kv_cache_recurrent::size_v_bytes`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentsize_v_bytes)
    - [`llama_kv_cache_recurrent::state_write`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentstate_write)
    - [`llama_kv_cache_recurrent::state_read`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentstate_read)
    - [`llama_kv_cache_recurrent::state_write_meta`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentstate_write_meta)
    - [`llama_kv_cache_recurrent::state_write_data`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentstate_write_data)
    - [`llama_kv_cache_recurrent::state_read_meta`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentstate_read_meta)
    - [`llama_kv_cache_recurrent::state_read_data`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrentstate_read_data)
- **Inherits From**:
    - [`llama_memory_i`](llama-memory.h.driver.md#llama_memory_i)

**Methods**

---
#### llama\_kv\_cache\_recurrent::\~llama\_kv\_cache\_recurrent<!-- {{#callable:llama_kv_cache_recurrent::~llama_kv_cache_recurrent}} -->
The destructor `~llama_kv_cache_recurrent` is a default destructor for the `llama_kv_cache_recurrent` class, which does not perform any specific cleanup operations.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as `default`, meaning it relies on the compiler-generated default behavior for destructors.
    - No explicit resource deallocation or cleanup logic is implemented in this destructor.
- **Output**: The destructor does not return any value or perform any operations; it simply allows the compiler to handle object destruction using default behavior.
- **See also**: [`llama_kv_cache_recurrent`](#llama_kv_cache_recurrent)  (Data Structure)



---
### kv\_cell<!-- {{#data_structure:llama_kv_cache_recurrent::kv_cell}} -->
- **Type**: `struct`
- **Members**:
    - `pos`: Represents the position of the kv_cell, initialized to -1.
    - `src`: An integer used to copy states, initialized to -1.
    - `tail`: An integer representing the tail, initialized to -1.
    - `seq_id`: A set of llama_seq_id that stores sequence identifiers associated with the kv_cell.
- **Description**: The `kv_cell` struct is a data structure used to represent a key-value cell in a cache system, specifically for graph computation in the llama framework. It contains positional information, state copying indicators, and a set of sequence identifiers to manage and track sequences associated with the cell. The struct provides methods to check if a sequence identifier exists, if the cell is empty, and if it matches another cell's sequence identifiers.
- **Member Functions**:
    - [`llama_kv_cache_recurrent::kv_cell::has_seq_id`](#kv_cellhas_seq_id)
    - [`llama_kv_cache_recurrent::kv_cell::is_empty`](#kv_cellis_empty)
    - [`llama_kv_cache_recurrent::kv_cell::is_same_seq`](#kv_cellis_same_seq)

**Methods**

---
#### kv\_cell::has\_seq\_id<!-- {{#callable:llama_kv_cache_recurrent::kv_cell::has_seq_id}} -->
The `has_seq_id` function checks if a given sequence ID exists within the `seq_id` set of a `kv_cell`.
- **Inputs**:
    - `id`: A constant reference to a `llama_seq_id` object that represents the sequence ID to be checked for existence in the `seq_id` set.
- **Control Flow**:
    - The function attempts to find the provided `id` in the `seq_id` set using the `find` method.
    - It compares the result of `find` with `seq_id.end()` to determine if the `id` is present in the set.
- **Output**: Returns a boolean value: `true` if the `id` is found in the `seq_id` set, otherwise `false`.
- **See also**: [`llama_kv_cache_recurrent::kv_cell`](#llama_kv_cache_recurrent::kv_cell)  (Data Structure)


---
#### kv\_cell::is\_empty<!-- {{#callable:llama_kv_cache_recurrent::kv_cell::is_empty}} -->
The `is_empty` function checks if the `seq_id` set in a `kv_cell` is empty.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `seq_id` member of the `kv_cell` structure.
    - It calls the `empty()` method on the `seq_id` set to determine if it contains any elements.
    - The result of the `empty()` method call is returned as the output of the function.
- **Output**: A boolean value indicating whether the `seq_id` set is empty (true if empty, false otherwise).
- **See also**: [`llama_kv_cache_recurrent::kv_cell`](#llama_kv_cache_recurrent::kv_cell)  (Data Structure)


---
#### kv\_cell::is\_same\_seq<!-- {{#callable:llama_kv_cache_recurrent::kv_cell::is_same_seq}} -->
The `is_same_seq` function checks if two `kv_cell` objects have identical sequence IDs.
- **Inputs**:
    - `other`: A reference to another `kv_cell` object whose sequence IDs are to be compared with the current object.
- **Control Flow**:
    - The function compares the `seq_id` set of the current `kv_cell` object with the `seq_id` set of the `other` `kv_cell` object using the equality operator.
- **Output**: A boolean value indicating whether the sequence IDs of the two `kv_cell` objects are the same.
- **See also**: [`llama_kv_cache_recurrent::kv_cell`](#llama_kv_cache_recurrent::kv_cell)  (Data Structure)



---
### llama\_kv\_cache\_recurrent\_state<!-- {{#data_structure:llama_kv_cache_recurrent_state}} -->
- **Type**: `class`
- **Members**:
    - `status`: Represents the memory status of the llama_kv_cache_recurrent_state.
    - `kv`: Pointer to a llama_kv_cache_recurrent object, representing the key-value cache.
    - `sbatch`: An instance of llama_sbatch, used for batch processing.
    - `i_next`: Index to track the next ubatch to process.
    - `ubatches`: A vector of llama_ubatch objects, representing unprocessed batches.
    - `is_full`: Boolean flag indicating if the cache is full.
- **Description**: The `llama_kv_cache_recurrent_state` class is a specialized memory state class that manages the state of a key-value cache for recurrent processing in a llama-based system. It inherits from `llama_memory_state_i` and provides functionality to handle memory status, batch processing, and cache management. The class maintains a reference to a key-value cache, tracks the current batch being processed, and provides methods to interact with the cache and retrieve information about its state. It is designed to support both full-cache and batch-based operations, with mechanisms to handle errors and manage the compute graph for the current batch.
- **Member Functions**:
    - [`llama_kv_cache_recurrent_state::llama_kv_cache_recurrent_state`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrent_statellama_kv_cache_recurrent_state)
    - [`llama_kv_cache_recurrent_state::llama_kv_cache_recurrent_state`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrent_statellama_kv_cache_recurrent_state)
    - [`llama_kv_cache_recurrent_state::llama_kv_cache_recurrent_state`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrent_statellama_kv_cache_recurrent_state)
    - [`llama_kv_cache_recurrent_state::~llama_kv_cache_recurrent_state`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrent_statellama_kv_cache_recurrent_state)
    - [`llama_kv_cache_recurrent_state::next`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrent_statenext)
    - [`llama_kv_cache_recurrent_state::apply`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrent_stateapply)
    - [`llama_kv_cache_recurrent_state::out_ids`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrent_stateout_ids)
    - [`llama_kv_cache_recurrent_state::get_status`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrent_stateget_status)
    - [`llama_kv_cache_recurrent_state::get_ubatch`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrent_stateget_ubatch)
    - [`llama_kv_cache_recurrent_state::get_n_kv`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrent_stateget_n_kv)
    - [`llama_kv_cache_recurrent_state::get_head`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrent_stateget_head)
    - [`llama_kv_cache_recurrent_state::get_size`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrent_stateget_size)
    - [`llama_kv_cache_recurrent_state::get_k_l`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrent_stateget_k_l)
    - [`llama_kv_cache_recurrent_state::get_v_l`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrent_stateget_v_l)
    - [`llama_kv_cache_recurrent_state::s_copy`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrent_states_copy)
    - [`llama_kv_cache_recurrent_state::s_mask`](llama-kv-cache-recurrent.cpp.driver.md#llama_kv_cache_recurrent_states_mask)
- **Inherits From**:
    - [`llama_memory_state_i`](llama-memory.h.driver.md#llama_memory_state_i)


