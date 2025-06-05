# Purpose
The provided C++ code defines a class `llama_kv_cache_unified` and its associated state class `llama_kv_cache_unified_state`, which are part of a system for managing key-value (KV) caches in a machine learning or data processing context. The primary purpose of this code is to handle the storage, retrieval, and manipulation of key-value pairs in a structured and efficient manner, likely for use in neural network models or similar applications. The `llama_kv_cache_unified` class inherits from `llama_memory_i`, indicating that it implements a memory interface, which includes methods for initializing, updating, and clearing memory states, as well as managing sequences within the cache.

The code provides a comprehensive API for interacting with the KV cache, including methods for preparing and applying updates, managing cache slots, and handling defragmentation. It also includes functionality for writing and reading the state of the cache, which is crucial for maintaining consistency and performance in dynamic environments. The use of structures like `defrag_info` and `kv_layer` suggests a focus on optimizing memory usage and ensuring efficient data access patterns. The `llama_kv_cache_unified_state` class further extends this functionality by managing the state of the cache during various operations, such as batch processing and updates, and provides additional methods for interacting with the cache's current state. Overall, this code is a specialized component designed to support complex data processing tasks by providing a robust and flexible KV caching mechanism.
# Imports and Dependencies

---
- `llama-batch.h`
- `llama-graph.h`
- `llama-kv-cells.h`
- `llama-memory.h`
- `unordered_map`
- `vector`


# Data Structures

---
### llama\_kv\_cache\_unified<!-- {{#data_structure:llama_kv_cache_unified}} -->
- **Type**: `class`
- **Members**:
    - `model`: A reference to the llama_model object associated with this cache.
    - `hparams`: A reference to the llama_hparams object containing hyperparameters for the model.
    - `v_trans`: A boolean indicating if the value tensor is transposed.
    - `head`: The current index from where the search for a free slot in the ring buffer of KV cells starts.
    - `n_seq_max`: The maximum number of sequences allowed, initialized to 1.
    - `n_pad`: The required padding for the cache, initialized to 1.
    - `n_swa`: The number of SWA (Stochastic Weight Averaging) steps, initialized to 0.
    - `swa_type`: The type of SWA being used, initialized to LLAMA_SWA_TYPE_NONE.
    - `ctxs`: A vector of ggml_context_ptr objects representing contexts for different buffer types.
    - `bufs`: A vector of ggml_backend_buffer_ptr objects representing backend buffers.
    - `cells`: An instance of llama_kv_cells_unified managing the key-value cells in the cache.
    - `layers`: A vector of kv_layer structs representing the layers in the cache.
    - `map_layer_ids`: An unordered map mapping model layer IDs to KV cache layer IDs.
- **Description**: The `llama_kv_cache_unified` class is a specialized data structure that manages a unified key-value cache for a llama model. It inherits from `llama_memory_i` and provides functionality to initialize, update, and manage the cache, including operations like adding, removing, and copying sequences. The class supports defragmentation and shifting of cache entries, and it is designed to handle multiple layers and sequences efficiently. It uses a ring buffer approach to manage the cache cells and provides APIs for interacting with the cache's state and structure.
- **Member Functions**:
    - [`llama_kv_cache_unified::~llama_kv_cache_unified`](#llama_kv_cache_unifiedllama_kv_cache_unified)
    - [`llama_kv_cache_unified::llama_kv_cache_unified`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedllama_kv_cache_unified)
    - [`llama_kv_cache_unified::clear`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedclear)
    - [`llama_kv_cache_unified::seq_rm`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedseq_rm)
    - [`llama_kv_cache_unified::seq_cp`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedseq_cp)
    - [`llama_kv_cache_unified::seq_keep`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedseq_keep)
    - [`llama_kv_cache_unified::seq_add`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedseq_add)
    - [`llama_kv_cache_unified::seq_div`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedseq_div)
    - [`llama_kv_cache_unified::seq_pos_min`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedseq_pos_min)
    - [`llama_kv_cache_unified::seq_pos_max`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedseq_pos_max)
    - [`llama_kv_cache_unified::init_batch`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedinit_batch)
    - [`llama_kv_cache_unified::init_full`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedinit_full)
    - [`llama_kv_cache_unified::init_update`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedinit_update)
    - [`llama_kv_cache_unified::prepare`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedprepare)
    - [`llama_kv_cache_unified::update`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedupdate)
    - [`llama_kv_cache_unified::find_slot`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedfind_slot)
    - [`llama_kv_cache_unified::apply_ubatch`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedapply_ubatch)
    - [`llama_kv_cache_unified::get_can_shift`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedget_can_shift)
    - [`llama_kv_cache_unified::get_size`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedget_size)
    - [`llama_kv_cache_unified::get_has_shift`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedget_has_shift)
    - [`llama_kv_cache_unified::get_n_kv`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedget_n_kv)
    - [`llama_kv_cache_unified::get_k`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedget_k)
    - [`llama_kv_cache_unified::get_v`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedget_v)
    - [`llama_kv_cache_unified::cpy_k`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedcpy_k)
    - [`llama_kv_cache_unified::cpy_v`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedcpy_v)
    - [`llama_kv_cache_unified::set_input_kq_mask`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedset_input_kq_mask)
    - [`llama_kv_cache_unified::set_input_k_shift`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedset_input_k_shift)
    - [`llama_kv_cache_unified::set_input_pos_bucket`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedset_input_pos_bucket)
    - [`llama_kv_cache_unified::total_size`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedtotal_size)
    - [`llama_kv_cache_unified::size_k_bytes`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedsize_k_bytes)
    - [`llama_kv_cache_unified::size_v_bytes`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedsize_v_bytes)
    - [`llama_kv_cache_unified::build_rope_shift`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedbuild_rope_shift)
    - [`llama_kv_cache_unified::build_graph_shift`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedbuild_graph_shift)
    - [`llama_kv_cache_unified::build_graph_defrag`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedbuild_graph_defrag)
    - [`llama_kv_cache_unified::defrag_prepare`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifieddefrag_prepare)
    - [`llama_kv_cache_unified::is_masked_swa`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedis_masked_swa)
    - [`llama_kv_cache_unified::state_write`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedstate_write)
    - [`llama_kv_cache_unified::state_read`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedstate_read)
    - [`llama_kv_cache_unified::state_write_meta`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedstate_write_meta)
    - [`llama_kv_cache_unified::state_write_data`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedstate_write_data)
    - [`llama_kv_cache_unified::state_read_meta`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedstate_read_meta)
    - [`llama_kv_cache_unified::state_read_data`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedstate_read_data)
    - [`llama_kv_cache_unified::get_padding`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unifiedget_padding)
- **Inherits From**:
    - [`llama_memory_i`](llama-memory.h.driver.md#llama_memory_i)

**Methods**

---
#### llama\_kv\_cache\_unified::\~llama\_kv\_cache\_unified<!-- {{#callable:llama_kv_cache_unified::~llama_kv_cache_unified}} -->
The destructor `~llama_kv_cache_unified()` is a default destructor for the `llama_kv_cache_unified` class, which performs no specific operations upon object destruction.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as `default`, meaning it relies on the compiler-generated default behavior for destructors.
    - No custom cleanup or resource deallocation is performed in this destructor.
- **Output**: There is no output from this destructor as it performs no operations.
- **See also**: [`llama_kv_cache_unified`](#llama_kv_cache_unified)  (Data Structure)



---
### defrag\_info<!-- {{#data_structure:llama_kv_cache_unified::defrag_info}} -->
- **Type**: `struct`
- **Members**:
    - `ids`: A vector of unsigned 32-bit integers that indicates the target position for each cell, where cell i moves to ids[i].
- **Description**: The `defrag_info` struct is designed to manage and store information about the movement of cells within a data structure. It contains a single member, `ids`, which is a vector of unsigned 32-bit integers. Each element in this vector represents the target position for a corresponding cell, indicating where each cell should be moved. If a cell's target position is the same as its current position or equal to the size of the vector, it implies that the cell is not moved. The struct also provides a method `empty()` to check if the `ids` vector is empty, which can be used to determine if there are any cell movements to process.
- **Member Functions**:
    - [`llama_kv_cache_unified::defrag_info::empty`](#defrag_infoempty)

**Methods**

---
#### defrag\_info::empty<!-- {{#callable:llama_kv_cache_unified::defrag_info::empty}} -->
The `empty` function checks if the `ids` vector in the `defrag_info` struct is empty.
- **Inputs**: None
- **Control Flow**:
    - The function calls the `empty` method on the `ids` vector, which is a standard library function that returns `true` if the vector is empty and `false` otherwise.
    - The result of the `ids.empty()` call is returned as the output of the `empty` function.
- **Output**: A boolean value indicating whether the `ids` vector is empty (`true`) or not (`false`).
- **See also**: [`llama_kv_cache_unified::defrag_info`](#llama_kv_cache_unified::defrag_info)  (Data Structure)



---
### kv\_layer<!-- {{#data_structure:llama_kv_cache_unified::kv_layer}} -->
- **Type**: `struct`
- **Members**:
    - `il`: Represents the layer index in the model, which may differ from the layer index in the KV cache.
    - `k`: A pointer to a ggml_tensor representing the key tensor for the layer.
    - `v`: A pointer to a ggml_tensor representing the value tensor for the layer.
- **Description**: The `kv_layer` struct is a data structure used to represent a layer within a model, specifically in the context of key-value caching. It contains an index `il` that identifies the layer within the model, which may not correspond directly to its index in the key-value cache. The struct also holds pointers to `ggml_tensor` objects `k` and `v`, which represent the key and value tensors associated with the layer, respectively. This struct is likely used in the management and manipulation of key-value pairs in a neural network model's caching mechanism.


---
### llama\_kv\_cache\_unified\_state<!-- {{#data_structure:llama_kv_cache_unified_state}} -->
- **Type**: `class`
- **Members**:
    - `status`: Represents the memory status of the llama_kv_cache_unified_state.
    - `kv`: Pointer to a llama_kv_cache_unified object, representing the key-value cache.
    - `lctx`: Pointer to a llama_context object, used for context management.
    - `do_shift`: Boolean flag indicating whether a shift operation should be performed.
    - `dinfo`: Holds defragmentation information for the cache.
    - `sbatch`: Represents a batch of sequences to be processed.
    - `i_next`: Index of the next ubatch to process.
    - `heads`: Stores the head positions for ubatches.
    - `ubatches`: Vector of ubatches to be processed.
    - `n_kv`: Heuristic value to avoid attending the full cache if not utilized.
    - `head`: Indicates the beginning of the current slot for ubatch insertion.
- **Description**: The `llama_kv_cache_unified_state` class is a specialized state management class for handling key-value cache operations in the llama framework. It extends the `llama_memory_state_i` interface and provides mechanisms to manage cache states, including full-cache, update, and decode states. The class maintains various attributes such as memory status, cache pointers, context, and batch processing states. It also includes methods for cache manipulation, such as copying and setting input tensors, and managing ubatch processing. The class is designed to optimize cache usage and facilitate efficient data processing in the llama framework.
- **Member Functions**:
    - [`llama_kv_cache_unified_state::llama_kv_cache_unified_state`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unified_statellama_kv_cache_unified_state)
    - [`llama_kv_cache_unified_state::llama_kv_cache_unified_state`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unified_statellama_kv_cache_unified_state)
    - [`llama_kv_cache_unified_state::llama_kv_cache_unified_state`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unified_statellama_kv_cache_unified_state)
    - [`llama_kv_cache_unified_state::llama_kv_cache_unified_state`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unified_statellama_kv_cache_unified_state)
    - [`llama_kv_cache_unified_state::~llama_kv_cache_unified_state`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unified_statellama_kv_cache_unified_state)
    - [`llama_kv_cache_unified_state::next`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unified_statenext)
    - [`llama_kv_cache_unified_state::apply`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unified_stateapply)
    - [`llama_kv_cache_unified_state::out_ids`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unified_stateout_ids)
    - [`llama_kv_cache_unified_state::get_status`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unified_stateget_status)
    - [`llama_kv_cache_unified_state::get_ubatch`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unified_stateget_ubatch)
    - [`llama_kv_cache_unified_state::get_n_kv`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unified_stateget_n_kv)
    - [`llama_kv_cache_unified_state::get_k`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unified_stateget_k)
    - [`llama_kv_cache_unified_state::get_v`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unified_stateget_v)
    - [`llama_kv_cache_unified_state::cpy_k`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unified_statecpy_k)
    - [`llama_kv_cache_unified_state::cpy_v`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unified_statecpy_v)
    - [`llama_kv_cache_unified_state::set_input_k_shift`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unified_stateset_input_k_shift)
    - [`llama_kv_cache_unified_state::set_input_kq_mask`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unified_stateset_input_kq_mask)
    - [`llama_kv_cache_unified_state::set_input_pos_bucket`](llama-kv-cache-unified.cpp.driver.md#llama_kv_cache_unified_stateset_input_pos_bucket)
- **Inherits From**:
    - [`llama_memory_state_i`](llama-memory.h.driver.md#llama_memory_state_i)


