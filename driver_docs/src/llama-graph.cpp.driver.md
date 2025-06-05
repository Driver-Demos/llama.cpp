# Purpose
The provided C++ source code file is part of a larger software system designed for processing and managing data in a machine learning context, specifically related to the "llama" architecture. This file appears to be a core component of a library or framework that handles various input configurations and operations for a neural network model. The code is structured around a series of classes and functions that set up and manipulate input data for different components of a neural network, such as embeddings, attention mechanisms, and pooling layers. The file includes numerous functions that configure input tensors, apply transformations, and manage the flow of data through the network's layers.

The code is highly modular, with each function focusing on a specific aspect of the input data processing, such as setting input embeddings, positional encodings, attention masks, and more. The functions utilize a variety of mathematical operations and data structures to prepare the input data for further processing by the neural network. The file also includes classes that encapsulate the context and parameters required for these operations, ensuring that the data is processed consistently and efficiently. This code is likely intended to be part of a larger library that can be imported and used in different machine learning applications, providing a flexible and extensible framework for building and training neural network models.
# Imports and Dependencies

---
- `llama-graph.h`
- `llama-impl.h`
- `llama-batch.h`
- `llama-cparams.h`
- `llama-kv-cache-unified.h`
- `llama-kv-cache-unified-iswa.h`
- `llama-kv-cache-recurrent.h`
- `cassert`
- `cmath`
- `cstring`


# Data Structures

---
### llm\_graph\_input\_embd<!-- {{#data_structure:llm_graph_input_embd}} -->
- **Description**: [See definition](llama-graph.h.driver.md#llm_graph_input_embd)
- **Member Functions**:
    - [`llm_graph_input_embd::llm_graph_input_embd`](llama-graph.h.driver.md#llm_graph_input_embdllm_graph_input_embd)
    - [`llm_graph_input_embd::~llm_graph_input_embd`](llama-graph.h.driver.md#llm_graph_input_embdllm_graph_input_embd)
    - [`llm_graph_input_embd::set_input`](#llm_graph_input_embdset_input)
- **Inherits From**:
    - `llm_graph_input_i`

**Methods**

---
#### llm\_graph\_input\_embd::set\_input<!-- {{#callable:llm_graph_input_embd::set_input}} -->
The `set_input` function sets the input tensors for tokens and embeddings in the `llm_graph_input_embd` class using data from a `llama_ubatch` object.
- **Inputs**:
    - `ubatch`: A pointer to a `llama_ubatch` object containing token and embedding data to be set in the input tensors.
- **Control Flow**:
    - Check if `ubatch->token` is not null.
    - If `ubatch->token` is not null, calculate the number of tokens and set the `tokens` tensor using [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set).
    - Check if `ubatch->embd` is not null.
    - If `ubatch->embd` is not null, calculate the number of embeddings and tokens, then set the `embd` tensor using [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set).
- **Output**: The function does not return a value; it modifies the `tokens` and `embd` tensors of the `llm_graph_input_embd` object in place.
- **Functions called**:
    - [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`ggml_element_size`](../ggml/src/ggml.c.driver.md#ggml_element_size)
- **See also**: [`llm_graph_input_embd`](llama-graph.h.driver.md#llm_graph_input_embd)  (Data Structure)



---
### llm\_graph\_input\_pos<!-- {{#data_structure:llm_graph_input_pos}} -->
- **Description**: [See definition](llama-graph.h.driver.md#llm_graph_input_pos)
- **Member Functions**:
    - [`llm_graph_input_pos::llm_graph_input_pos`](llama-graph.h.driver.md#llm_graph_input_posllm_graph_input_pos)
    - [`llm_graph_input_pos::~llm_graph_input_pos`](llama-graph.h.driver.md#llm_graph_input_posllm_graph_input_pos)
    - [`llm_graph_input_pos::set_input`](#llm_graph_input_posset_input)
- **Inherits From**:
    - `llm_graph_input_i`

**Methods**

---
#### llm\_graph\_input\_pos::set\_input<!-- {{#callable:llm_graph_input_pos::set_input}} -->
The `set_input` function sets the position tensor for a batch of tokens, converting 1D positions to 4D if necessary, and updates the tensor using the backend tensor set function.
- **Inputs**:
    - `ubatch`: A pointer to a `llama_ubatch` structure containing the batch of tokens and their positions.
- **Control Flow**:
    - Check if both `ubatch->pos` and `pos` are non-null.
    - Retrieve the number of tokens from `ubatch->n_tokens`.
    - If `ubatch->token` is non-null and `n_pos_per_embd` equals 4, convert 1D positions to 4D by repeating the position three times and setting the fourth dimension to zero.
    - Use [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set) to update the `pos` tensor with the converted or original position data.
- **Output**: The function does not return a value; it updates the `pos` tensor of the `llm_graph_input_pos` object.
- **Functions called**:
    - [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`ggml_element_size`](../ggml/src/ggml.c.driver.md#ggml_element_size)
- **See also**: [`llm_graph_input_pos`](llama-graph.h.driver.md#llm_graph_input_pos)  (Data Structure)



---
### llm\_graph\_input\_attn\_temp<!-- {{#data_structure:llm_graph_input_attn_temp}} -->
- **Description**: [See definition](llama-graph.h.driver.md#llm_graph_input_attn_temp)
- **Member Functions**:
    - [`llm_graph_input_attn_temp::llm_graph_input_attn_temp`](llama-graph.h.driver.md#llm_graph_input_attn_templlm_graph_input_attn_temp)
    - [`llm_graph_input_attn_temp::~llm_graph_input_attn_temp`](llama-graph.h.driver.md#llm_graph_input_attn_templlm_graph_input_attn_temp)
    - [`llm_graph_input_attn_temp::set_input`](#llm_graph_input_attn_tempset_input)
- **Inherits From**:
    - `llm_graph_input_i`

**Methods**

---
#### llm\_graph\_input\_attn\_temp::set\_input<!-- {{#callable:llm_graph_input_attn_temp::set_input}} -->
The `set_input` function calculates and sets the attention scale tensor based on the positions of tokens in a batch.
- **Inputs**:
    - `ubatch`: A pointer to a `llama_ubatch` structure containing the batch of tokens and their positions.
- **Control Flow**:
    - Check if `ubatch->pos` and `attn_scale` are not null.
    - Retrieve the number of tokens from `ubatch->n_tokens`.
    - Initialize a vector `attn_scale_data` with zeros, sized to the number of tokens.
    - Iterate over each token position in `ubatch->pos`.
    - For each position, calculate the attention scale using a logarithmic formula and store it in `attn_scale_data`.
    - Set the `attn_scale` tensor with the calculated data using [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set).
- **Output**: The function does not return a value; it modifies the `attn_scale` tensor in place.
- **Functions called**:
    - [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`ggml_element_size`](../ggml/src/ggml.c.driver.md#ggml_element_size)
- **See also**: [`llm_graph_input_attn_temp`](llama-graph.h.driver.md#llm_graph_input_attn_temp)  (Data Structure)



---
### llm\_graph\_input\_pos\_bucket<!-- {{#data_structure:llm_graph_input_pos_bucket}} -->
- **Description**: [See definition](llama-graph.h.driver.md#llm_graph_input_pos_bucket)
- **Member Functions**:
    - [`llm_graph_input_pos_bucket::llm_graph_input_pos_bucket`](llama-graph.h.driver.md#llm_graph_input_pos_bucketllm_graph_input_pos_bucket)
    - [`llm_graph_input_pos_bucket::~llm_graph_input_pos_bucket`](llama-graph.h.driver.md#llm_graph_input_pos_bucketllm_graph_input_pos_bucket)
    - [`llm_graph_input_pos_bucket::set_input`](#llm_graph_input_pos_bucketset_input)
- **Inherits From**:
    - `llm_graph_input_i`

**Methods**

---
#### llm\_graph\_input\_pos\_bucket::set\_input<!-- {{#callable:llm_graph_input_pos_bucket::set_input}} -->
The `set_input` function initializes the `pos_bucket` tensor with relative position bucket values based on the positions in the `ubatch`.
- **Inputs**:
    - `ubatch`: A pointer to a `llama_ubatch` structure containing token positions and other batch-related data.
- **Control Flow**:
    - Check if `pos_bucket` is not null.
    - Retrieve the number of tokens from `ubatch`.
    - Assert that `pos_bucket`'s buffer is hosted and `ubatch` does not have equal sequences.
    - Cast `pos_bucket->data` to an `int32_t` pointer.
    - Iterate over tokens to compute relative position buckets using [`llama_relative_position_bucket`](#llama_relative_position_bucket) and store them in `pos_bucket->data`.
- **Output**: The function does not return a value; it modifies the `pos_bucket` tensor in place.
- **Functions called**:
    - [`ggml_backend_buffer_is_host`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_is_host)
    - [`llama_relative_position_bucket`](#llama_relative_position_bucket)
- **See also**: [`llm_graph_input_pos_bucket`](llama-graph.h.driver.md#llm_graph_input_pos_bucket)  (Data Structure)



---
### llm\_graph\_input\_pos\_bucket\_kv<!-- {{#data_structure:llm_graph_input_pos_bucket_kv}} -->
- **Description**: [See definition](llama-graph.h.driver.md#llm_graph_input_pos_bucket_kv)
- **Member Functions**:
    - [`llm_graph_input_pos_bucket_kv::llm_graph_input_pos_bucket_kv`](llama-graph.h.driver.md#llm_graph_input_pos_bucket_kvllm_graph_input_pos_bucket_kv)
    - [`llm_graph_input_pos_bucket_kv::~llm_graph_input_pos_bucket_kv`](llama-graph.h.driver.md#llm_graph_input_pos_bucket_kvllm_graph_input_pos_bucket_kv)
    - [`llm_graph_input_pos_bucket_kv::set_input`](#llm_graph_input_pos_bucket_kvset_input)
- **Inherits From**:
    - `llm_graph_input_i`

**Methods**

---
#### llm\_graph\_input\_pos\_bucket\_kv::set\_input<!-- {{#callable:llm_graph_input_pos_bucket_kv::set_input}} -->
The `set_input` method of the `llm_graph_input_pos_bucket_kv` class sets the input position bucket for the key-value state if the position bucket is not null.
- **Inputs**:
    - `ubatch`: A pointer to a `llama_ubatch` object, which contains the input data for the operation.
- **Control Flow**:
    - Check if `pos_bucket` is not null.
    - If `pos_bucket` is not null, call `set_input_pos_bucket` on `kv_state` with `pos_bucket` and `ubatch` as arguments.
- **Output**: The function does not return any value; it performs an operation on the `kv_state` object.
- **See also**: [`llm_graph_input_pos_bucket_kv`](llama-graph.h.driver.md#llm_graph_input_pos_bucket_kv)  (Data Structure)



---
### llm\_graph\_input\_out\_ids<!-- {{#data_structure:llm_graph_input_out_ids}} -->
- **Description**: [See definition](llama-graph.h.driver.md#llm_graph_input_out_ids)
- **Member Functions**:
    - [`llm_graph_input_out_ids::llm_graph_input_out_ids`](llama-graph.h.driver.md#llm_graph_input_out_idsllm_graph_input_out_ids)
    - [`llm_graph_input_out_ids::~llm_graph_input_out_ids`](llama-graph.h.driver.md#llm_graph_input_out_idsllm_graph_input_out_ids)
    - [`llm_graph_input_out_ids::set_input`](#llm_graph_input_out_idsset_input)
- **Inherits From**:
    - `llm_graph_input_i`

**Methods**

---
#### llm\_graph\_input\_out\_ids::set\_input<!-- {{#callable:llm_graph_input_out_ids::set_input}} -->
The `set_input` function configures the output IDs for a model based on the input batch and model parameters.
- **Inputs**:
    - `ubatch`: A pointer to a `llama_ubatch` structure containing the input batch data, including the number of tokens and optional output flags.
- **Control Flow**:
    - Check if causal attention is enabled or if pooling type is none.
    - If `out_ids` is not initialized, log a warning.
    - If `out_ids` is initialized, retrieve the number of tokens from `ubatch`.
    - Assert that the `out_ids` buffer is hosted on the CPU.
    - If the number of outputs equals the number of tokens, populate `out_ids` with sequential indices.
    - If `ubatch->output` is available, populate `out_ids` with indices of tokens marked for output.
    - If there is only one output, set `out_ids` to the last token index.
    - Assert that the number of outputs is zero if none of the above conditions are met.
- **Output**: The function does not return a value but modifies the `out_ids` tensor within the `llm_graph_input_out_ids` class to reflect the indices of tokens to be output based on the input batch and model parameters.
- **Functions called**:
    - [`ggml_backend_buffer_is_host`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_is_host)
- **See also**: [`llm_graph_input_out_ids`](llama-graph.h.driver.md#llm_graph_input_out_ids)  (Data Structure)



---
### llm\_graph\_input\_mean<!-- {{#data_structure:llm_graph_input_mean}} -->
- **Description**: [See definition](llama-graph.h.driver.md#llm_graph_input_mean)
- **Member Functions**:
    - [`llm_graph_input_mean::llm_graph_input_mean`](llama-graph.h.driver.md#llm_graph_input_meanllm_graph_input_mean)
    - [`llm_graph_input_mean::~llm_graph_input_mean`](llama-graph.h.driver.md#llm_graph_input_meanllm_graph_input_mean)
    - [`llm_graph_input_mean::set_input`](#llm_graph_input_meanset_input)
- **Inherits From**:
    - `llm_graph_input_i`

**Methods**

---
#### llm\_graph\_input\_mean::set\_input<!-- {{#callable:llm_graph_input_mean::set_input}} -->
The `set_input` function initializes and populates the `mean` tensor with normalized values based on the sequence data from the `llama_ubatch` input when the pooling type is set to mean.
- **Inputs**:
    - `ubatch`: A pointer to a `llama_ubatch` structure containing sequence data, including the number of tokens, sequence tokens, sequences, and sequence IDs.
- **Control Flow**:
    - Check if embeddings are enabled and the pooling type is set to mean.
    - Retrieve the number of tokens, sequence tokens, and sequences from the `ubatch`.
    - Assert that the `mean` tensor is initialized and its buffer is hosted on the CPU.
    - Initialize the `mean` tensor data to zero.
    - Create a vector `sum` to accumulate the number of sequence tokens for each sequence ID.
    - Iterate over each sequence to update the `sum` vector with the number of sequence tokens for each sequence ID.
    - Create a vector `div` to store the inverse of the accumulated sequence token counts for normalization.
    - Iterate over the `sum` vector to calculate the inverse for non-zero counts and store in `div`.
    - Iterate over each sequence and sequence token to populate the `mean` tensor with normalized values using the `div` vector.
- **Output**: The function does not return a value; it modifies the `mean` tensor in place.
- **Functions called**:
    - [`ggml_backend_buffer_is_host`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_is_host)
    - [`ggml_element_size`](../ggml/src/ggml.c.driver.md#ggml_element_size)
- **See also**: [`llm_graph_input_mean`](llama-graph.h.driver.md#llm_graph_input_mean)  (Data Structure)



---
### llm\_graph\_input\_cls<!-- {{#data_structure:llm_graph_input_cls}} -->
- **Description**: [See definition](llama-graph.h.driver.md#llm_graph_input_cls)
- **Member Functions**:
    - [`llm_graph_input_cls::llm_graph_input_cls`](llama-graph.h.driver.md#llm_graph_input_clsllm_graph_input_cls)
    - [`llm_graph_input_cls::~llm_graph_input_cls`](llama-graph.h.driver.md#llm_graph_input_clsllm_graph_input_cls)
    - [`llm_graph_input_cls::set_input`](#llm_graph_input_clsset_input)
- **Inherits From**:
    - `llm_graph_input_i`

**Methods**

---
#### llm\_graph\_input\_cls::set\_input<!-- {{#callable:llm_graph_input_cls::set_input}} -->
The `set_input` method of the `llm_graph_input_cls` class configures the input data for a graph based on the specified pooling type and updates the `cls` tensor accordingly.
- **Inputs**:
    - `ubatch`: A pointer to a `llama_ubatch` structure containing batch data, including tokens, sequence IDs, and positions.
- **Control Flow**:
    - Check if embeddings are enabled and the pooling type is either CLS or RANK.
    - Initialize variables for the number of tokens, sequence tokens, and sequences from `ubatch`.
    - Assert that `cls` is valid and its buffer is hosted on the CPU.
    - Clear the `cls` data buffer.
    - Iterate over sequences and update `cls` data based on the first position of each sequence for CLS or RANK pooling.
    - Check if embeddings are enabled and the pooling type is LAST.
    - Initialize variables for the number of tokens, sequence tokens, and sequences from `ubatch`.
    - Assert that `cls` is valid and its buffer is hosted on the CPU.
    - Clear the `cls` data buffer.
    - Initialize vectors to track the last position and row for each token.
    - Iterate over sequences and update the last position and row for each token.
    - Update `cls` data with the last row indices for each token.
- **Output**: The method updates the `cls` tensor's data based on the specified pooling type, either setting it to the first or last position of each sequence.
- **Functions called**:
    - [`ggml_backend_buffer_is_host`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_is_host)
    - [`ggml_element_size`](../ggml/src/ggml.c.driver.md#ggml_element_size)
- **See also**: [`llm_graph_input_cls`](llama-graph.h.driver.md#llm_graph_input_cls)  (Data Structure)



---
### llm\_graph\_input\_s\_copy<!-- {{#data_structure:llm_graph_input_s_copy}} -->
- **Description**: [See definition](llama-graph.h.driver.md#llm_graph_input_s_copy)
- **Member Functions**:
    - [`llm_graph_input_s_copy::llm_graph_input_s_copy`](llama-graph.h.driver.md#llm_graph_input_s_copyllm_graph_input_s_copy)
    - [`llm_graph_input_s_copy::~llm_graph_input_s_copy`](llama-graph.h.driver.md#llm_graph_input_s_copyllm_graph_input_s_copy)
    - [`llm_graph_input_s_copy::set_input`](#llm_graph_input_s_copyset_input)
- **Inherits From**:
    - `llm_graph_input_i`

**Methods**

---
#### llm\_graph\_input\_s\_copy::set\_input<!-- {{#callable:llm_graph_input_s_copy::set_input}} -->
The `set_input` function updates the `s_copy` tensor with values from the `kv_state` object, assuming the copy destinations are within a specific range.
- **Inputs**:
    - `ubatch`: A pointer to a `llama_ubatch` object, which is not used in this function.
- **Control Flow**:
    - The function begins by marking the `ubatch` parameter as unused with `GGML_UNUSED` macro.
    - It retrieves the number of key-value pairs (`n_kv`) from the `kv_state` object.
    - If `s_copy` is not null, it asserts that the buffer of `s_copy` is hosted on the CPU.
    - It casts the data of `s_copy` to an `int32_t` pointer.
    - A loop iterates over the range of `n_kv`, updating each element of `s_copy`'s data with the corresponding value from `kv_state->s_copy`.
- **Output**: The function does not return any value; it modifies the `s_copy` tensor in place.
- **Functions called**:
    - [`ggml_backend_buffer_is_host`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_is_host)
- **See also**: [`llm_graph_input_s_copy`](llama-graph.h.driver.md#llm_graph_input_s_copy)  (Data Structure)



---
### llm\_graph\_input\_s\_mask<!-- {{#data_structure:llm_graph_input_s_mask}} -->
- **Description**: [See definition](llama-graph.h.driver.md#llm_graph_input_s_mask)
- **Member Functions**:
    - [`llm_graph_input_s_mask::llm_graph_input_s_mask`](llama-graph.h.driver.md#llm_graph_input_s_maskllm_graph_input_s_mask)
    - [`llm_graph_input_s_mask::~llm_graph_input_s_mask`](llama-graph.h.driver.md#llm_graph_input_s_maskllm_graph_input_s_mask)
    - [`llm_graph_input_s_mask::set_input`](#llm_graph_input_s_maskset_input)
- **Inherits From**:
    - `llm_graph_input_i`

**Methods**

---
#### llm\_graph\_input\_s\_mask::set\_input<!-- {{#callable:llm_graph_input_s_mask::set_input}} -->
The `set_input` function updates the `s_mask` tensor with values from the `kv_state`'s `s_mask` method for each key-value pair index.
- **Inputs**:
    - `ubatch`: A pointer to a `llama_ubatch` object, which is not used in this function.
- **Control Flow**:
    - The function begins by marking the `ubatch` parameter as unused with `GGML_UNUSED` macro.
    - It retrieves the number of key-value pairs (`n_kv`) from the `kv_state` object.
    - If `s_mask` is not null, it asserts that the buffer of `s_mask` is hosted on the CPU.
    - It then casts the data of `s_mask` to a float pointer.
    - A loop iterates over each key-value index up to `n_kv`, setting each element in `s_mask`'s data to the corresponding value from `kv_state->s_mask(i)`.
- **Output**: The function does not return any value; it modifies the `s_mask` tensor in place.
- **Functions called**:
    - [`ggml_backend_buffer_is_host`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_is_host)
- **See also**: [`llm_graph_input_s_mask`](llama-graph.h.driver.md#llm_graph_input_s_mask)  (Data Structure)



---
### llm\_graph\_input\_cross\_embd<!-- {{#data_structure:llm_graph_input_cross_embd}} -->
- **Description**: [See definition](llama-graph.h.driver.md#llm_graph_input_cross_embd)
- **Member Functions**:
    - [`llm_graph_input_cross_embd::llm_graph_input_cross_embd`](llama-graph.h.driver.md#llm_graph_input_cross_embdllm_graph_input_cross_embd)
    - [`llm_graph_input_cross_embd::~llm_graph_input_cross_embd`](llama-graph.h.driver.md#llm_graph_input_cross_embdllm_graph_input_cross_embd)
    - [`llm_graph_input_cross_embd::set_input`](#llm_graph_input_cross_embdset_input)
- **Inherits From**:
    - `llm_graph_input_i`

**Methods**

---
#### llm\_graph\_input\_cross\_embd::set\_input<!-- {{#callable:llm_graph_input_cross_embd::set_input}} -->
The `set_input` function sets the input tensor for cross-embedding in a graph if certain conditions are met.
- **Inputs**:
    - `ubatch`: A pointer to a `llama_ubatch` structure, which is not used in this function.
- **Control Flow**:
    - The function begins by marking the `ubatch` parameter as unused with `GGML_UNUSED` macro.
    - It checks if `cross_embd` is not null and if `cross->v_embd` is not empty.
    - If both conditions are true, it asserts that `cross_embd->type` is `GGML_TYPE_F32`.
    - It then sets the `cross_embd` tensor using [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set) with data from `cross->v_embd`.
- **Output**: The function does not return any value; it modifies the `cross_embd` tensor in place if conditions are met.
- **Functions called**:
    - [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
- **See also**: [`llm_graph_input_cross_embd`](llama-graph.h.driver.md#llm_graph_input_cross_embd)  (Data Structure)



---
### llm\_graph\_input\_attn\_no\_cache<!-- {{#data_structure:llm_graph_input_attn_no_cache}} -->
- **Description**: [See definition](llama-graph.h.driver.md#llm_graph_input_attn_no_cache)
- **Member Functions**:
    - [`llm_graph_input_attn_no_cache::llm_graph_input_attn_no_cache`](llama-graph.h.driver.md#llm_graph_input_attn_no_cachellm_graph_input_attn_no_cache)
    - [`llm_graph_input_attn_no_cache::~llm_graph_input_attn_no_cache`](llama-graph.h.driver.md#llm_graph_input_attn_no_cachellm_graph_input_attn_no_cache)
    - [`llm_graph_input_attn_no_cache::get_kq_mask`](llama-graph.h.driver.md#llm_graph_input_attn_no_cacheget_kq_mask)
    - [`llm_graph_input_attn_no_cache::set_input`](#llm_graph_input_attn_no_cacheset_input)
- **Inherits From**:
    - `llm_graph_input_i`

**Methods**

---
#### llm\_graph\_input\_attn\_no\_cache::set\_input<!-- {{#callable:llm_graph_input_attn_no_cache::set_input}} -->
The `set_input` function configures the key-query mask (`kq_mask`) for attention mechanisms in a neural network model based on the input batch and model parameters.
- **Inputs**:
    - `ubatch`: A pointer to a `llama_ubatch` structure containing the input batch data, including token counts, sequence IDs, and positions.
- **Control Flow**:
    - Check if `kq_mask` is not null.
    - If `cparams.causal_attn` is true, set up a causal attention mask using the token positions and sequence IDs from `ubatch`.
    - Iterate over sequences and tokens to compute the mask values, setting them to negative infinity by default.
    - If `hparams.use_alibi` is true, adjust the mask values based on the absolute difference in token positions; otherwise, set them to zero if conditions are met.
    - If `cparams.causal_attn` is false, set up a non-causal attention mask similarly, but without the position constraint.
    - Fill any remaining mask positions with negative infinity.
- **Output**: The function modifies the `kq_mask` tensor in place, setting its data based on the input batch and model parameters.
- **Functions called**:
    - [`ggml_backend_buffer_is_host`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_is_host)
- **See also**: [`llm_graph_input_attn_no_cache`](llama-graph.h.driver.md#llm_graph_input_attn_no_cache)  (Data Structure)



---
### llm\_graph\_input\_attn\_kv\_unified<!-- {{#data_structure:llm_graph_input_attn_kv_unified}} -->
- **Description**: [See definition](llama-graph.h.driver.md#llm_graph_input_attn_kv_unified)
- **Member Functions**:
    - [`llm_graph_input_attn_kv_unified::llm_graph_input_attn_kv_unified`](llama-graph.h.driver.md#llm_graph_input_attn_kv_unifiedllm_graph_input_attn_kv_unified)
    - [`llm_graph_input_attn_kv_unified::~llm_graph_input_attn_kv_unified`](llama-graph.h.driver.md#llm_graph_input_attn_kv_unifiedllm_graph_input_attn_kv_unified)
    - [`llm_graph_input_attn_kv_unified::get_kq_mask`](llama-graph.h.driver.md#llm_graph_input_attn_kv_unifiedget_kq_mask)
    - [`llm_graph_input_attn_kv_unified::set_input`](#llm_graph_input_attn_kv_unifiedset_input)
- **Inherits From**:
    - `llm_graph_input_i`

**Methods**

---
#### llm\_graph\_input\_attn\_kv\_unified::set\_input<!-- {{#callable:llm_graph_input_attn_kv_unified::set_input}} -->
The `set_input` function sets the input for the attention key-value unified state using a key-query mask if it exists.
- **Inputs**:
    - `ubatch`: A pointer to a `llama_ubatch` object, which contains the input data for the function.
- **Control Flow**:
    - Check if `self_kq_mask` is not null.
    - If `self_kq_mask` is not null, call `set_input_kq_mask` on `kv_state` with `self_kq_mask`, `ubatch`, and `cparams.causal_attn` as arguments.
- **Output**: This function does not return any value.
- **See also**: [`llm_graph_input_attn_kv_unified`](llama-graph.h.driver.md#llm_graph_input_attn_kv_unified)  (Data Structure)



---
### llm\_graph\_input\_attn\_kv\_unified\_iswa<!-- {{#data_structure:llm_graph_input_attn_kv_unified_iswa}} -->
- **Description**: [See definition](llama-graph.h.driver.md#llm_graph_input_attn_kv_unified_iswa)
- **Member Functions**:
    - [`llm_graph_input_attn_kv_unified_iswa::llm_graph_input_attn_kv_unified_iswa`](llama-graph.h.driver.md#llm_graph_input_attn_kv_unified_iswallm_graph_input_attn_kv_unified_iswa)
    - [`llm_graph_input_attn_kv_unified_iswa::~llm_graph_input_attn_kv_unified_iswa`](llama-graph.h.driver.md#llm_graph_input_attn_kv_unified_iswallm_graph_input_attn_kv_unified_iswa)
    - [`llm_graph_input_attn_kv_unified_iswa::get_kq_mask`](llama-graph.h.driver.md#llm_graph_input_attn_kv_unified_iswaget_kq_mask)
    - [`llm_graph_input_attn_kv_unified_iswa::get_kq_mask_swa`](llama-graph.h.driver.md#llm_graph_input_attn_kv_unified_iswaget_kq_mask_swa)
    - [`llm_graph_input_attn_kv_unified_iswa::set_input`](#llm_graph_input_attn_kv_unified_iswaset_input)
- **Inherits From**:
    - `llm_graph_input_i`

**Methods**

---
#### llm\_graph\_input\_attn\_kv\_unified\_iswa::set\_input<!-- {{#callable:llm_graph_input_attn_kv_unified_iswa::set_input}} -->
The `set_input` method configures the input key-query masks for both base and SWA states in a unified key-value attention graph.
- **Inputs**:
    - `ubatch`: A pointer to a `llama_ubatch` object, which contains the batch of data to be processed.
- **Control Flow**:
    - Check if `self_kq_mask` is not null.
    - If `self_kq_mask` is not null, call `set_input_kq_mask` on the base state of `kv_state` with `self_kq_mask`, `ubatch`, and `cparams.causal_attn`.
    - Check if `self_kq_mask_swa` is not null.
    - If `self_kq_mask_swa` is not null, call `set_input_kq_mask` on the SWA state of `kv_state` with `self_kq_mask_swa`, `ubatch`, and `cparams.causal_attn`.
- **Output**: This function does not return any value.
- **See also**: [`llm_graph_input_attn_kv_unified_iswa`](llama-graph.h.driver.md#llm_graph_input_attn_kv_unified_iswa)  (Data Structure)



---
### llm\_graph\_input\_attn\_cross<!-- {{#data_structure:llm_graph_input_attn_cross}} -->
- **Description**: [See definition](llama-graph.h.driver.md#llm_graph_input_attn_cross)
- **Member Functions**:
    - [`llm_graph_input_attn_cross::llm_graph_input_attn_cross`](llama-graph.h.driver.md#llm_graph_input_attn_crossllm_graph_input_attn_cross)
    - [`llm_graph_input_attn_cross::~llm_graph_input_attn_cross`](llama-graph.h.driver.md#llm_graph_input_attn_crossllm_graph_input_attn_cross)
    - [`llm_graph_input_attn_cross::get_kq_mask_cross`](llama-graph.h.driver.md#llm_graph_input_attn_crossget_kq_mask_cross)
    - [`llm_graph_input_attn_cross::set_input`](#llm_graph_input_attn_crossset_input)
- **Inherits From**:
    - `llm_graph_input_i`

**Methods**

---
#### llm\_graph\_input\_attn\_cross::set\_input<!-- {{#callable:llm_graph_input_attn_cross::set_input}} -->
The `set_input` function initializes the cross-attention key-query mask for a given micro-batch of tokens based on sequence IDs.
- **Inputs**:
    - `ubatch`: A pointer to a `llama_ubatch` structure containing the micro-batch of tokens and their associated sequence IDs.
- **Control Flow**:
    - Check if `cross_kq_mask` is not null, indicating that a cross-attention mask is needed.
    - Retrieve the number of encoder outputs (`n_enc`) and the number of tokens (`n_tokens`) from `cross_kq_mask` and `ubatch`, respectively.
    - Assert that the buffer for `cross_kq_mask` is hosted on the CPU and that `ubatch` does not have equal sequences.
    - Initialize a pointer to the data of `cross_kq_mask` for manipulation.
    - Iterate over each token and encoder output to set the mask value to `-INFINITY` by default.
    - For each token, iterate over its sequence IDs and check if any match the encoder's sequence IDs; if a match is found, set the mask value to `0.0f`.
    - Pad the remaining positions in the mask with `-INFINITY` to align with the required padding size.
- **Output**: The function modifies the `cross_kq_mask` tensor in-place, setting its values based on the presence of matching sequence IDs between the tokens and encoder outputs.
- **Functions called**:
    - [`ggml_backend_buffer_is_host`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_is_host)
- **See also**: [`llm_graph_input_attn_cross`](llama-graph.h.driver.md#llm_graph_input_attn_cross)  (Data Structure)



---
### llm\_graph\_context<!-- {{#data_structure:llm_graph_context}} -->
- **Description**: [See definition](llama-graph.h.driver.md#llm_graph_context)
- **Member Functions**:
    - [`llm_graph_context::llm_graph_context`](#llm_graph_contextllm_graph_context)
    - [`llm_graph_context::n_pos_per_embd`](#llm_graph_contextn_pos_per_embd)
    - [`llm_graph_context::cb`](#llm_graph_contextcb)
    - [`llm_graph_context::build_cvec`](#llm_graph_contextbuild_cvec)
    - [`llm_graph_context::build_lora_mm`](#llm_graph_contextbuild_lora_mm)
    - [`llm_graph_context::build_lora_mm_id`](#llm_graph_contextbuild_lora_mm_id)
    - [`llm_graph_context::build_norm`](#llm_graph_contextbuild_norm)
    - [`llm_graph_context::build_ffn`](#llm_graph_contextbuild_ffn)
    - [`llm_graph_context::build_moe_ffn`](#llm_graph_contextbuild_moe_ffn)
    - [`llm_graph_context::build_inp_embd`](#llm_graph_contextbuild_inp_embd)
    - [`llm_graph_context::build_inp_pos`](#llm_graph_contextbuild_inp_pos)
    - [`llm_graph_context::build_inp_attn_scale`](#llm_graph_contextbuild_inp_attn_scale)
    - [`llm_graph_context::build_inp_out_ids`](#llm_graph_contextbuild_inp_out_ids)
    - [`llm_graph_context::build_inp_mean`](#llm_graph_contextbuild_inp_mean)
    - [`llm_graph_context::build_inp_cls`](#llm_graph_contextbuild_inp_cls)
    - [`llm_graph_context::build_inp_s_copy`](#llm_graph_contextbuild_inp_s_copy)
    - [`llm_graph_context::build_inp_s_mask`](#llm_graph_contextbuild_inp_s_mask)
    - [`llm_graph_context::build_inp_cross_embd`](#llm_graph_contextbuild_inp_cross_embd)
    - [`llm_graph_context::build_inp_pos_bucket_enc`](#llm_graph_contextbuild_inp_pos_bucket_enc)
    - [`llm_graph_context::build_inp_pos_bucket_dec`](#llm_graph_contextbuild_inp_pos_bucket_dec)
    - [`llm_graph_context::build_pos_bias`](#llm_graph_contextbuild_pos_bias)
    - [`llm_graph_context::build_attn_mha`](#llm_graph_contextbuild_attn_mha)
    - [`llm_graph_context::build_attn_inp_no_cache`](#llm_graph_contextbuild_attn_inp_no_cache)
    - [`llm_graph_context::build_attn`](#llm_graph_contextbuild_attn)
    - [`llm_graph_context::build_attn_inp_kv_unified`](#llm_graph_contextbuild_attn_inp_kv_unified)
    - [`llm_graph_context::build_attn`](#llm_graph_contextbuild_attn)
    - [`llm_graph_context::build_attn_inp_kv_unified_iswa`](#llm_graph_contextbuild_attn_inp_kv_unified_iswa)
    - [`llm_graph_context::build_attn`](#llm_graph_contextbuild_attn)
    - [`llm_graph_context::build_attn_inp_cross`](#llm_graph_contextbuild_attn_inp_cross)
    - [`llm_graph_context::build_attn`](#llm_graph_contextbuild_attn)
    - [`llm_graph_context::build_copy_mask_state`](#llm_graph_contextbuild_copy_mask_state)
    - [`llm_graph_context::build_rwkv_token_shift_load`](#llm_graph_contextbuild_rwkv_token_shift_load)
    - [`llm_graph_context::build_rwkv_token_shift_store`](#llm_graph_contextbuild_rwkv_token_shift_store)
    - [`llm_graph_context::build_pooling`](#llm_graph_contextbuild_pooling)

**Methods**

---
#### llm\_graph\_context::llm\_graph\_context<!-- {{#callable:llm_graph_context::llm_graph_context}} -->
The `llm_graph_context` constructor initializes an instance of the `llm_graph_context` class using parameters from a `llm_graph_params` object.
- **Inputs**:
    - `params`: An instance of `llm_graph_params` containing various parameters needed to initialize the `llm_graph_context` object, including architecture, hyperparameters, context parameters, and other configuration settings.
- **Control Flow**:
    - The constructor initializes member variables of the `llm_graph_context` class using values from the `params` object.
    - It assigns values to various integer and float member variables, such as `n_embd`, `n_layer`, `n_rot`, `freq_base`, `freq_scale`, etc., using corresponding values from `hparams`, `cparams`, and `ubatch`.
    - It conditionally sets `n_expert_used` based on the `warmup` flag in `cparams`.
    - It initializes a unique pointer `res` to a new `llm_graph_result` object.
- **Output**: An initialized `llm_graph_context` object with its member variables set according to the provided `params`.
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::n\_pos\_per\_embd<!-- {{#callable:llm_graph_context::n_pos_per_embd}} -->
The `n_pos_per_embd` function returns the number of positions per embedding based on the rope type in the hyperparameters.
- **Inputs**: None
- **Control Flow**:
    - Check if `hparams.rope_type` is equal to `LLAMA_ROPE_TYPE_MROPE`.
    - If true, return 4.
    - If false, return 1.
- **Output**: An integer value indicating the number of positions per embedding, either 4 or 1.
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::cb<!-- {{#callable:llm_graph_context::cb}} -->
The `cb` function invokes a callback function `cb_func` with the current tensor, name, and index level if the callback function is defined.
- **Inputs**:
    - `cur`: A pointer to a `ggml_tensor` object representing the current tensor.
    - `name`: A constant character pointer representing the name associated with the current tensor.
    - `il`: An integer representing the index level or layer index.
- **Control Flow**:
    - Check if the callback function `cb_func` is defined.
    - If `cb_func` is defined, call it with `ubatch`, `cur`, `name`, and `il` as arguments.
- **Output**: This function does not return any value.
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_cvec<!-- {{#callable:llm_graph_context::build_cvec}} -->
The `build_cvec` function applies a transformation to a given tensor using a context and an integer index.
- **Inputs**:
    - `cur`: A pointer to a `ggml_tensor` object representing the current tensor to be transformed.
    - `il`: An integer representing an index or layer number used in the transformation process.
- **Control Flow**:
    - The function calls the `apply_to` method on the `cvec` object, passing `ctx0`, `cur`, and `il` as arguments.
    - The `apply_to` method performs the transformation on the `cur` tensor using the context `ctx0` and the index `il`.
- **Output**: Returns a pointer to a `ggml_tensor` object that is the result of applying the transformation to the input tensor `cur`.
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_lora\_mm<!-- {{#callable:llm_graph_context::build_lora_mm}} -->
The `build_lora_mm` function performs a matrix multiplication between two tensors and applies LoRA (Low-Rank Adaptation) transformations if applicable, returning the resulting tensor.
- **Inputs**:
    - `w`: A pointer to a `ggml_tensor` representing the weight matrix for the matrix multiplication.
    - `cur`: A pointer to a `ggml_tensor` representing the current tensor to be multiplied with the weight matrix.
- **Control Flow**:
    - Initialize `res` as the result of matrix multiplication between `w` and `cur` using [`ggml_mul_mat`](../ggml/src/ggml.c.driver.md#ggml_mul_mat).
    - Iterate over each LoRA configuration in `loras`.
    - For each LoRA, retrieve the corresponding weight using `get_weight` method on the LoRA object.
    - If the weight is not found, continue to the next LoRA configuration.
    - Calculate the scale using the LoRA's alpha and adapter scale.
    - Perform two matrix multiplications: first between `lw->a` and `cur`, then between `lw->b` and the result of the first multiplication.
    - Scale the result of the second multiplication by the calculated scale using [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale).
    - Add the scaled result to `res` using [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add).
    - Return the final `res` tensor.
- **Output**: A pointer to a `ggml_tensor` representing the result of the matrix multiplication and any applied LoRA transformations.
- **Functions called**:
    - [`ggml_mul_mat`](../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_lora\_mm\_id<!-- {{#callable:llm_graph_context::build_lora_mm_id}} -->
The `build_lora_mm_id` function performs a matrix multiplication with optional LoRA (Low-Rank Adaptation) adjustments on the input tensors and returns the resulting tensor.
- **Inputs**:
    - `w`: A `ggml_tensor` pointer representing the weight matrix for the multiplication.
    - `cur`: A `ggml_tensor` pointer representing the current tensor to be multiplied.
    - `ids`: A `ggml_tensor` pointer representing the IDs used in the matrix multiplication.
- **Control Flow**:
    - Initialize the result tensor `res` by performing a matrix multiplication with IDs using [`ggml_mul_mat_id`](../ggml/src/ggml.c.driver.md#ggml_mul_mat_id) on `w`, `cur`, and `ids`.
    - Iterate over each LoRA configuration in the `loras` collection.
    - For each LoRA, retrieve the LoRA weight `lw` for the weight matrix `w`.
    - If `lw` is not found, continue to the next LoRA configuration.
    - Calculate the scale factor using the LoRA's alpha, rank, and scale values.
    - Perform a nested matrix multiplication with IDs using [`ggml_mul_mat_id`](../ggml/src/ggml.c.driver.md#ggml_mul_mat_id) on `lw->a`, `cur`, and `ids`, followed by another multiplication with `lw->b` and `ids`.
    - Scale the resulting tensor `ab_cur` using [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale) with the calculated scale factor.
    - Add the scaled tensor `ab_cur` to the result tensor `res` using [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add).
    - Return the final result tensor `res`.
- **Output**: A `ggml_tensor` pointer representing the result of the matrix multiplication with optional LoRA adjustments.
- **Functions called**:
    - [`ggml_mul_mat_id`](../ggml/src/ggml.c.driver.md#ggml_mul_mat_id)
    - [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_norm<!-- {{#callable:llm_graph_context::build_norm}} -->
The `build_norm` function applies a specified normalization type to a tensor and optionally scales and biases it using provided weight and bias tensors.
- **Inputs**:
    - `cur`: A pointer to a `ggml_tensor` representing the current tensor to be normalized.
    - `mw`: A pointer to a `ggml_tensor` representing the weight tensor for scaling, which can be null.
    - `mb`: A pointer to a `ggml_tensor` representing the bias tensor for addition, which can be null.
    - `type`: An `llm_norm_type` enum value indicating the type of normalization to apply (e.g., LLM_NORM, LLM_NORM_RMS, LLM_NORM_GROUP).
    - `il`: An integer representing the layer index, used for callback purposes.
- **Control Flow**:
    - The function begins by switching on the `type` parameter to determine which normalization function to apply to `cur`.
    - If `type` is `LLM_NORM`, it applies [`ggml_norm`](../ggml/src/ggml.c.driver.md#ggml_norm) with a specified epsilon from `hparams`.
    - If `type` is `LLM_NORM_RMS`, it applies [`ggml_rms_norm`](../ggml/src/ggml.c.driver.md#ggml_rms_norm) with a specified epsilon from `hparams`.
    - If `type` is `LLM_NORM_GROUP`, it reshapes `cur` to 3D, applies [`ggml_group_norm`](../ggml/src/ggml.c.driver.md#ggml_group_norm), and then reshapes it back to 2D.
    - If either `mw` or `mb` is provided, it calls the callback function [`cb`](#llm_graph_contextcb) with the current tensor, "norm", and `il`.
    - If `mw` is provided, it multiplies `cur` by `mw` and calls the callback function [`cb`](#llm_graph_contextcb) with "norm_w" and `il` if `mb` is also provided.
    - If `mb` is provided, it adds `mb` to `cur`.
- **Output**: The function returns a pointer to the `ggml_tensor` that has been normalized and optionally scaled and biased.
- **Functions called**:
    - [`ggml_norm`](../ggml/src/ggml.c.driver.md#ggml_norm)
    - [`ggml_rms_norm`](../ggml/src/ggml.c.driver.md#ggml_rms_norm)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_group_norm`](../ggml/src/ggml.c.driver.md#ggml_group_norm)
    - [`ggml_reshape_2d`](../ggml/src/ggml.c.driver.md#ggml_reshape_2d)
    - [`llm_graph_context::cb`](#llm_graph_contextcb)
    - [`ggml_mul`](../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_ffn<!-- {{#callable:llm_graph_context::build_ffn}} -->
The `build_ffn` function constructs a feed-forward neural network (FFN) layer with optional gating and activation functions, applying various transformations and operations on input tensors based on specified types and parameters.
- **Inputs**:
    - `cur`: A pointer to a `ggml_tensor` representing the current tensor to be processed.
    - `up`: A pointer to a `ggml_tensor` used for the upward transformation matrix in the FFN.
    - `up_b`: A pointer to a `ggml_tensor` representing the bias for the upward transformation.
    - `up_s`: A pointer to a `ggml_tensor` representing the scaling factor for the upward transformation.
    - `gate`: A pointer to a `ggml_tensor` used for the gating mechanism in the FFN.
    - `gate_b`: A pointer to a `ggml_tensor` representing the bias for the gating mechanism.
    - `gate_s`: A pointer to a `ggml_tensor` representing the scaling factor for the gating mechanism.
    - `down`: A pointer to a `ggml_tensor` used for the downward transformation matrix in the FFN.
    - `down_b`: A pointer to a `ggml_tensor` representing the bias for the downward transformation.
    - `down_s`: A pointer to a `ggml_tensor` representing the scaling factor for the downward transformation.
    - `act_scales`: A pointer to a `ggml_tensor` used for scaling the activation output, applicable for certain activation types.
    - `type_op`: An `llm_ffn_op_type` enum indicating the type of activation function to apply (e.g., SILU, GELU, RELU).
    - `type_gate`: An `llm_ffn_gate_type` enum indicating the type of gating mechanism to use (e.g., sequential or parallel).
    - `il`: An integer representing the layer index, used for callback purposes.
- **Control Flow**:
    - Initialize `tmp` with the result of [`build_lora_mm`](#llm_graph_contextbuild_lora_mm) using `up` and `cur`, or just `cur` if `up` is null.
    - Apply bias `up_b` and scaling `up_s` to `tmp` if they are provided.
    - If `gate` is provided, apply the gating mechanism based on `type_gate`, and optionally add bias `gate_b` and scale `gate_s`.
    - Apply the specified activation function (`type_op`) to `cur`, with optional scaling using `act_scales` for GELU.
    - If `gate` is provided and `type_gate` is parallel, multiply `cur` with `tmp`.
    - Apply the downward transformation using `down`, with optional bias `down_b` and scaling `down_s`.
    - Return the final transformed tensor `cur`.
- **Output**: A pointer to a `ggml_tensor` representing the output of the feed-forward network after applying all specified transformations and operations.
- **Functions called**:
    - [`llm_graph_context::build_lora_mm`](#llm_graph_contextbuild_lora_mm)
    - [`llm_graph_context::cb`](#llm_graph_contextcb)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_mul`](../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`ggml_silu`](../ggml/src/ggml.c.driver.md#ggml_silu)
    - [`ggml_gelu`](../ggml/src/ggml.c.driver.md#ggml_gelu)
    - [`ggml_div`](../ggml/src/ggml.c.driver.md#ggml_div)
    - [`ggml_relu`](../ggml/src/ggml.c.driver.md#ggml_relu)
    - [`ggml_sqr`](../ggml/src/ggml.c.driver.md#ggml_sqr)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_element_size`](../ggml/src/ggml.c.driver.md#ggml_element_size)
    - [`ggml_mul_mat_set_prec`](../ggml/src/ggml.c.driver.md#ggml_mul_mat_set_prec)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_moe\_ffn<!-- {{#callable:llm_graph_context::build_moe_ffn}} -->
The `build_moe_ffn` function constructs a mixture of experts feed-forward network (MoE FFN) by selecting and weighting expert networks based on gating probabilities and applying transformations to the input tensor.
- **Inputs**:
    - `cur`: A pointer to a ggml_tensor representing the current input tensor to be processed.
    - `gate_inp`: A pointer to a ggml_tensor used as input for the gating mechanism.
    - `up_exps`: A pointer to a ggml_tensor representing the 'up' expert networks.
    - `gate_exps`: A pointer to a ggml_tensor representing the gating expert networks, if any.
    - `down_exps`: A pointer to a ggml_tensor representing the 'down' expert networks.
    - `exp_probs_b`: A pointer to a ggml_tensor representing the bias for expert selection probabilities, if any.
    - `n_expert`: An integer representing the total number of experts available.
    - `n_expert_used`: An integer representing the number of experts to be used in the computation.
    - `type_op`: An enum value of type llm_ffn_op_type indicating the type of activation function to apply (e.g., SILU, GELU).
    - `norm_w`: A boolean indicating whether to normalize the weights.
    - `scale_w`: A boolean indicating whether to scale the weights.
    - `w_scale`: A float representing the scaling factor for the weights.
    - `gating_op`: An enum value of type llama_expert_gating_func_type indicating the gating function to use (e.g., SOFTMAX, SIGMOID).
    - `il`: An integer representing the layer index for callback purposes.
- **Control Flow**:
    - Initialize embedding and token dimensions from the input tensor.
    - Determine if weights should be applied before the FFN based on architecture type.
    - Compute logits using the gating input and current tensor, then apply the specified gating function (softmax or sigmoid) to obtain probabilities.
    - Optionally add a bias to the probabilities for expert selection, unless the architecture is LLM_ARCH_LLAMA4.
    - Select the top experts based on the biased probabilities and compute the corresponding weights.
    - Normalize and/or scale the weights if specified by the input parameters.
    - Reshape the current tensor and apply weights before the FFN if required by the architecture.
    - Compute the 'up' transformation using the selected experts and apply the gating transformation if provided.
    - Apply the specified activation function (SILU or GELU) to the current tensor.
    - Compute the 'down' transformation using the selected experts and aggregate the results from all experts.
    - Return the final output tensor after ensuring it is contiguous if only one expert is used.
- **Output**: A pointer to a ggml_tensor representing the output of the mixture of experts feed-forward network.
- **Functions called**:
    - [`llm_graph_context::build_lora_mm`](#llm_graph_contextbuild_lora_mm)
    - [`llm_graph_context::cb`](#llm_graph_contextcb)
    - [`ggml_soft_max`](../ggml/src/ggml.c.driver.md#ggml_soft_max)
    - [`ggml_sigmoid`](../ggml/src/ggml.c.driver.md#ggml_sigmoid)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_top_k`](../ggml/src/ggml.c.driver.md#ggml_top_k)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_reshape_2d`](../ggml/src/ggml.c.driver.md#ggml_reshape_2d)
    - [`ggml_sum_rows`](../ggml/src/ggml.c.driver.md#ggml_sum_rows)
    - [`ggml_div`](../ggml/src/ggml.c.driver.md#ggml_div)
    - [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale)
    - [`ggml_repeat_4d`](../ggml/src/ggml.c.driver.md#ggml_repeat_4d)
    - [`ggml_mul`](../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`llm_graph_context::build_lora_mm_id`](#llm_graph_contextbuild_lora_mm_id)
    - [`ggml_silu`](../ggml/src/ggml.c.driver.md#ggml_silu)
    - [`ggml_gelu`](../ggml/src/ggml.c.driver.md#ggml_gelu)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_inp\_embd<!-- {{#callable:llm_graph_context::build_inp_embd}} -->
The `build_inp_embd` function constructs an input embedding tensor, optionally applying LoRA (Low-Rank Adaptation) adjustments, and returns the resulting tensor.
- **Inputs**:
    - `tok_embd`: A pointer to a `ggml_tensor` representing the token embeddings.
- **Control Flow**:
    - Retrieve the number of embeddings from `hparams.n_embd`.
    - Create a unique pointer to an `llm_graph_input_embd` object.
    - Initialize a `ggml_tensor` pointer `cur` to `nullptr`.
    - Check if `ubatch.token` is true; if so, create a new 1D tensor for tokens and set it as input.
    - Retrieve rows from `tok_embd` using the token tensor and assign to `cur`.
    - Iterate over `loras` to apply LoRA adjustments to the embeddings if applicable.
    - If `ubatch.token` is false, create a new 2D tensor for embeddings and set it as input, assigning it to `cur`.
    - If `hparams.f_embedding_scale` is non-zero, scale `cur` by this factor.
    - Invoke the callback function [`cb`](#llm_graph_contextcb) with `cur` and the name "inp_embd".
    - Add the input embedding to the result object `res`.
- **Output**: A pointer to a `ggml_tensor` representing the constructed input embedding tensor, potentially with LoRA adjustments and scaling applied.
- **Functions called**:
    - [`ggml_new_tensor_1d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale)
    - [`ggml_mul_mat`](../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_new_tensor_2d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d)
    - [`llm_graph_context::cb`](#llm_graph_contextcb)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_inp\_pos<!-- {{#callable:llm_graph_context::build_inp_pos}} -->
The `build_inp_pos` function creates and initializes a 1D tensor for positional input in a graph context, setting it as an input and adding it to the result inputs.
- **Inputs**: None
- **Control Flow**:
    - Create a unique pointer `inp` of type `llm_graph_input_pos` initialized with `n_pos_per_embd()`.
    - Reference `cur` to `inp->pos`.
    - Initialize `cur` as a new 1D tensor of type `GGML_TYPE_I32` with size `n_tokens * n_pos_per_embd()` using [`ggml_new_tensor_1d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_1d).
    - Set `cur` as an input using [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input).
    - Add `inp` to the result inputs using `res->add_input`.
    - Return the tensor `cur`.
- **Output**: Returns a pointer to a `ggml_tensor` representing the positional input tensor.
- **Functions called**:
    - [`llm_graph_context::n_pos_per_embd`](#llm_graph_contextn_pos_per_embd)
    - [`ggml_new_tensor_1d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_inp\_attn\_scale<!-- {{#callable:llm_graph_context::build_inp_attn_scale}} -->
The `build_inp_attn_scale` function creates and returns a 3D tensor for attention scaling, configured for broadcasting, and adds it as an input to the result object.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - Create a unique pointer to `llm_graph_input_attn_temp` using `hparams.n_attn_temp_floor_scale` and `hparams.f_attn_temp_scale`.
    - Assign a new 3D tensor to `cur` with dimensions 1x1xN, where N is `n_tokens`, using [`ggml_new_tensor_3d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_3d).
    - Set the tensor as an input using [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input).
    - Add the created input to the result object `res` using `add_input`.
    - Return the tensor `cur`.
- **Output**: Returns a pointer to a `ggml_tensor` object representing the attention scale tensor.
- **Functions called**:
    - [`ggml_new_tensor_3d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_3d)
    - [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_inp\_out\_ids<!-- {{#callable:llm_graph_context::build_inp_out_ids}} -->
The `build_inp_out_ids` function creates and returns a new 1D tensor for output IDs, setting it as an input in the graph context.
- **Inputs**: None
- **Control Flow**:
    - A unique pointer to `llm_graph_input_out_ids` is created with `hparams`, `cparams`, and `n_outputs` as arguments.
    - A reference to `cur` is obtained from `inp->out_ids`.
    - A new 1D tensor of type `GGML_TYPE_I32` with size `n_outputs` is created and assigned to `cur`.
    - The tensor `cur` is set as an input using [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input).
    - The input object `inp` is added to the result's input list using `res->add_input`.
    - The function returns the tensor `cur`.
- **Output**: A pointer to a `ggml_tensor` representing the newly created 1D tensor for output IDs.
- **Functions called**:
    - [`ggml_new_tensor_1d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_inp\_mean<!-- {{#callable:llm_graph_context::build_inp_mean}} -->
The `build_inp_mean` function creates and returns a 2D tensor for input mean pooling in a neural network graph context.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - A unique pointer to `llm_graph_input_mean` is created using `cparams`.
    - A reference to the `mean` tensor of the `inp` object is obtained.
    - A new 2D tensor of type `GGML_TYPE_F32` with dimensions `n_tokens` x `n_tokens` is created and assigned to `cur`.
    - The `cur` tensor is set as an input tensor using [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input).
    - The `inp` object is added to the `res` result object as an input.
    - The function returns the `cur` tensor.
- **Output**: The function returns a pointer to a `ggml_tensor` representing the mean input tensor.
- **Functions called**:
    - [`ggml_new_tensor_2d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d)
    - [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_inp\_cls<!-- {{#callable:llm_graph_context::build_inp_cls}} -->
The `build_inp_cls` function creates and initializes a classification input tensor for a graph context, setting it as an input and returning the tensor.
- **Inputs**: None
- **Control Flow**:
    - Create a unique pointer `inp` of type `llm_graph_input_cls` initialized with `cparams`.
    - Reference `cur` to `inp->cls`.
    - Create a new 1D tensor `cur` of type `GGML_TYPE_I32` with size `n_tokens` using [`ggml_new_tensor_1d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_1d).
    - Set `cur` as an input using [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input).
    - Add `inp` to the result's input list using `res->add_input`.
    - Return the tensor `cur`.
- **Output**: Returns a pointer to a `ggml_tensor` representing the classification input tensor.
- **Functions called**:
    - [`ggml_new_tensor_1d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_inp\_s\_copy<!-- {{#callable:llm_graph_context::build_inp_s_copy}} -->
The `build_inp_s_copy` function creates a new 1D tensor for storing a copy of the state from a recurrent key-value cache and adds it as an input to the graph result.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - Retrieve the `kv_state` from the member variable `mstate` and cast it to `llama_kv_cache_recurrent_state`.
    - Create a unique pointer `inp` of type `llm_graph_input_s_copy` initialized with `kv_state`.
    - Get the number of key-value pairs `n_kv` from `kv_state`.
    - Create a new 1D tensor `cur` of type `GGML_TYPE_I32` with size `n_kv` using the context `ctx0`.
    - Set the tensor `cur` as an input using [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input).
    - Add the input `inp` to the graph result `res` using `add_input`.
    - Return the tensor `cur`.
- **Output**: A pointer to a `ggml_tensor` representing the newly created 1D tensor for the state copy.
- **Functions called**:
    - [`ggml_new_tensor_1d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_inp\_s\_mask<!-- {{#callable:llm_graph_context::build_inp_s_mask}} -->
The `build_inp_s_mask` function creates and returns a new 2D tensor for the input state mask based on the number of key-value pairs in the recurrent state.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - Retrieve the `kv_state` from the member variable `mstate` and cast it to `llama_kv_cache_recurrent_state`.
    - Create a unique pointer `inp` of type `llm_graph_input_s_mask` initialized with `kv_state`.
    - Get the number of key-value pairs `n_kv` from `kv_state`.
    - Create a new 2D tensor `cur` of type `GGML_TYPE_F32` with dimensions 1 by `n_kv` using [`ggml_new_tensor_2d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d).
    - Set `cur` as an input tensor using [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input).
    - Add the `inp` object to the result's input list using `res->add_input`.
    - Return the tensor `cur`.
- **Output**: A pointer to a `ggml_tensor` representing the input state mask.
- **Functions called**:
    - [`ggml_new_tensor_2d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d)
    - [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_inp\_cross\_embd<!-- {{#callable:llm_graph_context::build_inp_cross_embd}} -->
The `build_inp_cross_embd` function constructs a cross-embedding input tensor for a graph context, using either pre-existing embeddings or creating a new tensor based on the context's parameters.
- **Inputs**:
    - `None`: This function does not take any direct input arguments.
- **Control Flow**:
    - Create a unique pointer `inp` of type `llm_graph_input_cross_embd` initialized with `cross`.
    - Reference `cur` to `inp->cross_embd`.
    - Check if `cross->t_embd` is available (commented out in the current code).
    - Determine `n_embd` and `n_enc` based on whether `cross->v_embd` is empty or not.
    - Create a new 2D tensor `cur` with dimensions `n_embd` by `n_enc` using [`ggml_new_tensor_2d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d).
    - Set `cur` as an input tensor using [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input).
    - Add `inp` to the result's input list using `res->add_input`.
    - Return the tensor `cur`.
- **Output**: Returns a pointer to a `ggml_tensor` representing the cross-embedding input tensor.
- **Functions called**:
    - [`ggml_new_tensor_2d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d)
    - [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_inp\_pos\_bucket\_enc<!-- {{#callable:llm_graph_context::build_inp_pos_bucket_enc}} -->
The `build_inp_pos_bucket_enc` function creates and returns a 2D tensor for position bucket encoding in a graph context.
- **Inputs**: None
- **Control Flow**:
    - Create a unique pointer `inp` of type `llm_graph_input_pos_bucket` initialized with `hparams`.
    - Access the `pos_bucket` member of `inp` and assign it a new 2D tensor of type `GGML_TYPE_I32` with dimensions `n_tokens` x `n_tokens` using [`ggml_new_tensor_2d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d).
    - Set the created tensor as an input using [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input).
    - Add the `inp` object to the `res` result object using `add_input`.
    - Return the created tensor `cur`.
- **Output**: Returns a pointer to a `ggml_tensor` representing the position bucket encoding tensor.
- **Functions called**:
    - [`ggml_new_tensor_2d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d)
    - [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_inp\_pos\_bucket\_dec<!-- {{#callable:llm_graph_context::build_inp_pos_bucket_dec}} -->
The `build_inp_pos_bucket_dec` function creates and returns a 2D tensor for position buckets in a decoder context, using the number of key-value pairs and tokens.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - Retrieve the `kv_state` from the member variable `mstate` and cast it to `llama_kv_cache_unified_state`.
    - Create a unique pointer `inp` of type `llm_graph_input_pos_bucket_kv` initialized with `hparams` and `kv_state`.
    - Get the number of key-value pairs `n_kv` from `kv_state`.
    - Create a 2D tensor `cur` of type `GGML_TYPE_I32` with dimensions `n_kv` by `n_tokens` using [`ggml_new_tensor_2d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d).
    - Set `cur` as an input tensor using [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input).
    - Add `inp` to the result's input list using `res->add_input`.
    - Return the tensor `cur`.
- **Output**: A pointer to a `ggml_tensor` representing the position bucket tensor for the decoder.
- **Functions called**:
    - [`ggml_new_tensor_2d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d)
    - [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_pos\_bias<!-- {{#callable:llm_graph_context::build_pos_bias}} -->
The `build_pos_bias` function constructs a positional bias tensor by reshaping and permuting input tensors based on positional bucket indices and attention relation biases.
- **Inputs**:
    - `pos_bucket`: A `ggml_tensor` representing the positional bucket indices, with dimensions indicating the number of elements in each dimension.
    - `attn_rel_b`: A `ggml_tensor` representing the attention relation biases, used to extract rows corresponding to the positional bucket indices.
- **Control Flow**:
    - Reshape `pos_bucket` into a 1D tensor `pos_bucket_1d` by multiplying its dimensions.
    - Log the reshaped `pos_bucket_1d` using the callback function [`cb`](#llm_graph_contextcb).
    - Extract rows from `attn_rel_b` using `pos_bucket_1d` to create `pos_bias`.
    - Reshape `pos_bias` into a 3D tensor with dimensions derived from `pos_bias` and `pos_bucket`.
    - Permute the dimensions of `pos_bias` to reorder them as specified.
    - Ensure `pos_bias` is contiguous in memory.
    - Log the final `pos_bias` using the callback function [`cb`](#llm_graph_contextcb).
- **Output**: A `ggml_tensor` representing the final positional bias, reshaped and permuted based on the input tensors.
- **Functions called**:
    - [`ggml_reshape_1d`](../ggml/src/ggml.c.driver.md#ggml_reshape_1d)
    - [`llm_graph_context::cb`](#llm_graph_contextcb)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_permute`](../ggml/src/ggml.c.driver.md#ggml_permute)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_attn\_mha<!-- {{#callable:llm_graph_context::build_attn_mha}} -->
The `build_attn_mha` function constructs a multi-head attention (MHA) mechanism using given query, key, and value tensors, along with optional bias, mask, and scaling parameters, and returns the resulting attention tensor.
- **Inputs**:
    - `gf`: A pointer to a `ggml_cgraph` object representing the computation graph.
    - `q`: A pointer to a `ggml_tensor` representing the query tensor.
    - `k`: A pointer to a `ggml_tensor` representing the key tensor.
    - `v`: A pointer to a `ggml_tensor` representing the value tensor.
    - `kq_b`: A pointer to a `ggml_tensor` representing the optional bias for the key-query product, can be null.
    - `kq_mask`: A pointer to a `ggml_tensor` representing the mask to be applied to the key-query product.
    - `v_mla`: A pointer to a `ggml_tensor` representing the optional matrix for linear attention, can be null.
    - `kq_scale`: A float representing the scaling factor for the key-query product.
- **Control Flow**:
    - Check if the value tensor `v` is transposed by comparing its strides.
    - Permute the dimensions of the query, key, and value tensors for compatibility with the attention mechanism.
    - Determine the number of tokens, heads, and key-value pairs from the dimensions of the query and key tensors.
    - If flash attention is enabled and conditions are met, use the flash attention mechanism with optional casting to half precision for key and value tensors.
    - If `v_mla` is provided, apply it using matrix multiplication with permutations to optimize the operation.
    - If flash attention is not used, compute the key-query product, apply precision settings, and optionally apply transformations based on architecture-specific conditions.
    - Apply softmax to the key-query product, optionally add bias, and compute the final attention output by multiplying with the value tensor.
    - If `v_mla` is provided, apply it to the attention output.
    - Permute and reshape the final attention output tensor for the desired format.
    - Expand the computation graph with the final attention output tensor.
- **Output**: A pointer to a `ggml_tensor` representing the final attention output tensor.
- **Functions called**:
    - [`ggml_permute`](../ggml/src/ggml.c.driver.md#ggml_permute)
    - [`ggml_transpose`](../ggml/src/ggml.c.driver.md#ggml_transpose)
    - [`ggml_cast`](../ggml/src/ggml.c.driver.md#ggml_cast)
    - [`ggml_flash_attn_ext`](../ggml/src/ggml.c.driver.md#ggml_flash_attn_ext)
    - [`ggml_flash_attn_ext_set_prec`](../ggml/src/ggml.c.driver.md#ggml_flash_attn_ext_set_prec)
    - [`ggml_reshape_4d`](../ggml/src/ggml.c.driver.md#ggml_reshape_4d)
    - [`ggml_mul_mat`](../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_reshape_2d`](../ggml/src/ggml.c.driver.md#ggml_reshape_2d)
    - [`ggml_mul_mat_set_prec`](../ggml/src/ggml.c.driver.md#ggml_mul_mat_set_prec)
    - [`ggml_tanh`](../ggml/src/ggml.c.driver.md#ggml_tanh)
    - [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_soft_max_ext`](../ggml/src/ggml.c.driver.md#ggml_soft_max_ext)
    - [`ggml_cont_2d`](../ggml/src/ggml.c.driver.md#ggml_cont_2d)
    - [`ggml_backend_sched_set_tensor_backend`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_set_tensor_backend)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_attn\_inp\_no\_cache<!-- {{#callable:llm_graph_context::build_attn_inp_no_cache}} -->
The `build_attn_inp_no_cache` function constructs an attention input object without a key-value cache for a graph context.
- **Inputs**: None
- **Control Flow**:
    - Create a unique pointer to an `llm_graph_input_attn_no_cache` object using `hparams` and `cparams`.
    - Initialize `kq_mask` as a 2D tensor with dimensions based on `n_tokens` and a padded size using `GGML_KQ_MASK_PAD`.
    - Set `kq_mask` as an input tensor using [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input).
    - Determine `kq_mask_cnv` based on whether `flash_attn` is enabled, casting `kq_mask` to `GGML_TYPE_F16` if true, otherwise using `kq_mask` directly.
    - Add the constructed input object to the result and return it.
- **Output**: Returns a pointer to an `llm_graph_input_attn_no_cache` object, which is added to the graph context's result.
- **Functions called**:
    - [`ggml_new_tensor_2d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d)
    - [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input)
    - [`ggml_cast`](../ggml/src/ggml.c.driver.md#ggml_cast)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_attn<!-- {{#callable:llm_graph_context::build_attn}} -->
The `build_attn` function constructs an attention mechanism within a computational graph, applying multi-head attention and optional linear transformations to input tensors.
- **Inputs**:
    - `inp`: A pointer to an `llm_graph_input_attn_no_cache` object, which provides the key-query mask for the attention mechanism.
    - `gf`: A pointer to a `ggml_cgraph` object representing the computational graph to which the attention nodes are added.
    - `wo`: A pointer to a `ggml_tensor` representing the weight matrix for the output linear transformation, or null if not used.
    - `wo_b`: A pointer to a `ggml_tensor` representing the bias for the output linear transformation, or null if not used.
    - `q_cur`: A pointer to a `ggml_tensor` representing the current query tensor.
    - `k_cur`: A pointer to a `ggml_tensor` representing the current key tensor.
    - `v_cur`: A pointer to a `ggml_tensor` representing the current value tensor.
    - `kq_b`: A pointer to a `ggml_tensor` representing the key-query bias, or null if not used.
    - `v_mla`: A pointer to a `ggml_tensor` used for multi-linear attention, or null if not used.
    - `kq_scale`: A float representing the scaling factor for the key-query product.
    - `il`: An integer representing the layer index or identifier for logging or debugging purposes.
- **Control Flow**:
    - The function begins by expanding the forward graph with the current query, key, and value tensors to ensure they are processed together.
    - It retrieves the key-query mask from the input object `inp`.
    - The function calls [`build_attn_mha`](#llm_graph_contextbuild_attn_mha) to perform multi-head attention using the query, key, value tensors, and other parameters, returning the result tensor `cur`.
    - If the weight matrix `wo` is provided, it applies a linear transformation to `cur` using [`build_lora_mm`](#llm_graph_contextbuild_lora_mm).
    - If the bias `wo_b` is provided, it adds this bias to `cur`.
    - Finally, the function returns the modified tensor `cur` as the output.
- **Output**: A pointer to a `ggml_tensor` representing the result of the attention mechanism, potentially transformed by additional linear operations.
- **Functions called**:
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
    - [`llm_graph_context::build_attn_mha`](#llm_graph_contextbuild_attn_mha)
    - [`llm_graph_context::cb`](#llm_graph_contextcb)
    - [`llm_graph_context::build_lora_mm`](#llm_graph_contextbuild_lora_mm)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_attn\_inp\_kv\_unified<!-- {{#callable:llm_graph_context::build_attn_inp_kv_unified}} -->
The `build_attn_inp_kv_unified` function constructs and returns a `llm_graph_input_attn_kv_unified` object, which is used for attention input with a unified key-value cache in a graph context.
- **Inputs**: None
- **Control Flow**:
    - Retrieve the `kv_state` from the member variable `mstate` and cast it to `llama_kv_cache_unified_state`.
    - Create a new `llm_graph_input_attn_kv_unified` object using `hparams`, `cparams`, and `kv_state`.
    - Assert that `hparams.swa_type` is `LLAMA_SWA_TYPE_NONE` to ensure the correct cache type is used.
    - Retrieve the number of key-value pairs (`n_kv`) from `kv_state`.
    - Create a new 2D tensor `self_kq_mask` with dimensions `n_kv` by `GGML_PAD(n_tokens, GGML_KQ_MASK_PAD)` and set it as an input.
    - If `cparams.flash_attn` is true, cast `self_kq_mask` to `GGML_TYPE_F16` and assign it to `self_kq_mask_cnv`; otherwise, assign `self_kq_mask` directly to `self_kq_mask_cnv`.
    - Add the constructed `llm_graph_input_attn_kv_unified` object to the result's input list and return it.
- **Output**: A pointer to a `llm_graph_input_attn_kv_unified` object, which is added to the graph's input list.
- **Functions called**:
    - [`ggml_new_tensor_2d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d)
    - [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input)
    - [`ggml_cast`](../ggml/src/ggml.c.driver.md#ggml_cast)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_attn\_inp\_kv\_unified\_iswa<!-- {{#callable:llm_graph_context::build_attn_inp_kv_unified_iswa}} -->
The `build_attn_inp_kv_unified_iswa` function constructs and returns an attention input object for a unified ISWA key-value cache, setting up necessary masks and converting them based on configuration parameters.
- **Inputs**:
    - `None`: This function does not take any explicit input arguments, but it uses member variables from the `llm_graph_context` class.
- **Control Flow**:
    - Retrieve the `llama_kv_cache_unified_iswa_state` from the member variable `mstate` and cast it appropriately.
    - Create a new `llm_graph_input_attn_kv_unified_iswa` object using `hparams`, `cparams`, and the retrieved `kv_state`.
    - Retrieve the number of key-value pairs (`n_kv`) from the base state of `kv_state`.
    - Create a new 2D tensor `self_kq_mask` with dimensions based on `n_kv` and padded `n_tokens`, and set it as an input.
    - Convert `self_kq_mask` to half-precision if `flash_attn` is enabled, storing the result in `self_kq_mask_cnv`.
    - Assert that the `swa_type` is not `LLAMA_SWA_TYPE_NONE`, indicating that SWA is being used.
    - Retrieve the number of key-value pairs (`n_kv`) from the SWA state of `kv_state`.
    - Create a new 2D tensor `self_kq_mask_swa` with dimensions based on `n_kv` and padded `n_tokens`, and set it as an input.
    - Convert `self_kq_mask_swa` to half-precision if `flash_attn` is enabled, storing the result in `self_kq_mask_swa_cnv`.
    - Add the constructed input object to the result set and return it.
- **Output**: The function returns a pointer to a `llm_graph_input_attn_kv_unified_iswa` object, which is added to the result set of the `llm_graph_context`.
- **Functions called**:
    - [`ggml_new_tensor_2d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d)
    - [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input)
    - [`ggml_cast`](../ggml/src/ggml.c.driver.md#ggml_cast)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_attn\_inp\_cross<!-- {{#callable:llm_graph_context::build_attn_inp_cross}} -->
The `build_attn_inp_cross` function constructs and returns a `llm_graph_input_attn_cross` object, initializing its cross-attention key-query mask tensor and optionally converting it to a different data type based on configuration parameters.
- **Inputs**: None
- **Control Flow**:
    - Create a unique pointer to a `llm_graph_input_attn_cross` object using the `cross` member of the `llm_graph_context` class.
    - Determine the number of encoder tokens (`n_enc`) based on whether the `v_embd` vector in `cross` is empty or not.
    - Create a new 2D tensor for the cross-attention key-query mask with dimensions `n_enc` by a padded version of `n_tokens`, using the `ctx0` context and `GGML_TYPE_F32` data type.
    - Set the created tensor as an input tensor using [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input).
    - Check if `flash_attn` is enabled in `cparams`; if so, cast the cross-attention key-query mask tensor to `GGML_TYPE_F16`, otherwise keep it as is.
    - Add the constructed `llm_graph_input_attn_cross` object to the result's input list and return it.
- **Output**: A pointer to the newly created `llm_graph_input_attn_cross` object, which is added to the result's input list.
- **Functions called**:
    - [`ggml_new_tensor_2d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d)
    - [`ggml_set_input`](../ggml/src/ggml.c.driver.md#ggml_set_input)
    - [`ggml_cast`](../ggml/src/ggml.c.driver.md#ggml_cast)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_copy\_mask\_state<!-- {{#callable:llm_graph_context::build_copy_mask_state}} -->
The `build_copy_mask_state` function manages the copying and masking of state tensors for a recurrent neural network model, ensuring that only relevant states are retained and modified.
- **Inputs**:
    - `gf`: A pointer to a `ggml_cgraph` object representing the computation graph.
    - `s`: A pointer to a `ggml_tensor` object representing the current state tensor.
    - `state_copy`: A pointer to a `ggml_tensor` object used to specify which rows of the state tensor should be copied.
    - `state_mask`: A pointer to a `ggml_tensor` object used to mask certain states, effectively clearing them.
    - `n_state`: An integer representing the number of states.
    - `n_seqs`: An integer representing the number of sequences.
- **Control Flow**:
    - Retrieve the number of key-value pairs (`n_kv`) and the head index (`kv_head`) from the recurrent state cache.
    - Reshape the input state tensor `s` into a 2D tensor with dimensions `n_state` by the size of the key-value state.
    - Copy the relevant rows from the reshaped state tensor using `state_copy`, assuming the copy destinations are within the range of `kv_head` to `kv_head + n_kv`.
    - Apply the `state_mask` to clear states of sequences that are starting at the beginning of the batch.
    - Copy the states that will not be changed further, specifically those between `n_seqs` and `n_kv`, into the computation graph `gf`.
    - Return a 2D view of the modified states tensor, focusing on the part that will be used and modified, with dimensions `n_state` by `n_seqs`.
- **Output**: A pointer to a `ggml_tensor` object representing the modified view of the states tensor, ready for further processing.
- **Functions called**:
    - [`ggml_reshape_2d`](../ggml/src/ggml.c.driver.md#ggml_reshape_2d)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_mul`](../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
    - [`ggml_cpy`](../ggml/src/ggml.c.driver.md#ggml_cpy)
    - [`ggml_view_1d`](../ggml/src/ggml.c.driver.md#ggml_view_1d)
    - [`ggml_element_size`](../ggml/src/ggml.c.driver.md#ggml_element_size)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_rwkv\_token\_shift\_load<!-- {{#callable:llm_graph_context::build_rwkv_token_shift_load}} -->
The `build_rwkv_token_shift_load` function constructs a token shift tensor by copying and reshaping state data for a given layer index in a recurrent neural network context.
- **Inputs**:
    - `gf`: A pointer to a `ggml_cgraph` object representing the computation graph.
    - `state_copy`: A pointer to a `ggml_tensor` object representing the state copy tensor.
    - `state_mask`: A pointer to a `ggml_tensor` object representing the state mask tensor.
    - `ubatch`: A constant reference to a `llama_ubatch` object containing batch information, including the number of sequences.
    - `il`: An integer representing the layer index for which the token shift is being constructed.
- **Control Flow**:
    - Retrieve the recurrent state from the member variable `mstate` and cast it to `llama_kv_cache_recurrent_state`.
    - Obtain the `token_shift_count` from the hyperparameters `hparams`.
    - Get the number of sequences `n_seqs` from the `ubatch` object.
    - Retrieve the `token_shift_all` tensor for the specified layer index `il` from the recurrent state.
    - Call [`build_copy_mask_state`](#llm_graph_contextbuild_copy_mask_state) to create a `token_shift` tensor by copying and masking the `token_shift_all` tensor using `state_copy` and `state_mask`.
    - Reshape the `token_shift` tensor to a 3D shape with dimensions `(n_embd, token_shift_count, n_seqs)` using the context `ctx0`.
    - Return the reshaped `token_shift` tensor.
- **Output**: A pointer to a `ggml_tensor` object representing the reshaped token shift tensor.
- **Functions called**:
    - [`llm_graph_context::build_copy_mask_state`](#llm_graph_contextbuild_copy_mask_state)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_rwkv\_token\_shift\_store<!-- {{#callable:llm_graph_context::build_rwkv_token_shift_store}} -->
The `build_rwkv_token_shift_store` function copies a 1D view of a token shift tensor into a specific location within a key-value cache tensor for a given layer index.
- **Inputs**:
    - `token_shift`: A `ggml_tensor` pointer representing the token shift tensor to be stored.
    - `ubatch`: A constant reference to a `llama_ubatch` object containing batch information, including the number of sequences.
    - `il`: An integer representing the layer index for which the token shift is being stored.
- **Control Flow**:
    - Retrieve the key-value state from the member variable `mstate` and cast it to `llama_kv_cache_recurrent_state` type.
    - Extract the `token_shift_count` and `n_embd` from the hyperparameters (`hparams`).
    - Determine the number of sequences (`n_seqs`) from the `ubatch` parameter.
    - Get the current head index of the key-value state using `get_head()`.
    - Create a 1D view of the `token_shift` tensor with dimensions `n_embd * n_seqs * token_shift_count`.
    - Create a 1D view of the key-value cache tensor for the specified layer index `il`, with dimensions `hparams.n_embd_k_s() * n_seqs` and an offset based on the current head index and element size.
    - Copy the `token_shift` view into the key-value cache view using `ggml_cpy()`.
- **Output**: Returns a `ggml_tensor` pointer representing the result of the copy operation.
- **Functions called**:
    - [`ggml_cpy`](../ggml/src/ggml.c.driver.md#ggml_cpy)
    - [`ggml_view_1d`](../ggml/src/ggml.c.driver.md#ggml_view_1d)
    - [`ggml_element_size`](../ggml/src/ggml.c.driver.md#ggml_element_size)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)


---
#### llm\_graph\_context::build\_pooling<!-- {{#callable:llm_graph_context::build_pooling}} -->
The `build_pooling` function constructs a pooling operation on a graph based on the specified pooling type and updates the graph with the resulting tensor.
- **Inputs**:
    - `gf`: A pointer to a `ggml_cgraph` object representing the computation graph.
    - `cls`: A pointer to a `ggml_tensor` object used for classification, can be null.
    - `cls_b`: A pointer to a `ggml_tensor` object representing the bias for classification, can be null.
    - `cls_out`: A pointer to a `ggml_tensor` object representing the output classification tensor, can be null.
    - `cls_out_b`: A pointer to a `ggml_tensor` object representing the bias for the output classification tensor, can be null.
- **Control Flow**:
    - Check if embeddings are enabled in `cparams`; if not, return immediately.
    - Retrieve the input tensor `inp` from the result object `res`.
    - Assert that `inp` is not null, ensuring the presence of a required tensor.
    - Determine the pooling operation based on the `pooling_type` enum value.
    - For `LLAMA_POOLING_TYPE_NONE`, set `cur` to `inp`.
    - For `LLAMA_POOLING_TYPE_MEAN`, compute the mean pooling using [`build_inp_mean`](#llm_graph_contextbuild_inp_mean) and matrix multiplication.
    - For `LLAMA_POOLING_TYPE_CLS` and `LLAMA_POOLING_TYPE_LAST`, use [`build_inp_cls`](#llm_graph_contextbuild_inp_cls) to select specific rows from `inp`.
    - For `LLAMA_POOLING_TYPE_RANK`, perform additional operations involving `cls`, `cls_b`, `cls_out`, and `cls_out_b` for classification, with assertions to ensure required tensors are not null.
    - Abort if an unknown pooling type is encountered.
    - Invoke the callback function [`cb`](#llm_graph_contextcb) with the resulting tensor `cur`.
    - Update the result object `res` with the pooled tensor `cur`.
    - Expand the computation graph `gf` with the new tensor `cur`.
- **Output**: The function outputs a pooled tensor `cur` which is added to the computation graph and stored in the result object `res`.
- **Functions called**:
    - [`llm_graph_context::build_inp_mean`](#llm_graph_contextbuild_inp_mean)
    - [`ggml_mul_mat`](../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_transpose`](../ggml/src/ggml.c.driver.md#ggml_transpose)
    - [`llm_graph_context::build_inp_cls`](#llm_graph_contextbuild_inp_cls)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_tanh`](../ggml/src/ggml.c.driver.md#ggml_tanh)
    - [`llm_graph_context::cb`](#llm_graph_contextcb)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)  (Data Structure)



# Functions

---
### llama\_relative\_position\_bucket<!-- {{#callable:llama_relative_position_bucket}} -->
The function `llama_relative_position_bucket` calculates a relative position bucket index for two positions, considering the number of buckets and whether the calculation is bidirectional.
- **Inputs**:
    - `x`: The first position, of type `llama_pos`, representing a point in a sequence.
    - `y`: The second position, of type `llama_pos`, representing another point in a sequence.
    - `n_buckets`: A `uint64_t` representing the total number of buckets available for categorizing relative positions.
    - `bidirectional`: A `bool` indicating whether the relative position calculation should consider both directions (positive and negative).
- **Control Flow**:
    - Initialize `max_distance` to 128, which is a constant used for scaling large distances.
    - If `bidirectional` is true, halve the number of buckets (`n_buckets`).
    - Calculate `max_exact` as half of `n_buckets`.
    - Compute the `relative_position` as the difference between `x` and `y`.
    - Initialize `relative_bucket` to 0.
    - If `bidirectional` is true, adjust `relative_bucket` and take the absolute value of `relative_position`.
    - If not `bidirectional`, ensure `relative_position` is non-negative by taking the minimum with 0 and negating it.
    - Calculate `relative_position_if_large` using a logarithmic scale for large distances, ensuring it does not exceed `n_buckets - 1`.
    - Add either `relative_position` or `relative_position_if_large` to `relative_bucket` based on whether `relative_position` is less than `max_exact`.
    - Return the computed `relative_bucket`.
- **Output**: The function returns an `int32_t` representing the bucket index for the relative position between `x` and `y`.


