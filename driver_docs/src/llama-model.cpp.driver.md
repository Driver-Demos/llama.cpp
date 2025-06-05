# Purpose
The provided C++ code is a comprehensive component of a larger machine learning framework, specifically tailored for managing and configuring various types of language models, such as LLaMA, GPT, and BERT. It is not a standalone executable but a library designed to be integrated into broader software systems, facilitating the construction and management of neural network models for natural language processing tasks. The code supports a wide range of model architectures and configurations, offering functionality for loading, configuring, and operating models, including the setup of layers like attention mechanisms and feed-forward networks common in transformer-based architectures. It includes classes and utility functions for handling model parameters, memory management, and ensuring compatibility with different hardware backends, such as CPUs and GPUs, making it flexible and portable across computing environments. Overall, this code serves as a backend system that supports higher-level API calls, providing developers with the tools necessary to build, customize, and deploy language models for training and inference in diverse applications.
# Imports and Dependencies

---
- `llama-model.h`
- `llama-impl.h`
- `llama-mmap.h`
- `llama-batch.h`
- `llama-cparams.h`
- `llama-model-loader.h`
- `llama-kv-cache-unified.h`
- `llama-kv-cache-unified-iswa.h`
- `llama-kv-cache-recurrent.h`
- `ggml-cpp.h`
- `algorithm`
- `cassert`
- `cmath`
- `cfloat`
- `cstring`
- `functional`
- `map`
- `regex`
- `sstream`
- `stdexcept`


# Global Variables

---
### LLAMA\_ROPE\_SCALING\_TYPES
- **Type**: `std::map<llama_rope_scaling_type, const char *>`
- **Description**: `LLAMA_ROPE_SCALING_TYPES` is a static constant map that associates `llama_rope_scaling_type` enumeration values with their corresponding string representations. This map is used to provide a human-readable description for each type of rope scaling defined in the `llama_rope_scaling_type` enumeration.
- **Use**: This variable is used to convert `llama_rope_scaling_type` enum values to their string equivalents for display or logging purposes.


# Data Structures

---
### layer\_dev<!-- {{#data_structure:llama_model::impl::layer_dev}} -->
- **Type**: `struct`
- **Members**:
    - `dev`: An instance of `ggml_backend_dev_t` representing the device.
    - `buft_list`: A pointer to a `buft_list_t` structure that holds a list of buffers.
- **Description**: The `layer_dev` structure is designed to encapsulate information about a device in a machine learning context, containing a backend device representation and a list of associated buffers.


---
### llm\_build\_llama<!-- {{#data_structure:llm_build_llama}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the model parameters used for building the Llama architecture.
    - `params`: Holds the graph parameters necessary for the Llama model.
    - `gf`: Pointer to the graph context used for building the computation graph.
    - `res`: Stores the results of the forward pass through the model.
- **Description**: The `llm_build_llama` structure is designed to construct a Llama model by utilizing various components such as input embeddings, attention mechanisms, and feed-forward networks, while managing the necessary parameters and tensors throughout the layers of the model.
- **Member Functions**:
    - [`llm_build_llama::llm_build_llama`](#llm_build_llamallm_build_llama)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_llama::llm\_build\_llama<!-- {{#callable:llm_build_llama::llm_build_llama}} -->
Constructs a LLaMA model graph by processing input embeddings through multiple layers of attention and feed-forward networks.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and weights.
    - `params`: A constant reference to `llm_graph_params` that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph for the model.
- **Control Flow**:
    - Initializes the number of embedding heads and asserts consistency in model parameters.
    - Builds input embeddings and positional encodings.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - For each layer, computes query (Q), key (K), and value (V) tensors, applying necessary transformations and RoPE (Rotary Positional Encoding).
    - Handles both standard and Mixture of Experts (MoE) configurations for feed-forward networks.
    - Adds residual connections and prepares the output for the next layer.
    - After processing all layers, applies final normalization and computes the output logits.
- **Output**: The function does not return a value directly but populates the `res` structure with the final embedding tensor and logits for the model output.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_llama`](#llm_build_llama)  (Data Structure)



---
### llm\_build\_llama\_iswa<!-- {{#data_structure:llm_build_llama_iswa}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the `llama_model` instance used for model parameters.
    - `params`: Holds the `llm_graph_params` configuration for the graph.
    - `gf`: Pointer to the `ggml_cgraph` structure used for graph operations.
    - `res`: Stores the results of the computations performed by the structure.
- **Description**: The `llm_build_llama_iswa` structure is a specialized implementation of a graph context that facilitates the construction and execution of a neural network model, specifically designed for the LLaMA architecture. It inherits from `llm_graph_context` and initializes various tensors and parameters necessary for the model's forward pass, including embedding inputs, attention mechanisms, and feed-forward networks. The structure is designed to handle multiple layers of the model, applying normalization and attention mechanisms while managing the flow of data through the network.
- **Member Functions**:
    - [`llm_build_llama_iswa::llm_build_llama_iswa`](#llm_build_llama_iswallm_build_llama_iswa)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_llama\_iswa::llm\_build\_llama\_iswa<!-- {{#callable:llm_build_llama_iswa::llm_build_llama_iswa}} -->
Constructs a llama model's computation graph for inference using self-attention and feed-forward layers.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and weights.
    - `params`: A constant reference to `llm_graph_params` that holds the configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph being built.
- **Control Flow**:
    - Initializes embedding and position tensors, and scales for attention based on model parameters.
    - Iterates through each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - For each layer, computes query (Q), key (K), and value (V) tensors, applying necessary transformations and RoPE if applicable.
    - Handles the output for the last layer by skipping unused tokens and computes the final output logits.
    - Builds the computation graph by expanding the forward operations for the tensors computed.
- **Output**: The function does not return a value but populates the `res` structure with the final embedding tensor and logits for the model output.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_mul`](../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`ggml_rms_norm`](../ggml/src/ggml.c.driver.md#ggml_rms_norm)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_llama_iswa`](#llm_build_llama_iswa)  (Data Structure)



---
### llm\_build\_deci<!-- {{#data_structure:llm_build_deci}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to the `llama_model` used for building the context.
    - `params`: Reference to the `llm_graph_params` that contains parameters for the graph.
    - `gf`: Pointer to a `ggml_cgraph` structure used for graph operations.
    - `res`: Pointer to a result structure that holds output tensors.
- **Description**: The `llm_build_deci` structure extends `llm_graph_context` and is designed to build a deep learning model's computation graph using various input tensors and parameters, facilitating operations such as attention mechanisms and feed-forward networks across multiple layers.
- **Member Functions**:
    - [`llm_build_deci::llm_build_deci`](#llm_build_decillm_build_deci)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_deci::llm\_build\_deci<!-- {{#callable:llm_build_deci::llm_build_deci}} -->
Constructs a deep learning model graph for Llama architecture using specified parameters and input tensors.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model's parameters and weights.
    - `params`: A constant reference to `llm_graph_params` that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph to be built.
- **Control Flow**:
    - Initializes embedding and position tensors using `build_inp_embd` and `build_inp_pos`.
    - Iterates over each layer of the model, performing operations based on the number of attention heads and feed-forward network configurations.
    - For layers with attention heads, computes query (Q), key (K), and value (V) tensors, applying normalization and RoPE (Rotary Positional Encoding) as necessary.
    - Handles special cases for layers without attention or feed-forward networks, skipping computations where applicable.
    - At the last layer, computes outputs for only the used tokens and applies final normalization.
    - Builds the output logits using the model's output layer and expands the graph for forward computation.
- **Output**: The function does not return a value directly but populates the `res` structure with the final embedding tensor (`t_embd`) and logits tensor (`t_logits`), which are used for further processing in the model.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_deci`](#llm_build_deci)  (Data Structure)



---
### llm\_build\_baichuan<!-- {{#data_structure:llm_build_baichuan}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the model parameters used for building the Baichuan LLM.
    - `params`: Holds the parameters for the LLM graph.
    - `gf`: Pointer to the graph structure used in the LLM.
    - `res`: Stores the results of the LLM computations.
- **Description**: `llm_build_baichuan` is a structure that extends `llm_graph_context` and is responsible for constructing the Baichuan language model by processing input embeddings, attention mechanisms, and feed-forward networks through multiple layers, ultimately producing output logits.
- **Member Functions**:
    - [`llm_build_baichuan::llm_build_baichuan`](#llm_build_baichuanllm_build_baichuan)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_baichuan::llm\_build\_baichuan<!-- {{#callable:llm_build_baichuan::llm_build_baichuan}} -->
Constructs a Baichuan language model graph using the provided model parameters and input tensors.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` structure that contains the model parameters and weights.
    - `params`: A constant reference to `llm_graph_params` that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph for the model.
- **Control Flow**:
    - Initializes the number of embedding heads and asserts consistency in model parameters.
    - Builds the input embeddings and positional encodings based on the model type.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - Handles different model types (7B and 13B) with specific operations for the attention mechanism.
    - Computes the output for the last layer while skipping unused tokens.
    - Applies normalization and computes the final output logits.
- **Output**: The function does not return a value but modifies the `res` structure to store the final embedding tensor and logits, and expands the computation graph.
- **Functions called**:
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_baichuan`](#llm_build_baichuan)  (Data Structure)



---
### llm\_build\_xverse<!-- {{#data_structure:llm_build_xverse}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds graph configuration.
    - `gf`: Pointer to a `ggml_cgraph` structure for graph operations.
    - `res`: Pointer to a result structure that stores output tensors.
- **Description**: The `llm_build_xverse` structure extends `llm_graph_context` and is designed to build a neural network architecture for language models, utilizing various tensor operations and configurations defined by the `llama_model` and `llm_graph_params`. It orchestrates the construction of input embeddings, attention mechanisms, and feed-forward networks across multiple layers, ultimately producing output tensors for further processing.
- **Member Functions**:
    - [`llm_build_xverse::llm_build_xverse`](#llm_build_xversellm_build_xverse)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_xverse::llm\_build\_xverse<!-- {{#callable:llm_build_xverse::llm_build_xverse}} -->
Constructs a neural network graph for a language model using specified parameters and model weights.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` structure containing the model's parameters and weights.
    - `params`: A constant reference to a `llm_graph_params` structure that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure representing the computational graph to be built.
- **Control Flow**:
    - Initializes the number of embedding heads and asserts consistency across model parameters.
    - Builds input embeddings and positional encodings for the model.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - For each layer, computes query, key, and value tensors, reshapes them, and applies rotary positional encodings.
    - Handles the last layer differently by skipping unused tokens and computing the final output.
    - Applies normalization and builds the final output logits for the model.
- **Output**: The function does not return a value but populates the `res` structure with the final embedding tensor and logits tensor.
- **Functions called**:
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_xverse`](#llm_build_xverse)  (Data Structure)



---
### llm\_build\_falcon<!-- {{#data_structure:llm_build_falcon}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the `llama_model` instance used for building the Falcon model.
    - `params`: Holds the `llm_graph_params` configuration for the graph context.
    - `gf`: Pointer to the `ggml_cgraph` structure used for graph operations.
    - `res`: Stores the results of the model's computations.
- **Description**: The `llm_build_falcon` structure is a specialized implementation of `llm_graph_context` designed to build and manage the Falcon model architecture, utilizing various tensors and layers defined in the `llama_model`. It orchestrates the construction of attention mechanisms, feed-forward networks, and normalization processes across multiple layers, ensuring the model's parameters are correctly applied and optimized for performance.
- **Member Functions**:
    - [`llm_build_falcon::llm_build_falcon`](#llm_build_falconllm_build_falcon)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_falcon::llm\_build\_falcon<!-- {{#callable:llm_build_falcon::llm_build_falcon}} -->
The `llm_build_falcon` function constructs a neural network graph for the Falcon model using specified parameters and input data.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model architecture and parameters.
    - `params`: A constant reference to `llm_graph_params` that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computational graph being built.
- **Control Flow**:
    - The function begins by extracting embedding dimensions and asserting consistency in model parameters.
    - It initializes input tensors for embeddings and positions, and prepares attention input tensors.
    - A loop iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - Within the loop, it builds attention and feed-forward layers, reshapes tensors, and applies rotary embeddings.
    - For the last layer, it skips unused tokens and computes the final output tensors.
    - Finally, it normalizes the output and builds the final logits tensor before expanding the graph.
- **Output**: The function does not return a value but populates the `res` structure with the final embedding tensor and logits tensor, which are used for further processing in the neural network.
- **Functions called**:
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_falcon`](#llm_build_falcon)  (Data Structure)



---
### llm\_build\_grok<!-- {{#data_structure:llm_build_grok}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds configuration parameters.
    - `gf`: Pointer to a `ggml_cgraph` structure used for graph operations.
    - `res`: Pointer to a result structure that stores output tensors.
- **Description**: The `llm_build_grok` structure is a specialized implementation of `llm_graph_context` designed to build and manage the forward pass of a neural network model, specifically for processing input embeddings, attention mechanisms, and feed-forward networks, while applying various normalization and scaling operations throughout the layers.
- **Member Functions**:
    - [`llm_build_grok::llm_build_grok`](#llm_build_grokllm_build_grok)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_grok::llm\_build\_grok<!-- {{#callable:llm_build_grok::llm_build_grok}} -->
The `llm_build_grok` function constructs a neural network graph for a language model using specified parameters and a llama model.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model's parameters and weights.
    - `params`: A constant reference to `llm_graph_params` that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computational graph being built.
- **Control Flow**:
    - The function begins by asserting that the number of embedding heads and rotations are consistent.
    - It initializes input embeddings and scales them by a predefined multiplier.
    - A loop iterates over the number of layers in the model, performing normalization, self-attention, and feed-forward operations for each layer.
    - Within the loop, it computes query (Q), key (K), and value (V) tensors, applying necessary transformations and additions.
    - If the last layer is reached, it skips computing outputs for unused tokens.
    - After processing each layer, it applies normalization and combines outputs before passing them to the next layer.
    - Finally, it normalizes the output, applies a linear transformation, scales the logits, and expands the graph for forward computation.
- **Output**: The function outputs the final embeddings and logits of the model, stored in the `res` structure, and expands the computational graph for further processing.
- **Functions called**:
    - [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_grok`](#llm_build_grok)  (Data Structure)



---
### llm\_build\_dbrx<!-- {{#data_structure:llm_build_dbrx}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds configuration parameters.
    - `gf`: Pointer to a `ggml_cgraph` structure used for graph operations.
    - `res`: Pointer to a result structure that stores output tensors.
- **Description**: The `llm_build_dbrx` structure extends `llm_graph_context` and is designed to build a deep learning model's computation graph using various input tensors and parameters, facilitating operations such as self-attention and feed-forward networks across multiple layers.
- **Member Functions**:
    - [`llm_build_dbrx::llm_build_dbrx`](#llm_build_dbrxllm_build_dbrx)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_dbrx::llm\_build\_dbrx<!-- {{#callable:llm_build_dbrx::llm_build_dbrx}} -->
Constructs a deep learning model graph for a language model using specified parameters and input tensors.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model's parameters and weights.
    - `params`: A constant reference to `llm_graph_params` that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph being built.
- **Control Flow**:
    - Initializes embedding and position tensors using the model's token embeddings and position information.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - For each layer, computes query, key, and value tensors, applies attention mechanisms, and processes the output through a feed-forward network.
    - Handles the last layer differently by skipping unused tokens and adjusting the output accordingly.
    - Finalizes the output by applying normalization and generating logits for the model's predictions.
- **Output**: The function does not return a value but populates the `res` structure with the final embedding tensor and logits for the model.
- **Functions called**:
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_dbrx`](#llm_build_dbrx)  (Data Structure)



---
### llm\_build\_starcoder<!-- {{#data_structure:llm_build_starcoder}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds graph configuration.
    - `gf`: Pointer to a `ggml_cgraph` structure for graph operations.
    - `res`: Pointer to a result structure that stores output tensors.
- **Description**: The `llm_build_starcoder` structure is a derived class from `llm_graph_context` that encapsulates the construction of a neural network model, specifically designed for processing inputs through multiple layers, utilizing various tensor operations and configurations defined by the `llama_model` and `llm_graph_params`. It manages the input embeddings, attention mechanisms, and feedforward networks, while also handling normalization and output generation, making it a crucial component in the model's architecture.
- **Member Functions**:
    - [`llm_build_starcoder::llm_build_starcoder`](#llm_build_starcoderllm_build_starcoder)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_starcoder::llm\_build\_starcoder<!-- {{#callable:llm_build_starcoder::llm_build_starcoder}} -->
Constructs a starcoder model using the provided llama model and graph parameters.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and embeddings.
    - `params`: A constant reference to `llm_graph_params` that holds the configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph for the model.
- **Control Flow**:
    - Initializes embedding and position tensors using the provided model.
    - Iterates through each layer of the model to build attention and feed-forward components.
    - For each layer, computes normalized inputs, applies self-attention, and adds residual connections.
    - Handles the last layer differently by skipping unused tokens and preparing the output.
    - Finalizes the model by applying normalization and computing the output logits.
- **Output**: The function does not return a value but populates the `res` structure with the final embeddings and logits for the model.
- **Functions called**:
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_starcoder`](#llm_build_starcoder)  (Data Structure)



---
### llm\_build\_refact<!-- {{#data_structure:llm_build_refact}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds configuration parameters.
    - `gf`: Pointer to a `ggml_cgraph` structure used for graph operations.
- **Description**: The `llm_build_refact` structure is a derived class from `llm_graph_context` that encapsulates the construction of a neural network model, specifically for a language model, utilizing various tensors and layers defined in the model. It initializes the model with embeddings, processes through multiple layers applying attention mechanisms and feed-forward networks, and ultimately prepares the output logits for further operations.
- **Member Functions**:
    - [`llm_build_refact::llm_build_refact`](#llm_build_refactllm_build_refact)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_refact::llm\_build\_refact<!-- {{#callable:llm_build_refact::llm_build_refact}} -->
Constructs a refactored LLM graph using the provided model and parameters.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and weights.
    - `params`: A constant reference to a `llm_graph_params` object that holds the graph configuration parameters.
    - `gf`: A pointer to a `ggml_cgraph` object that represents the computational graph to be built.
- **Control Flow**:
    - Initializes the number of embedding heads and asserts that the number of embedding heads for keys is equal to that for values.
    - Builds the input embeddings from the model's token embeddings.
    - Enters a loop over the number of layers in the model, performing operations for each layer.
    - For each layer, normalizes the input and computes the self-attention using query, key, and value matrices.
    - Reshapes the query, key, and value tensors for multi-head attention.
    - If processing the last layer, skips computing outputs for unused tokens.
    - Adds the output of the attention mechanism to the input for the feed-forward network.
    - Normalizes the feed-forward input and computes the feed-forward network output.
    - Adds the feed-forward output to the previous input and prepares it for the next layer.
    - After processing all layers, normalizes the final output and computes the logits using the model's output layer.
- **Output**: The function does not return a value but populates the `res` structure with the final embedding tensor and logits.
- **Functions called**:
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_refact`](#llm_build_refact)  (Data Structure)



---
### llm\_build\_bert<!-- {{#data_structure:llm_build_bert}} -->
- **Type**: `struct`
- **Members**:
    - `model`: The `llama_model` instance used for model parameters.
    - `params`: The `llm_graph_params` instance containing graph parameters.
    - `gf`: A pointer to a `ggml_cgraph` structure for graph operations.
    - `n_embd_head`: An integer representing the number of embedding heads.
    - `n_embd_gqa`: An integer representing the number of embedding dimensions for GQA.
    - `cur`: A pointer to the current `ggml_tensor` being processed.
    - `inpL`: A pointer to the input layer tensor.
    - `inp_pos`: A pointer to the input position tensor, initialized to nullptr.
    - `inp_attn`: A pointer to the input attention tensor.
    - `Qcur`: A pointer to the current query tensor.
    - `Kcur`: A pointer to the current key tensor.
    - `Vcur`: A pointer to the current value tensor.
    - `ffn_inp`: A pointer to the input tensor for the feed-forward network.
    - `res`: A pointer to the result structure containing output embeddings.
- **Description**: The `llm_build_bert` structure is a specialized implementation of a BERT-like model that inherits from `llm_graph_context`, designed to build and manage the forward pass of a transformer model using various tensor operations. It initializes input embeddings, processes multiple layers of self-attention and feed-forward networks, and applies normalization at each layer, ultimately producing output embeddings for further use.
- **Member Functions**:
    - [`llm_build_bert::llm_build_bert`](#llm_build_bertllm_build_bert)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_bert::llm\_build\_bert<!-- {{#callable:llm_build_bert::llm_build_bert}} -->
Constructs a BERT model graph using the provided model parameters and input tensors.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` structure that contains the architecture and parameters of the BERT model.
    - `params`: A constant reference to a `llm_graph_params` structure that holds the graph parameters for the model.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph for the model.
- **Control Flow**:
    - Initializes embedding dimensions and asserts consistency of embedding head sizes.
    - Builds input position embeddings if the model architecture is not JINA BERT V2.
    - Constructs input embeddings by combining token, type, and position embeddings.
    - Iterates through each layer of the model, performing self-attention and feed-forward operations.
    - Applies normalization and residual connections at each layer.
    - Handles different architectures and configurations for attention and feed-forward networks.
    - Builds the final output tensor and expands the computation graph.
- **Output**: The function outputs a tensor representing the final embeddings of the model, stored in `res->t_embd`, and expands the computation graph for further processing.
- **Functions called**:
    - [`ggml_view_1d`](../ggml/src/ggml.c.driver.md#ggml_view_1d)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_bert`](#llm_build_bert)  (Data Structure)



---
### llm\_build\_bloom<!-- {{#data_structure:llm_build_bloom}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds configuration parameters.
    - `gf`: Pointer to a `ggml_cgraph` structure used for graph computations.
    - `res`: Pointer to a result structure that stores output tensors.
- **Description**: The `llm_build_bloom` structure is a specialized type that extends `llm_graph_context`, designed to facilitate the construction of a neural network model using various layers and operations, leveraging parameters from a `llama_model` and `llm_graph_params` to build and process input embeddings, attention mechanisms, and feedforward networks, ultimately producing output tensors for further use.
- **Member Functions**:
    - [`llm_build_bloom::llm_build_bloom`](#llm_build_bloomllm_build_bloom)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_bloom::llm\_build\_bloom<!-- {{#callable:llm_build_bloom::llm_build_bloom}} -->
Constructs a Bloom model graph using the provided llama model and parameters.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and embeddings.
    - `params`: A constant reference to `llm_graph_params` that holds the configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph.
- **Control Flow**:
    - Initializes embedding and attention input tensors using the model's token embeddings and normalization parameters.
    - Iterates through each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - For each layer, computes the query, key, and value tensors from the input, reshapes them, and applies attention mechanisms.
    - Handles the last layer differently by skipping unused tokens and preparing the output tensors.
    - Applies normalization to the final output and constructs the logits tensor before expanding the graph.
- **Output**: The function does not return a value but populates the `res` structure with the final embedding and logits tensors.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_bloom`](#llm_build_bloom)  (Data Structure)



---
### llm\_build\_mpt<!-- {{#data_structure:llm_build_mpt}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds graph configuration.
    - `gf`: Pointer to a `ggml_cgraph` structure for managing computation graphs.
    - `cur`: Pointer to a `ggml_tensor` representing the current tensor in processing.
    - `pos`: Pointer to a `ggml_tensor` for position embeddings.
    - `inpL`: Pointer to a `ggml_tensor` for input embeddings.
- **Description**: The `llm_build_mpt` structure extends `llm_graph_context` and is designed to facilitate the construction of a multi-part tensor graph for a llama model, incorporating various layers and operations such as attention mechanisms and feed-forward networks, while managing input and output tensors throughout the process.
- **Member Functions**:
    - [`llm_build_mpt::llm_build_mpt`](#llm_build_mptllm_build_mpt)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_mpt::llm\_build\_mpt<!-- {{#callable:llm_build_mpt::llm_build_mpt}} -->
Constructs a multi-layer transformer model using the provided llama model and parameters.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model architecture and parameters.
    - `params`: A constant reference to `llm_graph_params` that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure used for building the computation graph.
- **Control Flow**:
    - Initializes embedding and attention input tensors based on the model's token embeddings.
    - Checks if positional embeddings are present and adds them to the input embeddings if so.
    - Iterates through each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - Handles layer normalization for query and key tensors if specified in the model.
    - Reshapes the query, key, and value tensors for multi-head attention.
    - Computes the output for the last layer by skipping unused tokens and adding the input to the feed-forward output.
    - Applies normalization and final output transformations before expanding the computation graph.
- **Output**: The function does not return a value but modifies the `res` structure to store the final embeddings and logits, and expands the computation graph for execution.
- **Functions called**:
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_mpt`](#llm_build_mpt)  (Data Structure)



---
### llm\_build\_stablelm<!-- {{#data_structure:llm_build_stablelm}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the model parameters used for building the stable LM.
    - `params`: Holds the parameters for the graph context.
    - `gf`: Pointer to the graph structure used in the model.
    - `res`: Stores the results of the model's computations.
- **Description**: The `llm_build_stablelm` structure is a derived type from `llm_graph_context` that encapsulates the construction of a stable language model using various layers and attention mechanisms, processing input embeddings, and generating output logits through a series of transformations and normalizations.
- **Member Functions**:
    - [`llm_build_stablelm::llm_build_stablelm`](#llm_build_stablelmllm_build_stablelm)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_stablelm::llm\_build\_stablelm<!-- {{#callable:llm_build_stablelm::llm_build_stablelm}} -->
Constructs a stable language model graph using the provided model parameters and input tensors.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` structure that contains the model parameters and weights.
    - `params`: A constant reference to `llm_graph_params` that holds the configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph to be built.
- **Control Flow**:
    - Initializes the embedding input tensor and position tensor.
    - Iterates over each layer of the model to build the attention and feed-forward components.
    - For each layer, computes the normalized input, self-attention outputs, and applies the feed-forward network.
    - Handles the last layer differently by skipping unused tokens and adjusting the output accordingly.
    - Applies normalization and computes the final logits for the output.
- **Output**: The function does not return a value but populates the `res` structure with the final embedding tensor and logits.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_stablelm`](#llm_build_stablelm)  (Data Structure)



---
### llm\_build\_qwen<!-- {{#data_structure:llm_build_qwen}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the model parameters used for building the LLM.
    - `params`: Holds the parameters for the LLM graph.
    - `gf`: Pointer to the graph structure used in the LLM.
    - `res`: Stores the results of the LLM computations.
- **Description**: The `llm_build_qwen` structure is a derived type from `llm_graph_context` that encapsulates the construction of a large language model (LLM) by processing input embeddings, attention mechanisms, and feed-forward networks across multiple layers, ultimately producing output logits and embeddings.
- **Member Functions**:
    - [`llm_build_qwen::llm_build_qwen`](#llm_build_qwenllm_build_qwen)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_qwen::llm\_build\_qwen<!-- {{#callable:llm_build_qwen::llm_build_qwen}} -->
Constructs a neural network graph for a language model using specified parameters and model weights.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` structure containing the model's weights and configurations.
    - `params`: A constant reference to a `llm_graph_params` structure that holds parameters for the graph construction.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computational graph being built.
- **Control Flow**:
    - Initializes the embedding input and position tensors.
    - Iterates over each layer of the model to build the attention and feed-forward components.
    - For each layer, computes the normalized input, applies self-attention, and processes the output through a feed-forward network.
    - Handles the last layer differently by skipping unused tokens and preparing the output.
    - Applies normalization and final transformations to produce the output logits.
- **Output**: The function does not return a value but modifies the `res` structure to store the final embeddings and logits, and expands the graph for forward computation.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_qwen`](#llm_build_qwen)  (Data Structure)



---
### llm\_build\_qwen2<!-- {{#data_structure:llm_build_qwen2}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the `llama_model` used for building the model.
    - `params`: Holds the `llm_graph_params` that configure the graph.
    - `gf`: Pointer to the `ggml_cgraph` used for graph operations.
    - `res`: Stores the results of the model's computations.
- **Description**: The `llm_build_qwen2` structure is a specialized implementation of a neural network model that extends `llm_graph_context`, designed to build and manage the forward pass of a transformer model using various tensor operations, including attention mechanisms and feed-forward networks, while ensuring proper normalization and handling of input embeddings.
- **Member Functions**:
    - [`llm_build_qwen2::llm_build_qwen2`](#llm_build_qwen2llm_build_qwen2)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_qwen2::llm\_build\_qwen2<!-- {{#callable:llm_build_qwen2::llm_build_qwen2}} -->
Constructs a neural network graph for the Qwen2 model using the provided llama model and parameters.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model's parameters and weights.
    - `params`: A constant reference to a `llm_graph_params` object that holds the configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computational graph being built.
- **Control Flow**:
    - Initializes the number of embedding heads and asserts their consistency with model parameters.
    - Builds input embeddings and positional encodings.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - Computes query (Q), key (K), and value (V) tensors, applying necessary transformations and RoPE (Rotary Positional Encoding).
    - Handles the last layer differently by skipping unused tokens and adjusting the input accordingly.
    - Applies normalization and builds the final output logits after processing through all layers.
- **Output**: The function does not return a value directly but populates the `res` structure with the final embedding tensor and logits for the model.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_qwen2`](#llm_build_qwen2)  (Data Structure)



---
### llm\_build\_qwen2vl<!-- {{#data_structure:llm_build_qwen2vl}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the model parameters used for building the layer.
    - `params`: Holds the parameters for the graph context.
    - `gf`: Pointer to the graph structure used in computations.
    - `n_embd_head`: Represents the number of embedding heads.
    - `cur`: A tensor used to hold intermediate results during processing.
    - `inpL`: Tensor representing the input embeddings.
    - `inp_pos`: Tensor containing positional information.
    - `inp_attn`: Tensor for attention input.
    - `sections`: Array holding sections for RoPE.
    - `il`: Index for the current layer being processed.
    - `ffn_inp`: Tensor for the input to the feed-forward network.
    - `res`: Structure to hold the final output tensors.
- **Description**: The `llm_build_qwen2vl` struct is designed to build a layer of a neural network model, specifically for a language model, by processing input embeddings, applying attention mechanisms, and utilizing feed-forward networks across multiple layers, while managing intermediate tensor states and ensuring proper dimensionality through reshaping and normalization.
- **Member Functions**:
    - [`llm_build_qwen2vl::llm_build_qwen2vl`](#llm_build_qwen2vlllm_build_qwen2vl)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_qwen2vl::llm\_build\_qwen2vl<!-- {{#callable:llm_build_qwen2vl::llm_build_qwen2vl}} -->
Constructs a neural network graph for the Qwen2 model using the provided llama model and parameters.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model's parameters and weights.
    - `params`: A constant reference to a `llm_graph_params` object that holds the configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` object that represents the computational graph being built.
- **Control Flow**:
    - Initializes the number of embedding heads and asserts consistency in model parameters.
    - Builds input embeddings and positional encodings.
    - Iterates over each layer of the model to perform normalization, self-attention, and feed-forward operations.
    - Computes query (Q), key (K), and value (V) tensors, applying necessary transformations and RoPE (Rotary Positional Encoding).
    - Handles the last layer differently by skipping unused tokens and preparing the output.
    - Applies normalization and computes the final output logits.
- **Output**: The function does not return a value directly but populates the `res` structure with the final embedding tensor and logits for the model output.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_multi`](../ggml/src/ggml.c.driver.md#ggml_rope_multi)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_qwen2vl`](#llm_build_qwen2vl)  (Data Structure)



---
### llm\_build\_qwen2moe<!-- {{#data_structure:llm_build_qwen2moe}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the model parameters used for building the architecture.
    - `params`: Holds the parameters for the graph context.
    - `gf`: Pointer to the graph structure used in computations.
    - `n_embd_head`: Represents the number of embedding heads.
    - `inpL`: Tensor representing the input embeddings.
    - `inp_pos`: Tensor containing positional embeddings.
    - `inp_attn`: Tensor for attention input.
    - `cur`: Current tensor being processed.
    - `ffn_inp`: Input tensor for the feed-forward network.
    - `moe_out`: Output tensor from the mixture of experts.
    - `res`: Structure to hold the results of the computation.
- **Description**: The `llm_build_qwen2moe` structure is a specialized implementation of a neural network architecture that extends the `llm_graph_context` class, designed to build and manage the layers of a language model using various tensor operations, including attention mechanisms and feed-forward networks, while incorporating mixture of experts (MoE) for enhanced performance.
- **Member Functions**:
    - [`llm_build_qwen2moe::llm_build_qwen2moe`](#llm_build_qwen2moellm_build_qwen2moe)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_qwen2moe::llm\_build\_qwen2moe<!-- {{#callable:llm_build_qwen2moe::llm_build_qwen2moe}} -->
Constructs a Qwen2Moe model graph using the provided llama model and parameters.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model architecture and parameters.
    - `params`: A constant reference to `llm_graph_params` that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph.
- **Control Flow**:
    - Initializes the embedding and position tensors using `build_inp_embd` and `build_inp_pos`.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - Computes query (Q), key (K), and value (V) tensors for self-attention, applying optional biases and reshaping them.
    - Applies rotary positional encoding to Q and K tensors.
    - Handles the final layer differently by skipping unused tokens and computing the output for only relevant tokens.
    - Constructs the feed-forward input by adding the current tensor to the input from the previous layer.
    - Processes the MoE (Mixture of Experts) branch and shared expert feed-forward network.
    - Normalizes the output and prepares it for the final linear transformation to logits.
- **Output**: The function does not return a value directly but populates the `res` structure with the final embedding tensor (`t_embd`) and logits tensor (`t_logits`).
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_div`](../ggml/src/ggml.c.driver.md#ggml_div)
    - [`ggml_silu`](../ggml/src/ggml.c.driver.md#ggml_silu)
    - [`ggml_mul`](../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_qwen2moe`](#llm_build_qwen2moe)  (Data Structure)



---
### llm\_build\_qwen3<!-- {{#data_structure:llm_build_qwen3}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the model parameters used for building the layer.
    - `params`: Holds the parameters for the graph context.
    - `gf`: Pointer to the graph structure used in computations.
    - `res`: Stores the results of the computations.
- **Description**: `llm_build_qwen3` is a structure that extends `llm_graph_context` and is designed to build a neural network model using various layers and operations, including attention mechanisms and feed-forward networks, while managing input embeddings and normalization across multiple layers.
- **Member Functions**:
    - [`llm_build_qwen3::llm_build_qwen3`](#llm_build_qwen3llm_build_qwen3)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_qwen3::llm\_build\_qwen3<!-- {{#callable:llm_build_qwen3::llm_build_qwen3}} -->
Constructs a neural network graph for the Qwen3 model using the provided llama model and parameters.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model's parameters and weights.
    - `params`: A constant reference to a `llm_graph_params` object that holds the configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` object that represents the computational graph being built.
- **Control Flow**:
    - Initializes the number of embedding heads and asserts their consistency with model parameters.
    - Builds input embeddings and position tensors.
    - Iterates over each layer of the model to perform normalization, self-attention, and feed-forward operations.
    - For each layer, computes query (Q), key (K), and value (V) tensors, applying normalization and rotary positional encoding.
    - Handles the last layer differently by skipping unused tokens and adjusting the input accordingly.
    - Adds the output of the attention mechanism to the input for the feed-forward network.
    - Normalizes the feed-forward input and computes the output using the feed-forward network.
    - Builds the final output tensor and applies normalization before storing the results.
- **Output**: The function does not return a value but populates the `res` structure with the final embedding tensor and logits for the model output.
- **Functions called**:
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_qwen3`](#llm_build_qwen3)  (Data Structure)



---
### llm\_build\_qwen3moe<!-- {{#data_structure:llm_build_qwen3moe}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the model parameters used for building the layer.
    - `params`: Holds the parameters for the graph context.
    - `gf`: Pointer to the graph structure used in computations.
    - `res`: Stores the results of the computations.
    - `hparams`: Holds hyperparameters related to the model.
    - `n_layer`: Indicates the number of layers in the model.
    - `n_embd_head`: Specifies the number of embedding heads.
    - `n_expert`: Defines the number of experts in the mixture of experts.
    - `n_expert_used`: Tracks the number of experts that are actually used.
    - `rope_type`: Specifies the type of rotary positional encoding.
    - `freq_base`: Base frequency for the rotary encoding.
    - `freq_scale`: Scale factor for the frequency.
    - `ext_factor`: External factor used in computations.
    - `attn_factor`: Attention factor used in the model.
    - `beta_fast`: Fast beta parameter for gating.
    - `beta_slow`: Slow beta parameter for gating.
- **Description**: `llm_build_qwen3moe` is a derived structure from `llm_graph_context` that encapsulates the construction of a neural network model, specifically designed for processing layers of attention and feedforward networks, utilizing various tensor operations and configurations based on the provided model and parameters.
- **Member Functions**:
    - [`llm_build_qwen3moe::llm_build_qwen3moe`](#llm_build_qwen3moellm_build_qwen3moe)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_qwen3moe::llm\_build\_qwen3moe<!-- {{#callable:llm_build_qwen3moe::llm_build_qwen3moe}} -->
Constructs a Qwen3 MoE model graph using the provided llama model and parameters.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and weights.
    - `params`: A constant reference to a `llm_graph_params` object that holds the graph configuration parameters.
    - `gf`: A pointer to a `ggml_cgraph` object that represents the computation graph for the model.
- **Control Flow**:
    - Initializes the number of embedding heads and asserts their consistency with model parameters.
    - Builds input embeddings and positional embeddings.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - Computes query (Q), key (K), and value (V) tensors, applying normalization and RoPE transformations.
    - Handles the last layer differently by skipping unused tokens and preparing the output.
    - Applies a mixture of experts (MoE) mechanism in the feed-forward network.
    - Finalizes the output by applying normalization and computing logits.
- **Output**: The function constructs the model's computation graph and stores the resulting embeddings and logits in the `res` object.
- **Functions called**:
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_qwen3moe`](#llm_build_qwen3moe)  (Data Structure)



---
### llm\_build\_phi2<!-- {{#data_structure:llm_build_phi2}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds graph configuration.
    - `gf`: Pointer to a `ggml_cgraph` structure for managing computation graphs.
    - `res`: Pointer to a result structure that stores output tensors.
- **Description**: The `llm_build_phi2` structure extends `llm_graph_context` and is designed to facilitate the construction of a neural network model, specifically for processing input embeddings and performing attention mechanisms across multiple layers, while managing tensor operations and ensuring proper normalization and scaling of outputs.
- **Member Functions**:
    - [`llm_build_phi2::llm_build_phi2`](#llm_build_phi2llm_build_phi2)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_phi2::llm\_build\_phi2<!-- {{#callable:llm_build_phi2::llm_build_phi2}} -->
The `llm_build_phi2` constructor initializes a layer of a neural network model by building its components, including attention and feedforward layers, while managing tensor operations and normalization.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` structure that contains the model parameters and layer configurations.
    - `params`: A constant reference to `llm_graph_params` that holds the parameters for the graph context.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph for the model.
- **Control Flow**:
    - The function begins by extracting the number of embedding heads and GQA embeddings from the model's hyperparameters.
    - It asserts that the number of embedding heads matches the expected value.
    - The input embeddings and positional embeddings are built using helper functions.
    - A loop iterates over each layer of the model, performing normalization, self-attention, and feedforward operations.
    - Within the loop, it builds query, key, and value tensors, applying necessary transformations and scaling.
    - If processing the last layer, it skips unused tokens by filtering the output.
    - After processing all layers, it applies normalization to the final output and computes the logits.
- **Output**: The function does not return a value directly; instead, it modifies the state of the `res` object to store the final embeddings and logits, which are used for further processing in the model.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_phi2`](#llm_build_phi2)  (Data Structure)



---
### llm\_build\_phi3<!-- {{#data_structure:llm_build_phi3}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds graph configuration.
    - `gf`: Pointer to a `ggml_cgraph` structure for graph operations.
    - `n_embd_head`: Integer representing the number of embedding heads.
    - `n_embd_gqa`: Integer representing the number of embedding dimensions for GQA.
    - `inpL`: Pointer to a `ggml_tensor` representing the input layer.
    - `inp_pos`: Pointer to a `ggml_tensor` containing positional embeddings.
    - `inp_attn`: Pointer to the attention input tensor, type determined by `iswa`.
    - `residual`: Pointer to a tensor used for residual connections.
    - `cur`: Pointer to a tensor representing the current output.
- **Description**: The `llm_build_phi3` struct is a template-based structure that extends `llm_graph_context`, designed to build and manage the forward pass of a neural network model, specifically for large language models. It initializes various components such as input embeddings, attention mechanisms, and feed-forward networks, while also handling residual connections and normalization layers across multiple layers of the model. The structure is parameterized by a boolean `iswa`, which influences the type of attention mechanism used, and it encapsulates the complexity of managing tensor operations and model parameters during the forward computation.
- **Member Functions**:
    - [`llm_build_phi3::llm_build_phi3`](#llm_build_phi3llm_build_phi3)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_phi3::llm\_build\_phi3<!-- {{#callable:llm_build_phi3::llm_build_phi3}} -->
Constructs a forward pass through a neural network model using attention mechanisms and feed-forward layers.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and weights.
    - `params`: A constant reference to a `llm_graph_params` object that holds the graph parameters for the model.
    - `gf`: A pointer to a `ggml_cgraph` object that represents the computation graph for the model.
- **Control Flow**:
    - Initializes embedding inputs and position tensors.
    - Determines the type of attention input based on the template parameter `iswa`.
    - Iterates over each layer of the model, performing self-attention and feed-forward operations.
    - For each layer, computes the attention scores using query, key, and value tensors, applying normalization and residual connections.
    - Handles the final layer differently by skipping unused tokens and applying output normalization.
    - Constructs the final output logits by applying a linear transformation and adding a bias if present.
- **Output**: The function outputs the final logits tensor, which is the result of the forward pass through the model, stored in `res->t_logits`.
- **Functions called**:
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_phi3`](#llm_build_phi3)  (Data Structure)



---
### llm\_build\_plamo<!-- {{#data_structure:llm_build_plamo}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the `llama_model` used for building the model.
    - `params`: Holds the `llm_graph_params` that configure the graph.
    - `gf`: Pointer to the `ggml_cgraph` used for graph operations.
    - `res`: Stores the results of the model's computations.
- **Description**: The `llm_build_plamo` structure is a derived class from `llm_graph_context` that encapsulates the construction of a neural network model using various components such as embeddings, attention mechanisms, and feed-forward networks, while managing the necessary parameters and tensors for the model's operations.
- **Member Functions**:
    - [`llm_build_plamo::llm_build_plamo`](#llm_build_plamollm_build_plamo)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_plamo::llm\_build\_plamo<!-- {{#callable:llm_build_plamo::llm_build_plamo}} -->
Constructs a transformer model's computation graph using the provided model parameters and input data.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` structure that contains the model's parameters and weights.
    - `params`: A constant reference to a `llm_graph_params` structure that holds the graph parameters for the model.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph being built.
- **Control Flow**:
    - Initializes the number of embedding heads and asserts consistency in model parameters.
    - Builds the input embeddings and position tensors.
    - Iterates over each layer of the model to construct the computation graph, performing normalization, self-attention, and feed-forward operations.
    - For each layer, computes the query (Q), key (K), and value (V) tensors, applying necessary reshaping and RoPE (Rotary Positional Encoding).
    - Handles the output for the last layer by skipping unused tokens and adjusting the input tensors accordingly.
    - Adds the results of the self-attention and feed-forward network to the current tensor.
    - Normalizes the final output tensor and prepares it for the output layer.
    - Builds the final output logits using the model's output layer.
- **Output**: The function does not return a value directly; instead, it modifies the `res` structure to store the final embedding tensor and logits for the model's output.
- **Functions called**:
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_plamo`](#llm_build_plamo)  (Data Structure)



---
### llm\_build\_gpt2<!-- {{#data_structure:llm_build_gpt2}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the model parameters used for building the GPT-2 architecture.
    - `params`: Holds the parameters for the graph context.
    - `gf`: Pointer to the graph structure used for computations.
    - `n_embd_head`: Represents the number of embedding heads.
    - `n_embd_gqa`: Indicates the number of embeddings for the GQA.
    - `cur`: Temporary tensor used during the building process.
    - `pos`: Tensor representing positional embeddings.
    - `inpL`: Input tensor for the layer.
    - `inp_pos`: Tensor containing position information.
    - `inp_attn`: Tensor for attention input.
    - `Qcur`: Tensor for query embeddings.
    - `Kcur`: Tensor for key embeddings.
    - `Vcur`: Tensor for value embeddings.
    - `ffn_inp`: Input tensor for the feedforward network.
    - `res`: Structure to hold the results of the computation.
- **Description**: The `llm_build_gpt2` structure is designed to build and manage the architecture of a GPT-2 model, inheriting from `llm_graph_context`. It initializes various tensors and parameters necessary for the model's layers, including embeddings, attention mechanisms, and feedforward networks, while ensuring the correct flow of data through the model's architecture.
- **Member Functions**:
    - [`llm_build_gpt2::llm_build_gpt2`](#llm_build_gpt2llm_build_gpt2)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_gpt2::llm\_build\_gpt2<!-- {{#callable:llm_build_gpt2::llm_build_gpt2}} -->
Constructs a GPT-2 model graph using the provided model parameters and input tensors.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and embeddings.
    - `params`: A constant reference to `llm_graph_params` that holds the configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph for the model.
- **Control Flow**:
    - Initializes embedding tensors and position embeddings using helper functions.
    - Iterates over each layer of the model to build the attention and feed-forward components.
    - For each layer, normalizes the input, computes self-attention, and applies feed-forward transformations.
    - Handles the last layer differently by skipping unused tokens and preparing the output.
    - Finalizes the output by normalizing and applying the output layer transformations.
- **Output**: The function constructs the model's computation graph and stores the resulting tensors in the `res` structure, specifically `t_embd` for embeddings and `t_logits` for logits.
- **Functions called**:
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_gpt2`](#llm_build_gpt2)  (Data Structure)



---
### llm\_build\_codeshell<!-- {{#data_structure:llm_build_codeshell}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the `llama_model` instance used for building the codeshell.
    - `params`: Holds the `llm_graph_params` configuration for the graph context.
    - `gf`: Pointer to the `ggml_cgraph` structure used for graph operations.
    - `res`: Stores the results of the computations performed in the codeshell.
- **Description**: The `llm_build_codeshell` structure extends `llm_graph_context` and is designed to facilitate the construction of a neural network graph for a language model, utilizing various input embeddings, attention mechanisms, and feedforward networks across multiple layers, while ensuring proper normalization and tensor operations.
- **Member Functions**:
    - [`llm_build_codeshell::llm_build_codeshell`](#llm_build_codeshellllm_build_codeshell)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_codeshell::llm\_build\_codeshell<!-- {{#callable:llm_build_codeshell::llm_build_codeshell}} -->
Constructs a code shell for a language model by building its computational graph.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and weights.
    - `params`: A constant reference to `llm_graph_params` that holds the configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computational graph being built.
- **Control Flow**:
    - Initializes embedding and position tensors using the model's token embeddings and position information.
    - Iterates over each layer of the model to build normalization, self-attention, and feed-forward components.
    - For each layer, computes the attention scores using query, key, and value tensors, applying necessary reshaping and positional encodings.
    - Handles the final layer differently by skipping unused tokens and preparing the output tensors.
    - Applies normalization and builds the final output logits before expanding the graph for forward computation.
- **Output**: The function does not return a value but populates the `res` structure with the final embedding and logits tensors.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_codeshell`](#llm_build_codeshell)  (Data Structure)



---
### llm\_build\_orion<!-- {{#data_structure:llm_build_orion}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds configuration parameters.
    - `gf`: Pointer to a `ggml_cgraph` structure used for graph operations.
    - `n_embd_head`: Integer representing the number of embedding heads.
    - `inpL`: Pointer to a `ggml_tensor` representing the input layer.
    - `inp_pos`: Pointer to a `ggml_tensor` containing positional embeddings.
    - `inp_attn`: Pointer to a `ggml_tensor` for attention input.
    - `cur`: Pointer to a `ggml_tensor` used for intermediate computations.
- **Description**: `llm_build_orion` is a derived structure from `llm_graph_context` that encapsulates the construction of a neural network model, specifically designed for processing input through multiple layers, utilizing attention mechanisms and feed-forward networks, while managing tensor operations and configurations based on the provided model and parameters.
- **Member Functions**:
    - [`llm_build_orion::llm_build_orion`](#llm_build_orionllm_build_orion)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_orion::llm\_build\_orion<!-- {{#callable:llm_build_orion::llm_build_orion}} -->
Constructs a neural network graph for a language model using specified parameters and model weights.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` structure containing the model's weights and configurations.
    - `params`: A constant reference to a `llm_graph_params` structure that holds parameters for building the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computational graph being built.
- **Control Flow**:
    - Initializes the number of embedding heads and asserts consistency in model parameters.
    - Builds input embeddings and positional encodings.
    - Iterates over each layer of the model to construct the attention and feed-forward components.
    - For each layer, computes normalized inputs, applies self-attention, and processes through a feed-forward network.
    - Handles the last layer differently by skipping unused tokens and preparing the output.
    - Applies normalization and computes the final logits for the model output.
- **Output**: The function does not return a value but populates the `res` structure with the final embeddings and logits, which are used for further processing in the language model.
- **Functions called**:
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_orion`](#llm_build_orion)  (Data Structure)



---
### llm\_build\_internlm2<!-- {{#data_structure:llm_build_internlm2}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds graph configuration.
    - `gf`: Pointer to a `ggml_cgraph` structure for graph operations.
    - `res`: Pointer to a result structure that stores output tensors.
- **Description**: The `llm_build_internlm2` structure extends `llm_graph_context` and is designed to build a neural network model using various layers and operations, including attention mechanisms and feed-forward networks, while managing input embeddings and output logits.
- **Member Functions**:
    - [`llm_build_internlm2::llm_build_internlm2`](#llm_build_internlm2llm_build_internlm2)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_internlm2::llm\_build\_internlm2<!-- {{#callable:llm_build_internlm2::llm_build_internlm2}} -->
Constructs a neural network graph for an internal language model using specified model parameters.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` structure that contains the model parameters and weights.
    - `params`: A constant reference to `llm_graph_params` that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computational graph being built.
- **Control Flow**:
    - Initializes the number of embedding heads and asserts consistency in model parameters.
    - Builds input embeddings and position tensors.
    - Iterates over each layer of the model to construct the attention and feed-forward components.
    - For each layer, computes normalized inputs, applies self-attention, and processes the output through a feed-forward network.
    - Handles the last layer differently by skipping unused tokens and preparing the output.
    - Applies normalization and computes the final logits for the model output.
- **Output**: The function does not return a value but populates the `res` structure with the final embedding tensor and logits for the model.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_internlm2`](#llm_build_internlm2)  (Data Structure)



---
### llm\_build\_minicpm3<!-- {{#data_structure:llm_build_minicpm3}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds graph configuration parameters.
    - `gf`: Pointer to a `ggml_cgraph` structure used for graph operations.
- **Description**: The `llm_build_minicpm3` structure extends `llm_graph_context` and is designed to build a computational graph for a language model using various parameters and tensors. It initializes with a model and parameters, and processes input embeddings, attention mechanisms, and feed-forward networks through multiple layers, ultimately producing output logits for the model.
- **Member Functions**:
    - [`llm_build_minicpm3::llm_build_minicpm3`](#llm_build_minicpm3llm_build_minicpm3)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_minicpm3::llm\_build\_minicpm3<!-- {{#callable:llm_build_minicpm3::llm_build_minicpm3}} -->
Constructs a mini CPM model graph using the provided llama model and parameters.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and weights.
    - `params`: A constant reference to a `llm_graph_params` object that holds the graph parameters.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph.
- **Control Flow**:
    - Initializes embedding parameters and scales the input embeddings.
    - Builds input position embeddings and attention input tensors.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - Handles the scaling of hidden states for residual connections and prepares inputs for the next layer.
    - Applies final normalization and scaling before producing output logits.
- **Output**: Outputs the final logits tensor after processing through the model layers and applying necessary transformations.
- **Functions called**:
    - [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale)
    - [`ggml_mul_mat`](../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_view_3d`](../ggml/src/ggml.c.driver.md#ggml_view_3d)
    - [`ggml_row_size`](../ggml/src/ggml.c.driver.md#ggml_row_size)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_concat`](../ggml/src/ggml.c.driver.md#ggml_concat)
    - [`ggml_repeat`](../ggml/src/ggml.c.driver.md#ggml_repeat)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_minicpm3`](#llm_build_minicpm3)  (Data Structure)



---
### llm\_build\_gemma<!-- {{#data_structure:llm_build_gemma}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the model parameters used for building the graph.
    - `params`: Holds the parameters for the graph context.
    - `gf`: Pointer to the graph structure used in computations.
    - `res`: Stores the results of the computations.
- **Description**: `llm_build_gemma` is a structure that extends `llm_graph_context` and is responsible for building a graph representation of a language model using various input tensors, attention mechanisms, and feed-forward networks, while managing the normalization and scaling of the data throughout the layers.
- **Member Functions**:
    - [`llm_build_gemma::llm_build_gemma`](#llm_build_gemmallm_build_gemma)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_gemma::llm\_build\_gemma<!-- {{#callable:llm_build_gemma::llm_build_gemma}} -->
The `llm_build_gemma` function constructs a neural network graph for a language model using specified model parameters and input tensors.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` structure that contains the model parameters and weights.
    - `params`: A constant reference to `llm_graph_params` that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computational graph being built.
- **Control Flow**:
    - The function initializes the embedding input tensor and scales it based on the number of embedding heads.
    - It builds positional embeddings and initializes attention input tensors.
    - A loop iterates over the number of layers in the model, performing normalization, self-attention calculations, and feed-forward network operations for each layer.
    - Within the loop, it computes query (Q), key (K), and value (V) tensors, applies reshaping and rotary positional encoding, and performs attention operations.
    - For the last layer, it skips computing outputs for unused tokens and adds the results of the self-attention to the input.
    - After processing all layers, it normalizes the final output and computes the logits for the model's output.
- **Output**: The function does not return a value but populates the `res` structure with the final embedding tensor and logits, and expands the computational graph for further processing.
- **Functions called**:
    - [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_gemma`](#llm_build_gemma)  (Data Structure)



---
### llm\_build\_gemma2\_iswa<!-- {{#data_structure:llm_build_gemma2_iswa}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the model parameters used for building the structure.
    - `params`: Holds the graph parameters necessary for the context.
    - `gf`: Pointer to the graph structure used in computations.
    - `res`: Stores the results of the computations performed by the structure.
    - `ctx0`: Context used for managing memory and operations during the build process.
    - `hparams`: Hyperparameters that control various aspects of the model's behavior.
    - `n_layer`: Indicates the number of layers in the model.
    - `n_embd`: Represents the dimensionality of the embedding space.
    - `n_embd_head`: Specifies the number of embedding heads used in attention.
    - `n_head`: Number of attention heads in the model.
    - `n_head_kv`: Number of key-value pairs in the attention mechanism.
    - `n_tokens`: Total number of tokens processed by the model.
    - `n_rot`: Number of rotations applied in the attention mechanism.
    - `rope_type`: Type of rotation encoding used in the model.
    - `n_ctx_orig`: Original context length for the model.
    - `freq_base`: Base frequency for the rotation encoding.
    - `freq_scale`: Scale factor for the frequency in the rotation encoding.
    - `ext_factor`: Factor for extending the attention mechanism.
    - `attn_factor`: Factor that influences attention calculations.
    - `beta_fast`: Fast beta parameter for attention.
    - `beta_slow`: Slow beta parameter for attention.
- **Description**: The `llm_build_gemma2_iswa` structure is a specialized implementation of a neural network model that builds a graph context for processing input data through multiple layers of attention and feed-forward networks, utilizing various hyperparameters and model configurations to optimize performance and output results.
- **Member Functions**:
    - [`llm_build_gemma2_iswa::llm_build_gemma2_iswa`](#llm_build_gemma2_iswallm_build_gemma2_iswa)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_gemma2\_iswa::llm\_build\_gemma2\_iswa<!-- {{#callable:llm_build_gemma2_iswa::llm_build_gemma2_iswa}} -->
Constructs a neural network graph for the Gemma2 model using input embeddings, attention mechanisms, and feed-forward layers.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` structure that contains the model parameters and weights.
    - `params`: A constant reference to a `llm_graph_params` structure that holds the graph parameters for the model.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph being built.
- **Control Flow**:
    - Initializes the embedding input and scales it based on the number of embedding heads.
    - Builds positional embeddings and initializes attention inputs.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - For each layer, computes query (Q), key (K), and value (V) tensors, applying necessary transformations and scaling.
    - Handles the last layer differently by skipping unused tokens and adjusting the input accordingly.
    - Applies normalization and combines outputs from the self-attention and feed-forward networks.
    - Finalizes the output by applying normalization and scaling before expanding the graph.
- **Output**: The function outputs the final logits tensor after processing through the model layers, which is stored in the `res->t_logits`.
- **Functions called**:
    - [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_tanh`](../ggml/src/ggml.c.driver.md#ggml_tanh)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_gemma2_iswa`](#llm_build_gemma2_iswa)  (Data Structure)



---
### llm\_build\_gemma3\_iswa<!-- {{#data_structure:llm_build_gemma3_iswa}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds graph configuration.
    - `gf`: Pointer to a `ggml_cgraph` structure for managing computation graphs.
    - `res`: Pointer to a result structure that stores output tensors.
- **Description**: The `llm_build_gemma3_iswa` structure extends `llm_graph_context` and is designed to build a complex neural network architecture, specifically for a language model, by processing input embeddings, attention mechanisms, and feed-forward networks across multiple layers, while managing tensor operations and normalization.
- **Member Functions**:
    - [`llm_build_gemma3_iswa::llm_build_gemma3_iswa`](#llm_build_gemma3_iswallm_build_gemma3_iswa)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_gemma3\_iswa::llm\_build\_gemma3\_iswa<!-- {{#callable:llm_build_gemma3_iswa::llm_build_gemma3_iswa}} -->
The `llm_build_gemma3_iswa` function constructs a neural network graph for processing input embeddings through multiple layers of attention and feed-forward networks.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and weights used for building the graph.
    - `params`: A constant reference to a `llm_graph_params` object that holds the parameters for the graph construction.
    - `gf`: A pointer to a `ggml_cgraph` object that represents the computational graph being built.
- **Control Flow**:
    - The function initializes the embedding input tensor and scales it if necessary based on the presence of tokens.
    - It constructs positional embeddings and prepares the attention input.
    - A loop iterates over the number of layers in the model, performing normalization, self-attention calculations, and feed-forward operations for each layer.
    - Within each layer, it computes query (Q), key (K), and value (V) tensors, applies normalization, and performs rotary positional encoding.
    - The output of each layer is combined with the input and passed through additional normalization and feed-forward layers.
    - The final output tensor is normalized and passed through a linear transformation to produce logits.
- **Output**: The function outputs the final tensor of logits, which represents the model's predictions, and updates the graph with the constructed operations.
- **Functions called**:
    - [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_gemma3_iswa`](#llm_build_gemma3_iswa)  (Data Structure)



---
### llm\_build\_starcoder2<!-- {{#data_structure:llm_build_starcoder2}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the model parameters used for building the structure.
    - `params`: Holds the parameters for the graph context.
    - `gf`: Pointer to the graph structure used in computations.
    - `res`: Stores the results of the computations performed by the structure.
- **Description**: `llm_build_starcoder2` is a structure that extends `llm_graph_context` and is designed to build and manage the components of a neural network model, specifically for processing input embeddings, attention mechanisms, and feed-forward networks across multiple layers, while ensuring proper normalization and tensor operations.
- **Member Functions**:
    - [`llm_build_starcoder2::llm_build_starcoder2`](#llm_build_starcoder2llm_build_starcoder2)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_starcoder2::llm\_build\_starcoder2<!-- {{#callable:llm_build_starcoder2::llm_build_starcoder2}} -->
Constructs a neural network graph for the Starcoder model using the provided model parameters and graph context.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` structure that contains the model parameters and weights.
    - `params`: A constant reference to a `llm_graph_params` structure that holds the graph parameters.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph.
- **Control Flow**:
    - Initializes the number of embedding heads and asserts consistency in model parameters.
    - Builds input embeddings and position tensors.
    - Iterates over each layer of the model to perform normalization, self-attention, and feed-forward operations.
    - For each layer, computes query (Q), key (K), and value (V) tensors, applying necessary transformations and additions.
    - Handles the last layer differently by skipping unused tokens and preparing the output.
    - Applies normalization and computes the final output logits.
- **Output**: The function outputs the final tensor embeddings and logits for the model, stored in the `res` structure.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_starcoder2`](#llm_build_starcoder2)  (Data Structure)



---
### llm\_build\_mamba<!-- {{#data_structure:llm_build_mamba}} -->
- **Type**: `struct`
- **Members**:
    - `model`: A constant reference to a `llama_model` instance used for model parameters.
- **Description**: The `llm_build_mamba` structure extends `llm_graph_context` and is designed to facilitate the construction of a neural network architecture, specifically for the Mamba model, utilizing various tensor operations and layers defined in the `llama_model`. It manages the input embeddings, layer normalization, and the forward pass through multiple layers, while also handling the state management for recurrent connections.
- **Member Functions**:
    - [`llm_build_mamba::llm_build_mamba`](#llm_build_mamballm_build_mamba)
    - [`llm_build_mamba::build_mamba_layer`](#llm_build_mambabuild_mamba_layer)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_mamba::llm\_build\_mamba<!-- {{#callable:llm_build_mamba::llm_build_mamba}} -->
The `llm_build_mamba` constructor initializes a neural network model by building its layers and processing input tensors through a series of transformations.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model's parameters and layer configurations.
    - `params`: A constant reference to a `llm_graph_params` object that holds the parameters for the graph context.
    - `gf`: A pointer to a `ggml_cgraph` object that represents the computation graph for the model.
- **Control Flow**:
    - The function begins by initializing input embeddings using `build_inp_embd` with the model's token embeddings.
    - It creates state tensors for copying and masking using `build_inp_s_copy` and `build_inp_s_mask`.
    - A loop iterates over each layer of the model, performing normalization, building the Mamba layer, and applying residual connections.
    - For the last layer, it skips computing outputs for unused tokens by filtering the input and current tensors.
    - After processing all layers, it applies a final normalization and computes the output logits using `build_lora_mm`.
- **Output**: The function does not return a value directly; instead, it populates the `res` structure with the final embedded tensor and logits, which are the results of the model's forward pass.
- **Functions called**:
    - [`llm_build_mamba::build_mamba_layer`](#llm_build_mambabuild_mamba_layer)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_mamba`](#llm_build_mamba)  (Data Structure)


---
#### llm\_build\_mamba::build\_mamba\_layer<!-- {{#callable:llm_build_mamba::build_mamba_layer}} -->
The `build_mamba_layer` function constructs a layer in a Mamba architecture by processing input tensors through convolution and state management.
- **Inputs**:
    - `gf`: A pointer to a `ggml_cgraph` structure representing the computation graph.
    - `cur`: A pointer to a `ggml_tensor` representing the current input tensor to be processed.
    - `state_copy`: A pointer to a `ggml_tensor` used for copying the state during processing.
    - `state_mask`: A pointer to a `ggml_tensor` that serves as a mask for the state.
    - `ubatch`: A reference to a `llama_ubatch` structure containing information about the batch of sequences.
    - `il`: An integer representing the index of the current layer being processed.
- **Control Flow**:
    - The function begins by extracting parameters from the `mstate` and initializing various dimensions based on hyperparameters.
    - Assertions are made to ensure the validity of the input batch, including checks for sequence equality and token counts.
    - The function retrieves the current key and value states from the KV cache and prepares them for processing.
    - It reshapes the input tensor `cur` to match the expected dimensions for the current layer.
    - The function performs a series of tensor operations including convolution, state management, and normalization, applying specific transformations based on the Mamba architecture.
    - Intermediate results are computed through matrix multiplications and reshaping, with specific handling for different tensor dimensions.
    - The final output tensor is reshaped to match the expected output dimensions before being returned.
- **Output**: The function returns a pointer to a `ggml_tensor` representing the output of the Mamba layer after processing the input tensor through various transformations.
- **Functions called**:
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_view_3d`](../ggml/src/ggml.c.driver.md#ggml_view_3d)
    - [`ggml_element_size`](../ggml/src/ggml.c.driver.md#ggml_element_size)
    - [`ggml_concat`](../ggml/src/ggml.c.driver.md#ggml_concat)
    - [`ggml_transpose`](../ggml/src/ggml.c.driver.md#ggml_transpose)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
    - [`ggml_cpy`](../ggml/src/ggml.c.driver.md#ggml_cpy)
    - [`ggml_view_1d`](../ggml/src/ggml.c.driver.md#ggml_view_1d)
    - [`ggml_ssm_conv`](../ggml/src/ggml.c.driver.md#ggml_ssm_conv)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_silu`](../ggml/src/ggml.c.driver.md#ggml_silu)
    - [`ggml_rms_norm`](../ggml/src/ggml.c.driver.md#ggml_rms_norm)
    - [`ggml_ssm_scan`](../ggml/src/ggml.c.driver.md#ggml_ssm_scan)
    - [`ggml_mul`](../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_reshape_2d`](../ggml/src/ggml.c.driver.md#ggml_reshape_2d)
- **See also**: [`llm_build_mamba`](#llm_build_mamba)  (Data Structure)



---
### llm\_build\_command\_r<!-- {{#data_structure:llm_build_command_r}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to the `llama_model` used for building the command.
    - `params`: Reference to the `llm_graph_params` that contains parameters for the graph.
    - `gf`: Pointer to a `ggml_cgraph` structure used in the graph construction.
    - `res`: Pointer to a result structure that holds output tensors.
- **Description**: The `llm_build_command_r` structure is a derived type from `llm_graph_context` that encapsulates the process of building a command for a language model, utilizing various tensors and parameters to construct layers of attention and feed-forward networks, while managing input embeddings and output logits.
- **Member Functions**:
    - [`llm_build_command_r::llm_build_command_r`](#llm_build_command_rllm_build_command_r)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_command\_r::llm\_build\_command\_r<!-- {{#callable:llm_build_command_r::llm_build_command_r}} -->
Constructs a command for a language model by processing input embeddings through multiple layers of attention and feed-forward networks.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` structure that contains the model parameters and weights.
    - `params`: A constant reference to a `llm_graph_params` structure that holds the graph parameters for the model.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph for the model.
- **Control Flow**:
    - Initializes embedding and position tensors using helper functions.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - Computes query (Q), key (K), and value (V) tensors, applying normalization and rotary positional encoding as needed.
    - Handles the last layer differently by skipping computations for unused tokens.
    - Combines outputs from the feed-forward network and self-attention with residual connections.
    - Applies final normalization and computes logits for the output.
- **Output**: The function outputs the final embeddings and logits for the model, stored in the `res` structure.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_command_r`](#llm_build_command_r)  (Data Structure)



---
### llm\_build\_cohere2\_iswa<!-- {{#data_structure:llm_build_cohere2_iswa}} -->
- **Type**: `struct`
- **Members**:
    - `model`: The `llama_model` instance used for model parameters and embeddings.
    - `params`: The `llm_graph_params` instance containing configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure representing the computation graph.
- **Description**: The `llm_build_cohere2_iswa` structure extends `llm_graph_context` and is designed to build a neural network architecture for language modeling, utilizing various components such as attention mechanisms and feed-forward networks, while managing input embeddings and normalization across multiple layers.
- **Member Functions**:
    - [`llm_build_cohere2_iswa::llm_build_cohere2_iswa`](#llm_build_cohere2_iswallm_build_cohere2_iswa)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_cohere2\_iswa::llm\_build\_cohere2\_iswa<!-- {{#callable:llm_build_cohere2_iswa::llm_build_cohere2_iswa}} -->
Constructs a neural network graph for the Cohere2 model using the provided llama model and parameters.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model architecture and parameters.
    - `params`: A constant reference to a `llm_graph_params` object that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` object that represents the computation graph being built.
- **Control Flow**:
    - Initializes the number of embedding heads and asserts that the number of heads for keys matches that for values.
    - Builds the input embeddings and position tensors.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - For each layer, computes the query, key, and value tensors, applying any necessary biases and reshaping them.
    - If the layer is the last one, it skips computing outputs for unused tokens.
    - Adds the results of the self-attention and feed-forward network to the input for the next layer.
    - After processing all layers, applies normalization to the final output and computes the logits.
- **Output**: The function outputs the final tensor embeddings and logits, which are stored in the `res` object.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_cohere2_iswa`](#llm_build_cohere2_iswa)  (Data Structure)



---
### llm\_build\_olmo<!-- {{#data_structure:llm_build_olmo}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds configuration parameters.
    - `gf`: Pointer to a `ggml_cgraph` structure used for graph operations.
    - `res`: Pointer to a result structure that stores output tensors.
- **Description**: The `llm_build_olmo` structure is a derived class from `llm_graph_context` that encapsulates the construction of a neural network model's forward pass, utilizing various tensor operations and layers defined in the model. It initializes with a model and parameters, and processes input embeddings through multiple layers, applying normalization, self-attention, and feed-forward networks, ultimately producing output tensors for logits and embeddings.
- **Member Functions**:
    - [`llm_build_olmo::llm_build_olmo`](#llm_build_olmollm_build_olmo)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_olmo::llm\_build\_olmo<!-- {{#callable:llm_build_olmo::llm_build_olmo}} -->
The `llm_build_olmo` constructor initializes a neural network model by building its layers and processing input embeddings through self-attention and feed-forward networks.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model's parameters and weights.
    - `params`: A constant reference to `llm_graph_params` that holds configuration parameters for the graph context.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph for the model.
- **Control Flow**:
    - The function begins by asserting that the number of embedding heads is consistent across various parameters.
    - It initializes input embeddings and position tensors, and prepares attention inputs.
    - A loop iterates over the number of layers in the model, processing each layer sequentially.
    - Within each layer, it computes normalization, self-attention (including query, key, and value tensors), and applies clamping if necessary.
    - The attention outputs are reshaped and combined with positional encodings.
    - If processing the last layer, it skips unused tokens and prepares the output.
    - The feed-forward network is built and applied, followed by another normalization step.
    - The final output tensor is computed and stored in the result structure.
- **Output**: The function outputs the final logits tensor, which represents the model's predictions, and stores it in the result structure.
- **Functions called**:
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_olmo`](#llm_build_olmo)  (Data Structure)



---
### llm\_build\_olmo2<!-- {{#data_structure:llm_build_olmo2}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds configuration parameters.
    - `gf`: Pointer to a `ggml_cgraph` structure used for graph operations.
    - `res`: Pointer to a result structure that stores output tensors.
- **Description**: The `llm_build_olmo2` structure is a specialized implementation of a neural network layer builder that inherits from `llm_graph_context`, designed to construct and manage the forward pass of a transformer model, utilizing various tensor operations and configurations defined by the provided model and parameters.
- **Member Functions**:
    - [`llm_build_olmo2::llm_build_olmo2`](#llm_build_olmo2llm_build_olmo2)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_olmo2::llm\_build\_olmo2<!-- {{#callable:llm_build_olmo2::llm_build_olmo2}} -->
Constructs a neural network graph for the LLM model using the provided model parameters and input tensors.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model's parameters and weights.
    - `params`: A constant reference to `llm_graph_params` that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph being built.
- **Control Flow**:
    - Initializes the number of embedding heads and asserts consistency across model parameters.
    - Builds input embeddings and position tensors.
    - Iterates over each layer of the model to compute self-attention and feed-forward network outputs.
    - For each layer, computes query (Q), key (K), and value (V) tensors, applies normalization, and reshapes them.
    - Applies rotary positional encoding to Q and K tensors.
    - Builds attention outputs and applies normalization after attention.
    - Handles the last layer differently by skipping unused tokens.
    - Computes the feed-forward network output and applies normalization.
    - Adds the feed-forward output to the input for residual connections.
    - Finalizes the output by applying normalization and building the output layer.
- **Output**: The function does not return a value but populates the `res` structure with the final embedding tensor and logits for the model output.
- **Functions called**:
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_olmo2`](#llm_build_olmo2)  (Data Structure)



---
### llm\_build\_olmoe<!-- {{#data_structure:llm_build_olmoe}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds configuration parameters.
    - `gf`: Pointer to a `ggml_cgraph` structure used for graph operations.
    - `res`: Pointer to a result structure that stores output tensors.
- **Description**: The `llm_build_olmoe` structure is a derived class from `llm_graph_context` that encapsulates the construction of a neural network model, specifically designed for processing input embeddings, attention mechanisms, and feedforward networks, while managing the flow of data through multiple layers and applying necessary transformations and normalizations.
- **Member Functions**:
    - [`llm_build_olmoe::llm_build_olmoe`](#llm_build_olmoellm_build_olmoe)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_olmoe::llm\_build\_olmoe<!-- {{#callable:llm_build_olmoe::llm_build_olmoe}} -->
Constructs a layer-wise graph for a language model using attention and feed-forward mechanisms.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` structure that contains the model parameters and weights.
    - `params`: A constant reference to `llm_graph_params` that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph being built.
- **Control Flow**:
    - Initializes the number of embedding heads and asserts consistency across model parameters.
    - Builds input embeddings and positional encodings for the model.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - For each layer, computes query (Q), key (K), and value (V) tensors, applying normalization and reshaping as necessary.
    - Applies rotary positional encoding to Q and K tensors.
    - Handles the final layer differently by skipping unused tokens and preparing the output.
    - Constructs the feed-forward input by adding the current tensor to the input from the previous layer.
    - Processes the feed-forward network with a mixture of experts (MoE) mechanism.
    - Normalizes the output and prepares it for the final output layer.
    - Builds the final logits tensor and expands the computation graph.
- **Output**: The function outputs the final logits tensor representing the model's predictions, stored in the `res->t_logits`.
- **Functions called**:
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_olmoe`](#llm_build_olmoe)  (Data Structure)



---
### llm\_build\_openelm<!-- {{#data_structure:llm_build_openelm}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the llama model used for building the openelm.
    - `params`: Holds the parameters for the llm graph.
    - `gf`: Pointer to the ggml computation graph.
    - `res`: Stores the results of the computation.
- **Description**: `llm_build_openelm` is a structure that extends `llm_graph_context` and is designed to build a layer of a neural network model, specifically for processing input embeddings, attention mechanisms, and feed-forward networks, while managing the computation graph and normalization steps throughout the layers.
- **Member Functions**:
    - [`llm_build_openelm::llm_build_openelm`](#llm_build_openelmllm_build_openelm)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_openelm::llm\_build\_openelm<!-- {{#callable:llm_build_openelm::llm_build_openelm}} -->
Constructs a neural network graph for a language model using specified parameters and model weights.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model's parameters and weights.
    - `params`: A constant reference to a `llm_graph_params` object that holds the configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` object that represents the computational graph being built.
- **Control Flow**:
    - Initializes embedding and position tensors using `build_inp_embd` and `build_inp_pos`.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - For each layer, computes query, key, and value tensors, applies normalization, and processes them with attention mechanisms.
    - Handles the last layer differently by skipping unused tokens and computing the final output.
    - Applies normalization to the final output and builds the forward graph expansion.
- **Output**: The function does not return a value directly but modifies the `res` object to store the resulting embeddings and logits, which are used for further processing in the model.
- **Functions called**:
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_view_3d`](../ggml/src/ggml.c.driver.md#ggml_view_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_openelm`](#llm_build_openelm)  (Data Structure)



---
### llm\_build\_gptneox<!-- {{#data_structure:llm_build_gptneox}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the model parameters for the LLM.
    - `params`: Holds the parameters for the LLM graph.
    - `gf`: Pointer to the graph structure used in computations.
    - `n_embd_head`: Number of embedding heads used in the model.
    - `n_embd_gqa`: Number of embedding dimensions for the GQA.
    - `inpL`: Tensor representing the input layer.
    - `inp_pos`: Tensor containing positional embeddings.
    - `inp_attn`: Tensor for attention input.
    - `cur`: Current tensor being processed.
    - `n_layer`: Number of layers in the model.
    - `hparams`: Hyperparameters for the model.
    - `ctx0`: Context for tensor operations.
    - `res`: Result structure holding output tensors.
- **Description**: The `llm_build_gptneox` structure is designed to build and manage the components of a GPT-NeoX model, inheriting from `llm_graph_context`. It initializes various tensors and parameters necessary for the model's architecture, including input embeddings, attention mechanisms, and feedforward networks, while also handling the normalization and output layers across multiple layers of the model.
- **Member Functions**:
    - [`llm_build_gptneox::llm_build_gptneox`](#llm_build_gptneoxllm_build_gptneox)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_gptneox::llm\_build\_gptneox<!-- {{#callable:llm_build_gptneox::llm_build_gptneox}} -->
Constructs a GPT-NeoX model graph using the provided model parameters and input tensors.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model architecture and parameters.
    - `params`: A constant reference to a `llm_graph_params` object that holds the graph parameters for the model.
    - `gf`: A pointer to a `ggml_cgraph` object that represents the computation graph for the model.
- **Control Flow**:
    - Initializes embedding and position tensors using `build_inp_embd` and `build_inp_pos`.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - For each layer, computes the attention outputs and applies normalization, followed by either parallel or sequential feed-forward processing based on the model's configuration.
    - Handles the last layer differently by skipping unused tokens and preparing the output tensors.
    - Finalizes the output by applying normalization and building the final logits tensor.
- **Output**: The function constructs the model's computation graph and populates the output tensors `t_embd` and `t_logits` with the final embeddings and logits, respectively.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_gptneox`](#llm_build_gptneox)  (Data Structure)



---
### llm\_build\_arctic<!-- {{#data_structure:llm_build_arctic}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the `llama_model` used for building the architecture.
    - `params`: Holds the parameters of type `llm_graph_params` for the graph context.
    - `gf`: Pointer to a `ggml_cgraph` structure used for graph operations.
    - `res`: Stores the results of the computations, including embeddings and logits.
- **Description**: The `llm_build_arctic` structure extends `llm_graph_context` and is designed to build a neural network architecture based on the provided `llama_model` and `llm_graph_params`. It initializes various tensors and processes them through multiple layers, applying operations such as normalization, self-attention, and feed-forward networks, ultimately producing output embeddings and logits for further use in a machine learning context.
- **Member Functions**:
    - [`llm_build_arctic::llm_build_arctic`](#llm_build_arcticllm_build_arctic)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_arctic::llm\_build\_arctic<!-- {{#callable:llm_build_arctic::llm_build_arctic}} -->
The `llm_build_arctic` constructor initializes a neural network model by building its architecture layer by layer, incorporating attention mechanisms and feed-forward networks.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` structure that contains the model parameters and weights used for building the neural network.
    - `params`: A constant reference to `llm_graph_params` that holds configuration parameters for the graph context.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph for the model.
- **Control Flow**:
    - The function begins by asserting that the number of embedding heads is consistent across various parameters.
    - It initializes input embeddings and position embeddings, followed by a loop that iterates over each layer of the model.
    - Within the loop, it builds normalization layers, computes self-attention using query, key, and value tensors, and applies rotary positional encoding.
    - For the last layer, it skips computing outputs for unused tokens and prepares inputs for the feed-forward network.
    - It constructs the feed-forward network, applies normalization, and integrates mixture of experts (MoE) if applicable.
    - Finally, it normalizes the output and prepares the logits for the model's predictions.
- **Output**: The output consists of the final tensor embeddings and logits stored in the result structure, which represent the processed output of the neural network.
- **Functions called**:
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_arctic`](#llm_build_arctic)  (Data Structure)



---
### llm\_build\_deepseek<!-- {{#data_structure:llm_build_deepseek}} -->
- **Type**: `struct`
- **Members**:
    - `model`: The `llama_model` instance used for model parameters.
    - `params`: The `llm_graph_params` instance containing graph parameters.
    - `gf`: A pointer to a `ggml_cgraph` structure for graph operations.
    - `hparams`: Hyperparameters related to the model's architecture.
    - `n_layer`: The number of layers in the model.
    - `n_embd_head`: The number of embedding heads used in the model.
    - `n_expert`: The number of experts in the mixture of experts (MoE) layer.
    - `n_expert_used`: The number of experts that are actually used.
    - `res`: A structure to hold the results of the forward pass.
- **Description**: The `llm_build_deepseek` struct is a specialized data structure that extends `llm_graph_context` to implement a deep learning model architecture, specifically designed for processing input through multiple layers of attention and feedforward networks, utilizing various tensor operations and hyperparameters to manage the model's complexity and performance.
- **Member Functions**:
    - [`llm_build_deepseek::llm_build_deepseek`](#llm_build_deepseekllm_build_deepseek)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_deepseek::llm\_build\_deepseek<!-- {{#callable:llm_build_deepseek::llm_build_deepseek}} -->
Constructs a deep learning model graph for the Llama architecture using specified parameters and model weights.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model's parameters and weights.
    - `params`: A constant reference to a `llm_graph_params` object that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` object that represents the computational graph being built.
- **Control Flow**:
    - Initializes embedding and position tensors using `build_inp_embd` and `build_inp_pos`.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - Computes query (Q), key (K), and value (V) tensors, applying necessary transformations and RoPE (Rotary Positional Encoding).
    - Handles the last layer differently by skipping unused tokens and computing the output only for relevant tokens.
    - Applies feed-forward network (FFN) operations, including a mixture of experts (MoE) for certain layers.
    - Finalizes the output by applying normalization and computing logits for the model's predictions.
- **Output**: The function outputs the final tensor representing the model's logits and embeddings, which are stored in the `res` object.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_deepseek`](#llm_build_deepseek)  (Data Structure)



---
### llm\_build\_deepseek2<!-- {{#data_structure:llm_build_deepseek2}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` containing graph configuration.
    - `gf`: Pointer to a `ggml_cgraph` structure for graph operations.
    - `cur`: Pointer to a `ggml_tensor` representing the current tensor in processing.
    - `inpL`: Pointer to a `ggml_tensor` for input embeddings.
- **Description**: The `llm_build_deepseek2` struct extends `llm_graph_context` and is designed to build a deep learning model architecture, specifically for processing input embeddings and managing attention mechanisms across multiple layers, utilizing various tensor operations and configurations defined by the model and parameters.
- **Member Functions**:
    - [`llm_build_deepseek2::llm_build_deepseek2`](#llm_build_deepseek2llm_build_deepseek2)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_deepseek2::llm\_build\_deepseek2<!-- {{#callable:llm_build_deepseek2::llm_build_deepseek2}} -->
Constructs a deep learning model graph for the Llama architecture using specified parameters and model weights.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` structure containing the model's weights and architecture.
    - `params`: A constant reference to a `llm_graph_params` structure that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph being built.
- **Control Flow**:
    - Initializes various parameters based on the model's hyperparameters, including checks for model type (lite or MLA).
    - Builds input embeddings and position tensors for the model.
    - Iterates over each layer of the model, performing normalization, self-attention, and feedforward operations.
    - Handles different configurations for attention mechanisms based on whether the model is MLA or not.
    - Concatenates and reshapes tensors as necessary for the attention and feedforward layers.
    - Applies normalization and computes the final output logits after processing through all layers.
- **Output**: The function does not return a value directly; instead, it modifies the `res` structure to store the final embeddings and logits for the model.
- **Functions called**:
    - [`ggml_mul_mat`](../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_view_3d`](../ggml/src/ggml.c.driver.md#ggml_view_3d)
    - [`ggml_row_size`](../ggml/src/ggml.c.driver.md#ggml_row_size)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_permute`](../ggml/src/ggml.c.driver.md#ggml_permute)
    - [`ggml_concat`](../ggml/src/ggml.c.driver.md#ggml_concat)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_repeat`](../ggml/src/ggml.c.driver.md#ggml_repeat)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_deepseek2`](#llm_build_deepseek2)  (Data Structure)



---
### llm\_build\_bitnet<!-- {{#data_structure:llm_build_bitnet}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds graph configuration.
    - `gf`: Pointer to a `ggml_cgraph` structure for graph operations.
    - `res`: Pointer to a result structure that stores output tensors.
- **Description**: The `llm_build_bitnet` structure is a specialized type that extends `llm_graph_context`, designed to build and manage the forward pass of a neural network model, specifically for a transformer architecture. It initializes various tensors and layers based on the provided model and parameters, performing operations such as normalization, attention mechanisms, and feed-forward computations across multiple layers, ultimately producing output tensors for further processing.
- **Member Functions**:
    - [`llm_build_bitnet::llm_build_bitnet`](#llm_build_bitnetllm_build_bitnet)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_bitnet::llm\_build\_bitnet<!-- {{#callable:llm_build_bitnet::llm_build_bitnet}} -->
The `llm_build_bitnet` function constructs a bitnet model by processing input embeddings through multiple layers of attention and feed-forward networks.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` structure that contains the model parameters and weights used for building the bitnet.
    - `params`: A constant reference to a `llm_graph_params` structure that holds the parameters for the graph context.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph for the model.
- **Control Flow**:
    - The function begins by asserting that the number of embedding heads is consistent.
    - It initializes input embeddings and position tensors.
    - A loop iterates over each layer of the model, performing normalization, self-attention calculations, and feed-forward operations.
    - Within each layer, it computes query (Q), key (K), and value (V) tensors, applying scaling and biases as necessary.
    - The function reshapes Q, K, and V tensors and applies rotary positional encoding.
    - Attention outputs are computed and normalized before being passed to the feed-forward network.
    - The final output is normalized and passed through a linear transformation to produce logits.
- **Output**: The function outputs the final tensor embeddings and logits, which are stored in the `res` structure, and expands the computation graph for further processing.
- **Functions called**:
    - [`ggml_mul`](../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_bitnet`](#llm_build_bitnet)  (Data Structure)



---
### llm\_build\_t5\_enc<!-- {{#data_structure:llm_build_t5_enc}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to the `llama_model` used for building the encoder.
    - `params`: Reference to the `llm_graph_params` that contains parameters for the graph.
    - `gf`: Pointer to a `ggml_cgraph` structure used for graph operations.
    - `n_embd_head`: Integer representing the number of embedding heads.
    - `cur`: Pointer to the current `ggml_tensor` being processed.
    - `inpL`: Pointer to the input tensor for the layer.
    - `pos_bucket_enc`: Pointer to the positional bucket encoding tensor.
    - `inp_attn`: Pointer to the input tensor for attention without cache.
- **Description**: `llm_build_t5_enc` is a structure that extends `llm_graph_context` and is designed to build the encoder part of a T5 model, utilizing various tensors and layers to process input embeddings, apply self-attention mechanisms, and execute feed-forward networks across multiple layers, ultimately producing a normalized output tensor.
- **Member Functions**:
    - [`llm_build_t5_enc::llm_build_t5_enc`](#llm_build_t5_encllm_build_t5_enc)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_t5\_enc::llm\_build\_t5\_enc<!-- {{#callable:llm_build_t5_enc::llm_build_t5_enc}} -->
Constructs the T5 encoder by processing input embeddings through multiple layers of normalization, self-attention, and feed-forward networks.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and weights used for the T5 architecture.
    - `params`: A constant reference to a `llm_graph_params` object that holds the parameters for the graph context.
    - `gf`: A pointer to a `ggml_cgraph` object that represents the computation graph for the model.
- **Control Flow**:
    - Initializes the number of embedding heads and asserts that the number of embedding heads for keys is equal to that for values.
    - Builds the input embeddings and position bucket encodings.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - For each layer, computes the query, key, and value tensors, reshapes them, and applies attention mechanisms.
    - If processing the last layer, skips unused tokens by filtering the output.
    - Adds the output of the attention to the input of the layer and processes it through a feed-forward network.
    - Normalizes the output of the feed-forward network and adds it to the input.
    - Builds the final output tensor and normalizes it before storing it in the result.
- **Output**: The function does not return a value directly but modifies the `res` object to store the final tensor representation of the embeddings after processing through the T5 encoder.
- **Functions called**:
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_t5_enc`](#llm_build_t5_enc)  (Data Structure)



---
### llm\_build\_t5\_dec<!-- {{#data_structure:llm_build_t5_dec}} -->
- **Type**: `struct`
- **Members**:
    - `model`: The `llama_model` instance used for model parameters.
    - `params`: The `llm_graph_params` instance containing graph configuration parameters.
    - `gf`: A pointer to the `ggml_cgraph` structure used for graph operations.
- **Description**: The `llm_build_t5_dec` structure is a derived class from `llm_graph_context` that encapsulates the construction of a T5 model's decoding process, utilizing various tensors and layers to perform operations such as self-attention, cross-attention, and feed-forward networks, while managing the input and output tensors throughout the layers.
- **Member Functions**:
    - [`llm_build_t5_dec::llm_build_t5_dec`](#llm_build_t5_decllm_build_t5_dec)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_t5\_dec::llm\_build\_t5\_dec<!-- {{#callable:llm_build_t5_dec::llm_build_t5_dec}} -->
Constructs the T5 decoder architecture using the provided model and parameters.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and weights.
    - `params`: A constant reference to a `llm_graph_params` object that holds the graph parameters for the decoder.
    - `gf`: A pointer to a `ggml_cgraph` object that represents the computation graph for the model.
- **Control Flow**:
    - Initializes the embedding and positional bucket tensors using the model's token embeddings.
    - Iterates over each layer of the T5 decoder, performing normalization, self-attention, and cross-attention operations.
    - For each layer, computes the attention outputs and applies feed-forward networks, updating the current tensor state.
    - Handles the last layer differently by skipping unused tokens and preparing the final output tensors.
    - Applies normalization to the final output and computes the logits for the model's predictions.
- **Output**: The function outputs the final tensor embeddings and logits, which are stored in the `res` object.
- **Functions called**:
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_t5_dec`](#llm_build_t5_dec)  (Data Structure)



---
### llm\_build\_jais<!-- {{#data_structure:llm_build_jais}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds configuration parameters.
    - `gf`: Pointer to a `ggml_cgraph` structure used for graph operations.
    - `res`: Pointer to a result structure that stores output tensors.
- **Description**: The `llm_build_jais` structure extends `llm_graph_context` and is designed to build a neural network model using various layers and operations defined in the `llama_model`. It initializes the model with embedding and attention mechanisms, processes input through multiple layers, and computes the final output tensors while ensuring proper normalization and configuration as specified by the parameters.
- **Member Functions**:
    - [`llm_build_jais::llm_build_jais`](#llm_build_jaisllm_build_jais)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_jais::llm\_build\_jais<!-- {{#callable:llm_build_jais::llm_build_jais}} -->
Constructs a neural network graph for a language model using specified parameters and model weights.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` structure containing the model's parameters and weights.
    - `params`: A constant reference to `llm_graph_params` that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computational graph being built.
- **Control Flow**:
    - Initializes embedding and attention input tensors using the model's token embeddings.
    - Iterates over each layer of the model, performing normalization, self-attention, and feedforward operations.
    - For each layer, computes the query, key, and value tensors from the input, reshapes them, and applies attention mechanisms.
    - Handles the last layer differently by skipping unused tokens and adjusting the input accordingly.
    - Adds the input to the output of the attention layer before passing it through the feedforward network.
    - Normalizes the final output and computes the logits for the model's predictions.
- **Output**: The function does not return a value directly but modifies the `res` structure to store the final embedding tensor and logits, and expands the computational graph for further processing.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_jais`](#llm_build_jais)  (Data Structure)



---
### llm\_build\_chatglm<!-- {{#data_structure:llm_build_chatglm}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the model parameters used for building the chat model.
    - `params`: Holds the parameters for the graph context.
    - `gf`: Pointer to the graph structure used in computations.
    - `res`: Stores the results of the computations.
- **Description**: The `llm_build_chatglm` structure is a derived type from `llm_graph_context` that encapsulates the functionality for building a chat model using a specified llama model and graph parameters. It initializes various tensors and processes input through multiple layers, applying attention mechanisms and feedforward networks, ultimately producing output tensors that represent the model's predictions.
- **Member Functions**:
    - [`llm_build_chatglm::llm_build_chatglm`](#llm_build_chatglmllm_build_chatglm)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_chatglm::llm\_build\_chatglm<!-- {{#callable:llm_build_chatglm::llm_build_chatglm}} -->
Constructs a chat model using the provided llama model and parameters, building the necessary layers and attention mechanisms.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model architecture and parameters.
    - `params`: A constant reference to a `llm_graph_params` object that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` object that represents the computation graph for the model.
- **Control Flow**:
    - Initializes embedding and position tensors using the provided model.
    - Iterates over each layer of the model to build normalization, self-attention, and feed-forward components.
    - Handles both standard and LoRA (Low-Rank Adaptation) weight configurations for attention mechanisms.
    - Applies rotary positional encoding to the query and key tensors.
    - Computes the output for the last layer while skipping unused tokens.
    - Combines the input with the output of the feed-forward network and normalizes the result.
    - Finalizes the output by applying the output layer and expanding the computation graph.
- **Output**: The function does not return a value but populates the `res` structure with the final embedding tensor and logits tensor.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_chatglm`](#llm_build_chatglm)  (Data Structure)



---
### llm\_build\_glm4<!-- {{#data_structure:llm_build_glm4}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds configuration parameters.
    - `gf`: Pointer to a `ggml_cgraph` structure used for graph operations.
    - `res`: Pointer to a result structure that stores output tensors.
- **Description**: The `llm_build_glm4` structure extends `llm_graph_context` and is designed to build a graph for a language model using various input tensors and parameters. It initializes the model with embeddings, processes multiple layers through attention and feed-forward networks, and applies normalization at different stages, ultimately producing output logits and embeddings.
- **Member Functions**:
    - [`llm_build_glm4::llm_build_glm4`](#llm_build_glm4llm_build_glm4)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_glm4::llm\_build\_glm4<!-- {{#callable:llm_build_glm4::llm_build_glm4}} -->
Constructs a GLM-4 model graph using the provided llama model and parameters.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model architecture and parameters.
    - `params`: A constant reference to `llm_graph_params` that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph.
- **Control Flow**:
    - Initializes embedding and position tensors using `build_inp_embd` and `build_inp_pos`.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - For each layer, computes query, key, and value tensors, applying LoRA if necessary.
    - Applies rotary embeddings to the query and key tensors.
    - Handles residual connections and normalization after attention and feed-forward layers.
    - Finalizes the output by applying a normalization and projection to the result.
- **Output**: The function outputs the final logits tensor through the `res->t_logits` member, which is derived from the processed input embeddings.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_glm4`](#llm_build_glm4)  (Data Structure)



---
### llm\_build\_nemotron<!-- {{#data_structure:llm_build_nemotron}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds graph configuration.
    - `gf`: Pointer to a `ggml_cgraph` structure for graph operations.
    - `res`: Pointer to a result structure that stores output tensors.
- **Description**: The `llm_build_nemotron` structure extends `llm_graph_context` and is designed to build a neural network architecture based on the provided model and parameters, managing the input embeddings, attention mechanisms, and feed-forward networks across multiple layers, ultimately producing output tensors for further processing.
- **Member Functions**:
    - [`llm_build_nemotron::llm_build_nemotron`](#llm_build_nemotronllm_build_nemotron)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_nemotron::llm\_build\_nemotron<!-- {{#callable:llm_build_nemotron::llm_build_nemotron}} -->
Constructs a neural network graph for a language model using specified parameters and model weights.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` structure containing the model's parameters and weights.
    - `params`: A constant reference to a `llm_graph_params` structure that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computational graph being built.
- **Control Flow**:
    - Initializes the embedding input tensor using `build_inp_embd` with the model's token embeddings.
    - Builds the position input tensor using `build_inp_pos`.
    - Enters a loop over the number of layers in the model, performing normalization, self-attention, and feed-forward operations for each layer.
    - Within each layer, computes the query (Q), key (K), and value (V) tensors using `build_lora_mm`, and applies optional biases.
    - Reshapes Q, K, and V tensors and applies rotary positional encoding using [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext).
    - If processing the last layer, skips unused tokens by filtering the output using [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows).
    - Adds the current layer's output to the input of the next layer and processes it through a feed-forward network.
    - Normalizes the final output and applies the output layer transformation using `build_lora_mm`.
- **Output**: The function outputs the final logits tensor for the language model, stored in `res->t_logits`, and expands the graph for forward computation.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_nemotron`](#llm_build_nemotron)  (Data Structure)



---
### llm\_build\_exaone<!-- {{#data_structure:llm_build_exaone}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds graph configuration.
    - `gf`: Pointer to a `ggml_cgraph` structure for graph operations.
    - `res`: Pointer to a result structure that stores output tensors.
- **Description**: The `llm_build_exaone` structure is a derived class from `llm_graph_context` that encapsulates the construction of a neural network model, specifically designed for processing input embeddings, attention mechanisms, and feed-forward networks, while managing the necessary tensor operations and configurations for each layer of the model.
- **Member Functions**:
    - [`llm_build_exaone::llm_build_exaone`](#llm_build_exaonellm_build_exaone)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_exaone::llm\_build\_exaone<!-- {{#callable:llm_build_exaone::llm_build_exaone}} -->
Constructs a neural network graph for a language model using the provided model parameters and input tensors.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and weights necessary for building the graph.
    - `params`: A constant reference to a `llm_graph_params` object that holds the configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` object that represents the computational graph being built.
- **Control Flow**:
    - Initializes the embedding layer and position inputs using `build_inp_embd` and `build_inp_pos`.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - For each layer, computes the query (Q), key (K), and value (V) tensors, applying necessary transformations and additions.
    - Handles the last layer differently by skipping unused tokens and adjusting the output accordingly.
    - Applies normalization and builds the final output logits using the model's output layer.
- **Output**: The function does not return a value directly but populates the `res` structure with the final embedding tensor (`t_embd`) and the output logits tensor (`t_logits`).
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_exaone`](#llm_build_exaone)  (Data Structure)



---
### llm\_build\_rwkv6\_base<!-- {{#data_structure:llm_build_rwkv6_base}} -->
- **Type**: `struct`
- **Members**:
    - `model`: A constant reference to a `llama_model` instance used for model operations.
- **Description**: The `llm_build_rwkv6_base` struct extends `llm_graph_context` and is designed to facilitate the construction of RWKV6 model layers, utilizing a reference to a `llama_model` for its operations and parameters.
- **Member Functions**:
    - [`llm_build_rwkv6_base::llm_build_rwkv6_base`](#llm_build_rwkv6_basellm_build_rwkv6_base)
    - [`llm_build_rwkv6_base::build_rwkv6_channel_mix`](#llm_build_rwkv6_basebuild_rwkv6_channel_mix)
    - [`llm_build_rwkv6_base::build_rwkv6_time_mix`](#llm_build_rwkv6_basebuild_rwkv6_time_mix)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_rwkv6\_base::llm\_build\_rwkv6\_base<!-- {{#callable:llm_build_rwkv6_base::llm_build_rwkv6_base}} -->
Constructs an instance of `llm_build_rwkv6_base` using a reference to a `llama_model` and `llm_graph_params`.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that represents the model to be used.
    - `params`: A constant reference to a `llm_graph_params` object that contains parameters for the graph context.
- **Control Flow**:
    - The constructor initializes the base class `llm_graph_context` with the provided `params`.
    - It then initializes the member variable `model` with the provided `model` reference.
- **Output**: The constructor does not return a value but initializes an instance of `llm_build_rwkv6_base`.
- **See also**: [`llm_build_rwkv6_base`](#llm_build_rwkv6_base)  (Data Structure)


---
#### llm\_build\_rwkv6\_base::build\_rwkv6\_channel\_mix<!-- {{#callable:llm_build_rwkv6_base::build_rwkv6_channel_mix}} -->
This function builds a channel mix tensor for the RWKV6 architecture by applying various transformations to the input tensors.
- **Inputs**:
    - `layer`: A pointer to a `llama_layer` structure that contains the parameters for the channel mixing.
    - `cur`: A pointer to a `ggml_tensor` representing the current tensor state.
    - `x_prev`: A pointer to a `ggml_tensor` representing the previous tensor state.
    - `arch`: An enumeration value of type `llm_arch` that specifies the architecture type, which influences the processing logic.
- **Control Flow**:
    - The function first computes the difference tensor `sx` by subtracting `cur` from `x_prev`.
    - It then checks the architecture type using a switch statement, specifically handling the `LLM_ARCH_RWKV6` case.
    - Within the RWKV6 case, it computes intermediate tensors `xk` and `xr` using linear combinations of `sx` and `cur`.
    - The function applies a sigmoid activation to compute tensor `r` and a ReLU followed by squaring to compute tensor `k`.
    - Finally, it updates `cur` by multiplying `r` with the result of a linear transformation on `k`.
- **Output**: The function returns a pointer to a `ggml_tensor` that represents the updated state after applying the channel mixing transformations.
- **Functions called**:
    - [`ggml_sub`](../ggml/src/ggml.c.driver.md#ggml_sub)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_mul`](../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`ggml_sigmoid`](../ggml/src/ggml.c.driver.md#ggml_sigmoid)
    - [`ggml_sqr`](../ggml/src/ggml.c.driver.md#ggml_sqr)
    - [`ggml_relu`](../ggml/src/ggml.c.driver.md#ggml_relu)
- **See also**: [`llm_build_rwkv6_base`](#llm_build_rwkv6_base)  (Data Structure)


---
#### llm\_build\_rwkv6\_base::build\_rwkv6\_time\_mix<!-- {{#callable:llm_build_rwkv6_base::build_rwkv6_time_mix}} -->
The `build_rwkv6_time_mix` function processes input tensors to compute a time-mixed representation for a recurrent neural network.
- **Inputs**:
    - `gf`: A pointer to a `ggml_cgraph` structure representing the computation graph.
    - `cur`: A pointer to a `ggml_tensor` representing the current state tensor.
    - `x_prev`: A pointer to a `ggml_tensor` representing the previous input tensor.
    - `state_copy`: A pointer to a `ggml_tensor` used for copying the state.
    - `state_mask`: A pointer to a `ggml_tensor` representing the state mask.
    - `ubatch`: A reference to a `llama_ubatch` structure containing batch information such as number of tokens and sequences.
    - `il`: An integer representing the index of the layer being processed.
- **Control Flow**:
    - The function begins by extracting parameters from the `ubatch` and the model layer specified by `il`.
    - It computes the difference between the previous input and the current tensor, reshaping them for further processing.
    - A series of tensor operations are performed, including reshaping, matrix multiplication, and activation functions (like `tanh` and `sigmoid`).
    - Depending on whether the layer uses fused weights, it either combines tensors directly or applies additional linear transformations.
    - The function builds the key, value, and reception tensors using the `build_lora_mm` function, applying biases if they exist.
    - It computes a decay weight tensor and applies it to the key tensor if the layer is of a specific type.
    - The function then computes the output using either gated linear attention or a specific RWKV function based on the layer type.
    - Finally, it reshapes the output tensor and returns it.
- **Output**: The function returns a pointer to a `ggml_tensor` that represents the reshaped output of the time-mixed computation.
- **Functions called**:
    - [`ggml_sub`](../ggml/src/ggml.c.driver.md#ggml_sub)
    - [`ggml_reshape_2d`](../ggml/src/ggml.c.driver.md#ggml_reshape_2d)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_mul`](../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`ggml_reshape_4d`](../ggml/src/ggml.c.driver.md#ggml_reshape_4d)
    - [`ggml_tanh`](../ggml/src/ggml.c.driver.md#ggml_tanh)
    - [`ggml_mul_mat`](../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_permute`](../ggml/src/ggml.c.driver.md#ggml_permute)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_sigmoid`](../ggml/src/ggml.c.driver.md#ggml_sigmoid)
    - [`ggml_silu`](../ggml/src/ggml.c.driver.md#ggml_silu)
    - [`ggml_new_tensor_4d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_4d)
    - [`ggml_repeat`](../ggml/src/ggml.c.driver.md#ggml_repeat)
    - [`ggml_exp`](../ggml/src/ggml.c.driver.md#ggml_exp)
    - [`ggml_neg`](../ggml/src/ggml.c.driver.md#ggml_neg)
    - [`ggml_gated_linear_attn`](../ggml/src/ggml.c.driver.md#ggml_gated_linear_attn)
    - [`ggml_rwkv_wkv6`](../ggml/src/ggml.c.driver.md#ggml_rwkv_wkv6)
    - [`ggml_view_1d`](../ggml/src/ggml.c.driver.md#ggml_view_1d)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
    - [`ggml_cpy`](../ggml/src/ggml.c.driver.md#ggml_cpy)
    - [`ggml_element_size`](../ggml/src/ggml.c.driver.md#ggml_element_size)
    - [`ggml_norm`](../ggml/src/ggml.c.driver.md#ggml_norm)
- **See also**: [`llm_build_rwkv6_base`](#llm_build_rwkv6_base)  (Data Structure)



---
### llm\_build\_rwkv6<!-- {{#data_structure:llm_build_rwkv6}} -->
- **Type**: `struct`
- **Members**:
    - `model`: The `llama_model` instance used for model parameters.
    - `params`: The `llm_graph_params` instance containing graph configuration.
    - `gf`: A pointer to the `ggml_cgraph` structure for graph operations.
- **Description**: The `llm_build_rwkv6` structure extends `llm_build_rwkv6_base` and is designed to build a RWKV6 model using specified model parameters and graph configurations, facilitating the construction of neural network layers and tensor operations for processing input data.
- **Member Functions**:
    - [`llm_build_rwkv6::llm_build_rwkv6`](#llm_build_rwkv6llm_build_rwkv6)
- **Inherits From**:
    - [`llm_build_rwkv6_base::llm_build_rwkv6_base`](#llm_build_rwkv6_basellm_build_rwkv6_base)

**Methods**

---
#### llm\_build\_rwkv6::llm\_build\_rwkv6<!-- {{#callable:llm_build_rwkv6::llm_build_rwkv6}} -->
Constructs a RWKV6 model layer by layer, applying various transformations and normalizations to input tensors.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and layers.
    - `params`: A constant reference to a `llm_graph_params` object that holds the graph parameters for the model.
    - `gf`: A pointer to a `ggml_cgraph` object used for managing the computation graph.
- **Control Flow**:
    - Asserts that the token shift count in hyperparameters is equal to 2.
    - Initializes input tensors and builds the input embedding and normalization.
    - Iterates over each layer of the model, performing tensor reshaping, attention and feedforward operations, and normalizations.
    - Concatenates and shifts tensors as needed for each layer, storing intermediate results.
    - Handles the last layer differently by skipping unused tokens and preparing the output tensors.
    - Applies channel mixing and scaling based on hyperparameters.
    - Finalizes the output by normalizing and applying a linear transformation.
- **Output**: The function outputs the final tensor embeddings and logits, storing them in the result structure.
- **Functions called**:
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_view_3d`](../ggml/src/ggml.c.driver.md#ggml_view_3d)
    - [`ggml_element_size`](../ggml/src/ggml.c.driver.md#ggml_element_size)
    - [`ggml_concat`](../ggml/src/ggml.c.driver.md#ggml_concat)
    - [`llm_build_rwkv6_base::build_rwkv6_time_mix`](#llm_build_rwkv6_basebuild_rwkv6_time_mix)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_reshape_2d`](../ggml/src/ggml.c.driver.md#ggml_reshape_2d)
    - [`llm_build_rwkv6_base::build_rwkv6_channel_mix`](#llm_build_rwkv6_basebuild_rwkv6_channel_mix)
    - [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale)
- **See also**: [`llm_build_rwkv6`](#llm_build_rwkv6)  (Data Structure)



---
### llm\_build\_rwkv6qwen2<!-- {{#data_structure:llm_build_rwkv6qwen2}} -->
- **Type**: `struct`
- **Members**:
    - `model`: The `llama_model` instance used for model parameters.
    - `params`: The `llm_graph_params` instance containing graph configuration.
    - `gf`: A pointer to `ggml_cgraph` representing the computation graph.
- **Description**: The `llm_build_rwkv6qwen2` structure is a derived class from `llm_build_rwkv6_base` that encapsulates the construction of a RWKV model's computation graph, utilizing various tensor operations and layers defined in the model, while managing input embeddings, state copies, and normalization processes across multiple layers.
- **Member Functions**:
    - [`llm_build_rwkv6qwen2::llm_build_rwkv6qwen2`](#llm_build_rwkv6qwen2llm_build_rwkv6qwen2)
- **Inherits From**:
    - [`llm_build_rwkv6_base::llm_build_rwkv6_base`](#llm_build_rwkv6_basellm_build_rwkv6_base)

**Methods**

---
#### llm\_build\_rwkv6qwen2::llm\_build\_rwkv6qwen2<!-- {{#callable:llm_build_rwkv6qwen2::llm_build_rwkv6qwen2}} -->
Constructs a RWKV model layer by layer, applying normalization and feed-forward operations.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and layers.
    - `params`: A constant reference to `llm_graph_params` that holds the graph parameters for the model.
    - `gf`: A pointer to a `ggml_cgraph` structure used for building the computation graph.
- **Control Flow**:
    - Asserts that the embedding dimension matches the expected size.
    - Initializes input embeddings and state tensors.
    - Iterates over each layer of the model, performing operations such as reshaping inputs, applying normalization, and building feed-forward networks.
    - Handles the last layer differently by skipping unused tokens and preparing the output.
    - Applies normalization and builds the final output logits.
- **Output**: The function does not return a value directly but populates the `res` structure with the final embedded tensor and logits.
- **Functions called**:
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_concat`](../ggml/src/ggml.c.driver.md#ggml_concat)
    - [`ggml_view_3d`](../ggml/src/ggml.c.driver.md#ggml_view_3d)
    - [`llm_build_rwkv6_base::build_rwkv6_time_mix`](#llm_build_rwkv6_basebuild_rwkv6_time_mix)
    - [`ggml_element_size`](../ggml/src/ggml.c.driver.md#ggml_element_size)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_reshape_2d`](../ggml/src/ggml.c.driver.md#ggml_reshape_2d)
- **See also**: [`llm_build_rwkv6qwen2`](#llm_build_rwkv6qwen2)  (Data Structure)



---
### llm\_build\_rwkv7\_base<!-- {{#data_structure:llm_build_rwkv7_base}} -->
- **Type**: `struct`
- **Members**:
    - `model`: A constant reference to a `llama_model` instance used for model operations.
- **Description**: The `llm_build_rwkv7_base` struct inherits from `llm_graph_context` and is designed to facilitate the construction of RWKV7 model layers, utilizing a reference to a `llama_model` for its operations.
- **Member Functions**:
    - [`llm_build_rwkv7_base::llm_build_rwkv7_base`](#llm_build_rwkv7_basellm_build_rwkv7_base)
    - [`llm_build_rwkv7_base::build_rwkv7_channel_mix`](#llm_build_rwkv7_basebuild_rwkv7_channel_mix)
    - [`llm_build_rwkv7_base::build_rwkv7_time_mix`](#llm_build_rwkv7_basebuild_rwkv7_time_mix)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_rwkv7\_base::llm\_build\_rwkv7\_base<!-- {{#callable:llm_build_rwkv7_base::llm_build_rwkv7_base}} -->
Constructs an instance of `llm_build_rwkv7_base` using a given `llama_model` and `llm_graph_params`.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model's parameters and structure.
    - `params`: A constant reference to a `llm_graph_params` object that holds the parameters for the graph context.
- **Control Flow**:
    - The constructor initializes the base class `llm_graph_context` with the provided `params`.
    - It then initializes the member variable `model` with the provided `model` reference.
- **Output**: The constructor does not return a value but initializes an instance of `llm_build_rwkv7_base`.
- **See also**: [`llm_build_rwkv7_base`](#llm_build_rwkv7_base)  (Data Structure)


---
#### llm\_build\_rwkv7\_base::build\_rwkv7\_channel\_mix<!-- {{#callable:llm_build_rwkv7_base::build_rwkv7_channel_mix}} -->
Builds a channel mix tensor for the RWKV7 architecture using the provided layer and input tensors.
- **Inputs**:
    - `layer`: A pointer to a `llama_layer` structure that contains parameters for the channel mixing.
    - `cur`: A pointer to a `ggml_tensor` representing the current tensor state.
    - `x_prev`: A pointer to a `ggml_tensor` representing the previous tensor state.
    - `arch`: An enumeration value of type `llm_arch` indicating the architecture type, specifically RWKV7 in this case.
- **Control Flow**:
    - Calculates the difference tensor `sx` by subtracting `cur` from `x_prev`.
    - Checks the architecture type using a switch statement; if it matches `LLM_ARCH_RWKV7`, it proceeds with the channel mixing calculations.
    - Computes an intermediate tensor `xk` by scaling `sx` with a mixing parameter and adding `cur`.
    - Generates a tensor `k` by applying a series of transformations including matrix multiplication, ReLU activation, and squaring.
    - Updates the `cur` tensor by applying another matrix multiplication with the `channel_mix_value` parameter.
    - If the architecture does not match, it triggers a fatal error.
- **Output**: Returns a pointer to the updated `cur` tensor after applying the channel mixing operations.
- **Functions called**:
    - [`ggml_sub`](../ggml/src/ggml.c.driver.md#ggml_sub)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_mul`](../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`ggml_sqr`](../ggml/src/ggml.c.driver.md#ggml_sqr)
    - [`ggml_relu`](../ggml/src/ggml.c.driver.md#ggml_relu)
- **See also**: [`llm_build_rwkv7_base`](#llm_build_rwkv7_base)  (Data Structure)


---
#### llm\_build\_rwkv7\_base::build\_rwkv7\_time\_mix<!-- {{#callable:llm_build_rwkv7_base::build_rwkv7_time_mix}} -->
The `build_rwkv7_time_mix` function processes input tensors to compute a mixed representation for a recurrent neural network layer.
- **Inputs**:
    - `gf`: A pointer to a `ggml_cgraph` structure representing the computation graph.
    - `cur`: A pointer to a `ggml_tensor` representing the current tensor state.
    - `x_prev`: A pointer to a `ggml_tensor` representing the previous input tensor.
    - `state_copy`: A pointer to a `ggml_tensor` used for copying the state.
    - `state_mask`: A pointer to a `ggml_tensor` used for masking the state.
    - `first_layer_value`: A reference to a pointer to a `ggml_tensor` that stores the first layer's value.
    - `ubatch`: A constant reference to a `llama_ubatch` structure containing batch information.
    - `il`: An integer representing the index of the layer being processed.
- **Control Flow**:
    - The function begins by extracting parameters from the `ubatch` and the model layer specified by `il`.
    - It computes the difference between the current tensor and the previous input tensor, and prepares a dummy tensor for further operations.
    - The function then performs a series of tensor operations including addition, multiplication, and reshaping to compute intermediate values.
    - It checks for gating conditions and applies gating mechanisms if necessary.
    - The function constructs the key, value, and reception tensors using learned weights and applies activation functions.
    - It handles residual connections for the first layer value and computes the final output tensor.
    - Finally, it applies normalization if specified and returns the reshaped output tensor.
- **Output**: The function returns a pointer to a `ggml_tensor` that represents the processed output of the recurrent layer, reshaped to match the expected dimensions.
- **Functions called**:
    - [`ggml_sub`](../ggml/src/ggml.c.driver.md#ggml_sub)
    - [`ggml_new_tensor_4d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_4d)
    - [`ggml_repeat`](../ggml/src/ggml.c.driver.md#ggml_repeat)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_mul`](../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_mul_mat`](../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_tanh`](../ggml/src/ggml.c.driver.md#ggml_tanh)
    - [`ggml_exp`](../ggml/src/ggml.c.driver.md#ggml_exp)
    - [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale)
    - [`ggml_sigmoid`](../ggml/src/ggml.c.driver.md#ggml_sigmoid)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_l2_norm`](../ggml/src/ggml.c.driver.md#ggml_l2_norm)
    - [`ggml_rwkv_wkv7`](../ggml/src/ggml.c.driver.md#ggml_rwkv_wkv7)
    - [`ggml_neg`](../ggml/src/ggml.c.driver.md#ggml_neg)
    - [`ggml_view_1d`](../ggml/src/ggml.c.driver.md#ggml_view_1d)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
    - [`ggml_cpy`](../ggml/src/ggml.c.driver.md#ggml_cpy)
    - [`ggml_element_size`](../ggml/src/ggml.c.driver.md#ggml_element_size)
    - [`ggml_norm`](../ggml/src/ggml.c.driver.md#ggml_norm)
    - [`ggml_reshape_2d`](../ggml/src/ggml.c.driver.md#ggml_reshape_2d)
    - [`ggml_sum_rows`](../ggml/src/ggml.c.driver.md#ggml_sum_rows)
- **See also**: [`llm_build_rwkv7_base`](#llm_build_rwkv7_base)  (Data Structure)



---
### llm\_build\_rwkv7<!-- {{#data_structure:llm_build_rwkv7}} -->
- **Type**: `struct`
- **Members**:
    - `model`: The `llama_model` instance used for model parameters.
    - `params`: The `llm_graph_params` instance containing graph parameters.
    - `gf`: A pointer to a `ggml_cgraph` structure used for graph operations.
    - `res`: A structure to hold the results of the computation.
- **Description**: The `llm_build_rwkv7` structure is a derived class from `llm_build_rwkv7_base` that encapsulates the construction and processing of a RWKV7 model, utilizing various tensor operations and layers defined in the model, while managing input embeddings, normalization, and output logits.
- **Member Functions**:
    - [`llm_build_rwkv7::llm_build_rwkv7`](#llm_build_rwkv7llm_build_rwkv7)
- **Inherits From**:
    - [`llm_build_rwkv7_base::llm_build_rwkv7_base`](#llm_build_rwkv7_basellm_build_rwkv7_base)

**Methods**

---
#### llm\_build\_rwkv7::llm\_build\_rwkv7<!-- {{#callable:llm_build_rwkv7::llm_build_rwkv7}} -->
Constructs a RWKV-7 model graph using the provided llama model and parameters.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model architecture and parameters.
    - `params`: A constant reference to a `llm_graph_params` object that holds the graph parameters for the model.
    - `gf`: A pointer to a `ggml_cgraph` object that represents the computation graph for the model.
- **Control Flow**:
    - Asserts that the token shift count in the hyperparameters is equal to 2.
    - Initializes input tensors and builds the input embedding and normalization.
    - Iterates over each layer of the model, reshaping inputs and building necessary tensors for attention and feedforward operations.
    - Constructs token shifts and applies normalization for attention and feedforward inputs.
    - Concatenates previous outputs and current inputs to prepare for the next layer.
    - Handles the final layer differently by skipping unused tokens and reshaping outputs.
    - Applies final normalization and builds the output logits.
- **Output**: The function does not return a value directly but populates the `res` structure with the final embedded tensor and logits after processing through the model.
- **Functions called**:
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_view_3d`](../ggml/src/ggml.c.driver.md#ggml_view_3d)
    - [`ggml_element_size`](../ggml/src/ggml.c.driver.md#ggml_element_size)
    - [`ggml_concat`](../ggml/src/ggml.c.driver.md#ggml_concat)
    - [`llm_build_rwkv7_base::build_rwkv7_time_mix`](#llm_build_rwkv7_basebuild_rwkv7_time_mix)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_reshape_2d`](../ggml/src/ggml.c.driver.md#ggml_reshape_2d)
    - [`llm_build_rwkv7_base::build_rwkv7_channel_mix`](#llm_build_rwkv7_basebuild_rwkv7_channel_mix)
- **See also**: [`llm_build_rwkv7`](#llm_build_rwkv7)  (Data Structure)



---
### llm\_build\_arwkv7<!-- {{#data_structure:llm_build_arwkv7}} -->
- **Type**: `struct`
- **Members**:
    - `model`: The `llama_model` instance used for model parameters.
    - `params`: The `llm_graph_params` instance containing graph parameters.
    - `gf`: A pointer to a `ggml_cgraph` structure for graph operations.
- **Description**: The `llm_build_arwkv7` structure is a derived class from `llm_build_rwkv7_base` that encapsulates the construction of a neural network model, specifically designed for processing input embeddings and managing the forward pass through multiple layers of the model, utilizing various tensor operations and normalization techniques.
- **Member Functions**:
    - [`llm_build_arwkv7::llm_build_arwkv7`](#llm_build_arwkv7llm_build_arwkv7)
- **Inherits From**:
    - [`llm_build_rwkv7_base::llm_build_rwkv7_base`](#llm_build_rwkv7_basellm_build_rwkv7_base)

**Methods**

---
#### llm\_build\_arwkv7::llm\_build\_arwkv7<!-- {{#callable:llm_build_arwkv7::llm_build_arwkv7}} -->
Constructs a RWKV model layer by layer, processing input embeddings and applying normalization and feed-forward operations.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and layers.
    - `params`: A constant reference to `llm_graph_params` that holds the graph parameters for the model.
    - `gf`: A pointer to a `ggml_cgraph` structure used for managing the computation graph.
- **Control Flow**:
    - Asserts that the number of embedding dimensions matches the expected parameter.
    - Initializes input embeddings and state tensors for processing.
    - Iterates over each layer of the model, reshaping inputs and applying various transformations.
    - Loads token shifts and applies normalization to attention outputs.
    - Concatenates previous outputs with current attention outputs for further processing.
    - Applies feed-forward network operations and normalization to the outputs.
    - Handles the final layer differently by skipping unused tokens and preparing the output.
    - Builds the final output logits and expands the computation graph.
- **Output**: The function does not return a value directly but populates the `res` structure with the final embedded tensor and logits for the model output.
- **Functions called**:
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_concat`](../ggml/src/ggml.c.driver.md#ggml_concat)
    - [`ggml_view_3d`](../ggml/src/ggml.c.driver.md#ggml_view_3d)
    - [`llm_build_rwkv7_base::build_rwkv7_time_mix`](#llm_build_rwkv7_basebuild_rwkv7_time_mix)
    - [`ggml_element_size`](../ggml/src/ggml.c.driver.md#ggml_element_size)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_reshape_2d`](../ggml/src/ggml.c.driver.md#ggml_reshape_2d)
- **See also**: [`llm_build_arwkv7`](#llm_build_arwkv7)  (Data Structure)



---
### llm\_build\_granite<!-- {{#data_structure:llm_build_granite}} -->
- **Type**: `struct`
- **Members**:
    - `model`: The `llama_model` instance used for model parameters.
    - `params`: The `llm_graph_params` instance containing configuration parameters.
    - `gf`: A pointer to a `ggml_cgraph` structure used for graph operations.
    - `use_rope`: A boolean flag indicating whether to use rotary positional encoding.
- **Description**: The `llm_build_granite` structure extends `llm_graph_context` and is designed to build a neural network architecture for language models, utilizing various components such as embeddings, attention mechanisms, and feed-forward networks, while also supporting optional rotary positional encoding.
- **Member Functions**:
    - [`llm_build_granite::llm_build_granite`](#llm_build_granitellm_build_granite)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_granite::llm\_build\_granite<!-- {{#callable:llm_build_granite::llm_build_granite}} -->
The `llm_build_granite` function constructs a neural network graph for a language model using specified parameters and an optional RoPE mechanism.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model's parameters and weights.
    - `params`: A constant reference to `llm_graph_params` that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph being built.
    - `use_rope`: A boolean flag indicating whether to use Rotary Positional Encoding (RoPE) in the model.
- **Control Flow**:
    - The function begins by asserting that the number of embedding heads is consistent across various parameters.
    - It initializes input embeddings and optionally positional embeddings based on the `use_rope` flag.
    - A loop iterates over the number of layers in the model, performing normalization, self-attention, and feed-forward operations for each layer.
    - Within the self-attention block, it computes query (Q), key (K), and value (V) tensors, applying RoPE if enabled.
    - The output of the attention mechanism is scaled and combined with the input for the next layer.
    - The feed-forward network is executed, with a conditional path for models using Mixture of Experts (MoE).
    - After processing all layers, the final output is normalized and passed through a linear transformation to produce logits.
- **Output**: The function outputs the final tensor embeddings and logits, which are stored in the `res` structure, and expands the computation graph for further processing.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_granite`](#llm_build_granite)  (Data Structure)



---
### llm\_build\_chameleon<!-- {{#data_structure:llm_build_chameleon}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the model parameters used for building the chameleon.
    - `params`: Holds the parameters for the graph context.
    - `gf`: Pointer to the graph structure used in computations.
    - `hparams`: Hyperparameters related to the model's architecture.
    - `n_layer`: Number of layers in the model.
    - `res`: Results structure to store output tensors.
- **Description**: The `llm_build_chameleon` struct is a specialized data structure that extends `llm_graph_context` to facilitate the construction of a complex neural network architecture, specifically designed for processing input embeddings, attention mechanisms, and feed-forward networks across multiple layers, while managing various tensor operations and hyperparameters.
- **Member Functions**:
    - [`llm_build_chameleon::llm_build_chameleon`](#llm_build_chameleonllm_build_chameleon)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_chameleon::llm\_build\_chameleon<!-- {{#callable:llm_build_chameleon::llm_build_chameleon}} -->
Constructs a chameleon model for language processing by building a series of layers with attention and feed-forward networks.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and weights.
    - `params`: A constant reference to `llm_graph_params` that holds the configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph for the model.
- **Control Flow**:
    - Initializes the number of embedding heads and asserts their consistency with model parameters.
    - Builds input embeddings and positional encodings.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - For each layer, computes query (Q), key (K), and value (V) tensors, applying normalization if specified.
    - Applies rotary positional encoding to Q and K tensors.
    - Constructs the attention output and applies normalization if required.
    - Handles the last layer differently by skipping unused tokens and preparing the output.
    - Adds the feed-forward input to the current tensor and processes it through the feed-forward network.
    - Normalizes the output of the feed-forward network if not using Swin normalization.
    - Builds the final output tensor and applies a linear transformation to produce logits.
- **Output**: The function does not return a value directly but populates the `res` structure with the final embedding tensor and logits for the model output.
- **Functions called**:
    - [`ggml_view_3d`](../ggml/src/ggml.c.driver.md#ggml_view_3d)
    - [`ggml_element_size`](../ggml/src/ggml.c.driver.md#ggml_element_size)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_new_tensor_1d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_set_1d`](../ggml/src/ggml.c.driver.md#ggml_set_1d)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_chameleon`](#llm_build_chameleon)  (Data Structure)



---
### llm\_build\_wavtokenizer\_dec<!-- {{#data_structure:llm_build_wavtokenizer_dec}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` object used for model parameters.
    - `params`: Reference to a `llm_graph_params` object containing graph parameters.
    - `gf`: Pointer to a `ggml_cgraph` object used for graph operations.
- **Description**: The `llm_build_wavtokenizer_dec` structure extends `llm_graph_context` and is designed to build a wave tokenizer model using various layers and operations defined in the `llama_model`. It initializes the model with input embeddings, processes through multiple layers including convolutional and normalization layers, and constructs the final output embeddings, facilitating the transformation of input data into a structured format suitable for further processing.
- **Member Functions**:
    - [`llm_build_wavtokenizer_dec::llm_build_wavtokenizer_dec`](#llm_build_wavtokenizer_decllm_build_wavtokenizer_dec)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_wavtokenizer\_dec::llm\_build\_wavtokenizer\_dec<!-- {{#callable:llm_build_wavtokenizer_dec::llm_build_wavtokenizer_dec}} -->
Constructs a wave tokenizer decoder using a llama model and graph parameters.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and layers.
    - `params`: A constant reference to `llm_graph_params` that holds the graph configuration parameters.
    - `gf`: A pointer to a `ggml_cgraph` structure used for building the computation graph.
- **Control Flow**:
    - Initializes input embeddings using `build_inp_embd` with the model's token embeddings.
    - Transposes the input tensor and applies a 1D convolution followed by an addition of a bias term.
    - Iterates through the positional network layers, applying normalization, convolutions, and attention mechanisms based on the layer index.
    - For specific layers, applies different operations including normalization, activation functions, and residual connections.
    - After processing the positional network, transposes the tensor and applies normalization with the model's token normalization parameters.
    - Processes the convolutional next layers similarly, applying depthwise convolutions, normalization, and feedforward networks.
    - Finalizes the output by applying normalization and a linear transformation to produce the result embedding.
- **Output**: The function outputs a tensor representing the result embedding, which is stored in `res->t_embd` and is also expanded in the computation graph.
- **Functions called**:
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_transpose`](../ggml/src/ggml.c.driver.md#ggml_transpose)
    - [`ggml_conv_1d_ph`](../ggml/src/ggml.c.driver.md#ggml_conv_1d_ph)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_mul`](../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`ggml_sigmoid`](../ggml/src/ggml.c.driver.md#ggml_sigmoid)
    - [`ggml_mul_mat`](../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_soft_max_ext`](../ggml/src/ggml.c.driver.md#ggml_soft_max_ext)
    - [`ggml_conv_1d_dw_ph`](../ggml/src/ggml.c.driver.md#ggml_conv_1d_dw_ph)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_wavtokenizer_dec`](#llm_build_wavtokenizer_dec)  (Data Structure)



---
### llm\_build\_plm<!-- {{#data_structure:llm_build_plm}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Reference to a `llama_model` instance used for model parameters.
    - `params`: Reference to `llm_graph_params` that holds graph configuration.
    - `gf`: Pointer to a `ggml_cgraph` structure for managing computation graphs.
    - `hparams`: Contains hyperparameters related to the model's architecture.
    - `n_layer`: Indicates the number of layers in the model.
    - `res`: Pointer to a result structure that stores output tensors.
- **Description**: The `llm_build_plm` structure extends `llm_graph_context` and is designed to build a language model by processing input embeddings, attention mechanisms, and feedforward networks across multiple layers, utilizing various tensor operations and hyperparameters to manage the model's architecture and output.
- **Member Functions**:
    - [`llm_build_plm::llm_build_plm`](#llm_build_plmllm_build_plm)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_plm::llm\_build\_plm<!-- {{#callable:llm_build_plm::llm_build_plm}} -->
Constructs a transformer model's forward pass using the provided llama model and parameters.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model's parameters and weights.
    - `params`: A constant reference to `llm_graph_params` that holds configuration parameters for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure used for managing the computation graph.
- **Control Flow**:
    - Initializes scaling factors and tensor dimensions based on model parameters.
    - Builds input embeddings and position tensors.
    - Iterates over each layer of the model, performing normalization, self-attention, and feed-forward operations.
    - For each layer, computes query, key, and value tensors, applying necessary transformations and concatenations.
    - Handles the last layer differently by skipping unused tokens and preparing the output.
    - Applies normalization and final transformations to produce the output logits.
- **Output**: The function outputs the final logits tensor and the embedded tensor, which are stored in the `res` structure.
- **Functions called**:
    - [`ggml_mul_mat`](../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_view_3d`](../ggml/src/ggml.c.driver.md#ggml_view_3d)
    - [`ggml_row_size`](../ggml/src/ggml.c.driver.md#ggml_row_size)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_concat`](../ggml/src/ggml.c.driver.md#ggml_concat)
    - [`ggml_repeat`](../ggml/src/ggml.c.driver.md#ggml_repeat)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_plm`](#llm_build_plm)  (Data Structure)



---
### llm\_build\_bailingmoe<!-- {{#data_structure:llm_build_bailingmoe}} -->
- **Type**: `struct`
- **Members**:
    - `model`: Contains the `llama_model` instance used for model parameters.
    - `params`: Holds the `llm_graph_params` configuration for the graph.
    - `gf`: Pointer to the `ggml_cgraph` structure used for graph operations.
    - `res`: Stores the results of the computations performed by the structure.
- **Description**: The `llm_build_bailingmoe` structure extends `llm_graph_context` and is designed to build and manage the forward pass of a neural network model, specifically for a Llama model, by processing input embeddings, attention mechanisms, and feedforward networks across multiple layers, while utilizing various tensor operations and configurations.
- **Member Functions**:
    - [`llm_build_bailingmoe::llm_build_bailingmoe`](#llm_build_bailingmoellm_build_bailingmoe)
- **Inherits From**:
    - [`llm_graph_context`](llama-graph.h.driver.md#llm_graph_context)

**Methods**

---
#### llm\_build\_bailingmoe::llm\_build\_bailingmoe<!-- {{#callable:llm_build_bailingmoe::llm_build_bailingmoe}} -->
The `llm_build_bailingmoe` function constructs a neural network graph for a language model using a series of layers, including attention and feedforward networks, while applying normalization and expert gating mechanisms.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model parameters and weights necessary for building the graph.
    - `params`: A constant reference to `llm_graph_params` that holds configuration parameters for the graph construction.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computational graph being built.
- **Control Flow**:
    - The function initializes input embeddings and position tensors using helper functions.
    - It iterates over each layer of the model, performing normalization, self-attention calculations, and feedforward operations.
    - Within each layer, it computes query (Q), key (K), and value (V) tensors, applying necessary transformations and RoPE (Rotary Positional Encoding) adjustments.
    - It conditionally skips output computation for unused tokens in the last layer.
    - The function combines outputs from the attention and feedforward networks, applies normalization, and prepares the final output tensors.
    - Finally, it expands the graph with the computed output tensor.
- **Output**: The function does not return a value directly; instead, it modifies the `res` structure to store the final embeddings and logits, which are used for further processing in the language model.
- **Functions called**:
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`llm_build_bailingmoe`](#llm_build_bailingmoe)  (Data Structure)



---
### llama\_model<!-- {{#data_structure:llama_model}} -->
- **Description**: [See definition](llama-model.h.driver.md#llama_model)
- **Member Functions**:
    - [`llama_model::llama_model`](#llama_modelllama_model)
    - [`llama_model::~llama_model`](#llama_modelllama_model)
    - [`llama_model::load_stats`](#llama_modelload_stats)
    - [`llama_model::load_arch`](#llama_modelload_arch)
    - [`llama_model::load_hparams`](#llama_modelload_hparams)
    - [`llama_model::load_vocab`](#llama_modelload_vocab)
    - [`llama_model::load_tensors`](#llama_modelload_tensors)
    - [`llama_model::arch_name`](#llama_modelarch_name)
    - [`llama_model::type_name`](#llama_modeltype_name)
    - [`llama_model::desc`](#llama_modeldesc)
    - [`llama_model::size`](#llama_modelsize)
    - [`llama_model::n_tensors`](#llama_modeln_tensors)
    - [`llama_model::n_devices`](#llama_modeln_devices)
    - [`llama_model::n_elements`](#llama_modeln_elements)
    - [`llama_model::print_info`](#llama_modelprint_info)
    - [`llama_model::dev_layer`](#llama_modeldev_layer)
    - [`llama_model::dev_output`](#llama_modeldev_output)
    - [`llama_model::select_buft`](#llama_modelselect_buft)
    - [`llama_model::has_tensor_overrides`](#llama_modelhas_tensor_overrides)
    - [`llama_model::get_tensor`](#llama_modelget_tensor)
    - [`llama_model::get_rope_freq_base`](#llama_modelget_rope_freq_base)
    - [`llama_model::get_rope_freq_scale`](#llama_modelget_rope_freq_scale)
    - [`llama_model::get_rope_factors`](#llama_modelget_rope_factors)
    - [`llama_model::create_memory`](#llama_modelcreate_memory)
    - [`llama_model::build_graph`](#llama_modelbuild_graph)

**Methods**

---
#### llama\_model::llama\_model<!-- {{#callable:llama_model::llama_model}} -->
The `llama_model` constructor initializes a `llama_model` instance with specified parameters and sets up internal state.
- **Inputs**:
    - `params`: A constant reference to a `llama_model_params` structure that contains configuration parameters for the model.
- **Control Flow**:
    - The constructor initializes the member variable `params` with the provided `params` argument.
    - It creates a unique pointer `pimpl` to an instance of the `impl` struct.
    - It checks if `tensor_buft_overrides` is present in `params` and if the first element has a `pattern`, setting `has_tensor_overrides` accordingly.
- **Output**: The constructor does not return a value but initializes the `llama_model` object, preparing it for further operations.
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::load\_stats<!-- {{#callable:llama_model::load_stats}} -->
The `load_stats` function initializes the statistics of a `llama_model` instance using data from a `llama_model_loader`.
- **Inputs**:
    - `ml`: A reference to a `llama_model_loader` object that contains the statistics to be loaded into the model.
- **Control Flow**:
    - The function directly accesses the `n_elements` and `n_bytes` attributes of the `llama_model_loader` instance.
    - It assigns these values to the corresponding attributes in the private implementation structure of the `llama_model`.
- **Output**: The function does not return a value; it modifies the internal state of the `llama_model` by updating its statistics.
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::load\_arch<!-- {{#callable:llama_model::load_arch}} -->
The `load_arch` function loads the architecture type of a llama model from a model loader.
- **Inputs**:
    - `ml`: A reference to a `llama_model_loader` object that provides the architecture information.
- **Control Flow**:
    - The function retrieves the architecture type using `ml.get_arch()`.
    - It checks if the retrieved architecture is `LLM_ARCH_UNKNOWN`.
    - If the architecture is unknown, it throws a runtime error with a descriptive message.
- **Output**: The function does not return a value; it either successfully sets the architecture or throws an exception if the architecture is unknown.
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::load\_hparams<!-- {{#callable:llama_model::load_hparams}} -->
Loads hyperparameters for a `llama_model` from a `llama_model_loader`.
- **Inputs**:
    - `ml`: A reference to a `llama_model_loader` object that provides access to the model's metadata and key-value pairs.
- **Control Flow**:
    - Retrieves metadata from the `llama_model_loader` and populates the `gguf_kv` map with key-value pairs, skipping any arrays.
    - Checks if the model is set to only load vocabulary parameters; if so, it exits early.
    - Retrieves various hyperparameters from the loader, including context length, embedding length, block count, and expert counts.
    - If the architecture is `LLM_ARCH_WAVTOKENIZER_DEC`, additional parameters related to features and positional networks are retrieved.
    - Performs assertions to ensure the integrity of expert counts and initializes arrays for heads and feed-forward lengths.
    - Retrieves optional parameters for attention heads and ROPE scaling, handling defaults where necessary.
    - Based on the architecture, retrieves architecture-specific parameters and determines the model type.
    - Sets the number of bytes and description string for the model, and checks for the use of ALiBi if applicable.
- **Output**: The function does not return a value but populates the `hparams` and other member variables of the `llama_model` instance with the loaded hyperparameters.
- **Functions called**:
    - [`gguf_get_n_kv`](../ggml/src/gguf.cpp.driver.md#gguf_get_n_kv)
    - [`gguf_get_kv_type`](../ggml/src/gguf.cpp.driver.md#gguf_get_kv_type)
    - [`gguf_get_key`](../ggml/src/gguf.cpp.driver.md#gguf_get_key)
    - [`llama_rope_scaling_type_from_string`](#llama_rope_scaling_type_from_string)
    - [`format`](llama-impl.cpp.driver.md#format)
    - [`llama_model::arch_name`](#llama_modelarch_name)
    - [`llama_model::type_name`](#llama_modeltype_name)
    - [`llama_model_rope_type`](#llama_model_rope_type)
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::load\_vocab<!-- {{#callable:llama_model::load_vocab}} -->
Loads the vocabulary for the `llama_model` from a specified loader.
- **Inputs**:
    - `ml`: A reference to a `llama_model_loader` object that provides the vocabulary data to be loaded.
- **Control Flow**:
    - Retrieves the key-value pairs associated with the architecture of the model using `LLM_KV(arch)`.
    - Calls the `load` method of the `vocab` member, passing the loader and the retrieved key-value pairs.
- **Output**: This function does not return a value; it modifies the internal state of the `vocab` member of the `llama_model` by loading vocabulary data.
- **Functions called**:
    - [`LLM_KV`](llama-arch.h.driver.md#LLM_KV)
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::load\_tensors<!-- {{#callable:llama_model::load_tensors}} -->
Loads model tensors from a specified loader into the llama model.
- **Inputs**:
    - `ml`: A reference to a `llama_model_loader` object that provides the necessary methods to load tensors.
- **Control Flow**:
    - Logs the start of the tensor loading process.
    - Builds a list of buffer types for CPU and GPU devices.
    - Calculates split points for tensor distribution across devices based on available memory or provided splits.
    - Assigns layers to either CPU or GPU based on the calculated splits and the model's architecture.
    - Creates tensors for model weights and initializes them based on the architecture.
    - Handles potential errors during tensor creation and logs warnings for unused tensors.
    - Creates backend buffers for the tensors and manages memory allocation.
    - Loads tensor data from the loader into the model's tensors.
- **Output**: Returns true if all tensors are successfully loaded; otherwise, returns false.
- **Functions called**:
    - [`make_cpu_buft_list`](#make_cpu_buft_list)
    - [`make_gpu_buft_list`](#make_gpu_buft_list)
    - [`llama_model::n_devices`](#llama_modeln_devices)
    - [`ggml_backend_dev_memory`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_memory)
    - [`format`](llama-impl.cpp.driver.md#format)
    - [`ggml_backend_dev_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_name)
    - [`ggml_tensor_overhead`](../ggml/src/ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_init`](../ggml/src/ggml.c.driver.md#ggml_init)
    - [`llm_tensor_info_for`](llama-arch.cpp.driver.md#llm_tensor_info_for)
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`ggml_type_name`](../ggml/src/ggml.c.driver.md#ggml_type_name)
    - [`ggml_backend_buft_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buft_name)
    - [`select_weight_buft`](#select_weight_buft)
    - [`ggml_get_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_tensor)
    - [`LLM_TN::LLM_TN`](llama-arch.h.driver.md#LLM_TNLLM_TN)
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_backend_dev_get_props`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_get_props)
    - [`ggml_get_max_tensor_size`](../ggml/src/ggml.c.driver.md#ggml_get_max_tensor_size)
    - [`ggml_backend_alloc_ctx_tensors_from_buft`](../ggml/src/ggml-alloc.c.driver.md#ggml_backend_alloc_ctx_tensors_from_buft)
    - [`ggml_backend_buffer_is_host`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_is_host)
    - [`ggml_backend_buffer_get_base`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_get_base)
    - [`ggml_backend_buffer_get_size`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_get_size)
    - [`ggml_backend_buffer_set_usage`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_set_usage)
    - [`ggml_backend_buffer_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_name)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`ggml_get_name`](../ggml/src/ggml.c.driver.md#ggml_get_name)
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::arch\_name<!-- {{#callable:llama_model::arch_name}} -->
Returns the architecture name of the `llama_model` instance.
- **Inputs**: None
- **Control Flow**:
    - The function calls [`llm_arch_name`](llama-arch.cpp.driver.md#llm_arch_name) with the `arch` member of the `llama_model` instance.
    - The result of [`llm_arch_name`](llama-arch.cpp.driver.md#llm_arch_name) is returned as a string.
- **Output**: A string representing the architecture name associated with the `arch` member of the `llama_model`.
- **Functions called**:
    - [`llm_arch_name`](llama-arch.cpp.driver.md#llm_arch_name)
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::type\_name<!-- {{#callable:llama_model::type_name}} -->
Returns the type name of the `llama_model` instance.
- **Inputs**: None
- **Control Flow**:
    - The function directly calls the [`llm_type_name`](#llm_type_name) function with the `type` member of the `llama_model` instance.
    - No conditional statements or loops are present in this function.
- **Output**: Returns a `std::string` representing the type name associated with the `type` member of the `llama_model`.
- **Functions called**:
    - [`llm_type_name`](#llm_type_name)
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::desc<!-- {{#callable:llama_model::desc}} -->
Returns the description string of the `llama_model` instance.
- **Inputs**:
    - `this`: A constant reference to the current instance of `llama_model`.
- **Control Flow**:
    - The function accesses the `pimpl` member, which is a pointer to an implementation detail of the `llama_model` class.
    - It retrieves the `desc_str` member from the `pimpl` structure.
    - The function then returns the retrieved description string.
- **Output**: A `std::string` containing the description of the model, as stored in the `desc_str` member of the `pimpl` structure.
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::size<!-- {{#callable:llama_model::size}} -->
Returns the size in bytes of the `llama_model` instance.
- **Inputs**:
    - `this`: A constant reference to the current instance of the `llama_model` class.
- **Control Flow**:
    - The function directly accesses the `n_bytes` member of the `pimpl` pointer, which is an implementation detail of the `llama_model` class.
    - No conditional statements or loops are present; the function simply returns the value of `n_bytes`.
- **Output**: Returns a `size_t` value representing the total number of bytes used by the model.
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::n\_tensors<!-- {{#callable:llama_model::n_tensors}} -->
Returns the number of tensors stored in the `tensors_by_name` member of the `llama_model` class.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `tensors_by_name` member, which is a vector of pairs containing tensor names and their corresponding tensor pointers.
    - It calls the `size()` method on `tensors_by_name` to retrieve the count of tensors.
- **Output**: Returns a `size_t` value representing the total number of tensors in the `tensors_by_name` vector.
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::n\_devices<!-- {{#callable:llama_model::n_devices}} -->
Returns the number of devices used in the `llama_model`.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `devices` member of the `llama_model` structure.
    - It returns the size of the `devices` vector, which indicates how many devices are associated with the model.
- **Output**: The function outputs a `size_t` value representing the count of devices in the `devices` vector.
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::n\_elements<!-- {{#callable:llama_model::n_elements}} -->
Returns the total number of elements in the `llama_model` instance.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the private implementation pointer `pimpl` of the `llama_model` class.
    - It retrieves the `n_elements` member from the `pimpl` structure.
- **Output**: Returns a `uint64_t` value representing the total number of elements in the model.
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::print\_info<!-- {{#callable:llama_model::print_info}} -->
The `print_info` method logs detailed information about the `llama_model`'s architecture and hyperparameters.
- **Inputs**:
    - `this`: A constant reference to the current instance of the `llama_model` class.
- **Control Flow**:
    - The method begins by determining the rope scaling type from the model's hyperparameters.
    - A lambda function `print_f` is defined to evaluate and format the output of hyperparameter functions based on the number of layers.
    - The method logs various hyperparameters and model characteristics using `LLAMA_LOG_INFO`, conditionally including additional parameters based on the `vocab_only` flag.
    - It checks the architecture type of the model and logs architecture-specific parameters accordingly.
    - Finally, it calls the `print_info` method of the `vocab` member to log vocabulary-related information.
- **Output**: The method does not return a value; instead, it outputs formatted log messages to provide insights into the model's configuration and parameters.
- **Functions called**:
    - [`llama_rope_scaling_type_name`](#llama_rope_scaling_type_name)
    - [`llama_model::arch_name`](#llama_modelarch_name)
    - [`llama_model::type_name`](#llama_modeltype_name)
    - [`llama_expert_gating_func_name`](#llama_expert_gating_func_name)
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::dev\_layer<!-- {{#callable:llama_model::dev_layer}} -->
The `dev_layer` function retrieves the device associated with a specific layer in the `llama_model`.
- **Inputs**:
    - `il`: An integer index representing the specific layer for which the device is being retrieved.
- **Control Flow**:
    - Accesses the `pimpl` member of the `llama_model` instance, which is a pointer to an implementation detail.
    - Uses the provided index `il` to access the corresponding layer's device from the `dev_layer` vector.
    - Returns the `dev` member of the retrieved layer.
- **Output**: Returns a `ggml_backend_dev_t` object representing the device associated with the specified layer.
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::dev\_output<!-- {{#callable:llama_model::dev_output}} -->
Retrieves the device associated with the output of the `llama_model`.
- **Inputs**:
    - `this`: A constant reference to the current instance of `llama_model`, allowing access to its private members.
- **Control Flow**:
    - The function directly accesses the `dev` member of the `dev_output` structure within the `pimpl` pointer.
    - No conditional logic or loops are present; the function simply returns the value.
- **Output**: Returns a `ggml_backend_dev_t` representing the device used for the model's output.
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::select\_buft<!-- {{#callable:llama_model::select_buft}} -->
The `select_buft` function selects a backend buffer type based on the specified layer index.
- **Inputs**:
    - `il`: An integer representing the index of the layer for which the backend buffer type is to be selected.
- **Control Flow**:
    - The function calls the global `select_buft` function, passing a reference to the buffer list of the specified layer.
    - A lambda function is defined and passed as a second argument to `select_buft`, which creates two new 1D tensors of type `GGML_TYPE_F32` with a size defined by `hparams.n_embd`.
    - The lambda function then adds these two tensors together using [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add) and returns the result.
- **Output**: The function returns a value of type `ggml_backend_buffer_type_t`, which represents the selected backend buffer type for the specified layer.
- **Functions called**:
    - [`ggml_new_tensor_1d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::has\_tensor\_overrides<!-- {{#callable:llama_model::has_tensor_overrides}} -->
The `has_tensor_overrides` function checks if the `llama_model` instance has tensor overrides enabled.
- **Inputs**:
    - `this`: A constant reference to the `llama_model` instance on which the method is called.
- **Control Flow**:
    - The function directly accesses the `has_tensor_overrides` member of the `pimpl` pointer, which is an instance of a private implementation structure.
    - It returns the value of `has_tensor_overrides`, which is expected to be a boolean indicating the presence of tensor overrides.
- **Output**: The function returns a boolean value that indicates whether tensor overrides are present in the model.
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::get\_tensor<!-- {{#callable:llama_model::get_tensor}} -->
Retrieves a pointer to a `ggml_tensor` associated with a given name from the `llama_model`.
- **Inputs**:
    - `name`: A C-style string representing the name of the tensor to retrieve.
- **Control Flow**:
    - Uses `std::find_if` to search through the `tensors_by_name` vector for a pair where the first element matches the provided `name`.
    - If the tensor is not found (i.e., the iterator reaches the end of the vector), the function returns a null pointer.
    - If the tensor is found, the function returns the second element of the found pair, which is a pointer to the corresponding `ggml_tensor`.
- **Output**: Returns a pointer to the `ggml_tensor` associated with the specified name, or nullptr if no such tensor exists.
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::get\_rope\_freq\_base<!-- {{#callable:llama_model::get_rope_freq_base}} -->
The `get_rope_freq_base` function retrieves the base frequency for rope embeddings based on the model's training state.
- **Inputs**:
    - `cparams`: A constant reference to a `llama_cparams` structure that contains configuration parameters for the model.
    - `il`: An integer representing the index of the layer for which the frequency base is being retrieved.
- **Control Flow**:
    - The function checks if the model is in Stochastic Weight Averaging (SWA) mode for the given layer index `il` using the `is_swa` method of `hparams`.
    - If the model is in SWA mode, it returns the `rope_freq_base_train_swa` value from `hparams`.
    - If not in SWA mode, it returns the `rope_freq_base` value from the `cparams` structure.
- **Output**: The function outputs a float value representing the base frequency for rope embeddings, either from the training SWA parameters or the provided configuration parameters.
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::get\_rope\_freq\_scale<!-- {{#callable:llama_model::get_rope_freq_scale}} -->
The `get_rope_freq_scale` function retrieves the appropriate rope frequency scale based on the training state.
- **Inputs**:
    - `cparams`: A constant reference to a `llama_cparams` structure that contains configuration parameters for the model.
    - `il`: An integer representing the index of the layer for which the frequency scale is being retrieved.
- **Control Flow**:
    - The function checks if the layer index `il` corresponds to a Stochastic Weight Averaging (SWA) state using the `is_swa` method of `hparams`.
    - If `is_swa(il)` returns true, it returns the `rope_freq_scale_train_swa` value from `hparams`.
    - If `is_swa(il)` returns false, it returns the `rope_freq_scale` value from `cparams`.
- **Output**: The function outputs a float value representing the rope frequency scale, which is determined based on the training state of the model.
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::get\_rope\_factors<!-- {{#callable:llama_model::get_rope_factors}} -->
The `get_rope_factors` function retrieves the appropriate rope frequency tensor based on the context size and layer index.
- **Inputs**:
    - `cparams`: A constant reference to a `llama_cparams` structure containing configuration parameters for the model.
    - `il`: An integer representing the index of the layer for which the rope factors are being requested.
- **Control Flow**:
    - Calculates the number of context tokens per sequence by dividing `n_ctx` by `n_seq_max` from `cparams`.
    - Checks if the `rope_freqs` for the specified layer is already initialized; if so, it returns that tensor.
    - If the context size per sequence exceeds a predefined threshold (`n_ctx_orig_yarn`), it returns the long rope factors for the layer.
    - Otherwise, it returns the short rope factors for the layer.
- **Output**: Returns a pointer to a `ggml_tensor` representing the appropriate rope frequency tensor (either long or short) based on the context size and layer index.
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::create\_memory<!-- {{#callable:llama_model::create_memory}} -->
Creates a memory structure based on the model architecture and provided parameters.
- **Inputs**:
    - `params`: A constant reference to a `llama_memory_params` structure that contains parameters for memory creation.
    - `cparams`: A reference to a `llama_cparams` structure that contains configuration parameters for the memory.
- **Control Flow**:
    - The function begins by declaring a pointer `res` to hold the resulting memory structure.
    - A switch statement evaluates the `arch` member of the `llama_model` to determine the model architecture.
    - For specific architectures (BERT variants), it initializes `res` to nullptr, indicating no memory is created.
    - For other architectures (like MAMBA and RWKV), it allocates a new `llama_kv_cache_recurrent` object with parameters derived from `cparams`.
    - For all other architectures, it calculates padding and adjusts `cparams.n_ctx`, then either creates a `llama_kv_cache_unified_iswa` or `llama_kv_cache_unified` based on the `swa_type` in `hparams`.
- **Output**: Returns a pointer to the created `llama_memory_i` structure, which may be a cache or a unified memory structure depending on the architecture and parameters.
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)


---
#### llama\_model::build\_graph<!-- {{#callable:llama_model::build_graph}} -->
The `build_graph` function constructs a graph representation for a language model based on its architecture and specified parameters.
- **Inputs**:
    - `params`: A constant reference to an `llm_graph_params` structure that contains parameters for building the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the graph to be built.
    - `type`: An enumeration value of type `llm_graph_type` that specifies the type of graph to build (e.g., encoder or decoder).
- **Control Flow**:
    - The function begins by declaring a unique pointer `llm` to hold the graph context.
    - A switch statement is used to determine the architecture of the model (`arch`), and based on this, it creates an appropriate graph builder object (e.g., `llm_build_llama`, `llm_build_bert`, etc.) using `std::make_unique`.
    - For the `LLM_ARCH_PHI3` architecture, an additional check on `hparams.swa_type` is performed to decide which version of the builder to instantiate.
    - After the appropriate builder is created, the function calls `build_pooling` on the `llm` object to add a pooling layer to the graph.
    - Finally, the function returns the result of the graph building process by moving the result from the `llm` object.
- **Output**: The function returns a pointer to the result of the graph building process, encapsulated in `llm_graph_result_ptr`, which represents the constructed graph.
- **See also**: [`llama_model`](llama-model.h.driver.md#llama_model)  (Data Structure)



# Functions

---
### llm\_type\_name<!-- {{#callable:llm_type_name}} -->
The `llm_type_name` function returns a string representation of a given `llm_type` enumeration value.
- **Inputs**:
    - `type`: An enumeration value of type `llm_type` representing a specific large language model type.
- **Control Flow**:
    - The function uses a switch statement to match the input `type` against various `llm_type` enumeration values.
    - For each case in the switch statement, a corresponding string literal is returned that represents the name of the model type.
    - If the input `type` does not match any of the predefined cases, the function returns the string "?B" as a default case.
- **Output**: A constant character pointer to a string literal representing the name of the large language model type.


---
### llama\_expert\_gating\_func\_name<!-- {{#callable:llama_expert_gating_func_name}} -->
The function `llama_expert_gating_func_name` returns the name of a gating function based on the provided type.
- **Inputs**:
    - `type`: An enumeration value of type `llama_expert_gating_func_type` that specifies the gating function type.
- **Control Flow**:
    - The function uses a switch statement to determine the output based on the input `type`.
    - If `type` is `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`, it returns the string "softmax".
    - If `type` is `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`, it returns the string "sigmoid".
    - For any other value of `type`, it returns the string "unknown".
- **Output**: A constant character pointer to a string representing the name of the gating function, or "unknown" if the type is not recognized.


---
### llama\_rope\_scaling\_type\_name<!-- {{#callable:llama_rope_scaling_type_name}} -->
The function `llama_rope_scaling_type_name` retrieves the name of a rope scaling type from a predefined map using the given scaling type as a key.
- **Inputs**:
    - `rope_scaling_type`: An enumeration value of type `llama_rope_scaling_type` that represents a specific rope scaling type.
- **Control Flow**:
    - The function accesses the `LLAMA_ROPE_SCALING_TYPES` map using the provided `rope_scaling_type` as a key.
    - It retrieves the corresponding string value from the map.
- **Output**: A `std::string` representing the name of the rope scaling type associated with the given `rope_scaling_type` key.


---
### llama\_rope\_scaling\_type\_from\_string<!-- {{#callable:llama_rope_scaling_type_from_string}} -->
The function `llama_rope_scaling_type_from_string` converts a string representation of a rope scaling type to its corresponding enumeration value.
- **Inputs**:
    - `name`: A constant reference to a string representing the name of the rope scaling type to be converted.
- **Control Flow**:
    - Iterates over the `LLAMA_ROPE_SCALING_TYPES` map, which contains pairs of enumeration values and their corresponding string names.
    - Checks if the current string name in the map matches the input string `name`.
    - If a match is found, returns the corresponding enumeration value cast to `llama_rope_scaling_type`.
    - If no match is found after iterating through the map, returns `LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED`.
- **Output**: Returns a `llama_rope_scaling_type` enumeration value corresponding to the input string, or `LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED` if no match is found.


---
### weight\_buft\_supported<!-- {{#callable:weight_buft_supported}} -->
The function `weight_buft_supported` checks if a given operation on a tensor is supported by a specific backend device and buffer type.
- **Inputs**:
    - `hparams`: A constant reference to a `llama_hparams` object containing hyperparameters used in certain operations.
    - `w`: A pointer to a `ggml_tensor` object representing the tensor on which the operation is to be performed.
    - `op`: A `ggml_op` enumeration value representing the operation to be checked for support.
    - `buft`: A `ggml_backend_buffer_type_t` value indicating the type of buffer to be used.
    - `dev`: A `ggml_backend_dev_t` value representing the backend device to check for operation support.
- **Control Flow**:
    - Assert that the tensor `w` is not null.
    - If the operation `op` is `GGML_OP_NONE`, return true immediately as no operation is needed.
    - Initialize a `ggml_context` with specific parameters and check for successful creation.
    - Depending on the operation `op`, create the necessary tensors and perform the operation using the `ggml` library functions, storing the result in `op_tensor`.
    - Allocate a temporary dummy buffer for the tensor `w` to check the buffer type compatibility.
    - Check if the backend device `dev` supports the operation on `op_tensor` using the allocated buffer.
    - Free the temporary buffer and reset the tensor's buffer to null.
    - Return the result of the support check as a boolean.
- **Output**: A boolean value indicating whether the specified operation is supported by the given backend device and buffer type.
- **Functions called**:
    - [`ggml_tensor_overhead`](../ggml/src/ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_init`](../ggml/src/ggml.c.driver.md#ggml_init)
    - [`format`](llama-impl.cpp.driver.md#format)
    - [`ggml_new_tensor_1d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`ggml_new_tensor_4d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_4d)
    - [`ggml_mul_mat`](../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_new_tensor_3d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_3d)
    - [`ggml_new_tensor_2d`](../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d)
    - [`ggml_mul_mat_id`](../ggml/src/ggml.c.driver.md#ggml_mul_mat_id)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_mul`](../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`ggml_div`](../ggml/src/ggml.c.driver.md#ggml_div)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_ssm_conv`](../ggml/src/ggml.c.driver.md#ggml_ssm_conv)
    - [`ggml_ssm_scan`](../ggml/src/ggml.c.driver.md#ggml_ssm_scan)
    - [`ggml_rwkv_wkv6`](../ggml/src/ggml.c.driver.md#ggml_rwkv_wkv6)
    - [`ggml_im2col`](../ggml/src/ggml.c.driver.md#ggml_im2col)
    - [`ggml_op_name`](../ggml/src/ggml.c.driver.md#ggml_op_name)
    - [`ggml_backend_dev_supports_op`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_supports_op)
    - [`ggml_backend_buffer_free`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_free)


---
### select\_weight\_buft<!-- {{#callable:select_weight_buft}} -->
The function `select_weight_buft` selects an appropriate buffer type for a given tensor operation based on the provided hyperparameters and a list of buffer types.
- **Inputs**:
    - `hparams`: A constant reference to `llama_hparams`, which contains hyperparameters that may influence buffer type selection.
    - `tensor`: A pointer to a `ggml_tensor` object, representing the tensor for which the buffer type is being selected.
    - `op`: A `ggml_op` object representing the operation to be performed on the tensor.
    - `buft_list`: A constant reference to a `buft_list_t`, which is a list of pairs containing a device and a buffer type to be considered for selection.
- **Control Flow**:
    - Assert that `buft_list` is not empty to ensure there are buffer types to select from.
    - Iterate over each pair in `buft_list`, extracting the current device and buffer type.
    - Check if the current buffer type is supported for the given hyperparameters, tensor, operation, and device using [`weight_buft_supported`](#weight_buft_supported).
    - If a supported buffer type is found, return it immediately.
    - If no supported buffer type is found after iterating through the list, return `nullptr`.
- **Output**: Returns a `ggml_backend_buffer_type_t` representing the selected buffer type, or `nullptr` if no suitable buffer type is found.
- **Functions called**:
    - [`weight_buft_supported`](#weight_buft_supported)


---
### make\_cpu\_buft\_list<!-- {{#callable:make_cpu_buft_list}} -->
The `make_cpu_buft_list` function constructs a list of buffer types for CPU and other devices, prioritizing ACCEL and host buffer types, and ensuring CPU buffer types are included.
- **Inputs**:
    - `devices`: A constant reference to a vector of `ggml_backend_dev_t` objects representing the devices to consider for buffer type inclusion.
- **Control Flow**:
    - Initialize an empty `buft_list_t` object to store buffer types.
    - Iterate over all available backend devices and add ACCEL buffer types to the list, skipping CPU buffer types.
    - Iterate over the provided `devices` vector to add a host buffer type, breaking after the first successful addition.
    - Check for the presence of a CPU device; if none is found, throw a runtime error.
    - Retrieve and add extra buffer types from the CPU device if no GPU device is present.
    - Iterate over all backend devices again to add CPU buffer types to the list.
    - Return the constructed `buft_list_t` object.
- **Output**: Returns a `buft_list_t` object containing buffer types associated with the specified devices, including ACCEL, host, extra, and CPU buffer types.
- **Functions called**:
    - [`ggml_backend_dev_count`](../ggml/src/ggml-backend-reg.cpp.driver.md#ggml_backend_dev_count)
    - [`ggml_backend_dev_type`](../ggml/include/ggml-backend.h.driver.md#ggml_backend_dev_type)
    - [`format`](llama-impl.cpp.driver.md#format)
    - [`ggml_backend_reg_get_proc_address`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_get_proc_address)


---
### make\_gpu\_buft\_list<!-- {{#callable:make_gpu_buft_list}} -->
The function `make_gpu_buft_list` creates a list of buffer types for a given GPU device based on the specified split mode and tensor split configuration.
- **Inputs**:
    - `dev`: A `ggml_backend_dev_t` object representing the GPU device for which the buffer list is being created.
    - `split_mode`: A `llama_split_mode` enum value indicating the mode of tensor splitting, specifically whether row splitting is requested.
    - `tensor_split`: A pointer to a float array representing the tensor split configuration.
- **Control Flow**:
    - Initialize an empty buffer list `buft_list`.
    - Check if the `split_mode` is `LLAMA_SPLIT_MODE_ROW`.
    - If `split_mode` is `LLAMA_SPLIT_MODE_ROW`, retrieve the backend registry for the device and obtain the function pointer for `ggml_backend_split_buffer_type`.
    - If the function pointer is valid, determine the device index within its backend registry.
    - Use the function pointer to get the buffer type for the device index and tensor split, and add it to `buft_list` if not null.
    - Add the default buffer type for the device to `buft_list`.
    - Return the populated `buft_list`.
- **Output**: A `buft_list_t` object containing buffer types for the specified GPU device.
- **Functions called**:
    - [`ggml_backend_reg_get_proc_address`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_get_proc_address)
    - [`ggml_backend_reg_dev_count`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_dev_count)
    - [`format`](llama-impl.cpp.driver.md#format)
    - [`ggml_backend_dev_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_name)


---
### impl<!-- {{#callable:llama_model::impl::impl}} -->
The `impl` function is a constructor and destructor pair for a class, both of which are empty and perform no operations.
- **Inputs**: None
- **Control Flow**:
    - The constructor `impl()` is called when an object of the class is created, but it contains no logic or operations.
    - The destructor `~impl()` is called when an object of the class is destroyed, but it also contains no logic or operations.
- **Output**: There is no output from either the constructor or the destructor as they are both empty.


---
### \~impl<!-- {{#callable:llama_model::impl::~impl}} -->
The `~impl` function is a destructor for a class, which performs no specific actions upon object destruction.
- **Inputs**: None
- **Control Flow**:
    - The function is defined as a destructor for a class, indicated by the tilde `~` prefix.
    - The function body is empty, meaning it does not perform any operations when an object of the class is destroyed.
- **Output**: There is no output from this function as it is a destructor with an empty body.


---
### buft\_supported<!-- {{#callable:buft_supported}} -->
The `buft_supported` function checks if a specific backend buffer type is supported by a given device using a provided function to create an operation tensor.
- **Inputs**:
    - `buft`: A `ggml_backend_buffer_type_t` value representing the type of backend buffer to be checked.
    - `dev`: A `ggml_backend_dev_t` value representing the device on which the buffer type support is being checked.
    - `fn`: A function reference that takes a `ggml_context_ptr` and returns a `ggml_tensor*`, used to create the operation tensor for checking support.
- **Control Flow**:
    - Initialize `ggml_init_params` with specific memory settings and no allocation.
    - Create a `ggml_context_ptr` using [`ggml_init`](../ggml/src/ggml.c.driver.md#ggml_init) with the initialized parameters.
    - Check if the context creation was successful; if not, throw a runtime error.
    - Allocate a backend buffer using `ggml_backend_buft_alloc_buffer` with the specified buffer type and size 0.
    - Invoke the provided function `fn` with the context to obtain an operation tensor.
    - Iterate over the source tensors of the operation tensor, ensuring each source tensor's buffer is set to the allocated buffer if it is not already set.
    - Check if the device supports the operation tensor using [`ggml_backend_dev_supports_op`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_supports_op).
    - Return the result of the support check as a boolean.
- **Output**: A boolean value indicating whether the specified backend buffer type is supported by the given device.
- **Functions called**:
    - [`ggml_tensor_overhead`](../ggml/src/ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_init`](../ggml/src/ggml.c.driver.md#ggml_init)
    - [`format`](llama-impl.cpp.driver.md#format)
    - [`ggml_backend_dev_supports_op`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_supports_op)


---
### select\_buft<!-- {{#callable:select_buft}} -->
The function `select_buft` iterates over a list of buffer types and returns the first one that is supported by a given condition.
- **Inputs**:
    - `buft_list`: A list of pairs, where each pair consists of a device and a buffer type.
    - `fn`: A callable object or function that determines the condition for buffer type support.
- **Control Flow**:
    - Iterate over each pair in `buft_list`, extracting the device and buffer type.
    - Check if the current buffer type is supported by calling [`buft_supported`](#buft_supported) with the current buffer type, device, and the provided function `fn`.
    - If a supported buffer type is found, return it immediately.
    - If no supported buffer type is found after iterating through the list, throw a runtime error.
- **Output**: The function returns a `ggml_backend_buffer_type_t` which is the first supported buffer type found in the list.
- **Functions called**:
    - [`buft_supported`](#buft_supported)
    - [`format`](llama-impl.cpp.driver.md#format)


---
### llama\_model\_default\_params<!-- {{#callable:llama_model_default_params}} -->
The `llama_model_default_params` function initializes and returns a `llama_model_params` structure with default configuration values for a llama model.
- **Inputs**: None
- **Control Flow**:
    - Initialize a `llama_model_params` structure named `result` with default values for various parameters such as `devices`, `tensor_buft_overrides`, `n_gpu_layers`, `split_mode`, `main_gpu`, `tensor_split`, `progress_callback`, `progress_callback_user_data`, `kv_overrides`, `vocab_only`, `use_mmap`, `use_mlock`, and `check_tensors`.
    - Check if `GGML_USE_METAL` is defined; if so, set `result.n_gpu_layers` to 999 to offload all layers to the GPU.
    - Return the `result` structure.
- **Output**: A `llama_model_params` structure initialized with default values.


---
### llama\_model\_get\_vocab<!-- {{#callable:llama_model_get_vocab}} -->
The function `llama_model_get_vocab` retrieves the vocabulary from a given llama model.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure from which the vocabulary is to be retrieved.
- **Control Flow**:
    - The function takes a pointer to a `llama_model` as input.
    - It accesses the `vocab` member of the `llama_model` structure.
    - It returns a pointer to the `vocab` member.
- **Output**: A pointer to the `llama_vocab` structure contained within the provided `llama_model`.


---
### llama\_free\_model<!-- {{#callable:llama_free_model}} -->
The `llama_free_model` function deallocates memory associated with a `llama_model` object.
- **Inputs**:
    - `model`: A pointer to a `llama_model` object that needs to be freed.
- **Control Flow**:
    - The function calls [`llama_model_free`](#llama_model_free) with the provided `model` pointer to release the allocated resources.
- **Output**: This function does not return any value.
- **Functions called**:
    - [`llama_model_free`](#llama_model_free)


---
### llama\_model\_free<!-- {{#callable:llama_model_free}} -->
The `llama_model_free` function deallocates memory for a `llama_model` object.
- **Inputs**:
    - `model`: A pointer to a `llama_model` object that needs to be deallocated.
- **Control Flow**:
    - The function takes a pointer to a `llama_model` object as an argument.
    - It uses the `delete` operator to deallocate the memory associated with the `llama_model` object.
- **Output**: The function does not return any value.


---
### llama\_model\_n\_ctx\_train<!-- {{#callable:llama_model_n_ctx_train}} -->
The function `llama_model_n_ctx_train` retrieves the number of training contexts from a given llama model.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure from which the training context count is to be retrieved.
- **Control Flow**:
    - Accesses the `hparams` member of the `llama_model` structure pointed to by `model`.
    - Returns the `n_ctx_train` member of the `hparams` structure.
- **Output**: An `int32_t` representing the number of training contexts (`n_ctx_train`) in the model's hyperparameters.


---
### llama\_model\_n\_embd<!-- {{#callable:llama_model_n_embd}} -->
The function `llama_model_n_embd` retrieves the number of embedding dimensions from a given llama model.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure from which the number of embedding dimensions is to be retrieved.
- **Control Flow**:
    - Access the `hparams` member of the `llama_model` structure pointed to by `model`.
    - Return the `n_embd` value from the `hparams` structure.
- **Output**: An `int32_t` representing the number of embedding dimensions in the model.


---
### llama\_model\_n\_layer<!-- {{#callable:llama_model_n_layer}} -->
The function `llama_model_n_layer` retrieves the number of layers from a given llama model's hyperparameters.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure from which the number of layers is to be retrieved.
- **Control Flow**:
    - Access the `hparams` member of the `llama_model` structure pointed to by `model`.
    - Return the `n_layer` value from the `hparams` structure.
- **Output**: The function returns an `int32_t` representing the number of layers in the model's hyperparameters.


---
### llama\_model\_n\_head<!-- {{#callable:llama_model_n_head}} -->
The function `llama_model_n_head` retrieves the number of attention heads from a given llama model's hyperparameters.
- **Inputs**:
    - `model`: A pointer to a `llama_model` object from which the number of attention heads is to be retrieved.
- **Control Flow**:
    - Access the `hparams` member of the `llama_model` object pointed to by `model`.
    - Call the `n_head()` method on the `hparams` object to get the number of attention heads.
- **Output**: Returns an `int32_t` representing the number of attention heads in the model's hyperparameters.


---
### llama\_model\_n\_head\_kv<!-- {{#callable:llama_model_n_head_kv}} -->
The function `llama_model_n_head_kv` retrieves the number of key-value heads from a given llama model's hyperparameters.
- **Inputs**:
    - `model`: A pointer to a `llama_model` object from which the number of key-value heads is to be retrieved.
- **Control Flow**:
    - The function accesses the `hparams` member of the `llama_model` object pointed to by `model`.
    - It calls the `n_head_kv()` method on the `hparams` object to get the number of key-value heads.
- **Output**: The function returns an `int32_t` representing the number of key-value heads in the model's hyperparameters.


---
### llama\_model\_n\_swa<!-- {{#callable:llama_model_n_swa}} -->
The function `llama_model_n_swa` retrieves the `n_swa` parameter from the hyperparameters of a given llama model.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure from which the `n_swa` parameter is to be retrieved.
- **Control Flow**:
    - Access the `hparams` member of the `llama_model` structure pointed to by `model`.
    - Return the `n_swa` value from the `hparams` structure.
- **Output**: The function returns an `int32_t` value representing the `n_swa` parameter of the model's hyperparameters.


---
### llama\_n\_ctx\_train<!-- {{#callable:llama_n_ctx_train}} -->
The function `llama_n_ctx_train` retrieves the number of training contexts from a given llama model.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure from which the number of training contexts is to be retrieved.
- **Control Flow**:
    - The function calls [`llama_model_n_ctx_train`](#llama_model_n_ctx_train) with the provided `model` as an argument.
    - It directly returns the result of the [`llama_model_n_ctx_train`](#llama_model_n_ctx_train) function call.
- **Output**: An `int32_t` representing the number of training contexts in the given llama model.
- **Functions called**:
    - [`llama_model_n_ctx_train`](#llama_model_n_ctx_train)


---
### llama\_n\_embd<!-- {{#callable:llama_n_embd}} -->
The function `llama_n_embd` retrieves the number of embeddings from a given llama model.
- **Inputs**:
    - `model`: A pointer to a `llama_model` object from which the number of embeddings is to be retrieved.
- **Control Flow**:
    - The function directly calls [`llama_model_n_embd`](#llama_model_n_embd) with the provided `model` as an argument.
    - It returns the result of the [`llama_model_n_embd`](#llama_model_n_embd) function call.
- **Output**: An `int32_t` representing the number of embeddings in the given llama model.
- **Functions called**:
    - [`llama_model_n_embd`](#llama_model_n_embd)


---
### llama\_n\_layer<!-- {{#callable:llama_n_layer}} -->
The function `llama_n_layer` retrieves the number of layers in a given llama model.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure, representing the model whose number of layers is to be retrieved.
- **Control Flow**:
    - The function calls another function [`llama_model_n_layer`](#llama_model_n_layer) with the provided `model` as an argument.
    - It directly returns the result of the [`llama_model_n_layer`](#llama_model_n_layer) function call.
- **Output**: An `int32_t` representing the number of layers in the specified llama model.
- **Functions called**:
    - [`llama_model_n_layer`](#llama_model_n_layer)


---
### llama\_n\_head<!-- {{#callable:llama_n_head}} -->
The function `llama_n_head` retrieves the number of heads from a given llama model.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure from which the number of heads is to be retrieved.
- **Control Flow**:
    - The function calls another function [`llama_model_n_head`](#llama_model_n_head) passing the `model` as an argument.
    - It directly returns the result obtained from [`llama_model_n_head`](#llama_model_n_head).
- **Output**: An `int32_t` representing the number of heads in the given llama model.
- **Functions called**:
    - [`llama_model_n_head`](#llama_model_n_head)


---
### llama\_model\_rope\_type<!-- {{#callable:llama_model_rope_type}} -->
The function `llama_model_rope_type` determines the type of RoPE (Rotary Position Embedding) to use based on the architecture of a given llama model.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure, which contains information about the model's architecture.
- **Control Flow**:
    - The function uses a switch statement to evaluate the `arch` field of the `model` structure.
    - If the architecture matches any of the cases that do not use RoPE, it returns `LLAMA_ROPE_TYPE_NONE`.
    - If the architecture matches any of the cases that use normal RoPE, it returns `LLAMA_ROPE_TYPE_NORM`.
    - If the architecture matches any of the cases where head values are offset by `n_rot/2`, it returns `LLAMA_ROPE_TYPE_NEOX`.
    - If the architecture is `LLM_ARCH_QWEN2VL`, it returns `LLAMA_ROPE_TYPE_MROPE`.
    - If the architecture is `LLM_ARCH_UNKNOWN`, it triggers an abort with an error message.
    - If no case matches, it defaults to returning `LLAMA_ROPE_TYPE_NONE`.
- **Output**: The function returns a `llama_rope_type` enumeration value indicating the type of RoPE to use for the given model architecture.


---
### llama\_model\_rope\_freq\_scale\_train<!-- {{#callable:llama_model_rope_freq_scale_train}} -->
The function `llama_model_rope_freq_scale_train` retrieves the rope frequency scale training parameter from a given llama model.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure from which the rope frequency scale training parameter is to be retrieved.
- **Control Flow**:
    - Access the `hparams` member of the `llama_model` structure pointed to by `model`.
    - Retrieve the `rope_freq_scale_train` value from the `hparams` structure.
- **Output**: Returns the `rope_freq_scale_train` value as a float.


---
### llama\_model\_meta\_val\_str<!-- {{#callable:llama_model_meta_val_str}} -->
The function `llama_model_meta_val_str` retrieves a string value associated with a given key from a model's metadata and stores it in a buffer.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure containing metadata in a key-value format.
    - `key`: A constant character pointer representing the key whose associated value is to be retrieved.
    - `buf`: A character buffer where the retrieved string value will be stored.
    - `buf_size`: The size of the buffer `buf` to ensure no overflow occurs.
- **Control Flow**:
    - The function searches for the `key` in the `gguf_kv` map of the `model`.
    - If the `key` is not found, it checks if `buf_size` is greater than 0 and sets the first character of `buf` to the null terminator, then returns -1.
    - If the `key` is found, it uses `snprintf` to copy the associated string value into `buf`, ensuring it does not exceed `buf_size`.
- **Output**: Returns -1 if the key is not found, otherwise returns the number of characters written to `buf` (excluding the null terminator).


---
### llama\_model\_meta\_count<!-- {{#callable:llama_model_meta_count}} -->
The function `llama_model_meta_count` returns the number of metadata entries in a given `llama_model` object.
- **Inputs**:
    - `model`: A pointer to a `llama_model` object whose metadata count is to be retrieved.
- **Control Flow**:
    - Access the `gguf_kv` member of the `llama_model` object pointed to by `model`.
    - Call the `size()` method on the `gguf_kv` member to get the number of metadata entries.
    - Cast the result of `size()` to an `int` and return it.
- **Output**: An `int32_t` representing the number of metadata entries in the `llama_model` object.


---
### llama\_model\_meta\_key\_by\_index<!-- {{#callable:llama_model_meta_key_by_index}} -->
The function `llama_model_meta_key_by_index` retrieves a metadata key from a `llama_model` object by its index and stores it in a buffer.
- **Inputs**:
    - `model`: A pointer to a `llama_model` object containing metadata key-value pairs.
    - `i`: An integer index specifying which metadata key to retrieve.
    - `buf`: A character buffer where the retrieved metadata key will be stored.
    - `buf_size`: The size of the buffer `buf`.
- **Control Flow**:
    - Check if the index `i` is out of bounds for the metadata key-value pairs in the model.
    - If `i` is out of bounds and `buf_size` is greater than 0, set the first character of `buf` to the null terminator and return -1.
    - If `i` is within bounds, advance an iterator to the `i`-th position in the metadata key-value pairs.
    - Use `snprintf` to copy the key at the `i`-th position into `buf`, ensuring not to exceed `buf_size`.
- **Output**: Returns the number of characters written to `buf` (excluding the null terminator) or -1 if the index is out of bounds.


---
### llama\_model\_meta\_val\_str\_by\_index<!-- {{#callable:llama_model_meta_val_str_by_index}} -->
The function `llama_model_meta_val_str_by_index` retrieves a string value from a model's metadata by index and stores it in a buffer.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure containing metadata in a key-value format.
    - `i`: An integer index specifying which metadata entry to retrieve.
    - `buf`: A character buffer where the retrieved string value will be stored.
    - `buf_size`: The size of the buffer `buf` to ensure no overflow occurs.
- **Control Flow**:
    - Check if the index `i` is out of bounds for the metadata size; if so, set the buffer to an empty string and return -1.
    - If the index is valid, advance an iterator to the `i`-th position in the metadata key-value map.
    - Use `snprintf` to copy the string value at the specified index into the buffer, ensuring it does not exceed `buf_size`.
- **Output**: Returns the number of characters written to the buffer, excluding the null terminator, or -1 if the index is out of bounds.


---
### llama\_model\_desc<!-- {{#callable:llama_model_desc}} -->
The `llama_model_desc` function formats the description of a llama model into a provided buffer.
- **Inputs**:
    - `model`: A pointer to a `llama_model` object whose description is to be formatted.
    - `buf`: A character buffer where the formatted description will be stored.
    - `buf_size`: The size of the buffer `buf` to ensure the description fits within the allocated space.
- **Control Flow**:
    - The function calls `snprintf` to format the description of the `llama_model` into the provided buffer `buf`.
    - It uses the `desc()` method of the `llama_model` object to obtain the description string.
    - The formatted string is truncated if it exceeds `buf_size`.
- **Output**: The function returns the number of characters that would have been written if `buf_size` was sufficiently large, not counting the terminating null byte.


---
### llama\_model\_size<!-- {{#callable:llama_model_size}} -->
The function `llama_model_size` returns the size of a given `llama_model` object.
- **Inputs**:
    - `model`: A pointer to a `llama_model` object whose size is to be determined.
- **Control Flow**:
    - The function accesses the `size()` method of the `llama_model` object pointed to by `model`.
    - It returns the result of the `size()` method call.
- **Output**: The function returns a `uint64_t` value representing the size of the `llama_model` object.


---
### llama\_model\_chat\_template<!-- {{#callable:llama_model_chat_template}} -->
The function `llama_model_chat_template` retrieves a chat template string associated with a given model and name, or returns a default or null value if not found.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure, which contains information about the model architecture and key-value pairs.
    - `name`: A constant character pointer representing the name used to generate a key for retrieving the chat template.
- **Control Flow**:
    - Generate a key using the model's architecture and the provided name, or a default key if the name is null.
    - Search for the generated key in the model's `gguf_kv` map.
    - If the key is not found, check if the model matches specific conditions (e.g., pre-type and layer size) to return a hardcoded template string.
    - If the conditions are not met, return a null pointer.
    - If the key is found, return the associated chat template string.
- **Output**: A constant character pointer to the chat template string if found, a hardcoded string for specific models, or null if not found.
- **Functions called**:
    - [`LLM_KV`](llama-arch.h.driver.md#LLM_KV)


---
### llama\_model\_n\_params<!-- {{#callable:llama_model_n_params}} -->
The function `llama_model_n_params` returns the number of parameters in a given llama model.
- **Inputs**:
    - `model`: A pointer to a `llama_model` object whose number of parameters is to be retrieved.
- **Control Flow**:
    - The function accesses the `n_elements` method of the `llama_model` object pointed to by `model`.
    - It returns the result of the `n_elements` method call.
- **Output**: The function returns a `uint64_t` value representing the number of parameters in the llama model.


---
### llama\_model\_has\_encoder<!-- {{#callable:llama_model_has_encoder}} -->
The function `llama_model_has_encoder` checks if a given llama model architecture includes an encoder.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure, which contains information about the model's architecture.
- **Control Flow**:
    - The function uses a switch statement to evaluate the `arch` field of the `model`.
    - If the architecture is `LLM_ARCH_T5`, the function returns `true`.
    - If the architecture is `LLM_ARCH_T5ENCODER`, the function returns `true`.
    - For any other architecture, the function returns `false`.
- **Output**: A boolean value indicating whether the model architecture includes an encoder (`true`) or not (`false`).


---
### llama\_model\_has\_decoder<!-- {{#callable:llama_model_has_decoder}} -->
The function `llama_model_has_decoder` checks if a given llama model architecture includes a decoder component.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure, which contains information about the model's architecture.
- **Control Flow**:
    - The function uses a switch statement to evaluate the `arch` field of the `llama_model` structure pointed to by `model`.
    - If the architecture is `LLM_ARCH_T5ENCODER`, the function returns `false`, indicating the model does not have a decoder.
    - For any other architecture, the function returns `true`, indicating the model has a decoder.
- **Output**: A boolean value indicating whether the model has a decoder (`true`) or not (`false`).


---
### llama\_model\_decoder\_start\_token<!-- {{#callable:llama_model_decoder_start_token}} -->
The function `llama_model_decoder_start_token` retrieves the decoder start token ID from a given llama model.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure from which the decoder start token ID is to be retrieved.
- **Control Flow**:
    - Access the `hparams` member of the `llama_model` structure pointed to by `model`.
    - Return the `dec_start_token_id` from the `hparams` structure.
- **Output**: The function returns a `llama_token`, which is the decoder start token ID from the model's hyperparameters.


---
### llama\_model\_is\_recurrent<!-- {{#callable:llama_model_is_recurrent}} -->
The function `llama_model_is_recurrent` determines if a given llama model architecture is recurrent.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure, which contains information about the model's architecture.
- **Control Flow**:
    - The function uses a switch statement to check the `arch` field of the `model` structure.
    - If the architecture matches any of the specified recurrent architectures (LLM_ARCH_MAMBA, LLM_ARCH_RWKV6, LLM_ARCH_RWKV6QWEN2, LLM_ARCH_RWKV7, LLM_ARCH_ARWKV7), the function returns true.
    - If the architecture does not match any of the specified cases, the function returns false.
- **Output**: A boolean value indicating whether the model's architecture is recurrent (true) or not (false).


---
### llama\_internal\_get\_tensor\_map<!-- {{#callable:llama_internal_get_tensor_map}} -->
The function `llama_internal_get_tensor_map` retrieves a reference to a vector of tensor name and pointer pairs from a given llama model.
- **Inputs**:
    - `model`: A pointer to a `llama_model` object from which the tensor map is to be retrieved.
- **Control Flow**:
    - The function directly accesses the `tensors_by_name` member of the `llama_model` structure.
    - It returns this member without any modification or additional processing.
- **Output**: A constant reference to a vector of pairs, where each pair consists of a string (tensor name) and a pointer to a `ggml_tensor`.


---
### llm\_build\_rwkv6\_base<!-- {{#callable:llm_build_rwkv6_base::llm_build_rwkv6_base}} -->
The `llm_build_rwkv6_base` constructor initializes an object with a given model and graph parameters.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object, representing the model to be used.
    - `params`: A constant reference to a `llm_graph_params` object, representing the parameters for the graph context.
- **Control Flow**:
    - The constructor initializes the base class `llm_graph_context` with `params`.
    - The member variable `model` is initialized with the provided `model` argument.
- **Output**: This constructor does not return a value as it is used to initialize an object of the class.


---
### llm\_build\_rwkv7\_base<!-- {{#callable:llm_build_rwkv7_base::llm_build_rwkv7_base}} -->
The `llm_build_rwkv7_base` constructor initializes an object with a given model and parameters by invoking the base class constructor.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object, representing the model to be used.
    - `params`: A constant reference to a `llm_graph_params` object, representing the parameters for the graph context.
- **Control Flow**:
    - The constructor initializes the base class `llm_graph_context` with `params`.
    - The member variable `model` is initialized with the provided `model` argument.
- **Output**: This constructor does not return a value as it is used to initialize an object of the class.


