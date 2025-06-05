# Purpose
This C++ source code file is part of a larger software system designed for processing and encoding images and audio using a model architecture similar to CLIP (Contrastive Language–Image Pretraining). The file provides a comprehensive implementation of various functionalities related to image and audio processing, including image resizing, normalization, and encoding into embeddings using different projector types. The code is structured to support multiple modalities, specifically vision and audio, and includes detailed implementations for handling different projector types such as LLaVA, Qwen2VL, and others.

The file defines several key components, including structures for handling image and audio data, functions for preprocessing and encoding images, and classes for managing the model's parameters and context. It also includes utility functions for image manipulation, such as resizing and cropping, and implements specific preprocessing strategies for different model configurations. The code is designed to be flexible and extensible, allowing for the integration of new projector types and preprocessing methods. It also includes detailed logging and error handling to facilitate debugging and ensure robustness. Overall, this file is a critical part of a system that enables the efficient processing and encoding of visual and audio data for machine learning applications.
# Imports and Dependencies

---
- `clip.h`
- `clip-impl.h`
- `ggml.h`
- `ggml-cpp.h`
- `ggml-cpu.h`
- `ggml-alloc.h`
- `ggml-backend.h`
- `gguf.h`
- `cassert`
- `cmath`
- `cstdlib`
- `cstring`
- `fstream`
- `map`
- `regex`
- `stdexcept`
- `unordered_set`
- `vector`
- `sstream`
- `cinttypes`
- `limits`
- `array`
- `numeric`
- `functional`


# Global Variables

---
### g\_logger\_state
- **Type**: `struct clip_logger_state`
- **Description**: The `g_logger_state` is a global variable of type `struct clip_logger_state` that holds the state of the logger used in the application. It is initialized with a default log level, a callback function for logging, and a null pointer for additional data.
- **Use**: This variable is used to manage and control the logging behavior throughout the application, including setting the log level and handling log messages.


# Data Structures

---
### ffn\_op\_type<!-- {{#data_structure:ffn_op_type}} -->
- **Type**: `enum`
- **Members**:
    - `FFN_GELU`: Represents the GELU (Gaussian Error Linear Unit) operation type.
    - `FFN_GELU_ERF`: Represents the GELU operation using the error function (erf) for approximation.
    - `FFN_SILU`: Represents the SiLU (Sigmoid Linear Unit) operation type.
    - `FFN_GELU_QUICK`: Represents a quick approximation of the GELU operation.
- **Description**: The `ffn_op_type` enum defines a set of operation types for feed-forward neural network layers, specifically focusing on different activation functions. These operation types include various implementations of the GELU activation function, such as the standard GELU, a version using the error function for approximation, and a quick approximation. Additionally, it includes the SiLU activation function. This enum is used to specify the type of activation function to be applied in the feed-forward layers of a neural network model.


---
### norm\_type<!-- {{#data_structure:norm_type}} -->
- **Type**: `enum`
- **Members**:
    - `NORM_TYPE_NORMAL`: Represents a normal normalization type.
    - `NORM_TYPE_RMS`: Represents a root mean square normalization type.
- **Description**: The `norm_type` enum defines two types of normalization methods: `NORM_TYPE_NORMAL` and `NORM_TYPE_RMS`. These are used to specify the type of normalization to be applied in various contexts, such as in machine learning models or data processing pipelines. The `NORM_TYPE_NORMAL` is typically used for standard normalization, while `NORM_TYPE_RMS` is used for root mean square normalization, which is often applied in signal processing or neural network layers.


---
### patch\_merge\_type<!-- {{#data_structure:patch_merge_type}} -->
- **Type**: `enum`
- **Members**:
    - `PATCH_MERGE_FLAT`: Represents a flat patch merge type.
    - `PATCH_MERGE_SPATIAL_UNPAD`: Represents a spatial unpadded patch merge type.
- **Description**: The `patch_merge_type` enum defines two types of patch merging strategies: `PATCH_MERGE_FLAT` and `PATCH_MERGE_SPATIAL_UNPAD`. These strategies are used to specify how patches are merged in a model, with `PATCH_MERGE_FLAT` indicating a straightforward, flat merging approach, and `PATCH_MERGE_SPATIAL_UNPAD` indicating a more complex, spatially aware merging that does not involve padding.


---
### clip\_hparams<!-- {{#data_structure:clip_hparams}} -->
- **Type**: `struct`
- **Members**:
    - `image_size`: Specifies the size of the image.
    - `patch_size`: Defines the size of each patch in the image.
    - `n_embd`: Represents the number of embedding dimensions.
    - `n_ff`: Indicates the number of feed-forward dimensions.
    - `projection_dim`: Specifies the dimension of the projection.
    - `n_head`: Denotes the number of attention heads.
    - `n_layer`: Indicates the number of layers in the model.
    - `proj_scale_factor`: A scaling factor for the projection, default is 0.
    - `image_mean`: An array representing the mean values for image normalization.
    - `image_std`: An array representing the standard deviation values for image normalization.
    - `warmup_image_size`: Specifies a smaller image size for model warmup, default is 0.
    - `warmup_audio_size`: Specifies a smaller audio size for model warmup, default is 3000.
    - `ffn_op`: Defines the type of feed-forward network operation, default is FFN_GELU.
    - `mm_patch_merge_type`: Specifies the type of patch merge, default is PATCH_MERGE_FLAT.
    - `eps`: A small epsilon value for numerical stability, default is 1e-6.
    - `rope_theta`: A parameter for rotational position encoding, default is 0.0.
    - `image_grid_pinpoints`: A vector of integers representing grid pinpoints for the image.
    - `image_crop_resolution`: Specifies the resolution for image cropping.
    - `vision_feature_layer`: An unordered set of integers indicating vision feature layers.
    - `attn_window_size`: Specifies the attention window size, default is 0.
    - `n_wa_pattern`: Indicates the number of window attention patterns, default is 0.
    - `spatial_merge_size`: Specifies the size for spatial merging, default is 0.
    - `n_mel_bins`: Indicates the number of mel bins for audio processing, default is 0.
    - `proj_stack_factor`: A factor for stacking projections, default is 0.
    - `has_llava_projector`: A boolean indicating if the model has a LLaVA projector, default is false.
    - `minicpmv_version`: Specifies the version of the minicpmv, default is 0.
- **Description**: The `clip_hparams` struct is a comprehensive configuration structure used to define various hyperparameters for a CLIP model, including image and audio processing parameters. It includes settings for image size, patch size, embedding dimensions, feed-forward dimensions, projection dimensions, attention heads, and layers. Additionally, it contains parameters for image normalization, model warmup, feed-forward network operations, patch merging, and position encoding. The struct also supports configurations for audio processing, such as mel bins and projection stacking, and includes legacy support for specific model versions.


---
### clip\_layer<!-- {{#data_structure:clip_layer}} -->
- **Type**: `struct`
- **Members**:
    - `k_w`: Pointer to a ggml_tensor representing the key weights for attention.
    - `k_b`: Pointer to a ggml_tensor representing the key biases for attention.
    - `q_w`: Pointer to a ggml_tensor representing the query weights for attention.
    - `q_b`: Pointer to a ggml_tensor representing the query biases for attention.
    - `v_w`: Pointer to a ggml_tensor representing the value weights for attention.
    - `v_b`: Pointer to a ggml_tensor representing the value biases for attention.
    - `o_w`: Pointer to a ggml_tensor representing the output weights for attention.
    - `o_b`: Pointer to a ggml_tensor representing the output biases for attention.
    - `k_norm`: Pointer to a ggml_tensor for key normalization.
    - `q_norm`: Pointer to a ggml_tensor for query normalization.
    - `ln_1_w`: Pointer to a ggml_tensor representing the weights for the first layer normalization.
    - `ln_1_b`: Pointer to a ggml_tensor representing the biases for the first layer normalization.
    - `ff_up_w`: Pointer to a ggml_tensor representing the weights for the feed-forward up projection.
    - `ff_up_b`: Pointer to a ggml_tensor representing the biases for the feed-forward up projection.
    - `ff_gate_w`: Pointer to a ggml_tensor representing the weights for the feed-forward gate.
    - `ff_gate_b`: Pointer to a ggml_tensor representing the biases for the feed-forward gate.
    - `ff_down_w`: Pointer to a ggml_tensor representing the weights for the feed-forward down projection.
    - `ff_down_b`: Pointer to a ggml_tensor representing the biases for the feed-forward down projection.
    - `ln_2_w`: Pointer to a ggml_tensor representing the weights for the second layer normalization.
    - `ln_2_b`: Pointer to a ggml_tensor representing the biases for the second layer normalization.
    - `ls_1_w`: Pointer to a ggml_tensor representing the weights for the first layer scale without bias.
    - `ls_2_w`: Pointer to a ggml_tensor representing the weights for the second layer scale without bias.
- **Description**: The `clip_layer` struct is a data structure used to represent a layer in a neural network model, specifically for attention mechanisms and feed-forward networks. It contains pointers to `ggml_tensor` objects that store weights and biases for various components of the layer, including attention (key, query, value, and output), layer normalization, and feed-forward networks. The struct is designed to facilitate the construction and manipulation of neural network layers, providing a modular approach to defining the parameters required for each layer's operations.


---
### clip\_model<!-- {{#data_structure:clip_model}} -->
- **Type**: `struct`
- **Members**:
    - `modality`: Specifies the modality of the clip model, defaulting to CLIP_MODALITY_VISION.
    - `proj_type`: Defines the type of projector used, defaulting to PROJECTOR_TYPE_MLP.
    - `hparams`: Holds hyperparameters for the clip model.
    - `class_embedding`: Pointer to a ggml_tensor representing the class embedding, initialized to nullptr.
    - `patch_embeddings_0`: Pointer to a ggml_tensor for the first patch embedding, initialized to nullptr.
    - `patch_embeddings_1`: Pointer to a ggml_tensor for the second patch embedding, initialized to nullptr.
    - `patch_bias`: Pointer to a ggml_tensor for the patch bias, initialized to nullptr.
    - `position_embeddings`: Pointer to a ggml_tensor for position embeddings, initialized to nullptr.
    - `pre_ln_w`: Pointer to a ggml_tensor for pre-layer normalization weights, initialized to nullptr.
    - `pre_ln_b`: Pointer to a ggml_tensor for pre-layer normalization biases, initialized to nullptr.
    - `layers`: Vector of clip_layer structures representing the layers of the model.
    - `post_ln_w`: Pointer to a ggml_tensor for post-layer normalization weights.
    - `post_ln_b`: Pointer to a ggml_tensor for post-layer normalization biases.
    - `projection`: Pointer to a ggml_tensor for the projection layer, intended to be renamed to fc.
    - `mm_fc_w`: Pointer to a ggml_tensor for the weights of the multimodal fully connected layer.
    - `mm_fc_b`: Pointer to a ggml_tensor for the biases of the multimodal fully connected layer.
    - `mm_input_norm_w`: Pointer to a ggml_tensor for input normalization weights in LLaVA projection, initialized to nullptr.
    - `mm_0_w`: Pointer to a ggml_tensor for the first weight in LLaVA projection, initialized to nullptr.
    - `mm_0_b`: Pointer to a ggml_tensor for the first bias in LLaVA projection, initialized to nullptr.
    - `mm_2_w`: Pointer to a ggml_tensor for the second weight in LLaVA projection, initialized to nullptr.
    - `mm_2_b`: Pointer to a ggml_tensor for the second bias in LLaVA projection, initialized to nullptr.
    - `image_newline`: Pointer to a ggml_tensor for image newline, initialized to nullptr.
    - `mm_1_w`: Pointer to a ggml_tensor for the first weight in Yi type models, initialized to nullptr.
    - `mm_1_b`: Pointer to a ggml_tensor for the first bias in Yi type models, initialized to nullptr.
    - `mm_3_w`: Pointer to a ggml_tensor for the third weight in Yi type models, initialized to nullptr.
    - `mm_3_b`: Pointer to a ggml_tensor for the third bias in Yi type models, initialized to nullptr.
    - `mm_4_w`: Pointer to a ggml_tensor for the fourth weight in Yi type models, initialized to nullptr.
    - `mm_4_b`: Pointer to a ggml_tensor for the fourth bias in Yi type models, initialized to nullptr.
    - `mm_model_adapter_conv_w`: Pointer to a ggml_tensor for the adapter convolution weights in GLMV-Edge projection, initialized to nullptr.
    - `mm_model_adapter_conv_b`: Pointer to a ggml_tensor for the adapter convolution biases in GLMV-Edge projection, initialized to nullptr.
    - `mm_glm_tok_boi`: Pointer to a ggml_tensor for the beginning of input token in GLMV-Edge projection, initialized to nullptr.
    - `mm_glm_tok_eoi`: Pointer to a ggml_tensor for the end of input token in GLMV-Edge projection, initialized to nullptr.
    - `mm_model_mlp_1_w`: Pointer to a ggml_tensor for the first MLP weight in MobileVLM projection, initialized to nullptr.
    - `mm_model_mlp_1_b`: Pointer to a ggml_tensor for the first MLP bias in MobileVLM projection, initialized to nullptr.
    - `mm_model_mlp_3_w`: Pointer to a ggml_tensor for the third MLP weight in MobileVLM projection, initialized to nullptr.
    - `mm_model_mlp_3_b`: Pointer to a ggml_tensor for the third MLP bias in MobileVLM projection, initialized to nullptr.
    - `mm_model_block_1_block_0_0_w`: Pointer to a ggml_tensor for the first block weight in MobileVLM projection, initialized to nullptr.
    - `mm_model_block_1_block_0_1_w`: Pointer to a ggml_tensor for the second block weight in MobileVLM projection, initialized to nullptr.
    - `mm_model_block_1_block_0_1_b`: Pointer to a ggml_tensor for the second block bias in MobileVLM projection, initialized to nullptr.
    - `mm_model_block_1_block_1_fc1_w`: Pointer to a ggml_tensor for the first fully connected layer weight in MobileVLM projection, initialized to nullptr.
    - `mm_model_block_1_block_1_fc1_b`: Pointer to a ggml_tensor for the first fully connected layer bias in MobileVLM projection, initialized to nullptr.
    - `mm_model_block_1_block_1_fc2_w`: Pointer to a ggml_tensor for the second fully connected layer weight in MobileVLM projection, initialized to nullptr.
    - `mm_model_block_1_block_1_fc2_b`: Pointer to a ggml_tensor for the second fully connected layer bias in MobileVLM projection, initialized to nullptr.
    - `mm_model_block_1_block_2_0_w`: Pointer to a ggml_tensor for the first weight in the second block of MobileVLM projection, initialized to nullptr.
    - `mm_model_block_1_block_2_1_w`: Pointer to a ggml_tensor for the second weight in the second block of MobileVLM projection, initialized to nullptr.
    - `mm_model_block_1_block_2_1_b`: Pointer to a ggml_tensor for the second bias in the second block of MobileVLM projection, initialized to nullptr.
    - `mm_model_block_2_block_0_0_w`: Pointer to a ggml_tensor for the first weight in the third block of MobileVLM projection, initialized to nullptr.
    - `mm_model_block_2_block_0_1_w`: Pointer to a ggml_tensor for the second weight in the third block of MobileVLM projection, initialized to nullptr.
    - `mm_model_block_2_block_0_1_b`: Pointer to a ggml_tensor for the second bias in the third block of MobileVLM projection, initialized to nullptr.
    - `mm_model_block_2_block_1_fc1_w`: Pointer to a ggml_tensor for the first fully connected layer weight in the third block of MobileVLM projection, initialized to nullptr.
    - `mm_model_block_2_block_1_fc1_b`: Pointer to a ggml_tensor for the first fully connected layer bias in the third block of MobileVLM projection, initialized to nullptr.
    - `mm_model_block_2_block_1_fc2_w`: Pointer to a ggml_tensor for the second fully connected layer weight in the third block of MobileVLM projection, initialized to nullptr.
    - `mm_model_block_2_block_1_fc2_b`: Pointer to a ggml_tensor for the second fully connected layer bias in the third block of MobileVLM projection, initialized to nullptr.
    - `mm_model_block_2_block_2_0_w`: Pointer to a ggml_tensor for the first weight in the fourth block of MobileVLM projection, initialized to nullptr.
    - `mm_model_block_2_block_2_1_w`: Pointer to a ggml_tensor for the second weight in the fourth block of MobileVLM projection, initialized to nullptr.
    - `mm_model_block_2_block_2_1_b`: Pointer to a ggml_tensor for the second bias in the fourth block of MobileVLM projection, initialized to nullptr.
    - `mm_model_mlp_0_w`: Pointer to a ggml_tensor for the first MLP weight in MobileVLM_V2 projection, initialized to nullptr.
    - `mm_model_mlp_0_b`: Pointer to a ggml_tensor for the first MLP bias in MobileVLM_V2 projection, initialized to nullptr.
    - `mm_model_mlp_2_w`: Pointer to a ggml_tensor for the second MLP weight in MobileVLM_V2 projection, initialized to nullptr.
    - `mm_model_mlp_2_b`: Pointer to a ggml_tensor for the second MLP bias in MobileVLM_V2 projection, initialized to nullptr.
    - `mm_model_peg_0_w`: Pointer to a ggml_tensor for the first weight in the PEG layer of MobileVLM_V2 projection, initialized to nullptr.
    - `mm_model_peg_0_b`: Pointer to a ggml_tensor for the first bias in the PEG layer of MobileVLM_V2 projection, initialized to nullptr.
    - `mm_model_pos_embed_k`: Pointer to a ggml_tensor for position embedding in MINICPMV projection, initialized to nullptr.
    - `mm_model_query`: Pointer to a ggml_tensor for the query in MINICPMV projection, initialized to nullptr.
    - `mm_model_proj`: Pointer to a ggml_tensor for the projection in MINICPMV projection, initialized to nullptr.
    - `mm_model_kv_proj`: Pointer to a ggml_tensor for the key-value projection in MINICPMV projection, initialized to nullptr.
    - `mm_model_attn_q_w`: Pointer to a ggml_tensor for the attention query weight in MINICPMV projection, initialized to nullptr.
    - `mm_model_attn_q_b`: Pointer to a ggml_tensor for the attention query bias in MINICPMV projection, initialized to nullptr.
    - `mm_model_attn_k_w`: Pointer to a ggml_tensor for the attention key weight in MINICPMV projection, initialized to nullptr.
    - `mm_model_attn_k_b`: Pointer to a ggml_tensor for the attention key bias in MINICPMV projection, initialized to nullptr.
    - `mm_model_attn_v_w`: Pointer to a ggml_tensor for the attention value weight in MINICPMV projection, initialized to nullptr.
    - `mm_model_attn_v_b`: Pointer to a ggml_tensor for the attention value bias in MINICPMV projection, initialized to nullptr.
    - `mm_model_attn_o_w`: Pointer to a ggml_tensor for the attention output weight in MINICPMV projection, initialized to nullptr.
    - `mm_model_attn_o_b`: Pointer to a ggml_tensor for the attention output bias in MINICPMV projection, initialized to nullptr.
    - `mm_model_ln_q_w`: Pointer to a ggml_tensor for the query layer normalization weight in MINICPMV projection, initialized to nullptr.
    - `mm_model_ln_q_b`: Pointer to a ggml_tensor for the query layer normalization bias in MINICPMV projection, initialized to nullptr.
    - `mm_model_ln_kv_w`: Pointer to a ggml_tensor for the key-value layer normalization weight in MINICPMV projection, initialized to nullptr.
    - `mm_model_ln_kv_b`: Pointer to a ggml_tensor for the key-value layer normalization bias in MINICPMV projection, initialized to nullptr.
    - `mm_model_ln_post_w`: Pointer to a ggml_tensor for the post layer normalization weight in MINICPMV projection, initialized to nullptr.
    - `mm_model_ln_post_b`: Pointer to a ggml_tensor for the post layer normalization bias in MINICPMV projection, initialized to nullptr.
    - `mm_input_proj_w`: Pointer to a ggml_tensor for the input projection weight in gemma3, initialized to nullptr.
    - `mm_soft_emb_norm_w`: Pointer to a ggml_tensor for the soft embedding normalization weight in gemma3, initialized to nullptr.
    - `token_embd_img_break`: Pointer to a ggml_tensor for the image break token embedding in pixtral, initialized to nullptr.
    - `mm_patch_merger_w`: Pointer to a ggml_tensor for the patch merger weight in pixtral, initialized to nullptr.
    - `conv1d_1_w`: Pointer to a ggml_tensor for the first 1D convolution weight in ultravox/whisper encoder, initialized to nullptr.
    - `conv1d_1_b`: Pointer to a ggml_tensor for the first 1D convolution bias in ultravox/whisper encoder, initialized to nullptr.
    - `conv1d_2_w`: Pointer to a ggml_tensor for the second 1D convolution weight in ultravox/whisper encoder, initialized to nullptr.
    - `conv1d_2_b`: Pointer to a ggml_tensor for the second 1D convolution bias in ultravox/whisper encoder, initialized to nullptr.
    - `mm_norm_pre_w`: Pointer to a ggml_tensor for the pre-normalization weight in ultravox/whisper encoder, initialized to nullptr.
    - `mm_norm_mid_w`: Pointer to a ggml_tensor for the mid-normalization weight in ultravox/whisper encoder, initialized to nullptr.
- **Description**: The `clip_model` struct is a comprehensive data structure designed to encapsulate the various components and configurations of a CLIP (Contrastive Language–Image Pretraining) model. It includes fields for specifying the modality and projector type, as well as a collection of hyperparameters. The struct also contains numerous pointers to `ggml_tensor` objects, which represent different layers and components of the model, such as embeddings, layer normalizations, and projections. These components are initialized to `nullptr` and are intended to be set up during the model's initialization phase. The struct supports various projection types and configurations, making it versatile for different CLIP model implementations.


---
### clip\_ctx<!-- {{#data_structure:clip_ctx}} -->
- **Type**: `struct`
- **Members**:
    - `model`: An instance of the `clip_model` structure, representing the model configuration and parameters.
    - `ctx_gguf`: A pointer to a `gguf_context` structure, used for managing GGUF context.
    - `ctx_data`: A pointer to a `ggml_context` structure, used for managing GGML context.
    - `buf_compute_meta`: A vector of `uint8_t` used to store metadata for computation buffers.
    - `backend_ptrs`: A vector of `ggml_backend_t` representing pointers to backend instances.
    - `backend_buft`: A vector of `ggml_backend_buffer_type_t` representing types of backend buffers.
    - `backend`: A `ggml_backend_t` representing the current backend in use.
    - `backend_cpu`: A `ggml_backend_t` representing the CPU backend.
    - `buf`: A pointer to a `ggml_backend_buffer` structure, used for managing backend buffers.
    - `max_nodes`: An integer representing the maximum number of nodes, defaulting to 8192.
    - `sched`: A pointer to a `ggml_backend_sched` structure, used for managing backend scheduling.
    - `debug_graph`: A boolean indicating whether debugging for the graph is enabled, defaulting to false.
    - `debug_print_tensors`: A vector of pointers to `ggml_tensor` used for storing tensors to be printed for debugging.
- **Description**: The `clip_ctx` structure is a comprehensive context for managing the execution and configuration of a CLIP model. It encapsulates various components such as the model parameters, backend contexts, and scheduling information. The structure is designed to facilitate the initialization, execution, and debugging of the CLIP model, providing pointers to necessary contexts and buffers, as well as options for backend management and debugging. It supports both CPU and GPU backends, allowing for flexible deployment and execution of the model.
- **Member Functions**:
    - [`clip_ctx::clip_ctx`](#clip_ctxclip_ctx)
    - [`clip_ctx::~clip_ctx`](#clip_ctxclip_ctx)
    - [`clip_ctx::proj_type`](#clip_ctxproj_type)

**Methods**

---
#### clip\_ctx::clip\_ctx<!-- {{#callable:clip_ctx::clip_ctx}} -->
The `clip_ctx` constructor initializes a CLIP context by setting up the CPU and optionally GPU backends, and configuring the backend scheduler for computation.
- **Inputs**:
    - `ctx_params`: A reference to a `clip_context_params` structure that contains parameters for initializing the CLIP context, including whether to use a GPU.
- **Control Flow**:
    - Check if the environment variable 'MTMD_DEBUG_GRAPH' is set to enable debug graph mode.
    - Initialize the CPU backend using `ggml_backend_init_by_type` with `GGML_BACKEND_DEVICE_TYPE_CPU`.
    - If the CPU backend initialization fails, throw a runtime error.
    - Initialize the GPU backend if `ctx_params.use_gpu` is true, otherwise set it to `nullptr`.
    - If the GPU backend is initialized, log the backend type and add it to the backend pointers and buffer types lists.
    - If the GPU backend is not initialized, set the backend to the CPU backend and log the usage of the CPU backend.
    - Add the CPU backend to the backend pointers and buffer types lists.
    - Reset the scheduler with the initialized backends and buffer types, setting the maximum nodes to 8192.
- **Output**: The function does not return a value; it initializes the `clip_ctx` object.
- **Functions called**:
    - [`ggml_backend_name`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_name)
- **See also**: [`clip_ctx`](#clip_ctx)  (Data Structure)


---
#### clip\_ctx::\~clip\_ctx<!-- {{#callable:clip_ctx::~clip_ctx}} -->
The destructor `~clip_ctx()` releases resources by freeing the backend and, if necessary, the CPU backend.
- **Inputs**: None
- **Control Flow**:
    - Call [`ggml_backend_free`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_free) to free the `backend`.
    - Check if `backend` is not equal to `backend_cpu`.
    - If true, call [`ggml_backend_free`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_free) to free `backend_cpu`.
- **Output**: The function does not return any value as it is a destructor.
- **Functions called**:
    - [`ggml_backend_free`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_free)
- **See also**: [`clip_ctx`](#clip_ctx)  (Data Structure)


---
#### clip\_ctx::proj\_type<!-- {{#callable:clip_ctx::proj_type}} -->
The `proj_type` function returns the projector type of the model within the `clip_ctx` structure.
- **Inputs**: None
- **Control Flow**:
    - Access the `model` member of the `clip_ctx` structure.
    - Return the `proj_type` member of the `model`.
- **Output**: The function returns a `projector_type` which indicates the type of projector used in the model.
- **See also**: [`clip_ctx`](#clip_ctx)  (Data Structure)



---
### clip\_graph<!-- {{#data_structure:clip_graph}} -->
- **Type**: `struct`
- **Members**:
    - `ctx`: A pointer to a `clip_ctx` structure, representing the context for the clip graph.
    - `model`: A reference to a `clip_model` structure, representing the model used in the clip graph.
    - `hparams`: A reference to a `clip_hparams` structure, representing the hyperparameters for the clip graph.
    - `img`: A reference to a `clip_image_f32` structure, representing the image data used in the clip graph.
    - `patch_size`: An integer representing the size of each patch in the image.
    - `n_patches_x`: An integer representing the number of patches along the x-axis.
    - `n_patches_y`: An integer representing the number of patches along the y-axis.
    - `n_patches`: An integer representing the total number of patches in the image.
    - `n_embd`: An integer representing the number of embeddings.
    - `n_head`: An integer representing the number of attention heads.
    - `d_head`: An integer representing the dimension of each attention head.
    - `n_layer`: An integer representing the number of layers in the model.
    - `eps`: A float representing the epsilon value used for numerical stability in normalization.
    - `kq_scale`: A float representing the scaling factor for the key-query attention mechanism.
    - `ctx0_ptr`: A smart pointer to a `ggml_context` structure, managing the context for the graph.
    - `ctx0`: A pointer to a `ggml_context` structure, representing the current context for the graph.
    - `gf`: A pointer to a `ggml_cgraph` structure, representing the computational graph for the clip model.
- **Description**: The `clip_graph` struct is a data structure used to represent a computational graph for processing images in a CLIP model. It contains references to the model, hyperparameters, and image data, as well as various parameters related to the structure of the model, such as the number of patches, embeddings, and layers. The struct also manages the context and computational graph used for processing the image data, allowing for the construction and execution of the graph to generate embeddings or other outputs from the input image.
- **Member Functions**:
    - [`clip_graph::clip_graph`](#clip_graphclip_graph)
    - [`clip_graph::build_siglip`](#clip_graphbuild_siglip)
    - [`clip_graph::build_pixtral`](#clip_graphbuild_pixtral)
    - [`clip_graph::build_qwen2vl`](#clip_graphbuild_qwen2vl)
    - [`clip_graph::build_minicpmv`](#clip_graphbuild_minicpmv)
    - [`clip_graph::build_internvl`](#clip_graphbuild_internvl)
    - [`clip_graph::build_llama4`](#clip_graphbuild_llama4)
    - [`clip_graph::build_llava`](#clip_graphbuild_llava)
    - [`clip_graph::build_whisper_enc`](#clip_graphbuild_whisper_enc)
    - [`clip_graph::cb`](#clip_graphcb)
    - [`clip_graph::build_vit`](#clip_graphbuild_vit)
    - [`clip_graph::build_inp`](#clip_graphbuild_inp)
    - [`clip_graph::build_inp_raw`](#clip_graphbuild_inp_raw)
    - [`clip_graph::build_norm`](#clip_graphbuild_norm)
    - [`clip_graph::build_ffn`](#clip_graphbuild_ffn)
    - [`clip_graph::build_attn`](#clip_graphbuild_attn)
    - [`clip_graph::build_rope_2d`](#clip_graphbuild_rope_2d)

**Methods**

---
#### clip\_graph::clip\_graph<!-- {{#callable:clip_graph::clip_graph}} -->
The `clip_graph` constructor initializes a graph for processing a single image using the CLIP model, setting up various parameters and contexts needed for computation.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure, which contains the model and other context information for the CLIP computation.
    - `img`: A constant reference to a `clip_image_f32` object representing the image to be processed.
- **Control Flow**:
    - Initialize member variables using the provided context and image, including model parameters and image dimensions.
    - Calculate the number of patches in the x and y dimensions based on the image size and patch size.
    - Compute the total number of patches by multiplying the number of patches in the x and y dimensions.
    - Set up the embedding dimension, number of heads, and other model hyperparameters from the context's model parameters.
    - Initialize a `ggml_init_params` structure with memory size and buffer information from the context.
    - Create a new `ggml_context` using the initialized parameters and store it in `ctx0_ptr`.
    - Retrieve the raw pointer to the `ggml_context` from `ctx0_ptr` and assign it to `ctx0`.
    - Create a new custom graph `gf` using the [`ggml_new_graph_custom`](../../ggml/src/ggml.c.driver.md#ggml_new_graph_custom) function with the initialized context.
- **Output**: The constructor does not return a value; it initializes the `clip_graph` object with the necessary parameters and contexts for further processing.
- **Functions called**:
    - [`ggml_init`](../../ggml/src/ggml.c.driver.md#ggml_init)
    - [`ggml_new_graph_custom`](../../ggml/src/ggml.c.driver.md#ggml_new_graph_custom)
- **See also**: [`clip_graph`](#clip_graph)  (Data Structure)


---
#### clip\_graph::build\_siglip<!-- {{#callable:clip_graph::build_siglip}} -->
The `build_siglip` function constructs a computational graph for a vision transformer model with specific projector types, processing input tensors and applying transformations based on the projector type.
- **Inputs**: None
- **Control Flow**:
    - Initialize input tensor `inp` using `build_inp()` and process it with `build_vit()` to get `cur`.
    - Check the projector type using `ctx->proj_type()`.
    - If the projector type is `PROJECTOR_TYPE_GEMMA3`, perform a series of tensor transformations including transposing, reshaping, pooling, and applying normalization and projection.
    - If the projector type is `PROJECTOR_TYPE_IDEFICS3`, reshape and permute the tensor `cur`, then apply a matrix multiplication with the model's projection tensor.
    - If the projector type is unsupported, abort with an error message.
    - Build the forward computational graph using `ggml_build_forward_expand(gf, cur)`.
    - Return the constructed computational graph `gf`.
- **Output**: The function returns a pointer to a `ggml_cgraph`, which represents the constructed computational graph.
- **Functions called**:
    - [`clip_graph::build_inp`](#clip_graphbuild_inp)
    - [`clip_graph::build_vit`](#clip_graphbuild_vit)
    - [`ggml_cont`](../../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_transpose`](../../ggml/src/ggml.c.driver.md#ggml_transpose)
    - [`ggml_reshape_4d`](../../ggml/src/ggml.c.driver.md#ggml_reshape_4d)
    - [`ggml_pool_2d`](../../ggml/src/ggml.c.driver.md#ggml_pool_2d)
    - [`ggml_reshape_3d`](../../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rms_norm`](../../ggml/src/ggml.c.driver.md#ggml_rms_norm)
    - [`ggml_mul`](../../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`ggml_mul_mat`](../../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_permute`](../../ggml/src/ggml.c.driver.md#ggml_permute)
    - [`ggml_build_forward_expand`](../../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`clip_graph`](#clip_graph)  (Data Structure)


---
#### clip\_graph::build\_pixtral<!-- {{#callable:clip_graph::build_pixtral}} -->
The `build_pixtral` function constructs a computational graph for processing image data using a vision transformer model with specific configurations and returns the graph.
- **Inputs**:
    - `None`: This function does not take any direct input parameters.
- **Control Flow**:
    - Initialize the number of merges from hyperparameters.
    - Create 2D input position tensors for height and width, set their names, and mark them as inputs.
    - Define a lambda function `add_pos` to add positional encoding using 2D RoPE to a given tensor.
    - Build the input tensor using [`build_inp`](#clip_graphbuild_inp) and pass it to [`build_vit`](#clip_graphbuild_vit) to construct the vision transformer model with RMS normalization and the `add_pos` function for positional encoding.
    - Check if the model has a patch merger weight; if so, apply RMS normalization, reshape the tensor to a 2D grid, and perform im2col operation for patch merging.
    - Apply a multi-modal projector using GELU activation, performing matrix multiplications and adding biases if available.
    - Arrange the [IMG_BREAK] token by reshaping the tensor, creating a new tensor for the token, and concatenating it to the end of each row.
    - Build the forward computational graph using [`ggml_build_forward_expand`](../../ggml/src/ggml.c.driver.md#ggml_build_forward_expand) with the final tensor.
    - Return the constructed computational graph.
- **Output**: The function returns a pointer to a `ggml_cgraph`, which represents the constructed computational graph for the image processing model.
- **Functions called**:
    - [`ggml_new_tensor_1d`](../../ggml/src/ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_set_name`](../../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_set_input`](../../ggml/src/ggml.c.driver.md#ggml_set_input)
    - [`clip_graph::build_rope_2d`](#clip_graphbuild_rope_2d)
    - [`clip_graph::build_inp`](#clip_graphbuild_inp)
    - [`clip_graph::build_vit`](#clip_graphbuild_vit)
    - [`ggml_mul`](../../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`ggml_rms_norm`](../../ggml/src/ggml.c.driver.md#ggml_rms_norm)
    - [`ggml_reshape_3d`](../../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_permute`](../../ggml/src/ggml.c.driver.md#ggml_permute)
    - [`ggml_cont`](../../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_view_3d`](../../ggml/src/ggml.c.driver.md#ggml_view_3d)
    - [`ggml_im2col`](../../ggml/src/ggml.c.driver.md#ggml_im2col)
    - [`ggml_reshape_2d`](../../ggml/src/ggml.c.driver.md#ggml_reshape_2d)
    - [`ggml_mul_mat`](../../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_add`](../../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_gelu`](../../ggml/src/ggml.c.driver.md#ggml_gelu)
    - [`ggml_new_tensor_3d`](../../ggml/src/ggml.c.driver.md#ggml_new_tensor_3d)
    - [`ggml_scale`](../../ggml/src/ggml.c.driver.md#ggml_scale)
    - [`ggml_concat`](../../ggml/src/ggml.c.driver.md#ggml_concat)
    - [`ggml_view_2d`](../../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_row_size`](../../ggml/src/ggml.c.driver.md#ggml_row_size)
    - [`ggml_build_forward_expand`](../../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`clip_graph`](#clip_graph)  (Data Structure)


---
#### clip\_graph::build\_qwen2vl<!-- {{#callable:clip_graph::build_qwen2vl}} -->
The `build_qwen2vl` function constructs a computation graph for the Qwen2VL model using M-RoPE and potentially windowed attention, processing image patches through a series of transformations and attention layers to produce multimodal embeddings.
- **Inputs**:
    - `None`: The function does not take any direct input parameters; it operates on the class members and context.
- **Control Flow**:
    - Assert that `model.patch_bias` and `model.class_embedding` are null, ensuring preconditions for the model.
    - Initialize constants and determine if windowed attention is used based on hyperparameters.
    - Create input tensors for raw image data and apply a 2D convolution to generate initial embeddings.
    - Assert image dimensions are compatible with the patch size, ensuring correct input size.
    - Apply a second convolution and reshape the input tensor to prepare it for processing through the model layers.
    - Initialize tensors for positions and window attention indices if applicable.
    - Apply pre-layer normalization if weights are available.
    - If windowed attention is used, prepare indices and masks for windowed attention processing.
    - Iterate over each layer, applying layer normalization, self-attention with M-RoPE, and feed-forward network transformations.
    - Re-add residual connections after each layer's transformations.
    - Apply post-layer normalization if weights are available.
    - Perform a multimodal projection using linear transformations and GELU activation.
    - If windowed attention is used, adjust the embeddings using window indices.
    - Build the computation graph with the final embeddings and return it.
- **Output**: The function returns a pointer to a `ggml_cgraph`, which represents the constructed computation graph for the Qwen2VL model.
- **Functions called**:
    - [`clip_graph::build_inp_raw`](#clip_graphbuild_inp_raw)
    - [`ggml_conv_2d`](../../ggml/src/ggml.c.driver.md#ggml_conv_2d)
    - [`ggml_add`](../../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_cont`](../../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_permute`](../../ggml/src/ggml.c.driver.md#ggml_permute)
    - [`ggml_reshape_4d`](../../ggml/src/ggml.c.driver.md#ggml_reshape_4d)
    - [`ggml_reshape_3d`](../../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_new_tensor_1d`](../../ggml/src/ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_set_name`](../../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_set_input`](../../ggml/src/ggml.c.driver.md#ggml_set_input)
    - [`clip_graph::build_norm`](#clip_graphbuild_norm)
    - [`ggml_new_tensor_2d`](../../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d)
    - [`ggml_reshape_2d`](../../ggml/src/ggml.c.driver.md#ggml_reshape_2d)
    - [`ggml_get_rows`](../../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`clip_graph::cb`](#clip_graphcb)
    - [`ggml_mul_mat`](../../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_rope_multi`](../../ggml/src/ggml.c.driver.md#ggml_rope_multi)
    - [`clip_graph::build_attn`](#clip_graphbuild_attn)
    - [`clip_graph::build_ffn`](#clip_graphbuild_ffn)
    - [`ggml_gelu`](../../ggml/src/ggml.c.driver.md#ggml_gelu)
    - [`ggml_build_forward_expand`](../../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`clip_graph`](#clip_graph)  (Data Structure)


---
#### clip\_graph::build\_minicpmv<!-- {{#callable:clip_graph::build_minicpmv}} -->
The `build_minicpmv` function constructs a computational graph for a MiniCPMV model using a series of tensor operations and transformations.
- **Inputs**: None
- **Control Flow**:
    - Initialize a batch size of 1 and assert that the model's class embedding is null.
    - Determine the number of positions (`n_pos`) based on the number of patches.
    - Create a 3D tensor for position embeddings and set its name and input status.
    - Create a 1D tensor for positions, set its name and input status, and retrieve learned position embeddings.
    - Build the input tensor using the [`build_inp`](#clip_graphbuild_inp) function.
    - Construct the ViT (Vision Transformer) embeddings using the [`build_vit`](#clip_graphbuild_vit) function with the input tensor and learned position embeddings.
    - Perform matrix multiplication to project the embeddings and queries using the model's query and key-value projection matrices.
    - Normalize the query and value tensors using the [`build_norm`](#clip_graphbuild_norm) function.
    - Add the position embeddings to the value tensor to form the key tensor.
    - Perform attention operations by calculating Q, K, and V matrices, reshaping them, and applying the [`build_attn`](#clip_graphbuild_attn) function.
    - Normalize the resulting embeddings using the [`build_norm`](#clip_graphbuild_norm) function.
    - Project the embeddings using matrix multiplication with the model's projection matrix.
    - Expand the computational graph with the final embeddings tensor.
    - Return the constructed computational graph.
- **Output**: The function returns a pointer to a `ggml_cgraph`, which represents the constructed computational graph for the MiniCPMV model.
- **Functions called**:
    - [`clip_n_mmproj_embd`](#clip_n_mmproj_embd)
    - [`ggml_new_tensor_3d`](../../ggml/src/ggml.c.driver.md#ggml_new_tensor_3d)
    - [`ggml_set_name`](../../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_set_input`](../../ggml/src/ggml.c.driver.md#ggml_set_input)
    - [`ggml_new_tensor_1d`](../../ggml/src/ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_get_rows`](../../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`clip_graph::build_inp`](#clip_graphbuild_inp)
    - [`clip_graph::build_vit`](#clip_graphbuild_vit)
    - [`ggml_mul_mat`](../../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`clip_graph::build_norm`](#clip_graphbuild_norm)
    - [`ggml_add`](../../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_reshape_3d`](../../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`clip_graph::cb`](#clip_graphcb)
    - [`clip_graph::build_attn`](#clip_graphbuild_attn)
    - [`ggml_build_forward_expand`](../../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`clip_graph`](#clip_graph)  (Data Structure)


---
#### clip\_graph::build\_internvl<!-- {{#callable:clip_graph::build_internvl}} -->
The `build_internvl` function constructs a computational graph for processing image data using a Vision Transformer (ViT) model with specific configurations for the InternVL model.
- **Inputs**:
    - `None`: This function does not take any direct input arguments.
- **Control Flow**:
    - Assert that `model.class_embedding` and `model.position_embeddings` are not null.
    - Calculate the number of positions `n_pos` as `n_patches + 1`.
    - Call `build_inp()` to construct the input tensor `inp`.
    - Concatenate the `class_embedding` to `inp` using [`ggml_concat`](../../ggml/src/ggml.c.driver.md#ggml_concat).
    - Determine the normalization type `norm_t` based on model parameters.
    - Call `build_vit()` to process the input tensor with the ViT model, using the determined normalization type and position embeddings.
    - Remove the CLS token from the processed tensor using [`ggml_view_2d`](../../ggml/src/ggml.c.driver.md#ggml_view_2d).
    - Perform a pixel shuffle operation on the tensor, involving reshaping and permuting dimensions.
    - Apply a projector with GELU activation, including normalization and matrix multiplications.
    - Build the computational graph using [`ggml_build_forward_expand`](../../ggml/src/ggml.c.driver.md#ggml_build_forward_expand).
    - Return the constructed computational graph `gf`.
- **Output**: The function returns a pointer to a `ggml_cgraph`, which represents the constructed computational graph for the InternVL model.
- **Functions called**:
    - [`clip_graph::build_inp`](#clip_graphbuild_inp)
    - [`ggml_concat`](../../ggml/src/ggml.c.driver.md#ggml_concat)
    - [`clip_graph::build_vit`](#clip_graphbuild_vit)
    - [`ggml_view_2d`](../../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_row_size`](../../ggml/src/ggml.c.driver.md#ggml_row_size)
    - [`ggml_reshape_4d`](../../ggml/src/ggml.c.driver.md#ggml_reshape_4d)
    - [`ggml_permute`](../../ggml/src/ggml.c.driver.md#ggml_permute)
    - [`ggml_cont`](../../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_reshape_2d`](../../ggml/src/ggml.c.driver.md#ggml_reshape_2d)
    - [`clip_graph::build_norm`](#clip_graphbuild_norm)
    - [`ggml_mul_mat`](../../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_add`](../../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_gelu`](../../ggml/src/ggml.c.driver.md#ggml_gelu)
    - [`ggml_build_forward_expand`](../../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`clip_graph`](#clip_graph)  (Data Structure)


---
#### clip\_graph::build\_llama4<!-- {{#callable:clip_graph::build_llama4}} -->
The `build_llama4` function constructs a computation graph for the Llama4 model, which processes image data through a series of transformations including convolution, position embedding, and multi-modal projection.
- **Inputs**:
    - `None`: The function does not take any direct input parameters.
- **Control Flow**:
    - Assert that `model.class_embedding` and `model.position_embeddings` are not null.
    - Calculate `n_pos` as `n_patches + 1` for the [CLS] token.
    - Create 1D tensors `pos_h` and `pos_w` for 2D input positions and set them as inputs.
    - Call `build_inp_raw()` to get the raw input tensor.
    - Perform a convolution operation using [`ggml_im2col`](../../ggml/src/ggml.c.driver.md#ggml_im2col) and [`ggml_mul_mat`](../../ggml/src/ggml.c.driver.md#ggml_mul_mat) to transform the input tensor.
    - Concatenate the [CLS] token to the input tensor.
    - Define a lambda function `add_pos` to add 2D position embeddings using [`build_rope_2d`](#clip_graphbuild_rope_2d).
    - Build a Vision Transformer (ViT) using [`build_vit`](#clip_graphbuild_vit) with the input tensor and position embeddings.
    - Remove the [CLS] token from the output of the ViT.
    - Perform a pixel shuffle operation to rearrange the tensor dimensions.
    - Apply a multi-layer perceptron (MLP) with GELU activation to the shuffled tensor.
    - Project the final tensor using [`ggml_mul_mat`](../../ggml/src/ggml.c.driver.md#ggml_mul_mat) with `model.mm_model_proj`.
    - Expand the computation graph with the final tensor and return it.
- **Output**: The function returns a pointer to a `ggml_cgraph`, which represents the constructed computation graph for the Llama4 model.
- **Functions called**:
    - [`ggml_new_tensor_1d`](../../ggml/src/ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_set_name`](../../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_set_input`](../../ggml/src/ggml.c.driver.md#ggml_set_input)
    - [`clip_graph::build_inp_raw`](#clip_graphbuild_inp_raw)
    - [`ggml_reshape_4d`](../../ggml/src/ggml.c.driver.md#ggml_reshape_4d)
    - [`ggml_im2col`](../../ggml/src/ggml.c.driver.md#ggml_im2col)
    - [`ggml_mul_mat`](../../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_reshape_2d`](../../ggml/src/ggml.c.driver.md#ggml_reshape_2d)
    - [`clip_graph::cb`](#clip_graphcb)
    - [`ggml_concat`](../../ggml/src/ggml.c.driver.md#ggml_concat)
    - [`clip_graph::build_rope_2d`](#clip_graphbuild_rope_2d)
    - [`clip_graph::build_vit`](#clip_graphbuild_vit)
    - [`ggml_view_2d`](../../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_row_size`](../../ggml/src/ggml.c.driver.md#ggml_row_size)
    - [`ggml_permute`](../../ggml/src/ggml.c.driver.md#ggml_permute)
    - [`ggml_cont`](../../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_gelu`](../../ggml/src/ggml.c.driver.md#ggml_gelu)
    - [`ggml_build_forward_expand`](../../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`clip_graph`](#clip_graph)  (Data Structure)


---
#### clip\_graph::build\_llava<!-- {{#callable:clip_graph::build_llava}} -->
The `build_llava` function constructs a computation graph for processing image data using a vision transformer model with optional projector types and feature layers.
- **Inputs**:
    - `None`: The function does not take any input parameters.
- **Control Flow**:
    - Initialize constants for batch size and position count based on model parameters.
    - Assert that only square images are supported by checking patch dimensions.
    - Determine the deepest feature layer based on hyperparameters and projector type.
    - Build the input tensor using the [`build_inp`](#clip_graphbuild_inp) function.
    - Concatenate class embeddings with patch embeddings if class embeddings are present.
    - Create a new tensor for positions and set it as input.
    - Add position embeddings to the input tensor.
    - Apply pre-layer normalization if weights are available.
    - Iterate over each layer up to the maximum feature layer, performing layer normalization, self-attention, and feed-forward network operations.
    - Store outputs of specified vision feature layers in an embedding stack.
    - Apply post-layer normalization if weights are available.
    - Process vision feature layers by stacking them if multiple are present.
    - Apply the appropriate projector type (e.g., MLP, MLP_NORM, LDP, LDPV2, GLM_EDGE) to the embeddings based on the model's projector type.
    - Build the computation graph using the final embeddings.
- **Output**: Returns a pointer to a `ggml_cgraph` object representing the constructed computation graph.
- **Functions called**:
    - [`clip_graph::build_inp`](#clip_graphbuild_inp)
    - [`ggml_concat`](../../ggml/src/ggml.c.driver.md#ggml_concat)
    - [`ggml_new_tensor_1d`](../../ggml/src/ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_set_name`](../../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_set_input`](../../ggml/src/ggml.c.driver.md#ggml_set_input)
    - [`ggml_add`](../../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_get_rows`](../../ggml/src/ggml.c.driver.md#ggml_get_rows)
    - [`clip_graph::build_norm`](#clip_graphbuild_norm)
    - [`clip_graph::cb`](#clip_graphcb)
    - [`ggml_mul_mat`](../../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_reshape_3d`](../../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`clip_graph::build_attn`](#clip_graphbuild_attn)
    - [`clip_graph::build_ffn`](#clip_graphbuild_ffn)
    - [`ggml_reshape_2d`](../../ggml/src/ggml.c.driver.md#ggml_reshape_2d)
    - [`ggml_gelu`](../../ggml/src/ggml.c.driver.md#ggml_gelu)
    - [`ggml_norm`](../../ggml/src/ggml.c.driver.md#ggml_norm)
    - [`ggml_mul`](../../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`ggml_cont`](../../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_permute`](../../ggml/src/ggml.c.driver.md#ggml_permute)
    - [`ggml_reshape_4d`](../../ggml/src/ggml.c.driver.md#ggml_reshape_4d)
    - [`ggml_conv_2d_dw`](../../ggml/src/ggml.c.driver.md#ggml_conv_2d_dw)
    - [`ggml_hardswish`](../../ggml/src/ggml.c.driver.md#ggml_hardswish)
    - [`ggml_pool_2d`](../../ggml/src/ggml.c.driver.md#ggml_pool_2d)
    - [`ggml_relu`](../../ggml/src/ggml.c.driver.md#ggml_relu)
    - [`ggml_hardsigmoid`](../../ggml/src/ggml.c.driver.md#ggml_hardsigmoid)
    - [`ggml_conv_2d`](../../ggml/src/ggml.c.driver.md#ggml_conv_2d)
    - [`ggml_gelu_inplace`](../../ggml/src/ggml.c.driver.md#ggml_gelu_inplace)
    - [`ggml_silu_inplace`](../../ggml/src/ggml.c.driver.md#ggml_silu_inplace)
    - [`ggml_build_forward_expand`](../../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`clip_graph`](#clip_graph)  (Data Structure)


---
#### clip\_graph::build\_whisper\_enc<!-- {{#callable:clip_graph::build_whisper_enc}} -->
The `build_whisper_enc` function constructs a computation graph for a Whisper encoder with a custom projector, processing input data through convolutional and transformer layers, and applying different projection types based on the context's projector type.
- **Inputs**:
    - `None`: The function does not take any explicit input parameters; it operates on the internal state of the `clip_graph` object, particularly using the `img`, `model`, `hparams`, and `ctx` attributes.
- **Control Flow**:
    - Initialize the number of frames and positions based on the image dimensions.
    - Assert that the model's position embeddings can accommodate the calculated positions.
    - Create a raw input tensor using [`build_inp_raw`](#clip_graphbuild_inp_raw).
    - Perform a series of 1D convolutions and GELU activations on the input tensor, followed by a transpose operation.
    - Assert the presence of necessary model layer parameters for sanity checks.
    - Select position embeddings based on the calculated positions.
    - Build a vision transformer graph using [`build_vit`](#clip_graphbuild_vit) with the input tensor and selected position embeddings.
    - Check the projector type from the context and apply the corresponding projection logic:
    - - For `PROJECTOR_TYPE_ULTRAVOX`, stack audio frames, apply pre-norm, perform a feed-forward network with SwiGLU activation, and apply mid-norm.
    - - For `PROJECTOR_TYPE_QWEN2A`, apply a simple matrix multiplication and addition for projection.
    - Abort if an unknown projector type is encountered.
    - Add the final projected tensor to the computation graph.
    - Return the constructed computation graph.
- **Output**: The function returns a pointer to a `ggml_cgraph` object, representing the constructed computation graph for the Whisper encoder.
- **Functions called**:
    - [`clip_graph::build_inp_raw`](#clip_graphbuild_inp_raw)
    - [`ggml_conv_1d_ph`](../../ggml/src/ggml.c.driver.md#ggml_conv_1d_ph)
    - [`ggml_add`](../../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_gelu_erf`](../../ggml/src/ggml.c.driver.md#ggml_gelu_erf)
    - [`ggml_cont`](../../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_transpose`](../../ggml/src/ggml.c.driver.md#ggml_transpose)
    - [`clip_graph::cb`](#clip_graphcb)
    - [`ggml_view_2d`](../../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`clip_graph::build_vit`](#clip_graphbuild_vit)
    - [`ggml_nelements`](../../ggml/src/ggml.c.driver.md#ggml_nelements)
    - [`ggml_view_1d`](../../ggml/src/ggml.c.driver.md#ggml_view_1d)
    - [`ggml_pad`](../../ggml/src/ggml.c.driver.md#ggml_pad)
    - [`ggml_row_size`](../../ggml/src/ggml.c.driver.md#ggml_row_size)
    - [`ggml_rms_norm`](../../ggml/src/ggml.c.driver.md#ggml_rms_norm)
    - [`ggml_mul`](../../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`ggml_mul_mat`](../../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_element_size`](../../ggml/src/ggml.c.driver.md#ggml_element_size)
    - [`ggml_silu`](../../ggml/src/ggml.c.driver.md#ggml_silu)
    - [`ggml_build_forward_expand`](../../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`clip_graph`](#clip_graph)  (Data Structure)


---
#### clip\_graph::cb<!-- {{#callable:clip_graph::cb}} -->
The `cb` function is a utility function that conditionally copies a tensor, assigns it a name, sets it as an output, and adds it to a debug list if debugging is enabled.
- **Inputs**:
    - `cur0`: A pointer to a `ggml_tensor` object representing the current tensor to be processed.
    - `name`: A constant character pointer representing the base name to be assigned to the tensor.
    - `il`: An integer representing the layer index, used to differentiate tensor names if non-negative.
- **Control Flow**:
    - Check if the `debug_graph` flag in the context (`ctx`) is true.
    - If true, create a copy of the tensor `cur0` using [`ggml_cpy`](../../ggml/src/ggml.c.driver.md#ggml_cpy) and [`ggml_dup_tensor`](../../ggml/src/ggml.c.driver.md#ggml_dup_tensor).
    - Construct a name for the tensor by appending the layer index `il` to `name` if `il` is non-negative, otherwise use `name` as is.
    - Set the name of the copied tensor using [`ggml_set_name`](../../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Mark the copied tensor as an output using [`ggml_set_output`](../../ggml/src/ggml.c.driver.md#ggml_set_output).
    - Add the copied tensor to the forward graph using [`ggml_build_forward_expand`](../../ggml/src/ggml.c.driver.md#ggml_build_forward_expand).
    - Append the copied tensor to the `debug_print_tensors` list in the context.
- **Output**: The function does not return any value; it operates by side effects on the context and the graph.
- **Functions called**:
    - [`ggml_cpy`](../../ggml/src/ggml.c.driver.md#ggml_cpy)
    - [`ggml_dup_tensor`](../../ggml/src/ggml.c.driver.md#ggml_dup_tensor)
    - [`ggml_set_name`](../../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_set_output`](../../ggml/src/ggml.c.driver.md#ggml_set_output)
    - [`ggml_build_forward_expand`](../../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
- **See also**: [`clip_graph`](#clip_graph)  (Data Structure)


---
#### clip\_graph::build\_vit<!-- {{#callable:clip_graph::build_vit}} -->
The `build_vit` function constructs a Vision Transformer (ViT) computational graph by applying layer normalization, self-attention, and feed-forward network operations over multiple layers, optionally incorporating learned positional embeddings and additional positional transformations.
- **Inputs**:
    - `inp`: A pointer to a `ggml_tensor` representing the input tensor to the Vision Transformer.
    - `n_pos`: An `int64_t` representing the number of positions or tokens in the input sequence.
    - `norm_t`: A `norm_type` enum value indicating the type of normalization to apply (e.g., normal or RMS normalization).
    - `ffn_t`: An `ffn_op_type` enum value specifying the type of feed-forward network operation to apply (e.g., GELU, SILU).
    - `learned_pos_embd`: A pointer to a `ggml_tensor` representing learned positional embeddings to be added to the input, if available.
    - `add_pos`: A `std::function` that takes a `ggml_tensor` and a `clip_layer` reference, used to apply additional positional transformations to the query and key tensors.
- **Control Flow**:
    - If `learned_pos_embd` is provided, add it to the input tensor `inp` and log the operation.
    - Initialize `inpL` with the input tensor `inp`.
    - If pre-layer normalization weights are available, apply pre-layer normalization to `inpL` and log the operation.
    - Iterate over each layer in the model, performing the following operations:
    - Apply layer normalization to the current layer's input and log the operation.
    - Compute query, key, and value tensors using the current layer's weights, optionally adding biases, and log the operations.
    - If normalization weights for query or key are available, apply normalization to the respective tensors and log the operations.
    - Reshape the query, key, and value tensors to 3D and log the reshaped tensors.
    - If `add_pos` is provided, apply it to the query and key tensors and log the operations.
    - Perform self-attention using the query, key, and value tensors, and log the output.
    - If layer scaling weights are available, scale the attention output and log the operation.
    - Add the residual connection from the input to the attention output and update `inpL`.
    - Apply layer normalization to the feed-forward network input and log the operation.
    - Apply the feed-forward network operation to the normalized input and log the output.
    - If layer scaling weights are available, scale the feed-forward network output and log the operation.
    - Add the residual connection from the input to the feed-forward network output and update `inpL`.
    - If the projector type is `PROJECTOR_TYPE_QWEN2A`, apply additional transformations to `inpL`.
    - If post-layer normalization weights are available, apply post-layer normalization to `inpL`.
- **Output**: Returns a pointer to a `ggml_tensor` representing the final output of the Vision Transformer after processing through all layers.
- **Functions called**:
    - [`ggml_add`](../../ggml/src/ggml.c.driver.md#ggml_add)
    - [`clip_graph::cb`](#clip_graphcb)
    - [`clip_graph::build_norm`](#clip_graphbuild_norm)
    - [`ggml_mul_mat`](../../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_reshape_3d`](../../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`clip_graph::build_attn`](#clip_graphbuild_attn)
    - [`ggml_mul`](../../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`clip_graph::build_ffn`](#clip_graphbuild_ffn)
    - [`ggml_transpose`](../../ggml/src/ggml.c.driver.md#ggml_transpose)
    - [`ggml_cont`](../../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_pool_1d`](../../ggml/src/ggml.c.driver.md#ggml_pool_1d)
- **See also**: [`clip_graph`](#clip_graph)  (Data Structure)


---
#### clip\_graph::build\_inp<!-- {{#callable:clip_graph::build_inp}} -->
The `build_inp` function constructs a 2D reshaped and optionally bias-adjusted tensor from raw input data using convolution and reshaping operations.
- **Inputs**:
    - `None`: The function does not take any explicit input parameters.
- **Control Flow**:
    - Call `build_inp_raw()` to obtain the raw input tensor `inp_raw`.
    - Perform a 2D convolution on `inp_raw` using `model.patch_embeddings_0` to produce `inp`.
    - Reshape `inp` into a 2D tensor with dimensions `n_patches` by `n_embd`.
    - Transpose and make `inp` contiguous.
    - If `model.patch_bias` is present, add it to `inp` and call [`cb`](#clip_graphcb) with the updated `inp`.
    - Return the final `inp` tensor.
- **Output**: A `ggml_tensor` pointer representing the processed input tensor.
- **Functions called**:
    - [`clip_graph::build_inp_raw`](#clip_graphbuild_inp_raw)
    - [`ggml_conv_2d`](../../ggml/src/ggml.c.driver.md#ggml_conv_2d)
    - [`ggml_reshape_2d`](../../ggml/src/ggml.c.driver.md#ggml_reshape_2d)
    - [`ggml_cont`](../../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_transpose`](../../ggml/src/ggml.c.driver.md#ggml_transpose)
    - [`ggml_add`](../../ggml/src/ggml.c.driver.md#ggml_add)
    - [`clip_graph::cb`](#clip_graphcb)
- **See also**: [`clip_graph`](#clip_graph)  (Data Structure)


---
#### clip\_graph::build\_inp\_raw<!-- {{#callable:clip_graph::build_inp_raw}} -->
The `build_inp_raw` function creates a new 3D tensor for image input with specified dimensions and sets it as an input tensor in the context.
- **Inputs**:
    - `channels`: An integer specifying the number of channels for the tensor, defaulting to 3.
- **Control Flow**:
    - A new 3D tensor `inp_raw` is created using [`ggml_new_tensor_3d`](../../ggml/src/ggml.c.driver.md#ggml_new_tensor_3d) with dimensions based on the image's width (`img.nx`), height (`img.ny`), and the specified number of channels.
    - The tensor `inp_raw` is named "inp_raw" using [`ggml_set_name`](../../ggml/src/ggml.c.driver.md#ggml_set_name).
    - The tensor `inp_raw` is set as an input tensor using [`ggml_set_input`](../../ggml/src/ggml.c.driver.md#ggml_set_input).
    - The tensor `inp_raw` is returned.
- **Output**: A pointer to the newly created `ggml_tensor` representing the raw input image tensor.
- **Functions called**:
    - [`ggml_new_tensor_3d`](../../ggml/src/ggml.c.driver.md#ggml_new_tensor_3d)
    - [`ggml_set_name`](../../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_set_input`](../../ggml/src/ggml.c.driver.md#ggml_set_input)
- **See also**: [`clip_graph`](#clip_graph)  (Data Structure)


---
#### clip\_graph::build\_norm<!-- {{#callable:clip_graph::build_norm}} -->
The `build_norm` function applies a normalization operation to a given tensor, optionally followed by scaling and bias addition, based on the specified normalization type and parameters.
- **Inputs**:
    - `cur`: A pointer to a `ggml_tensor` representing the current tensor to be normalized.
    - `mw`: A pointer to a `ggml_tensor` representing the weights for scaling the normalized tensor, or `nullptr` if not used.
    - `mb`: A pointer to a `ggml_tensor` representing the biases to be added to the scaled tensor, or `nullptr` if not used.
    - `type`: An enum value of type `norm_type` indicating the type of normalization to apply (either `NORM_TYPE_RMS` or `NORM_TYPE_NORMAL`).
    - `norm_eps`: A float representing the epsilon value used in the normalization process to prevent division by zero.
    - `il`: An integer representing the layer index, used for debugging purposes.
- **Control Flow**:
    - Determine the normalization type based on the `type` parameter and apply either RMS normalization or standard normalization to the `cur` tensor using [`ggml_rms_norm`](../../ggml/src/ggml.c.driver.md#ggml_rms_norm) or [`ggml_norm`](../../ggml/src/ggml.c.driver.md#ggml_norm), respectively.
    - If either `mw` or `mb` is provided, call the [`cb`](#clip_graphcb) function for debugging purposes with the current tensor, the name 'norm', and the layer index `il`.
    - If `mw` is provided, multiply the normalized tensor by `mw` using [`ggml_mul`](../../ggml/src/ggml.c.driver.md#ggml_mul), and if `mb` is also provided, call the [`cb`](#clip_graphcb) function again with the name 'norm_w'.
    - If `mb` is provided, add `mb` to the scaled tensor using [`ggml_add`](../../ggml/src/ggml.c.driver.md#ggml_add).
    - Return the final processed tensor.
- **Output**: A pointer to a `ggml_tensor` representing the normalized (and optionally scaled and biased) tensor.
- **Functions called**:
    - [`ggml_rms_norm`](../../ggml/src/ggml.c.driver.md#ggml_rms_norm)
    - [`ggml_norm`](../../ggml/src/ggml.c.driver.md#ggml_norm)
    - [`clip_graph::cb`](#clip_graphcb)
    - [`ggml_mul`](../../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`ggml_add`](../../ggml/src/ggml.c.driver.md#ggml_add)
- **See also**: [`clip_graph`](#clip_graph)  (Data Structure)


---
#### clip\_graph::build\_ffn<!-- {{#callable:clip_graph::build_ffn}} -->
The `build_ffn` function constructs a feed-forward neural network (FFN) layer with optional gating and bias additions, applying a specified activation function to the input tensor.
- **Inputs**:
    - `cur`: A pointer to a `ggml_tensor` representing the current input tensor to the FFN.
    - `up`: A pointer to a `ggml_tensor` representing the 'up' weight matrix for the FFN, used for matrix multiplication with the input tensor.
    - `up_b`: A pointer to a `ggml_tensor` representing the bias to be added after the 'up' matrix multiplication, if provided.
    - `gate`: A pointer to a `ggml_tensor` representing the gating weight matrix, used for matrix multiplication with the input tensor, if provided.
    - `gate_b`: A pointer to a `ggml_tensor` representing the bias to be added after the gating matrix multiplication, if provided.
    - `down`: A pointer to a `ggml_tensor` representing the 'down' weight matrix for the FFN, used for matrix multiplication with the output of the activation function.
    - `down_b`: A pointer to a `ggml_tensor` representing the bias to be added after the 'down' matrix multiplication, if provided.
    - `type_op`: An `ffn_op_type` enum value indicating the type of activation function to apply (e.g., SILU, GELU, etc.).
    - `il`: An integer representing the layer index, used for debugging or logging purposes.
- **Control Flow**:
    - Initialize `tmp` as the result of multiplying `up` with `cur` if `up` is provided, otherwise set `tmp` to `cur`.
    - If `up_b` is provided, add it to `tmp`.
    - If `gate` is provided, multiply it with `cur` and add `gate_b` if available; otherwise, set `cur` to `tmp`.
    - Apply the specified activation function (`type_op`) to `cur`.
    - If `gate` is provided, multiply `cur` with `tmp` to support parallel FFN.
    - If `down` is provided, multiply it with `cur`.
    - If `down_b` is provided, add it to `cur`.
    - Return the modified `cur` tensor.
- **Output**: A pointer to a `ggml_tensor` representing the output of the FFN after applying the specified operations and activation function.
- **Functions called**:
    - [`ggml_mul_mat`](../../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`clip_graph::cb`](#clip_graphcb)
    - [`ggml_add`](../../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_silu`](../../ggml/src/ggml.c.driver.md#ggml_silu)
    - [`ggml_gelu`](../../ggml/src/ggml.c.driver.md#ggml_gelu)
    - [`ggml_gelu_erf`](../../ggml/src/ggml.c.driver.md#ggml_gelu_erf)
    - [`ggml_gelu_quick`](../../ggml/src/ggml.c.driver.md#ggml_gelu_quick)
    - [`ggml_mul`](../../ggml/src/ggml.c.driver.md#ggml_mul)
- **See also**: [`clip_graph`](#clip_graph)  (Data Structure)


---
#### clip\_graph::build\_attn<!-- {{#callable:clip_graph::build_attn}} -->
The `build_attn` function constructs an attention mechanism by processing query, key, and value tensors, applying a mask and scaling, and optionally applying output weights and biases.
- **Inputs**:
    - `wo`: A pointer to a ggml_tensor representing the output weights for the attention mechanism.
    - `wo_b`: A pointer to a ggml_tensor representing the output biases for the attention mechanism.
    - `q_cur`: A pointer to a ggml_tensor representing the current query tensor.
    - `k_cur`: A pointer to a ggml_tensor representing the current key tensor.
    - `v_cur`: A pointer to a ggml_tensor representing the current value tensor.
    - `kq_mask`: A pointer to a ggml_tensor representing the mask to be applied to the key-query product.
    - `kq_scale`: A float representing the scaling factor for the key-query product.
    - `il`: An integer representing the layer index, used for debugging purposes.
- **Control Flow**:
    - Expand the forward graph with the query, key, and value tensors to prevent reordering.
    - Permute the query tensor dimensions to prepare for matrix multiplication.
    - Permute the key tensor dimensions similarly.
    - Permute and then make the value tensor contiguous for matrix multiplication.
    - Compute the matrix product of the key and query tensors to get the key-query product.
    - Apply a softmax function to the key-query product, using the mask and scaling factor.
    - Compute the matrix product of the value tensor and the softmaxed key-query product to get the attention output.
    - Permute the attention output to match the desired output dimensions.
    - Make the attention output contiguous in 2D for further processing.
    - If output weights are provided, multiply them with the attention output.
    - If output biases are provided, add them to the attention output.
- **Output**: A ggml_tensor pointer representing the final attention output after applying optional weights and biases.
- **Functions called**:
    - [`ggml_build_forward_expand`](../../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
    - [`ggml_permute`](../../ggml/src/ggml.c.driver.md#ggml_permute)
    - [`ggml_cont`](../../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_mul_mat`](../../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_soft_max_ext`](../../ggml/src/ggml.c.driver.md#ggml_soft_max_ext)
    - [`ggml_cont_2d`](../../ggml/src/ggml.c.driver.md#ggml_cont_2d)
    - [`clip_graph::cb`](#clip_graphcb)
    - [`ggml_add`](../../ggml/src/ggml.c.driver.md#ggml_add)
- **See also**: [`clip_graph`](#clip_graph)  (Data Structure)


---
#### clip\_graph::build\_rope\_2d<!-- {{#callable:clip_graph::build_rope_2d}} -->
The `build_rope_2d` function applies a 2D rotary positional encoding to a given tensor by splitting it into two halves and applying different frequency scaling to each half.
- **Inputs**:
    - `ctx0`: A pointer to the ggml_context, which is used for memory management and tensor operations.
    - `cur`: A pointer to the ggml_tensor that represents the current tensor to which the 2D rotary positional encoding will be applied.
    - `pos_a`: A pointer to the ggml_tensor representing the positions for the first half of the tensor.
    - `pos_b`: A pointer to the ggml_tensor representing the positions for the second half of the tensor.
    - `freq_base`: A float representing the base frequency used for the rotary positional encoding.
    - `interleave_freq`: A boolean indicating whether to interleave frequencies for the second half of the tensor.
- **Control Flow**:
    - Calculate the number of dimensions, heads, and positions from the current tensor `cur`.
    - Determine the frequency scaling factor for the odd dimensions based on the `interleave_freq` flag.
    - Create a view of the first half of the tensor and apply the rotary positional encoding using [`ggml_rope_ext`](../../ggml/src/ggml.c.driver.md#ggml_rope_ext).
    - Create a view of the second half of the tensor, make it contiguous, and apply the rotary positional encoding with adjusted frequency scaling.
    - Concatenate the processed first and second halves back into a single tensor.
- **Output**: Returns a pointer to the ggml_tensor that has been modified with the 2D rotary positional encoding applied.
- **Functions called**:
    - [`ggml_view_3d`](../../ggml/src/ggml.c.driver.md#ggml_view_3d)
    - [`ggml_row_size`](../../ggml/src/ggml.c.driver.md#ggml_row_size)
    - [`ggml_rope_ext`](../../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_element_size`](../../ggml/src/ggml.c.driver.md#ggml_element_size)
    - [`ggml_cont`](../../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_concat`](../../ggml/src/ggml.c.driver.md#ggml_concat)
- **See also**: [`clip_graph`](#clip_graph)  (Data Structure)



---
### clip\_model\_loader<!-- {{#data_structure:clip_model_loader}} -->
- **Type**: `struct`
- **Members**:
    - `ctx_meta`: A pointer to a ggml_context structure for metadata management.
    - `ctx_gguf`: A pointer to a gguf_context structure for handling GGUF file context.
    - `fname`: A string representing the filename of the CLIP model to be loaded.
    - `model_size`: A size_t variable indicating the size of the model in bytes.
    - `has_vision`: A boolean flag indicating if the model has a vision encoder.
    - `has_audio`: A boolean flag indicating if the model has an audio encoder.
- **Description**: The `clip_model_loader` struct is responsible for loading and managing the context of a CLIP model from a file. It holds pointers to context structures for metadata and GGUF file handling, as well as information about the model's filename, size, and available modalities (vision and audio). The struct provides functionality to initialize these contexts and load model parameters, ensuring the model is correctly set up for further processing.
- **Member Functions**:
    - [`clip_model_loader::clip_model_loader`](#clip_model_loaderclip_model_loader)
    - [`clip_model_loader::load_hparams`](#clip_model_loaderload_hparams)
    - [`clip_model_loader::load_tensors`](#clip_model_loaderload_tensors)
    - [`clip_model_loader::alloc_compute_meta`](#clip_model_loaderalloc_compute_meta)
    - [`clip_model_loader::get_bool`](#clip_model_loaderget_bool)
    - [`clip_model_loader::get_i32`](#clip_model_loaderget_i32)
    - [`clip_model_loader::get_u32`](#clip_model_loaderget_u32)
    - [`clip_model_loader::get_f32`](#clip_model_loaderget_f32)
    - [`clip_model_loader::get_string`](#clip_model_loaderget_string)
    - [`clip_model_loader::get_arr_int`](#clip_model_loaderget_arr_int)

**Methods**

---
#### clip\_model\_loader::clip\_model\_loader<!-- {{#callable:clip_model_loader::clip_model_loader}} -->
The `clip_model_loader` function initializes a CLIP model loader by loading model metadata and tensors from a specified file.
- **Inputs**:
    - `fname`: A constant character pointer representing the filename of the CLIP model to be loaded.
- **Control Flow**:
    - Initialize a `ggml_context` pointer `meta` to `nullptr`.
    - Set up `gguf_init_params` with `no_alloc` as `true` and `ctx` pointing to `meta`.
    - Initialize `ctx_gguf` by calling [`gguf_init_from_file`](../../ggml/src/gguf.cpp.driver.md#gguf_init_from_file) with `fname` and `params`.
    - Check if `ctx_gguf` is valid; if not, throw a runtime error indicating failure to load the model.
    - Reset `ctx_meta` with `meta`.
    - Retrieve the number of tensors using [`gguf_get_n_tensors`](../../ggml/src/gguf.cpp.driver.md#gguf_get_n_tensors).
    - Retrieve and log model information such as name, description, version, alignment, number of tensors, and key-value pairs.
    - Check for the presence of vision and audio encoders and log their availability.
    - Iterate over each tensor, retrieve its name, offset, type, and size, and accumulate the total model size.
- **Output**: The function does not return a value but initializes the `clip_model_loader` object with the loaded model data and logs relevant information.
- **Functions called**:
    - [`gguf_init_from_file`](../../ggml/src/gguf.cpp.driver.md#gguf_init_from_file)
    - [`string_format`](clip-impl.h.driver.md#string_format)
    - [`gguf_get_n_tensors`](../../ggml/src/gguf.cpp.driver.md#gguf_get_n_tensors)
    - [`clip_model_loader::get_string`](#clip_model_loaderget_string)
    - [`gguf_get_version`](../../ggml/src/gguf.cpp.driver.md#gguf_get_version)
    - [`gguf_get_alignment`](../../ggml/src/gguf.cpp.driver.md#gguf_get_alignment)
    - [`gguf_get_n_kv`](../../ggml/src/gguf.cpp.driver.md#gguf_get_n_kv)
    - [`clip_model_loader::get_bool`](#clip_model_loaderget_bool)
    - [`gguf_get_tensor_name`](../../ggml/src/gguf.cpp.driver.md#gguf_get_tensor_name)
    - [`gguf_get_tensor_offset`](../../ggml/src/gguf.cpp.driver.md#gguf_get_tensor_offset)
    - [`gguf_get_tensor_type`](../../ggml/src/gguf.cpp.driver.md#gguf_get_tensor_type)
    - [`ggml_get_tensor`](../../ggml/src/ggml.c.driver.md#ggml_get_tensor)
    - [`ggml_nbytes`](../../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`ggml_n_dims`](../../ggml/src/ggml.c.driver.md#ggml_n_dims)
    - [`ggml_type_name`](../../ggml/src/ggml.c.driver.md#ggml_type_name)
- **See also**: [`clip_model_loader`](#clip_model_loader)  (Data Structure)


---
#### clip\_model\_loader::load\_hparams<!-- {{#callable:clip_model_loader::load_hparams}} -->
The `load_hparams` function initializes and configures the hyperparameters of a CLIP model based on the specified modality (vision or audio).
- **Inputs**:
    - `model`: A reference to a `clip_model` object whose hyperparameters are to be loaded and configured.
    - `modality`: A `clip_modality` enum value indicating whether the model is for vision or audio processing.
- **Control Flow**:
    - Check if the specified modality is supported by asserting the presence of vision or audio capabilities in the model.
    - Set the model's modality to the specified modality.
    - Retrieve and set the projector type from a configuration key, and handle unknown projector types by throwing an exception.
    - Adjust the projector type for multimodal models if necessary.
    - Determine if the modality is vision or audio and set a prefix accordingly.
    - Load various hyperparameters such as embedding size, number of heads, feedforward size, number of layers, projection dimension, and layer normalization epsilon using the prefix.
    - For vision modality, load additional parameters like image size, patch size, crop resolution, grid pinpoints, and legacy versioning.
    - For audio modality, load the number of mel bins.
    - Set default warmup image size and determine if the model has a llava projector based on the projector type.
    - Check for mutually exclusive activation functions (GELU and SiLU) and set the feedforward operation type accordingly, logging the chosen operation.
    - Load the patch merge type if specified.
    - For vision modality, load image mean and standard deviation values from a configuration context, ensuring they are found.
    - Load vision feature layer indices if provided, converting them to a set for easy access.
    - Handle model-specific parameters based on the projector type, such as setting default values or loading additional configuration keys.
    - Log the loaded hyperparameters and model-specific parameters for debugging and verification.
- **Output**: The function does not return a value; it modifies the `model` object in place by setting its hyperparameters.
- **Functions called**:
    - [`clip_model_loader::get_string`](#clip_model_loaderget_string)
    - [`clip_projector_type_from_string`](clip-impl.h.driver.md#clip_projector_type_from_string)
    - [`string_format`](clip-impl.h.driver.md#string_format)
    - [`clip_model_loader::get_u32`](#clip_model_loaderget_u32)
    - [`clip_model_loader::get_f32`](#clip_model_loaderget_f32)
    - [`clip_model_loader::get_arr_int`](#clip_model_loaderget_arr_int)
    - [`clip_model_loader::get_i32`](#clip_model_loaderget_i32)
    - [`clip_model_loader::get_bool`](#clip_model_loaderget_bool)
    - [`gguf_find_key`](../../ggml/src/gguf.cpp.driver.md#gguf_find_key)
    - [`gguf_get_arr_data`](../../ggml/src/gguf.cpp.driver.md#gguf_get_arr_data)
    - [`ggml_get_mem_size`](../../ggml/src/ggml.c.driver.md#ggml_get_mem_size)
- **See also**: [`clip_model_loader`](#clip_model_loader)  (Data Structure)


---
#### clip\_model\_loader::load\_tensors<!-- {{#callable:clip_model_loader::load_tensors}} -->
The `load_tensors` function initializes and loads tensor data into a CLIP model context from a file, setting up the necessary structures and handling different projector types.
- **Inputs**:
    - `ctx_clip`: A reference to a `clip_ctx` object, which contains the model and context data for loading tensors.
- **Control Flow**:
    - Retrieve the model and hyperparameters from the `ctx_clip` object.
    - Determine the prefix for tensor names based on the model's modality (audio or video).
    - Iterate over all tensors in the GGUF context to calculate their offsets and store them in a map.
    - Initialize a GGML data context with calculated memory size and check for successful initialization.
    - Define a helper function `get_tensor` to retrieve and duplicate tensors, adding them to a list for loading.
    - Load various model components such as class embeddings, layer weights, and biases using the `get_tensor` function.
    - Iterate over each layer in the model, loading attention and feed-forward network weights and biases.
    - Handle legacy naming issues by swapping weights and biases if necessary.
    - Switch on the model's projector type to load specific projector-related tensors.
    - Open the file specified in `fname` and read tensor data into the context, handling both host and device memory scenarios.
    - Log the number of loaded tensors and close the file.
- **Output**: The function does not return a value but modifies the `ctx_clip` object by loading and setting up the tensors within its context.
- **Functions called**:
    - [`gguf_get_n_tensors`](../../ggml/src/gguf.cpp.driver.md#gguf_get_n_tensors)
    - [`gguf_get_tensor_name`](../../ggml/src/gguf.cpp.driver.md#gguf_get_tensor_name)
    - [`gguf_get_data_offset`](../../ggml/src/gguf.cpp.driver.md#gguf_get_data_offset)
    - [`gguf_get_tensor_offset`](../../ggml/src/gguf.cpp.driver.md#gguf_get_tensor_offset)
    - [`ggml_tensor_overhead`](../../ggml/src/ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_init`](../../ggml/src/ggml.c.driver.md#ggml_init)
    - [`string_format`](clip-impl.h.driver.md#string_format)
    - [`ggml_get_tensor`](../../ggml/src/ggml.c.driver.md#ggml_get_tensor)
    - [`ggml_dup_tensor`](../../ggml/src/ggml.c.driver.md#ggml_dup_tensor)
    - [`ggml_set_name`](../../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_backend_alloc_ctx_tensors_from_buft`](../../ggml/src/ggml-alloc.c.driver.md#ggml_backend_alloc_ctx_tensors_from_buft)
    - [`ggml_backend_buffer_set_usage`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_set_usage)
    - [`ggml_nbytes`](../../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`ggml_backend_buft_is_host`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buft_is_host)
    - [`ggml_backend_tensor_set`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
- **See also**: [`clip_model_loader`](#clip_model_loader)  (Data Structure)


---
#### clip\_model\_loader::alloc\_compute\_meta<!-- {{#callable:clip_model_loader::alloc_compute_meta}} -->
The `alloc_compute_meta` function allocates and initializes the compute metadata buffer for a CLIP context based on the model's parameters and backend configurations.
- **Inputs**:
    - `ctx_clip`: A reference to a `clip_ctx` object, which contains the model parameters, backend configurations, and other context-specific data for CLIP processing.
- **Control Flow**:
    - Retrieve the hyperparameters from the model within the `ctx_clip` context.
    - Resize the `buf_compute_meta` buffer in `ctx_clip` to accommodate the maximum number of nodes and overheads for tensors and graphs.
    - Initialize a fake batch with a single image or audio entry, depending on the modality of the model.
    - If the model's modality is vision, set the image dimensions to the warmup image size; otherwise, set them to the warmup audio size and number of mel bins.
    - Build a computation graph using the [`clip_image_build_graph`](#clip_image_build_graph) function with the fake batch.
    - Reserve backend scheduling resources for the computation graph using [`ggml_backend_sched_reserve`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_reserve).
    - Iterate over each backend pointer in `ctx_clip` to log the compute buffer size if it exceeds 1 MiB.
- **Output**: The function does not return a value; it modifies the `ctx_clip` object in place.
- **Functions called**:
    - [`ggml_tensor_overhead`](../../ggml/src/ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_graph_overhead`](../../ggml/src/ggml.c.driver.md#ggml_graph_overhead)
    - [`clip_image_build_graph`](#clip_image_build_graph)
    - [`ggml_backend_sched_reserve`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_reserve)
    - [`ggml_backend_sched_get_buffer_size`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_get_buffer_size)
    - [`ggml_backend_buft_name`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buft_name)
- **See also**: [`clip_model_loader`](#clip_model_loader)  (Data Structure)


---
#### clip\_model\_loader::get\_bool<!-- {{#callable:clip_model_loader::get_bool}} -->
The `get_bool` function retrieves a boolean value associated with a given key from a context and assigns it to an output variable, throwing an error if the key is not found and the value is required.
- **Inputs**:
    - `key`: A constant reference to a string representing the key to search for in the context.
    - `output`: A reference to a boolean variable where the retrieved value will be stored.
    - `required`: A boolean flag indicating whether the key is required; defaults to true.
- **Control Flow**:
    - Call [`gguf_find_key`](../../ggml/src/gguf.cpp.driver.md#gguf_find_key) to find the index of the key in the context.
    - Check if the index is negative, indicating the key was not found.
    - If the key is not found and `required` is true, throw a runtime error with a message indicating the key was not found.
    - If the key is found, call [`gguf_get_val_bool`](../../ggml/src/gguf.cpp.driver.md#gguf_get_val_bool) to retrieve the boolean value associated with the key and assign it to `output`.
- **Output**: The function does not return a value but modifies the `output` boolean reference with the retrieved value.
- **Functions called**:
    - [`gguf_find_key`](../../ggml/src/gguf.cpp.driver.md#gguf_find_key)
    - [`gguf_get_val_bool`](../../ggml/src/gguf.cpp.driver.md#gguf_get_val_bool)
- **See also**: [`clip_model_loader`](#clip_model_loader)  (Data Structure)


---
#### clip\_model\_loader::get\_i32<!-- {{#callable:clip_model_loader::get_i32}} -->
The `get_i32` function retrieves a 32-bit integer value associated with a given key from a context and assigns it to an output variable, throwing an error if the key is not found and the value is required.
- **Inputs**:
    - `key`: A constant reference to a `std::string` representing the key to search for in the context.
    - `output`: A reference to an integer where the retrieved 32-bit integer value will be stored.
    - `required`: A boolean flag indicating whether the key is required; defaults to `true`.
- **Control Flow**:
    - The function first attempts to find the index of the key in the context using [`gguf_find_key`](../../ggml/src/gguf.cpp.driver.md#gguf_find_key).
    - If the index is negative (key not found) and `required` is true, it throws a `std::runtime_error` with a message indicating the key was not found.
    - If the key is not required and not found, the function simply returns without modifying the output.
    - If the key is found, it retrieves the 32-bit integer value using [`gguf_get_val_i32`](../../ggml/src/gguf.cpp.driver.md#gguf_get_val_i32) and assigns it to the `output` variable.
- **Output**: The function does not return a value but modifies the `output` integer reference to store the retrieved 32-bit integer value.
- **Functions called**:
    - [`gguf_find_key`](../../ggml/src/gguf.cpp.driver.md#gguf_find_key)
    - [`gguf_get_val_i32`](../../ggml/src/gguf.cpp.driver.md#gguf_get_val_i32)
- **See also**: [`clip_model_loader`](#clip_model_loader)  (Data Structure)


---
#### clip\_model\_loader::get\_u32<!-- {{#callable:clip_model_loader::get_u32}} -->
The `get_u32` function retrieves a 32-bit unsigned integer value associated with a given key from a context and assigns it to an output variable, throwing an error if the key is not found and the value is required.
- **Inputs**:
    - `key`: A constant reference to a string representing the key to search for in the context.
    - `output`: A reference to an integer where the retrieved 32-bit unsigned integer value will be stored.
    - `required`: A boolean flag indicating whether the key is required; defaults to true.
- **Control Flow**:
    - The function begins by finding the index of the key in the context using [`gguf_find_key`](../../ggml/src/gguf.cpp.driver.md#gguf_find_key).
    - If the index is negative (key not found) and the key is required, it throws a runtime error with a message indicating the key was not found.
    - If the key is not required and not found, the function returns without doing anything.
    - If the key is found, it retrieves the 32-bit unsigned integer value using [`gguf_get_val_u32`](../../ggml/src/gguf.cpp.driver.md#gguf_get_val_u32) and assigns it to the output variable.
- **Output**: The function does not return a value but assigns the retrieved 32-bit unsigned integer to the provided output reference.
- **Functions called**:
    - [`gguf_find_key`](../../ggml/src/gguf.cpp.driver.md#gguf_find_key)
    - [`gguf_get_val_u32`](../../ggml/src/gguf.cpp.driver.md#gguf_get_val_u32)
- **See also**: [`clip_model_loader`](#clip_model_loader)  (Data Structure)


---
#### clip\_model\_loader::get\_f32<!-- {{#callable:clip_model_loader::get_f32}} -->
The `get_f32` function retrieves a float value associated with a given key from a context and assigns it to an output variable, throwing an error if the key is not found and the value is required.
- **Inputs**:
    - `key`: A constant reference to a string representing the key to search for in the context.
    - `output`: A reference to a float variable where the retrieved value will be stored.
    - `required`: A boolean flag indicating whether the key is required; defaults to true.
- **Control Flow**:
    - Call [`gguf_find_key`](../../ggml/src/gguf.cpp.driver.md#gguf_find_key) to find the index of the key in the context.
    - Check if the index is negative, indicating the key was not found.
    - If the key is not found and `required` is true, throw a runtime error with a message indicating the key was not found.
    - If the key is found, call [`gguf_get_val_f32`](../../ggml/src/gguf.cpp.driver.md#gguf_get_val_f32) to retrieve the float value associated with the key and assign it to `output`.
- **Output**: The function does not return a value but assigns the retrieved float value to the `output` parameter.
- **Functions called**:
    - [`gguf_find_key`](../../ggml/src/gguf.cpp.driver.md#gguf_find_key)
    - [`gguf_get_val_f32`](../../ggml/src/gguf.cpp.driver.md#gguf_get_val_f32)
- **See also**: [`clip_model_loader`](#clip_model_loader)  (Data Structure)


---
#### clip\_model\_loader::get\_string<!-- {{#callable:clip_model_loader::get_string}} -->
The `get_string` function retrieves a string value associated with a given key from a context and assigns it to an output variable, throwing an error if the key is not found and the value is required.
- **Inputs**:
    - `key`: A constant reference to a string representing the key to search for in the context.
    - `output`: A reference to a string where the retrieved value will be stored.
    - `required`: A boolean flag indicating whether the key is required; defaults to true.
- **Control Flow**:
    - Finds the index of the key in the context using [`gguf_find_key`](../../ggml/src/gguf.cpp.driver.md#gguf_find_key).
    - Checks if the index is negative, indicating the key was not found.
    - If the key is not found and `required` is true, throws a runtime error.
    - If the key is found, retrieves the string value using [`gguf_get_val_str`](../../ggml/src/gguf.cpp.driver.md#gguf_get_val_str) and assigns it to `output`.
- **Output**: The function does not return a value; it modifies the `output` string by reference.
- **Functions called**:
    - [`gguf_find_key`](../../ggml/src/gguf.cpp.driver.md#gguf_find_key)
    - [`gguf_get_val_str`](../../ggml/src/gguf.cpp.driver.md#gguf_get_val_str)
- **See also**: [`clip_model_loader`](#clip_model_loader)  (Data Structure)


---
#### clip\_model\_loader::get\_arr\_int<!-- {{#callable:clip_model_loader::get_arr_int}} -->
The `get_arr_int` function retrieves an array of integers associated with a given key from a context and stores it in the provided output vector.
- **Inputs**:
    - `key`: A constant reference to a string representing the key to search for in the context.
    - `output`: A reference to a vector of integers where the retrieved array will be stored.
    - `required`: A boolean flag indicating whether the key is required; defaults to true.
- **Control Flow**:
    - Find the index of the key in the context using [`gguf_find_key`](../../ggml/src/gguf.cpp.driver.md#gguf_find_key).
    - If the key is not found and `required` is true, throw a runtime error; otherwise, return.
    - Retrieve the number of elements in the array associated with the key using [`gguf_get_arr_n`](../../ggml/src/gguf.cpp.driver.md#gguf_get_arr_n).
    - Resize the output vector to match the number of elements in the array.
    - Retrieve the array data using [`gguf_get_arr_data`](../../ggml/src/gguf.cpp.driver.md#gguf_get_arr_data) and cast it to a pointer to `int32_t`.
    - Copy the array data into the output vector.
- **Output**: The function does not return a value; it modifies the `output` vector in place.
- **Functions called**:
    - [`gguf_find_key`](../../ggml/src/gguf.cpp.driver.md#gguf_find_key)
    - [`gguf_get_arr_n`](../../ggml/src/gguf.cpp.driver.md#gguf_get_arr_n)
    - [`gguf_get_arr_data`](../../ggml/src/gguf.cpp.driver.md#gguf_get_arr_data)
- **See also**: [`clip_model_loader`](#clip_model_loader)  (Data Structure)



---
### image\_manipulation<!-- {{#data_structure:image_manipulation}} -->
- **Type**: `struct`
- **Description**: The `image_manipulation` struct is a utility structure designed to provide a set of static functions for manipulating images, specifically for resizing and cropping operations. It includes methods for bilinear and bicubic resizing, resizing and padding images to maintain aspect ratios, and cropping images. The struct also contains private helper functions for clipping values and performing linear interpolation, which are used in the resizing processes. This struct is particularly useful in image processing tasks where resizing and aspect ratio preservation are required.
- **Member Functions**:
    - [`image_manipulation::bilinear_resize`](#image_manipulationbilinear_resize)
    - [`image_manipulation::bicubic_resize`](#image_manipulationbicubic_resize)
    - [`image_manipulation::resize_and_pad_image`](#image_manipulationresize_and_pad_image)
    - [`image_manipulation::crop_image`](#image_manipulationcrop_image)
    - [`image_manipulation::calc_size_preserved_ratio`](#image_manipulationcalc_size_preserved_ratio)
    - [`image_manipulation::clip`](#image_manipulationclip)
    - [`image_manipulation::lerp`](#image_manipulationlerp)

**Methods**

---
#### image\_manipulation::bilinear\_resize<!-- {{#callable:image_manipulation::bilinear_resize}} -->
The `bilinear_resize` function resizes an image using bilinear interpolation to a specified target width and height.
- **Inputs**:
    - `src`: A `clip_image_u8` object representing the source image to be resized.
    - `dst`: A `clip_image_u8` object where the resized image will be stored.
    - `target_width`: An integer specifying the desired width of the resized image.
    - `target_height`: An integer specifying the desired height of the resized image.
- **Control Flow**:
    - Set the dimensions of the destination image (`dst`) to the target width and height.
    - Resize the buffer of the destination image to accommodate the new size.
    - Calculate the horizontal (`x_ratio`) and vertical (`y_ratio`) scaling factors based on the source and target dimensions.
    - Iterate over each pixel position in the target image dimensions.
    - For each pixel, calculate the corresponding position in the source image using the scaling factors.
    - Determine the integer floor values and fractional parts for the source image coordinates.
    - Perform bilinear interpolation for each color channel (R, G, B) using the fractional parts to compute the weighted average of the four surrounding pixels in the source image.
    - Store the interpolated color value in the destination image buffer.
- **Output**: The function does not return a value; it modifies the `dst` object to contain the resized image.
- **Functions called**:
    - [`image_manipulation::lerp`](#image_manipulationlerp)
- **See also**: [`image_manipulation`](#image_manipulation)  (Data Structure)


---
#### image\_manipulation::bicubic\_resize<!-- {{#callable:image_manipulation::bicubic_resize}} -->
The `bicubic_resize` function resizes an image using bicubic interpolation to a specified target width and height.
- **Inputs**:
    - `img`: A `clip_image_u8` object representing the source image to be resized.
    - `dst`: A `clip_image_u8` object where the resized image will be stored.
    - `target_width`: An integer specifying the desired width of the resized image.
    - `target_height`: An integer specifying the desired height of the resized image.
- **Control Flow**:
    - Initialize the dimensions of the source image (`nx` and `ny`) and set the dimensions of the destination image (`dst`) to the target width and height.
    - Resize the buffer of the destination image to accommodate the new size.
    - Calculate the scaling factors `tx` and `ty` for the x and y dimensions based on the ratio of the source and target dimensions.
    - Iterate over each pixel in the target image dimensions, calculating the corresponding source image coordinates using the scaling factors.
    - For each color channel (R, G, B), perform bicubic interpolation by calculating the differences `d0`, `d2`, `d3` and coefficients `a0`, `a1`, `a2`, `a3` for the interpolation formula.
    - Compute the interpolated color value `Cc` for each pixel using the calculated coefficients and differences, and clamp the result to the range [0, 255].
    - Assign the computed color value to the corresponding pixel in the destination image buffer.
    - Return `true` to indicate successful resizing.
- **Output**: A boolean value `true` indicating that the resizing operation was successful.
- **Functions called**:
    - [`image_manipulation::clip`](#image_manipulationclip)
- **See also**: [`image_manipulation`](#image_manipulation)  (Data Structure)


---
#### image\_manipulation::resize\_and\_pad\_image<!-- {{#callable:image_manipulation::resize_and_pad_image}} -->
The `resize_and_pad_image` function resizes an input image to fit within a target resolution while maintaining its aspect ratio, and pads it with a specified color if necessary.
- **Inputs**:
    - `image`: A `clip_image_u8` object representing the source image to be resized and padded.
    - `dst`: A reference to a `clip_image_u8` object where the resized and padded image will be stored.
    - `target_resolution`: A `clip_image_size` object specifying the desired width and height for the output image.
    - `pad_color`: An optional `std::array<uint8_t, 3>` specifying the RGB color to use for padding, defaulting to black (0, 0, 0).
- **Control Flow**:
    - Calculate the scaling factors for width and height based on the target resolution and the original image dimensions.
    - Determine the new dimensions for the resized image by comparing the scaling factors and ensuring the aspect ratio is preserved.
    - Resize the original image to the new dimensions using bicubic interpolation.
    - Initialize a new image buffer for the padded image with the target resolution and fill it with the padding color.
    - Calculate the offsets needed to center the resized image within the padded image.
    - Copy the resized image into the center of the padded image buffer.
    - Move the padded image into the destination image reference.
- **Output**: The function outputs a resized and padded image stored in the `dst` parameter, which is a `clip_image_u8` object.
- **Functions called**:
    - [`image_manipulation::bicubic_resize`](#image_manipulationbicubic_resize)
- **See also**: [`image_manipulation`](#image_manipulation)  (Data Structure)


---
#### image\_manipulation::crop\_image<!-- {{#callable:image_manipulation::crop_image}} -->
The `crop_image` function extracts a specified rectangular region from a source image and stores it in a destination image buffer.
- **Inputs**:
    - `image`: A constant reference to a `clip_image_u8` object representing the source image from which a region will be cropped.
    - `dst`: A reference to a `clip_image_u8` object where the cropped image region will be stored.
    - `x`: An integer specifying the x-coordinate of the top-left corner of the cropping rectangle.
    - `y`: An integer specifying the y-coordinate of the top-left corner of the cropping rectangle.
    - `w`: An integer specifying the width of the cropping rectangle.
    - `h`: An integer specifying the height of the cropping rectangle.
- **Control Flow**:
    - Set the width (`nx`) and height (`ny`) of the destination image to the specified width (`w`) and height (`h`).
    - Resize the destination image buffer to accommodate the cropped region, which is `3 * w * h` bytes for RGB data.
    - Iterate over each row (`i`) from 0 to `h-1` in the cropping rectangle.
    - For each row, iterate over each column (`j`) from 0 to `w-1`.
    - Calculate the source index (`src_idx`) in the source image buffer for the current pixel using the formula `3 * ((y + i) * image.nx + (x + j))`.
    - Calculate the destination index (`dst_idx`) in the destination image buffer for the current pixel using the formula `3 * (i * w + j)`.
    - Copy the RGB values from the source image buffer at `src_idx` to the destination image buffer at `dst_idx`.
- **Output**: The function does not return a value; it modifies the `dst` image in place to contain the cropped region.
- **See also**: [`image_manipulation`](#image_manipulation)  (Data Structure)


---
#### image\_manipulation::calc\_size\_preserved\_ratio<!-- {{#callable:image_manipulation::calc_size_preserved_ratio}} -->
The `calc_size_preserved_ratio` function calculates the size of a resized image while preserving the aspect ratio, aligning the dimensions to the nearest multiple of a specified alignment size, and ensuring neither dimension exceeds a maximum value.
- **Inputs**:
    - `inp_size`: A `clip_image_size` structure representing the original width and height of the image.
    - `align_size`: An integer specifying the alignment size to which the dimensions should be aligned.
    - `max_dimension`: An integer specifying the maximum allowable dimension for either width or height.
- **Control Flow**:
    - Check if any of the input dimensions or parameters are non-positive; if so, return a size of {0, 0}.
    - Calculate the scaling factor to ensure neither dimension exceeds the `max_dimension`, while preserving the aspect ratio.
    - Compute the target width and height by applying the scaling factor to the original dimensions.
    - Align the target dimensions to the nearest multiple of `align_size` using the `CLIP_ALIGN` macro.
    - Return the aligned dimensions as a `clip_image_size` structure.
- **Output**: A `clip_image_size` structure containing the aligned width and height of the resized image.
- **See also**: [`image_manipulation`](#image_manipulation)  (Data Structure)


---
#### image\_manipulation::clip<!-- {{#callable:image_manipulation::clip}} -->
The `clip` function constrains an integer value to lie within a specified range defined by a lower and upper bound.
- **Inputs**:
    - `x`: The integer value to be constrained.
    - `lower`: The lower bound of the range.
    - `upper`: The upper bound of the range.
- **Control Flow**:
    - The function first calculates the minimum of `x` and `upper` using `std::min` to ensure `x` does not exceed the upper bound.
    - Then, it calculates the maximum of the result and `lower` using `std::max` to ensure the result is not below the lower bound.
    - The final result is returned, which is the constrained value of `x` within the specified range.
- **Output**: An integer value that is constrained to be within the range [lower, upper].
- **See also**: [`image_manipulation`](#image_manipulation)  (Data Structure)


---
#### image\_manipulation::lerp<!-- {{#callable:image_manipulation::lerp}} -->
The `lerp` function performs linear interpolation between two float values based on a given interpolation factor.
- **Inputs**:
    - `s`: The starting float value for interpolation.
    - `e`: The ending float value for interpolation.
    - `t`: The interpolation factor, a float value typically between 0 and 1, where 0 returns the start value and 1 returns the end value.
- **Control Flow**:
    - Calculate the difference between the end value `e` and the start value `s`.
    - Multiply the difference by the interpolation factor `t`.
    - Add the result to the start value `s` to get the interpolated value.
- **Output**: A float value representing the interpolated result between `s` and `e` based on the factor `t`.
- **See also**: [`image_manipulation`](#image_manipulation)  (Data Structure)



---
### llava\_uhd<!-- {{#data_structure:llava_uhd}} -->
- **Type**: `struct`
- **Members**:
    - `slice_coordinates`: A nested struct representing the coordinates and size of a slice.
    - `slice_instructions`: A nested struct containing instructions for slicing, including sizes and grid information.
- **Description**: The `llava_uhd` struct is designed to handle image slicing operations for the LLaVA-UHD model. It contains nested structs for defining slice coordinates and instructions, which include details about the overview size, refined size, grid size, and whether padding is required. The struct provides static methods to determine the maximum number of slices, generate slice instructions based on the original image size, and perform the actual slicing of images into smaller segments. This is particularly useful for processing large images by breaking them down into manageable pieces for further analysis or processing.
- **Member Functions**:
    - [`llava_uhd::get_max_slices`](#llava_uhdget_max_slices)
    - [`llava_uhd::get_slice_instructions`](#llava_uhdget_slice_instructions)
    - [`llava_uhd::slice_image`](#llava_uhdslice_image)
    - [`llava_uhd::get_best_resize`](#llava_uhdget_best_resize)
    - [`llava_uhd::select_best_resolution`](#llava_uhdselect_best_resolution)
    - [`llava_uhd::select_best_resolution`](#llava_uhdselect_best_resolution)
    - [`llava_uhd::ensure_divide`](#llava_uhdensure_divide)
    - [`llava_uhd::get_refine_size`](#llava_uhdget_refine_size)
    - [`llava_uhd::get_best_grid`](#llava_uhdget_best_grid)

**Methods**

---
#### llava\_uhd::get\_max\_slices<!-- {{#callable:llava_uhd::get_max_slices}} -->
The `get_max_slices` function determines the maximum number of slices that can be generated for a given context, returning 9 if the context is of type 'minicpmv', otherwise returning 0.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure, which contains the context information for determining the maximum number of slices.
- **Control Flow**:
    - Check if the context `ctx` is of type 'minicpmv' using the [`clip_is_minicpmv`](#clip_is_minicpmv) function.
    - If the context is 'minicpmv', return 9.
    - If the context is not 'minicpmv', return 0.
- **Output**: An integer representing the maximum number of slices, either 9 or 0, depending on the context type.
- **Functions called**:
    - [`clip_is_minicpmv`](#clip_is_minicpmv)
- **See also**: [`llava_uhd`](#llava_uhd)  (Data Structure)


---
#### llava\_uhd::get\_slice\_instructions<!-- {{#callable:llava_uhd::get_slice_instructions}} -->
The `get_slice_instructions` function calculates the slicing instructions for an image based on its original size and context parameters, determining how the image should be resized and divided into slices for processing.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains context information and parameters for image processing.
    - `original_size`: A `clip_image_size` structure representing the original dimensions (width and height) of the image to be processed.
- **Control Flow**:
    - Initialize a `slice_instructions` object to store the result.
    - Retrieve patch size, slice size, and maximum number of slices from the context.
    - Calculate the aspect ratio and area ratio of the original image to the slice size.
    - Determine if the image should be sliced based on the calculated ratios and maximum slice numbers.
    - Check if the context has predefined pinpoints for grid calculation.
    - If pinpoints are present, calculate the refined size using the best resolution from pinpoints and set the overview size to the slice size.
    - Iterate over the refined size to create slice coordinates and update the grid size.
    - If no pinpoints are present, calculate the best resize dimensions for the overview size.
    - If slicing is needed, calculate the best grid size and refined size, then iterate to create slice coordinates.
    - Return the populated `slice_instructions` object.
- **Output**: A `slice_instructions` object containing the overview size, refined size, grid size, and a list of slice coordinates for the image.
- **Functions called**:
    - [`clip_get_patch_size`](#clip_get_patch_size)
    - [`clip_get_image_size`](#clip_get_image_size)
    - [`llava_uhd::get_max_slices`](#llava_uhdget_max_slices)
    - [`llava_uhd::get_best_resize`](#llava_uhdget_best_resize)
    - [`llava_uhd::get_best_grid`](#llava_uhdget_best_grid)
    - [`llava_uhd::get_refine_size`](#llava_uhdget_refine_size)
- **See also**: [`llava_uhd`](#llava_uhd)  (Data Structure)


---
#### llava\_uhd::slice\_image<!-- {{#callable:llava_uhd::slice_image}} -->
The `slice_image` function slices an image into smaller parts based on given instructions, including resizing and padding options.
- **Inputs**:
    - `img`: A pointer to a `clip_image_u8` object representing the image to be sliced.
    - `inst`: A `slice_instructions` object containing the parameters for slicing, including overview size, refined size, grid size, and whether to pad the refined image.
- **Control Flow**:
    - Initialize an empty vector `output` to store the resulting image slices.
    - Create a resized image to the overview size using bicubic resizing and add it to `output`.
    - Check if there are any slices specified in `inst.slices`; if not, return the `output` containing only the resized image.
    - Create a refined image by resizing the original image to the refined size, using padding if specified in `inst.padding_refined`.
    - Iterate over each slice in `inst.slices`, cropping the refined image according to the slice's coordinates and dimensions, and add each cropped slice to `output`.
    - Return the `output` vector containing the overview image and any additional slices.
- **Output**: A vector of `clip_image_u8_ptr` objects, each representing a part of the original image, including the overview and any specified slices.
- **See also**: [`llava_uhd`](#llava_uhd)  (Data Structure)


---
#### llava\_uhd::get\_best\_resize<!-- {{#callable:llava_uhd::get_best_resize}} -->
The `get_best_resize` function calculates the optimal resized dimensions for an image while maintaining its aspect ratio and ensuring the dimensions are divisible by a specified patch size.
- **Inputs**:
    - `original_size`: A `clip_image_size` structure representing the original dimensions of the image.
    - `scale_resolution`: An integer representing the maximum resolution to scale the image to.
    - `patch_size`: An integer representing the size of the patch that the dimensions should be divisible by.
    - `allow_upscale`: A boolean flag indicating whether upscaling is allowed if the original dimensions are smaller than the scale resolution.
- **Control Flow**:
    - Extract the width and height from the `original_size` parameter.
    - Check if the product of width and height exceeds the square of `scale_resolution` or if `allow_upscale` is true.
    - If the condition is met, calculate the aspect ratio `r` and adjust the height and width to fit within the `scale_resolution` while maintaining the aspect ratio.
    - Create a `clip_image_size` structure `res` to store the new dimensions.
    - Use the [`ensure_divide`](#llava_uhdensure_divide) function to adjust the width and height to be divisible by `patch_size`.
    - Return the `res` structure with the adjusted dimensions.
- **Output**: A `clip_image_size` structure containing the adjusted width and height that are optimal for resizing the image.
- **Functions called**:
    - [`llava_uhd::ensure_divide`](#llava_uhdensure_divide)
- **See also**: [`llava_uhd`](#llava_uhd)  (Data Structure)


---
#### llava\_uhd::select\_best\_resolution<!-- {{#callable:llava_uhd::select_best_resolution}} -->
The `select_best_resolution` function selects the best resolution from a list of possible resolutions based on the original image size, maximizing effective resolution while minimizing wasted resolution.
- **Inputs**:
    - `original_size`: The original size of the image, represented as a `clip_image_size` object with width and height attributes.
    - `possible_resolutions`: A vector of `clip_image_size` objects representing the list of possible resolutions to choose from.
- **Control Flow**:
    - Initialize variables for original width and height, best fit resolution, maximum effective resolution, and minimum wasted resolution.
    - Iterate over each resolution in the possible resolutions list.
    - For each resolution, calculate the scale factor based on the original size and the current resolution.
    - Compute the downscaled width and height using the scale factor.
    - Calculate the effective resolution as the minimum of the downscaled area and the original area.
    - Determine the wasted resolution as the difference between the current resolution's area and the effective resolution.
    - If the effective resolution is greater than the current maximum or if it is equal but the wasted resolution is less than the current minimum, update the best fit resolution, maximum effective resolution, and minimum wasted resolution.
    - Return the best fit resolution.
- **Output**: The function returns a `clip_image_size` object representing the best fit resolution from the list of possible resolutions.
- **See also**: [`llava_uhd`](#llava_uhd)  (Data Structure)


---
#### llava\_uhd::select\_best\_resolution<!-- {{#callable:llava_uhd::select_best_resolution}} -->
The [`select_best_resolution`](#llava_uhdselect_best_resolution) function selects the best resolution from a list of possible resolutions based on the original image size and a list of pinpoints.
- **Inputs**:
    - `pinpoints`: A constant reference to a vector of int32_t representing pairs of width and height values.
    - `original_size`: A constant reference to a clip_image_size structure representing the original size of the image.
- **Control Flow**:
    - Initialize an empty vector of clip_image_size to store possible resolutions.
    - Iterate over the pinpoints vector in steps of 2 to construct clip_image_size objects and add them to the possible_resolutions vector.
    - Call the overloaded select_best_resolution function with original_size and possible_resolutions as arguments.
    - Return the result of the select_best_resolution function call.
- **Output**: Returns a clip_image_size object representing the best fit resolution from the possible resolutions.
- **Functions called**:
    - [`llava_uhd::select_best_resolution`](#llava_uhdselect_best_resolution)
- **See also**: [`llava_uhd`](#llava_uhd)  (Data Structure)


---
#### llava\_uhd::ensure\_divide<!-- {{#callable:llava_uhd::ensure_divide}} -->
The `ensure_divide` function rounds a given length to the nearest multiple of a specified patch size, ensuring it is at least the patch size.
- **Inputs**:
    - `length`: An integer representing the length to be adjusted.
    - `patch_size`: An integer representing the patch size to which the length should be rounded.
- **Control Flow**:
    - Convert the length to a float and divide it by the patch size.
    - Round the result to the nearest integer and multiply it back by the patch size to get the nearest multiple.
    - Use std::max to ensure the result is at least the patch size.
- **Output**: Returns an integer that is the nearest multiple of patch_size to the given length, but not less than patch_size.
- **See also**: [`llava_uhd`](#llava_uhd)  (Data Structure)


---
#### llava\_uhd::get\_refine\_size<!-- {{#callable:llava_uhd::get_refine_size}} -->
The `get_refine_size` function calculates the refined size of an image based on its original size, a grid size, scale resolution, patch size, and an optional upscale allowance.
- **Inputs**:
    - `original_size`: A `clip_image_size` structure representing the original dimensions of the image.
    - `grid`: A `clip_image_size` structure representing the grid dimensions to which the image should conform.
    - `scale_resolution`: An integer representing the target resolution for scaling.
    - `patch_size`: An integer representing the size of the patches to which the image should be aligned.
    - `allow_upscale`: A boolean flag indicating whether upscaling is permitted (default is false).
- **Control Flow**:
    - Extract the width and height from `original_size` and `grid`.
    - Calculate `refine_width` and `refine_height` by ensuring the original dimensions are divisible by the grid dimensions using [`ensure_divide`](#llava_uhdensure_divide).
    - Determine the `grid_size` by dividing `refine_width` and `refine_height` by the grid dimensions.
    - Call [`get_best_resize`](#llava_uhdget_best_resize) to find the best grid size based on the calculated `grid_size`, `scale_resolution`, `patch_size`, and `allow_upscale`.
    - Calculate the final `refine_size` by multiplying the best grid dimensions by the grid dimensions.
    - Return the `refine_size`.
- **Output**: A `clip_image_size` structure representing the refined dimensions of the image.
- **Functions called**:
    - [`llava_uhd::ensure_divide`](#llava_uhdensure_divide)
    - [`llava_uhd::get_best_resize`](#llava_uhdget_best_resize)
- **See also**: [`llava_uhd`](#llava_uhd)  (Data Structure)


---
#### llava\_uhd::get\_best\_grid<!-- {{#callable:llava_uhd::get_best_grid}} -->
The `get_best_grid` function determines the optimal grid size for slicing an image based on constraints and a target aspect ratio.
- **Inputs**:
    - `max_slice_nums`: The maximum number of slices allowed for the grid.
    - `multiple`: A base number used to generate candidate grid sizes, typically related to the desired number of slices.
    - `log_ratio`: The logarithm of the desired aspect ratio (width/height) for the grid.
- **Control Flow**:
    - Initialize an empty vector `candidate_split_grids_nums` to store potential grid sizes.
    - Iterate over the values `multiple - 1`, `multiple`, and `multiple + 1` to generate candidate grid sizes.
    - For each candidate, check if it is greater than 1 and less than or equal to `max_slice_nums`; if so, add it to `candidate_split_grids_nums`.
    - Initialize an empty vector `candidate_grids` to store potential grid dimensions.
    - For each candidate grid size in `candidate_split_grids_nums`, find all factor pairs (m, n) such that m * n equals the candidate size, and add these pairs as `clip_image_size` objects to `candidate_grids`.
    - Initialize `best_grid` as a `clip_image_size` with dimensions (1, 1) and `min_error` as infinity.
    - For each grid in `candidate_grids`, calculate the error as the absolute difference between `log_ratio` and the logarithm of the grid's aspect ratio.
    - If the calculated error is less than `min_error`, update `best_grid` to the current grid and `min_error` to the current error.
    - Return `best_grid` as the optimal grid size.
- **Output**: A `clip_image_size` object representing the best grid dimensions (width and height) for slicing the image.
- **See also**: [`llava_uhd`](#llava_uhd)  (Data Structure)



---
### slice\_coordinates<!-- {{#data_structure:llava_uhd::slice_coordinates}} -->
- **Type**: `struct`
- **Members**:
    - `x`: An integer representing the x-coordinate of the slice.
    - `y`: An integer representing the y-coordinate of the slice.
    - `size`: A `clip_image_size` structure representing the size of the slice.
- **Description**: The `slice_coordinates` struct is used to define the position and size of a slice within an image. It contains two integer fields, `x` and `y`, which specify the coordinates of the slice's top-left corner, and a `clip_image_size` field named `size` that specifies the dimensions of the slice. This struct is typically used in image processing tasks where an image is divided into smaller sections or slices for further analysis or processing.


---
### slice\_instructions<!-- {{#data_structure:llava_uhd::slice_instructions}} -->
- **Type**: `struct`
- **Members**:
    - `overview_size`: Represents the size of the downscaled image.
    - `refined_size`: Represents the size of the image right before slicing, which must be a multiple of the slice size.
    - `grid_size`: Represents the grid dimensions where grid_size.width * grid_size.height equals the number of slices.
    - `slices`: A vector containing the coordinates and sizes of each slice.
    - `padding_refined`: Indicates if the refined image will be padded to the grid size.
- **Description**: The `slice_instructions` struct is designed to manage the slicing of images for processing, particularly in the context of LLaVA-UHD. It contains information about the size of the downscaled image (`overview_size`), the size of the image before slicing (`refined_size`), and the grid dimensions (`grid_size`) that determine how the image is divided into slices. The `slices` vector holds the coordinates and sizes of each individual slice, while the `padding_refined` boolean indicates whether the refined image should be padded to fit the grid size. This struct is crucial for handling images that need to be processed in smaller, manageable parts, especially when dealing with high-resolution images.


# Functions

---
### clip\_image\_write\_image\_to\_ppm<!-- {{#callable:clip_image_write_image_to_ppm}} -->
The `clip_image_write_image_to_ppm` function writes a `clip_image_u8` image to a PPM file format.
- **Inputs**:
    - `img`: A constant reference to a `clip_image_u8` structure containing the image data to be written.
    - `filename`: A string representing the name of the file to which the image will be written.
- **Control Flow**:
    - Attempts to open the specified file in binary mode for writing.
    - If the file fails to open, logs an error message and exits the function.
    - Writes the PPM header, including the format identifier, image dimensions, and maximum color value.
    - Iterates through the pixel data in the image buffer, writing RGB values to the file in binary format.
    - Closes the file after writing all pixel data.
- **Output**: The function does not return a value; it writes the image data directly to the specified file.


---
### clip\_image\_save\_to\_bmp<!-- {{#callable:clip_image_save_to_bmp}} -->
Saves a `clip_image_u8` image to a BMP file format.
- **Inputs**:
    - `img`: A constant reference to a `clip_image_u8` structure containing the image data to be saved.
    - `filename`: A string representing the name of the file where the image will be saved.
- **Control Flow**:
    - Open a binary output file stream with the specified `filename`.
    - Check if the file stream is successfully opened; if not, log an error and return.
    - Calculate the total file size required for the BMP file format, including headers and pixel data.
    - Prepare the BMP file header and information header with the appropriate values.
    - Write the BMP file header and information header to the file.
    - Iterate over the image pixels in a bottom-to-top manner, converting the pixel format from RGB to BGR, and write the pixel data to the file.
    - Add necessary padding to each row of pixel data to ensure proper alignment.
    - Close the file stream after writing all data.
- **Output**: The function does not return a value; it writes the image data directly to the specified BMP file.


---
### clip\_image\_convert\_f32\_to\_u8<!-- {{#callable:clip_image_convert_f32_to_u8}} -->
Converts a floating-point image representation to an unsigned 8-bit image representation with clamping.
- **Inputs**:
    - `src`: A constant reference to a `clip_image_f32` structure representing the source image in floating-point format.
    - `dst`: A reference to a `clip_image_u8` structure where the converted image will be stored.
- **Control Flow**:
    - The function begins by copying the dimensions (nx, ny) from the source image (`src`) to the destination image (`dst`).
    - Next, it resizes the destination buffer to accommodate the RGB data for the image.
    - Then, it iterates over each pixel in the source image's buffer, scaling the floating-point values to the range of 0 to 255.
    - During this scaling, it uses `std::min` and `std::max` to ensure that the values are clamped within the valid range for an 8-bit unsigned integer.
    - Finally, the converted pixel values are stored in the destination image's buffer.
- **Output**: The function does not return a value; instead, it modifies the `dst` parameter in place to contain the converted image data.


---
### clip\_image\_build\_graph<!-- {{#callable:clip_image_build_graph}} -->
Builds a computation graph for processing a single image based on the specified projector type.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the context for the CLIP model, including the model parameters and state.
    - `imgs`: A reference to a `clip_image_f32_batch` structure containing a batch of images, which is expected to have exactly one image for processing.
- **Control Flow**:
    - The function asserts that the batch size of images is exactly one, throwing an error if not.
    - A `clip_graph` object is instantiated using the context and the first image from the batch.
    - A switch statement determines the projector type from the context and calls the corresponding graph building method.
    - The result of the graph building method is assigned to the `res` variable.
    - Finally, the function returns the constructed computation graph.
- **Output**: Returns a pointer to a `ggml_cgraph` structure representing the computation graph built for the specified image and projector type.


---
### clip\_init<!-- {{#callable:clip_init}} -->
Initializes the CLIP model by loading its parameters and allocating necessary resources.
- **Inputs**:
    - `fname`: A pointer to a string representing the filename of the model to be loaded.
    - `ctx_params`: A structure containing context parameters such as verbosity level for logging.
- **Control Flow**:
    - Sets the global verbosity threshold based on the provided context parameters.
    - Attempts to load the model using the `clip_model_loader` class, which initializes the model and checks for vision and audio capabilities.
    - If the model has a vision modality, it creates a new `clip_ctx` for vision, loads its hyperparameters, tensors, and allocates compute metadata.
    - If the model has an audio modality, it similarly creates a new `clip_ctx` for audio and performs the same loading and allocation steps.
    - If any exceptions occur during loading, it logs the error, cleans up any allocated contexts, and returns null pointers for both contexts.
- **Output**: Returns a `clip_init_result` structure containing pointers to the initialized `clip_ctx` for vision and audio, or null pointers if initialization fails.


---
### clip\_image\_size\_init<!-- {{#callable:clip_image_size_init}} -->
Initializes a `clip_image_size` structure with predefined width and height.
- **Inputs**: None
- **Control Flow**:
    - Allocates memory for a new `clip_image_size` structure using `new`.
    - Sets the `width` member of the structure to 448.
    - Sets the `height` member of the structure to 448.
    - Returns the pointer to the newly created `clip_image_size` structure.
- **Output**: Returns a pointer to a `clip_image_size` structure initialized with width and height set to 448.


---
### clip\_image\_u8\_init<!-- {{#callable:clip_image_u8_init}} -->
Initializes a new instance of the `clip_image_u8` structure.
- **Inputs**: None
- **Control Flow**:
    - The function directly allocates memory for a new `clip_image_u8` instance using the `new` operator.
    - It returns a pointer to the newly created instance.
- **Output**: Returns a pointer to a newly allocated `clip_image_u8` structure.


---
### clip\_image\_f32\_init<!-- {{#callable:clip_image_f32_init}} -->
Initializes a new instance of the `clip_image_f32` structure.
- **Inputs**: None
- **Control Flow**:
    - The function directly allocates memory for a new `clip_image_f32` object using the `new` operator.
    - It returns a pointer to the newly created object.
- **Output**: Returns a pointer to a newly allocated `clip_image_f32` structure.


---
### clip\_image\_f32\_batch\_init<!-- {{#callable:clip_image_f32_batch_init}} -->
Initializes a new instance of the `clip_image_f32_batch` structure.
- **Inputs**: None
- **Control Flow**:
    - The function allocates memory for a new `clip_image_f32_batch` structure using the `new` operator.
    - It returns a pointer to the newly created instance.
- **Output**: Returns a pointer to a newly allocated `clip_image_f32_batch` structure.


---
### clip\_image\_u8\_get\_data<!-- {{#callable:clip_image_u8_get_data}} -->
Retrieves the raw pixel data from a `clip_image_u8` structure and optionally sets the dimensions of the image.
- **Inputs**:
    - `img`: A pointer to a `clip_image_u8` structure that contains the image data.
    - `nx`: A pointer to a `uint32_t` variable where the width of the image will be stored, if not null.
    - `ny`: A pointer to a `uint32_t` variable where the height of the image will be stored, if not null.
- **Control Flow**:
    - Checks if the pointer `nx` is not null; if so, assigns the width of the image (`img->nx`) to the variable pointed to by `nx`.
    - Checks if the pointer `ny` is not null; if so, assigns the height of the image (`img->ny`) to the variable pointed to by `ny`.
    - Returns a pointer to the data buffer of the image (`img->buf.data()`), which contains the pixel data.
- **Output**: Returns a pointer to the raw pixel data of the image as an array of unsigned char.


---
### clip\_image\_size\_free<!-- {{#callable:clip_image_size_free}} -->
The `clip_image_size_free` function deallocates memory for a `clip_image_size` structure.
- **Inputs**:
    - `load_image_size`: A pointer to a `clip_image_size` structure that holds the dimensions of an image.
- **Control Flow**:
    - The function first checks if the `load_image_size` pointer is null.
    - If it is not null, it proceeds to delete the `load_image_size` object, freeing the allocated memory.
- **Output**: The function does not return any value; it simply frees the memory associated with the `load_image_size` pointer.


---
### clip\_image\_u8\_free<!-- {{#callable:clip_image_u8_free}} -->
The `clip_image_u8_free` function deallocates memory for a `clip_image_u8` structure if it is not null.
- **Inputs**:
    - `img`: A pointer to a `clip_image_u8` structure that holds the image data to be freed.
- **Control Flow**:
    - The function checks if the `img` pointer is not null.
    - If `img` is not null, it calls `delete` to free the memory allocated for the `clip_image_u8` structure.
- **Output**: The function does not return any value; it performs a memory deallocation operation.


---
### clip\_image\_f32\_free<!-- {{#callable:clip_image_f32_free}} -->
The `clip_image_f32_free` function deallocates memory for a `clip_image_f32` structure if it is not null.
- **Inputs**:
    - `img`: A pointer to a `clip_image_f32` structure that represents the image to be freed.
- **Control Flow**:
    - The function checks if the `img` pointer is not null.
    - If `img` is not null, it calls `delete` to free the memory allocated for the `clip_image_f32` structure.
- **Output**: The function does not return any value; it performs a memory deallocation operation.


---
### clip\_image\_u8\_batch\_free<!-- {{#callable:clip_image_u8_batch_free}} -->
The `clip_image_u8_batch_free` function deallocates memory for a `clip_image_u8_batch` structure if it is not null.
- **Inputs**:
    - `batch`: A pointer to a `clip_image_u8_batch` structure that holds a batch of images to be freed.
- **Control Flow**:
    - The function checks if the `batch` pointer is not null.
    - If the pointer is valid, it calls `delete` on the `batch` to free the allocated memory.
- **Output**: The function does not return any value; it performs a memory deallocation operation.


---
### clip\_image\_f32\_batch\_free<!-- {{#callable:clip_image_f32_batch_free}} -->
The `clip_image_f32_batch_free` function deallocates memory for a `clip_image_f32_batch` structure.
- **Inputs**:
    - `batch`: A pointer to a `clip_image_f32_batch` structure that holds image data to be freed.
- **Control Flow**:
    - The function checks if the `batch` pointer is not null.
    - If the pointer is valid, it calls `delete` on the `batch` to free the allocated memory.
- **Output**: The function does not return any value; it performs a memory deallocation operation.


---
### clip\_image\_f32\_batch\_n\_images<!-- {{#callable:clip_image_f32_batch_n_images}} -->
The `clip_image_f32_batch_n_images` function returns the number of entries in a `clip_image_f32_batch` structure.
- **Inputs**:
    - `batch`: A pointer to a `clip_image_f32_batch` structure that contains a collection of image entries.
- **Control Flow**:
    - The function accesses the `entries` member of the `batch` structure.
    - It calls the `size()` method on the `entries` vector to determine the number of images.
- **Output**: Returns the size of the `entries` vector, which indicates the number of images in the batch.


---
### clip\_image\_f32\_batch\_nx<!-- {{#callable:clip_image_f32_batch_nx}} -->
Retrieves the width (`nx`) of a specific image in a batch of float32 images.
- **Inputs**:
    - `batch`: A pointer to a `clip_image_f32_batch` structure that contains a collection of float32 images.
    - `idx`: An integer index specifying which image's width to retrieve from the batch.
- **Control Flow**:
    - Checks if the provided index `idx` is within the valid range of the batch entries.
    - If the index is invalid, logs an error message and returns 0.
    - If valid, retrieves and returns the width (`nx`) of the image at the specified index.
- **Output**: Returns the width of the specified image in the batch, or 0 if the index is invalid.


---
### clip\_image\_f32\_batch\_ny<!-- {{#callable:clip_image_f32_batch_ny}} -->
The `clip_image_f32_batch_ny` function retrieves the height (`ny`) of a specific image in a batch of float32 images.
- **Inputs**:
    - `batch`: A pointer to a `clip_image_f32_batch` structure that contains a collection of images.
    - `idx`: An integer index specifying which image's height to retrieve from the batch.
- **Control Flow**:
    - The function first checks if the provided index `idx` is within the valid range of the batch entries.
    - If the index is invalid (less than 0 or greater than or equal to the number of entries), an error is logged and the function returns 0.
    - If the index is valid, the function accesses the `ny` property of the image at the specified index and returns its value.
- **Output**: Returns the height (`ny`) of the image at the specified index in the batch, or 0 if the index is invalid.


---
### clip\_image\_f32\_get\_img<!-- {{#callable:clip_image_f32_get_img}} -->
Retrieves an image from a `clip_image_f32_batch` based on the specified index.
- **Inputs**:
    - `batch`: A pointer to a `clip_image_f32_batch` structure that contains a collection of images.
    - `idx`: An integer index specifying which image to retrieve from the batch.
- **Control Flow**:
    - Checks if the provided index `idx` is within the valid range of the batch entries.
    - If the index is invalid, logs an error message and returns a null pointer.
    - If the index is valid, retrieves and returns the image at the specified index from the batch.
- **Output**: Returns a pointer to the `clip_image_f32` image at the specified index, or nullptr if the index is invalid.


---
### clip\_build\_img\_from\_pixels<!-- {{#callable:clip_build_img_from_pixels}} -->
The `clip_build_img_from_pixels` function constructs a `clip_image_u8` object from raw RGB pixel data.
- **Inputs**:
    - `rgb_pixels`: A pointer to an array of unsigned char representing the RGB pixel data.
    - `nx`: An integer representing the width of the image in pixels.
    - `ny`: An integer representing the height of the image in pixels.
    - `img`: A pointer to a `clip_image_u8` structure where the constructed image will be stored.
- **Control Flow**:
    - The function begins by setting the width (`nx`) and height (`ny`) of the `img` structure.
    - It then resizes the buffer of `img` to accommodate the RGB pixel data, which is calculated as 3 times the product of width and height.
    - Finally, it copies the pixel data from the `rgb_pixels` array into the `buf` of the `img` structure using `memcpy`.
- **Output**: The function does not return a value; instead, it modifies the `img` structure to contain the image data built from the provided pixel data.


---
### normalize\_image\_u8\_to\_f32<!-- {{#callable:normalize_image_u8_to_f32}} -->
Normalizes an 8-bit unsigned image to a 32-bit floating-point representation using specified mean and standard deviation values.
- **Inputs**:
    - `src`: A constant reference to a `clip_image_u8` structure representing the source image in 8-bit unsigned format.
    - `dst`: A reference to a `clip_image_f32` structure where the normalized image will be stored in 32-bit floating-point format.
    - `mean`: An array of three floats representing the mean values for each color channel (R, G, B) used for normalization.
    - `std`: An array of three floats representing the standard deviation values for each color channel (R, G, B) used for normalization.
- **Control Flow**:
    - The function initializes the dimensions of the destination image `dst` to match those of the source image `src`.
    - The buffer of the destination image `dst` is resized to accommodate the same number of pixels as in the source image.
    - A loop iterates over each pixel in the source image, calculating the normalized value for each color channel (R, G, B) using the formula: ((pixel_value / 255.0) - mean[c]) / std[c].
    - The normalized values are stored in the corresponding positions in the destination image's buffer.
- **Output**: The function does not return a value; instead, it modifies the `dst` parameter in place to contain the normalized image data.


---
### clip\_image\_preprocess<!-- {{#callable:clip_image_preprocess}} -->
The `clip_image_preprocess` function preprocesses an input image based on the model's configuration and prepares it for further processing.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model context and parameters.
    - `img`: A pointer to a `clip_image_u8` structure representing the input image in 8-bit unsigned format.
    - `res_imgs`: A pointer to a `clip_image_f32_batch` structure where the processed images in float32 format will be stored.
- **Control Flow**:
    - The function begins by determining the original size of the input image and whether to pad the image to a square based on model parameters.
    - It checks the model's projector type and applies different preprocessing techniques based on the type, including slicing the image, resizing, and normalizing.
    - If the projector type is `PROJECTOR_TYPE_QWEN2VL` or `PROJECTOR_TYPE_QWEN25VL`, it resizes the image while preserving the aspect ratio.
    - For other projector types, it either pads the image to a square or slices it into smaller images based on specific instructions.
    - Finally, the processed images are normalized and stored in the `res_imgs` structure.
- **Output**: The function returns a boolean indicating the success of the preprocessing operation, with the processed images stored in the `res_imgs` structure.
- **Functions called**:
    - [`clip_is_minicpmv`](#clip_is_minicpmv)
    - [`normalize_image_u8_to_f32`](#normalize_image_u8_to_f32)


---
### clip\_get\_newline\_tensor<!-- {{#callable:clip_get_newline_tensor}} -->
The `clip_get_newline_tensor` function retrieves the `image_newline` tensor from the provided `clip_ctx` context.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model context from which the tensor is to be retrieved.
- **Control Flow**:
    - The function directly accesses the `model` member of the `ctx` structure.
    - It returns the `image_newline` tensor from the `model`.
- **Output**: Returns a pointer to a `ggml_tensor` representing the `image_newline` tensor from the model.


---
### clip\_free<!-- {{#callable:clip_free}} -->
The `clip_free` function deallocates memory for a `clip_ctx` object if it is not null.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that holds the context for the CLIP model.
- **Control Flow**:
    - The function first checks if the `ctx` pointer is null.
    - If `ctx` is not null, it proceeds to delete the `ctx` object, freeing the associated memory.
- **Output**: The function does not return any value; it performs a memory deallocation operation.


---
### clip\_embd\_nbytes<!-- {{#callable:clip_embd_nbytes}} -->
The `clip_embd_nbytes` function calculates the number of bytes required for the embeddings based on the image size defined in the context.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model parameters and context for the CLIP model.
- **Control Flow**:
    - The function retrieves the image size parameters (width and height) from the `ctx` structure.
    - It then calls the [`clip_embd_nbytes_by_img`](#clip_embd_nbytes_by_img) function, passing the context and the image dimensions to calculate the required number of bytes for the embeddings.
- **Output**: Returns the total number of bytes required for the embeddings based on the image size.
- **Functions called**:
    - [`clip_embd_nbytes_by_img`](#clip_embd_nbytes_by_img)


---
### clip\_embd\_nbytes\_by\_img<!-- {{#callable:clip_embd_nbytes_by_img}} -->
Calculates the number of bytes required for embedding based on the image dimensions.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model context.
    - `img_w`: An integer representing the width of the image.
    - `img_h`: An integer representing the height of the image.
- **Control Flow**:
    - A `clip_image_f32` structure is instantiated to hold the image dimensions.
    - The width and height of the image are assigned to the `nx` and `ny` members of the `clip_image_f32` structure.
    - The function [`clip_n_output_tokens`](#clip_n_output_tokens) is called with the context and the image structure to get the number of output tokens.
    - The function [`clip_n_mmproj_embd`](#clip_n_mmproj_embd) is called to get the embedding dimension size.
    - The total number of bytes is calculated by multiplying the number of output tokens, the embedding dimension size, and the size of a float.
- **Output**: Returns the total number of bytes required for the embedding as a `size_t` value.
- **Functions called**:
    - [`clip_n_output_tokens`](#clip_n_output_tokens)
    - [`clip_n_mmproj_embd`](#clip_n_mmproj_embd)


---
### clip\_get\_image\_size<!-- {{#callable:clip_get_image_size}} -->
The `clip_get_image_size` function retrieves the image size parameter from the model's hyperparameters.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model context, including hyperparameters.
- **Control Flow**:
    - Accesses the `model` member of the `ctx` structure.
    - Retrieves the `image_size` from the `hparams` member of the `model`.
- **Output**: Returns the image size as an `int32_t` value, which represents the size of the images used in the model.


---
### clip\_get\_patch\_size<!-- {{#callable:clip_get_patch_size}} -->
The `clip_get_patch_size` function retrieves the patch size parameter from the model's hyperparameters.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model context, including hyperparameters.
- **Control Flow**:
    - The function accesses the `model` member of the `ctx` structure.
    - It retrieves the `patch_size` from the `hparams` member of the `model`.
- **Output**: Returns the patch size as an `int32_t` value.


---
### clip\_get\_hidden\_size<!-- {{#callable:clip_get_hidden_size}} -->
Retrieves the hidden size of the model from the given `clip_ctx` structure.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model context, including its hyperparameters.
- **Control Flow**:
    - The function accesses the `model` member of the `ctx` structure.
    - It retrieves the `hparams` member from the `model`.
    - Finally, it returns the value of `n_embd` from `hparams`, which represents the hidden size.
- **Output**: Returns an integer representing the hidden size of the model, specifically the number of embedding dimensions.


---
### clip\_patch\_merge\_type<!-- {{#callable:clip_patch_merge_type}} -->
Returns a string representing the type of patch merging used in the model context.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model context and parameters.
- **Control Flow**:
    - Checks the `mm_patch_merge_type` field of the `hparams` structure within the `model` of the provided `ctx`.
    - If the `mm_patch_merge_type` is equal to `PATCH_MERGE_SPATIAL_UNPAD`, it returns the string 'spatial_unpad'.
    - Otherwise, it returns the string 'flat'.
- **Output**: A constant string indicating the type of patch merging, either 'spatial_unpad' or 'flat'.


---
### clip\_image\_grid<!-- {{#callable:clip_image_grid}} -->
The `clip_image_grid` function retrieves the pointer to the front of the `image_grid_pinpoints` vector if it contains elements, otherwise returns a null pointer.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model and its parameters.
- **Control Flow**:
    - The function checks if the `image_grid_pinpoints` vector in the `model.hparams` of the `ctx` structure is not empty.
    - If the vector is not empty, it returns a pointer to the first element of the vector.
    - If the vector is empty, it returns a null pointer.
- **Output**: Returns a pointer to the first element of the `image_grid_pinpoints` vector or a null pointer if the vector is empty.


---
### get\_clip\_image\_grid\_size<!-- {{#callable:get_clip_image_grid_size}} -->
The `get_clip_image_grid_size` function returns the size of the `image_grid_pinpoints` vector from the provided `clip_ctx` structure.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model parameters and context.
- **Control Flow**:
    - Accesses the `model` member of the `ctx` structure.
    - Retrieves the `hparams` member from the `model`.
    - Returns the size of the `image_grid_pinpoints` vector from `hparams`.
- **Output**: Returns a `size_t` representing the number of elements in the `image_grid_pinpoints` vector.


---
### clip\_n\_output\_tokens\_x<!-- {{#callable:clip_n_output_tokens_x}} -->
The `clip_n_output_tokens_x` function calculates the number of output tokens based on the input image dimensions and the model's projection type.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model context and parameters.
    - `img`: A pointer to a `clip_image_f32` structure representing the input image.
- **Control Flow**:
    - Retrieve the model hyperparameters from the `ctx` structure.
    - Call the [`clip_n_output_tokens`](#clip_n_output_tokens) function to get the total number of output tokens for the given image.
    - Check the projector type from the context.
    - If the projector type is either `PROJECTOR_TYPE_QWEN2VL` or `PROJECTOR_TYPE_QWEN25VL`, calculate the number of output tokens based on the image width and patch size.
    - Return the calculated number of output tokens or the total number of tokens based on the projector type.
- **Output**: Returns an integer representing the number of output tokens based on the input image and model configuration.
- **Functions called**:
    - [`clip_n_output_tokens`](#clip_n_output_tokens)


---
### clip\_n\_output\_tokens\_y<!-- {{#callable:clip_n_output_tokens_y}} -->
The `clip_n_output_tokens_y` function calculates the number of output tokens in the y-dimension based on the image height and model parameters.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model parameters and context for the CLIP model.
    - `img`: A pointer to a `clip_image_f32` structure representing the input image, which contains its dimensions.
- **Control Flow**:
    - The function retrieves the model parameters from the `ctx` structure.
    - It checks the projector type of the model using `ctx->proj_type()`.
    - If the projector type is either `PROJECTOR_TYPE_QWEN2VL` or `PROJECTOR_TYPE_QWEN25VL`, it calculates the number of output tokens in the y-dimension based on the image height and patch size.
    - If the projector type is not one of the specified types, it returns 1 as the output.
- **Output**: The function returns an integer representing the number of output tokens in the y-dimension, which is calculated based on the image height and model parameters.


---
### clip\_n\_output\_tokens<!-- {{#callable:clip_n_output_tokens}} -->
Calculates the number of output tokens based on the model's parameters and the input image dimensions.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model parameters and context for the operation.
    - `img`: A pointer to a `clip_image_f32` structure representing the input image in float32 format.
- **Control Flow**:
    - Retrieve the model's hyperparameters from the `ctx` structure.
    - Calculate the total number of patches in a square image based on the image size and patch size.
    - Determine the projector type using the `proj_type` method from the `ctx` structure.
    - Use a switch statement to handle different projector types, adjusting the number of patches accordingly.
    - For certain projector types, modify the number of patches based on specific conditions, such as the presence of special tokens or scaling factors.
    - Return the calculated number of output tokens.
- **Output**: Returns an integer representing the number of output tokens that will be generated based on the input image and model parameters.


---
### get\_1d\_sincos\_pos\_embed\_from\_grid\_new<!-- {{#callable:get_1d_sincos_pos_embed_from_grid_new}} -->
The `get_1d_sincos_pos_embed_from_grid_new` function generates a 3D positional embedding using sine and cosine functions based on a given grid of positions.
- **Inputs**:
    - `embed_dim`: An integer representing the dimensionality of the embedding, which must be even.
    - `pos`: A 2D vector of floats representing the positions for which the embeddings are to be calculated.
- **Control Flow**:
    - The function asserts that `embed_dim` is even to ensure valid calculations.
    - It initializes the height `H` and width `W` based on the size of the `pos` vector.
    - An `omega` vector is created to store frequency values calculated from `embed_dim`.
    - A 3D vector `emb` is initialized to hold the resulting embeddings.
    - Two nested loops iterate over the height and width of the `pos` vector, calculating sine and cosine values for each position and storing them in `emb`.
- **Output**: Returns a 3D vector of floats representing the sine and cosine positional embeddings for the given positions.


---
### get\_2d\_sincos\_pos\_embed\_from\_grid<!-- {{#callable:get_2d_sincos_pos_embed_from_grid}} -->
The `get_2d_sincos_pos_embed_from_grid` function generates a 2D sinusoidal positional embedding from a given grid.
- **Inputs**:
    - `embed_dim`: An integer representing the dimensionality of the embedding, which must be even.
    - `grid`: A 3D vector containing two 2D grids, where the first grid represents the height and the second grid represents the width.
- **Control Flow**:
    - The function asserts that `embed_dim` is even.
    - It calls [`get_1d_sincos_pos_embed_from_grid_new`](#get_1d_sincos_pos_embed_from_grid_new) twice to compute embeddings for height and width, each with half the `embed_dim`.
    - It initializes a 3D vector `emb` to store the final embeddings.
    - It iterates over the height and width of the embeddings, filling in the first half with height embeddings and the second half with width embeddings.
- **Output**: Returns a 3D vector containing the 2D sinusoidal positional embeddings, structured as (H, W, embed_dim).
- **Functions called**:
    - [`get_1d_sincos_pos_embed_from_grid_new`](#get_1d_sincos_pos_embed_from_grid_new)


---
### get\_2d\_sincos\_pos\_embed<!-- {{#callable:get_2d_sincos_pos_embed}} -->
The `get_2d_sincos_pos_embed` function generates a 2D sine-cosine positional embedding for a given image size and embedding dimension.
- **Inputs**:
    - `embed_dim`: An integer representing the dimensionality of the embedding.
    - `image_size`: A pair of integers representing the height and width of the image.
- **Control Flow**:
    - Extracts the height and width from the `image_size` pair.
    - Initializes two vectors, `grid_h` and `grid_w`, to hold the height and width indices.
    - Populates `grid_h` and `grid_w` with values from 0 to height-1 and 0 to width-1 respectively.
    - Creates a 2D grid initialized with the width values.
    - Fills the grid with height and width values to create a 2D grid representation.
    - Calls [`get_2d_sincos_pos_embed_from_grid`](#get_2d_sincos_pos_embed_from_grid) to compute the 3D positional embeddings from the 2D grid.
    - Reshapes the 3D positional embeddings into a 2D format suitable for output.
- **Output**: Returns a 2D vector containing the positional embeddings for the specified image size and embedding dimension.
- **Functions called**:
    - [`get_2d_sincos_pos_embed_from_grid`](#get_2d_sincos_pos_embed_from_grid)


---
### clip\_image\_encode<!-- {{#callable:clip_image_encode}} -->
Encodes a float32 image into a vector using a specified context and number of threads.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model and context for encoding.
    - `n_threads`: An integer representing the number of threads to use for encoding.
    - `img`: A pointer to a `clip_image_f32` structure representing the image to be encoded.
    - `vec`: A pointer to a float array where the resulting encoded vector will be stored.
- **Control Flow**:
    - Initializes a `clip_image_f32_batch` structure to hold the image batch.
    - Creates a copy of the input image and stores it in the batch.
    - Calls the [`clip_image_batch_encode`](#clip_image_batch_encode) function to perform the encoding using the context, number of threads, and the image batch.
- **Output**: Returns a boolean indicating the success or failure of the encoding operation.
- **Functions called**:
    - [`clip_image_batch_encode`](#clip_image_batch_encode)


---
### clip\_image\_batch\_encode<!-- {{#callable:clip_image_batch_encode}} -->
Encodes a batch of images into embeddings using a specified model context.
- **Inputs**:
    - `ctx`: A pointer to the `clip_ctx` structure that contains the model context and parameters for encoding.
    - `n_threads`: An integer representing the number of threads to use for processing.
    - `imgs_c_ptr`: A pointer to a `clip_image_f32_batch` structure containing the images to be encoded.
    - `vec`: A pointer to a float array where the resulting embeddings will be stored.
- **Control Flow**:
    - The function first dereferences the input batch of images and checks the batch size, returning false if it is not equal to 1.
    - It initializes the inference graph by resetting the scheduling context and building the graph based on the input images.
    - The function retrieves model parameters such as image dimensions and patch sizes from the context.
    - It prepares the input tensor for the raw pixel values of the image, ensuring the correct layout and data type.
    - Depending on whether the input is an image or audio, it sets the appropriate input tensor values.
    - The function then sets additional input tensors based on the model's projector type, which may include position embeddings and other parameters.
    - After setting all inputs, it schedules the graph for computation and executes it using the specified number of threads.
    - Finally, it retrieves the output embeddings from the graph and copies them to the provided output vector.
- **Output**: Returns a boolean indicating the success of the encoding operation, with the resulting embeddings stored in the provided vector.
- **Functions called**:
    - [`ggml_backend_sched_reset`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_reset)
    - [`clip_image_build_graph`](#clip_image_build_graph)
    - [`ggml_backend_sched_alloc_graph`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_alloc_graph)
    - [`ggml_graph_get_tensor`](../../ggml/src/ggml.c.driver.md#ggml_graph_get_tensor)
    - [`ggml_nelements`](../../ggml/src/ggml.c.driver.md#ggml_nelements)
    - [`ggml_backend_tensor_set`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`ggml_nbytes`](../../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`clip_n_mmproj_embd`](#clip_n_mmproj_embd)
    - [`get_2d_sincos_pos_embed`](#get_2d_sincos_pos_embed)
    - [`ggml_backend_reg_get_proc_address`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_get_proc_address)
    - [`ggml_backend_sched_graph_compute`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_graph_compute)
    - [`ggml_backend_tensor_get`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_get)
    - [`print_tensor_shape`](clip-impl.h.driver.md#print_tensor_shape)
    - [`print_tensor_data`](clip-impl.h.driver.md#print_tensor_data)
    - [`ggml_graph_node`](../../ggml/src/ggml.c.driver.md#ggml_graph_node)
    - [`clip_n_output_tokens`](#clip_n_output_tokens)


---
### clip\_n\_mmproj\_embd<!-- {{#callable:clip_n_mmproj_embd}} -->
The `clip_n_mmproj_embd` function retrieves the embedding dimension based on the projector type defined in the model context.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model and its parameters.
- **Control Flow**:
    - The function accesses the model's hyperparameters through the `ctx` pointer.
    - It uses a switch statement to determine the projector type from `ctx->model.proj_type`.
    - For each case of the projector type, it retrieves the corresponding embedding dimension from the model's parameters.
    - If the projector type is `PROJECTOR_TYPE_MINICPMV`, it checks the version and returns a fixed embedding size based on that version.
    - If the projector type is unknown, it calls `GGML_ABORT` to terminate the function.
- **Output**: The function returns an integer representing the embedding dimension based on the specified projector type.


---
### clip\_is\_minicpmv<!-- {{#callable:clip_is_minicpmv}} -->
Determines if the current `clip_ctx` is using the MINICPMV projector type and returns its version.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model and its parameters.
- **Control Flow**:
    - Checks if the projector type of the context is `PROJECTOR_TYPE_MINICPMV`.
    - If true, returns the `minicpmv_version` from the model's hyperparameters.
    - If false, returns 0.
- **Output**: Returns an integer representing the version of the MINICPMV projector if it is being used, otherwise returns 0.


---
### clip\_is\_glm<!-- {{#callable:clip_is_glm}} -->
The `clip_is_glm` function checks if the projector type of the given context is `PROJECTOR_TYPE_GLM_EDGE`.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the context for the CLIP model.
- **Control Flow**:
    - The function accesses the `proj_type` member of the `clip_model` structure contained within the `ctx` parameter.
    - It compares the `proj_type` to `PROJECTOR_TYPE_GLM_EDGE`.
    - The function returns true if the types match, otherwise it returns false.
- **Output**: Returns a boolean value indicating whether the projector type is `PROJECTOR_TYPE_GLM_EDGE`.


---
### clip\_is\_qwen2vl<!-- {{#callable:clip_is_qwen2vl}} -->
Determines if the projector type of the given `clip_ctx` is either `PROJECTOR_TYPE_QWEN2VL` or `PROJECTOR_TYPE_QWEN25VL`.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the context of the CLIP model, including the projector type.
- **Control Flow**:
    - The function retrieves the projector type from the `ctx` structure using the `proj_type()` method.
    - It checks if the retrieved projector type matches either `PROJECTOR_TYPE_QWEN2VL` or `PROJECTOR_TYPE_QWEN25VL`.
    - The function returns true if either condition is met, otherwise it returns false.
- **Output**: Returns a boolean value indicating whether the projector type is either `PROJECTOR_TYPE_QWEN2VL` or `PROJECTOR_TYPE_QWEN25VL`.


---
### clip\_is\_llava<!-- {{#callable:clip_is_llava}} -->
The `clip_is_llava` function checks if the model in the given context has a LLaVA projector.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model and its parameters.
- **Control Flow**:
    - The function accesses the `model` member of the `ctx` structure.
    - It checks the `has_llava_projector` boolean flag in the `hparams` of the model.
    - The function returns the value of the `has_llava_projector` flag.
- **Output**: Returns a boolean value indicating whether the model has a LLaVA projector.


---
### clip\_is\_gemma3<!-- {{#callable:clip_is_gemma3}} -->
The `clip_is_gemma3` function checks if the projector type of the given context is `PROJECTOR_TYPE_GEMMA3`.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the context for the CLIP model.
- **Control Flow**:
    - The function retrieves the projector type from the `ctx` structure using the `proj_type()` method.
    - It compares the retrieved projector type with `PROJECTOR_TYPE_GEMMA3`.
    - The function returns true if the types match, otherwise it returns false.
- **Output**: A boolean value indicating whether the projector type is `PROJECTOR_TYPE_GEMMA3`.


---
### clip\_has\_vision\_encoder<!-- {{#callable:clip_has_vision_encoder}} -->
The `clip_has_vision_encoder` function checks if the model modality of the given `clip_ctx` context is set to vision.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model context, including its modality.
- **Control Flow**:
    - The function accesses the `modality` member of the `model` structure within the `ctx` parameter.
    - It compares the `modality` to the constant `CLIP_MODALITY_VISION`.
    - The function returns a boolean value indicating whether the modality is set to vision.
- **Output**: Returns `true` if the model modality is `CLIP_MODALITY_VISION`, otherwise returns `false`.


---
### clip\_has\_audio\_encoder<!-- {{#callable:clip_has_audio_encoder}} -->
The `clip_has_audio_encoder` function checks if the audio encoder is present in the given clip context.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model context, including its modality.
- **Control Flow**:
    - The function accesses the `modality` field of the `model` member of the `ctx` structure.
    - It compares the `modality` to `CLIP_MODALITY_AUDIO` to determine if the audio encoder is present.
    - The function returns a boolean value based on the comparison.
- **Output**: Returns true if the modality is `CLIP_MODALITY_AUDIO`, indicating the presence of an audio encoder; otherwise, it returns false.


---
### clip\_has\_whisper\_encoder<!-- {{#callable:clip_has_whisper_encoder}} -->
The `clip_has_whisper_encoder` function checks if the projector type of the given context is either `PROJECTOR_TYPE_ULTRAVOX` or `PROJECTOR_TYPE_QWEN2A`.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the context of the CLIP model, including its projector type.
- **Control Flow**:
    - The function retrieves the projector type from the `ctx` structure using the `proj_type()` method.
    - It then checks if the retrieved projector type is equal to `PROJECTOR_TYPE_ULTRAVOX` or `PROJECTOR_TYPE_QWEN2A`.
    - The function returns true if either condition is met, otherwise it returns false.
- **Output**: A boolean value indicating whether the projector type of the context is either `PROJECTOR_TYPE_ULTRAVOX` or `PROJECTOR_TYPE_QWEN2A`.


---
### clip\_encode\_float\_image<!-- {{#callable:clip_encode_float_image}} -->
Encodes a float image into a vector representation using a specified context.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model context for encoding.
    - `n_threads`: An integer representing the number of threads to use for processing.
    - `img`: A pointer to a float array representing the image data, expected to be in RGB format.
    - `h`: An integer representing the height of the image.
    - `w`: An integer representing the width of the image.
    - `vec`: A pointer to a float array where the resulting vector representation will be stored.
- **Control Flow**:
    - A `clip_image_f32` structure is created to hold the image data.
    - The buffer of the `clip_image_f32` structure is resized to accommodate the image data.
    - A loop iterates over the height and width of the image to copy the pixel data from the input float array to the buffer of the `clip_image_f32` structure.
    - The dimensions of the `clip_image_f32` structure are set to the width and height of the image.
    - The [`clip_image_encode`](#clip_image_encode) function is called to perform the encoding using the context and the populated `clip_image_f32` structure.
    - The function returns true to indicate successful execution.
- **Output**: Returns a boolean value indicating the success of the encoding operation.
- **Functions called**:
    - [`clip_image_encode`](#clip_image_encode)


---
### clip\_get\_projector\_type<!-- {{#callable:clip_get_projector_type}} -->
Retrieves the `projector_type` from the given `clip_ctx` structure.
- **Inputs**:
    - `ctx`: A pointer to a `clip_ctx` structure that contains the model context, including the projector type.
- **Control Flow**:
    - The function accesses the `proj_type` member of the `model` field within the `ctx` structure.
    - It returns the value of `proj_type`, which indicates the type of projector being used.
- **Output**: Returns the `projector_type` enumeration value that specifies the type of projector associated with the given `clip_ctx`.


---
### clip\_image\_f32\_batch\_add\_mel<!-- {{#callable:clip_image_f32_batch_add_mel}} -->
The `clip_image_f32_batch_add_mel` function adds a mel spectrogram to a batch of float32 images.
- **Inputs**:
    - `batch`: A pointer to a `clip_image_f32_batch` structure that holds the batch of images.
    - `n_mel`: An integer representing the number of mel frequency bins.
    - `n_frames`: An integer representing the number of frames in the mel spectrogram.
    - `mel`: A pointer to a float array containing the mel spectrogram data.
- **Control Flow**:
    - A new `clip_image_f32` object is created to hold the mel spectrogram.
    - The dimensions of the mel spectrogram are set using `n_frames` and `n_mel`.
    - The buffer of the `clip_image_f32` object is resized to accommodate the mel spectrogram data.
    - The mel data is copied into the buffer of the `clip_image_f32` object.
    - The newly created `clip_image_f32` object is added to the `entries` vector of the `batch`.
- **Output**: The function does not return a value; it modifies the input `batch` by adding a new audio entry.


