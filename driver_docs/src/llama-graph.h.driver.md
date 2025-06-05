# Purpose
This C++ source file is part of a larger system designed to handle graph-based computations, likely in the context of machine learning or neural network models. The file defines a series of classes and structures that facilitate the construction and manipulation of computational graphs, specifically for models that may involve multi-modal inputs or complex attention mechanisms. The primary focus of the file is on defining input and result interfaces for these graphs, as well as the context and parameters necessary for their construction and execution.

The file includes several classes that implement the `llm_graph_input_i` interface, each tailored to handle different types of input data, such as embeddings, positional information, attention scaling, and cross-attention embeddings. These classes are designed to set up the input tensors required for graph computations. Additionally, the file defines the `llm_graph_result_i` interface and its implementation, which are responsible for managing the output tensors of the graph, such as tokens, logits, and embeddings. The `llm_graph_context` class encapsulates the parameters and methods needed to build and execute these graphs, including functions for constructing various components like attention mechanisms and feed-forward networks. The file is structured to be part of a larger library or framework, providing essential components for building and executing complex computational graphs in a modular and extensible manner.
# Imports and Dependencies

---
- `llama-arch.h`
- `llama-hparams.h`
- `llama-adapter.h`
- `cstdint`
- `vector`
- `memory`
- `set`
- `functional`


# Data Structures

---
### llm\_graph\_type<!-- {{#data_structure:llm_graph_type}} -->
- **Type**: `enum`
- **Members**:
    - `LLM_GRAPH_TYPE_DEFAULT`: Represents the default graph type.
    - `LLM_GRAPH_TYPE_ENCODER`: Represents the encoder graph type.
    - `LLM_GRAPH_TYPE_DECODER`: Represents the decoder graph type.
- **Description**: The `llm_graph_type` enum defines different types of graph structures that can be produced by certain models, typically multi-modal ones. It includes three types: `LLM_GRAPH_TYPE_DEFAULT`, `LLM_GRAPH_TYPE_ENCODER`, and `LLM_GRAPH_TYPE_DECODER`, which are used to specify the nature of the graph being utilized or generated in the context of a machine learning model.


---
### llm\_ffn\_op\_type<!-- {{#data_structure:llm_ffn_op_type}} -->
- **Type**: `enum`
- **Members**:
    - `LLM_FFN_SILU`: Represents the SiLU (Sigmoid Linear Unit) activation function.
    - `LLM_FFN_GELU`: Represents the GELU (Gaussian Error Linear Unit) activation function.
    - `LLM_FFN_RELU`: Represents the ReLU (Rectified Linear Unit) activation function.
    - `LLM_FFN_RELU_SQR`: Represents a variant of ReLU with squaring.
    - `LLM_FFN_SWIGLU`: Represents the SwiGLU (Swish Gated Linear Unit) activation function.
- **Description**: The `llm_ffn_op_type` enum defines a set of activation functions used in feed-forward neural network operations. Each enumerator corresponds to a specific activation function, such as SiLU, GELU, ReLU, a squared variant of ReLU, and SwiGLU, which are commonly used in neural network architectures to introduce non-linearity.


---
### llm\_ffn\_gate\_type<!-- {{#data_structure:llm_ffn_gate_type}} -->
- **Type**: `enum`
- **Members**:
    - `LLM_FFN_SEQ`: Represents a sequential feed-forward network gate type.
    - `LLM_FFN_PAR`: Represents a parallel feed-forward network gate type, where the ffn_gate is parallel to ffn_up.
- **Description**: The `llm_ffn_gate_type` is an enumeration that defines the types of gates used in a feed-forward network within a machine learning model. It includes two types: `LLM_FFN_SEQ`, which indicates a sequential gate type, and `LLM_FFN_PAR`, which indicates a parallel gate type where the gate operates in parallel to the feed-forward network's upward path. This enum is used to specify the configuration of the feed-forward network in the model's architecture.


---
### llm\_norm\_type<!-- {{#data_structure:llm_norm_type}} -->
- **Type**: `enum`
- **Members**:
    - `LLM_NORM`: Represents a standard normalization type.
    - `LLM_NORM_RMS`: Represents a root mean square normalization type.
    - `LLM_NORM_GROUP`: Represents a group normalization type.
- **Description**: The `llm_norm_type` enumeration defines different types of normalization that can be applied within a machine learning model. It includes standard normalization (`LLM_NORM`), root mean square normalization (`LLM_NORM_RMS`), and group normalization (`LLM_NORM_GROUP`). These normalization types are typically used to stabilize and improve the training of neural networks by ensuring that the input data to each layer has a consistent scale.


---
### llama\_cross<!-- {{#data_structure:llama_cross}} -->
- **Type**: `struct`
- **Members**:
    - `n_embd`: Represents the number of embeddings.
    - `n_enc`: Represents the number of encoders.
    - `v_embd`: A vector storing embeddings data copied to host memory.
    - `seq_ids_enc`: A vector of sets used to construct the cross-attention mask in the decoder.
- **Description**: The `llama_cross` struct is designed to facilitate the transfer of embedding data from an encoder to a decoder in a machine learning model. It contains fields for storing the number of embeddings and encoders, as well as a vector for temporarily holding embeddings data in host memory. Additionally, it includes a vector of sets to help construct the cross-attention mask required by the decoder, indicating its role in managing cross-attention mechanisms in sequence-to-sequence models.


---
### llm\_graph\_input\_i<!-- {{#data_structure:llm_graph_input_i}} -->
- **Type**: `class`
- **Description**: The `llm_graph_input_i` class is an abstract base class that defines a common interface for various types of graph input classes in a machine learning model. It contains a pure virtual function `set_input` which must be implemented by derived classes to set the input data for the graph, using a `llama_ubatch` object. This class serves as a foundation for creating specific input types that can be used in constructing and managing input data for different graph configurations in the model.
- **Member Functions**:
    - [`llm_graph_input_i::~llm_graph_input_i`](#llm_graph_input_illm_graph_input_i)

**Methods**

---
#### llm\_graph\_input\_i::\~llm\_graph\_input\_i<!-- {{#callable:llm_graph_input_i::~llm_graph_input_i}} -->
The destructor `~llm_graph_input_i` is a virtual default destructor for the `llm_graph_input_i` interface class.
- **Inputs**: None
- **Control Flow**:
    - The function is a virtual destructor, which means it is intended to be overridden by derived classes if necessary.
    - It is defined as `= default`, indicating that the compiler will generate the default implementation for the destructor.
    - Being virtual ensures that the destructor of the derived class is called when an object is deleted through a pointer to the base class.
- **Output**: There is no output from this destructor function; it ensures proper cleanup of derived class objects when deleted through a base class pointer.
- **See also**: [`llm_graph_input_i`](#llm_graph_input_i)  (Data Structure)



---
### llm\_graph\_input\_embd<!-- {{#data_structure:llm_graph_input_embd}} -->
- **Type**: `class`
- **Members**:
    - `tokens`: A pointer to a ggml_tensor representing integer tokens with dimensions [n_batch].
    - `embd`: A pointer to a ggml_tensor representing floating-point embeddings with dimensions [n_embd, n_batch].
- **Description**: The `llm_graph_input_embd` class is a specialized implementation of the `llm_graph_input_i` interface, designed to handle input embeddings for a graph-based model. It contains two primary data members: `tokens`, which stores integer token data, and `embd`, which stores the corresponding floating-point embeddings. The class provides a method `set_input` to initialize these tensors based on the input batch data (`llama_ubatch`). This class is part of a larger framework for managing and processing input data in a machine learning model, particularly in scenarios involving graph-based computations.
- **Member Functions**:
    - [`llm_graph_input_embd::llm_graph_input_embd`](#llm_graph_input_embdllm_graph_input_embd)
    - [`llm_graph_input_embd::~llm_graph_input_embd`](#llm_graph_input_embdllm_graph_input_embd)
    - [`llm_graph_input_embd::set_input`](llama-graph.cpp.driver.md#llm_graph_input_embdset_input)
- **Inherits From**:
    - [`llm_graph_input_i`](#llm_graph_input_i)

**Methods**

---
#### llm\_graph\_input\_embd::llm\_graph\_input\_embd<!-- {{#callable:llm_graph_input_embd::llm_graph_input_embd}} -->
The `llm_graph_input_embd` class is a specialized implementation of the `llm_graph_input_i` interface, designed to handle input embeddings for a graph, with default constructors and destructors.
- **Inputs**: None
- **Control Flow**:
    - The class `llm_graph_input_embd` inherits from `llm_graph_input_i` and provides default implementations for its constructor and destructor.
    - It declares two member variables, `tokens` and `embd`, which are pointers to `ggml_tensor` objects, intended to store token and embedding data respectively.
    - The class overrides the `set_input` method from the `llm_graph_input_i` interface, but the implementation details are not provided in the given code.
- **Output**: The class itself does not produce any output directly, but it is designed to manage input embeddings for a graph, potentially affecting the graph's processing and output.
- **See also**: [`llm_graph_input_embd`](#llm_graph_input_embd)  (Data Structure)


---
#### llm\_graph\_input\_embd::\~llm\_graph\_input\_embd<!-- {{#callable:llm_graph_input_embd::~llm_graph_input_embd}} -->
The destructor `~llm_graph_input_embd` is a virtual default destructor for the `llm_graph_input_embd` class.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as a virtual function, allowing for proper cleanup in derived classes if any.
    - It is marked as `default`, indicating that the compiler should generate the default implementation for the destructor.
- **Output**: There is no output from this destructor as it is used for cleanup when an object of `llm_graph_input_embd` is destroyed.
- **See also**: [`llm_graph_input_embd`](#llm_graph_input_embd)  (Data Structure)



---
### llm\_graph\_input\_pos<!-- {{#data_structure:llm_graph_input_pos}} -->
- **Type**: `class`
- **Members**:
    - `pos`: A pointer to a ggml_tensor representing position data, initialized to nullptr.
    - `n_pos_per_embd`: A constant integer representing the number of positions per embedding, defaulting to 1.
- **Description**: The `llm_graph_input_pos` class is a specialized input handler for graph-based models, inheriting from `llm_graph_input_i`. It manages position data for embeddings, with a focus on handling the number of positions per embedding. The class includes a method to set input data, which processes position data from a `llama_ubatch` object and adjusts it based on the number of positions per embedding, particularly for multi-dimensional position encoding scenarios.
- **Member Functions**:
    - [`llm_graph_input_pos::llm_graph_input_pos`](#llm_graph_input_posllm_graph_input_pos)
    - [`llm_graph_input_pos::~llm_graph_input_pos`](#llm_graph_input_posllm_graph_input_pos)
    - [`llm_graph_input_pos::set_input`](llama-graph.cpp.driver.md#llm_graph_input_posset_input)
- **Inherits From**:
    - [`llm_graph_input_i`](#llm_graph_input_i)

**Methods**

---
#### llm\_graph\_input\_pos::llm\_graph\_input\_pos<!-- {{#callable:llm_graph_input_pos::llm_graph_input_pos}} -->
The `llm_graph_input_pos` constructor initializes an object with a specified number of positions per embedding.
- **Inputs**:
    - `n_pos_per_embd`: An integer representing the number of positions per embedding to be used in the graph input.
- **Control Flow**:
    - The constructor initializes the member variable `n_pos_per_embd` with the provided argument value.
- **Output**: An instance of the `llm_graph_input_pos` class with the `n_pos_per_embd` member variable set.
- **See also**: [`llm_graph_input_pos`](#llm_graph_input_pos)  (Data Structure)


---
#### llm\_graph\_input\_pos::\~llm\_graph\_input\_pos<!-- {{#callable:llm_graph_input_pos::~llm_graph_input_pos}} -->
The destructor `~llm_graph_input_pos()` is a virtual default destructor for the `llm_graph_input_pos` class.
- **Inputs**: None
- **Control Flow**:
    - The destructor is declared as virtual, ensuring that the correct destructor is called for derived classes when an object is deleted through a base class pointer.
    - The destructor is defined as default, indicating that the compiler should generate the default implementation for it.
- **Output**: The destructor does not return any value or output.
- **See also**: [`llm_graph_input_pos`](#llm_graph_input_pos)  (Data Structure)



---
### llm\_graph\_input\_attn\_temp<!-- {{#data_structure:llm_graph_input_attn_temp}} -->
- **Type**: `class`
- **Members**:
    - `attn_scale`: A pointer to a ggml_tensor representing the attention scale, initialized to nullptr.
    - `n_attn_temp_floor_scale`: A constant unsigned 32-bit integer representing the floor scale for attention temperature.
    - `f_attn_temp_scale`: A constant float representing the scale factor for attention temperature.
- **Description**: The `llm_graph_input_attn_temp` class is a specialized input class for handling attention temperature tuning in a graph-based model, specifically used by llama4. It inherits from `llm_graph_input_i` and provides a mechanism to set input data for attention scaling. The class contains a pointer to a ggml_tensor for attention scaling and two constant parameters that define the scaling behavior: `n_attn_temp_floor_scale` and `f_attn_temp_scale`. These parameters are used to compute the attention scale based on the position of tokens in a batch, allowing for dynamic adjustment of attention temperature during model execution.
- **Member Functions**:
    - [`llm_graph_input_attn_temp::llm_graph_input_attn_temp`](#llm_graph_input_attn_templlm_graph_input_attn_temp)
    - [`llm_graph_input_attn_temp::~llm_graph_input_attn_temp`](#llm_graph_input_attn_templlm_graph_input_attn_temp)
    - [`llm_graph_input_attn_temp::set_input`](llama-graph.cpp.driver.md#llm_graph_input_attn_tempset_input)
- **Inherits From**:
    - [`llm_graph_input_i`](#llm_graph_input_i)

**Methods**

---
#### llm\_graph\_input\_attn\_temp::llm\_graph\_input\_attn\_temp<!-- {{#callable:llm_graph_input_attn_temp::llm_graph_input_attn_temp}} -->
The `llm_graph_input_attn_temp` constructor initializes an object with attention temperature scaling parameters.
- **Inputs**:
    - `n_attn_temp_floor_scale`: A `uint32_t` representing the floor scale for attention temperature.
    - `f_attn_temp_scale`: A `float` representing the scale factor for attention temperature.
- **Control Flow**:
    - The constructor initializes the member variables `n_attn_temp_floor_scale` and `f_attn_temp_scale` with the provided arguments.
- **Output**: An instance of the `llm_graph_input_attn_temp` class with initialized attention temperature scaling parameters.
- **See also**: [`llm_graph_input_attn_temp`](#llm_graph_input_attn_temp)  (Data Structure)


---
#### llm\_graph\_input\_attn\_temp::\~llm\_graph\_input\_attn\_temp<!-- {{#callable:llm_graph_input_attn_temp::~llm_graph_input_attn_temp}} -->
The destructor `~llm_graph_input_attn_temp` is a virtual default destructor for the `llm_graph_input_attn_temp` class, ensuring proper cleanup of derived class objects.
- **Inputs**: None
- **Control Flow**:
    - The destructor is declared as virtual to allow derived class destructors to be called correctly when an object is deleted through a base class pointer.
    - The destructor is defined as default, indicating that the compiler should generate the default implementation for it.
- **Output**: The destructor does not return any value or output.
- **See also**: [`llm_graph_input_attn_temp`](#llm_graph_input_attn_temp)  (Data Structure)



---
### llm\_graph\_input\_pos\_bucket<!-- {{#data_structure:llm_graph_input_pos_bucket}} -->
- **Type**: `class`
- **Members**:
    - `pos_bucket`: A pointer to a ggml_tensor representing a 2D integer tensor with dimensions [n_batch, n_batch].
    - `hparams`: A constant reference to a llama_hparams object, storing hyperparameters for the class.
- **Description**: The `llm_graph_input_pos_bucket` class is a specialized input handler for a graph-based model, inheriting from `llm_graph_input_i`. It is designed to manage position bucket inputs, which are represented as a 2D tensor (`pos_bucket`) that holds integer values. The class is initialized with a set of hyperparameters (`hparams`) and provides a method to set input data (`set_input`) using a `llama_ubatch` object. This class is part of a larger framework for handling various types of inputs in a machine learning model, particularly focusing on relative position encoding.
- **Member Functions**:
    - [`llm_graph_input_pos_bucket::llm_graph_input_pos_bucket`](#llm_graph_input_pos_bucketllm_graph_input_pos_bucket)
    - [`llm_graph_input_pos_bucket::~llm_graph_input_pos_bucket`](#llm_graph_input_pos_bucketllm_graph_input_pos_bucket)
    - [`llm_graph_input_pos_bucket::set_input`](llama-graph.cpp.driver.md#llm_graph_input_pos_bucketset_input)
- **Inherits From**:
    - [`llm_graph_input_i`](#llm_graph_input_i)

**Methods**

---
#### llm\_graph\_input\_pos\_bucket::llm\_graph\_input\_pos\_bucket<!-- {{#callable:llm_graph_input_pos_bucket::llm_graph_input_pos_bucket}} -->
The `llm_graph_input_pos_bucket` constructor initializes an instance of the class with given hyperparameters.
- **Inputs**:
    - `hparams`: A constant reference to a `llama_hparams` object, which contains hyperparameters for the model.
- **Control Flow**:
    - The constructor initializes the `hparams` member variable with the provided `hparams` argument.
    - The constructor does not perform any additional operations or logic.
- **Output**: An instance of the `llm_graph_input_pos_bucket` class is created with the specified hyperparameters.
- **See also**: [`llm_graph_input_pos_bucket`](#llm_graph_input_pos_bucket)  (Data Structure)


---
#### llm\_graph\_input\_pos\_bucket::\~llm\_graph\_input\_pos\_bucket<!-- {{#callable:llm_graph_input_pos_bucket::~llm_graph_input_pos_bucket}} -->
The destructor `~llm_graph_input_pos_bucket()` is a virtual default destructor for the `llm_graph_input_pos_bucket` class, ensuring proper cleanup of derived class objects.
- **Inputs**: None
- **Control Flow**:
    - The destructor is declared as virtual to allow derived class destructors to be called correctly when an object is deleted through a base class pointer.
    - The destructor is defined as default, indicating that the compiler should generate the default implementation for it.
- **Output**: The function does not produce any output as it is a destructor.
- **See also**: [`llm_graph_input_pos_bucket`](#llm_graph_input_pos_bucket)  (Data Structure)



---
### llm\_graph\_input\_pos\_bucket\_kv<!-- {{#data_structure:llm_graph_input_pos_bucket_kv}} -->
- **Type**: `class`
- **Members**:
    - `pos_bucket`: A pointer to a ggml_tensor representing position buckets, initialized to nullptr.
    - `hparams`: A constant reference to llama_hparams, storing hyperparameters for the model.
    - `kv_state`: A pointer to a llama_kv_cache_unified_state, representing the key-value cache state.
- **Description**: The `llm_graph_input_pos_bucket_kv` class is a specialized input class for handling position buckets in a graph input, particularly when working with key-value cache states. It inherits from `llm_graph_input_i` and is designed to manage position-related data in a neural network graph, using a position bucket tensor and references to hyperparameters and key-value cache states. The class provides a method to set input data, which interacts with the key-value cache state to update the position bucket based on the input batch.
- **Member Functions**:
    - [`llm_graph_input_pos_bucket_kv::llm_graph_input_pos_bucket_kv`](#llm_graph_input_pos_bucket_kvllm_graph_input_pos_bucket_kv)
    - [`llm_graph_input_pos_bucket_kv::~llm_graph_input_pos_bucket_kv`](#llm_graph_input_pos_bucket_kvllm_graph_input_pos_bucket_kv)
    - [`llm_graph_input_pos_bucket_kv::set_input`](llama-graph.cpp.driver.md#llm_graph_input_pos_bucket_kvset_input)
- **Inherits From**:
    - [`llm_graph_input_i`](#llm_graph_input_i)

**Methods**

---
#### llm\_graph\_input\_pos\_bucket\_kv::llm\_graph\_input\_pos\_bucket\_kv<!-- {{#callable:llm_graph_input_pos_bucket_kv::llm_graph_input_pos_bucket_kv}} -->
The `llm_graph_input_pos_bucket_kv` constructor initializes an object with hyperparameters and a key-value cache state for position bucket input in a graph.
- **Inputs**:
    - `hparams`: A constant reference to a `llama_hparams` object, which contains hyperparameters for the model.
    - `kv_state`: A pointer to a `llama_kv_cache_unified_state` object, representing the key-value cache state.
- **Control Flow**:
    - The constructor initializes the `hparams` member with the provided `hparams` argument.
    - The constructor initializes the `kv_state` member with the provided `kv_state` argument.
- **Output**: An instance of the `llm_graph_input_pos_bucket_kv` class is created with initialized hyperparameters and key-value cache state.
- **See also**: [`llm_graph_input_pos_bucket_kv`](#llm_graph_input_pos_bucket_kv)  (Data Structure)


---
#### llm\_graph\_input\_pos\_bucket\_kv::\~llm\_graph\_input\_pos\_bucket\_kv<!-- {{#callable:llm_graph_input_pos_bucket_kv::~llm_graph_input_pos_bucket_kv}} -->
The destructor `~llm_graph_input_pos_bucket_kv()` is a virtual default destructor for the `llm_graph_input_pos_bucket_kv` class, ensuring proper cleanup of derived class objects.
- **Inputs**: None
- **Control Flow**:
    - The destructor is declared as virtual to allow derived class destructors to be called correctly when an object is deleted through a base class pointer.
    - The destructor is defined as default, indicating that the compiler should generate the default implementation, which typically involves deallocating any resources owned by the class.
- **Output**: The destructor does not return any value or output.
- **See also**: [`llm_graph_input_pos_bucket_kv`](#llm_graph_input_pos_bucket_kv)  (Data Structure)



---
### llm\_graph\_input\_out\_ids<!-- {{#data_structure:llm_graph_input_out_ids}} -->
- **Type**: `class`
- **Members**:
    - `out_ids`: A pointer to a ggml_tensor representing output IDs, with a size of n_outputs.
    - `hparams`: A constant reference to llama_hparams, holding hyperparameters for the model.
    - `cparams`: A constant reference to llama_cparams, holding configuration parameters for the model.
    - `n_outputs`: A constant integer representing the number of outputs.
- **Description**: The `llm_graph_input_out_ids` class is a specialized input handler for a graph-based model, inheriting from `llm_graph_input_i`. It manages output IDs for a model, using hyperparameters and configuration parameters to determine the number of outputs. The class includes a method to set input data, which processes a batch of tokens and populates the output IDs tensor based on the model's requirements and the batch's characteristics.
- **Member Functions**:
    - [`llm_graph_input_out_ids::llm_graph_input_out_ids`](#llm_graph_input_out_idsllm_graph_input_out_ids)
    - [`llm_graph_input_out_ids::~llm_graph_input_out_ids`](#llm_graph_input_out_idsllm_graph_input_out_ids)
    - [`llm_graph_input_out_ids::set_input`](llama-graph.cpp.driver.md#llm_graph_input_out_idsset_input)
- **Inherits From**:
    - [`llm_graph_input_i`](#llm_graph_input_i)

**Methods**

---
#### llm\_graph\_input\_out\_ids::llm\_graph\_input\_out\_ids<!-- {{#callable:llm_graph_input_out_ids::llm_graph_input_out_ids}} -->
The `llm_graph_input_out_ids` constructor initializes an object with hyperparameters, compute parameters, and a specified number of output IDs.
- **Inputs**:
    - `hparams`: A constant reference to a `llama_hparams` object, which contains hyperparameters for the model.
    - `cparams`: A constant reference to a `llama_cparams` object, which contains compute parameters for the model.
    - `n_outputs`: An integer specifying the number of output IDs.
- **Control Flow**:
    - The constructor initializes the member variables `hparams`, `cparams`, and `n_outputs` with the provided arguments.
- **Output**: An instance of the `llm_graph_input_out_ids` class is created with initialized member variables.
- **See also**: [`llm_graph_input_out_ids`](#llm_graph_input_out_ids)  (Data Structure)


---
#### llm\_graph\_input\_out\_ids::\~llm\_graph\_input\_out\_ids<!-- {{#callable:llm_graph_input_out_ids::~llm_graph_input_out_ids}} -->
The destructor `~llm_graph_input_out_ids()` is a default virtual destructor for the `llm_graph_input_out_ids` class.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as a virtual function, ensuring that derived class destructors are called correctly when an object is deleted through a base class pointer.
    - The destructor is marked as `default`, indicating that the compiler should generate the default implementation.
- **Output**: There is no output from this destructor; it is used for cleanup when an object of `llm_graph_input_out_ids` is destroyed.
- **See also**: [`llm_graph_input_out_ids`](#llm_graph_input_out_ids)  (Data Structure)



---
### llm\_graph\_input\_mean<!-- {{#data_structure:llm_graph_input_mean}} -->
- **Type**: `class`
- **Members**:
    - `mean`: A pointer to a ggml_tensor representing the mean, with dimensions [n_batch, n_batch].
    - `cparams`: A constant reference to a llama_cparams object, used for configuration parameters.
- **Description**: The `llm_graph_input_mean` class is a specialized implementation of the `llm_graph_input_i` interface, designed to handle input data for a graph by computing the mean of input sequences. It contains a tensor `mean` to store the computed mean values and uses configuration parameters from `cparams` to determine how the input should be processed, particularly when the pooling type is set to mean. The class provides a method `set_input` to populate the `mean` tensor based on the input batch data.
- **Member Functions**:
    - [`llm_graph_input_mean::llm_graph_input_mean`](#llm_graph_input_meanllm_graph_input_mean)
    - [`llm_graph_input_mean::~llm_graph_input_mean`](#llm_graph_input_meanllm_graph_input_mean)
    - [`llm_graph_input_mean::set_input`](llama-graph.cpp.driver.md#llm_graph_input_meanset_input)
- **Inherits From**:
    - [`llm_graph_input_i`](#llm_graph_input_i)

**Methods**

---
#### llm\_graph\_input\_mean::llm\_graph\_input\_mean<!-- {{#callable:llm_graph_input_mean::llm_graph_input_mean}} -->
The `llm_graph_input_mean` constructor initializes an instance of the class with given configuration parameters.
- **Inputs**:
    - `cparams`: A constant reference to a `llama_cparams` object, which contains configuration parameters for the graph input.
- **Control Flow**:
    - The constructor initializes the `cparams` member variable with the provided `cparams` argument.
    - The constructor does not perform any additional operations beyond member initialization.
- **Output**: An instance of the `llm_graph_input_mean` class is created with the specified configuration parameters.
- **See also**: [`llm_graph_input_mean`](#llm_graph_input_mean)  (Data Structure)


---
#### llm\_graph\_input\_mean::\~llm\_graph\_input\_mean<!-- {{#callable:llm_graph_input_mean::~llm_graph_input_mean}} -->
The destructor `~llm_graph_input_mean` is a virtual default destructor for the `llm_graph_input_mean` class.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as a virtual function, allowing for proper cleanup in derived classes if any.
    - The destructor is defaulted, meaning it uses the compiler-generated default implementation.
- **Output**: There is no output from a destructor; it is used for cleanup when an object is destroyed.
- **See also**: [`llm_graph_input_mean`](#llm_graph_input_mean)  (Data Structure)



---
### llm\_graph\_input\_cls<!-- {{#data_structure:llm_graph_input_cls}} -->
- **Type**: `class`
- **Members**:
    - `cls`: A pointer to a ggml_tensor representing a tensor of integers with dimensions [n_batch].
    - `cparams`: A constant reference to a llama_cparams object, which holds configuration parameters for the class.
- **Description**: The `llm_graph_input_cls` class is a specialized implementation of the `llm_graph_input_i` interface, designed to handle input data for a graph in a machine learning model. It primarily manages a tensor (`cls`) that is used for classification purposes, with its behavior influenced by the configuration parameters (`cparams`). The class provides a method to set input data, which processes the input batch (`ubatch`) based on the specified pooling type and updates the `cls` tensor accordingly. This class is part of a larger framework for handling various types of input data in a machine learning context.
- **Member Functions**:
    - [`llm_graph_input_cls::llm_graph_input_cls`](#llm_graph_input_clsllm_graph_input_cls)
    - [`llm_graph_input_cls::~llm_graph_input_cls`](#llm_graph_input_clsllm_graph_input_cls)
    - [`llm_graph_input_cls::set_input`](llama-graph.cpp.driver.md#llm_graph_input_clsset_input)
- **Inherits From**:
    - [`llm_graph_input_i`](#llm_graph_input_i)

**Methods**

---
#### llm\_graph\_input\_cls::llm\_graph\_input\_cls<!-- {{#callable:llm_graph_input_cls::llm_graph_input_cls}} -->
The `llm_graph_input_cls` constructor initializes an instance of the class with given configuration parameters.
- **Inputs**:
    - `cparams`: A constant reference to a `llama_cparams` object, which contains configuration parameters for the class.
- **Control Flow**:
    - The constructor initializes the `cparams` member variable with the provided `cparams` argument.
- **Output**: An instance of the `llm_graph_input_cls` class is created with the specified configuration parameters.
- **See also**: [`llm_graph_input_cls`](#llm_graph_input_cls)  (Data Structure)


---
#### llm\_graph\_input\_cls::\~llm\_graph\_input\_cls<!-- {{#callable:llm_graph_input_cls::~llm_graph_input_cls}} -->
The destructor `~llm_graph_input_cls` is a virtual default destructor for the `llm_graph_input_cls` class, ensuring proper cleanup of derived class objects.
- **Inputs**: None
- **Control Flow**:
    - The destructor is declared as virtual to allow derived class destructors to be called correctly when an object is deleted through a base class pointer.
    - The destructor is defined as default, indicating that the compiler should generate the default implementation for it.
- **Output**: There is no explicit output from the destructor; it ensures proper resource cleanup when an object of `llm_graph_input_cls` or its derived class is destroyed.
- **See also**: [`llm_graph_input_cls`](#llm_graph_input_cls)  (Data Structure)



---
### llm\_graph\_input\_s\_copy<!-- {{#data_structure:llm_graph_input_s_copy}} -->
- **Type**: `class`
- **Members**:
    - `s_copy`: A pointer to a ggml_tensor representing a copy of state data, with an integer data type and size defined by kv_size.
    - `kv_state`: A constant pointer to a llama_kv_cache_recurrent_state object, representing the key-value cache state.
- **Description**: The `llm_graph_input_s_copy` class is a specialized input handler for a graph, inheriting from `llm_graph_input_i`. It is designed to manage and manipulate a copy of state data (`s_copy`) using a key-value cache state (`kv_state`). The class provides a method to set input data, which involves copying data from the key-value cache state into the `s_copy` tensor, ensuring that the data is stored in host memory. This class is part of a larger framework for handling various types of graph inputs in a machine learning model.
- **Member Functions**:
    - [`llm_graph_input_s_copy::llm_graph_input_s_copy`](#llm_graph_input_s_copyllm_graph_input_s_copy)
    - [`llm_graph_input_s_copy::~llm_graph_input_s_copy`](#llm_graph_input_s_copyllm_graph_input_s_copy)
    - [`llm_graph_input_s_copy::set_input`](llama-graph.cpp.driver.md#llm_graph_input_s_copyset_input)
- **Inherits From**:
    - [`llm_graph_input_i`](#llm_graph_input_i)

**Methods**

---
#### llm\_graph\_input\_s\_copy::llm\_graph\_input\_s\_copy<!-- {{#callable:llm_graph_input_s_copy::llm_graph_input_s_copy}} -->
The `llm_graph_input_s_copy` constructor initializes an instance of the class with a given key-value cache recurrent state.
- **Inputs**:
    - `kv_state`: A pointer to a `llama_kv_cache_recurrent_state` object, which represents the key-value cache recurrent state to be used by the instance.
- **Control Flow**:
    - The constructor initializes the `kv_state` member variable with the provided `kv_state` argument.
- **Output**: An instance of the `llm_graph_input_s_copy` class is created with the specified key-value cache recurrent state.
- **See also**: [`llm_graph_input_s_copy`](#llm_graph_input_s_copy)  (Data Structure)


---
#### llm\_graph\_input\_s\_copy::\~llm\_graph\_input\_s\_copy<!-- {{#callable:llm_graph_input_s_copy::~llm_graph_input_s_copy}} -->
The destructor `~llm_graph_input_s_copy()` is a default virtual destructor for the `llm_graph_input_s_copy` class, ensuring proper cleanup of derived class objects.
- **Inputs**: None
- **Control Flow**:
    - The destructor is declared as virtual to allow derived class destructors to be called correctly when an object is deleted through a base class pointer.
    - The destructor is defined as default, indicating that the compiler should generate the default implementation, which typically involves deallocating any resources owned by the object.
- **Output**: The destructor does not return any value or output.
- **See also**: [`llm_graph_input_s_copy`](#llm_graph_input_s_copy)  (Data Structure)



---
### llm\_graph\_input\_s\_mask<!-- {{#data_structure:llm_graph_input_s_mask}} -->
- **Type**: `class`
- **Members**:
    - `s_mask`: A pointer to a ggml_tensor representing a mask with dimensions F32 [1, n_kv].
    - `kv_state`: A constant pointer to a llama_kv_cache_recurrent_state object.
- **Description**: The `llm_graph_input_s_mask` class is a specialized input class for handling a mask tensor in a graph input context, inheriting from `llm_graph_input_i`. It is designed to work with a recurrent key-value cache state (`kv_state`) and manages a mask tensor (`s_mask`) that is used to clear unused states based on the number of key-value pairs (`n_kv`). The class provides a method to set the input, which updates the mask tensor according to the current state of the key-value cache.
- **Member Functions**:
    - [`llm_graph_input_s_mask::llm_graph_input_s_mask`](#llm_graph_input_s_maskllm_graph_input_s_mask)
    - [`llm_graph_input_s_mask::~llm_graph_input_s_mask`](#llm_graph_input_s_maskllm_graph_input_s_mask)
    - [`llm_graph_input_s_mask::set_input`](llama-graph.cpp.driver.md#llm_graph_input_s_maskset_input)
- **Inherits From**:
    - [`llm_graph_input_i`](#llm_graph_input_i)

**Methods**

---
#### llm\_graph\_input\_s\_mask::llm\_graph\_input\_s\_mask<!-- {{#callable:llm_graph_input_s_mask::llm_graph_input_s_mask}} -->
The `llm_graph_input_s_mask` constructor initializes an instance of the class with a given key-value cache recurrent state.
- **Inputs**:
    - `kv_state`: A pointer to a `llama_kv_cache_recurrent_state` object, which represents the key-value cache recurrent state to be used by the instance.
- **Control Flow**:
    - The constructor takes a single argument, `kv_state`, which is a pointer to a `llama_kv_cache_recurrent_state` object.
    - The constructor initializes the member variable `kv_state` with the provided argument.
- **Output**: An instance of the `llm_graph_input_s_mask` class is created with its `kv_state` member initialized to the provided argument.
- **See also**: [`llm_graph_input_s_mask`](#llm_graph_input_s_mask)  (Data Structure)


---
#### llm\_graph\_input\_s\_mask::\~llm\_graph\_input\_s\_mask<!-- {{#callable:llm_graph_input_s_mask::~llm_graph_input_s_mask}} -->
The destructor `~llm_graph_input_s_mask()` is a default virtual destructor for the `llm_graph_input_s_mask` class, ensuring proper cleanup of derived class objects.
- **Inputs**: None
- **Control Flow**:
    - The destructor is declared as virtual to allow derived class destructors to be called correctly when an object is deleted through a base class pointer.
    - The destructor is defined as default, indicating that the compiler should generate the default implementation for it.
- **Output**: The destructor does not return any value or output.
- **See also**: [`llm_graph_input_s_mask`](#llm_graph_input_s_mask)  (Data Structure)



---
### llm\_graph\_input\_cross\_embd<!-- {{#data_structure:llm_graph_input_cross_embd}} -->
- **Type**: `class`
- **Members**:
    - `cross_embd`: A pointer to a ggml_tensor representing the cross embeddings with dimensions [n_embd, n_outputs_enc].
    - `cross`: A constant pointer to a llama_cross structure, which contains data for cross-attention.
- **Description**: The `llm_graph_input_cross_embd` class is a specialized input class for handling cross embeddings in a graph-based model. It inherits from `llm_graph_input_i` and is designed to manage the input of cross embeddings, which are used in multi-modal models to facilitate cross-attention mechanisms. The class holds a tensor for cross embeddings and a reference to a `llama_cross` structure, which provides necessary data for constructing cross-attention masks. The `set_input` method is overridden to set the input data for the cross embeddings.
- **Member Functions**:
    - [`llm_graph_input_cross_embd::llm_graph_input_cross_embd`](#llm_graph_input_cross_embdllm_graph_input_cross_embd)
    - [`llm_graph_input_cross_embd::~llm_graph_input_cross_embd`](#llm_graph_input_cross_embdllm_graph_input_cross_embd)
    - [`llm_graph_input_cross_embd::set_input`](llama-graph.cpp.driver.md#llm_graph_input_cross_embdset_input)
- **Inherits From**:
    - [`llm_graph_input_i`](#llm_graph_input_i)

**Methods**

---
#### llm\_graph\_input\_cross\_embd::llm\_graph\_input\_cross\_embd<!-- {{#callable:llm_graph_input_cross_embd::llm_graph_input_cross_embd}} -->
The `llm_graph_input_cross_embd` constructor initializes an object with a reference to a `llama_cross` structure, which contains encoder output embeddings.
- **Inputs**:
    - `cross`: A pointer to a `llama_cross` structure, which holds encoder output embeddings and related data.
- **Control Flow**:
    - The constructor initializes the `cross` member variable with the provided `llama_cross` pointer.
- **Output**: An instance of the `llm_graph_input_cross_embd` class is created with the `cross` member variable set to the provided `llama_cross` pointer.
- **See also**: [`llm_graph_input_cross_embd`](#llm_graph_input_cross_embd)  (Data Structure)


---
#### llm\_graph\_input\_cross\_embd::\~llm\_graph\_input\_cross\_embd<!-- {{#callable:llm_graph_input_cross_embd::~llm_graph_input_cross_embd}} -->
The destructor `~llm_graph_input_cross_embd` is a virtual default destructor for the `llm_graph_input_cross_embd` class.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as a virtual function, ensuring that derived class destructors are called correctly when an object is deleted through a base class pointer.
    - The destructor is marked as `default`, indicating that the compiler should generate the default implementation.
- **Output**: The destructor does not produce any output; it is used for cleanup when an object of `llm_graph_input_cross_embd` is destroyed.
- **See also**: [`llm_graph_input_cross_embd`](#llm_graph_input_cross_embd)  (Data Structure)



---
### llm\_graph\_input\_attn\_no\_cache<!-- {{#data_structure:llm_graph_input_attn_no_cache}} -->
- **Type**: `class`
- **Members**:
    - `kq_mask`: A pointer to a ggml_tensor representing the key-query mask with dimensions [n_tokens, n_batch].
    - `kq_mask_cnv`: A pointer to a ggml_tensor representing the converted key-query mask with dimensions [n_tokens, n_batch].
    - `hparams`: A reference to a llama_hparams object containing hyperparameters.
    - `cparams`: A reference to a llama_cparams object containing configuration parameters.
- **Description**: The `llm_graph_input_attn_no_cache` class is a specialized implementation of the `llm_graph_input_i` interface, designed to handle attention input without caching in a machine learning model. It manages key-query masks for attention mechanisms, specifically for scenarios where caching is not utilized. The class holds references to hyperparameters and configuration parameters, and provides functionality to set input data and retrieve the converted key-query mask tensor.
- **Member Functions**:
    - [`llm_graph_input_attn_no_cache::llm_graph_input_attn_no_cache`](#llm_graph_input_attn_no_cachellm_graph_input_attn_no_cache)
    - [`llm_graph_input_attn_no_cache::~llm_graph_input_attn_no_cache`](#llm_graph_input_attn_no_cachellm_graph_input_attn_no_cache)
    - [`llm_graph_input_attn_no_cache::get_kq_mask`](#llm_graph_input_attn_no_cacheget_kq_mask)
    - [`llm_graph_input_attn_no_cache::set_input`](llama-graph.cpp.driver.md#llm_graph_input_attn_no_cacheset_input)
- **Inherits From**:
    - [`llm_graph_input_i`](#llm_graph_input_i)

**Methods**

---
#### llm\_graph\_input\_attn\_no\_cache::llm\_graph\_input\_attn\_no\_cache<!-- {{#callable:llm_graph_input_attn_no_cache::llm_graph_input_attn_no_cache}} -->
The `llm_graph_input_attn_no_cache` constructor initializes an object with hyperparameters and configuration parameters for a graph input without caching.
- **Inputs**:
    - `hparams`: A constant reference to a `llama_hparams` object, which contains hyperparameters for the model.
    - `cparams`: A constant reference to a `llama_cparams` object, which contains configuration parameters for the model.
- **Control Flow**:
    - The constructor initializes the `hparams` and `cparams` member variables with the provided arguments.
    - The destructor is defined as default, indicating no special cleanup is required.
- **Output**: An instance of the `llm_graph_input_attn_no_cache` class is created with initialized hyperparameters and configuration parameters.
- **See also**: [`llm_graph_input_attn_no_cache`](#llm_graph_input_attn_no_cache)  (Data Structure)


---
#### llm\_graph\_input\_attn\_no\_cache::\~llm\_graph\_input\_attn\_no\_cache<!-- {{#callable:llm_graph_input_attn_no_cache::~llm_graph_input_attn_no_cache}} -->
The destructor `~llm_graph_input_attn_no_cache()` is a default destructor for the `llm_graph_input_attn_no_cache` class, which performs no specific actions upon object destruction.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as `default`, meaning it relies on the compiler-generated default behavior for destructors.
    - No custom cleanup or resource deallocation is performed in this destructor.
- **Output**: There is no output from this destructor as it performs no operations.
- **See also**: [`llm_graph_input_attn_no_cache`](#llm_graph_input_attn_no_cache)  (Data Structure)


---
#### llm\_graph\_input\_attn\_no\_cache::get\_kq\_mask<!-- {{#callable:llm_graph_input_attn_no_cache::get_kq_mask}} -->
The `get_kq_mask` function returns the `kq_mask_cnv` tensor from the `llm_graph_input_attn_no_cache` class.
- **Inputs**: None
- **Control Flow**:
    - The function is a simple getter method that directly returns the `kq_mask_cnv` member variable of the class `llm_graph_input_attn_no_cache`.
- **Output**: A pointer to a `ggml_tensor`, specifically the `kq_mask_cnv` tensor.
- **See also**: [`llm_graph_input_attn_no_cache`](#llm_graph_input_attn_no_cache)  (Data Structure)



---
### llm\_graph\_input\_attn\_kv\_unified<!-- {{#data_structure:llm_graph_input_attn_kv_unified}} -->
- **Type**: `class`
- **Members**:
    - `self_kq_mask`: A pointer to a ggml_tensor representing the self-attention key-query mask with dimensions [n_kv, n_batch].
    - `self_kq_mask_cnv`: A pointer to a ggml_tensor representing the converted self-attention key-query mask with dimensions [n_kv, n_batch].
    - `hparams`: A reference to a llama_hparams object containing hyperparameters.
    - `cparams`: A reference to a llama_cparams object containing configuration parameters.
    - `kv_state`: A pointer to a llama_kv_cache_unified_state object representing the key-value cache state.
- **Description**: The `llm_graph_input_attn_kv_unified` class is a specialized input handler for attention mechanisms in a graph-based model, inheriting from `llm_graph_input_i`. It manages key-query masks for self-attention, utilizing a unified key-value cache state. The class holds references to hyperparameters and configuration parameters, and provides functionality to set input data and retrieve the converted key-query mask.
- **Member Functions**:
    - [`llm_graph_input_attn_kv_unified::llm_graph_input_attn_kv_unified`](#llm_graph_input_attn_kv_unifiedllm_graph_input_attn_kv_unified)
    - [`llm_graph_input_attn_kv_unified::~llm_graph_input_attn_kv_unified`](#llm_graph_input_attn_kv_unifiedllm_graph_input_attn_kv_unified)
    - [`llm_graph_input_attn_kv_unified::get_kq_mask`](#llm_graph_input_attn_kv_unifiedget_kq_mask)
    - [`llm_graph_input_attn_kv_unified::set_input`](llama-graph.cpp.driver.md#llm_graph_input_attn_kv_unifiedset_input)
- **Inherits From**:
    - [`llm_graph_input_i`](#llm_graph_input_i)

**Methods**

---
#### llm\_graph\_input\_attn\_kv\_unified::llm\_graph\_input\_attn\_kv\_unified<!-- {{#callable:llm_graph_input_attn_kv_unified::llm_graph_input_attn_kv_unified}} -->
The `llm_graph_input_attn_kv_unified` constructor initializes an object with hyperparameters, configuration parameters, and a key-value cache state for attention mechanisms in a graph input context.
- **Inputs**:
    - `hparams`: A constant reference to a `llama_hparams` object, which contains hyperparameters for the model.
    - `cparams`: A constant reference to a `llama_cparams` object, which contains configuration parameters for the model.
    - `kv_state`: A pointer to a `llama_kv_cache_unified_state` object, representing the key-value cache state used in the attention mechanism.
- **Control Flow**:
    - The constructor initializes the member variables `hparams`, `cparams`, and `kv_state` with the provided arguments.
    - The constructor does not perform any additional operations beyond member initialization.
- **Output**: The function does not return any value; it is a constructor for initializing an object of the `llm_graph_input_attn_kv_unified` class.
- **See also**: [`llm_graph_input_attn_kv_unified`](#llm_graph_input_attn_kv_unified)  (Data Structure)


---
#### llm\_graph\_input\_attn\_kv\_unified::\~llm\_graph\_input\_attn\_kv\_unified<!-- {{#callable:llm_graph_input_attn_kv_unified::~llm_graph_input_attn_kv_unified}} -->
The destructor `~llm_graph_input_attn_kv_unified()` is a default destructor for the `llm_graph_input_attn_kv_unified` class, which performs no specific cleanup operations.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as default, indicating it does not perform any custom operations and relies on the compiler-generated destructor.
    - It is automatically called when an object of `llm_graph_input_attn_kv_unified` is destroyed, ensuring proper cleanup of resources, if any, managed by the class.
- **Output**: The destructor does not produce any output or perform any specific actions.
- **See also**: [`llm_graph_input_attn_kv_unified`](#llm_graph_input_attn_kv_unified)  (Data Structure)


---
#### llm\_graph\_input\_attn\_kv\_unified::get\_kq\_mask<!-- {{#callable:llm_graph_input_attn_kv_unified::get_kq_mask}} -->
The `get_kq_mask` function returns a pointer to the `self_kq_mask_cnv` tensor, which represents a key-query mask in the context of attention mechanisms.
- **Inputs**: None
- **Control Flow**:
    - The function is a simple getter that directly returns the `self_kq_mask_cnv` member variable of the class `llm_graph_input_attn_kv_unified`.
- **Output**: A pointer to a `ggml_tensor` object, specifically the `self_kq_mask_cnv` tensor.
- **See also**: [`llm_graph_input_attn_kv_unified`](#llm_graph_input_attn_kv_unified)  (Data Structure)



---
### llm\_graph\_input\_attn\_kv\_unified\_iswa<!-- {{#data_structure:llm_graph_input_attn_kv_unified_iswa}} -->
- **Type**: `class`
- **Members**:
    - `self_kq_mask`: A pointer to a ggml_tensor representing the self key-query mask with dimensions [n_kv, n_batch].
    - `self_kq_mask_cnv`: A pointer to a ggml_tensor representing the converted self key-query mask with dimensions [n_kv, n_batch].
    - `self_kq_mask_swa`: A pointer to a ggml_tensor representing the self key-query mask for SWA with dimensions [n_kv, n_batch].
    - `self_kq_mask_swa_cnv`: A pointer to a ggml_tensor representing the converted self key-query mask for SWA with dimensions [n_kv, n_batch].
    - `hparams`: A reference to llama_hparams, holding hyperparameters for the model.
    - `cparams`: A reference to llama_cparams, holding configuration parameters for the model.
    - `kv_state`: A pointer to llama_kv_cache_unified_iswa_state, representing the state of the key-value cache for unified ISWA.
- **Description**: The `llm_graph_input_attn_kv_unified_iswa` class is a specialized input handler for attention mechanisms in a neural network graph, specifically designed to work with a unified key-value cache state for ISWA (Incremental Sliding Window Attention). It inherits from `llm_graph_input_i` and manages several tensor pointers that represent different forms of key-query masks, both standard and for SWA (Sliding Window Attention), used in the attention computation. The class also holds references to hyperparameters and configuration parameters, as well as a pointer to the key-value cache state, facilitating the integration of these components into the attention mechanism.
- **Member Functions**:
    - [`llm_graph_input_attn_kv_unified_iswa::llm_graph_input_attn_kv_unified_iswa`](#llm_graph_input_attn_kv_unified_iswallm_graph_input_attn_kv_unified_iswa)
    - [`llm_graph_input_attn_kv_unified_iswa::~llm_graph_input_attn_kv_unified_iswa`](#llm_graph_input_attn_kv_unified_iswallm_graph_input_attn_kv_unified_iswa)
    - [`llm_graph_input_attn_kv_unified_iswa::get_kq_mask`](#llm_graph_input_attn_kv_unified_iswaget_kq_mask)
    - [`llm_graph_input_attn_kv_unified_iswa::get_kq_mask_swa`](#llm_graph_input_attn_kv_unified_iswaget_kq_mask_swa)
    - [`llm_graph_input_attn_kv_unified_iswa::set_input`](llama-graph.cpp.driver.md#llm_graph_input_attn_kv_unified_iswaset_input)
- **Inherits From**:
    - [`llm_graph_input_i`](#llm_graph_input_i)

**Methods**

---
#### llm\_graph\_input\_attn\_kv\_unified\_iswa::llm\_graph\_input\_attn\_kv\_unified\_iswa<!-- {{#callable:llm_graph_input_attn_kv_unified_iswa::llm_graph_input_attn_kv_unified_iswa}} -->
The `llm_graph_input_attn_kv_unified_iswa` constructor initializes an object with hyperparameters, configuration parameters, and a key-value cache state for attention mechanisms in a neural network graph.
- **Inputs**:
    - `hparams`: A reference to a `llama_hparams` object containing hyperparameters for the model.
    - `cparams`: A reference to a `llama_cparams` object containing configuration parameters for the model.
    - `kv_state`: A pointer to a `llama_kv_cache_unified_iswa_state` object representing the key-value cache state for the attention mechanism.
- **Control Flow**:
    - The constructor initializes the member variables `hparams`, `cparams`, and `kv_state` with the provided arguments.
    - The destructor is defined as default, indicating no special cleanup is required.
- **Output**: An instance of the `llm_graph_input_attn_kv_unified_iswa` class is created with initialized member variables.
- **See also**: [`llm_graph_input_attn_kv_unified_iswa`](#llm_graph_input_attn_kv_unified_iswa)  (Data Structure)


---
#### llm\_graph\_input\_attn\_kv\_unified\_iswa::\~llm\_graph\_input\_attn\_kv\_unified\_iswa<!-- {{#callable:llm_graph_input_attn_kv_unified_iswa::~llm_graph_input_attn_kv_unified_iswa}} -->
The destructor `~llm_graph_input_attn_kv_unified_iswa()` is a default destructor for the `llm_graph_input_attn_kv_unified_iswa` class, which performs no specific cleanup operations.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as `default`, indicating that it relies on the compiler-generated default behavior.
    - No custom cleanup or resource deallocation is performed in this destructor.
- **Output**: There is no output from this destructor as it is a default destructor with no custom logic.
- **See also**: [`llm_graph_input_attn_kv_unified_iswa`](#llm_graph_input_attn_kv_unified_iswa)  (Data Structure)


---
#### llm\_graph\_input\_attn\_kv\_unified\_iswa::get\_kq\_mask<!-- {{#callable:llm_graph_input_attn_kv_unified_iswa::get_kq_mask}} -->
The `get_kq_mask` function returns a pointer to the `self_kq_mask_cnv` tensor, which is part of the `llm_graph_input_attn_kv_unified_iswa` class.
- **Inputs**: None
- **Control Flow**:
    - The function is a simple getter method that directly returns the `self_kq_mask_cnv` member variable of the class `llm_graph_input_attn_kv_unified_iswa`.
- **Output**: A pointer to a `ggml_tensor` object, specifically `self_kq_mask_cnv`.
- **See also**: [`llm_graph_input_attn_kv_unified_iswa`](#llm_graph_input_attn_kv_unified_iswa)  (Data Structure)


---
#### llm\_graph\_input\_attn\_kv\_unified\_iswa::get\_kq\_mask\_swa<!-- {{#callable:llm_graph_input_attn_kv_unified_iswa::get_kq_mask_swa}} -->
The `get_kq_mask_swa` function returns a pointer to the `self_kq_mask_swa_cnv` tensor, which is part of the `llm_graph_input_attn_kv_unified_iswa` class.
- **Inputs**: None
- **Control Flow**:
    - The function is a simple getter method that directly returns the `self_kq_mask_swa_cnv` member variable of the class `llm_graph_input_attn_kv_unified_iswa`.
- **Output**: A pointer to a `ggml_tensor`, specifically the `self_kq_mask_swa_cnv` tensor.
- **See also**: [`llm_graph_input_attn_kv_unified_iswa`](#llm_graph_input_attn_kv_unified_iswa)  (Data Structure)



---
### llm\_graph\_input\_attn\_cross<!-- {{#data_structure:llm_graph_input_attn_cross}} -->
- **Type**: `class`
- **Members**:
    - `cross_kq_mask`: A pointer to a ggml_tensor representing the cross-attention mask with dimensions [n_outputs_enc, n_batch].
    - `cross_kq_mask_cnv`: A pointer to a ggml_tensor representing the converted cross-attention mask with dimensions [n_outputs_enc, n_batch].
    - `cross`: A pointer to a llama_cross structure, which contains information necessary for constructing the cross-attention mask.
- **Description**: The `llm_graph_input_attn_cross` class is a specialized input handler for cross-attention mechanisms in a neural network graph, inheriting from `llm_graph_input_i`. It manages cross-attention masks, specifically `cross_kq_mask` and its converted form `cross_kq_mask_cnv`, which are used in the attention mechanism to handle interactions between encoder outputs and decoder inputs. The class is initialized with a `llama_cross` object, which provides the necessary sequence identifiers and embeddings data to construct these masks. This class is part of a larger framework for building and managing graph-based neural network computations, particularly in multi-modal or complex models that require cross-attention.
- **Member Functions**:
    - [`llm_graph_input_attn_cross::llm_graph_input_attn_cross`](#llm_graph_input_attn_crossllm_graph_input_attn_cross)
    - [`llm_graph_input_attn_cross::~llm_graph_input_attn_cross`](#llm_graph_input_attn_crossllm_graph_input_attn_cross)
    - [`llm_graph_input_attn_cross::get_kq_mask_cross`](#llm_graph_input_attn_crossget_kq_mask_cross)
    - [`llm_graph_input_attn_cross::set_input`](llama-graph.cpp.driver.md#llm_graph_input_attn_crossset_input)
- **Inherits From**:
    - [`llm_graph_input_i`](#llm_graph_input_i)

**Methods**

---
#### llm\_graph\_input\_attn\_cross::llm\_graph\_input\_attn\_cross<!-- {{#callable:llm_graph_input_attn_cross::llm_graph_input_attn_cross}} -->
The `llm_graph_input_attn_cross` constructor initializes an object with a reference to a `llama_cross` structure, which is used for cross-attention in a graph input context.
- **Inputs**:
    - `cross`: A pointer to a `llama_cross` structure, which contains data necessary for constructing cross-attention masks in a decoder.
- **Control Flow**:
    - The constructor initializes the `cross` member variable with the provided `llama_cross` pointer.
    - The destructor is defined as default, indicating no special cleanup is required.
- **Output**: An instance of the `llm_graph_input_attn_cross` class, initialized with the given `llama_cross` data.
- **See also**: [`llm_graph_input_attn_cross`](#llm_graph_input_attn_cross)  (Data Structure)


---
#### llm\_graph\_input\_attn\_cross::\~llm\_graph\_input\_attn\_cross<!-- {{#callable:llm_graph_input_attn_cross::~llm_graph_input_attn_cross}} -->
The destructor `~llm_graph_input_attn_cross()` is a default destructor for the `llm_graph_input_attn_cross` class.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as default, meaning it will automatically handle the destruction of the class's resources without any custom logic.
    - It is called when an object of the `llm_graph_input_attn_cross` class is destroyed.
- **Output**: There is no explicit output from the destructor; it ensures proper cleanup of the class's resources.
- **See also**: [`llm_graph_input_attn_cross`](#llm_graph_input_attn_cross)  (Data Structure)


---
#### llm\_graph\_input\_attn\_cross::get\_kq\_mask\_cross<!-- {{#callable:llm_graph_input_attn_cross::get_kq_mask_cross}} -->
The `get_kq_mask_cross` function returns the `cross_kq_mask_cnv` tensor from the `llm_graph_input_attn_cross` class.
- **Inputs**: None
- **Control Flow**:
    - The function is a simple getter method that directly returns the `cross_kq_mask_cnv` member variable of the `llm_graph_input_attn_cross` class.
- **Output**: A pointer to a `ggml_tensor` object, specifically the `cross_kq_mask_cnv` tensor.
- **See also**: [`llm_graph_input_attn_cross`](#llm_graph_input_attn_cross)  (Data Structure)



---
### llm\_graph\_result\_i<!-- {{#data_structure:llm_graph_result_i}} -->
- **Type**: `class`
- **Description**: The `llm_graph_result_i` class is an abstract interface that defines the structure for objects responsible for delivering the results from a graph build process back to a llama context. It includes virtual methods for retrieving various output tensors such as tokens, logits, embeddings, and pooled embeddings, as well as a method for setting input data. This interface is designed to be implemented by concrete classes that manage the specific details of handling these tensors and integrating them into the llama context's workflow.
- **Member Functions**:
    - [`llm_graph_result_i::~llm_graph_result_i`](#llm_graph_result_illm_graph_result_i)

**Methods**

---
#### llm\_graph\_result\_i::\~llm\_graph\_result\_i<!-- {{#callable:llm_graph_result_i::~llm_graph_result_i}} -->
The `~llm_graph_result_i` function is a virtual destructor for the `llm_graph_result_i` class, ensuring proper cleanup of derived class objects.
- **Inputs**: None
- **Control Flow**:
    - The function is defined as a virtual destructor, which means it is intended to be overridden by derived classes.
    - The destructor is marked as `default`, indicating that the compiler should generate the default implementation for it.
- **Output**: The function does not produce any output as it is a destructor; its purpose is to clean up resources when an object is destroyed.
- **See also**: [`llm_graph_result_i`](#llm_graph_result_i)  (Data Structure)



---
### llm\_graph\_result<!-- {{#data_structure:llm_graph_result}} -->
- **Type**: `class`
- **Members**:
    - `t_tokens`: A pointer to a ggml_tensor representing tokens.
    - `t_logits`: A pointer to a ggml_tensor representing logits.
    - `t_embd`: A pointer to a ggml_tensor representing embeddings.
    - `t_embd_pooled`: A pointer to a ggml_tensor representing pooled embeddings.
    - `inputs`: A vector of unique pointers to llm_graph_input_i, representing input nodes for the graph.
- **Description**: The `llm_graph_result` class is a concrete implementation of the `llm_graph_result_i` interface, designed to manage and deliver the results of a graph build process in a machine learning context. It holds pointers to important graph nodes such as tokens, logits, embeddings, and pooled embeddings, which are represented as `ggml_tensor` objects. Additionally, it maintains a collection of input nodes, allowing for dynamic input management through the `add_input` method. This class facilitates the setting of input data via the `set_inputs` method, enabling the population of input tensors with specific data for further processing.
- **Member Functions**:
    - [`llm_graph_result::~llm_graph_result`](#llm_graph_resultllm_graph_result)
    - [`llm_graph_result::get_tokens`](#llm_graph_resultget_tokens)
    - [`llm_graph_result::get_logits`](#llm_graph_resultget_logits)
    - [`llm_graph_result::get_embd`](#llm_graph_resultget_embd)
    - [`llm_graph_result::get_embd_pooled`](#llm_graph_resultget_embd_pooled)
    - [`llm_graph_result::set_inputs`](#llm_graph_resultset_inputs)
    - [`llm_graph_result::add_input`](#llm_graph_resultadd_input)
- **Inherits From**:
    - [`llm_graph_result_i`](#llm_graph_result_i)

**Methods**

---
#### llm\_graph\_result::\~llm\_graph\_result<!-- {{#callable:llm_graph_result::~llm_graph_result}} -->
The destructor `~llm_graph_result` is a virtual default destructor for the `llm_graph_result` class, ensuring proper cleanup of derived class objects.
- **Inputs**: None
- **Control Flow**:
    - The destructor is declared as virtual to allow derived class destructors to be called correctly when an object is deleted through a base class pointer.
    - The destructor is defined as default, indicating that the compiler should generate the default implementation for it.
- **Output**: There is no output from this destructor as it is used for cleanup purposes.
- **See also**: [`llm_graph_result`](#llm_graph_result)  (Data Structure)


---
#### llm\_graph\_result::get\_tokens<!-- {{#callable:llm_graph_result::get_tokens}} -->
The `get_tokens` function returns the `t_tokens` member variable of the `llm_graph_result` class.
- **Inputs**: None
- **Control Flow**:
    - The function is a simple getter method that directly returns the `t_tokens` member variable.
    - It overrides a virtual method from the `llm_graph_result_i` interface.
- **Output**: A pointer to a `ggml_tensor` object, specifically the `t_tokens` member of the `llm_graph_result` class.
- **See also**: [`llm_graph_result`](#llm_graph_result)  (Data Structure)


---
#### llm\_graph\_result::get\_logits<!-- {{#callable:llm_graph_result::get_logits}} -->
The `get_logits` function returns the `t_logits` tensor from the `llm_graph_result` class.
- **Inputs**: None
- **Control Flow**:
    - The function is a simple getter method that directly returns the `t_logits` member variable of the `llm_graph_result` class.
- **Output**: The function returns a pointer to a `ggml_tensor`, specifically the `t_logits` tensor.
- **See also**: [`llm_graph_result`](#llm_graph_result)  (Data Structure)


---
#### llm\_graph\_result::get\_embd<!-- {{#callable:llm_graph_result::get_embd}} -->
The `get_embd` function returns the `t_embd` tensor from the `llm_graph_result` class.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the member variable `t_embd` without any additional logic or computation.
- **Output**: A pointer to a `ggml_tensor` object, specifically the `t_embd` tensor.
- **See also**: [`llm_graph_result`](#llm_graph_result)  (Data Structure)


---
#### llm\_graph\_result::get\_embd\_pooled<!-- {{#callable:llm_graph_result::get_embd_pooled}} -->
The `get_embd_pooled` function returns a pointer to the pooled embedding tensor `t_embd_pooled`.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the member variable `t_embd_pooled`.
- **Output**: A pointer to a `ggml_tensor` representing the pooled embeddings.
- **See also**: [`llm_graph_result`](#llm_graph_result)  (Data Structure)


---
#### llm\_graph\_result::set\_inputs<!-- {{#callable:llm_graph_result::set_inputs}} -->
The `set_inputs` function iterates over a collection of input objects and sets their input using a provided `llama_ubatch` object.
- **Inputs**:
    - `ubatch`: A pointer to a `llama_ubatch` object, which is used to set the input for each input object in the `inputs` vector.
- **Control Flow**:
    - Iterate over each element in the `inputs` vector.
    - For each element, call its `set_input` method with the `ubatch` argument.
- **Output**: This function does not return any value; it modifies the state of the input objects in the `inputs` vector.
- **See also**: [`llm_graph_result`](#llm_graph_result)  (Data Structure)


---
#### llm\_graph\_result::add\_input<!-- {{#callable:llm_graph_result::add_input}} -->
The `add_input` function adds a new input to the `inputs` vector and returns a pointer to the newly added input.
- **Inputs**:
    - `input`: A unique pointer to an object of type `llm_graph_input_i`, representing the input to be added to the `inputs` vector.
- **Control Flow**:
    - The function takes a unique pointer `input` and moves it into the `inputs` vector using `emplace_back`.
    - It then returns a raw pointer to the last element in the `inputs` vector, which is the newly added input.
- **Output**: A raw pointer to the newly added `llm_graph_input_i` object in the `inputs` vector.
- **See also**: [`llm_graph_result`](#llm_graph_result)  (Data Structure)



---
### llm\_graph\_params<!-- {{#data_structure:llm_graph_params}} -->
- **Type**: `struct`
- **Members**:
    - `ctx`: A pointer to a ggml_context, representing the context for the graph.
    - `arch`: A constant llm_arch, indicating the architecture type.
    - `hparams`: A reference to llama_hparams, containing hyperparameters for the model.
    - `cparams`: A reference to llama_cparams, containing configuration parameters for the model.
    - `ubatch`: A reference to llama_ubatch, representing a unit batch of data.
    - `sched`: A ggml_backend_sched_t, representing the scheduling backend.
    - `backend_cpu`: A ggml_backend_t, representing the CPU backend.
    - `cvec`: A pointer to llama_adapter_cvec, representing adapter vectors.
    - `loras`: A pointer to llama_adapter_loras, representing adapter loras.
    - `mstate`: A pointer to llama_memory_state_i, representing memory state information.
    - `cross`: A pointer to llama_cross, representing cross-attention data.
    - `n_outputs`: An int32_t, indicating the number of outputs.
    - `cb`: A reference to llm_graph_cb, a callback function for custom tensor logic.
- **Description**: The `llm_graph_params` struct is a comprehensive data structure used to encapsulate various parameters and configurations necessary for constructing and managing a graph in a machine learning model. It includes pointers and references to contexts, architectures, hyperparameters, configuration parameters, and unit batches, as well as backend scheduling and CPU backend information. Additionally, it holds pointers to adapter vectors, loras, memory states, and cross-attention data, along with an integer specifying the number of outputs and a callback function for applying custom logic to tensors.


---
### llm\_graph\_context<!-- {{#data_structure:llm_graph_context}} -->
- **Type**: `struct`
- **Members**:
    - `arch`: Specifies the architecture type for the graph context.
    - `hparams`: References the hyperparameters used in the graph context.
    - `cparams`: References the configuration parameters used in the graph context.
    - `ubatch`: References the micro-batch used in the graph context.
    - `n_embd`: Specifies the number of embeddings.
    - `n_layer`: Specifies the number of layers.
    - `n_rot`: Specifies the number of rotations.
    - `n_ctx`: Specifies the user-defined context size.
    - `n_head`: Specifies the number of attention heads.
    - `n_head_kv`: Specifies the number of key-value attention heads.
    - `n_embd_head_k`: Specifies the number of embedding dimensions per key head.
    - `n_embd_k_gqa`: Specifies the number of embedding dimensions for grouped query attention.
    - `n_embd_head_v`: Specifies the number of embedding dimensions per value head.
    - `n_embd_v_gqa`: Specifies the number of embedding dimensions for grouped value attention.
    - `n_expert`: Specifies the number of experts.
    - `n_expert_used`: Specifies the number of experts used.
    - `freq_base`: Specifies the base frequency for positional encoding.
    - `freq_scale`: Specifies the scale factor for frequency in positional encoding.
    - `ext_factor`: Specifies an external factor for the graph context.
    - `attn_factor`: Specifies the attention factor for the graph context.
    - `beta_fast`: Specifies the fast beta parameter for the graph context.
    - `beta_slow`: Specifies the slow beta parameter for the graph context.
    - `norm_eps`: Specifies the epsilon value for normalization.
    - `norm_rms_eps`: Specifies the epsilon value for RMS normalization.
    - `n_tokens`: Specifies the number of tokens.
    - `n_outputs`: Specifies the number of outputs.
    - `n_ctx_orig`: Specifies the original context size for yarn.
    - `pooling_type`: Specifies the type of pooling used.
    - `rope_type`: Specifies the type of rotary positional encoding used.
    - `ctx0`: Pointer to the ggml context.
    - `sched`: Specifies the backend scheduling type.
    - `backend_cpu`: Specifies the backend type for CPU operations.
    - `cvec`: Pointer to the adapter vector for the graph context.
    - `loras`: Pointer to the adapter loras for the graph context.
    - `mstate`: Pointer to the memory state interface.
    - `cross`: Pointer to the cross-attention data.
    - `cb_func`: Reference to the callback function for tensor operations.
    - `res`: Unique pointer to the graph result object.
- **Description**: The `llm_graph_context` struct is a comprehensive data structure designed to encapsulate the configuration and state necessary for constructing and managing a graph in a machine learning model, particularly for large language models. It includes a wide array of parameters and references, such as architecture type, hyperparameters, configuration parameters, and various embedding and attention-related dimensions. Additionally, it holds pointers to context, scheduling, and backend information, as well as callback functions and result management. This struct is integral to managing the flow and operations within a graph-based computation framework, supporting complex operations like attention mechanisms, normalization, and pooling.
- **Member Functions**:
    - [`llm_graph_context::llm_graph_context`](llama-graph.cpp.driver.md#llm_graph_contextllm_graph_context)
    - [`llm_graph_context::n_pos_per_embd`](llama-graph.cpp.driver.md#llm_graph_contextn_pos_per_embd)
    - [`llm_graph_context::cb`](llama-graph.cpp.driver.md#llm_graph_contextcb)
    - [`llm_graph_context::build_cvec`](llama-graph.cpp.driver.md#llm_graph_contextbuild_cvec)
    - [`llm_graph_context::build_lora_mm`](llama-graph.cpp.driver.md#llm_graph_contextbuild_lora_mm)
    - [`llm_graph_context::build_lora_mm_id`](llama-graph.cpp.driver.md#llm_graph_contextbuild_lora_mm_id)
    - [`llm_graph_context::build_norm`](llama-graph.cpp.driver.md#llm_graph_contextbuild_norm)
    - [`llm_graph_context::build_ffn`](llama-graph.cpp.driver.md#llm_graph_contextbuild_ffn)
    - [`llm_graph_context::build_moe_ffn`](llama-graph.cpp.driver.md#llm_graph_contextbuild_moe_ffn)
    - [`llm_graph_context::build_inp_embd`](llama-graph.cpp.driver.md#llm_graph_contextbuild_inp_embd)
    - [`llm_graph_context::build_inp_pos`](llama-graph.cpp.driver.md#llm_graph_contextbuild_inp_pos)
    - [`llm_graph_context::build_inp_attn_scale`](llama-graph.cpp.driver.md#llm_graph_contextbuild_inp_attn_scale)
    - [`llm_graph_context::build_inp_out_ids`](llama-graph.cpp.driver.md#llm_graph_contextbuild_inp_out_ids)
    - [`llm_graph_context::build_inp_mean`](llama-graph.cpp.driver.md#llm_graph_contextbuild_inp_mean)
    - [`llm_graph_context::build_inp_cls`](llama-graph.cpp.driver.md#llm_graph_contextbuild_inp_cls)
    - [`llm_graph_context::build_inp_s_copy`](llama-graph.cpp.driver.md#llm_graph_contextbuild_inp_s_copy)
    - [`llm_graph_context::build_inp_s_mask`](llama-graph.cpp.driver.md#llm_graph_contextbuild_inp_s_mask)
    - [`llm_graph_context::build_inp_cross_embd`](llama-graph.cpp.driver.md#llm_graph_contextbuild_inp_cross_embd)
    - [`llm_graph_context::build_inp_pos_bucket_enc`](llama-graph.cpp.driver.md#llm_graph_contextbuild_inp_pos_bucket_enc)
    - [`llm_graph_context::build_inp_pos_bucket_dec`](llama-graph.cpp.driver.md#llm_graph_contextbuild_inp_pos_bucket_dec)
    - [`llm_graph_context::build_pos_bias`](llama-graph.cpp.driver.md#llm_graph_contextbuild_pos_bias)
    - [`llm_graph_context::build_attn_mha`](llama-graph.cpp.driver.md#llm_graph_contextbuild_attn_mha)
    - [`llm_graph_context::build_attn_inp_no_cache`](llama-graph.cpp.driver.md#llm_graph_contextbuild_attn_inp_no_cache)
    - [`llm_graph_context::build_attn`](llama-graph.cpp.driver.md#llm_graph_contextbuild_attn)
    - [`llm_graph_context::build_attn_inp_kv_unified`](llama-graph.cpp.driver.md#llm_graph_contextbuild_attn_inp_kv_unified)
    - [`llm_graph_context::build_attn`](llama-graph.cpp.driver.md#llm_graph_contextbuild_attn)
    - [`llm_graph_context::build_attn_inp_kv_unified_iswa`](llama-graph.cpp.driver.md#llm_graph_contextbuild_attn_inp_kv_unified_iswa)
    - [`llm_graph_context::build_attn`](llama-graph.cpp.driver.md#llm_graph_contextbuild_attn)
    - [`llm_graph_context::build_attn_inp_cross`](llama-graph.cpp.driver.md#llm_graph_contextbuild_attn_inp_cross)
    - [`llm_graph_context::build_attn`](llama-graph.cpp.driver.md#llm_graph_contextbuild_attn)
    - [`llm_graph_context::build_copy_mask_state`](llama-graph.cpp.driver.md#llm_graph_contextbuild_copy_mask_state)
    - [`llm_graph_context::build_rwkv_token_shift_load`](llama-graph.cpp.driver.md#llm_graph_contextbuild_rwkv_token_shift_load)
    - [`llm_graph_context::build_rwkv_token_shift_store`](llama-graph.cpp.driver.md#llm_graph_contextbuild_rwkv_token_shift_store)
    - [`llm_graph_context::build_pooling`](llama-graph.cpp.driver.md#llm_graph_contextbuild_pooling)


